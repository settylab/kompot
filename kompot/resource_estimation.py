"""Unified resource estimation and planning utilities for kompot.

This module provides tools to estimate and check memory and disk space requirements
before running computations, helping users plan their analyses and avoid failures.
"""

import numpy as np
import os
import tempfile
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import dask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

logger = logging.getLogger("kompot")


def human_readable_size(size_in_bytes: int) -> str:
    """Convert bytes to human-readable string (e.g., '1.23 GB')."""
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(size_in_bytes)
    unit_index = 0

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}" if size > 0 else "0.00 B"


@dataclass
class ResourceRequirement:
    """A single resource requirement (memory or disk)."""
    name: str  # Descriptive name (e.g., "Covariance matrix")
    size_bytes: int  # Size in bytes
    resource_type: str  # 'memory' or 'disk'
    shape: Optional[Tuple[int, ...]] = None  # Array shape if applicable
    field_name: Optional[str] = None  # AnnData field name if applicable
    overwrite: bool = False  # Whether this would overwrite existing data

    @property
    def size_human(self) -> str:
        """Human-readable size."""
        return human_readable_size(self.size_bytes)


@dataclass
class ResourceAvailability:
    """Current system resource availability."""
    memory_total: int
    memory_available: int
    disk_path: str
    disk_total: int
    disk_available: int

    @property
    def memory_total_human(self) -> str:
        return human_readable_size(self.memory_total)

    @property
    def memory_available_human(self) -> str:
        return human_readable_size(self.memory_available)

    @property
    def disk_total_human(self) -> str:
        return human_readable_size(self.disk_total)

    @property
    def disk_available_human(self) -> str:
        return human_readable_size(self.disk_available)


@dataclass
class ResourcePlan:
    """Complete resource usage plan and availability check."""
    requirements: List[ResourceRequirement] = field(default_factory=list)
    availability: Optional[ResourceAvailability] = None
    output_fields: Dict[str, List[str]] = field(default_factory=dict)  # Fields that will be created
    info: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def total_memory_required(self) -> int:
        """Total memory required in bytes."""
        return sum(r.size_bytes for r in self.requirements if r.resource_type == 'memory')

    @property
    def total_disk_required(self) -> int:
        """Total disk space required in bytes."""
        return sum(r.size_bytes for r in self.requirements if r.resource_type == 'disk')

    @property
    def memory_ratio(self) -> float:
        """Ratio of required to available memory."""
        if self.availability is None or self.availability.memory_available == 0:
            return float('inf')
        return self.total_memory_required / self.availability.memory_available

    @property
    def disk_ratio(self) -> float:
        """Ratio of required to available disk space."""
        if self.availability is None or self.availability.disk_available == 0:
            return float('inf')
        return self.total_disk_required / self.availability.disk_available

    @property
    def is_feasible(self) -> bool:
        """Whether the plan is feasible (no errors)."""
        return len(self.errors) == 0

    def add_requirement(self, name: str, size_bytes: int, resource_type: str,
                       shape: Optional[Tuple] = None, field_name: Optional[str] = None,
                       overwrite: bool = False):
        """Add a resource requirement to the plan."""
        req = ResourceRequirement(
            name=name,
            size_bytes=size_bytes,
            resource_type=resource_type,
            shape=shape,
            field_name=field_name,
            overwrite=overwrite
        )
        self.requirements.append(req)

    def check_availability(self, memory_threshold: float = 0.8, disk_threshold: float = 0.9):
        """
        Check resource availability and generate warnings/errors.

        Parameters
        ----------
        memory_threshold : float
            Maximum fraction of available memory to use (default 0.8 = 80%)
        disk_threshold : float
            Maximum fraction of available disk to use (default 0.9 = 90%)
        """
        if self.availability is None:
            self.warnings.append("Could not check resource availability (psutil not installed)")
            return

        # Check memory
        if self.total_memory_required > 0:
            if self.memory_ratio > 1.0:
                self.errors.append(
                    f"Insufficient memory: Need {human_readable_size(self.total_memory_required)}, "
                    f"but only {self.availability.memory_available_human} available"
                )
            elif self.memory_ratio > memory_threshold:
                self.warnings.append(
                    f"High memory usage: Will use {self.memory_ratio*100:.0f}% "
                    f"({human_readable_size(self.total_memory_required)}) "
                    f"of available memory ({self.availability.memory_available_human}). "
                    f"Consider using disk_storage_dir to offload to disk."
                )

        # Check disk
        if self.total_disk_required > 0:
            if self.disk_ratio > 1.0:
                # Get alternative suggestions
                alternatives = suggest_alternative_disk_locations()
                suitable_alternatives = [
                    (path, free_h) for path, free_h, free_bytes in alternatives
                    if free_bytes >= self.total_disk_required
                ][:3]  # Top 3

                error_msg = (
                    f"Insufficient disk space at {self.availability.disk_path}: "
                    f"Need {human_readable_size(self.total_disk_required)}, "
                    f"but only {self.availability.disk_available_human} available."
                )

                if suitable_alternatives:
                    error_msg += "\nSuggested locations with sufficient space:"
                    for path, free_h in suitable_alternatives:
                        error_msg += f"\n  - {path}: {free_h} available"
                    error_msg += f"\n  Use: disk_storage_dir='{suitable_alternatives[0][0]}'"
                else:
                    error_msg += (
                        "\nSolutions: (1) Use disk_storage_dir='/path/to/larger/disk', "
                        "(2) Set TMPDIR environment variable, "
                        "(3) Reduce n_landmarks or process fewer genes"
                    )

                self.errors.append(error_msg)
            elif self.disk_ratio > disk_threshold:
                self.warnings.append(
                    f"High disk usage: Will use {self.disk_ratio*100:.0f}% "
                    f"({human_readable_size(self.total_disk_required)}) "
                    f"of available disk space ({self.availability.disk_available_human}). "
                    f"Consider using a larger disk via disk_storage_dir parameter."
                )

    def format_report(self, verbose: bool = True) -> str:
        """
        Format a human-readable report of the resource plan.

        Parameters
        ----------
        verbose : bool
            If True, include detailed breakdown of all requirements

        Returns
        -------
        str
            Formatted report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("RESOURCE USAGE PLAN")
        lines.append("=" * 80)

        # System resources
        if self.availability:
            lines.append("\nSystem Resources:")
            lines.append(f"  Memory: {self.availability.memory_available_human} available "
                        f"(of {self.availability.memory_total_human} total)")
            lines.append(f"  Disk:   {self.availability.disk_available_human} available at "
                        f"{self.availability.disk_path}")

        # Requirements summary
        lines.append(f"\nTotal Requirements:")
        if self.total_memory_required > 0:
            ratio_pct = self.memory_ratio * 100 if self.availability else 0
            lines.append(f"  Memory: {human_readable_size(self.total_memory_required)} "
                        f"({ratio_pct:.0f}% of available)")
        if self.total_disk_required > 0:
            ratio_pct = self.disk_ratio * 100 if self.availability else 0
            lines.append(f"  Disk:   {human_readable_size(self.total_disk_required)} "
                        f"({ratio_pct:.0f}% of available)")

        # Detailed breakdown
        if verbose and self.requirements:
            mem_reqs = [r for r in self.requirements if r.resource_type == 'memory']
            disk_reqs = [r for r in self.requirements if r.resource_type == 'disk']

            if mem_reqs:
                lines.append("\nMemory Allocations:")
                for req in mem_reqs:
                    overwrite_mark = " [OVERWRITE]" if req.overwrite else ""
                    shape_info = f" {req.shape}" if req.shape else ""
                    field_info = f" → {req.field_name}" if req.field_name else ""
                    lines.append(f"  • {req.name}{shape_info}: {req.size_human}{field_info}{overwrite_mark}")

            if disk_reqs:
                lines.append("\nDisk Storage:")
                for req in disk_reqs:
                    overwrite_mark = " [OVERWRITE]" if req.overwrite else ""
                    shape_info = f" {req.shape}" if req.shape else ""
                    lines.append(f"  • {req.name}{shape_info}: {req.size_human}{overwrite_mark}")

        # Output fields that will be created
        if self.output_fields:
            lines.append("\nOutput Fields:")
            # Get the map of fields that will be overwritten (field_name -> run_id)
            overwrite_fields = getattr(self, '_overwrite_fields', {})

            for location, fields in sorted(self.output_fields.items()):
                if fields:
                    lines.append(f"  {location}:")
                    for field in fields:
                        # Check if this field will overwrite an existing one
                        if field in overwrite_fields:
                            run_id = overwrite_fields[field]
                            if run_id is not None:
                                overwrite_mark = f" [OVERWRITES run_id={run_id}]"
                            else:
                                overwrite_mark = " [OVERWRITE]"
                        else:
                            overwrite_mark = ""
                        lines.append(f"    - {field}{overwrite_mark}")

        # Info, warnings, and errors
        if self.info:
            lines.append("\nInfo:")
            for info_msg in self.info:
                lines.append(f"  ℹ {info_msg}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  ✗ {error}")

        # Status
        lines.append("\n" + "=" * 80)
        if self.errors:
            lines.append("STATUS: ✗ INFEASIBLE - Cannot proceed due to errors above")
        elif self.warnings:
            lines.append("STATUS: ⚠ FEASIBLE WITH WARNINGS - Proceed with caution")
        else:
            lines.append("STATUS: ✓ FEASIBLE - Sufficient resources available")
        lines.append("=" * 80)

        return "\n".join(lines)


def get_system_resources(disk_path: Optional[str] = None) -> ResourceAvailability:
    """
    Get current system resource availability.

    Parameters
    ----------
    disk_path : str, optional
        Path to check disk space for. If None, uses current working directory.

    Returns
    -------
    ResourceAvailability
        Current resource availability
    """
    if disk_path is None:
        disk_path = os.getcwd()

    # Get memory info
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        mem_total = mem.total
        mem_available = mem.available
    else:
        # Fallback estimates if psutil not available
        logger.warning("psutil not installed - using rough memory estimates")
        mem_total = 8 * 1024**3  # Assume 8GB
        mem_available = 4 * 1024**3  # Assume 4GB available

    # Get disk info
    if PSUTIL_AVAILABLE:
        disk = psutil.disk_usage(disk_path)
        disk_total = disk.total
        disk_free = disk.free
    else:
        # Fallback
        logger.warning("psutil not installed - using rough disk estimates")
        disk_total = 100 * 1024**3  # Assume 100GB
        disk_free = 50 * 1024**3  # Assume 50GB free

    return ResourceAvailability(
        memory_total=mem_total,
        memory_available=mem_available,
        disk_path=disk_path,
        disk_total=disk_total,
        disk_available=disk_free
    )


def suggest_alternative_disk_locations() -> List[Tuple[str, str, int]]:
    """
    Suggest alternative directories with more disk space.

    Returns
    -------
    list
        List of (directory_path, free_space_human, free_space_bytes) tuples,
        sorted by free space descending
    """
    candidates = []

    # Common locations to check
    paths_to_check = [
        os.path.expanduser("~"),  # Home directory
        "/tmp",
        os.environ.get("TMPDIR", ""),
        os.environ.get("SCRATCH", ""),
        os.environ.get("TEMP", ""),
    ]

    # Also check for /scratch if it exists (common on HPC systems)
    if os.path.exists("/scratch"):
        paths_to_check.append("/scratch")

    for path in paths_to_check:
        if path and os.path.exists(path) and os.path.isdir(path):
            try:
                avail = get_system_resources(path)
                candidates.append((path, avail.disk_available_human, avail.disk_available))
            except:
                pass

    # Sort by free space descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    return candidates


def estimate_array_size(shape: Tuple[int, ...], dtype=np.float64) -> int:
    """
    Estimate size in bytes for an array with given shape and dtype.

    Parameters
    ----------
    shape : tuple
        Array shape
    dtype : numpy dtype
        Data type (default float64)

    Returns
    -------
    int
        Size in bytes
    """
    dtype_size = np.dtype(dtype).itemsize
    num_elements = np.prod(shape)
    return int(num_elements * dtype_size)


def estimate_differential_expression_resources(
    adata,
    condition1: str,
    condition2: str,
    groupby: str,
    use_sample_variance: bool = False,
    store_arrays_on_disk: bool = False,
    disk_storage_dir: Optional[str] = None,
    landmarks: Optional[np.ndarray] = None,
    **kwargs
) -> ResourcePlan:
    """
    Estimate resource requirements for differential expression analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    condition1 : str
        First condition to compare
    condition2 : str
        Second condition to compare
    groupby : str
        Column in adata.obs that defines groups
    use_sample_variance : bool
        Whether sample variance estimation will be used
    store_arrays_on_disk : bool
        Whether arrays will be stored on disk
    disk_storage_dir : str, optional
        Directory for disk storage
    landmarks : np.ndarray, optional
        Landmark points (if None, uses all cells)
    **kwargs
        Additional parameters

    Returns
    -------
    ResourcePlan
        Complete resource plan with requirements and availability check
    """
    plan = ResourcePlan()

    # Get system resources
    # Determine disk path following the same logic as DiskStorage:
    # - If disk_storage_dir provided, use it
    # - Otherwise use system temp directory (respects TMPDIR, TEMP, TMP env vars)
    if disk_storage_dir:
        check_path = disk_storage_dir
    else:
        # Use Python's tempfile logic to determine where temp files would actually be created
        # Create and immediately remove a temp directory to get the actual path tempfile uses
        temp_test_dir = tempfile.mkdtemp(prefix="kompot_test_")
        check_path = os.path.dirname(temp_test_dir)
        os.rmdir(temp_test_dir)

    plan.availability = get_system_resources(check_path)

    # Get basic dimensions
    n_cells = adata.n_obs

    # Handle genes parameter
    genes_param = kwargs.get('genes')
    if genes_param is not None:
        n_genes = len(genes_param)
    else:
        n_genes = adata.n_vars

    # Handle null_genes parameter - these are ADDED to the gene count for null distribution
    null_genes_param = kwargs.get('null_genes', 2000)  # Default is 2000
    compute_mahalanobis = kwargs.get('compute_mahalanobis', True)

    n_null_genes = 0
    if null_genes_param is not None and null_genes_param != 0 and compute_mahalanobis:
        if isinstance(null_genes_param, int):
            n_null_genes = null_genes_param
        elif isinstance(null_genes_param, list):
            n_null_genes = len(null_genes_param)

        # Total genes processed includes null genes
        n_total_genes = n_genes + n_null_genes

        if n_null_genes > 0:
            plan.info.append(
                f"Null distribution will use {n_null_genes} additional genes "
                f"(total: {n_total_genes} genes processed)"
            )
    else:
        n_total_genes = n_genes

    # Handle landmarks
    landmarks_param = kwargs.get('landmarks')
    n_landmarks_param = kwargs.get('n_landmarks', 5000)

    if landmarks_param is not None:
        n_landmarks = len(landmarks_param)
    elif landmarks is not None:
        n_landmarks = len(landmarks)
    elif n_landmarks_param is not None:
        n_landmarks = min(n_landmarks_param, n_cells)
    else:
        n_landmarks = n_cells

    # Get the correct result_key (default is 'kompot_de' not 'de')
    result_key = kwargs.get('result_key', 'kompot_de')

    # Infer use_sample_variance from sample_col if not explicitly set
    inferred_use_sv = use_sample_variance or (kwargs.get('sample_col') is not None)

    # Estimate memory requirements

    # 1. Mellon's factorized precision matrices (L) - stored by each FunctionEstimator
    # These are Cholesky factors of the precision matrix.
    # When landmarks are used: shape (n_landmarks, n_landmarks)
    # When no landmarks: shape (n_training_cells, n_training_cells)
    # We have 2 conditions, so 2 FunctionEstimators

    # Determine effective training size (landmarks if used, otherwise full training set)
    if landmarks is not None:
        n_train_cond1 = n_landmarks
        n_train_cond2 = n_landmarks
        l_desc = f"{n_landmarks:,} landmarks"
    else:
        n_train_cond1 = (adata.obs[groupby] == condition1).sum()
        n_train_cond2 = (adata.obs[groupby] == condition2).sum()
        l_desc = f"{n_train_cond1:,}/{n_train_cond2:,} cells"

    L_size_cond1 = estimate_array_size((n_train_cond1, n_train_cond1))
    L_size_cond2 = estimate_array_size((n_train_cond2, n_train_cond2))

    plan.add_requirement(
        f"Mellon precision matrix L (condition 1, {l_desc})",
        L_size_cond1,
        'memory',
        shape=(n_train_cond1, n_train_cond1)
    )

    plan.add_requirement(
        f"Mellon precision matrix L (condition 2, {l_desc})",
        L_size_cond2,
        'memory',
        shape=(n_train_cond2, n_train_cond2)
    )

    # 2. Gene expression predictions - stored as LAYERS in AnnData
    # These are NOT temporary - they get stored as layers
    # NOTE: Uses n_total_genes (includes null genes if used for FDR)
    imputed_size = estimate_array_size((n_cells, n_total_genes))

    # Generate field names early so we can use them for requirements
    from .anndata.utils.field_tracking import generate_output_field_names

    field_names = generate_output_field_names(
        result_key=result_key,
        condition1=condition1,
        condition2=condition2,
        analysis_type="de",
        with_sample_suffix=inferred_use_sv
    )

    plan.add_requirement(
        f"Imputed expression (condition 1)",
        imputed_size,
        'memory',
        shape=(n_cells, n_total_genes),
        field_name=f"adata.layers['{field_names['imputed_key_1']}']"
    )

    plan.add_requirement(
        f"Imputed expression (condition 2)",
        imputed_size,
        'memory',
        shape=(n_cells, n_total_genes),
        field_name=f"adata.layers['{field_names['imputed_key_2']}']"
    )

    # Fold change layer
    plan.add_requirement(
        f"Fold change",
        imputed_size,
        'memory',
        shape=(n_cells, n_total_genes),
        field_name=f"adata.layers['{field_names['fold_change_key']}']"
    )

    # Standard deviation layers (if sample variance is used)
    if inferred_use_sv:
        plan.add_requirement(
            f"Standard deviation (condition 1)",
            imputed_size,
            'memory',
            shape=(n_cells, n_total_genes),
            field_name=f"adata.layers['{field_names['std_key_1']}']"
        )
        plan.add_requirement(
            f"Standard deviation (condition 2)",
            imputed_size,
            'memory',
            shape=(n_cells, n_total_genes),
            field_name=f"adata.layers['{field_names['std_key_2']}']"
        )

    # Fold change z-scores layer (if store_additional_stats is enabled)
    if kwargs.get('store_additional_stats', False):
        plan.add_requirement(
            f"Fold change z-scores",
            imputed_size,
            'memory',
            shape=(n_cells, n_total_genes),
            field_name=f"adata.layers['{field_names['fold_change_zscores_key']}']"
        )

    # Cell batching memory (temporary Kus matrix during prediction)
    # During prediction, mellon computes Kus = cov_func(X_batch, landmarks): (batch_cells, n_landmarks)
    # This is a temporary matrix that scales with batch_size
    cell_batch_size = kwargs.get('batch_size', 100 if 'anndata' in str(type(adata)) else 500)

    # Determine effective batch size (use all cells if batch_size is None/0 or >= n_cells)
    if cell_batch_size is None or cell_batch_size <= 0 or cell_batch_size >= n_cells:
        effective_cell_batch = n_cells
        batching_cells = False
    else:
        effective_cell_batch = cell_batch_size
        batching_cells = True

    # Temporary memory during prediction (per operation):
    # 1. Kus = cov_func(X_batch, landmarks): (batch_cells, n_landmarks)
    # 2. Temporary result from matmul before assignment: (batch_cells, n_genes)
    # We do 3-6 operations: predict_cond1, predict_cond2, uncertainty1, uncertainty2, [sample_var1, sample_var2]
    n_prediction_ops = 4 + (2 if inferred_use_sv else 0)

    kus_size = estimate_array_size((effective_cell_batch, n_landmarks))
    temp_result_size = estimate_array_size((effective_cell_batch, n_total_genes))
    total_temp_per_op = kus_size + temp_result_size

    # Always add as requirement (shows in Memory Allocations)
    plan.add_requirement(
        f"Temporary matrices during predictions (batch_size={cell_batch_size})",
        total_temp_per_op,
        'memory',
        shape=f"({effective_cell_batch}, {n_landmarks}) + ({effective_cell_batch}, {n_total_genes})"
    )

    if batching_cells:
        # Cell batching is active - explain the savings
        full_temp = estimate_array_size((n_cells, n_landmarks)) + estimate_array_size((n_cells, n_total_genes))
        plan.info.append(
            f"Cell batching reduces memory: Each of {n_prediction_ops} prediction operations uses "
            f"~{human_readable_size(total_temp_per_op)} temporary arrays instead of "
            f"{human_readable_size(full_temp)} (saving {human_readable_size(full_temp - total_temp_per_op)})."
        )
    else:
        # No cell batching - suggest improvement
        smaller_batch = min(500, n_cells)
        smaller_temp = estimate_array_size((smaller_batch, n_landmarks)) + estimate_array_size((smaller_batch, n_total_genes))
        plan.info.append(
            f"Cell batch_size ({cell_batch_size}) processes all {n_cells} cells at once. "
            f"Consider reducing batch_size (e.g., batch_size={smaller_batch}) to lower peak memory by "
            f"{human_readable_size(total_temp_per_op - smaller_temp)}."
        )

    # Intermediate arrays during predict() - CRITICAL FOR PEAK MEMORY
    # Even with cell batching, apply_batched() pre-allocates full output arrays (n_cells, n_genes)
    # During the predict() method in differential_expression.py, intermediate arrays coexist.
    #
    # Memory optimization history:
    # - Original (2025-10-12): ~30 arrays identified via SLURM MaxRSS
    # - zeros_like optimization (2025-10-13): Reduced to ~28 arrays
    #   For No SV case: condition1/2_sample_variance use scalar 0 instead of full arrays
    # - Manual optimizations (2025-10-13): Reduced to ~25 arrays
    #   1. Eliminated 'stds' intermediate array (inlined computation)
    #   2. Strategic del statements improve temporal locality (lines 825, 830, 835, 842)
    #   3. Early cleanup of uncertainties and total_variance arrays
    #
    # Remaining arrays include:
    # - 6 primary arrays from apply_batched (condition1/2_imputed, uncertainties)
    # - fold_change and derived quantities (z-scores, condition1/2_std, total_variance)
    # - Temporaries during numpy operations (addition, sqrt, division)
    # - Python/numpy internal buffers and copies
    #
    # These are created during computation but freed before final result is returned.
    # SLURM MaxRSS captures this peak; discrete memory measurements miss it due to GC.
    n_intermediate_arrays = 25  # Reduced from 28 via manual optimizations (2025-10-13)
    intermediate_array_size = estimate_array_size((n_cells, n_total_genes))
    total_intermediate_memory = n_intermediate_arrays * intermediate_array_size

    plan.add_requirement(
        f"Peak intermediate arrays during predictions (~{n_intermediate_arrays} arrays)",
        total_intermediate_memory,
        'memory',
        shape=f"{n_intermediate_arrays}×({n_cells}, {n_total_genes})"
    )

    plan.info.append(
        f"Prediction creates ~{n_intermediate_arrays} intermediate arrays of shape ({n_cells:,}, {n_total_genes}). "
        f"These coexist at peak memory ({human_readable_size(total_intermediate_memory)}) but are freed before completion."
    )

    # 2. Function predictor covariance matrices (ALWAYS created for Mahalanobis distance)
    # These are created by function_predictor.covariance(X, diag=False)
    cov_matrix_shape = (n_landmarks, n_landmarks)
    cov_size = estimate_array_size(cov_matrix_shape)

    resource_type = 'memory'  # Function predictor covs are always in memory
    plan.add_requirement(
        "Function predictor covariances (per condition)",
        cov_size * 2,  # cov1 and cov2
        resource_type,
        shape=cov_matrix_shape
    )

    # Combined covariance matrix (averaged)
    plan.add_requirement(
        "Combined covariance matrix",
        cov_size,  # (cov1 + cov2) / 2
        'memory',
        shape=cov_matrix_shape
    )

    # Cholesky decomposition of combined covariance (for Mahalanobis computation)
    if compute_mahalanobis:
        plan.add_requirement(
            "Cholesky decomposition (for Mahalanobis)",
            cov_size,  # Same size as covariance matrix
            'memory',
            shape=cov_matrix_shape
        )

    # 3. Sample variance covariance matrices (if enabled)
    if inferred_use_sv:
        # Count unique samples
        sample_col = kwargs.get('sample_column') or kwargs.get('sample_col')
        if sample_col and sample_col in adata.obs:
            n_samples_cond1 = adata.obs[adata.obs[groupby] == condition1][sample_col].nunique()
            n_samples_cond2 = adata.obs[adata.obs[groupby] == condition2][sample_col].nunique()
            n_samples = max(n_samples_cond1, n_samples_cond2)

            # Per-sample imputations: Each sample predictor imputes expression for all landmarks
            # Shape: (n_landmarks, n_total_genes) per sample (includes null genes)
            # Stacked: (n_samples, n_landmarks, n_total_genes)
            per_sample_impute_shape = (n_samples, n_landmarks, n_total_genes)
            per_sample_impute_size = estimate_array_size(per_sample_impute_shape)

            plan.add_requirement(
                f"Per-sample imputations ({n_samples} samples)",
                per_sample_impute_size * 2,  # Both conditions
                'memory',
                shape=per_sample_impute_shape
            )

            # Sample variance creates PER-GENE covariance matrices
            # Shape: (n_landmarks, n_landmarks, n_total_genes) - includes null genes
            sv_cov_shape = (n_landmarks, n_landmarks, n_total_genes)
            sv_cov_size = estimate_array_size(sv_cov_shape)

            resource_type = 'disk' if store_arrays_on_disk else 'memory'
            plan.add_requirement(
                f"Sample covariances (per condition, {n_samples} samples)",
                sv_cov_size * 2,  # variance1 and variance2 - both stored
                resource_type,
                shape=sv_cov_shape
            )

            # Note: combined_variance (variance1 + variance2) behavior differs by storage type:
            # - Disk storage: Creates a lazy Dask computation graph (no additional disk space)
            # - Memory storage: Creates a third in-memory tensor (3x total memory)
            if not store_arrays_on_disk:
                # In memory: variance1, variance2, and combined_variance all exist as separate arrays
                plan.add_requirement(
                    f"Combined sample covariances (temporary)",
                    sv_cov_size,
                    'memory',
                    shape=sv_cov_shape
                )
                total_sv = sv_cov_size * 3  # variance1 + variance2 + combined
                plan.warnings.append(
                    f"Sample variance covariance tensors ({human_readable_size(total_sv)}) "
                    f"will be stored in memory. Consider using disk_storage_dir for large datasets."
                )

            if store_arrays_on_disk:
                # Add note about disk location
                if disk_storage_dir:
                    plan.info.append(
                        f"Disk arrays will be stored at: {check_path}"
                    )
                else:
                    plan.info.append(
                        f"Disk arrays will be stored at system temp: {check_path}. "
                        f"To use a different location, set disk_storage_dir='/path/to/disk' "
                        f"or export TMPDIR=/path/to/disk"
                    )

                if not DASK_AVAILABLE:
                    plan.warnings.append(
                        "Disk storage requested but dask is not installed. "
                        "Install dask for 50x faster computation: pip install 'dask[array]'"
                    )

    # Add batch_size information
    batch_size = kwargs.get('batch_size', 100 if 'anndata' in str(type(adata)) else 500)

    if compute_mahalanobis:
        if not inferred_use_sv:
            # Without sample variance: batch_size affects Mahalanobis computation memory
            # Empirical testing shows JAX efficiently reuses memory during vmap operations.
            # While theoretically we might expect 4× (batch_diffs, solved, solved**2, workspace),
            # actual measurements show ~1× due to in-place operations and immediate deallocation.
            # Using 1.5× as a conservative estimate to account for temporary workspace.
            total_batch_genes = batch_size if batch_size > 0 and batch_size < n_total_genes else n_total_genes
            actual_batch_mem = estimate_array_size((total_batch_genes, n_landmarks))

            # Add this as a memory requirement so it shows in the allocations list
            batch_shape = (total_batch_genes, n_landmarks)
            plan.add_requirement(
                f"Mahalanobis batch processing (batch_size={batch_size})",
                actual_batch_mem,
                'memory',
                shape=batch_shape
            )

            plan.info.append(
                f"Mahalanobis computation processes {total_batch_genes} genes per batch. "
                f"Reduce batch_size to lower peak memory (currently {human_readable_size(actual_batch_mem)} for batch arrays)."
            )

            # If batch_size is 0 or greater than genes, warn about memory
            if batch_size == 0 or batch_size >= n_total_genes:
                plan.warnings.append(
                    f"batch_size ({batch_size}) will process all {n_total_genes} genes at once. "
                    f"Consider reducing batch_size (e.g., batch_size=500) to lower peak memory usage."
                )

    # 4. Check for existing results and what will be overwritten
    from .anndata.utils.field_tracking import detect_output_field_overwrite

    # field_names was already generated earlier, use it here
    all_patterns = field_names.get("all_patterns", {})

    has_overwrites = False
    existing_fields_with_location = []  # Store fields with location prefix
    overwrite_fields_map = {}  # Map field_name -> run_id
    prev_run = None

    # Try to get the field tracking metadata to get run_ids for each field
    storage_key = "kompot_de"
    field_tracking = None
    try:
        from .anndata.utils.json_utils import get_json_metadata
        field_tracking = get_json_metadata(adata, f"{storage_key}.anndata_fields")
    except Exception:
        pass

    for location, patterns in all_patterns.items():
        try:
            has_loc_overwrites, loc_fields, loc_prev_run = detect_output_field_overwrite(
                adata=adata,
                analysis_type="de",
                result_key=result_key,
                output_patterns=patterns,
                location=location
            )

            has_overwrites = has_overwrites or has_loc_overwrites
            # Fields from detect_output_field_overwrite already include location prefix
            # e.g., "var.kompot_de_Young_to_Old_mean_lfc"
            for field_with_location in loc_fields:
                existing_fields_with_location.append(field_with_location)
                # Strip location prefix to get just the field name for marking
                # e.g., "var.kompot_de_Young_to_Old_mean_lfc" -> "kompot_de_Young_to_Old_mean_lfc"
                if "." in field_with_location:
                    field_name = field_with_location.split(".", 1)[1]

                    # Look up the run_id for this field from tracking metadata
                    run_id = None
                    if field_tracking and location in field_tracking:
                        run_id = field_tracking[location].get(field_name)

                    overwrite_fields_map[field_name] = run_id
            if loc_prev_run is not None:
                prev_run = loc_prev_run
        except Exception as e:
            # If checking fails, continue without overwrite info
            # Silently ignore - this is expected if fields don't exist yet
            pass

    # Add overwrite information to the plan
    if has_overwrites:
        if prev_run:
            prev_timestamp = prev_run.get("timestamp", "unknown time")
            prev_params = prev_run.get("params", {})
            prev_cond1 = prev_params.get('condition1', 'unknown')
            prev_cond2 = prev_params.get('condition2', 'unknown')
            prev_use_sv = prev_params.get('use_sample_variance', False)
            prev_null_genes = prev_params.get('null_genes', None)

            # Try to get the run_id from tracking metadata
            run_id = None
            storage_key = "kompot_de"
            try:
                from .anndata.utils.json_utils import get_json_metadata
                tracking = get_json_metadata(adata, f"{storage_key}.anndata_fields")
                if tracking and "uns" in tracking and result_key in tracking["uns"]:
                    run_id = tracking["uns"][result_key]
            except Exception:
                pass

            overwrite_msg = f"Results with result_key='{result_key}' already exist"
            if run_id is not None:
                overwrite_msg += f" (run_id={run_id})"
            overwrite_msg += f". Previous run: {prev_timestamp} comparing {prev_cond1} to {prev_cond2}"

            # Add previous run details
            details = []
            if prev_use_sv:
                details.append("with sample variance")
            if prev_null_genes:
                details.append(f"null_genes={prev_null_genes}")
            if details:
                overwrite_msg += f" ({', '.join(details)})"

            overwrite_msg += f". Fields that will be overwritten: {', '.join(existing_fields_with_location[:5])}"
            if len(existing_fields_with_location) > 5:
                overwrite_msg += f" and {len(existing_fields_with_location) - 5} more"
        else:
            overwrite_msg = f"Will overwrite existing fields: {', '.join(existing_fields_with_location[:5])}"
            if len(existing_fields_with_location) > 5:
                overwrite_msg += f" and {len(existing_fields_with_location) - 5} more"

        plan.warnings.append(overwrite_msg)

    # Store the map of fields that will be overwritten (field_name -> run_id)
    plan._overwrite_fields = overwrite_fields_map

    # Start with basic DE fields
    plan.output_fields = {
        "adata.var": [
            field_names["mahalanobis_key"],
            field_names["mean_lfc_key"],
        ],
        "adata.layers": [
            field_names["imputed_key_1"],
            field_names["imputed_key_2"],
            field_names["fold_change_key"],
        ]
    }

    # Add FDR fields if null_genes is used
    if n_null_genes > 0:
        plan.output_fields["adata.var"].extend([
            field_names["mahalanobis_local_fdr_key"],
            field_names["is_de_key"],
        ])
        if kwargs.get('store_additional_stats', False):
            plan.output_fields["adata.var"].extend([
                field_names["mahalanobis_pvalue_key"],
                field_names["mahalanobis_tail_fdr_key"],
            ])

    # Add std layers if sample variance
    if inferred_use_sv:
        plan.output_fields["adata.layers"].extend([
            field_names["std_key_1"],
            field_names["std_key_2"],
        ])

    # Add fold change z-scores if store_additional_stats
    if kwargs.get('store_additional_stats', False):
        plan.output_fields["adata.layers"].append(field_names["fold_change_zscores_key"])

    # Add posterior covariance if store_posterior_covariance
    if kwargs.get('store_posterior_covariance', False):
        plan.output_fields["adata.obsp"] = [field_names["posterior_covariance_key"]]

    # Check availability
    plan.check_availability()

    return plan


def dry_run_differential_expression(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    verbose: bool = True,
    **kwargs
) -> ResourcePlan:
    """
    Estimate resource requirements for differential expression analysis.

    This is a planning tool that lets you explore different parameter configurations
    and understand resource requirements BEFORE attempting an actual run. Use this to:

    - Compare memory usage with/without sample_variance
    - Decide whether to use disk_storage_dir
    - Choose appropriate landmark subsampling
    - Understand which fields will be created/overwritten with run_id tracking
    - See detailed previous run information for fields that will be overwritten
    - Check if you have sufficient resources

    The actual kompot.compute_differential_expression() also checks resources,
    but this dry-run lets you experiment with parameters without waiting for
    the full computation.

    **Field Overwrite Detection:**

    The dry run shows which fields will be overwritten with their run_id in the
    Output Fields section (e.g., ``[OVERWRITES run_id=0]``). The warnings section
    provides detailed information about the previous run including timestamp,
    conditions, and parameters like ``use_sample_variance`` and ``null_genes``.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    groupby : str
        Column in adata.obs that defines groups
    condition1 : str
        First condition to compare
    condition2 : str
        Second condition to compare
    verbose : bool
        If True, print the resource plan report (default: True)
    **kwargs
        All other parameters that would be passed to compute_differential_expression
        (use_sample_variance, sample_column, disk_storage_dir, landmarks, etc.)

    Returns
    -------
    ResourcePlan
        Complete resource plan with requirements, availability, and feasibility check

    Examples
    --------
    **Compare different configurations:**

    >>> from kompot.resource_estimation import dry_run_differential_expression
    >>>
    >>> # Option 1: In-memory sample variance (highest memory)
    >>> plan1 = dry_run_differential_expression(
    ...     adata, 'treated', 'control', 'condition',
    ...     use_sample_variance=True,
    ...     sample_column='donor_id'
    ... )
    >>>
    >>> # Option 2: Disk-backed sample variance (lower memory, needs disk space)
    >>> plan2 = dry_run_differential_expression(
    ...     adata, 'treated', 'control', 'condition',
    ...     use_sample_variance=True,
    ...     sample_column='donor_id',
    ...     disk_storage_dir='/scratch/de_analysis'
    ... )
    >>>
    >>> # Option 3: No sample variance (lowest resources)
    >>> plan3 = dry_run_differential_expression(
    ...     adata, 'treated', 'control', 'condition',
    ...     use_sample_variance=False
    ... )
    >>>
    >>> # Compare memory usage
    >>> print(f"Option 1 memory: {plan1.requirements[0].size_human}")
    >>> print(f"Option 2 memory: {plan2.requirements[0].size_human}")
    >>> print(f"Option 3 memory: {plan3.requirements[0].size_human}")

    **Check field overwrites:**

    >>> plan = dry_run_differential_expression(
    ...     adata, 'Young', 'Old', 'age',
    ...     result_key='kompot_de'
    ... )
    >>> # The output shows which fields will be overwritten with their run_id:
    >>> # Output Fields:
    >>> #   adata.layers:
    >>> #     - kompot_de_Young_imputed [OVERWRITES run_id=0]
    >>> #     - kompot_de_Old_imputed [OVERWRITES run_id=0]
    >>> #   adata.var:
    >>> #     - kompot_de_Young_to_Old_mean_lfc [OVERWRITES run_id=0]
    >>> #
    >>> # Warnings:
    >>> #   ⚠ Results with result_key='kompot_de' already exist (run_id=0).
    >>> #     Previous run: 2025-10-02T12:30:00 comparing Young to Old
    >>> #     (null_genes=2000). Fields that will be overwritten: ...

    **Use with landmarks:**

    >>> # Test with different landmark counts to find sweet spot
    >>> import numpy as np
    >>> for n_landmarks in [500, 1000, 2000]:
    ...     landmarks = adata.obsm['X_pca'][::adata.n_obs//n_landmarks][:n_landmarks]
    ...     plan = dry_run_differential_expression(
    ...         adata, 'A', 'B', 'condition',
    ...         landmarks=landmarks,
    ...         use_sample_variance=True,
    ...         sample_column='donor',
    ...         verbose=False
    ...     )
    ...     print(f"{n_landmarks} landmarks: {plan.total_memory_required / 1024**3:.2f} GB")

    Notes
    -----
    This function only estimates resources. The actual kompot.compute_differential_expression()
    will perform its own checks before running. Use this dry-run to explore options and
    make informed decisions about parameters.
    """
    plan = estimate_differential_expression_resources(
        adata, condition1, condition2, groupby, **kwargs
    )

    if verbose:
        print(plan.format_report())

    return plan


# Backward compatibility wrappers for existing code
def analyze_covariance_memory_requirements(
    n_points: int,
    n_genes: int,
    max_memory_ratio: float = 0.8,
    analysis_name: str = "Covariance Matrix Memory Analysis",
    store_arrays_on_disk: bool = False,
    log_level: str = "info"
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for analyze_memory_requirements.

    This function maintains compatibility with existing code while using the
    new unified resource estimation system.

    Parameters
    ----------
    n_points : int
        Number of points (cells or landmarks)
    n_genes : int
        Number of genes
    max_memory_ratio : float
        Maximum fraction of available memory to use
    analysis_name : str
        Name for the analysis
    store_arrays_on_disk : bool
        Whether disk storage is enabled
    log_level : str
        Logging level

    Returns
    -------
    dict
        Analysis results compatible with old format
    """
    plan = ResourcePlan()
    plan.availability = get_system_resources()

    # Calculate covariance matrix size
    cov_shape = (n_points, n_points, n_genes)
    cov_size = estimate_array_size(cov_shape)

    # Add requirement
    plan.add_requirement(
        analysis_name,
        cov_size,
        'memory',
        shape=cov_shape
    )

    # Check availability
    plan.check_availability(memory_threshold=max_memory_ratio)

    # Format logs in the old style
    log_func = {
        'debug': logger.debug,
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error,
        'critical': logger.critical
    }.get(log_level.lower(), logger.info)

    # Use debug level if disk storage is already enabled
    if store_arrays_on_disk:
        log_func = logger.debug

    log_func(f"{analysis_name}:")
    log_func(f"  - Total memory required: {plan.requirements[0].size_human}")
    log_func(f"  - Available memory: {plan.availability.memory_available_human}")
    log_func(f"  - Memory usage ratio: {plan.memory_ratio:.2f}x")

    # Log warnings if not using disk storage
    if not store_arrays_on_disk:
        if plan.memory_ratio > max_memory_ratio:
            logger.warning(
                f"CRITICAL: Memory usage ({plan.requirements[0].size_human}) exceeds "
                f"{max_memory_ratio*100:.0f}% of available memory "
                f"({plan.availability.memory_available_human}).\n"
                f"Consider using store_arrays_on_disk=True"
                + (" with dask installed" if DASK_AVAILABLE else " (install dask for better performance)")
            )
        elif plan.memory_ratio > max_memory_ratio * 0.5:
            logger.warning(
                f"WARNING: High memory usage detected. Arrays will use {plan.memory_ratio:.2f}x "
                f"({plan.requirements[0].size_human}) of available memory "
                f"({plan.availability.memory_available_human}). "
                f"Consider using store_arrays_on_disk=True."
            )

    # Return in old format
    return {
        'array_sizes': [{
            'index': 0,
            'shape': cov_shape,
            'size_str': plan.requirements[0].size_human,
            'size_bytes': cov_size
        }],
        'total_size': plan.requirements[0].size_human,
        'total_bytes': cov_size,
        'available_memory': plan.availability.memory_available_human,
        'available_bytes': plan.availability.memory_available,
        'memory_ratio': plan.memory_ratio,
        'status': 'critical' if plan.memory_ratio > max_memory_ratio
                 else 'warning' if plan.memory_ratio > max_memory_ratio * 0.5
                 else 'ok',
        'should_use_disk': plan.memory_ratio > max_memory_ratio * 0.5
    }
