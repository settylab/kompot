"""Expression imputation for AnnData objects."""

import logging
import datetime
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List
from scipy import sparse

try:
    import anndata  # noqa: F401
except ImportError:
    raise ImportError("Please install anndata: pip install anndata")

from ..differential.expression_model import ExpressionModel
from ..settings import GPSettings, StorageSettings, OutputSettings, ModelSettings
from ..validation import (
    validate_obsm_key,
    validate_groupby,
    validate_layer,
    validate_genes,
    validate_sample_col,
)
from .utils import (
    _sanitize_name,
    detect_output_field_overwrite,
    get_environment_info,
    append_to_run_history,
)
from .utils.json_utils import set_json_metadata

logger = logging.getLogger("kompot")


def impute_expression(
    adata,
    groupby: Optional[str] = None,
    condition: Optional[str] = None,
    obsm_key: str = "DM_EigenVectors",
    layer: Optional[str] = None,
    genes: Optional[List[str]] = None,
    sample_col: Optional[str] = None,
    gp: "GPSettings | None" = None,
    storage: "StorageSettings | None" = None,
    output: "OutputSettings | None" = None,
    model: "ModelSettings | None" = None,
    **function_kwargs,
) -> Optional[Union[Dict[str, Any], Any]]:
    """Impute gene expression for a single condition using GP smoothing.

    Fits an :class:`ExpressionModel` on the selected cells and evaluates it
    on **all** cells in ``adata``. This means every cell gets an imputed value
    and uncertainty estimate, even if it was not part of the training condition.
    Stores the imputed values, posterior standard deviations, and (optionally)
    empirical and sample variance layers in ``adata``.

    The most common call is just::

        kompot.impute_expression(adata, "condition", "Young")

    Advanced options are available through the settings dataclasses
    (:class:`~kompot.GPSettings`, :class:`~kompot.StorageSettings`,
    :class:`~kompot.OutputSettings`).  Any field left at its default is
    equivalent to omitting it entirely.  Extra ``**function_kwargs`` are
    forwarded to mellon's :class:`~mellon.FunctionEstimator`.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    groupby : str, optional
        Column in ``adata.obs`` identifying conditions. Required when
        ``condition`` is specified.
    condition : str, optional
        Which group in ``groupby`` to impute. If *None* and ``groupby`` is
        *None*, all cells are used.
    obsm_key : str
        Key in ``adata.obsm`` for cell-state coordinates.
    layer : str, optional
        Layer to use as expression input. *None* means ``adata.X``.
    genes : list of str, optional
        Subset of genes to impute. *None* means all genes.
    sample_col : str, optional
        Column in ``adata.obs`` with biological-replicate labels.
    gp : GPSettings, optional
        GP model parameters (sigma, ls, n_landmarks, etc.).
    storage : StorageSettings, optional
        Output storage parameters (result_key, overwrite).
    output : OutputSettings, optional
        Return behavior (copy, inplace, return_full_results, progress).
    model : ModelSettings, optional
        Pre-fitted :class:`~kompot.ExpressionModel` to inject via
        ``model1``. When provided, skips internal fitting.
    **function_kwargs
        Forwarded to :class:`mellon.FunctionEstimator`.

    Returns
    -------
    None or dict
        *None* when results are stored in-place. If ``return_full_results``
        is *True*, a dictionary with keys ``"model"``, ``"table"``, and
        ``"field_names"``.
    """
    # --- Unpack settings ---
    # Impute has a different batch_size default (500) than DE (100),
    # so unpack with impute-specific defaults when gp is None.
    sigma = gp.sigma if gp is not None else 1.0
    ls = gp.ls if gp is not None else None
    ls_factor = gp.ls_factor if gp is not None else 10.0
    n_landmarks = gp.n_landmarks if gp is not None else 5000
    use_empirical_variance = gp.use_empirical_variance if gp is not None else True
    eps = gp.eps if gp is not None else 1e-8
    random_state = gp.random_state if gp is not None else None
    batch_size = gp.batch_size if gp is not None else 500

    _storage = storage or StorageSettings()
    result_key = _storage.result_key or "kompot_impute"
    overwrite = _storage.overwrite

    _output = output or OutputSettings()
    copy = _output.copy
    inplace = _output.inplace
    return_full_results = _output.return_full_results
    progress = _output.progress

    # Pre-fitted model
    model_settings = model or ModelSettings()
    pre_fitted_model = model_settings.model1

    if copy:
        adata = adata.copy()

    # --- Validate inputs ---
    validate_obsm_key(adata, obsm_key)
    validate_layer(adata, layer)
    validate_genes(adata, genes)
    validate_sample_col(adata, sample_col)

    if condition is not None and groupby is None:
        raise ValueError("'groupby' is required when 'condition' is specified.")

    if groupby is not None:
        validate_groupby(adata, groupby)

    # --- Build cell mask ---
    if groupby is not None and condition is not None:
        mask = (adata.obs[groupby] == condition).values
        if mask.sum() == 0:
            raise ValueError(
                f"Condition '{condition}' not found in column '{groupby}'."
            )
        logger.info(f"Imputing condition '{condition}': {mask.sum():,} cells")
    else:
        mask = np.ones(adata.n_obs, dtype=bool)
        logger.info(f"Imputing all {mask.sum():,} cells")

    # --- Build field names ---
    cond_label = _sanitize_name(condition) if condition else "all"
    field_names = _impute_field_names(
        result_key, cond_label, use_empirical_variance, sample_col is not None
    )

    # --- Overwrite detection ---
    all_fields = field_names["all_patterns"]
    has_ow = False
    ow_fields: list = []
    for loc, patterns in all_fields.items():
        h, f, _ = detect_output_field_overwrite(
            adata=adata,
            output_patterns=patterns,
            location=loc,
            analysis_type="impute",
            overwrite=overwrite is True,
        )
        has_ow = has_ow or h
        ow_fields.extend(f)

    if has_ow:
        msg = (
            f"Imputation results with result_key='{result_key}' already exist. "
            f"Fields: {', '.join(ow_fields[:5])}"
        )
        if overwrite is False:
            raise ValueError(msg + " Set overwrite=True or use a different result_key.")
        elif overwrite is None:
            logger.warning(msg + " Overwriting.")

    # --- Extract data ---
    X_all = adata.obsm[obsm_key]
    X_train = X_all[mask]

    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}"
            )
        y = adata.layers[layer][mask]
    else:
        y = adata.X[mask]

    if sparse.issparse(y):
        y = y.toarray()

    # Gene filtering
    if genes is not None:
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            raise ValueError(f"Genes not found: {missing[:10]}")
        gene_idx = [
            list(adata.var_names).index(g) for g in genes if g in set(adata.var_names)
        ]
        selected_genes = [adata.var_names[i] for i in gene_idx]
        y = y[:, gene_idx]
    else:
        selected_genes = adata.var_names.tolist()

    # Sample indices
    sample_indices = None
    if sample_col is not None:
        if sample_col not in adata.obs:
            raise ValueError(f"Column '{sample_col}' not found in adata.obs.")
        sample_indices = adata.obs[sample_col][mask].values
        n_samples = len(np.unique(sample_indices))
        logger.info(f"Using sample column '{sample_col}': {n_samples} sample(s)")

    # --- Fit or reuse ExpressionModel ---
    if pre_fitted_model is not None:
        expr_model = pre_fitted_model
        logger.info("Using pre-fitted ExpressionModel (skipping fit).")
    else:
        expr_model = ExpressionModel(
            n_landmarks=n_landmarks,
            use_empirical_variance=use_empirical_variance,
            eps=eps,
            random_state=random_state,
            batch_size=batch_size,
        )
        expr_model.fit(
            X_train,
            y,
            sigma=sigma,
            ls=ls,
            ls_factor=ls_factor,
            sample_indices=sample_indices,
            **function_kwargs,
        )

    # --- Predict on all cells ---
    # The model is trained on the condition cells but evaluated everywhere,
    # so every cell gets an imputed value and uncertainty estimate.
    imputed = np.asarray(
        expr_model.predict(X_all, batch_size=batch_size, progress=progress)
    )
    std_vals = np.asarray(
        expr_model.std(X_all, batch_size=batch_size, progress=progress)
    )
    # Broadcast (n, 1) to (n, n_genes) when only GP covariance is present
    n_genes_fitted = imputed.shape[1]
    if (
        isinstance(std_vals, np.ndarray)
        and std_vals.ndim == 2
        and std_vals.shape[1] == 1
        and n_genes_fitted > 1
    ):
        std_vals = np.broadcast_to(std_vals, (std_vals.shape[0], n_genes_fitted)).copy()

    # --- Store in adata ---
    n_all_genes = adata.n_vars
    n_cells = adata.n_obs

    def _expand_genes(values, n_cells, n_genes):
        """Place gene-subset values into a full (n_cells, n_genes) array."""
        out = np.full((n_cells, n_genes), np.nan)
        out[:, gene_idx] = values
        return out

    if inplace:
        if genes is not None:
            adata.layers[field_names["imputed_key"]] = _expand_genes(
                imputed, n_cells, n_all_genes
            )
            adata.layers[field_names["std_key"]] = _expand_genes(
                std_vals, n_cells, n_all_genes
            )
        else:
            adata.layers[field_names["imputed_key"]] = imputed
            adata.layers[field_names["std_key"]] = std_vals

        if use_empirical_variance:
            obs_var = np.asarray(
                expr_model.obs_variance(X_all, batch_size=batch_size, progress=progress)
            )
            if genes is not None:
                adata.layers[field_names["obs_variance_key"]] = _expand_genes(
                    obs_var, n_cells, n_all_genes
                )
            else:
                adata.layers[field_names["obs_variance_key"]] = obs_var

        if sample_col is not None and expr_model.has_sample_variance:
            sam_var = np.asarray(
                expr_model.sample_variance(
                    X_all, diag=True, batch_size=batch_size, progress=progress
                )
            )
            if isinstance(sam_var, np.ndarray):
                if genes is not None:
                    adata.layers[field_names["sample_variance_key"]] = _expand_genes(
                        sam_var, n_cells, n_all_genes
                    )
                else:
                    adata.layers[field_names["sample_variance_key"]] = sam_var

        # --- Metadata ---
        field_mapping = {
            field_names["imputed_key"]: {
                "location": "layers",
                "type": "imputed",
                "description": f"GP-imputed expression for {cond_label}",
            },
            field_names["std_key"]: {
                "location": "layers",
                "type": "std",
                "description": f"Total posterior standard deviation for {cond_label}",
            },
        }
        if use_empirical_variance:
            field_mapping[field_names["obs_variance_key"]] = {
                "location": "layers",
                "type": "obs_variance",
                "description": f"Per-gene empirical (aleatoric) variance for {cond_label}",
            }
        if sample_col is not None and expr_model.has_sample_variance:
            field_mapping[field_names["sample_variance_key"]] = {
                "location": "layers",
                "type": "sample_variance",
                "description": f"Biological-replicate variance for {cond_label}",
            }

        run_info = {
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis_type": "impute",
            "params": {
                "groupby": groupby,
                "condition": condition,
                "obsm_key": obsm_key,
                "layer": layer,
                "n_landmarks": n_landmarks,
                "sigma": sigma,
                "ls_factor": ls_factor,
                "use_empirical_variance": use_empirical_variance,
                "sample_col": sample_col,
                "n_genes": len(selected_genes),
                "n_cells": int(mask.sum()),
                "batch_size": batch_size,
                "result_key": result_key,
            },
            "field_names": field_names,
            "field_mapping": field_mapping,
            "environment": get_environment_info(),
        }

        if result_key not in adata.uns:
            adata.uns[result_key] = {}
        set_json_metadata(adata, f"{result_key}.last_run_info", run_info)
        append_to_run_history(adata, run_info, analysis_type="impute")

    # --- Return ---
    if copy:
        return adata
    if return_full_results:
        table = pd.DataFrame(index=selected_genes)
        table["mean_imputed"] = np.nanmean(imputed, axis=0)
        table["mean_std"] = np.nanmean(std_vals, axis=0)
        return {
            "model": expr_model,
            "table": table,
            "field_names": field_names,
        }
    return None


def compute_imputed_expression(
    adata,
    groupby: Optional[str] = None,
    condition: Optional[str] = None,
    obsm_key: str = "DM_EigenVectors",
    layer: Optional[str] = None,
    genes: Optional[List[str]] = None,
    n_landmarks: Optional[int] = 5000,
    sample_col: Optional[str] = None,
    sigma: float = 1.0,
    ls: Optional[float] = None,
    ls_factor: float = 10.0,
    use_empirical_variance: bool = True,
    eps: float = 1e-8,
    random_state: Optional[int] = None,
    batch_size: int = 500,
    result_key: str = "kompot_impute",
    return_full_results: bool = False,
    overwrite: Optional[bool] = None,
    progress: bool = True,
    **function_kwargs,
) -> Optional[Union[Dict[str, Any], Any]]:
    """Deprecated. Use :func:`kompot.impute_expression` with Settings dataclasses instead.

    .. deprecated::
        Use :func:`kompot.impute_expression` with Settings dataclasses instead.
    """
    warnings.warn(
        "compute_imputed_expression() is deprecated and will be removed "
        "in a future version. Use kompot.impute_expression() with "
        "Settings dataclasses instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return impute_expression(
        adata,
        groupby=groupby,
        condition=condition,
        obsm_key=obsm_key,
        layer=layer,
        genes=genes,
        sample_col=sample_col,
        gp=GPSettings(
            sigma=sigma,
            ls=ls,
            ls_factor=ls_factor,
            n_landmarks=n_landmarks,
            use_empirical_variance=use_empirical_variance,
            batch_size=batch_size,
            eps=eps,
            random_state=random_state,
        ),
        storage=StorageSettings(
            result_key=result_key,
            overwrite=overwrite,
        ),
        output=OutputSettings(
            return_full_results=return_full_results,
            progress=progress,
        ),
        **function_kwargs,
    )


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _impute_field_names(
    result_key: str,
    cond_label: str,
    use_empirical_variance: bool,
    has_sample_col: bool,
) -> Dict[str, Any]:
    """Generate standardised field names for imputation outputs."""
    field_names: Dict[str, Any] = {
        "imputed_key": f"{result_key}_{cond_label}_imputed",
        "std_key": f"{result_key}_{cond_label}_std",
    }

    layers = [field_names["imputed_key"], field_names["std_key"]]

    if use_empirical_variance:
        field_names["obs_variance_key"] = f"{result_key}_{cond_label}_obs_variance"
        layers.append(field_names["obs_variance_key"])

    if has_sample_col:
        field_names["sample_variance_key"] = (
            f"{result_key}_{cond_label}_sample_variance"
        )
        layers.append(field_names["sample_variance_key"])

    field_names["all_patterns"] = {"layers": layers}
    return field_names
