"""Expression imputation for AnnData objects."""

import logging
import datetime
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List
from scipy import sparse

try:
    import anndata
except ImportError:
    raise ImportError("Please install anndata: pip install anndata")

from ..differential.expression_model import ExpressionModel
from .utils import (
    _sanitize_name,
    generate_output_field_names,
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
    n_landmarks: Optional[int] = 5000,
    sample_col: Optional[str] = None,
    sigma: float = 1.0,
    ls: Optional[float] = None,
    ls_factor: float = 10.0,
    use_empirical_variance: bool = False,
    eps: float = 1e-8,
    random_state: Optional[int] = None,
    batch_size: int = 500,
    result_key: str = "kompot_impute",
    return_full_results: bool = False,
    overwrite: Optional[bool] = None,
    progress: bool = True,
    **function_kwargs,
) -> Optional[Union[Dict[str, Any], Any]]:
    """Impute gene expression for a single condition using GP smoothing.

    Fits an :class:`ExpressionModel` on the selected cells and stores the
    imputed values, posterior standard deviations, and (optionally) empirical
    and sample variance layers in ``adata``.

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
    n_landmarks : int, optional
        Number of Nystrom landmarks.
    sample_col : str, optional
        Column in ``adata.obs`` with biological-replicate labels.
    sigma, ls, ls_factor
        GP parameters forwarded to :meth:`ExpressionModel.fit`.
    use_empirical_variance : bool
        Estimate per-gene heteroscedastic noise.
    eps : float
        Numerical-stability constant.
    random_state : int, optional
        Seed for landmark selection.
    batch_size : int
        Cells per batch during prediction.
    result_key : str
        Base name for output fields.
    return_full_results : bool
        If *True*, return a dict with ``model``, ``table``, and
        ``field_names``.
    overwrite : bool or None
        *None*: warn on overwrite; *True*: silent; *False*: raise.
    progress : bool
        Show progress bars.
    **function_kwargs
        Forwarded to :class:`mellon.FunctionEstimator`.

    Returns
    -------
    None or dict
        *None* when results are stored in-place. If ``return_full_results``
        is *True*, a dictionary with keys ``"model"``, ``"table"``, and
        ``"field_names"``.
    """
    # --- Validate inputs ---
    if obsm_key not in adata.obsm:
        avail = list(adata.obsm.keys())
        msg = f"Key '{obsm_key}' not found in adata.obsm. Available: {avail}"
        if obsm_key == "DM_EigenVectors":
            msg += (
                "\n\nCompute diffusion maps with Palantir:\n"
                "  palantir.utils.run_diffusion_maps(adata)\n"
                "Or specify a different obsm_key such as 'X_pca'."
            )
        raise ValueError(msg)

    if condition is not None and groupby is None:
        raise ValueError("'groupby' is required when 'condition' is specified.")

    if groupby is not None and groupby not in adata.obs:
        raise ValueError(
            f"Column '{groupby}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

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
    field_names = _impute_field_names(result_key, cond_label,
                                      use_empirical_variance, sample_col is not None)

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
    X = adata.obsm[obsm_key][mask]

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
        gene_idx = [list(adata.var_names).index(g) for g in genes
                    if g in set(adata.var_names)]
        selected_genes = [adata.var_names[i] for i in gene_idx]
        y = y[:, gene_idx]
    else:
        selected_genes = adata.var_names.tolist()

    # Sample indices
    sample_indices = None
    if sample_col is not None:
        if sample_col not in adata.obs:
            raise ValueError(
                f"Column '{sample_col}' not found in adata.obs."
            )
        sample_indices = adata.obs[sample_col][mask].values
        n_samples = len(np.unique(sample_indices))
        logger.info(f"Using sample column '{sample_col}': {n_samples} sample(s)")

    # --- Fit ExpressionModel ---
    model = ExpressionModel(
        n_landmarks=n_landmarks,
        use_empirical_variance=use_empirical_variance,
        eps=eps,
        random_state=random_state,
        batch_size=batch_size,
    )
    model.fit(
        X, y,
        sigma=sigma, ls=ls, ls_factor=ls_factor,
        sample_indices=sample_indices,
        **function_kwargs,
    )

    # --- Predict ---
    imputed = model.predict(X, batch_size=batch_size, progress=progress)
    std_vals = model.std(X, batch_size=batch_size, progress=progress)
    # Broadcast (n, 1) to (n, n_genes) when only GP covariance is present
    n_genes_fitted = imputed.shape[1]
    if isinstance(std_vals, np.ndarray) and std_vals.ndim == 2 and std_vals.shape[1] == 1 and n_genes_fitted > 1:
        std_vals = np.broadcast_to(std_vals, (std_vals.shape[0], n_genes_fitted)).copy()

    # --- Store in adata ---
    n_all_genes = adata.n_vars
    n_cells = adata.n_obs

    def _expand_layer(values, mask, n_cells, n_genes):
        """Place values into a full (n_cells, n_genes) array."""
        out = np.full((n_cells, n_genes), np.nan)
        if genes is not None:
            out[np.ix_(np.where(mask)[0], gene_idx)] = values
        else:
            out[mask] = values
        return out

    adata.layers[field_names["imputed_key"]] = _expand_layer(
        imputed, mask, n_cells, n_all_genes
    )
    adata.layers[field_names["std_key"]] = _expand_layer(
        std_vals, mask, n_cells, n_all_genes
    )

    if use_empirical_variance:
        obs_var = model.obs_variance(X, batch_size=batch_size, progress=progress)
        adata.layers[field_names["obs_variance_key"]] = _expand_layer(
            obs_var, mask, n_cells, n_all_genes
        )

    if sample_col is not None and model.has_sample_variance:
        sam_var = model.sample_variance(X, diag=True, batch_size=batch_size, progress=progress)
        if isinstance(sam_var, np.ndarray):
            adata.layers[field_names["sample_variance_key"]] = _expand_layer(
                sam_var, mask, n_cells, n_all_genes
            )

    # --- Metadata ---
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
        },
        "environment": get_environment_info(),
    }

    if result_key not in adata.uns:
        adata.uns[result_key] = {}
    set_json_metadata(adata, f"{result_key}.last_run_info", run_info)
    append_to_run_history(adata, run_info, analysis_type="impute")

    # --- Return ---
    if return_full_results:
        table = pd.DataFrame(index=selected_genes)
        table["mean_imputed"] = np.nanmean(imputed, axis=0)
        table["mean_std"] = np.nanmean(std_vals, axis=0)
        return {
            "model": model,
            "table": table,
            "field_names": field_names,
        }
    return None


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
        field_names["sample_variance_key"] = f"{result_key}_{cond_label}_sample_variance"
        layers.append(field_names["sample_variance_key"])

    field_names["all_patterns"] = {"layers": layers}
    return field_names
