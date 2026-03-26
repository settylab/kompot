"""
Private helpers for compute_differential_abundance.

These functions decompose the monolithic compute_differential_abundance into
focused, testable pieces.
"""

import logging
import datetime
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger("kompot")


def _check_da_overwrites(
    adata,
    result_key: str,
    field_names: dict,
    sample_col: Optional[str],
    overwrite: Optional[bool],
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str,
    ls_factor: float,
):
    """Check for existing DA results and handle overwrite logic."""
    from .utils import detect_output_field_overwrite

    all_patterns = field_names["all_patterns"]
    column_patterns = all_patterns["obs"]

    has_overwrites, existing_fields, prev_run = detect_output_field_overwrite(
        adata=adata,
        result_key=result_key,
        output_patterns=column_patterns,
        with_sample_suffix=(sample_col is not None),
        sample_suffix="_sample_var" if sample_col is not None else "",
        result_type="differential abundance",
        analysis_type="da",
    )

    if not has_overwrites:
        return

    message = (
        f"Results with result_key='{result_key}' already exist in the dataset."
    )
    current_sample_var = sample_col is not None

    if prev_run:
        prev_timestamp = prev_run.get("timestamp", "unknown time")
        prev_params = prev_run.get("params", {})
        from .utils.params import params_get as _pg
        prev_conditions = (
            f"{_pg(prev_params, 'condition1', 'unknown')} to "
            f"{_pg(prev_params, 'condition2', 'unknown')}"
        )
        message += (
            f" Previous run was at {prev_timestamp} comparing {prev_conditions}."
        )

        if existing_fields:
            field_list = ", ".join(existing_fields[:5])
            if len(existing_fields) > 5:
                field_list += f" and {len(existing_fields) - 5} more fields"

            prev_sample_var = _pg(prev_params, "use_sample_variance", False)
            if prev_sample_var != current_sample_var:
                if current_sample_var:
                    message += (
                        f" Fields that will be overwritten: {field_list}. "
                        f"Note: Only fields NOT affected by sample variance "
                        f"(like lfc, log_density) will be overwritten since "
                        f"they don't use the sample variance suffix. These "
                        f"results will likely be identical if other parameters "
                        f"haven't changed."
                    )
                else:
                    message += (
                        f" Fields that will be overwritten: {field_list}. "
                        f"Note: Only fields NOT affected by sample variance "
                        f"will be overwritten since sample variance-specific "
                        f"fields use a different suffix."
                    )
            else:
                message += f" Fields that will be overwritten: {field_list}"

    if overwrite is False:
        message += " Set overwrite=True to overwrite or use a different result_key."
        raise ValueError(message)
    elif overwrite is None:
        params_match = False
        if prev_run:
            prev_params = prev_run.get("params", {})
            prev_sample_var = _pg(prev_params, "use_sample_variance", False)
            if current_sample_var and not prev_sample_var:
                params_match = True
                for param_name in [
                    "groupby", "condition1", "condition2", "obsm_key", "ls_factor",
                ]:
                    curr_val = locals().get(param_name)
                    prev_val = _pg(prev_params, param_name)
                    if curr_val != prev_val:
                        params_match = False
                        logger.debug(
                            f"Parameter mismatch: {param_name} "
                            f"(current: {curr_val}, previous: {prev_val})"
                        )

        if (
            prev_run
            and params_match
            and current_sample_var
            and not prev_sample_var
        ):
            logger.info(
                message
                + " This is a partial rerun with sample variance added to a "
                "previous analysis with matching parameters. "
                "Set overwrite=False to prevent overwriting or overwrite=True "
                "to silence this message."
            )
        else:
            logger.warning(
                message
                + " Set overwrite=False to prevent overwriting or overwrite=True "
                "to silence this message."
            )


def _extract_da_data(
    adata,
    groupby: str,
    condition1: str,
    condition2: str,
    obsm_key: str,
    sample_col: Optional[str],
) -> dict:
    """Validate inputs and extract arrays for DA from adata."""
    if obsm_key not in adata.obsm:
        error_msg = (
            f"Key '{obsm_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
        if obsm_key == "DM_EigenVectors":
            error_msg += (
                "\n\nTo compute DM_EigenVectors (diffusion map eigenvectors), "
                "use the Palantir package:\n"
                "```python\n"
                "import palantir\n"
                "palantir.utils.run_diffusion_maps(adata)\n"
                "```\n"
                "See https://github.com/dpeerlab/Palantir for installation.\n\n"
                "Alternatively, specify a different obsm_key that exists in "
                "your dataset, such as 'X_pca'."
            )
        raise ValueError(error_msg)

    if groupby not in adata.obs:
        raise ValueError(
            f"Column '{groupby}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    mask1 = adata.obs[groupby] == condition1
    mask2 = adata.obs[groupby] == condition2

    if np.sum(mask1) == 0:
        raise ValueError(f"Condition '{condition1}' not found in '{groupby}'.")
    if np.sum(mask2) == 0:
        raise ValueError(f"Condition '{condition2}' not found in '{groupby}'.")

    logger.info(f"Condition 1 ({condition1}): {np.sum(mask1):,} cells")
    logger.info(f"Condition 2 ({condition2}): {np.sum(mask2):,} cells")

    X1 = adata.obsm[obsm_key][mask1]
    X2 = adata.obsm[obsm_key][mask2]

    cond1_samples = None
    cond2_samples = None
    if sample_col is not None:
        if sample_col not in adata.obs:
            raise ValueError(
                f"Column '{sample_col}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        cond1_samples = adata.obs[sample_col][mask1].values
        cond2_samples = adata.obs[sample_col][mask2].values
        logger.info(
            f"Using sample column '{sample_col}' for sample variance estimation"
        )
        logger.info(
            f"Found {len(np.unique(cond1_samples))} unique sample(s) "
            f"in condition 1"
        )
        logger.info(
            f"Found {len(np.unique(cond2_samples))} unique sample(s) "
            f"in condition 2"
        )

    return {
        "X1": X1,
        "X2": X2,
        "condition1_sample_indices": cond1_samples,
        "condition2_sample_indices": cond2_samples,
    }


def _resolve_da_landmarks(
    adata,
    landmarks: Optional[np.ndarray],
    obsm_key: str,
    result_key: str,
) -> Optional[np.ndarray]:
    """Resolve landmarks for DA from provided array or adata.uns."""
    if landmarks is not None:
        logger.info(f"Using provided landmarks with shape {landmarks.shape}")
        return landmarks

    data_dim = adata.obsm[obsm_key].shape[1]

    # Check result_key
    if result_key in adata.uns and "landmarks" in adata.uns[result_key]:
        stored = adata.uns[result_key]["landmarks"]
        if stored.shape[1] == data_dim:
            logger.info(
                f"Using stored landmarks from adata.uns['{result_key}']"
                f"['landmarks'] with shape {stored.shape}"
            )
            return stored
        else:
            logger.warning(
                f"Stored landmarks have dimension {stored.shape[1]} but data "
                f"has dimension {data_dim}. Will check for other landmarks."
            )

    # Check default DA key
    storage_key = "kompot_da"
    if (
        storage_key in adata.uns
        and storage_key != result_key
        and "landmarks" in adata.uns[storage_key]
    ):
        other = adata.uns[storage_key]["landmarks"]
        if other.shape[1] == data_dim:
            logger.info(
                f"Reusing stored landmarks from adata.uns['{storage_key}']"
                f"['landmarks'] with shape {other.shape}"
            )
            return other
        else:
            logger.warning(
                f"Other stored landmarks have dimension {other.shape[1]} but "
                f"data has dimension {data_dim}. Will check for other landmarks."
            )

    # Check all kompot_* keys
    for key in adata.uns.keys():
        if (
            key.startswith("kompot_")
            and key != result_key
            and key != storage_key
            and "landmarks" in adata.uns[key]
        ):
            check = adata.uns[key]["landmarks"]
            if check.shape[1] == data_dim:
                logger.info(
                    f"Reusing landmarks from adata.uns['{key}']['landmarks'] "
                    f"with shape {check.shape}"
                )
                return check
            else:
                logger.debug(
                    f"Found landmarks in {key} but dimensions don't match: "
                    f"{check.shape[1]} vs {data_dim}"
                )

    # Check DE landmarks
    if "kompot_de" in adata.uns and "landmarks" in adata.uns["kompot_de"]:
        de = adata.uns["kompot_de"]["landmarks"]
        if de.shape[1] == data_dim:
            logger.info(
                f"Reusing differential expression landmarks from "
                f"adata.uns['kompot_de']['landmarks'] with shape {de.shape}"
            )
            return de
        else:
            logger.warning(
                f"DE landmarks have dimension {de.shape[1]} but data has "
                f"dimension {data_dim}. Computing new landmarks."
            )

    return None


def _store_da_landmarks(adata, diff_abundance, result_key, store_landmarks):
    """Store computed landmarks in adata.uns if requested."""
    if not (
        hasattr(diff_abundance, "computed_landmarks")
        and diff_abundance.computed_landmarks is not None
    ):
        logger.debug(
            "No computed landmarks found to store."
        )
        return

    if result_key not in adata.uns:
        adata.uns[result_key] = {}

    lm = diff_abundance.computed_landmarks
    landmarks_info = {
        "shape": list(lm.shape),
        "dtype": str(lm.dtype),
        "source": "computed",
        "n_landmarks": lm.shape[0],
        "timestamp": datetime.datetime.now().isoformat(),
    }

    adata.uns[result_key]["landmarks_info"] = landmarks_info

    storage_key = "kompot_da"
    if storage_key not in adata.uns:
        adata.uns[storage_key] = {}
    adata.uns[storage_key]["landmarks_info"] = landmarks_info.copy()

    if store_landmarks:
        adata.uns[result_key]["landmarks"] = lm
        logger.info(
            f"Stored landmarks in adata.uns['{result_key}']['landmarks'] "
            f"with shape {lm.shape} for future reuse"
        )
    else:
        logger.debug(
            "Landmark storage skipped (store_landmarks=False)."
        )


def _store_da_results(
    adata,
    abundance_results: dict,
    field_names: dict,
    condition1: str,
    condition2: str,
):
    """Store DA results in adata.obs and adata.uns."""
    from ..utils import KOMPOT_COLORS

    adata.obs[field_names["lfc_key"]] = abundance_results["log_fold_change"]
    adata.obs[field_names["zscore_key"]] = abundance_results[
        "log_fold_change_zscore"
    ]
    adata.obs[field_names["ptp_key"]] = abundance_results[
        "neg_log10_fold_change_ptp"
    ]

    direction_col = field_names["direction_key"]
    adata.obs[direction_col] = abundance_results["log_fold_change_direction"]
    adata.obs[direction_col] = pd.Categorical(
        adata.obs[direction_col], categories=["up", "neutral", "down"]
    )

    direction_colors = KOMPOT_COLORS["direction"]
    categories = adata.obs[direction_col].cat.categories
    color_list = [direction_colors.get(cat, "#d3d3d3") for cat in categories]
    adata.uns[f"{direction_col}_colors"] = color_list

    adata.obs[field_names["density_key_1"]] = abundance_results[
        "log_density_condition1"
    ]
    adata.obs[field_names["density_key_2"]] = abundance_results[
        "log_density_condition2"
    ]


def _record_da_run_info(
    adata,
    field_names: dict,
    condition1: str,
    condition2: str,
    sample_col: Optional[str],
    result_key: str,
    params_dict: dict,
):
    """Record run info, field tracking, and run history for DA."""
    from .utils import (
        get_environment_info,
        set_json_metadata,
        append_to_run_history,
        get_run_history,
        get_json_metadata,
    )

    current_timestamp = datetime.datetime.now().isoformat()
    storage_key = "kompot_da"

    current_run_info = {
        "timestamp": current_timestamp,
        "function": "compute_differential_abundance",
        "result_key": result_key,
        "analysis_type": "da",
        "lfc_key": field_names["lfc_key"],
        "zscore_key": field_names["zscore_key"],
        "ptp_key": field_names["ptp_key"],
        "direction_key": field_names["direction_key"],
        "density_keys": {
            "condition1": field_names["density_key_1"],
            "condition2": field_names["density_key_2"],
        },
        "field_names": field_names,
        "uses_sample_variance": sample_col is not None,
        "has_groups": False,
        "params": params_dict,
        "environment": get_environment_info(),
    }

    # Field mapping
    field_mapping = {
        field_names["lfc_key"]: {
            "location": "obs",
            "type": "log_fold_change",
            "description": "Log fold change values",
        },
        field_names["zscore_key"]: {
            "location": "obs",
            "type": "log_fold_change_zscore",
            "description": "Z-scores for fold changes",
        },
        field_names["ptp_key"]: {
            "location": "obs",
            "type": "ptp",
            "description": "Negative log10 PTPs (Posterior Tail Probabilities)",
        },
        field_names["direction_key"]: {
            "location": "obs",
            "type": "direction",
            "description": "Direction classification (up/down/neutral)",
        },
        field_names["density_key_1"]: {
            "location": "obs",
            "type": "density",
            "description": f"Log density for {condition1}",
        },
        field_names["density_key_2"]: {
            "location": "obs",
            "type": "density",
            "description": f"Log density for {condition2}",
        },
        f"{field_names['direction_key']}_colors": {
            "location": "uns",
            "type": "colors",
            "description": "Color mapping for direction categories",
        },
    }
    current_run_info["field_mapping"] = field_mapping

    # Initialize storage
    if storage_key not in adata.uns:
        adata.uns[storage_key] = {}

    # Run history
    if "run_history" not in adata.uns.get(storage_key, {}):
        adata.uns[storage_key]["run_history"] = "[]"
        new_run_id = 0
    else:
        current_history = get_run_history(adata, "da")
        new_run_id = len(current_history)

    logger.info(f"This run will have `run_id={new_run_id}`.")

    append_to_run_history(adata, current_run_info, "da")
    set_json_metadata(adata, f"{storage_key}.last_run_info", current_run_info)

    # Field tracking
    obs_fields = [
        field_names["lfc_key"],
        field_names["zscore_key"],
        field_names["ptp_key"],
        field_names["direction_key"],
        field_names["density_key_1"],
        field_names["density_key_2"],
    ]

    anndata_field_tracking = {"obs": {}, "uns": {}}
    for field in obs_fields:
        anndata_field_tracking["obs"][field] = new_run_id
    anndata_field_tracking["uns"][result_key] = new_run_id
    anndata_field_tracking["uns"][
        f"{field_names['direction_key']}_colors"
    ] = new_run_id

    if "anndata_fields" not in adata.uns[storage_key]:
        set_json_metadata(
            adata, f"{storage_key}.anndata_fields", anndata_field_tracking
        )
    else:
        existing = get_json_metadata(adata, f"{storage_key}.anndata_fields")
        if existing is None:
            existing = {}
        for section, fields in anndata_field_tracking.items():
            if section not in existing:
                existing[section] = {}
            for field, rid in fields.items():
                existing[section][field] = rid
        set_json_metadata(adata, f"{storage_key}.anndata_fields", existing)
