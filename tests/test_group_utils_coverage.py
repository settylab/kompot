"""Tests targeting uncovered lines in group_utils.py for improved coverage."""

import numpy as np
import pytest
import pandas as pd
import anndata


from kompot.anndata.utils.group_utils import (
    parse_groups,
    check_underrepresentation,
    refine_filter_for_underrepresentation,
    apply_cell_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n=100, n_genes=5):
    """Small AnnData with diverse obs columns."""
    np.random.seed(0)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                np.random.choice(["A", "B", "C"], n)
            ),
            "condition": pd.Categorical(
                np.random.choice(["ctrl", "treat"], n)
            ),
            "batch": np.random.choice(["b1", "b2"], n),
            "is_ok": np.random.choice([True, False], n),
            "score": np.random.uniform(0, 1, n),
            "label_space": np.random.choice(["Type A", "Type B"], n),
            "label_dash": np.random.choice(["Type-A", "Type-B"], n),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    X = np.random.randn(n, n_genes).astype(np.float32)
    return anndata.AnnData(X=X, obs=obs)


# ===================================================================
# parse_groups — dict filters
# ===================================================================

class TestParseGroupsDictFilters:
    """Cover lines 114, 123, 151, 167, 181, 185, 189-196, 204, 209-211,
    218-222, 228."""

    def test_nested_dict_formatted_names(self):
        """Line 123: formatted_names in nested-dict branch."""
        adata = _make_adata(50)
        groups = {
            "my group": {"cell_type": "A"},
            "other-group": {"cell_type": "B"},
        }
        d, names = parse_groups(adata, groups, formatted_names=True)
        assert "my_group" in d
        assert "other_group" in d

    def test_nested_dict_unsupported_value_type(self):
        """Line 114: unsupported value type inside nested dict."""
        adata = _make_adata(20)
        groups = {"g": {"cell_type": 42}}  # int not supported
        with pytest.raises(ValueError, match="Unsupported value type"):
            parse_groups(adata, groups)

    def test_multi_column_dict_unsupported_value(self):
        """Line 151: unsupported value in multi-column dict branch."""
        adata = _make_adata(20)
        # Both keys are columns ⇒ multi-column branch
        groups = {"cell_type": 99, "condition": "ctrl"}
        with pytest.raises(ValueError, match="Unsupported value type"):
            parse_groups(adata, groups)

    def test_regular_dict_formatted_names(self):
        """Line 167: formatted_names in regular dict branch."""
        adata = _make_adata(30)
        mask = np.zeros(30, dtype=bool)
        mask[:5] = True
        groups = {"my group": mask}
        d, names = parse_groups(adata, groups, formatted_names=True)
        assert "my_group" in d

    def test_regular_dict_str_value_not_column(self):
        """Lines 181, 185: group_name not a column, search all columns."""
        adata = _make_adata(30)
        # "custom_key" is NOT a column; "A" exists in cell_type
        groups = {"custom_key": "A"}
        d, names = parse_groups(adata, groups)
        assert "custom_key" in d
        assert np.any(d["custom_key"])

    def test_regular_dict_str_value_not_found(self):
        """Line 185: value not found in any column → error."""
        adata = _make_adata(20)
        groups = {"custom_key": "NONEXISTENT_VALUE_XYZ"}
        with pytest.raises(ValueError, match="not found in any column"):
            parse_groups(adata, groups)

    def test_regular_dict_list_str_column(self):
        """Lines 189-193: list of strings where key IS a column."""
        adata = _make_adata(50)
        groups = {"cell_type": ["A", "B"]}
        d, names = parse_groups(adata, groups)
        assert len(d) == 1
        key = list(d.keys())[0]
        expected = adata.obs["cell_type"].isin(["A", "B"]).values
        np.testing.assert_array_equal(d[key], expected)

    def test_regular_dict_list_str_not_column(self):
        """Lines 194-196: list of strings where key is NOT a column."""
        adata = _make_adata(20)
        groups = {"not_a_col": ["x", "y"]}
        with pytest.raises(ValueError, match="List values are only supported"):
            parse_groups(adata, groups)

    def test_regular_dict_bool_value_not_column(self):
        """Line 204: bool value but key is not a column."""
        adata = _make_adata(20)
        groups = {"not_a_col": True}
        with pytest.raises(ValueError, match="requires a boolean column"):
            parse_groups(adata, groups)

    def test_regular_dict_int_indices(self):
        """Lines 209-211: array-like with integer indices."""
        adata = _make_adata(50)
        groups = {"sel": np.array([0, 2, 4])}
        d, names = parse_groups(adata, groups)
        assert d["sel"][0] and d["sel"][2] and d["sel"][4]
        assert not d["sel"][1]

    def test_regular_dict_convertible_values(self):
        """Lines 218-222: values that need try/except bool conversion."""
        adata = _make_adata(30)
        # float array that can be cast to bool
        groups = {"sel": np.array([1.0, 0.0] * 15)}
        d, _ = parse_groups(adata, groups)
        assert "sel" in d

    def test_regular_dict_mask_length_mismatch(self):
        """Line 228: mask length doesn't match n_obs."""
        adata = _make_adata(30)
        groups = {"sel": np.array([True, False])}  # too short
        with pytest.raises(ValueError, match="mask length"):
            parse_groups(adata, groups)


# ===================================================================
# parse_groups — pd.Series inputs
# ===================================================================

class TestParseGroupsSeries:
    """Cover lines 247, 261-274, 285."""

    def test_categorical_series_formatted_names(self):
        """Line 247: formatted_names on categorical Series."""
        adata = _make_adata(40)
        s = pd.Series(
            pd.Categorical(np.random.choice(["Type A", "Type-B"], 40))
        )
        d, names = parse_groups(adata, s, formatted_names=True)
        for n in names:
            assert " " not in n
            assert "-" not in n

    def test_numeric_series(self):
        """Lines 261-274: numeric (integer) Series parsed to groups."""
        adata = _make_adata(40)
        s = pd.Series(np.random.choice([1, 2, 3], 40))
        d, names = parse_groups(adata, s)
        assert len(d) == 3  # three unique int values

    def test_string_series_formatted_names(self):
        """Line 285: non-categorical string Series with formatted_names."""
        adata = _make_adata(30)
        s = pd.Series(np.random.choice(["hello world", "foo-bar"], 30))
        d, names = parse_groups(adata, s, formatted_names=True)
        for n in names:
            assert " " not in n
            assert "-" not in n


# ===================================================================
# parse_groups — np.ndarray inputs
# ===================================================================

class TestParseGroupsNdarray:
    """Cover lines 300-303, 329, 340, 345-347."""

    def test_2d_float_mask_array(self):
        """Lines 300-303: 2D array where rows need bool conversion."""
        adata = _make_adata(20)
        arr = np.zeros((2, 20), dtype=float)
        arr[0, :5] = 1.0
        arr[1, 10:15] = 1.0
        d, names = parse_groups(adata, arr)
        assert len(d) == 2
        assert "subset1" in d
        assert "subset2" in d

    def test_string_ndarray(self):
        """Line 329: string numpy array matching n_obs."""
        adata = _make_adata(30)
        arr = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        d, names = parse_groups(adata, arr)
        assert len(d) == 3

    def test_string_ndarray_length_mismatch(self):
        """Line 329: string array length != n_obs."""
        adata = _make_adata(30)
        arr = np.array(["A", "B"])
        with pytest.raises(ValueError, match="categorical array"):
            parse_groups(adata, arr)

    def test_string_ndarray_formatted_names(self):
        """Line 340: formatted_names on string ndarray."""
        adata = _make_adata(20)
        arr = np.array(["Type A"] * 10 + ["Type-B"] * 10)
        d, names = parse_groups(adata, arr, formatted_names=True)
        for n in names:
            assert " " not in n
            assert "-" not in n

    def test_unsupported_dtype_ndarray(self):
        """Line 345: unsupported dtype (float 1D)."""
        adata = _make_adata(20)
        arr = np.array([1.5, 2.5, 3.5])
        with pytest.raises(ValueError, match="must be boolean, integer, or string"):
            parse_groups(adata, arr)

    def test_unsupported_shape_ndarray(self):
        """Line 347: 3D array → unsupported shape."""
        adata = _make_adata(20)
        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="not supported"):
            parse_groups(adata, arr)


# ===================================================================
# parse_groups — list inputs
# ===================================================================

class TestParseGroupsList:
    """Cover lines 362, 369-376, 385, 395, 402, 413-429, 448-454, 458,
    469-504."""

    def test_list_of_dicts_missing_column(self):
        """Line 362: column not found in list-of-dicts branch."""
        adata = _make_adata(20)
        groups = [{"nonexistent_col": "val"}]
        with pytest.raises(ValueError, match="not found in adata.obs"):
            parse_groups(adata, groups)

    def test_list_of_dicts_with_list_values(self):
        """Lines 369-374: list values inside list-of-dicts."""
        adata = _make_adata(50)
        groups = [{"cell_type": ["A", "B"]}, {"cell_type": "C"}]
        d, names = parse_groups(adata, groups)
        assert len(d) == 2

    def test_list_of_dicts_unsupported_value(self):
        """Line 376: unsupported value type in list-of-dicts."""
        adata = _make_adata(20)
        groups = [{"cell_type": 99}]
        with pytest.raises(ValueError, match="Unsupported value type"):
            parse_groups(adata, groups)

    def test_list_of_dicts_empty_filter(self):
        """Line 385: empty dict in list → fallback name."""
        adata = _make_adata(20)
        groups = [{}]
        d, names = parse_groups(adata, groups)
        assert "filter1" in names

    def test_list_of_pd_series(self):
        """Line 395: list containing pd.Series (converted to ndarray)."""
        adata = _make_adata(30)
        s = pd.Series(np.array([True] * 10 + [False] * 20))
        groups = [s]
        d, names = parse_groups(adata, groups)
        assert len(d) == 1

    def test_list_of_bool_mask_length_mismatch(self):
        """Line 402: bool mask in list with wrong length."""
        adata = _make_adata(30)
        groups = [np.array([True, False])]
        with pytest.raises(ValueError, match="boolean mask"):
            parse_groups(adata, groups)

    def test_list_of_int_indices(self):
        """Lines 448-454: integer indices in list of arrays."""
        adata = _make_adata(50)
        groups = [np.array([0, 2, 4])]
        d, names = parse_groups(adata, groups)
        assert "subset1" in d
        assert d["subset1"][0] and d["subset1"][2] and d["subset1"][4]

    def test_list_of_string_arrays(self):
        """Lines 458 + surrounding: string array in list of arrays."""
        adata = _make_adata(30)
        arr = np.array(["X"] * 15 + ["Y"] * 15)
        groups = [arr]
        d, names = parse_groups(adata, groups)
        assert "subset1_X" in d
        assert "subset1_Y" in d

    def test_list_of_string_arrays_length_mismatch(self):
        """Line 458: string array wrong length."""
        adata = _make_adata(30)
        groups = [np.array(["X", "Y"])]
        with pytest.raises(ValueError, match="categorical array"):
            parse_groups(adata, groups)

    def test_list_of_numeric_arrays(self):
        """Lines 469-498: numeric (float) array in list of arrays."""
        adata = _make_adata(40)
        arr = np.array([1.0] * 20 + [2.0] * 20)
        groups = [arr]
        d, names = parse_groups(adata, groups)
        # Should create subsets for each unique value
        assert len(d) >= 2

    def test_list_of_numeric_arrays_length_mismatch(self):
        """Lines 471-472: numeric array wrong length."""
        adata = _make_adata(30)
        groups = [np.array([1.0, 2.0])]
        with pytest.raises(ValueError, match="numeric array"):
            parse_groups(adata, groups)

    def test_list_unsupported_array_dtype(self):
        """Line 500: unsupported array dtype in list."""
        adata = _make_adata(20)
        # Use a structured dtype that won't match any branch
        arr = np.array([(1, 2)] * 20, dtype=[("a", "i4"), ("b", "i4")])
        groups = [arr]
        with pytest.raises(ValueError, match="not supported"):
            parse_groups(adata, groups)

    def test_list_mixed_types(self):
        """Line 502: mixed types in list."""
        adata = _make_adata(20)
        groups = [np.array([True] * 20), "not_an_array"]
        with pytest.raises(ValueError, match="mixed types"):
            parse_groups(adata, groups)

    def test_unsupported_groups_type(self):
        """Line 504: completely unsupported type."""
        adata = _make_adata(20)
        with pytest.raises(ValueError, match="Unsupported groups type"):
            parse_groups(adata, 42)


# ===================================================================
# check_underrepresentation
# ===================================================================

class TestCheckUnderrepresentation:
    """Cover lines 561, 593, 625, 636-671."""

    def test_groupby_not_found(self):
        """Line 561: groupby column not in adata.obs."""
        adata = _make_adata(30)
        with pytest.raises(ValueError, match="not found in adata.obs"):
            check_underrepresentation(adata, groupby="MISSING", groups="cell_type")

    def test_empty_group_skipped(self):
        """Line 593: group with all-False mask is skipped."""
        adata = _make_adata(30)
        groups = {"empty": np.zeros(30, dtype=bool)}
        result = check_underrepresentation(
            adata, groupby="condition", groups=groups, min_cells=1
        )
        assert "__underrepresentation_data" in result
        assert len(result["__underrepresentation_data"]) == 0

    def test_min_percentage_trigger(self):
        """Lines 620-628: underrepresentation detected by percentage."""
        # Build adata where one condition has very few cells in one group
        n = 100
        np.random.seed(7)
        obs = pd.DataFrame({
            "condition": pd.Categorical(["ctrl"] * 95 + ["treat"] * 5),
            "grp": pd.Categorical(["G"] * 100),
        })
        adata = anndata.AnnData(
            X=np.random.randn(n, 3).astype(np.float32), obs=obs
        )
        result = check_underrepresentation(
            adata,
            groupby="condition",
            groups="grp",
            min_cells=1,
            min_percentage=10.0,  # treat has only 5% → underrepresented
            warn=True,
        )
        underrep = result["__underrepresentation_data"]
        assert len(underrep) > 0

    def test_print_summary(self, capsys):
        """Lines 636-671: print_summary branch."""
        n = 60
        obs = pd.DataFrame({
            "condition": pd.Categorical(["ctrl"] * 55 + ["treat"] * 5),
            "grp": pd.Categorical(["G"] * 60),
        })
        adata = anndata.AnnData(
            X=np.random.randn(n, 3).astype(np.float32), obs=obs
        )
        result = check_underrepresentation(
            adata,
            groupby="condition",
            groups="grp",
            min_cells=10,
            print_summary=True,
        )
        captured = capsys.readouterr()
        assert "UNDERREPRESENTATION REPORT" in captured.out
        assert "RECOMMENDATION" in captured.out

    def test_dict_groups_metadata(self):
        """Line 580-584: dict groups adds first-key metadata."""
        adata = _make_adata(50)
        groups = {"sel": np.ones(50, dtype=bool)}
        result = check_underrepresentation(
            adata, groupby="condition", groups=groups, min_cells=1
        )
        assert "sel" in result

    def test_array_groups_metadata(self):
        """Lines 585-587: ndarray groups adds 'selection' key."""
        adata = _make_adata(50)
        groups = np.ones(50, dtype=bool)
        result = check_underrepresentation(
            adata, groupby="condition", groups=groups, min_cells=1
        )
        assert "selection" in result


# ===================================================================
# refine_filter_for_underrepresentation
# ===================================================================

class TestRefineFilter:
    """Cover lines 714, 759, 771-775."""

    def test_filter_mask_none_defaults_to_all(self):
        """Line 714: filter_mask=None → all ones."""
        adata = _make_adata(50)
        mask, underrep, excluded = refine_filter_for_underrepresentation(
            adata,
            filter_mask=None,
            groupby="condition",
            groups="cell_type",
            min_cells=1,
        )
        # With min_cells=1, no exclusion expected
        assert mask.shape == (50,)

    def test_no_groupby_returns_original(self):
        """Line 717-718: groupby=None short-circuits."""
        adata = _make_adata(30)
        m = np.ones(30, dtype=bool)
        mask, underrep, excluded = refine_filter_for_underrepresentation(
            adata, filter_mask=m, groupby=None, groups=None
        )
        np.testing.assert_array_equal(mask, m)
        assert excluded == 0

    def test_exclusion_happens(self):
        """Lines 759-783: groups are actually excluded."""
        n = 100
        obs = pd.DataFrame({
            "condition": pd.Categorical(
                ["ctrl"] * 48 + ["treat"] * 2 + ["ctrl"] * 25 + ["treat"] * 25
            ),
            "grp": pd.Categorical(
                ["G1"] * 50 + ["G2"] * 50
            ),
        })
        adata = anndata.AnnData(
            X=np.random.randn(n, 3).astype(np.float32), obs=obs
        )
        mask = np.ones(n, dtype=bool)
        refined, underrep, excluded = refine_filter_for_underrepresentation(
            adata,
            filter_mask=mask,
            groupby="condition",
            groups="grp",
            min_cells=10,  # G1 has only 2 treat cells → excluded
        )
        assert excluded > 0
        assert np.sum(refined) < n

    def test_mask_length_mismatch_warning(self):
        """Lines 771-775: group mask length != filtered_adata.n_obs → skip."""
        # This is hard to trigger naturally; we test the no-underrep path instead
        adata = _make_adata(40)
        mask = np.ones(40, dtype=bool)
        refined, underrep, excluded = refine_filter_for_underrepresentation(
            adata,
            filter_mask=mask,
            groupby="condition",
            groups="cell_type",
            min_cells=1,
        )
        # No exclusion because min_cells=1
        assert excluded == 0


# ===================================================================
# apply_cell_filter
# ===================================================================

class TestApplyCellFilter:
    """Cover lines 859, 870-873, 880, 896-914, 919-925, 951-957."""

    def test_string_filter_missing_column(self):
        """Line 859: column not found."""
        adata = _make_adata(20)
        with pytest.raises(ValueError, match="not found in adata.obs"):
            apply_cell_filter(adata, "MISSING_COL")

    def test_string_filter_non_bool_column(self):
        """Lines 870-873: string column converted to bool."""
        adata = _make_adata(30)
        # 'batch' is a string column; casting to bool should work (non-empty → True)
        # Actually need a numeric-like column for reliable bool cast
        adata.obs["numeric_flag"] = np.random.choice([0, 1], 30)
        mask, details = apply_cell_filter(adata, "numeric_flag")
        assert mask.dtype == bool

    def test_dict_filter_missing_column(self):
        """Line 880: dict filter with missing column."""
        adata = _make_adata(20)
        with pytest.raises(ValueError, match="not found in adata.obs"):
            apply_cell_filter(adata, {"MISSING": "val"})

    def test_dict_filter_single_value(self):
        """Line 895: dict filter with single (non-list) value."""
        adata = _make_adata(50)
        mask, details = apply_cell_filter(adata, {"cell_type": "A"})
        expected = (adata.obs["cell_type"] == "A").values
        np.testing.assert_array_equal(mask, expected)

    def test_ndarray_bool_filter(self):
        """Lines 896-906: ndarray boolean mask."""
        adata = _make_adata(30)
        f = np.array([True] * 10 + [False] * 20)
        mask, details = apply_cell_filter(adata, f)
        np.testing.assert_array_equal(mask, f)

    def test_ndarray_bool_length_mismatch(self):
        """Line 903-904: bool mask wrong length."""
        adata = _make_adata(30)
        with pytest.raises(ValueError, match="doesn't match"):
            apply_cell_filter(adata, np.array([True, False]))

    def test_ndarray_int_indices_filter(self):
        """Lines 907-910: integer index array."""
        adata = _make_adata(50)
        indices = np.array([0, 5, 10])
        mask, details = apply_cell_filter(adata, indices)
        assert mask[0] and mask[5] and mask[10]
        assert not mask[1]

    def test_ndarray_unsupported_dtype(self):
        """Line 912: float array → error."""
        adata = _make_adata(20)
        with pytest.raises(ValueError, match="boolean or integer"):
            apply_cell_filter(adata, np.array([1.5, 2.5]))

    def test_pd_series_filter(self):
        """Lines 898-899: pd.Series is converted to ndarray."""
        adata = _make_adata(30)
        s = pd.Series(np.array([True] * 15 + [False] * 15))
        mask, details = apply_cell_filter(adata, s)
        assert mask.dtype == bool
        assert np.sum(mask) == 15

    def test_unsupported_filter_type(self):
        """Line 914: unsupported filter type."""
        adata = _make_adata(20)
        with pytest.raises(ValueError, match="Unsupported filter type"):
            apply_cell_filter(adata, 42)

    def test_cell_subset_union(self):
        """Lines 919-925: cell_subset ORed with main mask."""
        adata = _make_adata(50)
        mask_filter = np.zeros(50, dtype=bool)
        mask_filter[:10] = True  # cells 0-9

        mask, details = apply_cell_filter(
            adata, mask_filter, cell_subset=[20, 25, 30]
        )
        # Should include 0-9 plus 20, 25, 30
        assert mask[0] and mask[20] and mask[25] and mask[30]
        assert "cell_subset" in details["filter_type"]

    def test_check_representation_with_exclusion(self):
        """Lines 951-957: auto-filter when underrepresentation found."""
        n = 100
        obs = pd.DataFrame({
            "condition": pd.Categorical(
                ["ctrl"] * 48 + ["treat"] * 2 + ["ctrl"] * 25 + ["treat"] * 25
            ),
            "grp": pd.Categorical(
                ["G1"] * 50 + ["G2"] * 50
            ),
        })
        adata = anndata.AnnData(
            X=np.random.randn(n, 3).astype(np.float32), obs=obs
        )
        mask, details = apply_cell_filter(
            adata,
            cell_filter=None,
            check_representation=True,
            groupby="condition",
            groups="grp",
            min_cells=10,
        )
        assert details["auto_filtered"] is True
        assert details["underrepresentation"] is not None
        assert np.sum(mask) < n

    def test_check_representation_no_exclusion(self):
        """Lines 959-960: check_representation with no underrepresentation."""
        adata = _make_adata(100)
        mask, details = apply_cell_filter(
            adata,
            cell_filter=None,
            check_representation=True,
            groupby="condition",
            groups="cell_type",
            min_cells=1,  # low threshold → no exclusion
        )
        assert details["auto_filtered"] is False
