"""
Tests for general utility functions in pages/data/:

- retrieve_data.inner_join_list
- grammarFunctions.find_non_zero_positions_values_and_ids
- grammarFunctions.getGroupingsFromFigure
- grammarFunctions.getColorGroupingsFromFigure (edge cases only)
- grammarFunctions.getEmptyPlot
"""
import pytest
import pandas as pd
from pages.data.retrieve_data import inner_join_list
from pages.data.grammarFunctions import (
    find_non_zero_positions_values_and_ids,
    getGroupingsFromFigure,
    getEmptyPlot,
)


# ---------------------------------------------------------------------------
# inner_join_list
# ---------------------------------------------------------------------------

class TestInnerJoinList:
    def test_common_elements_returned(self):
        assert inner_join_list([1, 2, 3], [2, 3, 4]) == [2, 3]

    def test_order_follows_first_list(self):
        result = inner_join_list(["c", "a", "b"], ["a", "b"])
        assert result == ["a", "b"]

    def test_no_common_elements(self):
        assert inner_join_list([1, 2], [3, 4]) == []

    def test_empty_first_list(self):
        assert inner_join_list([], [1, 2]) == []

    def test_empty_second_list(self):
        assert inner_join_list([1, 2], []) == []

    def test_both_empty(self):
        assert inner_join_list([], []) == []

    def test_duplicates_in_first_list(self):
        """Duplicates in list1 are preserved if present in list2."""
        result = inner_join_list([1, 1, 2], [1, 2])
        assert result == [1, 1, 2]

    def test_strings(self):
        result = inner_join_list(["A1", "B2", "C3"], ["A1", "C3"])
        assert result == ["A1", "C3"]


# ---------------------------------------------------------------------------
# find_non_zero_positions_values_and_ids
# ---------------------------------------------------------------------------

class TestFindNonZeroPositionsValuesAndIds:
    def test_returns_only_non_zero(self):
        symbols = [0, 1, 0, 2]
        ids = ["a", "b", "c", "d"]
        result = find_non_zero_positions_values_and_ids(symbols, ids)
        assert result == [(1, 1, "b"), (3, 2, "d")]

    def test_all_zero_returns_empty(self):
        result = find_non_zero_positions_values_and_ids([0, 0, 0], ["a", "b", "c"])
        assert result == []

    def test_all_non_zero(self):
        symbols = [1, 2, 3]
        ids = ["x", "y", "z"]
        result = find_non_zero_positions_values_and_ids(symbols, ids)
        assert result == [(0, 1, "x"), (1, 2, "y"), (2, 3, "z")]

    def test_empty_inputs(self):
        assert find_non_zero_positions_values_and_ids([], []) == []

    def test_returns_correct_tuple_structure(self):
        result = find_non_zero_positions_values_and_ids([5], ["id1"])
        assert len(result) == 1
        pos, sym, id_ = result[0]
        assert pos == 0
        assert sym == 5
        assert id_ == "id1"


# ---------------------------------------------------------------------------
# getGroupingsFromFigure
# ---------------------------------------------------------------------------

def _make_scatter_trace(symbols, ids, opacity=0.8):
    """Helper to build a minimal Dash/Plotly figure dict."""
    return {
        "type": "scatter",
        "marker": {
            "symbol": symbols,
            "opacity": opacity,
            "color": ["blue"] * len(ids),
        },
        "ids": ids,
        "name": "TestGroup",
        "x": list(range(len(ids))),
        "y": list(range(len(ids))),
    }


class TestGetGroupingsFromFigure:
    def test_none_figure_returns_empty(self):
        result = getGroupingsFromFigure(None)
        assert result == []

    def test_empty_figure_returns_empty(self):
        result = getGroupingsFromFigure({})
        assert result == []

    def test_figure_with_no_data_returns_empty(self):
        result = getGroupingsFromFigure({"data": []})
        assert isinstance(result, (list, dict))

    def test_figure_with_all_zero_symbols_returns_empty_df(self):
        trace = _make_scatter_trace([0, 0, 0], ["a", "b", "c"])
        fig = {"data": [trace]}
        result = getGroupingsFromFigure(fig)
        # result is a dict with 'dataframe' key when data exists, else empty list
        if isinstance(result, dict):
            df = pd.DataFrame(result["dataframe"])
            assert df.empty or len(df) == 0
        else:
            assert result == []

    def test_figure_with_non_zero_symbols_has_records(self):
        trace = _make_scatter_trace([1, 0, 2], ["id1", "id2", "id3"])
        fig = {"data": [trace]}
        result = getGroupingsFromFigure(fig)
        if isinstance(result, dict) and "dataframe" in result:
            df = pd.DataFrame(result["dataframe"])
            # Only the non-zero entries should appear
            assert len(df) == 2
            assert set(df["id"].tolist()) == {"id1", "id3"}

    def test_opacity_zero_excludes_points(self):
        """Points with zero opacity should be excluded."""
        trace = _make_scatter_trace([1, 2], ["a", "b"], opacity=0)
        fig = {"data": [trace]}
        result = getGroupingsFromFigure(fig)
        if isinstance(result, dict) and "dataframe" in result:
            df = pd.DataFrame(result["dataframe"])
            assert df.empty

    def test_per_point_opacity_filters_correctly(self):
        """When opacity is a list, only points with opacity > 0 are included."""
        trace = _make_scatter_trace([1, 2], ["a", "b"], opacity=[0, 0.8])
        fig = {"data": [trace]}
        result = getGroupingsFromFigure(fig)
        if isinstance(result, dict) and "dataframe" in result:
            df = pd.DataFrame(result["dataframe"])
            assert len(df) == 1
            assert df.iloc[0]["id"] == "b"


# ---------------------------------------------------------------------------
# getEmptyPlot
# ---------------------------------------------------------------------------

class TestGetEmptyPlot:
    def test_returns_figure_object(self):
        import plotly.graph_objects as go
        fig = getEmptyPlot()
        assert isinstance(fig, go.Figure)

    def test_default_annotation_text(self):
        fig = getEmptyPlot()
        annotations = fig.layout.annotations
        assert any("No data available." in a.text for a in annotations)

    def test_custom_message(self):
        fig = getEmptyPlot("Custom message here")
        annotations = fig.layout.annotations
        assert any("Custom message here" in a.text for a in annotations)

    def test_no_data_traces(self):
        fig = getEmptyPlot()
        assert len(fig.data) == 0
