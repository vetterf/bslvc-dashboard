import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pages.data.lexicalFunctions import (
    aggregate_lexical_facets,
    build_average_lexical_plot,
    build_lexical_dotplot,
    build_lexical_facet_plot,
    compute_average_lexical_data,
    compute_dotplot_data,
    compute_lexical_heatmap_by_item,
    get_x_axis_config,
)


@pytest.fixture
def lexical_rows():
    return pd.DataFrame([
        ("a", "20-29", "Female", "US", "numeric", -2.0),
        ("a", "30-39", "Female", "US", "numeric", 0.0),
        ("a", "30-39", "Male", "UK", "nx", np.nan),
        ("a", "40-49", "Male", "UK", "missing", np.nan),
        ("b", "20-29", "Male", "US", "numeric", 2.0),
        ("b", "30-39", "Non-binary", "UK", "numeric", 1.0),
        ("b", "40-49", None, None, "numeric", -1.0),
    ], columns=["Item", "XGroup", "Gender", "MainVariety", "ValueType", "NumericValue"])


@pytest.mark.parametrize("series_by", ["variety", "gender", "none"])
def test_dotplot_matches_pooled_facet_statistics(lexical_rows, series_by):
    original = lexical_rows.copy(deep=True)
    pooled = lexical_rows.copy()
    pooled["XGroup"] = "All"

    expected = aggregate_lexical_facets(pooled, series_by=series_by)
    actual = compute_dotplot_data(lexical_rows, series_by=series_by)

    assert_frame_equal(actual, expected)
    assert_frame_equal(lexical_rows, original)


@pytest.mark.parametrize("series_by", ["variety", "gender", "none"])
def test_dotplot_empty_input_matches_existing_aggregation(lexical_rows, series_by):
    empty = lexical_rows.iloc[0:0]
    assert_frame_equal(
        compute_dotplot_data(empty, series_by=series_by),
        aggregate_lexical_facets(empty, series_by=series_by),
    )


def test_dotplot_excludes_unsupported_genders(lexical_rows):
    rows = lexical_rows[lexical_rows["Gender"] == "Non-binary"]
    pooled = rows.assign(XGroup="All")
    assert_frame_equal(
        compute_dotplot_data(rows, series_by="gender"),
        aggregate_lexical_facets(pooled, series_by="gender"),
    )


def test_dotplot_and_existing_plot_builders(lexical_rows):
    metadata = {"a": {"american": "A", "british": "B"}, "b": {"american": "C", "british": "D"}}
    config = get_x_axis_config("apparent_time")
    dot = build_lexical_dotplot(compute_dotplot_data(lexical_rows), ["a", "b"], metadata)
    facets = build_lexical_facet_plot(
        aggregate_lexical_facets(lexical_rows, series_by="variety"),
        ["a", "b"], metadata, config, series_by="variety",
    )
    average = build_average_lexical_plot(
        compute_average_lexical_data(lexical_rows, series_by="variety"),
        config, series_by="variety",
    )
    heatmap_rows = lexical_rows.assign(RawValue=lexical_rows["NumericValue"])
    heatmap = compute_lexical_heatmap_by_item(heatmap_rows, config, item_meta_map=metadata)

    assert len(dot.data) == 3  # Two varieties and the mirrored-axis trace.
    assert facets["data"]
    assert average.data
    assert heatmap.data


def test_facets_skip_empty_traces_and_preserve_legend_and_data(lexical_rows):
    agg = aggregate_lexical_facets(lexical_rows, series_by="variety")
    config = get_x_axis_config("apparent_time")
    fig = build_lexical_facet_plot(agg, ["a", "b"], {}, config, series_by="variety")
    traces = [trace for trace in fig["data"] if trace["mode"] == "lines+markers"]

    # UK has no numeric mean for item a, but has one for b.
    assert len(traces) == 3
    by_name = {trace["name"]: trace for trace in traces}
    assert sum(trace["showlegend"] for trace in traces if trace["name"] == "US") == 1
    assert by_name["UK"]["showlegend"]
    assert by_name["UK"]["legendrank"] < by_name["US"]["legendrank"]
    assert by_name["UK"]["xaxis"] == "x2"
    idx = config["group_order"].index("30-39")
    assert by_name["UK"]["y"][idx] == 1.0
    assert by_name["UK"]["customdata"][idx][0] == 1


def test_facets_keep_one_legend_for_series_without_numeric_ratings(lexical_rows):
    rows = lexical_rows[lexical_rows["MainVariety"] == "UK"].copy()
    rows["ValueType"] = "nx"
    rows["NumericValue"] = np.nan
    agg = aggregate_lexical_facets(rows, series_by="variety")
    fig = build_lexical_facet_plot(
        agg, ["a", "b"], {}, get_x_axis_config("apparent_time"), series_by="variety",
    )

    assert len(fig["data"]) == 1
    trace = fig["data"][0]
    assert trace["name"] == "UK"
    assert trace["showlegend"]
    assert all(np.isnan(value) for value in trace["y"])
