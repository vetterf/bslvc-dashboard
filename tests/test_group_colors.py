import plotly.graph_objects as go

from pages.data.group_colors import tag_group_colors


def test_group_metadata_preserves_original_styles_and_existing_meta():
    fig = go.Figure(go.Scatter(
        name="England (spoken)", marker={"color": "#123456"},
        meta={"other": "kept"},
    ))
    result = tag_group_colors(fig, "variety")
    assert result is fig
    assert fig.layout.meta["group_color_kind"] == "variety"
    assert fig.data[0].meta == {
        "other": "kept",
        "group_color": {
            "group": "England", "marker": "#123456", "line": None, "fill": None,
        },
    }


def test_lexical_confidence_band_uses_legend_group():
    fig = go.Figure([
        go.Scatter(legendgroup="Female", fill="toself", fillcolor="rgba(1,2,3,0.08)"),
        go.Scatter(name="Female", legendgroup="Female", line={"color": "#123456"}),
    ])
    tag_group_colors(fig, "gender", use_legendgroup=True)
    assert [trace.meta["group_color"]["group"] for trace in fig.data] == ["Female", "Female"]
    assert fig.data[0].meta["group_color"]["fill"] == "rgba(1,2,3,0.08)"
    assert fig.data[1].meta["group_color"]["line"] == "#123456"


def test_vtype_grouping_gets_default_colors_and_tags():
    fig = go.Figure([go.Scatter(name="EFL"), go.Scatter(name="AI-ENL", marker={"color": "#123456"})])
    tag_group_colors(fig, "vtype_balanced")
    assert fig.layout.meta["group_color_kind"] == "vtype"
    assert fig.data[0].meta["group_color"]["marker"] == "#2ca02c"
    assert fig.data[1].meta["group_color"]["marker"] == "#123456"


def test_unknown_groupings_are_untouched():
    fig = go.Figure(go.Scatter(name="ENL"))
    tag_group_colors(fig, "region")
    assert fig.layout.meta is None
    assert fig.data[0].meta is None


def test_lexical_dotplot_pools_ages_and_tags_groups():
    import pandas as pd
    from pages.data.lexicalFunctions import compute_dotplot_data, build_lexical_dotplot

    rows = []
    for i, (gender, age, value) in enumerate([("Female", 20, -2), ("Female", 50, 0), ("Male", 30, 2), ("Male", 60, 1)]):
        rows.append({"InformantID": i, "Age": age, "Gender": gender, "MainVariety": "US", "XGroup": "x",
                     "Item": "a", "RawValue": value, "ValueType": "numeric", "NumericValue": float(value)})
    agg = compute_dotplot_data(pd.DataFrame(rows), series_by="gender")
    assert set(agg["XGroup"]) == {"All"}
    assert agg.set_index("Series")["Mean"].to_dict() == {"Female": -1.0, "Male": 1.5}

    fig = build_lexical_dotplot(agg, ["a"], {"a": {"american": "A", "british": "B"}}, series_by="gender")
    assert list(fig.layout.xaxis.range) == [-2.2, 2.2]
    assert fig.layout.yaxis2.ticktext == ("B",)


def test_tag_group_colors_accepts_figure_dicts():
    fig = {"data": [{"legendgroup": "US", "marker": {"color": "#111111"}, "meta": {"ci_band": True}}], "layout": {}}
    tag_group_colors(fig, "variety", use_legendgroup=True)
    assert fig["layout"]["meta"]["group_color_kind"] == "variety"
    assert fig["data"][0]["meta"]["group_color"]["group"] == "US"
    assert fig["data"][0]["meta"]["ci_band"] is True
