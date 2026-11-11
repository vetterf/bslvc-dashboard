import base64
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from pages.data import grammarFunctions as gf


@pytest.fixture
def umap_data():
    return pd.DataFrame([
        {
            "InformantID": "p1", "MainVariety": "England", "MainVariety_Original": "England",
            "Age": 20, "Gender": "Female", "RatioMainVariety": 0.8,
            "YearsLivedInMainVariety": 10, "CountryCollection": "UK", "Year": 2020,
            "LanguageHome": "English, French", "LanguageMother": "French",
            "LanguageFather": "English", "A1": 1,
        },
        {
            "InformantID": "p2", "MainVariety": "Malta", "MainVariety_Original": "Malta",
            "Age": 22, "Gender": "Male", "RatioMainVariety": 1,
            "YearsLivedInMainVariety": 20, "CountryCollection": "Malta", "Year": 2021,
            "LanguageHome": None, "LanguageMother": "", "LanguageFather": "  ",
            "A1": 2,
        },
    ], index=[4, 9])


@pytest.fixture
def mock_umap(monkeypatch, tmp_path):
    class FakeUMAP:
        def __init__(self, n_components, **kwargs):
            self.n_components = n_components

        def fit_transform(self, data):
            return np.arange(len(data) * self.n_components).reshape(len(data), self.n_components)

    monkeypatch.setattr(gf.umap, "UMAP", FakeUMAP)
    monkeypatch.setattr(gf, "compute_umap_quality_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(gf, "__file__", str(tmp_path / "grammarFunctions.py"))
    monkeypatch.setattr(gf, "performLeidenClustering", lambda *args, **kwargs: [0, 1])


@pytest.mark.parametrize(
    ("options", "expected_traces"),
    [({}, 2), ({"umap_3d": True}, 2), ({"umap_4d": True}, 4), ({"leiden": True}, 2)],
)
def test_language_hover_per_participant_and_mode(umap_data, mock_umap, options, expected_traces):
    fig, _ = gf.getUMAPplot(
        umap_data, ["A1"], informants=umap_data, items=["A1"], force_rerender=True, **options
    )
    traces = [trace for trace in fig.data if trace.type in ("scatter", "scatter3d")]
    assert len(traces) == expected_traces
    for trace in traces:
        row = trace.customdata[0]
        assert row[0] == trace.ids[0]
        assert row[4] in ("Female", "Male")
        for offset, label in enumerate(gf.UMAP_LANGUAGE_LABELS):
            assert f"{label}: %{{customdata[{len(row) - 3 + offset}]}}" in trace.hovertemplate
        if row[0] == "p1":
            assert list(row[-3:]) == ["English, French", "French", "English"]
        else:
            assert list(row[-3:]) == ["Not available"] * 3
        if options.get("leiden"):
            assert "Leiden Cluster: %{customdata[9]}" in trace.hovertemplate
            assert row[9] in (0, 1)


@pytest.mark.parametrize("with_cluster", [False, True])
def test_coordinate_export_keeps_language_column_names(umap_data, mock_umap, monkeypatch, with_cluster):
    import app
    from pages.grammar import download_umap_coordinates

    fig, _ = gf.getUMAPplot(
        umap_data, ["A1"], informants=umap_data, items=["A1"],
        force_rerender=True, leiden=with_cluster,
    )
    result = download_umap_coordinates(
        1, fig.to_plotly_json(), ["p1", "p2"], ["A1"], {}, 15, 0.1,
        "cosine", False, False, 2.0, False, False, False, False,
    )
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["content"]))) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        df = pd.read_csv(archive.open(csv_name)).set_index("InformantID")
    assert df.loc["p1", gf.UMAP_LANGUAGE_COLUMNS].tolist() == ["English, French", "French", "English"]
    assert df.loc["p2", gf.UMAP_LANGUAGE_COLUMNS].tolist() == ["Not available"] * 3
    if with_cluster:
        assert df.loc["p1", "LeidenCluster"] == 0


def test_old_cached_figures_are_not_reused(umap_data, mock_umap, monkeypatch):
    from pages.data import umap_background

    class Cache:
        def __init__(self):
            self.keys = []

        def get(self, key):
            self.keys.append(key)
            return None

        def set(self, key, value):
            self.keys.append(key)

    cache = Cache()
    monkeypatch.setattr(umap_background, "plot_cache", cache)
    monkeypatch.setattr(umap_background.retrieve_data, "getGrammarData", lambda **kwargs: umap_data)
    monkeypatch.setattr(umap_background, "GrammarItemsCols", ["A1"])
    args = (["p1", "p2"], ["A1"], 15, 0.1, "cosine", False, False, 2.0, False)
    key = umap_background.create_plot_cache_key(*args)
    monkeypatch.setattr(umap_background, "UMAP_FIGURE_VERSION", gf.UMAP_FIGURE_VERSION - 1)
    old_key = umap_background.create_plot_cache_key(*args)
    assert key != old_key
    monkeypatch.setattr(umap_background, "UMAP_FIGURE_VERSION", gf.UMAP_FIGURE_VERSION)
    fig, _ = umap_background.get_cached_umap_plot(*args, informants=umap_data)
    assert cache.keys == [f"umap_{key}", f"umap_{key}"]
    assert "Languages used at home" in next(trace for trace in fig.data if trace.type == "scatter").hovertemplate
