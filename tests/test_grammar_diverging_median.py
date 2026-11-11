import pandas as pd
import pytest

from pages.data import grammarFunctions


def make_plot(monkeypatch, rows, modes, *, pairs=False, perceptual=False):
    raw = pd.DataFrame(rows)
    monkeypatch.setattr(grammarFunctions.retrieve_data, "getGrammarData", lambda **kwargs: raw.copy())
    return grammarFunctions.create_diverging_stacked_bar_plot(
        pd.DataFrame(), ["A1"], modes, "variety", None, pairs, pd.DataFrame(), [],
        use_perceptual_scale=perceptual,
    )


def rows_for(ratings, *, mode=None, group="Test"):
    return [
        {"InformantID": index, "MainVariety": group, "A1": rating, **({"section": mode} if mode else {})}
        for index, rating in enumerate(ratings)
    ]


def median_traces(fig):
    return [trace for trace in fig.data if trace.name == "Median (50th percentile)"]


@pytest.mark.parametrize(
    ("ratings", "pairs", "perceptual", "position", "median"),
    [
        ([0, 1, 4, 5], False, False, 0, "2.5"),
        ([0, 1, 1, 5], False, False, -25, "1"),
        ([0, 2, 3, 5], False, True, 0, "47"),
        ([-2, 0, 0, 2], True, False, 0, "0"),
        ([-2, -1, 0, 2], True, False, -12.5, "-0.5"),
        ([0, 0, 0], True, False, 0, "0"),
        ([5, None, 5], False, False, 50, "5"),
    ],
)
def test_median_marker_tracks_distribution(monkeypatch, ratings, pairs, perceptual, position, median):
    fig = make_plot(monkeypatch, rows_for(ratings), ["single"], pairs=pairs, perceptual=perceptual)
    marker, = median_traces(fig)
    assert marker.x[0] == pytest.approx(position)
    assert marker.customdata[0] == ["A1", "Test", median, sum(value is not None for value in ratings)]
    assert "50th percentile" in marker.hovertemplate
    assert "Median" in marker.hovertemplate


def test_medians_are_independent_for_each_mode_and_group(monkeypatch):
    rows = (
        rows_for([0, 0, 5], mode="Spoken", group="A")
        + rows_for([5, 5, 5], mode="Written", group="A")
        + rows_for([0, 5, 5], mode="Written", group="B")
    )
    fig = make_plot(monkeypatch, rows, ["spoken", "written"])
    markers = median_traces(fig)
    assert len(markers) == 3
    by_key = {(trace.xaxis, trace.customdata[0][1]): trace for trace in markers}
    assert by_key["x", "A"].x[0] == pytest.approx(50 - 200 / 3)
    assert by_key["x", "A"].customdata[0][2] == "0"
    assert by_key["x2", "A"].x[0] == pytest.approx(50)
    assert by_key["x2", "B"].x[0] == pytest.approx(50 - 100 / 3)
    assert "Mode: Written" in by_key["x2", "B"].hovertemplate


def test_split_mode_perceptual_ratings_are_plotted(monkeypatch):
    rows = rows_for([1, 2, 3, 4], mode="Spoken") + rows_for([5], mode="Written")
    fig = make_plot(monkeypatch, rows, ["spoken", "written"], perceptual=True)
    spoken = [trace for trace in fig.data if trace.type == "bar" and trace.xaxis == "x"]
    assert sorted(float(trace.x[0]) for trace in spoken) == [-25, -25, 0, 0, 25, 25]
    marker = next(trace for trace in median_traces(fig) if trace.xaxis == "x")
    assert marker.x[0] == 0
    assert marker.customdata[0][2] == "47"
    assert "perceptual %" in marker.hovertemplate


def test_split_pair_neutral_is_stacked_next_to_zero(monkeypatch):
    rows = rows_for([-2, 0, 0, 2], mode="Spoken") + rows_for([0], mode="Written")
    fig = make_plot(monkeypatch, rows, ["spoken", "written"], pairs=True)
    spoken = [trace for trace in fig.data if trace.type == "bar" and trace.xaxis == "x"]
    assert spoken[0].name == "Rating 0 (left)"
    assert spoken[0].x[0] == -25
    assert median_traces(fig)[0].x[0] == 0
