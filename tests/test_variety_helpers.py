"""
Tests for variety classification / ordering / sorting helpers in
pages/data/grammarFunctions.py.

These functions are pure Python (no DB, no Dash app required) and form
the backbone of how varieties are ordered consistently across all plots.
"""
import pytest
from pages.data.grammarFunctions import (
    get_variety_classification,
    get_variety_mapping,
    get_ordered_varieties,
    sort_varieties_by_standard_order,
    sort_groups_for_plot,
)


# ---------------------------------------------------------------------------
# get_variety_classification
# ---------------------------------------------------------------------------

class TestGetVarietyClassification:
    def test_returns_dict(self):
        result = get_variety_classification()
        assert isinstance(result, dict)

    def test_has_three_types(self):
        result = get_variety_classification()
        assert set(result.keys()) == {"ENL", "ESL", "EFL"}

    def test_enl_varieties(self):
        result = get_variety_classification()
        assert set(result["ENL"]) == {"England", "Scotland", "US"}

    def test_esl_varieties(self):
        result = get_variety_classification()
        assert set(result["ESL"]) == {"Gibraltar", "India", "Malta", "Puerto Rico"}

    def test_efl_varieties(self):
        result = get_variety_classification()
        assert set(result["EFL"]) == {"Germany", "Slovenia", "Sweden"}

    def test_no_variety_appears_in_multiple_types(self):
        result = get_variety_classification()
        all_varieties = [v for varieties in result.values() for v in varieties]
        assert len(all_varieties) == len(set(all_varieties)), "A variety appears in more than one type"


# ---------------------------------------------------------------------------
# get_variety_mapping
# ---------------------------------------------------------------------------

class TestGetVarietyMapping:
    @pytest.fixture
    def mapping(self):
        return get_variety_mapping()

    def test_standard_varieties_mapped_to_types(self, mapping):
        assert mapping["England"] == "ENL"
        assert mapping["Scotland"] == "ENL"
        assert mapping["US"] == "ENL"
        assert mapping["Gibraltar"] == "ESL"
        assert mapping["India"] == "ESL"
        assert mapping["Malta"] == "ESL"
        assert mapping["Puerto Rico"] == "ESL"
        assert mapping["Germany"] == "EFL"
        assert mapping["Slovenia"] == "EFL"
        assert mapping["Sweden"] == "EFL"

    def test_ai_varieties_mapped_to_ai_types(self, mapping):
        assert mapping["AI-GPT-England"] == "AI-ENL"
        assert mapping["AI-GPT-Gibraltar"] == "AI-ESL"
        assert mapping["AI-GPT-Germany"] == "AI-EFL"

    def test_all_standard_varieties_have_ai_counterpart(self, mapping):
        classification = get_variety_classification()
        for varieties in classification.values():
            for variety in varieties:
                assert f"AI-GPT-{variety}" in mapping

    def test_ai_type_prefix(self, mapping):
        for key, value in mapping.items():
            if key.startswith("AI-GPT-"):
                assert value.startswith("AI-")


# ---------------------------------------------------------------------------
# get_ordered_varieties
# ---------------------------------------------------------------------------

class TestGetOrderedVarieties:
    @pytest.fixture
    def ordered(self):
        return get_ordered_varieties()

    def test_returns_list(self, ordered):
        assert isinstance(ordered, list)

    def test_contains_all_known_varieties(self, ordered):
        expected = {
            "England", "Scotland", "US",
            "Gibraltar", "India", "Malta", "Puerto Rico",
            "Germany", "Slovenia", "Sweden",
        }
        assert expected.issubset(set(ordered))

    def test_no_duplicates(self, ordered):
        assert len(ordered) == len(set(ordered))

    def test_enl_before_esl_before_efl(self, ordered):
        """ENL varieties appear before ESL, which appear before EFL."""
        enl = {"England", "Scotland", "US"}
        esl = {"Gibraltar", "India", "Malta", "Puerto Rico"}
        efl = {"Germany", "Slovenia", "Sweden"}
        enl_indices = [ordered.index(v) for v in enl]
        esl_indices = [ordered.index(v) for v in esl]
        efl_indices = [ordered.index(v) for v in efl]
        assert max(enl_indices) < min(esl_indices), "ENL should precede ESL"
        assert max(esl_indices) < min(efl_indices), "ESL should precede EFL"

    def test_no_ai_varieties_included(self, ordered):
        for v in ordered:
            assert not v.startswith("AI-GPT-")


# ---------------------------------------------------------------------------
# sort_varieties_by_standard_order
# ---------------------------------------------------------------------------

class TestSortVarietiesByStandardOrder:
    def test_already_sorted_stays_sorted(self):
        varieties = ["England", "Scotland", "US", "Gibraltar"]
        result = sort_varieties_by_standard_order(varieties)
        # ENL first, then ESL
        assert result.index("England") < result.index("Gibraltar")

    def test_reversed_input_is_sorted(self):
        varieties = ["Sweden", "Germany", "India", "US", "England"]
        result = sort_varieties_by_standard_order(varieties)
        assert result.index("England") < result.index("India")
        assert result.index("India") < result.index("Germany")

    def test_unknown_varieties_go_to_end(self):
        varieties = ["Sweden", "UnknownVariety", "England"]
        result = sort_varieties_by_standard_order(varieties)
        assert result[-1] == "UnknownVariety"

    def test_empty_list(self):
        assert sort_varieties_by_standard_order([]) == []

    def test_single_element(self):
        assert sort_varieties_by_standard_order(["Malta"]) == ["Malta"]

    def test_result_contains_same_elements(self):
        varieties = ["Sweden", "US", "Malta", "England"]
        result = sort_varieties_by_standard_order(varieties)
        assert set(result) == set(varieties)


# ---------------------------------------------------------------------------
# sort_groups_for_plot
# ---------------------------------------------------------------------------

class TestSortGroupsForPlot:
    def test_variety_mode_uses_standard_order(self):
        groups = ["Sweden", "England", "Gibraltar"]
        result = sort_groups_for_plot(groups, groupby="variety")
        assert result.index("England") < result.index("Gibraltar")
        assert result.index("Gibraltar") < result.index("Sweden")

    def test_vtype_mode_enl_before_esl_before_efl(self):
        groups = ["EFL", "ENL", "ESL"]
        result = sort_groups_for_plot(groups, groupby="vtype")
        assert result == ["ENL", "ESL", "EFL"]

    def test_vtype_mode_ai_types_after_main(self):
        groups = ["AI-ENL", "EFL", "ENL"]
        result = sort_groups_for_plot(groups, groupby="vtype")
        assert result.index("ENL") < result.index("EFL")
        assert result.index("EFL") < result.index("AI-ENL")

    def test_vtype_balanced_same_as_vtype(self):
        groups = ["EFL", "ESL", "ENL"]
        result_vtype = sort_groups_for_plot(groups, groupby="vtype")
        result_balanced = sort_groups_for_plot(groups, groupby="vtype_balanced")
        assert result_vtype == result_balanced

    def test_gender_mode_alphabetical(self):
        groups = ["Male", "Female", "Diverse"]
        result = sort_groups_for_plot(groups, groupby="gender")
        assert result == sorted(groups)

    def test_unknown_groupby_alphabetical(self):
        groups = ["Zebra", "Alpha", "Middle"]
        result = sort_groups_for_plot(groups, groupby="unknown_key")
        assert result == ["Alpha", "Middle", "Zebra"]

    def test_empty_groups(self):
        assert sort_groups_for_plot([], groupby="variety") == []
