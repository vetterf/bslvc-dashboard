"""
Tests for the color mapping infrastructure in pages/data/retrieve_data.py.

Covers:
- _hex_to_rgb / _rgb_to_hex round-trip
- _tone_down_color blending
- VarietyColorMap: existing manual mappings preserved
- VarietyColorMap: auto-generated colors for unmapped varieties are deterministic
- VarietyColorMap: AI-GPT varieties return toned-down versions of their base
- VarietyColorMap: dict-like API (__getitem__, get, in, items, keys, values)
"""
import pytest
from pages.data.retrieve_data import (
    _hex_to_rgb,
    _rgb_to_hex,
    _tone_down_color,
    VarietyColorMap,
    get_color_for_variety,
)


# ---------------------------------------------------------------------------
# _hex_to_rgb
# ---------------------------------------------------------------------------

class TestHexToRgb:
    def test_black(self):
        assert _hex_to_rgb("#000000") == (0, 0, 0)

    def test_white(self):
        assert _hex_to_rgb("#ffffff") == (255, 255, 255)

    def test_known_color(self):
        # England blue: #1f77b4  → (31, 119, 180)
        assert _hex_to_rgb("#1f77b4") == (31, 119, 180)

    def test_no_hash_prefix(self):
        assert _hex_to_rgb("ff0000") == (255, 0, 0)

    def test_uppercase(self):
        assert _hex_to_rgb("#FF7F0E") == (255, 127, 14)


# ---------------------------------------------------------------------------
# _rgb_to_hex
# ---------------------------------------------------------------------------

class TestRgbToHex:
    def test_black(self):
        assert _rgb_to_hex(0, 0, 0) == "#000000"

    def test_white(self):
        assert _rgb_to_hex(255, 255, 255) == "#ffffff"

    def test_known_color(self):
        assert _rgb_to_hex(31, 119, 180) == "#1f77b4"


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("hex_color", [
        "#1f77b4", "#7B3394", "#B22234", "#d62728",
        "#9467bd", "#FF9933", "#00A693", "#000000", "#ffffff",
    ])
    def test_round_trip(self, hex_color):
        r, g, b = _hex_to_rgb(hex_color)
        assert _rgb_to_hex(r, g, b) == hex_color


# ---------------------------------------------------------------------------
# _tone_down_color
# ---------------------------------------------------------------------------

class TestToneDownColor:
    def test_factor_1_returns_original(self):
        """factor=1 should return the exact original color."""
        assert _tone_down_color("#1f77b4", factor=1.0) == "#1f77b4"

    def test_factor_0_returns_white(self):
        """factor=0 should blend fully to white."""
        assert _tone_down_color("#1f77b4", factor=0.0) == "#ffffff"

    def test_result_is_lighter(self):
        """Default factor produces a lighter (higher average RGB) color."""
        original = "#1f77b4"
        toned = _tone_down_color(original)
        orig_r, orig_g, orig_b = _hex_to_rgb(original)
        tone_r, tone_g, tone_b = _hex_to_rgb(toned)
        assert (tone_r + tone_g + tone_b) > (orig_r + orig_g + orig_b)

    def test_result_is_valid_hex(self):
        result = _tone_down_color("#d62728")
        assert result.startswith("#")
        assert len(result) == 7


# ---------------------------------------------------------------------------
# VarietyColorMap – manual mappings preserved
# ---------------------------------------------------------------------------

KNOWN_VARIETIES = [
    ("England",                 "#1f77b4"),
    ("England_North",           "#4a90c4"),
    ("England_South",           "#0d5a8f"),
    ("England_UNCLEAR",         "#7bb3d9"),
    ("Scotland",                "#7B3394"),
    ("US",                      "#B22234"),
    ("Gibraltar",               "#d62728"),
    ("Malta",                   "#9467bd"),
    ("India",                   "#FF9933"),
    ("Puerto Rico",             "#00A693"),
    ("Slovenia",                "#7f7f7f"),
    ("Germany",                 "#FFCE00"),
    ("Sweden",                  "#006AA7"),
    ("Spain (Balearic Islands)","#393b79"),
    ("Other",                   "#c49c94"),
]

KNOWN_AI_VARIETIES = [
    ("AI-GPT-England",                  "#8fbbd9"),
    ("AI-GPT-England_North",            "#a5c8e2"),
    ("AI-GPT-England_South",            "#86adc7"),
    ("AI-GPT-England_UNCLEAR",          "#b8d5e8"),
    ("AI-GPT-Scotland",                 "#af84be"),
    ("AI-GPT-US",                       "#d07a85"),
    ("AI-GPT-Gibraltar",                "#eb9394"),
    ("AI-GPT-Malta",                    "#cab3de"),
    ("AI-GPT-India",                    "#ffc184"),
    ("AI-GPT-Puerto Rico",              "#66c9be"),
    ("AI-GPT-Slovenia",                 "#bfbfbf"),
    ("AI-GPT-Germany",                  "#ffe166"),
    ("AI-GPT-Sweden",                   "#66a5ca"),
    ("AI-GPT-Spain (Balearic Islands)", "#9c9dbc"),
    ("AI-GPT-Other",                    "#e2ceca"),
]


class TestVarietyColorMapManualMappings:
    @pytest.fixture
    def color_map(self):
        return VarietyColorMap()

    @pytest.mark.parametrize("variety,expected", KNOWN_VARIETIES)
    def test_known_variety_bracket_access(self, color_map, variety, expected):
        """Bracket access returns the manually defined color."""
        assert color_map[variety] == expected

    @pytest.mark.parametrize("variety,expected", KNOWN_VARIETIES)
    def test_known_variety_get(self, color_map, variety, expected):
        """get() returns the manually defined color."""
        assert color_map.get(variety) == expected

    @pytest.mark.parametrize("variety,expected", KNOWN_AI_VARIETIES)
    def test_known_ai_variety(self, color_map, variety, expected):
        """Manually-mapped AI-GPT colors are returned exactly."""
        assert color_map[variety] == expected

    @pytest.mark.parametrize("variety,_", KNOWN_VARIETIES)
    def test_known_variety_in_operator(self, color_map, variety, _):
        """'in' operator returns True for manually-mapped varieties."""
        assert variety in color_map


# ---------------------------------------------------------------------------
# VarietyColorMap – auto-generation for unmapped varieties
# ---------------------------------------------------------------------------

class TestVarietyColorMapAutoGeneration:
    @pytest.fixture
    def color_map(self):
        return VarietyColorMap()

    def test_unmapped_variety_returns_a_color(self, color_map):
        color = color_map["NewCountry"]
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7

    def test_unmapped_variety_is_deterministic(self, color_map):
        """Same variety always yields the same color."""
        c1 = color_map["NewCountry"]
        c2 = color_map["NewCountry"]
        assert c1 == c2

    def test_two_unmapped_varieties_are_independent(self, color_map):
        """Two different new varieties can produce different colors
        (palette has 10 slots; at least some must differ)."""
        varieties = [f"TestVariety{i}" for i in range(10)]
        colors = [color_map[v] for v in varieties]
        # Not all colors should be identical
        assert len(set(colors)) > 1

    def test_unmapped_variety_not_none(self, color_map):
        assert color_map.get("Nonexistent", None) is not None

    def test_unmapped_variety_cached(self, color_map):
        """Auto-generated colors are cached across multiple instances of the same map."""
        c1 = color_map["CachedVariety"]
        c2 = color_map["CachedVariety"]
        assert c1 == c2


# ---------------------------------------------------------------------------
# VarietyColorMap – AI-GPT auto-toning
# ---------------------------------------------------------------------------

class TestVarietyColorMapAIGPTAutoToning:
    @pytest.fixture
    def color_map(self):
        return VarietyColorMap()

    def test_ai_gpt_of_unmapped_variety_returns_lighter_color(self, color_map):
        """AI-GPT version of an unmapped variety should be lighter than the base."""
        base = color_map["NewVariety"]
        ai = color_map["AI-GPT-NewVariety"]
        base_r, base_g, base_b = _hex_to_rgb(base)
        ai_r, ai_g, ai_b = _hex_to_rgb(ai)
        assert (ai_r + ai_g + ai_b) >= (base_r + base_g + base_b)

    def test_ai_gpt_of_unmapped_variety_is_deterministic(self, color_map):
        c1 = color_map["AI-GPT-AnotherNewVariety"]
        c2 = color_map["AI-GPT-AnotherNewVariety"]
        assert c1 == c2

    def test_ai_gpt_differs_from_base(self, color_map):
        base = color_map["NewVariety2"]
        ai = color_map["AI-GPT-NewVariety2"]
        # They should NOT be identical (toning changes the color unless base is white)
        assert base != ai


# ---------------------------------------------------------------------------
# VarietyColorMap – dict-like API
# ---------------------------------------------------------------------------

class TestVarietyColorMapDictAPI:
    @pytest.fixture
    def color_map(self):
        return VarietyColorMap()

    def test_keys_contains_known_varieties(self, color_map):
        keys = list(color_map.keys())
        assert "England" in keys
        assert "Germany" in keys

    def test_values_are_hex_strings(self, color_map):
        for v in color_map.values():
            assert v.startswith("#")
            assert len(v) == 7

    def test_items_returns_tuples(self, color_map):
        for variety, color in color_map.items():
            assert isinstance(variety, str)
            assert color.startswith("#")

    def test_get_with_explicit_default_for_missing_key_uses_auto(self, color_map):
        """get() ignores the caller's default for known keys."""
        assert color_map.get("England", "#000000") == "#1f77b4"

    def test_unknown_key_not_in_operator(self, color_map):
        """Unmapped varieties are not 'in' the map (fixed_map only)."""
        assert "NewCountryXYZ" not in color_map


# ---------------------------------------------------------------------------
# get_color_for_variety factory function
# ---------------------------------------------------------------------------

class TestGetColorForVariety:
    def test_returns_variety_color_map_instance(self):
        result = get_color_for_variety()
        assert isinstance(result, VarietyColorMap)

    def test_type_parameter_does_not_break(self):
        """type param is accepted; currently both return same map."""
        result_grammar = get_color_for_variety(type="grammar")
        result_lexical = get_color_for_variety(type="lexical")
        assert result_grammar["England"] == result_lexical["England"]
