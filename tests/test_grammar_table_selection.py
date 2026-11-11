import pandas as pd
import pytest
from dash import no_update

from pages.data.grammarFunctions import getMetaTable


@pytest.mark.parametrize("pairs", [False, True])
def test_grammar_table_uses_multi_row_click_selection(pairs):
    metadata = pd.DataFrame([{
        "question_code": "A1",
        "question_code_written": "G21",
        "group_finegrained": "Group",
        "feature_ewave": "eWAVE",
        "feature_ewave_id": "1",
        "item": "Item",
        "group_ewave": "Area",
        "feature": "Feature",
        "section": "Spoken",
    }])

    grid = getMetaTable(metadata, pairs=pairs).children[1]
    assert grid.dashGridOptions["rowSelection"] == {
        "mode": "multiRow",
        "checkboxes": False,
        "headerCheckbox": False,
        "enableClickSelection": True,
    }
    assert grid.rowData[0]["Item Code"] == ("A1 - G21" if pairs else "A1")


@pytest.mark.parametrize(
    ("pairs", "rows", "codes"),
    [
        (False, [{"Item Code": "A1"}, {"Item Code": "A2"}], ["A1", "A2"]),
        (True, [{"Item Code": "A1 - G21"}, {"Item Code": "A2 - G22"}], ["A1-G21", "A2-G22"]),
    ],
)
def test_transfer_selected_rows_to_and_from_tree(pairs, rows, codes):
    import app  # Registers Dash pages before importing the callback module.
    from pages.grammar import deselect_table_rows_in_tree, select_table_rows_in_tree

    selected = select_table_rows_in_tree(1, rows, ["existing", codes[0]], pairs)
    assert set(selected) == {"existing", *codes}
    assert deselect_table_rows_in_tree(1, rows, selected, pairs) == ["existing"]
    assert select_table_rows_in_tree(1, [], selected, pairs) is no_update
    assert deselect_table_rows_in_tree(1, [], selected, pairs) is no_update
