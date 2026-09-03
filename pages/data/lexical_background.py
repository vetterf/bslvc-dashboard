"""Background computation for the Lexical Sets plots, isolated from pages/lexical_sets.py.

Ran into the same class of issue documented for grammar's UMAP background callback (see
/memories/repo/dash-background-callbacks.md): DiskcacheManager's background callbacks are
spawned in a separate process and need to pickle the callback's enclosing module globals.
Keeping this module free of Dash Mantine/Bootstrap component instances avoids that.
"""
import pandas as pd
import plotly.graph_objects as go

import pages.data.retrieve_data as retrieve_data
from pages.data.lexicalFunctions import (
    normalize_lexical_tree_selection,
    compute_lexical_data,
    build_lexical_facet_plot,
    prepare_lexical_long_data,
    get_x_axis_config,
    compute_average_lexical_data,
    build_average_lexical_plot,
    compute_lexical_heatmap_by_item,
)

from dash import callback, Output, Input, State
from dash.exceptions import PreventUpdate

# Data needed by the background computation, kept separate from lexical_sets.py's UI globals.
# lexical_sets.py imports these back so other call sites keep working.
Informants = retrieve_data.getInformantData(include_ai=False)
LexicalRaw = retrieve_data.getLexicalData(imputed=False)
LexicalItemsCols = retrieve_data.getLexicalItemsCols()
LexicalMeta = retrieve_data.getLexicalMeta()
LexicalItemMetaMap = LexicalMeta.set_index('column').to_dict('index')


@callback(
    Output('lexical-facet-fig', 'figure'),
    Input('lexical-render-trigger', 'data'),
    [State('participantsTreeLexical', 'checked'),
     State('lexicalItemsTree', 'checked'),
     State('lexical-plot-type', 'value'),
     State('lexical-series-by', 'value'),
     State('lexical-time-axis-mode', 'value'),
     State('lexical-exclude-small-birth-cohorts-switch', 'checked'),
     State('lexical-show-ci-switch', 'checked'),
     State('lexical-nx-opacity-switch', 'checked'),
     State('lexical-sort-by', 'value'),
     State('lexical-facet-cols', 'value'),
     State('lexical-heatmap-color-by-trend-switch', 'checked'),
     State('lexical-heatmap-sort-items', 'value'),
     State('lexical-informants-store', 'data')],
    prevent_initial_call=True,
    background=True,
    running=[(Output('lexical_running', 'data'), True, False)],
)
def compute_lexical_plot_background(trigger_data, selected_participants, selected_items, plot_type,
                                      series_by, time_axis_mode, exclude_small_cohorts, show_ci,
                                      encode_nx_opacity, sort_by, facet_cols, color_by_trend,
                                      sort_items, informants_data):
    """Compute the (potentially slow) lexical plot in a background process.
    
    Supports three plot types:
    - 'facets': faceted plot (one subplot per item)
    - 'average': single line showing average across all items (item-weighted)
    - 'heatmap': heatmap with items on Y-axis, age groups on X-axis
    """
    if trigger_data is None:
        raise PreventUpdate

    current_informants = pd.DataFrame(informants_data) if informants_data else Informants
    participants, items = normalize_lexical_tree_selection(
        selected_participants, selected_items, current_informants, LexicalItemsCols
    )

    if not participants or not items:
        fig = go.Figure()
        fig.update_layout(template="simple_white", annotations=[{
            "text": "Select at least one participant group and one lexical item.",
            "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16},
        }])
        return fig

    if plot_type == 'facets':
        # Faceted plot: multiple subplots, one per item
        _, agg_df, ordered_items, x_config = compute_lexical_data(
            lexical_raw=LexicalRaw,
            informants=current_informants,
            participant_ids=participants,
            items=items,
            mode=time_axis_mode,
            series_by=series_by,
            sort_by=sort_by,
            item_meta_map=LexicalItemMetaMap,
            exclude_small_cohorts=bool(exclude_small_cohorts),
        )
        return build_lexical_facet_plot(
            agg_df, ordered_items, LexicalItemMetaMap, x_config,
            series_by=series_by,
            facet_cols=int(facet_cols) if facet_cols else 4,
            show_ci=bool(show_ci),
            encode_nx_opacity=bool(encode_nx_opacity),
        )
    
    elif plot_type == 'average':
        # Average plot: single line showing mean across all items
        long_df, x_config = prepare_lexical_long_data(
            lexical_raw=LexicalRaw,
            informants=current_informants,
            participant_ids=participants,
            items=items,
            mode=time_axis_mode,
            exclude_small_cohorts=bool(exclude_small_cohorts),
        )
        if long_df.empty:
            fig = go.Figure()
            fig.update_layout(template="simple_white", annotations=[{
                "text": "No data available for the current selection.",
                "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16},
            }])
            return fig
        
        agg_df = compute_average_lexical_data(long_df, series_by=series_by)
        return build_average_lexical_plot(
            agg_df, x_config, series_by=series_by,
            show_ci=bool(show_ci),
            encode_nx_opacity=bool(encode_nx_opacity),
        )
    
    elif plot_type == 'heatmap':
        # Heatmap: items on Y-axis, age groups on X-axis
        # When color_by_trend=True: aggregate by variety and compute trend across age groups
        # When color_by_trend=False: aggregate across all ages by variety
        long_df, x_config = prepare_lexical_long_data(
            lexical_raw=LexicalRaw,
            informants=current_informants,
            participant_ids=participants,
            items=items,
            mode=time_axis_mode,
            exclude_small_cohorts=bool(exclude_small_cohorts),
        )
        if long_df.empty:
            fig = go.Figure()
            fig.update_layout(template="simple_white", annotations=[{
                "text": "No data available for the current selection.",
                "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16},
            }])
            return fig
        
        return compute_lexical_heatmap_by_item(
            long_df, x_config, color_by_trend=bool(color_by_trend),
            item_meta_map=LexicalItemMetaMap, sort_items=sort_items or 'average'
        )
    
    else:
        # Default fallback
        fig = go.Figure()
        fig.update_layout(template="simple_white", annotations=[{
            "text": f"Unknown plot type: {plot_type}",
            "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16},
        }])
        return fig
