"""Background UMAP computation, isolated from pages/grammar.py.
Ran into deserialization problems when packaging the app via PyInstaller on Windows, so had to isolate the callback functions.
"""
import hashlib
import os

import diskcache as dc
import pandas as pd

import pages.data.retrieve_data as retrieve_data
from pages.data.grammarFunctions import (
    getUMAPplot,
    getColorGroupingsFromFigure,
    expand_presets_to_items,
    generate_dynamic_presets,
    build_preset_multiselect_data,
)

from dash import callback, Output, Input, State
from dash.exceptions import PreventUpdate

# Use environment variable for Docker or create in a writable location
cache_dir = os.environ.get('CACHE_DIR',
                          os.path.join(os.environ.get('TMPDIR', '/tmp'), 'dash_cache', 'plot_cache'))

# Try to create the cache directory, fallback to temp if permissions fail
try:
    os.makedirs(cache_dir, exist_ok=True)
    # Test write permissions
    test_file = os.path.join(cache_dir, 'test_write.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print(f"[INFO] Using cache directory: {cache_dir}")
except (OSError, PermissionError) as e:
    print(f"[WARNING] Cannot create cache directory at {cache_dir}: {e}")
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), 'dash_cache', 'plot_cache')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[INFO] Using fallback cache directory: {cache_dir}")

plot_cache = dc.Cache(cache_dir)

# Data needed by the UMAP computation, kept separate from grammar.py's UI globals.
# grammar.py imports these back so existing call sites elsewhere keep working.
GrammarItemsCols = retrieve_data.getGrammarItemsCols()
GrammarItemsColsPairs = retrieve_data.getGrammarItemsCols("item_pairs")
grammarMeta = retrieve_data.getGrammarMeta()
Informants = retrieve_data.getInformantDataGrammar(imputed=True)

item_presets = generate_dynamic_presets(grammarMeta)
labels_dict = build_preset_multiselect_data(item_presets)


def create_plot_cache_key(participants, items, n_neighbours, min_dist, distance_metric, standardize, densemap, dens_lambda, pairs, regional_mapping=False, include_ai=False, umap_3d=False, umap_4d=False):
    """Create a unique cache key for plot parameters"""
    key_data = {
        'participants': sorted(participants) if participants else 'all',
        'items': sorted(items) if items else 'all', 
        'n_neighbours': n_neighbours,
        'min_dist': min_dist,
        'distance_metric': distance_metric,
        'standardize': standardize,
        'densemap': densemap,
        'dens_lambda': dens_lambda if densemap else None,
        'pairs': pairs,
        'regional_mapping': regional_mapping,
        'include_ai': include_ai,
        'umap_3d': umap_3d,
        'umap_4d': umap_4d
    }
    key_string = str(key_data)
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cached_umap_plot(participants, items, n_neighbours, min_dist, distance_metric, standardize, densemap, dens_lambda, pairs, informants=None, regional_mapping=False, include_ai=False, umap_3d=False, umap_4d=False):
    """Get UMAP plot (and quality metrics) from cache or compute if not exists.
    Returns (figure, metrics_dict)."""
    cache_key = f"umap_{create_plot_cache_key(participants, items, n_neighbours, min_dist, distance_metric, standardize, densemap, dens_lambda, pairs, regional_mapping, include_ai, umap_3d, umap_4d)}"
    
    cached = plot_cache.get(cache_key)
    if cached is not None:
        # Support old cached entries that stored just the figure
        if isinstance(cached, tuple):
            return cached
        return cached, {}
    
    # Use provided informants or fall back to module-level Informants
    if informants is None:
        informants_df = Informants
    else:
        # Convert from dict if needed
        if isinstance(informants, list):
            informants_df = pd.DataFrame(informants)
        else:
            informants_df = informants
    
    # Not in cache, compute it
    # Get data filtered by participants to ensure cache consistency
    if not pairs:
        grammarData = retrieve_data.getGrammarData(imputed=True, participants=participants, columns=items, regional_mapping=regional_mapping, include_ai=include_ai)
        grammarCols = GrammarItemsCols
    else:
        grammarData = retrieve_data.getGrammarData(imputed=True, participants=participants, columns=items, pairs=True, regional_mapping=regional_mapping, include_ai=include_ai)
        grammarCols = GrammarItemsColsPairs

    plot, quality_metrics = getUMAPplot(
        grammarData=grammarData,
        GrammarItemsCols=grammarCols,
        informants=informants_df,
        selected_informants=participants,
        items=items,
        n_neighbours=n_neighbours,
        min_dist=min_dist,
        distance_metric=distance_metric,
        standardize=standardize,
        densemap=densemap,
        dens_lambda=dens_lambda,
        pairs=pairs,
        regional_mapping=regional_mapping,
        umap_3d=umap_3d,
        umap_4d=umap_4d
    )
    
    plot_cache.set(cache_key, (plot, quality_metrics))
    return plot, quality_metrics


def normalize_tree_selection(selected_informants, selected_items):
    """
    Normalize tree selections - if only top-level is selected, expand to all items.
    
    Args:
        selected_informants: List of selected informant IDs or ['informants']
        selected_items: List of selected item IDs or ['grammaritems']
    
    Returns:
        Tuple of (normalized_informants, normalized_items)
    """
    if selected_informants == ['informants']:
        selected_informants = Informants['InformantID'].values.tolist()
    if selected_items == ['grammaritems']:
        selected_items = GrammarItemsCols
    return selected_informants, selected_items


# Callback 3: Background UMAP computation
@callback(	
    [Output('grammar_plots_UMAP', 'data', allow_duplicate=True),
    Output('UMAPgroup', 'data', allow_duplicate=True),
    Output('UMAPparticipants','data'),
    Output('UMAPitems','data'),
    Output('UMAPGroupsForRF', 'data', allow_duplicate=True),
    Output('umap-render-settings', 'data'),
    Output('umap-quality-metrics', 'data', allow_duplicate=True)],
    Input('umap-render-trigger', 'data'),
    [State("participantsTree", "checked"),
    State("grammarItemsTree", "checked"),
    State('UMAP_neighbours','value'),
    State('UMAP_mindist','value'),
    State('grammar-items-preset', 'value'), 
    State('umap-distance-metric-dropdown', 'value'), 
    State('umap-standardize-checkbox', 'value'),
    State('umap-densemap-checkbox', 'checked'),
    State('umap-dens-lambda-slider', 'value'),
    State('umap-3d-checkbox', 'checked'),
    State('umap-4d-checkbox', 'checked'),
    State('grammar-type-switch', 'checked'), 
    State('use-imputed-data-switch', 'checked'),
    State('informants-store', 'data'),  # Add informants store
    State('england-mapping-param', 'data'),  # Add england mapping parameter
    State('include-ai-param', 'data')],  # Add AI participants parameter
    prevent_initial_call=True,
    background=True,
    running=[(Output("grammar_running","data"),True,False)]
)
def compute_umap_background(trigger_data, selected_informants, items, n_neighbours, 
                           min_dist, selected_presets, distance_metric, 
                           standardize_participant_ratings, densemap, dens_lambda, umap_3d, umap_4d, pairs, use_imputed, informants_data, regional_mapping, include_ai):
    """Compute UMAP in background - this is the slow operation"""
    if trigger_data is None:
        raise PreventUpdate
    
    import hashlib
    
    def _hash_list(lst):
        return hashlib.md5(str(sorted(lst)).encode()).hexdigest() if lst else "all"
    
    # If presets were selected, expand them to item codes
    preset_items = None
    if selected_presets and isinstance(selected_presets, (list, tuple)):
        preset_items = expand_presets_to_items(selected_presets, item_presets)

    # Normalize tree selections using helper function. If preset_items is provided, use those instead of tree items
    if preset_items:
        selected_informants, items = normalize_tree_selection(selected_informants, preset_items)
    else:
        selected_informants, items = normalize_tree_selection(selected_informants, items)
    
    # Reset group counter when plot is re-rendered
    data = 0
    
    # Use cached UMAP plot generation for better performance
    figure, quality_metrics = get_cached_umap_plot(
        participants=selected_informants,
        items=items,
        n_neighbours=n_neighbours,
        min_dist=min_dist,
        distance_metric=distance_metric,
        standardize=standardize_participant_ratings,
        densemap=densemap,
        dens_lambda=dens_lambda if dens_lambda is not None else 2.0,
        pairs=pairs,
        informants=informants_data,
        regional_mapping=regional_mapping,
        include_ai=include_ai,
        umap_3d=umap_3d,
        umap_4d=umap_4d
    )
    groupsCache = getColorGroupingsFromFigure(figure)
    
    render_settings = {
        "pairs": pairs,
        "use_imputed": True  # UMAP always uses imputed data
    }
    
    return figure, data, selected_informants, items, groupsCache, render_settings, quality_metrics
