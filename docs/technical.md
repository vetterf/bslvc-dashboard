# Technical Documentation

This document provides technical details about the BSLVC Dashboard and all available functions.

## Table of Contents

1. [Key Components](#key-components)
2. [Caching System](#caching-system)
3. [URL Parameters](#url-parameters)

---


## Key components

The heart of the dashboard are the analysis modules for the grammar and the lexical data. Currently, only the grammar analysis module is finished. In what follows, you will find a description of all available options in the Dashboard.


### Grammar Sets: Interface Elements & Buttons

Below is a description of all major buttons and UI elements in the Grammar Sets module:

#### Main Tabs
- **Plot View**: Shows either the dimensionality reduction plot and the group comparison plot, or item ratings plot.
Note: The group comparison plot uses the same settings that were used for the dimensionality reduction. Any changes that have been applied to the settings in the UI are not taken into account. This is to ensure that the group comparison is linked to the dimensionality reduction plot.
- **Sociodemographic Details**: Displays participant information and summary plots. These plots update automatically upon changing settings in the UI.
- **Grammar Items**: Shows a table of all grammar items and metadata. This table supports dynamic filtering, and can also be used for modifying the selected items in the grammar items tree.

#### Analysis Type Selector
- **SegmentedControl**: Switch between "Participant Similarity" and "Item Ratings" plots. 

In the mode "Participant Similarity", the dashboard applies dimensionality reduction to the selected data using Uniform Manifold Approximation and Projection (UMAP; https://umap-learn.readthedocs.io/en/latest/). Additionally, users can compare groups in the plot. This can be done either by selectively hiding varieties, or by using the lasso tool to create custom groups. For detailed instructions, see the instructions for [custom group comparison](#custom-group-comparison).

In the mode "Item Ratings", users can compare the ratings of specific features. This mode is best suited for the comparison of a relatively small number of features. Some plot modes have limits of how much items and varieties can be displayed. Should you hit such a limit, the dashboard issues a warning notification. 

#### Primary Action
- **Render Plot**: Main button to generate the selected plot.

#### UMAP Group Buttons (visible in "Participant Similarity" mode)
These buttons facilitate custom group comparisons or excluding outliers from the selection. They all require a lasso selection of data points via the lasso tool in the mode bar at the top of the participant similarity plot as input. 

The "Add Group" and "Clear Group" buttons override the default variety-based grouping of the "Compare Selected Groups" function. More detailed instructions can ben found in the section [custom group comparison](#custom-group-comparison).

The "Deselect Lasso Selection" and "Select Only Lasso Selection" buttons modify the selection in the participants tree. These functions are helpful for deselecting outliers visible in the dimensionality reduction.

- **Add Group**: Add a group of participants for comparison.
- **Clear Groups**: Remove all defined groups.
- **Select Only Lasso Selection**: Select only participants currently selected via lasso tool.
- **Deselect Lasso Selection**: Deselect participants selected via lasso tool.
- **Compare Selected Groups**: Render a Random Forest plot comparing selected groups.

#### Participant Selection
Participants can be selected via the participant tree, either by clicking the checkboxes in the tree, by using presets or by using the buttons. Additionally, users can use the "Advanced Filters" section to select participants based on their sociodemographic details.  

- **Select All**: Select all participants in the tree.
- **Deselect All**: Deselect all participants.
- **Quick Selection**: Batch select by variety, age, gender, completeness, or balanced groups. Available presets for common selections:
    - **ENL**: Select all L1 varieties (England, Scotland, US)
    - **ESL**: Select all L2 varieties (Gibraltar, India, Malta, Puerto Rico)
    - **EFL**: Select all EFL varieties (Germany, Slovenia, Sweden)
    - **Balanced**: Select a balanced sample across all varieties
    - **Age Groups**: Select participants from specific age groups
    - **Gender**:
        - Female: Select all female participants
        - Male: Select all male participants
        - Balanced: Select a gender-balanced sample across all varieties
        - Balanced per Variety: Select a gender-balanced sample within each variety
    - **No missing values**: Select participanty without missing values
- **Advanced Filters**: Filter participants by gender, age, variety ratio, and main variety.
- **Apply Filters**: Apply selected participant filters. This overrides the current selection in the participant tree.

#### Grammar Item Selection
All grammar items can be selected in the grammar items tree. They are first grouped by mode, then by feature group. Items can be either selected by clicking the checkboxes, by using presets or the buttons above the tree, or via the grammar items table in the main view. The getting started section in the app describes how the grammar items table can be used for custom item selections.
- **Select All**: Select all grammar items.
- **Deselect All**: Deselect all grammar items.
- **Problematic**: Deselect problematic items. See [problematic items](#problematic-items) for more information. Currently the following items are flagged as problematic: 'M19', 'J23', 'C14' and 'A4' for use of currency; 'E22', 'D12' and 'E6' for different renderings of the item in some varieties.  
- **Select a Preset**: Multi-select dropdown for item presets (mode, feature groups, eWAVE feature groups).
- **Advanced Options**:
    - **Use item difference (spoken-written)**: Toggle to use difference between item pairs. Most items feature in the spoken and the written section of the BSLVC. These features are referred to as *twin items*. If this switch is set to true, the interface calculates the difference of the ratings for twin items (spoken - written) for each participant and uses this value for all subsequent plots. Naturally, items that feature only in the written section are excluded.
    - **Use imputed data**: Toggle between imputed and raw data. The dimensionality reduction always uses the imputed data, as UMAP cannot handle missing values. A description of the imputation process can be found in the supplementary BSLVC resources repository (https://github.com/vetterf/BSLVC-resources; available soon!). Imputation was performed with a random forest approach.
    - **Toggle Written-Only**: Toggle items which feature only in the written section.
    - **Currency/Unit**: Toggle currency/unit items.

#### UMAP Settings
- **Color**: Select coloring variable (Variety, Variety type, Gender). This setting does not trigger a rerender of the plot and can be changed after rendering the plot.
- **Distance metric**: Choose metric (Cosine, Euclidean, Manhattan).
- **Standardize participant ratings**: Checkbox to standardize ratings. Standardization is advised for use with Euclidean and Manhattan distances.
- **Use density-preserving embedding (DensMAP)**: Checkbox for DensMAP (https://umap-learn.readthedocs.io/en/latest/densmap_demo.html). By default, UMAP does not preserve densities of clusters well. DensMAP tries to preserve the density of clusters when reducing dimensionality.
- **Number of neighbours**: Slider for UMAP hyperparameter. Check the UMAP docs for more info.
- **Minimal distance**: Slider for UMAP hyperparameter. Check the UMAP docs for more info.

#### Item Plot Settings
- **Plot mode**: Select plot type for visualizing grammar items:
    - **Mean (95% CI)**: Plot mean values of features with confidence intervals
    - **Mean (95% CI - split varieties)**: Plot mean values of features with confidence intervals. Each variety is represented separately on the y-axis. This helps to avoid overplotting.
    - **Diverging stacked bars**: Show the distribution of ratings for each feature in a diverging stacked barchart.
    - **Informant mean of selected features (boxplot)**: Calculates a mean rating across all selected features for each participant and displays the distribution of these means in a boxplot.
    - **Correlation matrix**: Displays a correlation matrix showing the pairwise correlations between the selected features.
    - **Missing values heatmap**: Displays a heatmap indicating the presence of missing values across the selected features.
- **Group by**: Select grouping variable (Variety, Variety type, Gender).
- **Sort by**: Sort items by mean, SD, or alphabetically.

#### Group Comparison Settings
This is only available if the mode is set to "Participant Similarity". These functions can be used to filter out items that fall outside a certain rating rage, or to apply standardization of the ratings before training the random forest.
- **Filter by Average Rating**: Range slider to filter items by group average.
- **Use Z-Scores**: Checkbox to standardize ratings (participant-wise) before training the random forest.

#### Advanced Actions
- **Export Data**: Download current data selection. By default, this includes all sociodemographic columns, but only the selected participants and grammar items.
- **Copy Settings**: Copy current settings to clipboard. This function converts the current settings into a base64 code and copies it to the clipboard. This code can be used to share settings with other users or save the settings in a text file for later use.
- **Paste Settings**: Paste settings from clipboard. This button takes the base64 code created with the "Copy Settings" function to restore settings.
- **Save Settings**: Save current settings locally. Helpful for temporarily saving the settings.
- **Restore Settings**: Restore previously saved settings with the "Save Settings" function.

## Caching System

The BSLVC Dashboard implements a multi-layered caching strategy to optimize performance and reduce computation time for expensive operations. The caching system consists of three levels: server-side disk cache, in-memory Python cache, and browser-side storage.

### Server-Side Disk Cache

The dashboard uses **DiskCache** (via the `diskcache` library) for persistent storage of computationally expensive plot objects.

#### Cache Directory Configuration

```python
import diskcache as dc

# Cache directory with fallback mechanism
cache_dir = os.environ.get('CACHE_DIR', 
                          os.path.join(os.environ.get('TMPDIR', '/tmp'), 
                                      'dash_cache', 'plot_cache'))
plot_cache = dc.Cache(cache_dir)
```

**Default Locations:**
- **Development**: `/tmp/dash_cache/plot_cache`
- **Docker**: `/app/cache` (set via `CACHE_DIR` environment variable)
- **Fallback**: System temp directory if permissions fail

The cache directory is automatically created with write permission checks. If the default location is not writable, the system falls back to the system temp directory.

#### UMAP Plot Caching

UMAP (Uniform Manifold Approximation and Projection) computations are expensive and are cached based on input parameters:

```python
def get_cached_umap_plot(participants, items, n_neighbours, min_dist, 
                         distance_metric, standardize, densemap, pairs, 
                         informants=None, regional_mapping=False):
    """Get UMAP plot from cache or compute if not exists"""
    cache_key = f"umap_{create_plot_cache_key(...)}"
    
    cached_plot = plot_cache.get(cache_key)
    if cached_plot is not None:
        return cached_plot
    
    # Compute and cache
    plot = getUMAPplot(...)
    plot_cache.set(cache_key, plot)
    return plot
```

**Cache Key Components:**
- Participant list (sorted)
- Item list (sorted)
- UMAP hyperparameters (n_neighbours, min_dist, distance_metric)
- Standardization flag
- DensMAP flag
- Pairs mode flag
- Regional mapping flag

Cache keys are hashed using MD5 to create unique identifiers.

#### Random Forest Plot Caching

Random Forest comparison plots are also cached:

```python
def get_cached_rf_plot(data, importance_ratings, value_range, pairs, 
                       participants=None, split_by_variety=False):
    """Get RF plot from cache or compute if not exists"""
    key_data = {
        'data_shape': data.shape,
        'importance_ratings': importance_ratings,
        'value_range': value_range,
        'pairs': pairs,
        'participants': sorted(participants) if participants else None,
        'split_by_variety': split_by_variety,
        'data_hash': hashlib.md5(...).hexdigest()[:8]
    }
    cache_key = f"rf_{hashlib.md5(str(key_data).encode()).hexdigest()}"
    # ... cache logic
```

**Cache Invalidation:**
- Disk cache persists between sessions
- Cache is stored indefinitely (no expiration)
- Manual cache clearing requires deleting the cache directory

### In-Memory Python Cache (LRU)

The dashboard uses Python's `functools.lru_cache` decorator for in-memory caching of frequently accessed data.

#### Data Retrieval Functions

```python
@lru_cache(maxsize=2)
def get_grammar_data_cached(regional_mapping=False):
    return retrieve_data.getGrammarData(imputed=True, regional_mapping=regional_mapping)

@lru_cache(maxsize=2)
def get_grammar_data_pairs_cached(regional_mapping=False):
    return retrieve_data.getGrammarData(imputed=True, items=..., pairs=True, ...)

@lru_cache(maxsize=2)
def get_informants_cached(regional_mapping=False):
    return retrieve_data.getInformantDataGrammar(imputed=True, ...)

@lru_cache(maxsize=2)
def get_grammar_data_raw_cached(regional_mapping=False):
    return retrieve_data.getGrammarData(imputed=False, ...)

@lru_cache(maxsize=1)
def get_grammar_meta_cached():
    return retrieve_data.getGrammarMeta()

@lru_cache(maxsize=1)
def get_grammar_items_cols_cached():
    return retrieve_data.getGrammarItemsCols()
```

**LRU Cache Sizes:**
- **Grammar data**: maxsize=2 (for mapped/unmapped variants)
- **Metadata**: maxsize=1 (static data)
- **Item columns**: maxsize=1 (static data)

LRU (Least Recently Used) cache automatically evicts the least recently accessed items when the cache is full.

### Browser-Side Storage (Dash dcc.Store)

The dashboard leverages Dash's `dcc.Store` component for client-side data persistence with three storage types:

#### Memory Storage (`storage_type='memory'`)

Temporary storage cleared when page is refreshed:

```python
dcc.Store(id="UMAPgroup", storage_type="memory", data=0)
dcc.Store(id="UMAPparticipants", storage_type="memory", data=[])
dcc.Store(id="UMAPitems", storage_type="memory", data=[])
dcc.Store(id="UMAPGroupsForRF", storage_type="memory", data={"dataframe": ...})
dcc.Store(id="grammar_plots_UMAP", storage_type="memory", data=None)
dcc.Store(id="grammar_plots_item", storage_type="memory", data=...)
dcc.Store(id="leiden-cluster-data", storage_type="memory")
dcc.Store(id="umap-hoverinfo-store", storage_type="memory")
dcc.Store(id="umap-render-trigger", storage_type="memory")
dcc.Store(id="umap-render-settings", storage_type="memory")
dcc.Store(id="clipboard-settings-store", storage_type="memory")
```

**Use Cases:**
- Current UMAP group selections
- Active participant/item selections
- Plot data for current session
- Clustering results
- Temporary UI state

#### Session Storage (`storage_type='session'`)

Persists across page refreshes within the same browser tab/window:

```python
dcc.Store(id="last-rendered-item-plot", storage_type="session")
dcc.Store(id="last-rendered-umap-plot", storage_type="session")
dcc.Store(id="last-sociodemographic-settings", storage_type="session")
```

**Use Cases:**
- Last rendered plots (survive page refresh)
- Sociodemographic plot settings
- Session-specific cache

#### Local Storage (`storage_type='local'`)

Persists across browser sessions (stored in browser's localStorage):

```python
dcc.Store(id="saved-item-settings", storage_type="local")
dcc.Store(id="saved-umap-settings", storage_type="local")
```

**Use Cases:**
- User-saved settings
- Persistent user preferences
- Long-term storage across sessions

### UI Component Persistence

Many UI components use Dash's `persistence` and `persistence_type` properties:

```python
persistence_type = "memory"
persist_UI = True

# Example: UMAP settings sliders
dmc.Slider(
    id="UMAP_neighbours",
    value=25,
    persistence=persist_UI,
    persistence_type=persistence_type
)
```

**Persisted UI Elements:**
- UMAP hyperparameters (neighbours, min_dist)
- Distance metric selection
- Color/grouping selections
- Filter settings (gender, age, variety)
- Plot mode selections
- Advanced options (standardize, densemap, etc.)

**Persistence Type:** `"memory"` - UI state is preserved during the current browser session but cleared on page refresh.

### Cache Performance Benefits

**Without Cache:**
- UMAP computation: 5-30 seconds (depending on data size)
- Random Forest training: 2-10 seconds
- Data retrieval: 0.5-2 seconds

**With Cache:**
- UMAP retrieval: <100ms
- Random Forest retrieval: <50ms
- Data retrieval: <10ms

### Cache Management

**Clearing Caches:**

1. **Disk Cache**: Delete the cache directory
   ```bash
   rm -rf /tmp/dash_cache/plot_cache
   # or
   rm -rf /app/cache  # in Docker
   ```

2. **LRU Cache**: Restart the application (cache is in-memory only)

3. **Browser Storage**: 
   - Memory: Refresh the page
   - Session: Close the browser tab
   - Local: Clear browser data or use browser developer tools

**Environment Variables:**
- `CACHE_DIR`: Set custom cache directory path
- `TMPDIR`: Fallback temp directory (used if CACHE_DIR not set)

### How Caching Layers Work Together: UMAP Plot Rendering

When a user requests a UMAP plot, the three caching layers work in concert to minimize computation time. Here's the complete workflow:

#### Step 1: User Interaction
User selects participants, items, and UMAP settings (neighbors, distance metric, etc.), then clicks "Render Plot".

#### Step 2: Data Retrieval (LRU Cache)
```python
# First, retrieve data from SQLite database
grammarData = retrieve_data.getGrammarData(imputed=True, 
                                          participants=participants, 
                                          columns=items)
```

**LRU Cache Role:**
- `get_grammar_data_cached()` checks if full grammar dataset is already in memory
- If **cache hit**: Returns data instantly (~10ms)
- If **cache miss**: Queries SQLite database (~500-2000ms), stores result in LRU cache
- Subsequent requests with same `regional_mapping` parameter use cached data

**Why LRU for data?**
- Fast access (in-memory, no disk I/O)
- Limited memory footprint (maxsize=2 for mapped/unmapped variants)
- Automatic eviction prevents memory bloat
- Data is read-only and doesn't change during application runtime

#### Step 3: UMAP Computation (DiskCache)
```python
def get_cached_umap_plot(participants, items, n_neighbours, min_dist, ...):
    # Create unique cache key from all parameters
    cache_key = f"umap_{create_plot_cache_key(participants, items, 
                                               n_neighbours, min_dist, 
                                               distance_metric, standardize, 
                                               densemap, pairs, regional_mapping)}"
    
    # Check disk cache first
    cached_plot = plot_cache.get(cache_key)
    if cached_plot is not None:
        return cached_plot  # ~50-100ms
    
    # Cache miss - compute UMAP (expensive!)
    plot = getUMAPplot(grammarData, ...)  # 5-30 seconds
    
    # Store result on disk for future use
    plot_cache.set(cache_key, plot)
    return plot
```

**DiskCache Role:**
- Checks if this **exact combination** of parameters was computed before
- If **cache hit**: Loads pre-computed plot from disk (~50-100ms)
- If **cache miss**: Computes UMAP (5-30 seconds), saves to disk
- Cache persists across application restarts

**Why DiskCache for UMAP plots?**
- UMAP computation is extremely expensive (5-30 seconds)
- Plot objects are large (serialized Plotly figures with all trace data)
- Many possible parameter combinations (participants × items × settings)
- Results are deterministic - same inputs always produce same plot
- Persistence across sessions is valuable (users often revisit same views)

**Why not LRU cache for UMAP?**
- Would consume too much memory (each plot is several MB)
- LRU eviction would lose expensive computations too quickly
- Clears on application restart (losing valuable cached results)

#### Step 4: Browser Storage (dcc.Store)
```python
# After UMAP is rendered, store in browser session
@callback(
    Output('last-rendered-umap-plot', 'data'),
    Input('UMAPfig', 'figure'),
    prevent_initial_call=True
)
def save_umap_plot(fig):
    """Save UMAP plot to session storage"""
    return fig

# Restore plot when switching between plot types
@callback(
    [Output('ItemFig', 'figure', allow_duplicate=True),
     Output('UMAPfig', 'figure', allow_duplicate=True)],
    Input('grammar-plot-type', 'value'),
    [State('last-rendered-item-plot', 'data'),
     State('last-rendered-umap-plot', 'data')],
    prevent_initial_call=True
)
def restore_plots_on_type_change(plot_type, saved_item_plot, saved_umap_plot):
    """Restore last rendered plot when switching types"""
    if plot_type == 'item' and saved_item_plot:
        return saved_item_plot, no_update
    elif plot_type == 'umap' and saved_umap_plot:
        return no_update, saved_umap_plot
    return no_update, no_update
```

**Browser Storage Role:**
- Stores the most recent UMAP plot data in browser's sessionStorage
- When user switches between "Participant Similarity" and "Item Ratings" views, the last rendered plot is restored instantly
- Survives page refreshes within the same browser session

**Why Browser Storage?**
- Instant restoration when switching plot types (~0ms)
- No server round-trip or re-rendering needed
- Session-specific (different tabs maintain independent state)

**Important Note:** This only restores plots when switching between plot types in the same session, not when switching browser tabs or windows. The plot must be re-rendered if the page is refreshed or the application is reopened.

#### Complete Flow Example

**Scenario 1: First-time UMAP request**
1. User selects 100 participants, 50 items, 25 neighbors
2. **LRU Cache**: Miss → Query database (2000ms) → Cache full dataset
3. **DiskCache**: Miss → Compute UMAP (15000ms) → Save to disk
4. **Browser Store**: Save plot to sessionStorage
5. **Total time**: ~17 seconds

**Scenario 2: Same request later in session**
1. User selects same 100 participants, 50 items, 25 neighbors
2. **LRU Cache**: Hit → Return data (10ms)
3. **DiskCache**: Hit → Load plot from disk (80ms)
4. **Browser Store**: Save plot to sessionStorage
5. **Total time**: ~90ms (189× faster!)

**Scenario 3: Slightly different request (26 neighbors instead of 25)**
1. User changes only the n_neighbors parameter to 26
2. **LRU Cache**: Hit → Return data (10ms) - data hasn't changed
3. **DiskCache**: Miss → New cache key, compute UMAP (15000ms) → Save to disk
4. **Browser Store**: Save new plot to sessionStorage
5. **Total time**: ~15 seconds (LRU saved the database query time)

**Scenario 4: Switching between plot types**
1. User switches from UMAP view to Item Ratings view, then back to UMAP
2. **Browser Store**: Hit → Restore UMAP plot from sessionStorage (0ms)
3. **Total time**: Instant (no server communication or re-rendering)

**Note:** Page refresh or closing/reopening the browser tab will clear the plot and require re-rendering.

#### Why Three Cache Layers?

**Each layer optimizes a different bottleneck:**

| Layer | Optimizes | Speed | Persistence | Size Limit | Use Case |
|-------|-----------|-------|-------------|------------|----------|
| **LRU (Memory)** | Database queries | Fastest (~10ms) | Until restart | Small (2 items) | Static datasets |
| **DiskCache** | UMAP computation | Fast (~100ms) | Permanent | Large (GB) | Expensive computations |
| **Browser Store** | Plot type switching | Instant (0ms) | Session only | Medium (MB) | UI state, active plots |

**Cache Synergy:**
- **LRU + DiskCache**: Even on DiskCache miss, LRU provides data quickly for computation
- **DiskCache + Browser**: DiskCache serves across sessions; Browser optimizes plot type switching within a session
- **All three**: Maximum speed with minimal redundant computation and rendering

#### Cache Hit Rate Optimization

**High hit rates expected for:**
- Common parameter combinations (default settings)
- Repeated analysis of same participant groups
- Users exploring variations (changing one parameter at a time)

**Low hit rates expected for:**
- First-time users
- Highly customized parameter combinations
- Random exploration of many different participant/item sets

**Memory vs. Disk Trade-off:**
- LRU cache is small (only 2 datasets) but ultra-fast
- DiskCache is large (unlimited) but requires disk I/O
- This combination minimizes both memory usage and computation time

### Best Practices

1. **Development**: Cache directory in `/tmp` is automatically cleaned by the OS
2. **Production**: Use persistent volume mount for `/app/cache` in Docker
3. **Testing**: Clear cache between major data updates or code changes
4. **Monitoring**: Check cache directory size periodically (can grow large with many unique UMAP combinations)

---

**Last Updated**: November 2025  
**Version**: 0.1.2
