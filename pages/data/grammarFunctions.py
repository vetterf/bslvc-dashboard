import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, dash_table, html,ctx, callback, Output, Input, State, clientside_callback, no_update
import pages.data.retrieve_data as retrieve_data
import umap
import pickle
from dash.exceptions import PreventUpdate
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import polars as pl
import pyarrow as pa
from dash_iconify import DashIconify
import dash_ag_grid as dag
import dash_mantine_components as dmc
import plotly.figure_factory as ff
import os
import hashlib
# Add these imports for Leiden clustering
import igraph as ig
import leidenalg as la
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score
import seaborn as sns

# all symbols with -open; used for grouping data points in the UMAP plot
symbols = [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
colorMap = px.colors.qualitative.Dark24

# ============================================================================
# CENTRALIZED VARIETY ORDERING AND CLASSIFICATION
# ============================================================================

def get_variety_classification():
    """
    Central definition of variety types and their ordering.
    Returns a dictionary with variety mappings and ordered lists.
    """
    return {
        "ENL": ["England", "Scotland", "US"],
        "ESL": ["Gibraltar", "India", "Malta", "Puerto Rico"], 
        "EFL": ["Germany", "Slovenia", "Sweden"]
    }

def get_variety_mapping():
    """Get variety to type mapping (e.g., 'England' -> 'ENL')"""
    classification = get_variety_classification()
    mapping = {}
    for vtype, varieties in classification.items():
        for variety in varieties:
            mapping[variety] = vtype
    return mapping

def get_ordered_varieties():
    """Get all varieties in the standard order: ENL, ESL, EFL (alphabetical within each type)"""
    classification = get_variety_classification()
    ordered = []
    for vtype in ["ENL", "ESL", "EFL"]:
        ordered.extend(classification[vtype])
    return ordered

def sort_varieties_by_standard_order(varieties):
    """Sort a list of varieties according to the standard ordering"""
    standard_order = get_ordered_varieties()
    # Create a mapping of variety to its position in standard order
    order_map = {variety: idx for idx, variety in enumerate(standard_order)}
    # Sort varieties by their position in standard order, unknown varieties go to end
    return sorted(varieties, key=lambda x: order_map.get(x, 999))

def sort_groups_for_plot(groups, groupby="variety"):
    """
    Sort groups according to appropriate ordering rules.
    For varieties, use standard variety order. For other groups, use alphabetical.
    """
    if groupby == "variety":
        # These are actual variety names - use standard ordering
        return sort_varieties_by_standard_order(groups)
    elif groupby in ["vtype", "vtype_balanced"]:
        # These are variety types (ENL, ESL, EFL) - use standard type order
        type_order = ["ENL", "ESL", "EFL"]
        return sorted(groups, key=lambda x: type_order.index(x) if x in type_order else 999)
    else:
        # For gender or other categories, use alphabetical
        return sorted(groups)

def getEmptyPlot(msg="No data points remained after applying the filter. Please check the selected range."):
        # empty figure that says you should check the selected range
        fig = go.Figure()
        fig.add_annotation(
        text=msg,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20)
         )
        fig.update_layout(template="simple_white")
        return fig

def find_non_zero_positions_values_and_ids(symbols, ids):
    # Get positions, values and corresponding IDs for curves in scatterplot
    # This function is a helper function of getGroupingsFromFigure
    result = [(i, symbol, ids[i]) for i, symbol in enumerate(symbols) if symbol != 0]
    return result


def getGroupingsFromFigure(figure):
    # function is used to get the groupings in the scatter plot to the auxiliary plots / random forests
    marked_points = []
    
    # Handle None or empty figure
    if figure is None or 'data' not in figure:
        return marked_points
    
    for trace in figure['data']:
        if 'marker' in trace and 'symbol' in trace['marker'] and 'opacity' in trace['marker']:
            # Only consider markers with opacity > 0
            marker_opacity = trace['marker']['opacity']
            # marker_opacity can be a single value or a list
            if isinstance(trace['marker']['symbol'], list):
                symbols = trace['marker']['symbol']
                ids = trace['ids']
                # If opacity is a list, filter by each point; if scalar, apply to all
                if isinstance(marker_opacity, list):
                    filtered = [
                        (i, sym, id_)
                        for i, (sym, id_, op) in enumerate(zip(symbols, ids, marker_opacity))
                        if op > 0
                    ]
                else:
                    if marker_opacity > 0:
                        filtered = [
                            (i, sym, id_)
                            for i, (sym, id_) in enumerate(zip(symbols, ids))
                        ]
                    else:
                        filtered = []
                if filtered:
                    grouped_points = [(pos, sym, id_) for pos, sym, id_ in filtered if sym !=0]
                    marked_points.append(grouped_points)
    # concat to dataframe
    if marked_points:
        df = pd.concat([pd.DataFrame(group, columns=['position', 'symbol', 'id']) 
                       for group in marked_points], ignore_index=True)
    else:
        df = pd.DataFrame(columns=['position', 'symbol', 'id'])
    return {"dataframe": df.to_dict("records")}
"""
def getGroupingsFromFigure(figure):
    # function is used to get the groupings in the scatter plot to the auxiliary plots / random forests
    marked_points = []
    
    # Handle None or empty figure
    if figure is None or 'data' not in figure:
        return {"dataframe": []}
    
    for trace in figure['data']:
        if 'marker' in trace and 'symbol' in trace['marker']:
            if isinstance(trace['marker']['symbol'], list):
                symbols = trace['marker']['symbol']
                ids = trace['ids']
                grouped_points = find_non_zero_positions_values_and_ids(symbols, ids)
                marked_points.append(grouped_points)
    # comcat to dataframe
    df = pd.concat([pd.DataFrame(group, columns=['position', 'symbol', 'id']) 
                   for group in marked_points], ignore_index=True)
    return {"dataframe":df.to_dict("records")}
"""
def getColorGroupingsFromFigure(figure):
    # get groupings based on varietes from a figure.
    # assumes that there are no groupings in the figure
    curves_info = []
    
    # Handle None or empty figure
    if figure is None or 'data' not in figure:
        return curves_info
    
    # Check if the figure contains subplots
    if 'data' in figure:
        traces = figure['data']
        
        for trace in traces:
            # Only process scatter-type traces
            if trace['type'] == 'scatter' and 'marker' in trace:
                # Extract info from each trace
                if 'marker' in trace and 'color' in trace['marker']:
                    color = trace['marker']['color']
                    color = color[0] if isinstance(color, tuple) else color
                    ids = trace['ids'],
                    ids = ids[0] if isinstance(ids, tuple) else ids
                    name = [trace['name']] * len(color)
                    curve_info = pd.DataFrame({'color': color, 'ids': ids, 'name': name})
                    curves_info.append(curve_info)
    
    # Convert to DataFrame
    if curves_info:
        df = pd.concat(curves_info, ignore_index=True)
    else:
        df = pd.DataFrame(columns=['color', 'ids', 'name'])
        
    return {"dataframe": df.to_dict("records")}

def getAuxiliaryPlot(Informants,participants,items):
    # histogram of selected data points in UMAP plot;
    # to do: mode switch for groups vs age/gender selection
    if len(participants) == 0:
        fig = getEmptyPlot()
        return fig
    #data = retrieve_data.getInformantDataGrammar(participants=participants, imputed=True)
    data = Informants.copy(deep=True)
    data = data.loc[data['InformantID'].isin(participants)]
    data['Age'] = data['Age'].astype(float)
    # distribution of age and gender
    fig = px.histogram(data, x='Age', color='Gender', title='Histogram: Age/Gender', barmode="overlay", range_x=[0, 60])
    fig.update_layout(template="simple_white")
    return fig

def performLeidenClustering(data, n_neighbors=15, resolution=0.5, mode="connectivity",metric="cosine"):
    """
    Perform Leiden clustering on the data
    
    Args:
        data (pd.DataFrame): The filtered grammar data
        n_neighbors (int): Number of neighbors for k-NN graph construction
        resolution (float): Resolution parameter for Leiden clustering
    
    Returns:
        np.array: Cluster assignments for each data point
    """
    # Create k-NN graph
    knn_graph = kneighbors_graph(data, n_neighbors=n_neighbors, mode=mode,metric=metric)
    
    # Convert to igraph
    sources, targets = knn_graph.nonzero()
    g = ig.Graph()
    g.add_vertices(len(data))
    g.add_edges(list(zip(sources, targets)))
    
    # Perform Leiden clustering
    #partition = g.community_leiden(resolution=resolution)
    partition = la.find_partition(g, la.ModularityVertexPartition)
    return partition.membership

def getUMAPplot(grammarData, GrammarItemsCols, leiden=False, distance_metric='cosine',pairs=False, **kwargs):
    """
    Generate UMAP plot for grammar data with Leiden clustering
    
    Args:
        grammarData (pd.DataFrame): Grammar data
        GrammarItemsCols (list): List of grammar item columns
        distance_metric (str): Distance metric for UMAP ['euclidean','manhattan','cosine']
        **kwargs: Additional parameters including leiden_resolution
    """

    
    # to do: return no_update when no data is selected
        #data = retrieve_data.getGrammarData(imputed=True, items=items, participants=participants)
    data = grammarData
    items=[]
    selected_informants=[]
    ratio_as_opacity=False
    dataCols = GrammarItemsCols
    n_neighbours = 15
    min_dist = 0.1
    force_rerender = False
    leiden_resolution = 0.8
    standardize=False
    densemap=False
    # filter data
    for key, value in kwargs.items():
        if key == 'items':
            # filter data by grammatical items, supplied is a list of column names
            items = value
        if key == 'informants':
            informants = value
            informants = informants['InformantID'].to_list()
        if key == 'selected_informants':
            selected_informants = value
        if key == 'ratio_as_opacity':
            ratio_as_opacity = value
        if key == 'n_neighbours':
            n_neighbours = value
        if key == 'min_dist':
            min_dist = value
        if key == 'force_rerender':
            force_rerender = value
        if key == 'leiden_resolution':
            leiden_resolution = value
        if key == 'distance_metric':
            distance_metric = value
        if key == 'standardize':
            standardize = value
        if key == 'densemap':
            densemap = value


    # Determine which informants to use for UMAP calculation
    if selected_informants and len(selected_informants) > 0:
        # Use only selected informants for UMAP calculation
        umap_informants = selected_informants
        data = data.loc[data['InformantID'].isin(selected_informants)].reset_index()
    else:
        # If no specific selection, use all informants from the dataset
        umap_informants = informants
        data = data.loc[data['InformantID'].isin(informants)].reset_index()

    # Try to load prerendered UMAP plot if available
    preset_dir = os.path.join(os.path.dirname(__file__), "umap_presets")
    def _hash_list(lst):
        return hashlib.md5(str(sorted(lst)).encode()).hexdigest() if lst else "all"
    umap_informants_hash = _hash_list(umap_informants) # Cache based on actual informants used for UMAP
    items_hash = _hash_list(items)
    # --- include distance_metric in the hash/filename ---
    preset_filename = f"umap_{umap_informants_hash}_{items_hash}_{n_neighbours}_{min_dist}_{distance_metric}_{standardize}_{densemap}.pkl"
    preset_path = os.path.join(preset_dir, preset_filename)
    if os.path.exists(preset_path) and not force_rerender:
        try:
            with open(preset_path, "rb") as f:
                fig = pickle.load(f)
                # Since the cache is now based on the actual informants used, 
                # we can directly return the figure with full opacity
                for trace in fig['data']:
                    marker = trace['marker']
                    if isinstance(marker['opacity'], list):
                        marker['opacity'] = [0.8] * len(marker['opacity'])
                    else:
                        marker['opacity'] = 0.8
                    trace['marker'] = marker
            return fig
        except Exception:
            pass  # If loading fails, fall back to normal rendering
    
    if len(items)>0:
        dataCols = [item for item in dataCols if item in items]
    filtered_data = data.loc[:,dataCols]

    if standardize:
        # Standardize (Z-score) each participant's data row-wise
        filtered_data = filtered_data.apply(pd.to_numeric, errors='coerce')
        # Compute row-wise mean and std
        row_means = filtered_data.mean(axis=1)
        row_stds = filtered_data.std(axis=1).replace(0, 1).fillna(1)
        # Standardize row-wise (Z-score)
        filtered_data = filtered_data.sub(row_means, axis=0)
        filtered_data = filtered_data.div(row_stds, axis=0)

    # run umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbours,
        min_dist=min_dist,
        metric=distance_metric,
        n_jobs=-1,
        low_memory=False,
        densmap=densemap
    )
    embedding = reducer.fit_transform(filtered_data)


    # to dataframe for plotting
    embedding = pd.DataFrame(embedding, columns=['x', 'y'])

    embedding['CountryCollection'] = data['CountryCollection']
    embedding['MainVariety'] = data['MainVariety']
    embedding['InformantID'] = data['InformantID']
    embedding['Year'] = data['Year']
    embedding['Gender'] = data['Gender']
    embedding['RatioMainVariety'] = data['RatioMainVariety']
    embedding['YearsLivedInMainVariety'] = data['YearsLivedInMainVariety']
    embedding['Age'] = data['Age']


    # Assign an ascending number to each unique value of column "CountryCollection"
    #country_mapping = {country: idx for idx, country in enumerate(embedding['MainVariety'].unique())}
    #embedding['color_int'] = embedding['MainVariety'].map(country_mapping)
    
    # Create a color sequence for each unique CountryCollection
    #unique_countries = embedding['MainVariety'].unique()
    #color_sequence = px.colors.qualitative.Dark24[:len(unique_countries)]
    #color_sequence = px.colors.qualitative.Plotly[:len(unique_countries)]
    #color_map = {country: color_sequence[idx] for idx, country in enumerate(unique_countries)}
    #embedding['color'] = embedding['MainVariety'].map(color_map)
    #embedding['color'] = embedding['MainVariety'].apply(lambda variety: retrieve_data.get_color_for_variety(variety, type="grammar"))

    variety_to_color = retrieve_data.get_color_for_variety(type="grammar")

    # Apply the mapping efficiently
    embedding['color'] = embedding['MainVariety'].map(variety_to_color)
    
    if leiden:
        embedding['leiden_cluster'] = leiden_clusters
            # Perform Leiden clustering
        leiden_clusters = performLeidenClustering(filtered_data, n_neighbors=n_neighbours, resolution=leiden_resolution)
        # Use Leiden clusters for symbols instead of varieties
        unique_clusters = sorted(set(leiden_clusters))
        symbol_sequence = symbols[:len(unique_clusters)]
        cluster_symbol_map = {cluster: symbol_sequence[idx] for idx, cluster in enumerate(unique_clusters)}
        embedding['symbol'] = embedding['leiden_cluster'].map(cluster_symbol_map)
    # Create subplots: main UMAP plot and confusion matrix
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('UMAP: Informant similarity with Leiden clusters', 'Confusion Matrix: Varieties vs Clusters'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}]]
        )
    else:
        fig = go.Figure()
        


    # else:
    embedding['opacity'] = 0.8



    # Add UMAP traces - sort varieties alphabetically for consistent legend order
    for c in sorted(embedding['MainVariety'].unique()):
        df_color = embedding[embedding['MainVariety'] == c]
        if leiden:
            fig.add_trace(
                go.Scatter(
                    x=df_color['x'], 
                    y=df_color['y'],
                    name=c,
                    mode='markers',
                    text=df_color['InformantID'],
                    ids=df_color['InformantID'],
                    marker=dict(color=df_color['color'], size=5, opacity=0.8, symbol=df_color['symbol']), 
                    showlegend=True, 
                    customdata=df_color[['InformantID','MainVariety','Age','Gender','RatioMainVariety','YearsLivedInMainVariety','CountryCollection','Year','leiden_cluster']],
                    hovertemplate='<br>'.join([
                        'InformantID: %{customdata[0]}',
                        'Main Variety: %{customdata[1]}',
                        'Age: %{customdata[2]}',
                        'Gender: %{customdata[3]}',
                        'Ratio (Main Variety): %{customdata[4]}',
                        'Years lived in (Main Variety): %{customdata[5]}',
                        'CountryCollection: %{customdata[6]}',
                        'Year: %{customdata[7]}',
                        'Leiden Cluster: %{customdata[8]}',
                    ]), hoverinfo='text'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df_color['x'], 
                    y=df_color['y'],
                    name=c,
                    mode='markers',
                    text=df_color['InformantID'],
                    ids=df_color['InformantID'],
                    marker=dict(color=df_color['color'], size=5, opacity=0.8,symbol=0), 
                    showlegend=True, 
                    customdata=df_color[['InformantID','MainVariety','Age','Gender','RatioMainVariety','YearsLivedInMainVariety','CountryCollection','Year']],
                    hovertemplate='<br>'.join([
                        'InformantID: %{customdata[0]}',
                        'Main Variety: %{customdata[1]}',
                        'Age: %{customdata[2]}',
                        'Gender: %{customdata[3]}',
                        'Ratio (Main Variety): %{customdata[4]}',
                        'Years lived in (Main Variety): %{customdata[5]}',
                        'CountryCollection: %{customdata[6]}',
                        'Year: %{customdata[7]}',
                    ]), hoverinfo='text'
                )
            )
    if leiden:
        # Create confusion matrix
        cm = pd.crosstab(embedding['MainVariety'], embedding['leiden_cluster'].astype(str))
        varieties = sorted(embedding['MainVariety'].unique())
        clusters = sorted(str(c) for c in unique_clusters)
        
        # Add confusion matrix heatmap
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=[f'Cluster {c}' for c in clusters],
                y=varieties,
                colorscale='Blues',
                showscale=True,
                hovertemplate='Variety: %{y}<br>Cluster: %{x}<br>Count: %{z}<extra></extra>'
            ),
            row=1, col=2
        )

    # if all markers of a trace have the same opacity, plotly sets opacity to a single value
    # when hiding data points via opacity, this needs to be a list of the same value for each data point
    for trace in fig.data:
        if hasattr(trace, 'marker'):
            marker = getattr(trace, 'marker', None)
            ids = getattr(trace, 'ids', None)
            if marker and hasattr(marker, 'opacity'):
                opacity = marker.opacity
                if isinstance(opacity, (int, float)) and ids is not None:
                    marker.opacity = [opacity] * len(ids)

    fig.update_layout(template="simple_white")
    if leiden:
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    else:
        fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        )


        # Save the newly rendered figure as a preset for future use
    try:
        os.makedirs(preset_dir, exist_ok=True)
        with open(preset_path, "wb") as f:
            pickle.dump(fig, f)
    except Exception:
        pass  # Ignore errors in saving

    return fig

def trainRF(GrammarItemsCols,data,datacols,groupcol="MainVariety",pairs=False,use_zscores=False):
    if datacols is None:
        datacols = GrammarItemsCols
        #datacols=retrieve_data.getGrammarItemsCols()
    else:
        datacols = retrieve_data.inner_join_list(datacols,GrammarItemsCols)
    if not isinstance(groupcol,str):
        data = pd.merge(data,groupcol, left_on="InformantID",right_on="ids")
        groupcol = "group"
    else:
        data['group']=data['MainVariety']
    
    # convert all non-numeric values in datacols to missing values
    data[datacols] = data[datacols].apply(pd.to_numeric, errors='coerce')
    
    # Apply Z-score standardization row-wise if requested
    if use_zscores:
        
        # Standardize (Z-score) each participant's data row-wise
        data[datacols] = data[datacols].apply(pd.to_numeric, errors='coerce')
        # Compute row-wise mean and std
        row_means = data[datacols].mean(axis=1)
        row_stds = data[datacols].std(axis=1).replace(0, 1).fillna(1)
        # Standardize row-wise (Z-score)
        data[datacols] = data[datacols].sub(row_means, axis=0)
        data[datacols] = data[datacols].div(row_stds, axis=0)

        
    
    y = data[groupcol]
    X = data[datacols]

    # Split the data into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    #rf = RandomForestClassifier()
    #rf.fit(X_train, y_train)
    #y_pred = rf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print("Accuracy:", accuracy)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample') # use balanced class weights to account for class imbalance (e.g. more informants from one country than another)
    rf.fit(X, y)
    feature_importances = pd.DataFrame({'importance':rf.feature_importances_, 'item':X.columns}).sort_values(by='importance',ascending=False)
    if not pairs:
        spokenitems = pd.DataFrame({'item':retrieve_data.getGrammarItemsCols("spoken"),'mode':'spoken'})
        writtenitems =  pd.DataFrame({'item':retrieve_data.getGrammarItemsCols("written"),'mode':'written'})
        items = pd.concat([spokenitems, writtenitems], ignore_index=True)
        feature_importances = feature_importances.merge(items,on="item")
    else:
        feature_importances['mode'] = 'difference'
    return rf, feature_importances

def getRFplot(data, importanceRatings, value_range=[0,5],pairs=False, split_by_variety=False):
    if not pairs:
        Rating_map = {'0':'No-one','1':'Few','2':'Some','3':'Many','4':'Most','5':'Everyone'}
    else:
        Rating_map = {'-5': 'Written only', '-4':'-4', '-3':'-3', '-2':'-2', '-1':'-1', '0':'Neutral', '1':'1', '2':'2', '3':'3', '4':'4', '5':'Spoken only'}
    df = data.copy()
    # Ensure required columns exist
    required_cols = {'item', 'mean', 'group', 'mode'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")
    # Filter by mean range
    if not pairs:
        df = df.groupby('item', observed=True).filter(lambda x: x['mean'].between(value_range[0], value_range[1]).all()).reset_index(drop=True)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data points remained after applying the filter. Please check the selected range.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(template="simple_white")
        return fig
    if not pairs:
        meta = retrieve_data.getGrammarMeta()
    else:
        meta = retrieve_data.getGrammarMeta(type='item_pairs')
    # Merge meta info
    if not pairs:
        df = df.merge(meta, left_on="item", right_on="question_code", how="left")
    else:
        df = df.merge(meta, left_on="item", right_on="item_pair", how="left")

    # Rename columns if needed
    if 'item_y' in df.columns and 'item_x' in df.columns:
        df.rename(columns={'item_y':'sentence','item_x':'item'}, inplace=True)
    elif 'item' in df.columns and 'sentence' not in df.columns and 'feature' in df.columns:
        df['sentence'] = df['feature']
    # Always convert group to string for plotting
    df['group'] = df['group'].astype(str)
    # Merge importanceRatings if provided and not already present
    if importanceRatings is not None and 'importance' not in df.columns:
        if isinstance(importanceRatings, pd.DataFrame) and 'item' in importanceRatings.columns and 'importance' in importanceRatings.columns:
            df = df.merge(importanceRatings[['item', 'importance']], on='item', how='left')
        else:
            df['importance'] = 0
    elif 'importance' not in df.columns:
        df['importance'] = 0
    # Ensure error bar columns exist
    for col in ['upper_ci', 'lower_ci']:
        if col not in df.columns:
            df[col] = 0
    # Get color mapping for varieties if groups are main varieties
    variety_color_map = None
    known_varieties = get_ordered_varieties()
    if df['group'].str.contains('|'.join(known_varieties)).any():
        variety_color_map = retrieve_data.get_color_for_variety(type="grammar")
    
    # Check for split by variety mode
    if split_by_variety:
        # Get unique items and groups
        unique_items = sorted(df['item'].unique())
        unique_groups = sort_groups_for_plot(df['group'].unique(), groupby)
        
        # Create item-group position mapping
        item_positions = {}
        current_pos = 0
        
        for item in unique_items:
            item_positions[item] = {}
            for i, group in enumerate(unique_groups):
                item_positions[item][group] = current_pos + i * 0.8  # Space groups within each item
            current_pos += len(unique_groups) + 1  # Space between items
        
        # Create x-axis labels and tick positions
        x_labels = []
        x_positions = []
        
        for item in unique_items:
            for group in unique_groups:
                x_labels.append(f"{item} - {group}")
                x_positions.append(item_positions[item][group])
    
    # --- NEW: check if group contains "100" ---
    use_symbol_only = "100" in df['group'].unique()
    modes = df['mode'].unique()
    if len(modes) == 1:
        if not pairs:
            if split_by_variety:
                plot_groups = unique_groups  # Use the properly ordered groups for split by variety
            else:
                plot_groups = sorted(df['group'].unique())
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            if split_by_variety:
                # Plot with split by variety - each group gets its own x position per item
                for plot_group in plot_groups:
                    tempDF = df[df['group'] == plot_group]
                    color = None
                    if not use_symbol_only and variety_color_map and plot_group in variety_color_map:
                        color = variety_color_map[plot_group]
                    
                    # Calculate x positions for this group
                    x_vals = []
                    y_vals = []
                    upper_ci_vals = []
                    lower_ci_vals = []
                    custom_data_vals = []
                    
                    for _, row in tempDF.iterrows():
                        item = row['item']
                        if item in item_positions and plot_group in item_positions[item]:
                            x_vals.append(item_positions[item][plot_group])
                            y_vals.append(row['mean'])
                            upper_ci_vals.append(row['upper_ci'])
                            lower_ci_vals.append(row['lower_ci'])
                            
                            # Prepare custom data for hover
                            if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']):
                                custom_data_vals.append([row['item'], row['group'], row['mean'], row['count'], 
                                                       row.get('sentence', 'N/A'), row.get('variant_detail', 'N/A'), 
                                                       row.get('group_finegrained', 'N/A'), row.get('feature_ewave', 'N/A'), 
                                                       row.get('also_in_question', 'N/A')])
                    
                    if x_vals:  # Only add trace if there are data points
                        fig.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=y_vals,
                                name=plot_group,
                                mode="markers",
                                marker=dict(color=color) if color else None,
                                error_y=dict(
                                    type='data',
                                    symmetric=False,
                                    array=upper_ci_vals,
                                    arrayminus=lower_ci_vals
                                ),
                                customdata=custom_data_vals if custom_data_vals else None,
                                hovertemplate='<br>'.join([
                                    'Item: %{customdata[0]}',
                                    'Group: %{customdata[1]}',
                                    'Avg rating: %{customdata[2]:.2f}',
                                    'Number of participants: %{customdata[3]}',
                                    'Sentence: %{customdata[4]}',
                                    'Item name: %{customdata[5]}',
                                    'Item group: %{customdata[6]}',
                                    'Ewave feature: %{customdata[7]}',
                                    'Twin item: %{customdata[8]}'
                                ]) if custom_data_vals else None,
                                hoverinfo='text'
                            ), secondary_y=False
                        )
                
                # Update x-axis with custom labels and positions
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=x_positions,
                        ticktext=x_labels,
                        tickangle=45
                    )
                )
            else:
                # Original plotting logic for normal mode
                for plot_group in plot_groups:
                    tempDF = df[df['group'] == plot_group]
                    color = None
                    if not use_symbol_only and variety_color_map and plot_group in variety_color_map:
                        color = variety_color_map[plot_group]
                    # --- NEW: use symbol only if use_symbol_only is True ---
                    if use_symbol_only:
                        fig.add_trace(
                            go.Scatter(
                                x=tempDF['item'],
                                y=tempDF['mean'],
                                mode="markers",
                                marker=dict(
                                    symbol=int(plot_group),color="black"
                                ),
                                error_y=dict(
                                    type='data',
                                    symmetric=False,
                                    array=tempDF['upper_ci'],
                                    arrayminus=tempDF['lower_ci']
                                ),
                                customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                                hovertemplate='<br>'.join([
                                    'Item: %{customdata[0]}',
                                    'Group: %{customdata[1]}',
                                    'Avg rating: %{customdata[2]:.2f}',
                                    'Number of participants: %{customdata[3]}',
                                    'Sentence: %{customdata[4]}',
                                    'Item name: %{customdata[5]}',
                                    'Item group: %{customdata[6]}',
                                    'Ewave feature: %{customdata[7]}',
                                    'Twin item: %{customdata[8]}'
                                ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                                hoverinfo='text'
                            ), secondary_y=False
                        )
                    else:
                        if plot_group.isdigit():
                            fig.add_trace(
                                go.Scatter(
                                    x=tempDF['item'],
                                    y=tempDF['mean'],
                                    mode="markers",
                                    marker=dict(
                                        symbol=int(plot_group),
                                        color=color
                                    ),
                                    error_y=dict(
                                        type='data',
                                        symmetric=False,
                                        array=tempDF['upper_ci'],
                                        arrayminus=tempDF['lower_ci']
                                    ),
                                    customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                                    hovertemplate='<br>'.join([
                                        'Item: %{customdata[0]}',
                                        'Group: %{customdata[1]}',
                                        'Avg rating: %{customdata[2]}',
                            'Number of participants: %{customdata[3]}',
                                        'Sentence: %{customdata[4]}',
                                        'Item name: %{customdata[5]}',
                                        'Item group: %{customdata[6]}',
                                        'Ewave feature: %{customdata[7]}',
                                        'Twin item: %{customdata[8]}'
                                    ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                                    hoverinfo='text'
                                ), secondary_y=False
                            )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=tempDF['item'],
                                    y=tempDF['mean'],
                                    name=plot_group,
                                    mode="markers",
                                    marker=dict(color=color) if color else None,
                                    error_y=dict(
                                        type='data',
                                        symmetric=False,
                                        array=tempDF['upper_ci'],
                                        arrayminus=tempDF['lower_ci']
                                    ),
                                    customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                                    hovertemplate='<br>'.join([
                                        'Item: %{customdata[0]}',
                                        'Group: %{customdata[1]}',
                                        'Avg rating: %{customdata[2]}',
                            'Number of participants: %{customdata[3]}',
                                        'Sentence: %{customdata[4]}',
                                        'Item name: %{customdata[5]}',
                                        'Item group: %{customdata[6]}',
                                        'Ewave feature: %{customdata[7]}',
                                        'Twin item: %{customdata[8]}'
                                    ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                                    hoverinfo='text'
                                ), secondary_y=False
                            )
            
            # Add importance plot
            if not split_by_variety:
                # Original importance plot logic - sort by importance and update x-axis
                tempDF = df[['item','importance','mode']].drop_duplicates().sort_values("importance",ascending=False)
                fig.update_xaxes(categoryorder='array', categoryarray=tempDF['item'].to_list(), range=[-1,10])
                fig.add_trace(
                    go.Scatter(
                        y=tempDF['importance'],
                        x=tempDF['item'],
                        xaxis='x',
                        name="RF importance",
                    ), secondary_y=True
                )
            else:
                # Importance plot for split-by-variety mode
                tempDF = df[['item','importance','mode']].drop_duplicates().sort_values("importance",ascending=False)
                
                # Calculate x positions for importance values (use middle position for each item)
                importance_x_vals = []
                importance_y_vals = []
                
                for _, row in tempDF.iterrows():
                    item = row['item']
                    if item in item_positions:
                        # Use the middle position of all groups for this item
                        group_positions = list(item_positions[item].values())
                        middle_pos = sum(group_positions) / len(group_positions)
                        importance_x_vals.append(middle_pos)
                        importance_y_vals.append(row['importance'])
                
                fig.add_trace(
                    go.Scatter(
                        y=importance_y_vals,
                        x=importance_x_vals,
                        name="RF importance",
                        mode="markers+lines",
                        line=dict(color="red", dash="dash"),
                        marker=dict(color="red", symbol="diamond")
                    ), secondary_y=True
                )
            
            fig.update_layout(height=400)
        else:
            plot_groups = sorted(df['group'].unique())
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for plot_group in plot_groups:
                tempDF = df[df['group'] == plot_group]
                color = None
                if not use_symbol_only and variety_color_map and plot_group in variety_color_map:
                    color = variety_color_map[plot_group]
                # --- NEW: use symbol only if use_symbol_only is True ---
                if use_symbol_only:
                    fig.add_trace(
                        go.Scatter(
                            x=tempDF['item'],
                            y=tempDF['mean'],
                            mode="markers",
                            marker=dict(
                                symbol=int(plot_group),color="black"
                            ),
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=tempDF['upper_ci'],
                                arrayminus=tempDF['lower_ci']
                            ),
                            customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                            hovertemplate='<br>'.join([
                                'Item: %{customdata[0]}',
                                'Group: %{customdata[1]}',
                                'Avg rating: %{customdata[2]}',
                        'Number of participants: %{customdata[3]}',
                                'Sentence: %{customdata[4]}',
                                'Item name: %{customdata[5]}',
                                'Item group: %{customdata[6]}',
                                'Ewave feature: %{customdata[7]}',
                                'Twin item: %{customdata[8]}'
                            ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']) else None,
                            hoverinfo='text'
                        ), secondary_y=False
                    )
                else: 
                    if plot_group.isdigit():
                        fig.add_trace(
                            go.Scatter(
                                x=tempDF['item'],
                                y=tempDF['mean'],
                                mode="markers",
                                marker=dict(
                                    symbol=int(plot_group),
                                    color=color
                                ),
                                error_y=dict(
                                    type='data',
                                    symmetric=False,
                                    array=tempDF['upper_ci'],
                                    arrayminus=tempDF['lower_ci']
                                ),
                                customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave',]] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                                hovertemplate='<br>'.join([
                                    'Item: %{customdata[0]}',
                                    'Group: %{customdata[1]}',
                                    'Avg rating: %{customdata[2]}',
                        'Number of participants: %{customdata[3]}',
                                    'Sentence: %{customdata[4]}',
                                    'Item name: %{customdata[5]}',
                                    'Item group: %{customdata[6]}',
                                    'Ewave feature: %{customdata[7]}',
                                    'Twin item: %{customdata[8]}'
                                ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']) else None,
                                hoverinfo='text'
                            ), secondary_y=False
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=tempDF['item'],
                                y=tempDF['mean'],
                                name=plot_group,
                                mode="markers",
                                marker=dict(color=color) if color else None,
                                error_y=dict(
                                    type='data',
                                    symmetric=False,
                                    array=tempDF['upper_ci'],
                                    arrayminus=tempDF['lower_ci']
                                ),
                                customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                                hovertemplate='<br>'.join([
                                    'Item: %{customdata[0]}',
                                    'Group: %{customdata[1]}',
                                    'Avg rating: %{customdata[2]}',
                        'Number of participants: %{customdata[3]}',
                                    'Sentence: %{customdata[4]}',
                                    'Item name: %{customdata[5]}',
                                    'Item group: %{customdata[6]}',
                                    'Ewave feature: %{customdata[7]}',
                                    'Twin item: %{customdata[8]}'
                                ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']) else None,
                                hoverinfo='text'
                            ), secondary_y=False
                        )
            
            # Add importance plot
            if not split_by_variety:
                # Original importance plot logic - sort by importance and update x-axis
                tempDF = df[['item','importance','mode']].drop_duplicates().sort_values("importance",ascending=False)
                fig.update_xaxes(categoryorder='array', categoryarray=tempDF['item'].to_list(), range=[-1,10])
                fig.add_trace(
                    go.Scatter(
                        y=tempDF['importance'],
                        x=tempDF['item'],
                        xaxis='x',
                        name="RF importance",
                    ), secondary_y=True
                )
            else:
                # Importance plot for split-by-variety mode
                tempDF = df[['item','importance','mode']].drop_duplicates().sort_values("importance",ascending=False)
                
                # Calculate x positions for importance values (use middle position for each item)
                importance_x_vals = []
                importance_y_vals = []
                
                for _, row in tempDF.iterrows():
                    item = row['item']
                    if item in item_positions:
                        # Use the middle position of all groups for this item
                        group_positions = list(item_positions[item].values())
                        middle_pos = sum(group_positions) / len(group_positions)
                        importance_x_vals.append(middle_pos)
                        importance_y_vals.append(row['importance'])
                
                fig.add_trace(
                    go.Scatter(
                        y=importance_y_vals,
                        x=importance_x_vals,
                        name="RF importance",
                        mode="markers+lines",
                        line=dict(color="red", dash="dash"),
                        marker=dict(color="red", symbol="diamond")
                    ), secondary_y=True
                )
            
            fig.update_layout(height=400)
    else:
        # Multi-mode section
        axisorder = []
        fig = make_subplots(rows=2,cols=1,specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
        
        # Handle split by variety for multi-mode
        if split_by_variety:
            # Get unique items and groups for multi-mode
            unique_items = sorted(df['item'].unique())
            unique_groups = sort_groups_for_plot(df['group'].unique(), groupby)  # Use centralized ordering
            plot_groups = unique_groups  # Use properly ordered groups for multi-mode split by variety

            
            # Create item-group position mapping for multi-mode
            item_positions_multi = {}
            current_pos = 0
            
            for item in unique_items:
                item_positions_multi[item] = {}
                for i, group in enumerate(unique_groups):
                    item_positions_multi[item][group] = current_pos + i * 0.8  # Space groups within each item
                current_pos += len(unique_groups) + 1  # Space between items
            
            # Create x-axis labels and tick positions for multi-mode
            x_labels_multi = []
            x_positions_multi = []
            
            for item in unique_items:
                for group in unique_groups:
                    x_labels_multi.append(f"{item} - {group}")
                    x_positions_multi.append(item_positions_multi[item][group])
        
        for mode in modes:
            if mode == 'spoken':
                row=1
            else:
                row=2
            
            
            plot_groups = sorted(df['group'].unique(), reverse=True)

            if split_by_variety:
                # Split by variety logic for multi-mode
                for plot_group in plot_groups:
                    tempDF = df[((df['group'] == plot_group) & (df['mode'] == mode))].sort_values("importance",ascending=False)
                    
                    # Determine color for this group
                    color = None
                    if variety_color_map and plot_group in variety_color_map:
                        color = variety_color_map[plot_group]
                    
                    # Calculate x positions for this group in multi-mode
                    x_vals = []
                    y_vals = []
                    upper_ci_vals = []
                    lower_ci_vals = []
                    custom_data_vals = []
                    
                    for _, data_row in tempDF.iterrows():
                        item = data_row['item']
                        if item in item_positions_multi and plot_group in item_positions_multi[item]:
                            x_vals.append(item_positions_multi[item][plot_group])
                            y_vals.append(data_row['mean'])
                            upper_ci_vals.append(data_row['upper_ci'])
                            lower_ci_vals.append(data_row['lower_ci'])
                            custom_data_vals.append([data_row['item'], data_row['group'], data_row['mean'], data_row['count'], 
                                                   data_row.get('sentence', 'N/A'), data_row.get('variant_detail', 'N/A'), 
                                                   data_row.get('group_finegrained', 'N/A'), data_row.get('feature_ewave', 'N/A'), 
                                                   data_row.get('also_in_question', 'N/A')])
                    
                    if x_vals:  # Only add trace if there are data points
                        fig.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=y_vals,
                                name=f"{plot_group} ({mode})",
                                mode="markers",
                                marker=dict(color=color) if color else None,
                                error_y=dict(
                                    type='data',
                                    symmetric=False,
                                    array=upper_ci_vals,
                                    arrayminus=lower_ci_vals
                                ),
                                customdata=custom_data_vals if custom_data_vals else None,
                                hovertemplate='<br>'.join([
                                    'Item: %{customdata[0]}',
                                    'Group: %{customdata[1]}',
                                    'Avg rating: %{customdata[2]:.2f}',
                                    'Number of participants: %{customdata[3]}',
                                    'Sentence: %{customdata[4]}',
                                    'Item name: %{customdata[5]}',
                                    'Item group: %{customdata[6]}',
                                    'Ewave feature: %{customdata[7]}',
                                    'Twin item: %{customdata[8]}'
                                ]) if custom_data_vals else None,
                                hoverinfo='text'
                            ),row=row,col=1,secondary_y=False
                        )
                
                # Update x-axis with custom labels for this mode
                if row == 1:
                    fig.update_xaxes(
                        tickmode='array',
                        tickvals=x_positions_multi,
                        ticktext=x_labels_multi,
                        tickangle=45,
                        row=row, col=1
                    )
                else:
                    fig.update_xaxes(
                        tickmode='array',
                        tickvals=x_positions_multi,
                        ticktext=x_labels_multi,
                        tickangle=45,
                        row=row, col=1
                    )
            else:
                # Original multi-mode logic (non-split)

                for plot_group in plot_groups:
                    tempDF = df[((df['group'] == plot_group) & (df['mode'] == mode))].sort_values("importance",ascending=False)
                    
                    # Determine color for this group
                    color = None
                    if variety_color_map and plot_group in variety_color_map:
                        color = variety_color_map[plot_group]
                    if use_symbol_only:
                        fig.add_trace(
                        go.Scatter(
                            x=tempDF['item'],
                            y=tempDF['mean'],
                            name=f"{str(plot_group)} ({mode})",
                            mode="markers",
                            marker=dict(symbol=plot_group, color=color),
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=tempDF['upper_ci'],
                                arrayminus=tempDF['lower_ci']
                                ),
                            customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']],
                            hovertemplate='<br>'.join([
                                'Item: %{customdata[0]}',
                                'Group: %{customdata[1]}',
                                'Avg rating: %{customdata[2]}',
                            'Number of participants: %{customdata[3]}',
                                'Sentence: %{customdata[4]}',
                                'Item name: %{customdata[5]}',
                                'Item group: %{customdata[6]}',
                                'Ewave feature: %{customdata[7]}',
                                'Twin item: %{customdata[8]}'
                        ]), hoverinfo='text'
                        ),row=row,col=1,secondary_y=False
                    )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=tempDF['item'],
                                y=tempDF['mean'],
                                name=f"{plot_group} ({mode})",
                                mode="markers",
                                marker=dict(color=color) if color else None,
                                error_y=dict(
                                    type='data',
                                    symmetric=False,
                                    array=tempDF['upper_ci'],
                                    arrayminus=tempDF['lower_ci']
                                ),
                                customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']],
                                hovertemplate='<br>'.join([
                                    'Item: %{customdata[0]}',
                                    'Group: %{customdata[1]}',
                                    'Avg rating: %{customdata[2]}',
                                    'Number of participants: %{customdata[3]}',
                                    'Sentence: %{customdata[4]}',
                                    'Item name: %{customdata[5]}',
                                    'Item group: %{customdata[6]}',
                                    'Ewave feature: %{customdata[7]}',
                                    'Twin item: %{customdata[8]}'
                                ]), hoverinfo='text'
                    ),row=row,col=1,secondary_y=False
                )
            
            # Add importance plot for multi-mode
            tempDF=df[df['mode'] == mode][['item','importance','mode']].drop_duplicates().sort_values("importance",ascending=False)
            axisorder.append(tempDF)
            
            if not split_by_variety:
                fig.update_xaxes(categoryorder='array', categoryarray=tempDF['item'].to_list())
                fig.add_trace(
                    go.Scatter(
                        y=tempDF['importance'],
                        x=tempDF['item'],
                        name=f"RF importance ({mode})",
                    ),row=row,col=1, secondary_y=True
                )
            else:
                # Importance plot for split-by-variety multi-mode
                importance_x_vals = []
                importance_y_vals = []
                
                for _, imp_row in tempDF.iterrows():
                    item = imp_row['item']
                    if item in item_positions_multi:
                        # Use the middle position of all groups for this item
                        group_positions = list(item_positions_multi[item].values())
                        middle_pos = sum(group_positions) / len(group_positions)
                        importance_x_vals.append(middle_pos)
                        importance_y_vals.append(imp_row['importance'])
                
                fig.add_trace(
                    go.Scatter(
                        y=importance_y_vals,
                        x=importance_x_vals,
                        name=f"RF importance ({mode})",
                        mode="markers+lines",
                        line=dict(color="red", dash="dash"),
                        marker=dict(color="red", symbol="diamond")
                    ),row=row,col=1, secondary_y=True
                )
        
        axisorder = pd.concat(axisorder)
        axisorder.loc[:,'row'] = 1
        axisorder.loc[axisorder['mode']=='written','row'] = 2
        axisorder.sort_values(by=['row','importance'],inplace=True,ascending=[True,False])
        axisorder.reset_index(inplace=True)
        
        if not split_by_variety:
            fig['layout']['xaxis1']['categoryarray'] = tuple(axisorder.loc[axisorder['mode']=='spoken','item'].to_list())
            fig['layout']['xaxis2']['categoryarray'] = tuple(axisorder.loc[axisorder['mode']=='written','item'].to_list())
            fig.update_layout(
                height=800,
                xaxis1=dict(range=[-0.5,10]),xaxis2=dict(range=[-0.5,10])
            )
        else:
            # For split-by-variety, keep the custom tick positions and labels
            fig.update_layout(height=800)

    # Customize layout
    fig.update_layout(
        title='Mean with 95% Confidence Intervals',
        xaxis_title='Grammatical items',
        yaxis_title='Mean ratings',
        template='simple_white')
    
    fig.update_yaxes(title_text="Average rating", secondary_y=False,fixedrange=True,range=[0,5.1])
    if not pairs:
        fig.update_yaxes(
            title_text="Average rating",
            secondary_y=False,
            fixedrange=True,
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=[Rating_map[str(i)] for i in range(6)]
        )
    else:
        fig.update_yaxes(
            title_text="Average difference in ratings",
            secondary_y=False,
            fixedrange=True,
            tickvals=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
            ticktext=[Rating_map[str(i)] for i in range(-5, 5)]
        )
    return fig


def getAuxiliaryTable(Informants,participants):
    data = Informants.copy(deep=True)
    if len(participants) == 0:
        return dash_table.DataTable()

    columns = ['InformantID','MainVariety','AdditionalVarieties','Age','Gender','Nationality','EthnicSelfID','CountryID','LanguageHome','YearsLivedInMainVariety','RatioMainVariety','CountryCollection','Year','LanguageMother','LanguageFather','Qualifications','Occupation','OccupMother','OccupFather','OccupPartner','QualiMother','QualiFather','QualiPartner','YearsLivedOutside','YearsLivedInside','YearsLivedOtherEnglish','CommentsTimeline']
    data = data.loc[data['InformantID'].isin(participants),columns].reset_index()


    table = dag.AgGrid(
        id="umap-selection-table",
        rowData=data.to_dict("records"),
        columnDefs=[
            {"field": col, 
             "filter": "agTextColumnFilter", 
             "sortable": True,
             "resizable": True,
             "minWidth": 100,
             "flex": 1} 
            for col in data.columns
        ],
        defaultColDef={
            "filter": "agTextColumnFilter",
            "sortable": True,
            "resizable": True,
            "minWidth": 100,
            "flex": 1
        },
        className="ag-theme-quartz compact",
        columnSize="autoSize",
        style={"height": "400px"},
        dashGridOptions={
            "suppressMenuHide": True,
            "animateRows": True,
            "enableRangeSelection": True,
            "pagination": True,
            "paginationPageSize": 15,
            "headerHeight": 30
        }
    )       

    # distribution of age and gender
    return table

def getMetaTable(data):
    #data = data
    # filter out columns Standard_variety, "Control Item", and "Related Item"
    data = data.drop(columns=[col for col in ['standard_variety', 'control_item', 'related_item'] if col in data.columns])
    data.columns = [col.replace('_', ' ') for col in data.columns]
    
    # Add button and helper text for table features
    controls = dmc.Group([
        dmc.Button(
            "Show only selected items",
            id="filter-grammar-items-table",
            size="xs",
            variant="light",
            color="blue",
            leftSection=DashIconify(icon="tabler:filter", width=14),
        ),
        dmc.Text(
            "💡 Click headers to sort • Drag column borders to resize • Use filter boxes below headers to search.", 
            size="sm", c="dimmed", 
            style={"fontStyle": "italic"}
        ),
    ], mb=10, style={"backgroundColor": "#f8f9fa", "padding": "8px", "borderRadius": "4px", "border": "1px solid #e9ecef"})
    
    # Define which columns should have category filters (set filter) vs text filters
    category_columns = ['section', 'feature', 'group ewave', 'group finegrained', 'variant detail', 'feature ewave']
    
    table = dag.AgGrid(
                    id="grammar-items-table",
                    rowData=data.to_dict("records"),
                    columnDefs=[
                        {
                            "field": col, 
                            "headerName": col.replace("Item", "Item ID") if col == "Item" else col,
                            "filter": "agSetColumnFilter" if col in category_columns else "agTextColumnFilter",
                            "sortable": True,
                            "resizable": True,
                            "minWidth": 80 if col == "Item" else 120,
                            "flex": 1,
                            "cellStyle": {"textAlign": "left"},
                            "headerTooltip": f"Click to sort by {col}. Use filter below to search.",
                            "wrapText": True if col in ["Sentence", "Context"] else False,
                            "autoHeight": True if col in ["Sentence", "Context"] else False,
                            "filterParams": {
                                "buttons": ["reset", "apply"],
                                "closeOnApply": True
                            } if col in category_columns else {}
                        } 
                        for col in data.columns
                    ],
                    defaultColDef={
                        "filter": "agTextColumnFilter",
                        "sortable": True,
                        "resizable": True,
                        "minWidth": 100,
                        "flex": 1,
                        "headerTooltip": "Click header to sort, drag borders to resize, use filter below to search"
                    },
                    className="ag-theme-quartz compact",
                    columnSize="autoSize",
                    style={"height": "80vh"},
                    dashGridOptions={
                        "suppressMenuHide": True,
                        "animateRows": True,
                        "enableRangeSelection": True,
                        "pagination": True,
                        "paginationPageSize": 50,
                        "headerHeight": 30,
                        "suppressColumnVirtualisation": True,
                        "enableBrowserTooltips": True,
                        "tooltipShowDelay": 500,
                        "wrapHeaderText": True
                    }
                    )
    
    # Return both controls and table
    return html.Div([controls, table])

def get_balanced_informants(informants, groupby):
    """Apply balanced random sampling for variety types if needed"""
    if groupby != "vtype_balanced":
        return informants
    
    import numpy as np
    np.random.seed(42)  # For reproducible results
    
    # Get informant data to determine varieties
    informant_data = retrieve_data.getInformantDataGrammar(imputed=True)
    informant_data = informant_data[informant_data['InformantID'].isin(informants)]
    
    # Use centralized variety classification
    variety_mapping = get_variety_mapping()
    variety_classification = get_variety_classification()
    
    # Filter out unknown varieties
    informant_data = informant_data[informant_data['MainVariety'].isin(variety_mapping.keys())]
    
    # Apply balanced sampling within each variety type
    balanced_informants = []
    
    for vtype in ["ENL", "ESL", "EFL"]:
        # Get varieties within this type using centralized classification
        varieties_in_type = variety_classification[vtype]
        varieties_present = [v for v in varieties_in_type if v in informant_data['MainVariety'].unique()]
        
        if len(varieties_present) > 0:
            # Find minimum sample size across varieties in this type
            min_variety_size = min([
                informant_data[informant_data['MainVariety'] == variety]['InformantID'].nunique() 
                for variety in varieties_present
            ])
            
            # Sample equally from each variety within this type
            for variety in varieties_present:
                variety_informants = informant_data[informant_data['MainVariety'] == variety]['InformantID'].unique()
                if len(variety_informants) >= min_variety_size:
                    sampled_informants = np.random.choice(
                        variety_informants, 
                        size=min_variety_size, 
                        replace=False
                    )
                    balanced_informants.extend(sampled_informants)
    
    return balanced_informants

def getItemPlot(informants,items,sortby="mean",mean_cutoff_range=[0,5],groupby="variety",pairs=False,use_imputed=False,plot_mode="normal",split_by_variety=False):
    # groupby: group by column in the data, possible Values: "variety","vtype","vtype_balanced","gender"
    # sortby: column to sort by, . "mean", "sd","alpha"
    #if True:
    #    return go.Figure()
    # to do
    
    # Apply balanced sampling if needed
    balanced_informants = get_balanced_informants(informants, groupby)
    
    Rating_map={'0':'No-one','1':'Few','2':'Some','3':'Many','4':'Most','5':'Everyone'}
    df = retrieve_data.getGrammarData(imputed=use_imputed, items=items, participants=balanced_informants,pairs=pairs)
    #df = df.groupby('item').filter(lambda x: x['mean'].between(value_range[0], value_range[1]).all()).reset_index()
    
    if df.empty:
        fig = getEmptyPlot()
        return fig
    
    if not pairs:
        meta = retrieve_data.getGrammarMeta()
        metaCols = meta.columns
    else:
        meta = retrieve_data.getGrammarMeta(type='item_pairs')
        metaCols = meta.columns.to_list()


    if groupby == "variety":
        df['group'] = df['MainVariety']
        # Get color mapping for varieties
        variety_color_map = retrieve_data.get_color_for_variety(type="grammar")
    elif groupby == "vtype" or groupby == "vtype_balanced":
        # Use centralized variety mapping
        variety_mapping = get_variety_mapping()
        df['group'] = df['MainVariety'].map(variety_mapping).fillna("Other")
        df= df[df['group'] != "Other"]  # filter out "Other" group
        variety_color_map = None
    elif groupby == "gender":
        df['group'] = df['Gender_normalized'] if 'Gender_normalized' in df.columns else df['Gender']
        variety_color_map = None
    
    # to do: melt df, CI calculate, based on group and item
    infoCols = retrieve_data.getInformantDataGrammar(imputed=True).columns.to_list()
    # remove InformantID from the list of columns to melt
    infoCols.remove('InformantID')
    infoCols.append('group')
    df = df.melt(id_vars=infoCols,value_vars=items,var_name='item')

    # convert value to integer

    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Define modes early for use in sorting logic
    df['mode'] = df['section'].str.lower() if 'section' in df.columns else pd.Series(['single'] * len(df))
    modes = df['mode'].unique()

    meanplf = pl.from_pandas(df)
    meanplf = meanplf.group_by(['item']).agg(
        pl.mean('value').alias('mean'),
        pl.std('value').alias('std')
    )
    meandf = meanplf.to_pandas()
    
    # For multi-mode sorting OR single mode with twins, group twin items and calculate combined statistics
    if (len(modes) > 1 or len(modes) == 1) and not pairs:
        # Create twin item pairs
        meta = retrieve_data.getGrammarMeta()
        item_twins = {}
        
        for item in meandf['item']:
            if item in meta['question_code'].values:
                twin_item = meta[meta['question_code'] == item]['also_in_question'].iloc[0]
                if pd.notna(twin_item) and twin_item != '' and twin_item in meandf['item'].values:
                    item_twins[item] = twin_item
        
        # Calculate combined statistics for twin pairs (for sorting purposes)
        if item_twins and (sortby == "mean" or sortby == "sd"):
            combined_stats = []
            processed_items = set()
            
            for item in meandf['item']:
                if item in processed_items:
                    continue
                    
                if item in item_twins:
                    twin = item_twins[item]
                    if twin in meandf['item'].values:
                        # Calculate combined statistics for the pair
                        item1_data = df[df['item'] == item]['value'].dropna()
                        item2_data = df[df['item'] == twin]['value'].dropna()
                        combined_data = pd.concat([item1_data, item2_data])
                        
                        combined_mean = combined_data.mean()
                        combined_std = combined_data.std()
                        
                        # Create representative entry (use first item as key)
                        pair_key = min(item, twin)  # Use consistent ordering
                        combined_stats.append({
                            'item': pair_key,
                            'mean': combined_mean,
                            'std': combined_std,
                            'is_pair': True,
                            'twin_item': max(item, twin)
                        })
                        processed_items.add(item)
                        processed_items.add(twin)
                    else:
                        # Twin not available, treat as standalone
                        item_stats = meandf[meandf['item'] == item].iloc[0]
                        combined_stats.append({
                            'item': item,
                            'mean': item_stats['mean'],
                            'std': item_stats['std'],
                            'is_pair': False,
                            'twin_item': None
                        })
                        processed_items.add(item)
                else:
                    # Standalone item
                    item_stats = meandf[meandf['item'] == item].iloc[0]
                    combined_stats.append({
                        'item': item,
                        'mean': item_stats['mean'],
                        'std': item_stats['std'],
                        'is_pair': False,
                        'twin_item': None
                    })
                    processed_items.add(item)
            
            # Create new sorting DataFrame
            sort_df = pd.DataFrame(combined_stats)
            
            # Sort based on combined statistics
            if sortby == "sd":
                sort_df = sort_df.sort_values(by="std", ascending=False, ignore_index=True)
            else:  # sortby == "mean"
                sort_df = sort_df.sort_values(by="mean", ascending=True, ignore_index=True)
            
            # Create ordered item list for display
            ordered_items = []
            for _, row in sort_df.iterrows():
                ordered_items.append(row['item'])
                if row['is_pair'] and row['twin_item']:
                    ordered_items.append(row['twin_item'])
            
            # Reorder meandf based on this ordering
            meandf['sort_order'] = meandf['item'].apply(lambda x: ordered_items.index(x) if x in ordered_items else 999)
            meandf = meandf.sort_values(by='sort_order').drop(columns='sort_order').reset_index(drop=True)
        else:
            # Regular sorting for single mode or alpha sorting
            if sortby == "alpha":
                # Use natural sorting for item labels like A1, A2, ..., A10, A11, etc.
                import re
                def natural_key(s):
                    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
                meandf['item_sort'] = meandf['item'].apply(natural_key)
                meandf = meandf.sort_values(by='item_sort', ascending=False).drop(columns='item_sort').reset_index(drop=True)
            elif sortby == "sd":
                meandf = meandf.sort_values(by="std", ascending=False, ignore_index=True)
            else:
                meandf = meandf.sort_values(by=sortby, ascending=True, ignore_index=True)
    else:
        # Regular sorting for single mode or pairs
        if sortby == "alpha":
            # Use natural sorting for item labels like A1, A2, ..., A10, A11, etc.
            import re
            def natural_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
            meandf['item_sort'] = meandf['item'].apply(natural_key)
            meandf = meandf.sort_values(by='item_sort', ascending=False).drop(columns='item_sort').reset_index(drop=True)
        elif sortby == "sd":
            meandf = meandf.sort_values(by="std", ascending=False, ignore_index=True)
        else:
            meandf = meandf.sort_values(by=sortby, ascending=True, ignore_index=True)
    
    # filter by mean cutoff range
    if not pairs:
        meandf = meandf[meandf['mean'].between(mean_cutoff_range[0], mean_cutoff_range[1])].reset_index()
    if meandf.empty:
        fig = getEmptyPlot(msg="No data points remained after applying the filter. Please check the selected cutoff range.")
        return fig

    # filter df by what is left in meandf
    df = df[df['item'].isin(meandf['item'])].reset_index()

    # using polars here for CI calculation, pandas groupby is insanely slow?
    plf = pl.from_pandas(df)
    plf = plf.group_by(['group','item']).agg(
        pl.count('value').alias('count'),
        pl.mean('value').alias('mean'),
        pl.std('value').alias('std')
    )
    plf = plf.with_columns(pl.lit(1.96).alias('ci'))
    plf = plf.with_columns(plf['ci']*(plf['std']/pl.col('count').sqrt().alias('ci')))
    plf = plf.with_columns(plf['ci'].alias('lower_ci')) # plotting function expects difference, not absoulte y values
    plf = plf.with_columns(plf['ci'].alias('upper_ci'))
    df = plf.to_pandas()
    # sort df by item column in meandf
    df['item'] = pd.Categorical(df['item'], categories=meandf['item'].to_list(), ordered=True)
    df.sort_values(by='item',inplace=True,ignore_index=True)
    if not pairs:
        df = df.merge(meta,left_on="item",right_on="question_code",how="left")
        df.rename(columns={'item_y':'sentence','item_x':'item'},inplace=True)
    else:
        df = df.merge(meta,left_on="item",right_on="item_pair",how="left")
        df.rename(columns={'item_y':'sentence','item_x':'item'},inplace=True)
    
    if(not pd.api.types.is_integer_dtype(df['group'])):
        df['group'] = df['group'].astype(str)
    
    # Handle split by variety mode
    if split_by_variety:
        # Get unique items respecting the sorting order from meandf, and reverse group order
        unique_items = meandf['item'].to_list()  # Use meandf order which respects sorting
        unique_groups = list(reversed(sort_groups_for_plot(list(df['group'].unique()), groupby)))  # Use centralized variety ordering, then reverse
        total_combinations = len(unique_items) * len(unique_groups)
        
        max_combinations = 275
        if total_combinations > max_combinations:
            # Return warning plot
            fig = go.Figure()
            fig.add_annotation(
                text=f"Too many combinations to display ({total_combinations} > {max_combinations}).<br>"
                     f"Please reduce the number of items or groups.<br>"
                     f"Items: {len(unique_items)}, Groups: {len(unique_groups)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red"),
                align="center"
            )
            fig.update_layout(
                title="Too Many Combinations - Please Reduce Selection",
                template='simple_white',
                height=400
            )
            return fig
        
        # Create item-group position mapping
        item_positions = {}
        current_pos = 0
        
        for item in unique_items:
            item_positions[item] = {}
            for i, group in enumerate(unique_groups):
                item_positions[item][group] = current_pos + i * 0.8  # Space groups within each item
            current_pos += len(unique_groups) + 1  # Space between items
        
        # Create x-axis labels and tick positions
        x_labels = []
        x_positions = []
        
        for item in unique_items:
            for group in unique_groups:
                x_labels.append(f"{item} - {group}")
                x_positions.append(item_positions[item][group])
        
        # Create figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Create legend order (use standard ordering for legend)
        legend_groups = sort_groups_for_plot(df['group'].unique(), groupby)
        
        # Plot with split by variety - each group gets its own x position per item
        for plot_group in unique_groups:
            tempDF = df[df['group'] == plot_group]
            color = None
            if variety_color_map and plot_group in variety_color_map:
                color = variety_color_map[plot_group]
            
            # Calculate legend rank to maintain alphabetical order in legend
            legend_rank = legend_groups.index(plot_group) if plot_group in legend_groups else 999
            
            # Calculate x positions for this group
            x_vals = []
            y_vals = []
            upper_ci_vals = []
            lower_ci_vals = []
            custom_data_vals = []
            
            for _, row in tempDF.iterrows():
                item = row['item']
                if item in item_positions and plot_group in item_positions[item]:
                    x_vals.append(item_positions[item][plot_group])
                    y_vals.append(row['mean'])
                    upper_ci_vals.append(row['upper_ci'])
                    lower_ci_vals.append(row['lower_ci'])
                    
                    # Prepare custom data for hover
                    if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']):
                        custom_data_vals.append([row['item'], row['group'], row['mean'], row['count'], 
                                               row.get('sentence', 'N/A'), row.get('variant_detail', 'N/A'), 
                                               row.get('group_finegrained', 'N/A'), row.get('feature_ewave', 'N/A'), 
                                               row.get('also_in_question', 'N/A')])
            
            if x_vals:  # Only add trace if there are data points
                fig.add_trace(
                    go.Scatter(
                        x=y_vals,  # Swap: y-values (ratings) go on x-axis
                        y=x_vals,  # Swap: x-values (positions) go on y-axis
                        name=plot_group,
                        mode="markers",
                        marker=dict(color=color) if color else None,
                        legendrank=legend_rank,  # Maintain legend order
                        error_x=dict(  # Changed from error_y to error_x due to rotation
                            type='data',
                            symmetric=False,
                            array=upper_ci_vals,
                            arrayminus=lower_ci_vals
                        ),
                        customdata=custom_data_vals if custom_data_vals else None,
                        hovertemplate='<br>'.join([
                            'Item: %{customdata[0]}',
                            'Group: %{customdata[1]}',
                            'Avg rating: %{customdata[2]:.2f}',
                            'Number of participants: %{customdata[3]}',
                            'Sentence: %{customdata[4]}',
                            'Item name: %{customdata[5]}',
                            'Item group: %{customdata[6]}',
                            'Ewave feature: %{customdata[7]}',
                            'Twin item: %{customdata[8]}'
                        ]) if custom_data_vals else None,
                        hoverinfo='text'
                    ), secondary_y=False
                )
        
        # Update axes with custom labels and positions (rotated)
        fig.update_layout(
            yaxis=dict(  # Swapped: y-axis now has the item-group labels
                tickmode='array',
                tickvals=x_positions,
                ticktext=x_labels,
                # No angle rotation needed for y-axis labels
            ),
            title='Mean with 95% Confidence Intervals (Split by Variety - Rotated)',
            xaxis_title='Mean ratings',  # Swapped: x-axis shows ratings
            yaxis_title='Items by Group',  # Swapped: y-axis shows items
            template='simple_white',
            height=max(600, len(unique_items) * len(unique_groups) * 30)
        )
        

        
        # Add mirrored y-axis for better readability with labels
        fig.update_layout(
            yaxis=dict(  # Primary y-axis
                tickmode='array',
                tickvals=x_positions,
                ticktext=x_labels,
                mirror=True,
                side='left'
            ),
            yaxis2=dict(  # Secondary y-axis (mirror of primary)
                tickmode='array',
                tickvals=x_positions,
                ticktext=x_labels,
                overlaying='y',
                side='right',
                showgrid=False,
                zeroline=False
            ),
            title='Mean with 95% Confidence Intervals',
            xaxis_title='Mean ratings',
            yaxis_title='Items by Group',
            template='simple_white',
            height=max(600, len(unique_items) * len(unique_groups) * 30)
        )
        
        # Add secondary x-axis at top to repeat rating labels
        if not pairs:
            Rating_map={'0':'No-one','1':'Few','2':'Some','3':'Many','4':'Most','5':'Everyone'}
            fig.update_layout(
                xaxis2=dict(  # Secondary x-axis at top
                    tickvals=[0, 1, 2, 3, 4, 5],
                    ticktext=[Rating_map[str(i)] for i in range(6)],
                    overlaying='x1',
                    side='top',
                    showgrid=False,
                    zeroline=False,
                    range=[-0.2, 5.2]
                )
            )
            
            # Add repeated x-axis labels every 5 items as annotations
            if len(unique_items) > 5:
                for item_idx in range(0, len(unique_items), 5):
                    # Calculate y position for this set of items
                    y_pos = item_idx * (len(unique_groups) + 2)
                    
                    # Add rating labels as annotations every 5 items
                    for rating_val, rating_label in Rating_map.items():
                        fig.add_annotation(
                            x=float(rating_val),
                            y=y_pos,
                            text=rating_label,
                            showarrow=False,
                            yshift=15,  # Position above the line
                            font=dict(size=10, color="gray"),
                            bgcolor="white",
                            bordercolor="lightgray",
                            borderwidth=1
                        )
        # Update primary x-axis with all rating labels (swapped) - always show all labels for split by variety
        if not pairs:
            Rating_map={'0':'No-one','1':'Few','2':'Some','3':'Many','4':'Most','5':'Everyone'}
            fig.update_xaxes(  # Primary x-axis
                title_text="Average rating",
                fixedrange=True,
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=[Rating_map[str(i)] for i in range(6)],
                range=[-0.2, 5.2]  # Ensure all labels are visible
            )
        else:
            Rating_map = {'-5': 'Written only', '-4':'-4', '-3':'-3', '-2':'-2', '-1':'-1', '0':'Neutral', '1':'1', '2':'2', '3':'3', '4':'4', '5':'Spoken only'}
            fig.update_xaxes(  # Changed from update_yaxes to update_xaxes
                title_text="Average difference in ratings",
                fixedrange=True,
                tickvals=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                ticktext=[Rating_map[str(i)] for i in range(-5, 6)]
            )
        
        return fig
    
    # Handle diverging stacked bar chart mode
    if plot_mode == "diverging":
        return create_diverging_stacked_bar_plot(df, items, modes, groupby, variety_color_map, pairs, meta, balanced_informants, sortby, use_imputed)

    # Handle correlation matrix mode
    if plot_mode == "correlation_matrix":
        return create_correlation_matrix_plot(df, items, balanced_informants, pairs, use_imputed)
    
    # Handle split by variety mode
    if plot_mode == "split_by_variety":
        # Enable split_by_variety flag and use normal plot logic
        split_by_variety = True
    
    # Handle normal mode with axis rotation
    if plot_mode == "normal":
        return create_normal_plot_rotated(df, items, modes, groupby, variety_color_map, pairs, meandf)
    
    # Handle informant mean boxplot mode
    if plot_mode == "informant_boxplot":
        return create_informant_mean_boxplot(df, items, modes, groupby, variety_color_map, pairs, meandf, sortby, use_imputed, balanced_informants)
    
    # Legacy code for backward compatibility (will be removed)
    if len(modes) == 1 and not pairs:
    # for error bars:
        plot_groups = sorted(df['group'].unique(), reverse=True)  # Reverse order for plot, but legend will show normal order
        legend_groups = sorted(df['group'].unique())  # Keep original order for legend ranking
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for plot_group in plot_groups:
            tempDF = df[df['group'] == plot_group]
            
            # Get color for this group if we're grouping by variety
            color = None
            if variety_color_map and plot_group in variety_color_map:
                color = variety_color_map[plot_group]
            
            # Calculate legend rank to maintain alphabetical order in legend
            legend_rank = legend_groups.index(plot_group) if plot_group in legend_groups else 999
                
            if pd.api.types.is_integer_dtype(plot_group):
                fig.add_trace(
                go.Scatter(
                    x=tempDF['item'],
                    y=tempDF['mean'],
                    name=str(plot_group),
                    mode="markers",
                    marker=dict(
                        symbol=plot_group,
                        color=color
                    ),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=tempDF['upper_ci'],
                        arrayminus=tempDF['lower_ci']
                        ),
                        legendrank=legend_rank,  # Maintain legend order
                        customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                        hovertemplate='<br>'.join([
                            'Item: %{customdata[0]}',
                            'Group: %{customdata[1]}',
                            'Avg rating: %{customdata[2]:.2f}',
                            'Number of participants: %{customdata[3]}',
                            'Sentence: %{customdata[4]}',
                            'Item name: %{customdata[5]}',
                            'Item group: %{customdata[6]}',
                            'Ewave feature: %{customdata[7]}',
                            'Twin item: %{customdata[8]}'
                        ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                        hoverinfo='text'
                    ),secondary_y=False
                )
            else:
                fig.add_trace(
                go.Scatter(
                    x=tempDF['item'],
                    y=tempDF['mean'],
                    name=plot_group,
                    mode="markers",
                    marker=dict(color=color) if color else None,
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=tempDF['upper_ci'],
                        arrayminus=tempDF['lower_ci']
                        ),
                        legendrank=legend_rank,  # Maintain legend order
                        customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                        hovertemplate='<br>'.join([
                            'Item: %{customdata[0]}',
                            'Group: %{customdata[1]}',
                            'Avg rating: %{customdata[2]:.2f}',
                            'Number of participants: %{customdata[3]}',
                            'Sentence: %{customdata[4]}',
                            'Item name: %{customdata[5]}',
                            'Item group: %{customdata[6]}',
                            'Ewave feature: %{customdata[7]}',
                            'Twin item: %{customdata[8]}'
                        ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                        hoverinfo='text'
                    ),secondary_y=False
                )
        # importance plot
        tempdf = df
        # Use importance ordering for consistent item sorting
        tempDF = df[['item','importance','mode']].drop_duplicates().sort_values("importance",ascending=False)
        fig.update_xaxes(categoryorder='array', categoryarray=tempDF['item'].to_list(), range=[-1,10])
        
        fig.update_layout(
            height=400
        )
    elif len(modes) == 1 and pairs:

        plot_groups = sorted(df['group'].unique(), reverse=True)  # Reverse order for plot, but legend will show normal order
        legend_groups = sorted(df['group'].unique())  # Keep original order for legend ranking
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for plot_group in plot_groups:
            tempDF = df[df['group'] == plot_group]
            
            # Get color for this group if we're grouping by variety
            color = None
            if variety_color_map and plot_group in variety_color_map:
                color = variety_color_map[plot_group]
            
            # Calculate legend rank to maintain alphabetical order in legend
            legend_rank = legend_groups.index(plot_group) if plot_group in legend_groups else 999
                
            if pd.api.types.is_integer_dtype(plot_group):
                fig.add_trace(
                go.Scatter(
                    x=tempDF['item'],
                    y=tempDF['mean'],
                    name=str(plot_group),
                    mode="markers",
                    marker=dict(
                        symbol=plot_group,
                        color=color
                    ),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=tempDF['upper_ci'],
                        arrayminus=tempDF['lower_ci']
                        ),
                        legendrank=legend_rank,  # Maintain legend order
                        customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']) else None,
                        hovertemplate='<br>'.join([
                            'Item: %{customdata[0]}',
                            'Group: %{customdata[1]}',
                            'Avg rating: %{customdata[2]:.2f}',
                            'Number of participants: %{customdata[3]}',
                            'Sentence: %{customdata[4]}',
                            'Item name: %{customdata[5]}',
                            'Item group: %{customdata[6]}',
                            'Ewave feature: %{customdata[7]}'
                        ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']) else None,
                        hoverinfo='text'
                    ),secondary_y=False
                )
            else:
                fig.add_trace(
                go.Scatter(
                    x=tempDF['item'],
                    y=tempDF['mean'],
                    name=plot_group,
                    mode="markers",
                    marker=dict(color=color) if color else None,
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=tempDF['upper_ci'],
                        arrayminus=tempDF['lower_ci']
                        ),
                        legendrank=legend_rank,  # Maintain legend order
                        customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']) else None,
                        hovertemplate='<br>'.join([
                            'Item: %{customdata[0]}',
                            'Group: %{customdata[1]}',
                            'Avg rating: %{customdata[2]:.2f}',
                            'Number of participants: %{customdata[3]}',
                            'Sentence: %{customdata[4]}',
                            'Item name: %{customdata[5]}',
                            'Item group: %{customdata[6]}',
                            'Ewave feature: %{customdata[7]}'
                        ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']) else None,
                        hoverinfo='text'
                    ),secondary_y=False
                )
        # importance plot
        tempdf = df
        # Use importance ordering for consistent item sorting
        tempDF = df[['item','importance','mode']].drop_duplicates().sort_values("importance",ascending=False)
        fig.update_xaxes(categoryorder='array', categoryarray=tempDF['item'].to_list(), range=[-1,10])
        
        fig.update_layout(
            height=400
        )
    else:
        plot_groups = sorted(df['group'].unique(), reverse=True)  # Reverse order for plot, but legend will show normal order
        legend_groups = sorted(df['group'].unique())  # Keep original order for legend ranking
        axisorder = []
        fig = make_subplots(rows=2,cols=1,specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
        for mode in modes:
            if mode == 'spoken':
                row=1
            else:
                row=2
            for plot_group in plot_groups:
                tempDF = df[((df['group'] == plot_group) & (df['mode'] == mode))]
                
                # Get color for this group if we're grouping by variety
                color = None
                if variety_color_map and plot_group in variety_color_map:
                    color = variety_color_map[plot_group]
                
                # Calculate legend rank to maintain alphabetical order in legend
                legend_rank = legend_groups.index(plot_group) if plot_group in legend_groups else 999
                
                if pd.api.types.is_integer_dtype(plot_group):
                    fig.add_trace(
                    go.Scatter(
                        x=tempDF['item'],
                        y=tempDF['mean'],
                        name=f"{str(plot_group)} ({mode})",
                        mode="markers",
                        marker=dict(symbol=plot_group, color=color),
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=tempDF['upper_ci'],
                            arrayminus=tempDF['lower_ci']
                            ),
                        legendrank=legend_rank,  # Maintain legend order
                        customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']],
                        hovertemplate='<br>'.join([
                            'Item: %{customdata[0]}',
                            'Group: %{customdata[1]}',
                            'Avg rating: %{customdata[2]:.2f}',
                            'Number of participants: %{customdata[3]}',
                            'Sentence: %{customdata[4]}',
                            'Item name: %{customdata[5]}',
                            'Item group: %{customdata[6]}',
                            'Ewave feature: %{customdata[7]}',
                            'Twin item: %{customdata[8]}'
                    ]), hoverinfo='text'
                    ),row=row,col=1,secondary_y=False
                )
                else:
                    fig.add_trace(
                    go.Scatter(
                        x=tempDF['item'],
                        y=tempDF['mean'],
                        name=f"{plot_group} ({mode})",
                        mode="markers",
                        marker=dict(color=color) if color else None,
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=tempDF['upper_ci'],
                            arrayminus=tempDF['lower_ci']
                            ),
                        legendrank=legend_rank,  # Maintain legend order
                        customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']],
                        hovertemplate='<br>'.join([
                            'Item: %{customdata[0]}',
                            'Group: %{customdata[1]}',
                            'Avg rating: %{customdata[2]:.2f}',
                            'Number of participants: %{customdata[3]}',
                            'Sentence: %{customdata[4]}',
                            'Item name: %{customdata[5]}',
                            'Item group: %{customdata[6]}',
                            'Ewave feature: %{customdata[7]}',
                            'Twin item: %{customdata[8]}'
                    ]), hoverinfo='text'
                    ),row=row,col=1,secondary_y=False
                )


            tempDF=df[df['mode'] == mode][['item','mode']]
            axisorder.append(tempDF)
            # Use meandf ordering for consistent item sorting
            fig.update_xaxes(categoryorder='array', categoryarray=meandf['item'].to_list())
            
        axisorder = pd.concat(axisorder)
        axisorder.loc[:,'row'] = 1
        axisorder.loc[axisorder['mode']=='written','row'] = 2
        #axisorder.sort_values(by=['row','importance'],inplace=True,ascending=[True,False])
        axisorder.reset_index(inplace=True)
        #spokenrange = [-0.5,10]
        #writtenrange = axisorder.loc[axisorder['mode']=='written'].index[0:10].to_list()
        #writtenrange = [writtenrange[0]-0.5,writtenrange[-1]]
        # Use meandf ordering for both subplots to ensure consistent sorting
        fig['layout']['xaxis1']['categoryarray'] = tuple(meandf['item'].to_list())
        fig['layout']['xaxis2']['categoryarray'] = tuple(meandf['item'].to_list())
        fig.update_layout(
            height=800,
            xaxis1=dict(range=[-0.5,10]),xaxis2=dict(range=[-0.5,10])
        )

    # Customize layout
    fig.update_layout(
        title='Mean (95% CI)',
        xaxis_title='Grammatical items',
        yaxis_title='Mean ratings',
        template='simple_white')
    
    # Add top x-axis mirror for better readability
    if len(modes) == 1:
        # Single mode - add top axis with mirrored ticks and y-axis mirroring
        fig.update_xaxes(
            mirror=True,
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        )
        fig.update_yaxes(mirror=True)  # Add y-axis mirroring for single mode
        # Add divider lines every 3 items for better readability
        if not pairs and len(meandf) > 3:
            for i in range(3, len(meandf), 3):
                fig.add_vline(
                    x=i-0.5, 
                    line_dash="dot", 
                    line_color="gray", 
                    line_width=1,
                    opacity=0.5
                )
    else:
        # Multi-mode - add top axes for both subplots with mirrored ticks
        fig.update_xaxes(
            mirror=True,
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        )
        fig.update_yaxes(mirror=True)  # Add y-axis mirroring for multi-mode
        # Add divider lines every 3 items for both subplots
        if not pairs and len(meandf) > 3:
            for i in range(3, len(meandf), 3):
                fig.add_vline(
                    x=i-0.5, 
                    line_dash="dot", 
                    line_color="gray", 
                    line_width=1,
                    opacity=0.5,
                    row=1, col=1
                )
                fig.add_vline(
                    x=i-0.5, 
                    line_dash="dot", 
                    line_color="gray", 
                    line_width=1,
                    opacity=0.5,
                    row=2, col=1
                )
    
    fig.update_yaxes(title_text="Average rating", secondary_y=False,fixedrange=True,range=[0,5.1])
    fig.update_yaxes(title_text="RF importance", secondary_y=True,fixedrange=True)   
    if not pairs:
        # Always show all rating labels when sorting by mean or for better readability
        fig.update_yaxes(
            title_text="Average rating",
            secondary_y=False,
            fixedrange=True,
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=[Rating_map[str(i)] for i in range(6)],
            range=[-0.2, 5.2]  # Ensure all labels are visible
        )
    else:
        fig.update_yaxes(
            title_text="Stylistic difference",
            secondary_y=False,
            fixedrange=True,
            tickvals=[-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5],
            ticktext=['Written only','-4','-3','-2','-1','Neutral','1','2','3','4','Spoken only'],
            range=[-5,5]
        )

    return fig


def drawParticipantsTree(informants):
    #data = retrieve_data.getInformantDataGrammar(columns=['InformantID', 'MainVariety','Year'], imputed=True)
    data = informants.copy(deep=True)
    data = data.loc[:,['InformantID', 'MainVariety','Year']]
    # for each country draw a tree with the years and informants
    countries = data['MainVariety'].unique()
    treeData = []
    for country in countries:
        countryData = data[data['MainVariety'] == country]
        years = countryData['Year'].unique()
        countryTree = {
            'value': country,
            'label': country,
            'children': [{'label': year, 'value': country + '_' + year, 'children': [{'value': informant, 'label': informant} for informant in countryData[countryData['Year'] == year]['InformantID']]} for year in years]
        }
        treeData.append(countryTree)
    #treeData = [{'title': 'Informants', 'key': 'informants', 'children': treeData}]
    return treeData

def drawGrammarItemsTree(grammarMeta,pairs=False):
    #data = retrieve_data.getGrammarData(imputed=True)
    #data = grammarData.copy(deep=True)
    meta = grammarMeta.copy(deep=True)
    meta.loc[:,'letter'] = meta['question_code'].str[0]
    meta.loc[:,'numbering'] = meta['question_code'].str.extract(r'(\d+)')
    meta.loc[:,'numbering']= meta.loc[:,'numbering'].astype(int)
    meta.sort_values(by=['letter','numbering'],ascending=True,inplace=True)
    if not pairs:
        spokenLetters = ['A','B','C','D','E','F']
        writtenLetters = ['G','H','I','J','K','L','M','N']
        spokenChildren = [{'label': letter, 'value': letter, 'children':[{'label': x + ': ' + y, 'value': x} for x,y in zip(meta.loc[meta['letter']==letter]['question_code'],meta.loc[meta['letter']==letter]['feature'])]} for letter in spokenLetters ]
        writtenChildren = [{'label': letter, 'value': letter, 'children':[{'label': x + ': ' + y, 'value': x} for x,y in zip(meta.loc[meta['letter']==letter]['question_code'],meta.loc[meta['letter']==letter]['feature'])]} for letter in writtenLetters ]
        SpokenCols = {
        'label': 'Spoken section',
        'value': 'spoken',
        'children': spokenChildren
        }

        WrittenCols = {
        'label': 'Written section',
        'value': 'written',
        'children':writtenChildren
        }
        
        
        treeData = [SpokenCols, WrittenCols]
    else:
        spokenLetters = ['A','B','C','D','E','F']
        spokenChildren = [{'label': letter, 'value': letter, 'children':[{'label': x + ': ' + y, 'value': x} for x,y in zip(meta.loc[meta['letter']==letter]['item_pair'],meta.loc[meta['letter']==letter]['feature'])]} for letter in spokenLetters ]
        SpokenCols = {
        'label': 'Item pairs',
        'value': 'spoken',
        'children': spokenChildren
        }

        treeData = [SpokenCols]
    return treeData

def getAgeGenderPlot(informants):
    # Get age and gender data from informants
    try:
        data = informants.copy(deep=True)
        # Count NaN values in the Age and Gender columns
        nan_rows = data[data['Age'].isna() | data['Gender'].isna()]
        nan_count = len(nan_rows)  # Count of NaN rows
        
        # Drop rows with NaN in 'Age' or 'Gender'
        data = data.dropna(subset=['Age', 'Gender'])
        
        # Convert Age to float for plotting
        data['Age'] = data['Age'].astype(float)
        

        # Create a histogram using Plotly's built-in histogram
        fig = px.histogram(
            data,
            x='Age',
            color='Gender',
            template="simple_white",
            category_orders={"Gender": ["Female", "Male", "Non-binary", "NA"]},  # Order of gender categories
            nbins=25  # Adjust the number of bins as needed
        )
        
        # Update axes and layout
        fig.update_xaxes(title_text='Age')
        fig.update_yaxes(title_text='Informants')
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize padding
            showlegend=True,  # Show legend for gender categories
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = getEmptyPlot()  # Use an empty plot if there's an error

    return fig

def getMainVarietiesPlot(informants):
    try:
        # Get main variety data from informants
        data = informants.copy(deep=True)
        
        # Count occurrences of each variety
        variety_counts = data['MainVariety'].value_counts()
        
        # Group varieties with fewer than 10 informants into "Other"
        data['MainVariety'] = data['MainVariety'].apply(
            lambda x: x if variety_counts[x] >= 10 else 'Other'
        )
        
        # Recalculate counts grouped by MainVariety and Year
        grouped_counts = data.groupby(['MainVariety', 'Year'], observed=True).size().reset_index(name='counts')
        
        # Calculate overall frequency of each variety for sorting
        overall_counts = grouped_counts.groupby('MainVariety', observed=True)['counts'].sum().reset_index()
        overall_counts = overall_counts.sort_values(by='counts', ascending=False)
        
        # Ensure years are ordered
        grouped_counts['Year'] = grouped_counts['Year'].astype(int)  # Convert Year to string for consistent ordering
        grouped_counts = grouped_counts.merge(overall_counts, on='MainVariety', suffixes=('', '_total'))
        grouped_counts = grouped_counts.sort_values(by=['Year'],ascending=True)
        VarietyOrder = overall_counts['MainVariety'].tolist() + ['Other'] if 'Other' in overall_counts['MainVariety'].tolist() else overall_counts['MainVariety'].tolist()
        height = len(VarietyOrder) * 25
        if height < 150:
            height = 150
        # Create a bar plot with the grouped varieties, colored by year, and swap axes
        fig = px.bar(
            grouped_counts,
            y='MainVariety',
            x='counts',
            color='Year',
            template="simple_white",
            barmode='stack',  
            color_continuous_scale='Blues_r',
            category_orders={
                'MainVariety': VarietyOrder},
            height=height,
            hover_data={'counts': True, 'counts_total': True}   # Adjust height based on number of varieties
        )
        
        # Update axes and layout
        fig.update_xaxes(title_text='Informants')
        fig.update_traces(marker_line_width=0.5, marker_line_color="gray")
        fig.update_yaxes(title_text='Main Variety',automargin=True)  # Ensure all y-axis labels are visible
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize padding
            showlegend=False,  # Show legend for years
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = getEmptyPlot()  # Use an empty plot if there's an error
    return fig

def getYearsLivedOutsidePlot(informants):
    try:
        # Get data for years lived outside home country
        data = informants.copy(deep=True)
        
        # Count NaN values in 'YearsLivedOutside'
        nan_count = data['YearsLivedOutside'].isna().sum()
        
        # Drop rows with missing values in 'YearsLivedOutside'
        data = data.dropna(subset=['YearsLivedOutside'])
        
        # Create a histogram
        fig = px.histogram(
            data,
            x='YearsLivedOutside',
            nbins=20,  # Adjust the number of bins as needed
            title=f'Years Lived Outside Home Country (NA Count: {nan_count})',
            template="simple_white"
        )
        
        # Update axes and layout
        fig.update_xaxes(title_text='Years Lived Outside')
        fig.update_yaxes(title_text='Number of Informants')
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),  # Minimize padding
            showlegend=False,  # Hide legend if not needed
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = getEmptyPlot()  # Use an empty plot if there's an error
    return fig

def getYearsLivedOtherEnglishPlot(informants):
    try:
        # Get data for years lived in other English-speaking countries
        data = informants.copy(deep=True)
        
        # Count NaN values in 'YearsLivedOtherE'
        nan_count = data['YearsLivedOtherEnglish'].isna().sum()
        
        # Drop rows with missing values in 'YearsLivedOtherE'
        data = data.dropna(subset=['YearsLivedOtherEnglish'])
        
        # Create a histogram
        fig = px.histogram(
            data,
            x='YearsLivedOtherEnglish',
            nbins=20,  # Adjust the number of bins as needed
            title=f'Years Lived in Other English-Speaking Countries (NA Count: {nan_count})',
            template="simple_white"
        )
        
        # Update axes and layout
        fig.update_xaxes(title_text='Years Lived in Other English-Speaking Countries')
        fig.update_yaxes(title_text='Number of Informants')
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),  # Minimize padding
            showlegend=False,  # Hide legend if not needed
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = getEmptyPlot()  # Use an empty plot if there's an error
    return fig

def getRatioMainVarietyPlot(informants):
    try:
        # Get data for Ratio_MainVariety
        data = informants.copy(deep=True)
        
        # Count NaN values in 'Ratio_MainVariety'
        nan_count = data['RatioMainVariety'].isna().sum()
        
        # Drop rows with missing values in 'Ratio_MainVariety'
        data = data.dropna(subset=['RatioMainVariety'])
        
        # Create a histogram plot
        fig = px.histogram(
            data,
            x='RatioMainVariety',
            title=f'Ratio of Years Lived in Main Variety to Age (NA Count: {nan_count})',
            template="simple_white"
        )
        
        # Update axes and layout
        fig.update_xaxes(title_text='Ratio (Years Lived in Main Variety / Age)')
        fig.update_yaxes(title_text='Informants')
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize padding
            showlegend=False,  # Hide legend if not needed
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = getEmptyPlot()  # Use an empty plot if there's an error   
    return fig

def getFloatHistogramPlot (informants,ColName="RatioMainVariety"):
    try:
        # Get data for Ratio_MainVariety
        data = informants.copy(deep=True)
        
        # Count NaN values in 'Ratio_MainVariety'
        nan_count = data[ColName].isna().sum()
        
        # Drop rows with missing values in 'Ratio_MainVariety'
        data = data.dropna(subset=[ColName])
        hist_data = [np.array(data[ColName].to_list())]
        fig = ff.create_distplot(hist_data,show_hist=False,group_labels=[ColName])
            
        # Create a histogram plot
        """ fig = px.histogram(
            data,
            x=ColName,
            title=f'{ColName} (NA Count: {nan_count})',
            template="simple_white"
        ) """
        
        # Update axes and layout
        fig.update_xaxes(title_text=ColName)
        #fig.update_yaxes(title_text='Informants')
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize padding
            showlegend=False,
            template="simple_white",
            height=200  # Hide legend if not needed
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = getEmptyPlot()  # Use an empty plot if there's an error

    return fig

def getFrequencyBarPlot(informants, ColName="YearsLivedOutside"):
    """
    Creates a violin plot for numerical data like years lived outside.
    Better visualization for showing distribution shape and density.
    """
    try:
        data = informants.copy(deep=True)
        
        # Count NaN values
        nan_count = data[ColName].isna().sum()
        
        # Drop rows with missing values
        data = data.dropna(subset=[ColName])
        
        if len(data) == 0:
            return getEmptyPlot()
        
        # Create violin plot
        fig = px.violin(
            data, 
            y=ColName,
            title=f'{ColName} Distribution (NA Count: {nan_count})',
            template="simple_white",
            box=True,  # Show box plot inside violin
            points="outliers"  # Show outlier points
        )
        
        # Update layout for better appearance
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            template="simple_white",
            height=200,
            xaxis_title="",
            yaxis_title=ColName.replace("YearsLived", "Years lived ").replace("Ratio", "Ratio ")
        )
        
        # Remove x-axis ticks since we only have one violin
        fig.update_xaxes(showticklabels=False)
        
        # Add some styling to the violin
        fig.update_traces(
            fillcolor="lightblue",
            line_color="darkblue",
            opacity=0.7
        )
        
        fig.update_layout(modebar_remove=True)
        
    except Exception as e:
        print(f"Error creating violin plot: {e}")
        fig = getEmptyPlot()
    
    return fig


def getCategoryHistogramPlot(informants, ColName="PrimarySchool", GroupOther=True, split="", GenderDistribution=True):
    try: # do it right sometime, works for now
        data = informants.copy(deep=True)
        
        # Fill missing values and standardize empty strings
        data[ColName] = data[ColName].fillna('NA')
        data.loc[data[ColName] == "", ColName] = "NA"
        if split != "":
            data = data.assign(**{ColName: data[ColName].str.split(',')}).explode(ColName)
            data[ColName] = data[ColName].str.strip().str.capitalize()

        if GenderDistribution:
            col_counts = data.groupby([ColName,'Gender'], observed=True).size().reset_index(name='counts')
            col_counts.columns = [ColName, 'Gender', 'counts']
            # group catergories with fewer than 10 occurrences per Gender into "Other
            if GroupOther:
                col_counts[ColName] = col_counts[ColName].apply(
                    lambda x: x if ((col_counts[col_counts[ColName] == x]['counts'].sum() >= 10) | (x == 'NA') | (x == 'ND')) else 'Other'
                )
            col_counts = col_counts.groupby([ColName,'Gender'], as_index=False, observed=True)['counts'].sum()
            # calculate overall frequency of each category for sorting
            overall_counts = col_counts.groupby(ColName, observed=True)['counts'].sum().reset_index()
            overall_counts = overall_counts.sort_values(by='counts', ascending=False)
            col_counts = col_counts.merge(overall_counts, on=ColName, suffixes=('', '_total'))
            col_counts = col_counts.sort_values(by=['counts_total','Gender'], ascending=[False,True])
        else:
            col_counts = data[ColName].value_counts().reset_index()
            col_counts.columns = [ColName, 'counts']
            if GroupOther:
                col_counts[ColName] = col_counts[ColName].apply(
                    lambda x: x if ((col_counts.loc[col_counts[ColName] == x, 'counts'].values[0] >= 10) | (x == 'NA') | (x == 'ND')) else 'Other'
                )
            col_counts = col_counts.groupby(ColName, as_index=False)['counts'].sum()
            col_counts = col_counts.sort_values(by='counts', ascending=False)


        col_Order = list(dict.fromkeys(col_counts[ColName].tolist()))
        col_Order = [category for category in col_Order if category not in ['NA', 'ND', 'Other']] + [category for category in ['NA', 'ND', 'Other'] if category in col_Order]
        height = len(col_Order) * 25
        if height < 200:
            height = 200
    
        # If GenderDistribution is True, include Gender in the plot
        if GenderDistribution:
            
            # Create a bar plot with Gender as color
            fig = px.bar(
                col_counts,
                y=ColName,
                x='counts',
                color='Gender',
                template="simple_white",
                barmode='stack',
                category_orders={ColName: col_Order},
                height=height,
                hover_data={'counts': True}
            )
        else:
            # Create a bar plot without Gender
            fig = px.bar(
                col_counts,
                y=ColName,
                x='counts',
                template="simple_white",
                barmode='stack',
                category_orders={ColName: col_Order},
                height=height,
                hover_data={'counts': True}
            )

        # Update axes and layout
        fig.update_xaxes(title_text='Informants')
        fig.update_traces(marker_line_width=0.5, marker_line_color="gray")
        fig.update_yaxes(title_text=ColName, automargin=True)
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=GenderDistribution,  # Show legend only if GenderDistribution is True
        )
        fig.update_layout(modebar_remove=True)

    except Exception as e:
        # Use an empty plot if there's an error
        fig = getEmptyPlot()

    return fig

def get_matching_preset(checked_items, item_presets):
    """
    Check if list of checked items matches any preset and return preset name.
    
    Args:
        checked_items (list): List of selected grammar items
    
    Returns:
        str: Name of matching preset or "Custom" if no match found
    """
    # Sort checked items to ensure order-independent comparison
    checked_items = sorted(checked_items)
    
    # Compare with each preset
    for preset in item_presets:
        if sorted(preset['value']) == checked_items:
            return preset['label']
            
    return "Custom"
    # Sort checked items to ensure order-independent comparison
    checked_items = sorted(checked_items)
    
    # Compare with each preset
    for preset in item_presets:
        if sorted(preset['value']) == checked_items:
            return preset['label']
            
    return "Custom"

def construct_initial_hoverinfo(initialPlot):
    """Construct hoverinfo data from the initial UMAP plot at startup"""
    hoverinfo_data = {}
    
    if initialPlot is not None:
        import plotly.graph_objects as go
        fig = go.Figure(initialPlot)
        # Extract and store hoverinfo from all traces
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'hovertemplate') and trace.hovertemplate is not None:
                if hasattr(trace, 'ids') and trace.ids is not None:
                    # Store hovertemplate for each point by ID
                    for j, point_id in enumerate(trace.ids):
                        if isinstance(trace.hovertemplate, list):
                            hoverinfo_data[point_id] = trace.hovertemplate[j] if j < len(trace.hovertemplate) else trace.hovertemplate[0]
                        else:
                            hoverinfo_data[point_id] = trace.hovertemplate
                else:
                    # Fallback: store by trace index if no IDs
                    hoverinfo_data[f"trace_{i}"] = trace.hovertemplate
    
    return hoverinfo_data

# Utility function to hash a list of participant IDs
def _participants_hash(participants):
    if participants is None:
        return None
    # Convert to tuple for consistent hashing
    return hashlib.md5(str(tuple(sorted(participants))).encode()).hexdigest()

def create_diverging_stacked_bar_plot(df_orig, items, modes, groupby, variety_color_map, pairs, meta, informants, sortby="mean", use_imputed=True):
    """Create a diverging stacked bar chart for rating distributions"""
    
    # Get raw data with individual ratings (not aggregated)
    raw_df = retrieve_data.getGrammarData(imputed=use_imputed, items=items, pairs=pairs, participants=informants)

    # Apply same grouping logic as main function
    if groupby == "variety":
        raw_df['group'] = raw_df['MainVariety']
    elif groupby == "vtype" or groupby == "vtype_balanced":
        # Use centralized variety mapping
        variety_mapping = get_variety_mapping()
        raw_df['group'] = raw_df['MainVariety'].map(variety_mapping).fillna("Other")
        raw_df = raw_df[raw_df['group'] != "Other"]
    elif groupby == "gender":
        # Use normalized gender column
        raw_df['group'] = raw_df['Gender_normalized'] if 'Gender_normalized' in raw_df.columns else raw_df['Gender']
    
    # Melt the data to get individual responses
    info_cols = ['InformantID', 'group']
    if 'section' in raw_df.columns:
        info_cols.append('section')
        
    melted_df = raw_df.melt(id_vars=info_cols, value_vars=items, var_name='item', value_name='rating')
    melted_df['rating'] = pd.to_numeric(melted_df['rating'], errors='coerce')
    melted_df = melted_df.dropna(subset=['rating'])
    
    # Add mode column if section exists
    if 'section' in melted_df.columns:
        melted_df['mode'] = melted_df['section'].str.lower()
    
    # Handle pairs (spoken-written differences) - ratings range from -5 to 5
    if pairs:
        # For pairs, we need to handle negative ratings differently
        rating_range = list(range(-5, 6))  # -5 to 5
        # Define colors for difference scale (-5=red, 0=neutral, 5=green)
        rating_colors = {
            -5: '#b71c1c',  # Dark Red
            -4: '#d32f2f',  # Red
            -3: '#f57c00',  # Orange
            -2: '#ff9800',  # Light Orange
            -1: '#ffc107',  # Yellow
            0: '#9e9e9e',   # Gray (neutral)
            1: '#cddc39',   # Light Green
            2: '#8bc34a',   # Yellowish Green
            3: '#4caf50',   # Green
            4: '#2e7d32',   # Dark Green
            5: '#1b5e20'    # Very Dark Green
        }
    else:
        # Normal scale 0-5
        rating_range = list(range(0, 6))  # 0 to 5
        # Define colors for rating scale (0=red, 3-5=green hues)
        rating_colors = {
            0: '#d32f2f',   # Red
            1: '#f57c00',   # Orange
            2: '#fbc02d',   # Yellow
            3: '#8bc34a',   # Yellowish Green
            4: '#4caf50',   # Green
            5: '#2e7d32'    # Pure Green
        }
    
    # Create rating distribution for each item-group combination
    if 'mode' in melted_df.columns:
        rating_counts = melted_df.groupby(['item', 'group', 'mode', 'rating']).size().reset_index(name='count')
        rating_totals = melted_df.groupby(['item', 'group', 'mode']).size().reset_index(name='total')
        rating_data = rating_counts.merge(rating_totals, on=['item', 'group', 'mode'])
    else:
        rating_counts = melted_df.groupby(['item', 'group', 'rating']).size().reset_index(name='count')
        rating_totals = melted_df.groupby(['item', 'group']).size().reset_index(name='total')
        rating_data = rating_counts.merge(rating_totals, on=['item', 'group'])
    
    # Calculate percentages and add participant counts
    rating_data['percentage'] = (rating_data['count'] / rating_data['total']) * 100
    
    # Add participant counts by calculating unique participants for each rating/item/group
    participant_counts = melted_df.groupby(['item', 'group', 'rating'])['InformantID'].nunique().reset_index()
    participant_counts.columns = ['item', 'group', 'rating', 'participant_count']
    
    # Also get total participants per item-group for reference
    total_participants = melted_df.groupby(['item', 'group'])['InformantID'].nunique().reset_index()
    total_participants.columns = ['item', 'group', 'total_participants']
    
    # Merge participant information
    if 'mode' in rating_data.columns:
        rating_data = rating_data.merge(participant_counts, on=['item', 'group', 'rating'], how='left')
        rating_data = rating_data.merge(total_participants, on=['item', 'group'], how='left')
    else:
        rating_data = rating_data.merge(participant_counts, on=['item', 'group', 'rating'], how='left')
        rating_data = rating_data.merge(total_participants, on=['item', 'group'], how='left')
    
    # Merge with meta information to get sentences and other details
    try:
        if pairs:
            meta_cols = ['item_pair','item', 'feature', 'variant_detail', 'group_finegrained', 'feature_ewave']
            item_col = 'item_pair'
        else:
            meta_cols = ['question_code','item', 'feature', 'variant_detail', 'group_finegrained', 'feature_ewave']
            item_col = 'question_code'
            if 'sentence' in meta.columns:
                meta_cols.append('sentence')
        
        # Get available columns from meta and check if item_col exists
        available_meta_cols = [col for col in meta_cols if col in meta.columns]
        if available_meta_cols and item_col in meta.columns:
            meta_subset = meta[available_meta_cols + [item_col]].drop_duplicates()
            meta_subset = meta_subset.loc[:, ~meta_subset.columns.duplicated()]
            rating_data = rating_data.merge(
                meta_subset, 
                left_on='item', 
                right_on=item_col, 
                how='left'
            )
            if not pairs:
                # rename item_x and item_y
                rating_data = rating_data.rename(columns={'item_x': 'item', 'item_y': 'sentence'})
            else:
                rating_data = rating_data.rename(columns={'item_x': 'item', 'item_y': 'sentence'})
    except Exception as e:
        print(f"Warning: Could not merge meta information: {e}")
        # Continue without meta information if merge fails
    
    # Sort items by specified criteria
    if sortby == "alpha":
        # Use natural sorting for item labels like A1, A2, ..., A10, A11, etc.
        import re
        def natural_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
        item_means_df = pd.DataFrame({'item': melted_df['item'].unique()})
        item_means_df['item_sort'] = item_means_df['item'].apply(natural_key)
        sorted_items = item_means_df.sort_values(by='item_sort', ascending=False)['item'].tolist()
    elif sortby == "sd":
        item_stats = melted_df.groupby('item')['rating'].agg(['mean', 'std']).reset_index()
        sorted_items = item_stats.sort_values(by='std', ascending=False)['item'].tolist()
    else:  # sortby == "mean" or default
        item_means = melted_df.groupby('item')['rating'].mean().sort_values(ascending=True)
        sorted_items = item_means.index.tolist()
    
    # Get unique groups and check if we exceed 150 bars
    groups = list(reversed(sort_groups_for_plot(rating_data['group'].unique(), groupby)))
    total_bars = len(sorted_items) * len(groups)
    
    max_bars = 275
    if total_bars > max_bars:
        # Return warning plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"Too many bars to display ({total_bars} > {max_bars}).<br>"
                 f"Please reduce the number of items or use fewer groups.<br>"
                 f"Items: {len(sorted_items)}, Groups: {len(groups)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red"),
            align="center"
        )
        fig.update_layout(
            title="Too Many Bars - Please Reduce Selection",
            template='simple_white',
            height=400
        )
        return fig
    
    # Create figure based on number of modes
    if len(modes) <= 1 or 'mode' not in melted_df.columns:
        fig = go.Figure()
        
        # Create separate bars for each group-item combination
        groups = list(reversed(sort_groups_for_plot(rating_data['group'].unique(), groupby)))
        
        # Calculate spacing between items and groups
        item_positions = {}
        current_pos = 0
        
        for item in sorted_items:
            item_positions[item] = {}
            for i, group in enumerate(groups):
                item_positions[item][group] = current_pos + i * 0.8  # Space groups within each item
            current_pos += len(groups) + 1  # Space between items
        
        # Create y-axis labels and tick positions
        y_labels = []
        y_positions = []
        
        for item in sorted_items:
            for group in groups:
                y_labels.append(f"{item} - {group}")
                y_positions.append(item_positions[item][group])
        
        # Create diverging stacked bars
        if pairs:
            # For pairs: negative ratings (left side), neutral split, positive ratings (right side)
            negative_ratings = [r for r in rating_range if r < 0]
            positive_ratings = [r for r in rating_range if r > 0]
            neutral_rating = 0
            
            # Plot neutral rating left half first (closest to center)
            neutral_x_left = []
            neutral_x_right = []
            y_values_neutral = []
            neutral_custom_data = []
            
            for item in sorted_items:
                for group in groups:
                    group_item_data = rating_data[
                        (rating_data['rating'] == neutral_rating) & 
                        (rating_data['group'] == group) & 
                        (rating_data['item'] == item)
                    ]
                    
                    if len(group_item_data) > 0:
                        percentage = group_item_data['percentage'].iloc[0]
                        participant_count = group_item_data['participant_count'].iloc[0] if 'participant_count' in group_item_data.columns else 0
                        sentence = group_item_data['sentence'].iloc[0] if 'sentence' in group_item_data.columns else "N/A"
                        feature = group_item_data['feature'].iloc[0] if 'feature' in group_item_data.columns else "N/A"
                    else:
                        percentage = 0
                        participant_count = 0
                        sentence = "N/A"
                        feature = "N/A"
                        
                    half_percentage = percentage / 2
                    
                    neutral_x_left.append(-half_percentage)
                    neutral_x_right.append(half_percentage)
                    y_values_neutral.append(item_positions[item][group])
                    neutral_custom_data.append([percentage, participant_count, item, group, sentence, feature])
            
            # Add left half of neutral (closest to center on left side)
            fig.add_trace(go.Bar(
                x=neutral_x_left,
                y=y_values_neutral,
                name='Rating 0 (left)',
                marker_color=rating_colors[neutral_rating],
                orientation='h',
                showlegend=False,  # Don't show in legend to avoid confusion
                hovertemplate='<br>'.join([
                    'Rating 0 (Neutral): %{customdata[0]:.1f}% (total)',
                    'Item: %{customdata[2]}',
                    'Group: %{customdata[3]}',
                    'Participants: %{customdata[1]}',
                    'Sentence: %{customdata[4]}',
                    'Feature: %{customdata[5]}'
                ]) + '<extra></extra>',
                customdata=neutral_custom_data
            ))
            
            # Plot negative ratings (left side) - reversed order so -1 is closest to center
            for rating in reversed(negative_ratings):  # This makes -1, -2, -3, -4, -5 order
                x_values = []
                y_values = []
                custom_data = []
                
                for item in sorted_items:
                    for group in groups:
                        group_item_data = rating_data[
                            (rating_data['rating'] == rating) & 
                            (rating_data['group'] == group) & 
                            (rating_data['item'] == item)
                        ]
                        
                        if len(group_item_data) > 0:
                            percentage = group_item_data['percentage'].iloc[0]
                            participant_count = group_item_data['participant_count'].iloc[0] if 'participant_count' in group_item_data.columns else 0
                            sentence = group_item_data['sentence'].iloc[0] if 'sentence' in group_item_data.columns else "N/A"
                            feature = group_item_data['feature'].iloc[0] if 'feature' in group_item_data.columns else "N/A"
                        else:
                            percentage = 0
                            participant_count = 0
                            sentence = "N/A"
                            feature = "N/A"
                        
                        # Make negative ratings appear on the left (negative x)
                        x_values.append(-percentage)
                        y_values.append(item_positions[item][group])
                        custom_data.append([abs(percentage), participant_count, item, group, sentence, feature])
                
                fig.add_trace(go.Bar(
                    x=x_values,
                    y=y_values,
                    name=f'Rating {rating}',
                    marker_color=rating_colors[rating],
                    orientation='h',
                    legendrank=rating + 5,  # This controls legend order (convert -5 to 0, -4 to 1, etc.)
                    hovertemplate='<br>'.join([
                        f'Rating {rating}: %{{customdata[0]:.1f}}%',
                        'Item: %{customdata[2]}',
                        'Group: %{customdata[3]}',
                        'Participants: %{customdata[1]}',
                        'Sentence: %{customdata[4]}',
                        'Feature: %{customdata[5]}'
                    ]) + '<extra></extra>',
                    customdata=custom_data
                ))
            
            # Add right half of neutral
            fig.add_trace(go.Bar(
                x=neutral_x_right,
                y=y_values_neutral,
                name='Rating 0 (Neutral)',
                marker_color=rating_colors[neutral_rating],
                orientation='h',
                legendrank=5,  # This controls legend order (neutral rating = 0, so 0 + 5 = 5)
                hovertemplate='<br>'.join([
                    'Rating 0 (Neutral): %{customdata[0]:.1f}% (total)',
                    'Item: %{customdata[2]}',
                    'Group: %{customdata[3]}',
                    'Participants: %{customdata[1]}',
                    'Sentence: %{customdata[4]}',
                    'Feature: %{customdata[5]}'
                ]) + '<extra></extra>',
                customdata=neutral_custom_data
            ))
            
            # Plot positive ratings (right side)  
            for rating in positive_ratings:
                x_values = []
                y_values = []
                custom_data = []
                
                for item in sorted_items:
                    for group in groups:
                        group_item_data = rating_data[
                            (rating_data['rating'] == rating) & 
                            (rating_data['group'] == group) & 
                            (rating_data['item'] == item)
                        ]
                        
                        if len(group_item_data) > 0:
                            percentage = group_item_data['percentage'].iloc[0]
                            participant_count = group_item_data['participant_count'].iloc[0] if 'participant_count' in group_item_data.columns else 0
                            sentence = group_item_data['sentence'].iloc[0] if 'sentence' in group_item_data.columns else "N/A"
                            feature = group_item_data['feature'].iloc[0] if 'feature' in group_item_data.columns else "N/A"
                        else:
                            percentage = 0
                            participant_count = 0
                            sentence = "N/A"
                            feature = "N/A"
                        
                        x_values.append(percentage)
                        y_values.append(item_positions[item][group])
                        custom_data.append([percentage, participant_count, item, group, sentence, feature])
                
                fig.add_trace(go.Bar(
                    x=x_values,
                    y=y_values,
                    name=f'Rating {rating}',
                    marker_color=rating_colors[rating],
                    orientation='h',
                    legendrank=rating + 5,  # This controls legend order (convert 1 to 6, 2 to 7, etc.)
                    hovertemplate='<br>'.join([
                        f'Rating {rating}: %{{customdata[0]:.1f}}%',
                        'Item: %{customdata[2]}',
                        'Group: %{customdata[3]}',
                        'Participants: %{customdata[1]}',
                        'Sentence: %{customdata[4]}',
                        'Feature: %{customdata[5]}'
                    ]) + '<extra></extra>',
                    customdata=custom_data
                ))
        else:
            # For normal ratings: 0-2 on left, 3-5 on right
            left_ratings = [0, 1, 2]
            right_ratings = [3, 4, 5]
            
            # Define descriptive rating names
            rating_names = {
                0: 'No-one',
                1: 'Few', 
                2: 'Some',
                3: 'Many',
                4: 'Most',
                5: 'Everyone'
            }
            
            # Plot left side ratings (0-2) - reversed order so 2 is closest to center
            for rating in reversed(left_ratings):  # This makes 2, 1, 0 order
                x_values = []
                y_values = []
                custom_data = []
                
                for item in sorted_items:
                    for group in groups:
                        group_item_data = rating_data[
                            (rating_data['rating'] == rating) & 
                            (rating_data['group'] == group) & 
                            (rating_data['item'] == item)
                        ]
                        
                        if len(group_item_data) > 0:
                            percentage = group_item_data['percentage'].iloc[0]
                            participant_count = group_item_data['participant_count'].iloc[0] if 'participant_count' in group_item_data.columns else 0
                            sentence = group_item_data['sentence'].iloc[0] if 'sentence' in group_item_data.columns else "N/A"
                            feature = group_item_data['feature'].iloc[0] if 'feature' in group_item_data.columns else "N/A"
                        else:
                            percentage = 0
                            participant_count = 0
                            sentence = "N/A"
                            feature = "N/A"
                        
                        # Make left ratings appear on the left (negative x)
                        x_values.append(-percentage)
                        y_values.append(item_positions[item][group])
                        custom_data.append([abs(percentage), participant_count, item, group, sentence, feature])
                
                fig.add_trace(go.Bar(
                    x=x_values,
                    y=y_values,
                    name=rating_names[rating],
                    marker_color=rating_colors[rating],
                    orientation='h',
                    legendrank=rating,  # This controls legend order
                    hovertemplate='<br>'.join([
                        f'{rating_names[rating]}: %{{customdata[0]:.1f}}%',
                        'Item: %{customdata[2]}',
                        'Group: %{customdata[3]}',
                        'Participants: %{customdata[1]}',
                        'Sentence: %{customdata[4]}',
                        'Feature: %{customdata[5]}'
                    ]) + '<extra></extra>',
                    customdata=custom_data
                ))
            
            # Plot right side ratings (3-5)
            for rating in right_ratings:
                x_values = []
                y_values = []
                custom_data = []
                
                for item in sorted_items:
                    for group in groups:
                        group_item_data = rating_data[
                            (rating_data['rating'] == rating) & 
                            (rating_data['group'] == group) & 
                            (rating_data['item'] == item)
                        ]
                        
                        if len(group_item_data) > 0:
                            percentage = group_item_data['percentage'].iloc[0]
                            participant_count = group_item_data['participant_count'].iloc[0] if 'participant_count' in group_item_data.columns else 0
                            sentence = group_item_data['sentence'].iloc[0] if 'sentence' in group_item_data.columns else "N/A"
                            feature = group_item_data['feature'].iloc[0] if 'feature' in group_item_data.columns else "N/A"
                        else:
                            percentage = 0
                            participant_count = 0
                            sentence = "N/A"
                            feature = "N/A"
                        
                        x_values.append(percentage)
                        y_values.append(item_positions[item][group])
                        custom_data.append([percentage, participant_count, item, group, sentence, feature])
                
                fig.add_trace(go.Bar(
                    x=x_values,
                    y=y_values,
                    name=rating_names[rating],
                    marker_color=rating_colors[rating],
                    orientation='h',
                    legendrank=rating,  # This controls legend order
                    hovertemplate='<br>'.join([
                        f'{rating_names[rating]}: %{{customdata[0]:.1f}}%',
                        'Item: %{customdata[2]}',
                        'Group: %{customdata[3]}',
                        'Participants: %{customdata[1]}',
                        'Sentence: %{customdata[4]}',
                        'Feature: %{customdata[5]}'
                    ]) + '<extra></extra>',
                    customdata=custom_data
                ))
        
        # Add vertical line at x=0 for orientation
        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black", opacity=0.7)
        
        fig.update_layout(
            barmode='relative',  # Changed from 'stack' to 'relative' for diverging bars
            title='Diverging Rating Distribution by Group and Item',
            xaxis_title='← Lower Ratings | Higher Ratings →' if not pairs else '← Written Preference | Spoken Preference →',
            yaxis_title='Items by Group',
            template='simple_white',
            height=max(600, len(sorted_items) * len(groups) * 25),
            showlegend=True,
            legend=dict(
                traceorder='normal'  # This will order legend items by their order of creation
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=y_positions,
                ticktext=y_labels,
                tickfont=dict(size=10)
            ),
            xaxis=dict(
                range=[-100, 100],  # Set symmetrical range
                tickformat='.0f',
                side='bottom'
            ),
            xaxis2=dict(
                range=[-100, 100],  # Set symmetrical range
                tickformat='.0f',
                side='top',
                overlaying='x',
                showgrid=False
            )
        )
        
    else:
        # Create subplots for multiple modes (columns instead of rows for rotation)
        fig = make_subplots(
            rows=1, cols=len(modes),
            subplot_titles=[mode.title() for mode in sorted(modes)],
            shared_yaxes=True
        )
        
        groups = list(reversed(sort_groups_for_plot(rating_data['group'].unique(), groupby)))
        
        # Calculate spacing for each subplot
        item_positions = {}
        current_pos = 0
        
        for item in sorted_items:
            item_positions[item] = {}
            for i, group in enumerate(groups):
                item_positions[item][group] = current_pos + i * 0.8
            current_pos += len(groups) + 1
        
        # Create y-axis labels
        y_labels = []
        y_positions = []
        
        for item in sorted_items:
            for group in groups:
                y_labels.append(f"{item} - {group}")
                y_positions.append(item_positions[item][group])
        
        for col, mode in enumerate(sorted(modes), 1):
            mode_data = rating_data[rating_data['mode'] == mode]
            
            if pairs:
                # For pairs: negative ratings (left side), neutral split, positive ratings (right side)
                negative_ratings = [r for r in rating_range if r < 0]
                positive_ratings = [r for r in rating_range if r > 0]
                neutral_rating = 0
                
                # Plot negative ratings (left side) - reversed order so -1 is closest to center
                for rating in reversed(negative_ratings):  # This makes -1, -2, -3, -4, -5 order
                    x_values = []
                    y_values = []
                    
                    for item in sorted_items:
                        for group in groups:
                            group_item_data = mode_data[
                                (mode_data['rating'] == rating) & 
                                (mode_data['group'] == group) & 
                                (mode_data['item'] == item)
                            ]
                            
                            percentage = group_item_data['percentage'].iloc[0] if len(group_item_data) > 0 else 0
                            x_values.append(-percentage)
                            y_values.append(item_positions[item][group])
                    
                    fig.add_trace(go.Bar(
                        x=x_values,
                        y=y_values,
                        name=f'Rating {rating}' if col == 1 else None,
                        marker_color=rating_colors[rating],
                        orientation='h',
                        showlegend=(col == 1),
                        hovertemplate=f'Rating {rating}: %{{customdata:.1f}}%<br>' +
                                     f'{mode.title()}: %{{text}}<extra></extra>',
                        customdata=[abs(x) for x in x_values],
                        text=[f"{item} - {group}" for item in sorted_items for group in groups]
                    ), row=1, col=col)
                
                # Plot neutral rating (split in half)
                neutral_x_left = []
                neutral_x_right = []
                y_values_neutral = []
                
                for item in sorted_items:
                    for group in groups:
                        group_item_data = mode_data[
                            (mode_data['rating'] == neutral_rating) & 
                            (mode_data['group'] == group) & 
                            (mode_data['item'] == item)
                        ]
                        
                        percentage = group_item_data['percentage'].iloc[0] if len(group_item_data) > 0 else 0
                        half_percentage = percentage / 2
                        
                        neutral_x_left.append(-half_percentage)
                        neutral_x_right.append(half_percentage)
                        y_values_neutral.append(item_positions[item][group])
                
                # Add left half of neutral
                fig.add_trace(go.Bar(
                    x=neutral_x_left,
                    y=y_values_neutral,
                    name='Rating 0 (left)' if col == 1 else None,
                    marker_color=rating_colors[neutral_rating],
                    orientation='h',
                    showlegend=False,  # Don't show in legend
                    hovertemplate='Rating 0 (Neutral): %{customdata:.1f}% (total)<br>' +
                                 f'{mode.title()}: %{{text}}<extra></extra>',
                    customdata=[abs(x) * 2 for x in neutral_x_left],
                    text=[f"{item} - {group}" for item in sorted_items for group in groups]
                ), row=1, col=col)
                
                # Add right half of neutral
                fig.add_trace(go.Bar(
                    x=neutral_x_right,
                    y=y_values_neutral,
                    name='Rating 0 (Neutral)' if col == 1 else None,
                    marker_color=rating_colors[neutral_rating],
                    orientation='h',
                    showlegend=(col == 1),
                    hovertemplate='Rating 0 (Neutral): %{customdata:.1f}% (total)<br>' +
                                 f'{mode.title()}: %{{text}}<extra></extra>',
                    customdata=[x * 2 for x in neutral_x_right],
                    text=[f"{item} - {group}" for item in sorted_items for group in groups]
                ), row=1, col=col)
                
                # Plot positive ratings (right side)
                for rating in positive_ratings:
                    x_values = []
                    y_values = []
                    
                    for item in sorted_items:
                        for group in groups:
                            group_item_data = mode_data[
                                (mode_data['rating'] == rating) & 
                                (mode_data['group'] == group) & 
                                (mode_data['item'] == item)
                            ]
                            
                            percentage = group_item_data['percentage'].iloc[0] if len(group_item_data) > 0 else 0
                            x_values.append(percentage)
                            y_values.append(item_positions[item][group])
                    
                    fig.add_trace(go.Bar(
                        x=x_values,
                        y=y_values,
                        name=f'Rating {rating}' if col == 1 else None,
                        marker_color=rating_colors[rating],
                        orientation='h',
                        showlegend=(col == 1),
                        hovertemplate=f'Rating {rating}: %{{x:.1f}}%<br>' +
                                     f'{mode.title()}: %{{text}}<extra></extra>',
                        text=[f"{item} - {group}" for item in sorted_items for group in groups]
                    ), row=1, col=col)
            else:
                # For normal ratings: 0-2 on left, 3-5 on right
                left_ratings = [0, 1, 2]
                right_ratings = [3, 4, 5]
                
                # Plot left side ratings (0-2) - reversed order so 2 is closest to center
                for rating in reversed(left_ratings):  # This makes 2, 1, 0 order
                    x_values = []
                    y_values = []
                    
                    for item in sorted_items:
                        for group in groups:
                            group_item_data = mode_data[
                                (mode_data['rating'] == rating) & 
                                (mode_data['group'] == group) & 
                                (mode_data['item'] == item)
                            ]
                            
                            percentage = group_item_data['percentage'].iloc[0] if len(group_item_data) > 0 else 0
                            x_values.append(-percentage)
                            y_values.append(item_positions[item][group])
                    
                    fig.add_trace(go.Bar(
                        x=x_values,
                        y=y_values,
                        name=f'Rating {rating}' if col == 1 else None,
                        marker_color=rating_colors[rating],
                        orientation='h',
                        showlegend=(col == 1),
                        hovertemplate=f'Rating {rating}: %{{customdata:.1f}}%<br>' +
                                     f'{mode.title()}: %{{text}}<extra></extra>',
                        customdata=[abs(x) for x in x_values],
                        text=[f"{item} - {group}" for item in sorted_items for group in groups]
                    ), row=1, col=col)
                
                # Plot right side ratings (3-5)
                for rating in right_ratings:
                    x_values = []
                    y_values = []
                    
                    for item in sorted_items:
                        for group in groups:
                            group_item_data = mode_data[
                                (mode_data['rating'] == rating) & 
                                (mode_data['group'] == group) & 
                                (mode_data['item'] == item)
                            ]
                            
                            percentage = group_item_data['percentage'].iloc[0] if len(group_item_data) > 0 else 0
                            x_values.append(percentage)
                            y_values.append(item_positions[item][group])
                    
                    fig.add_trace(go.Bar(
                        x=x_values,
                        y=y_values,
                        name=f'Rating {rating}' if col == 1 else None,
                        marker_color=rating_colors[rating],
                        orientation='h',
                        showlegend=(col == 1),
                        hovertemplate=f'Rating {rating}: %{{x:.1f}}%<br>' +
                                     f'{mode.title()}: %{{text}}<extra></extra>',
                        text=[f"{item} - {group}" for item in sorted_items for group in groups]
                    ), row=1, col=col)
            
            # Add vertical line at x=0 for each subplot
            fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black", opacity=0.7, row=1, col=col)
        
        fig.update_layout(
            barmode='relative',
            title='Diverging Rating Distribution by Mode, Group and Item',
            template='simple_white',
            height=max(800, len(sorted_items) * len(groups) * 30)
        )
        
        # Update x-axes for all subplots
        for col in range(1, len(modes) + 1):
            fig.update_xaxes(
                title_text='← Lower Ratings | Higher Ratings →' if not pairs else '← Written | Spoken →', 
                range=[-100, 100],
                tickformat='.0f',
                row=1, col=col
            )
        
        # Update Y-axis for first column only
        fig.update_yaxes(
            title_text='Items by Group',
            tickmode='array',
            tickvals=y_positions,
            ticktext=y_labels,
            tickfont=dict(size=10),
            row=1, col=1
        )
    
    # Add center reference lines for each individual bar
    if len(modes) == 1:
        # Add a thick black line at the center of each item-group bar
        for item in sorted_items:
            for group in groups:
                # Get data to check if this combination exists
                item_group_data = melted_df[
                    (melted_df['item'] == item) & 
                    (melted_df['group'] == group)
                ]
                
                if len(item_group_data) > 0:
                    y_pos = item_positions[item][group]
                    
                    # Calculate the total extent of this bar (left + right percentages)
                    total_left = 0
                    total_right = 0
                    
                    if pairs:
                        # For pairs, calculate total extent differently
                        for rating in rating_data['rating'].unique():
                            rating_subset = rating_data[
                                (rating_data['rating'] == rating) & 
                                (rating_data['group'] == group) & 
                                (rating_data['item'] == item)
                            ]
                            if len(rating_subset) > 0:
                                percentage = rating_subset['percentage'].iloc[0]
                                if rating < 0:
                                    total_left += percentage
                                else:
                                    total_right += percentage
                    else:
                        # For normal ratings: left (0-2) and right (3-5)
                        left_ratings = [0, 1, 2]
                        right_ratings = [3, 4, 5]
                        
                        for rating in left_ratings:
                            rating_subset = rating_data[
                                (rating_data['rating'] == rating) & 
                                (rating_data['group'] == group) & 
                                (rating_data['item'] == item)
                            ]
                            if len(rating_subset) > 0:
                                total_left += rating_subset['percentage'].iloc[0]
                        
                        for rating in right_ratings:
                            rating_subset = rating_data[
                                (rating_data['rating'] == rating) & 
                                (rating_data['group'] == group) & 
                                (rating_data['item'] == item)
                            ]
                            if len(rating_subset) > 0:
                                total_right += rating_subset['percentage'].iloc[0]
                    
                    # Calculate the center position of this specific bar
                    # The bar extends from -total_left to +total_right, so center is at:
                    bar_center = (total_right - total_left) / 2
                    
                    # Add white circle marker at the center of this specific bar
                    fig.add_trace(go.Scatter(
                        x=[bar_center],
                        y=[y_pos],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=6,
                            color='white',
                            line=dict(
                                color='black',
                                width=1
                            )
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    return fig

def create_normal_plot_rotated(df, items, modes, groupby, variety_color_map, pairs, meandf):
    """Create normal plot with 90-degree rotation (items on Y-axis)"""
    
    Rating_map={'0':'No-one','1':'Few','2':'Some','3':'Many','4':'Most','5':'Everyone'}
    
    if len(modes) == 1 and not pairs:
        # Single mode plot - rotated
        plot_groups = sorted(df['group'].unique())
        fig = go.Figure()
        
        for plot_group in plot_groups:
            tempDF = df[df['group'] == plot_group]
            
            # Get color for this group if we're grouping by variety
            color = None
            if variety_color_map and plot_group in variety_color_map:
                color = variety_color_map[plot_group]
            
            # Create rotated scatter plot (Y = items, X = mean)
            fig.add_trace(
                go.Scatter(
                    y=tempDF['item'],  # Items on Y-axis
                    x=tempDF['mean'],  # Mean on X-axis
                    name=str(plot_group),
                    mode="markers",
                    marker=dict(color=color) if color else None,
                    error_x=dict(  # Error bars now horizontal
                        type='data',
                        symmetric=False,
                        array=tempDF['upper_ci'],
                        arrayminus=tempDF['lower_ci']
                    ),
                    customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                    hovertemplate='<br>'.join([
                        'Item: %{customdata[0]}',
                        'Group: %{customdata[1]}',
                        'Avg rating: %{customdata[2]:.2f}',
                        'Number of participants: %{customdata[3]}',
                        'Sentence: %{customdata[4]}',
                        'Item name: %{customdata[5]}',
                        'Item group: %{customdata[6]}',
                        'Ewave feature: %{customdata[7]}',
                        'Twin item: %{customdata[8]}'
                    ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']) else None,
                    hoverinfo='text'
                )
            )
        
        # Update layout for rotated plot
        fig.update_layout(
            title='Mean difference (95% CI)',
            xaxis_title='Mean ratings',
            yaxis_title='Grammatical items',
            template='simple_white',
            height=max(400, len(meandf) * 15),  # Adjust height based on number of items
            xaxis2=dict(
                fixedrange=True,
                range=[0, 5.1],
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=[Rating_map[str(i)] for i in range(6)] if not pairs else None,
                side='top',
                overlaying='x',
                showgrid=False
            )
        )
        
        # Update axes
        fig.update_xaxes(
            fixedrange=True,
            range=[0, 5.1],
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=[Rating_map[str(i)] for i in range(6)] if not pairs else None,
            mirror='ticks',  # Mirror ticks to top
            ticks='outside'
        )
        
        # Create custom y-axis labels with pair information and mode
        item_order = list(reversed(meandf['item'].tolist()))  # Reverse for top-to-bottom order
        custom_labels = []
        
        for item in item_order:
            # Get item data to check for twin
            item_data = df[df['item'] == item]
            if not item_data.empty:
                twin_item = item_data['also_in_question'].iloc[0] if 'also_in_question' in item_data.columns else None
                mode = item_data['section'].iloc[0] if 'section' in item_data.columns else ""
                
                # Format mode with proper capitalization
                mode_formatted = mode.capitalize() if mode else ""
                
                if pd.notna(twin_item) and twin_item != '':
                    # Item has a twin - check mode to determine order
                    if mode.lower() == 'spoken':
                        # For spoken items: spoken-written format
                        custom_labels.append(f"{item}-{twin_item}: {mode_formatted}")
                    else:
                        # For written items: spoken-written format (twin first)
                        custom_labels.append(f"{twin_item}-{item}: {mode_formatted}")
                else:
                    # No twin - format as "G26: Written" 
                    custom_labels.append(f"{item}: {mode_formatted}")
            else:
                # Fallback if no data found
                custom_labels.append(item)
        
        # Set custom y-axis labels
        fig.update_yaxes(
            tickmode='array',
            tickvals=item_order,
            ticktext=custom_labels
        )
        
    elif len(modes) == 1 and pairs:
        # Single mode pairs plot - rotated
        plot_groups = sorted(df['group'].unique())
        fig = go.Figure()
        
        for plot_group in plot_groups:
            tempDF = df[df['group'] == plot_group]
            
            # Get color for this group if we're grouping by variety
            color = None
            if variety_color_map and plot_group in variety_color_map:
                color = variety_color_map[plot_group]
            
            # Create rotated scatter plot (Y = items, X = mean)
            fig.add_trace(
                go.Scatter(
                    y=tempDF['item'],  # Items on Y-axis
                    x=tempDF['mean'],  # Mean on X-axis
                    name=str(plot_group),
                    mode="markers",
                    marker=dict(color=color) if color else None,
                    error_x=dict(  # Error bars now horizontal
                        type='data',
                        symmetric=False,
                        array=tempDF['upper_ci'],
                        arrayminus=tempDF['lower_ci']
                    ),
                    customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']] if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']) else None,
                    hovertemplate='<br>'.join([
                        'Item: %{customdata[0]}',
                        'Group: %{customdata[1]}',
                        'Avg difference: %{customdata[2]:.2f}',
                        'Number of participants: %{customdata[3]}',
                        'Sentence: %{customdata[4]}',
                        'Item name: %{customdata[5]}',
                        'Item group: %{customdata[6]}',
                        'Ewave feature: %{customdata[7]}'
                    ]) if all(col in tempDF.columns for col in ['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave']) else None,
                    hoverinfo='text'
                )
            )
        
        # Update layout for rotated pairs plot
        fig.update_layout(
            title='Mean difference (95% CI)',
            xaxis_title='Mean difference (spoken - written)',
            yaxis_title='Grammatical items',
            template='simple_white',
            height=max(400, len(meandf) * 15),  # Adjust height based on number of items
            xaxis2=dict(
                fixedrange=True,
                range=[-5.5, 5.5],
                tickvals=[-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5],
                ticktext=['Written','-4','-3','-2','-1','Neutral','1','2','3','4','Spoken'],
                side='top',
                overlaying='x',
                showgrid=False
            )
        )
        
        # Update axes for pairs data (-5 to 5 range)
        fig.update_xaxes(
            fixedrange=True,
            range=[-5.5, 5.5],
            tickvals=[-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5],
            ticktext=['Written','-4','-3','-2','-1','Neutral','1','2','3','4','Spoken'],
            mirror='ticks',  # Mirror ticks to top
            ticks='outside'
        )
        
        # Add vertical line at x=0 for pairs
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray", opacity=0.7)
        
        # Set item order on Y-axis
        fig.update_yaxes(
            categoryorder='array',
            categoryarray=list(reversed(meandf['item'].tolist()))  # Reverse for top-to-bottom order
        )
        
    elif len(modes) > 1:
        # Multiple modes - group twin items together
        plot_groups = sorted(df['group'].unique())
        
        # Group items by their twin pairs from also_in_question column
        # Use the meandf order which already accounts for sorting
        item_pairs = {}
        standalone_items = set()
        ordered_items = meandf['item'].tolist()  # Use sorted order from meandf
        
        for item in ordered_items:
            item_data = df[df['item'] == item]
            if not item_data.empty:
                twin_item = item_data['also_in_question'].iloc[0] if 'also_in_question' in item_data.columns else None
                
                if pd.notna(twin_item) and twin_item != '' and twin_item in df['item'].unique():
                    # Create a sorted pair key to avoid duplicates
                    pair_key = tuple(sorted([item, twin_item]))
                    if pair_key not in item_pairs:
                        item_pairs[pair_key] = True
                else:
                    standalone_items.add(item)
        
        # Create y-position mapping for grouped display following sort order
        # Twin items are displayed separately but grouped together
        y_positions = {}
        y_labels = []
        current_pos = 0
        processed_items = set()
        
        # Process items in the order they appear in meandf (which reflects sorting)
        for item in ordered_items:
            if item in processed_items:
                continue
                
            # Check if this item has a twin
            twin_item = None
            item_data = df[df['item'] == item]
            if not item_data.empty:
                potential_twin = item_data['also_in_question'].iloc[0] if 'also_in_question' in item_data.columns else None
                if pd.notna(potential_twin) and potential_twin != '' and potential_twin in ordered_items:
                    twin_item = potential_twin
            
            if twin_item:
                # This is a twin pair - display as separate entries but grouped together
                # Determine which is spoken and which is written by checking modes in df
                item_modes = df[df['item'] == item]['mode'].unique()
                twin_modes = df[df['item'] == twin_item]['mode'].unique()
                
                # Order: spoken first, then written
                if 'spoken' in item_modes:
                    spoken_item, written_item = item, twin_item
                elif 'spoken' in twin_modes:
                    spoken_item, written_item = twin_item, item
                else:
                    # If neither is clearly spoken, use alphabetical order
                    spoken_item, written_item = sorted([item, twin_item])
                
                # Add spoken item first
                y_labels.append(f"{spoken_item} - spoken")
                y_positions[spoken_item] = current_pos
                current_pos += 1
                
                # Add written item second (grouped closely)
                y_labels.append(f"{written_item} - written")
                y_positions[written_item] = current_pos
                current_pos += 1
                
                processed_items.add(item)
                processed_items.add(twin_item)
            else:
                # Standalone item - determine its mode for labeling
                item_modes = df[df['item'] == item]['mode'].unique()
                mode_suffix = f" - {item_modes[0]}" if len(item_modes) == 1 else ""
                y_labels.append(f"{item}{mode_suffix}")
                y_positions[item] = current_pos
                processed_items.add(item)
                current_pos += 1
        
        # Create single plot instead of subplots
        fig = go.Figure()
        
        # Define mode colors
        mode_colors = {'spoken': '#1f77b4', 'written': '#ff7f0e'}
        
        for mode in sorted(modes):
            for plot_group in plot_groups:
                tempDF = df[(df['group'] == plot_group) & (df['mode'] == mode)]
                
                if tempDF.empty:
                    continue
                
                # Get color for this group if we're grouping by variety
                base_color = None
                if variety_color_map and plot_group in variety_color_map:
                    base_color = variety_color_map[plot_group]
                else:
                    base_color = mode_colors.get(mode.lower(), '#1f77b4')
                
                # Map items to their y-positions
                y_vals = [y_positions[item] for item in tempDF['item']]
                
                fig.add_trace(
                    go.Scatter(
                        y=y_vals,  # Use mapped positions
                        x=tempDF['mean'],  # Mean on X-axis
                        name=f"{plot_group} ({mode})",
                        mode="markers",
                        marker=dict(
                            color=base_color,
                            symbol='circle' if mode.lower() == 'spoken' else 'diamond',
                            size=8
                        ),
                        error_x=dict(  # Error bars now horizontal
                            type='data',
                            symmetric=False,
                            array=tempDF['upper_ci'],
                            arrayminus=tempDF['lower_ci']
                        ),
                        customdata=tempDF[['item','group','mean','count','sentence','variant_detail','group_finegrained','feature_ewave','also_in_question']],
                        hovertemplate='<br>'.join([
                            'Item: %{customdata[0]}',
                            'Group: %{customdata[1]}',
                            'Mode: ' + mode,
                            'Avg rating: %{customdata[2]:.2f}',
                            'Number of participants: %{customdata[3]}',
                            'Sentence: %{customdata[4]}',
                            'Item name: %{customdata[5]}',
                            'Item group: %{customdata[6]}',
                            'Ewave feature: %{customdata[7]}',
                            'Twin item: %{customdata[8]}'
                        ]), 
                        hoverinfo='text'
                    )
                )
        
        fig.update_layout(
            title='Mean (95% CI) by mode (grouped twin items)',
            xaxis_title='Mean ratings',
            yaxis_title='Grammatical items',
            template='simple_white',
            height=max(600, len(y_labels) * 25)
        )
        
        # Update x-axis
        fig.update_xaxes(
            fixedrange=True,
            range=[0, 5.1],
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=[Rating_map[str(i)] for i in range(6)] if not pairs else None,
            mirror='ticks',  # Add ticks on top
            ticks='outside'
        )
        
        # Set custom y-axis with grouped labels
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
            title_text='Grammatical items'
        )
    
    return fig


def create_informant_mean_boxplot(df_orig, items, modes, groupby, variety_color_map, pairs, meandf, sortby="mean", use_imputed=True, informants=None):
    """Create boxplots showing distribution of individual participant means across all items for each mode"""
    
    # Get raw data with individual ratings (not aggregated)
    raw_df = retrieve_data.getGrammarData(imputed=use_imputed, items=items, pairs=pairs, participants=informants)
    
    # Get metadata and merge it to get section information
    if not pairs:
        meta = retrieve_data.getGrammarMeta()
    else:
        meta = retrieve_data.getGrammarMeta(type='item_pairs')
    
    # Apply same grouping logic as main function
    if groupby == "variety":
        raw_df['group'] = raw_df['MainVariety']
    elif groupby == "vtype" or groupby == "vtype_balanced":
        # Use centralized variety mapping
        variety_mapping = get_variety_mapping()
        raw_df['group'] = raw_df['MainVariety'].map(variety_mapping).fillna("Other")
        raw_df = raw_df[raw_df['group'] != "Other"]
    elif groupby == "gender":
        raw_df['group'] = raw_df['Gender_normalized'] if 'Gender_normalized' in raw_df.columns else raw_df['Gender']
    
    # Melt the data to get individual responses
    info_cols = ['InformantID', 'group']
    
    melted_df = raw_df.melt(id_vars=info_cols, value_vars=items, var_name='item', value_name='rating')
    melted_df['rating'] = pd.to_numeric(melted_df['rating'], errors='coerce')
    melted_df = melted_df.dropna(subset=['rating'])
    
    # Merge with metadata to get section information
    if not pairs:
        melted_df = melted_df.merge(meta, left_on='item', right_on='question_code', how='left')
        # After merge, the original 'item' column might be renamed or duplicated
        # Ensure we have a clean 'item' column for counting
        if 'item_x' in melted_df.columns:
            melted_df['item'] = melted_df['item_x']
        elif 'question_code' in melted_df.columns and 'item' not in melted_df.columns:
            melted_df['item'] = melted_df['question_code']
    else:
        melted_df = melted_df.merge(meta, left_on='item', right_on='item_pair', how='left')
        # After merge, the original 'item' column might be renamed or duplicated
        if 'item_x' in melted_df.columns:
            melted_df['item'] = melted_df['item_x']
        elif 'item_pair' in melted_df.columns and 'item' not in melted_df.columns:
            melted_df['item'] = melted_df['item_pair']
    
    # Add mode column from section
    if 'section' in melted_df.columns:
        melted_df['mode'] = melted_df['section'].str.lower()
    else:
        melted_df['mode'] = 'single'
    
    # Get the actual modes from the data
    actual_modes = melted_df['mode'].unique()
    
    # Calculate individual participant means across ALL items for each mode (not per item)
    if len(actual_modes) > 1:
        # For multi-mode: calculate means across all items for each participant-group-mode combination
        participant_means = melted_df.groupby(['InformantID', 'group', 'mode'])['rating'].mean().reset_index()
        participant_means.rename(columns={'rating': 'participant_mean'}, inplace=True)
    else:
        # For single mode: calculate means across all items for each participant-group combination
        participant_means = melted_df.groupby(['InformantID', 'group'])['rating'].mean().reset_index()
        participant_means.rename(columns={'rating': 'participant_mean'}, inplace=True)
        participant_means['mode'] = actual_modes[0] if len(actual_modes) > 0 else 'single'
    
    # Create the plot - always handle both single and multi-mode in a unified way
    fig = go.Figure()
    plot_groups = sorted(participant_means['group'].unique())
    
    # Calculate hover info data - number of items and informants
    # Get unique items per mode for counting - ensure we have the item column
    items_per_mode = melted_df.groupby('mode')['item'].nunique().to_dict()
    informants_per_group_mode = melted_df.groupby(['group', 'mode'])['InformantID'].nunique().reset_index()
    informants_per_group_mode.columns = ['group', 'mode', 'n_informants']
    
    if len(actual_modes) == 1:
        # Single mode - create one boxplot per group (horizontal orientation)
        for i, plot_group in enumerate(plot_groups):
            group_data = participant_means[participant_means['group'] == plot_group]['participant_mean']
            
            # Get hover info for this group
            mode = actual_modes[0]
            n_items = items_per_mode.get(mode, 0)
            n_informants = informants_per_group_mode[
                (informants_per_group_mode['group'] == plot_group) & 
                (informants_per_group_mode['mode'] == mode)
            ]['n_informants'].iloc[0] if len(informants_per_group_mode[
                (informants_per_group_mode['group'] == plot_group) & 
                (informants_per_group_mode['mode'] == mode)
            ]) > 0 else 0
            
            # Get color for this group
            color = None
            if variety_color_map and plot_group in variety_color_map:
                color = variety_color_map[plot_group]
            
            if not group_data.empty:
                fig.add_trace(
                    go.Box(
                        x=group_data,  # Horizontal orientation: data on x-axis
                        y=[plot_group] * len(group_data),  # Group names on y-axis
                        name=f"{plot_group}",
                        marker_color=color if color else None,
                        boxpoints='outliers',
                        orientation='h',  # Horizontal orientation
                        hovertemplate='<b>%{y}</b><br>' +
                                    'Informants: %{customdata[0]}<br>' +
                                    'Items: %{customdata[1]}<br>' +
                                    'Mean rating: %{x:.2f}<extra></extra>',
                        customdata=[[n_informants, n_items]] * len(group_data)
                    )
                )
        
        fig.update_layout(
            title='Informant means across all selected items',
            xaxis_title='Individual Participant Mean Ratings',
            yaxis_title='Groups',
            template='simple_white',
            height=max(400, len(plot_groups) * 60),
            showlegend=False
        )
        
        # Update x-axis with rating labels
        if pairs:
            # For pairs, use -5 to 5 scale (spoken - written difference)
            fig.update_xaxes(
                tickvals=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                ticktext=['Written', '-4', '-3', '-2', '-1', 'Neutral', '1', '2', '3', '4', 'Spoken'],
                range=[-5.5, 5.5]
            )
        else:
            # For regular items, use 0-5 scale
            Rating_map = {'0':'No-one','1':'Few','2':'Some','3':'Many','4':'Most','5':'Everyone'}
            fig.update_xaxes(
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=[Rating_map[str(i)] for i in range(6)],
                range=[0, 5.1]
            )
        
    else:
        # Multi-mode plot - create two boxplots per group (spoken and written)
        mode_colors = {'spoken': '#1f77b4', 'written': '#ff7f0e'}
        
        # Create y-axis positions for grouped boxplots
        y_positions = {}
        y_labels = []
        pos = 0
        
        # Create positions: for each group, place spoken above written
        for group in plot_groups:
            for mode in sorted(actual_modes, reverse=True):  # spoken first, then written
                y_positions[(group, mode)] = pos
                y_labels.append(f"{group} ({mode})")
                pos += 1
        
        for group in plot_groups:
            for mode in sorted(actual_modes, reverse=True):  # spoken first, then written
                mode_data = participant_means[
                    (participant_means['group'] == group) & 
                    (participant_means['mode'] == mode)
                ]['participant_mean']
                
                # Get hover info for this group-mode combination
                n_items = items_per_mode.get(mode, 0)
                n_informants = informants_per_group_mode[
                    (informants_per_group_mode['group'] == group) & 
                    (informants_per_group_mode['mode'] == mode)
                ]['n_informants'].iloc[0] if len(informants_per_group_mode[
                    (informants_per_group_mode['group'] == group) & 
                    (informants_per_group_mode['mode'] == mode)
                ]) > 0 else 0
                
                # Get color - prefer variety color, fallback to mode color
                color = None
                if variety_color_map and group in variety_color_map:
                    color = variety_color_map[group]
                else:
                    color = mode_colors.get(mode, '#1f77b4')
                
                if not mode_data.empty:
                    fig.add_trace(
                        go.Box(
                            x=mode_data,  # Horizontal orientation: data on x-axis
                            y=[y_positions[(group, mode)]] * len(mode_data),  # Position on y-axis
                            name=f"{group} ({mode})",
                            marker_color=color,
                            boxpoints='outliers',
                            orientation='h',  # Horizontal orientation
                            showlegend=True,
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Informants: %{customdata[0]}<br>' +
                                        'Items: %{customdata[1]}<br>' +
                                        'Mean rating: %{x:.2f}<extra></extra>',
                            customdata=[[n_informants, n_items]] * len(mode_data)
                        )
                    )
        
        fig.update_layout(
            title='Informant means across all selected items by mode',
            xaxis_title='Individual Participant Mean Ratings',
            yaxis_title='Groups and Modes',
            template='simple_white',
            height=max(600, len(y_labels) * 40),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(y_labels))),
                ticktext=y_labels
            )
        )
        
        # Update x-axis with rating labels
        if pairs:
            # For pairs, use -5 to 5 scale (spoken - written difference)
            fig.update_xaxes(
                tickvals=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                ticktext=['Written', '-4', '-3', '-2', '-1', 'Neutral', '1', '2', '3', '4', 'Spoken'],
                range=[-5.5, 5.5]
            )
        else:
            # For regular items, use 0-5 scale
            Rating_map = {'0':'No-one','1':'Few','2':'Some','3':'Many','4':'Most','5':'Everyone'}
            fig.update_xaxes(
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=[Rating_map[str(i)] for i in range(6)],
                range=[0, 5.1]
            )
    
    return fig

def create_correlation_matrix_plot(df, items, informants, pairs=False, use_imputed=False):
    """Create a correlation matrix heatmap for grammar items"""
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    
    # Get the data in wide format (participants as rows, items as columns)
    df_wide = retrieve_data.getGrammarData(imputed=use_imputed, items=items, participants=informants, pairs=pairs)
    
    if df_wide.empty:
        return getEmptyPlot("No data available for correlation matrix")
    
    # Select only the grammar item columns for correlation
    item_columns = [col for col in df_wide.columns if col in items]
    
    if len(item_columns) < 2:
        return getEmptyPlot("At least 2 items are required for correlation matrix")
    
    # Get meta data for sorting and hover information
    if not pairs:
        meta = retrieve_data.getGrammarMeta()
    else:
        meta = retrieve_data.getGrammarMeta(type='item_pairs')
    
    # Create a mapping from item to variant_detail and sentence for hover info
    item_info = {}
    for item in item_columns:
        meta_row = meta[meta['question_code'] == item]
        if not meta_row.empty:
            variant_detail = meta_row['variant_detail'].iloc[0] if 'variant_detail' in meta.columns else ""
            sentence = meta_row['item'].iloc[0] if 'item' in meta.columns else ""
            item_info[item] = {
                'variant_detail': variant_detail if pd.notna(variant_detail) else "",
                'sentence': sentence if pd.notna(sentence) else ""
            }
        else:
            item_info[item] = {'variant_detail': "", 'sentence': ""}
    
    # Sort items by variant_detail
    sorted_items = sorted(item_columns, key=lambda x: (item_info[x]['variant_detail'], x))
    
    # Calculate correlation matrix with sorted items
    correlation_data = df_wide[sorted_items].corr()
    
    # Create hover text with item information
    hover_text = []
    for i, y_item in enumerate(sorted_items):
        hover_row = []
        for j, x_item in enumerate(sorted_items):
            corr_value = correlation_data.iloc[i, j]
            y_info = item_info[y_item]
            x_info = item_info[x_item]
            
            hover_text_cell = (
                f"<b>X-axis:</b> {x_item}<br>"
                f"<b>X-item:</b> {x_info['variant_detail']}<br>"
                f"<b>X-sentence:</b> {x_info['sentence']}<br>"
                f"<b>Y-axis:</b> {y_item}<br>"
                f"<b>Y-item:</b> {y_info['variant_detail']}<br>"
                f"<b>Y-sentence:</b> {y_info['sentence']}<br>"
                f"<b>Correlation:</b> {corr_value:.3f}"
            )
            hover_row.append(hover_text_cell)
        hover_text.append(hover_row)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=sorted_items,
        y=sorted_items,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(correlation_data.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text
    ))
    
    fig.update_layout(
        title="Correlation Matrix of Grammar Items (sorted by variant detail)",
        xaxis_title="Grammar Items",
        yaxis_title="Grammar Items",
        height=max(600, len(sorted_items) * 25),
        width=max(600, len(sorted_items) * 25),
        template="simple_white"
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig