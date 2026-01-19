import dash_mantine_components as dmc
from dash import register_page
import pages.data.retrieve_data as retrieve_data
from pages.data.grammarFunctions import *
from pages.data.grammarFunctions import _participants_hash
from time import sleep

import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, dash_table, html,ctx, callback, Output, Input, State, clientside_callback, no_update
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
import plotly.figure_factory as ff
# Add Leiden clustering imports
import networkx as nx
import igraph as ig
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
dmc.add_figure_templates()

register_page(__name__, path="/grammar", name="Grammar Sets")

# load the CSS file from the CDN
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"


#UMAP_Grammar_initialPlot = go.Figure()
emptyFig = go.Figure()
emptyFig.update_layout(template="simple_white")
persistence_type="memory"
persist_UI=True

# all symbols with -open; used for grouping data points in the UMAP plot
symbols = [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
colorMap = px.colors.qualitative.Dark24

# Lazy data loading with caching
from functools import lru_cache
import diskcache as dc
import hashlib
import os

# Create persistent cache for expensive operations
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
    # Fallback to system temp directory
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), 'dash_cache', 'plot_cache')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[INFO] Using fallback cache directory: {cache_dir}")

plot_cache = dc.Cache(cache_dir)

def create_plot_cache_key(participants, items, n_neighbours, min_dist, distance_metric, standardize, densemap, pairs, regional_mapping=False):
    """Create a unique cache key for plot parameters"""
    key_data = {
        'participants': sorted(participants) if participants else 'all',
        'items': sorted(items) if items else 'all', 
        'n_neighbours': n_neighbours,
        'min_dist': min_dist,
        'distance_metric': distance_metric,
        'standardize': standardize,
        'densemap': densemap,
        'pairs': pairs,
        'regional_mapping': regional_mapping
    }
    key_string = str(key_data)
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cached_umap_plot(participants, items, n_neighbours, min_dist, distance_metric, standardize, densemap, pairs, informants=None, regional_mapping=False):
    """Get UMAP plot from cache or compute if not exists"""
    cache_key = f"umap_{create_plot_cache_key(participants, items, n_neighbours, min_dist, distance_metric, standardize, densemap, pairs, regional_mapping)}"
    
    cached_plot = plot_cache.get(cache_key)
    if cached_plot is not None:
        return cached_plot
    
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
        grammarData = retrieve_data.getGrammarData(imputed=True, participants=participants, columns=items, regional_mapping=regional_mapping)
        grammarCols = GrammarItemsCols
    else:
        grammarData = retrieve_data.getGrammarData(imputed=True, participants=participants, columns=items, pairs=True, regional_mapping=regional_mapping)
        grammarCols = GrammarItemsColsPairs
        
    plot = getUMAPplot(
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
        pairs=pairs,
        regional_mapping=regional_mapping
    )
    
    # Cache the result for 24 hours
    #plot_cache.set(cache_key, plot, expire=86400)
    plot_cache.set(cache_key, plot)
    return plot

@lru_cache(maxsize=2)  # Increased to cache both mapped and unmapped versions
def get_grammar_data_cached(regional_mapping=False):
    return retrieve_data.getGrammarData(imputed=True, regional_mapping=regional_mapping)

@lru_cache(maxsize=2)
def get_grammar_data_pairs_cached(regional_mapping=False):
    return retrieve_data.getGrammarData(imputed=True, items=retrieve_data.getGrammarItemsCols("item_pairs"), pairs=True, regional_mapping=regional_mapping)

@lru_cache(maxsize=2)
def get_informants_cached(regional_mapping=False):
    return retrieve_data.getInformantDataGrammar(imputed=True, regional_mapping=regional_mapping)

@lru_cache(maxsize=2)
def get_grammar_data_raw_cached(regional_mapping=False):
    return retrieve_data.getGrammarData(imputed=False, regional_mapping=regional_mapping)

@lru_cache(maxsize=2)
def get_grammar_data_pairs_raw_cached(regional_mapping=False):
    return retrieve_data.getGrammarData(imputed=False, items=retrieve_data.getGrammarItemsCols("item_pairs"), pairs=True, regional_mapping=regional_mapping)

@lru_cache(maxsize=1)
def get_grammar_meta_cached():
    return retrieve_data.getGrammarMeta()

@lru_cache(maxsize=1)
def get_grammar_meta_pairs_cached():
    return retrieve_data.getGrammarMeta("item_pairs")

@lru_cache(maxsize=1)
def get_grammar_items_cols_cached():
    return retrieve_data.getGrammarItemsCols()

@lru_cache(maxsize=1)
def get_grammar_items_cols_pairs_cached():
    return retrieve_data.getGrammarItemsCols("item_pairs")

def get_cached_rf_plot(data, importance_ratings, value_range, pairs, participants=None, split_by_variety=False):
    """Get RF plot from cache or compute if not exists"""
    # Create cache key from parameters
    key_data = {
        'data_shape': data.shape if hasattr(data, 'shape') else len(data),
        'importance_ratings': importance_ratings,
        'value_range': value_range,
        'pairs': pairs,
        'participants': sorted(participants) if participants is not None else None,
        'split_by_variety': split_by_variety,
        'data_hash': hashlib.md5(str(data.values.tolist() if hasattr(data, 'values') else str(data)).encode()).hexdigest()[:8]
    }
    cache_key = f"rf_{hashlib.md5(str(key_data).encode()).hexdigest()}"
    
    cached_plot = plot_cache.get(cache_key)
    if cached_plot is not None:
        return cached_plot
    
    # Not in cache, compute it
    plot = getRFplot(
        data=data,
        importanceRatings=importance_ratings,
        value_range=value_range,
        pairs=pairs,
        split_by_variety=split_by_variety
    )
    
    # Cache the result for 24 hours
    #plot_cache.set(cache_key, plot, expire=86400)
    plot_cache.set(cache_key, plot)
    return plot

# Load only essential data at startup - defer heavy computations
GrammarItemsCols = retrieve_data.getGrammarItemsCols()
GrammarItemsColsPairs = retrieve_data.getGrammarItemsCols("item_pairs")
grammarMeta = retrieve_data.getGrammarMeta()
grammarMetaPairs = retrieve_data.getGrammarMeta("item_pairs")

# Get minimal informants data for initial tree display
Informants = retrieve_data.getInformantDataGrammar(imputed=True)

# Create simple empty plots for initial display
UMAP_Grammar_initialPlot = go.Figure()
UMAP_Grammar_initialPlot.update_layout(
    template="simple_white"
)

itemPlot_Grammar_initialPlot = go.Figure()
itemPlot_Grammar_initialPlot.update_layout(
    template="simple_white"
)

initial_hoverinfo = construct_initial_hoverinfo(UMAP_Grammar_initialPlot)

# presets
# Dynamically generated from meta table (using imported function from grammarFunctions)
item_presets = generate_dynamic_presets(grammarMeta)
labels_dict = build_preset_multiselect_data(item_presets)




##############
## Layout
##############


customSetWarningModal = dmc.Modal(
    id="confirm-custom-modal",
    centered=True,
    closeOnClickOutside=False,
    closeOnEscape=False,
    withCloseButton=False,
    children=[
        dmc.Text("You are using a custom item selection. Computing the UMAP may take a few minutes and you should not change the view until the render is complete.", mb=20),
        dmc.Text("Do you want to proceed?", fw=700, mb=20),
        dmc.Group(
            [
                dmc.Button("Cancel", id="modal-cancel-button", variant="outline"),
                dmc.Button("OK", id="modal-ok-button", color="blue"),
            ],
        ),
    ],
)

UmapPlotContainer = dmc.Container([

        dmc.Grid(children=[
            dmc.GridCol(children=[
                # View toggle control
                dmc.Group([
                    dmc.SegmentedControl(
                        id="umap-view-toggle",
                        data=[
                            {"value": "umap-plot", "label": "Participant Similarity Plot"},
                            {"value": "rf-plot", "label": "Group Comparison"},
                        ],
                        value="umap-plot",
                        color="blue",
                        size="sm",
                        mb="sm"
                    ),
                ], justify="center"),
                
                # UMAP plot (conditional display)
                html.Div(
                    id="umap-plot-container",
                    children=[
                        dcc.Graph(id="UMAPfig", figure=UMAP_Grammar_initialPlot, style={'height': '70vh'}, config={
                            'toImageButtonOptions': {
                                'format': 'svg',
                                'filename': 'umap_plot',
                                'height': 600,
                                'width': 800,
                                'scale': 1
                            }
                        })
                    ],
                    style={"display": "block"}
                ),
                
                # RF plot (conditional display)
                html.Div(
                    id="rf-plot-container",
                    children=[
                        dcc.Graph(id="RFPlotFig", figure=emptyFig, style={'height': '70vh'}, config={
                            'toImageButtonOptions': {
                                'format': 'svg',
                                'filename': 'rf_plot',
                                'height': 600,
                                'width': 800,
                                'scale': 1
                            }
                        })
                    ],
                    style={"display": "none"}
                ),
            ],span=12),
        ])        
    ], fluid=True)


## Stacks for InformantPlotContainer
AgeGender = dmc.Stack([
    dmc.Text("Age/Gender"),
    dcc.Graph(id="AgeGenderPlotG", figure=getAgeGenderPlot(Informants),style={'height': '200px'})
    ])

MainVarieties = dmc.Stack([

    dmc.Text("Main varieties"),
    html.Div(id="NationalityPlotContainer", children=[
    dcc.Graph(id="MainVarietiesPlotG", figure=getMainVarietiesPlot(Informants))
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

Nationality = dmc.Stack([

    dmc.Text("Nationality"),
    html.Div(id="NationalityPlotContainer", children=[
    dcc.Graph(id="NationalityPlotG", figure=getCategoryHistogramPlot(Informants,"Nationality", True, ""))
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

EthnicSID = dmc.Stack([

    dmc.Text("Ethnic Self-ID"),
    html.Div(id="EIDPlotContainer", children=[
    dcc.Graph(id="EIDPlotG", figure=getCategoryHistogramPlot(Informants,"EthnicSelfID", True,""))
    ], style={'height': 'auto', 'max-height' : '300px','overflowY': 'scroll'}),
])

CountryID = dmc.Stack([

    dmc.Text("Country (or region) you identify with most"),
    html.Div(id="CIDPlotContainer", children=[
    dcc.Graph(id="CIDPlotG", figure=getCategoryHistogramPlot(Informants,"CountryID", True, ""))
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

LanguagesHome = dmc.Stack([

    dmc.Text("Languages used at home"),
    html.Div(id="LanguagesHomePlotContainer", children=[    dcc.Graph(id="LanguagesHomePlotG", figure=getCategoryHistogramPlot(Informants,"LanguageHome_normalized", True, ","))
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])


LanguageMother = dmc.Stack([

    dmc.Text("Mother's Native Language"),
    html.Div(id="LanguagesMotherPlotContainer", children=[
    dcc.Graph(id="LanguagesMotherPlotG", figure=getCategoryHistogramPlot(Informants,"LanguageMother_normalized", True, ","))
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

LanguageFather = dmc.Stack([

    dmc.Text("Father's Native Language"),
    html.Div(id="LanguagesFatherPlotContainer", children=[
    dcc.Graph(id="LanguagesFatherPlotG", figure=getCategoryHistogramPlot(Informants,"LanguageFather_normalized", True, ","))
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

PrimarySchool = dmc.Stack([
    
        dmc.Text("Primary School"),
        dcc.Graph(id="PrimarySchoolPlotG", figure=getCategoryHistogramPlot(Informants,"PrimarySchool",True))

])

SecondarySchool = dmc.Stack([
    
        dmc.Text("Secondary School"),
        dcc.Graph(id="SecondarySchoolPlotG", figure=getCategoryHistogramPlot(Informants,"SecondarySchool",True))

])
Qualifications = dmc.Stack([
    
        dmc.Text("Highest Qualification"),
        dcc.Graph(id="QualiPlotG", figure=getCategoryHistogramPlot(Informants,"Qualifications",True))

])

YearsLivedOutside = dmc.Stack([
    
        dmc.Text("Years lived outside home country"),
        dcc.Graph(id="YLOPlotG", figure=getFloatHistogramPlot(Informants,"YearsLivedOutside"))

])

YearsLivedOtherE = dmc.Stack([
    
        dmc.Text("Years lived in other English-speaking countries"),
        dcc.Graph(id="YLOEPlotG", figure=getFloatHistogramPlot(Informants,"YearsLivedOtherEnglish"))

])

RatioMainVariety = dmc.Stack([
    
        dmc.Text("Ratio Main Variety"),
        dcc.Graph(id="RatioMainVarietyPlotG", figure=getFloatHistogramPlot(Informants,"RatioMainVariety"))

])

PIAccordion = dmc.Accordion(
    children=[
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    "Languages",
                ),
                dmc.AccordionPanel(
                    children=[
                        LanguagesHome,
                        dmc.Divider(),
                        LanguageMother,
                        dmc.Divider(),
                        LanguageFather
                    ]
                    ),
            ],
            value="languages",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    "Regions & Identification",
                ),
                dmc.AccordionPanel(children=[
                    Nationality,
                    dmc.Divider(),
                    EthnicSID,
                    dmc.Divider(),
                    CountryID
                ]
                ),
            ],
            value="seldif",
        ),
    ], variant="default"
)

InformantsGrid = html.Div(children = [
    # View toggle switch at the top
    dmc.Group([
        dmc.SegmentedControl(
            id="informants-view-toggle",
            data=[
                {"value": "table", "label": "Table View"},
                {"value": "plots", "label": "Plot View"},
            ],
            value="table",
            color="blue",
            size="sm"
        ),
    ], justify="center", mb="lg"),
    
    # Table view (default)
    html.Div(
        id="informants-table-view",
        children=[
            dmc.Stack([
                dmc.Group([
                    dmc.Text("Participant Data Table", fw=700),
                    dmc.Button(
                        "Download Table Data",
                        id="download-informants-table-button",
                        size="xs",
                        variant="light",
                        leftSection=DashIconify(icon="tabler:download", width=16)
                    ),
                ], justify="space-between", mb="xs"),
                
                # Column selection checkboxes
                dmc.Accordion(
                    children=[
                        dmc.AccordionItem(
                            [
                                dmc.AccordionControl("Select Columns to Display", style={"fontSize": "14px"}),
                                dmc.AccordionPanel(
                                    dmc.Stack(gap="xs", children=[
                                        dmc.Group([
                                            dmc.Button("Select All", id="select-all-columns-button", size="xs", variant="outline"),
                                            dmc.Button("Deselect All", id="deselect-all-columns-button", size="xs", variant="outline"),
                                        ], mb="xs"),
                                        dmc.CheckboxGroup(
                                            id="informants-columns-checkbox",
                                            children=[
                                                dmc.Grid([
                                                    dmc.GridCol(dmc.Checkbox(label=col.replace("_", " ").replace("ID", " ID").replace("Occup", "Occupation").replace("Quali", "Qualification"), 
                                                                            value=col, size="xs"), span=3)
                                                    for col in ['Age', 'Gender', 'MainVariety', 'AdditionalVarieties',
                                                               'YearsLivedInMainVariety', 'RatioMainVariety', 'CountryCollection', 'Year',
                                                               'Nationality', 'EthnicSelfID', 'CountryID', 'YearsLivedOutside', 
                                                               'YearsLivedInside', 'YearsLivedOtherEnglish', 'LanguageHome_normalized',
                                                               'LanguageFather_normalized', 'LanguageMother_normalized', 'Qualifications_normalized',
                                                               'QualiMother_normalized', 'QualiFather_normalized', 'QualiPartner_normalized',
                                                               'PrimarySchool', 'SecondarySchool', 'Occupation', 'OccupMother', 'OccupFather',
                                                               'OccupPartner'] if col in Informants.columns
                                                ])
                                            ],
                                            value=['Age', 'Gender', 'MainVariety','AdditionalVarieties',
                                                               'YearsLivedInMainVariety', 'RatioMainVariety', 'CountryCollection', 'Year','LanguageHome_normalized',
                                                               'LanguageFather_normalized', 'LanguageMother_normalized'],
                                            persistence=persist_UI,
                                            persistence_type=persistence_type
                                        )
                                    ])
                                ),
                            ],
                            value="column-selection",
                        ),
                    ],
                    variant="contained",
                    mb="xs"
                ),
            ], gap="xs", mb="xs", style={"backgroundColor": "#f8f9fa", "padding": "8px", "borderRadius": "4px", "border": "1px solid #e9ecef","display": "flex","flex-direction":"column"}),
            
            # Table will be updated by callback
            dag.AgGrid(
                id="informants-table",
                rowData=[],  # Initialize empty for performance
                        columnDefs=[
                            {
                                "field": col, 
                                "headerName": col.replace("ID", " ID").replace("_", " ").replace("Occup", "Occupation").replace("Quali", "Qualification").replace("Ethnic", "Ethnic "),
                                "filter": "agTextColumnFilter", 
                                "sortable": True,
                                "resizable": True,
                                "minWidth": 120 if "Language" in col or "Variety" in col else 100,
                                "flex": 1,
                                "cellStyle": {"textAlign": "left"},
                                "headerTooltip": f"Click to sort by {col}. Use filter below to search."
                            } 
                            for col in ['InformantID', 'Age', 'Gender', 'MainVariety', 'Year'] if col in Informants.columns
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
                        style={"flex": "1"},
                        dashGridOptions={
                            "suppressMenuHide": True,
                            "animateRows": True,
                            "enableRangeSelection": True,
                            "pagination": True,
                            "paginationPageSize": 30,
                            "headerHeight": 30,
                            "suppressColumnVirtualisation": True,
                            "enableBrowserTooltips": True,
                            "tooltipShowDelay": 500
                        }
                )
        ],
        style={"display": "block"}
    ),
    html.Div(
        id="informants-plot-view",
        children=[
            dmc.Card(children=[
                dmc.CardSection(children=[
                    dmc.Image(src='../assets/img/UB_logo.png',styles={"root" : {"width":"100px","height":"100px","float":"right","margin-bottom":"10px"}}),
                    dmc.Text("University of Bamberg",size="xl",styles={"root":{"line-height":"1.1"}}),
                    dmc.Text("Chair of English Linguistics",size="sm",styles={"root":{"padding-bottom":"10px"}}),
                    dmc.Text("Bamberg Survey of Language Variation and Change",fw=700,size="sm"),
                    dmc.Text("Participant Information Sheet",size="sm"),
                    dmc.Divider(styles={"root":{"margin-top":"20px"}}),
                ],styles={"section":{"margin":"20px"}}),
                dmc.CardSection(children=[
                    dmc.Grid(children=[ 
                        # "Personal Information" & "Location Timeline"
                        dmc.GridCol(children=[
                            dmc.Card(children=[
                                dmc.Text("Personal Information",fw=700),
                                AgeGender,
                                dmc.Divider(),
                                MainVarieties,
                                dmc.Divider(),
                                PIAccordion,
                                ],
                                withBorder=True,
                                shadow="sm",
                                radius="md"),
                            dmc.Card(children=[
                                dmc.Text("Location Timeline",fw=500),
                                YearsLivedOutside,
                                dmc.Divider(),
                                YearsLivedOtherE,
                                dmc.Divider(),
                                RatioMainVariety],
                                withBorder=True,
                                shadow="sm",
                                radius="md")
                            ],span=6),
                        # "Education Profile"
                        dmc.GridCol(children=[
                            dmc.Card(children=[
                                dmc.Text("Education Profile",fw=500),
                                PrimarySchool,
                                dmc.Divider(),
                                SecondarySchool,
                                dmc.Divider(),
                                Qualifications],
                                withBorder=True,
                                shadow="sm",
                                radius="md"),
                            ],span=6),
                    ])
                ],styles={"section":{"margin-left":"20px","margin-right":"20px","margin-bottom":"20px"}}),
            ],withBorder=True,shadow="sm",radius="md",style={
                "maxHeight": "calc(100vh - 230px)",
                "overflowY": "auto",
                "overflowX": "hidden"
            })
        ],
        style={"display": "none"}
    )
])

################


ItemPlotContainer = dmc.Container([dmc.Grid(children=[
            dmc.GridCol(children=[
                dcc.Graph(id="ItemFig", figure=itemPlot_Grammar_initialPlot, style={'height': 'auto'}, config={
                    'toImageButtonOptions': {
                        'format': 'svg',
                        'filename': 'item_plot',
                        'height': 600,
                        'width': 800,
                        'scale': 1
                    }
                })
            ],span=12),
        ])        
    ], fluid=True)

InformantsPlotContainer = InformantsGrid



GrammarPlots = dmc.Container([
    dmc.Title("Grammar Items", order = 2),
    dmc.Text("Plots of sentences, etc."),
], fluid=True)

# Add Leiden clustering container
LeidenClusterContainer = dmc.Container([
    dmc.Grid(children=[
        dmc.GridCol(children=[
            dmc.Card(children=[
                dcc.Graph(id="leiden-cluster-fig", style={'height': '70vh'})
            ], withBorder=True,
            shadow="sm",
            radius="md")
        ],span=8),
        dmc.GridCol(children=[
            dmc.Card(children=[
                dmc.Stack([
                    dmc.Text("Cluster Statistics", fw=700, mb=10),
                    html.Div(id="cluster-stats-table"),
                    dmc.Divider(),
                    dmc.Text("Selected Cluster Details", fw=700, mb=10),
                    html.Div(id="cluster-details-table")
                ])
            ], withBorder=True,
            shadow="sm",
            radius="md")
        ],span=4),
    ])        
], fluid=True)

# accordion items for reuse




# Add Leiden settings accordion after umapGroupCompAccordion
leidenSettingsAccordion = dmc.AccordionItem(
    [
        dmc.AccordionControl(
            "Leiden clustering settings",
            icon=DashIconify(
                icon="tabler:cluster",
                color="#1f77b4",
                width=20,
            ),
        ),
        dmc.AccordionPanel(
            dmc.Stack(gap='xl',children=[
                dmc.Text("Leiden Clustering Parameters:"),
                dmc.Text("Resolution (higher values = more clusters)"),
                dmc.Slider(
                    id="leiden-resolution",
                    value=1.0,
                    min=0.1,
                    max=2.0,
                    step=0.1,            
                    marks=[
                        {"value": 0.5, "label": "0.5"},
                        {"value": 1.0, "label": "1.0"},
                        {"value": 1.5, "label": "1.5"},
                        {"value": 2.0, "label": "2.0"},
                    ],
                    persistence=persist_UI,
                    persistence_type=persistence_type
                ),
                dmc.Text("Similarity Threshold (minimum similarity for connections)"),
                dmc.Slider(
                    id="similarity-threshold",
                    value=0.5,
                    min=0.1,
                    max=0.9,
                    step=0.05,            
                    marks=[
                        {"value": 0.1, "label": "0.1"},
                        {"value": 0.5, "label": "0.5"},
                        {"value": 0.9, "label": "0.9"},
                    ],
                    persistence=persist_UI,
                    persistence_type=persistence_type
                ),
                dmc.Select(
                    label="Color points by",
                    id="leiden-color-dropdown",
                    value="cluster",
                    data=[
                        {"value": "cluster", "label": "Cluster"},
                        {"value": "variety", "label": "Variety"},
                        {"value": "gender", "label": "Gender"},
                    ],
                    allowDeselect=False,
                    persistence=persist_UI, 
                    persistence_type=persistence_type
                ),
                dmc.Switch(
                    id="leiden-pca-switch",
                    label="Apply PCA before clustering",
                    checked=True,
                    persistence=persist_UI, 
                    persistence_type=persistence_type
                ),
                dmc.NumberInput(
                    id="leiden-pca-components",
                    label="Number of PCA components",
                    value=10,
                    min=2,
                    max=50,
                    step=1,
                    persistence=persist_UI, 
                    persistence_type=persistence_type
                ),
                dmc.Text("Note: PCA is applied for dimensionality reduction before clustering and for visualization. Higher similarity thresholds result in sparser networks and potentially more clusters.", 
                         size="sm", 
                         c="dimmed")
            ])
        ),
    ],
    value="LeidenSettings",
)

#filter informants accordion
informantSelectionAccordion = dmc.AccordionItem(
    [
        dmc.AccordionControl(
            dmc.Group([
                DashIconify(icon="tabler:users-group", color="#1f77b4", width=18),
                dmc.Text("Participants", fw=500, size="sm"),
                dmc.Badge("0", id="participants-badge", color="blue", variant="filled", size="sm")
            ], gap="xs"),
        ),
        dmc.AccordionPanel(
            dmc.Stack(gap='md', children=[
                # --- Participants tree and buttons at the top ---
                dmc.Group(children=[
                    dmc.Button("Select All", id='select-all-participants', size="xs", variant="outline"),
                    dmc.Button("Deselect All", id='deselect-all-participants', size="xs", variant="outline"),
                ], mb="xs"),
                html.Div(id="lasso-selection-buttons", children=[
                    dmc.Group(children=[
                        dmc.Button(
                            "Select Lasso",
                            id='select-only-lasso-participants',
                            size="xs",
                            variant="outline",
                            disabled=True,
                            leftSection=DashIconify(icon="tabler:lasso", width=16),
                        ),
                        dmc.Button(
                            "Deselect Lasso",
                            id='deselect-selected-participants',
                            size="xs",
                            variant="outline",
                            disabled=True,
                            leftSection=DashIconify(icon="tabler:lasso-off", width=16),
                        ),
                    ], mb="xs", wrap="wrap", gap="xs"),
                ]),  # Visible by default (UMAP is default now)
                
                # Regional mapping switch
                dmc.Group([
                    dmc.Switch(
                        id='regional-mapping-switch',
                        label="Split England into regions (North/South)",
                        size="sm",
                        checked=False,
                    )
                ], mb="sm"),
                
                # Batch Operations - Quick Selection
                dmc.Accordion(
                    children=[
                        dmc.AccordionItem(
                            [
                                dmc.AccordionControl("Quick Selection", style={"fontSize": "12px"}),
                                dmc.AccordionPanel(
                                    dmc.Stack(gap="xs", children=[
                                        dmc.Text("Variety:", size="xs", fw=600, c="dimmed"),
                                        dmc.Group(children=[
                                            dmc.Button("ENL", id='batch-select-enl', size="xs", variant="light"),
                                            dmc.Button("ESL", id='batch-select-esl', size="xs", variant="light"),
                                            dmc.Button("EFL", id='batch-select-efl', size="xs", variant="light"),
                                            dmc.Button("Balanced Per Variety", id='batch-select-equal-per-variety', size="xs", variant="light"),
                                        ], gap="xs", mb="xs"),
                                        dmc.Text("Age:", size="xs", fw=600, c="dimmed"),
                                        dmc.Group(children=[
                                            dmc.Button("18-25", id='batch-select-age-18-25', size="xs", variant="light"),
                                            dmc.Button("26-35", id='batch-select-age-26-35', size="xs", variant="light"),
                                            dmc.Button("36-50", id='batch-select-age-36-50', size="xs", variant="light"),
                                            dmc.Button("50+", id='batch-select-age-50plus', size="xs", variant="light"),
                                        ], gap="xs", mb="xs"),
                                        dmc.Text("Gender:", size="xs", fw=600, c="dimmed"),
                                        dmc.Group(children=[
                                            dmc.Button("Female", id='batch-select-female', size="xs", variant="light"),
                                            dmc.Button("Male", id='batch-select-male', size="xs", variant="light"),
                                        ], gap="xs", mb="xs"),
                                        dmc.Group(children=[
                                            dmc.Button("Balanced", id='batch-select-balanced', size="xs", variant="light"),
                                            dmc.Button("Balanced Per Variety", id='batch-select-balanced-per-variety', size="xs", variant="light"),
                                        ], gap="xs", mb="xs"),
                                        dmc.Text("Missing data:", size="xs", fw=600, c="dimmed"),
                                        dmc.Group(children=[
                                            dmc.Button("0%", id='batch-select-missing-0', size="xs", variant="light"),
                                            dmc.Button("<5%", id='batch-select-missing-5', size="xs", variant="light"),
                                            dmc.Button("<10%", id='batch-select-missing-10', size="xs", variant="light"),
                                        ], gap="xs", mb="xs"),
  
                                    ])
                                ),
                            ],
                            value="batch-ops",
                        ),
                    ],
                    variant="contained",
                    mb="xs"
                ),
                dmc.Tree(id='participantsTree', data=drawParticipantsTree(Informants), checkboxes=True, checked=[]),
                # --- Nested Accordion for filters ---
                dmc.Accordion(
                    children=[
                        dmc.AccordionItem(
                            [
                                dmc.AccordionControl(
                                    dmc.Text("Advanced Filters", size="xs", fw=500)
                                ),
                                dmc.AccordionPanel(
                                    dmc.Stack(gap='sm', children=[
                                        dmc.CheckboxGroup(
                                            id="checkbox-grammar-filter-gender",
                                            label="Gender:",
                                            size="xs",
                                            children=dmc.Group([
                                                dmc.Checkbox(label="F", value="Female", size="xs"),
                                                dmc.Checkbox(label="M", value="Male", size="xs"),
                                                dmc.Checkbox(label="NB", value="Non-binary", size="xs"),
                                                dmc.Checkbox(label="NA", value="NA", size="xs"),
                                            ], gap="xs"),
                                            value=["Female", "Male", "Non-binary", "NA"],
                                            persistence=persist_UI, persistence_type=persistence_type
                                        ),
                                        dmc.Stack(gap="xs", children=[
                                            dmc.Group([
                                                dmc.Text("Age:", size="xs", fw=500),
                                                dmc.Checkbox(
                                                    id="checkbox-grammar-filter-age-missing",
                                                    label="Include NA",
                                                    value=True,
                                                    size="xs",
                                                    persistence=persist_UI, persistence_type=persistence_type
                                                ),
                                            ], justify="space-between"),
                                            dmc.RangeSlider(
                                                id="rangeslider-grammar-filter-age",
                                                min=0,
                                                max=100,
                                                step=1,
                                                size="xs",
                                                marks=[{"value": 0, "label": "0"},
                                                       {"value": 110, "label": "110"}],
                                                value=[0, 110],
                                                persistence=persist_UI, persistence_type=persistence_type
                                            ),
                                        ]),
                                        dmc.Stack(gap="xs", children=[
                                            dmc.Group([
                                                dmc.Text("Variety ratio:", size="xs", fw=500),
                                                dmc.Checkbox(
                                                    id="checkbox-grammar-filter-ratio-missing",
                                                    label="Include NA",
                                                    value=True,
                                                    size="xs",
                                                    persistence=persist_UI, persistence_type=persistence_type
                                                ),
                                            ], justify="space-between"),
                                            dmc.Text("Years in main variety / age", size="xs", c="dimmed"),
                                            dmc.RangeSlider(
                                                id="rangeslider-grammar-filter-ratio",
                                                min=0,
                                                max=100,
                                                step=1,
                                                size="xs",
                                                marks=[{"value": 0, "label": "0%"},
                                                       {"value": 50, "label": "50%"},
                                                       {"value": 100, "label": "100%"}],
                                                value=[20, 100],
                                                persistence=persist_UI, persistence_type=persistence_type
                                            ),
                                        ]),
                                        dmc.MultiSelect(
                                            id="multiselect-grammar-filter-mainvariety",
                                            label="Main Variety:",
                                            size="xs",
                                            data=[{"label": v, "value": v} for v in sorted(Informants["MainVariety"].dropna().unique())],
                                            value=sorted(Informants["MainVariety"].dropna().unique()),
                                            clearable=False,
                                            persistence=persist_UI, persistence_type=persistence_type
                                        ),
                                        dmc.Button(
                                            'Apply Filters',
                                            id='apply-grammar-filters',
                                            size="xs",
                                            variant="light",
                                            fullWidth=True,
                                            loaderProps={"type": "dots"}
                                        ),
                                    ])
                                ),
                            ],
                            value="filters",
                        ),
                    ],
                    variant="contained",
                    radius="md",
                    value=[],  # Closed by default for cleaner initial view
                ),
            ])
        ),
    ],
    value="LoadData",
)
itemSelectionAccordion = dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Group([
                        DashIconify(icon="tabler:list-check", color="#2f9e44", width=18),
                        dmc.Text("Grammar Items", fw=500, size="sm"),
                        dmc.Badge("0", id="items-badge", color="green", variant="filled", size="sm")
                    ], gap="xs"),
                ),
                dmc.AccordionPanel(
                    dmc.Stack(gap='md',children=[
                        # Select All / Deselect All buttons at the top
                        dmc.Group(children=[
                            dmc.Button("Select All", id='select-all-grammar-items', size="xs", variant="outline"),
                            dmc.Button("Deselect All", id='deselect-all-grammar-items', size="xs", variant="outline"),
                            dmc.Button([
                                DashIconify(icon="tabler:filter-x", width=14),
                                " Problematic"
                            ],
                                id="grammar_deselect_problematic",
                                variant="outline",
                                size="xs"
                            )
                                                    ], mb="xs"),
                        dmc.Text(
                            "💡 Tip: Use the 'Grammar Items' tab to browse and select items more easily using the interactive table.",
                            size="sm",
                            c="dimmed",
                            style={"fontStyle": "italic", "marginBottom": "8px"}
                        ),
                        html.Div([
                            dmc.Tree(
                                id='grammarItemsTree',
                                data=drawGrammarItemsTree(grammarMeta,pairs=False), 
                                checkboxes=True, 
                                checked=[],

                            )
                        ],
                        
                        className="grammar-tree-wrapper",
                        ),
                        dmc.MultiSelect(label="Select a Preset:",
                            placeholder="Select one or more presets",
                            id="grammar-items-preset",
                            value=[],
                            data=labels_dict,
                            searchable=True,
                            clearable=True,
                            nothingFoundMessage="Nothing found...",
                            size="xs",
                            style={"flex": 1},
                            comboboxProps={"position": "bottom"},
                            persistence=persist_UI,persistence_type=persistence_type),
                        # Use a wrapper div with custom CSS class
                        # Advanced item options sub-accordion
                        dmc.Accordion(
                            children=[
                                dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl(
                                            dmc.Text("Advanced Options", size="xs", fw=500)
                                        ),
                                        dmc.AccordionPanel(
                                            dmc.Stack(gap='sm', children=[
                                                dmc.Switch(
                                                    id="grammar-type-switch",
                                                    label="Use item difference (spoken-written)",
                                                    description="Use difference between item pairs, instead of raw ratings",
                                                    checked=False,
                                                    persistence=persist_UI,
                                                    persistence_type=persistence_type,
                                                    size="sm",
                                                ),
                                                dmc.Switch(
                                                    id="use-imputed-data-switch",
                                                    label="Use imputed data",
                                                    description="Toggle between imputed and raw data. UMAP always uses imputed data.",
                                                    checked=False,
                                                    persistence=persist_UI,
                                                    persistence_type=persistence_type,
                                                    size="sm",
                                                ),
                                                dmc.Group(children=[
                                                    dmc.Button("Toggle Written-Only",
                                                        id="grammar_toggle_written_only",
                                                        size="xs",
                                                        variant="light"
                                                    ),
                                                    dmc.Button([
                                                        DashIconify(icon="tabler:coin-off", width=14),
                                                        " Currency/Unit"
                                                    ],
                                                        id="grammar_toggle_currency",
                                                        size="xs",
                                                        variant="light"
                                                    ),
                                                ], gap="xs"),
                                            ])
                                        ),
                                    ],
                                    value="advanced-item-options",
                                ),
                            ],
                            variant="contained",
                            radius="md",
                            value=[],  # Closed by default
                        ),
                    ])
                ),
            ],
            value="LoadItems",
        )
umapSettingsAccordion = dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Group([
                        DashIconify(icon="tabler:radar", width=18),
                        dmc.Text("UMAP Settings", fw=500, size="sm"),
                    ], gap="xs"),
                ),
                dmc.AccordionPanel(
                    dmc.Stack(gap='md',children=[
                        dmc.Select(
                            label="Color:",
                            id="umap-color-dropdown",
                            value="Variety",
                            data=[
                                {"value": "Variety", "label": "Variety"},
                                {"value": "Variety type", "label": "Variety type"},
                                {"value": "Gender", "label": "Gender"},
                            ],
                            size="xs",
                            allowDeselect=False,
                            persistence=persist_UI, persistence_type=persistence_type
                        ),
                        dmc.Select(
                            label="Distance metric:",
                            id="umap-distance-metric-dropdown",
                            value="cosine",
                            data=[
                                {"value": "cosine", "label": "Cosine"},
                                {"value": "euclidean", "label": "Euclidean"},
                                {"value": "manhattan", "label": "Manhattan"},
                            ],
                            size="xs",
                            allowDeselect=False,
                            persistence=persist_UI, persistence_type=persistence_type
                        ),
                        dmc.Checkbox(
                            id="umap-standardize-checkbox",
                            label="Standardize participant ratings",
                            size="sm",
                            checked=False,
                            persistence=persist_UI, persistence_type=persistence_type
                        ),
                        dmc.Checkbox(
                            id="umap-densemap-checkbox",
                            label="Use density-preserving embedding (DensMAP)",
                            size="sm",
                            checked=False,
                            persistence=persist_UI, persistence_type=persistence_type
                        ),
                        dmc.Text("UMAP Hyperparameters:", size="xs", fw=600, c="dimmed", mt="xs"),
                        dmc.Stack(gap="xs", children=[
                            dmc.Text("Number of neighbours:", size="xs"),
                            dmc.Slider(id="UMAP_neighbours",value=25,min=0,max=100,step=1,
                                size="xs",
                                marks=[
                                    {"value": 25, "label": "25"},
                                    {"value": 50, "label": "50"},
                                    {"value": 75, "label": "75"},
                                ],
                                persistence=persist_UI,persistence_type=persistence_type),
                        ]),
                        dmc.Stack(gap="xs", children=[
                            dmc.Text("Minimal distance:", size="xs"),
                            dmc.Slider(id="UMAP_mindist",value=0.1,min=0,max=0.99,step=0.05,
                                size="xs",
                                marks=[
                                    {"value": 0, "label": "0"},
                                    {"value": 0.5, "label": "0.5"},
                                    {"value": 0.99, "label": "0.99"},
                                ],
                                persistence=persist_UI,persistence_type=persistence_type),
                        ]),
                    ])
                ),
            ],
            value="PlotSettings",
        )
itemPlotSettingsAccordion = dmc.AccordionItem([
    dmc.AccordionControl(
        dmc.Group([
            DashIconify(icon="tabler:settings", width=18),
            dmc.Text("Plot Settings", fw=500, size="sm"),
        ], gap="xs"),
    ),
    dmc.AccordionPanel(
        dmc.Stack(gap='md',children=[
            dmc.Select(
                label="Plot mode:",
                id="items-plot-mode",
                value="normal",
                data=[
                    {"value":"normal","label":"Mean (95% CI)"},
                    {"value":"split_by_variety","label":"Mean (95% CI - split varieties)"},
                    {"value":"diverging","label":"Diverging stacked bars"},
                    {"value":"informant_boxplot","label":"Informant mean of selected items (boxplot)"},
                    {"value":"correlation_matrix","label":"Correlation matrix"},
                    {"value":"missing_values_heatmap","label":"Missing values heatmap"},
                ],
                size="xs",
                allowDeselect=False,
                persistence=persist_UI,persistence_type=persistence_type),
            dmc.Select(
                label="Group by:",
                id="items-group-by",
                value="variety",
                data=[
                    {"value":"variety","label":"Variety"},
                    {"value":"vtype","label":"Variety type"},
                    {"value":"gender","label":"Gender"},
                ],
                size="xs",
                allowDeselect=False,
                persistence=persist_UI,persistence_type=persistence_type),
            dmc.Select(
                label="Sort by:",
                id="items-sort-by",
                value="mean",
                data=[
                    {"value":"mean","label":"Mean"},
                    {"value":"sd","label":"Standard deviation"},
                    {"value":"alpha","label":"Alphabetically"},
                ],
                size="xs",
                allowDeselect=False,
                persistence=persist_UI,persistence_type=persistence_type),
        ])
    )
],value="PlotSettings2",
)
umapGroupCompAccordion = dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Group([
                        DashIconify(icon="tabler:chart-dots", width=18),
                        dmc.Text("Group Comparison", fw=500, size="sm"),
                    ], gap="xs"),
                ),
                dmc.AccordionPanel(
                    dmc.Stack(gap='md',children=[
                        dmc.Stack(gap="xs", children=[
                            dmc.Text("Filter by Average Rating:", size="xs", fw=600, c="dimmed"),
                            dmc.Text("Show only items where all groups rate within this range", size="xs", c="dimmed"),
                            dmc.Box(
                                dmc.RangeSlider(
                                    id="RF_avg_range",
                                    value=[0,5],
                                    min=0,
                                    max=5,
                                    step=0.25,
                                    minRange=1,
                                    size="xs",
                                    marks=[
                                        {"value": 0, "label": "No-one"},
                                        {"value": 1, "label": "Few"},
                                        {"value": 2, "label": "Some"},
                                        {"value": 3, "label": "Many"},
                                        {"value": 4, "label": "Most"},
                                        {"value": 5, "label": "Everyone"},
                                    ]
                                ),
                                px="xs",  # Add horizontal padding to prevent edge labels from clipping
                                mb="md"  # Add bottom margin to prevent overlap
                            ),
                        ]),
                        dmc.Checkbox(
                            id="rf-use-zscores",
                            label="Use Z-Scores",
                            description="Standardize participant ratings row-wise before training",
                            size="sm",
                            checked=False,
                            persistence=persist_UI,
                            persistence_type=persistence_type
                        ),
                    ])
                ),
            ],
            value="CompareGroups"
        )

# Merged Settings for Grammar Analysis (Item Plot + UMAP)
SettingsGrammarAnalysis = dmc.Card([
    # Collapsible Quick Stats Panel
    dmc.Accordion(
        children=[
            dmc.AccordionItem([
                dmc.AccordionControl("Selection Overview"),
                dmc.AccordionPanel(
                    html.Div(id="stats-summary", children=[
                        dmc.Text("Select participants and items to begin", size="sm", c="dimmed"),
                    ])
                )
            ], value="selection-overview")
        ],
        value="selection-overview",  # Open by default
        variant="filled",
        radius="md",
        mb="md"
    ),
    
    # Simplified Analysis Type Selector
    dmc.Stack([
        dmc.Text("Analysis Type:", size="sm", fw=500, mb="xs"),
        dmc.SegmentedControl(
            id="grammar-plot-type",
            data=[
                {"value": "umap", "label": "Participant Similarity"},
                {"value": "item", "label": "Item Ratings"},
            ],
            value="umap",
            fullWidth=True,
            color="blue",
            size="sm"
        ),
        dmc.Text(
            id="plot-type-description",
            children="Apply dimensionality reduction (UMAP) to explore how similar participants are to each other based on their grammar ratings",
            size="xs",
            c="dimmed"
        )
    ], gap="xs", mb="md"),
    
    # Primary Action Button with Loading State
    dmc.Box(
        pos="relative",
        mb="xs",
        children=[
            dmc.LoadingOverlay(
                id="render-loading-overlay",
                visible=False,
                overlayProps={"radius": "md", "blur": 2},
                loaderProps={"color": "blue", "type": "dots", "size": "xl"},
            ),
            dmc.Button(
                'Render Plot',
                id='render-grammar-plot',
                size="md",
                leftSection=DashIconify(icon="tabler:chart-line", width=20),
                color="blue",
                fullWidth=True,
                disabled=False,
            ),
        ]
    ),
        # UMAP-specific buttons (shown only when plot type is UMAP)
    html.Div(id="umap-group-buttons", children=[
        dmc.Group(children=[
            dmc.Button('Add Group', id='Umap-add-group', variant="outline", disabled=True),
            dmc.Button('Clear Groups', id='Umap-clear-groups', variant="outline", disabled=True),
        ],
        grow=True,
        wrap="nowrap",
        mb="md"),
        dmc.Group(children=[
            dmc.Button('Compare Selected Groups', id='render-rf-plot', variant="outline", loading=False, disabled=True),
        ],
        grow=True,
        wrap="nowrap",
        mb="md"),
    ]),  # Visible by default (UMAP is default now)
    
    # Common accordions (always visible)
    dmc.Accordion(children=[
        informantSelectionAccordion,
        itemSelectionAccordion,
    ], 
    variant="contained",
    radius="md",
    mb="md",
    value=["LoadData", "LoadItems"]),  # Keep both accordions open by default
    
    # Plot-specific settings (conditionally visible)
    html.Div(id="item-plot-settings-container", children=[
        dmc.Accordion(children=[
            itemPlotSettingsAccordion,
        ], 
        variant="contained",
        radius="md"),
    ], style={"display": "none"}),  # Hidden by default (UMAP is default now)
    
    html.Div(id="umap-settings-container", children=[
        dmc.Accordion(children=[
            umapSettingsAccordion,
            umapGroupCompAccordion
        ], 
        variant="contained",
        radius="md"),
    ]),  # Visible by default (UMAP is default now)
    
    # Collapsible Advanced Actions Section (Moved to bottom)
    dmc.Accordion(
        children=[
            dmc.AccordionItem(
                value="advanced",
                children=[
                    dmc.AccordionControl(
                        dmc.Group([
                            DashIconify(icon="tabler:settings", width=18),
                            dmc.Text("Advanced Actions", size="sm", fw=500)
                        ], gap="xs")
                    ),
                    dmc.AccordionPanel([
                        # Data Export Section
                        dmc.Stack([
                            dmc.Group([
                                DashIconify(icon="tabler:database-export", width=16),
                                dmc.Text("Data Export", size="sm", fw=500)
                            ], gap="xs", mb="xs"),
                            # Export format options
                            dmc.Stack([
                                dmc.Checkbox(
                                    id='export-include-sociodemographic-checkbox',
                                    label="Include sociodemographic data",
                                    checked=True,
                                    size="xs"
                                ),
                                dmc.Checkbox(
                                    id='export-include-item-metadata-checkbox',
                                    label="Include item metadata",
                                    checked=False,
                                    size="xs"
                                ),
                            ], gap="xs", mb="xs"),
                            dmc.Group([
                                dmc.Button(
                                    "Export Raw Data",
                                    id='export-data-button',
                                    size="xs",
                                    variant="light",
                                    leftSection=DashIconify(icon="tabler:table-export", width=14),
                                    fullWidth=True
                                ),
                            ], grow=True),
                            dmc.Divider(orientation="horizontal", variant="solid", color="gray", mt="xs", mb="xs"),
                            # Export Distance Matrix button (only visible in Participant Similarity mode)
                            html.Div(
                                id='export-distance-matrix-container',
                                children=[
                                    dmc.Group([
                                        dmc.Button(
                                            "Export Distance Matrix",
                                            id='export-distance-matrix-button',
                                            size="xs",
                                            variant="light",
                                            leftSection=DashIconify(icon="tabler:chart-dots", width=14),
                                            fullWidth=True
                                        ),
                                    ], grow=True),
                                ],
                                style={"display": "block"}  # Visible by default (UMAP is default)
                            ),
                        ], gap="xs", mb="md"),
                        
                        # Settings Section
                        dmc.Stack([
                            dmc.Group([
                                DashIconify(icon="tabler:settings", width=16),
                                dmc.Text("Settings", size="sm", fw=500)
                            ], gap="xs", mb="xs"),
                            dmc.Group([
                                dmc.Button(
                                    "Copy Settings",
                                    id='copy-settings-button',
                                    size="xs",
                                    variant="light",
                                    leftSection=DashIconify(icon="tabler:copy", width=14),
                                    fullWidth=True
                                ),
                                dmc.Button(
                                    "Paste Settings",
                                    id='paste-settings-button',
                                    size="xs",
                                    variant="light",
                                    leftSection=DashIconify(icon="tabler:clipboard", width=14),
                                    fullWidth=True
                                ),
                            ], grow=True),
                            dmc.Group([
                                dmc.Button(
                                    "Save Settings",
                                    id='save-current-settings',
                                    size="xs",
                                    variant="subtle",
                                    color="gray",
                                    leftSection=DashIconify(icon="tabler:device-floppy", width=14),
                                    fullWidth=True
                                ),
                                dmc.Button(
                                    "Restore Settings",
                                    id='restore-saved-settings',
                                    size="xs",
                                    variant="subtle",
                                    color="gray",
                                    leftSection=DashIconify(icon="tabler:restore", width=14),
                                    fullWidth=True
                                ),
                            ], grow=True),
                        ], gap="xs"),
                    ])
                ]
            )
        ],
        variant="separated",
        radius="md",
        mb="md"
    ),
    
    # Download components (hidden)
    dcc.Download(id="download-data"),
    dcc.Download(id="download-distance-matrix"),
    
    # Clipboard store for settings (client-side only)
    dcc.Store(id='clipboard-settings-store', storage_type='memory'),
    
    # Modal for pasting settings
    dmc.Modal(
        id="paste-settings-modal",
        title="Paste Settings",
        size="lg",
        children=[
            dmc.Text("Paste the base64-encoded settings string you received:", size="sm", mb="xs"),
            dmc.Textarea(
                id="paste-settings-textarea",
                placeholder="Paste the settings code here...",
                minRows=4,
                autosize=True,
                mb="md"
            ),
            dmc.Group([
                dmc.Button("Load Settings", id="load-pasted-settings", color="blue"),
                dmc.Button("Cancel", id="cancel-paste-settings", variant="subtle"),
            ], justify="flex-end")
        ]
    ),
    
    # Hidden buttons for backward compatibility with existing callbacks
    html.Div(style={"display": "none"}, children=[
        dmc.Button('Render UMAP', id='render-UMAP-plot', loaderProps={"type": "dots"}),
        dmc.Button('Render plot', id='render-item-plot', loaderProps={"type": "dots"}),
    ]),
    

    
], withBorder=True, shadow="sm", radius="md", p="md", style={"height": "calc(100vh - 160px)", "overflowY": "auto"})

# Deleted: SettingsInformants (deprecated - merged into Grammar Analysis)

# Settings for Leiden clustering
SettingsLeiden = dmc.Container([
 dmc.Group(children=[
        dmc.Button('Run Leiden Clustering', id='render-leiden-plot', loaderProps={"type": "dots"}),
    ],
    grow=True,
    wrap="nowrap"),
    dmc.Accordion(children=[
        leidenSettingsAccordion,
    ], 
                  variant="default",
                  radius="md"),
], fluid=True)

# Merged Grammar Analysis Container (replaces legacy itemC and umapC containers)
grammarAnalysisC = dmc.Grid([
    dmc.GridCol(html.Div(children=[
        # Collapse toggle button
        dmc.ActionIcon(
            DashIconify(icon="tabler:layout-sidebar-right-collapse", width=20),
            id="toggle-sidebar-button",
            variant="subtle",
            size="lg",
            style={
                "position": "absolute",
                "right": "10px",
                "top": "10px",
                "zIndex": 1000
            }
        ),
        dmc.Card(children=[
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Plot View", value="plot-view", id="grammar-analysis-plot-tab"),
                            dmc.TabsTab("Sociodemographic Details", value="sociodemographic-details"),
                            dmc.TabsTab("Grammar Items", value="grammar-items-table"),
                        ]
                    ),
                    dmc.TabsPanel(
                        # Unified plot container that shows either Item Plot or UMAP based on plot type
                        html.Div(id="grammar-unified-plot-container", children=[
                            # Item Plot Container (hidden by default - UMAP is default now)
                            html.Div(id="item-plot-display", children=[ItemPlotContainer], style={"display": "none"}),
                            # UMAP Plot Container (visible by default - UMAP is default now)
                            html.Div(id="umap-plot-display", children=[UmapPlotContainer], style={"display": "block"}),
                        ]),
                        value="plot-view"
                    ),
                    dmc.TabsPanel(
                        InformantsPlotContainer,
                        value="sociodemographic-details"
                    ),
                    dmc.TabsPanel(
                        getMetaTable(grammarMeta, preset_data=labels_dict),
                        value="grammar-items-table"
                    ),
                ],
                id='grammar-analysis-tabs',
                color="blue",
                orientation="horizontal",
                variant="default",
                value="plot-view",
                style={"height": "calc(100vh - 180px)"}  # Full height minus header/footer
            )
        ], withBorder=True, shadow="sm", radius="md", style={"height": "calc(100vh - 160px)"})],
        id="grammar-analysis-tab-content",
        style={"height": "calc(100vh - 150px)"}),
        id="main-content-col",
        span=8),
    dmc.GridCol(
        SettingsGrammarAnalysis,
        id="sidebar-col",
        span=4,
        style={}
    ),
], gutter="xl", id="grammar-analysis-grid")

# Deleted: informantsC container (deprecated - merged into grammarAnalysisC)

# tab container for Leiden clustering
leidenC = dmc.Grid([
    dmc.GridCol(html.Div(children = [LeidenClusterContainer],id="leiden-plot-tab-content",style={"paddingTop": 10}),span=8),
    dmc.GridCol(SettingsLeiden,span=4,style={"padding-top":"10px","margin-top": "5px","border-left": "1px solid #f0f0f0","padding-left": "10px"}),
], gutter="xl")

GrammaticalItems = dmc.Container([dmc.Grid(children=[
            dmc.GridCol(children=[
                dmc.Card(children=[
                        getMetaTable(grammarMeta, preset_data=labels_dict) 
                ], withBorder=True,
                shadow="sm",
                radius="md")
            ],span=12),
        ])        
    ], fluid=True)



layout = html.Div([

    customSetWarningModal,
    dcc.Location(id='url', refresh=False),  # URL location component for parsing URL parameters
    dcc.Store(id="england-mapping-param", storage_type="memory", data=False),  # Store for EnglandMapping URL parameter
    dcc.Store(id="UMAPgroup", storage_type="memory",data=0),
    dcc.Store(id="UMAPparticipants",storage_type="memory",data=[]), # Start empty - no auto-selection
    dcc.Store(id="UMAPitems",storage_type="memory",data=[]), # Start empty - no auto-selection
    dcc.Store(id="UMAPGroupsForRF",storage_type="memory",data={"dataframe":pd.DataFrame().to_dict("records")}),
    dcc.Store(id="grammar_plots_UMAP",storage_type="memory",data=None),
    dcc.Store(id="grammar_plots_item",storage_type="memory",data=itemPlot_Grammar_initialPlot),
    dcc.Store(id="informants-store", data=Informants.to_dict("records")),  
    dcc.Store(id="leiden-cluster-data", storage_type="memory"),
    dcc.Store(id="umap-render-trigger", storage_type="memory"),  # Trigger for background UMAP computation
    dcc.Store(id="umap-render-settings", storage_type="memory", data={"pairs": False, "use_imputed": True}),  # Store settings used for UMAP render (for RF plot consistency)
    
    # Settings persistence stores
    dcc.Store(id="saved-item-settings", storage_type="local"),
    dcc.Store(id="saved-umap-settings", storage_type="local"),
    dcc.Store(id="last-rendered-item-plot", storage_type="session"),  # Plot persistence
    dcc.Store(id="last-rendered-umap-plot", storage_type="session"),
    dcc.Store(id="last-sociodemographic-settings", storage_type="session"),  # Cache for sociodemographic plot settings
    # Directly show Grammar Analysis content (no outer tabs)
    grammarAnalysisC,
    
    html.Div(
        [
            html.Div(id="notify-container"),
        ],
    )
    ])





##############
## Helper Functions for Callbacks
##############

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


def initialize_marker_arrays(figure_data, ids_len):
    """
    Initialize marker symbol and opacity arrays for a figure trace.
    Ensures arrays are properly sized and typed for numpy operations.
    
    Args:
        figure_data: Dictionary containing marker data
        ids_len: Expected length of marker arrays
    
    Returns:
        Tuple of (marker_symbol_array, marker_opacity_array) as numpy arrays
    """
    # Initialize symbol array
    marker_symbol = figure_data['marker'].get('symbol', 0)
    if isinstance(marker_symbol, int):
        marker_symbol = np.full(ids_len, marker_symbol, dtype=int)
    elif isinstance(marker_symbol, list):
        if len(marker_symbol) != ids_len:
            marker_symbol = np.full(ids_len, 0, dtype=int)
        else:
            marker_symbol = np.array(marker_symbol, dtype=int)
    else:
        marker_symbol = np.full(ids_len, 0, dtype=int)
    
    # Initialize opacity array
    marker_opacity = figure_data['marker'].get('opacity', 1)
    if isinstance(marker_opacity, (float, int)):
        marker_opacity = np.full(ids_len, marker_opacity, dtype=float)
    elif isinstance(marker_opacity, list):
        if len(marker_opacity) != ids_len:
            marker_opacity = np.full(ids_len, 1.0, dtype=float)
        else:
            marker_opacity = np.array(marker_opacity, dtype=float)
    else:
        marker_opacity = np.full(ids_len, 1.0, dtype=float)
    
    return marker_symbol, marker_opacity


def create_info_notification(message, color="orange", autoClose=2000):
    """
    Create a standardized DMC notification component.
    
    Args:
        message: Notification message text
        color: Notification color (default: "orange")
        autoClose: Auto-close time in ms (default: 2000)
    
    Returns:
        dmc.Notification component
    """
    return dmc.Notification(
        id="my-notification",
        title="Info",
        message=message,
        color=color,
        loading=False,
        action="show",
        autoClose=autoClose,
        position="top-right"
    )


##############
## Callbacks 
##############

# Consolidated loading state management for render buttons
# Note: UMAP and RF plot loading states are now managed by dedicated callbacks
# This callback only handles item plot loading and RF button disabled state
@callback(
    [
        Output('render-item-plot', 'loading', allow_duplicate=True),
        Output('render-rf-plot', 'disabled', allow_duplicate=True),
    ],
    [
        Input('render-item-plot', 'n_clicks'),
        Input('UMAPfig', 'figure'),
        Input('RFPlotFig', 'figure'),
        Input('ItemFig', 'figure'),
    ],
    [State('UMAPGroupsForRF', 'data')],
    prevent_initial_call=True
)
def manage_render_button_loading_states(item_click, umap_fig, rf_fig, item_fig, groups_data):
    """
    Manage item plot loading state and RF button disabled state.
    
    Note: UMAP and RF plot loading states are now managed by dedicated callbacks
    for better immediate feedback.
    
    Special handling for render-rf-plot button:
    - Only enable if UMAPfig contains actual data (not the empty initial plot)
    - Check if figure has data traces and if group count allows comparison
    """
    triggered = ctx.triggered_id
    
    # Default states
    item_loading = False
    rf_disabled = False
    
    # Handle item plot button click (start loading)
    if triggered == 'render-item-plot':
        item_loading = True
    
    # Handle figure updates (stop loading and manage RF button state)
    elif triggered == 'UMAPfig':
        # Check if this is a real UMAP plot (has data) or the empty initial plot
        is_real_umap = False
        if umap_fig and 'data' in umap_fig:
            # Real UMAP plots have at least one trace with actual data points
            for trace in umap_fig['data']:
                if 'x' in trace and len(trace.get('x', [])) > 0:
                    is_real_umap = True
                    break
        
        # Only enable RF button if we have a real UMAP plot
        # Also check group count: disable if exactly 1 group
        if is_real_umap:
            groups_df = pd.DataFrame(groups_data.get("dataframe", []))
            num_groups = len(groups_df)
            rf_disabled = (num_groups == 1)
        else:
            # Empty/initial plot - keep RF button disabled
            rf_disabled = True
            
    elif triggered in ['RFPlotFig', 'ItemFig']:
        # Clear item loading when item figure updates
        item_loading = False
    else:
        return no_update, no_update
    
    return item_loading, rf_disabled

# Callback to disable group informants and item sorting dropdowns when correlation matrix is selected
@callback(
    [Output('items-group-by', 'disabled'),
     Output('items-sort-by', 'disabled')],
    Input('items-plot-mode', 'value')
)
def disable_controls_for_correlation_matrix(plot_mode):
    if plot_mode == "correlation_matrix":
        return True, True  # Disable both dropdowns for correlation matrix
    elif plot_mode == "missing_values_heatmap":
        return True, False  # Disable group-by, enable sort-by for missing values heatmap
    return False, False  # Enable both dropdowns

# Deleted: 8 individual loading state callbacks replaced by consolidated manage_render_button_loading_states callback above

@callback(
    Output('participantsTree', 'checked', allow_duplicate=True),
    [Input('select-all-participants', 'n_clicks'),
     Input('deselect-all-participants', 'n_clicks')],
    prevent_initial_call=True
)
def update_participants_selection(select_all_clicks, deselect_all_clicks):
    button_clicked = ctx.triggered_id
    if button_clicked == 'select-all-participants':
        return Informants['InformantID'].values.tolist()
    elif button_clicked == 'deselect-all-participants':
        return []
    return no_update

# Callback to toggle sidebar visibility
@callback(
    [Output('sidebar-col', 'span'),
     Output('sidebar-col', 'style'),
     Output('main-content-col', 'span'),
     Output('toggle-sidebar-button', 'children')],
    Input('toggle-sidebar-button', 'n_clicks'),
    State('sidebar-col', 'span'),
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, current_span):
    """Toggle sidebar between visible (span=4) and hidden (span=0)"""
    if current_span == 4:
        # Collapse sidebar
        return (
            0,
            {"display": "none"},
            12,
            DashIconify(icon="tabler:layout-sidebar-right-expand", width=20)
        )
    else:
        # Expand sidebar
        return (
            4,
            {
                "height": "calc(100vh - 150px)"
            },
            8,
            DashIconify(icon="tabler:layout-sidebar-right-collapse", width=20)
        )

# Callback to show/hide UI elements based on plot type selection
@callback(
    [Output('item-plot-settings-container', 'style'),
     Output('umap-settings-container', 'style'),
     Output('item-plot-display', 'style'),
     Output('umap-plot-display', 'style'),
     Output('umap-group-buttons', 'style'),
     Output('lasso-selection-buttons', 'style'),
     Output('Umap-add-group', 'disabled', allow_duplicate=True),
     Output('Umap-clear-groups', 'disabled', allow_duplicate=True),
     Output('render-rf-plot', 'disabled', allow_duplicate=True),
     Output('plot-type-description', 'children'),
     Output('export-distance-matrix-container', 'style')],
    Input('grammar-plot-type', 'value'),
    prevent_initial_call=True
)
def toggle_plot_type_ui(plot_type):
    """Show/hide UI elements based on selected plot type"""
    if plot_type == 'item':
        # Show item plot settings and display, hide UMAP
        return (
            {"display": "block"},  # item settings
            {"display": "none"},   # umap settings
            {"display": "block"},  # item plot display
            {"display": "none"},   # umap plot display
            {"display": "none"},   # umap group buttons
            {"display": "none"},   # lasso selection buttons
            True,                  # disable add group
            True,                  # disable clear groups
            True,                  # disable compare groups
            "Compare how different groups rate grammar items",  # description
            {"display": "none"}    # hide distance matrix button
        )
    else:  # plot_type == 'umap'
        # Show UMAP settings and display, hide item plot
        # Buttons are always enabled - errors handled via notifications
        return (
            {"display": "none"},   # item settings
            {"display": "block"},  # umap settings
            {"display": "none"},   # item plot display
            {"display": "block"},  # umap plot display
            {"display": "block"},  # umap group buttons
            {"display": "block"},  # lasso selection buttons
            False,                 # enable add group (errors handled via notifications)
            False,                 # enable clear groups (errors handled via notifications)
            False,                 # enable compare groups (errors handled via notifications)
            "Apply dimensionality reduction (UMAP) to explore how similar participants are to each other based on their grammar ratings",  # description
            {"display": "block"}   # show distance matrix button
        )

# Callback to manage imputed data switch based on plot type
@callback(
    [Output('use-imputed-data-switch', 'disabled'),
     Output('use-imputed-data-switch', 'checked', allow_duplicate=True)],
    Input('grammar-plot-type', 'value'),
    State('use-imputed-data-switch', 'checked'),
    prevent_initial_call=True
)
def manage_imputed_data_switch(plot_type, current_checked):
    """Disable and force to True for UMAP, enable for item plot"""
    if plot_type == 'umap':
        # UMAP always uses imputed data
        return True, True  # disabled=True, checked=True
    else:  # plot_type == 'item'
        # Item plot can toggle - keep current value
        return False, current_checked  # disabled=False, checked=current_checked

# Callback to manage loading state during rendering
@callback(
    [Output('render-loading-overlay', 'visible'),
     Output('render-grammar-plot', 'disabled')],
    [Input('render-grammar-plot', 'n_clicks'),
     Input('ItemFig', 'figure'),
     Input('UMAPfig', 'figure')],
    [State('grammar-plot-type', 'value')],
    prevent_initial_call=True
)
def manage_loading_state(n_clicks, item_fig, umap_fig, plot_type):
    """Show loading overlay and disable button while rendering"""
    triggered_id = ctx.triggered_id
    
    if triggered_id == 'render-grammar-plot':
        # Button was clicked - start loading
        return True, True
    elif triggered_id in ['ItemFig', 'UMAPfig']:
        # Plot was updated - stop loading
        return False, False
    
    return False, False

# Callback to update quick stats panel and badges
@callback(
    [Output('stats-summary', 'children'),
     Output('participants-badge', 'children'),
     Output('items-badge', 'children')],
    [Input('participantsTree', 'checked'),
     Input('grammarItemsTree', 'checked'),
     Input('grammar-type-switch', 'checked')],
    prevent_initial_call=False
)
def update_quick_stats(selected_participants, selected_items, use_pairs):
    """Update the quick stats panel with current selection info and badge counts"""
    n_participants = len(selected_participants) if selected_participants else 0
    n_items = len(selected_items) if selected_items else 0
    total_participants = len(Informants)
    # Update total_items based on the grammar-type-switch state
    total_items = len(GrammarItemsColsPairs) if use_pairs else len(GrammarItemsCols)
    
    # Update badges with format "selected/total"
    participant_badge = f"{n_participants}/{total_participants}"
    items_badge = f"{n_items}/{total_items}"
    
    # If nothing selected, show placeholder
    if not selected_participants or not selected_items:
        return (
            dmc.Text("Select participants and items to begin", size="sm", c="dimmed"),
            participant_badge,
            items_badge
        )
    
    # Get participant stats
    participant_data = Informants[Informants['InformantID'].isin(selected_participants)]
    
    # Variety breakdown - show all varieties
    variety_counts = participant_data['MainVariety'].value_counts()
    variety_text = ", ".join([f"{variety} ({count})" for variety, count in variety_counts.items()])
    
    # Gender breakdown
    gender_counts = participant_data['Gender'].value_counts()
    gender_map = {'f': 'F', 'm': 'M', 'female': 'F', 'male': 'M', 'nb': 'NB', 'non-binary': 'NB'}
    gender_summary = {}
    for gender, count in gender_counts.items():
        mapped = gender_map.get(gender, gender)
        gender_summary[mapped] = gender_summary.get(mapped, 0) + count
    
    gender_text = ", ".join([f"{gender}: {count}" for gender, count in sorted(gender_summary.items())])
    
    # Age stats
    ages = participant_data['Age'].dropna()
    if len(ages) > 0:
        age_text = f"Age: {int(ages.min())}-{int(ages.max())} (median: {int(ages.median())})"
    else:
        age_text = "Age: N/A"
    
    # Build simplified stats display
    stats_display = dmc.Stack([
        dmc.Text(f"👥 {n_participants} participants, 📋 {n_items} items", size="sm", fw=500),
        dmc.Text(variety_text, size="xs", c="dimmed"),
        dmc.Text(f"{gender_text} • {age_text}", size="xs", c="dimmed"),
    ], gap="2px")
    
    return stats_display, participant_badge, items_badge

# Batch selection callbacks
@callback(
    Output('participantsTree', 'checked', allow_duplicate=True),
    [Input('batch-select-enl', 'n_clicks'),
     Input('batch-select-esl', 'n_clicks'),
     Input('batch-select-efl', 'n_clicks'),
     Input('batch-select-age-18-25', 'n_clicks'),
     Input('batch-select-age-26-35', 'n_clicks'),
     Input('batch-select-age-36-50', 'n_clicks'),
     Input('batch-select-age-50plus', 'n_clicks'),
     Input('batch-select-female', 'n_clicks'),
     Input('batch-select-male', 'n_clicks'),
     Input('batch-select-missing-0', 'n_clicks'),
     Input('batch-select-missing-5', 'n_clicks'),
     Input('batch-select-missing-10', 'n_clicks'),
     Input('batch-select-balanced', 'n_clicks'),
     Input('batch-select-balanced-per-variety', 'n_clicks'),
     Input('batch-select-equal-per-variety', 'n_clicks')],
    prevent_initial_call=True
)
def batch_select_participants(*args):
    """Handle batch selection operations"""
    button_id = ctx.triggered_id
    
    if not button_id:
        return no_update
    
    data = Informants.copy()
    
    # Variety type mapping
    def get_variety_type(variety):
        if variety in ["US", "England", "Scotland"]:
            return "ENL"
        elif variety in ["Gibraltar", "Malta", "India", "Puerto Rico"]:
            return "ESL"
        elif variety in ["Slovenia", "Germany", "Sweden", "Spain (Balearic Islands)"]:
            return "EFL"
        return "Other"
    
    if button_id == 'batch-select-enl':
        selected = data[data['MainVariety'].apply(get_variety_type) == "ENL"]['InformantID'].tolist()
    elif button_id == 'batch-select-esl':
        selected = data[data['MainVariety'].apply(get_variety_type) == "ESL"]['InformantID'].tolist()
    elif button_id == 'batch-select-efl':
        selected = data[data['MainVariety'].apply(get_variety_type) == "EFL"]['InformantID'].tolist()
    elif button_id == 'batch-select-age-18-25':
        selected = data[(data['Age'] >= 18) & (data['Age'] <= 25)]['InformantID'].tolist()
    elif button_id == 'batch-select-age-26-35':
        selected = data[(data['Age'] >= 26) & (data['Age'] <= 35)]['InformantID'].tolist()
    elif button_id == 'batch-select-age-36-50':
        selected = data[(data['Age'] >= 36) & (data['Age'] <= 50)]['InformantID'].tolist()
    elif button_id == 'batch-select-age-50plus':
        selected = data[data['Age'] > 50]['InformantID'].tolist()
    elif button_id == 'batch-select-female':
        selected = data[data['Gender'].isin(['f', 'female', 'Female'])]['InformantID'].tolist()
    elif button_id == 'batch-select-male':
        selected = data[data['Gender'].isin(['m', 'male', 'Male'])]['InformantID'].tolist()
    elif button_id == 'batch-select-missing-0':
        # Get participants with 0% missing data
        selected = retrieve_data.getParticipantsByMissingData(max_missing_percent=0)
    elif button_id == 'batch-select-missing-5':
        # Get participants with <5% missing data
        selected = retrieve_data.getParticipantsByMissingData(max_missing_percent=5)
    elif button_id == 'batch-select-missing-10':
        # Get participants with <10% missing data
        selected = retrieve_data.getParticipantsByMissingData(max_missing_percent=10)
    elif button_id == 'batch-select-balanced':
        # Select equal numbers of male and female
        females = data[data['Gender'].isin(['f', 'female', 'Female'])]
        males = data[data['Gender'].isin(['m', 'male', 'Male'])]
        min_count = min(len(females), len(males))
        selected = females.sample(n=min_count, random_state=42)['InformantID'].tolist() + \
                   males.sample(n=min_count, random_state=42)['InformantID'].tolist()
    elif button_id == 'batch-select-balanced-per-variety':
        # Select balanced gender (50/50) for EACH variety
        selected = []
        for variety in data['MainVariety'].unique():
            variety_data = data[data['MainVariety'] == variety]
            females = variety_data[variety_data['Gender'].isin(['f', 'female', 'Female'])]
            males = variety_data[variety_data['Gender'].isin(['m', 'male', 'Male'])]
            min_count = min(len(females), len(males))
            if min_count > 0:
                selected.extend(females.sample(n=min_count, random_state=42)['InformantID'].tolist())
                selected.extend(males.sample(n=min_count, random_state=42)['InformantID'].tolist())
    elif button_id == 'batch-select-equal-per-variety':
        # Select equal-sized subsample per variety
        variety_counts = data['MainVariety'].value_counts()
        min_per_variety = variety_counts.min()
        selected = []
        for variety in data['MainVariety'].unique():
            variety_data = data[data['MainVariety'] == variety]
            sampled = variety_data.sample(n=min(min_per_variety, len(variety_data)), random_state=42)
            selected.extend(sampled['InformantID'].tolist())
    else:
        return no_update
    
    return selected

# Deleted: Export plot callback (feature removed)

@callback(
    Output("download-data", "data"),
    Input("export-data-button", "n_clicks"),
    [State('participantsTree', 'checked'),
     State('grammarItemsTree', 'checked'),
     State('grammar-type-switch', 'checked'),
     State('use-imputed-data-switch', 'checked'),
     State('export-include-sociodemographic-checkbox', 'checked'),
     State('export-include-item-metadata-checkbox', 'checked'),
     State('england-mapping-param', 'data')],
    prevent_initial_call=True
)
def export_data(n_clicks, participants, items, pairs, use_imputed, include_sociodem, include_item_meta, regional_mapping):
    """Export current selection as CSV with optional metadata"""
    if not n_clicks or not participants or not items:
        return no_update
    
    from datetime import datetime
    import pages.data.grammarFunctions as gf
    
    # Get the data
    data = retrieve_data.getGrammarData(
        imputed=use_imputed,
        participants=participants,
        items=items,
        pairs=pairs,
        regional_mapping=regional_mapping
    )
    
    # Remove sensitive columns (privacy protection)
    data = gf.remove_sensitive_columns(data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Identify participant metadata columns
    metadata_cols = ['InformantID', 'Age', 'Gender', 'MainVariety', 'AdditionalVarieties',
                    'YearsLivedInMainVariety', 'RatioMainVariety', 'CountryCollection', 'Year',
                    'Nationality', 'EthnicSelfID', 'CountryID', 'YearsLivedOutside', 
                    'YearsLivedInside', 'YearsLivedOtherEnglish', 'LanguageHome_normalized',
                    'LanguageFather_normalized', 'LanguageMother_normalized', 'Qualifications_normalized',
                    'QualiMother_normalized', 'QualiFather_normalized', 'QualiPartner_normalized',
                    'PrimarySchool', 'SecondarySchool', 'Occupation', 'OccupMother', 'OccupFather',
                    'OccupPartner']
    participant_cols = [col for col in metadata_cols if col in data.columns]
    
    # Filter to only the requested item columns (those in the items parameter)
    # This ensures we only export the items the user selected
    item_cols = [col for col in items if col in data.columns]
    
    if include_item_meta:
        # TRANSPOSED FORMAT: Items as rows, participants as columns
        
        # Get grammar item metadata
        if pairs:
            item_meta = retrieve_data.getGrammarMeta(type="item_pairs")
            item_meta = item_meta.rename(columns={'question_code': 'item_code'})
        else:
            item_meta = retrieve_data.getGrammarMeta(type="all_items")
            item_meta = item_meta.rename(columns={'question_code': 'item_code'})
        
        # Create item metadata dataframe
        item_meta_cols = ['item_code', 'section', 'feature', 'group_finegrained', 
                         'group_ewave', 'feature_ewave', 'item', 'variant_detail']
        
        # Use extracted function for transposition
        result = gf.transpose_grammar_data_with_metadata(
            data, item_cols, item_meta, item_meta_cols, 
            participant_cols, include_sociodem
        )
        
        if include_sociodem:
            base_filename = f"grammar_data_transposed_with_metadata_{timestamp}"
        else:
            base_filename = f"grammar_data_transposed_{timestamp}"
    
    else:
        # STANDARD FORMAT: Participants as rows, items as columns
        if include_sociodem:
            # WIDE: Include all participant metadata columns (limited to metadata_cols)
            result = data[participant_cols + item_cols].copy()
            base_filename = f"grammar_data_wide_{timestamp}"
        else:
            # MINIMAL: Only InformantID and item ratings (no sociodemographic details)
            result = data[['InformantID'] + item_cols].copy()
            base_filename = f"grammar_data_minimal_{timestamp}"
    
    # Create log content and ZIP file using extracted functions
    export_format = "Transposed (items as rows)" if include_item_meta else "Standard (participants as rows)"
    log_content = gf.create_export_log_grammar(
        participants, items, result, use_imputed, pairs,
        regional_mapping, include_sociodem, include_item_meta, export_format
    )
    
    return gf.create_zip_download(base_filename, result.to_csv(index=False), log_content)

# Callback to export distance matrix
@callback(
    Output("download-distance-matrix", "data"),
    Input("export-distance-matrix-button", "n_clicks"),
    [State('participantsTree', 'checked'),
     State('grammarItemsTree', 'checked'),
     State('UMAP_neighbours','value'),
     State('UMAP_mindist','value'),
     State('umap-distance-metric-dropdown', 'value'),
     State('umap-standardize-checkbox', 'checked'),
     State('grammar-type-switch', 'checked'),
     State('use-imputed-data-switch', 'checked'),
     State('informants-store', 'data'),
     State('england-mapping-param', 'data')],
    prevent_initial_call=True
)
def export_distance_matrix(n_clicks, participants, items, n_neighbours, min_dist, 
                          distance_metric, standardize, pairs, use_imputed, informants, regional_mapping):
    """Export distance matrix using UMAP settings as ZIP with log file"""
    if not n_clicks or not participants or not items:
        return no_update
    
    from datetime import datetime
    import pages.data.grammarFunctions as gf
    import zipfile
    import io
    
    # Get the data (always use imputed data for distance matrix calculation)
    data = retrieve_data.getGrammarData(
        imputed=True,  # Distance matrix requires complete data
        participants=participants,
        items=items,
        pairs=pairs,
        regional_mapping=regional_mapping
    )
    
    # Get grammar items columns
    if pairs:
        GrammarItemsCols = retrieve_data.getGrammarItemsCols(type="item_pairs")
    else:
        GrammarItemsCols = retrieve_data.getGrammarItemsCols()
    
    # Get informants dataframe
    informants_df = pd.DataFrame(informants)
    
    # Compute distance matrix using the same settings as UMAP
    distance_df = gf.computeDistanceMatrix(
        grammarData=data,
        GrammarItemsCols=GrammarItemsCols,
        distance_metric=distance_metric,
        standardize=standardize,
        items=items,
        informants=informants_df
    )
    
    # Generate timestamp and create log/ZIP using extracted functions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_content = gf.create_export_log_distance_matrix(
        participants, items, distance_df, distance_metric,
        standardize, pairs, regional_mapping
    )
    
    return gf.create_zip_download(f"distance_matrix_{timestamp}", distance_df.to_csv(), log_content)


# Notify user that distance matrix export is processing
@callback(
    Output("notify-container", "children", allow_duplicate=True),
    Input("export-distance-matrix-button", "n_clicks"),
    [State('participantsTree', 'checked'), State('grammarItemsTree', 'checked')],
    prevent_initial_call=True
)
def notify_distance_matrix_processing(n_clicks, participants, items):
    if not n_clicks:
        return no_update

    if not participants or not items:
        return create_info_notification(
            "Please select participants and items before exporting the distance matrix.",
            color="red",
            autoClose=4000
        )

    return create_info_notification(
        "Processing distance matrix... the download will start shortly.",
        color="blue",
        autoClose=4000
    )

# Clientside export for grammar items table (exports currently visible/filtered rows)
clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) {
            return window.dash_clientside.no_update;
        }
        const api = window.dash_ag_grid && window.dash_ag_grid.getApi('grammar-items-table');
        if (!api) {
            return [{
                action: "show",
                message: "Table not ready yet. Please try again.",
                title: "Export",
                color: "red",
                autoClose: 4000
            }];
        }
        const timestamp = new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 14);
        api.exportDataAsCsv({
            fileName: `grammar_items_${timestamp}.csv`,
            onlySelected: false,
            allColumns: true,
        });
        return [{
            action: "show",
            message: "Exporting currently visible rows...",
            title: "Export",
            color: "blue",
            autoClose: 2000
        }];
    }
    """,
    Output("notify-container", "children", allow_duplicate=True),
    Input("download-grammar-items-table-button", "n_clicks"),
    prevent_initial_call=True
)

# Callback to filter grammar items table based on preset selection
@callback(
    [Output("grammar-items-preset-filter", "value", allow_duplicate=True),
     Output("grammar-items-table", "rowData", allow_duplicate=True)],
    [Input("grammar-items-preset-filter", "value"),
     Input('grammar-type-switch', 'checked')],
    prevent_initial_call='initial_duplicate'
)
def update_grammar_items_preset_filter(selected_presets, pairs):
    """Filter table based on selected presets"""
    from dash import ctx
    
    # Get metadata based on pairs mode
    if pairs:
        meta = retrieve_data.getGrammarMeta(type="item_pairs")
        meta = meta[meta['question_code_written'].notna() & (meta['question_code_written'] != '')].copy()
        meta['combined_item_code'] = meta['question_code'] + ' - ' + meta['question_code_written']
        
        column_mapping = {
            'combined_item_code': 'Item Code',
            'group_finegrained': 'Group',
            'feature_ewave': 'eWAVE',
            'item': 'Item',
            'group_ewave': 'eWAVE Area',
            'feature': 'Feature',
            'section': 'Section'
        }
        selected_columns = ['combined_item_code', 'group_finegrained', 'feature_ewave', 'item', 'group_ewave', 'feature', 'section']
        meta = meta[selected_columns].copy()
        meta = meta.rename(columns=column_mapping)
    else:
        meta = retrieve_data.getGrammarMeta()
        
        column_mapping = {
            'question_code': 'Item Code',
            'group_finegrained': 'Group',
            'feature_ewave': 'eWAVE',
            'item': 'Item',
            'group_ewave': 'eWAVE Area',
            'feature': 'Feature',
            'section': 'Section'
        }
        selected_columns = ['question_code', 'group_finegrained', 'feature_ewave', 'item', 'group_ewave', 'feature', 'section']
        available_columns = [col for col in selected_columns if col in meta.columns]
        meta = meta[available_columns].copy()
        meta = meta.rename(columns=column_mapping)
    
    # Check what triggered the callback
    triggered_id = ctx.triggered_id if ctx.triggered else None
    
    # If pairs mode changed, clear the filter
    if triggered_id == 'grammar-type-switch':
        return [], meta.to_dict("records")
    
    # Filter data if presets are selected
    if selected_presets and len(selected_presets) > 0:
        # Expand presets to get item codes
        item_codes = expand_presets_to_items(selected_presets, item_presets)
        
        if pairs:
            # For pairs mode, we need to filter based on the original question_code
            original_meta = retrieve_data.getGrammarMeta(type="item_pairs")
            original_meta = original_meta[original_meta['question_code_written'].notna() & (original_meta['question_code_written'] != '')].copy()
            
            # Filter to only include items that match the preset item codes
            filtered_original = original_meta[original_meta['question_code'].isin(item_codes)]
            
            # Create combined item codes for matching
            filtered_original['combined_item_code'] = filtered_original['question_code'] + ' - ' + filtered_original['question_code_written']
            
            # Filter meta by these combined codes
            filtered_meta = meta[meta['Item Code'].isin(filtered_original['combined_item_code'].tolist())]
        else:
            # For individual items, filter directly by item codes
            filtered_meta = meta[meta['Item Code'].isin(item_codes)]
        
        return selected_presets, filtered_meta.to_dict("records")
    else:
        return selected_presets, meta.to_dict("records")

# Callback to filter grammar items table to show only checked items from tree
@callback(
    Output("grammar-items-table", "rowData", allow_duplicate=True),
    Input('filter-grammar-items-table', 'n_clicks'),
    [State('grammarItemsTree', 'checked'),
     State('grammar-type-switch', 'checked')],
    prevent_initial_call=True
)
def filter_grammar_items_table(n_clicks, items, pairs):
    """Filter grammar items table to show only selected items from tree"""
    if not n_clicks or not items:
        return no_update
    
    # Get metadata based on pairs mode
    if pairs:
        meta = retrieve_data.getGrammarMeta(type="item_pairs")
        meta = meta[meta['question_code_written'].notna() & (meta['question_code_written'] != '')].copy()
        meta['combined_item_code'] = meta['question_code'] + ' - ' + meta['question_code_written']
        
        column_mapping = {
            'combined_item_code': 'Item Code',
            'group_finegrained': 'Group',
            'feature_ewave': 'eWAVE',
            'item': 'Item',
            'group_ewave': 'eWAVE Area',
            'feature': 'Feature',
            'section': 'Section'
        }
        selected_columns = ['combined_item_code', 'group_finegrained', 'feature_ewave', 'item', 'group_ewave', 'feature', 'section']
        meta = meta[selected_columns].copy()
        meta = meta.rename(columns=column_mapping)
        
        # Convert tree items from "A1-G21" to "A1 - G21" format for matching
        display_items = [item.replace('-', ' - ') for item in items]
        filtered_meta = meta[meta['Item Code'].isin(display_items)]
    else:
        meta = retrieve_data.getGrammarMeta()
        
        column_mapping = {
            'question_code': 'Item Code',
            'group_finegrained': 'Group',
            'feature_ewave': 'eWAVE',
            'item': 'Item',
            'group_ewave': 'eWAVE Area',
            'feature': 'Feature',
            'section': 'Section'
        }
        selected_columns = ['question_code', 'group_finegrained', 'feature_ewave', 'item', 'group_ewave', 'feature', 'section']
        available_columns = [col for col in selected_columns if col in meta.columns]
        meta = meta[available_columns].copy()
        meta = meta.rename(columns=column_mapping)
        
        # Filter by tree items directly
        filtered_meta = meta[meta['Item Code'].isin(items)]
    
    return filtered_meta.to_dict("records")

@callback(
    [Output('grammar-items-table', 'rowData', allow_duplicate=True),
     Output('grammar-items-preset-filter', 'value'),
     Output('grammar-items-quick-filter', 'value')],
    Input('show-all-grammar-items-table', 'n_clicks'),
    State('grammar-type-switch', 'checked'),
    prevent_initial_call=True
)
def clear_all_grammar_items_filters(n_clicks, pairs):
    """Clear all filters from the grammar items table"""
    if not n_clicks:
        return no_update, no_update, no_update
    
    # Get full metadata to reset table
    if pairs:
        meta = retrieve_data.getGrammarMeta(type="item_pairs")
        meta = meta[meta['question_code_written'].notna() & (meta['question_code_written'] != '')].copy()
        meta['combined_item_code'] = meta['question_code'] + ' - ' + meta['question_code_written']
        
        column_mapping = {
            'combined_item_code': 'Item Code',
            'group_finegrained': 'Group',
            'feature_ewave': 'eWAVE',
            'item': 'Item',
            'group_ewave': 'eWAVE Area',
            'feature': 'Feature',
            'section': 'Section'
        }
        selected_columns = ['combined_item_code', 'group_finegrained', 'feature_ewave', 'item', 'group_ewave', 'feature', 'section']
        meta = meta[selected_columns].copy()
        meta = meta.rename(columns=column_mapping)
    else:
        meta = retrieve_data.getGrammarMeta()
        
        column_mapping = {
            'question_code': 'Item Code',
            'group_finegrained': 'Group',
            'feature_ewave': 'eWAVE',
            'item': 'Item',
            'group_ewave': 'eWAVE Area',
            'feature': 'Feature',
            'section': 'Section'
        }
        selected_columns = ['question_code', 'group_finegrained', 'feature_ewave', 'item', 'group_ewave', 'feature', 'section']
        available_columns = [col for col in selected_columns if col in meta.columns]
        meta = meta[available_columns].copy()
        meta = meta.rename(columns=column_mapping)
    
    # Return full data, clear preset filter, and clear quick search
    return meta.to_dict("records"), [], ""

@callback(
    Output("grammar-items-table", "dashGridOptions"),
    Input("grammar-items-quick-filter", "value")
)
def update_grammar_items_quick_filter(filter_value):
    """Update quick filter text for grammar items table"""
    from dash import Patch
    newFilter = Patch()
    newFilter['quickFilterText'] = filter_value
    return newFilter

# Callback to select rows from table in the grammar items tree
@callback(
    Output('grammarItemsTree', 'checked', allow_duplicate=True),
    Input('select-rows-in-tree-button', 'n_clicks'),
    [State('grammar-items-table', 'selectedRows'),
     State('grammarItemsTree', 'checked'),
     State('grammar-type-switch', 'checked')],
    prevent_initial_call=True
)
def select_table_rows_in_tree(n_clicks, selected_rows, current_checked, pairs):
    """Add selected table rows to the grammar items tree selection"""
    if not n_clicks or not selected_rows:
        return no_update
    
    # Get item codes from selected rows based on pairs mode
    if pairs:
        # In pairs mode, extract the item_pair codes from the combined "Item Code" column
        # The format is "A1 - G21", we need to convert it back to "A1-G21" for the tree
        item_codes = []
        for row in selected_rows:
            item_code = row.get('Item Code', '')
            if item_code:
                # Convert "A1 - G21" to "A1-G21"
                item_codes.append(item_code.replace(' - ', '-'))
    else:
        # In individual mode, extract question_code from "Item Code" column
        item_codes = [row.get('Item Code', '') for row in selected_rows if row.get('Item Code')]
    
    # Combine with currently checked items (avoid duplicates)
    current_checked = current_checked or []
    updated_checked = list(set(current_checked + item_codes))
    
    return updated_checked

# Callback to deselect rows from table in the grammar items tree
@callback(
    Output('grammarItemsTree', 'checked', allow_duplicate=True),
    Input('deselect-rows-in-tree-button', 'n_clicks'),
    [State('grammar-items-table', 'selectedRows'),
     State('grammarItemsTree', 'checked'),
     State('grammar-type-switch', 'checked')],
    prevent_initial_call=True
)
def deselect_table_rows_in_tree(n_clicks, selected_rows, current_checked, pairs):
    """Remove selected table rows from the grammar items tree selection"""
    if not n_clicks or not selected_rows:
        return no_update
    
    # Get item codes from selected rows based on pairs mode
    if pairs:
        # In pairs mode, extract the item_pair codes from the combined "Item Code" column
        item_codes = []
        for row in selected_rows:
            item_code = row.get('Item Code', '')
            if item_code:
                # Convert "A1 - G21" to "A1-G21"
                item_codes.append(item_code.replace(' - ', '-'))
    else:
        # In individual mode, extract question_code from "Item Code" column
        item_codes = [row.get('Item Code', '') for row in selected_rows if row.get('Item Code')]
    
    # Remove selected items from currently checked items
    current_checked = current_checked or []
    updated_checked = [item for item in current_checked if item not in item_codes]
    
    return updated_checked

# Clientside callback to copy settings to clipboard (100% client-side for security)
clientside_callback(
    """
    function(n_clicks, participants, items, plot_type, group_by, sort_by, 
             plot_mode, n_neighbors, min_dist, distance_metric, 
             standardize, pairs, use_imputed) {
        if (!n_clicks) {
            return window.dash_clientside.no_update;
        }
        
        // Construct settings object
        const settings = {
            timestamp: new Date().toISOString(),
            plot_type: plot_type,
            participants: participants,
            items: items,
            pairs: pairs,
            use_imputed: use_imputed,
            item_plot_settings: {
                group_by: group_by || "variety",
                sort_by: sort_by || "mean",
                plot_mode: plot_mode || "normal"
            },
            umap_settings: {
                n_neighbors: n_neighbors,
                min_dist: min_dist,
                distance_metric: distance_metric,
                standardize: standardize
            }
        };
        
        // Convert to JSON and encode to base64 (client-side only)
        const jsonString = JSON.stringify(settings, null, 2);
        const base64Settings = btoa(unescape(encodeURIComponent(jsonString)));
        
        // Copy to clipboard
        navigator.clipboard.writeText(base64Settings).then(
            function() {
                // Show success notification
                const notification = {
                    action: "show",
                    message: "Settings copied to clipboard! Share this code with others.",
                    title: "Success",
                    icon: "tabler:check",
                    color: "green",
                    autoClose: 5000
                };
                return [notification];
            },
            function(err) {
                // Show error notification
                const notification = {
                    action: "show",
                    message: "Failed to copy to clipboard. Please check browser permissions.",
                    title: "Error",
                    icon: "tabler:x",
                    color: "red",
                    autoClose: 5000
                };
                return [notification];
            }
        );
        
        return window.dash_clientside.no_update;
    }
    """,
    Output("notify-container", "children", allow_duplicate=True),
    Input("copy-settings-button", "n_clicks"),
    [State('participantsTree', 'checked'),
     State('grammarItemsTree', 'checked'),
     State('grammar-plot-type', 'value'),
     State('items-group-by', 'value'),
     State('items-sort-by', 'value'),
     State('items-plot-mode', 'value'),
     State('UMAP_neighbours', 'value'),
     State('UMAP_mindist', 'value'),
     State('umap-distance-metric-dropdown', 'value'),
     State('umap-standardize-checkbox', 'checked'),
     State('grammar-type-switch', 'checked'),
     State('use-imputed-data-switch', 'checked')],
    prevent_initial_call=True
)

# Open/close paste settings modal
@callback(
    Output("paste-settings-modal", "opened"),
    [Input("paste-settings-button", "n_clicks"),
     Input("cancel-paste-settings", "n_clicks"),
     Input("load-pasted-settings", "n_clicks")],
    State("paste-settings-modal", "opened"),
    prevent_initial_call=True
)
def toggle_paste_modal(open_clicks, cancel_clicks, load_clicks, is_open):
    """Open/close the paste settings modal"""
    # Toggle modal state
    return not is_open

# Load pasted settings with validation (server-side for security)
@callback(
    [Output('participantsTree', 'checked', allow_duplicate=True),
     Output('grammarItemsTree', 'checked', allow_duplicate=True),
     Output('grammar-plot-type', 'value', allow_duplicate=True),
     Output('items-group-by', 'value', allow_duplicate=True),
     Output('items-sort-by', 'value', allow_duplicate=True),
     Output('items-plot-mode', 'value', allow_duplicate=True),
     Output('UMAP_neighbours', 'value', allow_duplicate=True),
     Output('UMAP_mindist', 'value', allow_duplicate=True),
     Output('umap-distance-metric-dropdown', 'value', allow_duplicate=True),
     Output('umap-standardize-checkbox', 'checked', allow_duplicate=True),
     Output('grammar-type-switch', 'checked', allow_duplicate=True),
     Output('use-imputed-data-switch', 'checked', allow_duplicate=True),
     Output("notify-container", "children", allow_duplicate=True),
     Output("paste-settings-textarea", "value")],
    Input("load-pasted-settings", "n_clicks"),
    State("paste-settings-textarea", "value"),
    prevent_initial_call=True
)
def load_pasted_settings(n_clicks, pasted_text):
    """Load and validate pasted settings with security checks"""
    if not n_clicks or not pasted_text:
        return [no_update] * 14
    
    import json
    import base64
    
    try:
        # Security check 1: Size limit (prevent DOS attacks)
        if len(pasted_text) > 100000:  # ~100KB limit
            notification = dmc.Notification(
                title="Error",
                message="Settings string too large. Maximum 100KB allowed.",
                color="red",
                action="show",
                autoClose=5000,
                icon=DashIconify(icon="tabler:x"),
            )
            return [no_update] * 12 + [notification, ""]
        
        # Security check 2: Base64 decode (will fail if not valid base64)
        try:
            decoded_bytes = base64.b64decode(pasted_text)
            decoded_string = decoded_bytes.decode('utf-8')
        except Exception:
            notification = dmc.Notification(
                title="Error",
                message="Invalid settings format. Please paste a valid settings code.",
                color="red",
                action="show",
                autoClose=5000,
                icon=DashIconify(icon="tabler:x"),
            )
            return [no_update] * 12 + [notification, ""]
        
        # Security check 3: JSON parse with size limit
        try:
            settings = json.loads(decoded_string)
        except json.JSONDecodeError:
            notification = dmc.Notification(
                title="Error",
                message="Invalid JSON format in settings.",
                color="red",
                action="show",
                autoClose=5000,
                icon=DashIconify(icon="tabler:x"),
            )
            return [no_update] * 12 + [notification, ""]
        
        # Security check 4: Validate expected structure
        required_keys = {'plot_type', 'participants', 'items', 'pairs', 'use_imputed', 
                        'item_plot_settings', 'umap_settings'}
        if not all(key in settings for key in required_keys):
            notification = dmc.Notification(
                title="Error",
                message="Settings missing required fields.",
                color="red",
                action="show",
                autoClose=5000,
                icon=DashIconify(icon="tabler:x"),
            )
            return [no_update] * 12 + [notification, ""]
        
        # Security check 5: Validate data types
        if not isinstance(settings['participants'], list):
            settings['participants'] = []
        if not isinstance(settings['items'], list):
            settings['items'] = []
        if not isinstance(settings['pairs'], bool):
            settings['pairs'] = False
        if not isinstance(settings['use_imputed'], bool):
            settings['use_imputed'] = False
        
        # Security check 6: Validate nested structures
        item_settings = settings.get('item_plot_settings', {})
        umap_settings = settings.get('umap_settings', {})
        
        if not isinstance(item_settings, dict) or not isinstance(umap_settings, dict):
            notification = dmc.Notification(
                title="Error",
                message="Invalid settings structure.",
                color="red",
                action="show",
                autoClose=5000,
                icon=DashIconify(icon="tabler:x"),
            )
            return [no_update] * 12 + [notification, ""]
        
        # Extract validated values with defaults
        participants = settings['participants']
        items = settings['items']
        plot_type = settings.get('plot_type', 'items')
        group_by = item_settings.get('group_by', 'variety')
        sort_by = item_settings.get('sort_by', 'frequency')
        plot_mode = item_settings.get('plot_mode', 'group')
        n_neighbors = umap_settings.get('n_neighbors', 25)
        min_dist = umap_settings.get('min_dist', 0.1)
        distance_metric = umap_settings.get('distance_metric', 'cosine')
        standardize = umap_settings.get('standardize', False)
        pairs = settings['pairs']
        use_imputed = settings['use_imputed']
        
        # Security check 7: Validate value ranges and types
        if not isinstance(n_neighbors, (int, float)) or n_neighbors < 2 or n_neighbors > 200:
            n_neighbors = 25
        if not isinstance(min_dist, (int, float)) or min_dist < 0 or min_dist > 1:
            min_dist = 0.1
        if distance_metric not in ['cosine', 'euclidean', 'manhattan']:
            distance_metric = 'cosine'
        if plot_type not in ['items', 'umap']:
            plot_type = 'items'
        if group_by not in ['variety', 'gender', 'age_group']:
            group_by = 'variety'
        if sort_by not in ['frequency', 'alphabetical']:
            sort_by = 'frequency'
        if plot_mode not in ['group', 'overlay']:
            plot_mode = 'group'
        
        notification = dmc.Notification(
            title="Success",
            message=f"Settings loaded successfully! ({len(participants)} participants, {len(items)} items)",
            color="green",
            action="show",
            autoClose=5000,
            icon=DashIconify(icon="tabler:check"),
        )
        
        return [
            participants,
            items,
            plot_type,
            group_by,
            sort_by,
            plot_mode,
            n_neighbors,
            min_dist,
            distance_metric,
            standardize,
            pairs,
            use_imputed,
            notification,
            ""  # Clear textarea
        ]
        
    except Exception as e:
        # Catch-all for any unexpected errors
        notification = dmc.Notification(
            title="Error",
            message="Failed to load settings: Unexpected error",
            color="red",
            action="show",
            autoClose=5000,
            icon=DashIconify(icon="tabler:x"),
        )
        return [no_update] * 12 + [notification, ""]

# Settings persistence callbacks
@callback(
    [Output('saved-item-settings', 'data'),
     Output('saved-umap-settings', 'data'),
     Output("notify-container", "children", allow_duplicate=True)],
    Input('save-current-settings', 'n_clicks'),
    [State('grammar-plot-type', 'value'),
     State('participantsTree', 'checked'),
     State('grammarItemsTree', 'checked'),
     State('items-group-by', 'value'),
     State('items-sort-by', 'value'),
     State('items-plot-mode', 'value'),
     State('UMAP_neighbours', 'value'),
     State('UMAP_mindist', 'value'),
     State('umap-distance-metric-dropdown', 'value'),
     State('umap-standardize-checkbox', 'checked'),
     State('grammar-type-switch', 'checked'),
     State('use-imputed-data-switch', 'checked')],
    prevent_initial_call=True
)
def save_settings(n_clicks, plot_type, participants, items, group_by, sort_by,
                  plot_mode, n_neighbors, min_dist, distance_metric, 
                  standardize, pairs, use_imputed):
    """Save current settings to localStorage"""
    if not n_clicks:
        return no_update, no_update, no_update
    
    item_settings = {
        "participants": participants,
        "items": items,
        "group_by": group_by,
        "sort_by": sort_by,
        "plot_mode": plot_mode,
        "pairs": pairs,
        "use_imputed": use_imputed  # Save user's choice for item plots
    }
    
    umap_settings = {
        "participants": participants,
        "items": items,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "distance_metric": distance_metric,
        "standardize": standardize,
        "pairs": pairs
        # Don't save use_imputed for UMAP - it's always True (forced by callback)
    }
    
    notification = dmc.Notification(
        title="Success",
        message=f"{plot_type.upper()} settings saved!",
        color="green",
        action="show",
        autoClose=2000,
        icon=DashIconify(icon="tabler:check"),
    )
    
    return item_settings, umap_settings, notification

@callback(
    [Output('participantsTree', 'checked', allow_duplicate=True),
     Output('grammarItemsTree', 'checked', allow_duplicate=True),
     Output('items-group-by', 'value', allow_duplicate=True),
     Output('items-sort-by', 'value', allow_duplicate=True),
     Output('items-plot-mode', 'value', allow_duplicate=True),
     Output('UMAP_neighbours', 'value', allow_duplicate=True),
     Output('UMAP_mindist', 'value', allow_duplicate=True),
     Output('umap-distance-metric-dropdown', 'value', allow_duplicate=True),
     Output('umap-standardize-checkbox', 'checked', allow_duplicate=True),
     Output("notify-container", "children", allow_duplicate=True)],
    Input('restore-saved-settings', 'n_clicks'),
    [State('grammar-plot-type', 'value'),
     State('saved-item-settings', 'data'),
     State('saved-umap-settings', 'data')],
    prevent_initial_call=True
)
def restore_settings(n_clicks, plot_type, item_settings, umap_settings):
    """Restore settings from localStorage"""
    if not n_clicks:
        return [no_update] * 10
    
    settings = item_settings if plot_type == 'item' else umap_settings
    
    if not settings:
        notification = dmc.Notification(
            title="Info",
            message="No saved settings found",
            color="orange",
            action="show",
            autoClose=2000,
        )
        return [no_update] * 9 + [notification]
    
    notification = dmc.Notification(
        title="Success",
        message=f"{plot_type.upper()} settings restored!",
        color="green",
        action="show",
        autoClose=2000,
        icon=DashIconify(icon="tabler:check"),
    )
    
    if plot_type == 'item':
        return (
            settings.get('participants', no_update),
            settings.get('items', no_update),
            settings.get('group_by', no_update),
            settings.get('sort_by', no_update),
            settings.get('plot_mode', no_update),
            no_update,  # UMAP n_neighbors
            no_update,  # UMAP min_dist
            no_update,  # UMAP distance_metric
            no_update,  # UMAP standardize
            notification
        )
    else:  # UMAP
        return (
            settings.get('participants', no_update),
            settings.get('items', no_update),
            no_update,  # item group_by
            no_update,  # item sort_by
            no_update,  # item plot_mode
            settings.get('n_neighbors', no_update),
            settings.get('min_dist', no_update),
            settings.get('distance_metric', no_update),
            settings.get('standardize', no_update),
            notification
        )

# Unified render callback that routes to item plot or UMAP based on plot type
@callback(
    [Output('render-item-plot', 'n_clicks', allow_duplicate=True),
     Output('render-UMAP-plot', 'n_clicks', allow_duplicate=True)],
    Input('render-grammar-plot', 'n_clicks'),
    [State('grammar-plot-type', 'value'),
     State('render-item-plot', 'n_clicks'),
     State('render-UMAP-plot', 'n_clicks')],
    prevent_initial_call=True
)
def unified_render_button(btn_clicks, plot_type, item_clicks, umap_clicks):
    """Route render button click to appropriate plot type"""
    if btn_clicks is None:
        return no_update, no_update
    
    if plot_type == 'item':
        # Trigger item plot render by incrementing its n_clicks
        return (item_clicks or 0) + 1, no_update
    else:  # plot_type == 'umap'
        # Trigger UMAP plot render by incrementing its n_clicks
        return no_update, (umap_clicks or 0) + 1

# Deleted: Callback to show/hide deselect button based on tab - outer tabs removed, always show button
# The button is now always available since grammarAnalysisC is directly visible

# Callback to enable/disable the lasso selection buttons based on UMAP selection
@callback(
    [Output('deselect-selected-participants', 'disabled'),
     Output('select-only-lasso-participants', 'disabled')],
    Input('UMAPfig', 'selectedData'),
    prevent_initial_call=False
)
def toggle_lasso_buttons_state(selectedData):
    if selectedData and selectedData.get('points') and len(selectedData['points']) > 0:
        return False, False  # Enable both buttons
    else:
        return True, True   # Disable both buttons

# Callback to handle "Select Only Lasso Selection" button click
@callback(
    Output('participantsTree', 'checked', allow_duplicate=True),
    Input('select-only-lasso-participants', 'n_clicks'),
    State('UMAPfig', 'selectedData'),
    prevent_initial_call=True
)
def select_only_lasso_participants(n_clicks, selectedData):
    if n_clicks and selectedData and selectedData.get('points'):
        # Get IDs of selected points
        selected_ids = [point.get('id') for point in selectedData['points']]
        
        # Return only the selected IDs
        if selected_ids:
            return selected_ids
    
    return no_update

# Callback to handle "Deselect selection" button click
@callback(
    Output('participantsTree', 'checked', allow_duplicate=True),
    Input('deselect-selected-participants', 'n_clicks'),
    [State('UMAPfig', 'selectedData'),
     State('participantsTree', 'checked')],
    prevent_initial_call=True
)
def deselect_selected_participants(n_clicks, selectedData, current_checked):
    if n_clicks and selectedData and selectedData.get('points'):
        # Get IDs of selected points (same approach as existing code)
        selected_ids = [point.get('id') for point in selectedData['points']]
        
        # Remove selected IDs from current checked list
        if current_checked and selected_ids:
            updated_checked = [id for id in current_checked if id not in selected_ids]
            return updated_checked
    
    return no_update

@callback(
    Output('grammarItemsTree', 'checked', allow_duplicate=True),
    [Input('select-all-grammar-items', 'n_clicks'),
     Input('deselect-all-grammar-items', 'n_clicks')],
    [State('grammar-type-switch', 'checked')],
    prevent_initial_call=True
)
def update_grammar_items_selection(select_all_clicks, deselect_all_clicks, pairs):
    button_clicked = ctx.triggered_id
    if button_clicked == 'select-all-grammar-items':
        return GrammarItemsColsPairs if pairs else GrammarItemsCols
    elif button_clicked == 'deselect-all-grammar-items':
        return []
    return no_update




# Callback 1: Fast group management (Add Group, Clear Groups)
@callback(	
    [Output('grammar_plots_UMAP', 'data', allow_duplicate=True),
    Output('UMAPgroup', 'data', allow_duplicate=True),
    Output('UMAPGroupsForRF', 'data', allow_duplicate=True),
    Output("notify-container", "children", allow_duplicate=True)],
    [Input('Umap-add-group', 'n_clicks'),
    Input('Umap-clear-groups', 'n_clicks')],
    [State('UMAPfig', 'selectedData'),
    State('grammar_plots_UMAP', 'data'),
    State("UMAPgroup", "data"),
    State('UMAPfig','figure')],
    prevent_initial_call=True
)
def manage_umap_groups(BTNaddgroup, BTNcleargroup, selectedData, figure, data, displayedFigure):
    """Handle fast group operations without re-rendering UMAP"""
    button_clicked = ctx.triggered_id
    
    # Clear all groups
    if button_clicked == 'Umap-clear-groups':
        if (BTNcleargroup == 0 or BTNcleargroup is None):
            return no_update, no_update, no_update, no_update
        newFig = displayedFigure
        for x in range(len(newFig['data'])):
            newFig['data'][x]['marker']['symbol'] = 0
        groupsCache = getColorGroupingsFromFigure(displayedFigure)
        return newFig, 0, groupsCache, no_update
    
    # Add group
    elif button_clicked == 'Umap-add-group':
        if (BTNaddgroup == 0 or BTNaddgroup is None):
            return no_update, no_update, no_update, no_update
        
        # Check if selectedData exists and has points
        if selectedData is None or 'points' not in selectedData or len(selectedData['points']) == 0:
            notification = dmc.Notification(
                id="my-notification",
                title="No Selection",
                message="Please select participants in the UMAP plot using the lasso or box select tool.",
                color="orange",
                loading=False,
                action="show",
                autoClose=4000,
                position="top-right"
            )
            return no_update, no_update, no_update, notification
        
        # Proceed with adding group
        if len(selectedData['points']) > 0:
            curve_numbers = set([point['curveNumber'] for point in selectedData['points']])
            newFig = displayedFigure
            for x in curve_numbers:
                selPoints = newFig['data'][x]['selectedpoints']
                ids_len = len(newFig['data'][x]['ids'])
                
                # Use helper function to initialize marker arrays
                marker_symbol, marker_opacity = initialize_marker_arrays(newFig['data'][x], ids_len)
                
                # Only update selected points that are visible (opacity > 0)
                visible_selPoints = [point for point in selPoints if marker_opacity[point] > 0]
                for point in visible_selPoints:
                    marker_symbol[point] = symbols[data]
                
                # Assign back as lists (Plotly expects lists)
                newFig['data'][x]['marker']['symbol'] = marker_symbol.tolist()
                newFig['data'][x]['marker']['opacity'] = marker_opacity.tolist()
            
            if(data==14):  # reset groups if maximum is reached
                data = 0
            else:
                data = data + 1
            groupsCache = getGroupingsFromFigure(displayedFigure)
            return newFig, data, groupsCache, no_update
    
    return no_update, no_update, no_update, no_update


# Callback 2: Initiate UMAP rendering (set loading states immediately)
@callback(
    [Output('render-UMAP-plot', 'loading', allow_duplicate=True),
     Output('Umap-add-group', 'disabled', allow_duplicate=True),
     Output('Umap-clear-groups', 'disabled', allow_duplicate=True),
     Output('render-rf-plot', 'disabled', allow_duplicate=True),
     Output('umap-render-trigger', 'data'),
     Output("confirm-custom-modal", "opened"),
     Output('umap-view-toggle', 'value', allow_duplicate=True),
     Output('render-loading-overlay', 'visible', allow_duplicate=True),
     Output('render-grammar-plot', 'disabled', allow_duplicate=True),
     Output("notify-container", "children", allow_duplicate=True)],  # Add notification output
    [Input('render-UMAP-plot','n_clicks'),
     Input('modal-ok-button', 'n_clicks'),
     Input('modal-cancel-button', 'n_clicks')],
    [State('grammar_plots_UMAP', 'data'),
     State("grammar_running","data"),
     State("participantsTree", "checked"),
     State("grammarItemsTree", "checked")],
    prevent_initial_call=True
)
def initiate_umap_rendering(BTNrenderPlot, modal_ok, modal_cancel, figure, running_state, participants, items):
    """Set loading states immediately when render is clicked"""
    button_clicked = ctx.triggered_id
    
    if button_clicked == 'modal-cancel-button':
        return no_update, no_update, no_update, no_update, no_update, False, no_update, no_update, no_update, no_update
    
    if button_clicked in ['render-UMAP-plot', 'modal-ok-button']:
        # Validation: Check minimum selections (handle None case)
        if not participants or len(participants) < 10:
            notification = dmc.Notification(
                id="my-notification",
                title="Insufficient Participants",
                message=f"Please select at least 10 participants for UMAP. Currently selected: {len(participants) if participants else 0}",
                color="orange",
                loading=False,
                action="show",
                autoClose=5000,
                position="top-right"
            )
            return False, no_update, no_update, no_update, no_update, False, no_update, False, False, notification
        
        if not items or len(items) < 5:
            notification = dmc.Notification(
                id="my-notification",
                title="Insufficient Grammar Items",
                message=f"Please select at least 5 grammar items for UMAP. Currently selected: {len(items) if items else 0}",
                color="orange",
                loading=False,
                action="show",
                autoClose=5000,
                position="top-right"
            )
            return False, no_update, no_update, no_update, no_update, False, no_update, False, False, notification
        
        # If already running, prevent duplicate trigger
        if running_state:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        # Validation passed - show rendering notification with imputed data info for UMAP
        notification = dmc.Notification(
            id="my-notification",
            title="Info",
            message="Rendering UMAP plot with imputed data (UMAP cannot handle missing data), please wait.",
            color="blue",
            loading=True,
            action="show",
            autoClose=4000,
            position="top-right"
        )
        
        # Immediately set loading states (this happens instantly)
        # Modal is disabled, so always proceed with rendering
        import time
        trigger_data = {'timestamp': time.time(), 'button': button_clicked}
        return True, True, True, True, trigger_data, False, 'umap-plot', no_update, no_update, notification  # Switch to UMAP view
    
    return no_update, no_update, no_update, no_update, no_update, False, no_update, no_update, no_update, no_update


# Callback 3: Background UMAP computation
@callback(	
    [Output('grammar_plots_UMAP', 'data', allow_duplicate=True),
    Output('UMAPgroup', 'data', allow_duplicate=True),
    Output('UMAPparticipants','data'),
    Output('UMAPitems','data'),
    Output('UMAPGroupsForRF', 'data', allow_duplicate=True),
    Output('umap-render-settings', 'data')],  # Store settings for RF plot consistency
    Input('umap-render-trigger', 'data'),
    [State("participantsTree", "checked"),
    State("grammarItemsTree", "checked"),
    State('UMAP_neighbours','value'),
    State('UMAP_mindist','value'),
    State('grammar-items-preset', 'value'), 
    State('umap-distance-metric-dropdown', 'value'), 
    State('umap-standardize-checkbox', 'checked'),
    State('umap-densemap-checkbox', 'checked'),
    State('grammar-type-switch', 'checked'), 
    State('use-imputed-data-switch', 'checked'),
    State('informants-store', 'data'),  # Add informants store
    State('england-mapping-param', 'data')],  # Add england mapping parameter
    prevent_initial_call=True,
    background=True,
    running=[(Output("grammar_running","data"),True,False)]
)
def compute_umap_background(trigger_data, selected_informants, items, n_neighbours, 
                           min_dist, selected_presets, distance_metric, 
                           standardize_participant_ratings, densemap, pairs, use_imputed, informants_data, regional_mapping):
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
    figure = get_cached_umap_plot(
        participants=selected_informants,
        items=items,
        n_neighbours=n_neighbours,
        min_dist=min_dist,
        distance_metric=distance_metric,
        standardize=standardize_participant_ratings,
        densemap=densemap,
        pairs=pairs,
        informants=informants_data,
        regional_mapping=regional_mapping
    )
    groupsCache = getColorGroupingsFromFigure(figure)
    
    # Store the settings used for this UMAP render
    # Note: We store the user's interface choice for use_imputed, even though UMAP itself always uses imputed data.
    # This way, the RF plot (group comparison) can respect what the user actually selected in the interface.
    render_settings = {
        "pairs": pairs,
        "use_imputed": use_imputed  # Store user's interface choice for RF plot
    }
    
    return figure, data, selected_informants, items, groupsCache, render_settings


# Callback 4: Handle UMAP completion (clear loading states)
@callback(
    [Output('render-UMAP-plot', 'loading', allow_duplicate=True),
     Output('Umap-add-group', 'disabled', allow_duplicate=True),
     Output('Umap-clear-groups', 'disabled', allow_duplicate=True),
     Output('render-rf-plot', 'disabled', allow_duplicate=True),
     Output('grammar-analysis-tabs', 'value', allow_duplicate=True)],
    Input('grammar_plots_UMAP', 'data'),
    State('grammar_plots_UMAP', 'data'),
    prevent_initial_call=True
)
def handle_umap_completion(new_figure, figure_state):
    """Clear loading states when UMAP computation completes and activate Plot View tab"""
    # Only respond to actual new plots (not initial state or group modifications)
    if new_figure is None or len(new_figure.get('data', [])) == 0:
        # No real UMAP plot yet, keep buttons in appropriate state
        return False, False, False, False, no_update
    
    # UMAP plot is ready - enable all buttons and activate Plot View tab
    return False, False, False, False, "plot-view"


# Callback to toggle between UMAP and RF plot views
@callback(
    [Output('umap-plot-container', 'style'),
     Output('rf-plot-container', 'style')],
    Input('umap-view-toggle', 'value'),
    prevent_initial_call=True
)
def toggle_umap_rf_view(selected_view):
    """Show/hide UMAP or RF plot based on toggle selection"""
    if selected_view == 'umap-plot':
        return {"display": "block"}, {"display": "none"}
    else:  # rf-plot
        return {"display": "none"}, {"display": "block"}


@callback(
    Output('UMAPfig', 'figure',allow_duplicate=True),
    Input('grammar_plots_UMAP', 'data'),
    prevent_initial_call=True
)
def set_umapfig_from_store(fig_data):
    if fig_data is not None:
        fig = go.Figure(fig_data)
        return fig
    # Return initial empty plot (not a real UMAP plot yet)
    return UMAP_Grammar_initialPlot


# Set loading state immediately when RF plot button is clicked
@callback(
    Output('render-rf-plot', 'loading', allow_duplicate=True),
    Input('render-rf-plot', 'n_clicks'),
    prevent_initial_call=True
)
def set_rf_plot_loading(n_clicks):
    """Set loading state immediately when RF plot render is clicked"""
    if n_clicks:
        return True
    return no_update


# Clear loading state when RF plot finishes OR when error notification is shown
@callback(
    Output('render-rf-plot', 'loading', allow_duplicate=True),
    [Input('RFPlotFig', 'figure'),
     Input('notify-container', 'children')],
    prevent_initial_call=True
)
def clear_rf_plot_loading(figure, notification):
    """Clear loading state when RF plot completes or error occurs"""
    return False


@callback(
    [Output('RFPlotFig','figure'),
     Output('Umap-add-group', 'disabled',allow_duplicate=True),
     Output('Umap-clear-groups', 'disabled',allow_duplicate=True),
     Output('render-rf-plot', 'disabled',allow_duplicate=True),
     Output("notify-container", "children",allow_duplicate=True), 
     Output('umap-view-toggle', 'value', allow_duplicate=True),
     Output('grammar-analysis-tabs', 'value', allow_duplicate=True)],  # Activate Plot View tab
    Input('render-rf-plot','n_clicks'),
    [State('UMAPGroupsForRF','data'),
    State('UMAPitems','data'),
    State('UMAPgroup','data'),
    State('RF_avg_range','value'),
    State('UMAPfig','figure'),
    State('UMAPparticipants','data'),
    State('umap-render-settings', 'data'),  # Use stored settings instead of UI state
    State('rf-use-zscores','checked')],
    prevent_initial_call=True
)
def renderRFPlot(BTN,groups,items,UMAPgroup,value_range,figure,umap_participants,render_settings,use_zscores):
    # Set default value for split_by_variety since checkbox was removed
    split_by_variety = False
    
    # Extract settings from the stored UMAP render settings
    pairs = render_settings.get('pairs', False)
    use_imputed = render_settings.get('use_imputed', True)
    
    button_clicked = ctx.triggered_id
    if button_clicked == 'render-rf-plot' and BTN is not None:
        # Regenerate groups from the current figure state
        # Use getGroupingsFromFigure for lasso groups (with symbols) or getColorGroupingsFromFigure for varieties
        if UMAPgroup != 0:
            # User has created groups with lasso tool - extract symbols
            groups = getGroupingsFromFigure(figure)
        else:
            # No lasso groups - use variety-based coloring
            groups = getColorGroupingsFromFigure(figure)
        
        # Extract only visible participants from the figure (those not hidden via Plotly legend)
        visible_participant_ids = []
        if figure and 'data' in figure:
            for trace in figure['data']:
                # Check if trace is visible (not hidden via legend click)
                # visible can be True, False, or 'legendonly'
                is_visible = trace.get('visible', True)
                if is_visible is True or is_visible == True:  # Explicitly visible
                    if 'ids' in trace:
                        trace_ids = trace['ids']
                        if isinstance(trace_ids, list):
                            visible_participant_ids.extend(trace_ids)
                        else:
                            visible_participant_ids.append(trace_ids)
        
        # Use items from UMAP render (stored), not current tree selection
        items = items  # Already comes from UMAPitems State
        
        # Check if exactly 1 group is selected
        if(UMAPgroup==1):
            notification = dmc.Notification(
                    id="my-notification",
                    title="Invalid Group Selection",
                    message="Please select at least two groups for comparison, or select no groups to compare varieties.",
                    color="orange",
                    loading=False,
                    action="show",
                    autoClose=5000,
                    position="top-right"
            )
            return no_update, False, False, False, notification, no_update, no_update
        df = pd.DataFrame(groups['dataframe'])
        if("id" in df.columns):
            # rename column id to ids
            df.rename(columns={"id":"ids"},inplace=True)
        
        # Filter to only include participants that were in the UMAP render AND are currently visible in the plot
        if umap_participants is not None and umap_participants != []:
            df = df[df['ids'].isin(umap_participants)]
        
        if visible_participant_ids:
            df = df[df['ids'].isin(visible_participant_ids)]
            
        if len(df) == 0:
            notification = dmc.Notification(
                id="my-notification",
                title="Warning",
                message="No visible participants in selected groups. Please adjust your participant filter or unhide traces in the plot.",
                color="orange",
                loading=False,
                action="show",
                autoClose=5000,
                position="top-right"
            )
            return no_update, False, False, False, notification, no_update, no_update

        #data = retrieve_data
        # retrieve grammar data, add items from session cache
        # separate item columns (session cache) and informat columns
        # pass this to trainRF
        # based on trainRF, get plots
        groupcol = df
        if UMAPgroup == 0:
            groupcol['group'] = groupcol['name'] # group by variety.
        else:
            groupcol['group'] = groupcol['symbol'] # group by color

        # Show notification with imputed data info
        if use_imputed:
            notification = dmc.Notification(
                id="my-notification",
                title="Info",
                message="Rendering comparison plot with imputed data. Note: Imputed data might skew the distribution for some items.",
                color="blue",
                loading=True,
                action="show",
                autoClose=4000,
                position="top-right"
            )
        else:
            notification = dmc.Notification(
                id="my-notification",
                title="Info",
                message="Rendering comparison plot, please wait.",
                color="blue",
                loading=True,
                action="show",
                autoClose=2000,
                position="top-right"
            )
        
        # Use lazy data loading for better performance
        data = retrieve_data.getGrammarData(imputed=use_imputed,participants=df['ids'],columns=items, pairs=pairs)
        rf, importanceRatings = trainRF(items,data,datacols=items,groupcol=groupcol,pairs=pairs,use_zscores=use_zscores)


        columns = importanceRatings['item'].to_list()
        data[columns] = data[columns].apply(pd.to_numeric, errors='coerce')
        #data.loc[:,columns] = data.loc[:,columns].astype(int)
        data = data.merge(groupcol, left_on='InformantID', right_on="ids", how="left")
        plotList = []
        df = data.copy(deep=True)
        df = df.melt(id_vars=['group'],value_vars=columns,var_name='item')
        # using polars here for CI calculation ,pandas groupby is insanely slow?
        # Replace polars groupby with pandas groupby:
        df = data.copy()
        df = df.melt(id_vars=['group'], value_vars=columns, var_name='item')
        plotDF = (
            df.groupby(['group', 'item'], observed=True)
            .agg(count=('value', 'count'), mean=('value', 'mean'), std=('value', 'std'))
            .reset_index()
        )
        # Calculate confidence intervals efficiently:
        plotDF['ci'] = 1.96 * (plotDF['std'] / plotDF['count']**0.5)
        plotDF['lower_ci'] = plotDF['ci'] # plotting function expects difference, not absolute values
        plotDF['upper_ci'] = plotDF['ci']
        plotDF = plotDF.merge(importanceRatings, on='item', how='left')

        # to do: merge meta info here for hoverinfo in plot
        RFPlot = get_cached_rf_plot(plotDF, importanceRatings, value_range, pairs=pairs, split_by_variety=split_by_variety)
        return RFPlot, False, False, False, notification, 'rf-plot', "plot-view"
    return no_update, no_update, no_update, no_update, no_update, no_update, no_update

# apply filter to participant selection tree
@callback(
    [Output('participantsTree', 'checked',allow_duplicate=True), Output("notify-container", "children", allow_duplicate=True)],
    [Input('apply-grammar-filters', 'n_clicks')],
    [
        State('checkbox-grammar-filter-gender', 'value'),
        State('rangeslider-grammar-filter-age', 'value'),
        State('checkbox-grammar-filter-age-missing', 'value'),
        State('rangeslider-grammar-filter-ratio', 'value'),
        State('checkbox-grammar-filter-ratio-missing', 'value'),
        State('multiselect-grammar-filter-mainvariety', 'value')
    ],
    prevent_initial_call=True
)
def updateParticipants(applyButton, genderFilter, ageFilter, ageMissing, ratioFilter, ratioMissing, mainVarietyFilter):
    button_clicked = ctx.triggered_id
    if button_clicked == 'apply-grammar-filters' and applyButton is not None:
        data = Informants.loc[:, :]  # shallow copy is enough for filtering
        # Filter by main variety
        if mainVarietyFilter:
            data = data.loc[data['MainVariety'].isin(mainVarietyFilter), :]

        # Gender filter
        if "nb" in genderFilter:
            genderFilter = genderFilter + ["non-binary"]
        gender_na = "NA" in genderFilter
        if gender_na:
            genderFilter = genderFilter + ["NA", "na"]

        # Prepare masks for each filter
        # Age filter
        if ageMissing:
            age_mask = (data['Age'].between(ageFilter[0], ageFilter[1])) | (data['Age'].isna())
        else:
            age_mask = data['Age'].between(ageFilter[0], ageFilter[1])

        # Ratio filter
        if ratioMissing:
            ratio_mask = (data['RatioMainVariety'].between(ratioFilter[0]/100, ratioFilter[1]/100)) | (data['RatioMainVariety'].isna())
        else:
            ratio_mask = data['RatioMainVariety'].between(ratioFilter[0]/100, ratioFilter[1]/100)

        # Gender filter
        if gender_na:
            gender_mask = (data['Gender'].isin(genderFilter)) | (data['Gender'].isna())
        else:
            gender_mask = data['Gender'].isin(genderFilter)

        # Combine all masks
        combined_mask = age_mask & ratio_mask & gender_mask
        data = data.loc[combined_mask, :]

        if len(data) > 0:
            return data['InformantID'].to_list(), no_update
        else:
            notification = dmc.Notification(
                id="my-notification",
                title="Info",
                message="Your filters resulted in an empty selection, please adjust the filters.",
                color="orange",
                loading=True,
                action="show",
                autoClose=3000,
                position="top-right"
            )
            return no_update, notification
    return no_update, no_update

# apply filter to Grammar selection tree
@callback(
    Output('grammarItemsTree', 'checked', allow_duplicate=True),
    [Input('grammar_toggle_written_only', 'n_clicks'),Input('grammar_toggle_currency','n_clicks'),Input('grammar_deselect_problematic','n_clicks')], #,Input('grammar_top20','n_clicks')
    State("grammarItemsTree", "checked"),
    prevent_initial_call=True
)
def updateGrammarItemsTree(wo_button,curr_button,prob_button,itemTree):
    button_clicked = ctx.triggered_id # which buttonw as clicked
    # reset all groups without rerendering the plot

    if (button_clicked == 'grammar_toggle_written_only'):
        items = grammarMeta.copy(deep=True)
        wo_items = items.loc[items['also_in_question'] == '','question_code'].to_list()
        if (len(set(itemTree).intersection(wo_items)) < len(wo_items)):
            items = wo_items + itemTree
            items = list(set(items))
        else:
            items = list(set(itemTree)-set(wo_items))
        return items
    elif (button_clicked == 'grammar_toggle_currency'):
        unit_items = ['C21','H4','A4','C14','J23','M19']
        # if one of the above is already in list, select all
        # if all are selected, deselect all
        if (len(set(itemTree).intersection(unit_items)) < len(unit_items)):
            items = unit_items + itemTree
            items = list(set(items))
        else:
            items = list(set(itemTree)-set(unit_items))
        return items
    elif (button_clicked == 'grammar_deselect_problematic'):
        problematic_items = ['M19', 'J23', 'C14', 'A4', 'E22', 'D12', 'E6','C21','H4']
        # Remove problematic items from current selection
        items = list(set(itemTree) - set(problematic_items))
        return items
    elif (button_clicked == 'grammar_top20'):
        items = ['A20', 'A23', 'A4', 'A8', 'B14', 'C13', 'D12', 'D22', 'D4', 'E19', 'E22', 'F2', 'F22', 'K5', 'L9']
        return items
    else:
        return no_update


#@callback(Output("burger-state", "children"), Input("burger-button", "opened"))
#def is_open(opened):
#    return str(opened)

@callback(
    [Output('ItemFig','figure'),
     Output('render-loading-overlay', 'visible', allow_duplicate=True),
     Output('render-grammar-plot', 'disabled', allow_duplicate=True),
     Output("notify-container", "children",allow_duplicate=True),
     Output('grammar-analysis-tabs', 'value', allow_duplicate=True)],
    Input('render-item-plot','n_clicks'),
    [State('participantsTree','checked'),State('grammarItemsTree','checked'),State('items-group-by','value'),State('items-sort-by','value'),State('items-plot-mode','value'),State('grammar-type-switch','checked'),State('use-imputed-data-switch', 'checked'),State('england-mapping-param', 'data')],
    prevent_initial_call=True
)
def renderItemPlot(BTN,informants,items,groupby,sortby,plot_mode,pairs,use_imputed,regional_mapping):
    button_clicked = ctx.triggered_id
    if button_clicked == 'render-item-plot' and BTN is not None:
        # Validation: Check minimum selections (handle None case)
        if not informants or len(informants) < 10:
            notification = dmc.Notification(
                id="my-notification",
                title="Insufficient Participants",
                message=f"Please select at least 10 participants. Currently selected: {len(informants) if informants else 0}",
                color="orange",
                loading=False,
                action="show",
                autoClose=5000,
                position="top-right"
            )
            return no_update, False, False, notification, no_update
        
        if not items or len(items) < 1:
            notification = dmc.Notification(
                id="my-notification",
                title="No Items Selected",
                message="Please select at least 1 grammar item.",
                color="orange",
                loading=False,
                action="show",
                autoClose=5000,
                position="top-right"
            )
            return no_update, False, False, notification, no_update
        
        # Validation passed - show rendering notification with imputed data info if applicable
        if use_imputed:
            notification = dmc.Notification(
                id="my-notification",
                title="Info",
                message="Rendering Item plot with imputed data. Note: Imputed data might skew the distribution for some items.",
                color="blue",
                loading=True,
                action="show",
                autoClose=4000,
                position="top-right"
            )
        else:
            notification = dmc.Notification(
                id="my-notification",
                title="Info",
                message="Rendering Item plot, please wait.",
                color="blue",
                loading=True,
                action="show",
                autoClose=2000,
                position="top-right"
            )

        # Use lazy data loading - only get data when needed
        if use_imputed:
            if pairs:
                data_source = get_grammar_data_pairs_cached(regional_mapping=regional_mapping)
            else:
                data_source = get_grammar_data_cached(regional_mapping=regional_mapping)
        else:
            if pairs:
                data_source = get_grammar_data_pairs_raw_cached(regional_mapping=regional_mapping)
            else:
                data_source = get_grammar_data_raw_cached(regional_mapping=regional_mapping)
        
        # to do: merge meta info here for hoverinfo in plot
        # Check if split_by_variety mode is selected
        split_by_variety = (plot_mode == "split_by_variety")
        itemPlot = getItemPlot(informants, items,groupby=groupby,sortby=sortby,pairs=pairs,use_imputed=use_imputed,plot_mode=plot_mode,split_by_variety=split_by_variety,regional_mapping=regional_mapping)
        return itemPlot, no_update, no_update, notification, "plot-view"
    return no_update, no_update, no_update, no_update, no_update

# Save rendered plots to session storage for persistence
@callback(
    Output('last-rendered-item-plot', 'data'),
    Input('ItemFig', 'figure'),
    prevent_initial_call=True
)
def save_item_plot(fig):
    """Save item plot to session storage"""
    return fig

@callback(
    Output('last-rendered-umap-plot', 'data'),
    Input('UMAPfig', 'figure'),
    prevent_initial_call=True
)
def save_umap_plot(fig):
    """Save UMAP plot to session storage"""
    return fig

# Restore plots when switching plot types
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
    # Don't override if no saved plot
    if plot_type == 'item' and saved_item_plot:
        return saved_item_plot, no_update
    elif plot_type == 'umap' and saved_umap_plot:
        return no_update, saved_umap_plot
    return no_update, no_update

@callback(
    [Output('grammarItemsTree', 'checked',allow_duplicate=True),Output('grammar-items-preset', 'value',allow_duplicate=True)],
    [Input('grammarItemsTree', 'checked'),Input('grammar-items-preset', 'value')],
    [State('grammarItemsTree', 'checked')],
    prevent_initial_call=True
)
def update_item_tree_based_on_select(checked_items,selected_presets, current_items):
    # check which input triggered the callback
    button_clicked = ctx.triggered_id
    
    # if it was grammarItemsTree, clear the preset multiselect
    if button_clicked == 'grammarItemsTree':
        # If the tree is changed manually, clear the preset selection
        return checked_items, []
    
    # If no presets selected, do nothing
    if not selected_presets:
        return current_items, no_update

    # Expand all selected presets to a union of item codes
    selected_list = selected_presets if isinstance(selected_presets, (list, tuple)) else [selected_presets]
    union_items = expand_presets_to_items(selected_list, item_presets)
    # Update the tree checked items to reflect the union of preset items
    return union_items, no_update


# Auto-update sociodemographic plots when tab is selected (with smart caching)
@callback(
    [
        Output('AgeGenderPlotG','figure'),
        Output('MainVarietiesPlotG','figure'),
        Output('NationalityPlotG','figure'),
        Output('EIDPlotG','figure'),
        Output('CIDPlotG','figure'),
        Output('LanguagesHomePlotG','figure'),
        Output('LanguagesMotherPlotG','figure'),
        Output('LanguagesFatherPlotG','figure'),
        Output('PrimarySchoolPlotG','figure'),
        Output('SecondarySchoolPlotG','figure'),
        Output('QualiPlotG','figure'),
        Output('YLOPlotG','figure'),
        Output('YLOEPlotG','figure'),
        Output('RatioMainVarietyPlotG','figure'),
        Output('informants-table', 'rowData'),
        Output('last-sociodemographic-settings', 'data')
    ],
    [
        Input('grammar-analysis-tabs', 'value'),  # Trigger when tab changes
        Input('participantsTree', 'checked')  # Also trigger when participant selection changes
    ],
    [State('last-sociodemographic-settings', 'data'),
     State('informants-store', 'data')],
    prevent_initial_call=False
)
def auto_update_sociodemographic_plots(active_tab, selected_participants, last_settings, informants_data):
    """
    Automatically update sociodemographic plots when:
    1. Switching to the Sociodemographic Details tab
    2. Participant selection has changed since last render
    
    Uses smart caching to avoid unnecessary re-renders.
    """
    # Only update when on sociodemographic details tab
    if active_tab != 'sociodemographic-details':
        raise PreventUpdate
    
    # If no participants selected, show empty plots
    if not selected_participants:
        # Create empty figure for all plots
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="simple_white"
        )
        return (
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            empty_fig, [],
            {'participants': []}  # Update the cache
        )
    
    # Create current settings hash
    current_settings = {
        'participants': sorted(selected_participants)
    }
    
    # Check if settings have changed (smart caching)
    if last_settings == current_settings:
        # Settings haven't changed, no need to re-render
        raise PreventUpdate
    
    # Settings have changed, render plots
    # Use informants from store (respects regional mapping)
    informants_df = pd.DataFrame(informants_data)
    informants = informants_df.loc[informants_df['InformantID'].isin(selected_participants), :]
    
    AgeGenderPlot = getAgeGenderPlot(informants)
    MainVarietiesPlot = getMainVarietiesPlot(informants)
    NationalityPlot = getCategoryHistogramPlot(informants, "Nationality", True, "")
    EIDPlot = getCategoryHistogramPlot(informants, "EthnicSelfID", True, "")
    CIDPlot = getCategoryHistogramPlot(informants, "CountryID", True, ",")
    LanguagesHomePlot = getCategoryHistogramPlot(informants, "LanguageHome_normalized", True, ",")
    LanguagesMotherPlot = getCategoryHistogramPlot(informants, "LanguageMother_normalized", True, ",")
    LanguagesFatherPlot = getCategoryHistogramPlot(informants, "LanguageFather_normalized", True, ",")
    PrimarySchoolPlot = getCategoryHistogramPlot(informants, "PrimarySchool", True)
    SecondarySchoolPlot = getCategoryHistogramPlot(informants, "SecondarySchool", True)
    QualiPlot = getCategoryHistogramPlot(informants, "Qualifications", True)
    YLOPlot = getFloatHistogramPlot(informants, "YearsLivedOutside")
    YLOEPlot = getFloatHistogramPlot(informants, "YearsLivedOtherEnglish")
    RatioMainVarietyPlot = getFloatHistogramPlot(informants, "RatioMainVariety")
    
    return (
        AgeGenderPlot, MainVarietiesPlot, NationalityPlot, EIDPlot, CIDPlot,
        LanguagesHomePlot, LanguagesMotherPlot, LanguagesFatherPlot,
        PrimarySchoolPlot, SecondarySchoolPlot, QualiPlot, YLOPlot, YLOEPlot,
        RatioMainVarietyPlot, informants.to_dict("records"),
        current_settings  # Update the cache
    )

# Callback to toggle between table and plot views
@callback(
    [Output("informants-table-view", "style"),
     Output("informants-plot-view", "style")],
    Input("informants-view-toggle", "value"),
    prevent_initial_call=False
)
def toggle_informants_view(view_mode):
    """Toggle between table and plot views"""
    if view_mode == "table":
        return {"display": "flex", "flex-direction": "column", "height" : "calc(-285px + 100vh)"}, {"display": "none"}
    else:
        return {"display": "none"}, {"display": "flex", "flex-direction": "column", "height": "calc(-285px + 100vh)"}

# Callback to update table columns based on checkbox selection
@callback(
    Output("informants-table", "columnDefs"),
    Input("informants-columns-checkbox", "value"),
    prevent_initial_call=False
)
def update_informants_table_columns(selected_columns):
    """Update visible columns in informants table based on checkbox selection"""
    if not selected_columns:
        selected_columns = []
    
    # Always include InformantID as the first column
    columns_to_show = ['InformantID'] + [col for col in selected_columns if col != 'InformantID']
    
    # Filter to only columns that exist in the data
    available_columns = [col for col in columns_to_show if col in Informants.columns]
    
    columnDefs = [
        {
            "field": col,
            "headerName": col.replace("ID", " ID").replace("_", " ").replace("Occup", "Occupation").replace("Quali", "Qualification").replace("Ethnic", "Ethnic "),
            "filter": "agTextColumnFilter",
            "sortable": True,
            "resizable": True,
            "minWidth": 120 if "Language" in col or "Variety" in col else 100,
            "flex": 1,
            "cellStyle": {"textAlign": "left"},
            "headerTooltip": f"Click to sort by {col}. Use filter below to search."
        }
        for col in available_columns
    ]
    
    return columnDefs

# Callback for Select All/Deselect All column buttons
@callback(
    Output("informants-columns-checkbox", "value"),
    [Input("select-all-columns-button", "n_clicks"),
     Input("deselect-all-columns-button", "n_clicks")],
    State("informants-columns-checkbox", "value"),
    prevent_initial_call=True
)
def toggle_all_columns(select_clicks, deselect_clicks, current_value):
    """Select or deselect all columns"""
    ctx_triggered = ctx.triggered_id
    
    all_columns = ['InformantID', 'Age', 'Gender', 'MainVariety', 'AdditionalVarieties',
                  'YearsLivedInMainVariety', 'RatioMainVariety', 'CountryCollection', 'Year',
                  'Nationality', 'EthnicSelfID', 'CountryID', 'YearsLivedOutside', 
                  'YearsLivedInside', 'YearsLivedOtherEnglish', 'LanguageHome_normalized',
                  'LanguageFather_normalized', 'LanguageMother_normalized', 'Qualifications_normalized',
                  'QualiMother_normalized', 'QualiFather_normalized', 'QualiPartner_normalized',
                  'PrimarySchool', 'SecondarySchool', 'Occupation', 'OccupMother', 'OccupFather',
                  'OccupPartner']
    
    # Filter to only columns that exist in the data
    available_columns = [col for col in all_columns if col in Informants.columns]
    
    if ctx_triggered == "select-all-columns-button":
        return available_columns
    elif ctx_triggered == "deselect-all-columns-button":
        return []
    
    return current_value

# Clientside callback to download informants table data
clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) {
            return window.dash_clientside.no_update;
        }
        const api = window.dash_ag_grid && window.dash_ag_grid.getApi('informants-table');
        if (!api) {
            return window.dash_clientside.no_update;
        }
        const timestamp = new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 14);
        api.exportDataAsCsv({
            fileName: `informants_table_${timestamp}.csv`,
            onlySelected: false,
            allColumns: true,
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("download-informants-table-button", "n_clicks"),
    Input("download-informants-table-button", "n_clicks"),
    prevent_initial_call=True
)

# Deleted: Commented-out clientside callback for filtering by hoverinfo (unused)

clientside_callback(
    """
    function(colorMode, fig, informants) {
    if (!fig || !fig.data || !informants) {
        return window.dash_clientside.no_update;
    }

    // Build info map
    var infoMap = {};
    informants.forEach(function(row) {
        infoMap[row.InformantID] = row;
    });

    function getVarietyType(mainVariety) {
        if (["US","England","Scotland"].includes(mainVariety)) return "ENL";
        if (["Gibraltar","Malta","India","Puerto Rico"].includes(mainVariety)) return "ESL";
        if (["Slovenia","Germany","Sweden","Spain (Balearic Islands)"].includes(mainVariety)) return "EFL";
        return "Other";
    }

    var varietyColorMap = {
        "England": "#1f77b4",
        "Scotland": "#ff7f0e", 
        "US": "#2ca02c",
        "Gibraltar": "#d62728",
        "Malta": "#9467bd",
        "India": "#8c564b",
        "Puerto Rico": "#e377c2",
        "Slovenia": "#7f7f7f",
        "Germany": "#bcbd22",
        "Sweden": "#17becf",
        "Other": "#c49c94"
    };

    // Only copy traces that need updating
    let changed = false;
    let newFig = {...fig, data: fig.data.map(trace => ({...trace}))};

    newFig.data.forEach(function(trace, i) {
        if (trace.ids && Array.isArray(trace.ids)) {
            let colors = trace.ids.map(function(id) {
                var info = infoMap[id];
                if (!info) return "#cccccc";
                if (colorMode === "Variety") {
                    return varietyColorMap[info.MainVariety] || "#cccccc";
                } else if (colorMode === "Variety type") {
                    var vType = getVarietyType(info.MainVariety);
                    if (vType === "ENL") return "#1f77b4";
                    if (vType === "ESL") return "#ff7f0e";
                    if (vType === "EFL") return "#2ca02c";
                    return "#cccccc";
                } else if (colorMode === "Gender") {
                    if (info.Gender === "Female" || info.Gender === "female") return "#e377c2";
                    if (info.Gender === "Male" || info.Gender === "male") return "#1f77b4"; 
                    if (info.Gender === "Non-binary" || info.Gender === "non-binary") return "#2ca02c";
                    return "#cccccc";
                }
                return "#cccccc";
            });
            // Only update if changed
            if (!trace.marker) trace.marker = {};
            if (JSON.stringify(trace.marker.color) !== JSON.stringify(colors)) {
                trace.marker.color = colors;
                changed = true;
            }
        }
    });

    if (!changed) {
        return window.dash_clientside.no_update;
    }
    return newFig;
    }
    """,
    Output('UMAPfig', 'figure', allow_duplicate=True),
    [Input('umap-color-dropdown', 'value')],
    [State('UMAPfig', 'figure'),State('informants-store', 'data')],
    prevent_initial_call=True
)


def generate_umap_plots_for_all_presets(grammarData, GrammarItemsCols, Informants, item_presets, output_dir="pages/data/umap_presets"):
    """
    Iterates over all presets in item_presets, creates UMAP plots for each (with all informants selected),
    and saves each plot as a .pkl file in the specified output directory.
    Filenames are generated using the same hash-based scheme as in getUMAPplot.
    """
    import os
    import hashlib
    from pages.data.grammarFunctions import getUMAPplot

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_informants = Informants['InformantID'].tolist()
    def _hash_list(lst):
        return hashlib.md5(str(sorted(lst)).encode()).hexdigest() if lst else "all"

    n_neighbours = 25
    min_dist = 0.1
    regional_mapping = False  # Generate presets for default (non-regional) view

    for preset in item_presets:
        preset_items = preset['value']
        if not preset_items:
            continue  # skip empty presets
        participants_hash = _hash_list(all_informants)
        items_hash = _hash_list(preset_items)
        # Include regional_mapping in filename
        regional_suffix = "_regional" if regional_mapping else ""
        preset_filename = f"umap_{participants_hash}_{items_hash}_{n_neighbours}_{min_dist}{regional_suffix}.pkl"
        filename = os.path.join(output_dir, preset_filename)
        fig = getUMAPplot(
            grammarData=grammarData,
            GrammarItemsCols=GrammarItemsCols,
            informants=Informants,
            items=preset_items,
            n_neighbours=n_neighbours,
            min_dist=min_dist,
            standardize=False,
            pairs=False
        )
       

        with open(filename, "wb") as f:
            pickle.dump(fig, f)
        print(f"Saved: {filename}")

# Example usage:
# generate_umap_plots_for_all_presets(grammarData, GrammarItemsCols, Informants, item_presets)


@callback(
    Output("confirm-custom-modal", "opened", allow_duplicate=True),
    Input("modal-ok-button", "n_clicks"),
    prevent_initial_call=True
)
def close_modal_on_ok(n_clicks):
    # Close the modal when OK is clicked
    return False

# Add Leiden clustering callback
@callback(
    [Output('leiden-cluster-fig', 'figure'),
     Output('cluster-stats-table', 'children'),
     Output('leiden-cluster-data', 'data'),
     Output('render-leiden-plot', 'loading'),
     Output("notify-container", "children", allow_duplicate=True)],
    Input('render-leiden-plot', 'n_clicks'),
    [State('participantsTree', 'checked'),
     State('grammarItemsTree', 'checked'),
     State('leiden-resolution', 'value'),
     State('similarity-threshold', 'value'),
     State('leiden-color-dropdown', 'value'),
     State('leiden-pca-switch', 'checked'),
     State('leiden-pca-components', 'value'),
     State('informants-store', 'data')],  # Add informants store
    prevent_initial_call=True
)
def run_leiden_clustering(n_clicks, selected_informants, selected_items, resolution, similarity_threshold, 
                         color_by, apply_pca, n_components, informants_data):
    if n_clicks is None:
        return no_update, no_update, no_update, False, no_update
    
    try:
        # Convert informants data from store
        informants_df = pd.DataFrame(informants_data) if informants_data else Informants
        
        # Filter data based on selections
        if selected_informants == ['informants']:
            selected_informants = informants_df['InformantID'].tolist()
        if selected_items == ['grammaritems']:
            selected_items = GrammarItemsCols        
        # Get filtered data
        data = retrieve_data.getGrammarData(imputed=True, participants=selected_informants, columns=selected_items)
        informant_data = informants_df[informants_df['InformantID'].isin(selected_informants)].copy()
        
        # Prepare feature matrix
        feature_cols = [col for col in data.columns if col in selected_items]
        X = data[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0)
        
        # Apply PCA if requested
        if apply_pca:
            # Check if n_components is greater than the number of features or samples
            max_components = min(X.shape[0], X.shape[1])
            if n_components > max_components:
                n_components = max_components
                notification = dmc.Notification(
                    title="Warning",
                    message=f"PCA components reduced to {max_components} (maximum possible with current data)",
                    color="yellow",
                    action="show",
                    autoClose=5000,
                    position="top-right"
                )
                return no_update, notification
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=n_components)
            X = pca.fit_transform(X_scaled)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(X)
        # Use sparse adjacency if threshold is high and N is large
        if similarity_threshold > 0.5 and similarity_matrix.shape[0] > 500:
            adjacency_matrix = (similarity_matrix >= similarity_threshold).astype(int)
            np.fill_diagonal(adjacency_matrix, 0)
            adjacency_sparse = csr_matrix(adjacency_matrix)
            # Convert to igraph edge list for memory efficiency
            sources, targets = adjacency_sparse.nonzero()
            edges = list(zip(sources.tolist(), targets.tolist()))
            G = ig.Graph(edges=edges, directed=False)
        else:
            adjacency_matrix = (similarity_matrix >= similarity_threshold).astype(int)
            np.fill_diagonal(adjacency_matrix, 0)
            adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
            G = ig.Graph.Adjacency(adjacency_matrix.tolist(), mode='undirected')
        
        # Run Leiden clustering
        clusters = G.community_leiden(resolution_parameter=resolution)
        cluster_labels = clusters.membership
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'InformantID': data['InformantID'].values,
            'cluster': cluster_labels
        })
        
        # Merge with informant data
        results_df = results_df.merge(informant_data, on='InformantID', how='left')
        
        # Create visualization using PCA for 2D projection
        if not apply_pca or X.shape[1] != 2:
            pca_vis = PCA(n_components=2)
            X_vis = pca_vis.fit_transform(StandardScaler().fit_transform(X))
        else:
            X_vis = X
        
        results_df['PC1'] = X_vis[:, 0]
        results_df['PC2'] = X_vis[:, 1]
        
        # Create the plot
        fig = go.Figure()
        
        if color_by == 'cluster':
            color_col = 'cluster'
            color_discrete_map = None
        elif color_by == 'variety':
            color_col = 'MainVariety'
            color_discrete_map = None
        elif color_by == 'gender':
            color_col = 'Gender'
            color_discrete_map = None
        
        # Add traces for each group
        for group in results_df[color_col].unique():
            if pd.isna(group):
                continue
            group_data = results_df[results_df[color_col] == group]
            
            fig.add_trace(go.Scatter(
                x=group_data['PC1'],
                y=group_data['PC2'],
                mode='markers',
                name=str(group),
                text=group_data['InformantID'],
                hovertemplate='<b>%{text}</b><br>' +
                             'PC1: %{x:.2f}<br>' +
                             'PC2: %{y:.2f}<br>' +
                             f'{color_col}: {group}<br>' +
                             '<extra></extra>',
                marker=dict(size=8, opacity=0.7)
            ))
        
        fig.update_layout(
            title=f'Leiden Clustering Results (Resolution: {resolution})',
            xaxis_title='PC1' if apply_pca else 'PCA Component 1',
            yaxis_title='PC2' if apply_pca else 'PCA Component 2',
            template='simple_white',
            height=600
        )
        
        # Create cluster statistics table
        cluster_stats = results_df.groupby('cluster', observed=True).agg({
            'InformantID': 'count',
            'MainVariety': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed',
            'Gender': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
        }).rename(columns={'InformantID': 'Size'}).reset_index()
        
        stats_table = dag.AgGrid(
            rowData=cluster_stats.to_dict("records"),
            columnDefs=[{"field": i, "sortable": True, "filter": True, "resizable": True} for i in cluster_stats.columns],
            className="ag-theme-quartz compact",
            columnSize="autoSize",
            style={"height": "300px"}
        )
        
        notification = dmc.Notification(
            title="Success",
            message=f"Leiden clustering completed. Found {len(cluster_stats)} clusters.",
            color="green",
            action="show",
            autoClose=3000,
            position="top-right"
        )
        
        return fig, stats_table, results_df.to_dict('records'), False, notification
        
    except Exception as e:
        error_notification = dmc.Notification(
            title="Error",
            message=f"Clustering failed: {str(e)}",
            color="red",
            action="show",
            autoClose=5000,
            position="top-right"
        )
        return no_update, no_update, no_update, False, error_notification

# Callback for cluster details when clicking on plot
@callback(
   
   
    Output('cluster-details-table', 'children'),
    Input('leiden-cluster-fig', 'clickData'),
    State('leiden-cluster-data', 'data'),
    prevent_initial_call=True
)
def show_cluster_details(clickData, cluster_data):
    if clickData is None or cluster_data is None:
        return "Click on a data point to see cluster details"
    
    try:
        # Get the clicked point
        point_data = clickData['points'][0]
        clicked_informant = point_data['text']
        
        # Convert cluster data back to DataFrame
        df = pd.DataFrame.from_records(cluster_data)
        
        # Find the cluster of the clicked informant
        clicked_cluster = df[df['InformantID'] == clicked_informant]['cluster'].iloc[0]
        
        # Get all informants in the same cluster
        cluster_members = df[df['cluster'] == clicked_cluster]
        
        # Create details table
        details_cols = ['InformantID', 'MainVariety', 'Gender', 'Age']
        available_cols = [col for col in details_cols if col in cluster_members.columns]
        
        details_table = dag.AgGrid(
            rowData=cluster_members[available_cols].to_dict("records"),
            columnDefs=[{"field": i, "sortable": True} for i in available_cols],
            className="ag-theme-quartz compact",
            columnSize="autoSize",
            style={"height": "200px"}
        )
        
        return [
            dmc.Text(f"Cluster {clicked_cluster} ({len(cluster_members)} members)", fw=700, mb=10),
            details_table
        ]
        
    except Exception as e:
        return f"Error displaying cluster details: {str(e)}"

# Simplified: Synchronize participants tree with stored participants (no tab switching needed)
@callback(
    Output('participantsTree', 'checked'),
    Input('UMAPparticipants', 'data'),
    State('participantsTree', 'checked'),
    prevent_initial_call=False
)
def sync_participants_tree_with_store(stored_participants, current_checked):
    """
    Sync the participants tree checked state with the stored participants.
    Only update if the selection has actually changed to avoid unnecessary large data transfer.
    """
    # If nothing stored, do nothing
    if stored_participants is None or stored_participants == []:
        return no_update
    # If current_checked is None (first load), set it
    if current_checked is None:
        return stored_participants
    # Only update if the selection has changed (compare hashes)
    if _participants_hash(current_checked) != _participants_hash(stored_participants):
        return stored_participants
    return no_update

# 1. Add a callback to update UMAPparticipants store when tree selection changes
@callback(
    Output('UMAPparticipants', 'data', allow_duplicate=True),
    Input('participantsTree', 'checked'),
    State('UMAPparticipants', 'data'),
    prevent_initial_call=True
)
def update_participants_store_on_tree_change(checked_participants, last_synced):
    """
    Update the UMAPparticipants store whenever the tree selection changes.
    Only update if the selection has actually changed.
    """
    if checked_participants is not None and checked_participants != [] and checked_participants != ['informants']:
        # Only update if changed
        if _participants_hash(checked_participants) != _participants_hash(last_synced):
            return checked_participants
    return no_update

# Add callback to update participants-count-text
# Deleted: Participants count text callback - redundant with badge display

# Deleted: Grammar items count text callback - redundant with badge display

# Add callback to enable/disable UI elements based on Type switch
@callback(
    [Output('grammar-items-preset', 'disabled'),
     Output('grammar_toggle_written_only', 'disabled'),
     Output('grammar_toggle_currency', 'disabled')],
    Input('grammar-type-switch', 'checked'),
    prevent_initial_call=False
)
def toggle_item_selection_ui(type_switch_checked):
    """
    Enable/disable UI elements based on Type switch state.
    If Type = "Mode difference" (checked=True), disable preset and toggle buttons.
    If Type = "Individual items" (checked=False), enable them.
    Note: The preset filter in the grammar table remains enabled.
    """
    # When checked=True, it's "Mode difference" - disable elements
    # When checked=False, it's "Individual items" - enable elements
    disabled = type_switch_checked
    return disabled, disabled, disabled

# Add callback to update preset options based on item difference mode
@callback(
    [Output('grammar-items-preset', 'data'),
     Output('grammar-items-preset-filter', 'data')],
    Input('grammar-type-switch', 'checked'),
    prevent_initial_call=False
)
def update_preset_options(type_switch_checked):
    """
    Update preset options based on Type switch state.
    If Type = "Mode difference" (checked=True), exclude "Top 15", "Mode: Spoken", "Mode: written".
    If Type = "Individual items" (checked=False), show all presets.
    """
    if type_switch_checked:
        # Filter out mode-related presets when in item difference mode
        filtered_presets = [
            p for p in item_presets 
            if p['label'] not in ['Top 15 spoken', 'Mode: Spoken', 'Mode: written']
        ]
        filtered_data = build_preset_multiselect_data(filtered_presets)
        # For the filter dropdown, we need a flat list of labels
        filter_data = [{'label': p['label'], 'value': p['label']} for p in filtered_presets]
        return filtered_data, filter_data
    else:
        # Return all presets in individual items mode
        filter_data = [{'label': p['label'], 'value': p['label']} for p in item_presets]
        return labels_dict, filter_data

@callback(
    [Output('grammarItemsTree', 'data'),
     Output('grammarItemsTree', 'checked')],
    Input('grammar-type-switch', 'checked'),
    prevent_initial_call=False
)
def update_grammar_items_tree(type_switch_checked):
    """
    Redraw the grammar items tree based on Type switch state.
    If Type = "Mode difference" (checked=True), show pairs and check GrammarItemsColsPairs.
    If Type = "Individual items" (checked=False), show individual items and check GrammarItemsCols.
    """
    # When checked=True, it's "Mode difference" - use pairs=True
    # When checked=False, it's "Individual items" - use pairs=False
    if not type_switch_checked:
        # Individual items, use the full grammarMeta
        meta = grammarMeta.copy(deep=True)
        checked_items = []  # Start empty, no auto-selection
    else:
        meta = grammarMetaPairs.copy(deep=True)
        # Mode difference, use pairs
        checked_items = []  # Start empty, no auto-selection
    pairs = type_switch_checked
    tree_data = drawGrammarItemsTree(meta, pairs=pairs)
    return tree_data, checked_items

# Callback to parse URL parameters and set England mapping flag
@callback(
    [Output('england-mapping-param', 'data'),
     Output('informants-store', 'data'),
     Output('participantsTree', 'data'),
     Output('participantsTree', 'checked', allow_duplicate=True),
     Output('regional-mapping-switch', 'checked')],
    [Input('url', 'search'),
     Input('regional-mapping-switch', 'checked')],
    prevent_initial_call='initial_duplicate'
)
def update_regional_mapping(search, switch_checked):
    """
    Update regional mapping based on URL parameters or switch toggle.
    Also reload informants data and rebuild participant tree.
    Accepts both 'RegionalMapping' and 'regional_variation' URL parameters.
    """
    from dash import ctx
    
    regional_mapping = False
    
    # Determine which input triggered the callback
    triggered_id = ctx.triggered_id if ctx.triggered else None
    
    if triggered_id == 'regional-mapping-switch':
        # Switch was toggled - use its state
        regional_mapping = switch_checked if switch_checked is not None else False
    else:
        # URL parameter or initial load
        if search:
            # Parse the URL parameters
            from urllib.parse import parse_qs
            params = parse_qs(search.lstrip('?'))
            
            # Check for RegionalMapping parameter (supports multiple spellings)
            if 'RegionalMapping' in params:
                value = params['RegionalMapping'][0].lower()
                regional_mapping = value in ['true', '1', 'yes']
            elif 'regional_variation' in params:
                value = params['regional_variation'][0].lower()
                regional_mapping = value in ['true', '1', 'yes']
    
    # Reload informants data with the regional mapping flag
    informants = retrieve_data.getInformantDataGrammar(imputed=True, regional_mapping=regional_mapping)
    
    # Rebuild the participants tree with the updated informants data
    tree_data = drawParticipantsTree(informants)
    
    # Clear checked items when regional mapping changes to force user to reselect
    return regional_mapping, informants.to_dict("records"), tree_data, [], regional_mapping
