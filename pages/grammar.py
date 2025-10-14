import dash_mantine_components as dmc
from dash import register_page
import pages.data.retrieve_data as retrieve_data
from pages.data.grammarFunctions import *
from pages.data.grammarFunctions import _participants_hash
from pages.components.grammar_overview import get_grammar_overview
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

def create_plot_cache_key(participants, items, n_neighbours, min_dist, distance_metric, standardize, densemap, pairs):
    """Create a unique cache key for plot parameters"""
    key_data = {
        'participants': sorted(participants) if participants else 'all',
        'items': sorted(items) if items else 'all', 
        'n_neighbours': n_neighbours,
        'min_dist': min_dist,
        'distance_metric': distance_metric,
        'standardize': standardize,
        'densemap': densemap,
        'pairs': pairs
    }
    key_string = str(key_data)
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cached_umap_plot(participants, items, n_neighbours, min_dist, distance_metric, standardize, densemap, pairs):
    """Get UMAP plot from cache or compute if not exists"""
    cache_key = f"umap_{create_plot_cache_key(participants, items, n_neighbours, min_dist, distance_metric, standardize, densemap, pairs)}"
    
    cached_plot = plot_cache.get(cache_key)
    if cached_plot is not None:
        return cached_plot
    
    # Not in cache, compute it
    # Get data filtered by participants to ensure cache consistency
    if not pairs:
        grammarData = retrieve_data.getGrammarData(imputed=True, participants=participants, columns=items)
        grammarCols = GrammarItemsCols
    else:
        grammarData = retrieve_data.getGrammarData(imputed=True, participants=participants, columns=items, pairs=True)
        grammarCols = GrammarItemsColsPairs
        
    plot = getUMAPplot(
        grammarData=grammarData,
        GrammarItemsCols=grammarCols,
        informants=Informants,
        selected_informants=participants,
        items=items,
        n_neighbours=n_neighbours,
        min_dist=min_dist,
        distance_metric=distance_metric,
        standardize=standardize,
        densemap=densemap,
        pairs=pairs
    )
    
    # Cache the result for 24 hours
    #plot_cache.set(cache_key, plot, expire=86400)
    plot_cache.set(cache_key, plot)
    return plot

@lru_cache(maxsize=1)
def get_grammar_data_cached():
    return retrieve_data.getGrammarData(imputed=True)

@lru_cache(maxsize=1)
def get_grammar_data_pairs_cached():
    return retrieve_data.getGrammarData(imputed=True, items=retrieve_data.getGrammarItemsCols("item_pairs"), pairs=True)

@lru_cache(maxsize=1)
def get_informants_cached():
    return retrieve_data.getInformantDataGrammar(imputed=True)

@lru_cache(maxsize=1)
def get_grammar_data_raw_cached():
    return retrieve_data.getGrammarData(imputed=False)

@lru_cache(maxsize=1)
def get_grammar_data_pairs_raw_cached():
    return retrieve_data.getGrammarData(imputed=False, items=retrieve_data.getGrammarItemsCols("item_pairs"), pairs=True)

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

def generate_dynamic_presets(meta_df):
    """Generate presets dynamically from meta data columns"""
    presets = []
    
    # Base presets
    presets.extend([
        {'label':'All','value':GrammarItemsCols},
        {'label':'Custom','value':[]},
        {'label':'Top 15 spoken','value':['A8','B1','B14','B22','C12','D4','D6','D22','E11','E12','E15','E19','F2','F20','F22']},
        # {'label':'Gender attribution','value':['A14','A18','B16','E10','F9','G16','G17','H2','K6','L10']},
        # # Keep existing manually curated presets
        # {'label':'Subjunctive','value':['A10','B20','B21','B7','I23','L24','N1','N4','N5','D2','D7','E17','I13','K5','N14','F2','G19','K24']},
        # {'label':'Gender attribution','value':['A14','A18','B16','E10','F9','G16','G17','H2','K6','L10']},
        # {'label':'Comparative strategies','value':['A12','A21','B12','B15','C11','C16','E14','E2','E8','F11','G20','G9','H25','I15','I24','J11','J7','K12','L21','L4','M11','M17','M25']},
        # {'label':'Coordinated subjects','value':['J24','D23','D5','G25','H22','N13','B4']},
        # {'label':'Agreement/concord','value':['B13','B17','E1','E5','J12','K13','M3','M6','N9','F10','G4','H23','M15','N23','C6','D12','D8','F4','I25','I7','J25','M14','A23','K15','A5','C17','E18','H15','J15','J16','K21','M1','N10']},
        # {'label':'(Invariant) question tags','value':['H7','A8','C3','D9','E23','F22','G24','G3','I16','K2','K4a','K4b','N21']},
        # {'label':'Levelling','value':['E21','A17','C4','D10','F21','F5','G12','G22','H13','H19','M4','N2','F3','J19']},
        # {'label':'Negation','value':['J19','A2','A3','B19','C18','E11','E16','H6','I21','I5','J18','J6','L11','N15','N7']},
        # {'label':'Articles','value':['B5','C15','D13','D14','D20','F14','F17','G1','G10','G14','G26','H26','H3','I12','I26','I8','J13','J26','K26','L26','N18','N19','N8']},
        # {'label':'Prepositions','value':['G13','G15','H14','I11','I6','J10','J17','J2','J8','L15','L17','L22','L5','L8','M16','M21','M23','M7','M8','N12']},
    ])
    
    # Dynamic presets from section column (Mode: prefix)
    if 'section' in meta_df.columns:
        section_groups = meta_df.groupby('section')['question_code'].apply(list).to_dict()
        for section, codes in sorted(section_groups.items()):
            if section and pd.notna(section):
                presets.append({
                    'label': f'Mode: {section}',
                    'value': codes
                })
    
    # Dynamic presets from group_finegrained column (Group: prefix)
    if 'group_finegrained' in meta_df.columns:
        group_groups = meta_df.groupby('group_finegrained')['question_code'].apply(list).to_dict()
        for group, codes in sorted(group_groups.items()):
            if group and pd.notna(group) and group.strip():
                presets.append({
                    'label': f'Group: {group}',
                    'value': codes
                })
    
    # Dynamic presets from feature_ewave column (eWAVE: prefix)
    if 'feature_ewave' in meta_df.columns:
        ewave_groups = meta_df.groupby('feature_ewave')['question_code'].apply(list).to_dict()
        for feature, codes in sorted(ewave_groups.items()):
            if feature and pd.notna(feature) and feature.strip():
                presets.append({
                    'label': f'eWAVE: {feature}',
                    'value': codes
                })
    
    return presets

# Lazy initial plot generation - only create when needed
@lru_cache(maxsize=1)
def get_initial_umap_plot():
    """Generate initial UMAP plot lazily"""
    grammarData = get_grammar_data_cached()
    return getUMAPplot(
        grammarData=grammarData,
        GrammarItemsCols=GrammarItemsCols,
        informants=Informants,
        items=GrammarItemsCols,
        n_neighbours=25,
        min_dist=0.1,
        distance_metric="cosine",
        standardize=False,
        pairs=False
    )

@lru_cache(maxsize=1)
def get_initial_item_plot():
    """Generate initial item plot lazily"""
    return getItemPlot(
        informants=Informants['InformantID'].tolist(),
        items=GrammarItemsCols,
        plot_mode="normal"
    )

# Create simple empty plots for initial display
UMAP_Grammar_initialPlot = go.Figure()
UMAP_Grammar_initialPlot.update_layout(
    template="simple_white",
    title="Click 'Render UMAP' to generate plot",
    annotations=[
        dict(text="UMAP plot will be generated when you click the render button",
             xref="paper", yref="paper",
             x=0.5, y=0.5, showarrow=False,
             font=dict(size=16))
    ]
)

itemPlot_Grammar_initialPlot = go.Figure()
itemPlot_Grammar_initialPlot.update_layout(
    template="simple_white", 
    title="Click 'Render plot' to generate item plot",
    annotations=[
        dict(text="Item plot will be generated when you click the render button",
             xref="paper", yref="paper", 
             x=0.5, y=0.5, showarrow=False,
             font=dict(size=16))
    ]
)

initial_hoverinfo = construct_initial_hoverinfo(UMAP_Grammar_initialPlot)

# presets
# Dynamically generated from meta table
item_presets = generate_dynamic_presets(grammarMeta)
labels_dict = [{'label': preset['label'], 'value': preset['label']} for preset in item_presets]




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
                # Simplified: No Card wrapper, just Tabs
                dmc.Tabs(
                    [
                        dmc.TabsList(
                            [
                                dmc.TabsTab("UMAP", value="umap-plot"),
                                dmc.TabsTab("Group comparison", value="rf-plot"),
                            ]
                        ),
                        dmc.TabsPanel(
                            dcc.Graph(id="UMAPfig", figure=UMAP_Grammar_initialPlot, style={'height': '70vh'}, config={
                                'toImageButtonOptions': {
                                    'format': 'svg',
                                    'filename': 'umap_plot',
                                    'height': 600,
                                    'width': 800,
                                    'scale': 1
                                }
                            }),
                            value="umap-plot"
                        ),
                        dmc.TabsPanel(
                            dcc.Graph(id="RFPlotFig", figure=emptyFig, style={'height': '70vh'}, config={
                                'toImageButtonOptions': {
                                    'format': 'svg',
                                    'filename': 'rf_plot',
                                    'height': 600,
                                    'width': 800,
                                    'scale': 1
                                }
                            }),
                            value="rf-plot"
                        ),
                    ],
                    id='grammar-UMAP-main-tabs',
                    color="blue",
                    orientation="horizontal",
                    variant="default",
                    value="umap-plot"
                )
            ],span=12),
            # Commented out: Sociodemographic data card (Table and Age/Gender histogram)
            # dmc.GridCol(children=[
            #     dmc.Card(children=[
            #          dmc.Tabs(
            #             [
            #                 dmc.TabsList(
            #                     [
            #                         dmc.TabsTab("Table: Sociodemographic data", value="selTable"),
            #                         dmc.TabsTab("Histogram: Age/Gender",value="selAgeGender"),
            #                     ]
            #                 ),
            #                 dmc.TabsPanel(
            #                         html.Div(id="AuxPlotTable",className="dbc")
            #                     , value="selTable"),
            #                 dmc.TabsPanel(dcc.Graph(id="AuxPlotFig", figure=emptyFig), value="selAgeGender"),
            #
            #             ],
            #             id='grammar-UMAP-aux-tabs',
            #             color="blue", # default is blue
            #             orientation="horizontal", # or "vertical"
            #             variant="default", # or "outline" or "pills"
            #             value="selTable"
            #         ),
            #         ], withBorder=True,
            #         shadow="sm",
            #         radius="md")
            # ],span=12),
        ])        
    ], fluid=True)


## Stacks for InformantPlotContainer
AgeGender = dmc.Stack([
    dmc.Text("Age/Gender"),
    dcc.Graph(id="AgeGenderPlotG", figure=getAgeGenderPlot(Informants),style={'height': '200px'},config={'displayModeBar': False})
    ])

MainVarieties = dmc.Stack([

    dmc.Text("Main varieties"),
    html.Div(id="NationalityPlotContainer", children=[
    dcc.Graph(id="MainVarietiesPlotG", figure=getMainVarietiesPlot(Informants),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

Nationality = dmc.Stack([

    dmc.Text("Nationality"),
    html.Div(id="NationalityPlotContainer", children=[
    dcc.Graph(id="NationalityPlotG", figure=getCategoryHistogramPlot(Informants,"Nationality", True, ""),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

EthnicSID = dmc.Stack([

    dmc.Text("Ethnic Self-ID"),
    html.Div(id="EIDPlotContainer", children=[
    dcc.Graph(id="EIDPlotG", figure=getCategoryHistogramPlot(Informants,"EthnicSelfID", True,""),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px','overflowY': 'scroll'}),
])

CountryID = dmc.Stack([

    dmc.Text("Country (or region) you identify with most"),
    html.Div(id="CIDPlotContainer", children=[
    dcc.Graph(id="CIDPlotG", figure=getCategoryHistogramPlot(Informants,"CountryID", True, ""),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

LanguagesHome = dmc.Stack([

    dmc.Text("Languages used at home"),
    html.Div(id="LanguagesHomePlotContainer", children=[    dcc.Graph(id="LanguagesHomePlotG", figure=getCategoryHistogramPlot(Informants,"LanguageHome_normalized", True, ","),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])


LanguageMother = dmc.Stack([

    dmc.Text("Mother's Native Language"),
    html.Div(id="LanguagesMotherPlotContainer", children=[
    dcc.Graph(id="LanguagesMotherPlotG", figure=getCategoryHistogramPlot(Informants,"LanguageMother_normalized", True, ","),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

LanguageFather = dmc.Stack([

    dmc.Text("Father's Native Language"),
    html.Div(id="LanguagesFatherPlotContainer", children=[
    dcc.Graph(id="LanguagesFatherPlotG", figure=getCategoryHistogramPlot(Informants,"LanguageFather_normalized", True, ","),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

PrimarySchool = dmc.Stack([
    
        dmc.Text("Primary School"),
        dcc.Graph(id="PrimarySchoolPlotG", figure=getCategoryHistogramPlot(Informants,"PrimarySchool",True),config={'displayModeBar': False})

])

SecondarySchool = dmc.Stack([
    
        dmc.Text("Secondary School"),
        dcc.Graph(id="SecondarySchoolPlotG", figure=getCategoryHistogramPlot(Informants,"SecondarySchool",True),config={'displayModeBar': False})

])
Qualifications = dmc.Stack([
    
        dmc.Text("Highest Qualification"),
        dcc.Graph(id="QualiPlotG", figure=getCategoryHistogramPlot(Informants,"Qualifications",True),config={'displayModeBar': False})

])

YearsLivedOutside = dmc.Stack([
    
        dmc.Text("Years lived outside home country"),
        dcc.Graph(id="YLOPlotG", figure=getFloatHistogramPlot(Informants,"YearsLivedOutside"),config={'displayModeBar': False})

])

YearsLivedOtherE = dmc.Stack([
    
        dmc.Text("Years lived in other English-speaking countries"),
        dcc.Graph(id="YLOEPlotG", figure=getFloatHistogramPlot(Informants,"YearsLivedOtherEnglish"),config={'displayModeBar': False})

])

RatioMainVariety = dmc.Stack([
    
        dmc.Text("Ratio Main Variety"),
        dcc.Graph(id="RatioMainVarietyPlotG", figure=getFloatHistogramPlot(Informants,"RatioMainVariety"),config={'displayModeBar': False})

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

    # Grid to mimic questionnaire layout
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
                        #Age,
                        #dmc.Divider(),
                        #Gender,
                        #dmc.Divider(),
                        AgeGender,
                        dmc.Divider(),
                        MainVarieties,
                        dmc.Divider(),
                        PIAccordion,
                        #LanguagesHome,
                        #dmc.Divider(),
                        #LanguageMother,
                        #dmc.Divider(),
                        #LanguageFather
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
    ],withBorder=True,shadow="sm",radius="md"),
    # --- Add informants table below plots ---
    dmc.Card(
        children=[
            dmc.Text("Informants Data Table", fw=700, mb=5),
            dmc.Text("💡 Click headers to sort • Drag column borders to resize • Use filter boxes below headers to search.", 
                     size="sm", c="dimmed", mb=10, 
                     style={"fontStyle": "italic", "backgroundColor": "#f8f9fa", "padding": "8px", "borderRadius": "4px", "border": "1px solid #e9ecef"}),
            # Table will be updated by callback, initial data is all informants
            dag.AgGrid(
                id="informants-table",
                rowData=Informants.to_dict("records"),
                columnDefs=[
                    {
                        "field": col, 
                        "headerName": col.replace("ID", " ID").replace("Occup", "Occupation").replace("Quali", "Qualification").replace("Ethnic", "Ethnic "),
                        "filter": "agTextColumnFilter", 
                        "sortable": True,
                        "resizable": True,
                        "minWidth": 120 if "Language" in col or "Variety" in col else 100,
                        "flex": 1,
                        "cellStyle": {"textAlign": "left"},
                        "headerTooltip": f"Click to sort by {col}. Use filter below to search."
                    } 
                    for col in Informants.columns
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
                style={"height": "400px"},
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
        withBorder=True,
        shadow="sm",
        radius="md",
        style={"marginTop": "20px"}
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

# Import overview content from separate component
Overview = get_grammar_overview()

GrammarPlots = dmc.Container([
    dmc.Title("Grammatical Items", order = 2),
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
                    dmc.Button(
                        "Deselect selection", 
                        id='deselect-selected-participants', 
                        size="xs", 
                        variant="outline",
                        color="orange",
                        disabled=True,
                        style={"display": "none"}  # Initially hidden
                    )
                ], mb="xs"),
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
                                            dmc.Button("ENL", id='batch-select-enl', size="xs", variant="light", color="blue"),
                                            dmc.Button("ESL", id='batch-select-esl', size="xs", variant="light", color="green"),
                                            dmc.Button("EFL", id='batch-select-efl', size="xs", variant="light", color="orange"),
                                            dmc.Button("Balanced", id='batch-select-equal-per-variety', size="xs", variant="light", color="indigo"),
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
                                            dmc.Button("Female", id='batch-select-female', size="xs", variant="light", color="pink"),
                                            dmc.Button("Male", id='batch-select-male', size="xs", variant="light", color="cyan"),
                                        ], gap="xs", mb="xs"),
                                        dmc.Group(children=[
                                            dmc.Button("Balanced", id='batch-select-balanced', size="xs", variant="light", color="violet"),
                                            dmc.Button("Balanced per variety", id='batch-select-balanced-per-variety', size="xs", variant="light", color="grape"),
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
                dmc.Tree(id='participantsTree', data=drawParticipantsTree(Informants), checkboxes=True, checked=Informants['InformantID'].tolist()),
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
                                            label="Main variety:",
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
                            checked=True,
                            persistence=persist_UI,
                            persistence_type=persistence_type,
                            size="sm",
                        ),
                        dmc.Stack(
                        children=[
                            dmc.Select(label="Select a preset:",
                            placeholder="All",
                            id="grammar-items-preset",
                            value="All",
                            data=labels_dict,
                            allowDeselect=False,
                            size="xs",
                            persistence=persist_UI,persistence_type=persistence_type),
                            dmc.Group(children=[
                                dmc.Button("Toggle written-only",
                                    id="grammar_toggle_written_only",
                                    size="xs",
                                    variant="light"
                                ),
                                dmc.Button("Toggle currency/unit",
                                    id="grammar_toggle_currency",
                                    size="xs",
                                    variant="light"
                                ),
                            ], gap="xs"),
                            dmc.Button("Deselect problematic items",
                                id="grammar_deselect_problematic",
                                color="red",
                                variant="outline",
                                size="xs"
                            ),
                        ], gap="xs"),
                        # better: include drop-down with presets
                        dmc.Group(children=[
                            dmc.Button("Select All", id='select-all-grammar-items', size="xs", variant="outline"),
                            dmc.Button("Deselect All", id='deselect-all-grammar-items', size="xs", variant="outline"),
                        ], mb="xs"),
                        html.Div([
                            dcc.Store(id="grammar-tree-css", data={}),  # Dummy store for CSS
                            dmc.Tree(
                                id='grammarItemsTree',
                                data=drawGrammarItemsTree(grammarMeta,pairs=False), 
                                checkboxes=True, 
                                checked=GrammarItemsCols,

                            )
                        ], 
                        # Use a wrapper div with custom CSS class
                        className="grammar-tree-wrapper",
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
                    {"value":"split_by_variety","label":"Split by variety"},
                    {"value":"diverging","label":"Diverging stacked bars"},
                    {"value":"informant_boxplot","label":"Informant mean (boxplot)"},
                    {"value":"correlation_matrix","label":"Correlation matrix"},
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
                            dmc.Text("Average rating filter:", size="xs", fw=600, c="dimmed"),
                            dmc.Text("Only display items where all groups fall within range:", size="xs"),
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
SettingsGrammarAnalysis = dmc.Container([
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
        dmc.SegmentedControl(
            id="grammar-plot-type",
            data=[
                {"value": "item", "label": "Item Plot"},
                {"value": "umap", "label": "UMAP"},
            ],
            value="item",
            fullWidth=True,
            color="blue",
            size="sm"
        ),
        dmc.Text(
            id="plot-type-description",
            children="Visualize grammatical item frequencies across participants",
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
                'Render Item Plot',
                id='render-grammar-plot',
                size="md",
                leftSection=DashIconify(icon="tabler:chart-bar", width=20),
                color="blue",
                fullWidth=True,
                disabled=False,
            ),
        ]
    ),
        # UMAP-specific buttons (shown only when plot type is UMAP)
    html.Div(id="umap-group-buttons", children=[
        dmc.Group(children=[
            dmc.Button('Add group', id='Umap-add-group', disabled=True),
            dmc.Button('Clear groups', id='Umap-clear-groups'),
        ],
        grow=True,
        wrap="nowrap",
        mb="md"),
        dmc.Group(children=[
            dmc.Button('Compare selected groups', id='render-rf-plot'),
        ],
        grow=True,
        wrap="nowrap",
        mb="md"),
    ], style={"display": "none"}),  # Hidden by default
    
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
    ]),  # Visible by default (no style attribute needed)
    
    html.Div(id="umap-settings-container", children=[
        dmc.Accordion(children=[
            umapSettingsAccordion,
            umapGroupCompAccordion
        ], 
        variant="contained",
        radius="md"),
    ], style={"display": "none"}),  # Hidden by default
    
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
                        # Export Actions (Color-coded)
                        dmc.Stack([
                            dmc.Group([
                                DashIconify(icon="tabler:download", width=16),
                                dmc.Text("Export & Share", size="sm", fw=500)
                            ], gap="xs", mb="xs"),
                            dmc.Group([
                                dmc.Button(
                                    "Export Data",
                                    id='export-data-button',
                                    size="xs",
                                    variant="light",
                                    color="teal",
                                    leftSection=DashIconify(icon="tabler:table-export", width=14),
                                    fullWidth=True
                                ),
                            ], grow=True),
                            dmc.Group([
                                dmc.Button(
                                    "Copy Settings",
                                    id='copy-settings-button',
                                    size="xs",
                                    variant="light",
                                    color="violet",
                                    leftSection=DashIconify(icon="tabler:copy", width=14),
                                    fullWidth=True
                                ),
                                dmc.Button(
                                    "Paste Settings",
                                    id='paste-settings-button',
                                    size="xs",
                                    variant="light",
                                    color="violet",
                                    leftSection=DashIconify(icon="tabler:clipboard", width=14),
                                    fullWidth=True
                                ),
                            ], grow=True),
                        ], gap="xs", mb="md"),
                        
                        # Settings Memory (Color-coded)
                        dmc.Stack([
                            dmc.Group([
                                DashIconify(icon="tabler:device-floppy", width=16),
                                dmc.Text("Settings Memory", size="sm", fw=500)
                            ], gap="xs", mb="xs"),
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
    

    # Common accordions (always visible)
    #dmc.Accordion(children=[
    #    informantSelectionAccordion,
    #    itemSelectionAccordion,
    #], 
    #variant="contained",
    #radius="md",
    #mb="md",
    #value=["LoadData", "LoadItems"]),  # Keep both accordions open by default
    
    # # Plot-specific settings (conditionally visible)
    # html.Div(id="item-plot-settings-container", children=[
    #     dmc.Accordion(children=[
    #         itemPlotSettingsAccordion,
    #     ], 
    #     variant="contained",
    #     radius="md"),
    # ]),  # Visible by default (no style attribute needed)
    
    # html.Div(id="umap-settings-container", children=[
    #     dmc.Accordion(children=[
    #         umapSettingsAccordion,
    #         umapGroupCompAccordion
    #     ], 
    #     variant="contained",
    #     radius="md"),
    # ], style={"display": "none"}),  # Hidden by default
], fluid=True)

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
        dmc.Card(children=[
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Plot View", value="plot-view", id="grammar-analysis-plot-tab"),
                            dmc.TabsTab("Sociodemographic Details", value="sociodemographic-details"),
                            dmc.TabsTab("Grammatical Items Table", value="grammar-items-table"),
                        ]
                    ),
                    dmc.TabsPanel(
                        # Unified plot container that shows either Item Plot or UMAP based on plot type
                        html.Div(id="grammar-unified-plot-container", children=[
                            # Item Plot Container (visible by default)
                            html.Div(id="item-plot-display", children=[ItemPlotContainer], style={"display": "block"}),
                            # UMAP Plot Container (hidden by default)
                            html.Div(id="umap-plot-display", children=[UmapPlotContainer], style={"display": "none"}),
                        ]),
                        value="plot-view"
                    ),
                    dmc.TabsPanel(
                        InformantsPlotContainer,
                        value="sociodemographic-details"
                    ),
                    dmc.TabsPanel(
                        getMetaTable(grammarMeta),
                        value="grammar-items-table"
                    ),
                ],
                id='grammar-analysis-tabs',
                color="blue",
                orientation="horizontal",
                variant="default",
                value="plot-view"
            )
        ], withBorder=True, shadow="sm", radius="md")],
        id="grammar-analysis-tab-content",
        style={"paddingTop": 10}),
        span=8),
    dmc.GridCol(SettingsGrammarAnalysis, span=4, style={"padding-top":"10px","margin-top": "5px","border-left": "1px solid #f0f0f0","padding-left": "10px"}),
], gutter="xl")

# Deleted: informantsC container (deprecated - merged into grammarAnalysisC)

# tab container for Leiden clustering
leidenC = dmc.Grid([
    dmc.GridCol(html.Div(children = [LeidenClusterContainer],id="leiden-plot-tab-content",style={"paddingTop": 10}),span=8),
    dmc.GridCol(SettingsLeiden,span=4,style={"padding-top":"10px","margin-top": "5px","border-left": "1px solid #f0f0f0","padding-left": "10px"}),
], gutter="xl")

GrammaticalItems = dmc.Container([dmc.Grid(children=[
            dmc.GridCol(children=[
                dmc.Card(children=[
                        getMetaTable(grammarMeta) 
                ], withBorder=True,
                shadow="sm",
                radius="md")
            ],span=12),
        ])        
    ], fluid=True)



layout = html.Div([

    customSetWarningModal, 
    dcc.Store(id="UMAPgroup", storage_type="memory",data=0),
    dcc.Store(id="UMAPparticipants",storage_type="memory",data=Informants['InformantID']), # fill those two cache lists with initial data (i.e. initial plot shows all participants)
    dcc.Store(id="UMAPitems",storage_type="memory",data=GrammarItemsCols),
    dcc.Store(id="UMAPGroupsForRF",storage_type="memory",data={"dataframe":pd.DataFrame().to_dict("records")}),
    dcc.Store(id="grammar_plots_UMAP",storage_type="memory",data=None),
    dcc.Store(id="grammar_plots_item",storage_type="memory",data=itemPlot_Grammar_initialPlot),
    dcc.Store(id="informants-store", data=Informants.to_dict("records")),  
    dcc.Store(id="leiden-cluster-data", storage_type="memory"),
    dcc.Store(id="leiden-cluster-figure", storage_type="memory"),
    dcc.Store(id="umap-hoverinfo-store", storage_type="memory"),  # New store for hoverinfo
    
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
# Note: Leiden button excluded as the Leiden tab is currently commented out
@callback(
    [
        Output('render-UMAP-plot', 'loading', allow_duplicate=True),
        Output('render-rf-plot', 'loading', allow_duplicate=True),
        Output('render-item-plot', 'loading', allow_duplicate=True),
        Output('render-UMAP-plot', 'disabled', allow_duplicate=True),
        Output('render-rf-plot', 'disabled', allow_duplicate=True),
    ],
    [
        Input('render-UMAP-plot', 'n_clicks'),
        Input('render-rf-plot', 'n_clicks'),
        Input('render-item-plot', 'n_clicks'),
        Input('UMAPfig', 'figure'),
        Input('RFPlotFig', 'figure'),
        Input('ItemFig', 'figure'),
    ],
    prevent_initial_call=True
)
def manage_render_button_loading_states(*args):
    """
    Consolidated callback to manage loading states for render buttons.
    When a button is clicked, set its loading=True and disable other UMAP/RF buttons.
    When a figure updates, set corresponding loading=False and re-enable buttons.
    """
    triggered = ctx.triggered_id
    
    # Default states: no loading, buttons enabled
    umap_loading = False
    rf_loading = False
    item_loading = False
    umap_disabled = False
    rf_disabled = False
    
    # Handle button clicks (start loading)
    if triggered == 'render-UMAP-plot':
        umap_loading = True
        rf_disabled = True  # Disable RF while UMAP is running
    elif triggered == 'render-rf-plot':
        rf_loading = True
        umap_disabled = True  # Disable UMAP while RF is running
    elif triggered == 'render-item-plot':
        item_loading = True
    
    # Handle figure updates (stop loading and re-enable buttons)
    elif triggered in ['UMAPfig', 'RFPlotFig', 'ItemFig']:
        # All loading states set to False by default
        # Re-enable all buttons when figures update
        umap_disabled = False
        rf_disabled = False
    else:
        return no_update, no_update, no_update, no_update, no_update
    
    return umap_loading, rf_loading, item_loading, umap_disabled, rf_disabled

# Callback to disable group informants and item sorting dropdowns when correlation matrix is selected
@callback(
    [Output('items-group-by', 'disabled'),
     Output('items-sort-by', 'disabled')],
    Input('items-plot-mode', 'value')
)
def disable_controls_for_correlation_matrix(plot_mode):
    if plot_mode == "correlation_matrix":
        return True, True  # Disable both dropdowns
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

# Callback to show/hide UI elements based on plot type selection
@callback(
    [Output('item-plot-settings-container', 'style'),
     Output('umap-settings-container', 'style'),
     Output('item-plot-display', 'style'),
     Output('umap-plot-display', 'style'),
     Output('umap-group-buttons', 'style'),
     Output('Umap-add-group', 'disabled', allow_duplicate=True),
     Output('Umap-clear-groups', 'disabled', allow_duplicate=True),
     Output('render-grammar-plot', 'children'),
     Output('render-grammar-plot', 'leftSection'),
     Output('plot-type-description', 'children')],
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
            True,                  # disable add group
            True,                  # disable clear groups
            "Render Item Plot",    # button text
            DashIconify(icon="tabler:chart-bar", width=20),  # button icon
            "Visualize grammatical item frequencies across participant groups"  # description
        )
    else:  # plot_type == 'umap'
        # Show UMAP settings and display, hide item plot
        return (
            {"display": "none"},   # item settings
            {"display": "block"},  # umap settings
            {"display": "none"},   # item plot display
            {"display": "block"},  # umap plot display
            {"display": "block"},  # umap group buttons
            True,                  # disable add group (will be enabled after selection)
            False,                 # enable clear groups
            "Render UMAP Plot",    # button text
            DashIconify(icon="tabler:radar", width=20),  # button icon
            "Explore participant similarity using UMAP dimensionality reduction"  # description
        )

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
     Input('grammarItemsTree', 'checked')],
    prevent_initial_call=False
)
def update_quick_stats(selected_participants, selected_items):
    """Update the quick stats panel with current selection info and badge counts"""
    n_participants = len(selected_participants) if selected_participants else 0
    n_items = len(selected_items) if selected_items else 0
    total_participants = len(Informants)
    total_items = len(GrammarItemsCols)
    
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
        elif variety in ["Slovenia", "Germany", "Sweden"]:
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
     State('use-imputed-data-switch', 'checked')],
    prevent_initial_call=True
)
def export_data(n_clicks, participants, items, pairs, use_imputed):
    """Export current selection as CSV"""
    if not n_clicks or not participants or not items:
        return no_update
    
    from datetime import datetime
    
    # Get the data
    data = retrieve_data.getGrammarData(
        imputed=use_imputed,
        participants=participants,
        items=items,  # Fixed: changed from columns=items to items=items
        pairs=pairs
    )
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"grammar_data_{timestamp}.csv"
    
    return dict(content=data.to_csv(index=False), filename=filename)

# Callback to filter grammar items table based on selected items
@callback(
    Output('grammar-items-table', 'rowData'),
    Input('filter-grammar-items-table', 'n_clicks'),
    [State('grammarItemsTree', 'checked'),
     State('grammar-type-switch', 'checked')],
    prevent_initial_call=True
)
def filter_grammar_items_table(n_clicks, items, pairs):
    """Filter grammar items table to show only selected items"""
    if not n_clicks or not items:
        return no_update
    
    # Get the full grammar meta data
    if pairs:
        meta = retrieve_data.getGrammarMeta(type="item_pairs")
        # For pairs, the 'item_pair' column contains the pair codes like 'A1-G21'
        # Filter to only show rows where item_pair is in the selected items
        filtered = meta[meta['item_pair'].isin(items)]
    else:
        meta = retrieve_data.getGrammarMeta(type="all_items")
        # For individual items, filter by 'question_code' column
        filtered = meta[meta['question_code'].isin(items)]
    
    # Drop columns and rename as in getMetaTable
    filtered = filtered.drop(columns=[col for col in ['Standard_variety', 'Control Item'] if col in filtered.columns])
    filtered.columns = [col.replace('_', ' ') for col in filtered.columns]
    
    return filtered.to_dict("records")

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
                group_by: group_by,
                sort_by: sort_by,
                plot_mode: plot_mode
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
        "use_imputed": use_imputed
    }
    
    umap_settings = {
        "participants": participants,
        "items": items,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "distance_metric": distance_metric,
        "standardize": standardize,
        "pairs": pairs,
        "use_imputed": use_imputed
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
     Output('render-UMAP-plot', 'n_clicks', allow_duplicate=True),
     Output("notify-container", "children", allow_duplicate=True)],
    Input('render-grammar-plot', 'n_clicks'),
    [State('grammar-plot-type', 'value'),
     State('render-item-plot', 'n_clicks'),
     State('render-UMAP-plot', 'n_clicks')],
    prevent_initial_call=True
)
def unified_render_button(btn_clicks, plot_type, item_clicks, umap_clicks):
    """Route render button click to appropriate plot type"""
    if btn_clicks is None:
        return no_update, no_update, no_update
    
    notification = dmc.Notification(
        id="my-notification",
        title="Info",
        message=f"Rendering {plot_type.upper()} plot, please wait.",
        color="blue",
        loading=True,
        action="show",
        autoClose=2000,
        position="top-right"
    )
    
    if plot_type == 'item':
        # Trigger item plot render by incrementing its n_clicks
        return (item_clicks or 0) + 1, no_update, notification
    else:  # plot_type == 'umap'
        # Trigger UMAP plot render by incrementing its n_clicks
        return no_update, (umap_clicks or 0) + 1, notification

# Deleted: Callback to show/hide deselect button based on tab - outer tabs removed, always show button
# The button is now always available since grammarAnalysisC is directly visible

# Callback to enable/disable the "Deselect selection" button based on UMAP selection
@callback(
    Output('deselect-selected-participants', 'disabled'),
    Input('UMAPfig', 'selectedData'),
    prevent_initial_call=False
)
def toggle_deselect_button_state(selectedData):
    if selectedData and selectedData.get('points') and len(selectedData['points']) > 0:
        return False  # Enable button
    else:
        return True   # Disable button

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

# Deleted: Tab content callback - outer tabs removed, grammarAnalysisC shown directly

# Commented out: callback to display selected data for auxiliary plots (Table and Age/Gender histogram)
# @callback(	
#     [Output('AuxPlotFig', 'figure'),Output('AuxPlotTable', 'children')],
#     Input('UMAPfig', 'selectedData'),
#     [State('UMAPparticipants','data'),State('UMAPitems','data')]
# )
# def display_selected_data(selectedData, participants, items):
#     # call is used to render the auxiliary plots, i.e. age & gender distribution of selected points in UMAP
#     if selectedData is None:
#         auxPlots= getAuxiliaryPlot(Informants,participants=participants,items=items)
#         dt = getAuxiliaryTable(Informants,participants=participants)
#         # if no data is selected, use all data points for the auxiliary plots
#         #auxPlots = dcc.Graph(id="UMAPauxfig", figure=auxPlots)
#         # retrieve those from user cache
#         return auxPlots, dt
#     else:
#         #selectedData = selectedData['points']
#         ids = [point.get('id') for point in selectedData['points']]
#         auxPlots= getAuxiliaryPlot(Informants,participants=ids,items=items)
#         #auxPlots = dcc.Graph(id="UMAPauxfig", figure=auxPlots)
#         dt = getAuxiliaryTable(Informants,participants=ids)
#         # if data is selected, update auxiliary plots based on user selection
#         return auxPlots, dt


# using second callback to disable buttons
# "running=" does not appear to be working?
# maybe it's dmc?
@callback(
    [Output('render-UMAP-plot', 'loading', allow_duplicate=True),Output('Umap-add-group', 'disabled', allow_duplicate=True),Output('Umap-clear-groups', 'disabled', allow_duplicate=True),Output('render-rf-plot', 'disabled', allow_duplicate=True),Output("notify-container", "children", allow_duplicate=True)],
    [Input('grammar_running', 'data')], prevent_initial_call=True
)
def disable_render_button(running):
    # if one of the buttons is clicked, , show notification

    if not running:
        notification = dmc.Notification(
                        id="my-notification",
                        title="Info",
                        message="UMAP plot complete.",
                        color="green",
                        loading=False,
                        action="show",
                        autoClose=2000,
                        position="top-right"
                        #icon=DashIconify(icon="akar-icons:circle-check"),
                )
        return running, running, running, running, notification
    else:
        # if the button is clicked, disable the buttons
        return running, running, running, running, no_update
# using second callback to disable buttons
# "running=" does not appear to be working?
# maybe it's dmc?
@callback(
    Output("notify-container", "children", allow_duplicate=True),
    [Input('render-UMAP-plot','n_clicks'),Input('render-rf-plot','n_clicks')], prevent_initial_call=True
)
def show_render_notifcation(UMAPrender, RFrender):
    # if one of the buttons is clicked, , show notification

    if not UMAPrender and not RFrender:
        # if no button was clicked, do not show notification
        return no_update
    else:
        notification = dmc.Notification(
                        id="my-notification",
                        title="Info",
                        message="Rendering new plot, please wait.",
                        color="orange",
                        loading=True,
                        action="show",
                        autoClose=2000,
                        position="top-right"
                        #icon=DashIconify(icon="akar-icons:circle-check"),
                )
        return notification

# callback to update umap plot with groupings
@callback(	
    [Output('grammar_plots_UMAP', 'data'),Output('UMAPgroup', 'data'),Output('render-UMAP-plot','loading'),Output('Umap-add-group', 'disabled'),Output('Umap-clear-groups', 'disabled'),Output('UMAPparticipants','data'),Output('UMAPitems','data'),Output('UMAPGroupsForRF', 'data'),Output('render-rf-plot', 'disabled', allow_duplicate=True),Output("notify-container", "children", allow_duplicate=True),Output("confirm-custom-modal", "opened")],
    [Input('Umap-add-group', 'n_clicks'),Input('Umap-clear-groups', 'n_clicks'),Input('render-UMAP-plot','n_clicks'),Input('modal-ok-button', 'n_clicks'),Input('modal-cancel-button', 'n_clicks')],
    [State('UMAPfig', 'selectedData'),State('grammar_plots_UMAP', 'data'),State("UMAPgroup", "data"),State("participantsTree", "checked"),State("grammarItemsTree", "checked"),State('UMAPparticipants','data'),State('UMAPitems','data'),State('UMAP_neighbours','value'),State('UMAP_mindist','value'),State('UMAPfig','figure'),State("grammar_running","data"),State('grammar-items-preset', 'value'), State('umap-distance-metric-dropdown', 'value'), State('umap-standardize-checkbox', 'checked'),State('umap-densemap-checkbox', 'checked'),State('grammar-type-switch', 'checked'), State('use-imputed-data-switch', 'checked')], 
    prevent_initial_call=True,background=True,running=[(Output("grammar_running","data"),True,False)]
)
def updateGroupsUMAP(BTNaddgroup, BTNcleargroup,BTNrenderPlot, modal_ok, modal_cancel,selectedData, figure,  data, selected_informants, items, participantsCache, itemsCache, n_neighbours, min_dist,displayedFigure,running_state, preset, distance_metric, standardize_participant_ratings, densemap, pairs, use_imputed):
    # need longer callback here as outputs can only be references in one callback, hence the need to split the callback
    button_clicked = ctx.triggered_id # which buttonw as clicked
    # reset all groups without rerendering the plot
    if button_clicked == 'Umap-clear-groups':
        if (BTNcleargroup == 0 or BTNcleargroup is None):
            return figure, data, False, False, False, no_update, no_update, no_update, False, no_update, False
        newFig = displayedFigure
        for x in range(len(newFig['data'])):
            newFig['data'][x]['marker']['symbol'] = 0
        groupsCache = getColorGroupingsFromFigure(displayedFigure)
        return newFig, 0, False, False, False, no_update, no_update, groupsCache, False, no_update, False
    # add group without rerendering the plot
    elif button_clicked == 'Umap-add-group':
        if (BTNaddgroup == 0 or BTNaddgroup is None):
            return no_update, no_update, False, False, False, no_update, no_update, no_update, False, no_update, False
        if len(selectedData['points']) > 0:
            currentGroup = data
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
            if(data==14): # reset groups if maximum is reached
                data = 0
            else:
                data = data + 1
            groupsCache = getGroupingsFromFigure(displayedFigure)
            return newFig, data, False, False, False, no_update, no_update, groupsCache, False, no_update, False
        else:
            return no_update, no_update, False, False, False, no_update, no_update, no_update, False, no_update, False
    # rerender plot with seletected data
    elif button_clicked == 'render-UMAP-plot':
        if (BTNrenderPlot == 0 or BTNrenderPlot is None or running_state):
            while running_state: # if a background callback is running, wait, once it's finished, plot should be updated.
                raise PreventUpdate
        
            if figure is None or len(figure.get('data', [])) == 0:
                data = 0
                return no_update, no_update, False, False, False, no_update, no_update, no_update, False, no_update, False
            else:
                groupsCache = getColorGroupingsFromFigure(figure)
                return figure, no_update, False, False, False, no_update, items, groupsCache, False, no_update, False
        
        # Check if precomputed UMAP plot exists instead of checking for "Custom"
        import os
        import hashlib
        
        def _hash_list(lst):
            return hashlib.md5(str(sorted(lst)).encode()).hexdigest() if lst else "all"
        
        # Normalize tree selections using helper function
        selected_informants, items = normalize_tree_selection(selected_informants, items)
        
        # Modal disabled - proceed directly to rendering
        
        data = 0 # reset groups when plot is rerendered
        
        # Use cached UMAP plot generation for better performance
        figure = get_cached_umap_plot(
            participants=selected_informants,
            items=items,
            n_neighbours=n_neighbours,
            min_dist=min_dist,
            distance_metric=distance_metric,
            standardize=standardize_participant_ratings,
            densemap=densemap,
            pairs=pairs
        )
        groupsCache = getColorGroupingsFromFigure(figure)
        return figure, data, False, False, False, selected_informants, items, groupsCache, False, no_update, False
    elif button_clicked == 'modal-cancel-button':
        return no_update,no_update,no_update,no_update,no_update,no_update,no_update,no_update,no_update,no_update, False
    elif button_clicked == 'modal-ok-button':
        if (BTNrenderPlot == 0 or BTNrenderPlot is None or running_state):
            while running_state: # if a background callback is running, wait, once it's finished, plot should be updated.
                raise PreventUpdate
        
            if figure is None or len(figure.get('data', [])) == 0:
                data = 0
            else:
                groupsCache = getColorGroupingsFromFigure(figure)
                return figure, no_update, False, False, False, selected_informants, items, groupsCache, False, no_update, False

        data = 0 # reset groups when plot is rerendered
        
        # Normalize tree selections using helper function
        selected_informants, items = normalize_tree_selection(selected_informants, items)
            
        # Use cached UMAP plot generation for better performance  
        figure = get_cached_umap_plot(
            participants=selected_informants,
            items=items,
            n_neighbours=n_neighbours,
            min_dist=min_dist,
            distance_metric=distance_metric,
            standardize=standardize_participant_ratings,
            densemap=densemap,
            pairs=pairs
        )
        groupsCache = getColorGroupingsFromFigure(figure)
        return figure, data, False, False, False, selected_informants, items, groupsCache, False, no_update, False
    else:
        raise PreventUpdate
    # if no button was clicked, do not update anything
    return no_update, no_update, False, False, False, no_update, no_update, no_update, False, no_update, False

@callback(
    Output('UMAPfig', 'figure',allow_duplicate=True),
    Input('grammar_plots_UMAP', 'data'),
    prevent_initial_call=True
)
def set_umapfig_from_store(fig_data):
    if fig_data is not None:
        fig = go.Figure(fig_data)
        return fig
    return UMAP_Grammar_initialPlot


@callback(
    [Output('RFPlotFig','figure'),
     Output('render-UMAP-plot','loading',allow_duplicate=True),
     Output('Umap-add-group', 'disabled',allow_duplicate=True),
     Output('Umap-clear-groups', 'disabled',allow_duplicate=True),
     Output('render-rf-plot', 'disabled',allow_duplicate=True),
     Output('render-rf-plot', 'loading',allow_duplicate=True),
     Output("notify-container", "children",allow_duplicate=True), 
     Output('grammar-UMAP-main-tabs', 'value')],
    Input('render-rf-plot','n_clicks'),
    [State('UMAPGroupsForRF','data'),State('UMAPitems','data'),State('UMAPgroup','data'),State('RF_avg_range','value'),State('UMAPfig','figure'),State('UMAPparticipants','data'),State('grammar-type-switch','checked'),State('use-imputed-data-switch', 'checked'),State('rf-use-zscores','checked')],
    prevent_initial_call=True
)
def renderRFPlot(BTN,groups,items,UMAPgroup,value_range,figure,visible_participants,pairs,use_imputed,use_zscores):
    # Set default value for split_by_variety since checkbox was removed
    split_by_variety = False
    
    button_clicked = ctx.triggered_id
    if button_clicked == 'render-rf-plot' and BTN is not None:
        if(UMAPgroup==1):
            notification = dmc.Notification(
                    id="my-notification",
                    title="Info",
                    message="Please select more than one group. If no group is selected, varieties will be used as groups.",
                    color="orange",
                    loading=False,
                    action="show",
                    autoClose=5000,
                    position="top-right"
                    #icon=DashIconify(icon="akar-icons:circle-check"),
            )
            return no_update, False, False, False, False, False, notification, no_update
        df = pd.DataFrame(groups['dataframe'])
        if("id" in df.columns):
            # rename column id to ids
            df.rename(columns={"id":"ids"},inplace=True)
        
        # Filter df to only include visible participants
        if visible_participants is not None and visible_participants != []:
            df = df[df['ids'].isin(visible_participants)]
            if len(df) == 0:
                notification = dmc.Notification(
                    id="my-notification",
                    title="Warning",
                    message="No visible participants in selected groups. Please adjust your participant filter.",
                    color="orange",
                    loading=False,
                    action="show",
                    autoClose=5000,
                    position="top-right"
                )
                return no_update, False, False, False, False, False, notification, no_update

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

        # to do: group here, then pass DF on to trainRF - need the grouping in getRFplot again
        notification = create_info_notification("Rendering new plot, please wait.")
        
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
            df.groupby(['group', 'item'])
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
        return RFPlot, False, False, False, False, False, notification, 'rf-plot'
    return no_update,no_update,no_update,no_update,no_update,no_update, no_update, no_update

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
        problematic_items = ['M19', 'J23', 'C14', 'A4', 'E22', 'D12', 'E6']
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
     Output("notify-container", "children",allow_duplicate=True)],
    Input('render-item-plot','n_clicks'),
    [State('participantsTree','checked'),State('grammarItemsTree','checked'),State('items-group-by','value'),State('items-sort-by','value'),State('items-plot-mode','value'),State('grammar-type-switch','checked'),State('use-imputed-data-switch', 'checked')],
    prevent_initial_call=True
)
def renderItemPlot(BTN,informants,items,groupby,sortby,plot_mode,pairs,use_imputed):
    button_clicked = ctx.triggered_id
    if button_clicked == 'render-item-plot' and BTN is not None:
        # to do: group here, then pass DF on to trainRF - need the grouping in getRFplot again
        notification = create_info_notification("Rendering new plot, please wait.")

        # Use lazy data loading - only get data when needed
        if use_imputed:
            if pairs:
                data_source = get_grammar_data_pairs_cached()
            else:
                data_source = get_grammar_data_cached()
        else:
            if pairs:
                data_source = get_grammar_data_pairs_raw_cached()
            else:
                data_source = get_grammar_data_raw_cached()
        
        # to do: merge meta info here for hoverinfo in plot
        # Check if split_by_variety mode is selected
        split_by_variety = (plot_mode == "split_by_variety")
        itemPlot = getItemPlot(informants, items,groupby=groupby,sortby=sortby,pairs=pairs,use_imputed=use_imputed,plot_mode=plot_mode,split_by_variety=split_by_variety)
        return itemPlot, notification
    return no_update,no_update

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
def update_item_tree_based_on_select(checked_items,selected_preset, current_items):
    # check which input triggered the callback
    button_clicked = ctx.triggered_id
    
    # if it was grammarItemsTree, return "Custom"
    if button_clicked == 'grammarItemsTree':
        # If the tree is changed, set the preset to "Custom"
        current_preset = get_matching_preset(checked_items,item_presets)
        return checked_items, current_preset
    if selected_preset is None or selected_preset == "Custom":
        return current_items, no_update

    # Find the preset in item_presets that matches the selected value
    matching_preset = next((preset for preset in item_presets if preset['label'] == selected_preset), None)

    if matching_preset:
        # Return the items associated with the selected preset
        return matching_preset['value'], no_update
    else:
        # If no matching preset is found, return no_update
        return current_items, no_update


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
    [State('last-sociodemographic-settings', 'data')],
    prevent_initial_call=False
)
def auto_update_sociodemographic_plots(active_tab, selected_participants, last_settings):
    """
    Automatically update sociodemographic plots when:
    1. Switching to the Sociodemographic Details tab
    2. Participant selection has changed since last render
    
    Uses smart caching to avoid unnecessary re-renders.
    """
    # Only update when on sociodemographic details tab
    if active_tab != 'sociodemographic-details':
        raise PreventUpdate
    
    # If no participants selected, use all participants
    if not selected_participants:
        selected_participants = Informants['InformantID'].tolist()
    
    # Create current settings hash
    current_settings = {
        'participants': sorted(selected_participants)
    }
    
    # Check if settings have changed (smart caching)
    if last_settings == current_settings:
        # Settings haven't changed, no need to re-render
        raise PreventUpdate
    
    # Settings have changed, render plots
    informants = Informants.loc[Informants['InformantID'].isin(selected_participants), :]
    
    AgeGenderPlot = getAgeGenderPlot(informants)
    MainVarietiesPlot = getMainVarietiesPlot(informants)
    NationalityPlot = getCategoryHistogramPlot(informants, "Nationality", True, "")
    EIDPlot = getCategoryHistogramPlot(informants, "EthnicSelfID", True, "")
    CIDPlot = getCategoryHistogramPlot(informants, "CountryID", True, ",")
    LanguagesHomePlot = getCategoryHistogramPlot(informants, "LanguageHome", True, ",")
    LanguagesMotherPlot = getCategoryHistogramPlot(informants, "LanguageMother", True, ",")
    LanguagesFatherPlot = getCategoryHistogramPlot(informants, "LanguageFather", True, ",")
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
        if (["Slovenia","Germany","Sweden"].includes(mainVariety)) return "EFL";
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

    for preset in item_presets:
        preset_items = preset['value']
        if not preset_items:
            continue  # skip empty presets
        participants_hash = _hash_list(all_informants)
        items_hash = _hash_list(preset_items)
        preset_filename = f"umap_{participants_hash}_{items_hash}_{n_neighbours}_{min_dist}.pkl"
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
     State('leiden-pca-components', 'value')],
    prevent_initial_call=True
)
def run_leiden_clustering(n_clicks, selected_informants, selected_items, resolution, similarity_threshold, 
                         color_by, apply_pca, n_components):
    if n_clicks is None:
        return no_update, no_update, no_update, False, no_update
    
    try:
        # Filter data based on selections
        if selected_informants == ['informants']:
            selected_informants = Informants['InformantID'].tolist()
        if selected_items == ['grammaritems']:
            selected_items = GrammarItemsCols        
        # Get filtered data
        data = retrieve_data.getGrammarData(imputed=True, participants=selected_informants, columns=selected_items)
        informant_data = Informants[Informants['InformantID'].isin(selected_informants)].copy()
        
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
        cluster_stats = results_df.groupby('cluster').agg({
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
    """
    # When checked=True, it's "Mode difference" - disable elements
    # When checked=False, it's "Individual items" - enable elements
    disabled = type_switch_checked
    return disabled, disabled, disabled

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
        checked_items = GrammarItemsCols
    else:
        meta = grammarMetaPairs.copy(deep=True)
        # Mode difference, use pairs
        checked_items = GrammarItemsColsPairs
    pairs = type_switch_checked
    tree_data = drawGrammarItemsTree(meta, pairs=pairs)
    return tree_data, checked_items

# Client-side callback to enable/disable "Add group" button based on UMAP plot selection
clientside_callback(
    """
    function(selectedData) {
        if (selectedData && selectedData.points && selectedData.points.length > 0) {
            return false;  // Enable the button (disabled=false)
        } else {
            return true;   // Disable the button (disabled=true)
        }
    }
    """,
    Output('Umap-add-group', 'disabled', allow_duplicate=True),
    Input('UMAPfig', 'selectedData'),
    prevent_initial_call=True
)
