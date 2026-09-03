import dash_mantine_components as dmc
from dash import register_page, dcc, html, ctx, callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
import dash_ag_grid as dag
import pandas as pd
import plotly.graph_objects as go

from pages.data.lexicalFunctions import (
    drawParticipantsTreeLexical,
    drawLexicalItemsTree,
    normalize_lexical_tree_selection,
    compute_lexical_data,
    build_lexical_facet_plot,
    compute_average_lexical_data,
    build_average_lexical_plot,
    compute_lexical_heatmap_by_item,
    prepare_lexical_long_data,
    get_x_axis_config,
    build_lexical_raw_export,
    create_export_log_lexical,
)
from pages.data.grammarFunctions import (
    getAgeGenderPlot,
    getMainVarietiesPlot,
    getCategoryHistogramPlot,
    getFloatHistogramPlot,
    create_zip_download,
    remove_sensitive_columns,
)
from pages.data.lexical_background import (
    Informants,
    LexicalRaw,
    LexicalItemsCols,
    LexicalMeta,
    LexicalItemMetaMap,
)

register_page(__name__, path="/lexical", name="Lexical Sets")

persistence_type = "memory"
persist_UI = True

##############
## Data loading
##############

# Informants, LexicalRaw, LexicalItemsCols, LexicalMeta, LexicalItemMetaMap are
# imported from pages.data.lexical_background above (shared with its background callback).

emptyFig = go.Figure()
emptyFig.update_layout(template="simple_white")

##############
## Layout
##############

participantSelectionAccordion = dmc.AccordionItem(
    [
        dmc.AccordionControl(
            dmc.Group([
                DashIconify(icon="tabler:users-group", color="#1f77b4", width=18),
                dmc.Text("Participants", fw=500, size="sm"),
                dmc.Badge("0", id="lexical-participants-badge", color="blue", variant="filled", size="sm"),
            ], gap="xs"),
        ),
        dmc.AccordionPanel(
            dmc.Stack(gap='md', children=[
                dmc.Group(children=[
                    dmc.Button("Select All", id='select-all-lexical-participants', size="xs", variant="outline"),
                    dmc.Button("Deselect All", id='deselect-all-lexical-participants', size="xs", variant="outline"),
                ], mb="xs"),
                dmc.Text(
                    "Only Main Variety, Gender and Year can be selected here (individual participants are not listed due to their number).",
                    size="xs", c="dimmed", style={"fontStyle": "italic"}
                ),
                dmc.Tree(
                    id='participantsTreeLexical',
                    data=drawParticipantsTreeLexical(Informants),
                    checkboxes=True,
                    checked=[],
                ),
            ])
        ),
    ],
    value="LoadDataLexical",
)

itemSelectionAccordion = dmc.AccordionItem(
    [
        dmc.AccordionControl(
            dmc.Group([
                DashIconify(icon="tabler:list-check", color="#2f9e44", width=18),
                dmc.Text("Lexical Items", fw=500, size="sm"),
                dmc.Badge("0", id="lexical-items-badge", color="green", variant="filled", size="sm"),
            ], gap="xs"),
        ),
        dmc.AccordionPanel(
            dmc.Stack(gap='md', children=[
                dmc.Group(children=[
                    dmc.Button("Select All", id='select-all-lexical-items', size="xs", variant="outline"),
                    dmc.Button("Deselect All", id='deselect-all-lexical-items', size="xs", variant="outline"),
                ], mb="xs"),
                html.Div([
                    dmc.Tree(
                        id='lexicalItemsTree',
                        data=drawLexicalItemsTree(LexicalMeta),
                        checkboxes=True,
                        checked=[],
                    )
                ], className="grammar-tree-wrapper", style={"maxHeight": "400px", "overflowY": "auto"}),
            ])
        ),
    ],
    value="LoadItemsLexical",
)

plotSettingsAccordion = dmc.AccordionItem(
    [
        dmc.AccordionControl(
            dmc.Group([
                DashIconify(icon="tabler:settings", width=18),
                dmc.Text("Plot Settings", fw=500, size="sm"),
            ], gap="xs"),
        ),
        dmc.AccordionPanel(
            dmc.Stack(gap='md', children=[
                dmc.Select(
                    label="Plot type:",
                    id="lexical-plot-type",
                    value="facets",
                    data=[
                        {"value": "facets", "label": "Faceted by age group"},
                        {"value": "average", "label": "Averaged across items"},
                        {"value": "heatmap", "label": "Heatmap"},
                    ],
                    size="xs",
                    allowDeselect=False,
                    persistence=persist_UI, persistence_type=persistence_type,
                ),
                dmc.SegmentedControl(
                    id="lexical-time-axis-mode",
                    fullWidth=True,
                    value="apparent_time",
                    data=[
                        {"value": "apparent_time", "label": "Apparent time (age)"},
                        {"value": "birth_year", "label": "Birth year cohorts"},
                    ],
                    size="xs",
                    persistence=persist_UI, persistence_type=persistence_type,
                ),
                dmc.Text(
                    "Birth year = Year of data collection minus Age, grouped into 10-year cohorts.",
                    size="xs", c="dimmed", style={"fontStyle": "italic"}
                ),
                dmc.Switch(
                    id='lexical-exclude-small-birth-cohorts-switch',
                    label="Exclude birth-year cohorts with fewer than 5 datapoints",
                    checked=False,
                    disabled=True,
                    persistence=persist_UI, persistence_type=persistence_type,
                ),
                # Facets-only controls
                html.Div(
                    id='lexical-facets-only-controls',
                    children=[
                        dmc.Select(
                            label="Sort facets by:",
                            id="lexical-sort-by",
                            value="trend",
                            data=[
                                {"value": "trend", "label": "Globalization Trend"},
                                {"value": "average", "label": "Average rating (weighted across gender/variety)"},
                                {"value": "alphabetical", "label": "Alphabetically"},
                            ],
                            size="xs",
                            allowDeselect=False,
                            persistence=persist_UI, persistence_type=persistence_type,
                        ),
                        dmc.NumberInput(
                            label="Facet columns",
                            id="lexical-facet-cols",
                            value=4,
                            min=1,
                            max=8,
                            step=1,
                            size="xs",
                            persistence=persist_UI, persistence_type=persistence_type,
                        ),
                    ],
                    style={"display": "block"}
                ),
                # Facets and averaged common controls
                html.Div(
                    id='lexical-series-controls',
                    children=[
                        dmc.Select(
                            label="Group data by:",
                            id="lexical-series-by",
                            value="variety",
                            data=[
                                {"value": "none", "label": "No grouping"},
                                {"value": "gender", "label": "Gender"},
                                {"value": "variety", "label": "Variety"},
                            ],
                            size="xs",
                            allowDeselect=False,
                            persistence=persist_UI, persistence_type=persistence_type,
                        ),
                    ],
                    style={"display": "block"}
                ),
                # Facets-only CI and opacity
                html.Div(
                    id='lexical-facets-ci-controls',
                    children=[
                        dmc.Switch(
                            id='lexical-show-ci-switch',
                            label="Show 95% confidence bands",
                            checked=True,
                            persistence=persist_UI, persistence_type=persistence_type,
                        ),
                        dmc.Switch(
                            id='lexical-nx-opacity-switch',
                            label="Fade markers by share of 'neither' (NX) responses",
                            description="Higher NX share \u2192 more transparent marker",
                            checked=True,
                            persistence=persist_UI, persistence_type=persistence_type,
                        ),
                    ],
                    style={"display": "block"}
                ),
                # Heatmap-specific controls
                html.Div(
                    id='lexical-heatmap-controls',
                    children=[
                        dmc.Switch(
                            id='lexical-heatmap-color-by-trend-switch',
                            label="Color cells by trend (slope) instead of mean value",
                            description="Red = downward trend, Blue = upward trend",
                            checked=False,
                            persistence=persist_UI, persistence_type=persistence_type,
                        ),
                        dmc.Select(
                            label="Sort items by:",
                            id="lexical-heatmap-sort-items",
                            value="average",
                            data=[
                                {"value": "alphabetically", "label": "Alphabetically"},
                                {"value": "average", "label": "Average (default)"},
                                {"value": "slope", "label": "Slope", "disabled": True},
                            ],
                            size="xs",
                            allowDeselect=False,
                            persistence=persist_UI, persistence_type=persistence_type,
                        ),
                    ],
                    style={"display": "none"}
                ),
            ])
        ),
    ],
    value="PlotSettingsLexical",
)

advancedActionsAccordion = dmc.Accordion(
    children=[
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Group([
                        DashIconify(icon="tabler:settings", width=18),
                        dmc.Text("Advanced Actions", size="sm", fw=500),
                    ], gap="xs"),
                ),
                dmc.AccordionPanel(
                    dmc.Stack(gap="xs", children=[
                        dmc.Checkbox(
                            id='lexical-export-include-sociodemographic-checkbox',
                            label="Include sociodemographic data",
                            checked=True,
                            size="xs",
                        ),
                        dmc.Button(
                            "Export Raw Data",
                            id='lexical-export-raw-data-button',
                            size="xs",
                            variant="light",
                            leftSection=DashIconify(icon="tabler:table-export", width=14),
                            fullWidth=True,
                        ),
                        dmc.Button(
                            "Download aggregated data",
                            id='download-lexical-aggregated-data-button',
                            size="xs",
                            variant="light",
                            leftSection=DashIconify(icon="tabler:chart-bar", width=14),
                            fullWidth=True,
                        ),
                    ])
                ),
            ],
            value="advanced-actions",
        ),
    ],
    variant="contained",
    radius="md",
    mb="xs",
)

SettingsLexicalAnalysis = dmc.Card([
    dmc.Accordion(
        children=[
            dmc.AccordionItem([
                dmc.AccordionControl("Selection Overview"),
                dmc.AccordionPanel(
                    html.Div(id="lexical-stats-summary", children=[
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

    dmc.Box(
        pos="relative",
        mb="md",
        children=[
            dmc.Button(
                'Render Plot',
                id='render-lexical-plot',
                size="md",
                leftSection=DashIconify(icon="tabler:chart-line", width=20),
                color="blue",
                fullWidth=True,
            ),
        ]
    ),

    dmc.Accordion(children=[
        participantSelectionAccordion,
        itemSelectionAccordion,
        plotSettingsAccordion,
        advancedActionsAccordion,
    ],
    variant="contained",
    radius="md",
    mb="md",
    value=["LoadDataLexical", "LoadItemsLexical"]),

], withBorder=True, shadow="sm", radius="md", p="sm", style={"height": "calc(100vh - 110px)", "overflowY": "auto"})


LexicalPlotContainer = dmc.Container([
    dmc.Group([
        dmc.Button(
            "Download plot data",
            id='download-lexical-plot-data-button',
            size="xs",
            variant="light",
            leftSection=DashIconify(icon="tabler:download", width=14),
        ),
    ], justify="flex-end", mt="xs", mb="xs"),
    dcc.Download(id="download-lexical-plot-data"),
    dcc.Download(id="download-lexical-aggregated-data"),
    dcc.Download(id="download-lexical-raw-data"),
    dmc.Grid(children=[
        dmc.GridCol(children=[
            html.Div(
                dcc.Graph(id="lexical-facet-fig", figure=emptyFig, config={
                    'toImageButtonOptions': {
                        'format': 'svg',
                        'filename': 'lexical_facet_plot',
                        'scale': 1,
                    }
                }),
                # Figure has a fixed intrinsic width (facets capped at 350px each), so this
                # wrapper scrolls both ways instead of squeezing/stretching facets.
                style={'maxHeight': 'calc(100vh - 220px)', 'overflowY': 'auto', 'maxWidth': "100%"},
            ),
        ], span=12),
    ])
], fluid=True)

##############
## Sociodemographic Details tab (mirrored from the grammar module)
##############

lexicalAgeGender = dmc.Stack([
    dmc.Text("Age/Gender"),
    dcc.Graph(id="lexical-AgeGenderPlotG", figure=getAgeGenderPlot(Informants), style={'height': '200px'})
])

lexicalMainVarieties = dmc.Stack([
    dmc.Text("Main varieties"),
    html.Div(id="lexical-MainVarietiesPlotContainer", children=[
        dcc.Graph(id="lexical-MainVarietiesPlotG", figure=getMainVarietiesPlot(Informants))
    ], style={'height': 'auto', 'max-height': '300px', 'overflowY': 'scroll'}),
])


def _lexical_histogram_stack(title, graph_id, col, split=""):
    return dmc.Stack([
        dmc.Text(title),
        html.Div(id=f"{graph_id}Container", children=[
            dcc.Graph(id=graph_id, figure=getCategoryHistogramPlot(Informants, col, True, split))
        ], style={'height': 'auto', 'max-height': '300px', 'overflowY': 'scroll'}),
    ])


lexicalNationality = _lexical_histogram_stack("Nationality", "lexical-NationalityPlotG", "Nationality")
lexicalEthnicSID = _lexical_histogram_stack("Ethnic Self-ID", "lexical-EIDPlotG", "EthnicSelfID")
lexicalCountryID = _lexical_histogram_stack("Country (or region) you identify with most", "lexical-CIDPlotG", "CountryID")
lexicalLanguagesHome = _lexical_histogram_stack("Languages used at home", "lexical-LanguagesHomePlotG", "LanguageHome", ",")
lexicalLanguageMother = _lexical_histogram_stack("Mother's Native Language", "lexical-LanguagesMotherPlotG", "LanguageMother", ",")
lexicalLanguageFather = _lexical_histogram_stack("Father's Native Language", "lexical-LanguagesFatherPlotG", "LanguageFather", ",")

lexicalPrimarySchool = dmc.Stack([
    dmc.Text("Primary School"),
    dcc.Graph(id="lexical-PrimarySchoolPlotG", figure=getCategoryHistogramPlot(Informants, "PrimarySchool", True))
])
lexicalSecondarySchool = dmc.Stack([
    dmc.Text("Secondary School"),
    dcc.Graph(id="lexical-SecondarySchoolPlotG", figure=getCategoryHistogramPlot(Informants, "SecondarySchool", True))
])
lexicalQualifications = dmc.Stack([
    dmc.Text("Highest Qualification"),
    dcc.Graph(id="lexical-QualiPlotG", figure=getCategoryHistogramPlot(Informants, "Qualifications", True))
])

lexicalYearsLivedOutside = dmc.Stack([
    dmc.Text("Years lived outside home country"),
    dcc.Graph(id="lexical-YLOPlotG", figure=getFloatHistogramPlot(Informants, "YearsLivedOutside"))
])
lexicalYearsLivedOtherE = dmc.Stack([
    dmc.Text("Years lived in other English-speaking countries"),
    dcc.Graph(id="lexical-YLOEPlotG", figure=getFloatHistogramPlot(Informants, "YearsLivedOtherEnglish"))
])
lexicalRatioMainVariety = dmc.Stack([
    dmc.Text("Ratio Main Variety"),
    dcc.Graph(id="lexical-RatioMainVarietyPlotG", figure=getFloatHistogramPlot(Informants, "RatioMainVariety"))
])

lexicalPIAccordion = dmc.Accordion(
    children=[
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Languages"),
                dmc.AccordionPanel(children=[
                    lexicalLanguagesHome, dmc.Divider(), lexicalLanguageMother, dmc.Divider(), lexicalLanguageFather,
                ]),
            ],
            value="languages",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Regions & Identification"),
                dmc.AccordionPanel(children=[
                    lexicalNationality, dmc.Divider(), lexicalEthnicSID, dmc.Divider(), lexicalCountryID,
                ]),
            ],
            value="seldif",
        ),
    ],
    variant="default",
)

lexicalInformantsColumnDefaults = ['Age', 'Gender', 'MainVariety', 'MainVariety_Original', 'AdditionalVarieties',
                                   'YearsLivedInMainVariety', 'RatioMainVariety', 'CountryCollection', 'Year',
                                   'Nationality', 'EthnicSelfID', 'CountryID', 'YearsLivedOutside',
                                   'YearsLivedInside', 'YearsLivedOtherEnglish', 'LanguageHome',
                                   'LanguageFather', 'LanguageMother', 'Qualifications',
                                   'QualiMother', 'QualiFather', 'QualiPartner',
                                   'PrimarySchool', 'SecondarySchool', 'Occupation', 'OccupMother', 'OccupFather',
                                   'OccupPartner']

LexicalInformantsGrid = html.Div(children=[
    dmc.Group([
        dmc.SegmentedControl(
            id="lexical-informants-view-toggle",
            data=[
                {"value": "plots", "label": "Plot"},
                {"value": "table", "label": "Table"},
            ],
            value="table",
            color="blue",
            size="sm"
        ),
    ], justify="center", mb="lg"),

    html.Div(
        id="lexical-informants-table-view",
        children=[
            dmc.Stack([
                dmc.Accordion(
                    children=[
                        dmc.AccordionItem(
                            [
                                dmc.AccordionControl("Select Columns to Display", style={"fontSize": "14px"}),
                                dmc.AccordionPanel(
                                    dmc.CheckboxGroup(
                                        id="lexical-informants-columns-checkbox",
                                        children=dmc.Grid([
                                            dmc.GridCol(dmc.Checkbox(
                                                label=col.replace("_", " ").replace("ID", " ID"), value=col, size="xs"
                                            ), span=3)
                                            for col in lexicalInformantsColumnDefaults if col in Informants.columns
                                        ]),
                                        value=['Age', 'Gender', 'MainVariety', 'Year'],
                                        persistence=persist_UI, persistence_type=persistence_type,
                                    )
                                ),
                            ],
                            value="column-selection",
                        ),
                    ],
                    variant="contained", mb="xs",
                ),
            ], gap="xs", mb="xs"),
            dag.AgGrid(
                id="lexical-informants-table",
                rowData=[],
                columnDefs=[
                    {"field": col, "headerName": col.replace("ID", " ID").replace("_", " "),
                     "filter": "agTextColumnFilter", "sortable": True, "resizable": True, "flex": 1}
                    for col in ['InformantID', 'Age', 'Gender', 'MainVariety', 'Year'] if col in Informants.columns
                ],
                defaultColDef={"filter": "agTextColumnFilter", "sortable": True, "resizable": True, "flex": 1},
                className="ag-theme-quartz compact",
                columnSize="autoSize",
                dashGridOptions={"pagination": True, "paginationPageSize": 30, "animateRows": True},
                style={"height": "calc(100vh - 300px)", "width": "100%"}
            )
        ],
        style={"display": "block", "height": "calc(100vh - 300px)", "overflowY": "auto"}
    ),
    html.Div(
        id="lexical-informants-plot-view",
        children=[
            dmc.Card(children=[
                dmc.Grid(children=[
                    dmc.GridCol(children=[
                        dmc.Card(children=[
                            dmc.Text("Personal Information", fw=700),
                            lexicalAgeGender, dmc.Divider(), lexicalMainVarieties, dmc.Divider(), lexicalPIAccordion,
                        ], withBorder=True, shadow="sm", radius="md"),
                        dmc.Card(children=[
                            dmc.Text("Location Timeline", fw=500),
                            lexicalYearsLivedOutside, dmc.Divider(), lexicalYearsLivedOtherE, dmc.Divider(), lexicalRatioMainVariety,
                        ], withBorder=True, shadow="sm", radius="md"),
                    ], span=6),
                    dmc.GridCol(children=[
                        dmc.Card(children=[
                            dmc.Text("Education Profile", fw=500),
                            lexicalPrimarySchool, dmc.Divider(), lexicalSecondarySchool, dmc.Divider(), lexicalQualifications,
                        ], withBorder=True, shadow="sm", radius="md"),
                    ], span=6),
                ])
            ], withBorder=True, shadow="sm", radius="md", style={"maxHeight": "calc(100vh - 300px)", "overflowY": "auto"})
        ],
        style={"display": "none"}
    )
])

lexicalAnalysisC = dmc.Grid([
    dmc.GridCol(html.Div(children=[
        # Collapse toggle button
        dmc.ActionIcon(
            DashIconify(icon="tabler:layout-sidebar-right-collapse", width=20),
            id="toggle-lexical-sidebar-button",
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
                    dmc.TabsList([
                        dmc.TabsTab("Plot View", value="plot-view"),
                        dmc.TabsTab("Sociodemographic Details", value="sociodemographic-details"),
                    ]),
                    dmc.TabsPanel(LexicalPlotContainer, value="plot-view"),
                    dmc.TabsPanel(LexicalInformantsGrid, value="sociodemographic-details"),
                ],
                id="lexical-analysis-tabs",
                color="blue",
                value="plot-view",
            ),
        ], withBorder=True, shadow="sm", radius="md", style={"height": "calc(100vh - 110px)", "overflowY": "auto"})],
        id="lexical-analysis-tab-content",
        style={"height": "calc(100vh - 100px)"}),
        id="lexical-main-content-col",
        span=8,
    ),
    dmc.GridCol(
        SettingsLexicalAnalysis,
        id="lexical-sidebar-col",
        span=4,
        style={}
    ),
], gutter="xl", id="lexical-analysis-grid")

layout = dmc.Container([
    dcc.Store(id="lexical-informants-store", data=Informants.to_dict("records")),
    dcc.Store(id="lexical-render-trigger", storage_type="memory"),  # Trigger for background plot computation
    lexicalAnalysisC,
], fluid=True)


##############
## Callbacks
##############

@callback(
    Output('participantsTreeLexical', 'checked', allow_duplicate=True),
    [Input('select-all-lexical-participants', 'n_clicks'),
     Input('deselect-all-lexical-participants', 'n_clicks')],
    prevent_initial_call=True,
)
def update_lexical_participants_selection(select_clicks, deselect_clicks):
    button_clicked = ctx.triggered_id
    if button_clicked == 'select-all-lexical-participants':
        return ['participantslexical']
    elif button_clicked == 'deselect-all-lexical-participants':
        return []
    return no_update


@callback(
    Output('lexicalItemsTree', 'checked', allow_duplicate=True),
    [Input('select-all-lexical-items', 'n_clicks'),
     Input('deselect-all-lexical-items', 'n_clicks')],
    prevent_initial_call=True,
)
def update_lexical_items_selection(select_clicks, deselect_clicks):
    button_clicked = ctx.triggered_id
    if button_clicked == 'select-all-lexical-items':
        return ['lexicalitems']
    elif button_clicked == 'deselect-all-lexical-items':
        return []
    return no_update


@callback(
    [Output('lexical-stats-summary', 'children'),
     Output('lexical-participants-badge', 'children'),
     Output('lexical-items-badge', 'children')],
    [Input('participantsTreeLexical', 'checked'),
     Input('lexicalItemsTree', 'checked')],
    State('lexical-informants-store', 'data'),
    prevent_initial_call=False,
)
def update_lexical_quick_stats(selected_participants, selected_items, informants_data):
    current_informants = pd.DataFrame(informants_data) if informants_data else Informants
    participants, items = normalize_lexical_tree_selection(
        selected_participants, selected_items, current_informants, LexicalItemsCols
    )

    n_participants = len(participants) if selected_participants else 0
    n_items = len(items) if selected_items else 0
    total_participants = len(current_informants)
    total_items = len(LexicalItemsCols)

    participant_badge = f"{n_participants}/{total_participants}"
    items_badge = f"{n_items}/{total_items}"

    if not selected_participants or not selected_items:
        return dmc.Text("Select participants and items to begin", size="sm", c="dimmed"), participant_badge, items_badge

    # Mirrors the grammar module's Selection Overview summary
    participant_data = current_informants[current_informants['InformantID'].isin(participants)]

    variety_counts = participant_data['MainVariety'].value_counts()
    variety_text = ", ".join([f"{variety} ({count})" for variety, count in variety_counts.items()])

    gender_counts = participant_data['Gender'].value_counts()
    gender_map = {'f': 'F', 'm': 'M', 'female': 'F', 'male': 'M', 'nb': 'NB', 'non-binary': 'NB'}
    gender_summary = {}
    for gender, count in gender_counts.items():
        mapped = gender_map.get(gender, gender)
        gender_summary[mapped] = gender_summary.get(mapped, 0) + count
    gender_text = ", ".join([f"{gender}: {count}" for gender, count in sorted(gender_summary.items())])

    ages = participant_data['Age'].dropna()
    if len(ages) > 0:
        age_text = f"Age: {int(ages.min())}-{int(ages.max())} (median: {int(ages.median())})"
    else:
        age_text = "Age: N/A"

    summary = dmc.Stack([
        dmc.Text(f"\U0001F465 {n_participants} participants, \U0001F4CB {n_items} items", size="sm", fw=500),
        dmc.Text(variety_text, size="xs", c="dimmed"),
        dmc.Text(f"{gender_text} \u2022 {age_text}", size="xs", c="dimmed"),
    ], gap="2px")

    return summary, participant_badge, items_badge


def _resolve_lexical_selection(selected_participants, selected_items, informants_data):
    current_informants = pd.DataFrame(informants_data) if informants_data else Informants
    participants, items = normalize_lexical_tree_selection(
        selected_participants, selected_items, current_informants, LexicalItemsCols
    )
    return current_informants, participants, items


@callback(
    [Output('render-lexical-plot', 'loading', allow_duplicate=True),
     Output('lexical-render-trigger', 'data')],
    Input('render-lexical-plot', 'n_clicks'),
    prevent_initial_call=True,
)
def initiate_lexical_render(n_clicks):
    """Set the loading state immediately (instant) and hand off the actual
    computation to the background callback in pages/data/lexical_background.py,
    so the UI doesn't hang while a large facet plot is being built."""
    import time
    return True, {'timestamp': time.time()}


@callback(
    Output('render-lexical-plot', 'loading', allow_duplicate=True),
    Input('lexical-facet-fig', 'figure'),
    prevent_initial_call=True,
)
def handle_lexical_render_completion(figure):
    """Clear the loading state once the background callback has updated the figure."""
    return False


@callback(
    Output('lexical-exclude-small-birth-cohorts-switch', 'disabled'),
    Input('lexical-time-axis-mode', 'value'),
)
def toggle_exclude_small_cohorts_switch(time_axis_mode):
    """Only usable when the x-axis is birth-year cohorts (cohort sizes are irrelevant for fixed age bins)."""
    return time_axis_mode != 'birth_year'


@callback(
    [Output('lexical-facets-only-controls', 'style'),
     Output('lexical-series-controls', 'style'),
     Output('lexical-facets-ci-controls', 'style'),
     Output('lexical-heatmap-controls', 'style')],
    Input('lexical-plot-type', 'value'),
)
def toggle_plot_type_controls(plot_type):
    """Show/hide control groups based on selected plot type."""
    # facets-only: Sort facets, Facet columns
    facets_only_show = "block" if plot_type == "facets" else "none"
    
    # series controls: Group data by (shown for facets and average)
    series_show = "block" if plot_type in ["facets", "average"] else "none"
    
    # CI and opacity: shown for facets and average
    ci_show = "block" if plot_type in ["facets", "average"] else "none"
    
    # heatmap-specific: Color by trend
    heatmap_show = "block" if plot_type == "heatmap" else "none"
    
    return (
        {"display": facets_only_show},
        {"display": series_show},
        {"display": ci_show},
        {"display": heatmap_show},
    )


@callback(
    Output('lexical-time-axis-mode', 'disabled'),
    [Input('lexical-plot-type', 'value'),
     Input('lexical-heatmap-color-by-trend-switch', 'checked')],
)
def toggle_time_axis_mode_availability(plot_type, color_by_trend):
    """
    Enable/disable time axis mode selection based on plot type and heatmap coloring mode.
    
    - Facets & Average: always enabled (time mode affects all plots)
    - Heatmap without trend: disabled (we aggregate across all ages)
    - Heatmap with trend: enabled (trend is computed across age groups)
    """
    if plot_type != "heatmap":
        # Facets and average plots always use time axis
        return False
    
    # For heatmap: only enabled when computing trends
    return not color_by_trend


@callback(
    Output('lexical-heatmap-sort-items', 'data'),
    Input('lexical-heatmap-color-by-trend-switch', 'checked'),
)
def update_heatmap_sort_options(color_by_trend):
    """Enable/disable the 'Slope' sort option based on whether trend coloring is enabled."""
    base_options = [
        {"value": "alphabetically", "label": "Alphabetically"},
        {"value": "average", "label": "Average (default)"},
    ]
    
    if color_by_trend:
        # Trend mode: enable slope sorting
        base_options.append({"value": "slope", "label": "Slope"})
    else:
        # No trend mode: keep slope disabled
        base_options.append({"value": "slope", "label": "Slope", "disabled": True})
    
    return base_options




@callback(
    Output('download-lexical-plot-data', 'data'),
    Input('download-lexical-plot-data-button', 'n_clicks'),
    [State('participantsTreeLexical', 'checked'),
     State('lexicalItemsTree', 'checked'),
     State('lexical-time-axis-mode', 'value'),
     State('lexical-exclude-small-birth-cohorts-switch', 'checked'),
     State('lexical-informants-store', 'data')],
    prevent_initial_call=True,
)
def download_lexical_plot_data(n_clicks, selected_participants, selected_items, time_axis_mode,
                                exclude_small_cohorts, informants_data):
    current_informants, participants, items = _resolve_lexical_selection(
        selected_participants, selected_items, informants_data
    )
    if not participants or not items:
        return no_update

    long_df, _, _, _ = compute_lexical_data(
        lexical_raw=LexicalRaw, informants=current_informants,
        participant_ids=participants, items=items, mode=time_axis_mode,
        exclude_small_cohorts=bool(exclude_small_cohorts),
    )
    if long_df.empty:
        return no_update

    log_content = create_export_log_lexical("Plot data (long format)", participants, items, long_df, mode=time_axis_mode)
    return create_zip_download("lexical_plot_data", long_df.to_csv(index=False), log_content)


@callback(
    Output('download-lexical-aggregated-data', 'data'),
    Input('download-lexical-aggregated-data-button', 'n_clicks'),
    [State('participantsTreeLexical', 'checked'),
     State('lexicalItemsTree', 'checked'),
     State('lexical-series-by', 'value'),
     State('lexical-time-axis-mode', 'value'),
     State('lexical-exclude-small-birth-cohorts-switch', 'checked'),
     State('lexical-sort-by', 'value'),
     State('lexical-informants-store', 'data')],
    prevent_initial_call=True,
)
def download_lexical_aggregated_data(n_clicks, selected_participants, selected_items, series_by,
                                      time_axis_mode, exclude_small_cohorts, sort_by, informants_data):
    current_informants, participants, items = _resolve_lexical_selection(
        selected_participants, selected_items, informants_data
    )
    if not participants or not items:
        return no_update

    _, agg_df, ordered_items, _ = compute_lexical_data(
        lexical_raw=LexicalRaw, informants=current_informants,
        participant_ids=participants, items=items, mode=time_axis_mode,
        series_by=series_by, sort_by=sort_by, item_meta_map=LexicalItemMetaMap,
        exclude_small_cohorts=bool(exclude_small_cohorts),
    )
    if agg_df.empty:
        return no_update
    agg_df = agg_df[agg_df['Item'].isin(ordered_items)]

    log_content = create_export_log_lexical("Aggregated data", participants, items, agg_df, mode=time_axis_mode,
                                              series_by=series_by, sort_by=sort_by)
    return create_zip_download("lexical_aggregated_data", agg_df.to_csv(index=False), log_content)


@callback(
    Output('download-lexical-raw-data', 'data'),
    Input('lexical-export-raw-data-button', 'n_clicks'),
    [State('participantsTreeLexical', 'checked'),
     State('lexicalItemsTree', 'checked'),
     State('lexical-export-include-sociodemographic-checkbox', 'checked'),
     State('lexical-informants-store', 'data')],
    prevent_initial_call=True,
)
def download_lexical_raw_data(n_clicks, selected_participants, selected_items, include_sociodem, informants_data):
    current_informants, participants, items = _resolve_lexical_selection(
        selected_participants, selected_items, informants_data
    )
    if not participants or not items:
        return no_update

    result = build_lexical_raw_export(LexicalRaw, current_informants, participants, items,
                                       include_sociodem=bool(include_sociodem))
    result = remove_sensitive_columns(result)
    if result.empty:
        return no_update

    log_content = create_export_log_lexical(
        "Raw data (as stored in the database)", participants, items, result,
        extra_lines=[f"Include Sociodemographic Data: {'Yes' if include_sociodem else 'No'}"],
    )
    return create_zip_download("lexical_raw_data", result.to_csv(index=False), log_content)


@callback(
    [
        Output('lexical-AgeGenderPlotG', 'figure'),
        Output('lexical-MainVarietiesPlotG', 'figure'),
        Output('lexical-NationalityPlotG', 'figure'),
        Output('lexical-EIDPlotG', 'figure'),
        Output('lexical-CIDPlotG', 'figure'),
        Output('lexical-LanguagesHomePlotG', 'figure'),
        Output('lexical-LanguagesMotherPlotG', 'figure'),
        Output('lexical-LanguagesFatherPlotG', 'figure'),
        Output('lexical-PrimarySchoolPlotG', 'figure'),
        Output('lexical-SecondarySchoolPlotG', 'figure'),
        Output('lexical-QualiPlotG', 'figure'),
        Output('lexical-YLOPlotG', 'figure'),
        Output('lexical-YLOEPlotG', 'figure'),
        Output('lexical-RatioMainVarietyPlotG', 'figure'),
        Output('lexical-informants-table', 'rowData'),
    ],
    [
        Input('lexical-analysis-tabs', 'value'),
        Input('participantsTreeLexical', 'checked'),
    ],
    State('lexical-informants-store', 'data'),
    prevent_initial_call=False,
)
def auto_update_lexical_sociodemographic(active_tab, selected_participants, informants_data):
    """Mirrors grammar.py's sociodemographic tab: populate on tab switch / participant selection change."""
    if active_tab != 'sociodemographic-details':
        raise PreventUpdate

    current_informants = pd.DataFrame(informants_data) if informants_data else Informants
    participants, _ = normalize_lexical_tree_selection(
        selected_participants, ['lexicalitems'], current_informants, LexicalItemsCols
    )

    if not selected_participants or not participants:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="simple_white")
        return tuple([empty_fig] * 14) + ([],)

    informants = current_informants.loc[current_informants['InformantID'].isin(participants), :]

    return (
        getAgeGenderPlot(informants),
        getMainVarietiesPlot(informants),
        getCategoryHistogramPlot(informants, "Nationality", True, ""),
        getCategoryHistogramPlot(informants, "EthnicSelfID", True, ""),
        getCategoryHistogramPlot(informants, "CountryID", True, ","),
        getCategoryHistogramPlot(informants, "LanguageHome", True, ","),
        getCategoryHistogramPlot(informants, "LanguageMother", True, ","),
        getCategoryHistogramPlot(informants, "LanguageFather", True, ","),
        getCategoryHistogramPlot(informants, "PrimarySchool", True),
        getCategoryHistogramPlot(informants, "SecondarySchool", True),
        getCategoryHistogramPlot(informants, "Qualifications", True),
        getFloatHistogramPlot(informants, "YearsLivedOutside"),
        getFloatHistogramPlot(informants, "YearsLivedOtherEnglish"),
        getFloatHistogramPlot(informants, "RatioMainVariety"),
        informants.to_dict("records"),
    )


@callback(
    [Output("lexical-informants-table-view", "style"),
     Output("lexical-informants-plot-view", "style")],
    Input("lexical-informants-view-toggle", "value"),
    prevent_initial_call=False,
)
def toggle_lexical_informants_view(view_mode):
    if view_mode == "table":
        return {"display": "flex", "flex-direction": "column"}, {"display": "none"}
    return {"display": "none"}, {"display": "flex", "flex-direction": "column"}


@callback(
    [Output('lexical-sidebar-col', 'span'),
     Output('lexical-sidebar-col', 'style'),
     Output('lexical-main-content-col', 'span'),
     Output('toggle-lexical-sidebar-button', 'children')],
    Input('toggle-lexical-sidebar-button', 'n_clicks'),
    State('lexical-sidebar-col', 'span'),
    prevent_initial_call=True,
)
def toggle_lexical_sidebar(n_clicks, current_span):
    """Toggle sidebar between visible (span=4) and hidden (span=0), mirroring the grammar module."""
    if current_span == 4:
        return (
            0,
            {"display": "none"},
            12,
            DashIconify(icon="tabler:layout-sidebar-right-expand", width=20)
        )
    return (
        4,
        {"height": "calc(100vh - 100px)"},
        8,
        DashIconify(icon="tabler:layout-sidebar-right-collapse", width=20)
    )


@callback(
    Output("lexical-informants-table", "columnDefs"),
    Input("lexical-informants-columns-checkbox", "value"),
    prevent_initial_call=False,
)
def update_lexical_informants_table_columns(selected_columns):
    selected_columns = selected_columns or []
    columns_to_show = ['InformantID'] + [col for col in selected_columns if col != 'InformantID']
    available_columns = [col for col in columns_to_show if col in Informants.columns]
    return [
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