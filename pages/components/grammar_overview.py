import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

def get_grammar_overview():
    """Return the overview content for the grammar section"""
    return dmc.Container([
        dmc.Title("Getting started", order = 2),
        dmc.Text("This part of the BSLVC dashboard allows you to interactively explore the grammar data of the BSLVC. It is structured into three parts:", mb=20),
        dcc.Markdown("""
        - **Informants**: Explore the sociodemographic participant data included in the BSLVC data.
        - **Grammatical Items**: Analyze grammatical features, their distributions, and compare ratings across groups.
        - **Informant Similarity**: Visualize participant similarity patterns using dimensionality reduction and explore group differences.

        Each section contains specific tools and visualizations tailored to its focus area.

        Below is a brief overview of the individual analysis tools available in each section.
        """),

        dmc.Accordion([
            dmc.AccordionItem([
                dmc.AccordionControl(
                    "Informants",
                    icon=DashIconify(
                        icon="tabler:users",
                        color="#1f77b4",
                        width=20,
                    ),
                ),
                dmc.AccordionPanel([
                    dcc.Markdown("""
**Purpose**: View sociodemographic details of participants who filled in the grammar section of the BSLVC.

**Participant Selection**: You can either directly select participants in the tree-like selection (**1b**), or apply advanced filters based on sociodemographic details in the advanced filter section (**1a**). If advanced filters are used, you need to click "Apply Filters". Note that selection in the item tree will be overwritten when using advanced filters.

**Updating Plot**: After selecting participants, click "Update Plot" (**2**) to refresh the visualizations.
                    """.strip()),
                    html.Img(
                        src="assets/img/gs_grammar_informants_all.png", 
                        alt="Grammar Informants Screenshot",
                        style={"width": "70%", "height": "auto", "display": "block", "margin": "10px 0"}
                    )
                ])
            ], value="informants"),
            
            dmc.AccordionItem([
                dmc.AccordionControl(
                    "Grammatical Items",
                    icon=DashIconify(
                        icon="tabler:list-details",
                        color="#1f77b4",
                        width=20,
                    ),
                ),
                dmc.AccordionPanel([
                    dcc.Markdown("""
**Purpose**: Compare ratings of grammatical items across groups. Explore grammar items in a table.

This section of the app is structured into two parts. In the tab "Item Plot", you can plot the data (see below for the available options). The tab "Grammatical Items" displays the metadata of the items as a table.

**Participant Selection**: You can either directly select participants in the tree-like selection, or apply advanced filters based on sociodemographic details in the advanced filter section. If advanced filters are used, you need to click "Apply Filters". Note that selection in the item tree will be overwritten when using advanced filters.

**Grammar Items**: You can select either individual items in the selection tree or use predefined item sets. You can choose whether you want to visualize ratings of individual items, or the difference between spoken and written responses for the same items (note: not every written item has spoken counterpart). Additionally, you can select whether the analysis should be based on the raw data or on the imputed data set (missing values have been filled in - for more information, see the full documentation).

**Plot settings**: In the plot settings, you can select a plot mode, how the informants should be grouped, and how the items should be sorted in the plot. The available plot modes include:
- "Mean (95% CI)": Visualize the mean ratings with 95% confidence intervals per group.
- "Diverging stacked bars": Plot the percentages of the ratings per group and item as colored, stacked bars. For practical reasons, the maximum number of bars is limited to 275 and a warning will be issued in case you have selected too many items or groups.
- "Informant mean (boxplot)": For all selected items, a mean value per participant will be calculated. The participant means will then be plotted as a boxplot per group.
""".strip()),
                    html.Img(
                        src="assets/img/gs_grammar_grammaritems.png", 
                        alt="Grammar Informants Screenshot",
                        style={"width": "60%", "height": "auto", "display": "inline", "margin": "10px 0"}
                    ),
                    html.Img(
                        src="assets/img/gs_grammar_grammaritems_items.png", 
                        alt="Grammar Informants Screenshot",
                        style={"width": "15%", "height": "auto", "display": "inline", "margin": "10px 0"}
                    ),
                    html.Img(
                        src="assets/img/gs_grammar_grammaritems_plotoptions.png", 
                        alt="Grammar Informants Screenshot",
                        style={"width": "15%", "height": "auto", "display": "inline", "margin": "10px 0"}
                    )
                ])
            ], value="items"),
            
            dmc.AccordionItem([
                dmc.AccordionControl(
                    "Informant Similarity",
                    icon=DashIconify(
                        icon="tabler:chart-scatter",
                        color="#1f77b4",
                        width=20,
                    ),
                ),
                dmc.AccordionPanel([
                    dcc.Markdown("""
**Purpose**: Visualize participant similarity patterns using UMAP dimensionality reduction.

**Functions**:
- Generate UMAP plots showing participant clustering
- Color points by variety, variety type, or gender
- Adjust UMAP parameters (neighbors, distance metrics, standardization)
- Select participant subsets for comparison
- Analyze spoken vs. written response differences
- Filter by grammatical categories or custom item sets
- Apply Leiden clustering algorithm to identify participant groups
                    """.strip())
                ])
            ], value="similarity")
        ], 
        variant="contained",
        radius="md",
        multiple=False,
        ),
        
        dmc.Divider(my="xl"),
        
        dmc.Title("Getting Started", order=3, mb=15),
        dmc.Text("Navigate between tabs to explore different aspects of the grammatical data. Each tab includes filtering options to focus your analysis on specific participant groups or grammatical features. Use the settings panels on the right side of each tab to customize visualizations and apply filters.", mb=10),
        dmc.Text("The dashboard supports both individual item analysis and mode difference analysis (comparing spoken vs. written responses for the same grammatical constructions).", style={"fontStyle": "italic"})
    ], fluid=True)
