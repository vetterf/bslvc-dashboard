import dash_mantine_components as dmc
from dash import register_page, html, dcc, callback, Input, Output
from dash_iconify import DashIconify
import os
from pages.components.case_study_1 import create_case_study_1, get_case_study_1_step_content
from pages.components.case_study_2 import create_case_study_2, get_case_study_2_step_content

register_page(__name__, path="/getting-started", name="Getting Started")

# Load markdown content
def load_markdown(filename):
    """Load markdown file from assets directory"""
    filepath = os.path.join(os.path.dirname(__file__), '..', 'assets', filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: Could not find {filename}"

# Helper function to create UI location visualizations
def create_ui_location(highlight_section="", highlight_element="", highlight_tab="", highlight_nav="", highlight_right=False):
    """Create a visual representation of the UI with optional highlighting
    
    Args:
        highlight_section: Which section to highlight ("left", "middle", "right")
        highlight_element: Which element in the options panel to highlight 
                          ("render", "mode", "participants", "grammar", "plot", "actions")
        highlight_tab: Which tab in the main view to highlight
                      ("plot", "sociodemographic", "grammar_items")
        highlight_nav: Which navigation item to highlight
                      ("grammar")
    """
    # Set consistent height for all panels
    panel_height = "180px"
    
    left_style = {"backgroundColor": "#e7f5ff" if highlight_section == "left" else "#f8f9fa", "padding": "8px", "minHeight": panel_height}
    middle_style = {"backgroundColor": "#e7f5ff" if highlight_section == "middle" else "#f8f9fa", "padding": "8px", "minHeight": panel_height}
    right_style = {"backgroundColor": "#f8f9fa", "padding": "8px", "minHeight": panel_height}
    
    # Initialize all border variables
    left_border = middle_border = "1px solid #dee2e6"
    right_border = "1px solid #dee2e6"
    # Conditionally highlight the right/options panel
    if highlight_right:
        right_style["backgroundColor"] = "#e7f5ff"
        right_border = "2px solid #228be6"
    if highlight_section == "left":
        left_border = "2px solid #228be6"
    elif highlight_section == "middle":
        middle_border = "2px solid #228be6"
    
    # Create navigation menu items - only Grammar Sets has text
    nav_items = []
    
    # Add blank navigation items (just colored blocks)
    for i in range(2):
        nav_items.append(
            html.Div(style={
                "backgroundColor": "#e9ecef",
                "height": "18px",
                "borderRadius": "4px",
                "marginBottom": "4px"
            })
        )
    
    # Add Grammar Sets with text (highlighted if active)
    is_grammar_highlighted = highlight_nav == "grammar"
    nav_items.append(
        dmc.Group([
            DashIconify(icon="tabler:language", width=10, color="white" if is_grammar_highlighted else "#495057"),
            dmc.Text("Grammar Sets", size="xs", c="white" if is_grammar_highlighted else "dark", fw=600 if is_grammar_highlighted else 400)
        ], gap="xs", style={
            "backgroundColor": "#228be6" if is_grammar_highlighted else "#e9ecef",
            "padding": "4px 6px",
            "borderRadius": "4px",
            "marginBottom": "4px"
        })
    )
    
    # Add more blank navigation items
    for i in range(2):
        nav_items.append(
            html.Div(style={
                "backgroundColor": "#e9ecef",
                "height": "18px",
                "borderRadius": "4px",
                "marginBottom": "4px"
            })
        )
    
    # Navigation panel content
    nav_content = dmc.Stack(nav_items, gap=0, justify="flex-start")
    
    # Create main view tabs - using actual tab names from interface
    main_view_tabs = []
    tab_configs = [
        ("plot", "Plot View", "tabler:chart-dots"),
        ("sociodemographic", "Sociodemographic Details", "tabler:users-group"),
        ("grammar_items", "Grammar Items", "tabler:list-check")
    ]
    
    for tab_id, tab_label, tab_icon in tab_configs:
        is_highlighted = highlight_tab == tab_id
        tab_style = {
            "backgroundColor": "#228be6" if is_highlighted else "#dee2e6",
            "padding": "4px 8px",
            "borderRadius": "6px 6px 0 0",
            "marginRight": "2px",
            "border": f"2px solid {'#228be6' if is_highlighted else '#dee2e6'}",
            "borderBottom": "none"
        }
        main_view_tabs.append(
            html.Div(
                dmc.Group([
                    DashIconify(icon=tab_icon, width=10, color="white" if is_highlighted else "#495057"),
                    dmc.Text(tab_label, size="xs", c="white" if is_highlighted else "dark", fw=600 if is_highlighted else 400)
                ], gap="xs"),
                style=tab_style
            )
        )
    
    # Main view content area (below tabs)
    content_area_style = {
        "backgroundColor": "#f8f9fa",
        "border": "2px solid #dee2e6",
        "borderRadius": "0 6px 6px 6px",
        "padding": "8px",
        "height": "120px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center"
    }
    
    # Main view content
    main_view_content = dmc.Stack([
        dmc.Group(main_view_tabs, gap=0, align="flex-end"),
        html.Div(
            dmc.Center(
                DashIconify(icon="tabler:chart-line", width=24, color="#228be6" if highlight_section == "middle" else "#adb5bd"),
            ),
            style=content_area_style
        )
    ], gap=0, justify="flex-start")
    
    # Create options panel elements - using actual icons from interface
    options_elements = []
    element_configs = [
        ("render", "Render Plot", "tabler:chart-line", "blue"),
        ("mode", "Analysis Mode", "tabler:cluster", "violet"),
        ("participants", "Participants", "tabler:users-group", "blue"),
        ("grammar", "Grammar Items", "tabler:list-check", "teal"),
        ("plot", "Plot Options", "tabler:settings", "orange"),
        ("actions", "Advanced Actions", "tabler:settings", "grape")
    ]
    
    for elem_id, elem_label, elem_icon, elem_color in element_configs:
        is_highlighted = highlight_element == elem_id
        elem_style = {
            "backgroundColor": "#228be6" if is_highlighted else "#e9ecef",
            "padding": "3px 6px",
            "borderRadius": "4px",
            "marginBottom": "3px"
        }
        options_elements.append(
            dmc.Group([
                DashIconify(icon=elem_icon, width=10, color="white" if is_highlighted else elem_color),
                dmc.Text(elem_label, size="xs", c="white" if is_highlighted else "dark", fw=600 if is_highlighted else 400)
            ], gap="xs", style=elem_style)
        )
    
    return dmc.Grid([
        dmc.GridCol([
            dmc.Paper([
                nav_content
            ], withBorder=True, p="xs", radius="md", style={**left_style, "border": left_border})
        ], span=2),
        dmc.GridCol([
            dmc.Paper([
                main_view_content
            ], withBorder=True, p="sm", radius="md", style={**middle_style, "border": middle_border})
        ], span=7),
        dmc.GridCol([
            dmc.Paper([
                dmc.Stack(options_elements, gap=0, justify="flex-start")
            ], withBorder=True, p="xs", radius="md", style={**right_style, "border": right_border})
        ], span=3)
    ], gutter="xs", mb="md")


def get_basic_workflow_step_content(step):
    """Return content for a specific basic workflow step
    
    Args:
        step: Step number (0-6)
    """
    if step == 0:  # Step 1
        return dmc.Paper([
            dmc.Badge("1", size="lg", variant="filled", color="violet", circle=True, mb="sm"),
            dmc.Title("Start Grammar Analysis Module", order=5, mb="md"),
            create_ui_location(highlight_section="left", highlight_element="", highlight_tab="", highlight_nav="grammar"),
            dcc.Markdown("""
To open the Grammar Analysis module, use the navigation bar on the left and click on "Grammar Sets". This is where you'll find all the tools for exploring the BSLVC grammar data. 
In the main view, you can switch between different tabs to access plots, sociodemographic details, and a tabular overview of the grammar items. The options panel on the right contains all the controls for customizing your analysis.
            """, className="markdown-content")
        ], p="md", withBorder=True, radius="md")
    
    elif step == 1:  # Step 2
        return dmc.Paper([
            dmc.Badge("2", size="lg", variant="filled", color="violet", circle=True, mb="sm"),
            dmc.Title("Choose an Analysis Mode", order=5, mb="md"),
            create_ui_location(highlight_section="", highlight_element="mode", highlight_tab="", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
The dashboard offers two different analysis modes for exploring the grammar data:

- **Participant Similarity**: In this mode, you can apply dimensionality reduction to the data to visualize how similar individual participants are based on their ratings.
- **Item Ratings**: Choose this mode, if you want to explore the distribution of individual grammar features.

Select your preferred mode in the options panel on the right.
            """, className="markdown-content")
        ], p="md", withBorder=True, radius="md")
    
    elif step == 2:  # Step 3
        return dmc.Paper([
            dmc.Badge("3", size="lg", variant="filled", color="violet", circle=True, mb="sm"),
            dmc.Title("Select Participants", order=5, mb="md"),
            create_ui_location(highlight_section="", highlight_element="participants", highlight_tab="", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
After choosing an analysis mode, use the participants tree in the right sidebar to select participants for your analysis. By clicking on the checkboxes, you can either select individual participants or entire groups based on variety and year.

Alternatively, you can use the "Select All" and "Deselect All" buttons, or choose one of the following presets for common selections:

- ENL: Select all L1 varieties (England, Scotland, US)
- ESL: Select all L2 varieties (Gibraltar, India, Malta, Puerto Rico)
- EFL: Select all EFL varieties (Germany, Slovenia, Sweden)
- Balanced: Select a balanced sample across all varieties
- Age Groups: Select participants from specific age groups
- Gender:
    - Female: Select all female participants
    - Male: Select all male participants
    - Balanced: Select a gender-balanced sample across all varieties
    - Balanced per Variety: Select a gender-balanced sample within each variety
            """, className="markdown-content")
        ], p="md", withBorder=True, radius="md")
    
    elif step == 3:  # Step 4
        return dmc.Paper([
            dmc.Badge("4", size="lg", variant="filled", color="violet", circle=True, mb="sm"),
            dmc.Title("Select Features", order=5, mb="md"),
            dcc.Markdown("""
Next, choose grammatical features that you want to include in your analysis. You can select features either directly in the grammar items tree or via the "Grammatical Items" tab.

**Selecting features in the Grammar Items tree**

This mode is recommended for selecting a larger number of features or entire feature groups.
The features are first grouped by mode, then by feature group. You can expand the groups to see individual features. By checking the boxes next to the feature names, you can select individual features or entire groups.
"""),
            create_ui_location(highlight_section="", highlight_element="grammar", highlight_tab="", highlight_nav="", highlight_right=True),
            dmc.Image(h="auto", w=300, fit="contain", src="/assets/img/gs_select_grammar_features_I.png"),
            dmc.Image(h="auto", w=300, fit="contain", src="/assets/img/gs_select_grammar_features_II.png"),
            dcc.Markdown("""


**Selecting features via the Grammatical Items tab**

This way of selecting features is recommended for a smaller number of specific features.
First, navigate to the "Grammatical Items" tab in the main view. 
There, you can use the search bar and preset filters to find specific features or groups of features. The table is filtered according to your input. You can select features by clicking invididual rows. By holding the Shift key, you can select multiple features at once. By holding the Ctrl (Cmd on Mac) key, you can add or remove individual features from your selection. 
Finally, click the green button labelled "Select rows" above the table. This selects the features of the highlighted rows in the grammar items tree on the right.
            """, className="markdown-content"),
            create_ui_location(highlight_section="middle", highlight_element="", highlight_tab="grammar_items", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
                    In the screenshot below, you can see that the table was filtered by selecting the preset "Existentials", and several features are selected in the filtered results. By clicking the green button "Select rows", these features are selected in the grammar items tree on the right.
            """, className="markdown-content"),
            dmc.Image(h="auto", w=500, fit="contain", src="/assets/img/gs_select_grammar_features_table.png"),
            dcc.Markdown("""

**Advanced Settings**

            """),
            create_ui_location(highlight_section="", highlight_element="grammar", highlight_tab="", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
**Deselecting problematic features**

A handful of features in the dataset could potentially skew the analysis. These features can be deselected with the button "Problematic" above the Grammar Item tree."""),
            dmc.Image(h="auto", w=300, fit="contain", src="/assets/img/gs_select_grammar_features_problematic.png"),
            dcc.Markdown("""

Items 'M19', 'J23', 'C14' and 'A4' contain currencies. Depending on the currency used in the country where the data were collected, the items differ in their wording (e.g., Euro vs. Dollar).

For items 'E22', 'D12' and 'E6', some of our speakers have either corrected the item (e.g., our Swedish speaker read out "This is exactly *what* I wanted" instead of "This is exactly *that* I wanted"), or read out the items in a way that blurs the distinction (e.g., compared to our other speakers, our Scottish speaker aspirated the t more heavily in "I'll tell Jane when I mee**t**s her"). We recommend that you do not use these items for a dimensionality reduction. 

            """, className="markdown-content"),
            dcc.Markdown("""
**Use item difference (spoken-written)**

Most items occur in both the spoken and the written section of the BSLVC. These items are referred to as *twin items*. 
If this switch is activated, the dashboard will calculate the difference between the spoken and written rating for each twin item and use these difference scores for the analysis instead of the raw ratings. This allows you to focus on how participants differentiate between spoken and written language use.
            """, className="markdown-content"),
            dcc.Markdown("""
**Use imputed data**

As the dataset contains missing values, we applied a random forest-based imputation method to estimate missing ratings. This allows us to include all features in the analysis without losing data due to incomplete responses.
The dimensionality reduction by default uses the imputed data, even if the switch is disabled, because the algorithm cannot handle missing data. For all other plots, this switch controls whether the imputed data are used or not.
We recommend you leave this switch disabled unless you have a specific reason to include imputed data in your analysis.
 
            """, className="markdown-content"),
        ], p="md", withBorder=True, radius="md")
    
    elif step == 4:  # Step 5
        return dmc.Paper([
            dmc.Badge("5", size="lg", variant="filled", color="violet", circle=True, mb="sm"),
            dmc.Title("Optional: Advanced Plot Settings", order=5, mb="md"),
            create_ui_location(highlight_section="", highlight_element="plot", highlight_tab="", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
### Advanced Plot Settings

In the advanced plot settings, you can customize the visualizations. Depending on the analysis mode, you have different options avalailabe.

**Analysis Mode "Participant Similarity":**
In this mode, you can mainly control the UMAP hyperparameters and how the data points should be colored. A description of the parameters and their effects can be found in the UMAP documentation at https://umap-learn.readthedocs.io/en/latest/
- Color by: Variety, Variety Type, Gender
- Adjust n_neighbors (controls local vs. global structure)
- Set min_dist (controls clustering tightness)
- Choose distance metric (cosine, euclidean, manhattan)
- Standardize data (recommended with euclidean and manhattan distances)
- Use density-preserving embedding (DensMAP; experimental)

**Analysis Mode "Item Ratings":**
In this mode, you can control the plot type, how the data should be grouped, and how the data should be sorted in the plot.
- Plot mode: 
    - "Mean (95% CI)": Plot mean values of features with confidence intervals
    - "Mean (95% CI - split varieties)": Plot mean values of features with confidence intervals. Each variety is represented separately on the y-axis. This helps to avoid the overplotting.
    - "Diverging stacked bars": Show the distribution of ratings for each feature in a diverging stacked barchart.
    - "Informant mean of selected features (boxplot)": Calculates a mean rating across all selected features for each participant and displays the distribution of these means in a boxplot.
    - "Correlation matrix": Displays a correlation matrix showing the pairwise correlations between the selected features.
    - "Missing values heatmap": Displays a heatmap indicating the presence of missing values across the selected features.
- Group by: Variety, Variety Type, Gender
- Sort by: Mean, Standard Deviation, Alphabetical

            """, className="markdown-content")
        ], p="md", withBorder=True, radius="md")
    
    elif step == 5:  # Step 6
        return dmc.Paper([
            dmc.Badge("6", size="lg", variant="filled", color="violet", circle=True, mb="sm"),
            dmc.Title("Render Plot", order=5, mb="md"),
            create_ui_location(highlight_section="", highlight_element="render", highlight_tab="", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
### Rendering Your Plot

Once you have selected your participants, grammar features, and configured your settings, click the **Render Plot** button to generate your visualization.

The render button is located in the options panel. Depending on your analysis mode and data size, rendering may take a few seconds. A loading indicator will appear while the plot is being generated.

After rendering, the plot will appear in the main view where you can interact with it, zoom, and explore the results.
            """, className="markdown-content")
        ], p="md", withBorder=True, radius="md")
    
    else:  # Step 7
        return dmc.Paper([
            dmc.Badge("7", size="lg", variant="filled", color="violet", circle=True, mb="sm"),
            dmc.Title("Export Plots & Data", order=5, mb="md"),
            create_ui_location(highlight_section="middle", highlight_element="", highlight_tab="plot", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
### Exporting Data and Plots

You can download all plots and export the data in a CSV format.

**Export Plots:**
Hover over any plot and click the camera icon in the mode bar located at the top right of the plot to download the plot as SVG file. 
SVG files are ideal for publications as they maintain high quality at any scale. They can be manually post-processed in vector graphic software (e.g. Inkscape) if needed.

**Export Data:**
To export the data, use the "Export data" button in the Advanced Actions section to download filtered datasets. The Data are exported in CSV format and includec your current participant and feature selection.
            """, className="markdown-content")
        ], p="md", withBorder=True, radius="md")


layout = dmc.Container([
    dmc.Card([
        # Main introduction content
        dcc.Markdown("""
## Getting Started with the BSLVC Dashboard

The BSLVC Dashboard allows you to interactively explore the data of the Bamberg Survey of Language Variation and Change.
Currently, only the grammar data are available for exploration. The lexical data will be added in future updates.

This section provides an overview of the interface, describes the basic workflow for utilizing the Dashboard to analyse the grammar data, and describes two case studies in which we illustrate how you can approach more specific research questions. 
A comprehensive documentation of the Dashboard's features and functionalities will be available soon.
        """),
        
        # UI Structure Overview
        dmc.Title("Interface Overview", order=3, mt="xl", mb="md"),
        dmc.Grid([
            dmc.GridCol([
                dmc.Paper([
                    dmc.Stack([
                        dmc.ThemeIcon(
                            DashIconify(icon="tabler:layout-sidebar-left", width=20),
                            size="lg",
                            radius="md",
                            variant="light",
                            color="blue"
                        ),
                        dmc.Text("Navigation", fw=600, size="sm", ta="center"),
                        #dmc.Text("Participants & Items", size="xs", c="dimmed", ta="center")
                    ], align="center", gap="xs", justify="center", style={"minHeight": "200px"})
                ], withBorder=True, p="md", radius="md", style={"backgroundColor": "#f8f9fa"})
            ], span=1),
            dmc.GridCol([
                dmc.Paper([
                    dmc.Stack([
                        dmc.ThemeIcon(
                            DashIconify(icon="tabler:layout", width=24),
                            size="xl",
                            radius="md",
                            variant="light",
                            color="violet"
                        ),
                        dmc.Text("Main View", fw=600, size="md", ta="center"),
                        dmc.Text("Text, plots & tables", size="sm", c="dimmed", ta="center")
                    ], align="center", gap="xs", justify="center", style={"minHeight": "200px"})
                ], withBorder=True, p="lg", radius="md", style={"backgroundColor": "#f8f9fa"})
            ], span=8),
            dmc.GridCol([
                dmc.Paper([
                    dmc.Stack([
                        dmc.ThemeIcon(
                            DashIconify(icon="tabler:settings", width=20),
                            size="lg",
                            radius="md",
                            variant="light",
                            color="orange"
                        ),
                        dmc.Text("Options", fw=600, size="sm", ta="center"),
                        dmc.Text("Action buttons, selection of participants & features, plot settings", size="xs", c="dimmed", ta="center")
                    ], align="center", gap="xs", justify="center", style={"minHeight": "200px"})
                ], withBorder=True, p="md", radius="md", style={"backgroundColor": "#f8f9fa"})
            ], span=3)
        ], gutter="md", mb="xl"),
        
        # Working with Grammar Data Section (combines workflow + case studies)
        dmc.Group([
            DashIconify(icon="tabler:language", width=28),
            dmc.Title("Working with the Grammar Data", order=3, mt="xl", mb="md")
        ], gap="sm", mt="xl", mb="md"),
        dmc.Text("Follow the basic workflow to get started, or explore practical examples through case studies.", c="dimmed", mb="md"),
        
        dmc.Accordion([
            # Basic Workflow
            dmc.AccordionItem([
                dmc.AccordionControl(
                    dmc.Group([
                        dmc.ThemeIcon(
                            DashIconify(icon="tabler:list-numbers", width=20),
                            size="lg",
                            radius="md",
                            variant="light",
                            color="violet"
                        ),
                        dmc.Text("Basic Workflow", fw=500, size="md")
                    ], gap="sm")
                ),
                dmc.AccordionPanel([
                    # Grid layout: Stepper on left, content on right
                    dmc.Grid([
                        # Left column: Stepper
                        dmc.GridCol([
                            dmc.Stepper(
                                id="basic-workflow-stepper",
                                active=0,
                                color="violet",
                                size="sm",
                                orientation="vertical",
                                iconSize=32,
                                children=[
                                    dmc.StepperStep(label="Start Grammar Analysis Module", description="Navigate to Grammar Sets"),
                                    dmc.StepperStep(label="Choose an Analysis Mode", description="Select analysis type"),
                                    dmc.StepperStep(label="Select Participants", description="Choose study participants"),
                                    dmc.StepperStep(label="Select Features", description="Pick grammar features"),
                                    dmc.StepperStep(label="Advanced Plot Settings", description="Optional customization"),
                                    dmc.StepperStep(label="Render Plot", description="Generate visualization"),
                                    dmc.StepperStep(label="Export Plots & Data", description="Download results"),
                                ]
                            )
                        ], span=3),
                        
                        # Right column: Content for all steps
                        dmc.GridCol([
                            html.Div(id="basic-workflow-content", children=get_basic_workflow_step_content(0))
                        ], span=9)
                    ], gutter="xl")
                ])
            ], value="basic_workflow"),
            
            # Case Study 1: Variety Clustering
            dmc.AccordionItem([
                dmc.AccordionControl(
                    dmc.Group([
                        dmc.ThemeIcon(
                            DashIconify(icon="tabler:chart-dots-3", width=20),
                            size="lg",
                            radius="md",
                            variant="light",
                            color="blue"
                        ),
                        dmc.Text("Case Study 1: A Typological Approach to the BSLVC Data", fw=500, size="md")
                    ], gap="sm")
                ),
                create_case_study_1(create_ui_location)
            ], value="case1"),
            
            # Case Study 2: Progressive Aspect
            dmc.AccordionItem([
                dmc.AccordionControl(
                    dmc.Group([
                        dmc.ThemeIcon(
                            DashIconify(icon="tabler:arrow-right-circle", width=20),
                            size="lg",
                            radius="md",
                            variant="light",
                            color="green"
                        ),
                        dmc.Text("Case Study 2: The Progressive Aspect in Selected Varieties of English", fw=500, size="md")
                    ], gap="sm")
                ),
                create_case_study_2(create_ui_location)
            ], value="case2")
        ], multiple=True, variant="contained", radius="md")
        
    ], withBorder=True, shadow="sm", radius="md", p="lg", mb="lg")
], fluid=True, style={"maxWidth": "1600px", "margin": "0 auto", "paddingLeft": "20px", "paddingRight": "20px"})


# Callback to update basic workflow content based on stepper step
@callback(
    Output("basic-workflow-content", "children"),
    Input("basic-workflow-stepper", "active")
)
def update_basic_workflow_content(active_step):
    """Update basic workflow content when stepper step changes"""
    if active_step is None:
        active_step = 0
    return get_basic_workflow_step_content(active_step)


# Callback to update case study 1 content based on stepper step
@callback(
    Output("case-study-1-content", "children"),
    Input("case-study-1-stepper", "active")
)
def update_case_study_content(active_step):
    """Update content when stepper step changes"""
    if active_step is None:
        active_step = 0
    return get_case_study_1_step_content(active_step, create_ui_location)


# Callback to update case study 2 content based on stepper step
@callback(
    Output("case-study-2-content", "children"),
    Input("case-study-2-stepper", "active")
)
def update_case_study_2_content(active_step):
    """Update case study 2 content when stepper step changes"""
    if active_step is None:
        active_step = 0
    return get_case_study_2_step_content(active_step, create_ui_location)
