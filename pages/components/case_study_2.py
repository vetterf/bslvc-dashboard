"""Case Study 2: The Progressive Aspect in Selected Varieties of English"""
import dash_mantine_components as dmc
from dash import html, dcc
from dash_iconify import DashIconify


def create_case_study_2(create_ui_location):
    """Create Case Study 2 content with stepper on left, content on right
    
    Args:
        create_ui_location: Function to create UI location visualizations
    """
    return dmc.AccordionPanel([
        # Research Questions
        dmc.Paper([
            dmc.Group([
                DashIconify(icon="tabler:bulb", width=24, color="#2b8a3e"),
                dmc.Title("Research Question", order=4)
            ], gap="sm", mb="sm"),
            dcc.Markdown("""
In case study 1, we found that Scottish participants have a markedly higher average rating than English participants for item D2: "What are you wanting?". 
In this case study, we follow up on this findings, and investigate all features related to the eWAVE Feature 88 "Wider range of uses of progrssive be + V-ing than in StE: extension to stative verbs" (https://ewave-atlas.org/parameters/88).
            """, className="markdown-content")
        ], p="md", withBorder=True, radius="md", mb="lg", style={"backgroundColor": "#f8f9fa"}),
        
        # Grid layout: Stepper on left, content on right
        dmc.Grid([
            # Left column: Stepper
            dmc.GridCol([
                dmc.Stepper(
                    id="case-study-2-stepper",
                    active=0,
                    color="green",
                    size="sm",
                    orientation="vertical",
                    iconSize=32,
                    children=[
                        dmc.StepperStep(label="Select Analysis Mode", description="Choose Item Ratings"),
                        dmc.StepperStep(label="Select Participants", description="England & Scotland only"),
                        dmc.StepperStep(label="Select Progressive Features", description="Use table filtering"),
                        dmc.StepperStep(label="Render Item Plot", description="Compare variety ratings"),
                    ]
                )
            ], span=3),
            
            # Right column: Content that changes based on active step
            dmc.GridCol([
                html.Div(id="case-study-2-content", children=get_case_study_2_step_content(0, create_ui_location))
            ], span=9)
        ], gutter="xl")
    ])


def get_case_study_2_step_content(step, create_ui_location):
    """Return content for a specific step
    
    Args:
        step: Step number (0-3)
        create_ui_location: Function to create UI location visualizations
    """
    if step == 0:  # Step 1
        return dmc.Paper([
            dmc.Badge("1", size="lg", variant="filled", color="green", circle=True, mb="sm"),
            dmc.Title("Select Analysis Mode", order=5, mb="md"),
            create_ui_location(highlight_section="", highlight_element="mode", highlight_tab="", highlight_nav="grammar", highlight_right=True),
            dcc.Markdown("""
Navigate to the **Grammar Sets** page and select **"Item Ratings"** from the analysis mode selector. 

This mode allows you to explore the distribution of ratings for individual grammatical features, making it ideal for comparing how different varieties rate specific constructions.
            """, className="markdown-content"),
        ], p="md", withBorder=True, radius="md")
    
    elif step == 1:  # Step 2
        return dmc.Paper([
            dmc.Badge("2", size="lg", variant="filled", color="green", circle=True, mb="sm"),
            dmc.Title("Select England and Scotland Participants", order=5, mb="md"),
            create_ui_location(highlight_section="", highlight_element="participants", highlight_tab="", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
In the **Participants** section of the settings panel:

1. Select **England** (check the box next to "England")
2. Select **Scotland** (check the box next to "Scotland")
3. Leave all other varieties unchecked
            """, className="markdown-content"),
        ], p="md", withBorder=True, radius="md")
    
    elif step == 2:  # Step 3
        return dmc.Paper([
            dmc.Badge("3", size="lg", variant="filled", color="green", circle=True, mb="sm"),
            dmc.Title("Select Features Using Table Filtering", order=5, mb="md"),
            create_ui_location(highlight_section="middle", highlight_element="", highlight_tab="grammar_items", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
The items in the BSLVC are grouped by feature type, and if applicable, also grouped by eWAVE features. 
To find specific features or groups more easily, we'll use the Grammar Items table and make use of the quick filter function:

1. Deselect all currently selected items in the grammar item tree selector on the right panel (click the red **"Deselect all"** button above the tree).
2. Navigate to the **"Grammar Items"** tab in the main view
3. In the **Quick search bar**, type "progressive" to filter items containing this term. Alternatively, you can select the preset "Group:Progressives" or "eWAVE: Wider range of uses of progressive be + V-ing than StE: extension to stative verbs" in the dropdown menu below the quick search bar.
4. If necessary, change the width of the eWAVE column.
5. Select all features in the relevant eWAVE group (click on rows, or use Shift/Ctrl to select multiple).
6. Click the green **"Select rows"** button above the table to apply the selection to the grammar item tree selector in the right panel.""", className = "markdown-content"),
            dmc.Image(h="auto", w=500, fit="contain", src="/assets/case_studies/screenshots/c2_grammar_table.png"),
            dcc.Markdown("""
The selected features will now be highlighted in the grammar items tree selector in the right panel.
            """, className="markdown-content"),

            dmc.Image(h="auto", w=300, fit="contain", src="/assets/case_studies/screenshots/c2_grammar_tree.png"),
        ], p="md", withBorder=True, radius="md")
    
    elif step == 3:  # Step 4
        return dmc.Paper([
            dmc.Badge("4", size="lg", variant="filled", color="green", circle=True, mb="sm"),
            dmc.Title("Render Item Plot", order=5, mb="md"),
            create_ui_location(highlight_section="", highlight_element="render", highlight_tab="", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
With England and Scotland participants selected and features chosen, click **"Render Plot"** to generate the visualization.

The resulting plot will display ratings for each BSLVC item related to eWAVE feature 88, grouped by variety. 
            """, className="markdown-content"),
            dmc.Image(h="auto", w=600, fit="contain", src="/assets/case_studies/screenshots/c2_item_plot.svg"),
            dcc.Markdown("""💡 *Advanced Tip: You can change the plot mode in the Plot Options section to view the data as diverging stacked bars or boxplots for alternative perspectives on the distribution.*
            """, className="markdown-content"),
        ], p="md", withBorder=True, radius="md")
    
    else:
        return dmc.Paper([], p="md", withBorder=True, radius="md")
