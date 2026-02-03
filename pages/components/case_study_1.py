"""Case Study 1: A Typological Approach to the BSLVC Data"""
import dash_mantine_components as dmc
from dash import html, dcc, callback, Input, Output
from dash_iconify import DashIconify


def create_case_study_1(create_ui_location):
    """Create Case Study 1 content with stepper on left, content on right
    
    Args:
        create_ui_location: Function to create UI location visualizations
    """
    return dmc.AccordionPanel([
        # Research Questions
        dmc.Paper([
            dmc.Group([
                DashIconify(icon="tabler:bulb", width=24, color="#228be6"),
                dmc.Title("Research Questions", order=4)
            ], gap="sm", mb="sm"),
            dcc.Markdown("""
This case study explores how similar participants are with respect to their ratings of the spoken grammar items. The guiding questions are:

1. Do participants cluster according to their variety based on spoken feature ratings?
2. If clustering occurs, do varieties group by type (e.g., ENL, ESL, EFL)?
3. Which grammatical features best distinguish English from Scottish participants?
            """, className="markdown-content")
        ], p="md", withBorder=True, radius="md", mb="lg", style={"backgroundColor": "#f8f9fa"}),
        
        # Grid layout: Stepper on left, content on right
        dmc.Grid([
            # Left column: Stepper
            dmc.GridCol([
                dmc.Stepper(
                    id="case-study-1-stepper",
                    active=0,
                    color="blue",
                    size="sm",
                    orientation="vertical",
                    iconSize=32,
                    children=[
                        dmc.StepperStep(label="Select Analysis Mode", description="Choose Participant Similarity"),
                        dmc.StepperStep(label="Select All Participants", description="Include all varieties"),
                        dmc.StepperStep(label="Select Spoken Features", description="Choose grammar items"),
                        dmc.StepperStep(label="Dimensionality Reduction", description="Render and analyze plot"),
                        dmc.StepperStep(label="Compare Groups", description="Compare varieties and features"),
                    ]
                )
            ], span=3),
            
            # Right column: Content that changes based on active step
            dmc.GridCol([
                html.Div(id="case-study-1-content", children=get_case_study_1_step_content(0, create_ui_location))
            ], span=9)
        ], gutter="xl")
    ])


def get_case_study_1_step_content(step, create_ui_location):
    """Return content for a specific step
    
    Args:
        step: Step number (0-4)
        create_ui_location: Function to create UI location visualizations
    """
    if step == 0:  # Step 1
        return dmc.Paper([
            dmc.Badge("1", size="lg", variant="filled", color="blue", circle=True, mb="sm"),
            dmc.Title("Select Analysis Mode", order=5, mb="md"),
           
            create_ui_location(highlight_section="", highlight_element="mode", highlight_tab="", highlight_nav="grammar", highlight_right=True),
             dcc.Markdown("""
Navigate to the **Grammar Sets** page and select **"Participant Similarity"** from the analysis mode selector. This mode visualizes how similar participants are with respect to their intuition-based ratings of the grammar items.
            """, className="markdown-content"),
            dmc.Paper([
                dmc.Stack([
                    dmc.Text("Analysis Mode:", size="sm", fw=500, mb="xs"),
                    dmc.SegmentedControl(
                        data=[
                            {"value": "umap", "label": "Participant Similarity"},
                            {"value": "item", "label": "Item Ratings"},
                        ],
                        value="umap",
                        fullWidth=True,
                        color="blue",
                        size="sm"
                    ),
                ], gap="xs")
            ], p="md", withBorder=True, radius="md", style={"backgroundColor": "#f8f9fa", "maxWidth": "400px",
                    "margin": "1rem 0",
                    "display": "block"
                }
            ),
        ], p="md", withBorder=True, radius="md")
    
    elif step == 1:  # Step 2
        return dmc.Paper([
            dmc.Badge("2", size="lg", variant="filled", color="blue", circle=True, mb="sm"),
            dmc.Title("Select All Participants", order=5, mb="md"),
            create_ui_location(highlight_section="", highlight_element="participants", highlight_tab="", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
In the **Participants** section of the settings panel, click the **"Select All"** button to include all available participants in your analysis.
""", className="markdown-content"),
            dmc.Image(h="auto", w=300, fit="contain", src="/assets/case_studies/screenshots/c1_select_all_participants.png"),
        ], p="md", withBorder=True, radius="md")
    
    elif step == 2:  # Step 3
        return dmc.Paper([
            dmc.Badge("3", size="lg", variant="filled", color="blue", circle=True, mb="sm"),
            dmc.Title("Select Spoken Features", order=5, mb="md"),
            create_ui_location(highlight_section="", highlight_element="grammar", highlight_tab="", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
In the **Grammar Items** section:

1. Click the checkbox next to "Spoken" to include all spoken grammar features.
2. Then click **"Problematic"** (marked with a filter icon) to exclude features that may have distorting effects.

For more details on problematic features, see either the basic workflow or the documentation.
            """, className="markdown-content"),
            dmc.Image(h="auto", w=300, fit="contain", src="/assets/case_studies/screenshots/c1_select_grammar_items.png"),
        ], p="md", withBorder=True, radius="md")
    
    elif step == 3:  # Step 4 - Combined Dimensionality Reduction
        return dmc.Paper([
            dmc.Badge("4", size="lg", variant="filled", color="blue", circle=True, mb="sm"),
            dmc.Title("Dimensionality Reduction", order=5, mb="md"),
            create_ui_location(highlight_section="", highlight_element="render", highlight_tab="", highlight_nav="", highlight_right=True),
            dcc.Markdown("""
With participants and features selected, click **"Render Plot"** to generate the visualization.

⏱️ *Note: Computing UMAP may take up to 30 seconds depending on the number of participants and features selected.*
            """, className="markdown-content"),
            create_ui_location(highlight_section="middle", highlight_element="", highlight_tab="plot", highlight_nav="", highlight_right=False),
            dcc.Markdown("""

The resulting plot displays each participant as a point. The color indicates variety.

Unlike traditional multidimensional scaling (MDS), UMAP (Uniform Manifold Approximation and Projection; see [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/) for technical details) does not try to preserve all pair-wise distances between data points in the low-dimensional space. Instead, the algorithm tries to preserve both local and global structure when reducing high-dimensional data to 2D.
The visualization serves to provide an intuitive overview of the structure in the data rather than precise distance measurements. Consequently, the distances should not be overinterpreted (e.g., do not interpret the distances à la "Participant A is exactly twice as different from B as from C").
            """, className="markdown-content"),
            dmc.Image(h="auto", w=600, fit="contain", src="/assets/case_studies/screenshots/c1_umap.svg"),
            dcc.Markdown("""
**Key Observations:**

1. **Variety Clustering**: Participants cluster by variety, indicating that speakers from the same variety share similar grammatical preferences.
2. **Variety-type Clustering**: Varieties also cluster typologically, i.e. ENL varieties (England, Scotland, US) cluster together, EFL varieties form a separate cluster, while ESL varieties appear more spread out.
            """, className="markdown-content")
        ], p="md", withBorder=True, radius="md")
    
    else:  # step == 4 - Combined Compare Groups
        return dmc.Paper([
            dmc.Badge("5", size="lg", variant="filled", color="blue", circle=True, mb="sm"),
            dmc.Title("Compare English and Scottish Participants", order=5, mb="md"),
            create_ui_location(highlight_section="middle", highlight_element="", highlight_tab="plot", highlight_nav="", highlight_right=False),
            dcc.Markdown("""
An intuitive follow-up question is: **Which features are most discriminative between groups in the plot?**

To answer this, we'll use the **"Compare Groups"** functionality, which trains a Random Forest classifier on the selected data and extracts feature importance rankings.

**How to compare specific varieties:**

1. Click on varieties in the plot legend to toggle visibility
2. Keep only **England** and **Scotland** visible (hide all other varieties)
3. Click the **"Compare Groups"** button
            """, className="markdown-content"),
            dmc.Image(h="auto", w=600, fit="contain", src="/assets/case_studies/screenshots/c1_umap_eng_sc.svg"),
            dcc.Markdown("""

Only the visible groups will be included in the comparison. The Random Forest model will identify which grammatical features best distinguish between English and Scottish participants.

💡 *Advanced Tip: You can also use the lasso selection tool to define custom groups. See the full documentation for examples.*
            """, className="markdown-content"),
            dmc.Image(h="auto", w=600, fit="contain", src="/assets/case_studies/screenshots/c1_group_comparison.svg"),
            dcc.Markdown("""

The newly generated plot displays the grammatical features that, according to the Random Forest model, have the highest discriminative power for distinguishing English from Scottish participants. The green line indicates the importance ratings for each feature.

💡 *Navigation Tip: Although only the first 10-15 items are visible by default, you can use the pan tool (hand icon in the toolbar) or scroll horizontally to explore all included features.*
            """, className="markdown-content"),
            
        ], p="md", withBorder=True, radius="md")
