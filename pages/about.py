import dash_mantine_components as dmc
from dash import register_page, dcc
from dash_iconify import DashIconify

register_page(__name__, path="/about", name="About")

layout = dmc.Container([
    dmc.Card([
        # Main Title
        dmc.Group([
            DashIconify(icon="tabler:info-square", width=32, color="#228be6"),
            dmc.Title("About the BSLVC Dashboard", order=2)
        ], gap="sm", mb="lg"),
        
        # Introduction
        dmc.Text(
            "The BSLVC Dashboard is an interactive web application for exploring and analyzing data from the Bamberg Survey of Language Variation and Change. "
            "It provides researchers, students, and language enthusiasts with powerful tools to visualize grammatical variation across World Englishes varieties through acceptability judgments and dimensionality reduction techniques.",
            size="md",
            mb="xl"
        ),
        
        # Project Homepage Section
        dmc.Paper([
            dmc.Group([
                dmc.ThemeIcon(
                    DashIconify(icon="tabler:world", width=20),
                    size="lg",
                    radius="md",
                    variant="light",
                    color="blue"
                ),
                dmc.Title("Project Information", order=4)
            ], gap="sm", mb="md"),
            dmc.Text(
                "For comprehensive information about the Bamberg Survey of Language Variation and Change project, "
                "including methodology, research objectives, and the team behind the corpus, please visit:",
                size="sm",
                mb="sm"
            ),
            dmc.Anchor(
                dmc.Button(
                    "Visit Project Homepage",
                    leftSection=DashIconify(icon="tabler:external-link", width=16),
                    variant="light",
                    color="blue"
                ),
                href="https://www.uni-bamberg.de/aspra/bslvc/",
                target="_blank",
                style={"textDecoration": "none"}
            )
        ], p="md", withBorder=True, radius="md", mb="xl", style={"backgroundColor": "#f8f9fa"}),
        
        # Citation Section
        dmc.Paper([
            dmc.Group([
                dmc.ThemeIcon(
                    DashIconify(icon="tabler:quote", width=20),
                    size="lg",
                    radius="md",
                    variant="light",
                    color="green"
                ),
                dmc.Title("How to Cite", order=4)
            ], gap="sm", mb="md"),
            dmc.Text(
                "If you use the BSLVC Dashboard or data from the corpus in your research, please cite it appropriately:",
                size="sm",
                mb="md"
            ),
            
            # Dashboard Citation
            dmc.Stack([
                dmc.Text("Dashboard Citation:", fw=600, size="sm"),
                dmc.Code(
                    "Vorberger, L., & Röthlisberger, M. (2025). BSLVC Interactive Dashboard (Version 0.1) [Software]. University of Bamberg. https://bslvc-dashboard.uni-bamberg.de",
                    block=True,
                    style={"whiteSpace": "pre-wrap", "padding": "12px"}
                ),
                
                dmc.Space(h="md"),
                
                # Corpus Citation
                dmc.Text("Corpus Citation:", fw=600, size="sm"),
                dmc.Code(
                    "Vorberger, L., & Röthlisberger, M. (2025). Bamberg Survey of Language Variation and Change (BSLVC) [Data set]. University of Bamberg. https://www.uni-bamberg.de/aspra/bslvc/",
                    block=True,
                    style={"whiteSpace": "pre-wrap", "padding": "12px"}
                ),
                
                dmc.Space(h="md"),
                
                # BibTeX
                dmc.Text("BibTeX:", fw=600, size="sm"),
                dmc.Code(
                    """@software{bslvc_dashboard2025,
  author = {Vorberger, Lars and Röthlisberger, Melanie},
  title = {BSLVC Interactive Dashboard},
  version = {0.1},
  year = {2025},
  publisher = {University of Bamberg},
  url = {https://bslvc-dashboard.uni-bamberg.de}
}

@dataset{bslvc2025,
  author = {Vorberger, Lars and Röthlisberger, Melanie},
  title = {Bamberg Survey of Language Variation and Change (BSLVC)},
  year = {2025},
  publisher = {University of Bamberg},
  url = {https://www.uni-bamberg.de/aspra/bslvc/}
}""",
                    block=True,
                    style={"whiteSpace": "pre-wrap", "padding": "12px"}
                ),
            ], gap="xs")
        ], p="md", withBorder=True, radius="md", mb="xl", style={"backgroundColor": "#f8f9fa"}),
        
        # Technical Information
        dmc.Paper([
            dmc.Group([
                dmc.ThemeIcon(
                    DashIconify(icon="tabler:code", width=20),
                    size="lg",
                    radius="md",
                    variant="light",
                    color="violet"
                ),
                dmc.Title("Technical Information", order=4)
            ], gap="sm", mb="md"),
            dmc.List([
                dmc.ListItem(dmc.Text(["Dashboard Version: ", dmc.Code("0.1.1")], size="sm")),
                dmc.ListItem(dmc.Text(["Built with: ", dmc.Code("Dash 3.2.0"), " and ", dmc.Code("dash-mantine-components 2.3.0")], size="sm")),
                dmc.ListItem(dmc.Text(["Data Analysis: ", dmc.Code("Python 3.13"), ", ", dmc.Code("scikit-learn"), ", ", dmc.Code("umap-learn")], size="sm")),
                dmc.ListItem(dmc.Text(["Visualizations: ", dmc.Code("Plotly")], size="sm")),
            ], size="sm")
        ], p="md", withBorder=True, radius="md", mb="xl", style={"backgroundColor": "#f8f9fa"}),
        
        # Contact Information
        dmc.Paper([
            dmc.Group([
                dmc.ThemeIcon(
                    DashIconify(icon="tabler:mail", width=20),
                    size="lg",
                    radius="md",
                    variant="light",
                    color="orange"
                ),
                dmc.Title("Contact & Support", order=4)
            ], gap="sm", mb="md"),
            dmc.Text(
                "For questions, feedback, or support regarding the BSLVC Dashboard, please contact the project team through the project homepage.",
                size="sm"
            )
        ], p="md", withBorder=True, radius="md", style={"backgroundColor": "#f8f9fa"}),
        
    ], withBorder=True, shadow="sm", radius="md", p="lg", mb="lg")
], fluid=True, style={"maxWidth": "1200px", "margin": "0 auto", "paddingLeft": "20px", "paddingRight": "20px"})
