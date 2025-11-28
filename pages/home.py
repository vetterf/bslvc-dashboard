import dash_mantine_components as dmc
from dash import register_page
from dash_iconify import DashIconify

register_page(__name__, path="/", name="Home")

layout = dmc.Container([
   
    # Welcome section
    dmc.Card([
        dmc.Group([
            dmc.Image(
                src="/assets/img/bslvc_logo.png",
                h=60,
                w="auto",
                fit="contain"
            ),
            dmc.Title("Welcome to the BSLVC Dashboard", order=3)
        ], align="flex-end", gap="md", mb="xl"),
    
        dmc.Text(
            "The BSLVC Dashboard provides interactive tools to explore the Bamberg Survey of Language "
            "Variation and Change database. Currently, the Grammar Analysis module is available, "
            "allowing you to visualize participant similarity, explore feature distributions, and identify "
            "the most discriminative features between groups of informants.",
            size="md",
            mb="lg"
        ),

        dmc.Divider(mb="lg"),

        # What you can do section
        dmc.Title("What You Can Do with the Grammar Module", order=4, mb="md"),
        
        dmc.List(
            [
                dmc.ListItem("Filter participants by variety, age, gender, and data completeness"),
                dmc.ListItem("Visualize participant similarity using dimensionality reduction (UMAP)"),
                dmc.ListItem("Explore individual feature distributions with interactive plots"),
                dmc.ListItem("Compare groups and identify the most discriminative features between groups using Random Forests"),
                dmc.ListItem("Export plots as SVG files and filtered data as CSV"),
            ],
            size="md",
            mb="lg"
        ),

        dmc.Divider(mb="lg"),

        # Getting started section
        dmc.Group([
            dmc.ThemeIcon(
                DashIconify(icon="tabler:rocket", width=24),
                size="lg",
                radius="md",
                variant="light",
                color="blue"
            ),
            dmc.Title("Get Started", order=4),
        ], mb="md"),
        
        dmc.Text(
            "New to the dashboard? Visit the Getting Started section for a step-by-step guide and example case studies. "
            "For detailed information about features and functionality, check the Documentation.",
            size="md",
            mb="md"
        ),

        dmc.Group([
            dmc.Anchor(
                dmc.Button(
                    "Getting Started",
                    leftSection=DashIconify(icon="tabler:rocket"),
                    variant="filled",
                ),
                href="/getting-started"
            ),
            dmc.Anchor(
                dmc.Button(
                    "Documentation",
                    leftSection=DashIconify(icon="tabler:book-2"),
                    variant="outline",
                ),
                href="https://vetterf.github.io/bslvc-dashboard",
                target="_blank"
            ),
        ], gap="md", mb="md"),

        dmc.Text(
            "Note: This platform is under active development. The lexical analysis module will be added in future updates.",
            size="sm",
            c="dimmed",
            mt="lg"
        ),

    ], withBorder=True, shadow="sm", radius="md", p="lg", mb="lg"),
    
], fluid=True, style={"maxWidth": "1200px", "margin": "0 auto", "paddingLeft": "20px", "paddingRight": "20px"})