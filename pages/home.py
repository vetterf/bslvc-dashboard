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
            "Variation and Change database. The heart of the dashboard are the two main modules: the Grammar Analysis module and the Lexical Analysis module. "
            "The Grammar Analysis module allows you to "
            "visualize participant similarity, explore feature distributions, and identify "
            "the most discriminative features between groups of informants. "
            "The Lexical Analysis module allows you to investigate apparent time trends in the lexical part of theBSLVC data.",
            size="md",
            mb="lg"
        ),

        dmc.Divider(mb="lg"),

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
            "A more detailed description of the features and functionality can be found in the documentation.",
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
            "Note: This platform is under active development. If you encounter any issues or have suggestions, please contact Fabian Vetter.",
            size="sm",
            c="dimmed",
            mt="lg"
        ),

    ], withBorder=True, shadow="sm", radius="md", p="md", mb="lg"),
    
], fluid=True)