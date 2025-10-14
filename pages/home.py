import dash_mantine_components as dmc
from dash import register_page

register_page(__name__, path="/", name="Home")

layout = dmc.Container([
    dmc.Group([
        dmc.Image(
            src="/assets/img/bslvc_logo.png",
            h=60,
            w="auto",
            fit="contain"
        ),
        dmc.Title("Bamberg Survey of Language Variation and Change", order = 2)
    ], align="center", gap="md"),
    dmc.Space(h=20),
    dmc.Text("Welcome to the BSLVC Dashboard", size="xl"),
    dmc.Space(h=15),
    dmc.Text("Use the navigation menu above to explore the various features of this linguistic research platform.", size="md", style={"fontStyle": "italic"}),
], fluid=True)