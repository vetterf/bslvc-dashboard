import dash_mantine_components as dmc
from dash import register_page, html

register_page(__name__, path="/about-project", name="About the Project")

layout = dmc.Container([
    dmc.Title("About the Project", order = 2),
    dmc.Space(h=30),
    dmc.Center([
        html.Img(
            src="/assets/img/under_construction.png",
            style={
                "maxWidth": "400px",
                "width": "100%",
                "height": "auto"
            }
        )
    ])
], fluid=True)
