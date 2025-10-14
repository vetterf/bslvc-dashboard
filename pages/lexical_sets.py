import dash_mantine_components as dmc
from dash import register_page, html

register_page(__name__, path="/lexical", name="Lexical Sets")

layout = dmc.Container([
    dmc.Title("Lexical Sets", order = 2),
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