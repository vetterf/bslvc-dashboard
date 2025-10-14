import dash_mantine_components as dmc
from dash import register_page

register_page(__name__, path="/bslvc", name="BSLVC: The Survey")

layout = dmc.Container([
    dmc.Title("Bamberg Survey of Language Contact and Change", order = 2),
   dmc.Text("- Description of the data plus collection regimes"),
dmc.Text("- Lexical data mostly in the streets"),
dmc.Text("- Grammar data mostly in university settings, hence gender, age, education bias"),
dmc.Text("- Exemplary questionnaire with audio file")
], fluid=True)