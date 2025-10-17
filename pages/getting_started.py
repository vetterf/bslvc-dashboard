import dash_mantine_components as dmc
from dash import register_page, dcc, html
import os

register_page(__name__, path="/getting-started", name="Getting Started")

# Read the markdown content
def read_documentation():
    doc_path = os.path.join(os.path.dirname(__file__), "..", "assets", "getting_started.md")
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "# Getting Started\n\nDocumentation file not found."

layout = dmc.Container([
    dcc.Markdown(
        children=read_documentation(),
        style={
            "maxWidth": "none",
            "lineHeight": "1.7",
            "fontSize": "15px"
        }
    )
], size="lg", py="xl")
