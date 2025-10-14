import dash_mantine_components as dmc
from dash import register_page, dcc, html
import os

register_page(__name__, path="/documentation", name="Documentation")

# Read the markdown content
def read_documentation():
    doc_path = os.path.join(os.path.dirname(__file__), "..", "assets", "documentation.md")
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "# Documentation\n\nDocumentation file not found."

layout = dmc.Container([
    dmc.Title("BSLVC Dashboard Documentation", order=1, mb="xl"),
    dcc.Markdown(
        children=read_documentation(),
        style={
            "maxWidth": "none",
            "lineHeight": "1.6",
            "fontSize": "14px"
        }
    )
], size="xl", py="xl")
