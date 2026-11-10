"""Shared palette controls for the analysis sidebars."""

import dash_mantine_components as dmc
from dash import ALL, ClientsideFunction, Input, Output, State, clientside_callback, html
from dash_iconify import DashIconify


def color_controls(page):
    return dmc.Accordion(
        [dmc.AccordionItem([
            dmc.AccordionControl(dmc.Group([
                DashIconify(icon="tabler:palette", width=18),
                dmc.Text("Group Colors", fw=500, size="sm"),
            ], gap="xs")),
            dmc.AccordionPanel([
                html.Div(id=f"{page}-group-color-controls"),
                dmc.Button("Reset all group colors", id=f"{page}-group-color-reset",
                           size="xs", variant="outline", mt="sm"),
            ]),
        ], value="group-colors")],
        variant="contained", radius="md", mb="md",
    )


for page in ("grammar", "lexical"):
    clientside_callback(
        ClientsideFunction(namespace="groupColors", function_name="savePalette"),
        Output("group-color-palette", "data", allow_duplicate=True),
        Input({"type": "group-color-input", "page": page, "kind": ALL,
               "group": ALL, "default": ALL}, "value"),
        Input(f"{page}-group-color-reset", "n_clicks"),
        State({"type": "group-color-input", "page": page, "kind": ALL,
               "group": ALL, "default": ALL}, "id"),
        State("group-color-palette", "data"),
        prevent_initial_call=True,
    )

clientside_callback(
    ClientsideFunction(namespace="groupColors", function_name="grammarControls"),
    Output("grammar-group-color-controls", "children"),
    Input("grammar-plot-type", "value"),
    Input("umap-view-toggle", "value"),
    Input("umap-color-dropdown", "value"),
    Input("UMAPfig", "figure"),
    Input("ItemFig", "figure"),
    Input("group-color-palette", "data"),
    State("informants-store", "data"),
)

clientside_callback(
    ClientsideFunction(namespace="groupColors", function_name="lexicalControls"),
    Output("lexical-group-color-controls", "children"),
    Input("lexical-facet-fig", "figure"),
    Input("group-color-palette", "data"),
)

for graph in ("ItemFig", "lexical-facet-fig"):
    clientside_callback(
        ClientsideFunction(namespace="groupColors", function_name="recolorGrouped"),
        Output(graph, "figure", allow_duplicate=True),
        Input(graph, "figure"),
        Input("group-color-palette", "data"),
        prevent_initial_call=True,
    )

clientside_callback(
    ClientsideFunction(namespace="groupColors", function_name="recolorUmap"),
    Output("UMAPfig", "figure", allow_duplicate=True),
    Input("umap-color-dropdown", "value"),
    Input("UMAPfig", "figure"),
    Input("group-color-palette", "data"),
    State("informants-store", "data"),
    prevent_initial_call=True,
)

# Browser-side display toggles (CI bands, NX fade) for the lexical figure
clientside_callback(
    ClientsideFunction(namespace="groupColors", function_name="lexicalDisplay"),
    Output("lexical-facet-fig", "figure", allow_duplicate=True),
    Input("lexical-facet-fig", "figure"),
    Input("lexical-show-ci-switch", "checked"),
    Input("lexical-nx-opacity-switch", "checked"),
    prevent_initial_call=True,
)
