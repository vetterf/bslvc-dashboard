import dash_mantine_components as dmc
from dash import register_page
from dash_iconify import DashIconify

register_page(__name__, path="/privacy", name="Privacy")

layout = dmc.Container([
    dmc.Card([
        dmc.Group([
            DashIconify(icon="tabler:shield-lock", width=32, color="#228be6"),
            dmc.Title("Privacy Statement", order=2)
        ], gap="sm", mb="lg"),

        dmc.Text(
            "This page explains what data the BSLVC Dashboard stores in your browser while you use it. "
            "The dashboard does not use cookies, analytics, or any third-party trackers, and it does not "
            "collect or transmit any information about you as a visitor.",
            size="md",
            mb="xl"
        ),

        dmc.Card([
            dmc.Group([
                dmc.ThemeIcon(DashIconify(icon="tabler:database-off", width=20), size="lg", radius="md",
                              variant="light", color="green"),
                dmc.Title("What is never stored in your browser", order=4)
            ], gap="sm", mb="md"),
            dmc.List([
                dmc.ListItem("Any tracking, analytics, or advertising cookies."),
            ]),
        ], p="md", withBorder=True, radius="md", mb="lg", style={"backgroundColor": "#f8f9fa"}),

        dmc.Card([
            dmc.Group([
                dmc.ThemeIcon(DashIconify(icon="tabler:clock-hour-4", width=20), size="lg", radius="md",
                              variant="light", color="orange"),
                dmc.Title("Session-only data (cleared when you close the browser tab)", order=4)
            ], gap="sm", mb="md"),
            dmc.Text("Stored in your browser's sessionStorage, and only for as long as the tab stays open:", mb="sm"),
            dmc.List([
                dmc.ListItem("General: your light/dark theme choice, whether the sidebar is collapsed, "
                              "and an internal flag used only when an admin-triggered cache reset is requested via URL."),
                dmc.ListItem("Grammar module: the most recently rendered Item/UMAP plot (so it can be redisplayed "
                              "without re-rendering after navigating away), and the list of currently selected "
                              "participant codes used for the Sociodemographic Details tab."),
            ]),
        ], p="md", withBorder=True, radius="md", mb="lg", style={"backgroundColor": "#f8f9fa"}),

        dmc.Card([
            dmc.Group([
                dmc.ThemeIcon(DashIconify(icon="tabler:device-floppy", width=20), size="lg", radius="md",
                              variant="light", color="red"),
                dmc.Title("Persistent data (kept until you clear your browser data)", order=4)
            ], gap="sm", mb="md"),
            dmc.Text(
                "The Grammar module's \u201cSave Settings\u201d button (Advanced Actions) is the only feature in "
                "the whole dashboard that writes to your browser's localStorage, where it remains until you clear "
                "it or use \u201cRestore Settings\u201d. It stores:",
                mb="sm"
            ),
            dmc.List([
                dmc.ListItem("The participant codes and grammar item codes you had selected."),
                dmc.ListItem("Your chosen plot settings (grouping, sorting, UMAP parameters, item-pairs/imputation toggles)."),
            ]),
            dmc.Text(
                "All other selections and switches across both the Grammar and Lexical modules (participant/item "
                "trees, plot-type toggles, sort order, facet options, etc.) use Dash's in-memory persistence: they "
                "are kept while you interact with the page, but are NOT written to browser storage and reset on a "
                "page refresh.",
                mt="sm", size="sm", c="dimmed"
            ),
        ], p="md", withBorder=True, radius="md", mb="lg", style={"backgroundColor": "#f8f9fa"}),

        dmc.Divider(mb="md"),
        dmc.Text(
            "To remove any of the above at any time, use your browser's \u201cClear site data\u201d / \u201cClear "
            "browsing data\u201d option for this site, or simply close the browser tab to clear session-only data.",
            size="sm", c="dimmed"
        ),
    ], withBorder=True, shadow="sm", radius="md", p="xl")
], fluid=False, size="md", py="xl")
