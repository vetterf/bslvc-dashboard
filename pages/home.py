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
    "The BSLVC Dashboard offers researchers interactive tools to explore the Bamberg Survey of Language "
    "Variation and Change database. Visualize the data from different perspectives, filter by sociodemographic variables, "
    "and export results — all with the latest version of the dataset. Navigate to any section on the left to begin.",
    size="md",
    mb="md"
),
dmc.Text(
    "Note: This platform is under active development. Some features are still being implemented, "
    "and existing functionality may change in future versions.",
    size="md",
    mb="md",
    c="dimmed"
),

    dmc.Divider(mb="lg"),
    # Key features section
    dmc.Title("Platform Features", order=3, mb="md"),
    
    dmc.Grid([
        dmc.GridCol([
            dmc.Card([
                dmc.Group([
                    dmc.ThemeIcon(
                        DashIconify(icon="mdi:account-filter", width=28),
                        size="xl",
                        radius="md",
                        variant="light",
                    ),
                    dmc.Title("Sociodemographic Filtering", order=5),
                ], mb="sm"),
                dmc.Text(
                    "Isolate specific participant groups for targeted analysis by filtering across "
                    "sociodemographic variables. Currently available filters: age, gender, variety and years lived in home country.",
                    size="sm"
                ),
            ], withBorder=True, shadow="sm", radius="md", p="md", h="100%"),
        ], span=6),

        dmc.GridCol([
            dmc.Card([
                dmc.Group([
                    dmc.ThemeIcon(
                        DashIconify(icon="mdi:chart-box-outline", width=28),
                        size="xl",
                        radius="md",
                        variant="light",
                    ),
                    dmc.Title("Interactive Visualizations", order=5),
                ], mb="sm"),
                dmc.Text(
                    "Explore the distribution of variables included in the BSLVC in interactive plots. "
                    "Switch between different plot types such as box plots, dot plots with error bars and stacked diverging bar charts.",
                    size="sm"
                ),
            ], withBorder=True, shadow="sm", radius="md", p="md", h="100%"),
        ], span=6),
        
        dmc.GridCol([
            dmc.Card([
                dmc.Group([
                    dmc.ThemeIcon(
                        DashIconify(icon="mdi:map-marker-distance", width=28),
                        size="xl",
                        radius="md",
                        variant="light",
                    ),
                    dmc.Title("Dimensionality Reduction & Group Comparison", order=5),
                ], mb="sm"),
                dmc.Text(
                    "Visualize participant similarity in a lower-dimensional space using UMAP (Uniform Manifold Approximation and Projection). "
                    "Select and compare different participant groups to identify distinguishing linguistic features using Random Forests.",
                    size="sm"
                ),
            ], withBorder=True, shadow="sm", radius="md", p="md", h="100%"),
        ], span=6),
        
        dmc.GridCol([
            dmc.Card([
                dmc.Group([
                    dmc.ThemeIcon(
                        DashIconify(icon="mdi:table-account", width=28),
                        size="xl",
                        radius="md",
                        variant="light",
                    ),
                    dmc.Title("Explore Sociodemographic Data", order=5),
                ], mb="sm"),
                dmc.Text(
                    "Explore the sociolinguistic profile of BSLVC participants through manually standardized demographic data including languages spoken at home, "
                    "educational qualifications (participants, partners, and parents), years spent abroad, and schooling history.",
                    size="sm"
                ),
            ], withBorder=True, shadow="sm", radius="md", p="md", h="100%"),
        ], span=6),
        
        dmc.GridCol([
            dmc.Card([
                dmc.Group([
                    dmc.ThemeIcon(
                        dmc.Text("💾", size="xl"),
                        size="xl",
                        radius="md",
                        variant="light",
                    ),
                    dmc.Title("Download Plots & Data", order=5),
                ], mb="sm"),
                dmc.Text(
                    "Download plots directly as SVG files. Export filtered data for use in external statistical software "
                    "for easy integration with R, Python, or other analytical tools.",
                    size="sm"
                ),
            ], withBorder=True, shadow="sm", radius="md", p="md", h="100%"),
        ], span=6),
        
        dmc.GridCol([
            dmc.Card([
                dmc.Group([
                    dmc.ThemeIcon(
                        DashIconify(icon="mdi:code-braces-box", width=28),
                        size="xl",
                        radius="md",
                        variant="light",
                    ),
                    dmc.Title("Open Source", order=5),
                ], mb="sm"),
                dmc.Text(
                    "The BSLVC Dashboard is fully open-source and hosted on the Open Science Framework (OSF). "
                    "You can access and download the complete codebase (soon).",
                    size="sm"
                ),
            ], withBorder=True, shadow="sm", radius="md", p="md", h="100%"),
        ], span=6),
    ], gutter="md", mb="lg"),
    

    ], withBorder=True, shadow="sm", radius="md", p="lg", mb="lg"),
    
    
   
    
], fluid=True, style={"maxWidth": "1400px", "margin": "0 auto"})