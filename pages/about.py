import dash_mantine_components as dmc
from dash import register_page, dcc
from dash_iconify import DashIconify

register_page(__name__, path="/about", name="About")

layout = dmc.Container([
    dmc.Card([
        # Main Title
        dmc.Group([
            DashIconify(icon="tabler:info-square", width=32, color="#228be6"),
            dmc.Title("About the BSLVC Dashboard", order=2)
        ], gap="sm", mb="lg"),
        
        # Introduction
        dmc.Text(
            "The BSLVC Dashboard is an interactive web application for exploring and analyzing data from the Bamberg Survey of Language Variation and Change. "
            "It provides researchers with powerful tools to visualize lexical (coming soon) and grammatical variation across varieties of English.",
            size="md",
            mb="xl"
        ),
        
        # Project Homepage Section
        dmc.Card([
            dmc.Group([
                dmc.ThemeIcon(
                    DashIconify(icon="tabler:world", width=20),
                    size="lg",
                    radius="md",
                    variant="light",
                    color="blue"
                ),
                dmc.Title("Project Information", order=4)
            ], gap="sm", mb="md"),
            dmc.Text(
                """The BSLVC project investigates the use of lexical and grammatical structures in varieties of English
around the world. Its overarching goal is to identify (supra-)
regional patterns, globalization trends and sociolinguistic variation through the systematic description, profiling and
comparison of questionnaire data. For this purpose, more than 6,000
questionnaires have (so far) been collected in over 10 countries where English is used as a native
language (ENL), a second language (ESL) or as a foreign language (EFL). The survey elicits
sociodemographic data, informant ratings for lexical preferences and stylistically differentiated
intuition ratings for a broad range of grammatical structures, including 56 features from the
electronic World Atlas of Varieties of English (https://ewave-atlas.org/).

While data collection started in 2008, the current phase of the project is funded by the DFG. For more information, please visit the project homepage.
""",
                mb="sm"
            ),
            dmc.Anchor(
                dmc.Button(
                    "Visit Project Homepage",
                    leftSection=DashIconify(icon="tabler:external-link", width=16),
                    variant="light",
                    color="blue"
                ),
                href="https://www.uni-bamberg.de/en/eng-ling/forschung/the-bslvc-project-dfg-funded/",
                target="_blank",
                style={"textDecoration": "none"}
            )
        ], p="md", withBorder=True, radius="md", mb="xl", style={"backgroundColor": "#f8f9fa"}),
        
        # Citation Section
        dmc.Card([
            dmc.Group([
                dmc.ThemeIcon(
                    DashIconify(icon="tabler:quote", width=20),
                    size="lg",
                    radius="md",
                    variant="light",
                    color="green"
                ),
                dmc.Title("How to Cite", order=4)
            ], gap="sm", mb="md"),
            dmc.Text(
                "If you use the BSLVC Dashboard or data from the BSLVC project in your research, please cite it appropriately:",
                mb="md"
            ),
            
            # Dashboard Citation
            dmc.Stack([
                dmc.Text("Dashboard Citation:", fw=600, size="sm"),
                dmc.Text("The Dashboard will have its own DOI and OSF repo soon. For the time being, cite as follows:"),
                dmc.Text(
                    "Vetter, Fabian. 2025. BSLVC Dashboard (Version 0.1.2) [Software]. University of Bamberg. https://bslvc.eng-ling.uni-bamberg.de",
                ),
                
                dmc.Space(h="md"),
                
                # Corpus Citation
                dmc.Text("Data Set Citation:", fw=600, size="sm"),
                dmc.Text("The BSLVC data set is currently unpublished. We expect to release the full data set in 2027 at the latest. Until then, please cite as follows:"),
                dmc.Text(
                    "Krug, Manfred, Fabian Vetter & Lukas Sönning. 2025. The Bamberg Survey of Language Variation and Change (Version 251118) [Data set]. University of Bamberg.",
                ),
                
                dmc.Space(h="md"),
            ], gap="xs")
        ], p="md", withBorder=True, radius="md", mb="xl", style={"backgroundColor": "#f8f9fa"}),
        
        # Contact Information
        dmc.Card([
            dmc.Group([
                dmc.ThemeIcon(
                    DashIconify(icon="tabler:mail", width=20),
                    size="lg",
                    radius="md",
                    variant="light",
                    color="orange"
                ),
                dmc.Title("Contact & Support", order=4)
            ], gap="sm", mb="md"),
            dmc.Text(
                "For questions, feedback, or support regarding the BSLVC Dashboard, please contact the project team through the project homepage.",
                size="sm"
            )
        ], p="md", withBorder=True, radius="md", style={"backgroundColor": "#f8f9fa"}),
        
    ], withBorder=True, shadow="sm", radius="md", p="lg", mb="lg")
], fluid=True, style={"maxWidth": "1600px", "margin": "0 auto", "paddingLeft": "20px", "paddingRight": "20px"})
