import dash_mantine_components as dmc
from dash import register_page, dcc, callback, Output, Input
import pages.data.retrieve_data as retrieve_data
import plotly.express as px
import pandas as pd
import numpy as np
from dash_iconify import DashIconify
import dash_ag_grid as dag

register_page(__name__, path="/data-overview", name="Data Overview")

# Load data
Informants = retrieve_data.getInformantData()
grammarData = retrieve_data.getGrammarData(imputed=False)
grammarMeta = retrieve_data.getGrammarMeta()
GrammarItemsCols = retrieve_data.getGrammarItemsCols()

try:
    lexicalData = retrieve_data.getLexicalData(imputed=False)
    LexicalItemsCols = retrieve_data.getLexicalItemsCols()
    HAS_LEXICAL = True
except:
    lexicalData = pd.DataFrame()
    LexicalItemsCols = []
    HAS_LEXICAL = False

# Feature group descriptions (manually curated)
FEATURE_GROUPS_MANUAL = {
    'Subjunctive': {
        'description': 'Constructions involving subjunctive mood, including present and past subjunctive forms in conditional and subordinate clauses.',
        'items': ['A10','B20','B21','B7','I23','L24','N1','N4','N5','D2','D7','E17','I13','K5','N14','F2','G19','K24']
    },
    'Gender attribution': {
        'description': 'Patterns of gender assignment and pronoun usage, including variation in animate and inanimate noun reference.',
        'items': ['A14','A18','B16','E10','F9','G16','G17','H2','K6','L10']
    },
    'Comparative strategies': {
        'description': 'Different ways of expressing comparison, including morphological (e.g., -er) and periphrastic (e.g., more) forms.',
        'items': ['A12','A21','B12','B15','C11','C16','E14','E2','E8','F11','G20','G9','H25','I15','I24','J11','J7','K12','L21','L4','M11','M17','M25']
    },
    'Coordinated subjects': {
        'description': 'Agreement patterns and word order in constructions with multiple subjects joined by conjunctions.',
        'items': ['J24','D23','D5','G25','H22','N13','B4']
    },
    'Agreement/concord': {
        'description': 'Subject-verb agreement and noun-modifier concord patterns, including variation in number and person agreement.',
        'items': ['B13','B17','E1','E5','J12','K13','M3','M6','N9','F10','G4','H23','M15','N23','C6','D12','D8','F4','I25','I7','J25','M14','A23','K15','A5','C17','E18','H15','J15','J16','K21','M1','N10']
    },
    '(Invariant) question tags': {
        'description': 'Use of tag questions at the end of statements, including invariant forms (e.g., "innit") and standard variable forms.',
        'items': ['H7','A8','C3','D9','E23','F22','G24','G3','I16','K2','K4a','K4b','N21']
    },
    'Levelling': {
        'description': 'Reduction of morphological distinctions, such as leveling of verb forms, pronouns, or other grammatical categories.',
        'items': ['E21','A17','C4','D10','F21','F5','G12','G22','H13','H19','M4','N2','F3','J19']
    },
    'Negation': {
        'description': 'Different negation strategies, including negative concord, placement of negation, and use of negative markers.',
        'items': ['J19','A2','A3','B19','C18','E11','E16','H6','I21','I5','J18','J6','L11','N15','N7']
    },
    'Articles': {
        'description': 'Use and omission of definite and indefinite articles, including contexts where article usage varies across varieties.',
        'items': ['B5','C15','D13','D14','D20','F14','F17','G1','G10','G14','G26','H26','H3','I12','I26','I8','J13','J26','K26','L26','N18','N19','N8']
    },
    'Prepositions': {
        'description': 'Variation in prepositional usage, including choice of preposition, omission, and use of different prepositional phrases.',
        'items': ['G13','G15','H14','I11','I6','J10','J17','J2','J8','L15','L17','L22','L5','L8','M16','M21','M23','M7','M8','N12']
    }
}

def generate_dynamic_feature_groups():
    """Generate feature groups dynamically from grammar metadata, similar to grammar.py"""
    groups = {
        'manual': {},
        'mode': {},
        'group': {},
        'ewave': {}
    }
    
    # Add manual groups
    groups['manual'] = FEATURE_GROUPS_MANUAL
    
    # Dynamic groups from section column (Mode)
    if 'section' in grammarMeta.columns:
        section_groups = grammarMeta.groupby('section', observed=True)['question_code'].apply(list).to_dict()
        for section, codes in sorted(section_groups.items()):
            if section and pd.notna(section):
                groups['mode'][section] = {
                    'description': f'Items from the {section} section of the questionnaire.',
                    'items': codes
                }
    
    # Dynamic groups from group_finegrained column
    if 'group_finegrained' in grammarMeta.columns:
        group_groups = grammarMeta.groupby('group_finegrained', observed=True)['question_code'].apply(list).to_dict()
        for group, codes in sorted(group_groups.items()):
            if group and pd.notna(group) and group.strip():
                groups['group'][group] = {
                    'description': f'Items related to {group.lower()}.',
                    'items': codes
                }
    
    # Dynamic groups from feature_ewave column
    if 'feature_ewave' in grammarMeta.columns:
        ewave_groups = grammarMeta.groupby('feature_ewave', observed=True)['question_code'].apply(list).to_dict()
        for feature, codes in sorted(ewave_groups.items()):
            if feature and pd.notna(feature) and feature.strip():
                groups['ewave'][feature] = {
                    'description': f'eWAVE feature: {feature}',
                    'items': codes
                }
    
    return groups

# Generate all feature groups
FEATURE_GROUPS = generate_dynamic_feature_groups()


def create_summary_card():
    """Create summary statistics card with map"""
    total_participants = len(Informants)
    total_grammar_items = len(GrammarItemsCols)
    total_lexical_items = len(LexicalItemsCols) if HAS_LEXICAL else 0
    varieties = Informants[Informants['MainVariety'] != 'Other']['MainVariety'].nunique()
    years = sorted(Informants['Year'].dropna().unique())
    year_range = f"{int(min(years))}-{int(max(years))}" if len(years) > 0 else "N/A"
    
    # Calculate completion rates
    grammar_participants = grammarData['InformantID'].nunique() if not grammarData.empty else 0
    lexical_participants = lexicalData['InformantID'].nunique() if HAS_LEXICAL and not lexicalData.empty else 0
    both_sections = len(set(grammarData['InformantID'].unique() if not grammarData.empty else []) & 
                         set(lexicalData['InformantID'].unique() if HAS_LEXICAL and not lexicalData.empty else []))
    
    return dmc.AccordionItem([
        dmc.AccordionControl(
            dmc.Group([
                DashIconify(icon="tabler:database", width=20, color="#1f77b4"),
                dmc.Text("Dataset Overview", fw=500)
            ], gap="sm")
        ),
        dmc.AccordionPanel([
            dmc.Stack([
                dmc.SimpleGrid([
                    dmc.Stack([
                        dmc.Text("Total Participants", size="sm", c="dimmed"),
                        dmc.Title(str(total_participants), order=2, c="#1f77b4")
                    ], gap=0, align="center"),
                    dmc.Stack([
                        dmc.Text("Grammar Items", size="sm", c="dimmed"),
                        dmc.Title(str(total_grammar_items), order=2, c="#1f77b4")
                    ], gap=0, align="center"),
                    dmc.Stack([
                        dmc.Text("Lexical Items", size="sm", c="dimmed"),
                        dmc.Title(str(total_lexical_items), order=2, c="#1f77b4")
                    ], gap=0, align="center"),
                    dmc.Stack([
                        dmc.Text("Varieties", size="sm", c="dimmed"),
                        dmc.Title(str(varieties), order=2, c="#1f77b4")
                    ], gap=0, align="center"),
                    dmc.Stack([
                        dmc.Text("Survey Years", size="sm", c="dimmed"),
                        dmc.Title(year_range, order=2, c="#1f77b4")
                    ], gap=0, align="center"),
                ], cols=5, spacing="lg", mb="md"),
                dmc.Group([
                    dmc.Text("Data Completion:", size="sm", fw=600),
                    dmc.Badge(f"Lexical: {lexical_participants}", color="green", variant="light"),
                    dmc.Badge(f"Lexical & Grammar: {grammar_participants}", color="blue", variant="light"),
                ], gap="xs", mb="md"),
                dmc.Divider(my="sm"),
                dmc.Text(
                    "Geographic distribution of survey participants. Markers indicate where data is available: blue markers show locations with both grammar and lexical data, green markers show locations with lexical data only. Varieties with fewer than 10 respondents are excluded from the map.",
                    size="sm",
                    c="dimmed",
                    mb="sm"
                ),
                dcc.Graph(
                    id='participants-map-overview',
                    figure=create_participants_map(),
                    config={'displayModeBar': True, 'scrollZoom': True}
                )
            ], gap="sm")
        ])
    ], value="overview")


def create_participants_plot():
    """Create stacked bar chart of participants per variety"""
    # Check if participants have grammar data by looking at A1 or G1 columns
    grammar_participants = set()
    if not grammarData.empty:
        # A participant has grammar data if A1 or G1 is not "ND"
        for col in ['A1', 'G1']:
            if col in grammarData.columns:
                has_data = grammarData[grammarData[col] != 'ND']['InformantID'].unique()
                grammar_participants.update(has_data)
    
    # All participants have lexical data (as stated)
    lexical_participants = set(Informants['InformantID'].unique())
    
    # Create participation summary
    participant_data = []
    for _, row in Informants.iterrows():
        informant_id = row['InformantID']
        variety = row['MainVariety']
        
        has_grammar = informant_id in grammar_participants
        has_lexical = informant_id in lexical_participants
        
        if has_grammar and has_lexical:
            category = 'Grammar + Lexical'
        elif has_lexical:
            category = 'Lexical only'
        else:
            # This shouldn't happen based on your description
            category = 'Grammar only'
        
        participant_data.append({
            'Variety': variety,
            'Category': category
        })
    
    df = pd.DataFrame(participant_data)
    counts = df.groupby(['Variety', 'Category'], observed=True).size().reset_index(name='Count')
    
    # Sort by total count
    variety_totals = counts.groupby('Variety', observed=True)['Count'].sum().sort_values(ascending=True)
    
    # Add total counts for hover information
    variety_total_dict = variety_totals.to_dict()
    counts['Total'] = counts['Variety'].map(variety_total_dict)
    
    fig = px.bar(
        counts,
        y='Variety',
        x='Count',
        color='Category',
        template='simple_white',
        category_orders={
            'Variety': variety_totals.index.tolist(),
            'Category': ['Grammar + Lexical', 'Lexical only', 'Grammar only']
        },
        color_discrete_map={
            'Grammar + Lexical': '#1f77b4',
            'Lexical only': '#2ca02c',
            'Grammar only': '#ff7f0e'
        },
        hover_data={'Total': True, 'Count': True, 'Category': True, 'Variety': False},
        height=max(400, len(variety_totals) * 25)
    )
    
    fig.update_layout(
        xaxis_title='Number of Participants',
        yaxis_title='Variety',
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="gray")
    
    return fig


def create_participants_map():
    """Create a map showing participant locations with markers.
    
    Shows markers for each variety location, colored by data type availability 
    (grammar + lexical vs. lexical only). All markers have uniform size.
    Varieties without geographic coordinates are excluded.
    """
    import plotly.graph_objects as go
    
    # Get grammar and lexical participant IDs
    grammar_ids = set(grammarData['InformantID'].unique() if not grammarData.empty else [])
    lexical_ids = set(lexicalData['InformantID'].unique() if HAS_LEXICAL and not lexicalData.empty else [])
    
    # Mapping varieties to coordinates
    variety_coords = {
        'England': {'lat': 52.3555, 'lon': -1.1743},
        'Scotland': {'lat': 56.4907, 'lon': -4.2026},
        'Wales': {'lat': 52.1307, 'lon': -3.7837},
        'Ireland': {'lat': 53.4129, 'lon': -8.2439},
        'Jersey': {'lat': 49.2144, 'lon': -2.1312},
        'US': {'lat': 37.0902, 'lon': -95.7129},
        'Hawaii': {'lat': 19.8968, 'lon': -155.5828},
        'Canada': {'lat': 56.1304, 'lon': -106.3468},
        'Gibraltar': {'lat': 36.1408, 'lon': -5.3536},
        'Malta': {'lat': 35.9375, 'lon': 14.3754},
        'Australia': {'lat': -25.2744, 'lon': 133.7751},
        'India': {'lat': 20.5937, 'lon': 78.9629},
        'China': {'lat': 35.8617, 'lon': 104.1954},
        'Puerto Rico': {'lat': 18.2208, 'lon': -66.5901},
        'Slovenia': {'lat': 46.1512, 'lon': 14.9955},
        'Serbia': {'lat': 44.0165, 'lon': 21.0059},
        'Germany': {'lat': 51.1657, 'lon': 10.4515},
        'Italy': {'lat': 41.8719, 'lon': 12.5674},
        'Denmark': {'lat': 56.2639, 'lon': 9.5018},
        'Sweden': {'lat': 60.1282, 'lon': 18.6435}
    }
    
    # Collect marker data
    markers = []
    
    for variety in Informants['MainVariety'].unique():
        if variety not in variety_coords:
            continue
            
        variety_informants = Informants[Informants['MainVariety'] == variety]['InformantID'].unique()
        
        # Count different categories
        both_count = len([id for id in variety_informants if id in grammar_ids and id in lexical_ids])
        lexical_only_count = len([id for id in variety_informants if id in lexical_ids and id not in grammar_ids])
        total_count = len(variety_informants)
        data_type = 'Grammar + Lexical' if both_count > 0 else 'Lexical only'
        color = '#1f77b4' if both_count > 0 else '#2ca02c'  # Blue for Grammar+Lexical, Green for Lexical only
        
        coords = variety_coords[variety]
        markers.append({
            'variety': variety,
            'lat': coords['lat'],
            'lon': coords['lon'],
            'Total': total_count,
            'Grammar + Lexical': both_count,
            'Lexical only': lexical_only_count,
            'data_type': data_type,
            'color': color
        })
    
    # Create figure
    fig = go.Figure()
    
    # Add markers for each data type
    if markers:
        df_markers = pd.DataFrame(markers)
        
        for data_type in ['Grammar + Lexical', 'Lexical only']:
            df_subset = df_markers[df_markers['data_type'] == data_type]
            if not df_subset.empty:
                color = '#1f77b4' if data_type == 'Grammar + Lexical' else '#2ca02c'  # Blue for Grammar+Lexical, Green for Lexical only
                fig.add_trace(go.Scattergeo(
                    lon=df_subset['lon'],
                    lat=df_subset['lat'],
                    text=df_subset['variety'],
                    customdata=df_subset[['Total', 'Grammar + Lexical', 'Lexical only']],
                    hovertemplate='<b>%{text}</b><br>Total: %{customdata[0]}<br>Grammar + Lexical: %{customdata[1]}<br>Lexical only: %{customdata[2]}<extra></extra>',
                    mode='markers',
                    marker=dict(
                        size=10,  # Uniform size for all markers
                        color=color,
                        line=dict(color='white', width=1)
                    ),
                    name=data_type,
                    legendgroup=data_type
                ))
    
    if not fig.data:
        return dmc.Alert(
            "No geographic data available for mapping.",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    fig.update_geos(
        showcountries=True,
        countrycolor="lightgray",
        showcoastlines=True,
        coastlinecolor="gray",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        showocean=True,
        oceancolor="rgb(230, 245, 255)",
        projection_type='natural earth'
    )
    
    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            title=None,
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        )
    )
    
    return fig


def create_participants_year_table():
    """Create heatmap showing distribution of participants per year and variety"""
    # Get grammar and lexical participant IDs
    grammar_ids = set(grammarData['InformantID'].unique() if not grammarData.empty else [])
    lexical_ids = set(lexicalData['InformantID'].unique() if HAS_LEXICAL and not lexicalData.empty else [])
    
    # Get data grouped by year and variety
    year_variety = Informants.groupby(['Year', 'MainVariety'], observed=True).size().reset_index(name='Count')
    year_variety_pivot = year_variety.pivot(index='MainVariety', columns='Year', values='Count').fillna(0).astype(int)
    
    # Sort alphabetically by variety (index)
    year_variety_pivot = year_variety_pivot.sort_index()
    
    # Sort year columns chronologically
    year_cols = sorted(year_variety_pivot.columns, key=lambda x: float(x))
    year_variety_pivot = year_variety_pivot[year_cols]
    
    # Calculate breakdown for each variety-year combination
    hover_data = []
    for variety in year_variety_pivot.index:
        row_hover = []
        for year in year_variety_pivot.columns:
            # Get informants for this variety-year combination
            mask = (Informants['MainVariety'] == variety) & (Informants['Year'] == year)
            variety_year_informants = Informants[mask]['InformantID'].unique()
            
            total = len(variety_year_informants)
            both_count = len([id for id in variety_year_informants if id in grammar_ids and id in lexical_ids])
            lexical_only_count = len([id for id in variety_year_informants if id in lexical_ids and id not in grammar_ids])
            
            hover_text = (
                f"<b>{variety}</b><br>"
                f"Year: {year}<br>"
                f"Total: {total}<br>"
                f"Grammar + Lexical: {both_count}<br>"
                f"Lexical only: {lexical_only_count}"
            )
            row_hover.append(hover_text)
        hover_data.append(row_hover)
    
    # Create a heatmap visualization instead of a table with many zeros
    fig = px.imshow(
        year_variety_pivot,
        labels=dict(x="Year", y="Variety", color="Participants"),
        x=year_variety_pivot.columns.astype(str),
        y=year_variety_pivot.index,
        color_continuous_scale='Blues',
        aspect='auto',
        height=max(300, len(year_variety_pivot) * 30)
    )
    
    # Add text annotations and custom hover
    fig.update_traces(
        text=year_variety_pivot.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_data
    )
    
    fig.update_layout(
        xaxis_title='Survey Year',
        yaxis_title='Variety',
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar=dict(
            title="Count"
        )
    )
    
    return fig


def create_age_gender_histogram():
    """Create histogram showing distribution of participants by age and gender"""
    # Prepare data
    df = Informants[['Age', 'Gender']].copy()
    
    # Remove missing values
    df = df.dropna(subset=['Age', 'Gender'])
    
    # Create age bins
    bins = [0, 18, 25, 35, 45, 55, 65, 100]
    labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    
    # Count by age group and gender
    counts = df.groupby(['Age Group', 'Gender'], observed=True).size().reset_index(name='Count')
    
    fig = px.bar(
        counts,
        x='Age Group',
        y='Count',
        color='Gender',
        barmode='group',
        template='simple_white',
        color_discrete_map={
            'Female': '#2a9d8f',  # Teal
            'Male': '#e76f51',    # Coral/Orange
            'Non-binary': '#9b59b6'  # Purple
        },
        height=350
    )
    
    fig.update_layout(
        xaxis_title='Age Group',
        yaxis_title='Number of Participants',
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(title='Gender', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_grammar_items_table():
    """Create table of grammar items with metadata.
    
    Each unique sentence is shown once. If a sentence appears in both Spoken and Written sections,
    the item codes are combined (e.g., 'A6, K14') and section shows 'Spoken & Written'.
    If a sentence appears only once, section shows 'Written only'.
    """
    meta = grammarMeta.copy()
    
    # Group by sentence and aggregate metadata
    # For sentences that appear twice, combine the item codes and mark as "Spoken & Written"
    grouped_data = []
    
    for sentence, group in meta.groupby('item', observed=True):
        # Get all question codes for this sentence
        question_codes = sorted(group['question_code'].tolist())
        
        # Determine section based on how many times the sentence appears
        if len(group) == 2:
            section = 'Spoken & Written'
            # Combine question codes with comma
            combined_code = ', '.join(question_codes)
        else:
            section = 'Written only'
            combined_code = question_codes[0]
        
        # Take the first row's metadata (all should be identical for the same sentence)
        row_data = {
            'question_code': combined_code,
            'item': sentence,
            'feature': group.iloc[0]['feature'],
            'section': section
        }
        
        # Add optional columns if they exist
        if 'group_finegrained' in group.columns:
            row_data['group_finegrained'] = group.iloc[0]['group_finegrained']
        if 'feature_ewave' in group.columns:
            row_data['feature_ewave'] = group.iloc[0]['feature_ewave']
        if 'group_ewave' in group.columns:
            row_data['group_ewave'] = group.iloc[0]['group_ewave']
        
        grouped_data.append(row_data)
    
    table_data = pd.DataFrame(grouped_data)
    
    # Sort by first question code letter and number
    table_data['letter'] = table_data['question_code'].str[0]
    table_data['numbering'] = table_data['question_code'].str.extract(r'(\d+)').astype(int)
    table_data = table_data.sort_values(by=['letter', 'numbering'])
    table_data = table_data.drop(['letter', 'numbering'], axis=1)
    
    # Select columns to display (excluding Variant Detail)
    available_cols = ['question_code', 'item', 'feature', 'section']
    optional_cols = ['group_finegrained', 'feature_ewave', 'group_ewave']
    
    for col in optional_cols:
        if col in table_data.columns:
            available_cols.append(col)
    
    table_data = table_data[available_cols].copy()
    
    # Rename columns for display
    column_names = {
        'question_code': 'Item Code',
        'item': 'Sentence',
        'feature': 'Feature',
        'section': 'Section',
        'group_finegrained': 'Group (Fine)',
        'feature_ewave': 'eWAVE Feature',
        'group_ewave': 'eWAVE Group'
    }
    table_data.columns = [column_names.get(col, col) for col in table_data.columns]
    
    columnDefs = [
        {'field': 'Item Code', 'width': 120, 'pinned': 'left'},
        {'field': 'Sentence', 'width': 350, 'wrapText': True, 'autoHeight': True},
        {'field': 'Feature', 'width': 180},
        {'field': 'Section', 'width': 140}
    ]
    
    # Add optional columns if they exist
    if 'Group (Fine)' in table_data.columns:
        columnDefs.append({'field': 'Group (Fine)', 'width': 150})
    if 'eWAVE Feature' in table_data.columns:
        columnDefs.append({'field': 'eWAVE Feature', 'width': 150})
    if 'eWAVE Group' in table_data.columns:
        columnDefs.append({'field': 'eWAVE Group', 'width': 150})
    
    return dag.AgGrid(
        id='grammar-items-table',
        rowData=table_data.to_dict('records'),
        columnDefs=columnDefs,
        defaultColDef={
            'sortable': True,
            'filter': True,
            'resizable': True,
            'wrapText': True
        },
        dashGridOptions={
            'pagination': True,
            'paginationPageSize': 20,
            'domLayout': 'normal'
        },
        style={'height': '600px'}
    )


def create_feature_groups_accordion():
    """Create tabbed accordion with feature group descriptions organized by category"""
    
    def create_accordion_for_category(category_groups):
        """Helper function to create an accordion for a specific category of groups"""
        items = []
        
        # Sort groups by name for consistent display
        sorted_groups = sorted(category_groups.items(), key=lambda x: x[0])
        
        for group_name, group_info in sorted_groups:
            # Get sentences for the items in this group
            group_items_meta = grammarMeta[grammarMeta['question_code'].isin(group_info['items'])][['question_code', 'item']].copy()
            
            if group_items_meta.empty:
                # If no items found, still show the group with a note
                items_with_sentences = [
                    dmc.Text("No items found in metadata", size="s", c="dimmed", fs="italic")
                ]
            else:
                # Deduplicate sentences and combine item codes
                sentence_dict = {}
                for _, row in group_items_meta.iterrows():
                    sentence = row['item']
                    code = row['question_code']
                    
                    if sentence not in sentence_dict:
                        sentence_dict[sentence] = []
                    sentence_dict[sentence].append(code)
                
                # Sort by first item code and create display items
                sorted_sentences = sorted(sentence_dict.items(), key=lambda x: x[1][0])
                
                items_with_sentences = []
                for sentence, codes in sorted_sentences:
                    # Combine codes if multiple
                    combined_codes = ', '.join(sorted(codes))
                    
                    items_with_sentences.append(
                        dmc.Text([
                            dmc.Text(f"{combined_codes}: ", size="sm", fw=600, span=True),
                            dmc.Text(sentence, size="sm", c="dark", span=True)
                        ], mb="xs")
                    )
                
                # Count unique sentences for badge
                unique_sentence_count = len(sentence_dict)
            
            items.append(
                dmc.AccordionItem([
                    dmc.AccordionControl(
                        dmc.Group([
                            dmc.Text(group_name, fw=500, size="sm"),
                            dmc.Badge(f"{unique_sentence_count if not group_items_meta.empty else 0} sentences", size="sm", variant="light")
                        ], justify="space-between")
                    ),
                    dmc.AccordionPanel([
                        dmc.Stack(items_with_sentences, gap="xs")
                    ])
                ], value=group_name)
            )
        
        if not items:
            return dmc.Text("No groups available", size="sm", c="dimmed", ta="center", p="md")
        
        return dmc.Accordion(
            children=items,
            variant="filled",
            radius="sm",
            styles={"item": {"maxHeight": "60vh", "overflowY": "auto"}}
        )
    
    # Merge manual curated groups and linguistic groups
    merged_groups = {}
    if FEATURE_GROUPS['manual']:
        merged_groups.update(FEATURE_GROUPS['manual'])
    if FEATURE_GROUPS['group']:
        merged_groups.update(FEATURE_GROUPS['group'])
    
    # Create tabs for each category (Mode removed, Curated + Linguistic merged)
    tabs = []
    
    # Merged curated and linguistic groups
    if merged_groups:
        tabs.append(
            dmc.TabsPanel(
                create_accordion_for_category(merged_groups),
                value="linguistic"
            )
        )
    
    # eWAVE feature groups
    if FEATURE_GROUPS['ewave']:
        tabs.append(
            dmc.TabsPanel(
                create_accordion_for_category(FEATURE_GROUPS['ewave']),
                value="ewave"
            )
        )
    
    return dmc.Tabs(
        [
            dmc.TabsList([
                dmc.TabsTab(
                    dmc.Group([
                        DashIconify(icon="tabler:book", width=16),
                        dmc.Text("Feature Groups", size="sm"),
                        dmc.Badge(str(len(merged_groups)), size="sm", variant="light")
                    ], gap="xs"),
                    value="linguistic"
                ),
                dmc.TabsTab(
                    dmc.Group([
                        DashIconify(icon="tabler:world", width=16),
                        dmc.Text("eWAVE Features", size="sm"),
                        dmc.Badge(str(len(FEATURE_GROUPS['ewave'])), size="sm", variant="light")
                    ], gap="xs"),
                    value="ewave"
                ),
            ]),
            *tabs
        ],
        value="linguistic",
        variant="outline",
        color="blue"
    )


def create_lexical_items_plot():
    """Create dot plot of lexical items with mean scores by variety"""
    if not HAS_LEXICAL or lexicalData.empty:
        return dmc.Alert(
            "Lexical data not available",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    # Calculate mean per item and variety
    plot_data = []
    item_overall_means = {}
    
    for item in LexicalItemsCols:
        if item in lexicalData.columns:
            # Convert to numeric first
            item_values = pd.to_numeric(lexicalData[item], errors='coerce')
            
            # Calculate overall mean for sorting
            overall_mean = item_values.mean()
            item_overall_means[item] = overall_mean if pd.notna(overall_mean) else 0
            
            # Calculate means per variety
            for variety in Informants['MainVariety'].unique():
                # Get informants for this variety
                variety_informants = Informants[Informants['MainVariety'] == variety]['InformantID'].unique()
                
                # Filter lexical data for this variety
                variety_mask = lexicalData['InformantID'].isin(variety_informants)
                variety_values = item_values[variety_mask]
                
                mean_val = variety_values.mean()
                if pd.notna(mean_val):
                    plot_data.append({
                        'Item': item,
                        'Variety': variety,
                        'Mean': mean_val
                    })
    
    if not plot_data:
        return dmc.Alert(
            "No valid lexical data to display",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    df_plot = pd.DataFrame(plot_data)
    
    # Sort items by overall mean
    item_order = sorted(item_overall_means.keys(), key=lambda x: item_overall_means[x], reverse=True)
    
    fig = px.scatter(
        df_plot,
        x='Mean',
        y='Item',
        color='Variety',
        template='simple_white',
        category_orders={'Item': item_order},
        height=max(400, len(item_order) * 15)
    )
    
    fig.update_traces(marker=dict(size=8))
    fig.update_xaxes(title_text='Mean Score', range=[-2, 2])
    fig.update_yaxes(
        title_text='Lexical Item',
        tickmode='linear',  # Show all tick labels
        dtick=1  # Show every item
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig


def create_lexical_items_heatmap():
    """Create heatmap of lexical items with mean scores by variety"""
    if not HAS_LEXICAL or lexicalData.empty:
        return dmc.Alert(
            "Lexical data not available",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    # Calculate mean per item and variety
    heatmap_data = {}
    item_overall_means = {}
    variety_overall_means = {}
    
    for item in LexicalItemsCols:
        if item in lexicalData.columns:
            # Convert to numeric first
            item_values = pd.to_numeric(lexicalData[item], errors='coerce')
            
            # Calculate overall mean for sorting
            overall_mean = item_values.mean()
            item_overall_means[item] = overall_mean if pd.notna(overall_mean) else 0
            
            # Calculate means per variety
            variety_means = {}
            for variety in Informants['MainVariety'].unique():
                # Get informants for this variety
                variety_informants = Informants[Informants['MainVariety'] == variety]['InformantID'].unique()
                
                # Filter lexical data for this variety
                variety_mask = lexicalData['InformantID'].isin(variety_informants)
                variety_values = item_values[variety_mask]
                
                mean_val = variety_values.mean()
                if pd.notna(mean_val):
                    variety_means[variety] = mean_val
                    # Accumulate for variety grand mean
                    if variety not in variety_overall_means:
                        variety_overall_means[variety] = []
                    variety_overall_means[variety].append(mean_val)
            
            heatmap_data[item] = variety_means
    
    if not heatmap_data:
        return dmc.Alert(
            "No valid lexical data to display",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    # Create DataFrame for heatmap
    df_heatmap = pd.DataFrame(heatmap_data).T
    
    # Sort items by overall mean
    item_order = sorted(item_overall_means.keys(), key=lambda x: item_overall_means[x], reverse=True)
    df_heatmap = df_heatmap.loc[item_order]
    
    # Calculate grand mean per variety and sort varieties by it
    variety_grand_means = {v: np.mean(means) for v, means in variety_overall_means.items()}
    variety_order = sorted(variety_grand_means.keys(), key=lambda x: variety_grand_means[x], reverse=True)
    
    # Reorder columns by variety grand mean
    df_heatmap = df_heatmap[[v for v in variety_order if v in df_heatmap.columns]]
    
    fig = px.imshow(
        df_heatmap,
        labels=dict(x="Variety", y="Lexical Item", color="Mean Score"),
        x=df_heatmap.columns,
        y=df_heatmap.index,
        color_continuous_scale='ylgnbu',  # Teal-Green scale (green to blue gradient)
        color_continuous_midpoint=0,
        aspect="auto",
        template='simple_white',
        height=max(500, len(df_heatmap) * 20)
    )
    
    fig.update_xaxes(side="top")
    fig.update_layout(
        margin=dict(l=10, r=10, t=80, b=10),
        xaxis={'tickangle': -45},
        coloraxis_colorbar=dict(
            title=dict(
                text="Mean Score<br>+2 = British (Blue)<br>-2 = American (Green)",
                side="right"
            )
        )
    )
    
    return fig


def create_lexical_statistics_panel():
    """Create statistical summary panel for lexical data"""
    if not HAS_LEXICAL or lexicalData.empty:
        return dmc.Alert(
            "Lexical data not available",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    # Calculate statistics per variety
    stats_by_variety = []
    
    for variety in sorted(Informants['MainVariety'].unique()):
        # Get informants for this variety
        variety_informants = Informants[Informants['MainVariety'] == variety]['InformantID'].unique()
        
        # Filter lexical data for this variety
        variety_data = lexicalData[lexicalData['InformantID'].isin(variety_informants)]
        
        # Collect all lexical values for this variety
        all_values = []
        for item in LexicalItemsCols:
            if item in variety_data.columns:
                values = pd.to_numeric(variety_data[item], errors='coerce').dropna()
                all_values.extend(values.tolist())
        
        if all_values:
            stats_by_variety.append({
                'Variety': variety,
                'Mean': round(np.mean(all_values), 2),
                'Median': round(np.median(all_values), 2),
                'Std Dev': round(np.std(all_values), 2),
                'Min': round(np.min(all_values), 2),
                'Max': round(np.max(all_values), 2),
                'N Values': len(all_values)
            })
    
    if not stats_by_variety:
        return dmc.Alert(
            "Unable to calculate statistics",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    df_stats = pd.DataFrame(stats_by_variety)
    
    columnDefs = [
        {'field': 'Variety', 'width': 150, 'pinned': 'left'},
        {'field': 'Mean', 'width': 100, 'type': 'numericColumn'},
        {'field': 'Median', 'width': 100, 'type': 'numericColumn'},
        {'field': 'Std Dev', 'width': 100, 'type': 'numericColumn'},
        {'field': 'Min', 'width': 90, 'type': 'numericColumn'},
        {'field': 'Max', 'width': 90, 'type': 'numericColumn'},
        {'field': 'N Values', 'width': 110, 'type': 'numericColumn'}
    ]
    
    return dag.AgGrid(
        id='lexical-stats-table',
        rowData=df_stats.to_dict('records'),
        columnDefs=columnDefs,
        defaultColDef={'sortable': True, 'resizable': True},
        dashGridOptions={'pagination': False, 'domLayout': 'autoHeight'},
        style={'height': 'auto'}
    )


def create_grammar_coverage_plot():
    """Create visualization showing grammar feature group coverage per variety"""
    if grammarData.empty:
        return dmc.Alert(
            "Grammar data not available",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    # Calculate coverage per feature group and variety (using manually curated groups)
    coverage_data = []
    
    for group_name, group_info in FEATURE_GROUPS['manual'].items():
        for variety in sorted(Informants['MainVariety'].unique()):
            # Get informants for this variety
            variety_informants = Informants[Informants['MainVariety'] == variety]['InformantID'].unique()
            
            # Filter grammar data for this variety
            variety_data = grammarData[grammarData['InformantID'].isin(variety_informants)]
            
            if variety_data.empty:
                continue
            
            # Count how many items in this group have data (not "ND")
            items_with_data = 0
            total_items = 0
            
            for item in group_info['items']:
                if item in variety_data.columns:
                    total_items += 1
                    # Check if any participant has data for this item
                    has_data = (variety_data[item] != 'ND').any()
                    if has_data:
                        items_with_data += 1
            
            if total_items > 0:
                coverage_pct = (items_with_data / total_items) * 100
                coverage_data.append({
                    'Feature Group': group_name,
                    'Variety': variety,
                    'Coverage (%)': coverage_pct,
                    'Items Covered': f"{items_with_data}/{total_items}"
                })
    
    if not coverage_data:
        return dmc.Alert(
            "Unable to calculate grammar coverage",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    df_coverage = pd.DataFrame(coverage_data)
    
    fig = px.bar(
        df_coverage,
        x='Coverage (%)',
        y='Feature Group',
        color='Variety',
        barmode='group',
        template='simple_white',
        height=max(400, len(FEATURE_GROUPS['manual']) * 40),
        hover_data=['Items Covered']
    )
    
    fig.update_layout(
        xaxis_title='Coverage (%)',
        yaxis_title='Feature Group',
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(range=[0, 100])
    
    return fig


def get_lexical_items_data():
    """Helper function to get lexical items data for table and download"""
    if not HAS_LEXICAL or lexicalData.empty:
        return pd.DataFrame()
    
    # Calculate mean per item and variety
    lexical_summary = []
    for item in LexicalItemsCols:
        if item in lexicalData.columns:
            # Convert to numeric first
            item_values = pd.to_numeric(lexicalData[item], errors='coerce')
            
            # Calculate overall mean
            overall_mean = item_values.mean()
            
            # Calculate means per variety
            variety_means = {}
            for variety in Informants['MainVariety'].unique():
                # Get informants for this variety
                variety_informants = Informants[Informants['MainVariety'] == variety]['InformantID'].unique()
                
                # Filter lexical data for this variety
                variety_mask = lexicalData['InformantID'].isin(variety_informants)
                variety_values = item_values[variety_mask]
                
                mean_val = variety_values.mean()
                variety_means[variety] = round(mean_val, 2) if pd.notna(mean_val) else 'N/A'
            
            lexical_summary.append({
                'Item': item,
                'Overall Mean': round(overall_mean, 2) if pd.notna(overall_mean) else 'N/A',
                **variety_means
            })
    
    return pd.DataFrame(lexical_summary)


def create_lexical_items_table():
    """Create table of lexical items with mean scores"""
    if not HAS_LEXICAL or lexicalData.empty:
        return dmc.Alert(
            "Lexical data not available",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    df_lexical = get_lexical_items_data()
    
    if df_lexical.empty:
        return dmc.Alert(
            "No lexical data available",
            title="Info",
            color="blue",
            icon=DashIconify(icon="tabler:info-circle")
        )
    
    columnDefs = [
        {'field': 'Item', 'pinned': 'left', 'width': 120},
        {'field': 'Overall Mean', 'width': 120, 'type': 'numericColumn'}
    ]
    
    for variety in Informants['MainVariety'].unique():
        if variety in df_lexical.columns:
            columnDefs.append({
                'field': variety,
                'width': 100,
                'type': 'numericColumn'
            })
    
    return dmc.Stack([
        dmc.Group([
            dmc.Button(
                "Download Table (CSV)",
                id="download-lexical-table-btn",
                leftSection=DashIconify(icon="tabler:download"),
                variant="light",
                color="blue"
            ),
            dcc.Download(id="download-lexical-table")
        ], justify="flex-end", mb="sm"),
        dag.AgGrid(
            id='lexical-items-table',
            rowData=df_lexical.to_dict('records'),
            columnDefs=columnDefs,
            defaultColDef={'sortable': True, 'filter': True, 'resizable': True},
            dashGridOptions={
                'pagination': True,
                'paginationPageSize': 20,
                'domLayout': 'normal'
            },
            style={'height': '600px'}
        )
    ], gap="xs")


# Layout
layout = dmc.Container([
    dmc.Stack([
        # Collapsible sections using Accordion
        dmc.Accordion([
            # Dataset Overview section (includes map)
            create_summary_card(),
            
            # Participants section
            dmc.AccordionItem([
                dmc.AccordionControl(
                    dmc.Group([
                        DashIconify(icon="tabler:users", width=20, color="#1f77b4"),
                        dmc.Text("Participants", fw=500)
                    ], gap="sm")
                ),
                dmc.AccordionPanel([
                    dmc.Stack([
                        dmc.Title("Age & Gender Distribution", order=5, mb="sm"),
                        dcc.Graph(
                            id='participants-age-gender-hist',
                            figure=create_age_gender_histogram(),
                            config={'displayModeBar': False}
                        ),
                        dmc.Divider(my="sm"),
                        dmc.Title("Participants per Variety", order=5, mb="sm"),
                        dcc.Graph(
                            id='participants-variety-plot',
                            figure=create_participants_plot(),
                            config={'displayModeBar': False}
                        ),
                        dmc.Divider(my="md"),
                        dmc.Title("Participants by Year and Variety", order=5, mb="sm"),
                        dcc.Graph(
                            id='participants-year-heatmap',
                            figure=create_participants_year_table(),
                            config={'displayModeBar': False}
                        )
                    ], gap="md")
                ])
            ], value="participants"),
            
            # Grammar items section
            dmc.AccordionItem([
                dmc.AccordionControl(
                    dmc.Group([
                        DashIconify(icon="tabler:library", width=20, color="#1f77b4"),
                        dmc.Text("Grammar Items", fw=500)
                    ], gap="sm")
                ),
                dmc.AccordionPanel([
                    dmc.Stack([
                        dmc.Title("All Grammar Items", order=5, mb="sm"),
                        dmc.Text(
                            "The table below shows all unique grammatical sentences in the survey. Each sentence is shown once. If a sentence appears in both Spoken and Written sections, both item codes are listed (e.g., 'A6, K14') and the section shows 'Spoken & Written'. Sentences appearing only in the written section show 'Written only'.",
                            size="sm",
                            c="dimmed",
                            mb="sm"
                        ),
                        create_grammar_items_table(),
                        dmc.Divider(my="md"),
                        dmc.Title("Feature Groups", order=5, mb="sm"),
                        dmc.Text(
                            "Grammar items are organized into feature groups available in the Grammar Sets page. Click each group to see its description and included items with sentences.",
                            size="sm",
                            c="dimmed",
                            mb="sm"
                        ),
                        create_feature_groups_accordion()
                    ], gap="md")
                ])
            ], value="grammar"),
            
            # Lexical items section (if available)
            dmc.AccordionItem([
                dmc.AccordionControl(
                    dmc.Group([
                        DashIconify(icon="tabler:book-2", width=20, color="#1f77b4"),
                        dmc.Text("Lexical Items", fw=500)
                    ], gap="sm")
                ),
                dmc.AccordionPanel([
                    dmc.Stack([
                        dmc.Title("Lexical Items - Heatmap", order=5, mb="sm"),
                        dmc.Text(
                            "Heatmap showing mean scores for all lexical items across varieties. Colors range from green (American variant, -2) to blue (British variant, +2). Varieties are ordered by their grand mean score.",
                            size="sm",
                            c="dimmed",
                            mb="sm"
                        ),
                        dcc.Graph(
                            id='lexical-items-heatmap',
                            figure=create_lexical_items_heatmap(),
                            config={'displayModeBar': False}
                        ),
                        dmc.Divider(my="md"),
                        dmc.Title("Lexical Items - Detailed Table", order=5, mb="sm"),
                        create_lexical_items_table()
                    ], gap="md")
                ])
            ], value="lexical") if HAS_LEXICAL else None,
            
        ], variant="separated", radius="md", multiple=True, value=["overview"]),
        
    ], gap="lg")
], size="xl", py="xl")


# Callback for downloading lexical table
@callback(
    Output("download-lexical-table", "data"),
    Input("download-lexical-table-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_lexical_table(n_clicks):
    """Download lexical items table as CSV"""
    df = get_lexical_items_data()
    if not df.empty:
        # Remove NameSchool column if it exists (privacy protection)
        if 'NameSchool' in df.columns:
            df = df.drop(columns=['NameSchool'])
        return dcc.send_data_frame(df.to_csv, "lexical_items_means.csv", index=False)
    return None
