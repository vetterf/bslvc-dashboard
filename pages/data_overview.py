import dash_mantine_components as dmc
from dash import register_page
import pages.data.retrieve_data as retrieve_data
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, dash_table, html,ctx, callback, Output, Input, State, clientside_callback, no_update
import pickle
from dash.exceptions import PreventUpdate
import polars as pl
import pyarrow as pa
from dash_iconify import DashIconify
import dash_ag_grid as dag
import plotly.graph_objects as go
import plotly.figure_factory as ff
dmc.add_figure_templates()
# register page for navigation
#register_page(__name__, path="/data_overview", name="Data Overview")

# load data
grammarData = retrieve_data.getGrammarData(imputed=True)
GrammarItemsCols = retrieve_data.getGrammarItemsCols()
Informants = retrieve_data.getInformantData()
grammarDataRaw = retrieve_data.getGrammarData(imputed=False)
grammarMeta = retrieve_data.getGrammarMeta()

# presets
# to do, base this on the meta table, not handcoded
item_presets = [
    {'label':'All','value':GrammarItemsCols},
    {'label':'Spoken','value':retrieve_data.getGrammarItemsCols("spoken")},
    {'label':'Written','value':retrieve_data.getGrammarItemsCols("written")},
    {'label':'Custom','value':[]},
    {'label':'Subjunctive','value':['A10','B20','B21','B7','I23','L24','N1','N4','N5','D2','D7','E17','I13','K5','N14','F2','G19','K24']},
    {'label':'Gender attribution','value':['A14','A18','B16','E10','F9','G16','G17','H2','K6','L10']},
    {'label':'Comparative strategies','value':['A12','A21','B12','B15','C11','C16','E14','E2','E8','F11','G20','G9','H25','I15','I24','J11','J7','K12','L21','L4','M11','M17','M25']},
    {'label':'Coordinated subjects','value':['J24','D23','D5','G25','H22','N13','B4']},
    {'label':'Agreement/concord','value':['B13','B17','E1','E5','J12','K13','M3','M6','N9','F10','G4','H23','M15','N23','C6','D12','D8','F4','I25','I7','J25','M14','A23','K15','A5','C17','E18','H15','J15','J16','K21','M1','N10']},
    {'label':'(Invariant) question tags','value':['H7','A8','C3','D9','E23','F22','G24','G3','I16','K2','K4a','K4b','N21']},
    {'label':'Levelling','value':['E21','A17','C4','D10','F21','F5','G12','G22','H13','H19','M4','N2','F3','J19']},
    {'label':'Negation','value':['J19','A2','A3','B19','C18','E11','E16','H6','I21','I5','J18','J6','L11','N15','N7']},
    {'label':'Articles','value':['B5','C15','D13','D14','D20','F14','F17','G1','G10','G14','G26','H26','H3','I12','I26','I8','J13','J26','K26','L26','N18','N19','N8']},
    {'label':'Prepositions','value':['G13','G15','H14','I11','I6','J10','J17','J2','J8','L15','L17','L22','L5','L8','M16','M21','M23','M7','M8','N12']},
    ]
labels_dict = [{'label': preset['label'], 'value': preset['label']} for preset in item_presets]

emptyPlot = go.Figure()
emptyPlot.update_layout(
    xaxis_title="",
    yaxis_title="",
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0),  # Minimize padding
    height=200,  # Set a fixed height for the empty plot
    width=200,  # Set a fixed width for the empty plot
    template="simple_white",
    modebar_remove=True,  # Disable modebar
)
emptyPlot.add_annotation(
        text="No data.",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20)
         )


def drawParticipantsTree():
    #data = retrieve_data.getInformantDataGrammar(columns=['InformantID', 'MainVariety','Year'], imputed=True)
    data = Informants.copy(deep=True)
    data = data.loc[:,['InformantID', 'MainVariety','Year']]
    # for each country draw a tree with the years and informants
    countries = data['MainVariety'].unique()
    treeData = []
    for country in countries:
        countryData = data[data['MainVariety'] == country]
        years = countryData['Year'].unique()
        countryTree = {
            'value': country,
            'label': country,
            'children': [{'label': year, 'value': country + '_' + year, 'children': [{'value': informant, 'label': informant} for informant in countryData[countryData['Year'] == year]['InformantID']]} for year in years]
        }
        treeData.append(countryTree)
    #treeData = [{'title': 'Informants', 'key': 'informants', 'children': treeData}]
    return treeData

def drawGrammarItemsTree():
    #data = retrieve_data.getGrammarData(imputed=True)
    #data = grammarData.copy(deep=True)
    meta = grammarMeta.copy(deep=True)
    meta.loc[:,'letter'] = meta['question_code'].str[0]
    meta.loc[:,'numbering'] = meta['question_code'].str.extract(r'(\d+)')
    meta.loc[:,'numbering']= meta.loc[:,'numbering'].astype(int)
    meta.sort_values(by=['letter','numbering'],ascending=True,inplace=True)
    spokenLetters = ['A','B','C','D','E','F']
    writtenLetters = ['G','H','I','J','K','L','M','N']
    spokenChildren = [{'label': letter, 'value': letter, 'children':[{'label': x + ': ' + y, 'value': x} for x,y in zip(meta.loc[meta['letter']==letter]['question_code'],meta.loc[meta['letter']==letter]['feature'])]} for letter in spokenLetters ]
    writtenChildren = [{'label': letter, 'value': letter, 'children':[{'label': x + ': ' + y, 'value': x} for x,y in zip(meta.loc[meta['letter']==letter]['question_code'],meta.loc[meta['letter']==letter]['feature'])]} for letter in writtenLetters ]
    SpokenCols = {
       'label': 'Spoken section',
       'value': 'spoken',
       'children': spokenChildren
    }

    WrittenCols = {
       'label': 'Written section',
       'value': 'written',
       'children':writtenChildren
    }
    
    
    treeData = [SpokenCols, WrittenCols]
    return treeData

def getAgeGenderPlot(informants):
    # Get age and gender data from informants
    try:
        data = informants.copy(deep=True)
        gender_mapping = {
            'nb': 'non-binary',
            'non-binary': 'non-binary',
            'malte': 'male',
            'NULL': 'NA',
            'f': 'female',
            'female': 'female',
            'm': 'male',
            'male': 'male'
        }
        data['Gender'] = data['Gender'].map(gender_mapping).fillna('NA')
        # Count NaN values in the Age and Gender columns
        nan_rows = data[data['Age'].isna() | data['Gender'].isna()]
        nan_count = len(nan_rows)  # Count of NaN rows
        
        # Drop rows with NaN in 'Age' or 'Gender'
        data = data.dropna(subset=['Age', 'Gender'])
        
        # Convert Age to float for plotting
        data['Age'] = data['Age'].astype(float)
        

        # Create a histogram using Plotly's built-in histogram
        fig = px.histogram(
            data,
            x='Age',
            color='Gender',
            template="simple_white",
            category_orders={"Gender": ["female", "male", "non-binary", "NA"]},  # Order of gender categories
            nbins=25  # Adjust the number of bins as needed
        )
        
        # Update axes and layout
        fig.update_xaxes(title_text='Age')
        fig.update_yaxes(title_text='Informants')
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize padding
            showlegend=True,  # Show legend for gender categories
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = emptyPlot  # Use an empty plot if there's an error

    return fig

def getMainVarietiesPlot(informants):
    try:
        # Get main variety data from informants
        data = informants.copy(deep=True)
        
        # Count occurrences of each variety
        variety_counts = data['MainVariety'].value_counts()
        
        # Group varieties with fewer than 10 informants into "Other"
        data['MainVariety'] = data['MainVariety'].apply(
            lambda x: x if variety_counts[x] >= 10 else 'Other'
        )
        
        # Recalculate counts grouped by MainVariety and Year
        grouped_counts = data.groupby(['MainVariety', 'Year']).size().reset_index(name='counts')
        
        # Calculate overall frequency of each variety for sorting
        overall_counts = grouped_counts.groupby('MainVariety')['counts'].sum().reset_index()
        overall_counts = overall_counts.sort_values(by='counts', ascending=False)
        
        # Ensure years are ordered
        grouped_counts['Year'] = grouped_counts['Year'].astype(int)  # Convert Year to string for consistent ordering
        grouped_counts = grouped_counts.merge(overall_counts, on='MainVariety', suffixes=('', '_total'))
        grouped_counts = grouped_counts.sort_values(by=['Year'],ascending=True)
        VarietyOrder = overall_counts['MainVariety'].tolist() + ['Other'] if 'Other' in overall_counts['MainVariety'].tolist() else overall_counts['MainVariety'].tolist()
        height = len(VarietyOrder) * 25
        if height < 150:
            height = 150
        # Create a bar plot with the grouped varieties, colored by year, and swap axes
        fig = px.bar(
            grouped_counts,
            y='MainVariety',
            x='counts',
            color='Year',
            template="simple_white",
            barmode='stack',  
            color_continuous_scale='Blues_r',
            category_orders={
                'MainVariety': VarietyOrder},
            height=height,
            hover_data={'counts': True, 'counts_total': True}   # Adjust height based on number of varieties
        )
        
        # Update axes and layout
        fig.update_xaxes(title_text='Informants')
        fig.update_traces(marker_line_width=0.5, marker_line_color="gray")
        fig.update_yaxes(title_text='Main Variety',automargin=True)  # Ensure all y-axis labels are visible
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize padding
            showlegend=False,  # Show legend for years
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = emptyPlot  # Use an empty plot if there's an error
    return fig

def getYearsLivedOutsidePlot(informants):
    try:
        # Get data for years lived outside home country
        data = informants.copy(deep=True)
        
        # Count NaN values in 'YearsLivedOutside'
        nan_count = data['YearsLivedOutside'].isna().sum()
        
        # Drop rows with missing values in 'YearsLivedOutside'
        data = data.dropna(subset=['YearsLivedOutside'])
        
        # Create a histogram
        fig = px.histogram(
            data,
            x='YearsLivedOutside',
            nbins=20,  # Adjust the number of bins as needed
            title=f'Years Lived Outside Home Country (NA Count: {nan_count})',
            template="simple_white"
        )
        
        # Update axes and layout
        fig.update_xaxes(title_text='Years Lived Outside')
        fig.update_yaxes(title_text='Number of Informants')
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),  # Minimize padding
            showlegend=False,  # Hide legend if not needed
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = emptyPlot  # Use an empty plot if there's an error
    return fig

def getYearsLivedOtherEnglishPlot(informants):
    try:
        # Get data for years lived in other English-speaking countries
        data = informants.copy(deep=True)
        
        # Count NaN values in 'YearsLivedOtherE'
        nan_count = data['YearsLivedOtherEnglish'].isna().sum()
        
        # Drop rows with missing values in 'YearsLivedOtherE'
        data = data.dropna(subset=['YearsLivedOtherEnglish'])
        
        # Create a histogram
        fig = px.histogram(
            data,
            x='YearsLivedOtherEnglish',
            nbins=20,  # Adjust the number of bins as needed
            title=f'Years Lived in Other English-Speaking Countries (NA Count: {nan_count})',
            template="simple_white"
        )
        
        # Update axes and layout
        fig.update_xaxes(title_text='Years Lived in Other English-Speaking Countries')
        fig.update_yaxes(title_text='Number of Informants')
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),  # Minimize padding
            showlegend=False,  # Hide legend if not needed
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = emptyPlot  # Use an empty plot if there's an error
    return fig

def getRatioMainVarietyPlot(informants):
    try:
        # Get data for Ratio_MainVariety
        data = informants.copy(deep=True)
        
        # Count NaN values in 'Ratio_MainVariety'
        nan_count = data['RatioMainVariety'].isna().sum()
        
        # Drop rows with missing values in 'Ratio_MainVariety'
        data = data.dropna(subset=['RatioMainVariety'])
        
        # Create a histogram plot
        fig = px.histogram(
            data,
            x='RatioMainVariety',
            title=f'Ratio of Years Lived in Main Variety to Age (NA Count: {nan_count})',
            template="simple_white"
        )
        
        # Update axes and layout
        fig.update_xaxes(title_text='Ratio (Years Lived in Main Variety / Age)')
        fig.update_yaxes(title_text='Informants')
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize padding
            showlegend=False,  # Hide legend if not needed
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = emptyPlot  # Use an empty plot if there's an error   
    return fig

def getFloatHistogramPlot (informants,ColName="RatioMainVariety"):
    try:
        # Get data for Ratio_MainVariety
        data = informants.copy(deep=True)
        
        # Count NaN values in 'Ratio_MainVariety'
        nan_count = data[ColName].isna().sum()
        
        # Drop rows with missing values in 'Ratio_MainVariety'
        data = data.dropna(subset=[ColName])
        hist_data = [np.array(data[ColName].to_list())]
        fig = ff.create_distplot(hist_data,show_hist=False,group_labels=[ColName])
            
        # Create a histogram plot
        """ fig = px.histogram(
            data,
            x=ColName,
            title=f'{ColName} (NA Count: {nan_count})',
            template="simple_white"
        ) """
        
        # Update axes and layout
        fig.update_xaxes(title_text=ColName)
        #fig.update_yaxes(title_text='Informants')
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize padding
            showlegend=False,
            template="simple_white",
            height=200  # Hide legend if not needed
        )
        fig.update_layout(modebar_remove=True)  # Disable modebar
    except Exception as e:
        #print(f"Error creating histogram: {e}")
        fig = emptyPlot  # Use an empty plot if there's an error

    return fig


def getCategoryHistogramPlot(informants, ColName="PrimarySchool", GroupOther=True, split="", GenderDistribution=True):
    try: # do it right sometime, works for now
        data = informants.copy(deep=True)
        
        # Fill missing values and standardize empty strings
        data[ColName] = data[ColName].fillna('NA')
        data.loc[data[ColName] == "", ColName] = "NA"
        if split != "":
            data = data.assign(**{ColName: data[ColName].str.split(',')}).explode(ColName)
            data[ColName] = data[ColName].str.strip().str.capitalize()

        if GenderDistribution:
            col_counts = data.groupby([ColName,'Gender']).size().reset_index(name='counts')
            col_counts.columns = [ColName, 'Gender', 'counts']
            # group catergories with fewer than 10 occurrences per Gender into "Other
            if GroupOther:
                col_counts[ColName] = col_counts[ColName].apply(
                    lambda x: x if ((col_counts[col_counts[ColName] == x]['counts'].sum() >= 10) | (x == 'NA') | (x == 'ND')) else 'Other'
                )
            col_counts = col_counts.groupby([ColName,'Gender'], as_index=False)['counts'].sum()
            # calculate overall frequency of each category for sorting
            overall_counts = col_counts.groupby(ColName)['counts'].sum().reset_index()
            overall_counts = overall_counts.sort_values(by='counts', ascending=False)
            col_counts = col_counts.merge(overall_counts, on=ColName, suffixes=('', '_total'))
            col_counts = col_counts.sort_values(by=['counts_total','Gender'], ascending=[False,True])
        else:
            col_counts = data[ColName].value_counts().reset_index()
            col_counts.columns = [ColName, 'counts']
            if GroupOther:
                col_counts[ColName] = col_counts[ColName].apply(
                    lambda x: x if ((col_counts.loc[col_counts[ColName] == x, 'counts'].values[0] >= 10) | (x == 'NA') | (x == 'ND')) else 'Other'
                )
            col_counts = col_counts.groupby(ColName, as_index=False)['counts'].sum()
            col_counts = col_counts.sort_values(by='counts', ascending=False)


        col_Order = list(dict.fromkeys(col_counts[ColName].tolist()))
        col_Order = [category for category in col_Order if category not in ['NA', 'ND', 'Other']] + [category for category in ['NA', 'ND', 'Other'] if category in col_Order]
        height = len(col_Order) * 25
        if height < 200:
            height = 200
    
        # If GenderDistribution is True, include Gender in the plot
        if GenderDistribution:
            
            # Create a bar plot with Gender as color
            fig = px.bar(
                col_counts,
                y=ColName,
                x='counts',
                color='Gender',
                template="simple_white",
                barmode='stack',
                category_orders={ColName: col_Order},
                height=height,
                hover_data={'counts': True}
            )
        else:
            # Create a bar plot without Gender
            fig = px.bar(
                col_counts,
                y=ColName,
                x='counts',
                template="simple_white",
                barmode='stack',
                category_orders={ColName: col_Order},
                height=height,
                hover_data={'counts': True}
            )

        # Update axes and layout
        fig.update_xaxes(title_text='Informants')
        fig.update_traces(marker_line_width=0.5, marker_line_color="gray")
        fig.update_yaxes(title_text=ColName, automargin=True)
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=GenderDistribution,  # Show legend only if GenderDistribution is True
        )
        fig.update_layout(modebar_remove=True)

    except Exception as e:
        # Use an empty plot if there's an error
        fig = emptyPlot

    return fig



AgeGender = dmc.Stack([
    dmc.Text("Age/Gender"),
    dcc.Graph(id="AgeGenderPlot", figure=getAgeGenderPlot(Informants),style={'height': '200px'},config={'displayModeBar': False})
    ])

MainVarieties = dmc.Stack([

    dmc.Text("Main varieties"),
    html.Div(id="NationalityPlotContainer", children=[
    dcc.Graph(id="MainVarietiesPlot", figure=getMainVarietiesPlot(Informants),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

Nationality = dmc.Stack([

    dmc.Text("Nationality"),
    html.Div(id="NationalityPlotContainer", children=[
    dcc.Graph(id="NationalityPlot", figure=getCategoryHistogramPlot(Informants,"Nationality", True, ""),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

EthnicSID = dmc.Stack([

    dmc.Text("Ethnic Self-ID"),
    html.Div(id="EIDPlotContainer", children=[
    dcc.Graph(id="EIDPlot", figure=getCategoryHistogramPlot(Informants,"EthnicSelfID", True,""),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px','overflowY': 'scroll'}),
])

CountryID = dmc.Stack([

    dmc.Text("Country (or region) you identify with most"),
    html.Div(id="CIDPlotContainer", children=[
    dcc.Graph(id="CIDPlot", figure=getCategoryHistogramPlot(Informants,"CountryID", True, ""),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

LanguagesHome = dmc.Stack([

    dmc.Text("Languages used at home"),
    html.Div(id="LanguagesHomePlotContainer", children=[
    dcc.Graph(id="LanguagesHomePlot", figure=getCategoryHistogramPlot(Informants,"LanguageHome", True, ","),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])


LanguageMother = dmc.Stack([

    dmc.Text("Mother's Native Language"),
    html.Div(id="LanguagesMotherPlotContainer", children=[
    dcc.Graph(id="LanguagesMotherPlot", figure=getCategoryHistogramPlot(Informants,"LanguageMother", True, ","),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

LanguageFather = dmc.Stack([

    dmc.Text("Father's Native Language"),
    html.Div(id="LanguagesFatherPlotContainer", children=[
    dcc.Graph(id="LanguagesFatherPlot", figure=getCategoryHistogramPlot(Informants,"LanguageFather", True, ","),config={'displayModeBar': False})
    ], style={'height': 'auto', 'max-height' : '300px', 'overflowY': 'scroll'}),
])

PrimarySchool = dmc.Stack([
    
        dmc.Text("Primary School"),
        dcc.Graph(id="PrimarySchoolPlot", figure=getCategoryHistogramPlot(Informants,"PrimarySchool",True),config={'displayModeBar': False})

])

SecondarySchool = dmc.Stack([
    
        dmc.Text("Secondary School"),
        dcc.Graph(id="SecondarySchoolPlot", figure=getCategoryHistogramPlot(Informants,"SecondarySchool",True),config={'displayModeBar': False})

])
Qualifications = dmc.Stack([
    
        dmc.Text("Highest Qualification"),
        dcc.Graph(id="QualiPlot", figure=getCategoryHistogramPlot(Informants,"Qualifications",True),config={'displayModeBar': False})

])

YearsLivedOutside = dmc.Stack([
    
        dmc.Text("Years lived outside home country"),
        dcc.Graph(id="YLOPlot", figure=getFloatHistogramPlot(Informants,"YearsLivedOutside"),config={'displayModeBar': False})

])

YearsLivedOtherE = dmc.Stack([
    
        dmc.Text("Years lived in other English-speaking countries"),
        dcc.Graph(id="YLOEPlot", figure=getFloatHistogramPlot(Informants,"YearsLivedOtherEnglish"),config={'displayModeBar': False})

])

RatioMainVariety = dmc.Stack([
    
        dmc.Text("Ratio Main Variety"),
        dcc.Graph(id="RatioMainVarietyPlot", figure=getFloatHistogramPlot(Informants,"RatioMainVariety"),config={'displayModeBar': False})

])


# Settings
#filter informants accordion
informantSelectionAccordion = dmc.AccordionItem(
            
            [
                dmc.AccordionControl(
                    "Data: Filter informants",
                    icon=DashIconify(
                        icon="tabler:users-group",
                        color="#1f77b4",
                        width=20,
                    ),
                ),
                dmc.AccordionPanel(
                    dmc.Stack(gap='xl',children=[
                        dmc.Text("In this section, you can select which informants should be included in the analysis. After specifying the filters, click on 'Apply filters' to filter the informants. Your previous selection in the tree selection below will be overwritten."),
                        dmc.CheckboxGroup(
                            id="checkbox-overview-filter-gender", label="Gender",mb=10,
                            children=dmc.Group([
                                dmc.Checkbox(label="Female", value="f"),
                                dmc.Checkbox(label="Male",value="m"),
                                dmc.Checkbox(label="Non-binary",value="nb"),
                                dmc.Checkbox(label="NA",value="NA"),
                            ]), 
                            value=["f","m","nb","NA"]
                        ),
                        dmc.Text("Age:"),
                        dmc.RangeSlider(
                            id="rangeslider-overview-filter-age",
                            min=0,
                            max=100,
                            step=1,
                            marks=[{"value": 0, "label": "0" },
                                   {"value": 110, "label": "110" }],
                            value=[0,110]
                        ),
                        dmc.Text("Ratio main variety (years lived in main variety/age):"),
                        dmc.RangeSlider(
                            id="rangeslider-overview-filter-ratio",
                            min=0,
                            max=100,
                            step=1,
                            marks=[{"value": 0, "label": "0%" },
                                   {"value": 50, "label": "50%" },
                                   {"value": 100, "label": "100%" }],
                            value=[20,100]
                        ),
                        dmc.Button('Apply filters', id='apply-overview-filters', loaderProps={"type": "dots"}),
                        dmc.Text('Selected informants:'),
                        dmc.Tree(id='participantsTreeOverview',data=drawParticipantsTree(), checkboxes=True,checked=Informants['InformantID']),
                          
                    ])
                ),
            ],
            value="LoadData",
        )
SettingsOverview = dmc.Container([
    dmc.Group(children=[
        dmc.Button('Update plots', id='render-Overview-plots', loaderProps={"type": "dots"}),
    ],
    grow=True,
    wrap="nowrap"),
    dmc.Accordion(children=[
        informantSelectionAccordion,
    ], 
                  variant="default",
                  radius="md"),

    
    # id="ratio_as_opacity",
    # size="xs",
    # radius="md",
    # label="Indicate ratio via opacity",
    # checked=False
    # ), 
], fluid=True)


PIAccordion = dmc.Accordion(
    children=[
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    "Languages",
                ),
                dmc.AccordionPanel(
                    children=[
                        LanguagesHome,
                        dmc.Divider(),
                        LanguageMother,
                        dmc.Divider(),
                        LanguageFather
                    ]
                    ),
            ],
            value="languages",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    "Regions & Identification",
                ),
                dmc.AccordionPanel(children=[
                    Nationality,
                    dmc.Divider(),
                    EthnicSID,
                    dmc.Divider(),
                    CountryID
                ]
                ),
            ],
            value="seldif",
        ),
    ], variant="default"
)


InformantsGrid = html.Div(children = [

    # Grid to mimic questionnaire layout
    dmc.Card(children=[
        dmc.CardSection(children=[
            dmc.Image(src='../assets/img/UB_logo.png',styles={"root" : {"width":"100px","height":"100px","float":"right","margin-bottom":"10px"}}),
            dmc.Text("University of Bamberg",size="xl",styles={"root":{"line-height":"1.1"}}),
            dmc.Text("Chair of English Linguistics",size="sm",styles={"root":{"padding-bottom":"10px"}}),
            dmc.Text("Bamberg Survey of Language Variation and Change",fw=700,size="sm"),
            dmc.Text("Participant Information Sheet",size="sm"),
            dmc.Divider(styles={"root":{"margin-top":"20px"}}),
        ],styles={"section":{"margin":"20px"}}),
        dmc.CardSection(children=[
            dmc.Grid(children=[ 
                # "Personal Information" & "Location Timeline"
                dmc.GridCol(children=[
                    dmc.Card(children=[
                        dmc.Text("Personal Information",fw=700),
                        #Age,
                        #dmc.Divider(),
                        #Gender,
                        #dmc.Divider(),
                        AgeGender,
                        dmc.Divider(),
                        MainVarieties,
                        dmc.Divider(),
                        PIAccordion,
                        #LanguagesHome,
                        #dmc.Divider(),
                        #LanguageMother,
                        #dmc.Divider(),
                        #LanguageFather
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md"),
                    dmc.Card(children=[
                        dmc.Text("Location Timeline",fw=700),
                        YearsLivedOutside,
                        dmc.Divider(),
                        YearsLivedOtherE,
                        dmc.Divider(),
                        RatioMainVariety],
                        withBorder=True,
                        shadow="sm",
                        radius="md")
                    ],span=6),
                # "Education Profile"
                dmc.GridCol(children=[
                    dmc.Card(children=[
                        dmc.Text("Education Profile",fw=700),
                        PrimarySchool,
                        dmc.Divider(),
                        SecondarySchool,
                        dmc.Divider(),
                        Qualifications],
                        withBorder=True,
                        shadow="sm",
                        radius="md"),
                    ],span=6),
            ])
        ],styles={"section":{"margin-left":"20px","margin-right":"20px","margin-bottom":"20px"}}),
    ],withBorder=True,shadow="sm",radius="md")
    ])

layout =  html.Div(
    [ dmc.Grid([
        dmc.GridCol(html.Div(children = [
        InformantsGrid,]
        ,id="informants-plot-tab-content",style={"paddingTop": 10}),span=8),
        dmc.GridCol(SettingsOverview,span=4,style={"padding-top":"10px","margin-top": "5px","border-left": "1px solid #f0f0f0","padding-left": "10px"}),
    ], gutter="xl"),
    html.Div(
        [
            html.Div(id="notify-container"),
        ],
    ),
    ])





# apply filter to participant selection tree
@callback(
    [Output('participantsTreeOverview', 'checked',allow_duplicate=True),Output("notify-container", "children", allow_duplicate=True)],
    [Input('apply-overview-filters', 'n_clicks')],
    [State('checkbox-overview-filter-gender', 'value'),State('rangeslider-overview-filter-age', 'value'),State('rangeslider-overview-filter-ratio', 'value')],
    prevent_initial_call=True
)
def updateParticipants(applyButton,genderFilter,ageFilter,ratioFilter):
    #if applyButton is not None:
    if (True):
        data = Informants.copy(deep=True)
        #InformantList = filterInformants(data,genderFilter,ageFilter,ratioFilter,otherFilter)
        #data = data.loc[(data)]
        # filter data

        data = data.loc[((data['Age'].between(ageFilter[0],ageFilter[1]))&(data['Gender'].isin(genderFilter))&(data['RatioMainVariety'].between(ratioFilter[0]/100,ratioFilter[1]/100))),:]    
        # apply filter to data
        if len(data) > 0:
            return data['InformantID'].to_list(), no_update
        
        # show warning if no participants are selected
        else:
            notification = dmc.Notification(
                    id="my-notification",
                    title="Info",
                    message="Your filters resulted in an empty selection, please modify the filters.",
                    color="orange",
                    loading=True,
                    action="show",
                    autoClose=3000,
                    position="top-right"
                    #icon=DashIconify(icon="akar-icons:circle-check"),
            )
            return no_update, notification
    else:
        return no_update, no_update


@callback(
    [Output('AgeGenderPlot','figure'),
    Output('MainVarietiesPlot','figure'),
    Output('NationalityPlot','figure'),
    Output('EIDPlot','figure'),
    Output('CIDPlot','figure'),
    Output('LanguagesHomePlot','figure'),
    Output('LanguagesMotherPlot','figure'),
    Output('LanguagesFatherPlot','figure'),
    Output('PrimarySchoolPlot','figure'),
    Output('SecondarySchoolPlot','figure'),
    Output('QualiPlot','figure'),
    Output('YLOPlot','figure'),
    Output('YLOEPlot','figure'),
    Output('RatioMainVarietyPlot','figure'),],
    Input('render-Overview-plots','n_clicks'),
    [State('participantsTreeOverview', 'checked')],
    prevent_initial_call=True
)
def renderOverviewPlots(BTN,informantsTree):
    button_clicked = ctx.triggered_id
    if button_clicked == 'render-Overview-plots' and BTN is not None:

        # get informants from tree  
        if informantsTree is not None:
            # get informants from tree
            informants = Informants.loc[Informants['InformantID'].isin(informantsTree),:]
            # get plots
        
            AgeGenderPlot = getAgeGenderPlot(informants)
            MainVarietiesPlot = getMainVarietiesPlot(informants)
            NationalityPlot = getCategoryHistogramPlot(informants,"Nationality", True, "")
            EIDPlot = getCategoryHistogramPlot(informants,"EthnicSelfID", True,"")
            CIDPlot = getCategoryHistogramPlot(informants,"CountryID", True, ",")
            LanguagesHomePlot = getCategoryHistogramPlot(informants,"LanguageHome", True, ",")
            LanguagesMotherPlot = getCategoryHistogramPlot(informants,"LanguageMother", True, ",")
            LanguagesFatherPlot = getCategoryHistogramPlot(informants,"LanguageFather", True, ",")
            PrimarySchoolPlot = getCategoryHistogramPlot(informants,"PrimarySchool",True)
            SecondarySchoolPlot = getCategoryHistogramPlot(informants,"SecondarySchool",True)
            QualiPlot = getCategoryHistogramPlot(informants,"Qualifications",True)
            YLOPlot = getFloatHistogramPlot(informants,"YearsLivedOutside")
            YLOEPlot = getFloatHistogramPlot(informants,"YearsLivedOtherEnglish")
            RatioMainVarietyPlot = getFloatHistogramPlot(informants,"RatioMainVariety")
            return AgeGenderPlot, MainVarietiesPlot, NationalityPlot, EIDPlot, CIDPlot, LanguagesHomePlot, LanguagesMotherPlot, LanguagesFatherPlot, PrimarySchoolPlot, SecondarySchoolPlot, QualiPlot, YLOPlot, YLOEPlot, RatioMainVarietyPlot
        return no_update,no_update,no_update,no_update,no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    return no_update,no_update,no_update,no_update,no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

