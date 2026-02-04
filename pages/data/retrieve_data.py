import os
from dataclasses import dataclass

import io
import sqlite3
import pandas as pd
from pages.data.db_credentials import sqlserver_db_config
import sqlalchemy as SA
from dash import dcc

import plotly.express as px

# This is a helper file to retrieve data from the database.
# Also includes further helper functions, e.g. for filtering and color mapping




@dataclass
class Conf:
    # Determine base app directory - use environment variable for Docker or calculate from file location
    appDir: str = os.environ.get('APP_DIR', 
                                os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Data directory - construct from app directory
    dataDir: str = os.environ.get('DATA_DIR',
                                 os.path.join(os.environ.get('APP_DIR', 
                                            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                            'assets', 'data'))
    
    # SQLite file path - construct from data directory
    sqliteFile: str = os.environ.get('DATABASE_PATH', 
                                   os.path.join(os.environ.get('DATA_DIR',
                                              os.path.join(os.environ.get('APP_DIR', 
                                                         os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                                         'assets', 'data')),
                                              'BSLVC_sqlite.db'))
    
    # England mapping CSV file path - construct from data directory (same pattern as database)
    englandMappingFile: str = os.environ.get('ENGLAND_MAPPING_PATH',
                                           os.path.join(os.environ.get('DATA_DIR',
                                                      os.path.join(os.environ.get('APP_DIR',
                                                                 os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                                                 'assets', 'data')),
                                                      'england_N_S_mapping.csv'))
    
    dataFileName: str = 'Questionnaire_db_export_all_inkl_grammar_231002.csv'
    columnsFileName: str = 'column_mapping_lexical.csv'
    metaFileName: str = 'meta_info.csv'
    debug: bool = True  # write print outputs in console for debugging
    source: str = 'sqlite'  # where to get the data from; options: 'sql_server' for sql server, 'sqlite' for sqlite db file
    SQLConfig = sqlserver_db_config[0]
    SQLConfigStr = '; '.join(
        [f"{key}={value}" for key, value in sqlserver_db_config[0].items()])

# to do: if local, use sqlite?
#if Conf.source == 'sqlite':
    #db_connection = sqlite3.connect(Conf.sqliteFile)
    #db_connection = SA.create_engine(f"sqlite:///{Conf.dataDir}BSLVC_sqlite.db")
#Conf.dataDir  = os.path.join(os.path.dirname(__file__), "assets/data", "BSLVC_sqlite.db")
# or
#Conf.sqliteFile = os.environ.get('DATABASE_PATH', '/app/assets/data/BSLVC_sqlite.db')
# Participants to exclude from all queries, appear to be outliers
EXCLUDED_PARTICIPANTS = ["GIB10-041m17","GIB10-042m17", "IND24-0105", "PR15-039m23", "SC16-405f16"]

# function to map variety names to colors
def get_color_for_variety(type="lexical"):
    # Fixed color mapping for varieties
    # Return the complete fixed map to ensure all varieties (including England subdivisions) have colors
    fixed_color_map = {
        "England": "#1f77b4",
        "England_North": "#4a90c4",  # Lighter blue for North England
        "England_South": "#0d5a8f",  # Darker blue for South England
        "England_UNCLEAR": "#7bb3d9",  # Even lighter blue for UNCLEAR England
        "Scotland": "#ff7f0e", 
        "US": "#2ca02c",
        "Gibraltar": "#d62728",
        "Malta": "#9467bd",
        "India": "#8c564b",
        "Puerto Rico": "#e377c2",
        "Slovenia": "#7f7f7f",
        "Germany": "#bcbd22",
        "Sweden": "#17becf",
        "Spain (Balearic Islands)": "#393b79",
        "Other": "#c49c94"
    }
    
    return fixed_color_map


def get_database_version():
    """Get the database version from the DatabaseMetadata table"""
    try:
        if Conf.source == 'sqlite':
            db_connection = sqlite3.connect(Conf.sqliteFile)
            cursor = db_connection.cursor()
            cursor.execute("SELECT Version FROM DatabaseMetadata LIMIT 1")
            result = cursor.fetchone()
            db_connection.close()
            if result:
                return result[0]
    except Exception as e:
        print(f"[WARNING] Could not retrieve database version: {e}")
    return "Unknown"


if Conf.source == 'sql_server':
    db_connection = SA.create_engine(f"mssql+pyodbc:///?odbc_connect={Conf.SQLConfigStr}")
# sharing a sql server connection between threads/callbacks is no problem
# for sqlite it is however, so if sqlite, then a connection if made for every function below. 
# performance-wise this shouldn't be too much of a problem, especially as the app will run locally most of the time.

# Helper function for inner join on two lists
def inner_join_list(list1, list2):
    return [item for item in list1 if item in list2]

def getInformantCols():
    # returns a list with the column names of the Informants table
    # Dynamically loads column names from the database

    db_connection = sqlite3.connect(Conf.sqliteFile)
    cursor = db_connection.cursor()
    cursor.execute("PRAGMA table_info(Informants)")
    columns = [row[1] for row in cursor.fetchall()]  # row[1] contains the column name
    db_connection.close()

    
    return columns

def getLexicalItemsCols():
    # returns a list with the column names of the LexicalItems table
    columns = [
    "aDropInTheOcean",
    "aTap",
    "Aluminium",
    "Anticlockwise",
    "Aubergine",
    "Autumn",
    "Backwards",
    "Bicentenary",
    "Biscuit",
    "Bookings",
    "Boot",
    "CarPark",
    "Centre",
    "Chemists",
    "Ill",
    "PotatoChips",
    "Chips",
    "Cinema",
    "Colour",
    "Cupboard",
    "DrivingLicence",
    "Pacifier",
    "DustBin",
    "FishFingers",
    "Football",
    "Forwards",
    "Globalisation",
    "Glocalisation",
    "Holiday",
    "Liberalisation",
    "JacketPotato",
    "Launderette",
    "PotatoCrisps",
    "Crisps",
    "toLicence",
    "Lift",
    "Localisation",
    "Lorry",
    "Maths",
    "MobilePhone",
    "Modernisation",
    "Nappies",
    "Organisation",
    "Parcel",
    "Pavement",
    "Petrol",
    "PetrolStation",
    "Postman",
    "Pushchair",
    "Railway",
    "Realisation",
    "Roundabout",
    "Rubber",
    "Rubbish",
    "ShoppingTrolley",
    "Sport",
    "StormInATeacup",
    "Subway",
    "toLet",
    "Torch",
    "TouchWood",
    "Trainers",
    "Whilst",
    "Windscreen",
    "aBookAboutChemistry",
    "CompareXToY",
    "TypicalOf",
    "Anyway"
    ]
    return columns

def getGrammarItemsCols(type="all"):
    # returns a list with the column names of the GrammarItems table
    if type=="all":
        columns = [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21", "A22", "A23",
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20", "B21", "B22", "B23",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23",
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19", "D20", "D21", "D22", "D23",
        "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18", "E19", "E20", "E21", "E22", "E23",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23",
        "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17", "G18", "G19", "G20", "G21", "G22", "G23", "G24", "G25", "G26",
        "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20", "H21", "H22", "H23", "H24", "H25", "H26",
        "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13", "I14", "I15", "I16", "I17", "I18", "I19", "I20", "I21", "I22", "I23", "I24", "I25", "I26",
        "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J19", "J20", "J21", "J22", "J23", "J24", "J25", "J26",
        "K1", "K2", "K3", "K4a", "K4b", "K5", "K6", "K7", "K8", "K9", "K10", "K11", "K12", "K13", "K14", "K15", "K16", "K17", "K18", "K19", "K20", "K21", "K22", "K23", "K24", "K25", "K26",
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19", "L20", "L21", "L22", "L23", "L24", "L25", "L26",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14", "M15", "M16", "M17", "M18", "M19", "M20", "M21", "M22", "M23", "M24", "M25",
        "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13", "N14", "N15", "N16", "N17", "N18", "N19", "N20", "N21", "N22", "N23", "N24", "N25"
        ]
    elif type=="spoken":
        columns = [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21", "A22", "A23",
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20", "B21", "B22", "B23",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23",
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19", "D20", "D21", "D22", "D23",
        "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18", "E19", "E20", "E21", "E22", "E23",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23",
        ]
    elif type=="written":
        columns = [
        "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17", "G18", "G19", "G20", "G21", "G22", "G23", "G24", "G25", "G26",
        "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20", "H21", "H22", "H23", "H24", "H25", "H26",
        "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13", "I14", "I15", "I16", "I17", "I18", "I19", "I20", "I21", "I22", "I23", "I24", "I25", "I26",
        "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J19", "J20", "J21", "J22", "J23", "J24", "J25", "J26",
        "K1", "K2", "K3", "K4a", "K4b", "K5", "K6", "K7", "K8", "K9", "K10", "K11", "K12", "K13", "K14", "K15", "K16", "K17", "K18", "K19", "K20", "K21", "K22", "K23", "K24", "K25", "K26",
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19", "L20", "L21", "L22", "L23", "L24", "L25", "L26",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14", "M15", "M16", "M17", "M18", "M19", "M20", "M21", "M22", "M23", "M24", "M25",
        "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13", "N14", "N15", "N16", "N17", "N18", "N19", "N20", "N21", "N22", "N23", "N24", "N25"
        ]
    elif type=="item_pairs":
        columns = ['A1-G21', 'A2-J18', 'A3-I5', 'A4-M19', 'A5-J16', 'A6-K14', 'A7-K3', 'A8-H7', 'A9-N3', 'A10-N1', 'A11-G8', 'A12-G9', 'A13-M20', 'A14-L10', 'A15-L20', 'A16-G6', 'A17-H13', 'A18-K6', 'A19-I14', 'A20-L9', 'A21-G20', 'A22-I20', 'A23-K15', 'B1-K10', 'B2-G5', 'B3-G18', 'B4-G25', 'B5-N18', 'B6-H11', 'B7-I23', 'B8-H16', 'B9-M5', 'B10-N25', 'B11-J5', 'B12-I24', 'B13-N9', 'B14-I9', 'B15-M17', 'B16-G17', 'B17-K13', 'B18-M10', 'B19-J6', 'B20-N4', 'B21-L24', 'B22-J9', 'B23-M24', 'C1-I1', 'C2-H8', 'C3-I16', 'C4-N2', 'C5-J14', 'C6-M14', 'C7-K7', 'C8-K16', 'C9-H21', 'C10-I2', 'C11-H25', 'C12-I4', 'C13-N20', 'C14-J23', 'C15-J13', 'C16-K12', 'C17-H15', 'C18-H6', 'C19-L7', 'C20-I18', 'C21-H4', 'C22-K23', 'C23-L1', 'D1-L23', 'D2-I13', 'D3-L2', 'D4-K9', 'D5-N13', 'D6-J24', 'D7-K5', 'D8-N10', 'D9-G24', 'D10-M4', 'D11-N6', 'D12-J25', 'D13-H3', 'D14-N8', 'D15-H10', 'D16-K1', 'D17-L25', 'D18-K18', 'D19-K11', 'D20-G10', 'D21-M18', 'D22-L13', 'D23-H22', 'E1-M6', 'E2-L21', 'E3-H17', 'E4-K19', 'E5-M3', 'E6-G2', 'E7-K22', 'E8-L4', 'E9-L14', 'E10-H2', 'E11-I21', 'E12-L12', 'E13-I22', 'E14-M25', 'E15-H1', 'E16-N7', 'E17-N14', 'E18-J15', 'E19-N11', 'E20-L19', 'E21-G22', 'E22-N22', 'E23-K4b', 'F1-K8', 'F2-K24', 'F3-J19', 'F4-I7', 'F5-H19', 'F6-N24', 'F7-G11', 'F8-N17', 'F9-G16', 'F10-H23', 'F11-J11', 'F12-J20', 'F13-M12', 'F14-N19', 'F15-G23', 'F16-L3', 'F17-I12', 'F18-K20', 'F19-M2', 'F20-N16', 'F21-G12', 'F22-K4a', 'F23-K25']
    return columns




def getInformantData(columns = None, informants = None, varieties = None):
    # to do: rewrite so that filtering is done in SQL not via pandas
    SQLstatement = 'SELECT * FROM Informants'
    if Conf.source == 'sqlite':
        db_connection = sqlite3.connect(Conf.sqliteFile)
    data = pd.read_sql(SQLstatement, con=db_connection)
    if Conf.source == 'sqlite':
        db_connection.close()
    
    # Exclude specified participants
    data = data[~data['InformantID'].isin(EXCLUDED_PARTICIPANTS)]
    
    # Remove NameSchool, signature, and CommentsTimeline columns if they exist (privacy protection)
    if 'NameSchool' in data.columns:
        data = data.drop(columns=['NameSchool'])
    if 'signature' in data.columns:
        data = data.drop(columns=['signature'])
    if 'CommentsTimeline' in data.columns:
        data = data.drop(columns=['CommentsTimeline'])
    
    float_columns = ['Age', 'YearsLivedOutside', 'YearsLivedInside', 'YearsLivedOtherEnglish', 'Ratio', 'YearsLivedInMainVariety', 'RatioMainVariety']
    data.loc[:,float_columns] = data.loc[:,float_columns].apply(pd.to_numeric, errors='coerce')
    variety_counts = data['MainVariety'].value_counts()

    data['MainVariety'] = data['MainVariety'].apply(lambda x: x if variety_counts.get(x, 0) >= 10 else 'Other')
    # if main variety unclear, set to "Other"
    data['MainVariety'] = data['MainVariety'].apply(lambda x: x if x != 'UNCLEAR' else 'Other')
    if informants is not None:
        data = data[data['InformantID'].isin(informants)]
    
    if varieties is not None:
        data = data[data['MainVariety'].isin(varieties)]
            # drop Gender column
    data = data.drop(columns=["Gender","PrimarySchool","SecondarySchool","Qualifications"], errors="ignore")


    data = data.rename(columns={"gender_normalized": "Gender","primary_school_normalized": "PrimarySchool", "secondary_school_normalized": "SecondarySchool", "highest_qualification": "Qualifications"})

    if columns is not None:
        data = data[columns]

    return data

def getInformantDataGrammar(columns = None, participants = None, varieties = None, imputed = False, regional_mapping = False):
    # Load all columns from Informants table, filtered by InformantIDs that exist in grammar tables
    
    if imputed:
        # Get InformantIDs that exist in the imputed grammar table
        grammar_ids_query = 'SELECT DISTINCT InformantID FROM BSLVC_Grammar_Imputed'
    else:
        # For non-imputed data, get InformantIDs that exist in the grammar table with valid grammar data
        # but only those that are also in the imputed dataset
        grammar_ids_query = '''SELECT DISTINCT G.InformantID FROM BSLVC_Grammar G 
                              WHERE ((G.G1 != 'ND' or G.G1 IS NULL) AND (G.A1 != 'ND' or G.A1 IS NULL))
                              AND G.InformantID IN (SELECT DISTINCT InformantID FROM BSLVC_Grammar_Imputed)'''
    
    if participants is not None:
        # Further filter by specific participants if provided
        participants_clause = ' AND InformantID IN (' + ', '.join(f"'{informant}'" for informant in participants) + ')'
        grammar_ids_query += participants_clause
    
    # Main query to get all Informants data filtered by grammar participants
    SQLstatement = f'''SELECT I.* FROM Informants I 
                      WHERE I.InformantID IN ({grammar_ids_query})'''
    
    if Conf.source == 'sqlite':
        db_connection = sqlite3.connect(Conf.sqliteFile)
    data = pd.read_sql(SQLstatement, con=db_connection)
    if Conf.source == 'sqlite':
        db_connection.close()
    
    # Exclude specified participants
    data = data[~data['InformantID'].isin(EXCLUDED_PARTICIPANTS)]
    
    # Remove NameSchool, signature, and CommentsTimeline columns if they exist (privacy protection)
    if 'NameSchool' in data.columns:
        data = data.drop(columns=['NameSchool'])
    if 'signature' in data.columns:
        data = data.drop(columns=['signature'])
    if 'CommentsTimeline' in data.columns:
        data = data.drop(columns=['CommentsTimeline'])
    
    float_columns = ['Age', 'YearsLivedOutside', 'YearsLivedInside', 'YearsLivedOtherEnglish', 'Ratio', 'YearsLivedInMainVariety', 'RatioMainVariety']
    data.loc[:,float_columns] = data.loc[:,float_columns].apply(pd.to_numeric, errors='coerce')
    data = data[~data['InformantID'].str.startswith('Unnamed')]
    
    # Recode MainVariety to "Other" if count of MainVariety is less than 10
    # For consistency, always use the imputed dataset as reference for variety counts
    if not imputed:
        # For non-imputed data, get variety counts from the imputed dataset for consistency
        if Conf.source == 'sqlite':
            db_connection_ref = sqlite3.connect(Conf.sqliteFile)
        
        # Get variety counts from imputed dataset
        imputed_variety_query = """
            SELECT I.MainVariety, COUNT(*) as count 
            FROM Informants I 
            JOIN BSLVC_Grammar_Imputed G ON I.InformantID = G.InformantID
            WHERE I.MainVariety IS NOT NULL 
            AND I.MainVariety != 'UNCLEAR'
            AND NOT (I.InformantID LIKE 'Unnamed%')
        """
        # Add exclusion of specified participants
        if EXCLUDED_PARTICIPANTS:
            exclusion_clause = " AND I.InformantID NOT IN (" + ', '.join(f"'{p}'" for p in EXCLUDED_PARTICIPANTS) + ")"
            imputed_variety_query += exclusion_clause
        
        imputed_variety_query += " GROUP BY I.MainVariety"
        
        variety_counts_df = pd.read_sql(imputed_variety_query, con=db_connection_ref)
        if Conf.source == 'sqlite':
            db_connection_ref.close()
        
        # Convert to dictionary for easy lookup
        variety_counts = dict(zip(variety_counts_df['MainVariety'], variety_counts_df['count']))
    else:
        # For imputed data, use the current data's variety counts
        variety_counts = data['MainVariety'].value_counts()
    
    data['MainVariety'] = data['MainVariety'].apply(lambda x: x if variety_counts.get(x, 0) >= 10 else 'Other')
    # if main variety unclear, set to "Other"
    data['MainVariety'] = data['MainVariety'].apply(lambda x: x if x != 'UNCLEAR' else 'Other')

    # Apply England North/South mapping if requested
    if regional_mapping:
        # Load the England mapping CSV - use configured path like database file
        regional_mapping_file = Conf.englandMappingFile
        if os.path.exists(regional_mapping_file):
            regional_map_df = pd.read_csv(regional_mapping_file)
            # Create a dictionary for quick lookup (note: CSV has "Informant ID" with space)
            # Rename column if necessary
            if 'Informant ID' in regional_map_df.columns:
                regional_map_df = regional_map_df.rename(columns={'Informant ID': 'InformantID'})
            regional_map_dict = dict(zip(regional_map_df['InformantID'], regional_map_df['north_south']))
            
            # Apply mapping to England participants
            def apply_regional_mapping(row):
                if row['MainVariety'] == 'England':
                    informant_id = row['InformantID']
                    region = regional_map_dict.get(informant_id, 'UNCLEAR')
                    if region == 'north':
                        return 'England_North'
                    elif region == 'south':
                        return 'England_South'
                    else:
                        return 'England_UNCLEAR'
                return row['MainVariety']
            
            data['MainVariety'] = data.apply(apply_regional_mapping, axis=1)

    if varieties is not None:
        data = data[data['MainVariety'].isin(varieties)]
    
    data = data.drop(columns=["Gender","PrimarySchool","SecondarySchool","Qualifications"], errors="ignore")


    data = data.rename(columns={"gender_normalized": "Gender","primary_school_normalized": "PrimarySchool", "secondary_school_normalized": "SecondarySchool", "highest_qualification": "Qualifications"})
    
    if columns is not None:
        data = data[columns] 

    
     
    return data

def getSurveyCount():
    # retrieves the counts of all questionnaires per Country and Year, and the count of questionnaires that have grammar written, spoken or both
    excluded_clause = "'" + "', '".join(EXCLUDED_PARTICIPANTS) + "'"
    SQLstatement = f"""SELECT 
            Year,
            MainVariety,
            COUNT(*) AS TotalCount,
            COUNT(CASE WHEN G1 != 'ND' or G1 IS NULL THEN 1 END) AS GrammarWrittenCount,
            COUNT(CASE WHEN A1 != 'ND' or A1 IS NULL THEN 1 END) AS GrammarSpokenCount,
            COUNT(CASE WHEN ((G1 != 'ND' or G1 IS NULL) AND (A1 != 'ND' or A1 IS NULL)) THEN 1 END) AS GrammarCompleteCount

        FROM 
            BSLVC_ALL
        WHERE 
            InformantID NOT IN ({excluded_clause})
        GROUP BY 
            Year, 
            MainVariety;
        """
    if Conf.source == 'sqlite':
        db_connection = sqlite3.connect(Conf.sqliteFile)
    data = pd.read_sql(SQLstatement, con=db_connection)
    if Conf.source == 'sqlite':
        db_connection.close()
    data['SpokenOnly'] = data['GrammarSpokenCount'] - data['GrammarCompleteCount']
    data['WrittenOnly'] = data['GrammarWrittenCount'] - data['GrammarCompleteCount']
    return data

def getParticipantsByMissingData(max_missing_percent=0):
    """
    Returns a list of InformantIDs filtered by percentage of missing data.
    Missing data includes both NULL values and 'ND' values.
    Only includes participants that are also present in the imputed grammar table.
    
    Args:
        max_missing_percent: Maximum percentage of missing values allowed (0-100).
                           For example, 0 means no missing data, 10 means <10% missing.
    
    Returns:
        List of InformantIDs meeting the criteria.
    """
    # Get all grammar item columns
    grammar_cols = getGrammarItemsCols(type="all")
    total_cols = len(grammar_cols)
    
    # Build SQL query to get all participants with their grammar data
    # Only include participants that are also in the imputed table
    excluded_clause = "'" + "', '".join(EXCLUDED_PARTICIPANTS) + "'"
    
    SQLstatement = f"""
        SELECT InformantID, {', '.join(grammar_cols)}
        FROM BSLVC_Grammar
        WHERE InformantID NOT IN ({excluded_clause})
        AND InformantID IN (SELECT DISTINCT InformantID FROM BSLVC_Grammar_Imputed)
        ORDER BY InformantID
    """
    
    if Conf.source == 'sqlite':
        db_connection = sqlite3.connect(Conf.sqliteFile)
    
    data = pd.read_sql(SQLstatement, con=db_connection)
    
    if Conf.source == 'sqlite':
        db_connection.close()
    
    # Count NULLs and NDs for each participant
    # For each row, count how many grammar columns are NULL or 'ND'
    null_count = data[grammar_cols].isnull().sum(axis=1)
    nd_count = (data[grammar_cols] == 'ND').sum(axis=1)
    total_missing = null_count + nd_count
    
    # Calculate percentage of missing data
    missing_percent = (total_missing / total_cols) * 100
    
    # Filter to participants with missing data below threshold
    if max_missing_percent == 0:
        # Exact match: no missing data at all
        filter_mask = missing_percent == 0
    else:
        # Less than threshold
        filter_mask = missing_percent < max_missing_percent
    
    filtered_participants = data.loc[filter_mask, 'InformantID']
    
    # Return list of InformantIDs
    return filtered_participants.tolist()

def getCompleteGrammarParticipants():
    """
    Returns a list of InformantIDs who have filled in all grammar items (non-imputed data).
    A participant is considered complete if they have non-ND values for all grammar item columns.
    
    This is a convenience wrapper around getParticipantsByMissingData(0).
    """
    return getParticipantsByMissingData(max_missing_percent=0)

def getLexicalData(imputed=False):
    if imputed:
        SQLstatement = "SELECT * FROM BSLVC_Lexical_Imputed"
    else:
        SQLstatement = "SELECT * FROM BSLVC_Lexical"
    if Conf.source == 'sqlite':
        db_connection = sqlite3.connect(Conf.sqliteFile)
    data = pd.read_sql(SQLstatement, con=db_connection)
    if Conf.source == 'sqlite':
        db_connection.close()
    
    # Exclude specified participants
    data = data[~data['InformantID'].isin(EXCLUDED_PARTICIPANTS)]
    
    # Remove NameSchool, signature, and CommentsTimeline columns if they exist (privacy protection)
    if 'NameSchool' in data.columns:
        data = data.drop(columns=['NameSchool'])
    if 'signature' in data.columns:
        data = data.drop(columns=['signature'])
    if 'CommentsTimeline' in data.columns:
        data = data.drop(columns=['CommentsTimeline'])
    
    return data


# implement filter here
def getGrammarData(imputed=False,pairs=False, regional_mapping=False, **kwargs):

    InformantCols = getInformantCols()
    GrammarCols = getGrammarItemsCols(type="all")
    
    # Build WHERE clause for participants filtering
    WhereClause = ""

    if 'items' not in kwargs and pairs:
        # if no items are specified, we assume we want all items
        # but if pairs are specified, we need to filter by item pairs
        item_pairs = getGrammarItemsCols(type="item_pairs")
        item_pairs = [item.split('-') for item in item_pairs]
        # flatten the list of item pairs and remove duplicates
        item_columns = list(set([col for pair in item_pairs for col in pair]))
        GrammarCols = inner_join_list(item_columns, GrammarCols)

    for key, value in kwargs.items():
        if key == 'items' and not pairs:
            # filter data by grammatical items, supplied is a list of column names
            if value is not None:
                if len(value) > 0:
                    GrammarCols = inner_join_list(value, GrammarCols)
        elif key == 'items' and pairs:
            # filter data by item pairs, supplied is a list of item pairs
            if value is not None:
                if len(value) > 0:
                    # item pairs are in the format 'A1-G21', so we need to split them and get the individual columns
                    item_pairs = [item.split('-') for item in value]
                    # flatten the list of item pairs and remove duplicates
                    item_columns = list(set([col for pair in item_pairs for col in pair]))
                    # filter the GrammarCols by the item columns
                    GrammarCols = inner_join_list(item_columns, GrammarCols)

        if key == 'participants':
            if value is not None:
                if len(value) > 0:
                # filter data by participants, supplied is a list of participant IDs
                    WhereClause = ' WHERE I.InformantID IN (' + ', '.join(f"'{informant}'" for informant in value) + ')'
    
    # Build column list for SELECT statement
    # Prefix informant columns with I. and grammar columns with G.
    informant_columns = [f"I.{col}" for col in InformantCols]
    grammar_columns = [f"G.{col}" for col in GrammarCols]
    all_columns = informant_columns + grammar_columns
    
    # Build SQL statement with JOIN to get informant data from Informants table and grammar data from grammar table
    if imputed:
        SQLstatement = f"""SELECT {', '.join(all_columns)} 
                          FROM Informants I 
                          JOIN BSLVC_Grammar_Imputed G ON I.InformantID = G.InformantID"""
    else:
        # For non-imputed data, only get participants that are also in the imputed dataset
        SQLstatement = f"""SELECT {', '.join(all_columns)} 
                          FROM Informants I 
                          JOIN BSLVC_Grammar G ON I.InformantID = G.InformantID
                          WHERE I.InformantID IN (SELECT DISTINCT InformantID FROM BSLVC_Grammar_Imputed)"""
    
    if WhereClause:
        if not imputed and "WHERE" in SQLstatement:
            # If we already have a WHERE clause for non-imputed data, add AND
            SQLstatement = SQLstatement.replace("WHERE", "WHERE") + " AND" + WhereClause.replace("WHERE", "")
        else:
            SQLstatement = SQLstatement + WhereClause
        
    if Conf.source == 'sqlite':
        db_connection = sqlite3.connect(Conf.sqliteFile)
    data = pd.read_sql(SQLstatement, con=db_connection)
    if Conf.source == 'sqlite':
        db_connection.close()
    
    # Exclude specified participants
    data = data[~data['InformantID'].isin(EXCLUDED_PARTICIPANTS)]
    
    # Remove NameSchool column if it exists (privacy protection)
    if 'NameSchool' in data.columns:
        data = data.drop(columns=['NameSchool'])
    if 'signature' in data.columns:
        data = data.drop(columns=['signature'])
    if 'CommentsTimeline' in data.columns:
        data = data.drop(columns=['CommentsTimeline'])
    
    # delete all records where InformantID starts with "Unnamed"

    float_columns = ['Age', 'YearsLivedOutside', 'YearsLivedInside', 'YearsLivedOtherEnglish', 'Ratio', 'YearsLivedInMainVariety', 'RatioMainVariety'] 
    data.loc[:,float_columns] = data.loc[:,float_columns].apply(pd.to_numeric, errors='coerce')

    data = data[~data['InformantID'].str.startswith('Unnamed')]
    
    # Recode MainVariety to "Other" if count of MainVariety is less than 10
    # For consistency, always use the imputed dataset as reference for variety counts
    if not imputed:
        # For non-imputed data, get variety counts from the imputed dataset for consistency
        if Conf.source == 'sqlite':
            db_connection = sqlite3.connect(Conf.sqliteFile)
        
        # Get variety counts from imputed dataset
        imputed_variety_query = """
            SELECT I.MainVariety, COUNT(*) as count 
            FROM Informants I 
            JOIN BSLVC_Grammar_Imputed G ON I.InformantID = G.InformantID
            WHERE I.MainVariety IS NOT NULL 
            AND I.MainVariety != 'UNCLEAR'
            AND NOT (I.InformantID LIKE 'Unnamed%')
        """
        # Add exclusion of specified participants
        if EXCLUDED_PARTICIPANTS:
            exclusion_clause = " AND I.InformantID NOT IN (" + ', '.join(f"'{p}'" for p in EXCLUDED_PARTICIPANTS) + ")"
            imputed_variety_query += exclusion_clause
        
        imputed_variety_query += " GROUP BY I.MainVariety"
        
        variety_counts_df = pd.read_sql(imputed_variety_query, con=db_connection)
        if Conf.source == 'sqlite':
            db_connection.close()
        
        # Convert to dictionary for easy lookup
        variety_counts = dict(zip(variety_counts_df['MainVariety'], variety_counts_df['count']))
    else:
        # For imputed data, use the current data's variety counts
        variety_counts = data['MainVariety'].value_counts()
    
    data.loc[:,'MainVariety'] = data.loc[:,'MainVariety'].apply(lambda x: x if variety_counts.get(x, 0) >= 10 else 'Other')
    data.loc[:,'MainVariety'] = data.loc[:,'MainVariety'].apply(lambda x: x if x != 'UNCLEAR' else 'Other')

    # Apply England North/South mapping if requested
    if regional_mapping:
        # Load the England mapping CSV - use configured path like database file
        regional_mapping_file = Conf.englandMappingFile
        if os.path.exists(regional_mapping_file):
            regional_map_df = pd.read_csv(regional_mapping_file)
            # Create a dictionary for quick lookup (note: CSV has "Informant ID" with space)
            # Rename column if necessary
            if 'Informant ID' in regional_map_df.columns:
                regional_map_df = regional_map_df.rename(columns={'Informant ID': 'InformantID'})
            regional_map_dict = dict(zip(regional_map_df['InformantID'], regional_map_df['north_south']))
            
            # Apply mapping to England participants
            def apply_regional_mapping(row):
                if row['MainVariety'] == 'England':
                    informant_id = row['InformantID']
                    region = regional_map_dict.get(informant_id, 'UNCLEAR')
                    if region == 'north':
                        return 'England_North'
                    elif region == 'south':
                        return 'England_South'
                    else:
                        return 'England_UNCLEAR'
                return row['MainVariety']
            
            data.loc[:, 'MainVariety'] = data.apply(apply_regional_mapping, axis=1)

    if pairs:
        # substract item pairs from each other, spoken item - written item
        #meta = getGrammarMeta(type="item_pairs")
        
        # Create all pair columns at once using pd.concat
        pair_columns = {}
        columns_to_drop = set()
        
        for pair in item_pairs:
            spoken_item = pair[0]
            written_item = pair[1]
            # new column name is the pair name
            pair_name = f"{spoken_item}-{written_item}"
            
            # convert the columns to numeric, errors='coerce' will turn non-numeric values into NaN
            spoken_numeric = pd.to_numeric(data[spoken_item], errors='coerce')
            written_numeric = pd.to_numeric(data[written_item], errors='coerce')
            
            # calculate the difference
            pair_columns[pair_name] = spoken_numeric - written_numeric
            
            # mark columns for dropping
            columns_to_drop.add(spoken_item)
            columns_to_drop.add(written_item)
        
        # Create DataFrame with all pair columns at once
        pair_df = pd.DataFrame(pair_columns, index=data.index)
        
        # Drop the original columns
        data = data.drop(columns=list(columns_to_drop), errors='ignore')
        
        # Concatenate the pair columns to the original data
        data = pd.concat([data, pair_df], axis=1)

            
    return data

def getAllData(imputed=False):
    if imputed:
        SQLstatement = "SELECT * FROM BSLVC_ALL_Imputed"
    else:
        SQLstatement = "SELECT * FROM BSLVC_ALL"
    if Conf.source == 'sqlite':
        db_connection = sqlite3.connect(Conf.sqliteFile)
    data = pd.read_sql(SQLstatement, con=db_connection)
    if Conf.source == 'sqlite':
        db_connection.close()
    
    # Exclude specified participants
    data = data[~data['InformantID'].isin(EXCLUDED_PARTICIPANTS)]
    
    # Remove NameSchool column if it exists (privacy protection)
    if 'NameSchool' in data.columns:
        data = data.drop(columns=['NameSchool'])
    if 'signature' in data.columns:
        data = data.drop(columns=['signature'])
    if 'CommentsTimeline' in data.columns:
        data = data.drop(columns=['CommentsTimeline'])
    
    return data

def getGrammarMeta(type="all_items"):
    # Columns to exclude from all queries
    excluded_columns = [
        "Reading instructions for recordings; stressed items = boldprint",
        "Comment; 2nd feature",
        "related_item"
    ]
    
    if type == "all_items":
        # Get all columns except the excluded ones
        SQLstatement = """
            SELECT question_code, section, item, flagged, control_item, feature, 
                   group_ewave, group_finegrained, variant_detail, feature_ewave, feature_ewave_id, also_in_item
            FROM bslvc_meta
        """
    elif type == "item_pairs":
        # Use SQLite string concatenation syntax (||) instead of CONCAT
        SQLstatement = "SELECT question_code, (question_code || '-' || also_in_item) as item_pair, also_in_item as question_code_written,  section, item, feature, group_ewave,group_finegrained, variant_detail, feature_ewave, feature_ewave_id FROM bslvc_meta where section = 'Spoken'"
    if Conf.source == 'sqlite':
        db_connection = sqlite3.connect(Conf.sqliteFile)
    data = pd.read_sql(SQLstatement, con=db_connection)
    if Conf.source == 'sqlite':
        db_connection.close()
    data.replace({'also_in_item':'x'},'',inplace=True)
    data.fillna('')
    return data