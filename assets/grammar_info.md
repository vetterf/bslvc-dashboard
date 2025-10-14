# Grammar Analysis Tools

## Purpose

This section analyzes grammatical variation patterns across participant groups using acceptability ratings for grammatical constructions.

## Data Structure

- **Participants**: Survey respondents with demographic and sociolinguistic metadata
- **Grammatical Items**: Sentences tested for acceptability, categorized by grammatical features  
- **Ratings**: Acceptability scores (1-5 scale) for each item-participant combination
- **Mode Differences**: Comparison between spoken and written response variants

## Analysis Functions

### Informants

**Purpose**: View participant data and demographic information.

**Functions**:
- Display participant demographics (age, gender, nationality, ethnicity)
- Show language background and location data
- Apply filters by participant characteristics
- Access education and location timeline information
- Browse data in sortable table format

### Items

**Purpose**: Analyze grammatical features and compare ratings across groups.

**Functions**:
- Compare acceptability ratings between language varieties
- Filter items by grammatical categories (subjunctive, negation, articles)
- Sort by mean ratings or standard deviation
- Group participants by variety, variety type, or gender
- View metadata for grammatical constructions

### Informant Similarity

**Purpose**: Visualize participant similarity patterns using UMAP dimensionality reduction.

**Functions**:
- Generate UMAP plots showing participant clustering
- Color points by variety, variety type, or gender
- Adjust UMAP parameters (neighbors, distance metrics, standardization)
- Select participant subsets for comparison
- Analyze spoken vs. written response differences
- Filter by grammatical categories or custom item sets
- Apply Leiden clustering algorithm to identify participant groups

## Analysis Methods

### Statistical Techniques
- UMAP dimensionality reduction for similarity visualization
- Random Forest analysis for feature importance
- Leiden clustering for participant grouping
- Descriptive statistics for rating distributions

### Data Processing
- Missing value handling through imputation
- Feature standardization options
- Categorical variable encoding
- Response mode comparison (spoken vs. written)

## Getting Started

Navigate between tabs to explore different aspects of the grammatical data. Each tab includes filtering options to focus analysis on specific participant groups or grammatical features. Use the settings panels on the right side of each tab to customize visualizations and apply filters.

The dashboard supports both individual item analysis and mode difference analysis (comparing spoken vs. written responses for the same grammatical constructions).
