# BSLVC Dashboard Documentation

## Overview

The Bamberg Survey of Language Variation and Change (BSLVC) Dashboard provides interactive analysis tools for exploring grammatical and lexical variation data. The dashboard consists of three main sections: Grammar Sets, Lexical Sets, and Data Overview.

## Grammar Sets Module

### Purpose
Analyze grammatical variation patterns across participant groups using acceptability ratings for grammatical constructions.

### Data Structure
- **Participants**: Survey respondents with demographic and sociolinguistic metadata
- **Grammatical Items**: Sentences tested for acceptability, categorized by grammatical features
- **Ratings**: Acceptability scores (1-5 scale) for each item-participant combination
- **Mode Differences**: Comparison between spoken and written response variants

### Available Tabs

#### 1. Info Tab
Provides overview of dashboard functionality and navigation instructions.

#### 2. Informants Tab
**Function**: Display and filter participant data.

**Data Fields**:
- Demographics: Age, gender, nationality, ethnicity
- Location: Current residence, place of birth, location timeline
- Language Background: Native languages, acquisition contexts
- Education: Qualification levels, institutional affiliations
- Survey Metadata: Participation dates, response completeness

**Filter Options**:
- Variety type (standard vs. dialectal)
- Gender categories
- Age ranges
- Geographic regions
- Education levels

**Table Features**:
- Sortable columns (click headers)
- Resizable columns (drag borders)
- Search filters (text boxes below headers)
- Export functionality

#### 3. Items Tab
**Function**: Analyze individual grammatical constructions and compare group responses.

**Plot Types**:
- Box plots: Distribution of ratings by participant groups
- Violin plots: Density distributions of acceptability scores
- Bar charts: Mean ratings with confidence intervals

**Grouping Variables**:
- Variety (standard German vs. dialectal varieties)
- Variety type (binary classification)
- Gender
- Custom participant selections

**Item Categories**:
- Subjunctive constructions
- Negation patterns
- Article usage
- Word order variations
- Modal verb constructions

**Analysis Features**:
- Sort items by mean rating or standard deviation
- Filter by grammatical category
- Compare spoken vs. written responses
- Statistical significance testing

#### 4. Informant Similarity Tab
**Function**: Visualize participant clustering based on grammatical response patterns.

**UMAP Parameters**:
- Number of neighbors (5-50): Controls local vs. global structure preservation
- Minimum distance (0.0-1.0): Determines cluster separation in embedding
- Distance metrics: Euclidean, Manhattan, Cosine, Correlation
- Standardization: Z-score normalization option

**Visualization Options**:
- Color coding: By variety, variety type, gender
- Point size: Based on response completeness
- Hover information: Participant metadata display

**Clustering Analysis**:
- Leiden algorithm implementation
- Resolution parameter adjustment (0.1-2.0)
- Similarity threshold configuration
- Cluster statistics and validation metrics

**Data Preprocessing**:
- Missing value handling
- Feature selection options
- Dimensionality reduction settings

### Technical Implementation

#### Performance Optimizations
- Lazy data loading with LRU caching
- Persistent disk caching for expensive computations
- Background callback processing
- Incremental data updates

#### Data Caching Strategy
- Grammar data cached at module level
- UMAP plots cached with parameter-specific keys
- RF model results cached with data hash validation
- Automatic cache invalidation on parameter changes

#### Computational Components
- UMAP dimensionality reduction (scikit-learn implementation)
- Random Forest feature importance analysis
- Leiden community detection algorithm
- Statistical testing (t-tests, ANOVA)

## Lexical Sets Module

### Purpose
Examine phonological and lexical variation patterns using word pronunciation and usage data.

### Data Structure
- **Lexical Items**: Words categorized by phonological features
- **Pronunciation Variants**: Different realizations recorded per item
- **Usage Contexts**: Formal vs. informal speech situations
- **Geographic Distribution**: Regional variation patterns

### Analysis Features
- Phoneme distribution analysis
- Vowel system comparisons
- Consonant variation patterns
- Lexical frequency analysis

## Data Overview Module

### Purpose
Provide summary statistics and metadata about the survey dataset.

### Content Areas
- Survey participation rates
- Response completion statistics
- Geographic distribution of participants
- Demographic balance analysis

## User Interface Components

### Navigation
- Tab-based interface within modules
- Breadcrumb navigation between sections
- Responsive design for multiple screen sizes

### Input Controls
- Multi-select dropdowns for participant filtering
- Slider controls for parameter adjustment
- Toggle switches for binary options
- Text input for custom filtering

### Output Displays
- Interactive Plotly visualizations
- AgGrid tables with advanced filtering
- Statistical summary cards
- Export functionality for plots and data

## Data Export Options

### Supported Formats
- PNG/SVG for plot exports
- CSV for tabular data
- JSON for processed analysis results

### Export Scope
- Full datasets or filtered subsets
- Individual plots or composite figures
- Statistical summaries and model outputs

## Performance Considerations

### Optimization Strategies
- Client-side caching of frequently accessed data
- Server-side computation caching
- Progressive data loading for large datasets
- Callback debouncing for real-time updates

### Memory Management
- Automatic cleanup of unused data objects
- Efficient data structure selection
- Lazy loading of heavy computational results

## Technical Requirements

### Dependencies
- Python 3.8+
- Dash 2.0+ with Mantine components
- Plotly for visualization
- Pandas/Polars for data manipulation
- Scikit-learn for statistical analysis
- NetworkX/iGraph for clustering

### Database Backend
- SQLite database storage
- Optimized query performance
- Data integrity constraints
- Backup and recovery procedures

## Troubleshooting

### Common Issues
- Slow loading times: Check network connection and cache status
- Missing data points: Verify participant selection criteria
- Plot rendering errors: Refresh browser cache
- Export failures: Check file permissions and disk space

### Error Messages
- "No data available": Adjust filter criteria
- "Computation timeout": Reduce dataset size or parameter complexity
- "Invalid parameter combination": Review input value ranges
- "Cache error": Clear browser storage and restart session
