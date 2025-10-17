# Getting Started with the BSLVC Dashboard

Welcome to the interactive data exploration interface for the Bamberg Survey of Language Variation and Change (BSLVC). This dashboard provides programmatic-free access to the complete BSLVC corpus, enabling quantitative analysis of morphosyntactic and lexical variation across World Englishes.

> **For comprehensive documentation**, including detailed methodological specifications, API references, and reproducible workflows, visit the [**BSLVC Dashboard Wiki**](https://your-wiki-url-here.com).

---

## Research Capabilities

The BSLVC Dashboard facilitates corpus-based variationist analysis through an integrated computational workflow:

- 📊 **Corpus exploration**: Examine participant demographics, sampling distributions, and metadata completeness
- 📝 **Morphosyntactic analysis**: Investigate grammatical feature distributions across English variety classifications (ENL, ESL, EFL)
- 🗣️ **Lexical analysis**: Explore lexeme variation patterns and regional preference indicators
- 🔍 **Subset generation**: Create filtered datasets based on sociolinguistic variables
- 📈 **Statistical visualization**: Generate publication-quality plots, UMAP dimensionality reductions, and Random Forest feature importance rankings
- 💾 **Data export**: Download filtered datasets in CSV format for external statistical analysis

---

## Recommended Analysis Workflow

### 1. **Corpus Familiarization**
Begin at **Data Overview** to assess:
- Sample composition and distribution across variety types
- Geographic representation and demographic stratification
- Data completeness rates for grammatical and lexical modules
- Temporal sampling patterns

### 2. **Participant Selection**
Navigate to **Grammar Sets** and construct your analytical sample using the hierarchical selection interface:
- Filter by variety classification (ENL/ESL/EFL)
- Apply demographic criteria (age cohorts, gender, educational background)
- Monitor real-time sample statistics in the **Quick Stats** panel
- Ensure minimum sample sizes for statistical power (recommended: n ≥ 10 per group)

### 3. **Feature Selection**
Specify dependent variables for analysis:
- Select individual morphosyntactic features by functional category
- Toggle between single items and item pairs for written/spoken comparison
- Utilize bulk selection tools for comprehensive feature sets
- Reference the grammar information panel for linguistic descriptions

### 4. **Statistical Visualization**
Generate analytical outputs:
- **Grammar Plots**: Distribution visualizations stratified by sociolinguistic variables
- **UMAP Projections**: Unsupervised dimensionality reduction revealing participant clustering patterns
- **Random Forest Analysis**: Supervised feature importance rankings identifying discriminatory linguistic variables
- **Demographic Plots**: Participant characteristic distributions for sample description

---

## Interface Components

### Hierarchical Selection Trees
- **Checkbox interaction**: Select/deselect individual items or entire categories
- **Node expansion**: Access nested subcategories via disclosure triangles
- **Bulk operations**: Clear or select all items efficiently using toolbar buttons

### Sample Statistics Panel
Real-time feedback displays:
- 🟦 Participant count (n)
- 🟩 Feature count (k)
- 🟨 Variety representation

### Visualization Controls
- **Parameter adjustment**: Modify UMAP hyperparameters (neighbors, minimum distance, distance metric)
- **Interactive inspection**: Hover tooltips provide detailed cell-level information
- **Export functionality**: Download plots as high-resolution PNG files for publication

### Advanced Analytical Features
- **Random Forest Classification**: Identify discriminatory features between pre-defined groups with permutation importance
- **UMAP Configuration**: Customize dimensionality reduction through adjustable hyperparameters
- **Multi-criteria Filtering**: Apply complex selection rules across multiple demographic variables

---

## Best Practices for Corpus Analysis

💡 **Sample Size Considerations**: Begin with focused subsets to ensure computational efficiency; expand systematically  
💡 **Statistical Monitoring**: Verify adequate cell counts using Quick Stats before analysis  
💡 **Hyperparameter Exploration**: Test multiple UMAP configurations to assess pattern robustness  
💡 **Targeted Exports**: Apply pre-export filtering to extract analysis-specific datasets  
💡 **Documentation**: Utilize hover tooltips for variable definitions and measurement specifications

---

## Common Research Workflows

### Cross-Variety Comparison
**Objective**: Identify morphosyntactic differences between variety types
1. Select balanced samples from 2-3 variety classifications (e.g., ENL vs. ESL vs. EFL)
2. Choose theoretically-motivated feature set
3. Generate Grammar Plot stratified by "Variety"
4. Apply Random Forest analysis to rank discriminatory features
5. Export filtered dataset for inferential statistical testing

### Age-Grading Analysis
**Objective**: Detect apparent-time variation patterns
1. Select participants from a single variety with age diversity
2. Choose features hypothesized to show generational variation
3. Generate Grammar Plot stratified by "Age"
4. Examine distribution patterns across age cohorts
5. Consider potential confounds (e.g., education, year of data collection)

### Exploratory Pattern Discovery
**Objective**: Uncover latent participant groupings
1. Select comprehensive participant sample (maximize n)
2. Include all available morphosyntactic features (maximize k)
3. Generate UMAP Projection colored by "Variety" or other demographic variable
4. Identify clustering patterns and outliers
5. Formulate hypotheses for confirmatory analysis

---

## Technical Requirements

- **Browser**: Chrome 90+, Firefox 88+, Safari 14+, or Edge 90+ (latest stable recommended)
- **JavaScript**: Must be enabled for interactive functionality
- **Display**: Minimum 1280×720 resolution; 1920×1080 recommended for optimal visualization
- **Network**: Stable connection required for initial data loading (approximate payload: 5-15 MB)
- **Computational**: Modern processor recommended for UMAP computation (intensive for n > 500)

---

## Support Resources

📚 **[Comprehensive Documentation Wiki](https://your-wiki-url-here.com)** – Methodological specifications and analytical protocols  
📖 **Data Dictionary** – Variable definitions, coding schemes, and measurement procedures  
📊 **Codebook** – Complete feature inventory with linguistic descriptions  
🎓 **Tutorial Videos** – Guided workflows for common research questions  
❓ **Frequently Asked Questions** – Troubleshooting and interpretation guidance  
✉️ **Research Support** – Contact for methodological consultation

---

**Ready to begin?** Navigate to **Grammar Sets** or **Lexical Sets** to initiate your analysis.

