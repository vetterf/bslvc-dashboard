# Technical Documentation

This document provides technical details about the BSLVC Dashboard and all available functions.

## Table of Contents

1. [Key Components](#key-components)

---


## Key components

The heart of the dashboard are the analysis modules for the grammar and the lexical data. Currently, only the grammar analysis module is finished. In what follows, you will find a description of all available options in the Dashboard.


### Grammar Sets: Interface Elements & Buttons

The Grammar Sets module provides a rich interactive interface. Below is a description of all major buttons and UI elements:

#### Main Tabs
- **Plot View**: Shows either the UMAP participant similarity plot or item ratings plot.
- **Sociodemographic Details**: Displays participant information and summary plots.
- **Grammar Items**: Shows a table of all grammar items and metadata.

#### Analysis Type Selector
- **SegmentedControl**: Switch between "Participant Similarity" (UMAP) and "Item Ratings" plots.

#### Primary Action
- **Render Plot**: Main button to generate the selected plot (UMAP or item ratings).

#### UMAP Group Buttons (visible in UMAP mode)
- **Add Group**: Add a group of participants for comparison.
- **Clear Groups**: Remove all defined groups.
- **Select Only Lasso Selection**: Select only participants currently selected via lasso tool.
- **Deselect Lasso Selection**: Deselect participants selected via lasso tool.
- **Compare Selected Groups**: Render a Random Forest plot comparing selected groups.

#### Participant Selection
- **Select All**: Select all participants in the tree.
- **Deselect All**: Deselect all participants.
- **Quick Selection**: Batch select by variety, age, gender, completeness, or balanced groups.
- **Participants Tree**: Hierarchical tree for manual selection.
- **Advanced Filters**: Filter participants by gender, age, variety ratio, and main variety.
- **Apply Filters**: Apply selected participant filters.

#### Grammar Item Selection
- **Select All**: Select all grammar items.
- **Deselect All**: Deselect all grammar items.
- **Problematic**: Deselect problematic items.
- **Grammar Items Tree**: Hierarchical tree for manual item selection.
- **Select a Preset**: Multi-select dropdown for item presets (manual, mode, group, eWAVE feature groups).
- **Advanced Options**:
    - **Use item difference (spoken-written)**: Toggle to use difference between item pairs.
    - **Use imputed data**: Toggle between imputed and raw data (always imputed for UMAP).
    - **Toggle Written-Only**: Select only written items.
    - **Currency/Unit**: Toggle currency/unit items.

#### UMAP Settings
- **Color**: Select coloring variable (Variety, Variety type, Gender).
- **Distance metric**: Choose metric (Cosine, Euclidean, Manhattan).
- **Standardize participant ratings**: Checkbox to standardize ratings.
- **Use density-preserving embedding (DensMAP)**: Checkbox for DensMAP.
- **Number of neighbours**: Slider for UMAP hyperparameter.
- **Minimal distance**: Slider for UMAP hyperparameter.

#### Item Plot Settings
- **Plot mode**: Select plot type (mean, split by variety, diverging bars, boxplot, correlation matrix, missing values heatmap).
- **Group by**: Select grouping variable (Variety, Variety type, Gender).
- **Sort by**: Sort items by mean, SD, or alphabetically.

#### Group Comparison Settings
- **Filter by Average Rating**: Range slider to filter items by group average.
- **Use Z-Scores**: Checkbox to standardize ratings before training.

#### Advanced Actions
- **Export Data**: Download current data selection.
- **Copy Settings**: Copy current settings to clipboard.
- **Paste Settings**: Paste settings from clipboard.
- **Save Settings**: Save current settings locally.
- **Restore Settings**: Restore previously saved settings.

#### Leiden Clustering (if enabled)
- **Run Leiden Clustering**: Button to run clustering.
- **Leiden Clustering Settings**: Sliders and dropdowns for resolution, similarity threshold, coloring, PCA options.

#### Miscellaneous
- **Badges**: Show count of selected participants/items.
- **Loading Overlays**: Indicate when plots are rendering.
- **Notifications**: Show info or error messages.

Refer to the User Guide for screenshots and step-by-step usage of each element.

**Getting Started** (`pages/getting_started.py`):
- Tutorial content
- Interactive steppers
- Case studies
- UI location helpers

**About Page** (`pages/about.py`):
- Project information
- Citations
- Technical details

### 4. Caching System

```python
from functools import lru_cache

@lru_cache(maxsize=10)
def get_cached_umap_plot(participants, items, n_neighbours,
                         min_dist, **kwargs):
    """Cache UMAP plots for reuse"""
    
@lru_cache(maxsize=50)
def get_cached_rf_plot(plotDF, importanceRatings,
                       value_range, **kwargs):
    """Cache Random Forest plots"""
```

**Cache Strategy:**
- UMAP plots: 10 most recent (expensive computation)
- RF plots: 50 most recent (moderate computation)
- Session storage: UI state, last rendered plots

---

**Last Updated**: November 2025  
**Version**: 0.1.2
