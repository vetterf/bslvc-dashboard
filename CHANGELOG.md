# Changelog

All notable changes to the BSLVC Dashboard will be documented in this file.

## [0.1.4] - 2026-01-13

### Added
- **Export Distance Matrix Button**: New button in Advanced Actions (visible only in Participant Similarity mode)
  - Exports pairwise distance matrix as CSV using the same settings as the UMAP plot
  - Uses selected distance metric (cosine/euclidean/manhattan), standardization, participants, and items
  - Participant IDs used as both row and column labels
- **Download Participant Metadata Button**: New button in Advanced Actions (visible in all analysis modes)
  - Downloads sociodemographic details for selected participants from database
  - Respects imputed/non-imputed data switch setting
  - Automatically removes sensitive columns (NameSchool, signature, CommentsTimeline)
- **Flexible Data Export Options**: Added two checkboxes in Data Export section for customizing export format
  - **Include sociodemographic data** (default: checked): Controls whether participant metadata columns are included
  - **Include item metadata** (default: unchecked): Controls whether to transpose data and add item metadata columns
  - Four export formats available:
    - Neither checked: Minimal format (InformantID + ratings only)
    - Only sociodemographic: Standard wide format (participants as rows, with metadata)
    - Only item metadata: Transposed format (items as rows, participants as columns, with item metadata)
    - Both checked: Transposed format with participant metadata as header rows
  - Filenames automatically indicate format type (minimal/wide/transposed)
- ** Items Table Download**: Added a download button for the grammar items

### Changed
- **Advanced Actions Restructuring**: Reorganized into two clear sections
  - **Data Export** section: All export/download buttons grouped together
  - **Settings** section: Copy/Paste and Save/Restore Settings grouped together
- **DB**: Updated database with new metadata for features.
- **UI Tweaks**: Minor ui tweaks

## [0.1.3] - 2025-11-28

### Changed
- **Regional mapping**: Added a switch in the interface instead of URL parameter
- **DB Version**: Now visible in footer
- **DB**: Added DB Version, added data for Balearic Islands, switched imputation from missForest to hdImpute
- **Citation**: Updated citation info

### Fixed
- **Regional mapping**: Switch was not working in production as paths for mapping CSV file were off in Docker file
- **Spain (Balearic Islands)**: Data point was not drawn on overview plot.

## [0.1.2] - 2025-11-18

### Added
- **About Page**: New dedicated About page with project information, citation information, technical details, and contact information
- **Auto-activate Plot View Tab**: Plot View tab now automatically activates when UMAP, Item, or RF plots finish rendering
- **Grammar item selection via Grammar Items table**: Users can now select grammar items in the grammar items table.
- **Regional mapping URL parameter**: URL parameter ?regional_mapping=true now activates the regional mapping of English participants to England_North and England_South
- **Added Documentation**

### Changed
- **License**: Changed from CC BY-SA 4.0 to MIT License for the dashboard software
  - Added MIT LICENSE file to repository
  - Updated footer to display MIT License with link
- **Grammar Items Preset Filtering**: 
  - Preset dropdown in table filter remains enabled when Item Difference mode is active
  - Mode-specific presets (Top 15, Mode: Spoken, Mode: Written) are now filtered out when Item Difference is enabled
- **Grammar Items Table Column Order**: Rearranged columns to: Item Code, Group, eWAVE, Item (more intuitive ordering)
- **Preset Dropdown Position**: Changed to open upward (`comboboxProps={"position": "top"}`) to prevent dropdown from being cut off
- **Getting Started Page**: Added a getting started section with a description of the grammar analysis module with three items (Basic Workflow, Case Study 1, Case Study 2)
- **Jump to Plot upon Finished Rendering**: The interface now automatically jumps to the plot tab, once rendering is finished.

### Removed
- **Superfluous Files**: Cleaned up unused project files:


## [0.1.1] - 2025-11-06

### Added
- **Version Number Display**: Added version number (v0.1.1) to the footer
- **Grammatical Items Table Enhancements**:
  - Added MultiSelect filter for presets (replacing item code filter)
  - Filter table by grammatical presets (e.g., "Top 15 spoken", "Mode: X", "Group: Y", "eWAVE: Z")
  - Added row selection functionality to the grammatical items table
  - "Select rows in tree" button to add selected table rows to the item selection tree
  - "Deselect rows in tree" button to remove selected table rows from the item selection tree
  - "Show only selected items" button to filter table based on tree selection
  - "Clear filters" button now also clears the quick search field
  - Helper text above grammar items tree directing users to the Grammatical Items tab
- **Tree Organization Improvements**:
  - Grammar items tree now organized by grammatical groups instead of section letters (A, B, C, etc.)
  - Maintained top-level distinction between "Spoken section" and "Written section"
  - Participants tree now sorted alphabetically by variety/country

### Changed
- **Grammar Items Table**:
  - Updated to show twin items when in pairs mode (item differences)
  - Items without twins are filtered out in pairs mode
  - Combined item codes displayed in format "A1 - G21" for pairs
  - Table filtering now uses rowData updates instead of AG Grid filterModel (more reliable)
- **Item Plots**:
  - Fixed x-axis range to -5 to 5 for item differences in "Mean (95% CI - split varieties)" mode
  - Previously used dynamic range which was inconsistent with other plot modes

### Fixed
- **AG Grid Compatibility**:
  - Filtering now works reliably

### Technical Details
- Modified `getMetaTable()` function to accept `preset_data` parameter
- Updated `drawGrammarItemsTree()` to organize by `group_finegrained` column
- Updated `drawParticipantsTree()` to sort varieties alphabetically
- Added new callbacks for preset-based filtering and table-tree integration
- Improved callback efficiency by using `prevent_initial_call='initial_duplicate'`

---

## [Previous Versions]

### [0.1.0] - 2024
- Initial release of BSLVC Dashboard
- UMAP visualization for participant similarity
- Grammar item analysis and plotting
- Interactive participant and item selection trees
- Preset management system
- Data export functionality
