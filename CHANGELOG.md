# Changelog

All notable changes to the BSLVC Dashboard will be documented in this file.


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
