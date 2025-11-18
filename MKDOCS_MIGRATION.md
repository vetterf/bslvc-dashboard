# MkDocs Migration Summary

## What Changed

Successfully migrated the BSLVC Dashboard documentation from Jekyll to MkDocs.

### Removed Files
- `docs/_config.yml` (Jekyll configuration)

### Added Files
- `mkdocs.yml` - MkDocs configuration at project root
- `requirements_docs.txt` - Documentation dependencies
- `docs/setup.md` - Comprehensive setup guide for MkDocs
- `docs/README_DOCS.md` - Quick reference for documentation development
- `docs/stylesheets/extra.css` - Custom CSS
- `.github/workflows/docs.yml` - GitHub Actions for auto-deployment
- Updated `.gitignore` to exclude `site/` build directory

### Modified Files
- `docs/index.md` - Fixed broken links (`.html` → `.md`)
- `docs/README.md` - Fixed broken links
- `docs/technical.md` - Removed broken anchor reference

## Key Features

### MkDocs with Material Theme
- Modern, responsive design
- Light/dark mode toggle
- Built-in search
- Code syntax highlighting
- Admonitions (note, warning, tip boxes)
- Image lightbox
- Navigation tabs

### Development Workflow
```bash
# Serve locally with live reload
.venv3.13/bin/mkdocs serve

# Build static site
.venv3.13/bin/mkdocs build

# Deploy to GitHub Pages
.venv3.13/bin/mkdocs gh-deploy
```

### Automatic Deployment
GitHub Actions automatically deploys documentation to GitHub Pages when changes are pushed to `main` branch.

## Quick Commands

```bash
# Install dependencies
pip install -r requirements_docs.txt

# Serve locally (with venv)
source .venv3.13/bin/activate
mkdocs serve

# Serve locally (without activation)
.venv3.13/bin/mkdocs serve

# Build
.venv3.13/bin/mkdocs build

# Deploy to GitHub Pages
.venv3.13/bin/mkdocs gh-deploy
```

## Documentation Structure

```
docs/
├── index.md              # Homepage
├── technical.md          # Technical docs & UI reference
├── setup.md              # MkDocs setup guide
├── README.md             # About & installation (excluded from nav due to conflict with index.md)
├── README_DOCS.md        # Development guide (not in nav)
└── stylesheets/
    └── extra.css         # Custom CSS
```

## Benefits Over Jekyll

1. **Simpler**: Pure Python, no Ruby dependencies
2. **Faster**: Quick builds and instant live reload
3. **Better Theme**: Material theme is modern and feature-rich
4. **Integrated**: Works seamlessly with Python projects
5. **Search**: Built-in search out of the box
6. **Markdown Extensions**: Rich formatting options (admonitions, code tabs, etc.)

## Next Steps

1. Customize `mkdocs.yml` with your GitHub repository URL
2. Test local build: `.venv3.13/bin/mkdocs serve`
3. Push to GitHub - docs will auto-deploy via GitHub Actions
4. Access at: `https://yourusername.github.io/bslvc-dashboard/`

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material Theme Docs](https://squidfunk.github.io/mkdocs-material/)
- See `docs/README_DOCS.md` for detailed development guide
