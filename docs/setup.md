# Documentation Setup Guide

This guide explains how to work with the MkDocs documentation for the BSLVC Dashboard.

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation

Install MkDocs and required plugins:

```bash
pip install mkdocs mkdocs-material mkdocs-glightbox
```

## Local Development

### Serve Documentation Locally

To preview the documentation with live reload:

```bash
mkdocs serve
```

Then visit `http://localhost:8000` in your browser. The documentation will automatically reload when you make changes to the source files.

### Custom Port

To use a different port:

```bash
mkdocs serve -a localhost:8080
```

## Building the Documentation

### Build Static Site

Generate the static HTML site:

```bash
mkdocs build
```

The output will be in the `site/` directory. This directory is gitignored and should not be committed.

### Clean Build

Remove the `site/` directory and rebuild:

```bash
mkdocs build --clean
```

## Deployment

### Deploy to GitHub Pages

MkDocs can automatically deploy to GitHub Pages:

```bash
mkdocs gh-deploy
```

This command will:
1. Build the documentation
2. Create/update the `gh-pages` branch
3. Push to GitHub

Your documentation will be available at `https://yourusername.github.io/bslvc-dashboard/`

### GitHub Actions (Recommended)

For automatic deployment on every push, create `.github/workflows/docs.yml`:

```yaml
name: docs
on:
  push:
    branches:
      - main
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: 3.x
      - run: pip install mkdocs-material mkdocs-glightbox
      - run: mkdocs gh-deploy --force
```

## Configuration

The documentation is configured in `mkdocs.yml` at the project root. Key settings:

- **site_name**: Title of the documentation
- **theme**: Material theme with custom colors
- **nav**: Navigation structure
- **markdown_extensions**: Enabled Markdown features
- **plugins**: Search and image lightbox

## File Structure

```
docs/
├── index.md              # Homepage
├── technical.md          # Technical documentation
├── README.md             # About/Installation
└── stylesheets/
    └── extra.css         # Custom CSS

mkdocs.yml                # MkDocs configuration
```

## Writing Documentation

### Basic Markdown

Use standard Markdown syntax. MkDocs supports:

- Headers: `# H1`, `## H2`, etc.
- Lists: `- item` or `1. item`
- Links: `[text](url)`
- Code blocks: ` ```language `
- Images: `![alt](path)`

### Admonitions

Create callout boxes:

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.
```

### Code Blocks with Syntax Highlighting

```markdown
```python
def hello():
    print("Hello, World!")
```
```

### Tabs

```markdown
=== "Tab 1"
    Content for tab 1

=== "Tab 2"
    Content for tab 2
```

### Tables

```markdown
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
```

## Customization

### Theme Colors

Edit `mkdocs.yml`:

```yaml
theme:
  palette:
    primary: indigo
    accent: blue
```

### Custom CSS

Add styles to `docs/stylesheets/extra.css`.

### Navigation

Update the `nav` section in `mkdocs.yml`:

```yaml
nav:
  - Home: index.md
  - Guide: guide.md
  - Reference: reference.md
```

## Troubleshooting

### Port Already in Use

```bash
mkdocs serve -a localhost:8001
```

### Build Errors

Check `mkdocs.yml` for syntax errors. Ensure all referenced files exist.

### Broken Links

Use relative paths: `[Link](page.md)` not `[Link](page.html)`

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material Theme](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)

---

**Need help?** Open an issue in the GitHub repository.
