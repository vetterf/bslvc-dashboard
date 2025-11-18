# Documentation

The BSLVC Dashboard documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## Quick Start

### Install Dependencies

```bash
# Using the virtual environment
source .venv3.13/bin/activate
pip install -r requirements_docs.txt

# Or install directly
pip install mkdocs mkdocs-material mkdocs-glightbox
```

### Serve Documentation Locally

```bash
# Activate venv first
source .venv3.13/bin/activate

# Serve with live reload
mkdocs serve

# Or use the venv directly without activation
.venv3.13/bin/mkdocs serve
```

Visit `http://localhost:8000` to preview the documentation.

### Build Documentation

```bash
# Activate venv first
source .venv3.13/bin/activate

# Build static site
mkdocs build

# Or use the venv directly
.venv3.13/bin/mkdocs build
```

The built site will be in the `site/` directory.

## Deployment

### GitHub Pages (Automatic)

The documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch. See `.github/workflows/docs.yml` for the workflow configuration.

### Manual Deployment

```bash
# Activate venv first
source .venv3.13/bin/activate

# Deploy to GitHub Pages
mkdocs gh-deploy

# Or use the venv directly
.venv3.13/bin/mkdocs gh-deploy
```

## File Structure

```
docs/
├── index.md              # Homepage
├── technical.md          # Technical documentation & UI reference
├── setup.md              # Documentation setup guide
└── stylesheets/
    └── extra.css         # Custom CSS

mkdocs.yml                # MkDocs configuration
requirements_docs.txt     # Documentation dependencies
.github/workflows/docs.yml # Auto-deployment workflow
```

## Configuration

The documentation is configured in `mkdocs.yml`. Key features:

- **Theme**: Material with light/dark mode toggle
- **Navigation**: Tabs for easy access
- **Search**: Built-in search functionality
- **Code Highlighting**: Syntax highlighting for code blocks
- **Admonitions**: Callout boxes for notes, warnings, tips
- **Image Lightbox**: Click to enlarge images

## Writing Documentation

### Markdown Files

All documentation is written in Markdown (`.md` files) in the `docs/` directory.

### Code Blocks

Use triple backticks with language identifier:

````markdown
```python
def example():
    print("Hello, World!")
```
````

### Admonitions

Create callout boxes:

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a helpful tip.
```

### Links

Use relative paths for internal links:

```markdown
[Technical Documentation](technical.md)
```

## Theme Customization

- **Colors**: Edit `mkdocs.yml` under `theme.palette`
- **CSS**: Add custom styles to `docs/stylesheets/extra.css`
- **Logo/Favicon**: Add images and reference in `mkdocs.yml`

## Troubleshooting

### Build Errors

If you get build errors, check:
- All links use `.md` extension (not `.html`)
- No broken internal links
- YAML syntax in `mkdocs.yml` is valid

### Port Already in Use

```bash
mkdocs serve -a localhost:8001
```

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)

---

**Need help?** Open an issue in the GitHub repository.
