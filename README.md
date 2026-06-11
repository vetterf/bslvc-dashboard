# BSLVC Dashboard

The BSLVC dashboard is an interactive visualization that allows researchers to explore the Bamberg Survey of Language Variation and Change (BSLVC; https://www.uni-bamberg.de/en/eng-ling/forschung/the-bslvc-project-dfg-funded/) data.

## Quick Start

Visit the [live dashboard](https://bslvc.uni-bamberg.de). The getting started section contains instructions how you can work with the dashboard.

For details on all available functions, see the [documentation](https://vetterf.github.io/bslvc-dashboard/).

## Export Notes

- In **Participant Similarity** mode, UMAP-related exports are available in **Advanced Actions → Data Export**.
- The export is provided as a ZIP archive containing:
  - a CSV with coordinate data and point-level visibility flags (including hidden points)
  - a log file with export timestamp, UMAP settings, and selection metadata

## Citation

If you use the BSLVC Dashboard or data from the BSLVC project in your research, please cite as follows:

Dashboard Citation:

Vetter, Fabian. 2026. _BSLVC Dashboard_ (Version 0.2.1). https://doi.org/10.17605/OSF.IO/4BUEF

Data Set Citation:

Krug, Manfred & Fabian Vetter. 2026. _The Bamberg Survey of Language Variation and Change_. Zenodo. DOI: 10.5281/zenodo.20157295

## Installation

The BSLVC Dashboard is written in Python and Dash and is designed to run as a docker container. Please consult the documentation of your systems docker/podman installation on how to create and run docker containers.

## Cache Management

- URL-based cache clearing is disabled by default (`ENABLE_URL_CACHE_CLEAR=false`).
- For manual cache clearing in development/admin workflows, run:

```bash
flask --app app clear-cache
```

---

*Funded by Deutsche Forschungsgemeinschaft (DFG) - Grant 548274092*
