# BSLVC Dashboard

[![DOI](https://zenodo.org/badge/1098053419.svg)](https://doi.org/10.5281/zenodo.23916143)
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The BSLVC dashboard is an interactive visualization that allows researchers to explore the Bamberg Survey of Language Variation and Change (BSLVC; https://www.uni-bamberg.de/en/eng-ling/forschung/the-bslvc-project-dfg-funded/) data.

## Quick Start

Visit the [live dashboard](https://bslvc.uni-bamberg.de). The getting started section contains instructions how you can work with the dashboard.

For details on all available functions, see the [documentation](https://vetterf.github.io/bslvc-dashboard/).

## Citation

If you use the BSLVC Dashboard or data from the BSLVC project in your research, please cite as follows:

Dashboard Citation:

Vetter, Fabian. 2026. _BSLVC Dashboard_ (Version 0.2.8). https://doi.org/10.5281/zenodo.23916143

Data Set Citation:

Krug, Manfred & Fabian Vetter. 2026. _The Bamberg Survey of Language Variation and Change_. Zenodo. DOI: 10.5281/zenodo.20157295

## Installation

The BSLVC Dashboard is written in Python and Dash and is designed to run as a docker container. Please consult the documentation of your systems docker/podman installation on how to create and run docker containers.

To create a local desktop-style version of the app, run the build script on your target OS or use the precompiled binaries attached to the most recent release:

- Linux: `bash packaging/build-linux.sh`
- macOS: `bash packaging/build-mac.sh`
- Windows: `packaging\\build-win.bat`

The build script uses PyInstaller for the current platform.

## Cache Management

- URL-based cache clearing is disabled by default (`ENABLE_URL_CACHE_CLEAR=false`).
- For manual cache clearing in development/admin workflows, run:

```bash
flask --app app clear-cache
```

---

*Funded by Deutsche Forschungsgemeinschaft (DFG) - Grant 548274092*
