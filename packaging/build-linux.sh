#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

venv_dir="packaging/.venv_build"

if [ ! -d "$venv_dir" ]; then
    python3 -m venv "$venv_dir"
fi

# shellcheck source=/dev/null
source "$venv_dir/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r packaging/requirements_desktop.txt

python packaging/build_desktop_bundle.py "$@"
