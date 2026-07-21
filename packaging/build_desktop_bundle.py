"""Build a platform-specific desktop bundle for the BSLVC dashboard.

The bundle is intended to be portable and inspectable:
- the executable is created with PyInstaller for the current OS
- a copy of the project source is placed in a sibling `source/` directory
- runtime assets and editable data files remain available in the bundle

Run this on each target operating system to produce that OS's build.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DIST_DIR = REPO_ROOT / "dist"
BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BUNDLE_PREFIX = "bslvc-dashboard"
DATABASE_REPO = "https://github.com/vetterf/bslvc-database"
DATABASE_FILENAME = "BSLVC_sqlite.db"
MAPPING_SOURCE_PATHS = [
    REPO_ROOT / "data" / "advanced_regional_mapping.csv",
    REPO_ROOT / "assets" / "data" / "advanced_regional_mapping.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-name",
        default=None,
        help="Override the bundle name. Defaults to a platform-specific name.",
    )
    parser.add_argument(
        "--database-path",
        default=None,
        help="Use an existing SQLite database file instead of cloning it from Git.",
    )
    parser.add_argument(
        "--icon",
        default=None,
        help="Optional application icon. Use .ico on Windows, .icns on macOS, or a platform-appropriate image.",
    )
    parser.add_argument(
        "--keep-build-artifacts",
        action="store_true",
        help="Keep the intermediate PyInstaller build directory.",
    )
    return parser.parse_args()


def platform_label() -> str:
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def default_bundle_name() -> str:
    return f"{DEFAULT_BUNDLE_PREFIX}-{platform_label()}"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def pyinstaller_args(bundle_name: str, icon: str | None) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        bundle_name,
        "desktop_launcher.py",
        # Include pages/ as data so Dash can discover pages on disk at runtime.
        "--add-data", "pages:pages",
        "--collect-submodules",
        "pages",
        "--collect-all",
        "dash",
        "--collect-all",
        "dash_bootstrap_components",
        "--collect-all",
        "dash_bootstrap_templates",
        "--collect-all",
        "dash_iconify",
        "--collect-all",
        "dash_mantine_components",
        "--collect-all",
        "dash_ag_grid",
        "--collect-all",
        "plotly",
        "--hidden-import", "tkinter",
        "--collect-all", "tkinter",
    ]

    if icon:
        args.extend(["--icon", icon])

    return args


def bundle_output_dir(bundle_name: str) -> Path:
    app_bundle = DIST_DIR / f"{bundle_name}.app"
    if app_bundle.exists():
        return app_bundle / "Contents" / "MacOS"

    folder_bundle = DIST_DIR / bundle_name
    if folder_bundle.exists():
        return folder_bundle

    raise FileNotFoundError(f"Could not locate bundle output for {bundle_name} in {DIST_DIR}")


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return

    for path in src.rglob("*"):
        # Skip broken symlinks (e.g. venv Python binaries pointing at unresolvable targets)
        if path.is_symlink() and not path.exists():
            continue

        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def ensure_database_file(runtime_root: Path, source_override: str | None) -> Path:
    data_dir = runtime_root / "assets" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / DATABASE_FILENAME

    if source_override:
        source_path = Path(source_override).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Database file not found: {source_path}")
        shutil.copy2(source_path, destination)
        return destination

    with tempfile.TemporaryDirectory(prefix="bslvc-database-") as temp_repo_dir:
        temp_repo = Path(temp_repo_dir)
        run(["git", "clone", "--depth", "1", DATABASE_REPO, str(temp_repo)])
        candidate = temp_repo / DATABASE_FILENAME
        if not candidate.exists():
            raise FileNotFoundError(f"Database file was not found in {DATABASE_REPO}")
        shutil.copy2(candidate, destination)

    return destination


def copy_mapping_file(runtime_root: Path) -> Path:
    data_dir = runtime_root / "assets" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / "advanced_regional_mapping.csv"

    for candidate in MAPPING_SOURCE_PATHS:
        if candidate.exists():
            shutil.copy2(candidate, destination)
            return destination

    raise FileNotFoundError("advanced_regional_mapping.csv was not found in data/ or assets/data/")


def write_manifest(runtime_root: Path, bundle_name: str) -> None:
    manifest = {
        "bundle_name": bundle_name,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(runtime_root),
    }
    (runtime_root / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bundle_name = args.bundle_name or default_bundle_name()

    run(pyinstaller_args(bundle_name, args.icon))

    runtime_root = bundle_output_dir(bundle_name)
    runtime_root.mkdir(parents=True, exist_ok=True)

    copy_tree(REPO_ROOT / "assets", runtime_root / "assets")
    ensure_database_file(runtime_root, args.database_path)
    copy_mapping_file(runtime_root)
    write_manifest(runtime_root, bundle_name)

    if not args.keep_build_artifacts and BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    print(f"Desktop bundle created at: {runtime_root}")


if __name__ == "__main__":
    main()