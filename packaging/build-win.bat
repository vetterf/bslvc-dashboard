@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"

set "VENV_DIR=packaging\.venv_build"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    py -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b %errorlevel%
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b %errorlevel%

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r packaging\requirements_desktop.txt
if errorlevel 1 exit /b %errorlevel%

python packaging\build_desktop_bundle.py %*
exit /b %errorlevel%
