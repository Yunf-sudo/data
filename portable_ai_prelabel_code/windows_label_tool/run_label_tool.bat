@echo off
setlocal
cd /d "%~dp0"

set "PY_CMD="

where py >nul 2>nul
if not errorlevel 1 set "PY_CMD=py"

if not defined PY_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo Python was not found.
    echo Please install Python 3 first and enable "Add Python to PATH".
    pause
    exit /b 1
)

if not exist ".label_tool_venv\Scripts\python.exe" (
    echo Creating local virtual environment...
    %PY_CMD% -m venv .label_tool_venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call ".label_tool_venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Installing required package...
python -m pip install --disable-pip-version-check -q -r requirements-label-tool.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    echo Please check your network connection and try again.
    pause
    exit /b 1
)

echo Starting label tool...
python simple_label_tool.py

if errorlevel 1 (
    echo The tool exited with an error.
    pause
    exit /b 1
)
