@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Prefer a known-compatible Python even when another version (for example 3.14)
rem is the system default.
rem NOTE: we compare the interpreter's own printed version string instead of
rem trusting "py -X.Y" errorlevel alone -- some py launcher installs return
rem exit code 0 even when the requested X.Y version does not exist.
set "PYTHON_CMD="
set "PY_VER="
for /f "delims=" %%v in ('py -3.12 -c "import sys; print(sys.version_info[:2])" 2^>nul') do set "PY_VER=%%v"
if "%PY_VER%"=="(3, 12)" set "PYTHON_CMD=py -3.12"

if not defined PYTHON_CMD (
  set "PY_VER="
  for /f "delims=" %%v in ('py -3.11 -c "import sys; print(sys.version_info[:2])" 2^>nul') do set "PY_VER=%%v"
  if "%PY_VER%"=="(3, 11)" set "PYTHON_CMD=py -3.11"
)

if not defined PYTHON_CMD (
  set "PY_VER="
  for /f "delims=" %%v in ('python -c "import sys; print(sys.version_info[:2])" 2^>nul') do set "PY_VER=%%v"
  if "%PY_VER%"=="(3, 11)" set "PYTHON_CMD=python"
  if "%PY_VER%"=="(3, 12)" set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
  echo.
  echo Python 3.11 or 3.12 was not found.
  echo You can keep other Python versions installed; this project will use 3.12 when available.
  echo Install Python 3.12 64-bit, then run this file again.
  echo.
  pause
  exit /b 1
)

echo Using:
%PYTHON_CMD% --version

rem If an old local environment was created with an unsupported interpreter,
rem rebuild only that local .venv. No system Python installation is removed.
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo Rebuilding the local environment with a compatible Python...
    rmdir /s /q .venv
  )
)

set "CREATED=0"
if not exist .venv (
  echo Creating local Python environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 pause & exit /b 1
  if not exist .venv\Scripts\python.exe (
    echo.
    echo Failed to create the local Python environment ^(.venv\Scripts\python.exe not found^).
    echo.
    pause
    exit /b 1
  )
  .venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)"
  if errorlevel 1 (
    echo.
    echo The local Python environment was created with an unsupported Python version.
    echo Delete the .venv folder and run this file again with Python 3.11 or 3.12 installed.
    echo.
    pause
    exit /b 1
  )
  set "CREATED=1"
)

call .venv\Scripts\activate.bat

rem Reuse an existing compatible environment when the required packages are already there.
if "%CREATED%"=="1" goto INSTALL_DEPS
python -c "import streamlit,pandas,numpy,scipy,plotly,openpyxl,xlrd,matplotlib,sqlalchemy,psycopg" >nul 2>nul
if errorlevel 1 goto INSTALL_DEPS
goto START_APP

:INSTALL_DEPS
echo Installing dependencies. This is only required once for this release...
python -m pip install --upgrade pip
if errorlevel 1 pause & exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 pause & exit /b 1

:START_APP
echo Starting AST Sensor Analytics...
python -m streamlit run app.py
if errorlevel 1 pause
endlocal
