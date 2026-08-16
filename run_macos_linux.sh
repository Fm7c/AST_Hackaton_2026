#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CREATED=0

if [ ! -d ".venv" ]; then
  echo "[AST] Primeira execução: a criar ambiente virtual .venv..."
  "$PYTHON_BIN" -m venv .venv
  CREATED=1
fi

source .venv/bin/activate

# Installing scientific dependencies is intentionally done only on the first
# execution of this project folder. Re-checking scipy/pandas/streamlit on every
# launch made startup much slower. Delete .venv if you ever need a clean rebuild.
if [ "$CREATED" -eq 1 ] || ! python -c "import streamlit,pandas,numpy,scipy,plotly,openpyxl,xlrd,matplotlib,sqlalchemy,psycopg" >/dev/null 2>&1; then
  echo "[AST] A instalar/atualizar dependências necessárias..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  touch .venv/.ast_dependencies_ready
else
  echo "[AST] Ambiente já preparado — a saltar instalação de dependências."
fi

echo "[AST] A abrir a plataforma..."
python -m streamlit run app.py
