#!/usr/bin/env bash
# Python 3.12 仮想環境から Streamlit を起動します。
cd "$(dirname "$0")"
if [[ ! -x .venv312/bin/streamlit ]]; then
  echo "先に: python3.12 -m venv .venv312 && .venv312/bin/pip install -r requirements.txt" >&2
  exit 1
fi
exec .venv312/bin/streamlit run us_states_rag_app.py "$@"
