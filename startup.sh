#!/bin/bash
set -e

echo "Building Chroma index..."
python scripts/build_chroma_index.py

echo "Building PageIndex..."
python scripts/build_pageindex.py

echo "Starting FastAPI app..."
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
