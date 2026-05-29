#!/bin/bash
set -e

echo "Building Chroma index..."
python build_chroma_index.py

echo "Building PageIndex..."
python build_pageindex.py
