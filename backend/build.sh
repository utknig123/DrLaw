#!/usr/bin/env bash
# exit on error
set -o errexit

# upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

# install binary packages first to ensure compatible versions
python -m pip install faiss-cpu==1.12.0 torch==2.0.1 torchvision==0.15.2

# install remaining requirements
python -m pip install -r requirements.txt