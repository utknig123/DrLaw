#!/usr/bin/env bash
# exit on error
set -o errexit

# ensure setuptools and wheel are available before anything else
python -m pip install --upgrade pip
python -m pip install setuptools==75.3.0 wheel==0.44.0 build

# install binary packages first
python -m pip install faiss-cpu==1.12.0 torch==2.0.1 torchvision==0.15.2

# install the rest
python -m pip install -r requirements.txt
