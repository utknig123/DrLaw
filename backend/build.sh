#!/usr/bin/env bash
set -o errexit

# upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

# install all requirements
python -m pip install -r requirements.txt






