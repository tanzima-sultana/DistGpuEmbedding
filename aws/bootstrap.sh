#!/bin/bash
set -e
echo "Starting bootstrap..."

sudo pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

sudo pip3 install --no-cache-dir \
    sentence-transformers==2.7.0 \
    transformers==4.40.0 \
    faiss-cpu \
    tqdm \
    pyyaml

echo "Bootstrap complete"