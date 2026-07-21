#!/bin/bash
set -e
echo "Starting bootstrap..."

# Git install
sudo yum install -y git

# Install dataset (--ignore-installed filelock avoids rpm-vs-pip uninstall conflict on this AMI)
sudo pip3 install --no-cache-dir --ignore-installed filelock datasets

# Install s3fs
sudo pip3 install --no-cache-dir s3fs

# --- Install torch (CUDA 12.6 build, verified against driver 560.35.03 / CUDA 12.6 on this AMI) ---
sudo pip3 install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# --- faiss-gpu requires numpy<2 (confirmed empirically: faiss 1.7.2 swig_ptr rejects numpy 2.x arrays) ---
sudo pip3 install --no-cache-dir "numpy<2"

# --- faiss-gpu only. Do NOT install faiss-cpu alongside it — they share the same
#     'faiss' module namespace and silently overwrite each other, breaking GPU support. ---
sudo pip3 install --no-cache-dir faiss-gpu

# --- Embedding stack ---
sudo pip3 install --no-cache-dir sentence-transformers transformers

# --- Utilities ---
sudo pip3 install --no-cache-dir tqdm pyyaml

# --- Persist LD_LIBRARY_PATH for nvidia .so files (cusparselt, cudnn, nccl, etc.)
#     Required because torch's pip-bundled nvidia-*-cu12 packages don't register
#     themselves on the system linker path by default. Written to /etc/environment
#     (not .bashrc) so it applies to non-interactive Spark executor processes too.
#     Filters out non-package entries (__pycache__, __init__.py) that a naive
#     os.listdir() would otherwise include as bogus paths. ---
NVIDIA_LIB_PATHS=$(python3 -c "
import nvidia, os
base = os.path.dirname(nvidia.__file__)
dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and not d.startswith('__')]
print(':'.join(os.path.join(base, d, 'lib') for d in dirs))
")
echo "LD_LIBRARY_PATH=${NVIDIA_LIB_PATHS}" | sudo tee /etc/environment > /dev/null

echo "Bootstrap complete"