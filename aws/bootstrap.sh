#!/bin/bash
set -e
echo "Starting bootstrap..."

# --- Install torch (CUDA 12.8 build, verified against driver 560.35.03 / CUDA 12.6) ---
sudo pip3 install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# --- faiss-gpu requires numpy<2 (ABI incompatibility with numpy 2.x, verified empirically) ---
sudo pip3 install --no-cache-dir "numpy<2"

# --- faiss-gpu only. Do NOT install faiss-cpu alongside it — they share the same
#     'faiss' module namespace and silently overwrite each other, breaking GPU support. ---
sudo pip3 install --no-cache-dir faiss-gpu

# --- Embedding stack ---
sudo pip3 install --no-cache-dir sentence-transformers transformers

# --- Utilities ---
sudo pip3 install --no-cache-dir tqdm pyyaml

# --- Persist LD_LIBRARY_PATH for nvidia .so files (cupti, nccl, cudnn, etc.)
#     Required because torch's pip-bundled nvidia-*-cu12 packages don't register
#     themselves on the system linker path by default. Written to /etc/environment
#     (not .bashrc) so it applies to non-interactive Spark executor processes too. ---
NVIDIA_LIB_PATHS=$(python3 -c "import nvidia, os; print(':'.join(os.path.join(os.path.dirname(nvidia.__file__), d, 'lib') for d in os.listdir(os.path.dirname(nvidia.__file__))))")
echo "LD_LIBRARY_PATH=${NVIDIA_LIB_PATHS}" | sudo tee -a /etc/environment

echo "Bootstrap complete"