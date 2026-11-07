#!/usr/bin/env bash
# Run a command in this app's venv with the proxy workaround this VM needs.
# The VM's default NO_PROXY contains bracketed IPv6 entries ([::1],
# [fd8b:...]) that httpx cannot parse, which breaks `import gradio`.
# Replacing NO_PROXY with plain localhost entries fixes the import while
# keeping the egress proxy (HTTP_PROXY etc.) for Hugging Face downloads.
# Usage: ./run.sh python test_headless.py
#        ./run.sh python app.py
set -e
cd "$(dirname "$0")"
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
# The egress proxy drops Hugging Face's XET download endpoint; force plain
# HTTP downloads for model files (HF_HUB_DISABLE_XET is a supported
# huggingface_hub flag). Without this, SentenceTransformer model downloads
# fail with ProxyError/RemoteDisconnected.
export HF_HUB_DISABLE_XET=1
export TMPDIR="$HOME/workspace/.pip-tmp"
exec ./venv/bin/"$@"
