#!/bin/bash
# Run Syntropic Swarm v2 (Gradio, 127.0.0.1:7861).
#
# VM quirk (2026-11-07): the environment's NO_PROXY contains bracketed
# IPv6 literals ([::1], [fd8b:...]). httpx parses NO_PROXY at import
# time inside gradio and crashes with `InvalidURL: Invalid port: ':1]'`.
# And gradio's launch() makes an outbound httpx call that dies through
# the egress proxy (RemoteProtocolError). The app needs no proxy at
# serve time (localhost LM Studio + local files + cached model), so
# this script drops all proxy vars for the app process and sets a clean
# no-proxy list for localhost. It does not change the machine's env.
# HF_HUB_OFFLINE=1: the embedding model is already cached locally
# (~/.cache/huggingface); without this, sentence-transformers does a
# hub HEAD request at startup, which fails with no proxy set.
cd "$(dirname "$0")"
exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
     -u ALL_PROXY -u all_proxy \
     NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ./venv/bin/python app.py
