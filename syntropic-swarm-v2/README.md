# Syntropic Swarm v2 — README

A working rebuild of Putt's **unified** "Architects of Reality: The
Syntropic Swarm (Master QCE)" — all 8 integrations. The canonical original
is saved untouched at `../SYNTROPIC_SWARM_UNIFIED_original.py` (19,162
bytes); the audit is `../SWARM_AUDIT.md`. The unified version superseded
the shorter 3-integration version (`../SYNTROPIC_SWARM_original.py`).

## Run it

```bash
./run.sh        # serves on http://127.0.0.1:7861
```

Use `run.sh`, not `python app.py` directly: the VM's NO_PROXY
contains bracketed IPv6 literals that httpx cannot parse at gradio
import time, and gradio's launch() makes an outbound call that dies
through the egress proxy. `run.sh` drops proxy vars for the app
process, sets a clean NO_PROXY for localhost, and sets
HF_HUB_OFFLINE=1 (the embedding model is cached locally).

First run builds the FAISS index from Putt's papers
(`build_index.py`, ~96 chunks) — a few minutes on CPU. The venv is
already built in `venv/` (pinned versions in `requirements.txt`).

## What v2 actually does

- **Chat (QCE)** — RAG over Putt's own papers (Third Revised Edition +
  UAF paper), via the reused companion-v2 engine: LM Studio first,
  extractive demo fallback when no LLM is running (labelled in the UI).
  The conversation is rendered as a scrollable HTML log, not
  gr.Chatbot — the Chatbot component crashes Gradio 5.4.0 /
  gradio-client 1.4.2 on the root page (its API schema contains a
  boolean `"additionalProperties": true`; gradio-client 1.4.3 and
  1.5.0 do not fix it). History is still a messages list in a
  gr.State; only the rendering changed.
- **Quantum collapse** — the original's 3-qubit circuit (H, CX, CX,
  RZ(phase·π)) run on a **local Aer simulator by default** (free, instant,
  1024 shots), with real measured counts and Shannon entropy. IBM Runtime
  only with saved credentials AND `AOR_ENABLE_IBM=1`; OFF disables it.
  No mock is ever presented as an IBM run.
- **VQE** — Tier 3 messages matching `V(Psi) = ...` SELECT between two
  named Hamiltonians by keyword ("cos" → cos-type ZZ + 0.5·IX − 0.15·ZI;
  otherwise generic ZZ + XX). The selection rule is stated in the UI.
  The VQE itself is real: Aer Estimator + TwoLocal + COBYLA on the
  simulator, reporting a genuine ground-state energy. The expression is
  NOT compiled into Pauli terms — a real term-by-term compiler is future
  work.
- **3D manifold** — Plotly PCA scatter of REAL embeddings: static corpus
  chunks re-embedded, reconstructed swarm vectors, live query. The
  original plotted random dummy vectors for the static layer.
- **EEG** — BrainFlow synthetic board used explicitly as a labelled demo
  (or numpy-simulated bands if BrainFlow is not installed); band
  indexing verified live (BrainFlow 5.16 returns 1-D per-band
  averages — the original's indexing was correct, the first audit's
  claim otherwise was wrong and is corrected in the audit);
  theta/alpha ratio labelled a relaxation proxy, not entropy; stop
  button included. Real Muse 2 path documented, needs a paired
  headset.
- **Biometrics (HRV)** — RMSSD over a rolling 30-beat RR window, mapped
  to a documented heuristic score (labelled heuristic, not entropy).
  Simulated-HRV demo mode (default) or the real WebBluetooth path.
- **Swarm ledger** — persisted to disk (`swarm_ledger.index` +
  `swarm_ledger.json`), **opt-in, default OFF, local only**.
- **Telemetry** — measured only: top L2 distance, chunks, latency, words,
  RMSSD, bio score, temperature, quantum backend/mode, VQE ground state,
  ledger size.
- **Two tabs** (Quantum Interface + Cybernetics & Billing), as in the
  unified original. **Binds 127.0.0.1 only** (the original bound
  0.0.0.0:7860, exposing the app to the network with no auth).

## Run

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt   # large: sentence-transformers pulls torch (~2 GB)
venv/bin/python test_headless.py           # headless test, exit 0 = all pass
venv/bin/python app.py                     # serves on http://127.0.0.1:7861
```

First start builds the FAISS index from the PDFs (downloads the
`all-mpnet-base-v2` embedding model once, ~420 MB). LM Studio is optional:
if it is not running on `http://localhost:1234/v1`, the app answers in
labelled demo mode.

## Env vars

- `AOR_QUANTUM_MODE=simulator|ibm|off` (default `simulator`)
- `AOR_ENABLE_IBM=1` — required, with saved IBM Quantum credentials, to
  submit a real-hardware job. Never set this unless you intend to spend
  IBM Quantum quota and wait in a hardware queue.
- `AOR_LLM_URL`, `AOR_LLM_MODEL` — LM Studio endpoint (defaults
  `http://localhost:1234/v1`, `local-model`).

## Honest limits

- Simulator by default. **No real-hardware job is ever submitted without
  saved credentials + `AOR_ENABLE_IBM=1`.** No IBM account is created here.
- No real biometric data unless you connect a device. Simulated HRV is
  labelled simulated. Real-device path needs Chrome/Edge, HTTPS or
  localhost, and a user gesture; a BLE chest strap is the realistic
  device — Apple Watch HRV is not reliably readable from a web page.
- The bio score is a heuristic (RMSSD mapped to [0.1, 1.0]); the
  temperature mapping (temp = 1 − score) is a design choice, not a law.
  The ledger thresholds (commit < 0.4, relevance L2 < 1.5) are arbitrary
  constants kept from the original, with no validation data.
- The ledger is local-only. There is no shared/community ledger, no
  network, no auth — a hosted shared ledger is future work, not a feature.
- Tier prices ($49/$299/$999) are display-only. There is no billing in
  this app; the companion v2 has the disabled-by-default billing stub.
- The entropy reported for a quantum run is the **Shannon** entropy of the
  measured outcome distribution, not a Von Neumann entropy.
- The VQE "compiler" is keyword selection between two fixed Hamiltonians,
  not compilation of an arbitrary math string. A real SymPy term-by-term
  compiler, a hosted shared ledger with auth, and a real Muse headset
  test are the next steps — see `../SWARM_v2_SUMMARY.md`.
