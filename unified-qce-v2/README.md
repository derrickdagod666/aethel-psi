# Architects of Reality: Unified QCE v2

**One app, not two.** This merges the Cognitive Companion v2 and the
Syntropic Swarm v2 (Master QCE) into a single Gradio app. The two v2
builds were already 90% the same code (identical `qce.py`,
`build_index.py`, `corpus_fallback.py`, `billing.py`); the swarm v2 had
already reused the companion's engine. The unified build finishes the
job: one engine, one FAISS index, one ledger, one billing stub, five
tabs, one launch on port 7862.

Canonical originals (untouched, one directory up):
- `../ARCHITECTS_original.py` — Cognitive Companion (RAG + tiers +
  Stripe + fake entropy/resonance gauges)
- `../SYNTROPIC_SWARM_UNIFIED_original.py` — the unified Swarm (8
  integrations), authoritative over `../SYNTROPIC_SWARM_original.py`
  (the shorter 3-integration version)
- Audits: `../AUDIT.md`, `../SWARM_AUDIT.md`
- The two in-flight v2 dirs (`../cognitive-companion-v2/`,
  `../syntropic-swarm-v2/`) were read, not overwritten. Their finished
  files were reused; the unified app is new work on top (see CHANGELOG).

## The five tabs

1. **QCE Chat** — RAG synthesis over Putt's own papers. LM Studio
   first; if it is not running, a clearly-labelled extractive demo
   responder answers ONLY from retrieved passages, with sources. Tier
   dropdown carries tier ids. Telemetry is measured only: top-chunk L2
   distance, chunks retrieved, latency, words streamed.
2. **3D Manifold** — Plotly PCA of REAL embeddings: static corpus
   chunks, swarm ledger vectors, live query. Layers with no data are
   absent, never faked with random vectors.
3. **Quantum Lab** — the 3-qubit Aethel-Psi circuit (H, CX, CX,
   RZ(phase·π)) on the Aer simulator (1024 shots, real counts, Shannon
   entropy). IBM Runtime only with saved credentials AND
   `AOR_ENABLE_IBM=1`; OFF disables it. Plus the VQE compiler: type a
   V(Psi) expression and it is COMPILED for real (see below), or run one
   of the two named presets from the originals.
4. **Biometrics** — simulated HRV demo (labelled) or the real
   WebBluetooth path (Chrome/Edge, HTTPS or localhost; BLE chest strap
   realistic, Apple Watch HRV not reliably readable from a web page).
   BrainFlow EEG: synthetic board labelled simulated, real Muse 2 path
   documented. Bio score from a 30-beat RMSSD window — a documented
   heuristic, not an entropy.
5. **Swarm Ledger & Billing** — ledger persisted to disk
   (`swarm_ledger.index` + `swarm_ledger.json`), opt-in default OFF,
   local only. Tier meanings listed. Stripe disabled by default.

## The VQE compiler actually compiles now

The originals claimed to compile `V(Psi) = ...` into a Hamiltonian and
did not — they substring-matched "cos" and returned a fixed Hamiltonian.
Unified v2 parses the expression for real, in two syntaxes:

- **Psi-function syntax** (`cos(Psi)`, `Psi**2 + 0.5*Psi`, `exp(Psi)`):
  Psi is encoded as Pauli Z on qubit 0 (a stated modelling choice).
  Since Z² = I, any function f compiles exactly to a·I + b·Z with
  a = (f(+1)+f(−1))/2, b = (f(+1)−f(−1))/2. Verified: cos(Psi) →
  0.5403·I, sin(Psi) → 0.8415·Z, exp(Psi) → 1.5431·I + 1.1752·Z.
- **Pauli-term syntax** (`0.5*ZZ + 1.0*IX`, `ZZ + XX`): parsed
  term-by-term into a SparsePauliOp (1–3 qubits).
- **Named presets** (`cos-type`, `generic-type`): the two fixed
  Hamiltonians from the originals, selectable by name in the Quantum
  Lab tab and labelled as presets — not compilation.

The VQE itself is real in all cases (Aer Estimator + TwoLocal + COBYLA).

## Tiers mean something real

| Tier | Passages | Depth | Quantum / VQE intercepts |
|---|---|---|---|
| Tier 1 Community ($49/mo) | 2 | conceptual | no |
| Tier 2 Researcher ($299/mo) | 3 | derivations | no |
| Tier 3 Enterprise ($999/mo) | 5 | full depth | yes |

Prices are display-only labels from the original; no payment is
possible in this build. Billing is disabled by default: no Stripe
account, products, prices, keys, or charges. The optional FastAPI stub
(`backend_stub.py`) returns 501 "billing not configured". To enable real
billing one day: Stripe account + products/prices, a real backend
implementing `/create-checkout-session`, then `AOR_BILLING_ENABLED=1`
and `AOR_BACKEND_URL`. No hardcoded demo email is used anywhere.

## Corpus

FAISS index built from Putt's own papers (see `build_index.py`):
- Third Revised Edition of the Aethel-Psi Synthesis (Nov 2026)
- UAF paper, *Quantum and Geometric Harmonic Resonance* (Oct 2026)

If PDF extraction ever yields nothing, a labelled fallback corpus of
summaries is used instead (`corpus_fallback.py`).

## Run it

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt   # large: torch ~2 GB
./venv/bin/python test_headless.py     # full headless test, exit 0 = pass
./venv/bin/python app.py              # serves on http://127.0.0.1:7862
```

First run downloads the `all-mpnet-base-v2` embedding model (~420 MB,
once) and builds the index (~1–2 min). First app start is slow
(~1 min) because the embedding model loads. To use full LLM mode: start
LM Studio, load a model, start its local server (default
http://localhost:1234/v1), then restart the app. The banner will read
"LLM mode". Override with `AOR_LLM_URL` / `AOR_LLM_MODEL`.

## Env vars

- `AOR_QUANTUM_MODE=simulator|ibm|off` (default `simulator`)
- `AOR_ENABLE_IBM=1` — required, with saved IBM Quantum credentials, to
  submit a real-hardware job. Never set this unless you intend to spend
  IBM Quantum quota and wait in a hardware queue.
- `AOR_LLM_URL`, `AOR_LLM_MODEL` — LM Studio endpoint.
- `AOR_BILLING_ENABLED=1` + `AOR_BACKEND_URL` — only to enable real billing.

## Files

| File | What it is |
|---|---|
| `app.py` | The one Gradio app (5 tabs, port 7862, 127.0.0.1) |
| `engine.py` | Unified engine: QCE + ledger + quantum + bio + EEG + VQE |
| `qce.py` | RAG engine: retrieval, LM Studio probe, demo responder, tiers |
| `vqe.py` | **Real VQE compiler** (Psi-function + Pauli-term syntax), presets, VQE runner |
| `quantum.py` | Simulator-first quantum link (Aer; IBM opt-in, OFF mode) |
| `biometrics.py` | HRV: 30-beat RMSSD window, simulated + WebBluetooth modes |
| `eeg.py` | BrainFlow EEG: synthetic (labelled) / Muse 2, theta-alpha proxy |
| `ledger.py` | Persisted, opt-in, local-only swarm ledger |
| `visualizer.py` | 3D manifold from real embeddings only |
| `build_index.py` | Builds `syntropic.index` from Putt's papers |
| `corpus_fallback.py` | Labelled summary corpus (only if PDFs fail) |
| `billing.py` | Billing, disabled by default |
| `backend_stub.py` | Optional FastAPI stub; 501 "billing not configured" |
| `test_headless.py` | Full headless test (index, retrieval, quantum, bio, ledger, demo, VQE compiler, tiers, visualizer, EEG, serve check) |
| `requirements.txt` | Pinned versions actually tested with |

## Honest limits

- Simulator by default. **No real-hardware job is ever submitted without
  saved credentials + `AOR_ENABLE_IBM=1`.** No IBM account is created here.
- The VQE compiler's Psi→Z encoding is a modelling choice, documented
  in `vqe.py` — it is not a derivation of the waveform from physics.
  A multi-variable compiler (Psi, Theta, ...) is future work.
- Demo responder is extractive, not generative — it cannot answer
  questions whose answer is not in the retrieved passages.
- Retrieval quality depends on the embedding model and chunking; the
  corpus is two papers, so coverage is limited to what they contain.
- The bio score is a heuristic (RMSSD mapped to [0.1, 1.0]); the
  temperature mapping (temp = 1 − score) is a design choice, not a law.
  The ledger thresholds (commit < 0.4, relevance L2 < 1.5) are arbitrary
  constants kept from the original, with no validation data.
- The ledger is local-only. There is no shared/community ledger, no
  network, no auth — a hosted shared ledger is future work, not a feature.
- No real biometric data unless you connect a device. Simulated HRV/EEG
  are labelled simulated. Real EEG needs a paired Muse 2 headset.
- Tier prices are display-only. There is no billing in this app.
- The entropy reported for a quantum run is the **Shannon** entropy of
  the measured outcome distribution, not a Von Neumann entropy.
- Local only. No hosted deployment, no accounts, no real billing.
- Slow first start (embedding model load); index rebuild needed if the
  papers change (`python build_index.py --force`).
