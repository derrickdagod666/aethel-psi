# Unified QCE v2 — Summary (one page)

## What it is

One Gradio app (`unified-qce-v2/`, port 7862, 127.0.0.1 only) merging
the Cognitive Companion v2 and the Syntropic Swarm v2 (Master QCE).
The two v2 builds were already ~90% the same code; this finishes the
merge: one engine, one FAISS index, one ledger, one billing stub, five
tabs — **QCE Chat, 3D Manifold, Quantum Lab, Biometrics, Swarm Ledger
& Billing**. Originals untouched (`ARCHITECTS_original.py`,
`SYNTROPIC_SWARM_UNIFIED_original.py`, `SYNTROPIC_SWARM_original.py`);
the two v2 dirs were read only, never overwritten. Nothing was pushed
to GitHub; no Stripe/IBM account created; no secrets copied.

## What was broken or fake in the originals (audit headline)

- **VQE "compiler" compiled nothing** — substring-matched "cos" and
  returned a fixed Hamiltonian while reporting "Parsed: ...".
- **Mock presented as IBM execution** — hardcoded `{'000': 512,
  '111': 512}` while the UI said "IBM QUANTUM EXECUTION".
- **3D visualizer plotted fake data** — static vectors were
  `np.random.rand(768)`.
- **"Cognitive Entropy" gauge and "+1.5 resonance" counter** were
  counters, not measurements. **HRV "bio-entropy"** was one RR
  interval's distance from 1000 ms. **EEG "neural entropy"** was a
  theta/alpha ratio with band indexing off by one level.
- **Ledger in-memory, no consent**, called a community swarm.
- **Billing POSTed to a backend that did not exist** with a hardcoded
  demo email. **Server bound 0.0.0.0:7860**, no auth.
- **Tiers were prompt wording only**, string-matched on labels.

## What v2 actually does

- **Real VQE compiler (the headline change):** `V(Psi)` expressions
  are parsed by SymPy — Psi-function syntax via an exact spectral
  mapping (Psi → Pauli Z on qubit 0; f(Z) = a·I + b·Z from f(±1)) and
  Pauli-term syntax parsed term-by-term. The two fixed Hamiltonians
  survive as named presets, labelled as presets.
- **Simulator-first quantum** (Aer, 1024 shots, real counts + Shannon
  entropy); IBM only with saved credentials + `AOR_ENABLE_IBM=1`.
- **Real-embedding manifold** (corpus chunks, ledger vectors, query);
  absent layers are absent, never faked.
- **Honest telemetry only** (top L2, chunks, latency, words); tiers
  mean something real (top_k 2/3/5, depth, Tier-3-only quantum/VQE).
- **RMSSD HRV** (30-beat window, documented heuristic), **labelled
  simulated EEG** (BrainFlow synthetic board) + real Muse path,
  **persisted opt-in local ledger**, **billing disabled** (501 stub).

## Test results — 39/39 PASS (fresh venv, `test_headless.py`)

- **Index:** 96 chunks, dim 768 (64 Third Revised Edition + 32 UAF
  paper). **Retrieval:** gamma/GW250114 query, top L2 = 0.82.
- **Quantum:** 1024 shots, 000:515 / 111:509, Shannon entropy 1.0000
  bits, GHZ 1024/1024. **Demo response:** 493 words, cites sources.
- **VQE compiler:** cos(Psi) → 0.5403, Psi² → 1.0000, sin(Psi) →
  −0.8415, 0.5·Psi+1 → 0.5000, exp(Psi) → 0.3679 (err 0.0000 vs
  analytic); ZZ+XX → −2.0000; cos-type preset → −1.2676 Ha.
  Tier 1 does NOT trigger VQE; Tier 3 compiles in chat (0.540302).
- **Biometrics:** RMSSD 26.59 ms, bio score 0.823. **Ledger:**
  round-trip persists. **EEG:** BrainFlow synthetic, value 0.950.
  **Serve:** HTTP 200 on 127.0.0.1:7862.

## Fixes found while testing (not in either v2)

The inherited pins had never actually run: qiskit-algorithms 0.3.1
is incompatible with qiskit 2.2.1 (→ 0.4.0 + EstimatorV2 +
decomposed ansatz); Gradio 5.4.0 crashed on the Chatbot schema
(→ 5.50.0 + gradio-client 1.14.0); BrainFlow needs setuptools<81
(→ 80.9.0); sandbox `no_proxy` broke Gradio's httpx import
(`app.py` resets it to localhost,127.0.0.1); pip needs TMPDIR off
the 512 MB /tmp tmpfs. All recorded in `requirements.txt`/`CHANGELOG.md`.

## What is still blocked / not real

- **No LM Studio here** — app runs in labelled demo mode (extractive
  only). Start LM Studio + a model for full synthesis.
- **No IBM Quantum credentials** — simulator only; real hardware needs
  an account + saved credentials + `AOR_ENABLE_IBM=1`.
- **No real biometrics** — simulated HRV/EEG only; needs a BLE chest
  strap / paired Muse 2 headset.
- **No billing** — no Stripe account, products, or prices; upgrade
  button reports "billing not configured".
- **No shared ledger** — local only; a hosted one needs auth/consent.
- **VQE compiler is single-variable** (Psi → Z on qubit 0) — a stated
  modelling choice, not physics; multi-variable is future work.
- First start is slow (~1 min model load; first index build 500 s on
  CPU). Venv is ~5 GB (CUDA torch; CPU-only torch would be smaller).

## Run it

```bash
cd unified-qce-v2
python -m venv venv
TMPDIR=$HOME/.pip-tmp venv/bin/pip install -r requirements.txt
venv/bin/python test_headless.py   # 39 checks, exit 0 = pass
venv/bin/python app.py             # http://127.0.0.1:7862
```
