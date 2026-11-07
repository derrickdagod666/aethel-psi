# Unified QCE v2 — CHANGELOG

## What was merged

Two separate v2 apps become ONE app (`unified-qce-v2/`, port 7862):

- **Cognitive Companion v2** (`../cognitive-companion-v2/`): RAG chat over
  Putt's papers, LM Studio probe + extractive demo fallback, tier ids,
  disabled-by-default billing, FastAPI billing stub. Reused as-is:
  `qce.py`, `build_index.py`, `corpus_fallback.py`, `billing.py`,
  `backend_stub.py` (the companion v2 shipped no requirements.txt; the
  unified build adds one).
- **Syntropic Swarm v2** (`../syntropic-swarm-v2/`): the same QCE engine
  plus quantum link, VQE, 3D visualizer, biometrics, EEG, ledger. Reused
  as-is: `quantum.py`, `biometrics.py`, `eeg.py`, `ledger.py`,
  `visualizer.py`, `engine.py` (then modified — see below),
  `requirements.txt` (then extended with fastapi/uvicorn).

The two v2 dirs were read only, never overwritten. The three original
files (`ARCHITECTS_original.py`, `SYNTROPIC_SWARM_original.py`,
`SYNTROPIC_SWARM_UNIFIED_original.py`) were not touched. The unified
app lives only in `unified-qce-v2/` plus the one-page summary
(`../UNIFIED_QCE_v2_SUMMARY.md`).

Note on the state of the two v2 builds when merged: both dirs were
complete on disk (code, READMEs, CHANGELOGs, summaries), but neither
had a built FAISS index and both venvs were empty (pip only) — so
neither headless test had actually run here. The unified build is the
first one to install, build the index, and run the full test suite;
its numbers are in `test_headless.py` output and the summary.

## What changed vs the Cognitive Companion original

1. Fake telemetry removed (was already gone in companion v2, kept gone
   here): the "Cognitive Entropy" gauge (0.95, −0.005 per token) and the
   "+1.5 Syntropic Resonance" counter are not in this app at all.
   Telemetry is measured: top-chunk L2, chunks, latency, words.
2. Tiers now mean something real: top_k 2/3/5 passages, answer depth,
   and quantum/VQE access on Tier 3 only (`qce.py` TIERS gained
   `features` and `quantum_access`; `engine.py` intercepts check
   `tier["quantum_access"]` instead of a hardcoded tier id).
3. Billing: no hardcoded demo email, no POST to a missing backend; the
   upgrade button reports "billing not configured" unless
   `AOR_BILLING_ENABLED=1` + `AOR_BACKEND_URL` are set.
4. One app instead of a companion-only app: the chat now sits alongside
   the manifold, quantum lab, biometrics, and ledger tabs.

## What changed vs the Syntropic Swarm unified original

1. **VQE compiler is real now (the headline change).** The original
   regex-matched "cos" and returned a fixed Hamiltonian while reporting
   "Parsed: ...". Unified v2 (`vqe.py`, rewritten) parses the expression
   for real: Psi-function syntax via an exact spectral mapping
   (Psi → Pauli Z on qubit 0; f(Z) = a·I + b·Z from f(±1)), and
   Pauli-term syntax parsed term-by-term. The two fixed Hamiltonians
   survive as named presets, labelled as presets. The Tier-3 chat
   intercept uses the real compiler; the old keyword selection is not
   used anywhere in the chat path.
2. One app instead of a swarm-only app: the swarm's chat tab is now the
   shared QCE Chat tab, and billing (which the swarm v2 borrowed from
   the companion) is in the same app.
3. `engine.py`: ledger creation guarded (engine no longer crashes if the
   embedding model failed to load — it degrades instead); VQE intercept
   uses `compile_expression`; tier gating via `tier["quantum_access"]`.
4. `app.py`: rewritten as five tabs (QCE Chat / 3D Manifold / Quantum
   Lab / Biometrics / Swarm Ledger & Billing) on port 7862. The chat's
   3D figure now updates the Manifold tab; a Refresh button re-renders
   it from the last query. Ledger contents are viewable in-tab.
5. Everything else from swarm v2 is unchanged: simulator-first quantum
   (no mock presented as IBM), real-embedding visualizer, RMSSD HRV,
   labelled synthetic EEG, persisted opt-in local ledger, 127.0.0.1 only.

## What was dropped and why (inherited from the v2 builds)

- "Compiles any math string into a Hamiltonian" as the original stated
  it — replaced by the real compiler above, which supports two defined
  syntaxes and fails cleanly on anything else. A general multi-variable
  compiler is future work.
- Real IBM hardware by default — simulator-first, opt-in only. Needs an
  IBM Quantum account, saved credentials, queue time and quota.
- Shared/community ledger — local-only. Needs auth, consent, moderation
  and hosting.
- Real Muse EEG data — labelled synthetic demo. Needs a paired Muse 2.
- Stripe billing — disabled. Needs a Stripe account, products, and a
  real backend.

## Test results

Headless test `test_headless.py`, run 2026-11-07 in a fresh venv:
**RESULT: 39/39 checks passed** (full log: `test_run.log`).
- Index: 96 chunks, dim 768 — 64 from the Third Revised Edition
  (21 pages, 36,903 chars), 32 from the UAF paper (10 pages, 18,630
  chars). First build took 500.3 s (embedding model download + CPU
  embedding); rebuilds are instant (index cached).
- Retrieval: "gamma limit from GW250114" → 3 chunks, top L2 = 0.82.
- Simulator quantum: 1024 shots, counts 000:515 / 111:509, Shannon
  entropy 1.0000 bits, GHZ outcomes 1024/1024.
- Biometrics: 30 simulated beats, RMSSD 26.59 ms, bio score 0.823,
  temperature 0.177.
- Ledger: disabled refuses, enabled commits, persists across reload,
  query returns the insight.
- Demo response: mode demo (no LM Studio), 493 words, cites sources,
  latency 0.24 s.
- Tier-3 quantum intercept: runs the circuit, labelled SIMULATOR.
- VQE compiler: cos(Psi) → 0.5403, Psi**2 → 1.0000, sin(Psi) →
  −0.8415, 0.5*Psi+1 → 0.5000, exp(Psi) → 0.3679 (all err 0.0000 vs
  analytic); Pauli syntax ZZ+XX → −2.0000; bad expression fails
  cleanly; cos-type preset → −1.2676 Ha.
- Tier gating: tier1 no quantum access (VQE intercept not triggered),
  tier3 compiles cos(Psi) in chat (vqe=0.540302).
- Visualizer: 10 real static embeddings (768-dim, deterministic),
  11 points in the figure.
- EEG: BrainFlow SYNTHETIC board (brainflow_installed=True after the
  setuptools fix below), value 0.950, stops cleanly.
- Serve check: HTTP 200 on 127.0.0.1:7862.

## Fixes found while testing this build (not in either v2)

1. **qiskit-algorithms 0.3.1 is incompatible with qiskit 2.2.1**
   (imports BaseSampler, removed in Qiskit 2.x) — the inherited pin
   never worked; neither v2 build had actually run its tests. Pinned
   qiskit-algorithms==0.4.0, and `run_vqe` now uses Aer EstimatorV2
   (0.4.0's VQE passes V2-style pubs) with the TwoLocal ansatz
   `.decompose()`d (Aer rejects the raw TwoLocal instruction).
2. **Gradio 5.4.0 crashes on the Chatbot component**: gradio-client
   1.4.2's `get_api_info()` raises TypeError on the Chatbot schema's
   `"additionalProperties": true`, which made `launch()` fail its
   localhost check. Upgraded to gradio==5.50.0 +
   gradio-client==1.14.0, where the bool case is handled.
3. **BrainFlow unimportable with default setuptools**: brainflow 5.16.0
   imports pkg_resources, removed in setuptools 81+. Pinned
   setuptools==80.9.0; EEG now uses the real BrainFlow synthetic board.
4. **Sandbox proxy env breaks Gradio import/launch**: `no_proxy`
   contains bracketed IPv6 entries httpx cannot parse. `app.py` sets
   no_proxy/NO_PROXY to `localhost,127.0.0.1` before importing gradio.
5. **pip needs TMPDIR off /tmp**: /tmp is a 512 MB tmpfs and the torch
   wheel (539 MB) does not fit — install with TMPDIR under $HOME.
   Noted in requirements.txt.
