# Syntropic Swarm v2 — Summary (one page)

## What the original was

Putt's unified "Architects of Reality: The Syntropic Swarm (Master QCE)"
— a Gradio app with 8 integrations: FAISS swarm ledger, Plotly 3D
visualizer, IBM Qiskit Runtime link, SymPy+VQE "Hamiltonian compiler",
BrainFlow EEG, WebBluetooth HRV, local LLM (LM Studio), Stripe billing.
Canonical original: `SYNTROPIC_SWARM_UNIFIED_original.py` (19,162 bytes,
verbatim, unedited). It superseded a shorter 3-integration version. Full
audit: `SWARM_AUDIT.md`.

## What was broken or fake

- **IBM link dead on arrival:** `QiskitRuntimeService()` with no saved
  credentials; no simulator mode at all.
- **Mock presented as IBM execution:** no backend → hardcoded
  `{'000': 512, '111': 512}` while the UI said "IBM QUANTUM EXECUTION /
  State vector physically collapsed".
- **VQE "compiler" compiled nothing:** the `V(Psi) = ...` string was
  ignored except for a "cos" substring test; both Hamiltonians were fixed
  constants, yet the UI reported "Parsed: ...".
- **3D visualizer plotted fake data:** static vectors were
  `np.random.rand(768)` — a manifold of nothing.
- **EEG "neural entropy" was a theta/alpha ratio**, not entropy.
  (Correction: the first audit claimed its band indexing was off by
  one level — wrong. BrainFlow 5.16 returns 1-D per-band averages, so
  the original's `bands[0][2]`/`bands[0][1]` were alpha/theta
  correctly. The real EEG gaps are the synthetic board with no label,
  no stop button, and the entropy label.)
- **HRV "bio-entropy" was one RR interval's distance from 1000 ms** —
  not HRV, not entropy.
- **Ledger in-memory, no consent:** lost on restart, called a community
  swarm, no opt-in.
- **No corpus:** `syntropic.index` did not exist; LM Studio assumed
  running; LLM stream had no error handling; Gradio 4 tuple history;
  app bound 0.0.0.0:7860 (network-exposed, no auth); billing POSTed to a
  backend that does not exist.

## What v2 actually does

`syntropic-swarm-v2/` — simulator-first quantum (Aer, free, local;
IBM only with saved credentials + `AOR_ENABLE_IBM=1`); real VQE on the
simulator with the keyword-selection rule stated honestly; 3D manifold
from real embeddings (re-embedded corpus chunks, reconstructed swarm
vectors, live query); EEG with fixed band indexing, labelled synthetic
demo, stop button, ratio called a relaxation proxy; HRV from a 30-beat
RMSSD window with a documented heuristic; ledger persisted to disk,
opt-in default OFF, local only; QCE reused from companion v2 (LM Studio
probe + extractive demo fallback); measured telemetry only; two tabs;
binds 127.0.0.1:7861; billing disabled by default.

## What was dropped and why

- **"Compiles any math string into a Hamiltonian"** — dropped to keyword
  selection between two named Hamiltonians. A real SymPy term-by-term
  compiler does not exist in the original and was not invented.
- **Real IBM hardware by default** — dropped to opt-in. It needs an IBM
  Quantum account, saved credentials, queue time and quota; none exist here.
- **Shared/community ledger** — dropped to local-only. A shared ledger
  needs auth, consent, moderation and hosting; none exist here.
- **Real Muse EEG data** — dropped to labelled synthetic demo. Needs a
  paired Muse headset; none is connected here.
- **Stripe billing** — kept disabled. Needs a Stripe account, products,
  and a real backend; none exist here.

## What each dropped piece would need next

- Real VQE compiler: a SymPy parser mapping parsed terms to Pauli
  operators, with tests per term.
- Real IBM runs: an IBM Quantum account + saved credentials +
  `AOR_ENABLE_IBM=1`, and a quota/queue display in the UI.
- Shared ledger: a hosted service with auth, per-user consent, and a
  moderation policy — plus a privacy review before any user data leaves
  the machine.
- Real EEG: a paired Muse 2 headset and a test session recording known
  states (eyes open/closed) to validate the theta/alpha proxy.
- Real billing: a Stripe account, tier products/prices, and a FastAPI
  backend implementing `/create-checkout-session` (see `billing.py`).

## Test results

See the delivery report and `CHANGELOG.md` (filled in after the headless
run — nothing claimed before it runs). Headless test:
`test_headless.py` (index build, retrieval, simulator quantum job, RMSSD
biometrics, ledger persistence round-trip, demo response, Tier-3
intercept, VQE, real-embedding visualizer, EEG synthetic start/stop).

**Final run (2026-11-07): 30/30 checks PASS** (`test_headless_output.txt`).
Headlines: simulator quantum 1024 shots, entropy 0.9981 bits, GHZ
000+111 = 1024/1024; VQE generic ground state −2.0000 Ha (exact),
cos-type −1.2679 Ha; demo answer 493 words in 0.23 s; EEG value 0.710
with updates=2 on the BrainFlow SYNTHETIC board. Gradio serve check:
**HTTP 200** on http://127.0.0.1:7861 via `run.sh`; user_submit
verified via the API. One gap: the full in-browser chat round-trip
(type → see answer) was not verified — the gradio_client call to the
chained bot_response endpoint could not be routed in the time
available; the engine path it calls is covered by the headless test.
