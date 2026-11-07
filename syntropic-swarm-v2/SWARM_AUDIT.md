# SWARM AUDIT — "Architects of Reality: The Syntropic Swarm (Master QCE)"
# UNIFIED version (canonical original)

Original: `SYNTROPIC_SWARM_UNIFIED_original.py` (19,162 bytes, as pasted by
Putt, 2026-11-07). **This unified version supersedes the shorter version**
(`SYNTROPIC_SWARM_original.py`) that the first audit covered: it has all 8
integrations (ledger, 3D visualizer, IBM link, VQE compiler, BrainFlow EEG,
WebBluetooth HRV, local LLM, Stripe billing) instead of 3. Findings from the
shorter version that still apply are kept below; new findings are marked
[UNIFIED].

Method: read the code in full, ran `python3 -m py_compile` (exit 0 — it
compiles), and traced every external dependency by hand. BrainFlow and IBM
Runtime could not be executed here at audit time (not installed, no
hardware, no credentials), so EEG/IBM findings are code-level review,
stated as such. No secrets were found in the pasted text; none copied. No
IBM Quantum account was created, no job was run on real hardware, nothing
was charged.

## Corrections to the task brief (verified against the file)

- The file does **NOT** end with `if name == "main":`. Line 360 is
  `if __name__ == "__main__":` with the dunders present. The paste is
  correct; the brief's claim was wrong.
- The feared `__init__` crash (Estimator/optimizer/ansatz created after
  the IBM try/except) does **NOT** exist as stated: `if not
  QISKIT_AVAILABLE: return` runs before those lines, and if `qiskit_aer`
  were missing the top-level import would already have set
  `QISKIT_AVAILABLE = False`. The lines only run when all imports
  succeeded. (Whether `TwoLocal()` with no `num_qubits` works with the
  2-qubit VQE Hamiltonian is still unverified until run — see §7.)

## What is real in the original

- The RAG structure is sound: embed query → FAISS search → system prompt
  with retrieved context → stream from an OpenAI-compatible local endpoint.
- The 3-qubit circuit (H, CX, CX, RZ(phase·π), measure-all) is a valid
  Qiskit circuit, and the Shannon entropy of measured counts is a real
  calculation from real counts — *when a backend exists*.
- The WebBluetooth JS parses the BLE heart-rate measurement correctly for
  the RR-interval flag (0x10) and converts 1/1024 s units to ms correctly.
- `QISKIT_AVAILABLE` / `BRAINFLOW_AVAILABLE` import guards exist, so the
  file imports without Qiskit or BrainFlow.
- Env-var configuration is real: `LLM_API_BASE`, `BILLING_API_URL`
  (defaults `http://localhost:1234/v1`, `http://localhost:8000`), demo
  email changed to `demo@architects.com`.

## What is broken or missing

1. **IBM Quantum link cannot work as written.** `QiskitRuntimeService()`
   with no arguments needs saved IBM Quantum credentials. None are in the
   pasted material. On a machine without them the constructor raises, the
   except prints "IBM Quantum offline", `backend` stays `None` — and the
   app then silently runs the mock (see §4) while telling the user it ran
   on IBM hardware.
2. **Real hardware is the default and only target.** `least_busy(
   simulator=False, operational=True)` picks real superconducting hardware:
   queue times, a limited free quota, no cost/usage display. There is no
   simulator mode, even though the same circuit runs identically on Aer
   locally for free.
3. **The static FAISS index does not exist.** `syntropic.index` /
   `syntropic_metadata.json` are loaded in a bare `except:` that swallows
   the failure and sets `static_index = None`. The app then runs with no
   corpus and no warning to the user.
4. **[UNIFIED] Mock results presented as IBM execution.** When no IBM
   backend exists, `execute_topological_collapse` returns the hardcoded
   `{'000': 512, '111': 512}` (entropy exactly 1.0) — labelled in a code
   comment "Fallback mock for local testing" — but `generate_response`
   then yields `"🌌 IBM QUANTUM EXECUTION\nMeasured Entropy: 1.0000\nState
   vector physically collapsed."` The UI message claims a physical
   collapse on IBM hardware that never happened. This is the single most
   misleading line in the file.
5. **LM Studio is assumed running.** The OpenAI client is constructed at
   import time and `generate_response` calls `chat.completions.create(...)`
   with **no try/except around the stream**. If LM Studio is down, the
   generator raises and the Gradio chat breaks for that turn.
6. **Gradio API is stale.** History is `[[user, ai]]` tuples; Gradio 5
   uses `messages` format by default. There is no submit button in this
   version, only `msg.submit`. `eeg_btn.click(fn=start_stream,
   outputs=eeg_btn)` feeds the returned string back into the button.
7. **[UNIFIED] VQE "compiler" does not compile anything.** `compile_vqe`
   parses `V(Psi) = ...` with a regex, then ignores the parsed expression
   except for a substring test: if `"cos"` appears anywhere, the
   Hamiltonian is the fixed `ZZ + 0.5·IX − 0.15·ZI`; otherwise it is the
   fixed `ZZ + XX`. The math string is never turned into a Hamiltonian —
   the two Hamiltonians are constants, and the "Parsed: ..." message
   implies the expression was compiled. A real compiler would map parsed
   terms to Pauli operators. Also unverified until run: `TwoLocal()` is
   constructed with no `num_qubits` while both Hamiltonians are 2-qubit,
   and `VQE(estimator, ansatz, optimizer)` depends on the installed
   qiskit-algorithms version.
8. **[UNIFIED] Swarm commit thread is fire-and-forget.** As in the shorter
   version: `threading.Thread(...).start()` with no join, no error
   handling — a failed commit dies silently.
9. **Tier gating is string matching.** `"Enterprise" in tier` matches the
   dropdown label; renaming a tier silently changes behaviour.
10. **[UNIFIED] The app binds 0.0.0.0:7860** — exposed on the local network,
    not just localhost. Combined with no auth, anyone on the network can
    use the LLM and read the ledger.

## What is fake, arbitrary, or mislabelled

11. **[UNIFIED] The 3D visualizer plots fake data.** When the index loads,
    `static_vectors` is filled with `np.random.rand(768)` — 50 random
    768-dim vectors — with a comment admitting they are "dummy vectors".
    The PCA scatter then shows random points labelled "Static Primer",
    i.e. a manifold plot of nothing. The swarm vectors are real (embedded
    queries), the static ones are not.
12. **"Bio-entropy" (HRV) is not HRV and not entropy.** `update_hrv`
    takes a *single* RR interval and computes `|1000 − rr| / 1000`. One
    heartbeat's distance from 1000 ms is not heart rate variability
    (which needs a window of intervals, e.g. RMSSD) and not an entropy.
13. **[UNIFIED] EEG "neural entropy" is a theta/alpha ratio, not entropy.**
    `_calculate_neural_entropy` computes `theta_power / alpha_power`,
    clips it to [0.1, 1.0], and smooths it 0.8/0.2. A band-power ratio is
    not an entropy (no distribution, no logarithm). It is a relaxation
    proxy wearing an entropy label.
14. **[UNIFIED] EEG band indexing — CORRECTION (first audit was wrong).**
    The first version of this audit claimed `bands[0][2], bands[0][1]`
    selected channels, not bands, because `avg_bands` was assumed to be
    per-channel. That was WRONG, and it was caught by executing the code:
    BrainFlow 5.16's `get_avg_band_powers` returns a tuple
    `(avg, std)` where `avg` is a **1-D array of length 5, [delta,
    theta, alpha, beta, gamma], already averaged over channels**
    (verified live on the synthetic board: avg shape `(5,)`). So
    `bands[0][2]` is alpha and `bands[0][1]` is theta — both scalars —
    and the original's ratio was correct. The detrend loop is also
    correct as written. What v2 changed instead: the loop body is now
    guarded so a bad data window cannot kill the streaming thread (the
    first v2 build assumed a 2-D return, crashed the thread on the
    first window, and silently froze the value at its 0.95 initial
    constant — caught by the headless test), and the code handles both
    the 1-D and 2-D return shapes.
15. **[UNIFIED] BrainFlow runs on the SYNTHETIC board by default.**
    `BoardIds.SYNTHETIC_BOARD` generates fake EEG in software — it is not
    a Muse headset. The comment says "Swap to MUSE_2_BOARD in prod", so
    the code is honest about it, but nothing in the UI says the EEG is
    simulated. There is also no stop: `is_streaming` is never set False,
    the daemon thread runs forever, and `start_stream(45000)` uses a
    45-second buffer with no explanation.
16. **Temperature modulation is a design choice, not a law.**
    `dynamic_temp = 1.0 − bio_entropy` ("stressed = grounded, flow =
    creative") is an invented mapping with no evidence behind it.
17. **Swarm thresholds are arbitrary constants.** Commit threshold
    `bio_entropy < 0.4` and relevance threshold `L2 < 1.5` have no
    validation, no units, no tuning data.
18. **The ledger is in-memory only and shared by design, with no consent.**
    `IndexFlatL2` + lists are lost on restart; no persistence, no auth,
    no opt-in. Any "coherent" synthesis is committed to a ledger the UI
    calls a community swarm, with no indication to the user that their
    words are being stored or shared.
19. **WebBluetooth reality check.** `navigator.bluetooth` only exists in
    Chrome/Edge, only over HTTPS or localhost, only with a user gesture.
    Apple Watch HRV is not exposed as a plain BLE heart-rate service to
    arbitrary web pages; a BLE chest strap (e.g. Polar H10) is the
    realistic device.
20. **Billing is a dead end.** `get_checkout_url` POSTs to
    `$BILLING_API_URL/create-checkout-session`, a backend that is not in
    the pasted material, and returns `["#"]` on any failure — the upgrade
    button then does nothing. Demo email `demo@architects.com` is
    hardcoded.

## What v2 changes (summary — see CHANGELOG.md in syntropic-swarm-v2/)

- Simulator-first quantum layer (Aer default, free, local); IBM only with
  saved credentials + `AOR_ENABLE_IBM=1`; never a mock presented as IBM.
- VQE: the message text SELECTS between two named, documented
  Hamiltonians (keyword rule stated in the UI/README); the VQE itself is
  real (Aer Estimator + TwoLocal + COBYLA on the simulator) and reports a
  genuine ground-state energy. The "compiles any math string" claim is
  dropped.
- 3D visualizer plots REAL embeddings: static chunks from the actual
  FAISS index (re-embedded from the corpus), swarm vectors, and the live
  query vector. No random dummy vectors.
- EEG: BrainFlow synthetic board used explicitly as a labelled demo
  (real BrainFlow, fake signal); band indexing verified live (1-D
  per-band averages in BrainFlow 5.16 — the original's indexing was
  correct; v2 handles both shapes and guards the streaming loop); a
  stop button; ratio labelled a relaxation proxy, not entropy. Real
  Muse board path documented, not claimed.
- HRV: rolling RMSSD window, documented heuristic, simulated demo mode.
- Ledger persisted to disk, opt-in default OFF, local-only.
- App binds 127.0.0.1 only; two tabs kept (Quantum Interface +
  Cybernetics & Billing); chat history rendered as an HTML log, not
  gr.Chatbot (the Chatbot component crashes gradio-client 1.4.2's
  api_info on the root page — see CHANGELOG §9); LLM stream wrapped
  in try/except; reused companion-v2 QCE (probe + demo fallback).
