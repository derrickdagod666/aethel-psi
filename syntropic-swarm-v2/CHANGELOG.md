# CHANGELOG — Syntropic Swarm v2

## v2 (2026-11-07)

Built from `../SYNTROPIC_SWARM_UNIFIED_original.py` (19,162 bytes, saved
verbatim, unedited) — the UNIFIED version, which superseded the shorter
`../SYNTROPIC_SWARM_original.py`. Full audit: `../SWARM_AUDIT.md`.
Two brief corrections recorded in the audit: the unified file DOES end
with `if __name__ == "__main__":` (dunders present), and the feared
Estimator `__init__` crash does not exist (early return guards it).

### Reused (from cognitive-companion-v2, copied unmodified)
- `build_index.py` + `corpus_fallback.py` — FAISS index builder over
  Putt's papers (Third Revised Edition + UAF paper), labelled corpus.
- `qce.py` — QCE engine: LM Studio probe, extractive demo fallback,
  tier-id dict, Gradio 5 messages format, measured telemetry.

### Changed vs the original (elaborated per Putt's rule)
1. **Quantum layer (`quantum.py`, new):** simulator-first (Aer, default,
   free, local). IBM Runtime only with saved credentials AND
   `AOR_ENABLE_IBM=1`; otherwise falls back to simulator with a printed
   reason. OFF mode added. Entropy correctly labelled Shannon (of the
   measured counts), not "Von Neumann / Shannon".
2. **Biometrics (`biometrics.py`, new):** RMSSD over a rolling 30-beat
   window instead of |1000 − rr| / 1000 for one interval; documented
   heuristic mapping to [0.1, 1.0]; simulated-HRV demo mode (labelled)
   plus the real WebBluetooth path with its limits stated.
3. **Ledger (`ledger.py`, new):** persisted to disk
   (`swarm_ledger.index` + `swarm_ledger.json`), opt-in default OFF,
   local-only; synchronous commit returning (committed, reason) instead
   of a fire-and-forget thread. Thresholds kept but labelled arbitrary.
4. **Engine (`engine.py`, new):** composes the reused QCE with the new
   layers; Tier-3 quantum intercept is mode-aware; LLM stream wrapped in
   try/except (the original had none); swarm + static context labelled
   by source.
5. **App (`app.py`, new):** measured telemetry only; WebBluetooth JS
   kept verbatim from the original. Chat history is a messages list
   held in a gr.State and rendered as a scrollable HTML log (see §9 —
   gr.Chatbot crashes this Gradio version).
6. **Unified additions:** `visualizer.py` (real embeddings, no dummy
   vectors), `vqe.py` (keyword selection + real VQE on Aer), `eeg.py`
   (labelled synthetic demo, stop button; band indexing verified live
   — the original's indexing was correct for BrainFlow 5.16's 1-D
   return, the first audit's claim otherwise was wrong and is
   corrected in the audit), two-tab UI with the 3D plot panel,
   127.0.0.1-only binding (the original bound 0.0.0.0:7860), billing
   reused from companion v2 (disabled by default).
7. **VQE implementation fix (found by the headless test):** first
   written with `qiskit_algorithms.VQE` — that import crashes under
   Qiskit 2.2.1 (qiskit-algorithms 0.3.1 needs the removed
   `BaseSampler`). Replaced with Aer `EstimatorV2` + scipy COBYLA,
   with the ansatz transpiled to Aer basis gates (raw `TwoLocal` is an
   unknown instruction to Aer) and the 0-d `evs` result flattened.
   `qiskit-algorithms` was dropped from requirements.txt.
8. **BrainFlow fix (found by the headless test):** BrainFlow 5.16
   imports `pkg_resources`, removed in setuptools 82+ — pinned
   `setuptools==81.0.0`. The EEG streaming loop now handles both the
   1-D and 2-D `get_avg_band_powers` return shapes and is guarded so
   a bad window cannot kill the thread (the first v2 version assumed
   2-D, crashed the thread, and froze the value at its 0.95 initial
   constant — the test now asserts `updates >= 1` so a frozen value
   cannot pass again).
9. **Chat log is HTML, not gr.Chatbot (found by the serve check):**
   Gradio 5.4.0 / gradio-client 1.4.2 crashes the server on the root
   page whenever a Chatbot is wired to an event — the Chatbot's API
   schema contains `"additionalProperties": true` (a boolean, not a
   schema) and `json_schema_to_python_type` dies with `TypeError:
   argument of type 'bool' is not iterable`. Verified by bisecting the
   app's config and by minimal probes; gradio-client 1.4.3 and 1.5.0
   do NOT fix it. The conversation is now rendered as a scrollable
   HTML log (`render_chat`, history still a messages list in a
   gr.State); only the rendering changed. Upgrading the whole Gradio
   stack is a separate decision — the companion app shares the 5.4.0
   pin and will hit the same crash.
10. **Serve-time env quirk (found by the serve check, in run.sh):**
    the VM's NO_PROXY contains bracketed IPv6 literals ([::1],
    [fd8b:...]) that httpx cannot parse at gradio import time, and
    gradio's launch() makes an outbound call that dies through the
    egress proxy. `run.sh` drops all proxy vars for the app process,
    sets a clean NO_PROXY for localhost, and sets HF_HUB_OFFLINE=1 so
    the cached embedding model loads without a hub HEAD request.

### Test results (2026-11-07, final run)
- Headless test `test_headless.py`: **30/30 checks PASS**
  (test_headless_output.txt). Index 96 chunks dim 768; retrieval top
  L2 0.82; simulator quantum 1024 shots, entropy 0.9981 bits, GHZ
  000+111 = 1024/1024; RMSSD 26.59 ms, bio score 0.823, temp 0.177;
  ledger disabled-refuses / enabled-commits / persists across reload
  / query returns the insight; demo answer 493 words in 0.23 s,
  telemetry present; Tier-3 entangle runs the circuit, labelled
  SIMULATOR (no mock as IBM); VQE generic ground state **-2.0000 Ha**
  (exact), cos-type **-1.2679 Ha**; visualizer 11 real points,
  embeddings deterministic; EEG value 0.710 with **updates=2** on the
  BrainFlow SYNTHETIC board (brainflow_installed=True).
- Gradio serve check: **HTTP 200**, 48,999 bytes, page contains
  "Syntropic Swarm v2" and "QCE Synthesis Terminal", on
  http://127.0.0.1:7861 via run.sh. API check: user_submit returns
  the rendered HTML with the user message. Full in-browser chat
  round-trip (type → see answer) was NOT verified here — the
  gradio_client call to the chained bot_response endpoint could not
  be routed in the time available; the engine path it calls is
  covered by the headless test.
- Environment notes: venv install took ~24 min (torch 539 MB over a
  flaky proxy, --resume-retries 10); model download needed one retry
  (proxy drop, resumed from cache). No IBM account, no hardware job,
  no charges, no secrets anywhere.
