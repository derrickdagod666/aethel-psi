# Architects of Reality: Cognitive Companion — v2 Summary

## What the original was

Putt pasted a Gradio app ("Architects of Reality: Cognitive Companion")
built around a Quantum Cognitive Engine: embed the user's question with
SentenceTransformers, search a FAISS index, feed the top passages as
context to a local LLM via LM Studio (`http://localhost:1234/v1`), and
stream the answer back, with tiered plans ($49 Community / $299 Architect /
$999 Enterprise) and a Stripe checkout button. Saved verbatim, unedited, at
`ARCHITECTS_original.py`. It compiles (`py_compile` OK) but does not run
as-is.

## What was broken or fake (audited, see `AUDIT.md`)

- **Missing pieces, verified:** the FAISS files `syntropic.index` /
  `syntropic_metadata.json` do not exist anywhere, and no
  `build_syntropic_index.py` was included — the app tells the user to run a
  script that was never pasted. No corpus ("Syntropic Primer") was included.
- **Assumed services:** LM Studio at `localhost:1234` was assumed running
  (connection refused on 2026-11-07); the Stripe checkout POSTs to a
  `localhost:8000` backend that was never included (connection refused), so
  the upgrade button silently did nothing; the embedding model download was
  assumed, with no cache present.
- **Fake telemetry, labelled as such:** "Cognitive Entropy" (starts 0.95,
  −0.005 per streamed token) and "Syntropic Resonance" (+1.5 per message)
  are a countdown and a message counter — they measure nothing.
- **Brittle logic:** tier handling string-matches the dropdown label
  (`"Community" in tier`); hardcoded demo email
  `demo@architectsofreality.com`; Gradio 4-style `[[user, ai]]` history
  tuples; version-sensitive `request.query_params` / `app.load`.
- **Real:** the RAG design itself — embed → FAISS top-k → system prompt
  with context → stream from an OpenAI-compatible local endpoint — is a
  sound design, and was kept.

## What v2 actually does (`cognitive-companion-v2/`)

- **Real index, built from his papers:** `build_index.py` extracts text
  from the Third Revised Edition PDF (21 pages) and the UAF paper PDF,
  chunks it (~800 chars), embeds with `all-mpnet-base-v2`, and writes
  `syntropic.index` + metadata — **96 chunks, dim 768**. If PDF extraction
  ever yields nothing, a labelled summary corpus is used instead and says so.
- **Honest LLM layer:** the engine probes LM Studio at startup. If it is
  running, it streams from the local model. If not (the state on
  2026-11-07), a demo responder answers **extractively, only from the
  retrieved passages, with sources cited**, and the UI banner and every
  demo answer say so. It never pretends the demo responder is the LLM.
- **Honest telemetry:** top-chunk L2 distance, chunks retrieved, response
  latency, words streamed, current mode — all measured, all labelled.
- **Billing, disabled by default:** tiers are keyed by id (`tier1/2/3`,
  top-k 2/3/5), the upgrade button returns a plain "billing is not
  configured" message, and the optional FastAPI stub returns 501. No Stripe
  account, product, price, or charge was created. `billing.py` documents
  exactly what enabling real Stripe would take.

## Test numbers (all run 2026-11-07, none faked)

- **Headless test: 10/10 PASS** (`test_headless_output.txt`) — index
  96 chunks / dim 768; retrieval returns 3 chunks, top L2 **0.815**;
  demo answer 2,953 chars / 493 words in **3.19 s**, cites its source
  PDFs and says it is a demo; billing disabled, no URL, no charge.
  (First run was 9/10 — the one FAIL was the test's own check string,
  not the app; the check was fixed and the re-run passed 10/10.)
- **Gradio serve check: PASS on Gradio 5.50.0** — root HTTP 200 on
  127.0.0.1:7860 (79,455 bytes, page contains "Architects of Reality"),
  `/gradio_api/info` HTTP 200 with valid JSON. Two earlier attempts on
  Gradio 5.4.0 failed — any `gr.Chatbot` crashed the API-info endpoint
  (found by component bisect) — and the upgrade to 5.50.0 fixed it; this
  re-check confirms the fix.
- **Environment fixes, recorded in `run.sh`:** the VM's `NO_PROXY`
  contains bracketed IPv6 entries that httpx cannot parse (replaced with
  plain `localhost,127.0.0.1`), and Hugging Face's XET endpoint is
  unreachable through the proxy (`HF_HUB_DISABLE_XET=1` forces plain HTTP
  downloads). `/tmp` is a 512 MB tmpfs, so pip used a workspace cache dir.

## What was dropped, and why

- The fake entropy/resonance gauges — replaced with measured values.
- The silent-fail Stripe checkout — replaced with an explicit
  not-configured message and a 501 stub. Nothing was charged, ever.
- The "Syntropic Primer" corpus — it was never in the pasted material;
  v2 uses his two theory PDFs instead, labelled as such.
- The hardcoded demo email and label-substring tier matching — replaced
  with tier ids and no email at all.

## What is needed next (per dropped piece)

- **Real LLM answers:** start LM Studio, load a model, start its local
  server at `http://localhost:1234/v1`, restart the app — the banner
  flips to "LLM mode". No code change needed.
- **Real Stripe billing:** a Stripe account, a backend implementing
  `/create-checkout-session` (the stub documents its shape), real price
  IDs for the three tiers, and keys in environment variables — Putt's
  step, with key rotation/sanitisation before anything ships.
- **Larger corpus:** add more of his papers/notes to `build_index.py`'s
  source list and rebuild (`python build_index.py --force`); retrieval
  quality is limited to the two papers today.
- **Not pushed to GitHub yet**, per instructions.

## Files

- `ARCHITECTS_original.py` — original, verbatim, unedited
- `AUDIT.md` — full audit of the original
- `cognitive-companion-v2/` — app, engine, index builder, tests, README,
  CHANGELOG, pinned `requirements.txt`
- `ARCHITECTS_v2_SUMMARY.md` — this file
