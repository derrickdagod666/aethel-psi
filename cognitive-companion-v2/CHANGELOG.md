# CHANGELOG — Architects of Reality v2 vs the original

## What changed, and why (per Putt's rule)

1. **Index builder added (`build_index.py`).** The original assumed
   `syntropic.index` + `syntropic_metadata.json` existed and told the user
   to run a `build_syntropic_index.py` that was never included. v2 ships
   the builder: it extracts text from Putt's Third Revised Edition PDF and
   the UAF paper PDF with pypdf, chunks (~800 chars, ~150 overlap, on
   sentence boundaries), embeds with `all-mpnet-base-v2`, and writes a
   FAISS IndexFlatL2 + metadata. If PDF extraction yields nothing, it
   falls back to a labelled summary corpus (`corpus_fallback.py`) instead
   of shipping an empty index.
2. **LM Studio probed at startup (`qce.py`).** The original created the
   OpenAI client unconditionally and failed on every query when LM Studio
   was down. v2 probes `/v1/models` (2 s timeout). If reachable, it uses
   the local model exactly as the original intended. If not, a demo
   responder answers extractively from retrieved passages only, with
   sources, and the UI banner says "DEMO MODE — extractive only".
3. **Fake telemetry removed.** The original "Cognitive Entropy" gauge
   (0.95 − 0.005 per token) and "Syntropic Resonance" (+1.5 per message)
   measured nothing. v2 shows measured values only: top-chunk L2 distance,
   chunks retrieved, response latency in seconds, words streamed, and the
   current mode.
4. **Tiers keyed by id, not label substrings.** The original matched
   `"Community" in tier` etc. v2 uses a `TIERS` dict (`tier1/2/3` → label,
   top_k, instruction) and the dropdown carries tier ids as values.
   Top-k retrieval varies by tier (2/3/5).
5. **Gradio 5 messages-format history, on Gradio 5.50.0.** The original
   used the Gradio 4 `[[user, ai]]` tuple format. v2 uses `type="messages"`
   with `{"role", "content"}` dicts. Version note: Gradio 5.4.0 /
   gradio-client 1.4.2 crashed on this format (API-info endpoint dies in
   `_json_schema_to_python_type` on the messages schema's
   `"additionalProperties": false`, which also kills `launch()`'s
   localhost check — found by bisecting each component). Upgrading to
   Gradio 5.50.0 / gradio-client 1.14.0 fixed it; a tuples-format detour
   was tried and reverted. `qce.py` accepts both tuple and dict history
   entries, so the engine does not care which format the UI uses.
6. **Billing disabled by default (`billing.py`, `backend_stub.py`).**
   The original POSTed to a `localhost:8000` backend that did not exist,
   with a hardcoded demo email, and the upgrade button silently failed.
   v2 returns an honest "billing not configured" message and never
   contacts Stripe. The optional FastAPI stub returns 501 for the same
   endpoint, documenting the real backend's shape. No Stripe account,
   product, price, or charge was created.
7. **Dropped the unused `numpy` import** and added `requirements.txt`
   (pinned, tested), `test_headless.py`, and this changelog.

## Test results (recorded in the delivery report)

- Headless test: `test_headless_output.txt`, 2026-11-07 — 10/10 PASS
  (index 96 chunks, dim 768; retrieval top L2 0.815; demo answer 2,953
  chars / 493 words in 3.19 s; billing disabled, no charge).
- Gradio serve check: 2026-11-07, Gradio 5.50.0 — root HTTP 200 on
  127.0.0.1:7860 (79,455 bytes, page contains "Architects of Reality"),
  `/gradio_api/info` HTTP 200 with valid JSON (named endpoints present).
  Two earlier serve attempts on Gradio 5.4.0 failed (any gr.Chatbot
  crashed the API-info endpoint); the 5.50.0 upgrade fixed it, and this
  re-check confirms the fix. (Note: the app prints two Gradio 6.0
  deprecation warnings on startup — `theme` in Blocks, Chatbot
  `allow_tags` default — cosmetic only.)
