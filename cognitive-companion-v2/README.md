# Architects of Reality: Cognitive Companion — v2

A working Gradio app: a Quantum Cognitive Engine (QCE) that answers questions
about Putt's Aethel-Psi framework by retrieving passages from his own papers
(FAISS + embeddings) and synthesising with a local LLM (LM Studio) — or, when
no LLM is running, answering extractively from the retrieved passages only,
and saying so plainly.

## What is real, and what is not

- **Real:** FAISS retrieval over Putt's papers (Third Revised Edition PDF +
  UAF paper PDF), chunked with source citations; LM Studio streaming when
  available; measured telemetry (top L2 distance, chunks retrieved, latency,
  words streamed).
- **Demo mode (labelled in the UI):** if LM Studio is not running, answers
  are assembled ONLY from retrieved passages. That is extractive, not
  generative — the app says so on screen and in every demo answer.
- **Not real / not wired up:** billing. There is no Stripe account, product,
  price, or checkout session. The upgrade button shows "billing not
  configured". See `billing.py` for what enabling real Stripe would take.
- **Corpus:** his papers, not "The Syntropic Primer" as a separate text —
  no primer file was in the pasted material. The index is built from the two
  PDFs; if extraction ever yields nothing, a labelled fallback corpus of
  summaries is used instead (see `corpus_fallback.py`).

## Run it

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python test_headless.py     # builds the index, runs retrieval + demo response
./venv/bin/python app.py              # serves on http://127.0.0.1:7860
```

First run downloads the `all-mpnet-base-v2` embedding model (~420 MB, once)
and builds the index (~1–2 min). First app start is slow (~1 min) because
the embedding model loads.

To use the full LLM mode: start LM Studio, load a model, start its local
server (default http://localhost:1234/v1), then restart the app. The banner
will read "LLM mode". Override with `AOR_LLM_URL` / `AOR_LLM_MODEL`.

## Files

| File | What it is |
|---|---|
| `app.py` | Gradio UI (chat, measured telemetry, tier dropdown, upgrade button) |
| `qce.py` | Engine: retrieval, LM Studio probe, demo responder, telemetry |
| `build_index.py` | Builds `syntropic.index` + `syntropic_metadata.json` from the PDFs |
| `corpus_fallback.py` | Labelled summary corpus, used only if PDF extraction fails |
| `billing.py` | Billing, disabled by default (no keys, no charges) |
| `backend_stub.py` | Optional FastAPI stub; `/create-checkout-session` returns 501 |
| `test_headless.py` | Headless test: index build + retrieval + demo response + billing check |
| `requirements.txt` | Pinned versions actually tested with |
| `CHANGELOG.md` | What changed vs the original, with reasons |

## Honest limits

- Local only. No hosted deployment, no accounts, no real billing.
- Demo responder is extractive, not generative — it cannot answer questions
  whose answer is not in the retrieved passages.
- Retrieval quality depends on the embedding model and the chunking; the
  corpus is two papers, so coverage is limited to what they contain.
- The tier prices ($49/$299/$999) are display-only labels from the original;
  no payment is possible in this build.
- Slow first start (embedding model load); index rebuild needed if the
  papers change (`python build_index.py --force`).
