# AUDIT — Architects of Reality: Cognitive Companion (original, as pasted)

Audited 2026-11-07 by reading `ARCHITECTS_original.py` in full and running
`python3 -m py_compile` on it, plus checking for the files/services it assumes.
No secrets were found in the pasted code; none are reproduced here.

## Headline

The original **compiles cleanly** (`py_compile` OK) — unlike the GENEIE
original, the syntax is fine. It does **not run usefully as-is**: every
external dependency it needs is missing or assumed, and two of its headline
features (the telemetry gauges and, in practice, the Stripe upgrade) are
cosmetic or dead. The RAG design itself — embed → FAISS search → system
prompt with context → stream from an OpenAI-compatible local endpoint — is
sound, and v2 keeps it.

## 1. Verified issues (each checked, not assumed)

| # | Issue | How verified |
|---|---|---|
| 1 | **FAISS index files do not exist.** `syntropic.index` and `syntropic_metadata.json` are loaded if present, else the engine prints a warning and sets `self.index = None`. | Searched `~/workspace/user/files/` for `*syntropic*` — no hits. No `build_syntropic_index.py` exists anywhere in the workspace either, despite the warning telling the user to run it. |
| 2 | **No corpus is included.** The index is supposed to hold "The Syntropic Primer", but no primer text, no PDF, and no build script were pasted. | The pasted material is only the frontend file. |
| 3 | **LM Studio is assumed running** at `http://localhost:1234/v1`, and the `SentenceTransformer("all-mpnet-base-v2")` model download (~420 MB) is assumed available. | `curl http://localhost:1234/v1/models` on this machine: connection refused — LM Studio is not running. No Hugging Face cache exists locally. Every query would hit the `except` branch and yield only the connection-error message. |
| 4 | **Stripe backend does not exist.** `get_checkout_url` POSTs to `http://localhost:8000/create-checkout-session` — no backend code was pasted. | `curl http://localhost:8000/` : connection refused. On failure the function prints an error, returns `["#"]`, and the JS redirect sends the user to `"#"` — the upgrade button does nothing visible. |
| 5 | **Hardcoded demo email** `demo@architectsofreality.com` is sent as `user_email` in the checkout call. | Read in the code (line: `json={"tier": tier_id, "user_email": "demo@architectsofreality.com"}`). No real user identity is ever used. |
| 6 | **Tier logic is string-matching on the dropdown label** (`"Community" in tier`, `"Researcher" in tier`, else Enterprise). | Read in the code. Renaming a label silently changes the tier behaviour; the `else` branch also catches any unexpected value as Enterprise. |
| 7 | **History format is Gradio 4 style.** `history + [[user_message, None]]` and iterating `for human_msg, ai_msg in history` is the old tuple format. Gradio 5's `gr.Chatbot` defaults to `type="messages"` (list of `{"role", "content"}` dicts). | Read in the code; confirmed against the installed Gradio 5.4.0 in the v2 venv (see README). |
| 8 | **`request.query_params`** in `handle_page_load` and `app.load(...)` are used without a fallback; if the Gradio version's request object differs, the page-load banner breaks. | Read in the code. Kept in v2 (verified working in the serve check) but noted as a version-sensitive call. |
| 9 | **No requirements file, no pinned versions, no README, no tests** in the pasted material. | Only the single frontend file was pasted. |
| 10 | **Unused imports**: `numpy` is imported but never used. | Read in the code. |

## 2. Fake / cosmetic — labelled as such

- **"Cognitive Entropy" gauge**: starts at `0.95` and drops `0.005` per
  streamed token (`self.entropy = max(0.1, self.entropy - 0.005)`). It
  measures nothing — it is a countdown tied to token count, and it resets
  only on restart. It is not entropy, uncertainty, or any property of the
  model or the answer.
- **"Syntropic Resonance" score**: starts at `0.0` and adds `+1.5` per
  message (`current_resonance + 1.5`). It measures nothing — it is a
  message counter times 1.5. It is not resonance, relevance, or any
  property of the retrieval.
- Both gauges were displayed as if they were live telemetry of a "Q-State".
  v2 replaces them with measured values (top L2 distance, chunks retrieved,
  latency, words streamed) and labels them as such.

## 3. What is real (kept in v2)

- The RAG pipeline design: embed the query → FAISS top-k search → build a
  system prompt with the retrieved context → stream completions from an
  OpenAI-compatible local endpoint. This is a sound, standard design.
- Tier-dependent synthesis depth (different instructions per tier) is a
  legitimate product feature; v2 keeps it, keyed on tier ids instead of
  label substrings.
- The Stripe success/cancelled banner handling via URL query params is a
  legitimate pattern; v2 keeps it (billing itself stays disabled).

## 4. What v2 changes (summary — full elaboration in the v2 code)

1. Index builder added (`build_index.py`): builds `syntropic.index` +
   `syntropic_metadata.json` from Putt's own papers (Third Revised Edition
   PDF + UAF paper PDF), with a labelled fallback corpus if extraction fails.
2. LM Studio is probed at startup; if unreachable, a clearly-labelled
   extractive demo responder answers only from retrieved passages (with
   sources) and the UI says so — never pretending to be the LLM.
3. Fake gauges replaced with measured telemetry.
4. Tiers are a dict of ids; dropdown carries tier ids as values.
5. Gradio 5 messages-format history.
6. Billing disabled by default (`billing.py` + `backend_stub.py`): no
   Stripe account, product, price, or charge; the upgrade button shows an
   honest "billing not configured" message. No hardcoded demo email.
7. `numpy` unused import dropped; requirements pinned and tested;
   headless test script added.
