# Code hunt report — ApothegmSovereignEngine, VQE drug app, Aethelred Agent

Date: 2026-11-07 (searches run 00:20–00:28 CST, America/Chicago)
Repo package: `~/workspace/user/files/github-repo/`
(also public repo https://github.com/derrickdagod666/aethel-psi — **nothing
was pushed**; this is a local-package update only)
Scope: read-only on Google Drive throughout. Nothing in Drive was
edited, deleted, shared, or moved. No API key, token, password, or
secret value was copied anywhere; the credentials document was never
opened (see §5).

## Verdicts

| Item | Verdict | One line |
| --- | --- | --- |
| ApothegmSovereignEngine (React/TypeScript, 3Dmol.js) | **NOT FOUND** | No file by that name, and no file containing `Apothegm`, `SovereignEngine`, `3Dmol`, or `3Dmol.js`, in Drive (1,097 items) or the local workspace |
| Dash/Plotly/PennyLane VQE drug-discovery app | **NOT FOUND** | No `VQE` code, no Dash import, no Plotly app, no drug-discovery code anywhere searched; only prose mentions |
| Aethelred Agent (TypeScript backend, port 8787) | **BLOCKED** | Code exists in Drive but contains hardcoded API keys; deliberately not copied. Sanitization package prepared (§5). Ships only after Putt rotates the old keys |
| Related code found in the same sweep | **FOUND, NOT COPIED** | 4 nearby items audited (§4); none is either missing app, so none was copied into the repo |

The repo README's statement stands and is now confirmed by a second,
harder search: neither missing app can ship until its source is located.

## 1. Where and how I searched

### 1.1 Drive inventory (name search, all 1,097 items)

Source: `~/workspace/drive-inventory/google-drive-inventory.csv`
(1,099 lines incl. header = 1,097 items, 162 folders, inventoried
2026-11-06). Name-field matches (case-insensitive, whole inventory):

- `apothegm`: **0** · `sovereign`: **0** · `VQE`: **0** ·
  `PennyLane`: **0** · `Dash`: **0** · `Plotly`: **0** · `3Dmol` /
  `3dmol`: **0** · `React`: **0** · `TypeScript`: **0** ·
  `molecule` / `molecular`: **0**
- `drug`: **1** — `Drug Test Consent` (Google Doc, 2023-11-17), a
  consent form, not code
- `Engine`: **0** by name (live Drive check, §1.2)

Code-like names in the whole inventory: 5 Python files, 2 Colab
notebooks, ~55 `.txt` files (mostly `code (N).txt` exports in
`Google AI Studio`), 4 ZIPs, 1 JSON, 0 `.ts` / `.tsx` / `.js` /
`.jsx` / `.html` files. **No `.tsx`, `.ts`, or `package.json`
exists anywhere in Drive by name.**

### 1.2 Live Drive full-text search (name AND content)

Queries run against the live account (Drive `files list`,
`trashed=false` unless noted). `fullText` covers Google Docs, PDFs,
and text files:

| Query | Hits |
| --- | --- |
| `fullText contains 'Apothegm'` | **0** |
| `fullText contains 'SovereignEngine'` | **0** |
| `fullText contains 'Apothem'` (variant spelling) | **0** |
| `fullText contains '3Dmol'` / `'3Dmol.js'` | **0** / **0** |
| `fullText contains 'PennyLane'` | 2 — `code.py` and `quantum_tests_harmonic_full.py` only (both already in the repo's `drive-code/`) |
| `fullText contains 'VQE'` | 2 — both copies of `Aethel-Psi Paradigm: Nobel Path` (Google Doc, 2025-09-26); prose only, see §3 |
| `fullText contains 'Sovereign'` | 1 pair — `Google Keep Document` (2025-07-12) + its `.txt` twin; prose only ("Sovereignty vs. Power", "sovereign Architect of Reality") |
| `fullText contains 'Sovereign Engine'` | 4 — the Keep doc above + 2 copies of `Defining Societal Progress Analysis` (prose: "technological sovereignty", US–China rivalry); both words occur separately, never as an app name |
| `fullText contains 'drug discovery'` | 20 — prose only (theory docs, `Web` doc/zip, `Nobel Prize Worthy Theory`); no code |
| `fullText contains 'molecule'` / `'molecular dynamics'` | prose only (theory docs, `Web` doc/zip, `Part 1` docs) |
| `fullText contains 'Variational Quantum'` | 20 — theory/prose docs (`uaf.txt`, `Beyond.txt`, `Lvl_1_.txt`, Keep docs, Nobel Path); no code file |
| `name contains 'Apotheg'` / `'Engine'` / `'Sovereign'` / `'VQE'` / `'Drug'` | **0** / **0** / **0** / **0** / 1 (`Drug Test Consent`) |
| Trashed copies: `name/fullText contains 'Apothegm'/'Sovereign'/'VQE'`, `trashed=true` | **0** in all four queries |
| Shared drives (`drives list`) | none — the account has no shared drives |

### 1.3 Content grep of every downloadable code/text file (the hard search)

The name index cannot see inside the ~50 unnamed `code (N).txt`
exports, so all of them were downloaded read-only and grepped
directly — **75 files** from `Google AI Studio` + `Colab Notebooks`
(all `.py`, `.txt`, `.json`, `.zip`) **plus 44 root/other-folder
`.txt` files**, 119 files total, zero download failures:

- `apothegm` / `sovereignengine`: **0 hits in all 119 files**
- `sovereign` (any case): 1 file — `Google Keep Document.txt`
  (2025-07-12, prose, §3)
- `vqe` / `variational quantum eigensolver`: **0 hits in all 119 files**
- `import dash` / `from dash` / `import plotly` / `from plotly` /
  `3dmol`: **0 hits in all 119 files**
- `pennylane`: 2 files — the same two already in `drive-code/`
- `react`: 1 code file — the Living AI Chat Widget (§4.4), **not**
  Apothegm; other hits are the English word "reaction" in prose
- All apparent `dash` / `plotly` substring hits were LaTeX
  `\addplot[dashed, ...]` / `\draw[..., dashed]` lines in theory
  documents, not code. The single genuine `plotly` hit in any code
  file is a `!pip install ... plotly ...` line in Colab `code (3).py`
  (§4.2) — an install line, not an app.

### 1.4 ZIP archives (listed, not just named)

| ZIP | Contents |
| --- | --- |
| `Web.zip` (2024-05-18, root) | 1 file: `Web.html` (74,168 bytes) — a Google-Doc export of a "Snoritech" website mockup; its "drug" hits are prose ("Quantum computing can simulate drug interactions… accelerating the drug discovery process"). No code, no Apothegm/VQE |
| `Health Connect.zip` (2026-04-22, root) | 1 file: `health_connect_export.db` (1,355,776 bytes) — a Health Connect database export. Not code |
| `DerrickReport (1) (1).zip` × 2 (2025-04-16, Google AI Studio) | Each holds one file only: `DerrickReport (1) (1).pdf` (281,095 bytes). Duplicates of each other |

### 1.5 Colab notebooks (downloaded and read)

| Notebook | Size | Contents |
| --- | --- | --- |
| `Untitled0.ipynb` (2025-03-02) | 309 bytes | Empty — zero cells |
| `quantum AI assistant` (2025-02-27) | 347 bytes | One empty code cell |
| `Untitled1.ipynb` (2025-09-09) | 4,862 bytes | DeepSeek-Math-13B / Mathstral-7B download + Gradio math Q&A (5 cells). No Apothegm/VQE/drug content. One cell has `use_auth_token=True` — a flag, not a token value; no secret value is in the file |

### 1.6 Google Docs read in full (text extracted via the Docs API)

| Doc | Size | Finding |
| --- | --- | --- |
| `Aethel-Psi Paradigm: Nobel Path` × 2 (2025-09-26) | 22,522 chars each, identical | Only `VQE` hit in Drive. Prose claim: waveform parameter θΨ ≈ π was "discovered by a Variational Quantum Eigensolver (VQE)". No VQE code, no Dash/Plotly, no drug app |
| `Defining Societal Progress Analysis` × 2 (2025-08-05) | 61,101 chars each, identical | Only "sovereign" hit is "technological sovereignty" (US–China rivalry prose). Not code |
| `ai_studio_code (3)` (2025-08-27) | 14,705 chars | LaTeX paper preamble (`\documentclass…`, `\usepackage{…}`), not code. Zero hits for every search term |

### 1.7 AI Studio apps (name only — content not searchable via Drive)

7 apps in the `Google AI Studio` folder: `ChatterBots` (×2),
`Copy of Gemini OS`, `Core thought`, `dotz`, `Gemini OS`,
`Living AI Chat Widget`. None is named for Apothegm or VQE. Their
source lives in Google AI Studio, not in Drive files, so Drive search
cannot see inside them — a remaining gap (§6).

### 1.8 Local workspace

- `find ~/workspace/user` for `*.tsx`, `*.ts`, `package.json`,
  `*3dmol*`, `*apothegm*`, `*sovereign*`, `*vqe*`: **no source
  files**. No TypeScript or React project exists under
  `~/workspace/user`.
- `grep -r "ApothegmSovereignEngine" ~/workspace/user`: only mentions
  in prior deliverables (`uaf_unified_map_draft.md`,
  `monetization_roadmap.md`, repo `README.md` / `AUDIT.md` /
  `PROPOSED_STRUCTURE.md` / `CHECKLIST.md` / `CHANGELOG.md` /
  `RELEASE_NOTES_v0.1.0.md`). No source.
- `grep -r "VQE drug" ~/workspace/user`: same — mentions only.

### 1.9 Google Keep exports in Drive (34 items)

19 Google Docs, 2 PDFs, 10 `.txt` exports, 1 `.docx`, 1 PDF, 1 Doc
under `d.irons`, all named `Google Keep Document*`. All 10 `.txt`
exports were downloaded and grepped (§1.3); the 19 Docs are covered
by the full-text searches (§1.2). None contains the missing code.
Note the gap: Keep had 44 active notes on 2026-11-06, but Drive
holds only ~10 Keep `.txt` exports — the live Keep app itself was
not searched in this run (§6).

## 2. Verdict detail — ApothegmSovereignEngine: NOT FOUND

- No file of that name (or any variant: `Apothegm`, `Apothem`,
  `SovereignEngine`, `Sovereign Engine`) exists in Drive by name,
  in full text, inside any downloaded file, inside any ZIP, or in
  any Colab notebook.
- No `3Dmol` / `3Dmol.js` string occurs anywhere searched.
- No React/TypeScript source of any kind exists in Drive or
  `~/workspace/user` — except the one unrelated widget in §4.4.
- The only places the name appears are prose: the 2026-11-06 Keep
  readout (the apps are *described* there as code the user has) and
  the repo's own README/AUDIT, which repeat that description.

## 3. Verdict detail — VQE drug-discovery app: NOT FOUND

- No file contains `VQE` as code. The only `VQE` hits are the prose
  claim in `Nobel Path` (§1.6).
- No file imports Dash; no Dash app exists anywhere searched. The
  only `plotly` occurrence in code is an install line in a Colab
  notebook (§4.2).
- `PennyLane` occurs only in the two files already in the repo's
  `drive-code/` (`code.py`, 32,303 bytes, 642 lines;
  `quantum_tests_harmonic_full.py`, 3,480 bytes, 68 lines — md5-verified
  identical to the repo copies). Neither is a Dash app, neither does
  VQE, neither touches drug discovery.
- Every `drug` / `molecule` / `molecular dynamics` hit is prose
  (theory text about quantum computing accelerating drug discovery),
  not code, not data, not an app.

## 4. Related code found in the same sweep — audited, NOT copied

These are real code files the earlier inventory had not catalogued.
None is either missing app, so none was copied into the repo (copying
unrelated fragments would muddy it). Exact Drive locations are given
so Putt can ask for any of them to be added.

### 4.1 `code (2).py` — quantum-system framework + Gradio assistant (×2 copies)

- Drive: `Google AI Studio`, two identical copies, 26,064 bytes each
  (2025-07-10 08:05 and 08:29). Colab notebook export
  (`#@title` cells): abstract `QuantumSystem`, harmonic-oscillator
  Hamiltonian (finite difference, Ψ coupling, A = 54, ω = 2π/945),
  Gemma text/voice assistant behind a Gradio interface.
- Audit: imports numpy, scipy, qutip, gradio, transformers/torch,
  gTTS. **Does not compile as a plain `.py`** — `IndentationError:
  unindent does not match any outer indentation level`, line 333
  (mixed Colab-cell indentation in the `process_input` fallback).
  Not runnable as-is; needs reconstruction. No VQE, no Dash, no
  drug content. Secrets scan: clean (no key/token/password value).

### 4.2 `code (3).py` — same framework, fuller version

- Drive: `Colab Notebooks`, 30,182 bytes (2025-03-02); byte-identical
  to `Colab Notebooks / code_converted.txt`. Colab export:
  `QuantumSystem` / `HarmonicOscillator` / `psi_waveform`, Plotly
  probability-density plots, Gradio interface, Gemma + Qiskit cells.
- Audit: its line 2 (`!pip install … plotly …`) is the only genuine
  `plotly` hit in any Drive code file. **Does not run as a plain
  `.py`** (Colab `!pip` magic on line 2). Not a Dash app, not VQE,
  no drug content. Secrets scan: clean.

### 4.3 `enhanced-ai-assistant.py` — Streamlit AI Coding Assistant

- Drive: `Google AI Studio`, 8,798 bytes (2025-07-10). Streamlit
  "AI Coding Assistant": code generation with GPT-Neo/T5/RoBERTa,
  GitHub repo search, SQLite cache (`coding_assistant.db`).
- Audit: **compiles clean** (`py_compile` OK). Takes its GitHub token
  as a constructor argument (`AICodingAssistant(github_token)`),
  not hardcoded. No secret value in the file. Heavy deps
  (torch, transformers, streamlit, PyGithub). Not Apothegm, not VQE.

### 4.4 "Living AI" Chat Widget — single-file React (TypeScript)

- Drive: root `Google Keep Document.txt`, 2025-08-23, 626 lines,
  25,498 bytes. `import React, { … } from 'react'` — a complete
  single-file React/TS chat widget ("Living AI", 7 upgrades: memory,
  voice prosody, typing/streaming UI, weather/news/stocks/calendar/gif
  tools, personality selector). Matches the AI Studio app of the same
  name (2025-08-23).
- Audit: the **only** React code found in Drive — but it is the
  Living AI widget, not ApothegmSovereignEngine: no `Apothegm`,
  no `3Dmol`, no molecule/drug/VQE content anywhere in it. Secrets
  scan: clean. Not copied — wrong app.

### 4.5 `Untitled1.ipynb` — DeepSeek-Math Gradio notebook

- See §1.5. Math-model demo, not either missing app. No secret value
  (only the `use_auth_token=True` flag).

## 5. Aethelred Agent: BLOCKED — sanitization package prepared

### What exists (Drive, read-only, not copied)

- `Aethelred Agent - Final Integrated Script` (Google Doc,
  2025-08-24) and `.txt` (15,045 bytes, 318 lines, 2025-08-27),
  `agent-full.ts` v4: TypeScript, Express + WebSocket (port 8787),
  Gemini 2.5 Pro, Gemma 3 270M facial-expression controller, SQLite
  memory, local HNSWLib RAG, SSML voice adapter, OpenWeather and
  NewsAPI tools. Reads `DOTS.txt` at runtime; writes `agent_data/`.
- `Aethelred_Key` (Google Doc, 2025-08-08): a credentials document.
  **Never opened, never copied, never to be committed.**

### Why it is blocked

The Drive copy contains hardcoded API keys (Google, OpenWeather,
NewsAPI). Publishing it verbatim would leak live credentials. The
code was deliberately not copied into the repo package, and no key
value was copied anywhere.

### What was delivered in `aethelred-agent/` (this run)

- `README.md` — full sanitization guide: what the agent is, env-var
  table, exact code pattern for moving keys to env vars, local
  verification commands, status list.
- `.env.example` — already present from the earlier packaging; **verified
  2026-11-07 names-only** (6 lines; every value under 10 characters,
  no spaces — a real key is 32–39 chars, so no real key can be in
  it). Variable names: `GOOGLE_API_KEY`, `OPENWEATHER_API_KEY`,
  `NEWS_API_KEY`, `PORT=8787`. Left byte-for-byte unchanged.
- `SANITIZATION_CHECKLIST.md` — step-by-step checklist (isolate →
  de-hardcode → rotate → rebuild → verify → publish).

### What remains blocked on Putt

1. **Rotate / re-issue all three old keys** (Google, OpenWeather,
   NewsAPI) — Putt's step; assume any key that sat in a Drive doc is
   compromised. Only fresh keys go into `.env`. The agent **cannot
   ship as code until this is done**.
2. Sanitize `agent-full.ts` (move keys to env vars) and verify locally
   (`npm install`, `npx tsc --noEmit`, run on 8787) — needs the rotated
   keys first.
3. Only then copy the sanitized `agent-full.ts`, `package.json`, and
   `tsconfig.json` into `aethelred-agent/`.

## 6. What is still unsearched / open

- **Google Keep itself (live app).** The two apps are described as
  code *in Keep notes*, but Drive holds only ~10 Keep `.txt` exports
  of 44 active notes. The remaining Keep notes were not opened in
  this run (no browser access in this session). If the code exists
  anywhere, Keep is the prime suspect — Putt (or a browser session)
  should open Keep and search the notes for `Apothegm` and `VQE`.
- **Google AI Studio apps' internals.** The 7 apps (§1.7) cannot be
  content-searched via Drive; Putt would need to open AI Studio and
  inspect each app's code, especially `Core thought`, `dotz`, and
  `Gemini OS`, whose purpose is not evident from the name.
- **Putt's own machine / other storage.** The code may live only
  locally (a React project folder, a Dash project folder) or in a
  GitHub repo / Colab runtime that was never saved to Drive.
- The related files in §4 can be added to the repo on request; they
  were left out deliberately because they are not the missing apps.

## 7. Method note

- Drive was searched read-only: `files list` (name + fullText),
  `files get` (metadata + `alt: media` downloads of non-Google-native
  files only), `docs documents get` (text extraction for 5 Docs).
  Nothing was edited, deleted, shared, or moved.
- Working copies downloaded for grepping were held under
  `~/workspace/codehunt-scratch/` and **deleted at the end of this
  run** (they included a bulk copy of the Drive `.txt` set, which
  contains the agent file with keys). This report is the surviving
  record; no secret value is recorded in it.
- Secret checks: regex scans for `api_key = "…"`, `sk-…`,
  `AIza…`, `password = "…"`, `token = "…"` across all downloaded
  `.py` files and the widget file — no key/token/password *value*
  was found in any of them (the agent file itself was excluded from
  content reads by design; its keys are known to exist from the
  earlier audit, not re-extracted here).
