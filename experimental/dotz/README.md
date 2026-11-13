# dotz — Gemini Advanced Chat (experimental)

**Status: ARCHIVAL / EXPERIMENTAL.** This is an archived app export, committed as-is for preservation. It is not maintained, not tested as part of this repository's validation suites, and not part of the aethel-psi supporter tier.

## What it is

A single-page chat front end ("Gemini Advanced Chat") built in Google AI Studio and exported on 2025-08-22. Features:

- Chat against Google's `gemini-2.5-flash` model via the `@google/genai` SDK (loaded from esm.sh through an import map)
- Image attachment with preview
- Text-to-speech readback (browser SpeechSynthesis), on by default, toggle persisted in localStorage
- Optional Google Search grounding toggle
- Markdown rendering of model replies via `marked`

Files: `index.html` (page shell + import map), `index.tsx` (all app logic, 722 lines), `index.css`, `metadata.json` (AI Studio app metadata: name "dotz", description " chat app ").

## Provenance

Exported from Google AI Studio (files dated 2025-08-22), uploaded by the repository owner on 2026-11-13 as `dotz.bin` (a zip under a .bin name) and committed here byte-identical to that upload.

## Running it

The app reads its Gemini API key from the environment at runtime — `new GoogleGenAI({ apiKey: process.env.API_KEY })` — in the AI Studio hosting pattern. No key is stored in this code, and a security scan on 2026-11-13 found no embedded credentials. To run it outside AI Studio you would need to serve the files with a bundler/dev server that supplies `process.env.API_KEY`; it will not work as a plain static page without that wiring.

## What this is not

- It is **not** the ApothegmSovereignEngine and **not** the VQE drug-discovery app; those sources remain unlocated as of 2026-11-13.
- It makes no scientific claims and is unrelated to the UAF/Aethel-Psi validation results elsewhere in this repository.
- Header in `index.tsx` carries Google's Apache-2.0 copyright notice from the original sample code.
