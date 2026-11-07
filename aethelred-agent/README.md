# aethelred-agent — BLOCKED, code intentionally NOT included

## What it is

The Aethelred Agent backend: TypeScript, Express + WebSocket server
(port 8787), Gemini 2.5 Pro for core logic, Gemma 3 270M for a
facial-expression controller, SQLite memory, local RAG (HNSWLib),
SSML voice adapter, Weather (OpenWeather) and News (NewsAPI) tools.
Source in Google Drive: `Aethelred Agent - Final Integrated Script`
(Google Doc, 2025-08-24) and `.txt` (15,045 bytes, 318 lines,
2025-08-27), file `agent-full.ts` v4.

Dependencies (from the file header):
```
npm i express cors body-parser uuid node-fetch ws sqlite3 sqlite @google/generative-ai hnswlib-node
npm i -D ts-node typescript @types/node @types/express @types/ws
```

## Why the code is NOT in this package

The Drive copy contains **hardcoded API keys** (Google, OpenWeather,
NewsAPI). Publishing it verbatim would leak live credentials. It
was deliberately excluded. A separate credentials document
(`Aethelred_Key`, Drive, 2025-08-08) was **not opened and not
copied**; it must never be committed.

## How to sanitize it before shipping

1. In `agent-full.ts`, replace every hardcoded key assignment with
   an environment-variable read, e.g.:
   ```ts
   const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY ?? "";
   const OPENWEATHER_API_KEY = process.env.OPENWEATHER_API_KEY ?? "";
   const NEWS_API_KEY = process.env.NEWS_API_KEY ?? "";
   if (!GOOGLE_API_KEY) throw new Error("GOOGLE_API_KEY is not set");
   ```
2. Copy `.env.example` (in this folder) to `.env` and fill in your
   own keys. `.env` is in the repo `.gitignore` — verify it stays
   untracked (`git status` must not list it).
3. **Rotate/re-issue the keys that were in the Drive copy** — assume
   any key that ever sat in a shared doc is compromised.
4. The script reads a `DOTS.txt` file at runtime and writes to an
   `agent_data/` directory (SQLite `memory.db`); both are gitignored.
   Provide a `DOTS.example.txt` with placeholder text, not the real one.
5. Verify locally before pushing:
   ```
   npm install
   npx tsc --noEmit
   PORT=8787 node --loader ts-node/esm agent-full.ts
   ```
   No run of this agent has been verified locally yet — record the
   result when you do.
6. Only then copy the sanitized `agent-full.ts`, `package.json`,
   and `tsconfig.json` into this folder.

## Status

- [ ] Keys moved to env vars
- [ ] Old keys rotated
- [ ] `tsc --noEmit` passes
- [ ] Local run verified
- [ ] Sanitized code copied here
