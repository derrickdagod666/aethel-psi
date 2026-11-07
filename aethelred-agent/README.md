# aethelred-agent — sanitized package (code NOT included)

## Status: BLOCKED — do not ship as code until Putt rotates the old keys

The Aethelred Agent backend exists in Google Drive, but the copy there
contains **hardcoded API keys** (Google, OpenWeather, NewsAPI). Publishing
it verbatim would leak live credentials. **No agent source code is in
this folder, and no key value has been copied anywhere in this repo.**
This folder holds only the sanitization package: this README,
`.env.example` (variable names only), and `SANITIZATION_CHECKLIST.md`.

**The one step nobody else can do is Putt's: rotate / re-issue every key
that was hardcoded in the Drive copy.** Assume any key that ever sat in a
shared doc is compromised. The agent cannot ship as code until that
rotation is done — see `SANITIZATION_CHECKLIST.md`, step 3.

## What the agent is

TypeScript backend, from the Drive file header and the earlier audit
(2026-11-07):

- Express + WebSocket server, **port 8787**
- **Gemini 2.5 Pro** for core logic; **Gemma 3 270M** for a
  facial-expression controller
- **SQLite** memory (`memory.db` in a runtime `agent_data/` directory)
- Local RAG via **HNSWLib**
- **SSML voice adapter**, plus **Weather (OpenWeather)** and
  **News (NewsAPI)** tools
- Reads a `DOTS.txt` file at runtime

Drive sources (read-only; neither was copied here):

- `Aethelred Agent - Final Integrated Script` (Google Doc, 2025-08-24)
- `Aethelred Agent - Final Integrated Script.txt`
  (15,045 bytes, 318 lines, 2025-08-27), file `agent-full.ts` v4
- A separate credentials document (`Aethelred_Key`, Drive, 2025-08-08)
  was **never opened and never copied**. It must never be committed.

Dependencies (from the file header):

```
npm i express cors body-parser uuid node-fetch ws sqlite3 sqlite @google/generative-ai hnswlib-node
npm i -D ts-node typescript @types/node @types/express @types/ws
```

## What is in this folder

| File | Contents |
| --- | --- |
| `README.md` | This file: what the agent is, and how to sanitize it |
| `.env.example` | Variable **names only** — no values. Verified 2026-11-07: 6 lines, every value under 10 characters (a real key is 32–39 chars), so no real key can be in it |
| `SANITIZATION_CHECKLIST.md` | Step-by-step checklist, including Putt's key-rotation step |

What is **deliberately absent**: `agent-full.ts`, `package.json`,
`tsconfig.json`, `DOTS.txt`, `memory.db`, `agent_data/`, `.env`.
All of those are gitignored or withheld (see `.gitignore` at repo root:
`.env`, `*.env`, `*.pem`, `*.key`, `Aethelred_Key*`, `agent_data/`,
`memory.db`, `*.db`, `node_modules/`, `dist/`).

## Environment variables

| Variable | Used for | Value comes from |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Gemini 2.5 Pro calls (`@google/generative-ai`) | Putt's Google AI Studio / Cloud key — a **freshly rotated** one |
| `OPENWEATHER_API_KEY` | Weather tool | Putt's OpenWeather key — a **freshly rotated** one |
| `NEWS_API_KEY` | News tool | Putt's NewsAPI key — a **freshly rotated** one |
| `PORT` | Server port | `8787` (default in `.env.example`) |

No other secrets are needed: Gemma 3 270M runs locally, SQLite and
HNSWLib are local files, and there is no other third-party account in
the header's dependency list. If sanitizing turns up another key
(for example a Hugging Face token for the Gemma download), add it to
`.env.example` as a name only and to this table — never paste a value.

## How to sanitize it before shipping

Do this on a local copy of the Drive file. Do not paste the Drive file
into a public repo, a gist, or a chat at any point.

1. **Replace every hardcoded key assignment with an environment-variable
   read.** The pattern, for each of the three keys:
   ```ts
   const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY ?? "";
   const OPENWEATHER_API_KEY = process.env.OPENWEATHER_API_KEY ?? "";
   const NEWS_API_KEY = process.env.NEWS_API_KEY ?? "";
   if (!GOOGLE_API_KEY) throw new Error("GOOGLE_API_KEY is not set");
   ```
   Then grep the sanitized file for the old key values and for
   `apiKey`, `api_key`, `sk-`, and `AIza` — every hit must be a
   variable name or a placeholder, never a value. Search the whole
   file, not just the top: keys can also hide in tool definitions,
   default arguments, and commented-out lines.
2. **Create the runtime files from examples, not from Drive.**
   Copy `.env.example` (this folder) to `.env` and fill in your own
   keys. `.env` is in the repo `.gitignore` — verify it stays
   untracked (`git status` must not list it). Provide a
   `DOTS.example.txt` with placeholder text, not the real `DOTS.txt`.
3. **Rotate / re-issue the old keys — Putt's step, not automatable
   here.** In Google AI Studio / Google Cloud, OpenWeather, and
   NewsAPI: revoke each key that was in the Drive copy and create a
   fresh one. Only the fresh keys go into `.env`. Do this even if the
   Drive copy is never published: the keys have sat in a document,
   and that alone is exposure.
4. **Add the missing project files.** The Drive copy is a single
   `agent-full.ts`. Ship it with a `package.json` (the dependencies
   above, pinned), a `tsconfig.json` (strict, `module: nodenext` or
   equivalent for the `ts-node/esm` loader), and this README.
5. **Verify locally before pushing:**
   ```
   npm install
   npx tsc --noEmit
   PORT=8787 node --loader ts-node/esm agent-full.ts
   ```
   Confirm the server starts, the WebSocket endpoint answers, and a
   request fails cleanly (not with a leaked key) when a key is unset.
   No run of this agent has been verified locally yet — record the
   result when you do, in `SANITIZATION_CHECKLIST.md`.
6. **Only then** copy the sanitized `agent-full.ts`, `package.json`,
   and `tsconfig.json` into this folder, and tick the checklist.

## Status

- [ ] Keys moved to env vars (sanitized `agent-full.ts` written)
- [ ] Old keys rotated by Putt (Google, OpenWeather, NewsAPI)
- [ ] `tsc --noEmit` passes
- [ ] Local run verified (server starts on 8787, fails cleanly without keys)
- [ ] Sanitized code copied here
- [ ] `.env` confirmed untracked (`git status` clean of it)

## History

- 2026-11-07: agent located in Drive; hardcoded keys found; code
  deliberately excluded from the repo package.
- 2026-11-07: sanitization package prepared (this README,
  `.env.example` verified names-only, `SANITIZATION_CHECKLIST.md`).
  No agent code copied, no key value copied, `Aethelred_Key` never
  opened. Full search record: `../CODE_HUNT_REPORT.md`.
