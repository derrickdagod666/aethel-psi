# Aethelred Agent — sanitization checklist

Work through this in order. Nothing ships until every box is ticked.
Steps marked **PUTT** can only be done by Putt (the user); the rest can
be done by whoever holds the sanitized copy.

## 1. Isolate the source

- [ ] Copy `Aethelred Agent - Final Integrated Script.txt` (Drive,
  2025-08-27, `agent-full.ts` v4) to a **private** local folder only.
      Never paste it into a public repo, gist, or chat.
- [ ] Confirm the copy is the same file: 15,045 bytes, 318 lines.
- [ ] Do **not** open `Aethelred_Key` (Drive, 2025-08-08). It is a
      credentials document and must never be committed or copied.

## 2. Remove the hardcoded keys

- [ ] Find every hardcoded key assignment in `agent-full.ts`
      (Google, OpenWeather, NewsAPI — and anything else the file
      turns up).
- [ ] Replace each with an environment-variable read (pattern in
      `README.md` § "How to sanitize it", step 1), failing loudly
      when the variable is unset.
- [ ] Grep the whole sanitized file for the old key values and for
      `apiKey`, `api_key`, `sk-`, `AIza`: every hit must be a variable
      name or placeholder, never a value. Check tool definitions,
      default arguments, and commented-out lines too.
- [ ] Add any newly found secret to `.env.example` as a **name only**.

## 3. Rotate the old keys — **PUTT**

- [ ] **PUTT:** In Google AI Studio / Google Cloud, revoke the Google
      key that was in the Drive copy and create a fresh one.
- [ ] **PUTT:** In OpenWeather, revoke the old key and create a fresh
      one.
- [ ] **PUTT:** In NewsAPI, revoke the old key and create a fresh one.
- [ ] **PUTT:** Put **only the fresh keys** into the local `.env`
      (copied from `.env.example`). The old keys are dead after this
      and must never be typed anywhere again.
- [ ] **PUTT:** If `Aethelred_Key` holds any other credential, rotate
      that too, and confirm it stays out of the repo.

Why this cannot be skipped: the keys sat in a Drive document. Even if
the document was never shared, that is exposure — assume compromise
and re-issue. The agent cannot ship as code until this step is done.

## 4. Rebuild the project files

- [ ] Write `package.json` with the dependencies from `README.md`,
      versions pinned.
- [ ] Write `tsconfig.json` (strict; `module: nodenext` or equivalent
      for the `ts-node/esm` loader).
- [ ] Create `DOTS.example.txt` with placeholder text (not the real
      `DOTS.txt`).
- [ ] Copy `.env.example` to `.env` and fill in the **fresh** keys.
- [ ] Confirm `.env`, `agent_data/`, `memory.db`, `*.db`,
      `DOTS.txt`, `node_modules/`, `dist/` are all gitignored
      (`git status` must not list `.env`).

## 5. Verify locally

- [ ] `npm install` succeeds.
- [ ] `npx tsc --noEmit` passes with no errors.
- [ ] `PORT=8787 node --loader ts-node/esm agent-full.ts` starts the
      server; WebSocket endpoint answers.
- [ ] With a key unset, the server fails cleanly with "not set" —
      and no key value appears in any log or error output.
- [ ] Record the verification result and date here:
      _verified: ____________________ (date, by whom, result)_

## 6. Publish

- [ ] Copy only the sanitized `agent-full.ts`, `package.json`,
      `tsconfig.json`, `README.md`, `.env.example`, and this checklist
      into `aethelred-agent/`.
- [ ] Final grep of the folder for key material before `git add`:
      no `AIza`, no 32-hex OpenWeather/NewsAPI-style token, no `.env`.
- [ ] `git status` shows no `.env`, no `agent_data/`, no `*.db`.
- [ ] Push only after all boxes above are ticked.

## Current state (2026-11-07)

- [x] Sanitization package prepared (`README.md`, `.env.example`
      verified names-only, this checklist)
- [ ] Steps 1–2: not started (no agent code copied)
- [ ] Step 3 (rotation): **blocked on Putt**
- [ ] Steps 4–6: not started
