# Checklist — what Putt must do himself

Nothing has been created on GitHub yet. No repo, no push, no App
access granted. These are the exact remaining steps.

## Step 1 — Choose (in chat, one reply)

- [ ] Repo name: **`aethel-psi`** (proposed; GitHub search
      2026-11-07 found 0 repos with that name), or an alternative:
      `aethel-psi-synthesis`, `uaf-gw-pipeline`,
      `primordial-waveform`.
- [ ] Visibility: **public** (recommended — the value is reuse and
      citation; the paid tier is support, not code access) or
      private (if you want a private first pass — note a private
      repo cannot be cited by labs until it is public).

## Step 2 — Repository selection (GitHub App)

- [ ] Open the install link and choose which repositories Muse
      can access:
      https://github.com/apps/meta-muse-ai/installations/new
- [ ] Come back to the chat when done — GitHub may not return
      here automatically.
- [ ] Note: OAuth is already connected as **derrickdagod666**
      (verified 2026-11-07). Do NOT redo OAuth; only the
      repository selection is pending.

## Step 3 — Create the repo

Either (a) Putt creates it on github.com (New repository →
`aethel-psi`, public, MIT licence, no README initialisation —
the package has its own), or (b) Putt tells dot to create it
via the connector (the connector exposes `create_repository`;
that call asks for fresh approval, and the repo name +
visibility from Step 1 must be stated first).

## Step 4 — Push this package

- [ ] Copy the contents of `workspace/user/files/github-repo/`
      into the new repo (all files, including the hidden
      `.gitignore`).
- [ ] First commit message suggestion:
      `Initial commit: aethel-psi pipeline, validation suites v2-v5,
      ACT DR6 test, Qiskit demo (Third Revised Edition)`
- [ ] Verify on github.com that: README renders, no `.hdf5` /
      `.npz` / venv files were pushed, and no secrets appear
      (search the repo for `api_key` — expect zero hits).

## Step 5 — After the push (do not skip)

- [ ] Add the Zenodo/arXiv record link to the repo README once
      the timestamp lands (sibling package `arxiv-zenodo/`).
- [ ] Do NOT add ApothegmSovereignEngine, the VQE app, or the
      Aethelred Agent code until their source is located /
      sanitized (see `AUDIT.md`).
- [ ] Do NOT add the GW231123 precessing-PE result until that
      run actually finishes (it died in a VM reboot; checkpoint
      exists, resume untested).
