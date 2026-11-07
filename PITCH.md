# Paid tier / consulting — outline only

No pricing promises, no sale. This is what labs and groups would
plausibly pay for, and what would have to exist first.

## The honest pitch

"We built a Bilby waveform plugin and ran it on five real LIGO/Virgo
events and real ACT DR6 data, with every null published. γ = 0.4 is
excluded; the bounded version is γ ≲ 0.08. Hire us to run the same
honest pipeline on your events / your model."

The product is the *pipeline and the discipline* (pre-registered
designs, fixed bugs documented, FAILs published), not a validated
theory. Never pitch γ = 0.4, a Nobel claim, or the unverified
">30σ" / "10/10 PASS" claims from Drive.

## Tier 1 — Open core (free, public repo)

- Everything in `aethel-psi`: waveform plugin, Gate 3 scripts,
  validation suites, ACT test, Qiskit demo. MIT licence.
- Value: citation, reputation, inbound consulting leads.

## Tier 2 — GW data-analysis plugin / support (paid)

What a lab would pay for:
- **UAF waveform as a maintained Bilby plugin**: pip-installable
  package (today it is loose scripts), versioned releases,
  changelog, CI running `test_qiskit.py` + the waveform modifier
  check on every commit.
- **Custom-dip analyses**: run the same profile/PE machinery with
  *their* waveform modification (different dip shape, different
  f_merger law) on *their* event list, pre-registered, with a
  written report (PASS/FAIL/OPEN per check, as in v5/v6).
- **Stacking as a service**: add events to the stack on request,
  with the same per-event pre-registration.
- **Support**: help integrating the plugin into a group's existing
  Bilby workflow.

What has to exist first: the pip package + CI + a versioned
release (v0.1.0). **Done 2026-11-07** (see `CHANGELOG.md` and
`PAID_TIER_PACKAGING.md`): `pyproject.toml` + `src/aethel_psi/`
(pip-installable, Bilby 2.8.2 / Qiskit 2.6.2 pinned, clean-install
tested in a fresh venv), CI at `.github/workflows/ci.yml`
(waveform modifier check + Qiskit stabilizer test on every push),
and v0.1.0 release files (`CHANGELOG.md`,
`RELEASE_NOTES_v0.1.0.md`). The GitHub release itself still has to
be created on the web (tag v0.1.0) — the connector cannot tag.
Tier 2 may be offered once that release exists; no prices are set
here and no sale has been made.

## Tier 3 — Consulting / collaboration (paid, scoped)

- **Precessing-event PE**: the GW231123 full precessing PE died
  in two VM reboots and was resumed twice from its checkpoint
  (latest 16279it, dlogz 6.92, target 0.1, 2026-11-07 — still
  running, no posterior/Bayes factor yet); the spin-marginalised
  GW200129 PE was launched 2026-11-07 with pre-registered priors
  and is also running. Finish both and report the posteriors +
  Bayes factors. See `PAID_TIER_REPORT.md`.
- **ACT joint analyses**: extend the TT+TE+EE ripple test to new
  templates / datasets on request.
- **Model-agnostic null testing**: apply the validation-suite
  discipline (falsifiable checks, pre-registration, published
  nulls) to another group's model.

Pricing: not set here. Comparable GW/quantum-software consulting
is scoped per project; any figure quoted to a lab must be a
written scope with deliverables (report + code + reproducibility
note), never a valuation of the theory.

## What is NOT for sale

- The theory itself (not patentable, $0 market value as a paper).
- γ = 0.4 as a result (excluded).
- Priority/timestamp (arXiv/Zenodo — free, already being done
  separately; see the sibling `arxiv-zenodo` package).
- Any application (QCI, RG-Fixed Point Intelligence, Holographic
  Data Compression, Phase-Vocoder) without a built, benchmarked
  prototype. No prototype, no patent, no valuation.

## Next steps to make Tier 2 real

1. Create the repo (see `CHECKLIST.md`). — Done, live at
   https://github.com/derrickdagod666/aethel-psi (first commit
   9803bd4b, PDF commit cbb16405).
2. Package `gw-pipeline/` as pip-installable (`pyproject.toml`,
   `aethel_psi` importable package, not loose scripts). — Done
   2026-11-07, v0.1.0; clean-install tested in a fresh venv.
3. Add CI (GitHub Actions): waveform modifier check + Qiskit
   test on every push. — Done 2026-11-07
   (`.github/workflows/ci.yml`).
4. Tag v0.1.0 with the Third Revised Edition PDF as the release
   note's reference. — **Still to do, on the web**: create tag
   `v0.1.0` and the GitHub release (see `PAID_TIER_PACKAGING.md`
   for the exact steps; the connector cannot tag).
5. Then — and only then — offer Tier 2 to GW groups. No prices
   set here; no sale until the release exists.
