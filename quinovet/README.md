# Quinovet — PsiGate Shor-code module

A self-contained, heavily tested reference implementation of the
[[9,1,3]] Shor code with the PsiGate family, by Derrick Irons
(Independent Researcher). NumPy only; no other dependencies.

## Contents

- `quinovet_psigate_v4.py` — the module: PsiGate with closed-form and
  numerical QFI that must agree, the Shor encoding, a transversal
  logical CNOT, the Berry phase of the canonical control loop computed
  by the Pancharatnam product (+2.2214 rad), two decoders, and 21
  self-tests. Run it directly: `python quinovet_psigate_v4.py`.
- **The favourable-family decoder** (new in v4): bit-flips at most one
  per 3-qubit block combined with phase errors inside a single block —
  1,408 multi-error patterns, proven uniquely decodable by a collision
  check that raises rather than guesses, with the boundary of the
  favourable region itself proven (phase errors spread across blocks
  collide with logical operators; 0 of 27 cross-block phase pairs are
  correctable).
- `quinovet_noise_benchmark.py` (+ results) — a pre-registered paired
  Monte Carlo: the favourable decoder cuts total logical failure by
  ~25–27% relative versus the single-error decoder across
  p = 0.005…0.05 — **and** has a higher *silent* logical failure rate,
  because it never erases where the other decoder gives up. Both
  metrics are in the results; the tradeoff is the finding.
- `quinovet_test_campaign.py` + `quinovet_followup_tests.py`
  (+ results) — nine further registered tests and two follow-ups:
  triplet anatomy, encoded Berry phase, the exhaustive weight-2/3
  error landscape, bias, correlated noise, readout, coherence, and a
  statevector cross-validation (0.200% vs 0.210%). Three of our own
  pre-registered predictions FAILED and are kept on record with
  diagnoses; under strong phase bias the favourable decoder's edge
  erodes to a statistical tie.
- `QUINOVET_V3_AUDIT.md` — the complete audit trail for this line of
  work, including the rejection of an inflated earlier draft.

## Scope

Statevector / Pauli-frame reference implementation. Not lattice
surgery, not fault-tolerant syndrome extraction, not a hardware
prediction. Nothing here is peer-reviewed.

## Companion note

"Correctable Beyond the Guarantee: An Enumerated Favourable-Family
Decoder for the Shor [[9,1,3]] Code, with Benchmarks and a Mapped
Failure Envelope" — preprint, November 2026 (Zenodo deposit in
preparation).

MIT license, as with the rest of this repository.
