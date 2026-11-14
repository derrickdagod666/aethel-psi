# Audit — quinovet_psigate_v3.py (uploaded file) and the pasted "Quinovetial Gate Unifier v2" text

Audited by dot, 2026-11-13. Two different programs arrived in one message:
an uploaded file (v3, 516 lines, NumPy-only) and a pasted class-based text
(v2-style, with Stim/FastAPI/lattice-surgery layers). They share a name and
a theme but almost no code. They are audited separately.

## Part 1 — The uploaded file: quinovet_psigate_v3.py — VERIFIED, adoptable

Its own header states the scope honestly: a statevector/reference
simulator; explicitly NOT lattice surgery, NOT fault-tolerant syndrome
extraction, NOT pulse control, NOT hardware. That disclaimer is accurate.

Verified by dot independently (not just its self-tests):

- **axis_rotation IS our PsiGate**: exp(−i·A/2·(sinΘ X + cosΘ Z)).
  Its numerical pure-state QFIM matches our symbolically verified
  closed-form metric (G_AA = sin²Θ, G_AΘ = sinA sinΘcosΘ,
  G_ΘΘ = 2(1−cosA) − sin²A sin²Θ) to ≤ 3.4e-10 at three random points.
- **Logical operators correctly labelled**: LOGICAL_X = Z^⊗9,
  LOGICAL_Z = X^⊗9 for the GHZ± Shor encoding, with the reason stated
  in a comment. (See Part 2 for why this matters.)
- **Transversal CNOT is real**: physical CNOTs q → q+9 on an 18-qubit
  statevector. Dot's independent checks: |0⟩|0⟩ → |0⟩|0⟩,
  |1⟩|0⟩ → |1⟩|1⟩, |+⟩|0⟩ → (|00⟩+|11⟩)/√2 — fidelity 1.0 on all three.
  The section header itself says "not lattice surgery".
- **Decoder fails loud**: syndromes outside the single-error lookup
  raise instead of guessing; the lookup itself is built with a
  stabilizer-equivalence collision check (22 equivalence classes).
  All 27 single-qubit Pauli errors corrected in its tests; three-round
  majority vote is a plain bitwise majority (odd rounds enforced).
- **QFI objective is fidelity-gated**: qfi_gate_objective returns inf
  below a 0.99 fidelity floor and only then rewards QFI gain — the same
  "QFI as constrained secondary objective" position as
  QFI_PULSE_PHASE1_PLAN.md, not the unconstrained −λ·G_QFI of the pasted
  platform spec.
- Self-tests: 13/13 PASS when run with the package venv.

Honest gaps (mostly self-declared):
- The ancilla interferometry is a correct primitive (⟨ψ|U|ψ⟩ coherence)
  but is never applied to a PsiGate loop — the file neither confirms nor
  contradicts the verified Berry phase γ = +2.2214 rad, and makes no
  Berry claim. Fine as far as it goes.
- psi_gate() (a generic Euler SU(2)) and axis_rotation() (the actual
  PsiGate) sit side by side; naming could confuse a future reader.
- CorrectionResult.corrected is always True — failures raise instead.
  Acceptable fail-loud design; the field itself is decorative.
- Single-error scope only — correct for a distance-3 code, and stated.

## Part 2 — The pasted v2-style text — SUPERSEDED, do not use

The same ambitions before the corrections landed. Faults, in order of
severity:

1. **Logical X and Z are swapped.** It sets logical_Z = Z^⊗9 and
   logical_X = X^⊗9 for the GHZ± encoding; the truth is the reverse
   (proven by v3's own logical_operator_action test). Consequence: its
   "lattice surgery CNOT" reads ⟨Z^⊗9⟩ = ⟨X_L⟩ on the control; for a
   control in |1_L⟩ that expectation is exactly 0, so p_ctrl_1 = 0.5,
   which fails its own > 0.5 trigger — **the headline gate silently
   never fires**.
2. **"Lattice surgery" is an expectation-value threshold**: a product
   ⟨Z_L1⟩·⟨Z_L2⟩ of two statevector expectation values plus a classical
   if-statement. No joint parity measurement, no projection, no
   feed-forward. Not surgery, and the [[18,2,3]] label is unearned.
3. **The "QFI-weighted" decoder never uses its weights.**
   compute_spacetime_weights feeds only two diagnostic echoes;
   filter_and_decode_spacetime's decisions are a hand-written lookup
   that ignores them. The geometry → QEC coupling is decorative; the
   worldline integrator's output likewise goes nowhere functional.
4. **The Berry "interferometer" is not one.** The built circuits are
   single-qubit (no ancilla, no controlled-U, despite the docstring);
   the "reverse" contour applies the same gates at reversed parameter
   points, which is not the inverse path; and γ = +2.223 rad is a
   hardcoded label — the stale grid value. Our converged simulation
   value is +2.2214 rad (see psigate_berry_loop.py).
5. **The Stim benchmark measures a different code** (rotated surface
   code, not the Shor tile) and reports the raw observable flip rate
   with no decoder — PyMatching is imported and never used.
6. Error rates of p_data = 0.45 / p_meas = 0.40 per cycle in the demo
   guarantee dramatic-looking trajectories and mean nothing physically.

## Verdict

The v3 FILE is the strongest code artifact in this line: it agrees with
our verified PsiGate geometry, its QEC is the real Shor code with a real
transversal CNOT, and its claims match its scope. The pasted text is its
superseded draft and should be treated the way we treat all pre-fix
drafts: kept for the record, never cited, never run as evidence.
Adoption of v3 into qce-improved (registered in run_tests.py like
PsiGate and the qutrit module) awaits Putt's approval, per standing rule.

## Addendum — v4 built 2026-11-13 on Putt's instruction ("improve it")

quinovet_psigate_v4.py sits beside the upload; v3 is untouched (MD5
verified unchanged). Changes, each answering a gap named above:

1. **The Berry phase is now computed, not just primitivised.**
   berry_phase_pancharatnam() evaluates the canonical (A, Θ) control
   loop by the Pancharatnam product. Convergence measured:
   +2.221533 (100/edge) → +2.221447 (400/edge) → +2.221442 (1600/edge),
   tested against the converged reference +2.2214 rad — never the
   note's 2.2234. The ancilla primitive's docstring now states exactly
   what it can and cannot measure.
2. **psigate_qfim_closed_form()** added (the verified metric), with a
   self-test that the numerical QFIM agrees at three points.
3. **Naming disambiguated**: the generic Euler gate is euler_su2
   (psi_gate kept as alias); psigate() is an explicit alias of
   axis_rotation, the actual PsiGate.
4. **Correction now verifies itself**: correct_single_pauli_error
   checks the corrected state against every stabilizer;
   CorrectionResult.corrected reports that verification instead of a
   hard-coded True. The docstring states the limit — a residual
   logical operator passes the check — and dot demonstrated the limit
   is real: a weight-8 error differing from X₀ by the logical X^⊗9
   returns corrected=True with logical fidelity 0.0000, exactly as
   documented.
5. **Stronger tests**: transversal CNOT truth table including
   entanglement generation; an exhaustive noisy-readout test — every
   single-qubit error × every syndrome bit × every round flipped
   (648 cases), all restored exactly. Suite: **17/17 PASS** (~5 s),
   plus a run_tests() in the qce-improved package format (also 17/17).

Still true in v4, by design: single-error scope, no lattice surgery,
no hardware claims, no Berry hardware run. Registration of v4 in
run_tests.py / adoption into qce-improved remains Putt's call.

## Addendum 2 — Audit of the pasted "9 qubits is the fault-tolerance threshold" document (2026-11-13)

A long pasted document claims 9 qubits (2^9 = 512) is "the exact
physical and mathematical phase transition" to self-correcting,
topologically protected matter, supported by QEC, topology, metrology,
secret-sharing, and 9/512 numerology sections. Verdict: real
ingredients, invented synthesis.

TRUE (the solid core):
- Shor [[9,1,3]] is the smallest degenerate concatenated code; Z
  errors within a block share syndromes (v4's own lookup: 27 errors →
  22 classes). [[5,1,3]] is perfect/non-degenerate by contrast.
- Favourable multi-error patterns correct beyond the distance
  guarantee — PROVEN in v4, see below.
- Distance-3 rotated surface code uses 9 DATA qubits (the document
  never mentions its 8 ancilla qubits; the real tile is 17).
- Noiseless 9-qubit GHZ: QFI = N² = 81; Heisenberg scaling 1/N. True
  as ideal-state arithmetic.
- 512 is a Dudeney number ((5+1+2)³ = 8³); Cl₉(R) ≅ C(16); 512-byte
  sectors and 512-point FFTs are real history.

FALSE / INFLATED (the headline layer):
- There is no fault-tolerance threshold at a qubit COUNT. The
  threshold is in noise RATE (~1% per op for surface codes); below
  it, larger codes help, above it they hurt. Below-threshold scaling
  was demonstrated by growing distance 3→5→7 (Google Willow,
  Λ ≈ 2.14), not by possessing 9 qubits. Break-even is an
  experimental result per device, never a construction guarantee —
  the document's own careful sentence ("once a tile achieves a lower
  error rate than its components") refutes its headline.
- The Shor code is NOT topological and not a lattice code; the
  document slides between Shor and the surface code as if one object.
  Nothing at 9 qubits is "self-correcting matter": correction is
  active (measure, decode, feed forward), and this project's own QEC
  v2 experiment left even an adaptive-decoding advantage unproven.
- 512 = 8³ "cubic phase space": true of ANY 9 qubits grouped 3+3+3 —
  a re-bracketing of the tensor product, conferring no simulation
  power. The octonion version is wrong (octonions are
  non-associative; O^⊗3 is not an algebra that runs SU(8) directly).
- Digit sums, Kaprekar, magic squares: facts about decimal notation
  (10 ≡ 1 mod 9), not physics; they vanish in base 8, qubits don't.
- Spin(18)'s CHIRAL spinor is 256-dimensional, not 512 (512 is the
  full spinor; Spin(19)'s spinor is 512 — the claim is half right).
- Error-corrected metrology is fenced by no-go results (generic noise
  restores the standard quantum limit); "past thermodynamic limits"
  is unsupported. "Unhackable" networks: QSS is real research, the
  guarantee language is marketing.

THE PROOF RUN (Putt chose "prove the multi-error claim in v4"):
v4 gained a collision-checked favourable-family decoder, built under
the module's own discipline (any inequivalent syndrome collision at
import raises instead of guessing):
- Family: X part at most one per block (64) × Z part inside a single
  block (22) = 1,408 patterns, all unique, resolving onto all 256
  syndrome classes with ZERO inequivalent collisions — the builder
  succeeding is the decodability proof.
- Degeneracy test: all 21 nonempty within-block Z patterns are
  stabilizer-equivalent to identity (even) or a single Z (odd).
- Spread test: all 27 weight-3 patterns (one X per block) decoded and
  corrected exactly on a general logical state (0.6|0_L⟩ + 0.8|1_L⟩).
- Boundary test: Z₀ vs Z₃Z₆ share a syndrome and differ by the logical
  Z₀Z₃Z₆ ∉ stabilizer span — the favourable region provably ends at
  the single-block condition for Z spread, exactly where the pasted
  document blurred it.
- Suite after extension: 21/21 PASS (was 17/17), standalone and via
  run_tests(). Single-error decoder and all scope limits unchanged.

CORRECTED EDITION PRODUCED 2026-11-13 (Putt: "Fix it improve it"):
workspace/your_files/nine-qubit-corrected/nine-qubits-corrected-edition.md
— the pasted document rewritten at the strength the evidence supports:
degeneracy and the proven favourable family (with the Z-spread
boundary) as the real headline; threshold reframed as a noise rate
with the Willow scaling result; surface-code tile stated as 17 qubits
and separated from the Shor code; metrology fenced by the no-go
results; secret sharing without guarantee language; Spin(18) chiral
spinor corrected to 256; digit-sum and 8³ material kept but quarantined
in a labelled "arithmetic curiosities" section with no physical
claims; the octonionic simulation claim removed. The document ends
with a nine-item corrections list, and a keyword sweep verified every
inflated term survives only in negated or quoted context.

## Addendum 3 — Noise benchmark: favourable vs single-error decoder (2026-11-13)

Pre-registered before running (design and bars in the header of
quinovet_noise_benchmark.py; results in
quinovet_noise_benchmark_results.json). Paired Monte Carlo, 20,000
trials per point, seed 20261113: depolarizing data errors at rate p
per qubit, syndrome readout flips at p_meas = p over 3 majority-voted
rounds, exact Pauli-frame outcome classification (success / erasure /
residual / logical). Grid p = 0.001 … 0.05.

RESULT — VERDICT: WIN for the favourable decoder, on the registered
bars (below the single-error decoder's total failure rate at 5 of 6
points — the sixth, p = 0.001, is an exact tie, as the P1 sanity
prediction required — with McNemar z > 2 at every p ≥ 0.005):
total failure A → B: 0.16% → 0.12% (p = .005), 0.56% → 0.41% (.01),
2.14% → 1.58% (.02), 11.43% → 8.57% (.05) — a consistent ~25–27%
relative reduction, driven by the single-error decoder turning
multi-error syndromes into erasures (1,212 of its 2,286 failures at
p = .05).

THE TRADEOFF, as pre-registered in P3 and confirmed: the favourable
decoder's SILENT logical failure rate is higher at every point
(p = .05: 3.04% vs 1.23%), because it always answers and a
minimum-weight representative sometimes lands on the logical side of
a class, while the single-error decoder's failures are mostly
erasures — known failures. Headline metric (total failure) was fixed
in advance and B wins it; which failure character matters is
application-dependent, and both numbers travel together in every
report of this benchmark. Scope: idealized Pauli-frame memory
experiment (perfect encoding and correction gates); a decoder
comparison, not a hardware prediction.

## Addendum 4 — Full test campaign PT1–PT9 (2026-11-13)

Predictions registered in quinovet_test_campaign.py's header before
the run; results in quinovet_test_campaign_results.json. 6 PASS,
3 FAIL — the failures are the informative part and stay on record.

PASS, with the numbers:
- PT1 triplet anatomy: one-block entropy of |0_L⟩ = 0 bits exactly,
  of |+_L⟩ = 1.000 bits exactly (the macro-GHZ tier carries exactly
  one ebit across every block cut); X-syndrome block-locality holds
  for all 9 qubits.
- PT2 Berry invariance: encoded-loop Pancharatnam phase equals the
  physical +2.2214 rad to < 1e-9 — the triplet encoding preserves
  the geometric phase exactly, cross-checking two code paths.
- PT3 exhaustive landscape (all 324 weight-2 and 2,268 weight-3
  patterns, noiseless readout): favourable decoder corrects 55.6%
  (w2) and 35.4% (w3) of ALL patterns vs the single-error decoder's
  13.9% and 8.1%; in-family patterns 100% at both weights; pure-Z
  same-block pairs 9/9; pure-Z CROSS-block pairs 0/27 — the boundary
  is absolute, not statistical.
- PT5 correlated noise: favourable decoder wins under adjacent-pair
  bursts and a 10× hot qubit at all 4 points.
- PT6 readout sweep: failure monotone in readout rate; 5 voted
  rounds ≤ 3 ≤ 1 at every rate.
- PT9 sabotage: all six malformed-input checks raise cleanly.

FAIL, with diagnoses:
- PT4 (bias): the prediction "favourable wins at all 8 biased
  points" FAILS under Z bias η=10 — at p = .005/.01/.02 it ties or
  trails the single-error decoder (winning only at .05), while under
  X bias it wins everywhere by wide margins. The paired sub-claim
  passed: its silent-logical share of failures under Z bias (48%)
  is double the X-bias share (23%) at p = .02.
- PT7 (coherent errors): the registered direction FAILS. At θ=0.20
  random-axis, coherent failures were 0/300 vs the twirl's 0.585%
  — coherent errors hurt LESS than their Pauli twirl at small
  angles (projective collapse strips most of the rotation); at
  θ=0.35 they agree (4.67% vs 4.50%). The Z-only condition was the
  mildest relative to its twirl (0.33% vs 0.925%), not the worst.
  Caveat recorded: the success criterion (logical expectations
  within 0.5) is blind to small residual coherent rotations.
- PT8 (statevector cross-check): FAIL as scored at p = .01
  (0/1000 statevector vs 0.205% frame) — a power artifact of the
  test's own design (a zero count gives σ=0 and an unpassable bar);
  the p = .02 point passed (0.70% vs 0.55%, within 3σ). Superseded
  methodologically by PT8b below; the FAIL stands.

## Addendum 5 — Follow-up tests PT4b / PT8b (registered after the campaign, before running)

- PT8b (powered cross-check, 6,000 statevector trials at p = .01):
  PASS, cleanly — statevector failure 0.200% (12/6000) vs frame
  0.210%, a 0.15σ difference. The Pauli-frame benchmark machinery
  is validated against full statevector simulation.
- PT4b (bias crossover map, η = 1…20 at p = .01/.02): its registered
  crossover prediction ALSO fails — the favourable decoder never
  actually goes underwater; its advantage at p = .02 shrinks from
  +0.54 points (η=1) through +0.44 (η=3) to +0.10…+0.20 at η = 5–20,
  i.e. a statistical TIE under strong Z bias (the campaign's one
  negative η=10 point was within noise of zero). Synthesis across
  PT4 + PT4b: the decoder's total-failure edge is real under
  balanced and X-biased noise (~25% relative), and erodes to a tie
  under strong Z bias — while its silent-logical share stays
  roughly double. That tie, not a win and not a loss, is the
  favourable decoder's honest operating envelope on its weak side.
