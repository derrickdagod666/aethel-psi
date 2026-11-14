#!/usr/bin/env python3
"""
quinovet_noise_benchmark.py -- Monte Carlo logical-error benchmark of
the two Shor-code decoders in quinovet_psigate_v4.py:

  A) the single-error lookup decoder (v3's original), and
  B) the favourable-family decoder (v4's multi-error extension).

PRE-REGISTERED DESIGN (fixed 2026-11-13, before this script was run)

Noise model, per trial:
  * Each of the 9 data qubits independently suffers a Pauli error with
    probability p (depolarizing: X, Y, Z equally likely given an error).
  * The syndrome is measured in 3 rounds; each of the 8 syndrome bits
    in each round is independently flipped with probability p_meas = p
    (the harsh choice: readout as noisy as the data).
  * Rounds are combined by bitwise majority vote (the module's rule).
  * The SAME sampled error and the SAME noisy rounds are fed to both
    decoders (paired comparison; differences are scored with the
    McNemar test on discordant trials).

Outcome classification (exact Pauli-frame bookkeeping -- a trial's
outcome depends only on the sampled error and the vote, never on a
chosen logical state):
  * erasure:        the decoder has no entry for the voted syndrome
                    and raises (only possible for decoder A).
  * success:        residual R = correction x error lies in the
                    stabilizer span (state restored exactly).
  * residual:       R has nonzero syndrome (state left the code space).
  * logical:        R has zero syndrome but is not a stabilizer
                    (silent logical corruption).

Grid and statistics:
  * p in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05}; 20,000 trials per p.
  * RNG seed fixed at 20261113. Rates reported with normal sigma.

PRE-REGISTERED PREDICTIONS AND BARS
  P1 (sanity): at p = 0.001 the two decoders' total failure rates
     (erasure + residual + logical) agree within 3 sigma -- single
     errors dominate there and both decoders treat them identically
     up to stabilizer equivalence.
  P2 (the claim): at every p >= 0.005, decoder B's total failure rate
     is strictly below decoder A's, the gap driven by A turning
     multi-error syndromes into erasures.
  P3 (the honesty check): decoder B's SILENT logical failure rate may
     exceed A's logical rate among decoded trials -- B always answers,
     and a minimum-weight representative sometimes lands on the logical
     side of a class. This metric is printed alongside the headline;
     an erasure (known failure) and a logical (silent corruption) are
     not the same kind of bad, and which matters is application-
     dependent. The headline metric is total failure, fixed here.
  VERDICT: B WINS if its total failure rate is below A's at >= 5 of
     the 6 grid points AND the paired McNemar z > 2 at every
     p >= 0.005. B LOSES if its total failure rate is above A's at
     any p >= 0.01. Anything else is MIXED and reported as such.

Scope: Pauli-frame Monte Carlo of an idealized memory experiment --
perfect encoding, perfect correction gates, one error burst then one
decode. It compares DECODERS for the Shor code; it is not a hardware
prediction and says nothing about any physical device's error rates.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quinovet_psigate_v4 as qv

SEED = 20261113
TRIALS = 20_000
P_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
ZERO_SYN = (0,) * 8


def sample_error(rng: np.random.Generator, p: float) -> qv.Pauli:
    ops = {}
    for q in range(qv.N_QUBITS):
        if rng.random() < p:
            ops[q] = str(rng.choice(["X", "Y", "Z"]))
    return qv.pauli_from_ops(ops)


def classify(correction, error) -> str:
    if correction is None:
        return "erasure"
    residual = qv.pauli_difference(correction, error)
    if residual in qv.STABILIZER_SPAN:
        return "success"
    if qv.syndrome(residual) != ZERO_SYN:
        return "residual"
    return "logical"


def run_point(rng: np.random.Generator, p: float) -> dict:
    counts = {
        d: {"success": 0, "erasure": 0, "residual": 0, "logical": 0}
        for d in ("A_single", "B_favourable")
    }
    discordant = {"A_fail_B_ok": 0, "B_fail_A_ok": 0}
    for _ in range(TRIALS):
        error = sample_error(rng, p)
        true_syn = qv.syndrome(error)
        rounds = [
            qv.simulate_syndrome_measurement(true_syn, p, rng)
            for _ in range(3)
        ]
        voted = qv.majority_vote_syndromes(rounds)

        try:
            corr_a = qv.decode_single_qubit_error(voted)
        except ValueError:
            corr_a = None
        corr_b = qv.decode_favourable_family(voted)

        out_a = classify(corr_a, error)
        out_b = classify(corr_b, error)
        counts["A_single"][out_a] += 1
        counts["B_favourable"][out_b] += 1
        a_fail = out_a != "success"
        b_fail = out_b != "success"
        if a_fail and not b_fail:
            discordant["A_fail_B_ok"] += 1
        elif b_fail and not a_fail:
            discordant["B_fail_A_ok"] += 1

    point = {"p": p, "trials": TRIALS, "decoders": {}}
    for name, c in counts.items():
        total_fail = c["erasure"] + c["residual"] + c["logical"]
        rate = total_fail / TRIALS
        point["decoders"][name] = {
            **c,
            "total_failures": total_fail,
            "total_failure_rate": rate,
            "sigma": math.sqrt(rate * (1.0 - rate) / TRIALS),
            "logical_rate_all_trials": c["logical"] / TRIALS,
        }
    b, cnum = discordant["A_fail_B_ok"], discordant["B_fail_A_ok"]
    point["mcnemar"] = {
        **discordant,
        "z": (b - cnum) / math.sqrt(b + cnum) if (b + cnum) > 0 else 0.0,
    }
    return point


def main() -> int:
    rng = np.random.default_rng(seed=SEED)
    results = [run_point(rng, p) for p in P_GRID]

    print(f"{'p':>7} | {'A total fail':>18} | {'B total fail':>18} | "
          f"{'McNemar z':>9} | {'A logical':>9} | {'B logical':>9}")
    print("-" * 84)
    for r in results:
        a = r["decoders"]["A_single"]
        b = r["decoders"]["B_favourable"]
        print(f"{r['p']:>7.3f} | {a['total_failure_rate']:>8.4f} "
              f"(e{a['erasure']:>5}) | {b['total_failure_rate']:>8.4f} "
              f"(e{b['erasure']:>5}) | {r['mcnemar']['z']:>9.1f} | "
              f"{a['logical_rate_all_trials']:>9.5f} | "
              f"{b['logical_rate_all_trials']:>9.5f}")

    # Score against the pre-registered bars.
    by_p = {r["p"]: r for r in results}
    a0 = by_p[0.001]["decoders"]["A_single"]
    b0 = by_p[0.001]["decoders"]["B_favourable"]
    diff0 = abs(a0["total_failure_rate"] - b0["total_failure_rate"])
    comb_sigma = math.sqrt(a0["sigma"] ** 2 + b0["sigma"] ** 2)
    p1 = diff0 < 3 * comb_sigma if comb_sigma > 0 else diff0 == 0
    p2 = all(
        by_p[p]["decoders"]["B_favourable"]["total_failure_rate"]
        < by_p[p]["decoders"]["A_single"]["total_failure_rate"]
        for p in (0.005, 0.01, 0.02, 0.05)
    )
    b_wins_points = sum(
        r["decoders"]["B_favourable"]["total_failure_rate"]
        < r["decoders"]["A_single"]["total_failure_rate"]
        for r in results
    )
    z_ok = all(by_p[p]["mcnemar"]["z"] > 2 for p in (0.005, 0.01, 0.02, 0.05))
    b_loses = any(
        by_p[p]["decoders"]["B_favourable"]["total_failure_rate"]
        > by_p[p]["decoders"]["A_single"]["total_failure_rate"]
        for p in (0.01, 0.02, 0.05)
    )
    if b_loses:
        verdict = "LOSE"
    elif b_wins_points >= 5 and z_ok:
        verdict = "WIN"
    else:
        verdict = "MIXED"
    print(f"\nP1 (sanity at p=0.001): {'PASS' if p1 else 'FAIL'} "
          f"(|diff| = {diff0:.5f}, 3-sigma bound = {3 * comb_sigma:.5f})")
    print(f"P2 (B below A at all p >= 0.005): {'PASS' if p2 else 'FAIL'}")
    print(f"VERDICT: {verdict} "
          f"(B below A at {b_wins_points}/6 points; McNemar z > 2 at all "
          f"p >= 0.005: {z_ok})")

    out = {
        "seed": SEED, "trials_per_point": TRIALS, "p_meas_equals_p": True,
        "results": results,
        "scoring": {"P1_sanity": bool(p1), "P2_claim": bool(p2),
                    "verdict": verdict},
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "quinovet_noise_benchmark_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
