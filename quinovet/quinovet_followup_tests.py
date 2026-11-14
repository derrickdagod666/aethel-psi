#!/usr/bin/env python3
"""
quinovet_followup_tests.py -- Two NEW tests registered after the main
campaign (quinovet_test_campaign.py), written 2026-11-13 BEFORE being
run. They do not regrade anything: PT4 and PT8 of the campaign remain
FAIL on the record. These follow-ups measure what the failures pointed
at.

PT4b (bias crossover map): the campaign showed the favourable
decoder's total-failure advantage disappearing under Z-biased noise
at eta = 10. Map the advantage across bias eta in {1, 2, 3, 5, 10, 20}
at p in {0.01, 0.02} (20,000 paired-distribution trials per point,
readout flips at p_meas = p, 3 voted rounds).
  REGISTERED PREDICTION: a crossover exists -- B beats A at eta <= 3
  and ties or loses at eta >= 10, at p = 0.02.

PT8b (powered statevector cross-check): the campaign's PT8 compared
1,000 statevector trials against a 0.205% frame rate at p = 0.01; a
zero count there cannot resolve the comparison (the 3-sigma bar used
sigma = 0 for the zero rate). PT8b reruns the statevector memory
experiment at p = 0.01 with 6,000 trials and scores with Poisson-
correct sigmas on both sides.
  REGISTERED PREDICTION: the statevector failure rate agrees with
  the Pauli-frame rate (p_meas = 0, recomputed at 20,000 trials)
  within 3 combined sigma.
"""

import json
import math
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quinovet_psigate_v4 as qv
from quinovet_test_campaign import (
    decode_a, decode_b, frame_trial, iid_error, run_frame_condition,
)

RESULTS = {}


def test_pt4b_crossover(trials: int = 20_000) -> dict:
    out = {}
    for p in (0.01, 0.02):
        out[str(p)] = {}
        for eta in (1, 2, 3, 5, 10, 20):
            s = run_frame_condition(
                f"PT4b_p{p}_eta{eta}",
                lambda rng, p=p, eta=eta: iid_error(rng, p, ("Z", float(eta))),
                trials, p_meas=p, rounds_n=3)
            out[str(p)][str(eta)] = {
                "A_total_failure": s["A"]["total_failure_rate"],
                "B_total_failure": s["B"]["total_failure_rate"],
                "B_advantage": s["A"]["total_failure_rate"]
                - s["B"]["total_failure_rate"],
            }
    adv = out["0.02"]
    out["pass"] = bool(
        adv["3"]["B_advantage"] > 0 and adv["10"]["B_advantage"] <= 0)
    RESULTS["PT4b_crossover"] = out
    return out


def test_pt8b_statevector(trials: int = 6000) -> dict:
    psi0 = qv.shor_encode(0.6, 0.8)
    p = 0.01
    rng = np.random.default_rng(seed=zlib.crc32(b"PT8b_p0.01"))
    fails = 0
    for _ in range(trials):
        error = iid_error(rng, p)
        psi = qv.apply_pauli(psi0, error)
        syn = []
        for s in qv.STABILIZERS:
            exp_val = np.vdot(psi, qv.apply_pauli(psi, s)).real
            syn.append(0 if exp_val > 0 else 1)
        corr = qv.decode_favourable_family(tuple(syn))
        fixed = qv.apply_pauli(psi, corr)
        if abs(np.vdot(psi0, fixed)) ** 2 < 0.999:
            fails += 1
    sv_rate = fails / trials
    frame = run_frame_condition(
        "PT8b_frame", lambda r: iid_error(r, p),
        20_000, p_meas=0.0, rounds_n=1, decoders=("B",))
    fr_rate = frame["B"]["total_failure_rate"]
    comb = math.sqrt(sv_rate * (1 - sv_rate) / trials
                     + fr_rate * (1 - fr_rate) / 20_000)
    out = {
        "statevector_trials": trials,
        "statevector_failures": fails,
        "statevector_failure_rate": sv_rate,
        "frame_failure_rate": fr_rate,
        "combined_sigma": comb,
        "pass": bool(abs(sv_rate - fr_rate) < 3 * comb),
    }
    RESULTS["PT8b_statevector_powered"] = out
    return out


def main() -> int:
    print("== PT4b bias crossover ==")
    print(json.dumps(test_pt4b_crossover(), indent=2))
    print("== PT8b powered statevector cross-check ==")
    print(json.dumps(test_pt8b_statevector(), indent=2))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "quinovet_followup_results.json")
    with open(path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
