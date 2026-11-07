"""Headless test for the VQE core — no Dash server needed.

Runs VQE at the equilibrium bond length for H2, LiH, H2O, NH3 and a
5-point H2 bond scan, and compares each VQE energy to exact
diagonalisation. Exit 0 only if every molecule is within tolerance;
exit 1 otherwise. Tolerances are honest for a hardware-efficient
ansatz on an analytic simulator (chemical accuracy is 0.0016 Ha;
we do NOT claim it here).
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from vqe_core import MOLECULES, bond_scan, run_vqe

TOLERANCE = {"H2": 0.005, "LiH": 0.005, "H2O": 0.01, "NH3": 0.01}


def main():
    overall_ok = True
    print("VQE headless test — simulator only, STO-3G, analytic (shots=None)")
    print("Ansatz: UCCSD from the HF reference (see vqe_core.py for why).")
    steps = {"H2": 40, "LiH": 40, "H2O": 60, "NH3": 60}
    for name, spec in MOLECULES.items():
        t0 = time.time()
        r = run_vqe(name, spec["default_bond"], steps=steps[name], layers=2,
                    optimizer="Adam", psi_overlay=False, seed=0)
        ok = abs(r["error"]) <= TOLERANCE[name]
        overall_ok &= ok
        print(
            f"{name:4s} bond={r['bond']:.2f} A  qubits={r['n_qubits']:2d}  "
            f"VQE={r['vqe_energy']:.6f} Ha  exact={r['exact_energy']:.6f} Ha  "
            f"err={r['error']:+.6f} Ha  tol={TOLERANCE[name]:.3f}  "
            f"{'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)"
        )

    print("H2 bond scan (5 points, 40 steps each):")
    scan = bond_scan("H2", [0.5, 0.74, 1.0, 1.4, 2.0], steps=40, seed=10)
    for r in scan:
        ok = abs(r["error"]) <= 0.01
        overall_ok &= ok
        print(
            f"  r={r['bond']:.2f} A  VQE={r['vqe_energy']:.6f}  "
            f"exact={r['exact_energy']:.6f}  err={r['error']:+.6f}  "
            f"{'PASS' if ok else 'FAIL'}"
        )

    print("RESULT:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
