"""GENEIE v2 headless test — runs the core with no Dash server.

Prints measured numbers; exits non-zero if any check fails. Run:
    python test_headless.py
"""
import sys
import numpy as np
from sklearn.metrics import adjusted_rand_score

import geneie_core as core

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def main():
    # 1. Synthetic genomic data + PCA/K-means
    df, y_true = core.synthetic_genomic_data()
    res = core.pca_kmeans(df)
    ari = adjusted_rand_score(y_true, res["labels"])
    check("genomic data shape", df.shape == (120, 14), str(df.shape))
    check("PCA variance PC1+PC2 > 0.3",
          sum(res["explained_variance_ratio"]) > 0.3,
          f"PC1={res['explained_variance_ratio'][0]:.3f} PC2={res['explained_variance_ratio'][1]:.3f}")
    check("K-means recovers planted clusters (ARI > 0.5)", ari > 0.5, f"ARI={ari:.3f}")
    check("t-SNE runs (n>=31)", res["tsne"] is not None and res["tsne"].shape == (120, 2))

    # 2. Drug model
    X, y = core.synthetic_drug_data()
    m = core.train_drug_model(X, y)
    check("drug model test accuracy 0.6-0.95", 0.6 <= m["test_accuracy"] <= 0.95,
          f"acc={m['test_accuracy']:.3f} n_test={m['n_test']} iters={m['n_iter']}")
    pred = core.predict_response(m, [0.0] * 10)
    check("drug prediction returns probability", 0.0 <= pred["p_response"] <= 1.0,
          f"p={pred['p_response']:.3f} pred={pred['prediction']}")

    # 3. DNA
    s = core.dna_stats(core.DEMO_SEQUENCES["BRCA1-fragment"])
    check("DNA GC content 30-70%", 30 <= s["gc_percent"] <= 70, f"GC={s['gc_percent']}% len={s['length']}")
    check("DNA base counts sum to length",
          s["a"] + s["t"] + s["g"] + s["c"] == s["length"])

    # 4. VQE H2 at equilibrium-ish (r=1.4 bohr, 40 steps for speed in test)
    r = core.vqe_energy(["H", "H"], [(0.0, 0.0, -0.7), (0.0, 0.0, 0.7)], max_steps=40)
    check("VQE H2 energy near exact (err < 0.05 Ha)", r["error"] < 0.05,
          f"VQE={r['vqe_energy']:.4f} exact={r['exact_energy']:.4f} err={r['error']:.4f} qubits={r['n_qubits']}")
    check("VQE H2 exact energy in known range (-1.2, -1.0) Ha",
          -1.2 < r["exact_energy"] < -1.0, f"{r['exact_energy']:.4f}")

    # 5. H2 bond scan — minimum near 1.4 bohr
    scan = core.h2_bond_scan(bond_lengths_bohr=(1.0, 1.4, 1.8), max_steps=30)
    energies = [s["vqe_energy"] for s in scan]
    check("H2 bond scan minimum at 1.4 bohr", int(np.argmin(energies)) == 1,
          f"energies={[round(e, 4) for e in energies]}")

    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
