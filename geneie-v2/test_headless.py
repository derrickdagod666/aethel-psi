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

    # 1b. Uploaded-CSV re-analysis (added 2026-11-16 with the upload fix:
    # uploads used to be parsed and ignored while PCA ran on the demo data)
    clean, rep = core.prepare_numeric_frame(df)
    check("upload: numeric features kept, text cols dropped",
          clean.shape == (120, 12) and set(rep["dropped_columns"]) == {"sample_id", "gene"},
          f"clean={clean.shape} dropped={rep['dropped_columns']}")
    res_u = core.pca_kmeans(clean)
    check("upload: PCA/K-means on the upload recovers clusters",
          adjusted_rand_score(y_true, res_u["labels"]) > 0.5,
          f"ARI={adjusted_rand_score(y_true, res_u['labels']):.3f}")
    messy = df.copy()
    messy["const"] = 3.14
    messy.loc[0, "feature_1"] = np.nan
    clean_m, rep_m = core.prepare_numeric_frame(messy)
    check("upload: constant column dropped, missing cell imputed and counted",
          rep_m["constant_columns"] == ["const"] and rep_m["imputed_cells"] >= 1
          and bool(np.isfinite(clean_m.to_numpy()).all()),
          f"rep={rep_m['constant_columns']}/{rep_m['imputed_cells']}")
    try:
        core.prepare_numeric_frame(df.head(2)); check("upload: rejects too few rows", False)
    except ValueError:
        check("upload: rejects too few rows", True)
    try:
        core.prepare_numeric_frame(df[["sample_id", "gene"]].assign(only=[1.0] * 120))
        check("upload: rejects <2 numeric features", False)
    except ValueError:
        check("upload: rejects <2 numeric features", True)

    # 1c. Per-run provenance (v2.2.0, evidence report's cheapest credibility
    # win: every run traceable to inputs, cleaning, parameters, versions)
    num = df.select_dtypes(include=[np.number])
    rec_demo = core.build_provenance(dataset_kind="demo", filename=None,
                                     input_shape=df.shape, analysed_frame=num,
                                     cleaning_report=None, analysis_result=res)
    check("provenance: demo record fields",
          rec_demo["app_version"] == core.APP_VERSION
          and rec_demo["dataset"] == {"kind": "demo", "filename": None}
          and rec_demo["input_shape"] == {"rows": 120, "columns": 14}
          and rec_demo["cleaned_shape"] == {"rows": 120, "columns": 12}
          and rec_demo["cleaning_report"] is None,
          f"v={rec_demo['app_version']} hash={rec_demo['cleaned_data_sha256_short']}")
    rec_demo2 = core.build_provenance(dataset_kind="demo", filename=None,
                                      input_shape=df.shape, analysed_frame=num,
                                      cleaning_report=None, analysis_result=res)
    check("provenance: SHA-256 is full-length and reproducible",
          len(rec_demo["cleaned_data_sha256"]) == 64
          and rec_demo2["cleaned_data_sha256"] == rec_demo["cleaned_data_sha256"],
          rec_demo["cleaned_data_sha256"][:16])
    rec_up = core.build_provenance(dataset_kind="upload", filename="demo.csv",
                                   input_shape=df.shape, analysed_frame=clean,
                                   cleaning_report=rep, analysis_result=res_u)
    # Hash identity note: cleaning the demo-shaped upload yields exactly the
    # demo's analysed frame, so its hash EQUALS the demo record's — that is
    # the hash doing its job (same data, same run identity). Distinctness is
    # therefore checked against the messy upload, whose imputed cell changes
    # the analysed values and must change the hash.
    rec_messy = core.build_provenance(dataset_kind="upload", filename="messy.csv",
                                      input_shape=messy.shape,
                                      analysed_frame=clean_m,
                                      cleaning_report=rep_m,
                                      analysis_result=res_u)
    check("provenance: upload record carries filename + cleaning report",
          rec_up["dataset"] == {"kind": "upload", "filename": "demo.csv"}
          and set(rec_up["cleaning_report"]["dropped_columns"]) == {"sample_id", "gene"}
          and rec_up["cleaning_report"]["imputed_cells"] == rep["imputed_cells"]
          and rec_up["cleaned_data_sha256"] == rec_demo["cleaned_data_sha256"]
          and rec_messy["cleaned_data_sha256"] != rec_demo["cleaned_data_sha256"],
          f"dropped={rec_up['cleaning_report']['dropped_columns']}")
    check("provenance: analysis parameters + library versions recorded",
          rec_demo["analysis"]["pca_components"] == 2
          and rec_demo["analysis"]["kmeans_clusters"] == 2
          and rec_demo["analysis"]["kmeans_seed"] == 42
          and set(rec_demo["libraries"]) == {"numpy", "pandas", "scikit-learn", "dash"}
          and all(v not in ("", "not installed")
                  for v in rec_demo["libraries"].values()),
          str(rec_demo["libraries"]))
    check("provenance: research-only notice attached to the record",
          "Research use only" in rec_demo["research_only"]
          and "not clinical diagnosis" in rec_demo["research_only"])
    md = core.provenance_markdown(rec_up)
    check("provenance: markdown renders hash, cleaning and notice",
          rec_up["cleaned_data_sha256"] in md and "sample_id" in md
          and "Research use only" in md)

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
