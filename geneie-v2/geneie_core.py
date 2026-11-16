"""GENEIE v2 — core logic (no web server needed to test this).

Changes vs the original, per Putt's rule (elaborate on every change):
- The original trained a Keras model inside a window constructor on CSVs that did
  not exist (genome.csv / drugs.csv), with X = drugs['Features'] assumed 2-D.
  v2 generates labelled SYNTHETIC demo data (clearly labelled), builds a real
  numeric feature matrix, and trains an sklearn MLPClassifier off the UI thread.
  Keras/TensorFlow is dropped: it was a heavy dependency for a 128->64 MLP that
  sklearn does identically on this scale, and it kept the original from installing.
- The original called SeqIO.read() on a sequence string (raises). v2 parses DNA
  strings directly with Bio.Seq and computes GC content and k-mer counts.
- The original's PyQuil `pq.QuantumAlgorithm(...)` does not exist. v2 does real VQE
  with PennyLane: qml.qchem.molecular_hamiltonian + UCCSD ansatz + Adam, and
  compares against exact diagonalisation of the same Hamiltonian.
- The original's Cirq one-qubit coin flip is dropped (it proved nothing).
- The original's blockchain / encrypt / database / analysis placeholders are
  dropped entirely (undefined names). No fake security theatre in v2.
- PCA + K-means from the original are kept, and actually run, on synthetic data.
- Everything here runs headlessly: see test_headless.py.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)

APP_VERSION = "2.2.0"

# The research-only boundary, as one canonical string: the app shows it in a
# persistent banner and build_provenance() attaches it to every run record,
# so the disclaimer travels with any exported result instead of living only
# in the README (evidence report, 2026-11-16: research-vs-clinical boundary).
RESEARCH_ONLY_NOTICE = (
    "Research use only: GENEIE is a research tool. Its outputs are hypotheses "
    "for research use — not clinical diagnosis, treatment, or therapeutic "
    "recommendations."
)

# ---------------------------------------------------------------- genomic demo data
GENE_NAMES = [
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "APOE", "CFTR", "HBB",
    "MLH1", "MSH2", "PTEN", "RB1", "VHL", "NF1", "ATM", "CHEK2",
]

DEMO_SEQUENCES = {
    "BRCA1-fragment": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGAACCTGATC",
    "TP53-fragment": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCCCTGTCATCTTCTGTCCCTTCCCAGAAAACCTACCAGGGCAGCTACGGTTTCCGTCTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACTTGCACGTACTCCCCTGCCCTCAACAAGATGTTTTGCCAACTGGCCAAGACCTGCCCTGTGCAGCTGTGGGTTGATTCCACACCCCCGCCCGGCACCCGCGTCCGCGCCATGGCCATCTACAAGCAGTCACAGCACATGACGGAGGTTGTGAGGCGCTGCCCCCACCATGAGCGCTGCTCAGATAGCGATGGTCTGGCCCCTCCTCAGCATCTTATCCGAGTGGAAGGAAATTTGCGTGTGGAGTATTTGGATGACAGAAACACTTTTCGACATAGTGTGGTGGTGCCCTATGAGCCGCCTGAGGTTGGCTCTGACTGTACCACCATCCACTACAACTACATGTGTAACAGTTCCTGCATGGGCGGCATGAACCGGAGGCCCATCCTCACCATCATCACACTGGAAGACTCCAGGTCAGGAGCCACTTGCTTACCAGGTGAAAACTTTAG",
    "EGFR-fragment": "ATGCGACCCTCCGGGACGGCCGGGGGCAGCGCTCCTGGCGCTGCTGGCTGCGCTCTGCCCGGCGAGTCGGGCTCTGGAGGAAAAGAAAGTTTGCCAAGGCACGAGTAACAAGCTCACGCAGTTGGGCACTTTTGAAGATCATTTTCTCAGCCTCCAGAGGATGTTCAATAACTGTGAGGTGGTCCTTGGGAATTTGGAAATTACCTATGTGCAGAGGAATTATGATCTTTCCTTAAAAGGAAATCACAGCAGGTTTGGTAAATTTTGTAAAAAACGTGGGAAAACAAAAAAA",
}

# ---------------------------------------------------------------- synthetic data
def synthetic_genomic_data(n_samples: int = 120, n_features: int = 12, seed: int = 42):
    """Synthetic multi-omics-style demo table. NOT real patient data.

    Two hidden clusters are planted (a mixture of two Gaussians) so PCA/K-means
    has real structure to find; labels are the planted cluster, used only to
    check that clustering recovers structure (adjusted Rand is reported by the test).
    """
    rng = np.random.default_rng(seed)
    n1 = n_samples // 2
    c1 = rng.normal(loc=0.0, scale=1.0, size=(n1, n_features))
    c2 = rng.normal(loc=2.2, scale=1.0, size=(n_samples - n1, n_features))
    X = np.vstack([c1, c2])
    y = np.array([0] * n1 + [1] * (n_samples - n1))
    cols = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "sample_id", [f"S{i:03d}" for i in range(n_samples)])
    df.insert(1, "gene", [GENE_NAMES[i % len(GENE_NAMES)] for i in range(n_samples)])
    return df, y


def prepare_numeric_frame(df: pd.DataFrame, min_rows: int = 4, min_features: int = 2):
    """Clean an uploaded table for PCA + K-means. Returns (clean_df, report).

    Added 2026-11-16 to close the v2 upload gap (uploads were parsed and then
    ignored; PCA always ran on the demo data). What this does, and why:

    - A column counts as numeric when at least half of its values parse as
      numbers and at least `min_rows` of them do. That keeps real feature
      columns (including numbers stored as text) and drops ID / gene / label
      text columns instead of crashing on them or silently coercing them.
    - Infinities become NaN; rows that are empty across every numeric column
      are dropped; remaining missing cells are median-imputed per column and
      counted, so the report can state exactly how much was filled in.
    - Constant (zero-variance) columns are dropped and named: after scaling
      they carry no signal and would otherwise poison the scaler with NaNs.
    - Nothing is invented silently: the report names every dropped column,
      the imputed-cell count, duplicate-row count and missing-value count,
      so the UI can show the user precisely what their data went through.
      (The duplicate / missing counts borrow the summary idea from the
      GENEIE_unified starter Putt uploaded the same day; that starter only
      summarised a CSV and never ran PCA on it, so the analysis wiring here
      is new.)

    Raises ValueError with a user-readable message when the upload cannot
    support the analysis (too few usable rows or numeric features).
    """
    if df is None or len(df) == 0:
        raise ValueError("The CSV contains no data rows.")
    if len(df) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} data rows to run PCA + K-means; "
            f"this CSV has {len(df)}.")
    numeric_cols, dropped_cols = [], []
    cleaned = {}
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_numeric = int(coerced.notna().sum())
        if n_numeric >= min_rows and n_numeric >= 0.5 * len(df):
            numeric_cols.append(col)
            cleaned[col] = coerced.replace([np.inf, -np.inf], np.nan)
        else:
            dropped_cols.append(str(col))
    if len(numeric_cols) < min_features:
        raise ValueError(
            f"Need at least {min_features} numeric feature columns to run "
            f"PCA + K-means; found {len(numeric_cols)} usable in this CSV. "
            f"Dropped as non-numeric or too sparse: {dropped_cols or 'none'}.")
    frame = pd.DataFrame(cleaned)
    missing_before = int(frame.isna().sum().sum())
    # Constant columns are dropped BEFORE empty-row removal: a row that is
    # empty in every varying feature but present in a constant column is
    # still an empty row, and an early test caught exactly that surviving
    # to be fully median-imputed.
    constant_cols = [c for c in frame.columns if frame[c].dropna().nunique() <= 1]
    if constant_cols:
        frame = frame.drop(columns=constant_cols)
    if frame.shape[1] < min_features:
        raise ValueError(
            f"Need at least {min_features} varying numeric features; "
            f"constant columns removed: {constant_cols}.")
    frame = frame.dropna(how="all")
    if len(frame) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} rows with numeric data; "
            f"only {len(frame)} remain after dropping empty rows.")
    imputed_cells = int(frame.isna().sum().sum())
    frame = frame.fillna(frame.median(numeric_only=True))
    if frame.shape[1] < min_features:
        raise ValueError(
            f"Need at least {min_features} varying numeric features; "
            f"constant columns removed: {constant_cols}.")
    report = {
        "n_rows_input": int(len(df)),
        "n_rows_used": int(len(frame)),
        "n_columns_input": int(len(df.columns)),
        "numeric_columns": [str(c) for c in frame.columns],
        "dropped_columns": dropped_cols,
        "constant_columns": [str(c) for c in constant_cols],
        "missing_values": missing_before,
        "imputed_cells": imputed_cells,
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return frame.reset_index(drop=True), report


# ---------------------------------------------------------------- provenance
def library_versions():
    """Versions of the key libraries a run actually depends on.

    Read from installed distribution metadata (not module __version__) so the
    record matches the environment that produced it, e.g. inside the deploy
    container. Missing packages are reported as such, never guessed.
    """
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for dist in ("numpy", "pandas", "scikit-learn", "dash"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = "not installed"
    return out


def build_provenance(*, dataset_kind: str, filename, input_shape,
                     analysed_frame: pd.DataFrame, cleaning_report,
                     analysis_result: dict, pca_components: int = 2,
                     n_clusters: int = 2, seed: int = 42) -> dict:
    """Build the per-run provenance record for one PCA + K-means run.

    Added 2026-11-16 (v2.2.0), per the GENEIE evidence report's cheapest
    credibility win: every result should be traceable to its inputs, cleaning,
    parameters, and software versions. The record is a plain JSON-safe dict;
    the app keeps it in session memory and offers it as a download — nothing
    is written to disk server-side.

    - `dataset_kind` is "demo" or "upload"; `filename` is the upload's name
      (None for the demo data).
    - `input_shape` is the (rows, columns) of the data as received, before
      cleaning; `analysed_frame` is the exact numeric frame fed to PCA, and
      its shape is recorded as the after-cleaning shape.
    - `cleaning_report` is the dict from prepare_numeric_frame for uploads;
      None for the demo data, which needs no cleaning (recorded as such).
    - The SHA-256 of the analysed frame's CSV bytes identifies the run's
      data: the same cleaned data always hashes the same, so a run can be
      recognised and repeated.
    - The research-only notice is attached to every record so it travels
      with any exported copy of the result.
    """
    digest = hashlib.sha256(
        analysed_frame.to_csv(index=False).encode("utf-8")).hexdigest()
    return {
        "app": "GENEIE",
        "app_version": APP_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"kind": dataset_kind, "filename": filename},
        "input_shape": {"rows": int(input_shape[0]),
                        "columns": int(input_shape[1])},
        "cleaned_shape": {"rows": int(analysed_frame.shape[0]),
                          "columns": int(analysed_frame.shape[1])},
        "cleaning_report": cleaning_report,
        "analysis": {
            "method": "PCA + K-means (numeric features only; the drug-response "
                      "model is separate and never retrains on uploads)",
            "pca_components": int(pca_components),
            "kmeans_clusters": int(n_clusters),
            "kmeans_seed": int(seed),
            "kmeans_n_init": 10,
            "explained_variance_ratio":
                [float(v) for v in analysis_result["explained_variance_ratio"]],
            "inertia": float(analysis_result["inertia"]),
        },
        "libraries": library_versions(),
        "cleaned_data_sha256": digest,
        "cleaned_data_sha256_short": digest[:12],
        "research_only": RESEARCH_ONLY_NOTICE,
    }


def provenance_markdown(record: dict) -> str:
    """Render a provenance record as a small human-readable Markdown document."""
    d = record["dataset"]
    dataset = ("built-in synthetic demo data" if d["kind"] == "demo"
               else f"upload: {d['filename']}")
    lines = [
        "# GENEIE run provenance",
        "",
        f"- App: GENEIE v{record['app_version']}",
        f"- Timestamp (UTC): {record['timestamp_utc']}",
        f"- Dataset: {dataset}",
        f"- Input shape: {record['input_shape']['rows']} rows x "
        f"{record['input_shape']['columns']} columns",
        f"- Analysed shape (after cleaning): {record['cleaned_shape']['rows']} rows x "
        f"{record['cleaned_shape']['columns']} columns",
    ]
    rep = record["cleaning_report"]
    if rep is None:
        lines.append("- Cleaning: none (demo data is used as generated)")
    else:
        lines += [
            "- Cleaning (prepare_numeric_frame):",
            f"  - numeric features used: {', '.join(rep['numeric_columns'])}",
            f"  - dropped as non-numeric: {', '.join(rep['dropped_columns']) or 'none'}",
            f"  - dropped as constant: {', '.join(rep['constant_columns']) or 'none'}",
            f"  - missing cells median-imputed: {rep['imputed_cells']}",
            f"  - duplicate rows in file: {rep['duplicate_rows']}",
        ]
    a = record["analysis"]
    lines += [
        f"- Analysis: PCA ({a['pca_components']} components) + K-means "
        f"(k={a['kmeans_clusters']}, seed={a['kmeans_seed']}, n_init={a['kmeans_n_init']})",
        f"- Explained variance ratio: "
        f"{', '.join(f'{v:.4f}' for v in a['explained_variance_ratio'])}",
        f"- K-means inertia: {a['inertia']:.2f}",
        "- Libraries: " + ", ".join(f"{k} {v}" for k, v in record["libraries"].items()),
        f"- Cleaned data SHA-256: {record['cleaned_data_sha256']}",
        "",
        f"_{record['research_only']}_",
        "",
    ]
    return "\n".join(lines)


def pca_kmeans(df: pd.DataFrame, n_clusters: int = 2, seed: int = 42):
    """PCA (2 components) + K-means on the numeric columns. Returns dict of results."""
    X = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if X.shape[0] < 4 or X.shape[1] < 2:
        raise ValueError(
            f"PCA + K-means needs at least 4 rows and 2 numeric features; "
            f"got {X.shape[0]} rows and {X.shape[1]} features.")
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=seed)
    pcs = pca.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    tsne = None
    if len(X) >= 31:  # t-SNE needs perplexity < n_samples; original crashed below 31
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed,
                    init="pca", learning_rate="auto").fit_transform(X)
    return {
        "pcs": pcs,
        "labels": labels,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cluster_centers_pca": km.cluster_centers_ @ pca.components_.T if False else None,
        "tsne": tsne,
        "inertia": float(km.inertia_),
    }


def synthetic_drug_data(n_samples: int = 400, n_features: int = 10, seed: int = 42):
    """Synthetic drug-response demo data. NOT a real drug-efficacy model.

    Response probability depends on a known linear score plus noise, so a
    classifier can learn it (expected test accuracy ~0.75-0.85) but cannot
    memorise it. Features are named like cheminformatics descriptors.
    """
    rng = np.random.default_rng(seed)
    names = ["mol_weight", "logP", "h_bond_donors", "h_bond_acceptors", "tpsa",
             "rot_bonds", "aromatic_rings", "qed", "solubility", "plasma_protein_binding"]
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    score = X @ w + rng.normal(scale=0.8, size=n_samples)
    y = (score > np.median(score)).astype(int)
    return pd.DataFrame(X, columns=names), pd.Series(y, name="response")


def train_drug_model(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """Train/test split + sklearn MLP (the original's Keras 128->64 shape, in sklearn).

    Returns accuracy and the fitted model. The original fit 100 epochs inside a
    GUI constructor; v2 trains once, on demand, and reports a test score.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                          max_iter=500, random_state=seed, early_stopping=True)
    model.fit(Xtr, y_train)
    acc = float(model.score(Xte, y_test))
    return {"model": model, "scaler": scaler, "test_accuracy": acc,
            "n_train": len(X_train), "n_test": len(X_test),
            "n_iter": int(model.n_iter_)}


def predict_response(model_dict, features):
    x = np.asarray(features, dtype=float).reshape(1, -1)
    x = model_dict["scaler"].transform(x)
    proba = model_dict["model"].predict_proba(x)[0]
    return {"prediction": int(model_dict["model"].predict(x)[0]),
            "p_response": float(proba[1])}

# ---------------------------------------------------------------- DNA
def dna_stats(seq: str):
    """GC content and k-mer counts for a raw DNA string (Bio.Seq, not SeqIO.read on a string)."""
    from Bio.Seq import Seq
    s = Seq(seq.strip().upper())
    n = len(s)
    gc = 100.0 * (s.count("G") + s.count("C")) / n if n else 0.0
    kmers = {}
    for k in (2, 3):
        for i in range(n - k + 1):
            kmer = str(s[i:i + k])
            kmers[kmer] = kmers.get(kmer, 0) + 1
    top_kmers = sorted(kmers.items(), key=lambda kv: -kv[1])[:10]
    return {"length": n, "gc_percent": round(gc, 2),
            "a": int(s.count("A")), "t": int(s.count("T")),
            "g": int(s.count("G")), "c": int(s.count("C")),
            "top_kmers": top_kmers}

# ---------------------------------------------------------------- molecules (Plotly 3D)
# Approximate geometries (angstrom), for the 3D viewer only — not used by VQE.
MOLECULES = {
    "H2": {"symbols": ["H", "H"], "coords": [(0.0, 0.0, -0.37), (0.0, 0.0, 0.37)]},
    "LiH": {"symbols": ["Li", "H"], "coords": [(0.0, 0.0, -0.80), (0.0, 0.0, 0.80)]},
    "H2O": {"symbols": ["O", "H", "H"],
            "coords": [(0.0, 0.0, 0.0), (0.76, 0.59, 0.0), (-0.76, 0.59, 0.0)]},
    "NH3": {"symbols": ["N", "H", "H", "H"],
            "coords": [(0.0, 0.0, 0.0), (0.94, 0.0, -0.35), (-0.47, 0.81, -0.35), (-0.47, -0.81, -0.35)]},
}

# ---------------------------------------------------------------- VQE (PennyLane)
def vqe_energy(symbols, coordinates, max_steps: int = 60, seed: int = 42):
    """Real VQE with PennyLane. Returns dict with vqe_energy, exact_energy, error.

    Uses qml.qchem.molecular_hamiltonian + UCCSD ansatz + Adam. The original's
    PyQuil QuantumAlgorithm object does not exist; this is the working equivalent.
    `coordinates` are in bohr for the Hamiltonian.
    """
    import pennylane as qml
    from pennylane import numpy as pnp
    np.random.seed(seed)
    # PennyLane 0.41 requires coordinates as a numpy array (list of tuples
    # raises AttributeError: 'list' object has no attribute 'flatten').
    coordinates = np.array(coordinates, dtype=float)
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    electrons = {"H2": 2, "LiH": 4, "H2O": 10, "NH3": 10}
    n_elec = electrons.get("".join(symbols), 2)
    hf_state = qml.qchem.hf_state(n_elec, qubits)
    singles, doubles = qml.qchem.excitations(n_elec, qubits)
    # PennyLane 0.41: excitations must be mapped to wire pairs first —
    # passing raw doubles ([i,j,k,l]) to UCCSD raises ValueError.
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev)
    def circuit(params):
        qml.UCCSD(params, wires=range(qubits), s_wires=s_wires,
                  d_wires=d_wires, init_state=hf_state)
        return qml.expval(H)

    params = pnp.zeros(len(singles) + len(doubles), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    energy = float(circuit(params))
    for _ in range(max_steps):
        params, energy = opt.step_and_cost(circuit, params)
        energy = float(energy)
    exact = float(np.linalg.eigvalsh(qml.matrix(H))[0])
    return {"vqe_energy": energy, "exact_energy": exact,
            "error": abs(energy - exact), "n_qubits": qubits, "n_electrons": n_elec}


def h2_bond_scan(bond_lengths_bohr=(0.8, 1.0, 1.4, 1.8, 2.2, 2.8), max_steps: int = 40):
    """VQE bond-length scan for H2. Coordinates in bohr; H-H distance = bond length."""
    out = []
    for r in bond_lengths_bohr:
        coords = [(0.0, 0.0, -r / 2), (0.0, 0.0, r / 2)]
        res = vqe_energy(["H", "H"], coords, max_steps=max_steps)
        out.append({"bond_length_bohr": r, **res})
    return out


def lih_energy(max_steps: int = 60):
    # LiH equilibrium ~3.0 bohr
    coords = [(0.0, 0.0, -1.5), (0.0, 0.0, 1.5)]
    return vqe_energy(["Li", "H"], coords, max_steps=max_steps)
