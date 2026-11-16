"""GENEIE v2 fast test suite — core cleaning + the actual Dash callbacks.

No server, no VQE: this imports app.py and calls its callback functions
directly, the way the v2.1.0 upload-fix build was verified. It covers the
upload cleaning cases, the provenance wiring (v2.2.0), and the research-only
banner's presence in the layout. The full suite (with VQE) is
test_headless.py. Run:
    python test_fast.py
Exits non-zero if any check fails.
"""
import base64
import io
import sys

import numpy as np
import pandas as pd
from dash import no_update

import geneie_core as core

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def _walk(comp, ids, texts):
    cid = getattr(comp, "id", None)
    if cid:
        ids.add(cid)
    ch = getattr(comp, "children", None)
    if isinstance(ch, str):
        texts.append(ch)
    elif isinstance(ch, (list, tuple)):
        for c in ch:
            _walk(c, ids, texts)
    elif ch is not None:
        _walk(ch, ids, texts)


def main():
    # ---- core: upload cleaning (mirrors test_headless.py section 1b)
    df, y_true = core.synthetic_genomic_data()
    clean, rep = core.prepare_numeric_frame(df)
    check("core: numeric features kept, text cols dropped",
          clean.shape == (120, 12)
          and set(rep["dropped_columns"]) == {"sample_id", "gene"},
          f"clean={clean.shape}")
    from sklearn.metrics import adjusted_rand_score
    res_u = core.pca_kmeans(clean)
    check("core: PCA/K-means on upload recovers clusters",
          adjusted_rand_score(y_true, res_u["labels"]) > 0.5)
    messy = df.copy()
    messy["const"] = 3.14
    messy.loc[0, "feature_1"] = np.nan
    clean_m, rep_m = core.prepare_numeric_frame(messy)
    check("core: constant dropped, missing imputed and counted",
          rep_m["constant_columns"] == ["const"] and rep_m["imputed_cells"] >= 1
          and bool(np.isfinite(clean_m.to_numpy()).all()))
    try:
        core.prepare_numeric_frame(df.head(2))
        check("core: rejects too few rows", False)
    except ValueError:
        check("core: rejects too few rows", True)
    try:
        core.prepare_numeric_frame(df[["sample_id", "gene"]].assign(only=[1.0] * 120))
        check("core: rejects <2 numeric features", False)
    except ValueError:
        check("core: rejects <2 numeric features", True)

    # ---- app import (trains the demo drug model once, as the server does)
    import app as geneie_app

    # ---- layout: research-only banner + provenance UI present
    ids, texts = set(), []
    _walk(geneie_app.app.layout, ids, texts)
    joined = " ".join(texts)
    check("layout: research-only banner present and explicit",
          "research-only-banner" in ids and "RESEARCH USE ONLY" in joined
          and "not clinical diagnosis" in joined)
    check("layout: research-only footer present",
          "research-only-footer" in ids
          and "not clinical diagnosis" in core.RESEARCH_ONLY_NOTICE)
    check("layout: provenance panel, store and download buttons present",
          {"provenance-panel", "provenance-store",
           "prov-json-btn", "prov-md-btn"} <= ids)

    # ---- initial (demo) provenance record seeded at startup
    rec0 = geneie_app._INITIAL_PROV
    check("provenance: initial demo record seeded",
          rec0["dataset"]["kind"] == "demo"
          and rec0["app_version"] == core.APP_VERSION
          and len(rec0["cleaned_data_sha256"]) == 64,
          f"hash={rec0['cleaned_data_sha256_short']}")

    # ---- upload callback drives provenance (demo-shaped CSV)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    contents = "data:text/csv;base64," + base64.b64encode(csv_bytes).decode()
    out = geneie_app.upload(contents, "demo-shaped.csv")
    status, _tdata, _tcols, _fig, stored, _panel, prov = out
    check("callback upload: provenance recorded for the upload",
          prov["dataset"] == {"kind": "upload", "filename": "demo-shaped.csv"}
          and prov["input_shape"] == {"rows": 120, "columns": 14}
          and prov["cleaned_shape"] == {"rows": 120, "columns": 12}
          and set(prov["cleaning_report"]["dropped_columns"]) == {"sample_id", "gene"},
          f"status starts: {status[:40]}")
    check("callback upload: cleaning report kept with stored data",
          stored.get("report", {}).get("imputed_cells") == rep["imputed_cells"])

    # ---- upload callback failure leaves provenance untouched
    bad = "data:text/csv;base64," + base64.b64encode(
        df.head(2).to_csv(index=False).encode("utf-8")).decode()
    out_bad = geneie_app.upload(bad, "tiny.csv")
    check("callback upload: bad CSV fails cleanly, provenance untouched",
          out_bad[0].startswith("Upload failed")
          and out_bad[5] is no_update and out_bad[6] is no_update)

    # ---- Re-run callback: provenance for upload-active and demo runs
    _f, _s, _p, prov_rerun_up = geneie_app.rerun_pca(1, stored)
    check("callback rerun: upload run keeps filename + cleaning report",
          prov_rerun_up["dataset"] == {"kind": "upload",
                                       "filename": "demo-shaped.csv"}
          and prov_rerun_up["cleaning_report"] is not None
          and prov_rerun_up["input_shape"] == {"rows": 120, "columns": 14})
    _f, _s, _p, prov_rerun_demo = geneie_app.rerun_pca(1, None)
    check("callback rerun: demo run recorded as demo",
          prov_rerun_demo["dataset"] == {"kind": "demo", "filename": None}
          and prov_rerun_demo["cleaning_report"] is None)

    # ---- provenance downloads carry the record + the notice
    dl = geneie_app.download_provenance_json(1, prov)
    import json as _json
    parsed = _json.loads(dl["content"])
    check("download: JSON carries the full record",
          parsed["cleaned_data_sha256"] == prov["cleaned_data_sha256"]
          and "Research use only" in parsed["research_only"]
          and dl["filename"].endswith(".json"))
    dl_md = geneie_app.download_provenance_md(1, prov)
    check("download: Markdown carries hash + research-only notice",
          prov["cleaned_data_sha256"] in dl_md["content"]
          and "Research use only" in dl_md["content"]
          and dl_md["filename"].endswith(".md"))

    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
