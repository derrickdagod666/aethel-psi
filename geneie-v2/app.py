"""GENEIE v2 — Dash + Plotly web app (replaces the PyQt5 desktop GUI).

Change vs the original, per Putt's rule: the original was a PyQt5 QMainWindow whose
buttons did nothing (all analysis methods were `pass`), whose VR/AR engines were
imports of packages that do not exist, and which trained a model inside the window
constructor. v2 is a browser app: every button runs real code (see geneie_core.py),
training happens on demand, and VR/AR are dropped and listed as roadmap only.
Run: python app.py  (then open http://127.0.0.1:8050)
"""
import base64
import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update

import geneie_core as core

app = Dash(__name__, title=f"GENEIE v{core.APP_VERSION}")
server = app.server

# Pre-train the drug model once at startup on synthetic data (labelled as such in the UI).
_DRUG_X, _DRUG_Y = core.synthetic_drug_data()
_DRUG_MODEL = core.train_drug_model(_DRUG_X, _DRUG_Y)
_GEN_DF, _GEN_Y = core.synthetic_genomic_data()

TAB_STYLE = {"padding": "20px"}

def _pca_fig(df, source="demo data"):
    res = core.pca_kmeans(df)
    pcs, labels = res["pcs"], res["labels"]
    fig = px.scatter(x=pcs[:, 0], y=pcs[:, 1], color=labels.astype(str),
                     labels={"x": "PC1", "y": "PC2", "color": "K-means cluster"},
                     title=f"PCA + K-means (2 clusters, {source}) — variance explained "
                           f"PC1 {res['explained_variance_ratio'][0]:.2f}, "
                           f"PC2 {res['explained_variance_ratio'][1]:.2f}")
    return fig, res

def _analysed_frame(df):
    """The exact numeric frame pca_kmeans consumes (it selects numeric dtypes
    itself) — provenance hashes and measures THIS frame, so the record always
    describes the data the analysis really ran on."""
    return df.select_dtypes(include=[np.number])

def _run_provenance(kind, filename, input_shape, df, cleaning_report, res):
    return core.build_provenance(
        dataset_kind=kind, filename=filename, input_shape=input_shape,
        analysed_frame=_analysed_frame(df), cleaning_report=cleaning_report,
        analysis_result=res)

def _provenance_children(rec):
    """Visible summary of a provenance record + the full record in a details
    fold. The short line is what a reader sees; the JSON is what they can
    verify or file with the result."""
    if not rec:
        return "No analysis run recorded yet this session."
    d = rec["dataset"]
    ds = ("built-in synthetic demo data" if d["kind"] == "demo"
          else f"upload: {d['filename']}")
    a = rec["analysis"]
    summary = (f"Run {rec['cleaned_data_sha256_short']} · {rec['timestamp_utc']} · "
               f"GENEIE v{rec['app_version']} · dataset: {ds} · "
               f"input {rec['input_shape']['rows']}x{rec['input_shape']['columns']} -> "
               f"analysed {rec['cleaned_shape']['rows']}x{rec['cleaned_shape']['columns']} · "
               f"PCA {a['pca_components']} components, K-means k={a['kmeans_clusters']} "
               f"(seed {a['kmeans_seed']}) · SHA-256 {rec['cleaned_data_sha256']}")
    return html.Div([
        html.P(summary),
        html.Details([html.Summary("Full provenance record (JSON)"),
                      html.Pre(json.dumps(rec, indent=2, default=str))]),
    ])

# The initial view is itself a run (demo data), so it gets a record too —
# computed once here and used for the initial figure, panel, and Store.
_INITIAL_FIG, _INITIAL_RES = _pca_fig(_GEN_DF)
_INITIAL_PROV = _run_provenance("demo", None, _GEN_DF.shape, _GEN_DF, None,
                                _INITIAL_RES)

def _mol_fig(name):
    mol = core.MOLECULES[name]
    xs = [c[0] for c in mol["coords"]]; ys = [c[1] for c in mol["coords"]]; zs = [c[2] for c in mol["coords"]]
    fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode="markers+text",
                                       text=mol["symbols"], textposition="top center",
                                       marker=dict(size=12))])
    fig.update_layout(title=f"{name} — atom positions (angstrom, schematic, not a VR render)",
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    return fig

genomic_tab = html.Div([
    html.H3("Genomic data — synthetic demo (NOT real patient data)"),
    html.P("Upload a CSV or use the built-in synthetic demo table (120 samples, 12 features, "
           "two planted clusters). On upload, PCA + K-means re-run on YOUR data: the table "
           "and the graph below switch to the upload, and the status line reports exactly "
           "what was used. Rules: numeric feature columns only (text/ID columns are dropped "
           "and named), at least 4 rows and 2 varying numeric features, missing cells are "
           "median-imputed and counted, constant columns are dropped and named. Uploads are "
           "processed in memory only — nothing is saved to disk or sent anywhere. "
           "Limits: 5 MB, 5,000 rows."),
    dcc.Upload(id="upload-csv", children=html.Button("Upload CSV"), multiple=False),
    dcc.Store(id="analysis-data"),
    html.Div(id="upload-status"),
    dash_table.DataTable(id="genomic-table", page_size=10,
                         data=_GEN_DF.head(50).to_dict("records"),
                         columns=[{"name": c, "id": c} for c in _GEN_DF.columns],
                         filter_action="native", sort_action="native"),
    dcc.Graph(id="pca-graph", figure=_INITIAL_FIG),
    html.Button("Re-run PCA + K-means", id="rerun-pca", n_clicks=0),
    html.Div(id="pca-status"),
    html.H4("Run provenance"),
    html.P("Every analysis run (demo, upload, or Re-run) records what it ran on: "
           "dataset, shapes before/after cleaning, the cleaning report, parameters, "
           "library versions, and a SHA-256 of the analysed data, so a run can be "
           "identified and repeated. Kept in session memory only."),
    dcc.Store(id="provenance-store", data=_INITIAL_PROV),
    html.Div(id="provenance-panel", children=_provenance_children(_INITIAL_PROV)),
    html.Button("Download provenance (.json)", id="prov-json-btn", n_clicks=0),
    html.Button("Download provenance (.md)", id="prov-md-btn", n_clicks=0),
    dcc.Download(id="prov-json-download"),
    dcc.Download(id="prov-md-download"),
], style=TAB_STYLE)

drug_tab = html.Div([
    html.H3("Drug response prediction — synthetic demo model"),
    html.P(f"sklearn MLP (128 -> 64, the original's Keras shape, in sklearn), trained on "
           f"synthetic data. Test accuracy: {_DRUG_MODEL['test_accuracy']:.3f} "
           f"(n_test={_DRUG_MODEL['n_test']}). This is NOT a real drug-efficacy model — "
           "no real compounds, no clinical claims."),
    html.P("Enter the 10 descriptor values (or leave the defaults) and predict:"),
    html.Div([dcc.Input(id=f"feat-{i}", type="number", value=0.0, step=0.1,
                        style={"width": "90px", "margin": "3px"})
              for i in range(10)]),
    html.Br(), html.Button("Predict response", id="predict-btn", n_clicks=0),
    html.Div(id="predict-out"),
], style=TAB_STYLE)

vqe_tab = html.Div([
    html.H3("VQE drug discovery — PennyLane, simulator only"),
    html.P("Real VQE (UCCSD + Adam) on H2 / LiH, compared with exact diagonalisation of "
           "the same Hamiltonian. Click to run — H2 takes ~10-30 s, LiH longer."),
    dcc.Dropdown(id="vqe-mol", options=[{"label": m, "value": m} for m in ("H2", "LiH")],
                 value="H2"),
    html.Button("Run VQE", id="vqe-btn", n_clicks=0),
    html.Button("Run H2 bond scan", id="scan-btn", n_clicks=0),
    html.Div(id="vqe-out"),
    dcc.Graph(id="vqe-graph"),
], style=TAB_STYLE)

dna_tab = html.Div([
    html.H3("DNA sequence"),
    html.P("Paste a DNA sequence (or use a demo fragment). GC content and top k-mers "
           "are computed with Bio.Seq — the original called SeqIO.read() on a string, "
           "which raises; v2 does not."),
    dcc.Dropdown(id="dna-demo", options=[{"label": k, "value": k} for k in core.DEMO_SEQUENCES],
                 value="BRCA1-fragment"),
    dcc.Textarea(id="dna-input", value=core.DEMO_SEQUENCES["BRCA1-fragment"],
                 style={"width": "100%", "height": 80}),
    html.Button("Analyse", id="dna-btn", n_clicks=0),
    html.Div(id="dna-out"),
], style=TAB_STYLE)

mol_tab = html.Div([
    html.H3("3D molecule viewer — Plotly 3D (not PyVista VR)"),
    dcc.Dropdown(id="mol-pick", options=[{"label": m, "value": m} for m in core.MOLECULES],
                 value="H2O"),
    dcc.Graph(id="mol-graph", figure=_mol_fig("H2O")),
], style=TAB_STYLE)

# Research-only boundary (v2.2.0): a persistent banner, always visible above
# the tabs — not a modal, not dismissible — plus the same line as a footer.
# The full notice also rides inside every provenance record (geneie_core).
RESEARCH_BANNER = html.Div(
    [html.Strong("RESEARCH USE ONLY — "),
     "GENEIE is a research tool. Its outputs are hypotheses for research use: "
     "not clinical diagnosis, not treatment advice, and not therapeutic "
     "recommendations. Demo data is synthetic; uploaded data is processed in "
     "memory only and is never saved."],
    id="research-only-banner",
    style={"border": "2px solid #8a6d00", "backgroundColor": "#fff8e1",
           "color": "#4a3a00", "padding": "10px 14px", "marginBottom": "12px",
           "borderRadius": "4px"})

app.layout = html.Div([
    RESEARCH_BANNER,
    html.H1(f"GENEIE v{core.APP_VERSION}"),
    html.P("Genomic research + drug discovery demo. Synthetic demo data only. "
           "No real patient data, no clinical claims, simulator only. "
           "Federated learning, blockchain, Clara/Parabricks, DeepVariant, Riva, "
           "FHIR, Kinesis, VR/AR and QGAN drug generation are roadmap items, not features — "
           "see README.md."),
    dcc.Tabs([
        dcc.Tab(label="Genomic data", children=[genomic_tab]),
        dcc.Tab(label="Drug response", children=[drug_tab]),
        dcc.Tab(label="VQE", children=[vqe_tab]),
        dcc.Tab(label="DNA sequence", children=[dna_tab]),
        dcc.Tab(label="3D molecule", children=[mol_tab]),
    ]),
    html.Hr(),
    html.P(core.RESEARCH_ONLY_NOTICE, id="research-only-footer",
           style={"fontSize": "0.85em", "color": "#555"}),
])

@app.callback(Output("pca-graph", "figure"), Output("pca-status", "children"),
              Output("provenance-panel", "children"),
              Output("provenance-store", "data"),
              Input("rerun-pca", "n_clicks"), State("analysis-data", "data"))
def rerun_pca(n, stored):
    if not n:
        return no_update, "", no_update, no_update
    # Re-run on the uploaded data when an upload is active, else the demo data.
    # Before the 2026-11-16 fix this always re-ran on the demo frame and never
    # updated the graph at all — only the status text changed.
    if stored and stored.get("records"):
        df = pd.DataFrame(stored["records"])
        filename = stored.get("filename", "uploaded CSV")
        source = f"upload: {filename}"
        # The upload's cleaning report was kept with the stored data, so a
        # Re-run's provenance still shows how the data was cleaned; the raw
        # input shape comes from that report when present.
        report = stored.get("report")
        input_shape = ((report["n_rows_input"], report["n_columns_input"])
                       if report else df.shape)
        fig, res = _pca_fig(df, source=source)
        prov = _run_provenance("upload", filename, input_shape, df, report, res)
    else:
        df, source = _GEN_DF, "demo data"
        fig, res = _pca_fig(df, source=source)
        prov = _run_provenance("demo", None, _GEN_DF.shape, df, None, res)
    return fig, (f"Re-ran PCA + K-means on {source}: "
                 f"PC1 {res['explained_variance_ratio'][0]:.2f}, "
                 f"PC2 {res['explained_variance_ratio'][1]:.2f}, inertia {res['inertia']:.1f}."), \
        _provenance_children(prov), prov

@app.callback(Output("predict-out", "children"), Input("predict-btn", "n_clicks"),
              [State(f"feat-{i}", "value") for i in range(10)])
def predict(n, *vals):
    if not n:
        return ""
    vals = [0.0 if v is None else v for v in vals]
    r = core.predict_response(_DRUG_MODEL, vals)
    return (f"Prediction: {'response' if r['prediction'] else 'no response'} "
            f"(P(response) = {r['p_response']:.3f}). Synthetic demo model — not a clinical result.")

@app.callback(Output("vqe-out", "children"), Output("vqe-graph", "figure"),
              Input("vqe-btn", "n_clicks"), Input("scan-btn", "n_clicks"),
              State("vqe-mol", "value"))
def run_vqe(n1, n2, mol):
    from dash import ctx
    if not ctx.triggered_id:
        return "", go.Figure()
    if ctx.triggered_id == "scan-btn":
        scan = core.h2_bond_scan(max_steps=40)
        fig = px.line(x=[s["bond_length_bohr"] for s in scan],
                      y=[s["vqe_energy"] for s in scan], markers=True,
                      labels={"x": "H-H distance (bohr)", "y": "Energy (Ha)"},
                      title="H2 VQE bond scan (vs exact: errors shown in text)")
        txt = " | ".join(f"r={s['bond_length_bohr']}: E={s['vqe_energy']:.4f} "
                         f"(exact {s['exact_energy']:.4f}, err {s['error']:.4f})" for s in scan)
        return txt, fig
    if mol == "H2":
        coords = [(0.0, 0.0, -0.7), (0.0, 0.0, 0.7)]
        res = core.vqe_energy(["H", "H"], coords, max_steps=60)
    else:
        res = core.lih_energy(max_steps=60)
    return (f"{mol}: VQE E = {res['vqe_energy']:.4f} Ha, exact = {res['exact_energy']:.4f} Ha, "
            f"|error| = {res['error']:.4f} Ha, qubits = {res['n_qubits']}, "
            f"electrons = {res['n_electrons']}."), go.Figure()

@app.callback(Output("dna-out", "children"), Input("dna-btn", "n_clicks"),
              State("dna-input", "value"))
def analyse_dna(n, seq):
    if not n or not seq:
        return ""
    s = core.dna_stats(seq)
    kmers = ", ".join(f"{k}:{v}" for k, v in s["top_kmers"][:6])
    return (f"Length {s['length']} bp, GC {s['gc_percent']}%, "
            f"A {s['a']} T {s['t']} G {s['g']} C {s['c']}. Top k-mers: {kmers}.")

@app.callback(Output("mol-graph", "figure"), Input("mol-pick", "value"))
def pick_mol(name):
    return _mol_fig(name)

@app.callback(Output("dna-input", "value"), Input("dna-demo", "value"))
def load_demo(name):
    return core.DEMO_SEQUENCES[name]

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_ROWS = 5000

@app.callback(Output("upload-status", "children"),
              Output("genomic-table", "data"), Output("genomic-table", "columns"),
              Output("pca-graph", "figure"), Output("analysis-data", "data"),
              Output("provenance-panel", "children"),
              Output("provenance-store", "data"),
              Input("upload-csv", "contents"), State("upload-csv", "filename"))
def upload(contents, filename):
    """Upload fix (2026-11-16): an upload now drives the whole tab.

    Before: the CSV was parsed, a status line admitted the table still showed
    demo data, and PCA never touched the upload. Now: the CSV is cleaned by
    core.prepare_numeric_frame (numeric columns kept, text/constant columns
    dropped and named, missing cells median-imputed and counted), PCA +
    K-means run on the cleaned upload, the table switches to the uploaded
    rows, the graph switches to the upload's clusters, and the cleaned
    numeric frame is kept in a session Store so 'Re-run PCA + K-means'
    re-runs on the upload, not the demo. Processed in memory only; the file
    is never written to disk. On any failure the demo view stays in place
    and the status line says why — nothing silently falls back.
    """
    if not contents:
        return "", no_update, no_update, no_update, no_update, no_update, no_update
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError(f"File is {len(raw)/1e6:.1f} MB; the limit is 5 MB.")
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
        if len(df) > MAX_UPLOAD_ROWS:
            raise ValueError(f"CSV has {len(df)} rows; the limit is {MAX_UPLOAD_ROWS}.")
        clean, report = core.prepare_numeric_frame(df)
        fig, res = _pca_fig(clean, source=f"upload: {filename}")
        table_df = df.head(50).astype(object).where(pd.notnull(df.head(50)), None)
        status = (
            f"Uploaded {filename}: {report['n_rows_input']} rows, "
            f"{report['n_columns_input']} columns. PCA + K-means ran on YOUR upload — "
            f"{report['n_rows_used']} rows x {len(report['numeric_columns'])} numeric features "
            f"({', '.join(report['numeric_columns'])}). "
            f"PC1 {res['explained_variance_ratio'][0]:.2f}, "
            f"PC2 {res['explained_variance_ratio'][1]:.2f}. "
            f"Dropped as non-numeric: {', '.join(report['dropped_columns']) or 'none'}; "
            f"dropped as constant: {', '.join(report['constant_columns']) or 'none'}; "
            f"missing cells median-imputed: {report['imputed_cells']}; "
            f"duplicate rows in file: {report['duplicate_rows']}. "
            f"Table (first 50 rows) and graph now show your data, not the demo.")
        # The cleaning report is stored alongside the cleaned records so a
        # later Re-run's provenance can still show how this data was cleaned.
        stored = {"filename": filename,
                  "records": clean.to_dict("records"),
                  "report": report}
        prov = _run_provenance("upload", filename, df.shape, clean, report, res)
        return (status, table_df.to_dict("records"),
                [{"name": c, "id": c} for c in df.columns], fig, stored,
                _provenance_children(prov), prov)
    except Exception as e:  # noqa: BLE001 - surface the error to the user
        return (f"Upload failed: {e} The demo data is still shown.",
                no_update, no_update, no_update, no_update, no_update, no_update)

@app.callback(Output("prov-json-download", "data"),
              Input("prov-json-btn", "n_clicks"),
              State("provenance-store", "data"), prevent_initial_call=True)
def download_provenance_json(n, rec):
    if not n or not rec:
        return no_update
    return dcc.send_string(
        json.dumps(rec, indent=2, default=str),
        filename=f"geneie-provenance-{rec['cleaned_data_sha256_short']}.json")

@app.callback(Output("prov-md-download", "data"),
              Input("prov-md-btn", "n_clicks"),
              State("provenance-store", "data"), prevent_initial_call=True)
def download_provenance_md(n, rec):
    if not n or not rec:
        return no_update
    return dcc.send_string(
        core.provenance_markdown(rec),
        filename=f"geneie-provenance-{rec['cleaned_data_sha256_short']}.md")

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
