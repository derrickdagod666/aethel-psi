#!/usr/bin/env python3
"""Architects of Reality: UNIFIED QCE v2 — one app, not two.

Merges the Cognitive Companion v2 and the Syntropic Swarm v2 (Master QCE)
into a single Gradio app with five tabs:

  1. QCE Chat        — RAG synthesis over Putt's papers (LM Studio first,
                       labelled extractive demo fallback), sources cited,
                       measured telemetry only.
  2. 3D Manifold     — Plotly PCA of REAL embeddings only (static corpus
                       chunks, swarm ledger vectors, live query). No random
                       dummy vectors; absent layers are labelled absent.
  3. Quantum Lab     — 3-qubit collapse on the Aer simulator (real counts
                       + Shannon entropy), and the VQE compiler: V(Psi)
                       expressions are COMPILED for real (SymPy spectral
                       mapping, or Pauli-term syntax). Named presets from
                       the originals are available and labelled as presets.
  4. Biometrics      — simulated HRV demo (labelled) + real WebBluetooth
                       path (limits stated); BrainFlow EEG, synthetic board
                       labelled simulated, real Muse 2 path documented.
                       Bio score from a 30-beat RMSSD window (documented
                       heuristic, not entropy).
  5. Swarm Ledger & Billing — ledger persisted to disk, opt-in default
                       OFF, local only; tiers with real meanings; Stripe
                       disabled by default (no keys, no charges).

What changed vs each original (per Putt's rule, elaborated here):
  vs the Cognitive Companion original:
    - The fake "Cognitive Entropy" gauge (0.95, -0.005 per token) and the
      "+1.5 Syntropic Resonance" counter are GONE. Telemetry is measured:
      top-chunk L2 distance, chunks retrieved, latency, words streamed.
    - Tiers are carried by id (tier1/tier2/tier3), not string-matched
      labels, and they now mean something real: top_k 2/3/5 passages,
      answer depth, and quantum/VQE access on Tier 3 only.
    - The upgrade button no longer POSTs to a missing backend with a
      hardcoded demo email; billing is disabled by default (billing.py).
    - FAISS index is built from Putt's own papers (Third Revised Edition
      + UAF paper), not assumed to exist.
  vs the Syntropic Swarm unified original:
    - IBM link is simulator-first; the hardcoded mock counts
      {'000': 512, '111': 512} presented as "IBM QUANTUM EXECUTION" are
      gone. IBM Runtime needs saved credentials AND AOR_ENABLE_IBM=1.
    - VQE "compiler" now REALLY compiles: SymPy parses V(Psi) and maps
      Psi -> Pauli Z on qubit 0 (spectral mapping, exact because Z**2=I),
      or parses Pauli-term syntax. See vqe.py. The two fixed Hamiltonians
      survive as named presets, labelled as presets.
    - 3D visualizer no longer plots np.random.rand(768) dummy vectors.
    - EEG band indexing fixed; theta/alpha ratio is a relaxation proxy,
      not "neural entropy"; synthetic board labelled simulated; stop added.
    - HRV is a 30-beat RMSSD window, not one RR interval's distance from
      1000 ms. Single-interval "bio-entropy" is gone.
    - Ledger persisted to disk, opt-in default OFF, local only; the
      0.4 commit / 1.5 retrieval thresholds are labelled arbitrary.
    - Server binds 127.0.0.1 only (original bound 0.0.0.0:7860).
    - Two separate v2 apps (companion on 7860, swarm on 7861) become ONE
      app on 7862. Nothing is duplicated: one engine, one index, one
      ledger, one billing stub.

Run:  python app.py        (serves on http://127.0.0.1:7862)
Env:  AOR_QUANTUM_MODE=simulator|ibm|off (default simulator)
      AOR_ENABLE_IBM=1   (required, with saved IBM credentials, for ibm)
      AOR_LLM_URL / AOR_LLM_MODEL (LM Studio endpoint, defaults
      http://localhost:1234/v1, local-model)
      AOR_BILLING_ENABLED=1 + AOR_BACKEND_URL (only to enable real billing)
"""
import os

# Environment fix (this VM): the sandbox sets no_proxy with bracketed IPv6
# entries like [fd8b:4f84:7d32:99::1], which httpx — imported by Gradio at
# import time — cannot parse ("Invalid port: ':1]'"), so `import gradio`
# crashes before the app even starts. The bare IPv6 leftovers also break
# Gradio's launch-time localhost check (httpx.head goes through the
# egress proxy and fails). This app is local-only, so no_proxy is set to
# exactly localhost + 127.0.0.1. The http(s)_proxy vars are untouched
# (they are only needed for downloads, and the embedding model is cached
# after the first run anyway).
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import gradio as gr

from billing import get_checkout_url
from engine import SwarmEngine
from qce import TIERS
from vqe import compile_expression, run_vqe, select_hamiltonian

engine = SwarmEngine()
LAST_QUERY_VEC = {"vec": None}

MODE_BANNER = (
    "🟢 **LLM mode:** LM Studio reachable — answers are synthesised by the "
    "local model, grounded in retrieved passages from Putt's papers."
    if engine.qce.llm_available else
    "🟡 **DEMO MODE — extractive only:** LM Studio is not reachable "
    f"({engine.qce.llm_status}). Answers are assembled ONLY from retrieved "
    "passages of Putt's papers, with sources. Start LM Studio with a model "
    "loaded and restart the app for full LLM synthesis."
)

# WebBluetooth JS — kept verbatim from the original. Limits (see README):
# Chrome/Edge only, HTTPS or localhost, user gesture required; a BLE chest
# strap is the realistic device, Apple Watch HRV is not reliably readable
# from a web page.
js_bluetooth = """
async function connectBiometrics() {
    try {
        const device = await navigator.bluetooth.requestDevice({ filters: [{ services: ['heart_rate'] }] });
        const server = await device.gatt.connect();
        const service = await server.getPrimaryService('heart_rate');
        const characteristic = await service.getCharacteristic('heart_rate_measurement');
        await characteristic.startNotifications();
        characteristic.addEventListener('characteristicvaluechanged', (event) => {
            const value = event.target.value;
            const flags = value.getUint8(0);
            let index = 1;
            if (flags & 0x01) index += 2; else index += 1;
            let rrInterval = 0;
            if (flags & 0x10) rrInterval = (value.getUint16(index, true) / 1024.0) * 1000.0;
            const bioInput = document.querySelector('#bio-payload textarea');
            if (bioInput) {
                bioInput.value = JSON.stringify({ rr: rrInterval });
                bioInput.dispatchEvent(new Event('input', { bubbles: true }));
            }
        });
        return "Link Active";
    } catch (e) { return "Link Failed"; }
}
"""

theme = gr.themes.Monochrome(
    primary_hue="slate",
    font=[gr.themes.GoogleFont("Noto Sans"), "ui-sans-serif", "system-ui"],
)

TIER_MEANINGS = "\n".join(
    f"- **{t['label']}** — {t['features']}"
    for t in TIERS.values()
)

with gr.Blocks(theme=theme, title="QCE — Unified Architects of Reality v2") as app:
    status_banner = gr.Markdown(visible=False)
    gr.Markdown(
        "# 🌌 Architects of Reality: Quantum Cognitive Engine \n"
        "Unified QCE v2 — Cognitive Companion + Syntropic Swarm in one app "
        "(simulator-first, local-only)"
    )
    gr.Markdown(MODE_BANNER)
    gr.Markdown(
        f"**Quantum mode:** {engine.quantum.mode} "
        f"(`{engine.quantum.backend_name}`) — {engine.quantum.status}"
    )

    with gr.Tabs():
        # ---- Tab 1: QCE Chat -------------------------------------------
        with gr.Tab("QCE Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=500,
                                         label="QCE Synthesis Terminal",
                                         type="messages")
                    with gr.Row():
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Enter a physics query, or "
                            "V(Psi) = cos(Psi) (Tier 3 compiles it)...",
                            scale=4)
                        submit_btn = gr.Button("Initialize Synthesis",
                                               variant="primary", scale=1)
                    tier_dropdown = gr.Dropdown(
                        choices=[(t["label"], tid)
                                 for tid, t in TIERS.items()],
                        value="tier2", label="Active Tier",
                        interactive=True)
                    gr.Markdown(
                        "Tiers change what you get, not just the prompt: "
                        "Tier 1 retrieves 2 passages, Tier 2 retrieves 3 "
                        "with derivations, Tier 3 retrieves 5 and unlocks "
                        "the quantum collapse and VQE compiler intercepts. "
                        "See the Swarm Ledger & Billing tab for details.")
                with gr.Column(scale=2):
                    gr.Markdown("### 🎛️ Telemetry (measured)")
                    top_dist = gr.Number(value=None,
                                         label="Top-chunk L2 distance",
                                         interactive=False)
                    chunks_n = gr.Number(value=0, label="Chunks retrieved",
                                         interactive=False)
                    latency = gr.Number(value=None,
                                        label="Response latency (seconds)",
                                        interactive=False)
                    words_n = gr.Number(value=0, label="Words streamed",
                                        interactive=False)
                    mode_box = gr.Textbox(
                        value="LLM" if engine.qce.llm_available
                        else "DEMO (extractive)",
                        label="Mode", interactive=False)
                    vqe_box = gr.Number(value=None,
                                        label="VQE ground state (Hartree)",
                                        interactive=False)
                    ledger_commit_box = gr.Textbox(
                        value="", label="Ledger commit", interactive=False)
                    gr.Markdown(
                        "No 'Cognitive Entropy' gauge and no resonance "
                        "counter here: the originals faked both. Every "
                        "number above is measured from the actual run.")

        # ---- Tab 2: 3D Manifold -----------------------------------------
        with gr.Tab("3D Manifold"):
            plot_3d = gr.Plot(label="Topological Swarm Manifold "
                                    "(real embeddings)")
            refresh_plot = gr.Button("Refresh manifold", variant="secondary")
            gr.Markdown(
                "Blue = static primer passages (real embeddings of Putt's "
                "papers), green = swarm ledger, amber = your latest query. "
                "Layers with no data are simply absent — the original "
                "plotted random dummy vectors for the static layer; this "
                "app never does.")

        # ---- Tab 3: Quantum Lab ------------------------------------------
        with gr.Tab("Quantum Lab"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚛️ Quantum collapse (3-qubit circuit)")
                    gr.Markdown(
                        "The original's circuit (H, CX, CX, RZ(phase·π)) "
                        "on the configured backend. Simulator by default "
                        "(Aer, local, free, 1024 shots, real counts + "
                        "Shannon entropy). IBM Runtime only with saved "
                        "credentials AND `AOR_ENABLE_IBM=1`; OFF disables it.")
                    quantum_btn = gr.Button(
                        "Run 3-qubit collapse", variant="secondary")
                    quantum_out = gr.Markdown("")
                with gr.Column():
                    gr.Markdown("### 🧮 VQE compiler — expressions compiled "
                                "for real")
                    gr.Markdown(
                        "Type an expression. Psi-function syntax "
                        "(`cos(Psi)`, `Psi**2 + 0.5*Psi`, `exp(Psi)`) is "
                        "compiled with Psi encoded as Pauli Z on qubit 0 "
                        "(exact spectral mapping: f(Z) = a·I + b·Z). "
                        "Pauli-term syntax (`0.5*ZZ + 1.0*IX`) is parsed "
                        "term-by-term. Tier 3 only in chat; this tab works "
                        "for any tier.")
                    vqe_expr = gr.Textbox(
                        value="cos(Psi)",
                        label="V(Psi) = ... (expression only, no 'V(Psi) =')")
                    vqe_btn = gr.Button("Compile & run VQE",
                                        variant="primary")
                    vqe_out = gr.Markdown("")
                    gr.Markdown("### Named presets (from the originals — "
                                "selected, NOT compiled)")
                    preset_dropdown = gr.Dropdown(
                        choices=[("cos-type: ZZ + 0.5*IX - 0.15*ZI",
                                  "cos-type"),
                                 ("generic-type: ZZ + XX", "generic-type")],
                        value="cos-type", label="Preset Hamiltonian")
                    preset_btn = gr.Button("Run preset VQE",
                                           variant="secondary")
                    preset_out = gr.Markdown("")

        # ---- Tab 4: Biometrics -------------------------------------------
        with gr.Tab("Biometrics"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🧬 EEG (BrainFlow)")
                    eeg_mode = gr.Dropdown(
                        choices=[("Simulated EEG (BrainFlow synthetic "
                                  "board / numpy demo)", "synthetic"),
                                 ("Real Muse 2 headset", "muse")],
                        value="synthetic", label="EEG source",
                        interactive=True)
                    eeg_btn = gr.Button("Start neural link",
                                        variant="primary")
                    eeg_stop = gr.Button("Stop neural link",
                                         variant="secondary")
                    eeg_status = gr.Markdown("")
                    eeg_value = gr.Slider(minimum=0.0, maximum=1.0,
                                          value=0.95,
                                          label="Theta/alpha relaxation "
                                          "proxy (NOT entropy)",
                                          interactive=False)
                    gr.Markdown(
                        "The EEG value is a theta/alpha band-power ratio, "
                        "smoothed 0.8/0.2 — a relaxation proxy, not an "
                        "entropy. Simulated mode is labelled simulated; a "
                        "real Muse headset is needed for real data.")
                with gr.Column():
                    gr.Markdown("### ❤️ HRV (WebBluetooth / simulated)")
                    ble_btn = gr.Button(
                        "Initialize WebBluetooth (HRV)", variant="secondary")
                    bio_score = gr.Slider(minimum=0.0, maximum=1.0,
                                          value=0.95,
                                          label="Bio-coherence score "
                                          "(HRV heuristic — input value, "
                                          "not a measured entropy)",
                                          interactive=False)
                    rmssd_box = gr.Number(
                        value=0.0, label="RMSSD (ms, 30-beat window)",
                        interactive=False)
                    temp_box = gr.Number(
                        value=0.05, label="LLM temperature (= 1 - score)",
                        interactive=False)
                    sim_btn = gr.Button("Simulate one heartbeat (demo)",
                                        variant="secondary")
                    bio_payload = gr.Textbox(visible=False,
                                             elem_id="bio-payload")
                    gr.Markdown(
                        "Bio-coherence score = 1 - RMSSD/150 ms over the "
                        "last 30 beats (documented heuristic). WebBluetooth "
                        "limits: Chrome/Edge, HTTPS or localhost, user "
                        "gesture required. A BLE chest strap works; Apple "
                        "Watch HRV is not reliably readable from a web page.")

        # ---- Tab 5: Swarm Ledger & Billing -------------------------------
        with gr.Tab("Swarm Ledger & Billing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🐝 Swarm ledger (local only)")
                    ledger_toggle = gr.Checkbox(
                        value=False,
                        label="Enable ledger (opt-in): store coherent "
                              "syntheses ON THIS MACHINE ONLY")
                    ledger_size = gr.Number(value=0,
                                            label="Ledger nodes stored",
                                            interactive=False)
                    ledger_list = gr.Markdown(
                        "Ledger is OFF — enable it to store insights.")
                    gr.Markdown(
                        "Default OFF. Nothing is stored or shared unless "
                        "you enable it. There is no network ledger. "
                        "Commit threshold (bio score < 0.4) and relevance "
                        "threshold (L2 < 1.5) are ARBITRARY constants kept "
                        "from the original — no validation data exists for "
                        "either value.")
                with gr.Column():
                    gr.Markdown("### 💎 Tiers — what they actually change")
                    gr.Markdown(TIER_MEANINGS)
                    gr.Markdown(
                        "Prices ($49/$299/$999 per month) are display-only "
                        "labels from the original. No payment is possible "
                        "in this build.")
                    upgrade_btn = gr.Button("Upgrade via Stripe",
                                            variant="primary")
                    billing_note = gr.Markdown("")

    # -- page load banner ----------------------------------------------------
    def handle_page_load(request: gr.Request):
        status = request.query_params.get("payment")
        if status == "success":
            return gr.update(
                value="<div style='padding: 15px; background-color: #059669; "
                      "color: white; border-radius: 5px; text-align: center; "
                      "margin-bottom: 15px;'><b>🎉 Payment Successful!</b> "
                      "QCE Access Granted.</div>", visible=True)
        if status == "cancelled":
            return gr.update(
                value="<div style='padding: 15px; background-color: #dc2626; "
                      "color: white; border-radius: 5px; text-align: center; "
                      "margin-bottom: 15px;'><b>⚠️ Checkout Cancelled.</b>"
                      "</div>", visible=True)
        return gr.update(visible=False)

    app.load(fn=handle_page_load, inputs=None, outputs=status_banner)

    # -- billing -------------------------------------------------------------
    def do_upgrade(tier_id):
        url, message = get_checkout_url(tier_id)
        if url:
            return f"✅ Checkout session created: {url}"
        return f"ℹ️ {message}"

    upgrade_btn.click(fn=do_upgrade, inputs=[tier_dropdown],
                      outputs=[billing_note])

    # -- EEG -------------------------------------------------------------------
    def _eeg_start(mode):
        engine.eeg.mode = mode
        result = engine.eeg.start_stream()
        return result, engine.eeg.current_value

    eeg_btn.click(fn=_eeg_start, inputs=[eeg_mode],
                  outputs=[eeg_status, eeg_value])

    def _eeg_stop():
        result = engine.eeg.stop_stream()
        return result, engine.eeg.current_value

    eeg_stop.click(fn=_eeg_stop, inputs=None,
                   outputs=[eeg_status, eeg_value])

    # -- biometrics ------------------------------------------------------------
    ble_btn.click(fn=None, inputs=None, outputs=None, js=js_bluetooth)

    def _bio_from_payload(payload):
        engine.bio.mode = "real"
        engine.bio.update_from_payload(payload)
        return (engine.bio.current_bio_score, engine.bio.current_rmssd,
                engine.bio.temperature())

    bio_payload.change(fn=_bio_from_payload, inputs=[bio_payload],
                       outputs=[bio_score, rmssd_box, temp_box])

    def _bio_simulate():
        engine.bio.mode = "simulated"
        engine.bio.simulate_step()
        return (engine.bio.current_bio_score, engine.bio.current_rmssd,
                engine.bio.temperature())

    sim_btn.click(fn=_bio_simulate, inputs=None,
                  outputs=[bio_score, rmssd_box, temp_box])

    # -- ledger toggle ---------------------------------------------------------
    def _set_ledger(enabled):
        if engine.ledger is None:
            return 0, "Ledger unavailable (no embedding model loaded)."
        engine.ledger.enabled = bool(enabled)
        return engine.ledger.size(), _ledger_list_text()

    def _ledger_list_text():
        if engine.ledger is None or not engine.ledger.swarm_memory:
            return "Ledger is empty."
        lines = ["**Stored insights (local only):**\n"]
        for i, insight in enumerate(engine.ledger.swarm_memory[-10:], 1):
            lines.append(f"{i}. {insight}")
        return "\n\n".join(lines)

    ledger_toggle.change(fn=_set_ledger, inputs=[ledger_toggle],
                         outputs=[ledger_size, ledger_list])

    # -- quantum lab -----------------------------------------------------------
    def _run_quantum():
        return engine._quantum_message(engine.bio.current_bio_score)

    quantum_btn.click(fn=_run_quantum, inputs=None, outputs=[quantum_out])

    def _run_vqe_compile(expr):
        try:
            hamiltonian, explanation = compile_expression(expr)
            gs, _ = run_vqe(hamiltonian)
            return (
                f"**Compiled Hamiltonian:** `{hamiltonian.to_list()}`\n\n"
                f"{explanation}\n\n"
                f"**Ground-state energy (real VQE, Aer simulator): "
                f"{gs:.6f} Hartree**")
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            return f"**VQE compilation failed:** {exc}"

    vqe_btn.click(fn=_run_vqe_compile, inputs=[vqe_expr],
                  outputs=[vqe_out])

    def _run_preset(preset_name):
        try:
            name, hamiltonian, explanation = select_hamiltonian(
                "cos" if preset_name == "cos-type" else "generic")
            gs, _ = run_vqe(hamiltonian)
            return (
                f"**Preset selected:** {name} "
                f"`{hamiltonian.to_list()}`\n\n{explanation}\n\n"
                f"**Ground-state energy (real VQE, Aer simulator): "
                f"{gs:.6f} Hartree**")
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            return f"**Preset VQE failed:** {exc}"

    preset_btn.click(fn=_run_preset, inputs=[preset_dropdown],
                     outputs=[preset_out])

    # -- manifold refresh ------------------------------------------------------
    def _refresh_manifold():
        return engine._figure(LAST_QUERY_VEC["vec"])

    refresh_plot.click(fn=_refresh_manifold, inputs=None, outputs=[plot_3d])

    # -- chat flow ---------------------------------------------------------------
    def user_submit(user_message, history):
        history = history or []
        return "", history + [{"role": "user", "content": user_message}]

    def bot_response(history, tier_id):
        user_message = history[-1]["content"]
        history = history + [{"role": "assistant", "content": ""}]
        generator = engine.generate_response(user_message, history[:-1],
                                             tier_id)
        for partial_resp, telemetry, fig in generator:
            history[-1]["content"] = partial_resp
            # Remember the query vector for the Manifold tab refresh.
            if engine.qce.embed_model is not None:
                LAST_QUERY_VEC["vec"] = engine.qce.embed_model.encode(
                    [user_message]).astype("float32")[0]
            yield (history, fig,
                   telemetry["top_l2_distance"],
                   telemetry["chunks_retrieved"],
                   telemetry["latency_s"],
                   telemetry["words_streamed"],
                   telemetry["vqe_ground_state"],
                   telemetry["ledger_commit"],
                   telemetry["bio_score"],
                   telemetry["rmssd_ms"],
                   telemetry["temperature"],
                   telemetry["ledger_size"])

    chat_outputs = [chatbot, plot_3d, top_dist, chunks_n, latency, words_n,
                    vqe_box, ledger_commit_box, bio_score, rmssd_box,
                    temp_box, ledger_size]

    submit_btn.click(user_submit, [msg, chatbot], [msg, chatbot],
                     queue=False).then(
        bot_response, [chatbot, tier_dropdown], chat_outputs)

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot],
               queue=False).then(
        bot_response, [chatbot, tier_dropdown], chat_outputs)

if __name__ == "__main__":
    app.queue().launch(share=False, server_name="127.0.0.1",
                       server_port=7862)
