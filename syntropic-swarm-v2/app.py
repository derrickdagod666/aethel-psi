#!/usr/bin/env python3
"""Architects of Reality: The Syntropic Swarm (Master QCE) — v2 (Gradio).

Two tabs, as in the unified original: Quantum Interface and Cybernetics
& Billing.

Changes vs the unified original (per Putt's rule, elaborated here):
  1. Quantum execution is simulator-first. The original only targeted real
     IBM hardware, and its fallback returned a hardcoded {'000': 512,
     '111': 512} while the UI claimed an IBM execution. v2 defaults to a
     local Aer simulator (free, instant); IBM Runtime needs saved
     credentials AND AOR_ENABLE_IBM=1; OFF disables it. A mock is never
     presented as a real run.
  2. VQE: the message text SELECTS between two named Hamiltonians
     (keyword rule, stated in the UI/README); the VQE itself is real (Aer
     Estimator + TwoLocal + COBYLA). The "compiles any math string" claim
     is dropped (see vqe.py).
  3. 3D visualizer plots REAL embeddings (visualizer.py): static chunks
     re-embedded from the corpus, reconstructed swarm vectors, live query.
     The original plotted np.random.rand(768) dummy vectors for the
     static layer.
  4. EEG: BrainFlow synthetic board used explicitly as a labelled demo;
     band indexing fixed (avg, _ unpacking); stop button added; the value
     is labelled a theta/alpha relaxation proxy, not "neural entropy".
     Real Muse path documented, not claimed.
  5. HRV: rolling RMSSD window (biometrics.py), not one RR interval;
     simulated demo mode plus the real WebBluetooth path with its limits
     stated.
  6. Ledger: persisted to disk, OPT-IN default OFF, local only.
  7. Chat history is rendered as an HTML log, NOT gr.Chatbot. The
     original used a Chatbot; v2 first kept it (type="messages"), but
     Gradio 5.4.0 / gradio-client 1.4.2 has a bug that crashes the
     server on the root page whenever a Chatbot is wired to an event:
     the Chatbot's API schema contains `"additionalProperties": true`
     (a boolean, not a schema), and `json_schema_to_python_type`
     crashes with `TypeError: argument of type 'bool' is not
     iterable` — verified by bisecting the app's config and by minimal
     probes (Chatbot in an event crashes; HTML + State does not).
     gradio-client 1.4.3 and 1.5.0 do NOT fix it, so v2 renders the
     conversation as a scrollable HTML log instead of upgrading the
     whole Gradio stack. The history is still a messages list in a
     gr.State; only the rendering changed.
  8. Telemetry is measured, not faked. The original's entropy gauge and
     resonance counter are gone; v2 shows L2 distance, chunks, latency,
     words, RMSSD, bio score, temperature, quantum backend/mode, ledger
     size and commit reason.
  9. The server binds 127.0.0.1 only. The original bound 0.0.0.0:7860,
     exposing the app (no auth) to the local network.
  10. Billing reuses the companion v2 billing.py: disabled by default, no
      keys, no charges; the upgrade button reports that plainly.

Run:  python app.py        (serves on http://127.0.0.1:7861)
Env:  AOR_QUANTUM_MODE=simulator|ibm|off (default simulator)
      AOR_ENABLE_IBM=1   (required, with saved IBM credentials, for ibm)
"""
import html

import gradio as gr

from billing import get_checkout_url
from engine import SwarmEngine
from qce import TIERS


def render_chat(history):
    """Render the messages-format history as a scrollable HTML log.

    See the module docstring (§7): gr.Chatbot is avoided because its
    API schema crashes gradio-client 1.4.2 on the root page. Content
    is HTML-escaped; line breaks are preserved.
    """
    parts = [
        "<div style='height:500px; overflow-y:auto; padding:10px; "
        "border:1px solid #444; border-radius:8px;'>"
    ]
    if not history:
        parts.append("<p style='color:#888;'>Ask a question about "
                     "Putt's papers — e.g. 'What is the gamma limit "
                     "from GW250114?'</p>")
    for turn in history or []:
        role = turn.get("role", "")
        content = html.escape(turn.get("content", "")).replace("\n", "<br>")
        if role == "user":
            parts.append(
                "<div style='text-align:right; margin:8px 0;'>"
                "<span style='display:inline-block; background:#1e3a5f; "
                "color:#fff; padding:8px 12px; border-radius:12px; "
                f"max-width:85%; text-align:left;'>{content}</span></div>")
        else:
            parts.append(
                "<div style='text-align:left; margin:8px 0;'>"
                "<span style='display:inline-block; background:#222; "
                "color:#eee; padding:8px 12px; border-radius:12px; "
                f"max-width:85%;'>{content}</span></div>")
    parts.append("</div>")
    return "".join(parts)

engine = SwarmEngine()

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

with gr.Blocks(theme=theme, title="QCE Master — Syntropic Swarm v2") as app:
    status_banner = gr.Markdown(visible=False)
    gr.Markdown(
        "# 🌌 Architects of Reality: Quantum Cognitive Engine \n"
        "Aethel-Psi Syntropic Swarm — v2 (simulator-first, local-only)"
    )
    gr.Markdown(MODE_BANNER)
    gr.Markdown(
        f"**Quantum mode:** {engine.quantum.mode} "
        f"(`{engine.quantum.backend_name}`) — {engine.quantum.status}"
    )

    with gr.Tabs():
        with gr.Tab("Quantum Interface"):
            with gr.Row():
                with gr.Column(scale=2):
                    chat_log = gr.HTML(value=render_chat([]),
                                       label="QCE Synthesis Terminal")
                    history_state = gr.State(value=[])
                    with gr.Row():
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Enter a physics query or V(Psi) = "
                            "cos(Psi) (Tier 3)...",
                            scale=4)
                        submit_btn = gr.Button("Initialize Synthesis",
                                               variant="primary", scale=1)
                with gr.Column(scale=2):
                    plot_3d = gr.Plot(label="Topological Swarm Manifold "
                                            "(real embeddings)")
                    gr.Markdown(
                        "Blue = static primer passages (real embeddings), "
                        "green = swarm ledger, amber = your query. "
                        "The original plotted random dummy vectors here; "
                        "v2 does not.")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚛️ Quantum collapse")
                    quantum_btn = gr.Button(
                        "Run 3-qubit collapse (simulator)", variant="secondary")
                    quantum_out = gr.Markdown("")
                with gr.Column():
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

        with gr.Tab("Cybernetics & Billing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🧬 Biometric Entanglement")
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
                        "real Muse headset is needed for real data."
                    )
                    ble_btn = gr.Button(
                        "Initialize WebBluetooth (HRV)", variant="secondary")
                    bio_score = gr.Slider(minimum=0.0, maximum=1.0,
                                          value=0.95,
                                          label="Bio-coherence score "
                                          "(HRV heuristic, lower = more "
                                          "coherent)", interactive=False)
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
                        "WebBluetooth limits: Chrome/Edge, HTTPS or "
                        "localhost, user gesture required. A BLE chest "
                        "strap works; Apple Watch HRV is not reliably "
                        "readable from a web page."
                    )
                    gr.Markdown("### 🐝 Swarm ledger (local only)")
                    ledger_toggle = gr.Checkbox(
                        value=False,
                        label="Enable ledger (opt-in): store coherent "
                              "syntheses ON THIS MACHINE ONLY")
                    ledger_size = gr.Number(value=0,
                                            label="Ledger nodes stored",
                                            interactive=False)
                    gr.Markdown(
                        "Default OFF. Nothing is stored or shared unless "
                        "you enable it. There is no network ledger."
                    )
                with gr.Column():
                    gr.Markdown("### 💎 SaaS Upgrades")
                    tier_dropdown = gr.Dropdown(
                        choices=[(t["label"], tid)
                                 for tid, t in TIERS.items()],
                        value="tier2", label="Active Tier",
                        interactive=True)
                    upgrade_btn = gr.Button("Upgrade via Stripe",
                                            variant="primary")
                    billing_note = gr.Markdown("")
                    gr.Markdown(
                        "Billing is disabled by default: no Stripe keys, "
                        "no charges. See billing.py for what real billing "
                        "would need."
                    )

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
        engine.ledger.enabled = bool(enabled)
        return engine.ledger.size()

    ledger_toggle.change(fn=_set_ledger, inputs=[ledger_toggle],
                         outputs=[ledger_size])

    # -- quantum button --------------------------------------------------------
    def _run_quantum():
        return engine._quantum_message(engine.bio.current_bio_score)

    quantum_btn.click(fn=_run_quantum, inputs=None, outputs=[quantum_out])

    # -- chat flow ---------------------------------------------------------------
    def user_submit(user_message, history):
        history = history or []
        history = history + [{"role": "user", "content": user_message}]
        return "", history, render_chat(history)

    def bot_response(history, tier_id):
        user_message = history[-1]["content"]
        history = history + [{"role": "assistant", "content": ""}]
        generator = engine.generate_response(user_message, history[:-1],
                                             tier_id)
        for partial_resp, telemetry, fig in generator:
            history[-1]["content"] = partial_resp
            yield (render_chat(history), fig,
                   telemetry["top_l2_distance"],
                   telemetry["chunks_retrieved"],
                   telemetry["latency_s"],
                   telemetry["words_streamed"],
                   telemetry["vqe_ground_state"],
                   telemetry["bio_score"],
                   telemetry["rmssd_ms"],
                   telemetry["temperature"],
                   telemetry["ledger_size"])

    submit_btn.click(user_submit, [msg, history_state],
                     [msg, history_state, chat_log],
                     queue=False).then(
        bot_response, [history_state, tier_dropdown],
        [chat_log, plot_3d, top_dist, chunks_n, latency, words_n, vqe_box,
         bio_score, rmssd_box, temp_box, ledger_size])

    msg.submit(user_submit, [msg, history_state],
               [msg, history_state, chat_log],
               queue=False).then(
        bot_response, [history_state, tier_dropdown],
        [chat_log, plot_3d, top_dist, chunks_n, latency, words_n, vqe_box,
         bio_score, rmssd_box, temp_box, ledger_size])

if __name__ == "__main__":
    app.queue().launch(share=False, server_name="127.0.0.1",
                       server_port=7861)
