#!/usr/bin/env python3
"""Architects of Reality: Cognitive Companion — v2 (Gradio).

Changes vs the original (per Putt's rule, elaborated here):
  1. Startup no longer crashes or silently degrades when LM Studio, the
     FAISS index, or the billing backend are missing. The engine probes LM
     Studio, builds the index from Putt's papers if absent, and the UI shows
     a plain-language mode banner ("LLM mode" vs "DEMO MODE — extractive
     only") instead of failing on the first query.
  2. Telemetry is measured, not faked. The original "Cognitive Entropy"
     slider (0.95 minus 0.005 per token) and "Syntropic Resonance" number
     (+1.5 per message) were counters pretending to be measurements. v2
     shows: top-chunk L2 retrieval distance, chunks retrieved, response
     latency (seconds), words streamed, and the current mode.
  3. Chat history uses Gradio 5 "messages" format (list of {"role",
     "content"} dicts). Note: Gradio 5.4.0 / gradio-client 1.4.2 crashed
     on this format — its API-info endpoint dies in
     _json_schema_to_python_type on the messages schema's
     "additionalProperties": false, which also kills launch's localhost
     check. Upgrading to Gradio 5.50.0 / gradio-client 1.14.0 fixed it
     (verified by a component bisect: all cases OK, including both
     Chatbot formats). A tuples-format detour was tried and reverted.
  4. The tier dropdown now carries tier ids as values
     ("tier1"/"tier2"/"tier3") instead of string-matching on the label
     ("Community" in tier). Prices are display-only.
  5. The upgrade button no longer POSTs to a backend that does not exist
     and no longer redirects to "#" on failure. It calls billing.py, which
     is disabled by default and shows an honest "billing not configured"
     message. No Stripe account, product, price, or charge exists in v2.

Run:  python app.py        (serves on http://127.0.0.1:7860)
"""
import gradio as gr

from billing import get_checkout_url
from qce import TIERS, QuantumCognitiveEngine

qce = QuantumCognitiveEngine()

MODE_BANNER = (
    "🟢 **LLM mode:** LM Studio reachable — answers are synthesised by the "
    "local model, grounded in retrieved passages from Putt's papers."
    if qce.llm_available else
    "🟡 **DEMO MODE — extractive only:** LM Studio is not reachable "
    f"({qce.llm_status}). Answers are assembled ONLY from retrieved "
    "passages of Putt's papers, with sources. Start LM Studio with a model "
    "loaded and restart the app for full LLM synthesis."
)

theme = gr.themes.Monochrome(
    primary_hue="slate",
    font=[gr.themes.GoogleFont("Noto Sans"), "ui-sans-serif", "system-ui"],
)

with gr.Blocks(theme=theme, title="Architects of Reality - QCE") as app:
    status_banner = gr.Markdown(visible=False)

    gr.Markdown(
        """
        # 🌌 Architects of Reality: Cognitive Companion
        **The Syntropic Primer (Phase 0)** | Powered by the Aethel-Psi Quantum Cognitive Engine
        """
    )
    gr.Markdown(MODE_BANNER)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=550, label="QCE Synthesis Terminal",
                                 type="messages")
            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Query the Aethel-Psi framework or your Syntropic Primer coursework...",
                    scale=4,
                )
                submit_btn = gr.Button("Initialize Synthesis",
                                       variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Retrieval Telemetry (measured)")
            top_dist = gr.Number(value=None, label="Top-chunk L2 distance (lower = closer match)",
                                 interactive=False)
            chunks_n = gr.Number(value=0, label="Chunks retrieved",
                                 interactive=False)
            latency = gr.Number(value=None, label="Response latency (seconds)",
                                interactive=False)
            words_n = gr.Number(value=0, label="Words streamed",
                                interactive=False)
            mode_box = gr.Textbox(
                value="LLM" if qce.llm_available else "DEMO (extractive)",
                label="Mode", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 💎 SaaS Subscription")
            tier_dropdown = gr.Dropdown(
                choices=[(t["label"], tid) for tid, t in TIERS.items()],
                value="tier2",
                label="Select Upgrade Tier",
                interactive=True,
            )
            upgrade_btn = gr.Button("Upgrade via Stripe", variant="secondary")
            billing_note = gr.Markdown("")

    def handle_page_load(request: gr.Request):
        """Reads URL parameters to display Stripe success/failure banners."""
        payment_status = request.query_params.get("payment")
        if payment_status == "success":
            return gr.update(
                value="<div style='padding: 15px; background-color: #059669; "
                      "color: white; border-radius: 5px; text-align: center; "
                      "margin-bottom: 15px;'><b>🎉 Payment Successful!</b> "
                      "Welcome to your upgraded Architects of Reality tier. "
                      "Your QCE access is now active.</div>",
                visible=True)
        if payment_status == "cancelled":
            return gr.update(
                value="<div style='padding: 15px; background-color: #dc2626; "
                      "color: white; border-radius: 5px; text-align: center; "
                      "margin-bottom: 15px;'><b>⚠️ Checkout Cancelled.</b> "
                      "Your subscription was not processed.</div>",
                visible=True)
        return gr.update(visible=False)

    app.load(fn=handle_page_load, inputs=None, outputs=status_banner)

    def do_upgrade(tier_id):
        url, message = get_checkout_url(tier_id)
        if url:
            return (f"✅ Checkout session created: {url}")
        return f"ℹ️ {message}"

    upgrade_btn.click(fn=do_upgrade, inputs=[tier_dropdown],
                      outputs=[billing_note])

    def user_submit(user_message, history):
        history = history or []
        return "", history + [{"role": "user", "content": user_message}]

    def bot_response(history, tier_id):
        user_message = history[-1]["content"]
        history = history + [{"role": "assistant", "content": ""}]
        generator = qce.generate_response(user_message, history[:-1], tier_id)
        for partial_resp, telemetry in generator:
            history[-1]["content"] = partial_resp
            yield (history,
                   telemetry["top_l2_distance"],
                   telemetry["chunks_retrieved"],
                   telemetry["latency_s"],
                   telemetry["words_streamed"])

    submit_btn.click(user_submit, [msg, chatbot], [msg, chatbot],
                     queue=False).then(
        bot_response, [chatbot, tier_dropdown],
        [chatbot, top_dist, chunks_n, latency, words_n])

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot],
               queue=False).then(
        bot_response, [chatbot, tier_dropdown],
        [chatbot, top_dist, chunks_n, latency, words_n])

if __name__ == "__main__":
    app.queue().launch(share=False, server_port=7860)
