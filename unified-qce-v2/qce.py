"""Quantum Cognitive Engine (QCE) v2 — RAG over Putt's own papers.

Changes vs the original (per Putt's rule, elaborated here):
  1. The engine no longer assumes LM Studio is running. At startup it probes
     http://localhost:1234/v1/models (2 s timeout). If reachable it uses the
     local model via the OpenAI-compatible API, exactly as the original
     intended. If not, it falls back to a DEMO RESPONDER that answers ONLY
     from the retrieved FAISS chunks (extractive, with source citations) and
     the UI says so in plain text. The original would raise a connection
     error on every query when LM Studio was down.
  2. The FAISS index is no longer assumed to exist: if it is missing, the
     engine builds it on first start via build_index.py (or reports the
     failure instead of silently running on "base model weights").
  3. The fake telemetry is gone. The original's "Cognitive Entropy" gauge
     started at 0.95 and dropped 0.005 per streamed token — a counter, not a
     measurement — and "Syntropic Resonance" added +1.5 per message. v2
     reports measured values only: top-chunk L2 retrieval distance, number
     of chunks retrieved, response latency in seconds, and words streamed.
  4. Tier handling no longer string-matches the dropdown label
     ("Community" in tier). Tiers are a dict of ids -> {label, top_k,
     instruction}; the dropdown value is the tier id.
  5. History is handled in Gradio 5 "messages" format (list of
     {"role": ..., "content": ...} dicts) instead of the old [[user, ai]]
     tuple format, which Gradio 5 no longer uses by default.
  6. No hardcoded demo email and no Stripe call live in this module; see
     billing.py (disabled by default, no keys, no charges).

Env vars:
  AOR_LLM_URL   LM Studio base URL (default http://localhost:1234/v1)
  AOR_LLM_MODEL model name for LM Studio (default "local-model")
"""
import json
import os
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "syntropic.index")
META_PATH = os.path.join(HERE, "syntropic_metadata.json")
EMBED_MODEL = "all-mpnet-base-v2"

TIERS = {
    "tier1": {
        "label": "Tier 1: Community ($49/mo)",
        "top_k": 2,
        "instruction": (
            "Provide a high-level, clear conceptual explanation suitable for "
            "a community student."
        ),
        # What the tier actually changes (unified v2 — tiers mean
        # something real, not just prompt wording):
        "features": "2 retrieved passages per answer; conceptual depth.",
        "quantum_access": False,  # no quantum/VQE intercepts
    },
    "tier2": {
        "label": "Tier 2: Researcher ($299/mo)",
        "top_k": 3,
        "instruction": (
            "Include deep mathematical derivations, topological cohomology "
            "analysis, and rigorous physics."
        ),
        "features": "3 retrieved passages per answer; full derivations.",
        "quantum_access": False,  # no quantum/VQE intercepts
    },
    "tier3": {
        "label": "Tier 3: Enterprise ($999/mo)",
        "top_k": 5,
        "instruction": (
            "Provide full theoretical depth, raw tensor logic, API endpoints, "
            "and unconstrained synthesis."
        ),
        "features": ("5 retrieved passages per answer; full depth; "
                     "quantum collapse + VQE compiler intercepts enabled."),
        "quantum_access": True,  # Tier 3 only: quantum + VQE intercepts
    },
}

# Prices are display-only. No billing is wired up (see billing.py).
TIER_LABELS = {tid: t["label"] for tid, t in TIERS.items()}


class QuantumCognitiveEngine:
    def __init__(self, llm_url=None, llm_model=None):
        self.llm_url = (llm_url or os.environ.get("AOR_LLM_URL")
                        or "http://localhost:1234/v1")
        self.llm_model = (llm_model or os.environ.get("AOR_LLM_MODEL")
                          or "local-model")
        self.embed_model = None
        self.index = None
        self.metadata = []
        self.llm_available = False
        self.llm_status = "not probed"
        self._load_index()
        self._probe_llm()

    # -- setup -------------------------------------------------------------
    def _load_index(self):
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            print("FAISS index not found — building it from Putt's papers...")
            try:
                import build_index
                build_index.main()
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                print(f"Index build failed: {exc}")
                return
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            print("WARNING: no FAISS index available. Retrieval disabled.")
            return
        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer

        self.embed_model = SentenceTransformer(EMBED_MODEL)
        import faiss

        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"FAISS index loaded: {self.index.ntotal} chunks, "
              f"dim {self.index.d}")

    def _probe_llm(self):
        try:
            resp = requests.get(f"{self.llm_url}/models", timeout=2)
            if resp.status_code == 200:
                self.llm_available = True
                self.llm_status = f"LM Studio reachable at {self.llm_url}"
                print(self.llm_status)
                return
            self.llm_status = (f"LM Studio answered HTTP {resp.status_code} "
                               f"at {self.llm_url} — demo mode")
        except Exception as exc:  # noqa: BLE001 - any failure means demo mode
            self.llm_status = (f"LM Studio not reachable at {self.llm_url} "
                               f"({exc}) — demo mode")
        print(self.llm_status)

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, query, top_k=3):
        """Return (chunks, distances). chunks are metadata dicts."""
        if self.index is None or self.embed_model is None:
            return [], []
        vec = self.embed_model.encode([query]).astype("float32")
        distances, indices = self.index.search(vec, top_k)
        chunks, dists = [], []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and 0 <= idx < len(self.metadata):
                chunks.append(self.metadata[idx])
                dists.append(float(dist))
        return chunks, dists

    def retrieve_context(self, query, top_k=3):
        chunks, dists = self.retrieve(query, top_k)
        if not chunks:
            return ("No vector database found. Operating on base model "
                    "weights."), [], []
        text = ""
        for chunk, dist in zip(chunks, dists):
            text += (f"[Source: {chunk['source']} | L2 Dist: {dist:.2f}]\n"
                     f"{chunk['text']}\n\n")
        return text.strip(), chunks, dists

    # -- generation --------------------------------------------------------
    def _demo_response(self, message, chunks, dists, tier_id):
        """Extractive demo answer. NOT LLM-generated; says so explicitly."""
        header = (
            "**DEMO MODE — no LLM is running.** LM Studio was not reachable, "
            "so this answer is assembled ONLY from the retrieved passages "
            "below (extractive, not generated). Start LM Studio with a model "
            "loaded and restart the app for full synthesis.\n\n"
        )
        if not chunks:
            return header + "No passages were retrieved for this query."
        body = "**Retrieved passages:**\n\n"
        for i, (chunk, dist) in enumerate(zip(chunks, dists), start=1):
            body += (f"**[{i}] {chunk['source']} (L2 distance {dist:.2f})**\n"
                     f"{chunk['text']}\n\n")
        body += ("*These passages are the most relevant parts of Putt's own "
                 "papers for your query, ranked by embedding distance. "
                 "Lower L2 distance = closer match.*")
        return header + body

    def generate_response(self, message, history, tier_id="tier2"):
        """Yield (partial_text, telemetry_dict).

        Telemetry is measured, not faked: top L2 distance of the best
        retrieved chunk, number of chunks retrieved, response latency in
        seconds, and words streamed so far.
        """
        tier = TIERS.get(tier_id, TIERS["tier2"])
        started = time.time()
        context, chunks, dists = self.retrieve_context(
            message, top_k=tier["top_k"])
        telemetry = {
            "top_l2_distance": round(dists[0], 3) if dists else None,
            "chunks_retrieved": len(chunks),
            "latency_s": None,
            "words_streamed": 0,
            "mode": "llm" if self.llm_available else "demo",
        }

        if not self.llm_available:
            text = self._demo_response(message, chunks, dists, tier_id)
            telemetry["latency_s"] = round(time.time() - started, 2)
            telemetry["words_streamed"] = len(text.split())
            yield text, telemetry
            return

        system_prompt = (
            "You are the Quantum Cognitive Engine (QCE), an AI assistant "
            "for the Aethel-Psi Synthesis framework.\n"
            "Answer the user's query using ONLY the provided mathematical "
            "and theoretical context.\n"
            f"{tier['instruction']}\n\n"
            "AVAILABLE CONTEXT FROM THE SYNTROPIC PRIMER:\n"
            f"{context}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history or []:
            if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"],
                                 "content": turn.get("content", "")})
        messages.append({"role": "user", "content": message})

        from openai import OpenAI

        client = OpenAI(base_url=self.llm_url, api_key="lm-studio")
        try:
            stream = client.chat.completions.create(
                model=self.llm_model, messages=messages,
                temperature=0.3, stream=True)
            partial = (f"**FAISS Context Retrieved:**\n```text\n"
                       f"{context[:300]}...\n```\n\n**Synthesis:**\n")
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    partial += delta
                    telemetry["words_streamed"] = len(partial.split())
                    telemetry["latency_s"] = round(time.time() - started, 2)
                    yield partial, telemetry
            telemetry["latency_s"] = round(time.time() - started, 2)
            yield partial, telemetry
        except Exception as exc:  # noqa: BLE001 - surface the failure honestly
            telemetry["latency_s"] = round(time.time() - started, 2)
            yield (f"**QCE Error:** LLM call failed. Ensure LM Studio is "
                   f"running on {self.llm_url}.\nDetails: {exc}"), telemetry
