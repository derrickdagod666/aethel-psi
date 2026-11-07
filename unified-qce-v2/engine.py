"""Swarm Engine v2 — composes the QCE, ledger, quantum link, biometrics,
VQE compiler, EEG tracker and 3D visualizer.

Changes vs the original QuantumCognitiveEngine (per Putt's rule, elaborated):
  1. The base QCE is REUSED from cognitive-companion-v2/qce.py (copied
     into this directory, unmodified): LM Studio probe, extractive demo
     fallback, tier-id dict, Gradio 5 messages format, measured telemetry.
  2. Quantum intercept is mode-aware. The original ran every Enterprise
     "entangle/execute/qpu" request on real IBM hardware and raised when
     no backend existed — and the unified version returned a hardcoded
     mock while claiming an IBM execution. v2 runs on the configured
     QuantumLink (simulator by default, IBM only with credentials +
     AOR_ENABLE_IBM=1, OFF raises a plain message) and reports backend,
     shots, counts and Shannon entropy. A mock is never presented as a
     real run.
  3. VQE intercept (unified version). The original "compiled" V(Psi) =
     ... into a Hamiltonian; it did not — it selected between two fixed
     Hamiltonians by substring. v2 keeps the same two Hamiltonians and
     the same keyword rule, but says so: see vqe.py. The VQE itself is
     real (Aer Estimator + TwoLocal + COBYLA on the simulator).
  4. Biometric temperature modulation uses the windowed RMSSD-based score
     from biometrics.py, not a single RR interval, and the LLM stream is
     wrapped in try/except — the original had no error handling around
     the stream, so a failed stream crashed the generator. When the EEG
     tracker is streaming, its theta/alpha proxy overrides the HRV score
     for temperature (labelled in the telemetry).
  5. Swarm commit is synchronous with a reported reason. The original
     fired a threading.Thread with no join and no error handling.
  6. Every yield now carries the 3D manifold figure as well as text and
     telemetry. The figure plots REAL embeddings (visualizer.py) — the
     unified version plotted random dummy vectors for the static layer.
"""
import re
import time

from biometrics import BiometricLayer
from eeg import EEGBiometricTracker
from ledger import SyntropicLedger
from qce import TIERS, QuantumCognitiveEngine
from quantum import QuantumLink
from visualizer import TopologicalVisualizer, real_static_embeddings
from vqe import compile_expression, run_vqe

QUANTUM_KEYWORDS = ("entangle", "execute", "qpu")


class SwarmEngine:
    def __init__(self, quantum_mode=None, bio_mode="simulated",
                 ledger_enabled=False):
        self.qce = QuantumCognitiveEngine()
        self.quantum = QuantumLink(mode=quantum_mode)
        self.bio = BiometricLayer(mode=bio_mode)
        self.eeg = EEGBiometricTracker(mode="synthetic")
        # The ledger needs an embedding model; if the index failed to
        # build, the engine still runs (chat in demo mode degrades to
        # "no passages retrieved") instead of crashing here.
        if self.qce.embed_model is not None:
            self.ledger = SyntropicLedger(self.qce.embed_model,
                                          enabled=ledger_enabled)
        else:
            self.ledger = None
        self.visualizer = TopologicalVisualizer()
        self._static_embeddings = None
        self.tier_id = "tier2"

    # -- helpers -----------------------------------------------------------
    def _figure(self, query_vec):
        if self._static_embeddings is None:
            if self.qce.embed_model is None or not self.qce.metadata:
                self._static_embeddings = []
            else:
                self._static_embeddings = real_static_embeddings(
                    self.qce.embed_model, self.qce.metadata)
        swarm_vecs = []
        if self.ledger is not None \
                and self.ledger.swarm_index is not None \
                and self.ledger.swarm_index.ntotal > 0:
            # IndexFlatL2 stores the vectors, so they can be reconstructed
            # exactly — these are the real embedded queries of committed
            # insights, not dummy data.
            n = min(self.ledger.swarm_index.ntotal, 50)
            swarm_vecs = [self.ledger.swarm_index.reconstruct(i)
                          .astype("float32") for i in range(n)]
        return self.visualizer.generate_3d_manifold(
            self._static_embeddings, swarm_vecs, query_vec)

    def _bio_score(self):
        """EEG proxy overrides HRV when the EEG tracker is streaming."""
        if self.eeg.is_streaming:
            return self.eeg.current_value, "eeg-theta/alpha-proxy"
        return self.bio.current_bio_score, "hrv-rmssd-heuristic"

    def _quantum_message(self, phase):
        """Run the circuit; return the chat text for the turn."""
        try:
            qc, counts, entropy = \
                self.quantum.execute_topological_collapse(phase)
        except ValueError as exc:
            return (f"**Quantum execution unavailable:** {exc}\n\n"
                    f"Mode: {self.quantum.describe()['mode']} — "
                    f"{self.quantum.describe()['status']}")
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        lines = ", ".join(f"{k}: {v}" for k, v in top)
        return (
            f"**🌌 QUANTUM EXECUTION COMPLETE ({self.quantum.mode.upper()})**\n"
            f"**Backend:** `{self.quantum.backend_name}`\n"
            f"**Shots:** 1024 | **Measured Shannon entropy:** "
            f"`{entropy:.4f} bits`\n"
            f"**Top outcomes:** {lines}\n\n"
            "The 3-qubit Aethel-Psi circuit (H, CX, CX, RZ(phase·π)) was "
            "executed and the outcome distribution above is the measured "
            "result — not a simulation of a claim, the counts themselves."
        )

    # -- main generation ---------------------------------------------------
    def generate_response(self, message, history, tier_id="tier2"):
        """Yield (partial_text, telemetry_dict, figure)."""
        tier = TIERS.get(tier_id, TIERS["tier2"])
        started = time.time()
        bio_score, bio_source = self._bio_score()
        telemetry = {
            "top_l2_distance": None, "chunks_retrieved": 0,
            "latency_s": None, "words_streamed": 0,
            "mode": "llm" if self.qce.llm_available else "demo",
            "quantum_mode": self.quantum.mode,
            "quantum_backend": self.quantum.backend_name,
            "quantum_entropy_bits": None,
            "vqe_ground_state": None,
            "bio_score": round(bio_score, 3),
            "bio_source": bio_source,
            "rmssd_ms": round(self.bio.current_rmssd, 2),
            "temperature": round(1.0 - bio_score, 3),
            "ledger_enabled": self.ledger.enabled if self.ledger else False,
            "ledger_size": self.ledger.size() if self.ledger else 0,
            "ledger_commit": "",
        }
        query_vec = None
        if self.qce.embed_model is not None:
            query_vec = self.qce.embed_model.encode(
                [message]).astype("float32")[0]

        # 1. VQE intercept (Tier 3 only, via tier["quantum_access"]):
        # V(Psi) = ... — COMPILED FOR REAL in unified v2 (see vqe.py):
        # Psi-function syntax via the spectral mapping, or Pauli-term
        # syntax. The old keyword-selection presets are NOT used here;
        # they live in the Quantum Lab tab, labelled as presets.
        if tier.get("quantum_access") and "=" in message:
            match = re.search(r"V\(Psi\)\s*=\s*(.*)", message)
            if match:
                expr = match.group(1).strip()
                yield ("Compiling V(Psi) into a Hamiltonian "
                       "(real SymPy compilation — see vqe.py)...",
                       telemetry, self._figure(query_vec))
                try:
                    hamiltonian, explanation = compile_expression(expr)
                    gs, _ = run_vqe(hamiltonian)
                    telemetry["vqe_ground_state"] = round(gs, 6)
                    text = (
                        f"⚛️ DYNAMIC VQE EXECUTION\n"
                        f"Expression received: `V(Psi) = {expr}`\n"
                        f"Compiled Hamiltonian: "
                        f"`{hamiltonian.to_list()}`\n"
                        f"{explanation}\n\n"
                        f"Ground-state energy (real VQE on the Aer "
                        f"simulator): **{gs:.6f} Hartree**")
                except Exception as exc:  # noqa: BLE001 - report, not crash
                    text = (f"**VQE compilation failed:** {exc}\n\n"
                            "Try a function of Psi (e.g. `V(Psi) = "
                            "cos(Psi)`, `V(Psi) = Psi**2 + 0.5*Psi`), "
                            "Pauli-term syntax (e.g. `V(Psi) = ZZ + XX`), "
                            "or run a named preset from the Quantum Lab tab.")
                telemetry["latency_s"] = round(time.time() - started, 2)
                telemetry["words_streamed"] = len(text.split())
                yield text, telemetry, self._figure(query_vec)
                return

        # 2. Quantum hardware intercept (Tier 3 only, as in the original).
        if tier.get("quantum_access") and any(
                kw in message.lower() for kw in QUANTUM_KEYWORDS):
            text = self._quantum_message(bio_score)
            if "Shannon entropy" in text:
                m = re.search(r"entropy:\*\* `([\d.]+) bits`", text)
                if m:
                    telemetry["quantum_entropy_bits"] = float(m.group(1))
            telemetry["latency_s"] = round(time.time() - started, 2)
            telemetry["words_streamed"] = len(text.split())
            yield text, telemetry, self._figure(query_vec)
            return

        # 3. Context retrieval (swarm + static), via the reused QCE.
        swarm_context = (self.ledger.query_swarm(message)
                         if self.ledger else "")
        static_context, chunks, dists = self.qce.retrieve_context(
            message, top_k=tier["top_k"])
        telemetry["top_l2_distance"] = \
            round(dists[0], 3) if dists else None
        telemetry["chunks_retrieved"] = len(chunks)

        if not self.qce.llm_available:
            text = self.qce._demo_response(message, chunks, dists, tier_id)
            if swarm_context:
                text += (f"\n\n**Swarm ledger insight:**\n{swarm_context}")
            telemetry["latency_s"] = round(time.time() - started, 2)
            telemetry["words_streamed"] = len(text.split())
            if self.ledger is not None:
                committed, reason = self.ledger.commit_resonance(
                    message, text, bio_score)
                telemetry["ledger_commit"] = \
                    ("committed" if committed else f"not committed: {reason}")
                telemetry["ledger_size"] = self.ledger.size()
            else:
                telemetry["ledger_commit"] = \
                    "not committed: no embedding model, ledger unavailable"
            yield text, telemetry, self._figure(query_vec)
            return

        # 4. LLM synthesis with biometric temperature (design choice).
        context_parts = []
        if swarm_context:
            context_parts.append(f"--- SWARM LEDGER ---\n{swarm_context}")
        if static_context:
            context_parts.append(
                f"--- STATIC PRIMER ---\n{static_context}")
        full_context = "\n\n".join(context_parts) or "(no context retrieved)"
        system_prompt = (
            "You are the QCE. Use context to answer. "
            f"SaaS Tier: {tier['label']}\n\nCONTEXT:\n{full_context}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history or []:
            if isinstance(turn, dict) and turn.get("role") in ("user",
                                                              "assistant"):
                messages.append({"role": turn["role"],
                                 "content": turn.get("content", "")})
        messages.append({"role": "user", "content": message})

        from openai import OpenAI

        client = OpenAI(base_url=self.qce.llm_url, api_key="lm-studio")
        try:
            stream = client.chat.completions.create(
                model=self.qce.llm_model, messages=messages,
                temperature=1.0 - bio_score, stream=True)
            partial = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    partial += delta
                    telemetry["words_streamed"] = len(partial.split())
                    telemetry["latency_s"] = round(time.time() - started, 2)
                    yield partial, telemetry, self._figure(query_vec)
            telemetry["latency_s"] = round(time.time() - started, 2)
            if self.ledger is not None:
                committed, reason = self.ledger.commit_resonance(
                    message, partial, bio_score)
                telemetry["ledger_commit"] = \
                    ("committed" if committed else f"not committed: {reason}")
                telemetry["ledger_size"] = self.ledger.size()
            else:
                telemetry["ledger_commit"] = \
                    "not committed: no embedding model, ledger unavailable"
            yield partial, telemetry, self._figure(query_vec)
        except Exception as exc:  # noqa: BLE001 - surface, do not crash
            telemetry["latency_s"] = round(time.time() - started, 2)
            yield (f"**QCE Error:** LLM call failed. Ensure LM Studio is "
                   f"running on {self.qce.llm_url}.\nDetails: {exc}"), \
                telemetry, self._figure(query_vec)
