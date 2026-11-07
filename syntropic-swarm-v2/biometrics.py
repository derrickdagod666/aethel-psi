"""Biometric layer for the Syntropic Swarm v2.

Changes vs the original update_biometrics (per Putt's rule, elaborated):
  1. The original's "bio-entropy" was |1000 - rr| / 1000 for a SINGLE RR
     interval — one heartbeat's distance from 1000 ms. That is not heart
     rate variability and not an entropy. v2 keeps a rolling window of the
     last 30 RR intervals and computes RMSSD, the standard short-window HRV
     measure: sqrt(mean of squared successive RR differences), in ms.
  2. The window value is mapped to a heuristic score with a DOCUMENTED
     formula (below), and the score is labelled "bio-coherence heuristic",
     not entropy: high RMSSD (relaxed, variable heart) -> low score;
     low RMSSD (stressed, rigid heart) -> high score. The formula is a
     product choice, not a measured law, and the README says so.
  3. Two modes. SIMULATED (default, clearly labelled): generates plausible
     RR intervals from a seeded RNG so the UI and the temperature mapping
     can be exercised with no device at all. REAL: parses the same
     WebBluetooth JSON payload as the original ({rr: <ms>}) and feeds the
     window; the JS in app.py is unchanged from the original, and the UI
     states its limits (Chrome/Edge, HTTPS or localhost, BLE chest strap
     realistic, Apple Watch not reliably readable from a web page).
  4. Payloads with no RR value, a non-positive RR, or malformed JSON are
     ignored instead of silently corrupting state (the original dropped
     them silently too, but it also overwrote the entropy with a stale
     single-interval value on every payload).

Formula (heuristic, documented):
    rmssd_ms = sqrt(mean(diff(rr_window)^2))        # ms, window of <=30
    bio_score = clip(1.0 - rmssd_ms / 150.0, 0.1, 1.0)
  Rationale for the 150 ms scale: resting RMSSD is typically ~20-100 ms,
  so 150 ms maps a very relaxed window to ~0.1 and a rigid window
  (RMSSD ~0) to 1.0. It is a scaling choice, not a calibration.
  Temperature mapping kept from the original (temp = 1 - bio_score) and
  labelled as a design choice.
"""
import json
import math
import random

WINDOW = 30
RMSSD_SCALE_MS = 150.0


def rmssd_ms(rr_intervals):
    """Root mean square of successive RR differences, in ms."""
    if len(rr_intervals) < 2:
        return 0.0
    diffs = [b - a for a, b in zip(rr_intervals, rr_intervals[1:])]
    return float(math.sqrt(sum(d * d for d in diffs) / len(diffs)))


def bio_score_from_rmssd(rmssd):
    """Heuristic coherence score in [0.1, 1.0]. See module docstring."""
    return float(min(1.0, max(0.1, 1.0 - rmssd / RMSSD_SCALE_MS)))


class BiometricLayer:
    def __init__(self, mode="simulated", seed=42):
        self.mode = mode  # "simulated" or "real"
        self.rr_window = []
        self._rng = random.Random(seed)
        self.current_rmssd = 0.0
        self.current_bio_score = 0.95  # default matches the original's start
        self.samples_seen = 0

    # -- ingestion ---------------------------------------------------------
    def ingest_rr(self, rr_ms):
        """Add one RR interval (ms) to the window; return the new score."""
        try:
            rr = float(rr_ms)
        except (TypeError, ValueError):
            return self.current_bio_score
        if rr <= 0 or rr > 3000:
            return self.current_bio_score
        self.rr_window.append(rr)
        self.rr_window = self.rr_window[-WINDOW:]
        self.samples_seen += 1
        self.current_rmssd = rmssd_ms(self.rr_window)
        self.current_bio_score = bio_score_from_rmssd(self.current_rmssd)
        return self.current_bio_score

    def update_from_payload(self, payload_str):
        """Parse the original's WebBluetooth JSON payload {"rr": ms}."""
        if not payload_str:
            return self.current_bio_score
        try:
            data = json.loads(payload_str)
            rr = data.get("rr", 0)
        except (json.JSONDecodeError, AttributeError):
            return self.current_bio_score
        if rr and float(rr) > 0:
            return self.ingest_rr(rr)
        return self.current_bio_score

    def simulate_step(self):
        """Generate one plausible RR interval (labelled SIMULATED).

        Resting heart ~60-75 bpm with natural beat-to-beat variation:
        base 800 ms +/- slow drift, plus per-beat jitter ~N(0, 25 ms).
        """
        drift = 40.0 * math.sin(self.samples_seen / 12.0)
        rr = 820.0 + drift + self._rng.gauss(0, 25.0)
        return self.ingest_rr(max(400.0, rr))

    # -- reporting ---------------------------------------------------------
    def temperature(self):
        """LLM temperature mapping, kept from the original: 1 - bio_score.

        Design choice, not a law: a low (coherent) score gives a high
        temperature (more creative synthesis); a high score gives a low
        temperature (more grounded answers).
        """
        return float(1.0 - self.current_bio_score)

    def describe(self):
        return {
            "mode": self.mode,
            "rr_window_n": len(self.rr_window),
            "samples_seen": self.samples_seen,
            "rmssd_ms": round(self.current_rmssd, 2),
            "bio_score": round(self.current_bio_score, 3),
            "temperature": round(self.temperature(), 3),
        }
