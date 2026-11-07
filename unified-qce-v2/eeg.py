"""EEG biometric tracker v2 (BrainFlow).

Changes vs the original EEGBiometricTracker (per Putt's rule, elaborated):
  1. BAND INDEXING FIXED. The original did
     `bands = DataFilter.get_avg_band_powers(...)` then
     `bands[0][2], bands[0][1]`. But get_avg_band_powers returns a TUPLE
     (avg_bands, std_bands), where avg_bands is per-channel
     [delta, theta, alpha, beta, gamma]. So bands[0] was the whole avg
     array and [2]/[1] selected *channels*, not bands — and the "ratio"
     became a numpy array, making current_entropy an array that would
     break the slider and temperature math downstream. v2 unpacks
     `avg, _ = get_avg_band_powers(...)` and uses avg[ch][2] (alpha) and
     avg[ch][1] (theta), averaged over the EEG channels.
  2. THE VALUE IS LABELLED A RELAXATION PROXY, NOT ENTROPY. The original
     called theta/alpha "neural entropy". A band-power ratio is not an
     entropy (no distribution, no logarithm). v2 calls it the
     theta/alpha relaxation proxy and documents the 0.8/0.2 smoothing as
     a display choice.
  3. SYNTHETIC BOARD IS LABELLED SIMULATED. The original defaulted to
     BoardIds.SYNTHETIC_BOARD (fake EEG generated in software) with a
     comment saying to swap to MUSE_2_BOARD in prod, but nothing in the
     UI said the EEG was simulated. v2's default mode is "synthetic"
     (BrainFlow synthetic board when installed, else a numpy simulation
     of band powers) and the UI labels it simulated. "muse" mode
     (MUSE_2_BOARD) is only attempted when explicitly selected, and
     requires a real paired Muse headset.
  4. STOP EXISTS. The original set is_streaming=True and never set it
     False; its daemon thread ran forever. v2 has stop_stream() (sets the
     stop event, stops the board, releases the session) and the thread
     checks the event each loop.
  5. No BrainFlow, no crash. If brainflow is not installed, the tracker
     runs in numpy-simulated mode and reports that plainly.
"""
import threading
import time

import numpy as np

try:
    from brainflow.board_shim import (
        BoardShim, BrainFlowInputParams, BoardIds)
    from brainflow.data_filter import DataFilter, DetrendOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False


class EEGBiometricTracker:
    def __init__(self, mode="synthetic"):
        self.mode = mode  # "synthetic" or "muse"
        self.is_streaming = False
        self.current_value = 0.95  # theta/alpha proxy, smoothed
        self.source = "not started"
        self._stop = threading.Event()
        self._thread = None
        self._rng = np.random.default_rng(7)
        self.board = None
        self.board_id = None
        self.eeg_channels = []
        self.sampling_rate = 250

    # -- lifecycle ---------------------------------------------------------
    def start_stream(self):
        if self.is_streaming:
            return "Neural link already active."
        self._stop.clear()
        if BRAINFLOW_AVAILABLE:
            try:
                params = BrainFlowInputParams()
                if self.mode == "muse":
                    self.board_id = BoardIds.MUSE_2_BOARD
                    self.source = "Muse 2 headset (real device)"
                else:
                    self.board_id = BoardIds.SYNTHETIC_BOARD
                    self.source = ("BrainFlow SYNTHETIC board — simulated "
                                   "EEG, not a real headset")
                self.board = BoardShim(self.board_id, params)
                self.board.prepare_session()
                self.board.start_stream(45000)
                self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
                self.sampling_rate = BoardShim.get_sampling_rate(
                    self.board_id)
                self.is_streaming = True
                self._thread = threading.Thread(
                    target=self._calculate_loop, daemon=True)
                self._thread.start()
                return f"Neural link active ({self.source})."
            except Exception as exc:  # noqa: BLE001 - fall back, say so
                self.source = (f"BrainFlow start failed ({exc}) — "
                               "using numpy-simulated bands")
        else:
            self.source = ("BrainFlow not installed — using numpy-simulated "
                           "band powers (labelled simulated)")
        self.is_streaming = True
        self._thread = threading.Thread(target=self._simulate_loop,
                                        daemon=True)
        self._thread.start()
        return f"Neural link active ({self.source})."

    def stop_stream(self):
        self._stop.set()
        self.is_streaming = False
        if self.board is not None:
            try:
                self.board.stop_stream()
                self.board.release_session()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                pass
            self.board = None
        return "Neural link stopped."

    # -- loops -------------------------------------------------------------
    def _calculate_loop(self):
        """Real BrainFlow data path (synthetic board or Muse)."""
        while not self._stop.is_set():
            time.sleep(2.0)
            try:
                data = self.board.get_current_board_data(256)
            except Exception:  # noqa: BLE001 - keep streaming
                continue
            if data.shape[1] < 256:
                continue
            for ch in self.eeg_channels:
                DataFilter.detrend(data[ch],
                                   DetrendOperations.LINEAR.value)
            avg, _ = DataFilter.get_avg_band_powers(
                data, self.eeg_channels, self.sampling_rate, True)
            # avg is per-channel [delta, theta, alpha, beta, gamma].
            theta = float(np.mean(avg[:, 1]))
            alpha = float(np.mean(avg[:, 2]))
            self._update_value(theta, alpha)

    def _simulate_loop(self):
        """Numpy-simulated band powers (labelled simulated)."""
        while not self._stop.is_set():
            time.sleep(2.0)
            theta = float(self._rng.uniform(0.3, 0.9))
            alpha = float(self._rng.uniform(0.5, 1.2))
            self._update_value(theta, alpha)

    def _update_value(self, theta_power, alpha_power):
        ratio = theta_power / (alpha_power + 1e-6)
        mapped = float(np.clip(ratio, 0.1, 1.0))
        # Smoothing kept from the original (0.8 old / 0.2 new) — a
        # display choice, documented here, not a measurement property.
        self.current_value = 0.8 * self.current_value + 0.2 * mapped
