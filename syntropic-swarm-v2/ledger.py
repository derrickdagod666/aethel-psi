"""Syntropic Ledger v2 — persisted, opt-in, local-only.

Changes vs the original SyntropicLedger (per Putt's rule, elaborated):
  1. PERSISTED. The original kept its FAISS index in memory and its
     insights in a Python list — everything was lost on restart. v2 saves
     the index to swarm_ledger.index and the insights to swarm_ledger.json
     in the app directory after every commit, and reloads both on start.
  2. OPT-IN, DEFAULT OFF. The original committed any "coherent" synthesis
     to a ledger the UI called a community swarm, with no consent and no
     indication. v2 only commits when the user enables the ledger toggle
     in the UI (or passes enabled=True). Default is OFF: nothing is stored.
  3. LOCAL-ONLY. Committed insights live on this machine only. There is
     no network, no shared ledger, no other user. The word "community" no
     longer appears in the ledger UI; a hosted shared ledger with auth is
     listed in the README as future work, not a feature.
  4. Synchronous commit with error reporting. The original fired a
     threading.Thread with no join and no error handling, so failures died
     silently. v2 commits synchronously and returns (committed, reason);
     the engine reports the reason in the telemetry.
  5. Thresholds unchanged but documented. Commit threshold
     bio_score < 0.4 and relevance threshold L2 < 1.5 are kept exactly as
     the original wrote them, and are now labelled ARBITRARY in the code
     and README (no validation data exists for either value).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "swarm_ledger.index")
MEMORY_PATH = os.path.join(HERE, "swarm_ledger.json")

COMMIT_THRESHOLD = 0.4   # arbitrary, kept from the original
RELEVANCE_THRESHOLD = 1.5  # arbitrary L2 distance, kept from the original


class SyntropicLedger:
    def __init__(self, embed_model, enabled=False,
                 index_path=INDEX_PATH, memory_path=MEMORY_PATH):
        self.embed_model = embed_model
        self.enabled = enabled
        self.index_path = index_path
        self.memory_path = memory_path
        self.dimension = self.embed_model.get_sentence_embedding_dimension()
        self.swarm_memory = []
        self.swarm_index = None
        self._load_or_create()

    # -- persistence -------------------------------------------------------
    def _load_or_create(self):
        import faiss

        if (os.path.exists(self.index_path)
                and os.path.exists(self.memory_path)):
            try:
                self.swarm_index = faiss.read_index(self.index_path)
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    self.swarm_memory = json.load(f)
                if self.swarm_index.ntotal != len(self.swarm_memory):
                    print("Ledger mismatch: index holds "
                          f"{self.swarm_index.ntotal} vectors but memory "
                          f"holds {len(self.swarm_memory)} insights — "
                          "keeping both, queries use the shorter list.")
                print(f"Ledger loaded: {len(self.swarm_memory)} insights.")
                return
            except Exception as exc:  # noqa: BLE001 - start fresh, say so
                print(f"Ledger load failed ({exc}) — starting empty.")
        self.swarm_index = faiss.IndexFlatL2(self.dimension)
        self.swarm_memory = []

    def _save(self):
        import faiss

        faiss.write_index(self.swarm_index, self.index_path)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.swarm_memory, f, ensure_ascii=False, indent=1)

    # -- API ---------------------------------------------------------------
    def commit_resonance(self, query, synthesis, bio_score):
        """Commit one insight. Returns (committed: bool, reason: str)."""
        if not self.enabled:
            return False, "ledger disabled (opt-in toggle is OFF)"
        if bio_score >= COMMIT_THRESHOLD:
            return False, (f"bio_score {bio_score:.2f} >= commit threshold "
                           f"{COMMIT_THRESHOLD} (not coherent enough)")
        insight = (f"Swarm Insight (bio_score {bio_score:.2f}): "
                   f"{synthesis[:500]}...")
        vector = self.embed_model.encode([query]).astype("float32")
        self.swarm_index.add(vector)
        self.swarm_memory.append(insight)
        self._save()
        return True, (f"committed — ledger now holds "
                      f"{self.swarm_index.ntotal} nodes")

    def query_swarm(self, query, top_k=1):
        """Return the best stored insight, or '' if none is close enough."""
        if self.swarm_index is None or self.swarm_index.ntotal == 0:
            return ""
        vector = self.embed_model.encode([query]).astype("float32")
        distances, indices = self.swarm_index.search(vector, top_k)
        if indices[0][0] == -1:
            return ""
        if distances[0][0] >= RELEVANCE_THRESHOLD:
            return ""
        idx = int(indices[0][0])
        if idx >= len(self.swarm_memory):
            return ""
        return self.swarm_memory[idx]

    def size(self):
        return len(self.swarm_memory)
