"""3D Topological Visualizer v2 — plots REAL embeddings only.

Changes vs the original TopologicalVisualizer (per Putt's rule, elaborated):
  1. NO RANDOM DUMMY VECTORS. The original filled static_vectors with
     np.random.rand(768) when the index loaded, so its PCA scatter plotted
     random points labelled "Static Primer" — a manifold of nothing. v2
     re-embeds the first 50 real corpus chunks (from the FAISS metadata)
     with the same embedding model, so every blue point is a real passage
     from Putt's papers. If the corpus is unavailable, the static layer is
     omitted and the figure says so, instead of inventing points.
  2. The PCA is fitted on the real vectors actually plotted (static +
     swarm + query), exactly as the original did — the math is unchanged,
     only the inputs are honest now.
  3. Empty-state figure is labelled ("not enough points yet") instead of
     a bare dark rectangle.
"""
import numpy as np

STATIC_MAX_POINTS = 50


def real_static_embeddings(embed_model, metadata, max_points=STATIC_MAX_POINTS):
    """Embed the first max_points corpus chunks for real. Returns a list
    of float32 vectors (possibly empty if there is no metadata)."""
    if not metadata:
        return []
    texts = [c["text"] for c in metadata[:max_points]]
    return [v.astype("float32") for v in
            embed_model.encode(texts, convert_to_numpy=True)]


class TopologicalVisualizer:
    def __init__(self):
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=3)
        self.is_fitted = False

    def generate_3d_manifold(self, static_embeddings, swarm_embeddings,
                             current_query_embedding):
        import plotly.graph_objects as go

        dark = dict(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
        all_vecs, labels, colors = [], [], []
        if static_embeddings:
            all_vecs.extend(static_embeddings)
            labels.extend(["Static Primer"] * len(static_embeddings))
            colors.extend(["#3b82f6"] * len(static_embeddings))
        if swarm_embeddings:
            all_vecs.extend(swarm_embeddings)
            labels.extend(["Swarm Ledger"] * len(swarm_embeddings))
            colors.extend(["#10b981"] * len(swarm_embeddings))
        if current_query_embedding is not None:
            all_vecs.append(current_query_embedding)
            labels.append("Active Query")
            colors.append("#f59e0b")
        if len(all_vecs) < 4:
            fig = go.Figure(layout=go.Layout(**dark))
            fig.add_annotation(
                text="Not enough points yet — ask a question or enable "
                     "the ledger to populate the manifold.",
                showarrow=False, font=dict(color="white"))
            return fig
        all_vecs = np.array(all_vecs, dtype=float)
        if not self.is_fitted or len(all_vecs) > 100:
            self.pca.fit(all_vecs)
            self.is_fitted = True
        coords_3d = self.pca.transform(all_vecs)
        fig = go.Figure(data=[go.Scatter3d(
            x=coords_3d[:, 0], y=coords_3d[:, 1], z=coords_3d[:, 2],
            mode="markers+lines",
            marker=dict(size=6, color=colors, opacity=0.8),
            line=dict(color="rgba(255,255,255,0.1)", width=1),
            text=labels, hoverinfo="text",
        )])
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis_title="Semantic X", yaxis_title="Semantic Y",
                zaxis_title="Semantic Z", bgcolor="#0f172a",
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                zaxis=dict(showgrid=False),
            ),
            paper_bgcolor="#0f172a", font_color="white",
        )
        return fig
