"""Aethel-Psi — importable Bilby waveform plugin (v0.1.0).

This is the pip-installable form of ``gw-pipeline/uaf_bilby_plugin.py``.
The waveform itself is unchanged: a Gaussian dip at the merger frequency
on top of ``bilby.gw.source.lal_binary_black_hole`` (IMRPhenomPv2),
with the two documented fixes — width is a *fraction* of ``f_merger``
(FIX 1) and ``f_merger`` is passed explicitly, never ``argmax(|h|)``
(FIX 2). See ``gw-pipeline/README.md`` for the full Gate 3 story.

What the package does NOT claim (see the root README and CHANGELOG):
gamma = 0.4 is excluded on real GW250114 data (profile dlnL = -55.2;
full dynesty PE: gamma median 0.0235, 95% upper 0.0813, ln B = -3.06
+/- 0.23 vs GR). Prediction 1 is gamma ≲ 0.08 (95%) for GW250114-like
mergers. The spin-gamma hypothesis is CLOSED (v6). The theory itself
is not validated and is not for sale; the pipeline is the product.
"""

from .waveform import (
    F_MAX,
    F_MIN,
    WAVEFORM_ARGS,
    dip_modifier,
    uaf_blunted_waveform,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "dip_modifier",
    "uaf_blunted_waveform",
    "WAVEFORM_ARGS",
    "F_MIN",
    "F_MAX",
]
