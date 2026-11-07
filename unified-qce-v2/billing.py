"""Billing for the Architects of Reality QCE (v2) — DISABLED BY DEFAULT.

What changed vs the original (per Putt's rule, elaborated here):
  - The original POSTed straight to http://localhost:8000/create-checkout-
    session, a backend that was never included in the pasted code, with a
    hardcoded demo email (demo@architectsofreality.com). If that backend was
    not running, the upgrade button silently failed and the JS redirect sent
    the user to "#".
  - v2 never charges anything and never calls Stripe. get_checkout_url()
    returns (None, message) unless billing is explicitly enabled with
    AOR_BILLING_ENABLED=1 AND AOR_BACKEND_URL set. The optional FastAPI stub
    in backend_stub.py returns HTTP 501 "billing not configured" for the same
    endpoint, so a developer can wire the real backend later without the
    frontend ever pretending a payment happened.

To enable real Stripe billing one day (NOT done here, no accounts created):
  1. Create a Stripe account and products/prices for tier1/tier2/tier3.
  2. Implement /create-checkout-session in a real FastAPI backend using the
     Stripe secret key (server-side only, never in this frontend).
  3. Set AOR_BILLING_ENABLED=1 and AOR_BACKEND_URL to that backend's URL.
  4. Replace the hardcoded email with the signed-in user's email.
"""
import os

import requests

TIER_IDS = {"tier1": "tier1", "tier2": "tier2", "tier3": "tier3"}


def billing_enabled():
    return (os.environ.get("AOR_BILLING_ENABLED") == "1"
            and bool(os.environ.get("AOR_BACKEND_URL")))


def get_checkout_url(tier_id):
    """Return (checkout_url or None, status message).

    Disabled by default: returns (None, message) and never contacts Stripe.
    """
    tier_id = TIER_IDS.get(tier_id, "tier2")
    if not billing_enabled():
        return None, (
            "Billing is not configured. This is a demo build: no Stripe "
            "account, product, or price has been created and no charge can "
            "be made. To enable real billing, run the FastAPI backend with "
            "Stripe keys (see backend_stub.py) and set AOR_BILLING_ENABLED=1 "
            "and AOR_BACKEND_URL."
        )
    backend = os.environ["AOR_BACKEND_URL"].rstrip("/")
    try:
        resp = requests.post(
            f"{backend}/create-checkout-session",
            json={"tier": tier_id},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("checkout_url"), "Checkout session created."
    except Exception as exc:  # noqa: BLE001 - report, do not redirect blindly
        return None, f"Checkout backend error: {exc}"
