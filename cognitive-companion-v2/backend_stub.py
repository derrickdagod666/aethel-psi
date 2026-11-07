"""Optional FastAPI billing stub for the Architects of Reality QCE (v2).

This is a STUB, not a billing backend. It exists so the frontend's upgrade
flow has something real to call during development, and so the shape of the
real endpoint is documented.

  POST /create-checkout-session {"tier": "tier1"|"tier2"|"tier3"}
      -> 501 {"detail": "Billing not configured: no Stripe keys, no prices,
                         no charges. See billing.py for enablement steps."}

Run it (only for development):  uvicorn backend_stub:app --port 8000
It creates no Stripe account, product, price, or checkout session, and it
charges nothing. The real backend must implement this endpoint server-side
with the Stripe secret key.
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Architects of Reality — billing stub (disabled)")


class CheckoutRequest(BaseModel):
    tier: str = "tier2"


@app.post("/create-checkout-session")
def create_checkout_session(req: CheckoutRequest):
    return {
        "error": "billing_not_configured",
        "detail": (
            "Billing is not configured: no Stripe account, products, or "
            "prices exist for this demo build, so no checkout session can "
            "be created and no charge can be made. See billing.py for the "
            "steps to enable real Stripe billing."
        ),
        "tier_requested": req.tier,
    }, 501


@app.get("/health")
def health():
    return {"status": "stub", "billing": "disabled"}
