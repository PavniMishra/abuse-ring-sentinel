"""
explain_flag.py

Turns a flagged account's top contributing features into a plain-language
reason. Template-based by default (no API key needed, fully deterministic
for the demo). If ANTHROPIC_API_KEY is set, uses Claude to phrase it more
naturally — same underlying facts either way, so the explanation is always
grounded in real feature values, never invented.
"""

import os
from typing import Optional


def template_explanation(account_row: dict) -> str:
    reasons = []

    if account_row.get("component_size", 1) > 1:
        fraud_peers = account_row.get("fraud_in_component", 0)
        if fraud_peers > 0:
            reasons.append(
                f"linked (shared device/IP/bank fingerprint) to a cluster of "
                f"{int(account_row['component_size'])} accounts, {int(fraud_peers)} "
                f"of which are known fraud"
            )

    dist = account_row.get("distance_to_fraud", float("inf"))
    if dist not in (None, float("inf")) and dist < 5:
        reasons.append(f"{dist:.1f} weighted hops from a known-fraud account")

    if account_row.get("is_new_account"):
        reasons.append(f"account is only {account_row.get('account_age_days', 0):.1f} days old")

    if account_row.get("velocity_per_day", 0) > 3:
        reasons.append(f"unusually high transaction velocity ({account_row['velocity_per_day']:.1f}/day)")

    if not reasons:
        return "Flagged by model score; no single dominant graph or velocity signal."

    return "This account was flagged because it is " + "; and ".join(reasons) + "."


def llm_explanation(account_row: dict) -> Optional[str]:
    """Optional: use Claude API to phrase the same facts more naturally.
    Falls back to the template if no API key is configured or the call fails.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        facts = template_explanation(account_row)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite this fraud-risk explanation in one concise, plain-language "
                    "sentence for a merchant dashboard. Do not invent any facts beyond "
                    f"what is stated:\n\n{facts}"
                ),
            }],
        )
        return response.content[0].text.strip()
    except Exception:
        return None


def explain(account_row: dict) -> str:
    llm_result = llm_explanation(account_row)
    return llm_result or template_explanation(account_row)


if __name__ == "__main__":
    example = {
        "component_size": 4,
        "fraud_in_component": 2,
        "distance_to_fraud": 1.7,
        "is_new_account": 1,
        "account_age_days": 2.3,
        "velocity_per_day": 5.1,
    }
    print(explain(example))
