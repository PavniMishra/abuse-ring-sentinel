"""
audit_log.py

Every scoring decision gets written to a structured, append-only JSONL
audit log: timestamp, account, score, verdict, contributing features,
and the explanation given. This is the evidence trail the track
explicitly asks for ("show the audit trail").
"""

import json
import os
from datetime import datetime, timezone


def log_decision(
    log_path: str,
    account_id: str,
    score: float,
    verdict: str,
    threshold: float,
    features: dict,
    explanation: str,
):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account_id": account_id,
        "risk_score": round(float(score), 4),
        "threshold": threshold,
        "verdict": verdict,          # "flagged" | "clear"
        "contributing_features": features,
        "explanation": explanation,
    }

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def read_audit_log(log_path: str):
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    entry = log_decision(
        log_path="audit/audit_log.jsonl",
        account_id="acc_demo123",
        score=0.87,
        verdict="flagged",
        threshold=0.5,
        features={"component_size": 4, "distance_to_fraud": 1.7},
        explanation="This account is linked to a cluster containing known fraud accounts.",
    )
    print("Logged:", entry)
