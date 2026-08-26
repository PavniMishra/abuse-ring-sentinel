"""
routes.py

Flask blueprint with the scoring endpoints. Kept separate from app.py
so the web layer stays thin and the scoring logic stays testable
independent of Flask.
"""

import os
import pickle
import sys

import pandas as pd
from flask import Blueprint, jsonify, request

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "explain"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "audit"))

from train_model import FEATURE_COLUMNS, load_config  # noqa: E402
from explain_flag import explain  # noqa: E402
from audit_log import log_decision  # noqa: E402

api = Blueprint("api", __name__)

_cfg = load_config()
with open(_cfg["paths"]["model_path"], "rb") as f:
    _model = pickle.load(f)
_features_df = pd.read_csv(_cfg["paths"]["features_csv"]).set_index("account_id")


@api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@api.route("/score/<account_id>", methods=["GET"])
def score_account(account_id):
    if account_id not in _features_df.index:
        return jsonify({"error": f"unknown account_id: {account_id}"}), 404

    row = _features_df.loc[account_id]
    X = row[FEATURE_COLUMNS].to_frame().T
    score = float(_model.predict_proba(X)[:, 1][0])
    threshold = _cfg["model"]["risk_threshold"]
    verdict = "flagged" if score >= threshold else "clear"

    feature_dict = row[FEATURE_COLUMNS].to_dict()
    explanation = explain(row.to_dict())

    log_decision(
        log_path=_cfg["paths"]["audit_log_path"],
        account_id=account_id,
        score=score,
        verdict=verdict,
        threshold=threshold,
        features=feature_dict,
        explanation=explanation,
    )

    return jsonify({
        "account_id": account_id,
        "risk_score": round(score, 4),
        "threshold": threshold,
        "verdict": verdict,
        "explanation": explanation,
        "contributing_features": feature_dict,
    })


@api.route("/accounts", methods=["GET"])
def list_accounts():
    """Convenience endpoint for the demo UI to populate a dropdown."""
    limit = int(request.args.get("limit", 50))
    return jsonify(_features_df.index.tolist()[:limit])
