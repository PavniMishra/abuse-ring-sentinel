"""
evaluate.py

Evaluates the trained model on the held-out test set ONLY.
Reports precision, recall, F1 — and, per the track's explicit
requirement, the false-positive cost: what it costs the merchant
in rupees to wrongly block a legitimate customer, weighed against
the cost of missing real fraud.
"""

import os
import pickle

import pandas as pd
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from train_model import FEATURE_COLUMNS, LABEL_COLUMN, load_config


def evaluate(cfg):
    model_path = cfg["paths"]["model_path"]
    test_path = model_path.replace("model.pkl", "held_out_test.csv")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    test_df = pd.read_csv(test_path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN]

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = cfg["model"]["risk_threshold"]
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    fp_cost = cfg["cost"]["false_positive_cost_inr"]
    fn_cost = cfg["cost"]["false_negative_cost_inr"]
    total_fp_cost = fp * fp_cost
    total_fn_cost = fn * fn_cost
    total_cost = total_fp_cost + total_fn_cost

    report = {
        "threshold": threshold,
        "n_test": len(y_test),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_cost_inr_total": round(total_fp_cost, 2),
        "false_negative_cost_inr_total": round(total_fn_cost, 2),
        "total_cost_inr": round(total_cost, 2),
    }

    print("=== Evaluation on held-out test set ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    return report


if __name__ == "__main__":
    cfg = load_config()
    evaluate(cfg)
