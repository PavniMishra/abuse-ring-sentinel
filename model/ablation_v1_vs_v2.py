"""
ablation_v1_vs_v2.py

Trains and evaluates TWO models on the exact same train/test split:
  - "v1_only"  : connected-components features only (component_size,
                 fraud_in_component) — no distance-to-fraud.
  - "v1_plus_v2": full feature set, including the Dijkstra
                 distance-to-fraud feature.

This isolates exactly what the v2 addition buys you, holding
everything else (data, split, model type, hyperparameters) constant —
a real ablation, not two independently-tuned models.

Outputs a JSON summary and a bar chart (recall, precision, F1, and
total cost) comparing the two.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from feature_engineering import build_feature_table  # noqa: E402


FULL_FEATURES = [
    "account_age_days", "is_new_account", "txn_count", "amount_mean",
    "amount_std", "velocity_per_day", "component_size", "fraud_in_component",
    "distance_to_fraud",
]
V1_ONLY_FEATURES = [f for f in FULL_FEATURES if f != "distance_to_fraud"]
LABEL_COLUMN = "is_fraud"


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def train_and_eval(features_df, feature_cols, cfg, label, hard_subset_ids=None):
    X = features_df[feature_cols]
    y = features_df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, features_df["account_id"],
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_seed"],
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        random_state=cfg["model"]["random_seed"],
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = cfg["model"]["risk_threshold"]
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    fp_cost = cfg["cost"]["false_positive_cost_inr"]
    fn_cost = cfg["cost"]["false_negative_cost_inr"]
    total_cost = fp * fp_cost + fn * fn_cost

    result = {
        "label": label,
        "n_features": len(feature_cols),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "total_cost_inr": round(float(total_cost), 2),
    }

    # The metric that actually matters for this ablation: recall computed
    # ONLY on fraud accounts that v1's strong-signal clustering can't see
    # at all (no bank/device link to anything). Overall recall stays
    # near-perfect regardless, because most fraud is in easy tight rings —
    # this subset isolates exactly what the graph-distance feature is for.
    if hard_subset_ids is not None:
        test_df = pd.DataFrame({"account_id": ids_test, "y_true": y_test.values, "y_pred": y_pred})
        hard_test = test_df[test_df["account_id"].isin(hard_subset_ids) & (test_df["y_true"] == 1)]
        if len(hard_test) > 0:
            hard_recall = (hard_test["y_pred"] == 1).mean()
            result["hard_subset_n"] = int(len(hard_test))
            result["hard_subset_recall"] = round(float(hard_recall), 4)
        else:
            result["hard_subset_n"] = 0
            result["hard_subset_recall"] = None

    print(f"--- {label} ---")
    for k, v in result.items():
        if k != "label":
            print(f"  {k}: {v}")
    return result


def plot_comparison(v1_result, v2_result, out_path):
    metrics = ["precision", "recall", "f1"]
    v1_vals = [v1_result[m] for m in metrics]
    v2_vals = [v2_result[m] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#111111")

    # --- Left: precision/recall/F1 comparison ---
    ax = axes[0]
    ax.set_facecolor("#111111")
    x = range(len(metrics))
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], v1_vals, width,
                   label="v1 only (connected components)", color="#9aa5b1")
    bars2 = ax.bar([i + width / 2 for i in x], v2_vals, width,
                   label="v1 + v2 (+ distance-to-fraud)", color="#e0a458")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.capitalize() for m in metrics], color="#f0e6d2")
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="#f0e6d2")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.set_title("Detection Quality: v1 vs v1+v2", color="#f0e6d2", fontsize=13)
    ax.legend(facecolor="#1c1c1c", edgecolor="#333333", labelcolor="#f0e6d2", fontsize=9)
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", color="#f0e6d2", fontsize=9)

    # --- Right: total cost comparison ---
    ax2 = axes[1]
    ax2.set_facecolor("#111111")
    cost_labels = ["v1 only", "v1 + v2"]
    cost_vals = [v1_result["total_cost_inr"], v2_result["total_cost_inr"]]
    bars = ax2.bar(cost_labels, cost_vals, color=["#9aa5b1", "#e0a458"], width=0.5)
    ax2.tick_params(colors="#f0e6d2")
    for spine in ax2.spines.values():
        spine.set_color("#444444")
    ax2.set_title("Total Mistake Cost on Held-Out Set (₹)", color="#f0e6d2", fontsize=13)
    ax2.set_ylabel("₹ (false-positive + false-negative cost)", color="#f0e6d2")
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"₹{h:,.0f}", (b.get_x() + b.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 4),
                     ha="center", color="#f0e6d2", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved chart to {out_path}")


def main():
    cfg = load_config()
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    features_df = build_feature_table(cfg)

    # The "hard" subset: fraud accounts v1 (strong-signal clustering) can't
    # see at all — component_size == 1 in the strong graph, meaning no
    # bank/device link to anything. If v2 (distance-to-fraud, computed on
    # the full graph including weak IP edges) is doing real work, recall
    # on THIS subset specifically should be higher for v1+v2 than v1-only.
    # Overall recall is not a useful metric here — most fraud sits in easy
    # tight rings and swamps the signal from the hard cases.
    hard_subset_ids = set(
        features_df[(features_df["is_fraud"] == 1) & (features_df["component_size"] == 1)]["account_id"]
    )
    print(f"Hard subset (fraud invisible to v1's clustering): {len(hard_subset_ids)} accounts\n")

    v1_result = train_and_eval(features_df, V1_ONLY_FEATURES, cfg, "v1_only", hard_subset_ids)
    v2_result = train_and_eval(features_df, FULL_FEATURES, cfg, "v1_plus_v2", hard_subset_ids)

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
        json.dump({"v1_only": v1_result, "v1_plus_v2": v2_result,
                    "hard_subset_total": len(hard_subset_ids)}, f, indent=2)

    plot_comparison(v1_result, v2_result, os.path.join(out_dir, "ablation_v1_vs_v2.png"))

    recall_delta = v2_result["recall"] - v1_result["recall"]
    print(f"\nOverall recall delta from adding v2: {recall_delta:+.2%}")
    if v1_result.get("hard_subset_recall") is not None and v2_result.get("hard_subset_recall") is not None:
        hard_delta = v2_result["hard_subset_recall"] - v1_result["hard_subset_recall"]
        print(f"Hard-subset recall — v1 only: {v1_result['hard_subset_recall']:.2%} "
              f"({v1_result['hard_subset_n']} accounts)")
        print(f"Hard-subset recall — v1+v2:   {v2_result['hard_subset_recall']:.2%} "
              f"({v2_result['hard_subset_n']} accounts)")
        print(f"Hard-subset recall delta: {hard_delta:+.2%}")
    else:
        print("Not enough hard-subset fraud accounts landed in the test split to report a hard-subset recall.")


if __name__ == "__main__":
    main()