"""
visualize_ablation.py

Runs an ablation study to compare v1 (Connected Components only)
vs v2 (Distance-to-Fraud added). Generates a side-by-side bar chart
showing the improvement in Recall and the reduction in missed fraud cost
for the pitch deck.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
import train_model
import evaluate

def run_ablation():
    cfg = train_model.load_config()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save the original features list (v2)
    original_features = list(train_model.FEATURE_COLUMNS)
    
    # --- RUN V1 (No distance_to_fraud) ---
    print("Running V1 (Connected Components Only)...")
    v1_features = [f for f in original_features if f != "distance_to_fraud"]
    
    # FIX: Ensure BOTH modules get the updated, smaller feature list
    train_model.FEATURE_COLUMNS = v1_features
    evaluate.FEATURE_COLUMNS = v1_features
    
    train_model.train(cfg)
    v1_report = evaluate.evaluate(cfg)
    
    # --- RUN V2 (All features including Dijkstra) ---
    print("\nRunning V2 (Dijkstra Distance Added)...")
    
    # FIX: Restore the full feature list in BOTH modules
    train_model.FEATURE_COLUMNS = original_features
    evaluate.FEATURE_COLUMNS = original_features
    
    train_model.train(cfg)
    v2_report = evaluate.evaluate(cfg)
    
    # --- PLOT RESULTS ---
    labels = ['Recall (Catch Rate)', 'Precision']
    v1_scores = [v1_report["recall"] * 100, v1_report["precision"] * 100]
    v2_scores = [v2_report["recall"] * 100, v2_report["precision"] * 100]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#111111")
    ax.set_facecolor("#111111")
    
    rects1 = ax.bar(x - width/2, v1_scores, width, label='v1 (Cluster Only)', color='#9aa5b1')
    rects2 = ax.bar(x + width/2, v2_scores, width, label='v2 (Cluster + Distance)', color='#e05858')
    
    ax.set_ylabel('Percentage (%)', color='#f0e6d2', fontsize=12)
    ax.set_title('v1 vs v2: Failure Recovery (Hold-out Test Set)', color='#f0e6d2', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='#f0e6d2', fontsize=12)
    ax.tick_params(axis='y', colors='#f0e6d2')
    
    ax.legend(facecolor='#1c1c1c', edgecolor='#333333', labelcolor='#f0e6d2')
    
    # Add data labels on top of bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', color='#f0e6d2', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#555555')
    ax.spines['left'].set_color('#555555')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "ablation_chart.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"\nSaved ablation chart to {out_path}")
    
    # Calculate the exact monetary difference
    cost_saved = v1_report['false_negative_cost_inr_total'] - v2_report['false_negative_cost_inr_total']
    print(f"Cost Impact: v2 saved ₹{cost_saved} in missed fraud costs on the test set compared to v1.")

if __name__ == "__main__":
    run_ablation()