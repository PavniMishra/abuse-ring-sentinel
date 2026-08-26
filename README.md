# Abuse-Ring Sentinel

**Track 02 — AI Risk Manager** (Razorpay AI Buildathon)

## Problem

Fraud rarely happens as a single isolated account. Coordinated fraud rings
share devices, IPs, or bank fingerprints across multiple accounts. A
merchant needs to know, *before* a payment completes, whether the payer or
payee is part of such a ring — not just whether a single transaction looks
odd in isolation.

## Approach

1. **Synthetic dataset** with two deliberately different fraud patterns:
   - **Tight rings** (60% of fraud): 3-6 accounts sharing one strong
     signal (bank fingerprint). Trivially caught by connected-components.
   - **Distributed fraud** (40%): weakly linked, 1-2 hops out via a
     shared IP only. Requires distance-based scoring to catch.
   Plenty of innocent coincidental overlaps (shared public IPs between
   honest users) are included so the model can't just flag any shared
   attribute.

2. **Graph construction**: accounts are nodes; edges are weighted by
   shared-attribute strength (bank fingerprint > device > IP).

3. **Ring detection**:
   - v1 — connected components (fast, catches obvious rings)
   - v2 — multi-source Dijkstra distance-to-nearest-known-fraud (catches
     the distributed fraud that v1 alone misses)

4. **Model**: Random Forest on graph features + account-level features
   (age, transaction velocity, amount anomaly). Chosen deliberately over
   an LLM for this step — it's a tabular classification problem where a
   classical model is faster, cheaper, and more auditable.

5. **Evaluation**: precision/recall/F1 on a genuinely held-out test set,
   plus false-positive cost and false-negative cost (see `config.yaml`)
   — a wrongly-blocked legitimate customer and a missed fraud account
   don't cost the same, and the evaluation reflects that.

6. **Explanation layer**: an LLM is used *only* here — to turn feature
   values into a plain-language reason for a flagged account. This is
   the one place in the pipeline where AI is the right tool; everywhere
   else, deterministic/classical methods are more auditable.

7. **Audit trail**: every scoring decision is logged to
   `audit/audit_log.jsonl` with timestamp, score, contributing features,
   and explanation.

**Defense-only**: this system only detects and scores. It never acts
against a suspected fraud source.

## What v1 alone misses (failure-recovery story)

Connected-components alone (v1) catches the tight rings cleanly but
misses the distributed fraud accounts entirely, since they aren't in any
tight cluster with more than 1-2 edges. Adding the Dijkstra
distance-to-fraud feature (v2) recovers most of these by scoring
proximity to fraud rather than requiring hard cluster membership. Run
`evaluate.py` with `distance_to_fraud` removed from `FEATURE_COLUMNS` in
`model/train_model.py` to reproduce the v1-only recall drop.

## How to run

```bash
pip install -r requirements.txt

# 1. Generate synthetic data
python data/generate_synthetic_data.py

# 2. Build graph + detect rings (sanity check, optional standalone run)
python graph/ring_detection.py

# 3. Build feature table, train model, evaluate
python model/train_model.py
python model/evaluate.py

# 4. Run tests
pytest tests/ -v

# 5. Launch demo
python demo/app.py
# then open http://localhost:5000
```

## File structure

```
abuse-ring-sentinel/
├── config.yaml                  # all costs, weights, hyperparameters
├── data/
│   ├── generate_synthetic_data.py
│   ├── accounts.csv             # generated
│   └── transactions.csv         # generated
├── graph/
│   ├── build_graph.py
│   └── ring_detection.py
├── features/
│   └── feature_engineering.py
├── model/
│   ├── train_model.py
│   ├── evaluate.py
│   └── model.pkl                # generated
├── explain/
│   └── explain_flag.py
├── audit/
│   └── audit_log.py
├── api/
│   └── routes.py
├── demo/
│   ├── app.py
│   └── templates/index.html
├── tests/
│   └── test_ring_detection.py
└── requirements.txt
```
