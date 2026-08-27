"""
generate_synthetic_data.py

Creates accounts.csv and transactions.csv.

Fraud accounts are distributed two ways (per config.yaml):
  - tight_ring_fraction: small, obvious rings sharing a bank fingerprint
    (should be caught by plain connected-components).
  - distributed_fraction: thin, indirect links via weaker shared IPs,
    2 hops from a known-fraud node (needs distance-based scoring to catch).

Plenty of innocent coincidental overlaps (e.g. shared public IP) are
also injected so the detector can't just flag "any shared attribute".
"""

import csv
import random
import uuid
from datetime import datetime, timedelta
import yaml

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def random_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def random_date(days_back_max=365):
    days_back = random.randint(0, days_back_max)
    return (datetime.now() - timedelta(days=days_back)).isoformat()

def make_pool_attribute(prefix, size):
    return [random_id(prefix) for _ in range(size)]

def generate_accounts(cfg):
    n = cfg["data"]["n_accounts"]
    fraud_pct = cfg["data"]["n_fraud_accounts_pct"]
    tight_frac = cfg["data"]["tight_ring_fraction"]
    n_fraud = max(1, int(n * fraud_pct))
    n_tight = int(n_fraud * tight_frac)
    n_distributed = n_fraud - n_tight

    # --- DYNAMIC SEED LOGIC ---
    seed_val = cfg["data"].get("random_seed")
    if seed_val == "random" or seed_val is None:
        random.seed()
        print("Using purely random seed: Every run will generate a unique topology.")
    else:
        random.seed(seed_val)
        print(f"Using fixed seed: {seed_val}")
    # --------------------------

    accounts = []

    fraud_ip_pool = make_pool_attribute("ip_fraud", size=4)
    general_ip_pool = make_pool_attribute("ip", size=max(8, n // 60))

    P_SHARE_PUBLIC_IP = 0.10
    P_SHARE_FRAUD_IP_DISTRIBUTED = 0.85
    P_CROSSOVER_LEGIT_ON_FRAUD_IP = 0.03

    def draw_ip(fraud_prob=0.0, general_prob=P_SHARE_PUBLIC_IP):
        if random.random() < fraud_prob:
            return random.choice(fraud_ip_pool)
        if random.random() < general_prob:
            return random.choice(general_ip_pool)
        return random_id("ip")

    tight_ring_accounts = []
    i = 0
    while i < n_tight:
        ring_size = min(random.randint(3, 6), n_tight - i)
        ring_bank_fp = random_id("bank_ring")
        ring_device = random_id("dev_ring")
        for _ in range(ring_size):
            acc_id = random_id("acc")
            accounts.append({
                "account_id": acc_id,
                "created_at": random_date(days_back_max=10),
                "device_id": ring_device if random.random() < 0.7 else random_id("dev"),
                "ip": draw_ip(),
                "bank_fp": ring_bank_fp,
                "is_fraud": 1,
            })
            tight_ring_accounts.append(acc_id)
        i += ring_size

    distributed_accounts = []
    for _ in range(n_distributed):
        acc_id = random_id("acc")
        blends_in = random.random() < 0.5
        accounts.append({
            "account_id": acc_id,
            "created_at": random_date(days_back_max=300 if blends_in else 60),
            "device_id": random_id("dev"),
            "ip": draw_ip(fraud_prob=P_SHARE_FRAUD_IP_DISTRIBUTED),
            "bank_fp": random_id("bank"),
            "is_fraud": 1,
            "_low_velocity": blends_in,
        })
        distributed_accounts.append(acc_id)

    n_legit = n - n_fraud
    for _ in range(n_legit):
        looks_suspicious = random.random() < 0.08
        accounts.append({
            "account_id": random_id("acc"),
            "created_at": random_date(days_back_max=10 if looks_suspicious else 365),
            "device_id": random_id("dev"),
            "ip": draw_ip(fraud_prob=P_CROSSOVER_LEGIT_ON_FRAUD_IP),
            "bank_fp": random_id("bank"),
            "is_fraud": 0,
            "_high_velocity": looks_suspicious,
        })

    random.shuffle(accounts)
    return accounts

def generate_transactions(cfg, accounts):
    avg_txn = cfg["data"]["n_transactions_per_account_avg"]
    transactions = []
    for acc in accounts:
        n_txn = max(1, int(random.gauss(avg_txn, 2)))

        if acc.get("_low_velocity"):
            pass
        elif acc["is_fraud"] == 1 and random.random() < 0.5:
            n_txn += random.randint(5, 15)

        if acc.get("_high_velocity"):
            n_txn += random.randint(5, 15)

        base_amount = random.uniform(200, 5000)
        for _ in range(n_txn):
            amount = max(10, random.gauss(base_amount, base_amount * 0.3))
            transactions.append({
                "txn_id": random_id("txn"),
                "account_id": acc["account_id"],
                "amount": round(amount, 2),
                "timestamp": random_date(days_back_max=30),
            })
    return transactions

def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def main():
    cfg = load_config()
    accounts = generate_accounts(cfg)
    transactions = generate_transactions(cfg, accounts)

    write_csv(
        accounts,
        cfg["paths"]["accounts_csv"],
        fieldnames=["account_id", "created_at", "device_id", "ip", "bank_fp", "is_fraud"],
    )
    write_csv(
        transactions,
        cfg["paths"]["transactions_csv"],
        fieldnames=["txn_id", "account_id", "amount", "timestamp"],
    )

    n_fraud = sum(a["is_fraud"] for a in accounts)
    print(f"Generated {len(accounts)} accounts ({n_fraud} fraud, {len(accounts) - n_fraud} legit)")
    print(f"Generated {len(transactions)} transactions")
    print(f"Wrote {cfg['paths']['accounts_csv']} and {cfg['paths']['transactions_csv']}")

if __name__ == "__main__":
    main()