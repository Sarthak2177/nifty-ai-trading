"""
test_pipeline.py
=================
Automated tests that run in the CI/CD pipeline.
These verify that each phase produced correct outputs.
If any test fails, the CI/CD pipeline fails and GitHub shows a red X.
"""

import os
import pandas as pd
import numpy as np
import sys

TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK",
           "ICICIBANK", "ITC", "SBIN", "WIPRO"]

passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ PASS — {name}")
        passed += 1
    else:
        print(f"  ❌ FAIL — {name} {detail}")
        failed += 1


print("\n🧪 Running Pipeline Tests")
print("=" * 50)

# ── Phase 1 Tests ──
print("\n[Phase 1] Feature Engineering")
for ticker in TICKERS:
    path = f"data/{ticker}_features.csv"
    test(f"{ticker} features file exists", os.path.exists(path))
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        test(f"{ticker} has enough rows", len(df) > 1000,
             f"(got {len(df)})")
        test(f"{ticker} has enough features", len(df.columns) >= 30,
             f"(got {len(df.columns)})")
        test(f"{ticker} no NaN values", df.isnull().sum().sum() == 0,
             f"(found {df.isnull().sum().sum()} NaNs)")

# ── Phase 2 Tests ──
print("\n[Phase 2] Label Generation")
for ticker in TICKERS:
    path = f"data/{ticker}_labelled.csv"
    test(f"{ticker} labelled file exists", os.path.exists(path))
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        test(f"{ticker} has label column", "label" in df.columns)
        test(f"{ticker} labels are valid (0/1/2)",
             set(df["label"].unique()).issubset({0, 1, 2}))
        counts = df["label"].value_counts(normalize=True)
        test(f"{ticker} no class dominates >80%",
             counts.max() < 0.80, f"(max={counts.max():.2f})")

# ── Phase 3 Tests ──
print("\n[Phase 3] Model Training")
for ticker in TICKERS:
    test(f"{ticker} XGBoost model saved",
         os.path.exists(f"models/{ticker}_xgb.json"))
    test(f"{ticker} LSTM model saved",
         os.path.exists(f"models/{ticker}_lstm.keras"))
    test(f"{ticker} scaler saved",
         os.path.exists(f"models/{ticker}_scaler.pkl"))

# ── Phase 4 Tests ──
print("\n[Phase 4] Ensemble + Backtest")
test("Signal snapshot CSV exists",
     os.path.exists("outputs/signal_snapshot.csv"))

if os.path.exists("outputs/signal_snapshot.csv"):
    snap = pd.read_csv("outputs/signal_snapshot.csv")
    test("All tickers in snapshot",
         len(snap) == len(TICKERS), f"(got {len(snap)})")
    test("Signal column has valid values",
         set(snap["Signal"].unique()).issubset({"BUY","HOLD","SELL"}))
    test("All SHAP charts saved",
         all(os.path.exists(f"outputs/{t}_shap.png") for t in TICKERS))
    test("All backtest charts saved",
         all(os.path.exists(f"outputs/{t}_backtest.png") for t in TICKERS))

# ── Summary ──
print(f"\n{'='*50}")
print(f"  Tests passed : {passed}")
print(f"  Tests failed : {failed}")
print(f"  Total        : {passed + failed}")

if failed > 0:
    print("\n❌ Pipeline has errors — check failed tests above")
    sys.exit(1)
else:
    print("\n✅ All tests passed — pipeline is healthy!")
    sys.exit(0)
