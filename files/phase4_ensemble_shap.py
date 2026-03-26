"""
PHASE 4 — Ensemble + Confidence + SHAP + Backtesting
======================================================
Combines XGBoost + LSTM via weighted average,
derives confidence scores, explains with SHAP,
and backtests the strategy vs buy-and-hold.
"""

import pandas as pd
import numpy as np
import os, glob, joblib, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR  = "data"
MODEL_DIR = "models"
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK",
           "ICICIBANK", "ITC", "SBIN", "WIPRO"]

LABEL_MAP    = {0: "SELL", 1: "HOLD", 2: "BUY"}
COLORS       = {"BUY": "#00e5a0", "HOLD": "#f5c542", "SELL": "#ff4d6d"}
XGB_WEIGHT   = 0.55
LSTM_WEIGHT  = 0.45
SEQUENCE_LEN = 20
TEST_SPLIT   = 0.2

FEATURE_COLS = [
    "ema_9","ema_21","ema_50","sma_20","macd","macd_signal","macd_diff","adx",
    "rsi_14","rsi_7","stoch_k","stoch_d","williams_r","cci",
    "bb_width","bb_pct","atr_14","obv","cmf","volume_ratio",
    "returns_1d","returns_5d","returns_10d","high_low_pct","close_open_pct",
    "dist_ema9","dist_ema21","dist_ema50",
]


def ensemble_predict(xgb_proba, lstm_proba):
    combined   = XGB_WEIGHT * xgb_proba + LSTM_WEIGHT * lstm_proba
    signal     = np.argmax(combined, axis=1)
    confidence = np.max(combined, axis=1) * 100
    return signal, confidence


def compute_shap_plot(ticker, X_scaled, avail, xgb_model):
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(X_scaled)
    mean_shap = np.abs(sv).mean(axis=(0,2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
    feat_imp  = pd.Series(mean_shap, index=avail).sort_values(ascending=True).tail(12)
    top3      = feat_imp.iloc[-3:][::-1].index.tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0a0b0f"); ax.set_facecolor("#0d0e14")
    ax.barh(feat_imp.index, feat_imp.values, color="#4a9eff", edgecolor="none", height=0.6)
    ax.set_title(f"{ticker} — SHAP Feature Importance", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#aaa", labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor("#333")
    ax.set_xlabel("Mean |SHAP value|", color="#888")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{ticker}_shap.png", dpi=150, bbox_inches="tight", facecolor="#0a0b0f")
    plt.close()
    return top3


def backtest(df_test, signals, confidence, min_conf=55.0):
    close     = df_test["Close"].values.flatten()
    n         = min(len(signals), len(close) - 1)
    cash, pos = 10000.0, 0.0
    equity    = [10000.0]
    bh_shares = 10000.0 / close[0]
    bh_curve  = [10000.0]
    buys, sells = [], []

    for i in range(n):
        p, s, c = close[i], signals[i], confidence[i]
        if s == 2 and c >= min_conf and pos == 0:
            pos = cash / p; cash = 0.0; buys.append(p)
        elif s == 0 and c >= min_conf and pos > 0:
            cash = pos * p; pos = 0.0; sells.append(p)
        equity.append(cash + pos * close[i + 1])
        bh_curve.append(bh_shares * close[i + 1])

    ret_s  = (equity[-1]   - 10000) / 10000 * 100
    ret_bh = (bh_curve[-1] - 10000) / 10000 * 100
    eq     = np.array(equity)
    dr     = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / (dr.std() + 1e-9) * np.sqrt(252)
    pairs  = list(zip(buys, sells))
    wr     = sum(1 for b, s in pairs if s > b) / max(len(pairs), 1) * 100

    return equity, bh_curve, ret_s, ret_bh, sharpe, len(buys), wr


def plot_backtest(ticker, equity, bh_curve, ret_s, ret_bh, sharpe, n_trades, wr):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#0a0b0f"); ax.set_facecolor("#0d0e14")
    ax.plot(equity,   color="#00e5a0", lw=1.5, label="Strategy")
    ax.plot(bh_curve, color="#4a9eff", lw=1.2, linestyle="--", label="Buy & Hold")
    ax.axhline(10000, color="#444", lw=0.8, linestyle=":")
    ax.set_title(f"{ticker} — Strategy vs Buy & Hold (₹10,000 starting capital)",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#888")
    for s in ax.spines.values(): s.set_edgecolor("#333")
    ax.legend(facecolor="#1a1b20", edgecolor="#333", labelcolor="white")
    ax.set_xlabel(
        f"Strategy: {ret_s:+.1f}%   B&H: {ret_bh:+.1f}%   "
        f"Sharpe: {sharpe:.2f}   Trades: {n_trades}   Win Rate: {wr:.0f}%",
        color="#aaa", fontsize=8)
    ax.set_ylabel("Portfolio Value (₹)", color="#888")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{ticker}_backtest.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0b0f")
    plt.close()


def run_ensemble():
    from tensorflow import keras

    print("\n🎯 Phase 4: Ensemble + SHAP + Backtesting")
    print("=" * 50)
    print(f"  Ensemble  : XGBoost {XGB_WEIGHT*100:.0f}% + LSTM {LSTM_WEIGHT*100:.0f}%")
    print(f"  Min confidence to trade: 55%\n")

    rows = []

    for ticker in TICKERS:
        fpath = f"{DATA_DIR}/{ticker}_labelled.csv"
        if not os.path.exists(fpath):
            print(f"  ❌ {ticker}: labelled file not found"); continue
        if not os.path.exists(f"{MODEL_DIR}/{ticker}_xgb.json"):
            print(f"  ❌ {ticker}: model not found, run Phase 3 first"); continue

        print(f"  ── {ticker} ──")
        df     = pd.read_csv(fpath, index_col=0, parse_dates=True)
        avail  = [c for c in FEATURE_COLS if c in df.columns]
        split  = int(len(df) * (1 - TEST_SPLIT))
        df_test= df.iloc[split:].copy()
        X_test = df_test[avail].values

        scaler    = joblib.load(f"{MODEL_DIR}/{ticker}_scaler.pkl")
        X_scaled  = scaler.transform(X_test)

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(f"{MODEL_DIR}/{ticker}_xgb.json")
        xgb_proba = xgb_model.predict_proba(X_scaled)

        lstm_model = keras.models.load_model(f"{MODEL_DIR}/{ticker}_lstm.keras")
        X_seq = np.array([X_scaled[i-SEQUENCE_LEN:i]
                          for i in range(SEQUENCE_LEN, len(X_scaled))])
        lstm_proba = lstm_model.predict(X_seq, verbose=0)

        n = len(lstm_proba)
        xgb_aligned  = xgb_proba[-n:]
        df_test_align= df_test.iloc[-n:].copy()

        signals, confidence = ensemble_predict(xgb_aligned, lstm_proba)
        latest_signal = LABEL_MAP[signals[-1]]
        latest_conf   = confidence[-1]
        print(f"  Latest signal : {latest_signal} (confidence {latest_conf:.1f}%)")

        top3 = compute_shap_plot(ticker, X_scaled, avail, xgb_model)
        print(f"  SHAP top drivers: {top3}")

        equity, bh_curve, ret_s, ret_bh, sharpe, n_trades, wr = \
            backtest(df_test_align, signals, confidence)
        plot_backtest(ticker, equity, bh_curve, ret_s, ret_bh, sharpe, n_trades, wr)
        print(f"  Backtest  : Strategy {ret_s:+.1f}%  B&H {ret_bh:+.1f}%  "
              f"Sharpe {sharpe:.2f}  Trades {n_trades}  WinRate {wr:.0f}%\n")

        rows.append({
            "Ticker": ticker, "Signal": latest_signal,
            "Confidence": f"{latest_conf:.0f}%",
            "Top_Feature_1": top3[0], "Top_Feature_2": top3[1],
            "Top_Feature_3": top3[2] if len(top3) > 2 else "-",
            "Strategy_Return": f"{ret_s:+.1f}%",
            "BH_Return": f"{ret_bh:+.1f}%",
            "Sharpe": f"{sharpe:.2f}",
            "Win_Rate": f"{wr:.0f}%",
        })

    snapshot = pd.DataFrame(rows)
    snapshot.to_csv(f"{OUT_DIR}/signal_snapshot.csv", index=False)

    print("\n  📋 Final Signal Snapshot:")
    print(f"  {'Ticker':<12} {'Signal':<6} {'Conf':>5}  {'Strategy':>9}  {'B&H':>7}  {'Sharpe':>7}  {'WinRate':>8}")
    print(f"  {'-'*60}")
    for _, r in snapshot.iterrows():
        print(f"  {r['Ticker']:<12} {r['Signal']:<6} {r['Confidence']:>5}  "
              f"{r['Strategy_Return']:>9}  {r['BH_Return']:>7}  {r['Sharpe']:>7}  {r['Win_Rate']:>8}")

    print(f"\n  All charts saved to {OUT_DIR}/")
    print("\n✅ Phase 4 complete — Full ML pipeline done!")
    print("   Next: Phase 5 — Frontend UI 🎨\n")


if __name__ == "__main__":
    run_ensemble()
