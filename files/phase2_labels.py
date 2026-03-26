"""
PHASE 2 — Label Generation
===========================
Labels each trading day as BUY / HOLD / SELL
based on forward return over next 5 days.

  future_return > +2%  → BUY  (2)
  future_return < -2%  → SELL (0)
  otherwise            → HOLD (1)
"""

import pandas as pd
import numpy as np
import os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DATA_DIR       = "data"
OUTPUT_DIR     = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOOKAHEAD_DAYS = 5
THRESHOLD      = 0.02
LABEL_MAP      = {0: "SELL", 1: "HOLD", 2: "BUY"}
LABEL_COLORS   = {0: "#ff4d6d", 1: "#f5c542", 2: "#00e5a0"}

TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK",
           "ICICIBANK", "ITC", "SBIN", "WIPRO"]


def generate_labels(df):
    close = df["Close"].squeeze()
    df["future_return"] = close.shift(-LOOKAHEAD_DAYS) / close - 1
    conditions = [df["future_return"] > THRESHOLD, df["future_return"] < -THRESHOLD]
    df["label"] = np.select(conditions, [2, 0], default=1)
    df = df.iloc[:-LOOKAHEAD_DAYS].copy()
    df["label_str"] = df["label"].map(LABEL_MAP)
    return df


def plot_label_distribution(all_dfs):
    tickers = list(all_dfs.keys())
    fig, axes = plt.subplots(1, len(tickers), figsize=(16, 4))
    fig.patch.set_facecolor("#0a0b0f")
    for ax, ticker in zip(axes, tickers):
        df = all_dfs[ticker]
        counts = df["label_str"].value_counts().reindex(["BUY","HOLD","SELL"]).fillna(0)
        colors = [LABEL_COLORS[2], LABEL_COLORS[1], LABEL_COLORS[0]]
        bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="none")
        ax.set_facecolor("#0d0e14")
        ax.set_title(ticker, color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    str(int(val)), ha="center", color="white", fontsize=7)
    fig.suptitle("BUY / HOLD / SELL Label Distribution — NIFTY 50",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/label_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0b0f")
    plt.close()
    print(f"  📊 Saved → {OUTPUT_DIR}/label_distribution.png")


def plot_signals_on_price(ticker, df):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0a0b0f"); ax.set_facecolor("#0d0e14")
    close = df["Close"].squeeze()
    ax.plot(df.index, close, color="#4a9eff", lw=1.2, label="Close Price", zorder=2)
    buys  = df[df["label"] == 2]
    sells = df[df["label"] == 0]
    ax.scatter(buys.index,  buys["Close"],  color="#00e5a0", s=10, label="BUY",  zorder=3, alpha=0.7)
    ax.scatter(sells.index, sells["Close"], color="#ff4d6d", s=10, label="SELL", zorder=3, alpha=0.7)
    ax.set_title(f"{ticker} — BUY/SELL Signals on Price", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#888", labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor("#333")
    ax.legend(facecolor="#1a1b20", edgecolor="#333", labelcolor="white", fontsize=9)
    ax.set_ylabel("Price (₹)", color="#888")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{ticker}_signals.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0b0f")
    plt.close()


def run_labelling():
    print("\n🏷️  Phase 2: Label Generation")
    print("=" * 50)
    print(f"  Lookahead : {LOOKAHEAD_DAYS} trading days")
    print(f"  Threshold : ±{THRESHOLD*100:.0f}% return\n")

    all_dfs = {}
    for ticker in TICKERS:
        fpath = f"{DATA_DIR}/{ticker}_features.csv"
        if not os.path.exists(fpath):
            print(f"  ❌ {ticker}: features file not found, run Phase 1 first")
            continue
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        df = generate_labels(df)
        df.to_csv(f"{DATA_DIR}/{ticker}_labelled.csv")
        all_dfs[ticker] = df

        counts = df["label_str"].value_counts()
        total  = len(df)
        print(f"  ✅ {ticker:<12} BUY:{counts.get('BUY',0):>4} ({counts.get('BUY',0)/total*100:.0f}%)  "
              f"HOLD:{counts.get('HOLD',0):>4} ({counts.get('HOLD',0)/total*100:.0f}%)  "
              f"SELL:{counts.get('SELL',0):>4} ({counts.get('SELL',0)/total*100:.0f}%)")

    plot_label_distribution(all_dfs)
    print(f"\n  Plotting signal overlays...")
    for ticker, df in all_dfs.items():
        plot_signals_on_price(ticker, df)
        print(f"    → {OUTPUT_DIR}/{ticker}_signals.png")

    print("\n✅ Phase 2 complete → Ready for Phase 3: Model Training\n")
    return all_dfs


if __name__ == "__main__":
    run_labelling()
