"""
PHASE 1 — Data Pipeline & Feature Engineering
==============================================
Loads real NIFTY 50 data from Kaggle CSVs and engineers
32 technical indicators as ML features.
"""

import pandas as pd
import numpy as np
import ta
import os
import warnings
warnings.filterwarnings("ignore")

DATA_SRC   = r"C:\Users\Sarthak\OneDrive\Desktop\ML\archive"
DATA_DIR   = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS    = ["RELIANCE", "TCS", "INFY", "HDFCBANK",
              "ICICIBANK", "ITC", "SBIN", "WIPRO"]
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"


def load_ticker(ticker):
    path = os.path.join(DATA_SRC, f"{ticker}.csv")
    df   = pd.read_csv(path, parse_dates=["Date"])
    df   = df.sort_values("Date").reset_index(drop=True)
    df   = df[["Date","Open","High","Low","Close","Volume"]].copy()
    df.set_index("Date", inplace=True)
    df   = df.loc[START_DATE:END_DATE]
    df   = df[(df["Close"] > 0) & (df["Volume"] > 0)]
    df.dropna(inplace=True)
    return df


def engineer_features(df):
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    df["ema_9"]       = ta.trend.ema_indicator(close, window=9)
    df["ema_21"]      = ta.trend.ema_indicator(close, window=21)
    df["ema_50"]      = ta.trend.ema_indicator(close, window=50)
    df["sma_20"]      = ta.trend.sma_indicator(close, window=20)

    macd              = ta.trend.MACD(close)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    df["adx"]         = ta.trend.adx(high, low, close, window=14)

    df["rsi_14"]      = ta.momentum.rsi(close, window=14)
    df["rsi_7"]       = ta.momentum.rsi(close, window=7)
    stoch             = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"]     = stoch.stoch()
    df["stoch_d"]     = stoch.stoch_signal()
    df["williams_r"]  = ta.momentum.williams_r(high, low, close, lbp=14)
    df["cci"]         = ta.trend.cci(high, low, close, window=20)

    bb                = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = bb.bollinger_wband()
    df["bb_pct"]      = bb.bollinger_pband()
    df["atr_14"]      = ta.volatility.average_true_range(high, low, close, window=14)

    df["obv"]         = ta.volume.on_balance_volume(close, volume)
    df["cmf"]         = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
    df["volume_sma"]  = volume.rolling(20).mean()
    df["volume_ratio"]= volume / df["volume_sma"]

    df["returns_1d"]     = close.pct_change(1)
    df["returns_5d"]     = close.pct_change(5)
    df["returns_10d"]    = close.pct_change(10)
    df["high_low_pct"]   = (high - low) / close
    df["close_open_pct"] = (close - df["Open"].squeeze()) / df["Open"].squeeze()
    df["dist_ema9"]      = (close - df["ema_9"])  / close
    df["dist_ema21"]     = (close - df["ema_21"]) / close
    df["dist_ema50"]     = (close - df["ema_50"]) / close

    df.dropna(inplace=True)
    return df


def run_pipeline():
    print("\n📥 Phase 1: Data Pipeline & Feature Engineering")
    print("=" * 52)
    print(f"  Source     : Kaggle NIFTY 50 dataset (real data)")
    print(f"  Date range : {START_DATE} → {END_DATE}")
    print(f"  Tickers    : {TICKERS}\n")

    all_dfs = {}
    for ticker in TICKERS:
        try:
            raw      = load_ticker(ticker)
            featured = engineer_features(raw)
            featured.to_csv(f"{DATA_DIR}/{ticker}_features.csv")
            all_dfs[ticker] = featured
            print(f"  ✅ {ticker:<12} {len(featured):>4} rows × {len(featured.columns)} features"
                  f"  [{featured.index[0].date()} → {featured.index[-1].date()}]")
        except FileNotFoundError:
            print(f"  ❌ {ticker}: CSV not found")

    print(f"\n✅ Phase 1 complete → Ready for Phase 2\n")
    return all_dfs


if __name__ == "__main__":
    run_pipeline()
