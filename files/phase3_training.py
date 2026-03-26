"""
PHASE 3 — Model Training: XGBoost + LSTM
==========================================
Trains two models per ticker:
  1. XGBoost  — gradient boosted trees on all 32 features
  2. LSTM     — recurrent network on 20-day sequences

Both output class probabilities used later for ensemble.
"""

import pandas as pd
import numpy as np
import os, glob, joblib, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR   = "data"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK",
           "ICICIBANK", "ITC", "SBIN", "WIPRO"]

SEQUENCE_LEN = 20
TEST_SPLIT   = 0.2

FEATURE_COLS = [
    "ema_9","ema_21","ema_50","sma_20","macd","macd_signal","macd_diff","adx",
    "rsi_14","rsi_7","stoch_k","stoch_d","williams_r","cci",
    "bb_width","bb_pct","atr_14","obv","cmf","volume_ratio",
    "returns_1d","returns_5d","returns_10d","high_low_pct","close_open_pct",
    "dist_ema9","dist_ema21","dist_ema50",
]


def train_xgboost(X_train, y_train, X_test, y_test):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    sw = np.array([dict(zip(classes, weights))[yi] for yi in y_train])

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sw,
              eval_set=[(X_test, y_test)], verbose=False)
    acc   = accuracy_score(y_test, model.predict(X_test))
    proba = model.predict_proba(X_test)
    return model, proba, acc


def train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq, n_features):
    from tensorflow import keras
    from tensorflow.keras import layers

    classes = np.unique(y_train_seq)
    weights = compute_class_weight("balanced", classes=classes, y=y_train_seq)
    cw = dict(enumerate(weights))

    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LEN, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=30, batch_size=32,
        class_weight=cw, verbose=1,
    )
    proba = model.predict(X_test_seq, verbose=0)
    acc   = accuracy_score(y_test_seq, np.argmax(proba, axis=1))
    return model, proba, acc, history


def plot_lstm_training(history, ticker):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0a0b0f")
    epochs = range(1, len(history.history["loss"]) + 1)
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0d0e14")
        ax.tick_params(colors="#888")
        for s in ax.spines.values(): s.set_edgecolor("#333")

    ax1.plot(epochs, history.history["loss"],     color="#4a9eff", label="Train")
    ax1.plot(epochs, history.history["val_loss"],  color="#ff4d6d", label="Val", linestyle="--")
    ax1.set_title(f"{ticker} LSTM — Loss", color="white"); ax1.legend(facecolor="#1a1b20", edgecolor="#333", labelcolor="white")

    ax2.plot(epochs, history.history["accuracy"],    color="#00e5a0", label="Train")
    ax2.plot(epochs, history.history["val_accuracy"], color="#f5c542", label="Val", linestyle="--")
    ax2.set_title(f"{ticker} LSTM — Accuracy", color="white"); ax2.legend(facecolor="#1a1b20", edgecolor="#333", labelcolor="white")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{ticker}_lstm_training.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0b0f")
    plt.close()


def run_training():
    print("\n🤖 Phase 3: Model Training (XGBoost + LSTM)")
    print("=" * 50)

    results = {}

    for ticker in TICKERS:
        fpath = f"{DATA_DIR}/{ticker}_labelled.csv"
        if not os.path.exists(fpath):
            print(f"\n  ❌ {ticker}: labelled file not found, run Phase 2 first")
            continue

        print(f"\n  ── {ticker} ──")
        df     = pd.read_csv(fpath, index_col=0, parse_dates=True)
        avail  = [c for c in FEATURE_COLS if c in df.columns]
        X      = df[avail].values
        y      = df["label"].values
        split  = int(len(X) * (1 - TEST_SPLIT))

        X_train_raw, X_test_raw = X[:split], X[split:]
        y_train, y_test         = y[:split], y[split:]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)
        joblib.dump(scaler, f"{MODEL_DIR}/{ticker}_scaler.pkl")

        # XGBoost
        print(f"  [XGBoost] Training...")
        xgb_model, xgb_proba, xgb_acc = train_xgboost(X_train, y_train, X_test, y_test)
        xgb_model.save_model(f"{MODEL_DIR}/{ticker}_xgb.json")
        print(f"  [XGBoost] Accuracy: {xgb_acc*100:.1f}%")

        # LSTM sequences
        def make_seq(X, y):
            Xs, ys = [], []
            for i in range(SEQUENCE_LEN, len(X)):
                Xs.append(X[i-SEQUENCE_LEN:i]); ys.append(y[i])
            return np.array(Xs), np.array(ys)

        X_seq,      y_seq      = make_seq(X_train, y_train)
        X_seq_test, y_seq_test = make_seq(X_test,  y_test)

        print(f"  [LSTM]    Training (30 epochs)...")
        lstm_model, lstm_proba, lstm_acc, hist = train_lstm(
            X_seq, y_seq, X_seq_test, y_seq_test, len(avail))
        lstm_model.save(f"{MODEL_DIR}/{ticker}_lstm.keras")
        plot_lstm_training(hist, ticker)
        print(f"  [LSTM]    Accuracy: {lstm_acc*100:.1f}%")

        n = len(lstm_proba)
        results[ticker] = {
            "xgb_proba":  xgb_proba[-n:],
            "lstm_proba": lstm_proba,
            "y_test":     y_test[-n:],
            "xgb_acc":    xgb_acc,
            "lstm_acc":   lstm_acc,
        }

    print("\n\n  📈 Training Summary:")
    print(f"  {'Ticker':<12} {'XGBoost':>10} {'LSTM':>10}")
    print(f"  {'-'*34}")
    for t, r in results.items():
        print(f"  {t:<12} {r['xgb_acc']*100:>9.1f}% {r['lstm_acc']*100:>9.1f}%")

    print("\n✅ Phase 3 complete → Models saved to models/")
    print("   Ready for Phase 4: Ensemble + SHAP + Backtesting\n")
    return results


if __name__ == "__main__":
    run_training()
