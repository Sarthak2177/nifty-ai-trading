"""
app.py — NIFTY 50 AI Trading Signal Dashboard
===============================================
Run with: streamlit run app.py

Pages:
  1. Dashboard   — Signal cards for all 8 stocks
  2. Backtest    — Performance charts per stock
  3. SHAP        — Feature importance charts
  4. AI Chatbot  — LangChain + Groq chatbot
  5. About       — Project explanation
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px

load_dotenv()

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title  = "NIFTY 50 AI Signal Generator",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ──────────────────────────────────────────
# CUSTOM CSS — Dark Theme
# ──────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0b0f; color: #ffffff; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0d0e14; border-right: 1px solid #1e1f26; }
    
    /* Cards */
    .signal-card {
        background: #0d0e14;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        border-left: 4px solid;
    }
    .buy-card  { border-color: #00e5a0; }
    .sell-card { border-color: #ff4d6d; }
    .hold-card { border-color: #f5c542; }

    .signal-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
        letter-spacing: 2px;
    }
    .buy-badge  { background: rgba(0,229,160,0.15); color: #00e5a0; border: 1px solid #00e5a040; }
    .sell-badge { background: rgba(255,77,109,0.15); color: #ff4d6d; border: 1px solid #ff4d6d40; }
    .hold-badge { background: rgba(245,197,66,0.15); color: #f5c542; border: 1px solid #f5c54240; }

    .metric-box {
        background: #13141a;
        border: 1px solid #1e1f26;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 26px; font-weight: bold; color: #ffffff; }
    .metric-label { font-size: 12px; color: #666; margin-top: 4px; }

    /* Chat */
    .chat-msg-user { background: #1a1b25; border-radius: 10px; padding: 12px 16px; margin: 6px 0; border-left: 3px solid #4a9eff; }
    .chat-msg-ai   { background: #0f1018; border-radius: 10px; padding: 12px 16px; margin: 6px 0; border-left: 3px solid #00e5a0; }

    /* Headers */
    h1, h2, h3 { color: #ffffff !important; }
    .stMetric label { color: #888 !important; }
    
    /* Plotly chart background fix */
    .js-plotly-plot { border-radius: 12px; }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    
    div[data-testid="stMetricValue"] { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────
TICKERS = ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","ITC","SBIN","WIPRO"]
SIGNAL_COLORS = {"BUY": "#00e5a0", "HOLD": "#f5c542", "SELL": "#ff4d6d"}

@st.cache_data
def load_snapshot():
    path = "outputs/signal_snapshot.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_labelled(ticker):
    path = f"data/{ticker}_labelled.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    return None


# ──────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 NIFTY 50 AI Signals")
    st.markdown("*Ensemble ML · XGBoost + LSTM*")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "📊 Backtest Charts",
        "🔍 SHAP Explainability",
        "🤖 AI Chatbot",
        "ℹ️ About Project",
    ])

    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- Dataset: NIFTY 50 (Kaggle)")
    st.markdown("- Period: 2015 – 2021")
    st.markdown("- Companies: 8")
    st.markdown("- Features: 32")
    st.markdown("- Models: XGBoost + LSTM")
    st.markdown("---")
    st.caption("⚠️ Decision support only. Not financial advice.")


# ──────────────────────────────────────────
# PAGE 1 — DASHBOARD
# ──────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown("# 📈 NIFTY 50 AI Trading Signal Dashboard")
    st.markdown("*Real-time signals powered by Ensemble ML (XGBoost + LSTM)*")
    st.markdown("---")

    snapshot = load_snapshot()

    if snapshot is None:
        st.error("Signal data not found. Please run phase4_ensemble_shap.py first.")
        st.stop()

    # ── Top metrics ──
    col1, col2, col3, col4 = st.columns(4)
    buy_count  = len(snapshot[snapshot["Signal"] == "BUY"])
    hold_count = len(snapshot[snapshot["Signal"] == "HOLD"])
    sell_count = len(snapshot[snapshot["Signal"] == "SELL"])
    avg_conf   = snapshot["Confidence"].str.replace("%","").astype(float).mean()

    with col1:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-value' style='color:#00e5a0'>{buy_count}</div>
            <div class='metric-label'>BUY Signals</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-value' style='color:#f5c542'>{hold_count}</div>
            <div class='metric-label'>HOLD Signals</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-value' style='color:#ff4d6d'>{sell_count}</div>
            <div class='metric-label'>SELL Signals</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-value'>{avg_conf:.0f}%</div>
            <div class='metric-label'>Avg Confidence</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Signal Cards")

    # Filter
    filter_signal = st.selectbox("Filter by Signal", ["ALL", "BUY", "HOLD", "SELL"])
    filtered = snapshot if filter_signal == "ALL" else snapshot[snapshot["Signal"] == filter_signal]

    # ── Signal Cards ──
    cols = st.columns(2)
    for i, (_, row) in enumerate(filtered.iterrows()):
        signal   = row["Signal"]
        color    = SIGNAL_COLORS[signal]
        css_cls  = signal.lower()

        with cols[i % 2]:
            conf_val = int(row["Confidence"].replace("%",""))
            conf_bar = "█" * (conf_val // 5) + "░" * (20 - conf_val // 5)

            st.markdown(f"""
            <div class='signal-card {css_cls}-card'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px'>
                    <span style='font-size:20px; font-weight:bold; color:white'>{row['Ticker']}</span>
                    <span class='signal-badge {css_cls}-badge'>{signal}</span>
                </div>
                <div style='margin-bottom:10px'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:4px'>
                        <span style='font-size:12px; color:#666'>Confidence</span>
                        <span style='font-size:12px; color:{color}; font-weight:bold'>{row['Confidence']}</span>
                    </div>
                    <div style='background:#1a1b25; border-radius:4px; height:6px'>
                        <div style='width:{conf_val}%; background:{color}; height:6px; border-radius:4px'></div>
                    </div>
                </div>
                <div style='margin-bottom:10px'>
                    <span style='font-size:11px; color:#555; text-transform:uppercase; letter-spacing:1px'>SHAP Drivers</span><br>
                    <span style='font-size:12px; color:{color}'>① {row['Top_Feature_1']}</span> &nbsp;
                    <span style='font-size:12px; color:{color}'>② {row['Top_Feature_2']}</span> &nbsp;
                    <span style='font-size:12px; color:{color}'>③ {row['Top_Feature_3']}</span>
                </div>
                <div style='display:flex; gap:16px'>
                    <div>
                        <span style='font-size:11px; color:#555'>Strategy</span><br>
                        <span style='font-size:14px; font-weight:bold; color:white'>{row['Strategy_Return']}</span>
                    </div>
                    <div>
                        <span style='font-size:11px; color:#555'>B&H</span><br>
                        <span style='font-size:14px; color:#888'>{row['BH_Return']}</span>
                    </div>
                    <div>
                        <span style='font-size:11px; color:#555'>Sharpe</span><br>
                        <span style='font-size:14px; color:white'>{row['Sharpe']}</span>
                    </div>
                    <div>
                        <span style='font-size:11px; color:#555'>Win Rate</span><br>
                        <span style='font-size:14px; color:white'>{row['Win_Rate']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Signal distribution pie chart ──
    st.markdown("### Signal Distribution")
    counts = snapshot["Signal"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.6,
        marker_colors=[SIGNAL_COLORS.get(l, "#888") for l in counts.index],
        textinfo="label+percent",
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        paper_bgcolor="#0d0e14", plot_bgcolor="#0d0e14",
        font=dict(color="white"),
        showlegend=False, height=320,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────
# PAGE 2 — BACKTEST CHARTS
# ──────────────────────────────────────────
elif page == "📊 Backtest Charts":
    st.markdown("# 📊 Backtest Performance")
    st.markdown("*Strategy returns vs Buy & Hold — ₹10,000 starting capital*")
    st.markdown("---")

    snapshot = load_snapshot()

    # Summary table
    if snapshot is not None:
        st.markdown("### Performance Summary")
        display = snapshot[["Ticker","Signal","Confidence","Strategy_Return","BH_Return","Sharpe","Win_Rate"]].copy()
        display.columns = ["Ticker","Signal","Confidence","Strategy","Buy & Hold","Sharpe","Win Rate"]

        def color_return(val):
            v = float(val.replace("%","").replace("+",""))
            color = "#00e5a0" if v > 0 else "#ff4d6d"
            return f"color: {color}"

        st.dataframe(
            display.style.applymap(color_return, subset=["Strategy","Buy & Hold"]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("### Backtest Charts")
    ticker = st.selectbox("Select Stock", TICKERS)

    img_path = f"outputs/{ticker}_backtest.png"
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning(f"Backtest chart not found for {ticker}. Run phase4_ensemble_shap.py first.")

    # Strategy vs BH bar chart
    if snapshot is not None:
        st.markdown("### Strategy vs Buy & Hold — All Stocks")
        strat = snapshot["Strategy_Return"].str.replace("%","").str.replace("+","").astype(float)
        bh    = snapshot["BH_Return"].str.replace("%","").str.replace("+","").astype(float)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Strategy", x=snapshot["Ticker"], y=strat,
            marker_color=[SIGNAL_COLORS[s] for s in snapshot["Signal"]],
            text=[f"{v:+.1f}%" for v in strat], textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Buy & Hold", x=snapshot["Ticker"], y=bh,
            marker_color="#4a9eff", opacity=0.5,
            text=[f"{v:+.1f}%" for v in bh], textposition="outside",
        ))
        fig.update_layout(
            barmode="group", paper_bgcolor="#0d0e14", plot_bgcolor="#0d0e14",
            font=dict(color="white"), height=400,
            yaxis=dict(title="Return %", gridcolor="#1e1f26"),
            xaxis=dict(gridcolor="#1e1f26"),
            legend=dict(bgcolor="#0d0e14", bordercolor="#1e1f26"),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────
# PAGE 3 — SHAP
# ──────────────────────────────────────────
elif page == "🔍 SHAP Explainability":
    st.markdown("# 🔍 SHAP Feature Explainability")
    st.markdown("*Which technical indicators drive each signal?*")
    st.markdown("---")

    st.info("💡 SHAP (SHapley Additive Explanations) shows which features the model relied on most to make each prediction. Higher value = more important.")

    ticker = st.selectbox("Select Stock", TICKERS)

    img_path = f"outputs/{ticker}_shap.png"
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning(f"SHAP chart not found for {ticker}.")

    # Show all SHAP charts in grid
    st.markdown("### All Stocks — Top SHAP Features")
    cols = st.columns(2)
    for i, t in enumerate(TICKERS):
        path = f"outputs/{t}_shap.png"
        if os.path.exists(path):
            with cols[i % 2]:
                st.markdown(f"**{t}**")
                st.image(path, use_container_width=True)

    # SHAP driver table from snapshot
    snapshot = load_snapshot()
    if snapshot is not None:
        st.markdown("### Top 3 SHAP Drivers per Stock")
        shap_df = snapshot[["Ticker","Signal","Top_Feature_1","Top_Feature_2","Top_Feature_3"]].copy()
        shap_df.columns = ["Ticker","Signal","Driver 1","Driver 2","Driver 3"]
        st.dataframe(shap_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────
# PAGE 4 — AI CHATBOT
# ──────────────────────────────────────────
elif page == "🤖 AI Chatbot":
    st.markdown("# 🤖 AI Trading Signal Chatbot")
    st.markdown("*Ask anything about your NIFTY 50 signals — powered by LangChain + Groq*")
    st.markdown("---")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found! Add it to your .env file.")
        st.code("GROQ_API_KEY=your_key_here", language="bash")
        st.stop()

    # Load signal data for context
    snapshot = load_snapshot()
    if snapshot is None:
        st.error("Signal data not found. Run phase4_ensemble_shap.py first.")
        st.stop()

    # Build context string
    signal_context = "CURRENT NIFTY 50 SIGNALS:\n\n"
    for _, row in snapshot.iterrows():
        signal_context += (
            f"{row['Ticker']}: {row['Signal']} | Confidence: {row['Confidence']} | "
            f"Drivers: {row['Top_Feature_1']}, {row['Top_Feature_2']}, {row['Top_Feature_3']} | "
            f"Strategy: {row['Strategy_Return']} | B&H: {row['BH_Return']} | "
            f"Sharpe: {row['Sharpe']} | WinRate: {row['Win_Rate']}\n"
        )

    SYSTEM_PROMPT = f"""You are an AI trading signal analyst for NIFTY 50 Indian stocks.
You were built using XGBoost + LSTM ensemble ML models trained on 6 years of NSE data.
You provide decision support — NOT financial advice.
Always mention confidence scores and SHAP drivers when relevant.
Always add a disclaimer that past performance doesn't guarantee future returns.

{signal_context}"""

    # Init chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Example questions
    st.markdown("**💡 Try asking:**")
    example_cols = st.columns(4)
    examples = [
        "Which stocks have BUY signal?",
        "Why is SBIN showing SELL?",
        "Best Sharpe ratio stock?",
        "Compare RELIANCE and TCS",
    ]
    for i, (col, q) in enumerate(zip(example_cols, examples)):
        with col:
            if st.button(q, key=f"ex_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.rerun()

    st.markdown("---")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""<div class='chat-msg-user'>
                <span style='color:#4a9eff; font-size:12px; font-weight:bold'>YOU</span><br>
                {msg['content']}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='chat-msg-ai'>
                <span style='color:#00e5a0; font-size:12px; font-weight:bold'>AI ANALYST</span><br>
                {msg['content']}
            </div>""", unsafe_allow_html=True)

    # Generate AI response for last unanswered user message
    if (st.session_state.chat_history and
            st.session_state.chat_history[-1]["role"] == "user"):
        with st.spinner("Analysing signals..."):
            try:
                from langchain_groq import ChatGroq
                from langchain.schema import SystemMessage, HumanMessage, AIMessage

                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1024,
                )
                messages = [SystemMessage(content=SYSTEM_PROMPT)]
                for msg in st.session_state.chat_history[-10:]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))

                response = llm.invoke(messages)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # Chat input
    user_input = st.chat_input("Ask about any NIFTY 50 stock signal...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ──────────────────────────────────────────
# PAGE 5 — ABOUT
# ──────────────────────────────────────────
elif page == "ℹ️ About Project":
    st.markdown("# ℹ️ About This Project")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Project Title")
        st.info("AI-Based Smart Trading Signal Generator for NIFTY 50 using Ensemble ML with Explainable AI")

        st.markdown("### 📦 Dataset")
        st.markdown("""
- **Source:** Kaggle — NIFTY 50 Stock Market Dataset
- **Origin:** NSE India (National Stock Exchange)
- **Period:** 2015 – 2021
- **Companies:** 8 major NIFTY 50 stocks
- **Records:** ~1,517 trading days per stock
- **Raw features:** Open, High, Low, Close, Volume
        """)

        st.markdown("### 🔧 Feature Engineering")
        st.markdown("""
32 technical indicators across 5 categories:
- **Trend:** EMA, SMA, MACD, ADX
- **Momentum:** RSI, Stochastic, Williams %R, CCI
- **Volatility:** Bollinger Bands, ATR
- **Volume:** OBV, CMF, Volume Ratio
- **Price:** Returns, Distance from MAs
        """)

    with col2:
        st.markdown("### 🤖 Models Used")
        st.markdown("""
**XGBoost (Gradient Boosting)**
- Input: 32 features (flat vector)
- 300 estimators, depth 5
- Class-weighted for balance
- Best for tabular pattern detection

**LSTM (Recurrent Neural Network)**
- Input: 20-day sequence × 32 features
- Architecture: LSTM(64) → LSTM(32) → Dense(3)
- Dropout layers prevent overfitting
- Best for temporal pattern detection

**Ensemble**
- XGBoost 55% + LSTM 45% weighted average
- Output: Signal + Confidence Score
        """)

        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
| Component | Technology |
|---|---|
| Data | Kaggle, Pandas |
| Features | TA-Lib (ta) |
| ML Models | XGBoost, TensorFlow/Keras |
| Explainability | SHAP |
| Backtesting | Custom engine |
| AI Chatbot | LangChain + Groq |
| CI/CD | GitHub Actions |
| Frontend | Streamlit |
        """)

    st.markdown("---")
    st.markdown("### 🗺️ Project Pipeline")
    st.markdown("""
```
Raw OHLCV Data (Kaggle NIFTY 50)
        ↓
Phase 1: Feature Engineering → 32 Technical Indicators
        ↓
Phase 2: Label Generation → BUY / HOLD / SELL
        ↓
Phase 3: Model Training → XGBoost + LSTM
        ↓
Phase 4: Ensemble + SHAP + Backtesting
        ↓
Phase 5: LangChain AI Chatbot (Groq LLaMA 3)
        ↓
Frontend: Streamlit Dashboard (this app)
```
    """)

    st.warning("⚠️ This system provides decision support only and is NOT financial advice. Always conduct your own research before investing.")
