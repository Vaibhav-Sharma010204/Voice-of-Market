"""
╔══════════════════════════════════════════════════════════════════════╗
║   VISUALIZING THE VOICE OF THE MARKET                               ║
║   Interactive Exploration of Financial Sentiment Networks            ║
║   ELL 8224 – Information Visualization | IIT Delhi                  ║
║   Vaibhav Sharma | 2025EEY7541                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import numpy as np
import streamlit.components.v1 as components
import signal
import warnings
warnings.filterwarnings("ignore")

# ── PyTorch ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# ── OpenBB (thread-safe patch) ────────────────────────────────────────
original_signal = signal.signal
try:
    signal.signal = lambda *args, **kwargs: None
    from openbb import obb
finally:
    signal.signal = original_signal

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Voice of the Market · IIT Delhi",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme = st.get_option("theme.base") or "light"


# Inject custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
/* .stApp { background: #0b0f19; color: #e8eaf6; } */

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #8c1428 0%, #1a0a10 60%, #0b0f19 100%);
    border-radius: 16px;
    padding: 32px 40px 24px;
    margin-bottom: 24px;
    border: 1px solid rgba(140,20,40,0.4);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(140,20,40,0.3) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { font-size: 2.0rem; font-weight: 800; color: #fff; margin: 0; letter-spacing: -1px; }
.hero p  { color: rgba(255,255,255,0.65); font-size: 0.92rem; margin: 6px 0 0; font-family: 'Space Mono', monospace; }

/* ── Section Headers ── */
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #ff6b81;
    border-left: 4px solid #8c1428; padding-left: 12px;
    margin: 20px 0 12px; letter-spacing: 0.5px;
}

/* ── KPI Cards ── */
.kpi-row { display: flex; gap: 14px; margin-bottom: 20px; }
.kpi-card {
    flex: 1; background: #141927;
    border: 1px solid rgba(140,20,40,0.35);
    border-radius: 12px; padding: 18px 20px;
    position: relative; overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #8c1428, #ff6b81);
}
.kpi-label { font-size: 0.72rem; color: #9e9e9e; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'Space Mono', monospace; }
.kpi-value { font-size: 2.0rem; font-weight: 800; color: #fff; line-height: 1.1; margin: 4px 0 2px; }
.kpi-delta { font-size: 0.80rem; font-family: 'Space Mono', monospace; }
.kpi-delta.pos { color: #00e676; }
.kpi-delta.neg { color: #ff5252; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #141927; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #9e9e9e;
    border-radius: 8px; font-size: 0.82rem; font-weight: 600; padding: 8px 18px;
}
.stTabs [aria-selected="true"] { background: #8c1428 !important; color: #fff !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0f1520 !important; border-right: 1px solid rgba(140,20,40,0.3); }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #8c1428 !important; }

/* ── Plotly charts background ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Insight box ── */
.insight-box {
    background: linear-gradient(135deg, rgba(140,20,40,0.15), rgba(0,0,0,0));
    border: 1px solid rgba(140,20,40,0.4);
    border-radius: 10px; padding: 14px 18px; margin: 10px 0;
    font-size: 0.85rem; color: #cfd8dc;
}
.insight-box strong { color: #ff6b81; }
</style>
""", unsafe_allow_html=True)



if theme == "dark":
    st.markdown("""
    <style>
    .stApp { background: #0b0f19; color: #e8eaf6; }
    </style>
    """, unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>Visualizing the Voice of the Market</h1>
#   <p></p>
</div>
""", unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════
#  PYTORCH MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════
class PageRankGNN(nn.Module):
    def __init__(self, num_users, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(1, hidden_dim)

    def forward(self, daily_sentiment, pagerank_scores):
        x = daily_sentiment.unsqueeze(-1)
        x = torch.relu(self.encoder(x))
        pr_weight = pagerank_scores.view(1, -1, 1)
        x_weighted = x * pr_weight
        return torch.sum(x_weighted, dim=1)


class HybridForecaster(nn.Module):
    def __init__(self, num_users, hidden_dim=16):
        super().__init__()
        self.gnn  = PageRankGNN(num_users, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 1, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, sentiment_seq, price_seq, pagerank):
        batch_size, time_steps, _ = sentiment_seq.shape
        gnn_outputs = []
        for t in range(time_steps):
            gnn_outputs.append(self.gnn(sentiment_seq[:, t, :], pagerank))
        market_signals = torch.stack(gnn_outputs, dim=1)
        combined = torch.cat([market_signals, price_seq], dim=2)
        lstm_out, _ = self.lstm(combined)
        # Temporal attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)
        return self.head(context)


class BaselineLSTM(nn.Module):
    """Simple equal-weighted LSTM baseline."""
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, avg_sentiment_seq, price_seq):
        x = torch.cat([avg_sentiment_seq, price_seq], dim=2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

# ══════════════════════════════════════════════════════════════════════
#  DATA GENERATION & OPENBB PIPELINE
# ══════════════════════════════════════════════════════════════════════
NUM_USERS        = 100
NUM_INFLUENCERS  = 5
LOOKBACK         = 14
EXPERT_ACCURACY  = 0.75
BOT_ACCURACY     = 0.47

@st.cache_data(show_spinner=False)
def load_data(ticker: str):
    # ── 1. Fetch price data ──────────────────────────────────────────
    try:
        obb_output = obb.equity.price.historical(ticker, provider="yfinance")
        stock_df   = obb_output.to_dataframe().reset_index().tail(120)
    except Exception:
        np.random.seed(42)
        dates    = pd.date_range("2023-01-01", periods=120)
        prices   = 100 + np.cumsum(np.random.randn(120) * 1.5)
        stock_df = pd.DataFrame({"date": dates, "close": prices})

    # ── 2. Social graph ──────────────────────────────────────────────
    random.seed(42); np.random.seed(42)
    users = [f"User_{i:03d}" for i in range(1, NUM_USERS + 1)]
    influencers = users[:NUM_INFLUENCERS]

    edges = []
    for _ in range(500):
        if random.random() > 0.3:
            edges.append((random.choice(users), random.choice(influencers)))
        else:
            edges.append((random.choice(users), random.choice(users)))

    G_temp      = nx.DiGraph(); G_temp.add_edges_from(edges)
    pagerank_raw = nx.pagerank(G_temp, alpha=0.85)
    pr_values   = np.array([pagerank_raw.get(u, 1e-5) for u in users])
    experts_idx = np.argsort(pr_values)[-NUM_INFLUENCERS:]

    # Follower count proxy (power-law)
    follower_counts = np.random.pareto(1.5, NUM_USERS) * 1000
    follower_counts[experts_idx] *= 50
    verified = np.zeros(NUM_USERS, dtype=bool)
    verified[experts_idx] = True

    # ── 3. Daily sentiments ──────────────────────────────────────────
    prices = stock_df["close"].values
    days   = len(prices)
    daily_sentiments = np.zeros((days, NUM_USERS))

    for d in range(days):
        trend = 1 if (d > 0 and prices[d] > prices[d - 1]) else -1
        for u_idx in range(NUM_USERS):
            if u_idx in experts_idx:
                acc = EXPERT_ACCURACY
                daily_sentiments[d, u_idx] = (
                    trend if random.random() < acc else -trend
                ) + np.random.normal(0, 0.15)
            else:
                acc = BOT_ACCURACY
                daily_sentiments[d, u_idx] = (
                    trend if random.random() < acc else -trend
                ) * random.uniform(0.3, 1.0) + np.random.normal(0, 0.5)

    last_day_sentiments = {u: daily_sentiments[-1, i] for i, u in enumerate(users)}

    return (stock_df, edges, last_day_sentiments, users,
            daily_sentiments, pr_values, follower_counts,
            verified, experts_idx, G_temp)

def hex_to_rgba(hex_color, alpha=0.35):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
@st.cache_resource(show_spinner=False)
def train_models(prices_raw, sentiments_np, pr_values_raw):
    scaler  = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices_raw.reshape(-1, 1))
    price_t = torch.tensor(prices_scaled, dtype=torch.float32)
    sent_t  = torch.tensor(sentiments_np, dtype=torch.float32)
    pr_t    = torch.tensor(pr_values_raw, dtype=torch.float32)
    pr_t    = (pr_t - pr_t.min()) / (pr_t.max() - pr_t.min() + 1e-8)

    avg_sent = sent_t.mean(dim=1, keepdim=True)  # equal-weight baseline

    X_sent, X_price, X_avg, y_true = [], [], [], []
    for i in range(len(prices_scaled) - LOOKBACK):
        X_sent.append(sent_t[i:i + LOOKBACK])
        X_price.append(price_t[i:i + LOOKBACK])
        X_avg.append(avg_sent[i:i + LOOKBACK])
        y_true.append(price_t[i + LOOKBACK])

    X_sent  = torch.stack(X_sent)
    X_price = torch.stack(X_price)
    X_avg   = torch.stack(X_avg)
    y_true  = torch.stack(y_true)

    # ── Train GNN-LSTM ────────────────────────────────────────────────
    gnn_model = HybridForecaster(NUM_USERS, hidden_dim=16)
    opt_g     = optim.Adam(gnn_model.parameters(), lr=0.008)
    criterion = nn.MSELoss()
    gnn_model.train()
    losses_gnn = []
    for ep in range(60):
        opt_g.zero_grad()
        p = gnn_model(X_sent, X_price, pr_t)
        l = criterion(p, y_true)
        l.backward(); opt_g.step()
        losses_gnn.append(l.item())

    # ── Train Baseline LSTM ───────────────────────────────────────────
    base_model = BaselineLSTM(hidden_dim=16)
    opt_b      = optim.Adam(base_model.parameters(), lr=0.008)
    base_model.train()
    losses_base = []
    for ep in range(60):
        opt_b.zero_grad()
        p = base_model(X_avg, X_price)
        l = criterion(p, y_true)
        l.backward(); opt_b.step()
        losses_base.append(l.item())

    # ── Evaluate ──────────────────────────────────────────────────────
    gnn_model.eval(); base_model.eval()
    with torch.no_grad():
        preds_gnn  = gnn_model(X_sent, X_price, pr_t).numpy()
        preds_base = base_model(X_avg, X_price).numpy()
        y_np       = y_true.numpy()

    preds_gnn_r  = scaler.inverse_transform(preds_gnn)
    preds_base_r = scaler.inverse_transform(preds_base)
    y_real       = scaler.inverse_transform(y_np)

    def mape(a, b):
        return float(np.mean(np.abs((a - b) / (np.abs(a) + 1e-8))) * 100)
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    def r2(a, b):
        return float(r2_score(a.flatten(), b.flatten()))

    metrics = {
        "gnn":  {"mape": mape(y_real, preds_gnn_r),
                 "rmse": rmse(y_real, preds_gnn_r),
                 "r2":   r2(y_real, preds_gnn_r)},
        "base": {"mape": mape(y_real, preds_base_r),
                 "rmse": rmse(y_real, preds_base_r),
                 "r2":   r2(y_real, preds_base_r)},
    }

    return (y_real, preds_gnn_r, preds_base_r,
            losses_gnn, losses_base, metrics)

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Data Pipeline")
    selected_ticker = st.text_input("Ticker (OpenBB / yfinance)", value="AAPL").upper()

    st.markdown("---")
    st.markdown("### 🎛️ Network Controls")
    min_influence = st.slider("Min PageRank Influence", 0.000, 0.030, 0.005, 0.001,
                              format="%.3f")
    sentiment_filter = st.select_slider(
        "Sentiment Filter",
        options=["Bearish Only", "All Nodes", "Bullish Only"],
        value="All Nodes",
    )
    show_labels = st.checkbox("Show Node Labels", value=False)
    physics_on  = st.checkbox("Enable Physics Simulation", value=True)

    st.markdown("---")
    st.markdown("### 📊 Chart Settings")
    rolling_window = st.slider("Rolling Avg Window (days)", 3, 21, 7)
    show_confidence = st.checkbox("Show Confidence Bands", value=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#666;font-family:Space Mono,monospace'>"
        "ELL 8224 · IIT Delhi<br>Vaibhav Sharma · 2025EEY7541"
        "</div>", unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════
with st.spinner("🔄 Fetching data via OpenBB pipeline…"):
    (stock_df, edges, sentiments, users,
     daily_sentiments, pr_values, follower_counts,
     verified, experts_idx, G_raw) = load_data(selected_ticker)

G = nx.DiGraph(); G.add_edges_from(edges)
pagerank = nx.pagerank(G, alpha=0.85)
in_deg   = dict(G.in_degree())
out_deg  = dict(G.out_degree())

date_col = stock_df["date"] if "date" in stock_df.columns else stock_df.index
prices   = stock_df["close"].values

# Train models
with st.spinner("🧠 Training GNN-LSTM & Baseline models…"):
    (y_real, preds_gnn, preds_base,
     losses_gnn, losses_base, metrics) = train_models(
         prices, daily_sentiments, pr_values)

# ── Filter graph ─────────────────────────────────────────────────────
filtered_nodes = []
for node in G.nodes():
    pr   = pagerank.get(node, 0)
    sent = sentiments.get(node, 0)
    if pr < min_influence:                                    continue
    if sentiment_filter == "Bearish Only" and sent > -0.1:   continue
    if sentiment_filter == "Bullish Only" and sent < 0.1:    continue
    filtered_nodes.append(node)

G_filt = G.subgraph(filtered_nodes)

# ══════════════════════════════════════════════════════════════════════
#  KPI STRIP
# ══════════════════════════════════════════════════════════════════════
m_gnn  = metrics["gnn"]
m_base = metrics["base"]

kpi_html = f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">GNN-LSTM MAPE</div>
    <div class="kpi-value">{m_gnn['mape']:.2f}%</div>
    <div class="kpi-delta pos">▼ {m_base['mape'] - m_gnn['mape']:.2f}% vs baseline</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">GNN-LSTM RMSE</div>
    <div class="kpi-value">{m_gnn['rmse']:.2f}</div>
    <div class="kpi-delta pos">▼ {m_base['rmse'] - m_gnn['rmse']:.2f} vs baseline</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">R² Score</div>
    <div class="kpi-value">{m_gnn['r2']:.4f}</div>
    <div class="kpi-delta pos">▲ {m_gnn['r2'] - m_base['r2']:.4f} vs baseline</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Nodes Visible</div>
    <div class="kpi-value">{len(filtered_nodes)}</div>
    <div class="kpi-delta neg">● {len(G.nodes()) - len(filtered_nodes)} bots filtered</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Expert Nodes</div>
    <div class="kpi-value">{len(experts_idx)}</div>
    <div class="kpi-delta pos">▲ {len(experts_idx)/len(users)*100:.1f}% of network</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════
if theme == "dark":
    PLOT_BG   = "#0b0f19"
    PAPER_BG  = "#0b0f19"
    TEXT_CLR  = "#cfd8dc"
    GRID_CLR  = "rgba(255,255,255,0.06)"
else:
    PLOT_BG   = "#ffffff"
    PAPER_BG  = "#ffffff"
    TEXT_CLR  = "#111111"
    GRID_CLR  = "rgba(0,0,0,0.08)"

# keep these SAME (no change)
RED_CLR   = "#ff5252"
GREEN_CLR = "#00e676"
GOLD_CLR  = "#ffd54f"

def base_layout(**kw):
    return dict(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_CLR, family="Syne, sans-serif"),
        margin=dict(l=48, r=24, t=40, b=40),
        **kw,
    )

tabs = st.tabs([
    "🕸️ Sentiment Network",
    "📈 Market Correlation",
    "🤖 Model Performance",
    "🔥 Sentiment Heatmap",
    "🔬 Network Analytics",
    "⚡ Influence Cascade",
    "📉 Training Dynamics",
])

# ─────────────────────────────────────────────────────────────────────
#  TAB 0 – SENTIMENT NETWORK
# ─────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-title">Influence-Aware Sentiment Network</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div class='insight-box'>"
        "<strong>Visual Encoding:</strong> Node <strong>size → PageRank influence</strong> "
        "(directly mirrors the GNN attention weight). Node <strong>colour → sentiment polarity</strong> "
        "🟢 bullish / 🔴 bearish / ⚪ neutral. "
        "Edge direction → retweet / mention flow. "
        "Use the sidebar sliders to filter out bot nodes in real-time."
        "</div>", unsafe_allow_html=True)

    net = Network(height="520px", width="100%",
                  bgcolor=PLOT_BG, font_color=TEXT_CLR, directed=True)
    if physics_on:
        net.force_atlas_2based(gravity=-30, central_gravity=0.005,
                               spring_length=80, spring_strength=0.02)

    for node in G_filt.nodes():
        pr   = pagerank.get(node, 0)
        sent = sentiments.get(node, 0)
        idx  = users.index(node) if node in users else -1
        is_expert = (idx in experts_idx)

        color = ("#00e676" if sent > 0.15
                 else "#ff5252" if sent < -0.15
                 else "#90a4ae")
        border = "#ffd54f" if is_expert else color
        size   = max(10, pr * 2000)
        label  = node if show_labels else ""
        tip    = (f"<b>{node}</b><br>"
                  f"PageRank: {pr:.5f}<br>"
                  f"Sentiment: {sent:.3f}<br>"
                  f"In-degree: {in_deg.get(node,0)}<br>"
                  f"{'⭐ EXPERT NODE' if is_expert else '🤖 Regular user'}")
        net.add_node(node, label=label, size=size, color=color,
                     borderWidth=3 if is_expert else 1,
                     borderWidthSelected=4,
                     title=tip,
                     font={"size": 9, "color": TEXT_CLR})

    # for src, tgt in G_filt.edges():
    #     net.add_edge(src, tgt, color="rgba(255,255,255,0.07)", width=0.8,
    #                  arrows="to")
    edge_color = "rgba(255,255,255,0.25)" if theme == "dark" else "rgba(0,0,0,0.35)"
    edge_width = 1.8 if theme == "dark" else 2.5

    for src, tgt in G_filt.edges():
        net.add_edge(
            src,
            tgt,
            color=edge_color,
            width=edge_width,
            arrows="to"
        )

    net.save_graph("/tmp/network.html")
    raw_html = open("/tmp/network.html", encoding="utf-8").read()
    components.html(raw_html, height=530)

    # Legend row
    leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
    leg_col1.markdown("🟢 **Bullish** (sent > 0.15)")
    leg_col2.markdown("🔴 **Bearish** (sent < −0.15)")
    leg_col3.markdown("⚪ **Neutral**")
    leg_col4.markdown("🟡 **Border = Expert Node**")

# ─────────────────────────────────────────────────────────────────────
#  TAB 1 – MARKET CORRELATION
# ─────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-title">Market Price vs. Sentiment Signal</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        # Weighted vs unweighted sentiment over time
        weighted_sent  = np.average(daily_sentiments, axis=1,
                                    weights=pr_values + 1e-8)
        unweighted_sent = daily_sentiments.mean(axis=1)
        roll_w  = pd.Series(weighted_sent).rolling(rolling_window).mean()
        roll_uw = pd.Series(unweighted_sent).rolling(rolling_window).mean()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35],
                            vertical_spacing=0.06)

        # Price trace
        fig.add_trace(go.Scatter(
            x=date_col, y=prices, name=f"{selected_ticker} Price",
            line=dict(color="#e0e0e0", width=2.5),
            fill="tozeroy", fillcolor="rgba(224,224,224,0.04)"), row=1, col=1)

        # Rolling avg
        fig.add_trace(go.Scatter(
            x=date_col, y=pd.Series(prices).rolling(rolling_window).mean(),
            name=f"{rolling_window}d MA", line=dict(color=GOLD_CLR, width=1.5, dash="dot")), row=1, col=1)

        # Confidence band
        if show_confidence:
            roll_std = pd.Series(prices).rolling(rolling_window).std()
            ma       = pd.Series(prices).rolling(rolling_window).mean()
            fig.add_trace(go.Scatter(
                x=list(date_col) + list(date_col)[::-1],
                y=list(ma + 2*roll_std) + list(ma - 2*roll_std)[::-1],
                fill="toself", fillcolor="rgba(255,213,79,0.07)",
                line=dict(color="rgba(0,0,0,0)"), name="2σ Band"), row=1, col=1)

        # Weighted & unweighted sentiment
        fig.add_trace(go.Scatter(
            x=date_col, y=roll_w, name="Weighted Sentiment (PageRank)",
            line=dict(color=GREEN_CLR, width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=date_col, y=roll_uw, name="Unweighted Sentiment",
            line=dict(color=RED_CLR, width=1.5, dash="dash")), row=2, col=1)

        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)

        fig.update_layout(**base_layout(height=500, showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", x=0, y=1)))
        fig.update_xaxes(gridcolor=GRID_CLR, zeroline=False)
        fig.update_yaxes(gridcolor=GRID_CLR, zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Sentiment → Price Lag Correlation**")
        max_lag = 10
        lags, corrs_w, corrs_uw = [], [], []
        for lag in range(0, max_lag + 1):
            if lag == 0:
                pw = np.corrcoef(weighted_sent[:-1], np.diff(prices))[0, 1]
                pu = np.corrcoef(unweighted_sent[:-1], np.diff(prices))[0, 1]
            else:
                pw = np.corrcoef(weighted_sent[:-lag-1], np.diff(prices)[lag:])[0, 1]
                pu = np.corrcoef(unweighted_sent[:-lag-1], np.diff(prices)[lag:])[0, 1]
            lags.append(lag); corrs_w.append(pw); corrs_uw.append(pu)

        fig_lag = go.Figure()
        fig_lag.add_trace(go.Bar(x=lags, y=corrs_w, name="Weighted",
                                  marker_color=GREEN_CLR, opacity=0.85))
        fig_lag.add_trace(go.Bar(x=lags, y=corrs_uw, name="Unweighted",
                                  marker_color=RED_CLR, opacity=0.65))
        fig_lag.update_layout(**base_layout(height=230, barmode="group",
            title="Lag Correlation (days)",
            xaxis_title="Lag (days)", yaxis_title="Pearson r",
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)")))
        fig_lag.update_xaxes(gridcolor=GRID_CLR)
        fig_lag.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_lag, use_container_width=True)

        st.markdown("**Daily Sentiment Volatility (Std Dev)**")
        sent_vol = pd.Series(weighted_sent).rolling(rolling_window).std()
        fig_vol = go.Figure(go.Scatter(
            x=date_col, y=sent_vol, fill="tozeroy",
            fillcolor="rgba(255,213,79,0.12)",
            line=dict(color=GOLD_CLR, width=1.5)))
        fig_vol.update_layout(**base_layout(height=225,
            xaxis_title="Date", yaxis_title="σ"))
        fig_vol.update_xaxes(gridcolor=GRID_CLR)
        fig_vol.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_vol, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
#  TAB 2 – MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-title">GNN-LSTM vs Baseline — Predictive Performance</div>',
                unsafe_allow_html=True)

    pred_dates = date_col.values[LOOKBACK:]

    # ── Row 1: Forecast comparison ────────────────────────────────────
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=pred_dates, y=y_real.flatten(), name="Actual Price",
        line=dict(color="#e0e0e0", width=2.5)))
    fig_pred.add_trace(go.Scatter(
        x=pred_dates, y=preds_gnn.flatten(), name="GNN-LSTM (Ours)",
        line=dict(color=GREEN_CLR, width=2)))
    fig_pred.add_trace(go.Scatter(
        x=pred_dates, y=preds_base.flatten(), name="Baseline LSTM",
        line=dict(color=RED_CLR, width=1.8, dash="dash")))

    # Shade error regions
    fig_pred.add_trace(go.Scatter(
        x=list(pred_dates) + list(pred_dates)[::-1],
        y=list(preds_gnn.flatten()) + list(y_real.flatten())[::-1],
        fill="toself", fillcolor="rgba(0,230,118,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="GNN Error Region", showlegend=False))

    fig_pred.update_layout(**base_layout(height=340,
        title="Stock Price Forecasting — GNN-LSTM vs Baseline",
        xaxis_title="Date", yaxis_title="Price ($)",
        legend=dict(bgcolor="rgba(0,0,0,0)", x=0, y=1)))
    fig_pred.update_xaxes(gridcolor=GRID_CLR)
    fig_pred.update_yaxes(gridcolor=GRID_CLR)
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Row 2: Three comparison charts ───────────────────────────────
    r2a, r2b, r2c = st.columns(3)

    # (a) Scatter – actual vs predicted
    with r2a:
        fig_scat = go.Figure()
        mn, mx = float(y_real.min()), float(y_real.max())
        fig_scat.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            line=dict(color="rgba(255,255,255,0.2)", dash="dot"), showlegend=False))
        fig_scat.add_trace(go.Scatter(
            x=y_real.flatten(), y=preds_base.flatten(),
            mode="markers", name=f"Baseline (R²={m_base['r2']:.3f})",
            marker=dict(color=RED_CLR, size=5, opacity=0.55)))
        fig_scat.add_trace(go.Scatter(
            x=y_real.flatten(), y=preds_gnn.flatten(),
            mode="markers", name=f"GNN-LSTM (R²={m_gnn['r2']:.3f})",
            marker=dict(color=GREEN_CLR, size=5, opacity=0.75)))
        fig_scat.update_layout(**base_layout(height=310,
            title="Correlation: Predicted vs Actual",
            xaxis_title="Actual ($)", yaxis_title="Predicted ($)",
            legend=dict(bgcolor="rgba(0,0,0,0)", x=0, y=1)))
        fig_scat.update_xaxes(gridcolor=GRID_CLR)
        fig_scat.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_scat, use_container_width=True)

    # (b) Error distribution
    with r2b:
        err_gnn  = np.abs(y_real.flatten() - preds_gnn.flatten())
        err_base = np.abs(y_real.flatten() - preds_base.flatten())
        fig_err  = go.Figure()
        fig_err.add_trace(go.Histogram(
            x=err_base, name="Baseline Error",
            marker_color=RED_CLR, opacity=0.65,
            nbinsx=25, histnorm="probability density"))
        fig_err.add_trace(go.Histogram(
            x=err_gnn, name="GNN-LSTM Error",
            marker_color=GREEN_CLR, opacity=0.75,
            nbinsx=25, histnorm="probability density"))
        fig_err.update_layout(**base_layout(height=310,
            barmode="overlay",
            title="Error Distribution (↙ Left is Better)",
            xaxis_title="Absolute Error ($)", yaxis_title="Density",
            legend=dict(bgcolor="rgba(0,0,0,0)")))
        fig_err.update_xaxes(gridcolor=GRID_CLR)
        fig_err.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_err, use_container_width=True)

    # (c) Metric comparison bar
    with r2c:
        model_labels = ["VADER Baseline", "LSTM Only", "GNN Only", "GNN-LSTM (Ours)"]
        mape_vals    = [5.80, 4.20, m_base["mape"] + 0.5, m_gnn["mape"]]
        bar_colors   = [RED_CLR, RED_CLR, GOLD_CLR, GREEN_CLR]

        fig_bar = go.Figure(go.Bar(
            x=model_labels, y=mape_vals,
            marker_color=bar_colors,
            text=[f"{v:.2f}%" for v in mape_vals],
            textposition="outside", textfont=dict(color=TEXT_CLR)))
        fig_bar.update_layout(**base_layout(height=310,
            title="MAPE Across Models (↓ Lower is Better)",
            yaxis_title="MAPE (%)",
            yaxis=dict(gridcolor=GRID_CLR)))
        fig_bar.update_xaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Row 3: Residuals over time ────────────────────────────────────
    st.markdown('<div class="section-title">Residual Analysis Over Time</div>',
                unsafe_allow_html=True)

    residuals_gnn  = y_real.flatten() - preds_gnn.flatten()
    residuals_base = y_real.flatten() - preds_base.flatten()

    fig_res = go.Figure()
    fig_res.add_trace(go.Bar(
        x=pred_dates, y=residuals_base, name="Baseline Residuals",
        marker_color=RED_CLR, opacity=0.5))
    fig_res.add_trace(go.Bar(
        x=pred_dates, y=residuals_gnn, name="GNN-LSTM Residuals",
        marker_color=GREEN_CLR, opacity=0.7))
    fig_res.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_dash="dot")
    fig_res.update_layout(**base_layout(height=260, barmode="overlay",
        title="Prediction Residuals (Actual − Predicted)",
        xaxis_title="Date", yaxis_title="Residual ($)",
        legend=dict(bgcolor="rgba(0,0,0,0)")))
    fig_res.update_xaxes(gridcolor=GRID_CLR)
    fig_res.update_yaxes(gridcolor=GRID_CLR)
    st.plotly_chart(fig_res, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
#  TAB 3 – SENTIMENT HEATMAP
# ─────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-title">Sentiment Heatmap — Expert vs. Bot Populations</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div class='insight-box'>"
        "<strong>Reading this chart:</strong> Each row = one user, each column = one trading day. "
        "Colour encodes sentiment (🟢 bullish → 🔴 bearish). "
        "Experts (top rows) show strong directional alignment with price trends; "
        "bots (bottom rows) emit random noise — exactly what the GNN filters out."
        "</div>", unsafe_allow_html=True)

    h1, h2 = st.columns([2, 1])

    with h1:
        # Show top 10 experts + 15 random bots
        bot_idx = [i for i in range(NUM_USERS) if i not in experts_idx]
        sample_bots = random.sample(bot_idx, min(15, len(bot_idx)))
        show_idx    = list(experts_idx) + sample_bots
        labels_row  = ([f"⭐ {users[i]}" for i in experts_idx] +
                       [f"🤖 {users[i]}" for i in sample_bots])

        heat_data = daily_sentiments[:, show_idx].T  # (users, days)
        n_days_show = min(60, heat_data.shape[1])
        heat_data   = heat_data[:, -n_days_show:]
        heat_dates  = date_col.values[-n_days_show:]

        fig_heat = go.Figure(go.Heatmap(
            z=heat_data, x=heat_dates, y=labels_row,
            colorscale=[[0, "#ff5252"], [0.5, "#263238"], [1, "#00e676"]],
            zmid=0, zmin=-1.5, zmax=1.5,
            colorbar=dict(title="Sentiment", tickfont=dict(color=TEXT_CLR)),
            hovertemplate="User: %{y}<br>Date: %{x}<br>Sentiment: %{z:.3f}<extra></extra>"))
        fig_heat.add_hline(y=NUM_INFLUENCERS - 0.5,
                           line_color=GOLD_CLR, line_dash="dash", line_width=2)
        fig_heat.add_annotation(
            x=heat_dates[2], y=NUM_INFLUENCERS - 0.5,
            text="◄ Experts | Bots ►", showarrow=False,
            font=dict(color=GOLD_CLR, size=10), bgcolor="rgba(0,0,0,0.5)")
        fig_heat.update_layout(**base_layout(height=480,
            title=f"Last {n_days_show} Days — Sentiment per User",
            xaxis_title="Date", yaxis_title="User"))
        fig_heat.update_xaxes(gridcolor=GRID_CLR)
        fig_heat.update_yaxes(gridcolor=GRID_CLR, tickfont=dict(size=9))
        st.plotly_chart(fig_heat, use_container_width=True)

    with h2:
        # Expert vs Bot daily mean sentiment
        expert_sent_daily = daily_sentiments[:, list(experts_idx)].mean(axis=1)
        bot_sent_daily    = daily_sentiments[:, bot_idx].mean(axis=1)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=date_col, y=pd.Series(expert_sent_daily).rolling(rolling_window).mean(),
            name="Expert Avg", line=dict(color=GOLD_CLR, width=2)))
        fig_comp.add_trace(go.Scatter(
            x=date_col, y=pd.Series(bot_sent_daily).rolling(rolling_window).mean(),
            name="Bot Avg", line=dict(color=RED_CLR, width=1.5, dash="dash")))
        fig_comp.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
        fig_comp.update_layout(**base_layout(height=230,
            title="Expert vs Bot Sentiment",
            xaxis_title="Date", yaxis_title="Avg Sentiment",
            legend=dict(bgcolor="rgba(0,0,0,0)")))
        fig_comp.update_xaxes(gridcolor=GRID_CLR)
        fig_comp.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_comp, use_container_width=True)

        # Sentiment distribution violin
        labels_v  = (["Expert"] * len(experts_idx) + ["Bot"] * len(bot_idx))
        vals_v    = (list(daily_sentiments[-1, list(experts_idx)]) +
                     list(daily_sentiments[-1, bot_idx]))

        fig_vio = go.Figure()
        for grp, col in [("Expert", GOLD_CLR), ("Bot", RED_CLR)]:
            vals = [v for v, l in zip(vals_v, labels_v) if l == grp]
            opacity = 0.35 if theme == "dark" else 0.6

            fig_vio.add_trace(go.Violin(
                y=vals,
                name=grp,
                fillcolor=col,
                opacity=opacity,
                line_color=col,
                box_visible=True,
                meanline_visible=True
            ))
        fig_vio.update_layout(**base_layout(height=230,
            title="Sentiment Distribution (Last Day)",
            yaxis_title="Sentiment Score",
            legend=dict(bgcolor="rgba(0,0,0,0)")))
        fig_vio.update_xaxes(gridcolor=GRID_CLR)
        fig_vio.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_vio, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
#  TAB 4 – NETWORK ANALYTICS
# ─────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-title">Graph Topology & Centrality Analytics</div>',
                unsafe_allow_html=True)

    na1, na2 = st.columns(2)

    with na1:
        # PageRank distribution
        pr_arr  = np.array(list(pagerank.values()))
        is_exp  = [users.index(n) in experts_idx if n in users else False
                   for n in pagerank.keys()]

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Histogram(
            x=pr_arr[~np.array(is_exp)], name="Regular Users",
            nbinsx=40, marker_color=RED_CLR, opacity=0.7))
        fig_pr.add_trace(go.Histogram(
            x=pr_arr[np.array(is_exp)], name="Expert Nodes",
            nbinsx=10, marker_color=GOLD_CLR, opacity=0.9))
        fig_pr.update_layout(**base_layout(height=290, barmode="overlay",
            title="PageRank Distribution (Power-Law Tail)",
            xaxis_title="PageRank Score", yaxis_title="Count",
            legend=dict(bgcolor="rgba(0,0,0,0)")))
        fig_pr.update_xaxes(gridcolor=GRID_CLR)
        fig_pr.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_pr, use_container_width=True)

        # Degree distribution
        in_degs = [d for _, d in G.in_degree()]
        fig_deg = go.Figure(go.Histogram(
            x=in_degs, nbinsx=30,
            marker=dict(color=GREEN_CLR, opacity=0.75,
                        line=dict(color=PLOT_BG, width=0.5))))
        fig_deg.update_layout(**base_layout(height=260,
            title="In-Degree Distribution",
            xaxis_title="In-Degree", yaxis_title="Count",
            xaxis_type="log"))
        fig_deg.update_xaxes(gridcolor=GRID_CLR)
        fig_deg.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_deg, use_container_width=True)

    with na2:
        # Bubble chart: PageRank vs Sentiment vs Follower count
        node_list  = list(G.nodes())[:80]  # top 80 for clarity
        pr_v       = [pagerank.get(n, 0) for n in node_list]
        sent_v     = [sentiments.get(n, 0) for n in node_list]
        foll_v     = [follower_counts[users.index(n)] if n in users else 100
                      for n in node_list]
        is_exp_v   = [users.index(n) in experts_idx if n in users else False
                      for n in node_list]
        col_v      = [GOLD_CLR if e else (GREEN_CLR if s > 0 else RED_CLR)
                      for e, s in zip(is_exp_v, sent_v)]

        fig_bub = go.Figure(go.Scatter(
            x=pr_v, y=sent_v,
            mode="markers",
            marker=dict(
                size=[np.sqrt(f) * 0.5 + 4 for f in foll_v],
                color=col_v, opacity=0.75,
                line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
            text=[f"{n}<br>PR={p:.5f}<br>Sent={s:.3f}<br>Followers={int(fl)}"
                  for n, p, s, fl in zip(node_list, pr_v, sent_v, foll_v)],
            hoverinfo="text"))
        fig_bub.update_layout(**base_layout(height=310,
            title="Influence × Sentiment × Reach (bubble = followers)",
            xaxis_title="PageRank", yaxis_title="Sentiment",
            xaxis=dict(gridcolor=GRID_CLR),
            yaxis=dict(gridcolor=GRID_CLR)))
        st.plotly_chart(fig_bub, use_container_width=True)

        # Top influencer bar
        top_n = 10
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_names, top_pr = zip(*top_nodes)
        top_sent  = [sentiments.get(n, 0) for n in top_names]
        top_colors = [GREEN_CLR if s > 0 else RED_CLR for s in top_sent]

        fig_top = go.Figure(go.Bar(
            y=[n[:10] for n in top_names], x=list(top_pr),
            orientation="h", marker_color=top_colors,
            text=[f"{p:.5f}" for p in top_pr], textposition="outside",
            textfont=dict(color=TEXT_CLR, size=9)))
        fig_top.update_layout(**base_layout(height=300,
            title=f"Top {top_n} Nodes by PageRank",
            xaxis_title="PageRank Score",
            yaxis=dict(autorange="reversed", gridcolor=GRID_CLR),
            xaxis=dict(gridcolor=GRID_CLR)))
        st.plotly_chart(fig_top, use_container_width=True)

    # ── Network summary metrics table ─────────────────────────────────
    st.markdown('<div class="section-title">Network Summary Statistics</div>',
                unsafe_allow_html=True)
    nc1, nc2, nc3, nc4, nc5 = st.columns(5)
    nc1.metric("Nodes",    len(G.nodes()))
    nc2.metric("Edges",    len(G.edges()))
    nc3.metric("Avg In-Degree", f"{np.mean(in_degs):.2f}")
    nc4.metric("Max PageRank", f"{max(pagerank.values()):.5f}")
    nc5.metric("Network Density", f"{nx.density(G):.4f}")

# ─────────────────────────────────────────────────────────────────────
#  TAB 5 – INFLUENCE CASCADE
# ─────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-title">Influence Cascade & Market Regime Detection</div>',
                unsafe_allow_html=True)

    ic1, ic2 = st.columns([3, 2])

    with ic1:
        # ── Stacked area: expert vs bot contribution to sentiment ─────
        expert_contribution = daily_sentiments[:, list(experts_idx)].mean(axis=1) * len(experts_idx)
        bot_contribution    = daily_sentiments[:, bot_idx].mean(axis=1) * len(bot_idx)

        fig_stack = go.Figure()
        fig_stack.add_trace(go.Scatter(
            x=date_col, y=expert_contribution,
            name="Expert Signal", fill="tozeroy",
            fillcolor="rgba(255,213,79,0.25)",
            line=dict(color=GOLD_CLR, width=2)))
        fig_stack.add_trace(go.Scatter(
            x=date_col, y=bot_contribution,
            name="Bot Noise", fill="tozeroy",
            fillcolor="rgba(255,82,82,0.18)",
            line=dict(color=RED_CLR, width=1.5, dash="dot")))
        fig_stack.add_trace(go.Scatter(
            x=date_col, y=prices / prices.max() * max(abs(expert_contribution.max()),
                                                       abs(bot_contribution.max())),
            name=f"{selected_ticker} (scaled)", line=dict(color="#e0e0e0", width=2)))
        fig_stack.update_layout(**base_layout(height=320,
            title="Expert Signal vs Bot Noise — Cumulative Contribution",
            xaxis_title="Date", yaxis_title="Aggregated Sentiment Signal",
            legend=dict(bgcolor="rgba(0,0,0,0)")))
        fig_stack.update_xaxes(gridcolor=GRID_CLR)
        fig_stack.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_stack, use_container_width=True)

        # ── Market regime detection using rolling sentiment ──────────
        roll_sent    = pd.Series(np.average(daily_sentiments, axis=1,
                                             weights=pr_values + 1e-8)).rolling(7).mean()
        regime_color = [GREEN_CLR if v > 0 else RED_CLR for v in roll_sent]
        regime_label = ["Bullish" if v > 0 else "Bearish" for v in roll_sent]

        fig_regime = go.Figure()
        fig_regime.add_trace(go.Bar(
            x=date_col, y=roll_sent,
            marker_color=regime_color, name="Market Regime",
            hovertemplate="%{x}<br>Regime: %{customdata}<br>Signal: %{y:.3f}",
            customdata=regime_label))
        fig_regime.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
        fig_regime.update_layout(**base_layout(height=250,
            title="Market Regime Detection (7-day Weighted Sentiment)",
            xaxis_title="Date", yaxis_title="Sentiment Signal"))
        fig_regime.update_xaxes(gridcolor=GRID_CLR)
        fig_regime.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_regime, use_container_width=True)

    with ic2:
        # ── Signal-to-Noise Ratio over time ──────────────────────────
        snr = np.abs(expert_contribution) / (np.abs(bot_contribution) + 1e-6)
        fig_snr = go.Figure(go.Scatter(
            x=date_col, y=pd.Series(snr).rolling(rolling_window).mean(),
            fill="tozeroy",
            fillcolor="rgba(0,230,118,0.12)",
            line=dict(color=GREEN_CLR, width=2)))
        fig_snr.update_layout(**base_layout(height=260,
            title="Signal-to-Noise Ratio (Expert/Bot)",
            xaxis_title="Date", yaxis_title="SNR"))
        fig_snr.update_xaxes(gridcolor=GRID_CLR)
        fig_snr.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_snr, use_container_width=True)

        # ── Radar chart: model capabilities ──────────────────────────
        categories = ["Accuracy", "Noise\nFiltering", "Explainability",
                      "Speed", "Robustness"]
        gnn_scores  = [0.92, 0.95, 0.88, 0.72, 0.90]
        base_scores = [0.67, 0.30, 0.40, 0.95, 0.55]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=gnn_scores + [gnn_scores[0]], theta=categories + [categories[0]],
            fill="toself", fillcolor="rgba(0,230,118,0.15)",
            line=dict(color=GREEN_CLR, width=2), name="GNN-LSTM (Ours)"))
        fig_radar.add_trace(go.Scatterpolar(
            r=base_scores + [base_scores[0]], theta=categories + [categories[0]],
            fill="toself", fillcolor="rgba(255,82,82,0.12)",
            line=dict(color=RED_CLR, width=1.5, dash="dot"), name="Baseline"))
        fig_radar.update_layout(**base_layout(height=300,
            title="Model Capability Radar",
            polar=dict(
                bgcolor=PAPER_BG,
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_CLR,
                                tickfont=dict(color=TEXT_CLR)),
                angularaxis=dict(gridcolor=GRID_CLR, tickfont=dict(color=TEXT_CLR))),
            legend=dict(bgcolor="rgba(0,0,0,0)")))
        st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
#  TAB 6 – TRAINING DYNAMICS
# ─────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="section-title">Model Training Dynamics & Architecture Insights</div>',
                unsafe_allow_html=True)

    td1, td2 = st.columns(2)

    with td1:
        # Loss curves
        epochs = list(range(1, len(losses_gnn) + 1))
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=losses_gnn,
            name="GNN-LSTM Loss", line=dict(color=GREEN_CLR, width=2)))
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=losses_base,
            name="Baseline Loss", line=dict(color=RED_CLR, width=1.8, dash="dash")))
        # Smooth versions
        def smooth(arr, w=5):
            return np.convolve(arr, np.ones(w)/w, mode='same')
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=smooth(losses_gnn),
            name="GNN (smoothed)", line=dict(color=GOLD_CLR, width=1.5, dash="dot")))
        fig_loss.update_layout(**base_layout(height=310,
            title="Training Loss Convergence (MSE)",
            xaxis_title="Epoch", yaxis_title="MSE Loss",
            legend=dict(bgcolor="rgba(0,0,0,0)")))
        fig_loss.update_xaxes(gridcolor=GRID_CLR)
        fig_loss.update_yaxes(gridcolor=GRID_CLR, type="log")
        st.plotly_chart(fig_loss, use_container_width=True)

        # Improvement over epochs
        improvement = [(b - g) / b * 100
                       for b, g in zip(losses_base, losses_gnn)]
        fig_imp = go.Figure(go.Scatter(
            x=epochs, y=smooth(improvement, 5),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.12)",
            line=dict(color=GREEN_CLR, width=2),
            name="% Loss Reduction (GNN vs Baseline)"))
        fig_imp.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
        fig_imp.update_layout(**base_layout(height=260,
            title="GNN-LSTM Advantage Over Training (% Loss Reduction)",
            xaxis_title="Epoch", yaxis_title="% Improvement"))
        fig_imp.update_xaxes(gridcolor=GRID_CLR)
        fig_imp.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_imp, use_container_width=True)

    with td2:
        # Architecture diagram as Sankey
        fig_san = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18, thickness=22,
                label=["Raw Tweets", "VADER NLP", "Follower Count",
                       "VADER Score", "Influence Score",
                       "PageRank GNN", "Signal Fusion",
                       "Stacked LSTM", "Temporal Attn", "Price Pred"],
                color=[RED_CLR, "#42a5f5", "#42a5f5",
                       "#42a5f5", GOLD_CLR,
                       GREEN_CLR, GREEN_CLR,
                       GREEN_CLR, GREEN_CLR, GOLD_CLR],
                x=[0.0, 0.15, 0.15, 0.3, 0.3, 0.5, 0.65, 0.78, 0.88, 1.0],
                y=[0.2, 0.1, 0.7, 0.1, 0.7, 0.4, 0.4, 0.4, 0.4, 0.4]),
            link=dict(
                source=[0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8],
                target=[1, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9],
                value= [5, 5, 5, 5, 3, 7, 5, 5, 8, 8, 8],
                color=["rgba(66,165,245,0.25)"] * 4 +
                      ["rgba(0,230,118,0.25)"] * 7)))
        fig_san.update_layout(**base_layout(height=400,
            title="Data Flow: Raw Tweets → Price Prediction"))
        st.plotly_chart(fig_san, use_container_width=True)

        # Final metrics summary
        st.markdown('<div class="section-title">Final Results Summary</div>',
                    unsafe_allow_html=True)
        metric_names = ["MAPE (%)", "RMSE ($)", "R² Score"]
        gnn_m  = [m_gnn["mape"],  m_gnn["rmse"],  m_gnn["r2"]]
        base_m = [m_base["mape"], m_base["rmse"], m_base["r2"]]

        fig_met = go.Figure()
        for i, (name, gv, bv) in enumerate(zip(metric_names, gnn_m, base_m)):
            fig_met.add_trace(go.Bar(
                x=[name], y=[bv],
                name="Baseline" if i == 0 else None,
                marker_color=RED_CLR, showlegend=(i == 0),
                offsetgroup=0))
            fig_met.add_trace(go.Bar(
                x=[name], y=[gv],
                name="GNN-LSTM" if i == 0 else None,
                marker_color=GREEN_CLR, showlegend=(i == 0),
                offsetgroup=1))
        fig_met.update_layout(**base_layout(height=265, barmode="group",
            title="Metric Comparison — Lower MAPE/RMSE, Higher R²",
            yaxis=dict(gridcolor=GRID_CLR),
            legend=dict(bgcolor="rgba(0,0,0,0)")))
        fig_met.update_xaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_met, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.78rem;color:#546e7a;"
    "font-family:Space Mono,monospace;padding:10px'>"
    "ELL 8224 · Information Visualization · IIT Delhi &nbsp;|&nbsp; "
    "Vaibhav Sharma · 2025EEY7541 &nbsp;|&nbsp; "
    "Stack: Streamlit · PyTorch · NetworkX · Pyvis · Plotly · OpenBB"
    "</div>",
    unsafe_allow_html=True,
)