import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Page setup + CYBERPUNK NEON THEME
# -----------------------------
st.set_page_config(page_title="Client Risk Command Center", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Electrolize&family=Share+Tech+Mono&display=swap');
    
    /* ==== GLOBAL CYBERPUNK BACKGROUND ==== */
    .stApp {
        background: linear-gradient(135deg, #0a0014 0%, #1a0033 25%, #0d001a 50%, #1a0033 75%, #0a0014 100%);
        background-attachment: fixed;
        animation: bgPulse 8s ease-in-out infinite;
    }
    
    @keyframes bgPulse {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* ==== MAIN CONTENT AREA ==== */
    .main {
        background: rgba(10, 0, 20, 0.6);
        backdrop-filter: blur(12px);
    }
    
    /* ==== NEON TYPOGRAPHY ==== */
    h1, h2, h3, .big-title {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 3px;
        text-transform: uppercase;
        background: linear-gradient(135deg, #ff00ff 0%, #00ffff 50%, #ff00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255, 0, 255, 0.6), 0 0 60px rgba(0, 255, 255, 0.4);
        font-weight: 900 !important;
        animation: titleGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { 
            filter: drop-shadow(0 0 10px rgba(255, 0, 255, 0.6)) 
                    drop-shadow(0 0 20px rgba(0, 255, 255, 0.4)); 
        }
        to { 
            filter: drop-shadow(0 0 20px rgba(255, 0, 255, 0.9)) 
                    drop-shadow(0 0 40px rgba(0, 255, 255, 0.7)); 
        }
    }
    
    .big-title {
        font-size: 3.5rem !important;
        margin-bottom: 0px;
        text-align: center;
    }
    
    .subtle {
        font-family: 'Electrolize', sans-serif !important;
        color: #00ffff !important;
        font-size: 18px !important;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-top: 10px;
        text-align: center;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.8);
        animation: subtitlePulse 2s ease-in-out infinite;
    }
    
    @keyframes subtitlePulse {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }
    
    /* ==== BODY TEXT ==== */
    body, .stMarkdown, p, div, span, label {
        font-family: 'Rajdhani', sans-serif !important;
        color: #e0e6ff !important;
        font-size: 17px !important;
        letter-spacing: 0.8px;
    }
    
    /* ==== NEON CARDS ==== */
    .card {
        padding: 20px 24px;
        border-radius: 16px;
        border: 2px solid;
        border-image: linear-gradient(135deg, #ff00ff, #00ffff, #ff00ff) 1;
        background: linear-gradient(135deg, rgba(255, 0, 255, 0.08) 0%, rgba(0, 255, 255, 0.08) 100%);
        box-shadow: 
            0 0 20px rgba(255, 0, 255, 0.3),
            0 0 40px rgba(0, 255, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ff00ff, #00ffff, #ff00ff);
        border-radius: 16px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .card:hover::before {
        opacity: 0.3;
        animation: borderRotate 2s linear infinite;
    }
    
    @keyframes borderRotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 
            0 0 30px rgba(255, 0, 255, 0.5),
            0 0 60px rgba(0, 255, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    /* ==== METRIC CARDS ==== */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #ff00ff 0%, #00ffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(255, 0, 255, 0.8);
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Share Tech Mono', monospace !important;
        color: #00ffff !important;
        font-size: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.6);
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 0, 255, 0.1) 0%, rgba(0, 255, 255, 0.1) 100%);
        border: 2px solid;
        border-image: linear-gradient(135deg, #ff00ff, #00ffff) 1;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 
            0 0 15px rgba(255, 0, 255, 0.3),
            0 0 30px rgba(0, 255, 255, 0.2),
            inset 0 2px 4px rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 0 25px rgba(255, 0, 255, 0.5),
            0 0 50px rgba(0, 255, 255, 0.4),
            inset 0 2px 4px rgba(255, 255, 255, 0.15);
    }
    
    /* ==== SIDEBAR ==== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0014 0%, #1a0033 50%, #0a0014 100%) !important;
        border-right: 3px solid;
        border-image: linear-gradient(180deg, #ff00ff, #00ffff, #ff00ff) 1;
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.3);
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ff00ff !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 15px rgba(255, 0, 255, 0.8);
    }
    
    /* ==== RADIO BUTTONS (Navigation) ==== */
    .stRadio > label {
        font-family: 'Electrolize', sans-serif !important;
        color: #00ffff !important;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.6);
    }
    
    .stRadio > div {
        background: rgba(255, 0, 255, 0.05);
        border-radius: 12px;
        padding: 10px;
    }
    
    .stRadio > div > label {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #e0e6ff !important;
        padding: 12px 16px;
        margin: 4px 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stRadio > div > label:hover {
        background: rgba(0, 255, 255, 0.15);
        border-color: #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
    }
    
    /* ==== BUTTONS ==== */
    .stButton > button {
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700;
        font-size: 18px;
        letter-spacing: 2px;
        text-transform: uppercase;
        background: linear-gradient(135deg, #ff00ff 0%, #00ffff 100%);
        color: #000 !important;
        border: none;
        border-radius: 12px;
        padding: 16px 40px;
        box-shadow: 
            0 0 20px rgba(255, 0, 255, 0.5),
            0 0 40px rgba(0, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ffff 0%, #ff00ff 100%);
        box-shadow: 
            0 0 30px rgba(255, 0, 255, 0.7),
            0 0 60px rgba(0, 255, 255, 0.5);
        transform: translateY(-3px) scale(1.05);
    }
    
    /* ==== DOWNLOAD BUTTON ==== */
    .stDownloadButton > button {
        font-family: 'Share Tech Mono', monospace !important;
        font-weight: 600;
        background: rgba(0, 255, 255, 0.2);
        color: #00ffff !important;
        border: 2px solid #00ffff;
        border-radius: 10px;
        letter-spacing: 1.5px;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
    }
    
    .stDownloadButton > button:hover {
        background: rgba(0, 255, 255, 0.3);
        border-color: #ff00ff;
        box-shadow: 0 0 25px rgba(255, 0, 255, 0.6);
    }
    
    /* ==== INPUT FIELDS ==== */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        font-family: 'Rajdhani', sans-serif !important;
        background: rgba(10, 0, 20, 0.8) !important;
        border: 2px solid rgba(0, 255, 255, 0.4) !important;
        border-radius: 10px;
        color: #e0e6ff !important;
        box-shadow: inset 0 0 10px rgba(0, 255, 255, 0.2);
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {
        border-color: #ff00ff !important;
        box-shadow: 
            0 0 20px rgba(255, 0, 255, 0.5),
            inset 0 0 15px rgba(255, 0, 255, 0.2);
    }
    
    /* ==== SLIDER ==== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
    }
    
    .stSlider > div > div > div > div {
        background: #ff00ff;
        border: 2px solid #00ffff;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.8);
    }
    
    /* ==== DATAFRAME ==== */
    .stDataFrame {
        border: 2px solid;
        border-image: linear-gradient(135deg, #ff00ff, #00ffff, #ff00ff) 1;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
    }
    
    [data-testid="stDataFrame"] {
        background: rgba(10, 0, 20, 0.7);
    }
    
    /* ==== DIVIDER ==== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #ff00ff, #00ffff, #ff00ff, transparent);
        margin: 2.5rem 0 !important;
        box-shadow: 0 0 10px rgba(255, 0, 255, 0.5);
    }
    
    /* ==== SUCCESS/INFO/WARNING ==== */
    .stSuccess {
        background: rgba(0, 255, 136, 0.15);
        border: 2px solid #00ff88;
        border-radius: 12px;
        color: #00ff88 !important;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.4);
    }
    
    .stInfo {
        background: rgba(0, 255, 255, 0.15);
        border: 2px solid #00ffff;
        border-radius: 12px;
        color: #00ffff !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
    }
    
    .stWarning {
        background: rgba(255, 195, 0, 0.15);
        border: 2px solid #ffc300;
        border-radius: 12px;
        color: #ffc300 !important;
        box-shadow: 0 0 15px rgba(255, 195, 0, 0.4);
    }
    
    /* ==== PROGRESS BAR ==== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.6);
    }
    
    /* ==== CAPTIONS ==== */
    .caption {
        font-family: 'Share Tech Mono', monospace !important;
        color: #00ffff !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.5);
    }
    
    /* ==== SUBHEADERS GLOW ==== */
    .stMarkdown h3::before {
        content: '▸▸ ';
        color: #ff00ff;
        text-shadow: 0 0 15px rgba(255, 0, 255, 0.8);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">⚡ CLIENT RISK COMMAND CENTER ⚡</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">B2B RISK SCORING • CHURN PREDICTION • RETENTION ACTIONS • RESPONSIBLE AI</div>', unsafe_allow_html=True)

# -----------------------------
# Configure matplotlib for NEON CYBERPUNK theme
# -----------------------------
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0a0014'
plt.rcParams['axes.facecolor'] = '#1a0033'
plt.rcParams['axes.edgecolor'] = '#ff00ff'
plt.rcParams['axes.labelcolor'] = '#00ffff'
plt.rcParams['text.color'] = '#e0e6ff'
plt.rcParams['xtick.color'] = '#00ffff'
plt.rcParams['ytick.color'] = '#00ffff'
plt.rcParams['grid.color'] = '#ff00ff'
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 10

# Neon color palette
NEON_PINK = '#ff00ff'
NEON_CYAN = '#00ffff'
NEON_GREEN = '#00ff88'
NEON_YELLOW = '#ffff00'
NEON_ORANGE = '#ff6600'
NEON_PURPLE = '#cc00ff'

RISK_COLORS = {
    'Low Risk': NEON_GREEN,
    'Medium Risk': NEON_YELLOW,
    'High Risk': NEON_PINK
}

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("B2B_Client_Churn_5000.csv")

df = load_data()

# -----------------------------
# Column checks (prevents crash)
# -----------------------------
required_cols = [
    "Client_ID", "Industry", "Region",
    "Monthly_Usage_Score", "Payment_Delay_Days",
    "Contract_Length_Months", "Support_Tickets_Last30Days",
    "Monthly_Revenue_USD", "Renewal_Status"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"CSV columns missing: {missing}")
    st.write("Columns found:", list(df.columns))
    st.stop()

# Optional columns for better UI
for col, default in [("Company_Name", "NA"), ("Plan", "NA"), ("Lead_Source", "NA"), ("Account_Age_Months", 0)]:
    if col not in df.columns:
        df[col] = default

# Churn flag
df["Churned"] = df["Renewal_Status"].map({"Yes": 0, "No": 1}).fillna(0).astype(int)

# -----------------------------
# Part B: Risk scoring (Quantile-based)
# -----------------------------
delay_q = df["Payment_Delay_Days"].quantile([0.50, 0.80]).values
usage_q = df["Monthly_Usage_Score"].quantile([0.20, 0.50]).values
contract_q = df["Contract_Length_Months"].quantile([0.20, 0.50]).values
tickets_q = df["Support_Tickets_Last30Days"].quantile([0.50, 0.80]).values

def risk_points(row):
    pts = 0
    if row["Payment_Delay_Days"] >= delay_q[1]:
        pts += 3
    elif row["Payment_Delay_Days"] >= delay_q[0]:
        pts += 2
    elif row["Payment_Delay_Days"] > 0:
        pts += 1

    if row["Monthly_Usage_Score"] <= usage_q[0]:
        pts += 3
    elif row["Monthly_Usage_Score"] <= usage_q[1]:
        pts += 2
    elif row["Monthly_Usage_Score"] <= df["Monthly_Usage_Score"].quantile(0.70):
        pts += 1

    if row["Contract_Length_Months"] <= contract_q[0]:
        pts += 2
    elif row["Contract_Length_Months"] <= contract_q[1]:
        pts += 1

    if row["Support_Tickets_Last30Days"] >= tickets_q[1]:
        pts += 2
    elif row["Support_Tickets_Last30Days"] >= tickets_q[0]:
        pts += 1

    return pts

df["Risk_Score"] = df.apply(risk_points, axis=1)

def risk_bucket(x):
    if x >= 7:
        return "High Risk"
    elif x >= 4:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Category"] = df["Risk_Score"].apply(risk_bucket)

# -----------------------------
# Sidebar: navigation + filters
# -----------------------------
st.sidebar.header("⚡ NAVIGATION")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📌 Segmentation", "🤖 Model Lab", "🛠 Action Center", "⚖️ Responsible AI", "📄 Data Export"],
    index=0
)

st.sidebar.divider()
st.sidebar.header("🔍 FILTERS")

all_regions = sorted(df["Region"].dropna().unique())
all_industries = sorted(df["Industry"].dropna().unique())
risk_levels = ["Low Risk", "Medium Risk", "High Risk"]

sel_region = st.sidebar.multiselect("Region", all_regions, default=all_regions)
sel_industry = st.sidebar.multiselect("Industry", all_industries, default=all_industries)
sel_risk = st.sidebar.multiselect("Risk Category", risk_levels, default=risk_levels)

rev_min, rev_max = float(df["Monthly_Revenue_USD"].min()), float(df["Monthly_Revenue_USD"].max())
sel_rev = st.sidebar.slider("Revenue Range (USD)", rev_min, rev_max, (rev_min, rev_max))

f = df[
    df["Region"].isin(sel_region) &
    df["Industry"].isin(sel_industry) &
    df["Risk_Category"].isin(sel_risk) &
    (df["Monthly_Revenue_USD"] >= sel_rev[0]) &
    (df["Monthly_Revenue_USD"] <= sel_rev[1])
].copy()

# -----------------------------
# KPI helper
# -----------------------------
def kpi_row(data):
    total = len(data)
    high = int((data["Risk_Category"] == "High Risk").sum())
    churn_pct = (data["Churned"].mean() * 100) if total else 0
    avg_rev = data["Monthly_Revenue_USD"].mean() if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Clients", f"{total}")
    c2.metric("High Risk Clients", f"{high}")
    c3.metric("Churn Rate %", f"{churn_pct:.2f}%")
    c4.metric("Avg Revenue / Client", f"${avg_rev:,.2f}")

# =============================
# PAGE 1: OVERVIEW (ENHANCED NEON VISUALS)
# =============================
if page == "🏠 Overview":
    st.subheader("EXECUTIVE SNAPSHOT")
    kpi_row(f)

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 🔥 Risk Category Distribution (Neon Radar)")
        counts = f["Risk_Category"].value_counts().reindex(risk_levels).fillna(0)

        # POLAR/RADAR STYLE BAR CHART (more cyberpunk)
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
        theta = np.linspace(0, 2 * np.pi, len(risk_levels), endpoint=False)
        width = 2 * np.pi / len(risk_levels)
        
        colors = [RISK_COLORS[level] for level in risk_levels]
        bars = ax.bar(theta, counts.values, width=width, bottom=0, color=colors, 
                      edgecolor=NEON_CYAN, linewidth=3, alpha=0.8)
        
        # Add glow effect
        for bar, color in zip(bars, colors):
            bar.set_alpha(0.7)
            
        ax.set_xticks(theta)
        ax.set_xticklabels(risk_levels, color=NEON_CYAN, fontweight='bold', fontsize=11)
        ax.set_ylim(0, counts.max() * 1.2)
        ax.grid(color=NEON_PINK, alpha=0.3, linestyle='--')
        ax.set_facecolor('#0a0014')
        fig.patch.set_facecolor('#0a0014')
        st.pyplot(fig)
        plt.close()
        st.caption("SHOWS OVERALL CLIENT RISK MIX IN RADAR VIEW")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 💎 Revenue vs Risk (Neon Scatter)")
        
        # ENHANCED SCATTER with gradient colors and glow
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        # Create size based on revenue
        sizes = (f["Monthly_Revenue_USD"] - f["Monthly_Revenue_USD"].min() + 100) / 30
        
        # Color by risk category
        colors_map = f["Risk_Category"].map(RISK_COLORS)
        
        scatter = ax2.scatter(
            f["Risk_Score"], 
            f["Monthly_Revenue_USD"], 
            s=sizes,
            c=colors_map,
            alpha=0.6,
            edgecolors=NEON_CYAN,
            linewidths=1.5
        )
        
        ax2.set_xlabel("Risk Score (0–10)", color=NEON_CYAN, fontweight='bold', fontsize=12)
        ax2.set_ylabel("Monthly Revenue (USD)", color=NEON_CYAN, fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.2, color=NEON_PINK, linestyle='--')
        ax2.set_facecolor('#1a0033')
        fig2.patch.set_facecolor('#0a0014')
        
        # Add border glow
        for spine in ax2.spines.values():
            spine.set_edgecolor(NEON_PINK)
            spine.set_linewidth(2)
        
        st.pyplot(fig2)
        plt.close()
        st.caption("FIND HIGH-REVENUE CLIENTS WITH HIGH RISK TO PRIORITIZE RETENTION")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### ⚡ Contract Length vs Churn (Neon Line Graph)")
    if len(f) > 0:
        tmp = f.copy()
        tmp["Contract_Bin"] = pd.cut(tmp["Contract_Length_Months"], bins=6)
        churn_by_contract = tmp.groupby("Contract_Bin")["Churned"].mean() * 100

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        
        # Neon line with markers and glow
        x_pos = range(len(churn_by_contract))
        ax3.plot(x_pos, churn_by_contract.values, 
                marker='o', markersize=12, linewidth=3,
                color=NEON_PINK, markerfacecolor=NEON_CYAN, 
                markeredgecolor=NEON_PINK, markeredgewidth=2)
        
        # Add fill under curve for glow effect
        ax3.fill_between(x_pos, churn_by_contract.values, alpha=0.3, color=NEON_PINK)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([str(x) for x in churn_by_contract.index], 
                           rotation=45, ha="right", color=NEON_CYAN, fontweight='bold')
        ax3.set_ylabel("Churn %", color=NEON_CYAN, fontweight='bold', fontsize=12)
        ax3.set_xlabel("Contract Length Bin", color=NEON_CYAN, fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.2, color=NEON_PINK, linestyle='--')
        ax3.set_facecolor('#1a0033')
        fig3.patch.set_facecolor('#0a0014')
        
        for spine in ax3.spines.values():
            spine.set_edgecolor(NEON_CYAN)
            spine.set_linewidth(2)
        
        st.pyplot(fig3)
        plt.close()
    else:
        st.info("No data after filters.")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# PAGE 2: SEGMENTATION (ENHANCED VISUALS)
# =============================
elif page == "📌 Segmentation":
    st.subheader("SEGMENTATION & PRIORITIZATION")
    kpi_row(f)

    st.divider()
    colA, colB = st.columns([1.1, 0.9])

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 🎯 Industry-wise Risk Analysis (Neon Stack)")
        pivot = pd.pivot_table(
            f, index="Industry", columns="Risk_Category",
            values="Client_ID", aggfunc="count", fill_value=0
        ).reindex(columns=risk_levels, fill_value=0)

        # HORIZONTAL STACKED BAR with neon colors
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(pivot.index))
        left = np.zeros(len(pivot.index))
        
        for cat in risk_levels:
            color = RISK_COLORS[cat]
            ax.barh(y_pos, pivot[cat].values, left=left, 
                   label=cat, color=color, alpha=0.8,
                   edgecolor=NEON_CYAN, linewidth=2)
            left += pivot[cat].values
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pivot.index, color=NEON_CYAN, fontweight='bold')
        ax.set_xlabel("Clients", color=NEON_CYAN, fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', facecolor='#1a0033', edgecolor=NEON_PINK, 
                 labelcolor=NEON_CYAN, framealpha=0.9)
        ax.grid(True, alpha=0.2, color=NEON_PINK, linestyle='--', axis='x')
        ax.set_facecolor('#1a0033')
        fig.patch.set_facecolor('#0a0014')
        
        for spine in ax.spines.values():
            spine.set_edgecolor(NEON_PINK)
            spine.set_linewidth(2)
        
        st.pyplot(fig)
        plt.close()
        st.caption("HIGHLIGHTS WHICH INDUSTRIES HAVE MORE HIGH RISK CLIENTS")
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 📊 Risk Score Distribution (Neon Histogram)")
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        # Gradient histogram
        n, bins, patches = ax2.hist(f["Risk_Score"], bins=11, 
                                    edgecolor=NEON_CYAN, linewidth=2, alpha=0.7)
        
        # Color gradient from green to pink
        for i, patch in enumerate(patches):
            ratio = i / len(patches)
            r = int(ratio * 255)
            g = int((1 - ratio) * 255)
            patch.set_facecolor(f'#{r:02x}{g:02x}ff')
        
        ax2.set_xlabel("Risk Score (0–10)", color=NEON_CYAN, fontweight='bold', fontsize=12)
        ax2.set_ylabel("Count", color=NEON_CYAN, fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.2, color=NEON_PINK, linestyle='--')
        ax2.set_facecolor('#1a0033')
        fig2.patch.set_facecolor('#0a0014')
        
        for spine in ax2.spines.values():
            spine.set_edgecolor(NEON_PINK)
            spine.set_linewidth(2)
        
        st.pyplot(fig2)
        plt.close()
        st.caption("SHOWS HOW RISK IS SPREAD ACROSS THE FILTERED POPULATION")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### 🎯 Top 20 High-Risk Clients (Action List)")
    top20 = f.sort_values(["Risk_Score", "Monthly_Revenue_USD"], ascending=[False, False]).head(20)

    cols_show = [
        "Client_ID", "Company_Name", "Industry", "Region", "Plan",
        "Monthly_Usage_Score", "Payment_Delay_Days", "Contract_Length_Months",
        "Support_Tickets_Last30Days", "Monthly_Revenue_USD", "Risk_Score",
        "Risk_Category", "Renewal_Status"
    ]
    cols_show = [c for c in cols_show if c in top20.columns]

    def highlight_risk(s):
        return ["font-weight:700;" if v >= 7 else "" for v in s]

    st.dataframe(top20[cols_show].style.apply(highlight_risk, subset=["Risk_Score"]), use_container_width=True)
    st.caption("USE THIS AS THE IMMEDIATE CALL LIST FOR RETENTION OUTREACH")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔎 CLIENT DRILL-DOWN")
    pick = st.selectbox("Select a client", sorted(f["Client_ID"].astype(str).unique()) if len(f) else [])
    if pick:
        row = f[f["Client_ID"].astype(str) == str(pick)].head(1)
        if len(row):
            r = row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Risk Score", f"{r['Risk_Score']:.0f}/10")
            m2.metric("Risk Category", r["Risk_Category"])
            m3.metric("Revenue", f"${r['Monthly_Revenue_USD']:,.0f}")
            m4.metric("Renewal Status", r["Renewal_Status"])
            st.dataframe(row[cols_show], use_container_width=True)

# =============================
# PAGE 3: MODEL LAB (ENHANCED VISUALS)
# =============================
elif page == "🤖 Model Lab":
    st.subheader("CHURN PREDICTION (DECISION TREE)")
    st.caption("TARGET: RENEWAL_STATUS (YES/NO). SHOWS ACCURACY, CONFUSION MATRIX, AND FEATURE IMPORTANCE.")

    feature_cols = [
        "Industry", "Region", "Plan", "Lead_Source",
        "Account_Age_Months", "Contract_Length_Months",
        "Monthly_Usage_Score", "Support_Tickets_Last30Days",
        "Payment_Delay_Days", "Monthly_Revenue_USD", "Risk_Score"
    ]
    X = df[feature_cols].copy()
    y = df["Renewal_Status"].map({"Yes": 1, "No": 0})

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    depth = st.slider("Tree Depth (controls complexity)", 2, 14, 6)
    min_leaf = st.slider("Min Samples per Leaf", 1, 50, 10)

    model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    a1, a2, a3 = st.columns(3)
    a1.metric("Accuracy", f"{acc:.4f}")
    a2.metric("Test Size", f"{len(X_test)}")
    a3.metric("Features Used", f"{X.shape[1]}")

    st.write("**CONFUSION MATRIX** (rows = actual, columns = predicted)")
    cm = confusion_matrix(y_test, pred)
    
    # Visualize confusion matrix with neon colors
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, cmap='plasma', alpha=0.8)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Churn', 'Renew'], color=NEON_CYAN, fontweight='bold')
    ax_cm.set_yticklabels(['Churn', 'Renew'], color=NEON_CYAN, fontweight='bold')
    ax_cm.set_xlabel('Predicted', color=NEON_CYAN, fontweight='bold')
    ax_cm.set_ylabel('Actual', color=NEON_CYAN, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax_cm.text(j, i, cm[i, j], ha="center", va="center", 
                            color=NEON_CYAN, fontsize=20, fontweight='bold')
    
    ax_cm.set_facecolor('#1a0033')
    fig_cm.patch.set_facecolor('#0a0014')
    st.pyplot(fig_cm)
    plt.close()

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
    st.write("**TOP 12 FEATURE IMPORTANCES (Neon Bars)**")
    
    # Enhanced horizontal bar chart for feature importance
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    colors_gradient = [NEON_PINK if i < 4 else NEON_CYAN if i < 8 else NEON_GREEN 
                      for i in range(len(imp))]
    
    bars = ax_imp.barh(range(len(imp)), imp.values, color=colors_gradient, 
                      edgecolor=NEON_CYAN, linewidth=2, alpha=0.8)
    ax_imp.set_yticks(range(len(imp)))
    ax_imp.set_yticklabels(imp.index, color=NEON_CYAN, fontweight='bold', fontsize=10)
    ax_imp.set_xlabel('Importance', color=NEON_CYAN, fontweight='bold', fontsize=12)
    ax_imp.grid(True, alpha=0.2, color=NEON_PINK, linestyle='--', axis='x')
    ax_imp.set_facecolor('#1a0033')
    fig_imp.patch.set_facecolor('#0a0014')
    
    for spine in ax_imp.spines.values():
        spine.set_edgecolor(NEON_PINK)
        spine.set_linewidth(2)
    
    st.pyplot(fig_imp)
    plt.close()
    
    st.caption("INTERPRETATION: HIGHER IMPORTANCE MEANS STRONGER INFLUENCE ON CHURN PREDICTION.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎲 PREDICTED CHURN PROBABILITY FOR A CLIENT (DEMO)")
    client_pick = st.selectbox("Pick a client to score", sorted(df["Client_ID"].astype(str).unique()))
    if client_pick:
        row = df[df["Client_ID"].astype(str) == str(client_pick)].copy()
        rowX = pd.get_dummies(row[feature_cols], drop_first=True)
        rowX = rowX.reindex(columns=X.columns, fill_value=0)

        classes = model.classes_
        churn_index = list(classes).index(0)
        churn_proba = model.predict_proba(rowX)[0][churn_index]

        st.write(f"Client **{client_pick}** estimated churn probability:")
        st.progress(float(churn_proba))
        st.write(f"**{churn_proba*100:.2f}%** (higher means more likely to churn)")

# =============================
# PAGE 4: ACTION CENTER
# =============================
elif page == "🛠 Action Center":
    st.subheader("RETENTION STRATEGY GENERATOR")
    st.caption("CLICK THE BUTTON TO GENERATE 3–5 ACTIONS BASED ON THE SELECTED (FILTERED) POPULATION.")

    if len(f) > 0:
        avg_delay = f["Payment_Delay_Days"].mean()
        avg_usage = f["Monthly_Usage_Score"].mean()
        avg_tickets = f["Support_Tickets_Last30Days"].mean()

        x1, x2, x3 = st.columns(3)
        x1.metric("Avg Payment Delay (days)", f"{avg_delay:.1f}")
        x2.metric("Avg Usage Score", f"{avg_usage:.1f}")
        x3.metric("Avg Tickets (30d)", f"{avg_tickets:.1f}")
    else:
        st.info("No rows in current filters. Adjust filters to generate insights.")

    st.divider()
    if st.button("⚡ GENERATE RETENTION STRATEGY"):
        st.success("RECOMMENDED RETENTION ACTIONS")
        st.write("1) **Payment recovery:** For clients with payment delay > 30 days, offer a flexible plan + early-pay discount.")
        st.write("2) **Adoption boost:** For low usage clients, conduct onboarding refresh + training + weekly usage nudges.")
        st.write("3) **Support stabilization:** For high-ticket clients, assign a dedicated account manager and priority SLA.")
        st.write("4) **Renewal upgrade:** For short contracts, offer long-term incentives (discount, add-ons, feature bundles).")
        st.write("5) **Protect revenue:** For high-revenue & high-risk accounts, schedule leadership call + custom success roadmap.")

    st.divider()
    st.markdown("### 🎯 QUICK TARGET LIST (HIGH RISK + HIGH REVENUE)")
    if len(f) > 0:
        target = f[(f["Risk_Category"] == "High Risk")].sort_values("Monthly_Revenue_USD", ascending=False).head(15)
        st.dataframe(target[["Client_ID","Company_Name","Industry","Region","Monthly_Revenue_USD","Risk_Score","Renewal_Status"]], use_container_width=True)
    else:
        st.write("-")

# =============================
# PAGE 5: RESPONSIBLE AI
# =============================
elif page == "⚖️ Responsible AI":
    st.subheader("ETHICAL IMPLICATIONS OF PREDICTING CLIENT CHURN")
    st.markdown(
        """
**1) BIAS IN PREDICTIVE MODELS**  
If some industries/regions historically churn more due to external conditions, the model can learn those patterns and unfairly label them as risky.

**2) IMPACT OF LABELING CLIENTS AS "HIGH RISK"**  
A "high risk" label can change how teams treat clients. If used to reduce service or increase strictness, it may increase churn (self-fulfilling outcome).

**3) DATA PRIVACY CONCERNS**  
Usage, payment behavior, and support history are sensitive. Access should be role-based, data minimized, and stored securely.

**4) RESPONSIBLE DECISION-MAKING**  
Predictions are probabilities, not facts. They should support account managers—not replace human judgment and relationship context.

**5) TRANSPARENCY & MONITORING**  
Explain key drivers (feature importance), re-check performance over time, and audit for fairness across industries/regions.
        """
    )

# =============================
# PAGE 6: DATA EXPORT
# =============================
elif page == "📄 Data Export":
    st.subheader("DATA VIEW & EXPORT")
    kpi_row(f)

    st.divider()
    st.write("Preview (first 200 rows of filtered data):")
    st.dataframe(f.head(200), use_container_width=True)

    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=f.to_csv(index=False).encode("utf-8"),
        file_name="filtered_b2b_clients.csv",
        mime="text/csv"
    )

    st.download_button(
        "⬇️ Download Top 20 High Risk (CSV)",
        data=f.sort_values(["Risk_Score","Monthly_Revenue_USD"], ascending=[False,False]).head(20).to_csv(index=False).encode("utf-8"),
        file_name="top20_high_risk.csv",
        mime="text/csv"
    )
