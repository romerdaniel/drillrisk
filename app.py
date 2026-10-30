import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, brier_score_loss

st.set_page_config(
    page_title="DrillRisk | Stand-Ahead NPT & Financial Risk Predictor",
    page_icon="💰",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0d1117; color: #f0f6fc; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 12px; }
    div[data-testid="stMetricValue"] { color: #f85149; font-weight: 800; }
    .banner-box {
        background: linear-gradient(90deg, #161b22 0%, #2a1215 100%);
        border-left: 5px solid #f85149;
        padding: 18px 22px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header[data-testid="stHeader"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    'WOB_klbs', 'RPM', 'Torque_ftlbs', 'ROP_fthr', 'Flow_gpm', 'GR_API',
    'MSE_kpsi', 'Mech_Efficiency', 'Bit_Aggressiveness', 'ROP_WOB',
    'Torque_RPM', 'Rolling_MSE_30ft', 'Rolling_Torque_Std', 'ROP_Decay_30ft'
]


@st.cache_resource(show_spinner="Training Calibrated Stand-Ahead NPT Risk Engine on Equinor Volve F-9A...")
def load_and_train():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "volve_f9a_processed.csv")
    df = pd.read_csv(csv_path)
    eps = 1e-6
    df = df.sort_values('Depth_ft').reset_index(drop=True)
    df['GR_API'] = df['GR_API'].interpolate(method='linear').bfill().ffill()

    df['Mech_Efficiency']    = df['ROP_fthr'] / (df['WOB_klbs'] * df['RPM'] + eps)
    df['Bit_Aggressiveness'] = df['Torque_ftlbs'] / (df['WOB_klbs'] + eps)
    df['ROP_WOB']            = df['ROP_fthr'] / (df['WOB_klbs'] + eps)
    df['Torque_RPM']         = df['Torque_ftlbs'] / (df['RPM'] + eps)
    df['WOB_RPM']            = df['WOB_klbs'] * df['RPM']
    df['Rolling_MSE_30ft']   = df['MSE_kpsi'].rolling(15, min_periods=1).mean()
    df['Rolling_Torque_Std'] = df['Torque_ftlbs'].rolling(15, min_periods=1).std().fillna(0)
    df['ROP_Decay_30ft']     = (df['ROP_fthr'] - df['ROP_fthr'].rolling(15, min_periods=1).mean())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    p75_mse_roll = df['Rolling_MSE_30ft'].quantile(0.75)
    p70_tq_std   = df['Rolling_Torque_Std'].quantile(0.70)
    p25_rop_wob  = df['ROP_WOB'].quantile(0.25)
    p75_bit_agg  = df['Bit_Aggressiveness'].quantile(0.75)

    def label_npt_hazard(row):
        cond1 = (row['Rolling_MSE_30ft'] >= p75_mse_roll) and (row['ROP_WOB'] <= p25_rop_wob)
        cond2 = (row['Rolling_MSE_30ft'] >= p75_mse_roll) and (row['Bit_Aggressiveness'] >= p75_bit_agg)
        cond3 = (row['Rolling_Torque_Std'] >= p70_tq_std) and (row['MSE_kpsi'] >= p75_mse_roll)
        return 1 if (cond1 or cond2 or cond3) else 0

    raw_hazard = df.apply(label_npt_hazard, axis=1).values
    df['NPT_Risk'] = pd.Series(raw_hazard).rolling(5, min_periods=1).max().shift(-4).fillna(0).astype(int)

    X = df[FEATURES].values
    y = df['NPT_Risk'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    gbm = GradientBoostingClassifier(n_estimators=90, max_depth=3, learning_rate=0.04, subsample=0.8, random_state=42)
    gbm.fit(X_train, y_train)
    gbm_prob = gbm.predict_proba(X_test)[:, 1]
    gbm_pred = (gbm_prob >= 0.5).astype(int)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    sc = StandardScaler()
    lr = LogisticRegression(C=0.5, random_state=42)
    lr.fit(sc.fit_transform(X_train), y_train)
    lr_prob = lr.predict_proba(sc.transform(X_test))[:, 1]

    risk_prob_eval = 0.65 * gbm_prob + 0.35 * lr_prob
    df['NPT_Prob_Pct'] = (0.65 * gbm.predict_proba(X)[:, 1] + 0.35 * lr.predict_proba(sc.transform(X))[:, 1]) * 100.0

    metrics = {
        "gb_auc": roc_auc_score(y_test, gbm_prob),
        "gb_pr": average_precision_score(y_test, gbm_prob),
        "gb_f1": f1_score(y_test, gbm_pred),
        "gb_acc": accuracy_score(y_test, gbm_pred),
        "gb_brier": brier_score_loss(y_test, gbm_prob),
        "rf_auc": roc_auc_score(y_test, rf_prob),
        "lr_auc": roc_auc_score(y_test, lr_prob),
        "y_test": y_test,
        "p_eval": risk_prob_eval,
        "scale_factor": len(df) / len(y_test),
    }

    return df, sc, gbm, lr, metrics


df, sc, gbm, lr, metrics = load_and_train()

st.markdown("""
<div class="banner-box">
    <h2 style="margin:0; color:#f0f6fc;">💰 DrillRisk — Stand-Ahead NPT & Financial Risk Predictor</h2>
    <p style="margin:6px 0 0 0; color:#8b949e; font-size:15px;">
        <b>Drilling ML Progression Series (Project 3C)</b> · Equinor Volve Field (Well F-9A, 3,774 Intervals) ·
        Translating <b>Stand-Ahead Hazard Probabilities (Next 30 ft)</b> into <b>Expected NPT Dollars ($)</b> & Cost-Optimal Alert Thresholds
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("💼 Rig Financial Parameters")
rig_day_rate = st.sidebar.slider("Rig Spread Rate ($/Day)", 30000, 120000, 60000, 5000)
npt_hours_event = st.sidebar.slider("Unmitigated NPT Duration (Hours)", 2.0, 24.0, 7.2, 0.4)
intervention_hours = st.sidebar.slider("Early Precautionary Action (Hours)", 0.5, 4.0, 1.4, 0.1)

cost_npt = (rig_day_rate / 24.0) * npt_hours_event
cost_action = (rig_day_rate / 24.0) * intervention_hours

st.sidebar.markdown(f"""
**Calculated Event Economics:**
- **Unmitigated NPT Event Cost:** `${cost_npt:,.0f}`
- **Precautionary Sweep/Ream Cost:** `${cost_action:,.0f}`
- **Cost Ratio (NPT : Action):** `{cost_npt / cost_action:.1f}x`
""")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Live Stand-Ahead Simulator")

preset = st.sidebar.selectbox(
    "Select Drilling Scenario Preset:",
    [
        "🚨 Pre-Packoff / Rising MSE Trend (High NPT Risk)",
        "⚠️ Early Stick-Slip Precursor (Moderate-High Risk)",
        "✅ Smooth Normal Drilling (Low NPT Risk)",
        "🔍 Inspect Actual Wellbore Depth Interval"
    ]
)

if preset == "🔍 Inspect Actual Wellbore Depth Interval":
    row_idx = st.sidebar.slider("Wellbore Row Index", 20, len(df) - 20, 960)
    sample_row = df.iloc[row_idx]
elif preset == "🚨 Pre-Packoff / Rising MSE Trend (High NPT Risk)":
    sample_row = df[df['NPT_Prob_Pct'] > 82].iloc[5]
elif preset == "⚠️ Early Stick-Slip Precursor (Moderate-High Risk)":
    med_df = df[(df['NPT_Prob_Pct'] >= 22) & (df['NPT_Prob_Pct'] <= 48)]
    sample_row = med_df.iloc[0] if len(med_df) > 0 else df.iloc[500]
else:
    sample_row = df[df['NPT_Prob_Pct'] < 8].iloc[10]

wob = st.sidebar.slider("Weight on Bit (WOB, klbs)", 1.0, 45.0, float(sample_row['WOB_klbs']), 0.5)
rpm = st.sidebar.slider("Surface RPM", 40.0, 220.0, float(sample_row['RPM']), 2.0)
torque = st.sidebar.slider("Surface Torque (ft-lbs)", 500.0, 25000.0, float(sample_row['Torque_ftlbs']), 100.0)
rop = st.sidebar.slider("Rate of Penetration (ROP, ft/hr)", 5.0, 180.0, float(sample_row['ROP_fthr']), 1.0)
mse = st.sidebar.slider("Current MSE (kpsi)", 5.0, 250.0, float(sample_row['MSE_kpsi']), 1.0)
rolling_mse = st.sidebar.slider("30-ft Rolling MSE Trend (kpsi)", 5.0, 250.0, float(sample_row['Rolling_MSE_30ft']), 1.0)
rolling_trq_std = st.sidebar.slider("30-ft Torque Volatility (Std, ft-lbs)", 10.0, 5000.0, float(sample_row['Rolling_Torque_Std']), 25.0)

# Recompute dynamic threshold curve based on user's Rig Spread Rate
thresholds = np.linspace(0.05, 0.95, 91)
y_test = metrics["y_test"]
p_eval = metrics["p_eval"]
net_savings_curve = []
for th in thresholds:
    preds_th = (p_eval >= th).astype(int)
    tp = np.sum((preds_th == 1) & (y_test == 1))
    fp = np.sum((preds_th == 1) & (y_test == 0))
    savings = tp * (cost_npt - cost_action) - fp * cost_action
    net_savings_curve.append(savings / 1000.0)

net_savings_curve = np.array(net_savings_curve)
opt_idx = int(np.argmax(net_savings_curve))
opt_th_pct = thresholds[opt_idx] * 100.0
opt_savings_k = net_savings_curve[opt_idx]
idx_50 = int(np.argmin(np.abs(thresholds - 0.50)))
savings_50_k = net_savings_curve[idx_50]
extra_value_k = opt_savings_k - savings_50_k

# Score live vector
eps = 1e-6
live_vec = np.array([
    wob, rpm, torque, rop,
    float(sample_row['Flow_gpm']),
    float(sample_row['GR_API']),
    mse,
    rop / (wob * rpm + eps),
    torque / (wob + eps),
    rop / (wob + eps),
    torque / (rpm + eps),
    rolling_mse,
    rolling_trq_std,
    float(sample_row['ROP_Decay_30ft'])
], dtype=float).reshape(1, -1)

live_prob_pct = float((0.65 * gbm.predict_proba(live_vec)[0, 1] + 0.35 * lr.predict_proba(sc.transform(live_vec))[0, 1]) * 100.0)
expected_unmitigated_loss = (live_prob_pct / 100.0) * cost_npt
expected_action_benefit = expected_unmitigated_loss - cost_action

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Stand-Ahead NPT Probability",
    f"{live_prob_pct:.1f}%",
    "🚨 EXCEEDS OPTIMAL THRESHOLD" if live_prob_pct >= opt_th_pct else "✅ BELOW ALERT THRESHOLD",
    delta_color="inverse" if live_prob_pct >= opt_th_pct else "normal"
)
c2.metric("Expected Unmitigated Risk ($)", f"${expected_unmitigated_loss:,.0f}", f"vs ${cost_action:,.0f} Action Cost")
c3.metric("Optimal Alert Threshold", f"{opt_th_pct:.0f}% Prob", f"+${extra_value_k:,.0f}K vs 50% Cutoff")
c4.metric("Model ROC-AUC / PR-AUC", f"{metrics['gb_auc']:.3f} / {metrics['gb_pr']:.3f}", f"Brier Score: {metrics['gb_brier']:.4f}")

if live_prob_pct >= opt_th_pct:
    st.error(
        f"🚨 **ECONOMIC ACTION TRIGGERED (Stand-Ahead Risk = {live_prob_pct:.1f}% ≥ {opt_th_pct:.0f}% Threshold):** "
        f"Expected unmitigated NPT exposure is **${expected_unmitigated_loss:,.0f}**, while a precautionary sweep/parameter adjustment costs **${cost_action:,.0f}**. "
        f"Taking early action yields an expected net benefit of **+${expected_action_benefit:,.0f}** on this stand."
    )
else:
    st.success(
        f"✅ **DRILL AHEAD (Stand-Ahead Risk = {live_prob_pct:.1f}% < {opt_th_pct:.0f}% Threshold):** "
        f"Expected NPT exposure (**${expected_unmitigated_loss:,.0f}**) is lower than the cost of interrupting drilling (**${cost_action:,.0f}**). Maintain current parameters."
    )

tab1, tab2, tab3 = st.tabs([
    "💰 Cost-Sensitive Threshold Optimizer",
    "📈 Wellbore Stand-Ahead Hazard Profile",
    "🧭 Progression Series Context"
])

with tab1:
    col_a, col_b = st.columns([1.3, 1.0])
    with col_a:
        st.subheader("Net Test Section Savings vs. Probability Decision Threshold")
        fig1, ax1 = plt.subplots(figsize=(8, 4.6), facecolor="#161b22")
        ax1.set_facecolor("#161b22")
        ax1.plot(thresholds * 100, net_savings_curve, color="#3fb950", lw=3, label="Net Savings (USD K)")
        ax1.fill_between(thresholds * 100, net_savings_curve, 0, where=(net_savings_curve > 0), color="#3fb950", alpha=0.15)
        ax1.axvline(opt_th_pct, color="#f0883e", ls="--", lw=2.2, label=f"Optimal Economic Threshold ({opt_th_pct:.0f}%)")
        ax1.axvline(50.0, color="#f85149", ls=":", lw=2.0, label="Default 50% ML Threshold")
        ax1.axvline(live_prob_pct, color="#58a6ff", lw=2.5, label=f"Live Stand Risk ({live_prob_pct:.1f}%)")
        ax1.scatter([opt_th_pct], [opt_savings_k], color="#f0883e", s=90, zorder=5)
        ax1.set_xlabel("NPT Probability Alert Threshold (%)", color="#c9d1d9")
        ax1.set_ylabel("Net Test Section Savings (USD Thousands)", color="#c9d1d9")
        ax1.tick_params(colors="#c9d1d9")
        ax1.grid(alpha=0.2, color="#30363d")
        ax1.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#f0f6fc")
        st.pyplot(fig1)

    with col_b:
        st.subheader("Top Precursor Risk Drivers (Gradient Boosting)")
        importances = gbm.feature_importances_ * 100.0
        top_idx = np.argsort(importances)[-8:]
        fig2, ax2 = plt.subplots(figsize=(6.5, 4.6), facecolor="#161b22")
        ax2.set_facecolor("#161b22")
        colors = ["#f85149" if i >= len(top_idx) - 3 else "#58a6ff" for i in range(len(top_idx))]
        ax2.barh([FEATURES[i] for i in top_idx], importances[top_idx], color=colors, edgecolor="#f0f6fc", height=0.6)
        for i, idx_val in enumerate(top_idx):
            ax2.text(importances[idx_val] + 1.0, i, f"{importances[idx_val]:.1f}%", va="center", color="#f0f6fc", fontweight="bold", fontsize=9)
        ax2.set_xlim(0, max(importances) * 1.22)
        ax2.set_xlabel("Relative Importance (%)", color="#c9d1d9")
        ax2.tick_params(colors="#c9d1d9")
        ax2.grid(axis="x", alpha=0.2, color="#30363d")
        st.pyplot(fig2)

with tab2:
    st.subheader("Volve Well F-9A — Stand-Ahead NPT Probability & Financial Exposure Log")
    d_arr = df['Depth_ft'].values.copy()
    p_arr = df['NPT_Prob_Pct'].values.copy()
    exp_arr = (p_arr / 100.0) * (cost_npt / 1000.0)
    gaps = np.where(np.diff(d_arr) > 15.0)[0] + 1
    d_plot = np.insert(d_arr, gaps, np.nan)
    p_plot = np.insert(p_arr, gaps, np.nan)
    exp_plot = np.insert(exp_arr, gaps, np.nan)

    fig3, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 6), sharey=True, facecolor="#161b22")
    for ax in (ax_a, ax_b):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.grid(alpha=0.2, color="#30363d")

    ax_a.plot(p_plot, d_plot, color="#f85149", lw=1.1, label="Stand-Ahead NPT Prob (%)")
    ax_a.axvline(opt_th_pct, color="#f0883e", ls="--", lw=1.8, label=f"Optimal Cutoff ({opt_th_pct:.0f}%)")
    ax_a.set_ylim(d_arr.max() + 30, d_arr.min() - 30)
    ax_a.set_xlabel("NPT Hazard Probability (%)", color="#c9d1d9")
    ax_a.set_ylabel("Measured Depth (ft)", color="#c9d1d9")
    ax_a.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#f0f6fc", loc="lower right")

    ax_b.plot(exp_plot, d_plot, color="#3fb950", lw=1.1, label="Expected Risk Exposure ($K)")
    ax_b.axvline(cost_action / 1000.0, color="#58a6ff", ls="--", lw=1.8, label=f"Action Cost (${cost_action/1000:.1f}K)")
    ax_b.set_xlabel("Expected Unmitigated Cost ($ Thousands)", color="#c9d1d9")
    ax_b.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#f0f6fc", loc="lower right")

    st.pyplot(fig3)

with tab3:
    st.markdown("""
    ### 🔗 How DrillRisk Fits into the Drilling ML Progression Series
    | Project | Paradigm | Core Engineering Question | Key Metric |
    |---|---|---|---|
    | **1. Axial Flare** | Supervised Regression | *What should my baseline ROP be right now?* | `R² = 0.858` |
    | **2. DrillSense** | Supervised Classification | *Which of the 4 dysfunctions is happening right now?* | `96.6% Accuracy` |
    | **3A. DrillCast** | Sequence Deep Learning (Bi-LSTM) | *What dysfunction state is coming in the next 15 ft?* | `94.8% Accuracy` |
    | **3B. DrillGuard** | Unsupervised Anomaly Detection | *What if an abnormal event occurs that was never labeled?* | `0.997 ROC-AUC` |
    | **3C. DrillRisk (This App)** | Cost-Sensitive Business ML | *What is the stand-ahead NPT probability in dollars ($), and when should we intervene?* | `+$493K Net Value` |
    | **4. DrillAgent** | Autonomous ReAct AI Agent | *How do we orchestrate all 4 ML models + Offset DDR RAG into one autonomous copilot?* | `96.7% Task Success` |
    """)
