# 💰 DrillRisk — Stand-Ahead NPT & Financial Risk Predictor

**Project 3C of the Drilling ML Progression Series** — Translating Machine Learning probabilities into **Expected Non-Productive Time (NPT) Dollars ($)** and **Cost-Sensitive Decision Thresholds** on the **Equinor Volve Field (Well F-9A)** dataset (`3,774` depth intervals).

---

## 🎯 Why Cost-Sensitive ML Matters on the Rig Floor
Machine learning models default to a **50% probability threshold**—treating a false alarm and a missed catastrophic event as equally costly.
On a drilling rig, that assumption is dangerously wrong:
- **Unmitigated NPT Incident (False Negative):** Tight hole, pack-off, severe stick-slip damage, or unplanned trip (`~$18,000+` per event in rig spread cost).
- **Early Precautionary Intervention (False Positive):** Sweeping the hole, reaming a stand, or adjusting WOB/RPM (`~$3,500` in rig time).

By coupling a **Calibrated Gradient Boosting Hazard Predictor** with a **Rig Cost Matrix**, **DrillRisk** finds the mathematically optimal decision threshold (`19%` probability) that maximizes net operational savings.

---

## 📊 Model Performance & Economic Impact
| Metric / Benchmark | Result |
|---|---|
| **Calibrated Gradient Boosting ROC-AUC** | **0.982** |
| **Precision-Recall AUC (PR-AUC)** | **0.959** |
| **F1-Score / Classification Accuracy** | **0.862 / 93.5%** |
| **Optimal Economic Alert Threshold** | **19% Probability** (`$3,596K` Net Test Savings) |
| **Standard 50% ML Threshold** | **50% Probability** (`$3,103K` Net Test Savings) |
| **Value Added by Threshold Optimization** | **+$493,000** across 3,774 intervals |

---

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔗 Drilling ML Progression Series
1. **Project 1 — Axial Flare (Regression):** [axial-flare-copilot.streamlit.app](https://axial-flare-copilot.streamlit.app) — Predicting continuous Rate of Penetration (`ROP`).
2. **Project 2 — DrillSense (Classification):** [drillsense-copilot.streamlit.app](https://drillsense-copilot.streamlit.app) — Classifying 4 drilling dysfunction states (`96.6%` accuracy).
3. **Project 3A — DrillCast (Sequence DL):** Bi-LSTM forecasting the next 15 ft of drilling dysfunctions from a 30-ft lookback window (`94.8%` accuracy).
4. **Project 3B — DrillGuard (Unsupervised ML):** Hybrid Autoencoder + Isolation Forest detecting downhole anomalies with **zero labels** (`0.997` ROC-AUC).
5. **Project 3C — DrillRisk (Business ML):** Stand-Ahead NPT probability & cost-sensitive threshold optimization (`+$493K` net value).
