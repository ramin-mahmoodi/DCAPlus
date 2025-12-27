# ⚡ DCA-Plus

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production-success)

**DCA-Plus** is a state-of-the-art **Petroleum Engineering Analytics Platform** designed to modernize reservoir forecasting. It merges traditional Decline Curve Analysis (DCA) with advanced Physics-based modeling (RTA, PVT) and cutting-edge Data Science (Bayesian MCMC, Neural Networks).

---

## 🚀 Features

### 📉 Core Engineering Engine
*   **Robust Fitting**: Arps (Exponential, Harmonic, Hyperbolic) using `scipy.optimize`.
*   **Ensemble Modeling**: Automatically averages top models weighted by AIC.
*   **Probabilistic Forecasting**: Monte Carlo simulation for P10/P50/P90 UR uncertainty.
*   **Economics**: Discounted Cash Flow (DCF), NPV, ROI, and Payout analysis.

### 🛑 Advanced Physics (v5.0)
*   **Rate Transient Analysis (RTA)**: Blasingame Type Curves for pressure-rate analysis.
*   **PVT simulator**: 0D Material Balance tank model for reservoir pressure tracking.
*   **Auto-History Matching**: Inverse solver to determine Oil-In-Place (N) and Productivity Index (J).

### 🧠 Analytics & AI
*   **Neural Forecasting**: LSTM-based architecture for complex time-series prediction.
*   **Bayesian Inference**: Metropolis-Hastings MCMC sampler for true posterior probability distributions.
*   **Waterflood CRM**: Capacitance Resistance Modeling for Injector-Producer connectivity.
*   **Geo Heatmaps**: Spatial interpolation of EUR for acreage grading.
*   **AI Copilot**: Natural Language Query engine ("What is the EUR for Well A?").

### 🏢 Enterprise Tools
*   **Portfolio Optimizer**: Capital allocation using Knapsack-style optimization.
*   **Smart Reporting**: Auto-generated PDF reports and AI-narrated summaries.
*   **Headless API**: FastAPI backend for integration with external dashboards (PowerBI/Spotfire).

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dca-plus.git
cd dca-plus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🖥️ Usage

Run the web application locally:

```bash
streamlit run app.py
```

### Headless API Mode
To run the REST API for external integrations:

```bash
uvicorn api:app --reload
```

---

## 📂 Project Structure

```
dca-plus/
├── app.py                 # Main Streamlit Application (Frontend)
├── api.py                 # FastAPI Gateway (Backend)
├── dca/                   # Core Library
│   ├── models.py          # Arps Equations
│   ├── fit.py             # Optimization Logic (Curve Fit)
│   ├── bayesian.py        # MCMC Sampler
│   ├── rta.py             # Rate Transient Analysis
│   ├── neural.py          # ML Forecasting
│   └── ...
├── data/                  # Sample Datasets
│   ├── sample_well.csv
│   └── sample_waterflood.csv
├── tests/                 # Unit Tests
└── requirements.txt       # Dependencies
```

## 📊 Sample Data Format

Input CSV files should follow this format:

| date       | oil_rate | pressure (optional) | water_injection (optional) |
|------------|----------|---------------------|----------------------------|
| 2023-01-01 | 500.0    | 3000                | 0                          |
| 2023-01-02 | 498.5    | 2995                | 100                        |

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

