# ⚡ QuantEngine

> **High-Performance Quantitative Research & Statistical Arbitrage Engine**  
> Accelerated with Apple Silicon Metal Performance Shaders (MPS / NPU), Kalman Filters, Hierarchical Risk Parity (HRP), and GARCH Volatility Modeling.

---

## 🏛️ Architecture Overview

QuantEngine is a modular algorithmic trading and quantitative analytics framework designed for statistical arbitrage, portfolio optimization, dynamic risk management, and live broker execution.

```
                     ┌──────────────────────────────────────┐
                     │            Live Market Data          │
                     └──────────────────┬───────────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌──────────────────────────┐                             ┌──────────────────────────┐
│   Statistical Arbitrage  │                             │   Portfolio Allocation   │
├──────────────────────────┤                             ├──────────────────────────┤
│ • Engle-Granger Cointeg  │                             │ • Hierarchical Risk      │
│ • Kalman Filter State    │                             │   Parity (HRP)           │
│ • Ornstein-Uhlenbeck SDE │                             │ • GARCH(1,1) Volatility  │
│ • Metal GPU / NPU ResNet │                             │ • FX Normalized Risk     │
└────────────┬─────────────┘                             └────────────┬─────────────┘
             │                                                        │
             └──────────────────────────┬─────────────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │       Broker Bridge & Risk Router    │
                     ├──────────────────────────────────────┤
                     │ • Position Sizing & Slippage Control │
                     │ • Alpaca Markets Execution Bridge    │
                     │ • Live WebSocket Web Dashboard       │
                     └──────────────────────────────────────┘
```

---

## 🔬 Core Mathematical & Neural Modules

| Module | Core Concept | Description |
| :--- | :--- | :--- |
| **`npu_neural.py`** | **Deep Residual Alpha Network** | PyTorch ResNet architecture with LayerNorm, GELU activations, and Bayesian Weight Regularization optimized for Apple Silicon MPS (Metal Performance Shaders) to estimate statistical arbitrage alpha confidence scores. |
| **`hrp.py`** | **Hierarchical Risk Parity (HRP)** | Marcos López de Prado's tree-clustering portfolio allocation. Uses hierarchical single-linkage clustering, matrix quasi-diagonalization, and recursive bisection to construct robust risk-parity weights without matrix inversion instability. |
| **`kalman.py`** | **State-Space Filtering** | Adaptive dynamic hedge ratio ($\beta$) tracking. Recursively updates the state transition matrix and measurement covariance to adapt to regime shifts and mean-reversion drift in real time. |
| **`cointegration.py`** | **Cointegration & Stationarity** | Automated pair selection engine utilizing Augmented Dickey-Fuller (ADF), Engle-Granger two-step cointegration, and Johansen test vector error correction. |
| **`garch.py`** | **Conditional Heteroskedasticity** | GARCH(1,1) volatility engine for time-varying variance estimation, volatility clustering identification, and dynamic Value-at-Risk (VaR) calculation. |
| **`stochastic.py`** | **Stochastic Calculus Models** | Ornstein-Uhlenbeck mean-reversion parameter estimation (speed of mean reversion $\theta$, long-term mean $\mu$, volatility $\sigma$), optimal entry/exit threshold solving, and jump-diffusion simulation. |
| **`broker_bridge.py`** | **Execution & Risk Management** | Production-grade broker integration with Alpaca Markets REST/WebSocket APIs, automated safety caps, and circuit breakers. |
| **`web_server.py`** | **Interactive Web Dashboard** | FastAPI & WebSocket dashboard providing real-time PnL monitoring, live spread charting, and tear sheet visualizations. |

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/ameerhmz/quant_engine.git
cd quant_engine
pip install -r requirements.txt
```

### 2. Running the Engine

Start the quantitative web server and dashboard:

```bash
python web_server.py
```

Open your browser at `http://localhost:8000` to monitor live signals, backtests, and portfolio allocations.

---

## 🛡️ License

MIT License. Developed for research and algorithmic trading experiments.
