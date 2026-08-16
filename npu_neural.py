"""
Apple Silicon NPU & Metal GPU ResNet Deep Learning Engine (v15.0)
Uses PyTorch Residual Network Blocks (ResNet) with LayerNorm, GELU, and Bayesian Weight Regularization
on Apple Silicon MPS (Metal Performance Shaders) for statistical arbitrage spread alpha confidence estimation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class ResidualBlock(nn.Module):
    """Deep Residual Block with GELU activation, LayerNorm, and Skip Connection."""
    def __init__(self, dim=64):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        out = self.act(out + residual)
        return out

class ResNetNPUAlphaModel(nn.Module):
    """Institutional ResNet Architecture for Statistical Arbitrage Alpha Time Series."""
    def __init__(self, input_dim=9, hidden_dim=64):
        super(ResNetNPUAlphaModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.input_layer(x))
        x = self.res1(x)
        x = self.res2(x)
        out = torch.sigmoid(self.out_layer(x))
        return out

class AppleNPUNeuralEngine:
    def __init__(self, input_dim=9):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.is_npu_active = True
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.is_npu_active = True
        else:
            self.device = torch.device("cpu")
            self.is_npu_active = False

        self.model = ResNetNPUAlphaModel(input_dim=input_dim, hidden_dim=64).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
        self.criterion = nn.BCELoss()
        self.history_features = []
        self.history_returns = []

    def extract_pair_features(self, spread_series, kalman_z=0.0, ou_z=0.0, ou_hl=10.0, garch_vol=0.001, beta_velocity=0.0, trend_div=0.0):
        """Extracts 9-dimensional statistical arbitrage feature vector."""
        if len(spread_series) < 21:
            return None

        s = np.asarray(spread_series, dtype=float)
        ret1 = np.clip((s[-1] - s[-2]) / (abs(s[-2]) + 1e-5), -0.05, 0.05)
        ret5 = np.clip((s[-1] - s[-6]) / (abs(s[-6]) + 1e-5), -0.08, 0.08)
        ret20 = np.clip((s[-1] - s[-20]) / (abs(s[-20]) + 1e-5), -0.15, 0.15)

        kz_norm = np.clip(kalman_z / 3.0, -2.0, 2.0)
        ouz_norm = np.clip(ou_z / 3.0, -2.0, 2.0)
        hl_norm = np.clip(ou_hl / 30.0, 0.0, 3.0)
        vol_norm = np.clip(garch_vol * 100.0, 0.0, 5.0)
        beta_v_norm = math.tanh(beta_velocity * 10.0)
        trend_norm = math.tanh(trend_div * 5.0)

        feat = np.array([ret1, ret5, ret20, kz_norm, ouz_norm, hl_norm, vol_norm, beta_v_norm, trend_norm], dtype=np.float32)
        return feat

    def predict_alpha_confidence(self, spread_series, kalman_z=0.0, ou_z=0.0, ou_hl=10.0, garch_vol=0.001, beta_velocity=0.0, trend_div=0.0):
        """Runs PyTorch ResNet Forward Pass to predict spread reversion confidence."""
        feat = self.extract_pair_features(spread_series, kalman_z, ou_z, ou_hl, garch_vol, beta_velocity, trend_div)
        if feat is None:
            return 0.50

        self.model.eval()
        x_tensor = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw_prob = self.model(x_tensor).item()

        # Clamp confidence to [0.10, 0.90] to prevent extreme overconfidence
        prob = max(0.10, min(0.90, raw_prob))

        # Record features for offline batch training
        self.history_features.append(feat)
        if len(spread_series) >= 2:
            future_diff = float(spread_series.iloc[-1] - spread_series.iloc[-2]) if hasattr(spread_series, 'iloc') else float(spread_series[-1] - spread_series[-2])
            if abs(future_diff) > 0.0001:
                target = 1.0 if future_diff > 0 else 0.0
                self.history_returns.append(target)
            else:
                self.history_returns.append(0.5)

        if len(self.history_features) >= 40:
            self.train_online_step()

        return float(prob)

    def train_online_step(self):
        """Executes regularized AdamW backprop on Apple Metal GPU / NPU with gradient clipping."""
        if len(self.history_features) < 30 or len(self.history_returns) < 30:
            return

        self.model.train()
        X = torch.tensor(np.array(self.history_features[-40:]), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.array(self.history_returns[-40:]), dtype=torch.float32, device=self.device).unsqueeze(1)

        self.optimizer.zero_grad()
        preds = self.model(X)
        loss = self.criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
