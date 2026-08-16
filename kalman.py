"""
Dynamic State-Space Kalman Filter Module for Cointegrated Pairs
Estimates smooth time-varying hedge ratios (beta_t) and intercept (alpha_t),
ensuring residuals capture genuine statistical spread deviations.
"""

import math
import numpy as np
import pandas as pd

class DynamicKalmanFilter:
    def __init__(self, delta=1e-5, R=1e-3):
        """
        State Equation: [beta_t, alpha_t]^T = [beta_{t-1}, alpha_{t-1}]^T + w_t, w_t ~ N(0, W)
        Measurement Equation: Y_t = beta_t * X_t + alpha_t + v_t, v_t ~ N(0, R)
        """
        self.delta = delta
        self.R = R
        self.P = np.eye(2) * 1e-4  # State estimation covariance
        self.W = np.eye(2) * (delta / (1.0 - delta))  # Smooth process noise covariance
        self.state = np.zeros(2)  # [beta (hedge ratio), alpha (intercept)]
        self.is_initialized = False
        self.residual_history = []

    def initialize_with_prior(self, beta_init, alpha_init=0.0):
        """Optionally initializes Kalman filter with OLS prior."""
        self.state = np.array([beta_init, alpha_init], dtype=float)
        self.is_initialized = True

    def update(self, x_price, y_price):
        """
        Updates dynamic state estimate for pair observation (X_t, Y_t).
        Returns time-varying beta, alpha, innovation residual e_t, and normalized Z-score.
        """
        if not self.is_initialized:
            self.state = np.array([y_price / max(1e-5, x_price), 0.0])
            self.is_initialized = True

        # Normalized measurement observation vector H = [X_t, 1.0]
        H = np.array([x_price, 1.0])

        # 1. State prediction
        P_pred = self.P + self.W

        # 2. Measurement prediction variance
        Q_t = float(H @ P_pred @ H.T + self.R)

        # 3. Innovation residual: e_t = Y_t - (beta * X_t + alpha)
        y_pred = float(H @ self.state)
        e_t = y_price - y_pred

        # 4. Kalman Gain with smooth damping factor
        K_t = (P_pred @ H.T) / max(1e-6, Q_t)

        # 5. State update (Smooth evolution to avoid absorbing transient mispricings)
        self.state = self.state + K_t * e_t * 0.05

        # 6. Covariance update
        self.P = (np.eye(2) - np.outer(K_t, H)) @ P_pred

        # 7. True Rolling Spread Z-Score: Z = (e_t - mean(e)) / std(e)
        self.residual_history.append(e_t)
        if len(self.residual_history) > 60:
            self.residual_history.pop(0)

        mean_res = np.mean(self.residual_history)
        std_res = max(1e-4, float(np.std(self.residual_history)))
        z_score = (e_t - mean_res) / std_res

        return {
            'hedge_ratio': float(self.state[0]),
            'intercept': float(self.state[1]),
            'y_pred': float(y_pred),
            'residual': float(e_t),
            'residual_variance': float(Q_t),
            'sigma_innovation': float(std_res),
            'z_score': float(z_score)
        }
