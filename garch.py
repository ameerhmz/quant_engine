"""
GARCH(1,1) Dynamic Volatility Engine Module
Forecasts conditional variance sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
Used for dynamic Value-at-Risk (VaR) caps and adaptive volatility trailing stop-losses.
"""

import math
import numpy as np
import pandas as pd

class GarchVolatilityEngine:
    def __init__(self, omega=2e-6, alpha=0.08, beta=0.90):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.sigma2 = 1e-4

    def update(self, log_return):
        """
        Updates GARCH(1,1) variance with new log return observation.
        Returns forecasted conditional volatility (sigma_t).
        """
        self.sigma2 = self.omega + (self.alpha * (log_return ** 2)) + (self.beta * self.sigma2)
        cond_vol = math.sqrt(max(1e-8, self.sigma2))
        return cond_vol

    def calculate_var_limit(self, capital, confidence=0.99):
        """Calculates 99% Parametric Value-at-Risk (VaR) limit in dollars."""
        cond_vol = math.sqrt(max(1e-8, self.sigma2))
        z_99 = 2.326  # 99% Z-score
        var_pct = z_99 * cond_vol
        var_dollar = capital * var_pct
        return var_dollar, var_pct
