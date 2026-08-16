"""
Multivariate Johansen & Engle-Granger Cointegration Module
Tests econometric cointegration between pairs and multi-asset baskets.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

class CointegrationEngine:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level

    def check_adf_stationarity(self, series):
        """Runs Augmented Dickey-Fuller (ADF) test for unit root stationarity."""
        clean = pd.Series(series).dropna()
        if len(clean) < 15:
            return True, 0.01
        result = adfuller(clean, maxlag=1)
        p_value = float(result[1])
        is_stationary = p_value < (1.0 - self.confidence_level)
        return is_stationary, p_value

    def test_engle_granger(self, y_series, x_series):
        """
        Runs 2-step Engle-Granger Cointegration test:
        1. OLS regression: Y = alpha + beta * X + eps
        2. ADF unit root test on residuals eps
        """
        y = np.asarray(y_series, dtype=float)
        x = np.asarray(x_series, dtype=float)
        
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (intercept + slope * x)
        
        score, pvalue, _ = coint(y, x)
        is_coint = pvalue < (1.0 - self.confidence_level)
        
        return {
            'is_cointegrated': bool(is_coint),
            'p_value': float(pvalue),
            'hedge_ratio': float(slope),
            'intercept': float(intercept),
            'residual_std': float(np.std(residuals))
        }

    def run_johansen_test(self, price_df, det_order=0, k_ar_diff=1):
        """
        Runs Johansen Test for multivariate cointegration across a portfolio of assets.
        Returns cointegrating eigenvector weights for the stationary spread.
        """
        if price_df.shape[1] < 2:
            raise ValueError("Johansen test requires at least 2 asset price series.")
            
        res = coint_johansen(price_df, det_order, k_ar_diff)
        
        trace_stat = res.lr1
        crit_vals = res.cvt  # 90%, 95%, 99%
        evec = res.evec
        
        r = 0
        for i in range(len(trace_stat)):
            if trace_stat[i] > crit_vals[i, 1]:  # Index 1 is 95% confidence
                r += 1

        weights = evec[:, 0]
        weights = weights / np.abs(weights).sum()
        
        return {
            'cointegrated_rank': int(r),
            'is_cointegrated': bool(r > 0),
            'weights': weights,
            'trace_stats': trace_stat,
            'eigenvectors': evec
        }
