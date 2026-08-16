"""
PhD Institutional Quantitative Trading Engine Package
"""

from .cointegration import CointegrationEngine
from .kalman import DynamicKalmanFilter
from .stochastic import OrnsteinUhlenbeckProcess
from .hrp import HierarchicalRiskParity
from .garch import GarchVolatilityEngine
from .charting import RealisticCandleChart
from .gpu_accelerator import MetalGPUQuantEngine
from .npu_neural import AppleNPUNeuralEngine
from .execution import ExecutionEngine
from .broker_bridge import AlpacaPaperBridge, LiveMarketPaperEngine
from .report_generator import AutoReportGenerator
from .web_server import launch_institutional_web_server
from .fx_rates import get_live_usd_inr_rate, usd_to_inr, usd_to_inr_lakhs

__all__ = [
    "CointegrationEngine",
    "DynamicKalmanFilter",
    "OrnsteinUhlenbeckProcess",
    "HierarchicalRiskParity",
    "GarchVolatilityEngine",
    "RealisticCandleChart",
    "MetalGPUQuantEngine",
    "AppleNPUNeuralEngine",
    "ExecutionEngine",
    "AlpacaPaperBridge",
    "LiveMarketPaperEngine",
    "AutoReportGenerator",
    "launch_institutional_web_server",
    "get_live_usd_inr_rate",
    "usd_to_inr",
    "usd_to_inr_lakhs"
]
