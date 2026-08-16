"""
Institutional Execution & Market Friction Engine Module
Simulates realistic passive Limit Order execution (Maker Rebates) and smart order routing.
"""

import math
import random
import numpy as np

class ExecutionEngine:
    def __init__(self, taker_fee_pct=0.0001, maker_rebate_pct=0.00005, half_spread_pct=0.00010, slippage_coeff=0.00005):
        """
        Institutional Smart Order Router using Passive Limit Orders (Maker Rebates).
        """
        self.taker_fee_pct = taker_fee_pct
        self.maker_rebate_pct = maker_rebate_pct
        self.half_spread_pct = half_spread_pct
        self.slippage_coeff = slippage_coeff
        
        self.total_commissions_paid = 0.0
        self.total_slippage_incurred = 0.0
        self.total_trades_executed = 0

    def calculate_fill_price(self, mid_price, order_type='LIMIT', side='BUY', size_usd=5000.0, current_volatility=0.001):
        """
        Calculates institutional passive limit order fill price capturing maker spread and rebates.
        """
        spread_impact = mid_price * self.half_spread_pct
        
        if order_type.upper() == 'LIMIT':
            # Passive Limit Order: Fills at mid or inside the spread, earning maker rebate
            if side.upper() == 'BUY':
                fill_price = mid_price - spread_impact * 0.25
            else:
                fill_price = mid_price + spread_impact * 0.25
            fee_rate = -self.maker_rebate_pct  # Maker rebate earned
            slippage_impact = 0.0
        else:
            # Taker Market Order
            size_factor = math.sqrt(max(100.0, size_usd) / 5000.0)
            vol_factor = max(0.5, current_volatility / 0.001)
            slippage_impact = mid_price * (self.slippage_coeff * size_factor * vol_factor)
            if side.upper() == 'BUY':
                fill_price = mid_price + spread_impact + slippage_impact
            else:
                fill_price = mid_price - spread_impact - slippage_impact
            fee_rate = self.taker_fee_pct

        trade_notional = fill_price * (size_usd / max(0.01, mid_price))
        commission = max(0.0, trade_notional * fee_rate)
        slippage_dollar = abs(fill_price - mid_price) * (size_usd / max(0.01, mid_price))
        
        self.total_commissions_paid += commission
        self.total_slippage_incurred += slippage_dollar
        self.total_trades_executed += 1

        return {
            'mid_price': mid_price,
            'fill_price': fill_price,
            'commission': commission,
            'slippage_dollar': slippage_dollar,
            'order_type': order_type,
            'side': side
        }
