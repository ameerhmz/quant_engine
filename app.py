import os
import sys
import time
import json
import threading
import gradio as gr
import pandas as pd
import numpy as np

from broker_bridge import LiveMarketPaperEngine
from web_server import launch_institutional_web_server

# Initialize Paper Trading Engine & background server
engine = LiveMarketPaperEngine()
local_server_url = launch_institutional_web_server(engine, port=8888, auto_open_browser=False, host="127.0.0.1")

def get_telemetry():
    balance = engine.get_account_balance()
    trades = engine.trade_history[-10:] if engine.trade_history else []
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["pair", "type", "entry_spread", "exit_spread", "net_pnl", "reason"])
    
    status_text = f"""
    ### ⚡ QuantEngine Live Status (24/7 Cloud)
    * **Portfolio Equity:** ${balance.get('equity', 100000.0):,.2f}
    * **Available Cash:** ${balance.get('cash', 100000.0):,.2f}
    * **Total Trades Executed:** {balance.get('total_trades', 0)}
    * **Execution Engine:** Active (Kalman Filter + Ornstein-Uhlenbeck + ResNet Alpha)
    """
    return status_text, trades_df

def create_ui():
    with gr.Blocks(title="QuantEngine - Statistical Arbitrage & Quantitative Analytics") as demo:
        gr.Markdown("# ⚡ QuantEngine - Statistical Arbitrage Cloud Terminal")
        gr.Markdown("24/7 Automated High-Frequency Statistical Arbitrage, Kalman Filtering, & HRP Portfolio Allocation.")
        
        with gr.Row():
            status_markdown = gr.Markdown()
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Live Telemetry", variant="primary")
        
        with gr.Tab("📈 Live Trade History"):
            trade_table = gr.Dataframe(headers=["pair", "type", "entry_spread", "exit_spread", "net_pnl", "reason"])
            
        with gr.Tab("🖥️ Full Institutional Web Dashboard"):
            gr.HTML(f"""
            <iframe src="{local_server_url}" width="100%" height="700px" style="border:none; border-radius:12px;"></iframe>
            """)

        demo.load(get_telemetry, outputs=[status_markdown, trade_table])
        refresh_btn.click(get_telemetry, outputs=[status_markdown, trade_table])

    return demo

demo = create_ui()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        ssr=False,
        show_error=True
    )
