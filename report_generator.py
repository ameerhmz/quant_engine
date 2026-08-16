"""
Automated Institutional Trading Report & Live Dashboard Generator Module
Generates high-definition Markdown reports (PAPER_TRADING_REPORT.md) and
interactive HTML Dashboards (paper_trading_dashboard.html) automatically upon every ledger update.
Includes live dynamic USD to INR conversion and session uptime tracking.
"""

import os
import json
import datetime
import numpy as np
try:
    from fx_rates import get_live_usd_inr_rate, usd_to_inr, usd_to_inr_lakhs
except ImportError:
    from .fx_rates import get_live_usd_inr_rate, usd_to_inr, usd_to_inr_lakhs

class AutoReportGenerator:
    def __init__(self, md_path=None, html_path=None):
        home = os.path.expanduser("~")
        self.md_path = md_path or os.path.join(home, "PAPER_TRADING_REPORT.md")
        self.html_path = html_path or os.path.join(home, "paper_trading_dashboard.html")

    def generate(self, ledger_data):
        """Generates both Markdown and HTML reports from live ledger data."""
        self._generate_markdown(ledger_data)
        self._generate_html(ledger_data)

    def _generate_markdown(self, data):
        equity = data.get("equity", 10467.78)
        initial_cap = data.get("initial_capital", 10467.78)
        trades = data.get("trade_history", [])
        total_pnl = equity - initial_cap
        pnl_pct = (total_pnl / initial_cap) * 100.0 if initial_cap > 0 else 0.0
        
        live_fx = get_live_usd_inr_rate()
        inr_equity = usd_to_inr(equity)
        inr_pnl = usd_to_inr(total_pnl)
        inr_lakhs = usd_to_inr_lakhs(equity)

        wins = [t for t in trades if t.get("net_pnl", 0) > 0]
        losses = [t for t in trades if t.get("net_pnl", 0) <= 0]
        total_trades = len(trades)
        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
        
        tot_gain = sum(t.get("net_pnl", 0) for t in wins)
        tot_loss = abs(sum(t.get("net_pnl", 0) for t in losses))
        profit_factor = (tot_gain / tot_loss) if tot_loss > 0 else (tot_gain if tot_gain > 0 else 1.0)
        avg_win = (tot_gain / len(wins)) if wins else 0.0
        avg_loss = (tot_loss / len(losses)) if losses else 0.0

        pnl_badge = f"+${total_pnl:,.2f} (+{pnl_pct:.2f}%) | +₹{inr_pnl:,.0f} INR" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f} ({pnl_pct:.2f}%) | -₹{abs(inr_pnl):,.0f} INR"
        now_str = datetime.datetime.now().strftime("%d %b %Y, %H:%M:%S")

        # Calculate Running Time / Uptime
        start_time_str = trades[0].get("entry_time", "--:--:--") if trades else "--:--:--"
        last_time_str = trades[-1].get("exit_time", "--:--:--") if trades else "--:--:--"

        # Per-Pair Breakdown
        pairs_stats = {}
        for t in trades:
            p_name = t.get("pair", "Unknown")
            if p_name not in pairs_stats:
                pairs_stats[p_name] = {"trades": 0, "wins": 0, "pnl": 0.0}
            pairs_stats[p_name]["trades"] += 1
            if t.get("net_pnl", 0) > 0:
                pairs_stats[p_name]["wins"] += 1
            pairs_stats[p_name]["pnl"] += t.get("net_pnl", 0)

        md = []
        md.append("# 📊 Live Institutional Paper Trading Performance Report")
        md.append(f"> **Status:** 🟢 Live Engine Active | **Last Updated:** `{now_str}` | **Live FX:** `1 USD = ₹{live_fx:.2f} INR`")
        md.append(f"> **Session Activity:** First Trade `{start_time_str}` ➔ Last Trade `{last_time_str}`")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 🏆 Executive Portfolio Summary")
        md.append("")
        md.append(f"* **Current Portfolio Equity (USD):** **`${equity:,.2f}`**")
        md.append(f"* **Current Portfolio Equity (INR):** **`₹{inr_equity:,.2f} INR`** (**₹{inr_lakhs:.2f} Lakhs**)")
        md.append(f"* **Realized Net Alpha PnL:** **`{pnl_badge}`**")
        md.append(f"* **Total Completed Trades:** **`{total_trades}`**")
        md.append(f"* **Win Rate:** **`{win_rate:.1f}%`** ({len(wins)} Wins / {len(losses)} Losses)")
        md.append(f"* **Profit Factor:** **`{profit_factor:.2f}`**")
        md.append(f"* **Average Win / Average Loss:** **`${avg_win:.2f}` / `${avg_loss:.2f}`**")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 📈 Performance Breakdown by Cointegrated Pair")
        md.append("")
        md.append("| Cointegrated Pair | Total Trades | Win Rate | Net PnL (USD) | Equiv Net PnL (INR) |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for p_name, s in pairs_stats.items():
            p_wr = (s["wins"] / s["trades"] * 100.0) if s["trades"] > 0 else 0.0
            p_inr = usd_to_inr(s["pnl"])
            p_pnl_str = f"+${s['pnl']:,.2f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):,.2f}"
            p_inr_str = f"+₹{p_inr:,.0f}" if p_inr >= 0 else f"-₹{abs(p_inr):,.0f}"
            md.append(f"| **{p_name}** | {s['trades']} | **{p_wr:.1f}%** | `{p_pnl_str}` | `{p_inr_str}` |")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 📜 Recent Executed Trade Journal (Last 15 Trades)")
        md.append("")
        md.append("| ID | Cointegrated Pair | Direction | Entry Time | Exit Time | Net PnL ($) | Equiv INR | Cumulative Equity | Strategy Reason |")
        md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        for t in trades[-15:]:
            t_pnl = t.get("net_pnl", 0)
            t_inr = usd_to_inr(t_pnl)
            pnl_col = f"+${t_pnl:.2f}" if t_pnl >= 0 else f"-${abs(t_pnl):.2f}"
            inr_col = f"+₹{t_inr:.1f}" if t_inr >= 0 else f"-₹{abs(t_inr):.1f}"
            p_type = "**LONG**" if "LONG" in t.get("type", "") else "**SHORT**"
            md.append(f"| #{t.get('id', 0)} | {t.get('pair', '')[:28]} | {p_type} | {t.get('entry_time', '')} | {t.get('exit_time', '')} | `{pnl_col}` | `{inr_col}` | `${t.get('portfolio_equity', 0):,.2f}` | {t.get('reason', '')} |")
        md.append("")
        md.append("---")
        md.append("*Generated automatically by the PhD Quantitative Trading Engine.*")

        try:
            with open(self.md_path, "w") as f:
                f.write("\n".join(md))
        except Exception:
            pass

    def _generate_html(self, data):
        equity = data.get("equity", 10467.78)
        initial_cap = data.get("initial_capital", 10467.78)
        trades = data.get("trade_history", [])
        total_pnl = equity - initial_cap
        pnl_pct = (total_pnl / initial_cap) * 100.0 if initial_cap > 0 else 0.0
        
        live_fx = get_live_usd_inr_rate()
        inr_equity = usd_to_inr(equity)
        inr_pnl = usd_to_inr(total_pnl)
        inr_lakhs = usd_to_inr_lakhs(equity)

        wins = [t for t in trades if t.get("net_pnl", 0) > 0]
        losses = [t for t in trades if t.get("net_pnl", 0) <= 0]
        total_trades = len(trades)
        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
        
        tot_gain = sum(t.get("net_pnl", 0) for t in wins)
        tot_loss = abs(sum(t.get("net_pnl", 0) for t in losses))
        profit_factor = (tot_gain / tot_loss) if tot_loss > 0 else (tot_gain if tot_gain > 0 else 1.0)
        
        pnl_color = "#00c087" if total_pnl >= 0 else "#ff3b30"
        now_str = datetime.datetime.now().strftime("%d %b %Y, %H:%M:%S")

        # Table rows
        table_rows = ""
        for t in reversed(trades[-25:]):
            t_pnl = t.get("net_pnl", 0)
            t_inr = usd_to_inr(t_pnl)
            c = "#00c087" if t_pnl >= 0 else "#ff3b30"
            sign = "+" if t_pnl >= 0 else ""
            t_badge = "<span style='background:rgba(0,192,135,0.15);color:#00c087;padding:3px 8px;border-radius:4px;font-weight:600;'>LONG</span>" if "LONG" in t.get("type", "") else "<span style='background:rgba(255,59,48,0.15);color:#ff3b30;padding:3px 8px;border-radius:4px;font-weight:600;'>SHORT</span>"
            table_rows += f"""
            <tr>
                <td>#{t.get('id',0)}</td>
                <td><strong>{t.get('pair','')}</strong></td>
                <td>{t_badge}</td>
                <td>{t.get('entry_time','')}</td>
                <td>{t.get('exit_time','')}</td>
                <td style='color:{c};font-weight:bold;'>{sign}${t_pnl:.2f} ({sign}₹{t_inr:.1f})</td>
                <td>${t.get('portfolio_equity',0):,.2f}</td>
                <td style='color:#8b949e;font-size:11px;'>{t.get('reason','')}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Paper Trading Executive Dashboard</title>
    <style>
        body {{ background:#090d14; color:#e6edf3; font-family:-apple-system,BlinkMacSystemFont,sans-serif; margin:0; padding:20px; font-size:12px; }}
        .header {{ display:flex; justify-content:space-between; align-items:center; background:#111622; padding:12px 18px; border-radius:6px; border:1px solid #232c3d; margin-bottom:15px; }}
        .grid {{ display:grid; grid-template-columns:repeat(5, 1fr); gap:12px; margin-bottom:20px; }}
        .card {{ background:#111622; padding:12px; border-radius:6px; border:1px solid #232c3d; }}
        .title {{ color:#8b949e; font-size:10px; text-transform:uppercase; margin-bottom:5px; }}
        .val {{ font-size:18px; font-weight:700; font-family:monospace; }}
        .sub {{ font-size:10px; color:#57606a; margin-top:3px; }}
        table {{ width:100%; border-collapse:collapse; background:#111622; border-radius:6px; overflow:hidden; border:1px solid #232c3d; font-family:monospace; }}
        th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #1e273a; }}
        th {{ background:#0d121c; color:#8b949e; font-size:10px; text-transform:uppercase; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <strong style="font-size:14px; color:#ffffff;">🎓 INSTITUTIONAL QUANT TRADING REPORT</strong>
            <span style="background:#171e2e; border:1px solid #232c3d; padding:2px 8px; border-radius:3px; margin-left:10px; color:#8b949e;">FX: 1 USD = ₹{live_fx:.2f} INR</span>
        </div>
        <div style="color:#8b949e; font-family:monospace;">Last Updated: {now_str}</div>
    </div>
    <div class="grid">
        <div class="card">
            <div class="title">Portfolio Equity</div>
            <div class="val" style="color:#ffd700;">${equity:,.2f}</div>
            <div class="sub">₹{inr_lakhs:.2f} Lakhs INR (₹{inr_equity:,.0f})</div>
        </div>
        <div class="card">
            <div class="title">Net Realized Alpha</div>
            <div class="val" style="color:{pnl_color};">{"+" if total_pnl>=0 else ""}${total_pnl:,.2f}</div>
            <div class="sub">{"+" if inr_pnl>=0 else ""}₹{inr_pnl:,.0f} INR ({"+" if pnl_pct>=0 else ""}{pnl_pct:.2f}%)</div>
        </div>
        <div class="card">
            <div class="title">Win Rate</div>
            <div class="val" style="color:#00c087;">{win_rate:.1f}%</div>
            <div class="sub">{total_trades} Completed Trades</div>
        </div>
        <div class="card">
            <div class="title">Profit Factor</div>
            <div class="val" style="color:#2f81f7;">{profit_factor:.2f}</div>
            <div class="sub">Gross Win/Loss Ratio</div>
        </div>
        <div class="card">
            <div class="title">Execution Engine</div>
            <div class="val" style="color:#00c087; font-size:14px;">ACTIVE</div>
            <div class="sub">Alpaca / Live Feed Paper</div>
        </div>
    </div>
    <table>
        <thead>
            <tr>
                <th>ID</th><th>Pair</th><th>Side</th><th>Entry</th><th>Exit</th><th>Net Alpha ($ / INR)</th><th>Equity</th><th>Strategy Reason</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
</body>
</html>
"""
        try:
            with open(self.html_path, "w") as f:
                f.write(html_content)
        except Exception:
            pass
