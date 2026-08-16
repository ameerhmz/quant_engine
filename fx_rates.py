"""
Live Real-Time FX Exchange Rate Module
Fetches real-time market exchange rates (USD -> INR) via live currency APIs.
Caches and auto-refreshes in background to ensure zero latency on UI streams.
"""

import time
import json
import threading
import urllib.request

class LiveFXEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LiveFXEngine, cls).__new__(cls)
                cls._instance._rate = 95.50
                cls._instance._last_fetch = 0
                cls._instance.refresh_rate()
                # Start background thread to refresh every 5 minutes
                threading.Thread(target=cls._instance._auto_refresh_loop, daemon=True).start()
            return cls._instance

    def _auto_refresh_loop(self):
        while True:
            time.sleep(300)
            self.refresh_rate()

    def refresh_rate(self):
        """Fetches live USD/INR exchange rate from open exchange rate APIs."""
        # 1. Primary Live API (open.er-api.com)
        try:
            url = "https://open.er-api.com/v6/latest/USD"
            req = urllib.request.Request(url, headers={"User-Agent": "InstitutionalFXEngine/1.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                data = json.loads(response.read().decode("utf-8"))
                if "rates" in data and "INR" in data["rates"]:
                    self._rate = float(data["rates"]["INR"])
                    self._last_fetch = time.time()
                    return self._rate
        except Exception:
            pass

        # 2. Secondary Live API (frankfurter.app)
        try:
            url = "https://api.frankfurter.app/latest?from=USD&to=INR"
            req = urllib.request.Request(url, headers={"User-Agent": "InstitutionalFXEngine/1.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                data = json.loads(response.read().decode("utf-8"))
                if "rates" in data and "INR" in data["rates"]:
                    self._rate = float(data["rates"]["INR"])
                    self._last_fetch = time.time()
                    return self._rate
        except Exception:
            pass

        return self._rate

    def get_rate(self):
        """Returns the cached live USD to INR exchange rate."""
        if time.time() - self._last_fetch > 600:
            self.refresh_rate()
        return self._rate

    def to_inr(self, usd_val):
        return float(usd_val) * self.get_rate()

    def to_inr_lakhs(self, usd_val):
        return (float(usd_val) * self.get_rate()) / 100000.0

fx_engine = LiveFXEngine()

def get_live_usd_inr_rate():
    return fx_engine.get_rate()

def usd_to_inr(usd_amount):
    return fx_engine.to_inr(usd_amount)

def usd_to_inr_lakhs(usd_amount):
    return fx_engine.to_inr_lakhs(usd_amount)
