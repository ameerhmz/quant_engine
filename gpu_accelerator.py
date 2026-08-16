"""
Apple Metal GPU (MPS) Ultra-High-Performance Quantitative Accelerator Module
Batches 100,000 parallel Monte Carlo SDE paths across all cointegrated pairs simultaneously
over multi-minute mean-reversion horizons using 3D PyTorch GPU Tensor operations.
"""

import math
import torch
import numpy as np

class MetalGPUQuantEngine:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.is_gpu_active = True
            self.device_name = "Apple Metal GPU (MPS 100K Batch)"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.is_gpu_active = True
            self.device_name = "NVIDIA CUDA GPU"
        else:
            self.device = torch.device("cpu")
            self.is_gpu_active = False
            self.device_name = "Host CPU"

    def run_gpu_monte_carlo(self, current_spread, theta, mu, sigma_ou, tp_dist=0.015, sl_dist=0.010, num_paths=100000, steps=30):
        """
        Executes 100,000 parallel path Monte Carlo SDE simulation on Apple Metal GPU for a single pair spread.
        """
        res = self.run_batched_gpu_monte_carlo(
            {'PAIR': current_spread},
            {'PAIR': {'theta': theta, 'mu': mu, 'sigma_ou': sigma_ou, 'tp_dist': tp_dist, 'sl_dist': sl_dist}},
            num_paths=num_paths,
            steps=steps
        )
        return res['PAIR']

    def run_batched_gpu_monte_carlo(self, spreads_dict, ou_params_dict, num_paths=100000, steps=30):
        """
        Executes 100,000 parallel path Monte Carlo SDE simulations on Apple Metal GPU
        batched across all cointegrated pairs over a 30-minute mean-reversion horizon.
        """
        if not spreads_dict:
            return {}

        pairs = list(spreads_dict.keys())
        N = len(pairs)

        # Allocate 3D GPU Tensors: (N_pairs, 1)
        s0 = torch.tensor([spreads_dict[p] for p in pairs], dtype=torch.float32, device=self.device).unsqueeze(1)
        mu = torch.tensor([ou_params_dict[p]['mu'] for p in pairs], dtype=torch.float32, device=self.device).unsqueeze(1)
        theta = torch.tensor([max(0.05, ou_params_dict[p]['theta']) for p in pairs], dtype=torch.float32, device=self.device).unsqueeze(1)
        sigma = torch.tensor([max(1e-4, ou_params_dict[p]['sigma_ou']) for p in pairs], dtype=torch.float32, device=self.device).unsqueeze(1)
        
        tp_dist = torch.tensor([max(1e-4, ou_params_dict[p].get('tp_dist', 0.015)) for p in pairs], dtype=torch.float32, device=self.device).unsqueeze(1)
        sl_dist = torch.tensor([max(1e-4, ou_params_dict[p].get('sl_dist', 0.010)) for p in pairs], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Step size: 1 minute (60s)
        dt = 60.0 / 86400.0
        sqrt_dt = math.sqrt(dt)

        curr = s0.expand(N, num_paths).clone()  # (N, 100000)

        long_tp = s0 + tp_dist
        long_sl = s0 - sl_dist
        short_tp = s0 - tp_dist
        short_sl = s0 + sl_dist

        long_hits_tp = torch.zeros((N, num_paths), dtype=torch.bool, device=self.device)
        long_hits_sl = torch.zeros((N, num_paths), dtype=torch.bool, device=self.device)
        short_hits_tp = torch.zeros((N, num_paths), dtype=torch.bool, device=self.device)
        short_hits_sl = torch.zeros((N, num_paths), dtype=torch.bool, device=self.device)

        for _ in range(steps):
            dW = torch.randn((N, num_paths), device=self.device) * sqrt_dt
            ds = theta * (mu - curr) * dt + sigma * dW
            curr = curr + ds

            # Track boundary hits
            long_hits_tp |= (curr >= long_tp)
            long_hits_sl |= (curr <= long_sl) & ~long_hits_tp

            short_hits_tp |= (curr <= short_tp)
            short_hits_sl |= (curr >= short_sl) & ~short_hits_tp

        results = {}
        for i, p in enumerate(pairs):
            # Directional Probability of Profit from 100K simulated paths
            rev_long = float((curr[i] > s0[i]).float().mean().item())
            rev_short = float((curr[i] < s0[i]).float().mean().item())

            results[p] = {
                'pop_long': float(rev_long),
                'pop_short': float(rev_short),
                'gpu_paths_evaluated': num_paths,
                'device': self.device_name
            }

        return results
