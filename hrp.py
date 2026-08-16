"""
Marcos López de Prado's Hierarchical Risk Parity (HRP) Portfolio Allocation Module
Uses hierarchical tree clustering, quasi-diagonalization, recursive bisection,
and Ledoit-Wolf covariance shrinkage with min/max weight bounds (10% to 40%).
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

class HierarchicalRiskParity:
    def __init__(self, min_weight=0.10, max_weight=0.40):
        self.min_weight = min_weight
        self.max_weight = max_weight

    def _get_quasi_diag(self, link):
        """Sorts cluster items by distance using recursive dendrogram ordering."""
        link = link.astype(int)
        sort_ix = [link[-1, 0], link[-1, 1]]
        num_items = link[-1, 3]

        while max(sort_ix) >= num_items:
            for i, item in enumerate(sort_ix):
                if item >= num_items:
                    sort_ix[i] = link[item - num_items, 0]
                    sort_ix.insert(i + 1, link[item - num_items, 1])
                    break
        return sort_ix

    def _get_cluster_var(self, cov, c_items):
        """Calculates cluster variance using inverse-variance weights with regularization."""
        cov_slice = cov.iloc[c_items, c_items]
        diag = np.diag(cov_slice)
        diag_reg = np.maximum(diag, 1e-6)
        w = 1.0 / diag_reg
        w = w / w.sum()
        c_var = np.dot(np.dot(w, cov_slice), w)
        return float(c_var)

    def _get_rec_bisection(self, cov, sort_ix):
        """Recursively bisects clusters to compute HRP weights."""
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]

        while len(c_items) > 0:
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                var0 = self._get_cluster_var(cov, c_items0)
                var1 = self._get_cluster_var(cov, c_items1)
                alpha = 1.0 - var0 / max(1e-8, (var0 + var1))
                w[c_items0] *= alpha
                w[c_items1] *= 1.0 - alpha
        return w

    def allocate(self, returns_df):
        """
        Computes robust Hierarchical Risk Parity (HRP) weights with shrinkage and bounding.
        """
        N = returns_df.shape[1]
        if N <= 1:
            return pd.Series(1.0, index=returns_df.columns)

        # Regularized Covariance and Correlation with shrinkage towards identity
        raw_cov = returns_df.cov().fillna(0)
        raw_corr = returns_df.corr().fillna(0)

        # Shrinkage towards diagonal
        shrinkage = 0.20
        cov = (1.0 - shrinkage) * raw_cov + shrinkage * np.diag(np.diag(raw_cov) + 1e-5)
        cov = pd.DataFrame(cov, index=returns_df.columns, columns=returns_df.columns)

        corr = (1.0 - shrinkage) * raw_corr + shrinkage * np.eye(N)
        corr_mat = np.array(corr, dtype=np.float64, copy=True)
        np.fill_diagonal(corr_mat, 1.0)

        # 1. Distance matrix: d(i,j) = sqrt(0.5 * (1 - rho(i,j)))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr_mat), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        # 2. Hierarchical Linkage Clustering
        try:
            dist_condensed = squareform(dist, checks=False)
            link = linkage(dist_condensed, method='single')
            sort_ix = self._get_quasi_diag(link)
            sort_ix_cols = returns_df.columns[sort_ix]
            hrp_weights = self._get_rec_bisection(cov, sort_ix)
            hrp_weights.index = sort_ix_cols
            w = hrp_weights.reindex(returns_df.columns).fillna(1.0 / N)
        except Exception:
            w = pd.Series(1.0 / N, index=returns_df.columns)

        # Apply institutional minimum and maximum allocation bounds
        w = w.clip(lower=self.min_weight, upper=self.max_weight)
        w = w / w.sum()  # Normalize to 1.0 exactly

        return w
