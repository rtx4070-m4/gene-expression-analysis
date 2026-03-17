"""
pca.py — PCA baseline for gene expression dimensionality reduction.
Provides fit/transform, variance explained, loadings, and 2D projection.
"""
import numpy as np
from sklearn.decomposition import PCA as _PCA
from sklearn.preprocessing import StandardScaler


class GeneExpressionPCA:
    """Wrapper around sklearn PCA with gene-expression-specific helpers."""

    def __init__(self, n_components: int = 50, scale: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.scaler_ = StandardScaler() if scale else None
        self.pca_ = _PCA(n_components=n_components, random_state=42)
        self.fitted_ = False

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray) -> "GeneExpressionPCA":
        Xs = self.scaler_.fit_transform(X) if self.scale else X
        self.pca_.fit(Xs)
        self.fitted_ = True
        return self

    # ── Transform ─────────────────────────────────────────────────────────────
    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted_, "Call fit() first."
        Xs = self.scaler_.transform(X) if self.scale else X
        return self.pca_.transform(Xs)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def variance_explained(self) -> np.ndarray:
        return self.pca_.explained_variance_ratio_

    @property
    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.pca_.explained_variance_ratio_)

    def n_components_for_variance(self, threshold: float = 0.90) -> int:
        return int(np.argmax(self.cumulative_variance >= threshold)) + 1

    def top_gene_loadings(self, pc: int = 0, n: int = 20,
                          gene_names=None) -> list:
        """Return top-n genes by absolute loading on principal component `pc`."""
        loadings = self.pca_.components_[pc]
        idx = np.argsort(np.abs(loadings))[::-1][:n]
        names = gene_names if gene_names is not None else [f"Gene_{i}" for i in idx]
        return [(names[i] if i < len(names) else f"Gene_{i}", loadings[i])
                for i in idx]

    def reconstruction_error(self, X: np.ndarray) -> float:
        Xs = self.scaler_.transform(X) if self.scale else X
        Z  = self.pca_.transform(Xs)
        Xr = self.pca_.inverse_transform(Z)
        return float(np.mean((Xs - Xr) ** 2))
