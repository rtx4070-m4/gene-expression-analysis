"""
tsne.py — t-SNE dimensionality reduction for gene expression data.

SIMPLE EXPLANATION
------------------
Imagine you have 500 genes per sample, making each sample a point in
500-dimensional space. t-SNE asks: "Which samples look similar?"
It then places them on a 2D map so that similar samples end up close
together. Unlike PCA (which finds straight-line directions of variance),
t-SNE preserves LOCAL NEIGHBOURHOOD structure — great for finding
disease subtypes, even when clusters are non-linearly separated.

WHEN TO USE t-SNE vs PCA
  PCA   → Linear relationships, interpretable axes, preprocessing step
  t-SNE → Visualising clusters, complex nonlinear structure, final plot

TECHNICAL EXPLANATION
---------------------
t-SNE (van der Maaten & Hinton, 2008):
  1. Compute pairwise Gaussian similarities in high-d space:
       p_{ij} = exp(-||x_i-x_j||²/2σ²) / Σ exp(...)
  2. Compute pairwise t-distribution similarities in 2D:
       q_{ij} = (1+||y_i-y_j||²)^{-1} / Σ (1+||y_k-y_l||²)^{-1}
  3. Minimise KL divergence between P and Q via gradient descent:
       KL(P||Q) = Σ p_{ij} log(p_{ij}/q_{ij})
  The t-distribution in step 2 has heavier tails than Gaussian,
  preventing the "crowding problem" in low-d space.

KEY HYPERPARAMETER — perplexity:
  Controls effective neighbourhood size (≈ 2^entropy of P_i).
  Rule of thumb: perplexity ∈ [5, 50]; larger datasets need higher values.
  Typical range for genomics: 10–50.
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _PCA


class GeneTSNE:
    """t-SNE wrapper with PCA pre-reduction and perplexity grid search."""

    def __init__(self,
                 n_components: int = 2,
                 perplexity: float = 30.0,
                 learning_rate: float = 200.0,
                 n_iter: int = 1000,
                 pca_preprocess: int = 50,
                 random_state: int = 42):
        self.n_components    = n_components
        self.perplexity      = perplexity
        self.learning_rate   = learning_rate
        self.n_iter          = n_iter
        self.pca_preprocess  = pca_preprocess
        self.random_state    = random_state
        self.embedding_      = None
        self.kl_divergence_  = None

    # ── Fit-transform ─────────────────────────────────────────────────────────
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        PCA pre-reduce to `pca_preprocess` dims (speeds up t-SNE significantly),
        then run t-SNE.
        """
        # Pre-reduce with PCA if dimensionality is large
        if self.pca_preprocess and X.shape[1] > self.pca_preprocess:
            X_pre = _PCA(n_components=self.pca_preprocess,
                         random_state=self.random_state).fit_transform(X)
        else:
            X_pre = X

        tsne = TSNE(
            n_components   = self.n_components,
            perplexity     = self.perplexity,
            learning_rate  = self.learning_rate,
            max_iter       = self.n_iter,
            random_state   = self.random_state,
            init           = 'pca',
            method         = 'barnes_hut',
        )
        self.embedding_     = tsne.fit_transform(X_pre)
        self.kl_divergence_ = tsne.kl_divergence_
        return self.embedding_

    # ── Perplexity grid search ────────────────────────────────────────────────
    def perplexity_sweep(self, X: np.ndarray,
                         perplexities=(5, 10, 20, 30, 50),
                         n_iter: int = 500) -> dict:
        """
        Run t-SNE with multiple perplexity values.
        Returns dict {perplexity: (embedding, kl_divergence)}.
        
        Interpretation guide:
          Low perplexity  (5-10)  → emphasises very local structure,
                                    may over-fragment true clusters
          High perplexity (40-50) → more global view, clusters may merge
          Sweet spot      (20-35) → usually optimal for genomics (n≈100-500)
        """
        results = {}
        if self.pca_preprocess and X.shape[1] > self.pca_preprocess:
            X_pre = _PCA(n_components=self.pca_preprocess,
                         random_state=self.random_state).fit_transform(X)
        else:
            X_pre = X

        for p in perplexities:
            tsne = TSNE(n_components=2, perplexity=p, learning_rate=200,
                        max_iter=n_iter, random_state=self.random_state,
                        init='pca', method='barnes_hut')
            emb = tsne.fit_transform(X_pre)
            results[p] = (emb, tsne.kl_divergence_)
            print(f"  Perplexity={p:3d}  KL={tsne.kl_divergence_:.4f}")
        return results
