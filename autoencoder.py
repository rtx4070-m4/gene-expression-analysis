"""
autoencoder.py — Neural Autoencoder for gene expression latent space learning.

SIMPLE EXPLANATION
------------------
An autoencoder is like a bottleneck. Imagine trying to describe 500 genes
using only 8 numbers — the autoencoder learns which 8 numbers capture the
most important biological information. The first half (encoder) compresses;
the second half (decoder) reconstructs. If the bottleneck can reconstruct
the data accurately, those 8 numbers (the "latent space") are a rich,
compact summary of the gene expression profile.

Unlike PCA (which finds linear combinations), the autoencoder can learn
NONLINEAR compressions — capturing complex cancer biology that PCA misses.

ARCHITECTURE
------------
Input(500) → Dense(256,ReLU) → Dense(128,ReLU) → Latent(2-32)
                → Dense(128,ReLU) → Dense(256,ReLU) → Output(500)

ENCODER → BOTTLENECK (latent space) → DECODER

TECHNICAL EXPLANATION
---------------------
Autoencoder minimises reconstruction loss:
  L = (1/N) Σ ||x_i - f_θ(g_φ(x_i))||²
where:
  g_φ = encoder (φ = encoder weights)
  f_θ = decoder (θ = decoder weights)
  f_θ(g_φ(x)) = reconstruction

Trained by backpropagation with Adam optimiser.

VARIATIONAL AUTOENCODER (VAE) extension:
  Instead of a point z, the encoder outputs μ and σ (mean/variance).
  Adds KL regularisation: KL(q(z|x)||N(0,I))
  Enables generative sampling from the latent space.

Implementation uses a layered MLP with manual backprop via numpy
(production systems should use PyTorch/TensorFlow for GPU acceleration).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _PCA


# ── Activation functions ──────────────────────────────────────────────────────
def relu(x):       return np.maximum(0, x)
def relu_grad(x):  return (x > 0).astype(float)
def sigmoid(x):    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def linear(x):     return x
def linear_grad(x): return np.ones_like(x)


class DenseLayer:
    """Single fully-connected layer with He initialisation."""
    def __init__(self, in_dim, out_dim, activation='relu', rng=None):
        rng = rng or np.random.RandomState(42)
        scale = np.sqrt(2.0 / in_dim)                   # He init
        self.W = rng.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.act     = activation
        self.cache   = {}
        # Adam state
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t  = 0

    def forward(self, X):
        Z = X @ self.W + self.b
        if self.act == 'relu':
            A = relu(Z)
        elif self.act == 'sigmoid':
            A = sigmoid(Z)
        else:
            A = linear(Z)
        self.cache = {'X': X, 'Z': Z, 'A': A}
        return A

    def backward(self, dA):
        Z = self.cache['Z']; X = self.cache['X']
        if self.act == 'relu':
            dZ = dA * relu_grad(Z)
        elif self.act == 'sigmoid':
            s = sigmoid(Z)
            dZ = dA * s * (1 - s)
        else:
            dZ = dA
        n   = X.shape[0]
        self.dW = X.T @ dZ / n
        self.db = dZ.mean(axis=0)
        return dZ @ self.W.T

    def adam_update(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for param, grad, m, v, key in [
            (self.W, self.dW, self.mW, self.vW, 'W'),
            (self.b, self.db, self.mb, self.vb, 'b'),
        ]:
            m[:] = beta1 * m + (1-beta1) * grad
            v[:] = beta2 * v + (1-beta2) * grad**2
            m_hat = m / (1 - beta1**self.t)
            v_hat = v / (1 - beta2**self.t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)


class GeneAutoencoder:
    """
    Deep autoencoder for gene expression latent space extraction.
    
    Architecture (default, latent_dim=8):
      Encoder: 500 → 256 → 128 → 8  (ReLU activations)
      Decoder: 8   → 128 → 256 → 500 (ReLU/Linear)
    
    Training: mini-batch SGD with Adam optimiser, MSE loss.
    """

    def __init__(self,
                 input_dim: int = 500,
                 hidden_dims: tuple = (256, 128),
                 latent_dim: int = 8,
                 learning_rate: float = 5e-3,
                 n_epochs: int = 300,
                 batch_size: int = 32,
                 pca_preprocess: int = None,
                 random_state: int = 42):
        self.input_dim      = input_dim
        self.hidden_dims    = hidden_dims
        self.latent_dim     = latent_dim
        self.lr             = learning_rate
        self.n_epochs       = n_epochs
        self.batch_size     = batch_size
        self.pca_preprocess = pca_preprocess
        self.random_state   = random_state
        self.rng            = np.random.RandomState(random_state)
        self.history_       = {'loss': [], 'val_loss': []}
        self.scaler_        = StandardScaler()
        self.pca_pre_       = None
        self._build()

    def _build(self):
        """Construct encoder and decoder layers."""
        rng = self.rng
        # Effective input dim after optional PCA pre-reduction
        in_d = self.pca_preprocess if self.pca_preprocess else self.input_dim

        # Encoder layers
        self.enc_layers = []
        prev = in_d
        for h in self.hidden_dims:
            self.enc_layers.append(DenseLayer(prev, h, 'relu', rng))
            prev = h
        self.enc_layers.append(DenseLayer(prev, self.latent_dim, 'linear', rng))

        # Decoder layers (mirror of encoder)
        self.dec_layers = []
        prev = self.latent_dim
        for h in reversed(self.hidden_dims):
            self.dec_layers.append(DenseLayer(prev, h, 'relu', rng))
            prev = h
        self.dec_layers.append(DenseLayer(prev, in_d, 'linear', rng))

    def _encode(self, X):
        out = X
        for layer in self.enc_layers:
            out = layer.forward(out)
        return out

    def _decode(self, Z):
        out = Z
        for layer in self.dec_layers:
            out = layer.forward(out)
        return out

    def _forward(self, X):
        Z    = self._encode(X)
        Xhat = self._decode(Z)
        return Z, Xhat

    def _backward(self, X, Xhat):
        """MSE loss backward pass."""
        n   = X.shape[0]
        dL  = 2 * (Xhat - X) / n           # dMSE/dXhat
        # Decoder backward
        grad = dL
        for layer in reversed(self.dec_layers):
            grad = layer.backward(grad)
        # Encoder backward
        for layer in reversed(self.enc_layers):
            grad = layer.backward(grad)

    def _adam_step(self):
        for layer in self.enc_layers + self.dec_layers:
            layer.adam_update(lr=self.lr)

    def fit(self, X: np.ndarray, validation_split: float = 0.1,
            verbose: bool = True) -> "GeneAutoencoder":
        # Pre-process
        Xs = self.scaler_.fit_transform(X)
        if self.pca_preprocess:
            self.pca_pre_ = _PCA(n_components=self.pca_preprocess,
                                  random_state=self.random_state)
            Xs = self.pca_pre_.fit_transform(Xs)

        n         = Xs.shape[0]
        n_val     = max(1, int(n * validation_split))
        idx       = self.rng.permutation(n)
        X_val     = Xs[idx[:n_val]]
        X_train   = Xs[idx[n_val:]]
        n_train   = X_train.shape[0]

        for epoch in range(self.n_epochs):
            # Shuffle
            perm = self.rng.permutation(n_train)
            X_tr = X_train[perm]
            # Mini-batch
            batch_losses = []
            for start in range(0, n_train, self.batch_size):
                Xb = X_tr[start:start+self.batch_size]
                _, Xhat = self._forward(Xb)
                loss = float(np.mean((Xb - Xhat)**2))
                batch_losses.append(loss)
                self._backward(Xb, Xhat)
                self._adam_step()
            # Validation
            _, Xhat_val = self._forward(X_val)
            val_loss    = float(np.mean((X_val - Xhat_val)**2))
            train_loss  = float(np.mean(batch_losses))
            self.history_['loss'].append(train_loss)
            self.history_['val_loss'].append(val_loss)
            if verbose and (epoch % 50 == 0 or epoch == self.n_epochs-1):
                print(f"  Epoch {epoch+1:4d}/{self.n_epochs}  "
                      f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Extract latent space representation."""
        Xs = self.scaler_.transform(X)
        if self.pca_pre_:
            Xs = self.pca_pre_.transform(Xs)
        return self._encode(Xs)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        return self._decode(Z)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Xs   = self.scaler_.transform(X)
        if self.pca_pre_:
            Xs = self.pca_pre_.transform(Xs)
        _, Xhat = self._forward(Xs)
        return Xhat

    def reconstruction_error(self, X: np.ndarray) -> float:
        Xs = self.scaler_.transform(X)
        if self.pca_pre_:
            Xs = self.pca_pre_.transform(Xs)
        _, Xhat = self._forward(Xs)
        return float(np.mean((Xs - Xhat)**2))

    def latent_2d(self, X: np.ndarray) -> np.ndarray:
        """
        If latent_dim > 2, apply t-SNE to the latent space for 2D visualisation.
        If latent_dim == 2, return directly.
        """
        Z = self.encode(X)
        if Z.shape[1] == 2:
            return Z
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, random_state=self.random_state,
                    init='pca', max_iter=500).fit_transform(Z)
