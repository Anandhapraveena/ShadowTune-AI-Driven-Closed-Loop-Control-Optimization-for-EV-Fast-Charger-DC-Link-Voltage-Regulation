import math
from typing import List, Tuple, Optional


class MatrixOps:
    """Lightweight, robust matrix operations for GP covariance computations."""

    @staticmethod
    def mat_vec_mul(A: List[List[float]], x: List[float]) -> List[float]:
        n = len(A)
        m = len(x)
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            row = A[i]
            for j in range(m):
                s += row[j] * x[j]
            out[i] = s
        return out

    @staticmethod
    def dot(x: List[float], y: List[float]) -> float:
        return sum(a * b for a, b in zip(x, y))

    @staticmethod
    def cholesky(A: List[List[float]], jitter: float = 1e-6) -> List[List[float]]:
        """Compute lower triangular Cholesky factor L such that A + jitter*I = L * L^T."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] + jitter - s
                    if val <= 0.0:
                        val = 1e-8  # fallback regularization
                    L[i][j] = math.sqrt(val)
                else:
                    if L[j][j] == 0.0:
                        L[i][j] = 0.0
                    else:
                        L[i][j] = (A[i][j] - s) / L[j][j]
        return L

    @staticmethod
    def solve_lower(L: List[List[float]], b: List[float]) -> List[float]:
        """Solve L * y = b via forward substitution."""
        n = len(L)
        y = [0.0] * n
        for i in range(n):
            s = sum(L[i][j] * y[j] for j in range(i))
            denom = L[i][i] if abs(L[i][i]) > 1e-12 else 1e-12
            y[i] = (b[i] - s) / denom
        return y

    @staticmethod
    def solve_upper(U: List[List[float]], b: List[float]) -> List[float]:
        """Solve U * x = b via backward substitution (where U is upper triangular)."""
        n = len(U)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(U[i][j] * x[j] for j in range(i + 1, n))
            denom = U[i][i] if abs(U[i][i]) > 1e-12 else 1e-12
            x[i] = (b[i] - s) / denom
        return x

    @staticmethod
    def solve_cholesky(L: List[List[float]], b: List[float]) -> List[float]:
        """Solve (L * L^T) * x = b."""
        n = len(L)
        y = MatrixOps.solve_lower(L, b)
        # L^T is upper triangular
        LT = [[L[j][i] for j in range(n)] for i in range(n)]
        return MatrixOps.solve_upper(LT, y)


class GaussianProcessRegressor:
    """
    Gaussian Process Regressor with Automatic Relevance Determination (ARD) kernel.
    Models mapping from (Kp, Ki) -> DC-link voltage performance metric.
    """

    def __init__(
        self,
        length_scales: Tuple[float, float] = (1.5, 15.0),
        signal_variance: float = 1.0,
        noise_variance: float = 1e-3,
        jitter: float = 1e-5,
    ):
        self.length_scales = list(length_scales)  # [ell_Kp, ell_Ki]
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.jitter = jitter

        self.X_train: List[List[float]] = []
        self.y_train: List[float] = []

        # Cached Cholesky factor and weights
        self.L: Optional[List[List[float]]] = None
        self.alpha: Optional[List[float]] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def _kernel_single(self, x1: List[float], x2: List[float]) -> float:
        """Squared Exponential kernel with ARD length scales."""
        d_sq = 0.0
        for i in range(len(x1)):
            ell = self.length_scales[i] if i < len(self.length_scales) else 1.0
            diff = (x1[i] - x2[i]) / ell
            d_sq += diff * diff
        return self.signal_variance * math.exp(-0.5 * d_sq)

    def fit(self, X: List[List[float]], y: List[float]):
        """Fit GP model to training gain pairs X=[[Kp, Ki], ...] and targets y."""
        self.X_train = [list(x) for x in X]
        n = len(y)
        if n == 0:
            return

        # Normalize targets for numerical stability
        self.y_mean = sum(y) / n
        var = sum((v - self.y_mean) ** 2 for v in y) / max(1, n - 1)
        self.y_std = math.sqrt(var) if var > 1e-8 else 1.0
        self.y_train = [(v - self.y_mean) / self.y_std for v in y]

        # Compute Gram matrix K
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                val = self._kernel_single(self.X_train[i], self.X_train[j])
                if i == j:
                    val += self.noise_variance
                K[i][j] = val
                K[j][i] = val

        self.L = MatrixOps.cholesky(K, jitter=self.jitter)
        self.alpha = MatrixOps.solve_cholesky(self.L, self.y_train)

    def predict(self, x: List[float]) -> Tuple[float, float]:
        """
        Predict mean and standard deviation (uncertainty) at candidate x = [Kp, Ki].
        Returns: (predicted_mean, predicted_std)
        """
        if not self.X_train or self.L is None or self.alpha is None:
            return 0.0, math.sqrt(self.signal_variance)

        n = len(self.X_train)
        k_star = [self._kernel_single(x, x_tr) for x_tr in self.X_train]

        # Predicted normalized mean: mu_norm = k_star^T * alpha
        mu_norm = MatrixOps.dot(k_star, self.alpha)

        # Predicted variance: v = L \ k_star; var_norm = k(x, x) - v^T * v
        v = MatrixOps.solve_lower(self.L, k_star)
        k_xx = self.signal_variance
        v_sq = MatrixOps.dot(v, v)
        var_norm = max(1e-9, k_xx - v_sq)

        # Unnormalize
        mu = mu_norm * self.y_std + self.y_mean
        std = math.sqrt(var_norm) * self.y_std

        return mu, std
