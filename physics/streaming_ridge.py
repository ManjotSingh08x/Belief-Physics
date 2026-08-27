import torch

class StreamingRidge:
    """
    A numerically stable, streaming implementation of Ridge Regression.
    Mathematically identical to sklearn.linear_model.Ridge(fit_intercept=True).
    """
    def __init__(self, x_dim, y_dim, device="cpu"):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.device = device
        self.N = 0
        
        # Accumulate in float64 to prevent numerical drift over millions of samples
        self.sum_x = torch.zeros(x_dim, device=device, dtype=torch.float64)
        self.sum_y = torch.zeros(y_dim, device=device, dtype=torch.float64)
        self.sum_xx = torch.zeros(x_dim, x_dim, device=device, dtype=torch.float64)
        self.sum_xy = torch.zeros(x_dim, y_dim, device=device, dtype=torch.float64)
        
    def update(self, X, y):
        """
        Stream a batch of data into the accumulators.
        X: (batch, x_dim)
        y: (batch, y_dim)
        """
        X = X.to(self.device, dtype=torch.float64)
        y = y.to(self.device, dtype=torch.float64)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
            
        self.N += X.shape[0]
        self.sum_x += X.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.sum_xx += X.T @ X
        self.sum_xy += X.T @ y
        
    def solve(self, alpha=1.0):
        """
        Solves the ridge regression problem (X_centered^T X_centered + alpha * I) beta = X_centered^T y_centered.
        Uses pseudo-inverse for numerical stability on highly collinear/ill-conditioned data.
        Returns: beta (x_dim, y_dim) as float32
        """
        if self.N == 0:
            return None
            
        mu_x = self.sum_x / self.N
        mu_y = self.sum_y / self.N
        
        # Centered covariance matrices
        XtX = self.sum_xx - self.N * torch.outer(mu_x, mu_x)
        Xty = self.sum_xy - self.N * torch.outer(mu_x, mu_y)
        
        # L2 Regularization (matches sklearn's un-scaled alpha penalty)
        I = torch.eye(self.x_dim, device=self.device, dtype=torch.float64)
        A = XtX + alpha * I
        
        # Use pseudo-inverse (pinv) to handle ill-conditioned matrices robustly,
        # preventing explosive, unstable weights (acting as sklearn's safe fallback).
        beta = torch.linalg.pinv(A) @ Xty
        
        return beta.float(), mu_x.float(), mu_y.float()

class StreamingEvaluator:
    """
    Evaluates streaming R2 score for models fit using StreamingRidge.
    """
    def __init__(self, y_dim, device="cpu"):
        self.y_dim = y_dim
        self.device = device
        self.N = 0
        self.sum_y = torch.zeros(y_dim, device=device, dtype=torch.float64)
        self.sum_yy = torch.zeros(y_dim, device=device, dtype=torch.float64)
        self.sum_res = torch.zeros(y_dim, device=device, dtype=torch.float64)

    def update(self, X, y, beta, mu_x, mu_y_train):
        if beta is None:
            return
        X = X.to(self.device, dtype=torch.float64)
        y = y.to(self.device, dtype=torch.float64)
        beta = beta.to(self.device, dtype=torch.float64)
        mu_x = mu_x.to(self.device, dtype=torch.float64)
        mu_y_train = mu_y_train.to(self.device, dtype=torch.float64)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
            
        y_pred = (X - mu_x) @ beta + mu_y_train
        
        self.N += X.shape[0]
        self.sum_y += y.sum(dim=0)
        self.sum_yy += (y ** 2).sum(dim=0)
        self.sum_res += ((y - y_pred) ** 2).sum(dim=0)

    def get_r2(self):
        if self.N == 0:
            return 0.0
        mu_y = self.sum_y / self.N
        ss_tot = self.sum_yy - self.N * (mu_y ** 2)
        # Prevent division by zero
        r2 = 1.0 - (self.sum_res / torch.clamp(ss_tot, min=1e-8))
        return r2.mean().item()
