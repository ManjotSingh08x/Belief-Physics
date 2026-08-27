import sys
import os
import torch
import numpy as np
from sklearn.linear_model import Ridge

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.streaming_ridge import StreamingRidge

def main():
    print("Testing StreamingRidge vs sklearn.linear_model.Ridge...")
    np.random.seed(42)
    
    # 1. Generate highly collinear (ill-conditioned) data 
    # This mimics neural network activations where features are highly correlated
    N = 10000
    features = 128
    targets = 3
    
    # Create base latent factors to force ill-conditioning
    latent = np.random.randn(N, 10)
    mixing = np.random.randn(10, features)
    X = latent @ mixing + 0.01 * np.random.randn(N, features) # Add slight noise
    
    # True weights and targets
    W = np.random.randn(features, targets)
    y = X @ W + np.random.randn(N, targets)
    
    print("Fitting sklearn Ridge...")
    # 2. Sklearn fit
    sk_ridge = Ridge(alpha=1.0, fit_intercept=True)
    sk_ridge.fit(X, y)
    sk_beta = sk_ridge.coef_.T # (features, targets)
    
    print("Fitting StreamingRidge in chunks...")
    # 3. StreamingRidge fit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream_ridge = StreamingRidge(features, targets, device=device)
    
    # Simulate streaming by passing in chunks (simulates our batch loop)
    chunk_size = 500
    for i in range(0, N, chunk_size):
        X_chunk = torch.tensor(X[i:i+chunk_size])
        y_chunk = torch.tensor(y[i:i+chunk_size])
        stream_ridge.update(X_chunk, y_chunk)
        
    sr_beta = stream_ridge.solve(alpha=1.0).cpu().numpy()
    
    # 4. Compare
    diff = np.max(np.abs(sk_beta - sr_beta))
    print(f"Maximum absolute difference in weights: {diff:.8f}")
    
    if np.allclose(sk_beta, sr_beta, atol=1e-4):
        print("✅ SUCCESS: StreamingRidge perfectly matches sklearn, even on ill-conditioned data!")
    else:
        print("❌ FAILURE: Weights do not match. Difference is too large.")

if __name__ == "__main__":
    main()
