import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    sys.path.append(os.getcwd())

from models.architecture import TokenTransformer
from physics.dataset import PendulumIterableDataset
from physics import params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
SEQ_LEN = 250
BATCH_SIZE = 256
VOCAB_SIZE = 1801
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 1
MAX_SEQ_LEN = 1024
OUTPUT_DIR = "experiments/outputs-04"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K = params.DEFAULT_N
EFF_EMBED_DIM = NUM_LAYERS * EMBED_DIM
BELIEF_DIM = K * EFF_EMBED_DIM

def chunk_batch(batch_tensor, seq_len):
    total_time = batch_tensor.size(1)
    chunks = []
    for start_idx in range(0, total_time - seq_len, seq_len):
        chunk = batch_tensor[:, start_idx : start_idx + seq_len]
        chunks.append(chunk)
    return chunks

def worker_init_fn(worker_id):
    import time
    np.random.seed((int(time.time() * 1000) + worker_id) % (2**32 - 1))

def project_simplex_2d(belief):
    if isinstance(belief, torch.Tensor):
        belief = belief.cpu().numpy()
    A = np.array([0, 1])
    B = np.array([-np.sqrt(3)/2, -0.5])
    C = np.array([np.sqrt(3)/2, -0.5])
    x = belief[:, 0] * A[0] + belief[:, 1] * B[0] + belief[:, 2] * C[0]
    y = belief[:, 0] * A[1] + belief[:, 1] * B[1] + belief[:, 2] * C[1]
    return np.stack([x, y], axis=-1)

class StreamingRidge:
    def __init__(self, in_features, out_features, device='cuda'):
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        self.XtX = torch.zeros((in_features, in_features), device=device, dtype=torch.float32)
        self.Xty = torch.zeros((in_features, out_features), device=device, dtype=torch.float32)
        self.sum_X = torch.zeros(in_features, device=device, dtype=torch.float32)
        self.sum_y = torch.zeros(out_features, device=device, dtype=torch.float32)
        self.count = 0
        
    def partial_fit(self, X, y):
        if y.ndim == 1:
            y = y.unsqueeze(1)
        
        X = X.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32)
        
        self.count += X.shape[0]
        self.sum_X += X.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        
        self.XtX.addmm_(X.T, X)
        self.Xty.addmm_(X.T, y)
        
    def solve_and_pca(self, alphas, k_pca=2):
        mu_x = self.sum_X / self.count
        mu_y = self.sum_y / self.count
        
        self.XtX -= self.count * torch.outer(mu_x, mu_x)
        self.Xty -= self.count * torch.outer(mu_x, mu_y)
        
        print(f"[{self.in_features}D] Computing top {k_pca} Principal Components using SVD...")
        U, S, V = torch.svd_lowrank(self.XtX, q=k_pca)
        pca_components = U 
        
        print(f"[{self.in_features}D] Solving Ridge Regression for alphas: {alphas}...")
        var_x = torch.diag(self.XtX) / self.count
        std_x = torch.sqrt(var_x + 1e-8)
        
        self.XtX /= std_x.unsqueeze(0)
        self.XtX /= std_x.unsqueeze(1)
        self.Xty /= std_x.unsqueeze(1)
        
        I = torch.eye(self.in_features, device=self.device, dtype=torch.float32)
        
        results = {}
        for alpha in alphas:
            beta_std = torch.linalg.solve(self.XtX + alpha * I, self.Xty)
            beta = beta_std / std_x.unsqueeze(1)
            intercept = mu_y - (mu_x.unsqueeze(1) * beta).sum(dim=0)
            results[alpha] = {'beta': beta, 'intercept': intercept}
            
        del self.XtX
        del self.Xty
        torch.cuda.empty_cache()
        
        return results, pca_components, mu_x

def evaluate_r2(model_dict, X, y):
    if y.ndim == 1:
        y = y.unsqueeze(1)
    
    y_pred = X @ model_dict['beta'] + model_dict['intercept']
    ss_res = torch.sum((y - y_pred) ** 2)
    ss_tot = torch.sum((y - y.mean(dim=0)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2.item(), y_pred

def main():
    print(f"Initializing PendulumIterableDataset (Batch Size: {BATCH_SIZE})...")
    dataset = PendulumIterableDataset(batch_size=BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, worker_init_fn=worker_init_fn, prefetch_factor=2)
    data_iter = iter(dataloader)

    model_path = "experiments/outputs-03/hmm_discrete_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return

    probe_model = TokenTransformer(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS, 
        num_heads=NUM_HEADS, max_seq_len=MAX_SEQ_LEN
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    probe_model.load_state_dict(clean_state_dict)
    probe_model.eval()

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    handles = [probe_model.blocks[i].register_forward_hook(get_activation(f'layer{i}')) for i in range(NUM_LAYERS)]

    # Initialize Streaming Regressors
    vel_regressor = StreamingRidge(in_features=EFF_EMBED_DIM, out_features=1, device=device)
    belief_regressor = StreamingRidge(in_features=BELIEF_DIM, out_features=3, device=device)

    # --- PHASE 1: STREAMING TRAIN ---
    TRAIN_BATCHES = 50
    print(f"\\n--- PHASE 1: Training ({TRAIN_BATCHES} Batches) ---")
    for _ in tqdm(range(TRAIN_BATCHES), desc="Accumulating Covariances"):
        batch = next(data_iter)
        
        theta_bins = torch.clamp(torch.round(batch["theta"] * (180.0 / np.pi)), -90, 90).long() + 90
        theta_bins = theta_bins.to(device)
        velocities = batch["velocity"].to(device)
        beliefs = batch["belief_state"].to(device)
        
        coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
        vel_chunks = chunk_batch(velocities, SEQ_LEN + K)
        belief_chunks = chunk_batch(beliefs, SEQ_LEN + K)
        
        for c_idx in range(len(coord_chunks)):
            inputs = coord_chunks[c_idx][:, :-K]
            vel_targets = vel_chunks[c_idx][:, :-K].flatten()
            belief_targets = belief_chunks[c_idx][:, :-K, :].reshape(-1, SEQ_LEN//K, K, 3)[:, :, 0, :].reshape(-1, 3)
            
            with torch.no_grad():
                probe_model(inputs)
            
            act_list = [activations[f'layer{i}'] for i in range(NUM_LAYERS)]
            act = torch.cat(act_list, dim=-1) # (batch, seq_len, 512)
            
            # Velocity features (flattened)
            X_vel = act.reshape(-1, EFF_EMBED_DIM)
            vel_regressor.partial_fit(X_vel, vel_targets)
            
            # Belief features (per-token concatenated window)
            act_tokens = act.reshape(act.shape[0], SEQ_LEN // K, K, EFF_EMBED_DIM)
            X_belief = act_tokens.reshape(-1, K * EFF_EMBED_DIM)
            belief_regressor.partial_fit(X_belief, belief_targets)

    # --- PHASE 2: SOLVE & VALIDATION ---
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    print(f"\\n--- PHASE 2: Solving & Cross-Validation ---")
    vel_models, _, _ = vel_regressor.solve_and_pca(alphas, k_pca=2)
    belief_models, pca_components, pca_mu = belief_regressor.solve_and_pca(alphas, k_pca=2)
    
    VAL_BATCHES = 10
    vel_r2_scores = {a: [] for a in alphas}
    belief_r2_scores = {a: [] for a in alphas}
    
    for _ in tqdm(range(VAL_BATCHES), desc="Validating Alphas"):
        batch = next(data_iter)
        theta_bins = torch.clamp(torch.round(batch["theta"] * (180.0 / np.pi)), -90, 90).long() + 90
        theta_bins = theta_bins.to(device)
        
        coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
        vel_chunks = chunk_batch(batch["velocity"].to(device), SEQ_LEN + K)
        belief_chunks = chunk_batch(batch["belief_state"].to(device), SEQ_LEN + K)
        
        for c_idx in range(len(coord_chunks)):
            inputs = coord_chunks[c_idx][:, :-K]
            vel_targets = vel_chunks[c_idx][:, :-K].flatten()
            belief_targets = belief_chunks[c_idx][:, :-K, :].reshape(-1, SEQ_LEN//K, K, 3)[:, :, 0, :].reshape(-1, 3)
            
            with torch.no_grad():
                probe_model(inputs)
            act = torch.cat([activations[f'layer{i}'] for i in range(NUM_LAYERS)], dim=-1)
            X_vel = act.reshape(-1, EFF_EMBED_DIM)
            X_belief = act.reshape(act.shape[0], SEQ_LEN // K, K, EFF_EMBED_DIM).reshape(-1, BELIEF_DIM)
            
            for a in alphas:
                r2_v, _ = evaluate_r2(vel_models[a], X_vel, vel_targets)
                r2_b, _ = evaluate_r2(belief_models[a], X_belief, belief_targets)
                vel_r2_scores[a].append(r2_v)
                belief_r2_scores[a].append(r2_b)

    best_vel_alpha = max(alphas, key=lambda a: np.mean(vel_r2_scores[a]))
    best_belief_alpha = max(alphas, key=lambda a: np.mean(belief_r2_scores[a]))
    print(f"Best Velocity Alpha: {best_vel_alpha} (Val R2: {np.mean(vel_r2_scores[best_vel_alpha]):.4f})")
    print(f"Best Belief Alpha: {best_belief_alpha} (Val R2: {np.mean(belief_r2_scores[best_belief_alpha]):.4f})")
    
    best_vel_model = vel_models[best_vel_alpha]
    best_belief_model = belief_models[best_belief_alpha]

    # --- PHASE 3: TEST & SUBSPACE VISUALIZATION ---
    TEST_BATCHES = 10
    print(f"\\n--- PHASE 3: Testing ({TEST_BATCHES} Batches) ---")
    
    test_y_belief_true = []
    test_y_belief_pred = []
    test_pca_proj = []
    
    for _ in tqdm(range(TEST_BATCHES), desc="Testing Final Model"):
        batch = next(data_iter)
        theta_bins = torch.clamp(torch.round(batch["theta"] * (180.0 / np.pi)), -90, 90).long() + 90
        theta_bins = theta_bins.to(device)
        
        coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
        belief_chunks = chunk_batch(batch["belief_state"].to(device), SEQ_LEN + K)
        
        for c_idx in range(len(coord_chunks)):
            inputs = coord_chunks[c_idx][:, :-K]
            belief_targets = belief_chunks[c_idx][:, :-K, :].reshape(-1, SEQ_LEN//K, K, 3)[:, :, 0, :].reshape(-1, 3)
            
            with torch.no_grad():
                probe_model(inputs)
            act = torch.cat([activations[f'layer{i}'] for i in range(NUM_LAYERS)], dim=-1)
            X_belief = act.reshape(act.shape[0], SEQ_LEN // K, K, EFF_EMBED_DIM).reshape(-1, BELIEF_DIM)
            
            _, y_pred = evaluate_r2(best_belief_model, X_belief, belief_targets)
            
            # Project onto Subspace
            X_centered = X_belief - pca_mu
            pca_proj = X_centered @ pca_components # (N, 2)
            
            test_y_belief_true.append(belief_targets)
            test_y_belief_pred.append(y_pred)
            test_pca_proj.append(pca_proj)

    y_true = torch.cat(test_y_belief_true, dim=0)
    y_pred = torch.cat(test_y_belief_pred, dim=0)
    pca_proj = torch.cat(test_pca_proj, dim=0)
    
    # Final Test R2
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean(dim=0)) ** 2)
    test_r2 = 1 - (ss_res / ss_tot)
    print(f"FINAL TEST R2 (Belief State): {test_r2.item():.4f}")
    
    # Plotting
    print("Saving Subspace Analysis Plots...")
    y_true_np = y_true.cpu().numpy()
    pca_proj_np = pca_proj.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_proj_np[:, 0], pca_proj_np[:, 1], c=y_true_np, alpha=0.5, s=2)
    plt.title(f'Belief Subspace: Top 2 Principal Components\\nColored by True Belief State (R2: {test_r2.item():.3f})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, 'subspace_pca_projection.png'), dpi=300)
    
    # Project to 2D Simplex for prediction accuracy
    y_pred_np = torch.clamp(y_pred, 0, 1).cpu().numpy()
    y_pred_np = y_pred_np / np.sum(y_pred_np, axis=-1, keepdims=True)
    
    y_2d_true = project_simplex_2d(y_true_np)
    y_2d_pred = project_simplex_2d(y_pred_np)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_2d_true[:, 0], y_2d_true[:, 1], color='gray', alpha=0.02, s=1)
    plt.scatter(y_2d_pred[:, 0], y_2d_pred[:, 1], c=y_true_np, alpha=0.4, s=2)
    plt.title(f'Test Set: Belief State Predictions (R²: {test_r2.item():.3f})')
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, 'subspace_simplex_predictions.png'), dpi=300)
    
    for handle in handles:
        handle.remove()

if __name__ == '__main__':
    main()
