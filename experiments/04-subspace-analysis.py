import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc

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
VOCAB_SIZE = params.VOCAB_SIZE
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 1
MAX_SEQ_LEN = 1024
OUTPUT_DIR = "experiments/outputs-04"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N = parans.DEFAULT_N
K = N  # K=10c
NUM_BLOCKS = (SEQ_LEN - N) // N + 1

TRAIN_BATCHES = 200
TEST_BATCHES = 30
ALPHA = 1.0
PEAK_T = 9

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

class StreamingRidge:
    def __init__(self, x_dim, y_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.N = 0
        self.sum_x = torch.zeros(x_dim, device=device, dtype=torch.float64)
        self.sum_y = torch.zeros(y_dim, device=device, dtype=torch.float64)
        self.sum_xx = torch.zeros(x_dim, x_dim, device=device, dtype=torch.float64)
        self.sum_xy = torch.zeros(x_dim, y_dim, device=device, dtype=torch.float64)
        self.sum_yy = torch.zeros(y_dim, device=device, dtype=torch.float64)
        
    def update(self, X, y):
        X = X.to(device, dtype=torch.float64)
        y = y.to(device, dtype=torch.float64)
        self.N += X.shape[0]
        self.sum_x += X.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.sum_xx += X.T @ X
        self.sum_xy += X.T @ y
        self.sum_yy += (y ** 2).sum(dim=0)
        
    def solve(self, alpha=1.0):
        if self.N == 0:
            return None, None, None
            
        mu_x = self.sum_x / self.N
        mu_y = self.sum_y / self.N
        
        XtX = self.sum_xx - self.N * torch.outer(mu_x, mu_x)
        Xty = self.sum_xy - self.N * torch.outer(mu_x, mu_y)
        
        I = torch.eye(self.x_dim, device=device, dtype=torch.float64)
        beta = torch.linalg.solve(XtX + alpha * self.N * I, Xty)
        
        return beta.float(), mu_x.float(), mu_y.float()

class StreamingEvaluator:
    def __init__(self, y_dim):
        self.y_dim = y_dim
        self.N = 0
        self.sum_y = torch.zeros(y_dim, device=device, dtype=torch.float64)
        self.sum_yy = torch.zeros(y_dim, device=device, dtype=torch.float64)
        self.sum_res = torch.zeros(y_dim, device=device, dtype=torch.float64)
        
    def update(self, X, y, beta, mu_x, mu_y_train):
        if beta is None:
            return
            
        X = X.to(device, dtype=torch.float64)
        y = y.to(device, dtype=torch.float64)
        beta = beta.to(device, dtype=torch.float64)
        mu_x = mu_x.to(device, dtype=torch.float64)
        mu_y_train = mu_y_train.to(device, dtype=torch.float64)
        
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
        r2 = 1.0 - (self.sum_res / ss_tot)
        return r2.mean().item()

def subspace_angle(beta_belief, beta_vel):
    if beta_belief is None or beta_vel is None:
        return 0.0
    Q, _ = torch.linalg.qr(beta_belief)
    v_norm = beta_vel / torch.norm(beta_vel)
    proj = Q.T @ v_norm
    cos_theta = torch.norm(proj)
    cos_theta = torch.clamp(cos_theta, 0.0, 1.0)
    angle_rad = torch.acos(cos_theta)
    return (angle_rad * 180.0 / np.pi).item()

def process_batches(model, data_iter, num_batches, mode="Train", accumulators=None):
    desc = f"{mode} Stream"
    
    # Pre-allocate indices
    b_indices = [b * N for b in range(NUM_BLOCKS)]
    
    for _ in tqdm(range(num_batches), desc=desc):
        batch = next(data_iter)
        theta_bins = params.discretize_theta(batch["theta"])
        theta_bins = theta_bins.to(device)
        
        coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
        vel_chunks = chunk_batch(batch["velocity"].to(device), SEQ_LEN + K)
        belief_chunks = chunk_batch(batch["belief_state"].to(device), SEQ_LEN + K)
        
        for c_idx in range(len(coord_chunks)):
            inputs = coord_chunks[c_idx][:, :-K]
            
            # Extract targets
            b_targets = belief_chunks[c_idx][:, b_indices, :].reshape(-1, 3)
            v_targets_start = vel_chunks[c_idx][:, b_indices].reshape(-1, 1)
            
            with torch.no_grad():
                model(inputs)
            
            # Cache activations to avoid repetitive dict lookups
            acts = [model.activations[f'layer{l}'] for l in range(NUM_LAYERS)]
            
            for t in range(N):
                indices = [b * N + t for b in range(NUM_BLOCKS)]
                
                # Extract the correct velocity target for this exact timestep
                v_targets_t = vel_chunks[c_idx][:, indices].reshape(-1, 1)
                
                # 1. Update L x T grids
                for l in range(NUM_LAYERS):
                    act_lt = acts[l][:, indices, :].reshape(-1, EMBED_DIM)
                    
                    if mode == "Train":
                        accumulators["lxt_b"][l][t].update(act_lt, b_targets)
                        accumulators["lxt_v"][l][t].update(act_lt, v_targets_t)
                    else:
                        beta_b, mux_b, muy_b = accumulators["lxt_b_weights"][l][t]
                        accumulators["lxt_b_eval"][l][t].update(act_lt, b_targets, beta_b, mux_b, muy_b)
                        
                        beta_v, mux_v, muy_v = accumulators["lxt_v_weights"][l][t]
                        accumulators["lxt_v_eval"][l][t].update(act_lt, v_targets_t, beta_v, mux_v, muy_v)
                
                # 2. Update T Sweep grid (layers concatenated)
                act_t_concat = torch.cat([acts[l][:, indices, :].reshape(-1, EMBED_DIM) for l in range(NUM_LAYERS)], dim=1)
                if mode == "Train":
                    accumulators["t_b"][t].update(act_t_concat, b_targets)
                    accumulators["t_v"][t].update(act_t_concat, v_targets_t)
                else:
                    beta_b, mux_b, muy_b = accumulators["t_b_weights"][t]
                    accumulators["t_b_eval"][t].update(act_t_concat, b_targets, beta_b, mux_b, muy_b)
                    
                    beta_v, mux_v, muy_v = accumulators["t_v_weights"][t]
                    accumulators["t_v_eval"][t].update(act_t_concat, v_targets_t, beta_v, mux_v, muy_v)
            
            # 3. Update L Sweep grid (timesteps concatenated)
            for l in range(NUM_LAYERS):
                act_l_concat = torch.cat([acts[l][:, [b * N + t for b in range(NUM_BLOCKS)], :].reshape(-1, EMBED_DIM) for t in range(N)], dim=1)
                if mode == "T            rain":
                    accumulators["l_b"][l].update(act_l_concat, b_targets)
                    accumulators["l_v"][l].update(act_l_concat, v_targets_start)
                else:
                    beta_b, mux_b, muy_b = accumulators["l_b_weights"][l]
                    accumulators["l_b_eval"][l].update(act_l_concat, b_targets, beta_b, mux_b, muy_b)
                    
                    beta_v, mux_v, muy_v = accumulators["l_v_weights"][l]
                    accumulators["l_v_eval"][l].update(act_l_concat, v_targets_start, beta_v, mux_v, muy_v)
        
        # Aggressive memory cleanup per batch
        del inputs, b_targets, v_targets_start, coord_chunks, vel_chunks, belief_chunks, theta_bins, batch, acts
        model.activations.clear()
        gc.collect()
        torch.cuda.empty_cache()

def main():
    model_path = "experiments/outputs-03/hmm_discrete_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")
        
    probe_model = TokenTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    probe_model.load_state_dict(clean_state_dict)
    probe_model.eval()
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook

    for i in range(NUM_LAYERS):
        probe_model.blocks[i].register_forward_hook(get_activation(f'layer{i}'))
    probe_model.activations = activations

    dataset = PendulumIterableDataset(batch_size=BATCH_SIZE)
    data_iter = iter(DataLoader(dataset, batch_size=BATCH_SIZE, worker_init_fn=worker_init_fn))

    print(f"\n--- PHASE 1: Training Stream ({TRAIN_BATCHES} Batches) ---")
    train_accums = {
        "lxt_b": [[StreamingRidge(EMBED_DIM, 3) for _ in range(N)] for _ in range(NUM_LAYERS)],
        "lxt_v": [[StreamingRidge(EMBED_DIM, 1) for _ in range(N)] for _ in range(NUM_LAYERS)],
        "t_b": [StreamingRidge(EMBED_DIM * NUM_LAYERS, 3) for _ in range(N)],
        "t_v": [StreamingRidge(EMBED_DIM * NUM_LAYERS, 1) for _ in range(N)],
        "l_b": [StreamingRidge(EMBED_DIM * N, 3) for _ in range(NUM_LAYERS)],
        "l_v": [StreamingRidge(EMBED_DIM * N, 1) for _ in range(NUM_LAYERS)],
    }
    
    process_batches(probe_model, data_iter, TRAIN_BATCHES, mode="Train", accumulators=train_accums)
    
    print("\n--- PHASE 2: Solving Regressions ---")
    solved_weights = {
        "lxt_b_weights": [[train_accums["lxt_b"][l][t].solve(ALPHA) for t in range(N)] for l in range(NUM_LAYERS)],
        "lxt_v_weights": [[train_accums["lxt_v"][l][t].solve(ALPHA) for t in range(N)] for l in range(NUM_LAYERS)],
        "t_b_weights": [train_accums["t_b"][t].solve(ALPHA) for t in range(N)],
        "t_v_weights": [train_accums["t_v"][t].solve(ALPHA) for t in range(N)],
        "l_b_weights": [train_accums["l_b"][l].solve(ALPHA) for l in range(NUM_LAYERS)],
        "l_v_weights": [train_accums["l_v"][l].solve(ALPHA) for l in range(NUM_LAYERS)],
    }
    
    # Compute angles instantly from solved betas
    angle_matrix = np.zeros((NUM_LAYERS, N))
    t_angle = np.zeros(N)
    l_angle = np.zeros(NUM_LAYERS)
    
    for l in range(NUM_LAYERS):
        for t in range(N):
            beta_b = solved_weights["lxt_b_weights"][l][t][0]
            beta_v = solved_weights["lxt_v_weights"][l][t][0]
            angle_matrix[l, t] = subspace_angle(beta_b, beta_v)
            
        beta_b_l = solved_weights["l_b_weights"][l][0]
        beta_v_l = solved_weights["l_v_weights"][l][0]
        l_angle[l] = subspace_angle(beta_b_l, beta_v_l)
            
    for t in range(N):
        beta_b = solved_weights["t_b_weights"][t][0]
        beta_v = solved_weights["t_v_weights"][t][0]
        t_angle[t] = subspace_angle(beta_b, beta_v)
    
    # Free up training accumulators
    del train_accums
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n--- PHASE 3: Testing Stream ({TEST_BATCHES} Batches) ---")
    test_accums = {
        **solved_weights,
        "lxt_b_eval": [[StreamingEvaluator(3) for _ in range(N)] for _ in range(NUM_LAYERS)],
        "lxt_v_eval": [[StreamingEvaluator(1) for _ in range(N)] for _ in range(NUM_LAYERS)],
        "t_b_eval": [StreamingEvaluator(3) for _ in range(N)],
        "t_v_eval": [StreamingEvaluator(1) for _ in range(N)],
        "l_b_eval": [StreamingEvaluator(3) for _ in range(NUM_LAYERS)],
        "l_v_eval": [StreamingEvaluator(1) for _ in range(NUM_LAYERS)],
    }
    
    process_batches(probe_model, data_iter, TEST_BATCHES, mode="Test", accumulators=test_accums)
    
    print("\n--- PHASE 4: Extracting Results ---")
    belief_r2_matrix = np.zeros((NUM_LAYERS, N))
    vel_r2_matrix = np.zeros((NUM_LAYERS, N))
    for l in range(NUM_LAYERS):
        for t in range(N):
            belief_r2_matrix[l, t] = test_accums["lxt_b_eval"][l][t].get_r2()
            vel_r2_matrix[l, t] = test_accums["lxt_v_eval"][l][t].get_r2()
            
    t_belief_r2 = np.zeros(N)
    t_vel_r2 = np.zeros(N)
    for t in range(N):
        t_belief_r2[t] = test_accums["t_b_eval"][t].get_r2()
        t_vel_r2[t] = test_accums["t_v_eval"][t].get_r2()
        
    l_belief_r2 = np.zeros(NUM_LAYERS)
    l_vel_r2 = np.zeros(NUM_LAYERS)
    for l in range(NUM_LAYERS):
        l_belief_r2[l] = test_accums["l_b_eval"][l].get_r2()
        l_vel_r2[l] = test_accums["l_v_eval"][l].get_r2()

    print("\n--- PHASE 5: Generating Plots ---")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    layer_labels = [f"Layer {i}" for i in range(NUM_LAYERS)]
    timestep_labels = [f"t={t}" for t in range(N)]
    
    # ROW 0: L x T Heatmaps
    sns.heatmap(belief_r2_matrix, ax=axes[0, 0], annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                yticklabels=layer_labels, xticklabels=timestep_labels)
    axes[0, 0].set_title("Belief Test $R^2$ (LxT)")
    axes[0, 0].set_xlabel("Relative Timestep")
    axes[0, 0].set_ylabel("Layer")

    sns.heatmap(vel_r2_matrix, ax=axes[0, 1], annot=True, fmt=".2f", cmap="magma", vmin=0, vmax=1,
                yticklabels=layer_labels, xticklabels=timestep_labels)
    axes[0, 1].set_title("Velocity Test $R^2$ (LxT)")
    axes[0, 1].set_xlabel("Relative Timestep")

    sns.heatmap(angle_matrix, ax=axes[0, 2], annot=True, fmt=".1f", cmap="coolwarm", vmin=0, vmax=90,
                yticklabels=layer_labels, xticklabels=timestep_labels)
    axes[0, 2].set_title("Subspace Angle (LxT)")
    axes[0, 2].set_xlabel("Relative Timestep")

    # ROW 1: T Sweep (Layers Concatenated)
    axes[1, 0].plot(range(N), t_belief_r2, marker='o', color='green')
    axes[1, 0].set_title("Belief Test $R^2$ (T-Sweep, 512D)")
    axes[1, 0].set_xlabel("Relative Timestep")
    axes[1, 0].set_ylim(0, 1)
    
    axes[1, 1].plot(range(N), t_vel_r2, marker='o', color='purple')
    axes[1, 1].set_title("Velocity Test $R^2$ (T-Sweep, 512D)")
    axes[1, 1].set_xlabel("Relative Timestep")
    axes[1, 1].set_ylim(0, 1)

    axes[1, 2].plot(range(N), t_angle, marker='o', color='red')
    axes[1, 2].set_title("Subspace Angle (T-Sweep)")
    axes[1, 2].set_xlabel("Relative Timestep")
    axes[1, 2].set_ylim(0, 90)

    # ROW 2: L Sweep (Timesteps Concatenated)
    axes[2, 0].plot(range(NUM_LAYERS), l_belief_r2, marker='o', color='blue')
    axes[2, 0].set_title(f"Belief Test $R^2$ (L-Sweep, {EMBED_DIM * N}D)")
    axes[2, 0].set_xlabel("Layer")
    axes[2, 0].set_xticks(range(NUM_LAYERS))
    axes[2, 0].set_ylim(0, 1)

    axes[2, 1].plot(range(NUM_LAYERS), l_vel_r2, marker='o', color='orange')
    axes[2, 1].set_title(f"Velocity Test $R^2$ (L-Sweep, {EMBED_DIM * N}D)")
    axes[2, 1].set_xlabel("Layer")
    axes[2, 1].set_xticks(range(NUM_LAYERS))
    axes[2, 1].set_ylim(0, 1)

    axes[2, 2].plot(range(NUM_LAYERS), l_angle, marker='o', color='brown')
    axes[2, 2].set_title(f"Subspace Angle (L-Sweep, {EMBED_DIM * N}D)")
    axes[2, 2].set_xlabel("Layer")
    axes[2, 2].set_xticks(range(NUM_LAYERS))
    axes[2, 2].set_ylim(0, 90)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'sweep_analysis.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved heatmaps to {output_path}")

if __name__ == "__main__":
    main()
