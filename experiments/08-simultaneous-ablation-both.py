import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    sys.path.append(os.getcwd())
    
from models.architecture import TokenTransformer
from physics.dataset import PendulumIterableDataset
from physics import params

# --- Toggles ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "experiments/outputs-08"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
SEQ_LEN = 250
BATCH_SIZE = 512
VOCAB_SIZE = params.VOCAB_SIZE
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 1
MAX_SEQ_LEN = 1024
K = params.DEFAULT_N

def chunk_batch(batch_tensor, seq_len):
    total_time = batch_tensor.size(1)
    chunks = []
    for start_idx in range(0, total_time - seq_len + 1, seq_len):
        chunk = batch_tensor[:, start_idx : start_idx + seq_len]
        chunks.append(chunk)
    return chunks

def main():
    print(f"Using device: {device}")
    required_m = max(params.DEFAULT_M, (SEQ_LEN + K + params.DEFAULT_N - 1) // params.DEFAULT_N)
    dataset = PendulumIterableDataset(batch_size=BATCH_SIZE, m=required_m)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    data_iter = iter(dataloader)

    model_path = "experiments/outputs-03/Mess3_theta_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return

    probe_model = TokenTransformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        num_layers=NUM_LAYERS, 
        num_heads=NUM_HEADS, 
        max_seq_len=MAX_SEQ_LEN
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

    handles = []
    for i in range(NUM_LAYERS):
        handles.append(probe_model.blocks[i].register_forward_hook(get_activation(f'layer{i}')))

    # 1. Collect data for probing
    print("Collecting data for probing...")
    acts_list = {i: [] for i in range(NUM_LAYERS)}
    vels_list = []
    beliefs_list = []

    with torch.no_grad():
        for _ in tqdm(range(10), desc="Collecting Batches"):
            batch = next(data_iter)
            theta_bins = params.discretize_theta(batch["theta"]).to(device)
            velocities = batch["velocity"]
            beliefs = batch["belief_state"]
            
            coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
            vel_chunks = chunk_batch(velocities, SEQ_LEN + K)
            belief_chunks = chunk_batch(beliefs, SEQ_LEN + K)
            
            for c_idx in range(len(coord_chunks)):
                inputs = coord_chunks[c_idx][:, :-K]
                vel_targets = vel_chunks[c_idx][:, :-K].numpy()
                belief_targets = belief_chunks[c_idx][:, :-K, :].numpy()
                
                probe_model(inputs)
                
                for i in range(NUM_LAYERS):
                    acts_list[i].append(activations[f'layer{i}'].cpu().numpy())
                
                vels_list.append(vel_targets)
                beliefs_list.append(belief_targets)

    # Prepare datasets for probing
    X_train_layers = {i: np.concatenate(acts_list[i], axis=0) for i in range(NUM_LAYERS)}
    y_vel_train = np.concatenate(vels_list, axis=0)
    y_belief_train = np.concatenate(beliefs_list, axis=0)

    # Flatten for velocity (token-wise)
    X_vel_train = {i: X_train_layers[i].reshape(-1, EMBED_DIM) for i in range(NUM_LAYERS)}
    y_vel_train_flat = y_vel_train.reshape(-1, 1)

    # For belief state (concatenated block-wise)
    X_belief_train = {}
    for i in range(NUM_LAYERS):
        act_blocks = X_train_layers[i].reshape(-1, SEQ_LEN // K, K, EMBED_DIM)
        X_belief_train[i] = act_blocks.reshape(-1, K * EMBED_DIM)
        
    y_belief_train_blocks = y_belief_train.reshape(-1, SEQ_LEN // K, K, 3)[:, :, 0, :].reshape(-1, 3)

    # Train probes and get projection matrices
    print("Training probes and computing projection matrices...")
    vel_projections = {}
    belief_projections = {}

    print("Using sklearn.linear_model.Ridge for probing...")
    for i in range(NUM_LAYERS):
        # Velocity Probing
        reg_vel = Ridge(alpha=1.0).fit(X_vel_train[i], y_vel_train_flat)
        C_vel = reg_vel.coef_
        if C_vel.ndim == 1:
            C_vel = C_vel.reshape(1, -1)
        inv_term = np.linalg.inv(C_vel @ C_vel.T + 1e-8 * np.eye(C_vel.shape[0]))
        vel_projections[i] = C_vel.T @ inv_term @ C_vel
        
        # Belief State Probing
        reg_belief = Ridge(alpha=1.0).fit(X_belief_train[i], y_belief_train_blocks)
        C_belief = reg_belief.coef_
        if C_belief.ndim == 1:
            C_belief = C_belief.reshape(1, -1)
        inv_term = np.linalg.inv(C_belief @ C_belief.T + 1e-8 * np.eye(C_belief.shape[0]))
        belief_projections[i] = C_belief.T @ inv_term @ C_belief
        
        print(f"Layer {i} Vel R2: {reg_vel.score(X_vel_train[i], y_vel_train_flat):.3f}, Belief R2: {reg_belief.score(X_belief_train[i], y_belief_train_blocks):.3f}")

    # Remove hooks
    for handle in handles:
        handle.remove()

    # 2. Define custom forward pass with simultaneous ablation
    def simultaneous_ablated_forward(model, x, proj_matrices, feature_type):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)
        h = model.token_embed(x) + model.pos_embed(pos)
        h = model.drop(h)
        
        for i, block in enumerate(model.blocks):
            h = block(h)
            
            # Ablate feature from the output of EVERY layer
            proj_tensor = torch.tensor(proj_matrices[i], dtype=h.dtype, device=h.device)
            if feature_type == "vel":
                delta = h @ proj_tensor
                h = h - delta
            elif feature_type == "belief":
                h_grouped = h.reshape(b, t // K, K * EMBED_DIM)
                delta = h_grouped @ proj_tensor
                h = h - delta.reshape(b, t, EMBED_DIM)
                
        h = model.ln_f(h)
        return model.lm_head(h)

    # 3. Evaluate Ablation
    print("Evaluating Simultaneous Ablation...")
    criterion = nn.CrossEntropyLoss()
    
    # Collect a fixed test set
    test_data = []
    with torch.no_grad():
        for _ in range(10):
            batch = next(data_iter)
            test_data.append(batch)
            
    def evaluate_ablation(feature_type=None):
        total_loss = 0
        total_chunks = 0
        with torch.no_grad():
            for batch in test_data:
                theta_bins = params.discretize_theta(batch["theta"]).to(device)
                velocities = batch["velocity"]
                beliefs = batch["belief_state"]
                
                coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
                vel_chunks = chunk_batch(velocities, SEQ_LEN + K)
                belief_chunks = chunk_batch(beliefs, SEQ_LEN + K)
                
                for c_idx in range(len(coord_chunks)):
                    inputs = coord_chunks[c_idx][:, :-K]
                    targets = coord_chunks[c_idx][:, K:]
                    
                    if feature_type is None:
                        with torch.amp.autocast('cuda'):
                            out = probe_model(inputs)
                            loss = criterion(out.reshape(-1, VOCAB_SIZE), targets.flatten())
                    else:
                        proj_matrices = vel_projections if feature_type == "vel" else belief_projections
                        with torch.amp.autocast('cuda'):
                            out = simultaneous_ablated_forward(probe_model, inputs, proj_matrices, feature_type)
                            loss = criterion(out.reshape(-1, VOCAB_SIZE), targets.flatten())
                            
                    total_loss += loss.item()
                    total_chunks += 1
        return total_loss / total_chunks

    baseline_loss = evaluate_ablation(feature_type=None)
    print(f"Baseline Loss: {baseline_loss:.4f}")

    simultaneous_vel_loss = evaluate_ablation(feature_type="vel")
    print(f"Simultaneous Velocity Ablation Loss: {simultaneous_vel_loss:.4f}")

    simultaneous_belief_loss = evaluate_ablation(feature_type="belief")
    print(f"Simultaneous Belief Ablation Loss: {simultaneous_belief_loss:.4f}")

    # 4. Plot Results
    categories = ['Baseline', 'Simultaneous\nVelocity Ablation', 'Simultaneous\nBelief Ablation']
    losses = [baseline_loss, simultaneous_vel_loss, simultaneous_belief_loss]
    
    plt.figure(figsize=(8, 6))
    plt.bar(categories, losses, color=['gray', 'red', 'blue'])
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Impact of Simultaneous Feature Ablation across All Layers')
    for i, v in enumerate(losses):
        plt.text(i, v + 0.05, f"{v:.4f}", ha='center')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'simultaneous_ablation.png'), dpi=300)
    print(f"Plot saved to {os.path.join(OUTPUT_DIR, 'simultaneous_ablation.png')}")

if __name__ == "__main__":
    main()
