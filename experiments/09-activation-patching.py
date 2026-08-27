import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
OUTPUT_DIR = "experiments/outputs-09"
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
        
    # Remove hooks
    for handle in handles:
        handle.remove()

    # 2. Define custom patched forward pass
    def patched_forward(model, x_target, x_source, proj_matrix, patch_layer, feature_type):
        b, t = x_target.size()
        
        pos_target = torch.arange(0, t, dtype=torch.long, device=x_target.device)
        h_target = model.token_embed(x_target) + model.pos_embed(pos_target)
        h_target = model.drop(h_target)
        
        pos_source = torch.arange(0, t, dtype=torch.long, device=x_source.device)
        h_source = model.token_embed(x_source) + model.pos_embed(pos_source)
        h_source = model.drop(h_source)
        
        for i, block in enumerate(model.blocks):
            if i <= patch_layer:
                h_target = block(h_target)
                h_source = block(h_source)
                
                if i == patch_layer:
                    proj_tensor = torch.tensor(proj_matrix, dtype=h_target.dtype, device=h_target.device)
                    if feature_type == "vel":
                        c_target = h_target @ proj_tensor
                        c_source = h_source @ proj_tensor
                        h_target = h_target - c_target + c_source
                    elif feature_type == "belief":
                        h_target_grouped = h_target.reshape(b, t // K, K * EMBED_DIM)
                        h_source_grouped = h_source.reshape(b, t // K, K * EMBED_DIM)
                        
                        c_target = h_target_grouped @ proj_tensor
                        c_source = h_source_grouped @ proj_tensor
                        
                        delta = c_target - c_source
                        h_target = h_target - delta.reshape(b, t, EMBED_DIM)
            else:
                h_target = block(h_target)
                
        h_target = model.ln_f(h_target)
        return model.lm_head(h_target)

    # 3. Evaluate Patching
    print("Evaluating Subspace Activation Patching...")
    
    # Collect a fixed test set
    test_data = []
    with torch.no_grad():
        for _ in range(5):
            batch = next(data_iter)
            test_data.append(batch)
            
    def evaluate_patching(feature_type, patch_layer):
        total_kl = 0
        total_chunks = 0
        
        proj_matrices = vel_projections if feature_type == "vel" else belief_projections
        
        with torch.no_grad():
            for batch in test_data:
                theta_bins = params.discretize_theta(batch["theta"]).to(device)
                
                coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
                
                for c_idx in range(len(coord_chunks)):
                    x_target = coord_chunks[c_idx][:, :-K]
                    
                    # Create a source sequence by randomly rolling the batch dimension
                    x_source = torch.roll(x_target, shifts=1, dims=0)
                    
                    with torch.amp.autocast('cuda'):
                        # 1. Baseline predictions (original target)
                        orig_logits = probe_model(x_target)
                        orig_probs = F.softmax(orig_logits, dim=-1)
                        
                        # 2. Patched predictions
                        patched_logits = patched_forward(probe_model, x_target, x_source, proj_matrices[patch_layer], patch_layer, feature_type)
                        patched_log_probs = F.log_softmax(patched_logits, dim=-1)
                        
                        # 3. Compute KL Divergence (patched diverges from original)
                        # We use batchmean over the (B*T, VOCAB_SIZE) tensor
                        kl_div = F.kl_div(patched_log_probs.view(-1, VOCAB_SIZE), orig_probs.view(-1, VOCAB_SIZE), reduction='batchmean')
                        
                    total_kl += kl_div.item()
                    total_chunks += 1
                    
        return total_kl / total_chunks

    kl_vel_by_layer = []
    kl_belief_by_layer = []

    for layer in range(NUM_LAYERS):
        kl_vel = evaluate_patching("vel", layer)
        kl_belief = evaluate_patching("belief", layer)
        kl_vel_by_layer.append(kl_vel)
        kl_belief_by_layer.append(kl_belief)
        print(f"Layer {layer} | Vel KL Div: {kl_vel:.4f} | Belief KL Div: {kl_belief:.4f}")

    # 4. Plot Results
    layers = list(range(NUM_LAYERS))
    plt.figure(figsize=(8, 6))
    
    plt.plot(layers, kl_vel_by_layer, marker='o', label='Velocity Patched', color='red')
    plt.plot(layers, kl_belief_by_layer, marker='s', label='Belief State Patched', color='blue')
    
    plt.xlabel('Layer Patched')
    plt.ylabel('KL Divergence (Causal Effect Size)')
    plt.title('Causal Effect of Subspace Activation Patching')
    plt.xticks(layers)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'activation_patching.png'), dpi=300)
    print(f"Plot saved to {os.path.join(OUTPUT_DIR, 'activation_patching.png')}")

if __name__ == "__main__":
    main()
