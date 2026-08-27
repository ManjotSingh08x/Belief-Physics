# %% [markdown]
# # Experiment 03: Discrete Physics HMM (Training & Probing)
# This notebook trains a `TokenTransformer` on a quantized physical pendulum dataset (theta discretized into 181 bins).
# By predicting discrete tokens using Cross-Entropy Loss, the model is mathematically forced to learn the full Bayesian Belief State Simplex!

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add root directory to path to import local modules
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'Belief-Physics'))
    sys.path.append('/kaggle/working/Belief-Physics')
    
    if os.path.exists("/kaggle/input"):
        import glob
        for path in glob.glob("/kaggle/input/*/"):
            sys.path.append(path)
        for path in glob.glob("/kaggle/input/*/*/"):
            sys.path.append(path)

from models.architecture import TokenTransformer
from physics.dataset import PendulumIterableDataset
from physics import params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# --- Hyperparameters ---
SEQ_LEN = 250 # 10 full tokens of history (10 * 25 steps)
SAMPLES_PER_EPOCH = 128000
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3

VOCAB_SIZE = params.VOCAB_SIZE
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 1
MAX_SEQ_LEN = 1024
OUTPUT_DIR = "experiments/outputs-03"

K = params.DEFAULT_N

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    A = np.array([0, 1])
    B = np.array([-np.sqrt(3)/2, -0.5])
    C = np.array([np.sqrt(3)/2, -0.5])
    x = belief[:, 0] * A[0] + belief[:, 1] * B[0] + belief[:, 2] * C[0]
    y = belief[:, 0] * A[1] + belief[:, 1] * B[1] + belief[:, 2] * C[1]
    return np.stack([x, y], axis=-1)

# %% [markdown]
# ## 2. Initialization

# %%
def main():

    print(f"Initializing PendulumIterableDataset (Batch Size: {BATCH_SIZE})...")
    dataset = PendulumIterableDataset(batch_size=BATCH_SIZE)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=4, 
        worker_init_fn=worker_init_fn,
        prefetch_factor=2
    )

    model = TokenTransformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        num_layers=NUM_LAYERS, 
        num_heads=NUM_HEADS, 
        max_seq_len=MAX_SEQ_LEN
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs in DataParallel mode!")
        model = nn.DataParallel(model)

    model = model.to(device)
    model = torch.compile(model) # Fuses GPU kernels for 20-30% speedup

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') # For Mixed Precision

    total_yields_per_epoch = max(1, (SAMPLES_PER_EPOCH // BATCH_SIZE))
    train_yields = int(total_yields_per_epoch * 0.8)
    test_yields = total_yields_per_epoch - train_yields

    data_iter = iter(dataloader)

    # %% [markdown]
    # ## 3. Training Loop

    # %%
    model_save_path = os.path.join(OUTPUT_DIR, "hmm_discrete_model.pt")
    
    if os.path.exists(model_save_path):
        print(f"Model already exists at {model_save_path}. Skipping training phase.")
    else:
        epoch_pbar = tqdm(range(EPOCHS), desc="Total Progress")
        for ep in epoch_pbar:
            # --- TRAINING PHASE ---
            model.train()
            train_loss = 0
            train_chunks = 0
            
            for _ in range(train_yields):
                batch = next(data_iter)
                
                # Discretize theta to bins
                theta = batch["theta"]
                theta_bins = params.discretize_theta(theta)
                theta_bins = theta_bins.to(device)
                
                coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
                
                for chunk in coord_chunks:
                    inputs = chunk[:, :-K]
                    targets = chunk[:, K:]
                    
                    optimizer.zero_grad()
                    
                    with torch.amp.autocast('cuda'):
                        out = model(inputs) # Shape: [batch, seq_len, 181]
                        
                        # CrossEntropy requires shape (N, C) for outputs, and (N) for targets
                        loss = criterion(out.reshape(-1, VOCAB_SIZE), targets.flatten())
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    train_chunks += 1
                    
                epoch_pbar.set_postfix(phase="Train", ep=ep+1, train_ce=f"{train_loss / max(1, train_chunks):.4f}")
                    
            # --- TESTING PHASE ---
            model.eval()
            test_loss = 0
            test_chunks = 0
            
            with torch.no_grad():
                for _ in range(test_yields):
                    batch = next(data_iter)
                    
                    theta = batch["theta"]
                    theta_bins = params.discretize_theta(theta)
                    theta_bins = theta_bins.to(device)
                    
                    coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
                    for chunk in coord_chunks:
                        inputs = chunk[:, :-K]
                        targets = chunk[:, K:]
                        
                        with torch.amp.autocast('cuda'):
                            out = model(inputs)
                            loss = criterion(out.reshape(-1, VOCAB_SIZE), targets.flatten())
                        
                        test_loss += loss.item()
                        test_chunks += 1
                        
                        epoch_pbar.set_postfix(phase="Test", ep=ep+1, train_ce=f"{train_loss / max(1, train_chunks):.4f}", test_ce=f"{test_loss / max(1, test_chunks):.4f}")
                    
            tqdm.write(f"Epoch {ep+1}/{EPOCHS} | Train CE: {train_loss / max(1, train_chunks):.4f} | Test CE: {test_loss / max(1, test_chunks):.4f}")
    
        # Save Model
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(model_state, model_save_path)
        print(f"Model saved to {model_save_path}")


    # %% [markdown]
    # ## 4. Load Model & Extract Activations

    # %%
    print(f"Initializing PendulumIterableDataset for Probing...")
    dataset = PendulumIterableDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    data_iter = iter(dataloader)

    model_path = os.path.join(OUTPUT_DIR, "hmm_discrete_model.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        
    # Initialize a CLEAN model to avoid DataParallel hook issues
    probe_model = TokenTransformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        num_layers=NUM_LAYERS, 
        num_heads=NUM_HEADS, 
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    # Remove the '_orig_mod.' prefix added by torch.compile
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    probe_model.load_state_dict(clean_state_dict)
    probe_model.eval()

    # Global variable to hook into model activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hook on all layers
    handles = []
    for i in range(NUM_LAYERS):
        handles.append(probe_model.blocks[i].register_forward_hook(get_activation(f'layer{i}')))
        
    acts_list = []
    vels_list = []

    belief_acts_list = []
    belief_list = []
    states_list = []

    print("Running forward passes to extract internal representations...")
    with torch.no_grad():
        for _ in range(15): # Collect 5 large batches for the probe dataset
            batch = next(data_iter)
            
            # Discretize theta to bins
            theta = batch["theta"]
            theta_bins = params.discretize_theta(theta)
            theta_bins = theta_bins.to(device)
            
            velocities = batch["velocity"]
            beliefs = batch["belief_state"]
            hmm_states = batch["hmm_state"]
            
            K_val = K
            coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K_val)
            vel_chunks = chunk_batch(velocities, SEQ_LEN + K_val)
            belief_chunks = chunk_batch(beliefs, SEQ_LEN + K_val)
            state_chunks = chunk_batch(hmm_states, SEQ_LEN + K_val)
            
            for c_idx in range(len(coord_chunks)):
                inputs = coord_chunks[c_idx][:, :-K_val]
                vel_targets = vel_chunks[c_idx][:, :-K_val].numpy()
                belief_targets = belief_chunks[c_idx][:, :-K_val, :].numpy()
                state_targets = state_chunks[c_idx][:, :-K_val].numpy()
                
                probe_model(inputs)

                act_list_layers = []
                for i in range(NUM_LAYERS) :
                    layer_act = activations[f'layer{i}'].cpu().numpy()
                    # Normalize across the embedding/feature dimension
                    layer_mean = layer_act.mean(axis=-1, keepdims=True)
                    layer_std = layer_act.std(axis=-1, keepdims=True) + 1e-8
                    layer_act_norm = (layer_act - layer_mean) / layer_std
                    act_list_layers.append(layer_act_norm)
                
                # Concatenate activations from all layers
                act_list_layers = [activations[f'layer{i}'].cpu().numpy() for i in range(NUM_LAYERS)]
                act = np.concatenate(act_list_layers, axis=-1)
                
                EFF_EMBED_DIM = NUM_LAYERS * EMBED_DIM
                
                # 1. Step-by-step feature extraction (for Velocity)
                # Flatten across batch and sequence length
                acts_list.append(act.reshape(-1, EFF_EMBED_DIM))
                vels_list.append(vel_targets.reshape(-1))
                
                # 2. Per-token feature extraction (for Belief State and Hidden State)
                act_tokens = act.reshape(act.shape[0], SEQ_LEN // params.DEFAULT_N, params.DEFAULT_N, EFF_EMBED_DIM)
                act_tokens_concat = act_tokens.reshape(act.shape[0], SEQ_LEN // params.DEFAULT_N, params.DEFAULT_N * EFF_EMBED_DIM)
                
                # Subsample belief targets (first step of each 10-step token)
                belief_tokens = belief_targets.reshape(belief_targets.shape[0], SEQ_LEN // params.DEFAULT_N, params.DEFAULT_N, 3)[:, :, 0, :]
                state_tokens = state_targets.reshape(state_targets.shape[0], SEQ_LEN // K, K)[:, :, 0]
                
                # Just extract, concat, and pass through
                belief_acts_list.append(act_tokens_concat.reshape(-1, params.DEFAULT_N * EFF_EMBED_DIM))
                belief_list.append(belief_tokens.reshape(-1, 3))
                states_list.append(state_tokens.reshape(-1))

    for handle in handles:
        handle.remove()

    X_probe = np.concatenate(acts_list, axis=0)
    y_vel = np.concatenate(vels_list, axis=0)

    X_belief_probe = np.concatenate(belief_acts_list, axis=0)
    y_belief = np.concatenate(belief_list, axis=0)
    y_states = np.concatenate(states_list, axis=0)

    # %% [markdown]
    # ### Probing: Velocity

    # %%
    X_probe = StandardScaler().fit_transform(X_probe)
    reg_vel = Ridge(alpha=1.0).fit(X_probe, y_vel)
    y_vel_pred = reg_vel.predict(X_probe)
    score_vel = reg_vel.score(X_probe, y_vel)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_vel, y_vel_pred, alpha=0.1, color='blue', s=2)
    plt.plot([y_vel.min(), y_vel.max()], [y_vel.min(), y_vel.max()], 'r--', linewidth=2)
    plt.xlabel('Ground Truth Velocity Magnitude')
    plt.ylabel('Probed Predicted Velocity')
    plt.title(f'Velocity Probing (R²: {score_vel:.3f})')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'discrete_probe_velocity.png'), dpi=300)
    plt.show()

    print(f"Velocity Probing R² Score: {score_vel:.4f}")

    # %% [markdown]
    # ### Probing: Belief State Geometry

    # %%
    print("Fitting Ridge Regression for Belief State Geometry (Concatenated Tokens)...")
    X_belief_probe = StandardScaler().fit_transform(X_belief_probe)
    reg_belief = Ridge(alpha=1.0).fit(X_belief_probe, y_belief)
    y_belief_pred = reg_belief.predict(X_belief_probe)

    # Enforce simplex bounds
    y_belief_pred = np.clip(y_belief_pred, 0, 1)
    y_belief_pred = y_belief_pred / np.sum(y_belief_pred, axis=-1, keepdims=True)

    score_belief = reg_belief.score(X_belief_probe, y_belief)

    # Project to 2D
    y_2d_true = project_simplex_2d(y_belief)
    y_2d_pred = project_simplex_2d(y_belief_pred)

    plt.figure(figsize=(10, 8))

    plt.scatter(y_2d_true[:, 0], y_2d_true[:, 1], color='gray', alpha=0.02, s=1)

    # Plot Predicted points, colored dynamically by the TRUE belief state (Continuous RGB)
    plt.scatter(y_2d_pred[:, 0], y_2d_pred[:, 1], c=y_belief, alpha=0.4, s=2)
    
    plt.title(f'Discrete Token Belief State Probing (R²: {score_belief:.3f})\\nColored by True Belief State')
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, 'discrete_probe_belief_state_geometry.png'), dpi=300)
    plt.show()

    print(f"Discrete Belief State Probing R² Score: {score_belief:.4f}")


if __name__ == '__main__':
    main()