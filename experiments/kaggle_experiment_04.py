# %% [markdown]
# # Experiment 04: Discrete Physics HMM RRXOR (Training & Probing)
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
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

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
SEQ_LEN = 100 # 10 full tokens of history (10 * 25 steps)
SAMPLES_PER_EPOCH = 128000
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3

VOCAB_SIZE = 181 # Bins from -90.0 to +90.0 degrees with 0.1 degree resolution
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 1
MAX_SEQ_LEN = 1024
OUTPUT_DIR = "outputs-04"

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

# %% [markdown]
# ## 2. Initialization

# %%
print(f"Initializing PendulumIterableDataset (Batch Size: {BATCH_SIZE})...")
dataset = PendulumIterableDataset(batch_size=BATCH_SIZE, hmm_type='rrxor')
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

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

total_yields_per_epoch = max(1, (SAMPLES_PER_EPOCH // (500 // SEQ_LEN)) // BATCH_SIZE)
train_yields = int(total_yields_per_epoch * 0.8)
test_yields = total_yields_per_epoch - train_yields

data_iter = iter(dataloader)

# %% [markdown]
# ## 3. Training Loop

# %%
for ep in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    train_loss = 0
    train_chunks = 0
    
    train_pbar = tqdm(range(train_yields), desc=f"Epoch {ep+1}/{EPOCHS} [Train]", leave=False)
    for _ in train_pbar:
        batch = next(data_iter)
        
        # Discretize theta to bins (0.1 degree resolution)
        theta = batch["theta"]
        theta_deg = theta * (180.0 / np.pi)
        theta_bins = torch.clamp(torch.round(theta_deg), -90, 90).long() + 90
        theta_bins = theta_bins.to(device)
        
        coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
        
        for chunk in coord_chunks:
            inputs = chunk[:, :-K]
            targets = chunk[:, K:]
            
            optimizer.zero_grad()
            out = model(inputs) # Shape: [batch, seq_len, 181]
            
            # CrossEntropy requires shape (N, C) for outputs, and (N) for targets
            loss = criterion(out.reshape(-1, VOCAB_SIZE), targets.flatten())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_chunks += 1
            
        train_pbar.set_postfix(ce_loss=f"{train_loss / max(1, train_chunks):.4f}")
            
    # --- TESTING PHASE ---
    model.eval()
    test_loss = 0
    test_chunks = 0
    
    test_pbar = tqdm(range(test_yields), desc=f"Epoch {ep+1}/{EPOCHS} [Test ]", leave=False)
    with torch.no_grad():
        for _ in test_pbar:
            batch = next(data_iter)
            
            theta = batch["theta"]
            theta_deg = theta * (180.0 / np.pi)
            theta_bins = torch.clamp(torch.round(theta_deg), -90, 90).long() + 90
            theta_bins = theta_bins.to(device)
            
            coord_chunks = chunk_batch(theta_bins, SEQ_LEN + K)
            for chunk in coord_chunks:
                inputs = chunk[:, :-K]
                targets = chunk[:, K:]
                
                out = model(inputs)
                loss = criterion(out.reshape(-1, VOCAB_SIZE), targets.flatten())
                
                test_loss += loss.item()
                test_chunks += 1
                
            test_pbar.set_postfix(ce_loss=f"{test_loss / max(1, test_chunks):.4f}")
            
    print(f"Epoch {ep+1}/{EPOCHS} | Train CE: {train_loss / max(1, train_chunks):.4f} | Test CE: {test_loss / max(1, test_chunks):.4f}")

# Save Model
model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
torch.save(model_state, os.path.join(OUTPUT_DIR, "hmm_discrete_model.pt"))
print(f"Model saved to {OUTPUT_DIR}/hmm_discrete_model.pt")


# %% [markdown]
# ## 4. Load Model & Extract Activations

# %%
print(f"Initializing PendulumIterableDataset for Probing...")
dataset = PendulumIterableDataset(batch_size=BATCH_SIZE, hmm_type='rrxor')
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

probe_model.load_state_dict(torch.load(model_path, map_location=device))
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
    for _ in range(5): # Collect 5 large batches for the probe dataset
        batch = next(data_iter)
        
        theta = batch["theta"]
        theta_deg = theta * (180.0 / np.pi)
        theta_bins = torch.clamp(torch.round(theta_deg), -90, 90).long() + 90
        theta_bins = theta_bins.to(device)
        
        velocities = batch["velocity"]
        beliefs = batch["belief_state"]
        hmm_states = batch["hmm_state"]
        
        K_val = params.DEFAULT_N
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
            
            # Concatenate activations from all layers
            act_list_layers = [activations[f'layer{i}'].cpu().numpy() for i in range(NUM_LAYERS)]
            act = np.concatenate(act_list_layers, axis=-1)
            
            EFF_EMBED_DIM = NUM_LAYERS * EMBED_DIM
            
            # 1. Step-by-step feature extraction (for Velocity)
            acts_list.append(act.reshape(-1, EFF_EMBED_DIM))
            vels_list.append(vel_targets.reshape(-1))
            
            # 2. Per-token feature extraction (for Belief State and Hidden State)
            act_tokens = act.reshape(act.shape[0], SEQ_LEN // K_val, K_val, EFF_EMBED_DIM)
            
            # Subsample belief targets (first step of each token, note RRXOR is 5D)
            belief_tokens = belief_targets.reshape(belief_targets.shape[0], SEQ_LEN // K_val, K_val, 5)[:, :, 0, :]
            state_tokens = state_targets.reshape(state_targets.shape[0], SEQ_LEN // K_val, K_val)[:, :, 0]
            
            # --- FRACTAL SHARPNESS FIX ---
            # 1. Concat the 10 size sequence across the 4 layers (5120 dims)
            act_tokens_concat = act_tokens.reshape(act.shape[0], SEQ_LEN // K_val, K_val * EFF_EMBED_DIM)
            
            # 2. Slice off the first 5 tokens to remove the corner blobs!
            act_tokens_fractal = act_tokens_concat
            belief_tokens_fractal = belief_tokens
            state_tokens_fractal = state_tokens
            
            # Extract for Probing
            belief_acts_list.append(act_tokens_fractal.reshape(-1, K_val * EFF_EMBED_DIM))
            belief_list.append(belief_tokens_fractal.reshape(-1, 5))
            states_list.append(state_tokens_fractal.reshape(-1))

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
print("Fitting Ridge Regression for Belief State Geometry (Mean-Pooled Tokens)...")
reg_belief = Ridge(alpha=1.0).fit(X_belief_probe, y_belief)
y_belief_pred = reg_belief.predict(X_belief_probe)

# Enforce simplex bounds
y_belief_pred = np.clip(y_belief_pred, 0, 1)
y_belief_pred = y_belief_pred / np.sum(y_belief_pred, axis=-1, keepdims=True)

score_belief = reg_belief.score(X_belief_probe, y_belief)

# Project to 2D using RRXOR's symmetric projection
hmm_proc = dataset.hmm
y_2d_true = hmm_proc.project_symmetric(y_belief)
y_2d_pred = hmm_proc.project_symmetric(y_belief_pred)

dominant_state = np.argmax(y_belief, axis=1)

# --- 1. True Belief State Plot ---
plt.figure(figsize=(10, 8))
scatter_true = plt.scatter(y_2d_true[:, 0], y_2d_true[:, 1], c=dominant_state, cmap='Set1', alpha=0.4, s=2)
plt.title('RRXOR True Belief State Geometry\\nColored by Dominant State')
plt.axis('equal')
plt.colorbar(scatter_true, ticks=range(5), label='Dominant Hidden State')
plt.savefig(os.path.join(OUTPUT_DIR, 'discrete_true_belief_state_geometry.png'), dpi=300)
plt.show()

# --- 2. Probed Belief State Plot ---
plt.figure(figsize=(10, 8))

# True Belief State Backbone
plt.scatter(y_2d_true[:, 0], y_2d_true[:, 1], color='gray', alpha=0.02, s=1)

# Plot Predicted points
scatter = plt.scatter(y_2d_pred[:, 0], y_2d_pred[:, 1], c=dominant_state, cmap='Set1', alpha=0.4, s=2)

plt.title(f'RRXOR Belief State Probing (R²: {score_belief:.3f})\\nColored by Dominant State')
plt.axis('equal')
plt.colorbar(scatter, ticks=range(5), label='Dominant Hidden State')
plt.savefig(os.path.join(OUTPUT_DIR, 'discrete_probe_belief_state_geometry.png'), dpi=300)
plt.show()

print(f"Discrete Belief State Probing R² Score: {score_belief:.4f}")
