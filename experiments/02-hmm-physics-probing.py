import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Add root directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.architecture import ContinuousTransformer
from physics.dataset import PendulumIterableDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
SEQ_LEN = 50
SAMPLES_PER_EPOCH = 16000
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 1
MAX_SEQ_LEN = 1024
OUTPUT_DIR = "experiments/outputs-02"

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# We use global variable to hook into model activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def project_simplex_2d(belief):
    """
    Projects a 3D simplex belief state (b1+b2+b3=1) onto a 2D equilateral triangle.
    Uses standard barycentric coordinates.
    """
    # Define vertices of an equilateral triangle
    A = np.array([0, 1])
    B = np.array([-np.sqrt(3)/2, -0.5])
    C = np.array([np.sqrt(3)/2, -0.5])
    
    # Calculate weighted average based on belief probabilities
    x = belief[:, 0] * A[0] + belief[:, 1] * B[0] + belief[:, 2] * C[0]
    y = belief[:, 0] * A[1] + belief[:, 1] * B[1] + belief[:, 2] * C[1]
    
    return np.stack([x, y], axis=-1)

def chunk_batch(batch_tensor, seq_len):
    """
    Splits a tensor of shape [batch, total_time, ...] into 
    multiple tensors of shape [batch, seq_len, ...].
    """
    total_time = batch_tensor.size(1)
    chunks = []
    for start_idx in range(0, total_time - seq_len, seq_len):
        chunk = batch_tensor[:, start_idx : start_idx + seq_len]
        chunks.append(chunk)
    return chunks

def probe_velocity(acts, vels):
    X = np.concatenate(acts, axis=0)
    y = np.concatenate(vels, axis=0)
    
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    score = reg.score(X, y)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.1, color='blue', s=2)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    plt.xlabel('Ground Truth Velocity Magnitude')
    plt.ylabel('Probed Predicted Velocity')
    plt.title(f'Velocity Probing (R²: {score:.3f})')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'probe_velocity.png'), dpi=300)
    plt.close()
    
    print(f"Velocity Probing R² Score: {score:.4f}")

def probe_belief_state(acts, beliefs):
    X = np.concatenate(acts, axis=0)
    y = np.concatenate(beliefs, axis=0)  # Shape: [N, 3]
    
    # Train multi-output linear regression
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Enforce simplex bounds for the prediction (Optional but helps visualization)
    y_pred = np.clip(y_pred, 0, 1)
    y_pred = y_pred / np.sum(y_pred, axis=-1, keepdims=True)
    
    # Calculate R^2 score across all dimensions
    score = reg.score(X, y)
    
    # Project to 2D for visualization
    y_2d_true = project_simplex_2d(y)
    y_2d_pred = project_simplex_2d(y_pred)
    
    plt.figure(figsize=(10, 8))
    # We plot the ground truth simplex background
    plt.scatter(y_2d_true[:, 0], y_2d_true[:, 1], color='gray', alpha=0.05, s=1, label="Ground Truth")
    # We plot the predicted points
    plt.scatter(y_2d_pred[:, 0], y_2d_pred[:, 1], color='red', alpha=0.1, s=2, label="Predicted")
    
    plt.title(f'Belief State Geometry Probing (R²: {score:.3f})')
    plt.legend()
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, 'probe_belief_state_geometry.png'), dpi=300)
    plt.close()
    
    print(f"Belief State Probing R² Score: {score:.4f}")

def main():
    print(f"Initializing PendulumIterableDataset (Batch Size: {BATCH_SIZE})...")
    dataset = PendulumIterableDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    model = ContinuousTransformer(
        input_dim=2, 
        embed_dim=EMBED_DIM, 
        num_layers=NUM_LAYERS, 
        num_heads=NUM_HEADS, 
        max_seq_len=MAX_SEQ_LEN
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Calculate how many DataLoader batches we need to reach SAMPLES_PER_EPOCH
    # We also have to account for chunking. 
    # If total_time = 500, we get roughly 500 // 50 = 10 chunks per simulated sequence.
    # So 1 sequence yields 10 chunks.
    # To get 16000 chunks, we need 1600 sequences.
    # At BATCH_SIZE=64 sequences per dataloader yield, we need 1600 / 64 = 25 yields per epoch.
    total_yields_per_epoch = max(1, (SAMPLES_PER_EPOCH // (500 // SEQ_LEN)) // BATCH_SIZE)
    train_yields = int(total_yields_per_epoch * 0.8)
    test_yields = total_yields_per_epoch - train_yields
    
    print(f"Training for {EPOCHS} epochs, approx {SAMPLES_PER_EPOCH} sample chunks per epoch...")
    print(f"80/20 Split -> Train Yields: {train_yields}, Test Yields: {test_yields}")
    
    data_iter = iter(dataloader)
    
    for ep in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        train_chunks = 0
        
        train_pbar = tqdm(range(train_yields), desc=f"Epoch {ep+1}/{EPOCHS} [Train]", leave=False)
        for _ in train_pbar:
            batch = next(data_iter)
            x = batch["x"].unsqueeze(-1)
            y = batch["y"].unsqueeze(-1)
            coords = torch.cat([x, y], dim=-1).to(device)
            
            coord_chunks = chunk_batch(coords, SEQ_LEN + 1)
            for chunk in coord_chunks:
                inputs = chunk[:, :-1, :]
                targets = chunk[:, 1:, :]
                
                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_chunks += 1
                
            train_pbar.set_postfix(mse=f"{train_loss / max(1, train_chunks):.5f}")
                
        # --- TESTING PHASE ---
        model.eval()
        test_loss = 0
        test_chunks = 0
        
        test_pbar = tqdm(range(test_yields), desc=f"Epoch {ep+1}/{EPOCHS} [Test ]", leave=False)
        with torch.no_grad():
            for _ in test_pbar:
                batch = next(data_iter)
                x = batch["x"].unsqueeze(-1)
                y = batch["y"].unsqueeze(-1)
                coords = torch.cat([x, y], dim=-1).to(device)
                
                coord_chunks = chunk_batch(coords, SEQ_LEN + 1)
                for chunk in coord_chunks:
                    inputs = chunk[:, :-1, :]
                    targets = chunk[:, 1:, :]
                    
                    out = model(inputs)
                    loss = criterion(out, targets)
                    
                    test_loss += loss.item()
                    test_chunks += 1
                    
                test_pbar.set_postfix(mse=f"{test_loss / max(1, test_chunks):.5f}")
                
        print(f"Epoch {ep+1}/{EPOCHS} | Train MSE: {train_loss / max(1, train_chunks):.6f} | Test MSE: {test_loss / max(1, test_chunks):.6f}")
        
    # --- PROBING PHASE ---
    print("\nTraining complete. Initiating probing...")
    model.eval()
    
    # Register hook on the target layer
    # We probe layer 3 (index 2) as requested.
    handle = model.blocks[2].register_forward_hook(get_activation('layer3'))
    
    acts_list = []
    vels_list = []
    belief_list = []
    
    # Collect data for probing
    with torch.no_grad():
        for _ in range(5): # Collect 5 large batches for the probe dataset
            batch = next(data_iter)
            
            coords = torch.cat([batch["x"].unsqueeze(-1), batch["y"].unsqueeze(-1)], dim=-1).to(device)
            velocities = batch["velocity"]
            beliefs = batch["belief_state"]
            
            coord_chunks = chunk_batch(coords, SEQ_LEN + 1)
            vel_chunks = chunk_batch(velocities, SEQ_LEN + 1)
            belief_chunks = chunk_batch(beliefs, SEQ_LEN + 1)
            
            for c_idx in range(len(coord_chunks)):
                inputs = coord_chunks[c_idx][:, :-1, :]
                vel_targets = vel_chunks[c_idx][:, :-1].numpy()
                belief_targets = belief_chunks[c_idx][:, :-1, :].numpy()
                
                model(inputs)
                act = activations['layer3'].cpu().numpy()
                
                acts_list.append(act.reshape(-1, EMBED_DIM))
                vels_list.append(vel_targets.reshape(-1))
                belief_list.append(belief_targets.reshape(-1, 3))

    handle.remove()
    
    print("Fitting Linear Regression for Velocity...")
    probe_velocity(acts_list, vels_list)
    
    print("Fitting Linear Regression for Belief State Geometry...")
    probe_belief_state(acts_list, belief_list)
    
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "hmm_physics_model.pt"))
    print(f"All outputs and models saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
