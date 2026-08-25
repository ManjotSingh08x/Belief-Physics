import torch
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.dataset import PendulumIterableDataset
import matplotlib.pyplot as plt
from physics import params

def main():
    print("Initializing Pendulum Iterable Dataset...")
    dataset = PendulumIterableDataset()
    
    # Set batch_size=8 here to test the DataLoader pulling from the IterableDataset
    dataloader = DataLoader(dataset, batch_size=8)
    
    print("Fetching one batch from DataLoader...")
    batch = next(iter(dataloader))
    
    print("\n--- Batch Dimensions ---")
    for key, tensor in batch.items():
        print(f"{key}: {tensor.shape}  | dtype: {tensor.dtype}")
        
    print("\nExtracting single sequence for visualization...")
    # Get the very first sequence in the batch
    x_trajectory = batch["x"][0].numpy()
    y_trajectory = batch["y"][0].numpy()
    
    # Plot the trajectory and color it by time to show motion
    time_steps = list(range(len(x_trajectory)))
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(x_trajectory, y_trajectory, c=time_steps, cmap='viridis', s=15, alpha=0.7)
    plt.colorbar(scatter, label='Time Step')
    
    # Also plot the pivot point of the pendulum
    plt.scatter([0], [0], color='red', marker='P', s=200, label='Pivot Origin (0,0)')
    
    plt.title(f"Pendulum XY Trajectory\n(1 sequence from Dataset)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Keep aspect ratio square so the arc looks circular
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory_test.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nTrajectory visualization saved to: {save_path}")
    
    # --- Velocity Histogram Visualization ---
    print("Generating velocity histogram for the same sequence...")
    velocity_trajectory = batch["velocity"][0].numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(velocity_trajectory, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title("Pendulum Velocity Histogram\n(1 sequence from Dataset)")
    plt.xlabel("Velocity (rad/s)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    
    hist_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "velocity_histogram_test.png")
    plt.savefig(hist_save_path, dpi=300)
    print(f"Velocity histogram saved to: {hist_save_path}")
    
    # --- Theta (Token Bin) Histogram Visualization ---
    print("\nGenerating Theta (Token Bin) histogram to visualize exact vocabulary utilization...")
    
    # We take the theta trajectory from ALL sequences in the batch to get a good distribution
    theta_trajectory = batch["theta"].numpy().flatten()
    import numpy as np
    theta_deg = theta_trajectory * (180.0 / np.pi)
    
    # Calculate the exact Token Bins used by the model
    theta_bins = np.clip(np.round(theta_deg * 10.0), -900, 900) + 900
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram across the full 1801 Vocabulary Size
    plt.hist(theta_bins, bins=1801, range=(0, 1800), color='purple', alpha=0.7)
    
    plt.title("Vocabulary Utilization (Token Bins 0 to 1800)\\nNotice how many of the 1801 tokens are completely unused!")
    plt.xlabel("Token ID (Bin)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    
    # Zoom in slightly to the active region for better visibility if it's super narrow
    active_min = max(0, np.min(theta_bins) - 50)
    active_max = min(1800, np.max(theta_bins) + 50)
    plt.xlim(active_min, active_max)
    
    theta_hist_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theta_histogram_test.png")
    plt.savefig(theta_hist_save_path, dpi=300)
    print(f"Token Bin histogram saved to: {theta_hist_save_path}\n")

if __name__ == "__main__":
    main()
