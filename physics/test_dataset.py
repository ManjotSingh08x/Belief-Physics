import torch
from torch.utils.data import DataLoader
from dataset import PendulumIterableDataset
import matplotlib.pyplot as plt
import os
import params

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
    print(f"Velocity histogram saved to: {hist_save_path}\n")

if __name__ == "__main__":
    main()
