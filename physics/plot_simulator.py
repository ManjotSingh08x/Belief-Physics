import matplotlib.pyplot as plt
from simulator import PendulumSimulator
import params
import sys
import os
import argparse

# Ensure hmm.py can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hmm import Mess3Process, RRXORProcess

def main():
    parser = argparse.ArgumentParser(description="Plot Pendulum Simulator")
    parser.add_argument("--hmm_type", type=str, default="mess3", choices=["mess3", "rrxor"], help="HMM type to test")
    args = parser.parse_args()
    
    # Parameters
    n = params.DEFAULT_N    # steps between HMM transitions
    m = params.DEFAULT_M    # number of HMM transitions
    t = params.DEFAULT_T    # step duration in ms
    num_simulations = params.PLOT_NUM_SIMULATIONS

    # 1. No HMM
    sim_no_hmm = PendulumSimulator(n=n, m=m, t=t, hmm=None, mu=params.DEFAULT_MU, g=params.DEFAULT_G, l=params.DEFAULT_L)
    res_no_hmm = sim_no_hmm.simulate_pendulum(num_simulations, initial_velocity=params.DEFAULT_INITIAL_V, initial_theta=params.DEFAULT_INITIAL_THETA)
    vel_no_hmm = res_no_hmm["velocity"][0].numpy()

    # 2. With HMM
    if args.hmm_type == 'mess3':
        hmm_process = Mess3Process(alpha=params.DEFAULT_HMM_ALPHA, x=params.DEFAULT_HMM_X)
    elif args.hmm_type == 'rrxor':
        rrxor_alpha = getattr(params, 'DEFAULT_RRXOR_ALPHA', 1.0)
        hmm_process = RRXORProcess(alpha=rrxor_alpha)
        
    sim_with_hmm = PendulumSimulator(n=n, m=m, t=t, hmm=hmm_process, mu=params.DEFAULT_MU, g=params.DEFAULT_G, l=params.DEFAULT_L, delta_v=params.DEFAULT_DELTA_V)
    res_with_hmm = sim_with_hmm.simulate_pendulum(num_simulations, initial_velocity=params.DEFAULT_INITIAL_V, initial_theta=params.DEFAULT_INITIAL_THETA)
    vel_with_hmm = res_with_hmm["velocity"][0].numpy()
    
    # Extract impulse steps
    step_types = res_with_hmm["step_type"][0].numpy()
    impulse_vals = res_with_hmm["impulse_value"][0].numpy()
    impulse_indices = (step_types == 1).nonzero()[0]
    
    # Time axis
    dt = t / 1000.0
    time = [i * dt for i in range(len(vel_no_hmm))]

    plt.figure(figsize=(12, 6))
    plt.plot(time, vel_no_hmm, label="No HMM (Free oscillation with damping)", color='blue', alpha=0.7)
    plt.plot(time, vel_with_hmm, label=f"With HMM ({args.hmm_type.upper()})", color='red', alpha=0.7)
    
    # Mark impulses
    if params.PLOT_MARKERS:
        # Use abs() < 1e-6 for float comparison safety
        pos_idx = [i for i in impulse_indices if impulse_vals[i] > 1e-6]
        neg_idx = [i for i in impulse_indices if impulse_vals[i] < -1e-6]
        zero_idx = [i for i in impulse_indices if abs(impulse_vals[i]) <= 1e-6]

        # Always call scatter (even with empty lists) so the legend never randomly disappears
        plt.scatter([time[i] for i in pos_idx], [vel_with_hmm[i] for i in pos_idx], 
                    color='green', zorder=5, label="+ Impulse", marker='^', s=100)
        
        plt.scatter([time[i] for i in neg_idx], [vel_with_hmm[i] for i in neg_idx], 
                    color='purple', zorder=5, label="- Impulse", marker='v', s=100)
        
        plt.scatter([time[i] for i in zero_idx], [vel_with_hmm[i] for i in zero_idx], 
                    color='black', zorder=5, label="No Change (0 Impulse)", marker='x', s=100)

    plt.title(f"Pendulum Velocity Map: No HMM vs HMM ({args.hmm_type.upper()})")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "velocity_map.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
