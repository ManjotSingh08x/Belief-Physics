import torch
import numpy as np

# Default Pendulum Physics Parameters

# mu: Coefficient for linear viscous damping (-mu * v). Controls how fast the pendulum loses energy.
DEFAULT_MU = 0.5

# g: Acceleration due to gravity (determines the restoring force along with L).
DEFAULT_G = 9.8

# l: Length of the pendulum (affects the period of oscillation).
DEFAULT_L = 1

DEFAULT_INITIAL_V = 1 

# delta_v: The base magnitude of the velocity impulse applied by the HMM. 
# An HMM token of '0' applies -delta_v, '2' applies +delta_v, and '1' applies 0.
DEFAULT_DELTA_V = 0.3

# device: PyTorch device to run the tensor operations on (e.g., 'cpu' or 'cuda').
DEFAULT_DEVICE = "cpu"


# Plotting Simulator Parameters

# n: Number of physical simulation steps (Euler integration steps) between two HMM state transitions.
# An impulse is injected exactly once every 'n' steps.
DEFAULT_N = 10

# m: Total number of HMM state transitions (and thus impulses generated) in a single simulation run.
# The total number of simulation steps will be m * n.
DEFAULT_M = 50

# t: Duration of a single physical simulation step in milliseconds.
# In the code, this is converted to seconds for physics integration (dt = t / 1000.0).
DEFAULT_T = 20

# num_simulations: The batch size, or number of pendulums to simulate in parallel.
PLOT_NUM_SIMULATIONS = 1

PLOT_MU = 0.1
PLOT_G = 9.8
PLOT_L = 1.0

# initial_velocity: The starting angular velocity (v) of the pendulum bob at t=0.
PLOT_INITIAL_VELOCITY = 1

# initial_theta: The starting angular displacement (theta) of the pendulum bob at t=0.
PLOT_INITIAL_THETA = 0

PLOT_DELTA_V = 0.1

# plot_markers: Whether to visually mark the applied impulses on the output plot
PLOT_MARKERS = False


# HMM Parameters for Plotting (Specifically for Mess3Process)

# alpha: The probability of the HMM emitting its "dominant" or primary token for its current state.
# (e.g. State 0 emitting 'A', State 1 emitting 'B').
DEFAULT_HMM_ALPHA = 0.7

# x: The probability of the HMM transitioning to a specific neighboring state.
# In Mess3Process (a triangle), there are 2 neighbors, so the chance to stay in the current state is (1 - 2*x).
DEFAULT_HMM_X = 0.15

# Default HMM Type ('mess3' or 'rrxor')
DEFAULT_HMM_TYPE = 'mess3'

# RRXOR Specific Parameters
DEFAULT_RRXOR_ALPHA = 0.7


# Dataset Generation Parameters
DATASET_BATCH_SIZE = 4096
# --- Discretization ---

VOCAB_SIZE = 181 # Bins from -90.0 to +90.0 degrees with 0.1 degree resolution

def discretize_theta(theta):
    """
    Discretizes continuous theta (in radians) into VOCAB_SIZE integer bins.
    Assumes theta is a torch Tensor.
    """
    theta_deg = theta * (180.0 / np.pi)
    return torch.clamp(torch.round(theta_deg), -90, 90).long() + 90