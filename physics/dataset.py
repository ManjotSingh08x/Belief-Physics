import torch
from torch.utils.data import IterableDataset
import params
from hmm import Mess3Process
from simulator import PendulumSimulator

class PendulumIterableDataset(IterableDataset):
    def __init__(self, n=None, m=None, t=None, 
                 batch_size=None, 
                 mu=None, g=None, l=None, 
                 initial_velocity = None,
                 delta_v=None,
                 alpha=None, x=None):
        super().__init__()
        
        # Fallback to params if not explicitly provided
        self.n = n if n is not None else params.DEFAULT_N
        self.m = m if m is not None else params.DEFAULT_M
        self.t = t if t is not None else params.DEFAULT_T
        self.batch_size = batch_size if batch_size is not None else params.DATASET_BATCH_SIZE
        
        self.mu = mu if mu is not None else params.DEFAULT_MU
        self.g = g if g is not None else params.DEFAULT_G
        self.l = l if l is not None else params.DEFAULT_L
        self.delta_v = delta_v if delta_v is not None else params.DEFAULT_DELTA_V
        self.initial_v = initial_velocity if initial_velocity is not None else params.DEFAULT_INITIAL_V

        self.alpha = alpha if alpha is not None else params.DEFAULT_HMM_ALPHA
        self.x = x if x is not None else params.DEFAULT_HMM_X
        
        self.hmm = Mess3Process(alpha=self.alpha, x=self.x)
        self.simulator = PendulumSimulator(
            n=self.n, m=self.m, t=self.t, hmm=self.hmm, 
            mu=self.mu, g=self.g, l=self.l, delta_v=self.delta_v
        )

    def __iter__(self):
        """
        Infinitely yields individual simulation samples.
        It generates them internally in fast, parallelized micro-batches to save time.
        """
        while True:
            # 1. Generate a micro-batch of simulations concurrently
            # This implicitly calls generate_batch and optimal_batch on the HMM.
            batch_results = self.simulator.simulate_pendulum(self.batch_size, initial_velocity=self.initial_v)
            
            # 2. Yield them one-by-one so DataLoader can easily batch them 
            # across multiple workers if needed.
            for i in range(self.batch_size):
                sample = {
                    "step_type": batch_results["step_type"][i],
                    "theta": batch_results["theta"][i],
                    "x": batch_results["x_cord"][i],
                    "y": batch_results["y_cord"][i],
                    "velocity": batch_results["velocity"][i],
                    "hmm_state": batch_results["hmm_state"][i],
                    "belief_state": batch_results["belief_state"][i],
                    "impulse_value": batch_results["impulse_value"][i]
                }
                yield sample
