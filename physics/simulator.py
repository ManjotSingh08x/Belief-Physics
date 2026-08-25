import torch
import numpy as np
import params


class PendulumSimulator:
    def __init__(self, n, m, t, hmm=None, mu=params.DEFAULT_MU, g=params.DEFAULT_G, l=params.DEFAULT_L, delta_v=params.DEFAULT_DELTA_V):
        """
        Args:
            n: No. of steps between two state transitions in the HMM
            m: No. of Hidden Markov Model transitions in each simulation
            t: Duration of each step in ms
            hmm: Instance of HMM (e.g. Mess3Process) or None
            mu: Coefficient to do linear damping (-mu * v)
            g: Gravity constant
            l: Pendulum length
            delta_v: The velocity change magnitude for the impulses
        """
        self.n = n
        self.m = m
        self.t = t
        self.hmm = hmm
        self.mu = mu
        self.g = g
        self.l = l
        self.delta_v = delta_v

    @staticmethod
    def theta_to_xy(theta, length):
        """
        Converts polar angle theta to cartesian x, y coordinates.
        Supports both PyTorch tensors and Python floats.
        """
        if isinstance(theta, torch.Tensor):
            x = length * torch.sin(theta)
            y = -length * torch.cos(theta)
        else:
            import math
            x = length * math.sin(theta)
            y = -length * math.cos(theta)
        return x, y

    def simulate_pendulum(self, num_simulations, initial_velocity=0.0, initial_theta=0.0):
        """
        Runs the simulation in a vectorized batch.
        
        Args:
            num_simulations: Batch size (e.g., 512)
            initial_velocity: Starting velocity for all bobs (float or tensor)
            initial_theta: Starting angle for all bobs (float or tensor)
            
        Returns:
            Dictionary containing stacked tensors of the simulation history:
            - step_type: (num_simulations, m * n)
            - theta: (num_simulations, m * n)
            - x_cord: (num_simulations, m * n)
            - y_cord: (num_simulations, m * n)
            - velocity: (num_simulations, m * n)
            - hmm_state: (num_simulations, m * n) or None
            - belief_state: (num_simulations, m * n, num_states) or None
            - impulse_value: (num_simulations, m * n)
        """
        device = torch.device(params.DEFAULT_DEVICE) # Can be moved to GPU if needed
        dt = self.t / 1000.0
        total_steps = self.m * self.n
        
        # Prepare HMM Sequences if provided
        if self.hmm is not None:
            if hasattr(self.hmm, 'generate_batch'):
                hmm_states_np, all_obs = self.hmm.generate_batch(num_simulations, self.m)
                hmm_beliefs_np = self.hmm.optimal_batch(all_obs)
                
                hmm_states = torch.tensor(hmm_states_np, dtype=torch.long, device=device)
                hmm_beliefs = torch.tensor(hmm_beliefs_np, dtype=torch.float32, device=device)
                
                tokens = sorted(self.hmm.tokens)
                obs_indices_np = np.zeros_like(all_obs, dtype=int)
                for i, tok in enumerate(tokens):
                    obs_indices_np[all_obs == tok] = i
                obs_indices = torch.tensor(obs_indices_np, dtype=torch.long, device=device)
            else:
                all_states = []
                all_obs = []
                all_beliefs = []
                for _ in range(num_simulations):
                    s, o = self.hmm.generate_sequence(self.m)
                    b = self.hmm.optimal(o)
                    all_states.append(s)
                    all_obs.append(o)
                    all_beliefs.append(b)
                    
                hmm_states = torch.tensor(np.array(all_states), dtype=torch.long, device=device)
                hmm_beliefs = torch.tensor(np.array(all_beliefs), dtype=torch.float32, device=device)
                
                # Map HMM tokens to delta_v impulses.
                tokens = sorted(self.hmm.tokens)
                obs_indices = []
                for obs in all_obs:
                    obs_indices.append([tokens.index(tok) for tok in obs])
                obs_indices = torch.tensor(obs_indices, dtype=torch.long, device=device)
            
            dv_tensor = torch.zeros_like(obs_indices, dtype=torch.float32, device=device)
            dv_tensor[obs_indices == 0] = -self.delta_v
            dv_tensor[obs_indices == 1] = 0.0
            dv_tensor[obs_indices == 2] = self.delta_v
        else:
            hmm_states = None
            hmm_beliefs = None
            dv_tensor = None

        # Initialize physical states
        theta = torch.full((num_simulations,), initial_theta, dtype=torch.float32, device=device)
        v = torch.full((num_simulations,), initial_velocity, dtype=torch.float32, device=device)
        
        # Output Tensors
        out_step_type = torch.zeros((num_simulations, total_steps), dtype=torch.long, device=device)
        out_theta = torch.zeros((num_simulations, total_steps), dtype=torch.float32, device=device)
        out_x = torch.zeros((num_simulations, total_steps), dtype=torch.float32, device=device)
        out_y = torch.zeros((num_simulations, total_steps), dtype=torch.float32, device=device)
        out_v = torch.zeros((num_simulations, total_steps), dtype=torch.float32, device=device)
        out_impulse_value = torch.zeros((num_simulations, total_steps), dtype=torch.float32, device=device)
        
        if self.hmm is not None:
            out_hmm_state = torch.zeros((num_simulations, total_steps), dtype=torch.long, device=device)
            out_belief_state = torch.zeros((num_simulations, total_steps, self.hmm.num_states), dtype=torch.float32, device=device)
        else:
            out_hmm_state = None
            out_belief_state = None

        # Simulation Loop
        for step in range(total_steps):
            phase = step // self.n
            is_impulse_step = (step % self.n == 0)
            
            # Apply Impulse
            if is_impulse_step and self.hmm is not None:
                v += dv_tensor[:, phase]
                out_step_type[:, step] = 1 # Impulse was given
                out_impulse_value[:, step] = dv_tensor[:, phase]
                
            # Physics Step (Semi-Implicit Euler)
            acceleration = -(self.g / self.l) * torch.sin(theta) - self.mu * v
            v = v + acceleration * dt
            theta = theta + v * dt
            
            # Calculate coordinates
            x, y = self.theta_to_xy(theta, self.l)
            
            # Record Data
            out_theta[:, step] = theta
            out_x[:, step] = x
            out_y[:, step] = y
            out_v[:, step] = v
            
            if self.hmm is not None:
                out_hmm_state[:, step] = hmm_states[:, phase]
                out_belief_state[:, step, :] = hmm_beliefs[:, phase, :]
                
        return {
            "step_type": out_step_type,
            "theta": out_theta,
            "x_cord": out_x,
            "y_cord": out_y,
            "velocity": out_v,
            "hmm_state": out_hmm_state,
            "belief_state": out_belief_state,
            "impulse_value": out_impulse_value
        }
