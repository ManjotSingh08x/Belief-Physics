# Pendulum Dataset Documentation

This document serves as a complete guide on how to utilize the `dataset.py` and `test_dataset.py` scripts to plug the pendulum physics engine directly into PyTorch's Deep Learning training loops.

## 1. How to Use the Dataset

The `PendulumIterableDataset` is designed to be effortlessly plugged into a standard `torch.utils.data.DataLoader`.

### Example Usage
```python
from torch.utils.data import DataLoader
from physics.dataset import PendulumIterableDataset

# 1. Initialize Dataset
# It automatically pulls default hyperparameters from params.py, but you can override them.
dataset = PendulumIterableDataset(batch_size=64, delta_v=0.1)

# 2. Initialize DataLoader
# NOTE: The dataloader batch_size dictates how many samples it yields per iteration,
# but the dataset internally generates them in chunks defined by dataset.batch_size.
dataloader = DataLoader(dataset, batch_size=8)

# 3. Training Loop
for batch in dataloader:
    theta = batch["theta"]             # torch.Size([8, m*n])
    velocity = batch["velocity"]       # torch.Size([8, m*n])
    belief_state = batch["belief_state"] # torch.Size([8, m*n, 3])
    
    # ... pass to Transformer / Neural Network ...
```

---

## 2. Component Reference

### `PendulumIterableDataset` (in `dataset.py`)
**Purpose:** An infinite generator inheriting from `torch.utils.data.IterableDataset`. It wraps the `PendulumSimulator` and yields formatting training dictionaries.

#### `__init__(self, n, m, t, batch_size, mu, g, l, initial_velocity, delta_v, alpha, x)`
- **Purpose:** Initializes the underlying `Mess3Process` and `PendulumSimulator`.
- **Args:** All arguments are optional. If left as `None`, they will automatically pull the `DEFAULT_` or `PLOT_` fallbacks configured inside `params.py`.
  - `batch_size`: The "micro-batch" size. This dictates how many sequences the simulator computes simultaneously in memory *before* yielding them one by one. Higher = faster generation, but requires more RAM.

#### `__iter__(self)`
- **Purpose:** The PyTorch generator function.
- **Outputs:** An infinite stream of `sample` dictionaries.
  - `sample["step_type"]`: Shape `(time,)`. `1` for impulse steps.
  - `sample["theta"]`: Shape `(time,)`. Angular position.
  - `sample["x"]`, `sample["y"]`: Shape `(time,)`. Cartesian position.
  - `sample["velocity"]`: Shape `(time,)`. Angular velocity.
  - `sample["hmm_state"]`: Shape `(time,)`. True hidden state.
  - `sample["belief_state"]`: Shape `(time, 3)`. Optimal belief simplex.
  - `sample["impulse_value"]`: Shape `(time,)`. Velocity delta injected.

---

## 3. Dataset Testing & Visualization

### `test_dataset.py`
**Purpose:** A robust verification script that ensures the `DataLoader` successfully communicates with the `PendulumIterableDataset`, verifies tensor dimensions, and visualizes the generated data.

**How to Use:**
```bash
python3 physics/test_dataset.py
```

**Outputs:**
1. **Console Output:** Prints the `torch.Size` and `torch.dtype` of every key inside the batch.
2. **`trajectory_test.png`:** A visual scatter plot of the X/Y coordinates of a *single* sequence pulled from the batch, color-coded by time step to reveal the exact physical path of the pendulum bob.
3. **`velocity_histogram_test.png`:** A 100-bin frequency histogram of the velocity tensor for that same sequence, useful for debugging physics instability or confirming standard deviation boundaries.
