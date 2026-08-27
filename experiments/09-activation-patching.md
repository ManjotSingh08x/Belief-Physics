# Experiment 9: Activation Patching (Subspace Patching)

## Overview
In previous experiments (like Experiment 8), we performed **Ablation**—we projected out (deleted) the linear belief state to see if the model's loss would increase. When the loss didn't change significantly, it raised a question: *Does the model actually use this linear belief state, or is the linear probe just finding a spurious correlation?*

**Activation Patching** (also known as Causal Tracing or Interchange Interventions) is a much stronger test for causality. Instead of deleting information, we **swap** it. 

## How It Works
1. We take a batch of sequences (`Sequence A`) and compute their baseline predictions.
2. We take a different, random batch of sequences (`Sequence B`, our source).
3. At a specific layer $L$, we extract the exact linear belief state vector from `Sequence B`.
4. We surgically inject `Sequence B`'s belief state into `Sequence A`, while removing `Sequence A`'s original belief state.
5. We let the model finish the rest of the layers and output a final prediction.

### The Metric: KL Divergence
To measure the causal effect, we calculate the **KL Divergence** between the model's original predictions for `Sequence A` and the new *patched* predictions. 
- KL Divergence measures how much one probability distribution differs from another. 
- **If the KL Divergence is high (> 0):** It means the patched predictions shifted significantly. This proves that swapping the belief state *caused* the model to change its mind, confirming the linear representation is causally used.
- **If the KL Divergence is ~0:** It means the model completely ignored the swapped belief state and output the exact same prediction as before, confirming the linear representation is just a spurious artifact or not used downstream.

## Running the Experiment
We perform subspace activation patching for both **Velocity** and the **Belief State** layer-by-layer to compare their causal impacts.

```bash
python3 experiments/09-activation-patching.py
```

## Interpreting the Results
When you run the script, it generates a graph (`experiments/outputs-09/activation_patching.png`).

- **Velocity Patched (Red):** You should expect a large KL Divergence when patching velocity. Because velocity dictates the deterministic physics update, injecting the velocity of a completely different pendulum will cause the model to predict drastically different positions, shifting the probability distribution heavily.
- **Belief State Patched (Blue):** The belief state controls the *uncertainty* (stochastic impulse probabilities). If the KL Divergence for the belief state is greater than 0 (especially in later layers like Layer 2 or 3), it means the model **is** causally using that exact 3D linear representation to adjust its uncertainty, proving the correlation is real! The effect size will naturally be smaller than velocity since uncertainty shifts the distribution less than a completely different velocity trajectory.
