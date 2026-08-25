# Experiment 02: HMM Physics Belief State Probing

## Goal
The goal of this experiment is to test the emergence of belief-state geometry in a Transformer model. Instead of feeding the discrete tokens directly into the model (as in standard next-token prediction), we feed the model a stream of continuous Cartesian coordinates `(x, y)` from a physical pendulum system that is secretly driven by a Hidden Markov Model (HMM). We then train the Transformer to predict the *next* coordinate `(x, y)`.

If the model is able to infer the underlying HMM states from the noisy physical data in order to optimally predict the trajectory, its internal representations should inherently learn a Bayesian belief-state simplex. 

## Dataset and Chunking
- We utilize the `PendulumIterableDataset` to generate data infinitely.
- Each generated sequence represents $m \times n = 500$ physics steps. 
- To train efficiently, these long sequences are chunked into shorter sequences of `SEQ_LEN = 50`.
- We defined an "Epoch" arbitrarily as `SAMPLES_PER_EPOCH = 16000` sequence chunks. 
- Within each epoch, we use an 80/20 train/test split. 80% of the generated batches are used for gradient descent (training), and 20% are used strictly for tracking validation MSE loss.

## Subspace Extraction & Probing
Once training concludes, we run a "probing" phase:
1. We freeze the model and extract internal representations from a specific layer (by default, `layer3`, which is the second-to-last transformer block).
2. We pull out 5 large batches of validation data.
3. We extract the internal activations corresponding to these inputs.
4. We train a standard `sklearn.linear_model.LinearRegression` to map the high-dimensional internal activations (e.g., 128-dimensional) to two physical properties:
   - **Velocity**: Predicting the 1D velocity magnitude.
   - **Belief State**: Predicting the 3D optimal belief state probability simplex.

## Barycentric Projection
Because the belief state is a 3-dimensional simplex ($b_1 + b_2 + b_3 = 1$), it is geometrically constrained to a 2D plane. To visualize it accurately, we mathematically project the 3D ground truth vectors and the 3D model predictions into 2D barycentric coordinates (an equilateral triangle). 

If the model successfully learned the belief geometry, the plot (`probe_belief_state_geometry.png`) will display a structured (often fractal) pattern bounded within the triangle!
