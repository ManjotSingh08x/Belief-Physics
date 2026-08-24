# Belief-State Geometry Under Physical Observation: A Pendulum-Perturbation Study

## Research Question

Belief-state geometry has been shown to emerge in transformers trained on next-token prediction over hidden Markov processes (e.g., the mess3 process), where the optimal predictive distribution requires tracking a Bayesian posterior over hidden states, and this posterior traces out a specific (often fractal) geometric structure in the residual stream.

That prior work gives the model the tokens directly as input and asks it to predict the next token. This project asks a harder question: **if the model never sees the tokens at all, and must instead infer them indirectly from their physical consequences, does the same belief-state geometry still emerge — and is it functionally necessary for optimal prediction, or merely an artifact of how the earlier tasks were set up?**

We operationalize "physical consequences" as a driven pendulum: each mess3 token deterministically triggers an impulse (left / right / none) to a pendulum initially at rest. The model observes only the resulting (x, y) trajectory and must predict future position. To do this well, it must implicitly (a) invert the physics to recover impulse history, (b) maintain a Bayesian belief over the hidden mess3 state consistent with that inferred history, and (c) combine that belief with ordinary physical state (position, velocity) to extrapolate the trajectory.

## Why This Is a Meaningful Extension

- Prior belief-geometry results are tied to a specific loss (next-token cross-entropy) and a specific observation channel (tokens). Whether the phenomenon survives a different loss (trajectory MSE) and a different, lossy observation channel (physical trajectory) is untested.
- If belief geometry is found here, it would suggest the phenomenon reflects something more general about how transformers represent uncertainty over discrete latent processes, not an artifact of token-level pretraining objectives.
- If belief geometry is *not* found, or is entangled/degenerate, that's informative too: it would suggest the geometry is contingent on the observation channel directly exposing the relevant sufficient statistics.

## Task Definition

1. Sample a hidden-state trajectory and token sequence from the mess3 process.
2. Map each token to an impulse applied to a pendulum: token 1 → left impulse, token 2 → right impulse, token 0 → no impulse.
3. Impulses are applied every *n* timesteps; total sequence length is *n·m* timesteps for *m* impulse events per sample (initial target m ≈ 10, matching known mess3 synchronization-length requirements).
4. Simulate pendulum dynamics (damped oscillator) between and across impulse events; pendulum starts at rest.
5. Train a transformer on (x, y) trajectories only, predicting position some fixed horizon *k* steps ahead.
6. Log, per sample and never expose to the model: token sequence, hidden mess3 state sequence, true belief-state trajectory (via HMM forward algorithm), true velocity, true impulse history.

*n*, *m*, and *k* are hyperparameters to be tuned jointly so that the task is neither solvable by pure physics extrapolation (belief-tracking unnecessary) nor by pure belief-tracking (physics unnecessary) — both should be required for near-optimal loss.

## Regime Structure

- **Regime A (initial phase):** Fully deterministic, noiseless physics and observation. Token history is exactly recoverable from the trajectory in principle. This isolates the "added computational depth" question — does belief-state tracking over the hidden mess3 state still emerge and remain necessary when the model must first invert physics to get at the tokens, even with no genuine physical uncertainty?
- **Regime B (later phase):** Observation and/or impulse-magnitude noise introduced, making token recovery itself probabilistic. This is closer to a continuous-emission HMM filtering problem and is deferred until Regime A groundwork (noise sweep design, Bayes-optimal loss computation) is established.

## Core Experiments

1. **Belief-state decodability.** Train linear probes from residual-stream activations to the ground-truth mess3 belief vector (forward-algorithm posterior). Measure R² and compare the recovered geometric structure (e.g., barycentric/simplex projection) against the known mess3 geometry from prior literature.
2. **Velocity decodability (sanity check).** Train linear probes to true velocity at each timestep. Establishes that the model represents ordinary physical state at all, and provides a second subspace for the orthogonality analysis.
3. **Subspace orthogonality analysis.** Using principal angles (CCA/SVD-based) between the belief-probe subspace and the velocity-probe subspace, test whether belief-state and physical-state information are represented in separable directions, or entangled/superposed. Control for shared trivial correlation with position-in-sequence before drawing conclusions.
4. **Third subspace: physical impulse echo.** Because the pendulum starts at rest and damping is finite, raw physical state itself retains a decaying "memory" of past impulses independent of any Bayesian belief computation. This is logged separately (true impulse history) and included as a third subspace in the orthogonality analysis, to ensure it isn't mistaken for belief-state representation.

## Ablations

1. **Belief-subspace zeroing.** Zero out the identified belief-state subspace at points just before an upcoming impulse event and observe whether predicted trajectory reverts to the un-perturbed damped/free-oscillation baseline. This is the primary *necessity* test — decodability (probes) shows presence, this shows causal reliance.
2. **Velocity-subspace zeroing.** Same procedure on the velocity subspace, as a cross-check that ablations produce physically sensible, dissociable effects (i.e., zeroing velocity should distort trajectory shape differently than zeroing belief state).
3. **Impulse-echo subspace zeroing.** Same procedure on the third (physical echo) subspace, to confirm it is dissociable from both of the above.

## Controls

1. **Token-given control.** Identical task and physics, but the model additionally receives the true token history as input. Compares belief-state geometry (presence, cleanliness, R²) against the physics-only condition to isolate the effect of the physics-inversion layer itself.
2. **Bayes-optimal loss bound (planned, methodology TBD in follow-up discussion).** Before large training runs, compute via statistical simulation the achievable loss under (a) full Bayesian belief-tracking and (b) a naive no-belief-tracking baseline, at a given (n, m, k) setting, to confirm a meaningful performance gap exists — i.e., that the task hyperparameters actually make belief tracking worthwhile before spending compute on training.

## Hypotheses

- **H1:** Belief-state geometry consistent with the known mess3 structure will be recoverable from residual-stream activations in Regime A, despite the model never observing tokens directly.
- **H2:** The belief-state subspace and velocity subspace will show partial but incomplete orthogonality (some entanglement expected due to superposition under limited model width), rather than clean separation.
- **H3:** Zeroing the belief-state subspace will causally degrade prediction specifically around upcoming impulse events, while leaving baseline damped-oscillation prediction intact — establishing necessity, not just decodability.
- **H4:** The token-given control will show cleaner/higher-R² belief geometry than the physics-only condition, indicating the physics-inversion layer partially degrades or entangles the representation rather than preserving it exactly.

## Further Experiments (later phases)

- Regime B: introduce observation and/or impulse-magnitude noise; sweep noise level and track belief-probe R² and geometry shape as a function of noise, turning the "is belief tracking necessary" question into an empirical curve rather than a binary claim.
- Alternative physical systems beyond the pendulum (e.g., planetary motion with discrete perturbation events analogous to meteor collisions), to test whether findings generalize across physics or are specific to damped-oscillator dynamics.
- Formal Bayes-optimal loss computation methodology and its use as a pre-registration-style check before each new (n, m, k, noise) configuration is trained.

## Open Items for Next Discussion

- Exact damping model and pendulum parameterization (linear damped oscillator recommended for Regime A to keep the physics analytically tractable).
- Concrete (n, m, k) sweep range and stopping criterion for "balance found."
- Probe training details (regularization, train/test split conventions) and subspace-analysis tooling (CCA implementation, PCA visualization pipeline) — deferred per your note.
- Bayes-optimal loss simulation methodology — deferred per your note.
