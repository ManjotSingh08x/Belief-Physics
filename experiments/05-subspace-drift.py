import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    sys.path.append(os.getcwd())

from models.architecture import TokenTransformer
from physics.dataset import PendulumIterableDataset
from physics import params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 250
BATCH_SIZE = 256
VOCAB_SIZE = params.VOCAB_SIZE
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 1
MAX_SEQ_LEN = 1024
K = params.DEFAULT_N
NUM_BLOCKS = (SEQ_LEN - N) // K + 1

# theta encoding: Original training used 1-degree bins over [-90, 90]
THETA_SCALE = 1.0
THETA_OFFSET = 90

TRAIN_BATCHES = 200
TEST_BATCHES = 30
ALPHA_REGULAR = 1.0
ALPHA_ZERO = 1e-6
PEAK_T = 9
NUM_FOLDS = 5
NULL_DIST_SAMPLES = 500

OUTPUT_DIR = "experiments/outputs-05"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = "experiments/outputs-03/hmm_discrete_model.pt"


def theta_to_bins(theta_rad):
    return params.discretize_theta(theta_rad)


def to_barycentric(belief_3d):
    v1 = torch.tensor([0.0, 1.0], device=DEVICE)
    v2 = torch.tensor([np.sqrt(3) / 2, -0.5], device=DEVICE)
    v3 = torch.tensor([-np.sqrt(3) / 2, -0.5], device=DEVICE)
    return belief_3d[..., 0:1] * v1 + belief_3d[..., 1:2] * v2 + belief_3d[..., 2:3] * v3


def normalize_activations(acts):
    mean = acts.mean(dim=0, keepdim=True)
    std = acts.std(dim=0, keepdim=True) + 1e-8
    return (acts - mean) / std


def subspace_angle(beta1, beta2):
    if beta1 is None or beta2 is None:
        return 90.0
    Q1, _ = torch.linalg.qr(beta1.to(DEVICE, dtype=torch.float64))
    Q2, _ = torch.linalg.qr(beta2.to(DEVICE, dtype=torch.float64))
    if len(Q1.shape) == 1:
        Q1 = Q1.unsqueeze(1)
    if len(Q2.shape) == 1:
        Q2 = Q2.unsqueeze(1)
    S = torch.linalg.svdvals(Q1.T @ Q2)
    cos_theta = torch.clamp(S[0], 0.0, 1.0)
    return (torch.acos(cos_theta) * 180.0 / np.pi).item()


def generate_null_distribution(ambient_dim, dim1, dim2, n_samples):
    angles = []
    for _ in range(n_samples):
        M1 = torch.randn(ambient_dim, dim1, dtype=torch.float64)
        M2 = torch.randn(ambient_dim, dim2, dtype=torch.float64)
        Q1, _ = torch.linalg.qr(M1)
        Q2, _ = torch.linalg.qr(M2)
        S = torch.linalg.svdvals(Q1.T @ Q2)
        cos_theta = torch.clamp(S[0], 0.0, 1.0)
        angles.append((torch.acos(cos_theta) * 180.0 / np.pi).item())
    return np.array(angles)


def chunk_batch(batch_tensor, seq_len):
    total_time = batch_tensor.size(1)
    return [batch_tensor[:, s:s + seq_len] for s in range(0, total_time - seq_len, seq_len)]


def worker_init_fn(worker_id):
    import time
    np.random.seed((int(time.time() * 1000) + worker_id) % (2 ** 32 - 1))


def windowed_drift_test(signals, threshold):
    drifts = [i for i in range(1, len(signals)) if signals[i] >= threshold]
    if len(drifts) == 0:
        return "Stable"
    if len(drifts) > len(signals) // 2:
        return "Gradual Drift"
    return f"Abrupt at {drifts[0]}"


class StreamingRidge:
    def __init__(self, x_dim, y_dim):
        self.x_dim = x_dim
        self.N = 0
        self.sum_x = torch.zeros(x_dim, device=DEVICE, dtype=torch.float64)
        self.sum_y = torch.zeros(y_dim, device=DEVICE, dtype=torch.float64)
        self.sum_xx = torch.zeros(x_dim, x_dim, device=DEVICE, dtype=torch.float64)
        self.sum_xy = torch.zeros(x_dim, y_dim, device=DEVICE, dtype=torch.float64)

    def update(self, X, y):
        X = X.to(DEVICE, dtype=torch.float64)
        y = y.to(DEVICE, dtype=torch.float64)
        self.N += X.shape[0]
        self.sum_x += X.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.sum_xx += X.T @ X
        self.sum_xy += X.T @ y

    def solve(self, alpha=1.0):
        if self.N == 0:
            return None, None, None
        mu_x = self.sum_x / self.N
        mu_y = self.sum_y / self.N
        XtX = self.sum_xx - self.N * torch.outer(mu_x, mu_x)
        Xty = self.sum_xy - self.N * torch.outer(mu_x, mu_y)
        I = torch.eye(self.x_dim, device=DEVICE, dtype=torch.float64)
        beta = torch.linalg.solve(XtX + alpha * self.N * I, Xty)
        return beta.float(), mu_x.float(), mu_y.float()


class StreamingEvaluator:
    def __init__(self, y_dim):
        self.N = 0
        self.sum_y = torch.zeros(y_dim, device=DEVICE, dtype=torch.float64)
        self.sum_yy = torch.zeros(y_dim, device=DEVICE, dtype=torch.float64)
        self.sum_res = torch.zeros(y_dim, device=DEVICE, dtype=torch.float64)

    def update(self, X, y, beta, mu_x, mu_y_train):
        if beta is None:
            return
        X = X.to(DEVICE, dtype=torch.float64)
        y = y.to(DEVICE, dtype=torch.float64)
        beta = beta.to(DEVICE, dtype=torch.float64)
        mu_x = mu_x.to(DEVICE, dtype=torch.float64)
        mu_y_train = mu_y_train.to(DEVICE, dtype=torch.float64)
        y_pred = (X - mu_x) @ beta + mu_y_train
        self.N += X.shape[0]
        self.sum_y += y.sum(dim=0)
        self.sum_yy += (y ** 2).sum(dim=0)
        self.sum_res += ((y - y_pred) ** 2).sum(dim=0)

    def get_r2(self):
        if self.N == 0:
            return 0.0
        mu_y = self.sum_y / self.N
        ss_tot = self.sum_yy - self.N * (mu_y ** 2)
        r2 = 1.0 - (self.sum_res / torch.clamp(ss_tot, min=1e-8))
        return r2.mean().item()


class PipelineRunner:
    def __init__(self):
        self.b_dim = 2
        self.v_dim = 1
        self.load_model()
        self.setup_data()
        self.check_encoding_sanity()

        # separate nulls: belief-vs-belief (drift) and belief-vs-velocity (orthogonality)
        self.null_dist_bb = generate_null_distribution(EMBED_DIM, self.b_dim, self.b_dim, NULL_DIST_SAMPLES)
        self.null_dist_bv = generate_null_distribution(EMBED_DIM, self.b_dim, self.v_dim, NULL_DIST_SAMPLES)
        self.null_bb_threshold = np.percentile(self.null_dist_bb, 5)
        self.null_bv_threshold = np.percentile(self.null_dist_bv, 5)
        print(f"Belief-belief null (5th pct): {self.null_bb_threshold:.2f} deg")
        print(f"Belief-velocity null (5th pct): {self.null_bv_threshold:.2f} deg")

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Train it first.")
        self.model = TokenTransformer(
            vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS, num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN
        ).to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state_dict.items()})
        self.model.eval()

        self.activations = {}

        def get_activation(name):
            def hook(module, inp, out):
                self.activations[name] = out
            return hook

        for i in range(NUM_LAYERS):
            self.model.blocks[i].register_forward_hook(get_activation(f'layer{i}'))

    def setup_data(self):
        self.dataset = PendulumIterableDataset(batch_size=BATCH_SIZE)
        self.data_iter = iter(DataLoader(self.dataset, batch_size=BATCH_SIZE, worker_init_fn=worker_init_fn))

    def check_encoding_sanity(self):
        batch = next(self.data_iter)
        bins = theta_to_bins(batch["theta"])
        with torch.no_grad():
            out = self.model(bins.to(DEVICE)[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            out.reshape(-1, VOCAB_SIZE), bins.to(DEVICE)[:, 1:].reshape(-1)
        ).item()
        uniform_loss = float(np.log(VOCAB_SIZE))
        print(f"Sanity check: next-token CE loss = {loss:.4f}, uniform-random baseline = {uniform_loss:.4f}")
        if loss > uniform_loss:
            print("WARNING: model loss exceeds uniform-random baseline. Encoding or model mismatch likely.")
        self.data_iter = iter(DataLoader(self.dataset, batch_size=BATCH_SIZE, worker_init_fn=worker_init_fn))

    def run_phase_0_cv_check(self):
        print("\n--- Phase 0: Full-concat CV check ---")
        X_all, Y_all = [], []
        for _ in range(5):
            batch = next(self.data_iter)
            bins = theta_to_bins(batch["theta"]).to(DEVICE)
            with torch.no_grad():
                self.model(bins[:, :-1])
            b_targets = to_barycentric(batch["belief_state"].to(DEVICE))
            acts = torch.cat([self.activations[f'layer{l}'] for l in range(NUM_LAYERS)], dim=2)
            X_all.append(acts[:, :N, :].reshape(-1, acts.shape[2]).cpu())
            Y_all.append(b_targets[:, :N, :].reshape(-1, 2).cpu())
            self.activations.clear()

        X = torch.cat(X_all, dim=0).numpy()
        Y = torch.cat(Y_all, dim=0).numpy()
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        r2_scores = []
        for train_idx, test_idx in kf.split(X):
            ridge = Ridge(alpha=ALPHA_REGULAR)
            ridge.fit(X[train_idx], Y[train_idx])
            r2_scores.append(ridge.score(X[test_idx], Y[test_idx]))
        mean_r2, std_r2 = np.mean(r2_scores), np.std(r2_scores)
        print(f"Full-concat belief R2 across {NUM_FOLDS} folds: {mean_r2:.4f} +/- {std_r2:.4f}")
        if std_r2 > 0.1:
            print("WARNING: concatenated R2 unstable across folds.")

    def run_training_sweeps(self):
        print("\n--- Training per-cell probes ---")
        self.train_accums = {
            "b": [[StreamingRidge(EMBED_DIM, self.b_dim) for _ in range(N)] for _ in range(NUM_LAYERS)],
            "v": [[StreamingRidge(EMBED_DIM, self.v_dim) for _ in range(N)] for _ in range(NUM_LAYERS)],
        }
        b_indices = [b * K for b in range(NUM_BLOCKS)]

        for _ in tqdm(range(TRAIN_BATCHES), desc="train"):
            batch = next(self.data_iter)
            bins = theta_to_bins(batch["theta"])
            coord_chunks = chunk_batch(bins.to(DEVICE), SEQ_LEN + K)
            vel_chunks = chunk_batch(batch["velocity"].to(DEVICE), SEQ_LEN + K)
            belief_chunks = chunk_batch(to_barycentric(batch["belief_state"].to(DEVICE)), SEQ_LEN + K)

            for c_idx in range(len(coord_chunks)):
                inputs = coord_chunks[c_idx][:, :-K]
                b_targets = belief_chunks[c_idx][:, b_indices, :].reshape(-1, self.b_dim)
                v_targets = vel_chunks[c_idx][:, b_indices].reshape(-1, self.v_dim)
                with torch.no_grad():
                    self.model(inputs)
                acts = [normalize_activations(self.activations[f'layer{l}']) for l in range(NUM_LAYERS)]
                for t in range(N):
                    idx = [b * K + t for b in range(NUM_BLOCKS)]
                    for l in range(NUM_LAYERS):
                        act_lt = acts[l][:, idx, :].reshape(-1, EMBED_DIM)
                        self.train_accums["b"][l][t].update(act_lt, b_targets)
                        self.train_accums["v"][l][t].update(act_lt, v_targets)
            self.activations.clear()
            torch.cuda.empty_cache()

        self.solved_regular = {
            "b": [[self.train_accums["b"][l][t].solve(ALPHA_REGULAR) for t in range(N)] for l in range(NUM_LAYERS)],
            "v": [[self.train_accums["v"][l][t].solve(ALPHA_REGULAR) for t in range(N)] for l in range(NUM_LAYERS)],
        }
        self.solved_zero = {
            "b": [[self.train_accums["b"][l][t].solve(ALPHA_ZERO) for t in range(N)] for l in range(NUM_LAYERS)],
        }

    def run_testing_sweeps(self):
        print("\n--- Testing per-cell probes ---")
        self.evals_regular = {
            "b": [[StreamingEvaluator(self.b_dim) for _ in range(N)] for _ in range(NUM_LAYERS)],
            "v": [[StreamingEvaluator(self.v_dim) for _ in range(N)] for _ in range(NUM_LAYERS)],
        }
        self.evals_zero = {
            "b": [[StreamingEvaluator(self.b_dim) for _ in range(N)] for _ in range(NUM_LAYERS)],
        }
        b_indices = [b * K for b in range(NUM_BLOCKS)]

        for _ in tqdm(range(TEST_BATCHES), desc="test"):
            batch = next(self.data_iter)
            bins = theta_to_bins(batch["theta"])
            coord_chunks = chunk_batch(bins.to(DEVICE), SEQ_LEN + K)
            vel_chunks = chunk_batch(batch["velocity"].to(DEVICE), SEQ_LEN + K)
            belief_chunks = chunk_batch(to_barycentric(batch["belief_state"].to(DEVICE)), SEQ_LEN + K)

            for c_idx in range(len(coord_chunks)):
                inputs = coord_chunks[c_idx][:, :-K]
                b_targets = belief_chunks[c_idx][:, b_indices, :].reshape(-1, self.b_dim)
                v_targets = vel_chunks[c_idx][:, b_indices].reshape(-1, self.v_dim)
                with torch.no_grad():
                    self.model(inputs)
                acts = [normalize_activations(self.activations[f'layer{l}']) for l in range(NUM_LAYERS)]
                for t in range(N):
                    idx = [b * K + t for b in range(NUM_BLOCKS)]
                    for l in range(NUM_LAYERS):
                        act_lt = acts[l][:, idx, :].reshape(-1, EMBED_DIM)
                        beta_b, mux_b, muy_b = self.solved_regular["b"][l][t]
                        self.evals_regular["b"][l][t].update(act_lt, b_targets, beta_b, mux_b, muy_b)
                        beta_v, mux_v, muy_v = self.solved_regular["v"][l][t]
                        self.evals_regular["v"][l][t].update(act_lt, v_targets, beta_v, mux_v, muy_v)
                        beta_b0, mux_b0, muy_b0 = self.solved_zero["b"][l][t]
                        self.evals_zero["b"][l][t].update(act_lt, b_targets, beta_b0, mux_b0, muy_b0)
            self.activations.clear()
            torch.cuda.empty_cache()

    def analyze_drift(self):
        print("\n--- Drift analysis ---")
        self.belief_r2 = np.zeros((NUM_LAYERS, N))
        for l in range(NUM_LAYERS):
            for t in range(N):
                self.belief_r2[l, t] = self.evals_regular["b"][l][t].get_r2()

        # constrained reference: must allow causal propagation to PEAK_T via later layers
        mask = np.full_like(self.belief_r2, -np.inf)
        mask[:NUM_LAYERS - 1, :PEAK_T + 1] = self.belief_r2[:NUM_LAYERS - 1, :PEAK_T + 1]
        flat_idx = np.argmax(mask)
        self.ref_l = flat_idx // N
        self.ref_t = flat_idx % N
        assert self.ref_l < NUM_LAYERS - 1, "reference layer must allow forward propagation"
        assert self.ref_t <= PEAK_T, "reference timestep must be at or before eval timestep"
        print(f"Reference cell: layer {self.ref_l}, t={self.ref_t} (R2={self.belief_r2[self.ref_l, self.ref_t]:.4f})")

        self.ref_beta_b = self.solved_regular["b"][self.ref_l][self.ref_t][0]

        self.angle_vs_l = [subspace_angle(self.ref_beta_b, self.solved_regular["b"][l][self.ref_t][0]) for l in range(NUM_LAYERS)]
        self.angle_vs_t = [subspace_angle(self.ref_beta_b, self.solved_regular["b"][self.ref_l][t][0]) for t in range(N)]

        self.drift_label_l = windowed_drift_test(self.angle_vs_l, self.null_bb_threshold)
        self.drift_label_t = windowed_drift_test(self.angle_vs_t, self.null_bb_threshold)
        print(f"Drift vs layer: {self.drift_label_l}")
        print(f"Drift vs timestep: {self.drift_label_t}")

        self.bv_angle_grid = np.zeros((NUM_LAYERS, N))
        for l in range(NUM_LAYERS):
            for t in range(N):
                beta_b = self.solved_regular["b"][l][t][0]
                beta_v = self.solved_regular["v"][l][t][0]
                self.bv_angle_grid[l, t] = subspace_angle(beta_b, beta_v)

    def run_causal_ablations(self):
        print("\n--- Causal ablations and patching ---")
        Q_b, _ = torch.linalg.qr(self.ref_beta_b.to(DEVICE))
        P_b = Q_b @ Q_b.T
        P_b_perp = torch.eye(EMBED_DIM, device=DEVICE) - P_b

        v_beta = self.solved_regular["v"][self.ref_l][self.ref_t][0].to(DEVICE)
        Q_v, _ = torch.linalg.qr(v_beta)
        P_v = Q_v @ Q_v.T
        P_v_perp = torch.eye(EMBED_DIM, device=DEVICE) - P_v

        Q_r, _ = torch.linalg.qr(torch.randn_like(self.ref_beta_b).to(DEVICE))
        P_r = Q_r @ Q_r.T
        P_r_perp = torch.eye(EMBED_DIM, device=DEVICE) - P_r

        def make_projection_hook(projection_matrix, target_t):
            def hook(module, inp, output):
                for b in range(NUM_BLOCKS):
                    idx = b * K + target_t
                    if idx < output.shape[1]:
                        output[:, idx, :] = output[:, idx, :] @ projection_matrix
                return output
            return hook

        criterion = torch.nn.CrossEntropyLoss()
        losses = {"Baseline": [], "Belief Ablated": [], "Velocity Ablated": [], "Random Ablated": []}

        for _ in range(5):
            batch = next(self.data_iter)
            bins = theta_to_bins(batch["theta"])
            inputs = bins.to(DEVICE)[:, :-1]
            targets = bins.to(DEVICE)[:, 1:]
            eval_indices = [b * K + PEAK_T for b in range(NUM_BLOCKS) if b * K + PEAK_T < targets.size(1)]

            with torch.no_grad():
                out = self.model(inputs)
                losses["Baseline"].append(criterion(out[:, eval_indices, :].transpose(1, 2), targets[:, eval_indices]).item())

                for name, proj in [("Belief Ablated", P_b_perp), ("Velocity Ablated", P_v_perp), ("Random Ablated", P_r_perp)]:
                    h = self.model.blocks[self.ref_l].register_forward_hook(make_projection_hook(proj, self.ref_t))
                    out = self.model(inputs)
                    losses[name].append(criterion(out[:, eval_indices, :].transpose(1, 2), targets[:, eval_indices]).item())
                    h.remove()

            self.activations.clear()
            torch.cuda.empty_cache()

        patching_losses = []
        for _ in range(5):
            batch = next(self.data_iter)
            bins = theta_to_bins(batch["theta"])
            inputs = bins.to(DEVICE)[:, :-1]
            targets = bins.to(DEVICE)[:, 1:]
            shuffle_idx = torch.randperm(inputs.size(0))

            def swap_hook(module, inp, output):
                for b in range(NUM_BLOCKS):
                    idx = b * K + self.ref_t
                    if idx < output.shape[1]:
                        belief_comp = output[:, idx, :] @ P_b
                        other_comp = output[:, idx, :] - belief_comp
                        output[:, idx, :] = other_comp + belief_comp[shuffle_idx]
                return output

            h = self.model.blocks[self.ref_l].register_forward_hook(swap_hook)
            with torch.no_grad():
                out = self.model(inputs)
                donor_targets = targets[shuffle_idx]
                eval_indices = [b * K + PEAK_T for b in range(NUM_BLOCKS) if b * K + PEAK_T < targets.size(1)]
                patching_losses.append(criterion(out[:, eval_indices, :].transpose(1, 2), donor_targets[:, eval_indices]).item())
            h.remove()
            self.activations.clear()
            torch.cuda.empty_cache()

        self.ablation_results = {k: np.mean(v) for k, v in losses.items()}
        self.ablation_results["Patched (Donor Target)"] = np.mean(patching_losses)
        print("Ablation results (CE @ perturbation event):")
        for k, v in self.ablation_results.items():
            print(f"  {k}: {v:.4f}")

    def plot_results(self):
        print("\n--- Generating plots ---")
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))

        sns.heatmap(self.belief_r2, ax=axes[0, 0], annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
        axes[0, 0].set_title("Belief Test R2 (alpha=1.0)")
        axes[0, 0].set_xlabel("Timestep")
        axes[0, 0].set_ylabel("Layer")

        sns.heatmap(self.bv_angle_grid, ax=axes[0, 1], annot=True, fmt=".0f", cmap="coolwarm", vmin=0, vmax=90)
        axes[0, 1].set_title(f"Belief-vs-Velocity Angle (null 5th pct={self.null_bv_threshold:.1f})")
        axes[0, 1].set_xlabel("Timestep")

        r2_zero = np.array([[self.evals_zero["b"][l][t].get_r2() for t in range(N)] for l in range(NUM_LAYERS)])
        sns.heatmap(r2_zero, ax=axes[0, 2], annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
        axes[0, 2].set_title("Belief Test R2 (alpha=1e-6 control)")

        axes[1, 0].plot(range(NUM_LAYERS), self.angle_vs_l, marker='o')
        axes[1, 0].axhline(self.null_bb_threshold, color='r', linestyle='--', label="null 5th pct")
        axes[1, 0].set_title(f"Drift vs Layer: {self.drift_label_l}")
        axes[1, 0].set_ylim(0, 90)
        axes[1, 0].legend()

        axes[1, 1].plot(range(N), self.angle_vs_t, marker='o')
        axes[1, 1].axhline(self.null_bb_threshold, color='r', linestyle='--', label="null 5th pct")
        axes[1, 1].set_title(f"Drift vs Timestep: {self.drift_label_t}")
        axes[1, 1].set_ylim(0, 90)
        axes[1, 1].legend()

        names = list(self.ablation_results.keys())
        values = list(self.ablation_results.values())
        axes[1, 2].bar(names, values, color=['gray', 'red', 'blue', 'orange', 'purple'])
        axes[1, 2].set_title("Causal Ablation Loss (CE @ perturbation)")
        axes[1, 2].tick_params(axis='x', rotation=45)

        sns.histplot(self.null_dist_bb, ax=axes[2, 0], bins=30, stat="density")
        axes[2, 0].axvline(self.null_bb_threshold, color='r', linestyle='--')
        axes[2, 0].set_title("Null: belief-vs-belief (2D vs 2D)")

        sns.histplot(self.null_dist_bv, ax=axes[2, 1], bins=30, stat="density")
        axes[2, 1].axvline(self.null_bv_threshold, color='r', linestyle='--')
        axes[2, 1].set_title("Null: belief-vs-velocity (2D vs 1D)")

        axes[2, 2].axis('off')
        summary = (
            f"Reference cell: layer {self.ref_l}, t={self.ref_t}\n"
            f"Belief R2: {self.belief_r2[self.ref_l, self.ref_t]:.4f}\n"
            f"Drift (layer): {self.drift_label_l}\n"
            f"Drift (timestep): {self.drift_label_t}\n\n"
            "Ablation deltas vs baseline:\n"
        )
        base = self.ablation_results["Baseline"]
        for k in ["Belief Ablated", "Velocity Ablated", "Random Ablated"]:
            summary += f"  {k}: {self.ablation_results[k] - base:+.4f}\n"
        axes[2, 2].text(0.0, 0.5, summary, fontsize=11, va='center', family='monospace')

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'subspace_drift_analysis_fixed.png')
        plt.savefig(output_path, dpi=300)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    pipeline = PipelineRunner()
    pipeline.run_phase_0_cv_check()
    pipeline.run_training_sweeps()
    pipeline.run_testing_sweeps()
    pipeline.analyze_drift()
    pipeline.run_causal_ablations()
    pipeline.plot_results()