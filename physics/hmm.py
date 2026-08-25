import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from torch.utils.data import IterableDataset
from scipy.linalg import null_space


# --- Helper Functions ---
def stringify(list):
    return "".join(' ' + e for e in list)

def destringify(string):
    return [c for c in string if c != ' ']

# ==========================================
# 1. MESS 3 (Triangle Topology)
# ==========================================
class Mess3Process:
    def __init__(self, alpha=0.7, x=0.1):
        # alpha: Probability of emitting the dominant token
        # x: Probability of switching to ANY neighbor
        self.x = np.clip(x, 0.0, 0.5)
        self.alpha = np.clip(alpha, 0.0, 1.0)
        
        self.tokens = ['0', '1', '2']
        self.num_states = 3
        
        # 1. Base Transition Matrix (Fully Connected)
        self.T_base = np.array([
            [1 - 2*self.x, self.x, self.x],        # From State 0
            [self.x, 1 - 2*self.x, self.x],        # From State 1
            [self.x, self.x, 1 - 2*self.x]         # From State 2
        ])
        
        # 2. Emission Matrix
        noise = (1.0 - self.alpha) / 2.0
        self.E = np.array([
            [self.alpha, noise, noise],  # State 0 -> A
            [noise, self.alpha, noise],  # State 1 -> B
            [noise, noise, self.alpha]   # State 2 -> C
        ])
        
        # 3. Joint Matrices
        self.matrices = {}
        for token_idx, token in enumerate(self.tokens):
            emission_probs = self.E[:, token_idx]
            self.matrices[token] = self.T_base * emission_probs.reshape(-1, 1)

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=int)
        observations = []
        current_state = np.random.choice(self.num_states)
        
        for t in range(length):
            states[t] = current_state
            
            # Emit
            token = np.random.choice(self.tokens, p=self.E[current_state])
            observations.append(token)
            
            # Transition
            current_state = np.random.choice(self.num_states, p=self.T_base[current_state])

        return states, observations

    def optimal(self, observations):
        T_len = len(observations)
        belief_states = np.zeros((T_len, self.num_states))
        b = np.array([1/3, 1/3, 1/3])
        
        for t, token in enumerate(observations):
            if token in self.matrices:
                b = np.dot(b, self.matrices[token])
                norm = np.sum(b)
                if norm > 0: b /= norm
                else: b = np.ones(3)/3
            belief_states[t] = b
            
            
        return belief_states
    
    def generate_batch(self, batch_size, length):
        states = np.zeros((batch_size, length), dtype=int)
        obs_array = np.empty((batch_size, length), dtype=object)
        current_states = np.random.choice(self.num_states, size=batch_size)
        
        E_cdf = np.cumsum(self.E, axis=1)
        T_cdf = np.cumsum(self.T_base, axis=1)
        tokens_np = np.array(self.tokens)
        
        for t in range(length):
            states[:, t] = current_states
            
            # Emit
            rand_vals_E = np.random.rand(batch_size, 1)
            current_E_cdf = E_cdf[current_states]
            emissions = (rand_vals_E > current_E_cdf).sum(axis=1)
            obs_array[:, t] = tokens_np[emissions]
            
            # Transition
            rand_vals_T = np.random.rand(batch_size, 1)
            current_T_cdf = T_cdf[current_states]
            current_states = (rand_vals_T > current_T_cdf).sum(axis=1)
            
        return states, obs_array

    def optimal_batch(self, observations_batch):
        batch_size, length = observations_batch.shape
        belief_states = np.zeros((batch_size, length, self.num_states))
        b = np.full((batch_size, self.num_states), 1.0 / self.num_states)
        
        for t in range(length):
            tokens_t = observations_batch[:, t]
            b_new = np.zeros_like(b)
            for token in self.tokens:
                mask = (tokens_t == token)
                if np.any(mask):
                    b_new[mask] = np.dot(b[mask], self.matrices[token])
            
            b = b_new
            norm = np.sum(b, axis=1, keepdims=True)
            zero_mask = (norm == 0).squeeze(-1)
            if np.any(zero_mask):
                b[zero_mask] = 1.0 / self.num_states
                norm[zero_mask] = 1.0
            
            b /= norm
            belief_states[:, t, :] = b
            
        return belief_states
    
    def plot_simplex(self, observations, hidden_states=None, show_triangle=True, title="Mess3 Optimal Belief Simplex"):
        beliefs = self.optimal(observations)
        X = beliefs[:, 1] + 0.5 * beliefs[:, 2]
        Y = (np.sqrt(3) / 2) * beliefs[:, 2]
        
        plt.figure(figsize=(10, 8))
        if show_triangle:
            plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=2, alpha=0.6)
            plt.text(-0.05, -0.05, 'State 0 (A)', ha='center', fontsize=12)
            plt.text(1.05, -0.05, 'State 1 (B)', ha='center', fontsize=12)
            plt.text(0.5, np.sqrt(3)/2 + 0.05, 'State 2 (C)', ha='center', fontsize=12)
            plt.axis('equal')
            plt.axis('off')

        if hidden_states is not None:
            colors = ['#E41A1C', '#4DAF4A', '#377EB8'] 
            cmap = ListedColormap(colors)
            plt.scatter(X, Y, s=1, c=hidden_states, cmap=cmap, alpha=0.6)
        else:
            plt.scatter(X, Y, s=1, c='blue', alpha=0.5)
            
        plt.title(f"{title}\n(alpha={self.alpha}, x={self.x})")
        plt.tight_layout()
        plt.show()

class Mess3Dataset(Dataset):
    def __init__(self, process: Mess3Process, num_samples, tokenizer, max_length=1024):
        self.process = process
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.generate_sequences(num_samples, max_length)
        
        self.token_map = {
            'A': tokenizer.encode(" A")[0],
            'B': tokenizer.encode(" B")[0],
            'C': tokenizer.encode(" C")[0]
        }
        
    def generate_sequences(self, num_samples, size):
        sequences = []
        for _ in tqdm(range(num_samples), desc="Gen Mess3"):
            hidden, obs = self.process.generate_sequence(size)
            beliefs = self.process.optimal(obs)
            encoding = self.tokenizer(stringify(obs), truncation=True, return_tensors='pt')
            
            sample = {
                "hidden": hidden.tolist(),
                "raw_obs": stringify(obs),
                "beliefs": beliefs,
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            }
            sequences.append(sample)
        return sequences

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


# ==========================================
# 2. MESS 5 (Ring Topology)
# ==========================================
class Mess5RingProcess:
    def __init__(self, alpha=0.7, x=0.1):
        # alpha: Probability of emitting the dominant token
        # x: Probability of switching to LEFT or RIGHT neighbor only
        self.x = np.clip(x, 0.0, 0.5)
        self.alpha = np.clip(alpha, 0.0, 1.0)
        
        self.tokens = ['A', 'B', 'C', 'D', 'E']
        self.num_states = 5
        
        # 1. Base Transition Matrix (Ring Topology)
        # S0 connects to S4 (Left) and S1 (Right)
        self.T_base = np.zeros((5, 5))
        for i in range(5):
            self.T_base[i, i] = 1 - 2 * self.x       # Stay
            self.T_base[i, (i + 1) % 5] = self.x     # Right
            self.T_base[i, (i - 1) % 5] = self.x     # Left

        # 2. Emission Matrix
        # S0->A, S1->B, S2->C, S3->D, S4->E
        noise = (1.0 - self.alpha) / 4.0
        self.E = np.full((5, 5), noise)
        np.fill_diagonal(self.E, self.alpha)
        
        # 3. Joint Matrices
        self.matrices = {}
        for token_idx, token in enumerate(self.tokens):
            emission_probs = self.E[:, token_idx]
            self.matrices[token] = self.T_base * emission_probs.reshape(-1, 1)

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=int)
        observations = []
        current_state = np.random.choice(self.num_states)
        
        for t in range(length):
            states[t] = current_state
            
            # Emit
            token = np.random.choice(self.tokens, p=self.E[current_state])
            observations.append(token)
            
            # Transition
            
            current_state = np.random.choice(self.num_states, p=self.T_base[current_state])

        return states, observations

    def optimal(self, observations):
        T_len = len(observations)
        belief_states = np.zeros((T_len, self.num_states))
        b = np.ones(5) / 5
        
        for t, token in enumerate(observations):
            if token in self.matrices:
                b = np.dot(b, self.matrices[token])
                norm = np.sum(b)
                if norm > 0: b /= norm
                else: b = np.ones(5)/5
            belief_states[t] = b
            
        return belief_states
    
    def plot_pca(self, observations, hidden_states=None, title="Mess5 Optimal Belief PCA"):
        """
        Projects 5D beliefs to 2D using PCA. 
        For Ring Topology, this should reveal a circular structure.
        """
        beliefs = self.optimal(observations)
        
        pca = PCA(n_components=2)
        projected = pca.fit_transform(beliefs)
        
        plt.figure(figsize=(8, 8))
        
        if hidden_states is not None:
            # 5 Colors for 5 States
            colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
            cmap = ListedColormap(colors)
            scatter = plt.scatter(projected[:, 0], projected[:, 1], s=5, c=hidden_states, cmap=cmap, alpha=0.6)
            plt.legend(handles=scatter.legend_elements()[0], labels=[f'State {i}' for i in range(5)])
        else:
            plt.scatter(projected[:, 0], projected[:, 1], s=5, c=range(len(beliefs)), cmap='viridis', alpha=0.6)
            plt.colorbar(label="Time Step")

        plt.title(f"{title}\n(alpha={self.alpha}, x={self.x})")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

class Mess5Dataset(Dataset):
    def __init__(self, process: Mess5RingProcess, num_samples, tokenizer, max_length=1024):
        self.process = process
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.generate_sequences(num_samples, max_length)
        
        # Token Map for 5 tokens
        self.token_map = {
            'A': tokenizer.encode(" A")[0],
            'B': tokenizer.encode(" B")[0],
            'C': tokenizer.encode(" C")[0],
            'D': tokenizer.encode(" D")[0],
            'E': tokenizer.encode(" E")[0]
        }
        
    def generate_sequences(self, num_samples, size):
        sequences = []
        for _ in tqdm(range(num_samples), desc="Gen Mess5"):
            hidden, obs = self.process.generate_sequence(size)
            beliefs = self.process.optimal(obs)
            encoding = self.tokenizer(stringify(obs), truncation=True, return_tensors='pt')
            
            sample = {
                "hidden": hidden.tolist(),
                "raw_obs": stringify(obs),
                "beliefs": beliefs,
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            }
            sequences.append(sample)
        return sequences

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
    
    

class RRXORProcess:
    def __init__(self, alpha=1, random_state=False):
        # 5 States: [S_S, S_0, S_1, S_T, S_F]
        self.states = ['S_S', 'S_0', 'S_1', 'S_T', 'S_F']
        self.num_states = 5
        self.random_state = random_state
        self.alpha = alpha
        # Transition Matrices T(s'|s, x)
        self.T0 = np.zeros((5, 5))
        self.T1 = np.zeros((5, 5))
        
        # --- Transition Rules ---
        # S_S -> S_0 (0), S_S -> S_1 (1)
        self.T0[0, 1] = 0.5; self.T1[0, 2] = 0.5
        
        # S_0 -> S_F (0), S_0 -> S_T (1)
        self.T0[1, 4] = 0.5; self.T1[1, 3] = 0.5
        
        # S_1 -> S_T (0), S_1 -> S_F (1)
        self.T0[2, 3] = 0.5; self.T1[2, 4] = 0.5
        
        # S_T -> S_S (1), S_F -> S_S (0)
        self.T1[3, 0] = alpha; self.T0[4, 0] = alpha
        self.T1[4, 0] = 1- alpha; self.T0[3, 0] = 1 - alpha
        self.matrices = {
            '0': self.T0,
            '1': self.T1
            }
        self.tokens = list(self.matrices.keys())

    def generate_sequence(self, length):
        states_seq = np.zeros(length, dtype=int)
        observations = []
        current_state = 0 
        if self.random_state:
            current_state = np.random.randint(0, 5)
        
        for t in range(length):
            states_seq[t] = current_state
            
            p0 = np.sum(self.T0[current_state])
            p1 = np.sum(self.T1[current_state])
            probs = np.array([p0, p1]) / (p0 + p1)
            
            token = np.random.choice(self.tokens, p=probs)
            observations.append(token)
            
            if token == '0': next_state = np.argmax(self.T0[current_state])
            else: next_state = np.argmax(self.T1[current_state])
            current_state = next_state
            
        return states_seq, observations

    def optimal(self, observations):
        """Computes the belief state evolution."""
        T_len = len(observations)
        belief_states = np.zeros((T_len, self.num_states))
        b = np.ones(self.num_states) / self.num_states # Start uniform
        
        for t, token in enumerate(observations):
            if token in self.matrices:
                b = np.dot(b, self.matrices[token])
                norm = np.sum(b)
                if norm > 1e-9: b /= norm
                else: b = np.ones(self.num_states) / self.num_states
            belief_states[t] = b
            
        return belief_states
        
    def generate_batch(self, batch_size, length):
        states = np.zeros((batch_size, length), dtype=int)
        obs_array = np.empty((batch_size, length), dtype=object)
        current_states = np.zeros(batch_size, dtype=int)
        if self.random_state:
            current_states = np.random.randint(0, 5, size=batch_size)
            
        tokens_np = np.array(self.tokens)
        
        for t in range(length):
            states[:, t] = current_states
            
            p0 = self.T0[current_states].sum(axis=1)
            p1 = self.T1[current_states].sum(axis=1)
            p0_norm = p0 / (p0 + p1)
            
            # Emit
            rand_vals_E = np.random.rand(batch_size)
            # 0 if rand < p0_norm, 1 otherwise
            emissions = (rand_vals_E > p0_norm).astype(int) 
            obs_array[:, t] = tokens_np[emissions]
            
            # Transition
            # Argmax gives the deterministic destination state for the corresponding token
            next_state_0 = np.argmax(self.T0[current_states], axis=1)
            next_state_1 = np.argmax(self.T1[current_states], axis=1)
            
            current_states = np.where(emissions == 0, next_state_0, next_state_1)
            
        return states, obs_array

    def optimal_batch(self, observations_batch):
        batch_size, length = observations_batch.shape
        belief_states = np.zeros((batch_size, length, self.num_states))
        b = np.full((batch_size, self.num_states), 1.0 / self.num_states)
        
        for t in range(length):
            tokens_t = observations_batch[:, t]
            b_new = np.zeros_like(b)
            for token in self.tokens:
                mask = (tokens_t == token)
                if np.any(mask):
                    b_new[mask] = np.dot(b[mask], self.matrices[token])
            
            b = b_new
            norm = np.sum(b, axis=1, keepdims=True)
            zero_mask = (norm == 0).squeeze(-1)
            if np.any(zero_mask):
                b[zero_mask] = 1.0 / self.num_states
                norm[zero_mask] = 1.0
            
            b /= norm
            belief_states[:, t, :] = b
            
        return belief_states
    
    def get_helmert_matrix(self, n):
        H = np.zeros((n-1, n))
        for i in range(n-1):
            H[i, i] = -np.sqrt((n - 1-i) / (n - i))
            for j in range(i + 1, n):
                H[i, j] = 1.0 / np.sqrt((n - 1 - i) * (n - i))
        return H.T
    
    def project_hermit(self, beliefs):
        X = beliefs.copy()
        H5 = self.get_helmert_matrix(5)
        X = np.dot(X, H5)
        
        # Step 2: 4D -> 3D (Orthogonal to 1,1,1,1)
        H4 = self.get_helmert_matrix(4) # Shape (4, 3)
        X = np.dot(X, H4)
        
        # Step 3: 3D -> 2D (Orthogonal to 1,1,1)
        H3 = self.get_helmert_matrix(3) # Shape (3, 2)
        X = np.dot(X, H3)
        
        return X
    def project_simplex_pca(self, beliefs):
        """
        1. Projects 5D beliefs onto the 4D simplex hyperplane defined by normal (1,1,1,1,1).
        2. Applies PCA to reduce from 4D to 2D.
        """
        # Step 1: Get Basis for the Simplex (Orthogonal to ones vector)
        # We find the null space of the 1x5 matrix [[1, 1, 1, 1, 1]]
        # This returns a (5, 4) orthonormal matrix.
        ones_vector = np.ones((1, 5))
        basis_4d = null_space(ones_vector) # Shape (5, 4)
        
        # Step 2: Project 5D -> 4D isometric coordinates
        # beliefs (N, 5) @ basis (5, 4) -> (N, 4)
        X_4d = np.dot(beliefs, basis_4d)
        
        # Step 3: PCA 4D -> 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_4d)
        
        return X_2d, pca

    def generate_runs(self, num_sequences=100, length=100):
        all_beliefs = []
        all_obs = []
        print(f"Generating {num_sequences} sequences of length {length}...")
        for _ in range(num_sequences):
            _, obs = self.generate_sequence(length)
            beliefs = self.optimal(obs)
            all_beliefs.append(beliefs)
            all_obs.extend(obs)
            
        return np.vstack(all_beliefs), all_obs
    
    def project_symmetric(self, beliefs):
        P = np.array([
            [0, 0],
            [-1, 1],
            [1, 1], 
            [1, -1], 
            [-1, -1]
        ], dtype=float)
        
        projected = np.dot(beliefs, P)
        return projected
    def project_mds(self, beliefs):
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto')
        return mds.fit_transform(beliefs)
    def project_to_pentagon(self, beliefs):
        angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 6)[:-1]
        Vx = np.cos(angles)
        Vy = np.sin(angles)
        
        X = np.dot(beliefs, Vx)
        Y = np.dot(beliefs, Vy)
        return X, Y, Vx, Vy
    def get_belief_clusters(self, belief_states, n_clusters=36):
        """
        Helper: Identifies the 36 distinct belief states using K-Means.
        Returns cluster labels and the fitted KMeans object.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(belief_states)
        return labels, kmeans

    def plot_belief_clusters(self, num_sequences=1000, seq_len=1000):
        beliefs, _ = self.generate_runs(num_sequences, seq_len)
        unique_b = np.unique(np.round(beliefs, 32), axis=0)
        print(f"Total Belief States: {len(beliefs)}")
        print(f"Unique Belief States Found: {len(unique_b)}")
        n_clusters = min(36, len(unique_b))
        print(unique_b)
        kmeans = KMeans(n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(beliefs)
        centroids = kmeans.cluster_centers_
        # print(centroids)
        # pca = PCA(n_components=2)
        # projected_centres = pca.fit_transform(centroids)
        # projected = pca.transform(beliefs)
        # projected = pca.fit_transform(unique_b)
        # projected = self.project_hermit(beliefs)
        projected = self.project_symmetric(centroids)
        plt.figure(figsize=(12, 10))
        pure_map = {
            0: {'color': '#7f8c8d', 'label': '$S_S$ (Sync)', 'marker': 'D'}, 
            1: {'color': '#E41A1C', 'label': '$S_0$', 'marker': 'o'},        
            2: {'color': '#4DAF4A', 'label': '$S_1$', 'marker': 'o'},        
            3: {'color': '#377EB8', 'label': '$S_T$', 'marker': 's'},        
            4: {'color': '#FF7F00', 'label': '$S_F$', 'marker': 's'}         
        }
        is_pure = np.zeros(n_clusters, dtype=bool)
        pure_indices = {}
        eye = np.eye(5)
        for i in range(5):
            dists = np.linalg.norm(centroids - eye[i], axis=1)
            closest_idx = np.argmin(dists)
            if dists[closest_idx] < 0.01:
                is_pure[closest_idx] = True
                pure_indices[closest_idx] = i 
                
        non_pure_mask = ~is_pure
        print(non_pure_mask)
        # Use a distinct colormap to separate 36 clusters
        # 'turbo' or 'nipy_spectral' provide high contrast for many classes
        cmap = cm.get_cmap('nipy_spectral', n_clusters)
        
        # scatter = plt.scatter(
        #     projected[:, 0], projected[:, 1], 
        #     c=labels, 
        #     cmap=cmap, 
        #     s=15, 
        #     alpha=0.7
        # )
        scatter = plt.scatter(
            projected[non_pure_mask, 0],
            projected[non_pure_mask, 1],
            c = cm.nipy_spectral(np.linspace(0,1, np.sum(non_pure_mask))),
            s=100,
            alpha=0.5,
            edgecolors='none'
        )
        for c_idx, state_idx in pure_indices.items():
            props = pure_map[state_idx]
            plt.scatter(
                projected[c_idx, 0], 
                projected[c_idx, 1], 
                c=props['color'], 
                s=250, 
                marker=props['marker'],
                edgecolors='black',
                linewidth=1.5,
                zorder=10,
                label=props['label']
            )
            plt.text(
                projected[c_idx, 0], 
                projected[c_idx, 1] + 0.04, 
                props['label'], 
                fontsize=12, 
                fontweight='bold', 
                ha='center',
                zorder=11
            )
        plt.colorbar(scatter, ticks=range(0, 36, 5), label="Belief Cluster ID")
        plt.title(f"RRXOR Belief Geometry (Batch Mode)\n{num_sequences} x {seq_len} steps | Found {len(unique_b)} Unique States")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return kmeans

class RRXORDataset(Dataset):
    def __init__(self, process: RRXORProcess, num_samples, tokenizer, max_length=1024):
        self.process = process
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-fit a clusterer on a reference sequence to ensure consistent labeling
        print("Pre-calculating belief clusters...")
        _, ref_obs = process.generate_sequence(10000)
        ref_beliefs = process.optimal(ref_obs)
        _, self.kmeans = process.get_belief_clusters(ref_beliefs, n_clusters=36)
        
        self.data = self.generate_sequences(num_samples, max_length)
        
    def generate_sequences(self, num_samples, size):
        sequences = []
        for _ in tqdm(range(num_samples), desc="Gen RRXOR"):
            hidden, obs = self.process.generate_sequence(size)
            beliefs = self.process.optimal(obs)
            
            # Assign belief cluster IDs
            belief_ids = self.kmeans.predict(beliefs)
            
            text = stringify(obs)
            encoding = self.tokenizer(text, truncation=True, max_length=size, return_tensors='pt')
            
            sample = {
                "hidden": hidden.tolist(),
                "belief_ids": belief_ids, # The 'Y' target for a probe training on discrete beliefs
                "beliefs": beliefs,       # The continuous target
                "raw_obs": text,
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            }
            sequences.append(sample)
        return sequences

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# --- Helper Function ---
def stringify(obs):
    """Converts a list of tokens into a space-separated string."""
    return "".join([" " + str(x) for x in obs])

# --- Nemo Process (Model b) ---
class NemoProcess:
    def __init__(self, p=0.5, x = 0.0):
        # p: Probability of emitting '0' and staying in State A
        self.x = np.clip(x, 0.0, 0.5)
        self.p = np.clip(p, 0.0, 1.0)
        self.tokens = ['0', '1']
        self.num_states = 3
        
        # State mapping: A=0, B=1, C=2
        # Transition/Emission Definitions (Unifilar):
        # State 0 (A): 0->0 (p), 1->1 (1-p)
        # State 1 (B): 1->2 (1.0)
        # State 2 (C): 0->0 (0.5), 1->1 (0.5)
        def noisy_row(prob_emit, target_idx):
            row = np.full(3, self.x)
            row[target_idx] = 1 - 2 * self.x
            return row * prob_emit
        # Joint Matrices M[token] where M_ij = P(S_next=j, Token=token | S_curr=i)
        self.matrices = {}
        
        # Matrix for Token '0'
        # A emits 0 -> A (p)
        # B emits 0 -> Impossible (0)
        # C emits 0 -> A (0.5)
        self.matrices['0'] = np.array([
            noisy_row(self.p, 0),  # A -> A
            noisy_row(0.0, 1),     # B -> ?
            noisy_row(0.5, 0)      # C -> A
        ])
        
        # Matrix for Token '1'
        # A emits 1 -> B (1-p)
        # B emits 1 -> C (1.0)
        # C emits 1 -> B (0.5)
        self.matrices['1'] = np.array([
             noisy_row(1.0 - self.p, 1),
             noisy_row(1.0, 2),
             noisy_row(0.5, 1)# C -> B
        ])
        print(self.matrices)

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=int)
        observations = []
        # Start randomly or at A (usually 0 is a good default, but we'll choose random for generality)
        current_state = np.random.choice(self.num_states)
        
        for t in range(length):
            states[t] = current_state
            
            # Determine probabilities of emitting '0' or '1' from current state
            p_0 = np.sum(self.matrices['0'][current_state])
            p_1 = np.sum(self.matrices['1'][current_state])
            
            # Normalize (handle states that might have 0 sum if strictly defined, though here all sum to 1)
            norm = p_0 + p_1
            if norm == 0: norm = 1.0 
            
            # Sample token
            token = np.random.choice(self.tokens, p=[p_0/norm, p_1/norm])
            observations.append(token)
            
            # Transition deterministically based on token (Unifilar property)
            # We look at the row for the current state in the matrix for the chosen token
            next_state_probs = self.matrices[token][current_state]
            next_state_probs = next_state_probs / np.sum(next_state_probs) # Should be 1-hot for unifilar
            current_state = np.random.choice(self.num_states, p=next_state_probs)

        return states, observations

    def optimal(self, observations):
        T_len = len(observations)
        belief_states = np.zeros((T_len, self.num_states))
        b = np.array([1/3, 1/3, 1/3]) # Initial uniform belief
        
        for t, token in enumerate(observations):
            if token in self.matrices:
                b = np.dot(b, self.matrices[token])
                norm = np.sum(b)
                if norm > 0: b /= norm
                else: b = np.ones(3)/3 # Reset if lost
            belief_states[t] = b
            
        return belief_states
    
    def plot_simplex(self, observations, hidden_states=None, show_triangle=True, title="Nemo Optimal Belief Simplex"):
        beliefs = self.optimal(observations)
        X = beliefs[:, 1] + 0.5 * beliefs[:, 2]
        Y = (np.sqrt(3) / 2) * beliefs[:, 2]
        
        plt.figure(figsize=(10, 8))
        if show_triangle:
            plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=2, alpha=0.6)
            plt.text(-0.05, -0.05, 'State A (0)', ha='center', fontsize=12)
            plt.text(1.05, -0.05, 'State B (1)', ha='center', fontsize=12)
            plt.text(0.5, np.sqrt(3)/2 + 0.05, 'State C (2)', ha='center', fontsize=12)
            plt.axis('equal')
            plt.axis('off')

        if hidden_states is not None:
            colors = ['#E41A1C', '#4DAF4A', '#377EB8'] 
            cmap = ListedColormap(colors)
            plt.scatter(X, Y, s=30, c=hidden_states, cmap=cmap, alpha=0.6)
        else:
            plt.scatter(X, Y, s=30, c='blue', alpha=0.5)
            
        plt.title(f"{title}\n(p={self.p})")
        plt.tight_layout()
        plt.show()

# --- Two-Step Erase Process (Model c) ---
class TwoStepEraseProcess:
    def __init__(self, p0=0.5, q0=0.5, r0=0.5, x=0.0):
        # Parameters represent P(emit 0) for states A, B, C respectively
        self.p0 = np.clip(p0, 0.0, 1.0)
        self.q0 = np.clip(q0, 0.0, 1.0)
        self.r0 = np.clip(r0, 0.0, 1.0)
        
        self.p1 = 1.0 - self.p0
        self.q1 = 1.0 - self.q0
        self.r1 = 1.0 - self.r0
        
        self.tokens = ['0', '1']
        self.num_states = 3
        self.x=x
        # State mapping: A=0, B=1, C=2
        self.matrices = {}
        
        # Matrix for Token '0' (Reset/Erase paths)
        # A(0) -> A (p0)
        # B(1) -> A (q0)
        # C(2) -> A (r0)
        def noisy_row(prob_emit, target_idx):
            row = np.full(3, self.x)
            row[target_idx] = 1.0 - 2*self.x
            return row * prob_emit
        self.matrices['0'] = np.array([
            noisy_row(self.p0, 0),
            noisy_row(self.q0, 0),
            noisy_row(self.r0, 0)
        ])
        
        # Matrix for Token '1' (Progress paths)
        # A(0) -> B (p1)
        # B(1) -> C (q1)
        # C(2) -> C (r1)
        self.matrices['1'] = np.array([
            noisy_row(self.p1, 1),
            noisy_row(self.q1, 2),
            noisy_row(self.r1, 2)
        ])

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=int)
        observations = []
        current_state = np.random.choice(self.num_states)
        
        for t in range(length):
            states[t] = current_state
            
            # Determine probabilities
            p_0 = np.sum(self.matrices['0'][current_state])
            p_1 = np.sum(self.matrices['1'][current_state])
            
            # Sample token
            token = np.random.choice(self.tokens, p=[p_0, p_1])
            observations.append(token)
            
            # Transition
            next_state_probs = self.matrices[token][current_state]
            next_state_probs = next_state_probs / np.sum(next_state_probs)
            current_state = np.random.choice(self.num_states, p=next_state_probs)

        return states, observations

    def optimal(self, observations):
        T_len = len(observations)
        belief_states = np.zeros((T_len, self.num_states))
        b = np.array([1/3, 1/3, 1/3])
        
        for t, token in enumerate(observations):
            if token in self.matrices:
                b = np.dot(b, self.matrices[token])
                norm = np.sum(b)
                if norm > 0: b /= norm
                else: b = np.ones(3)/3
            belief_states[t] = b
            
        return belief_states
    
    def plot_simplex(self, observations, hidden_states=None, show_triangle=True, title="Two-Step Erase Optimal Belief Simplex"):
        beliefs = self.optimal(observations)
        X = beliefs[:, 1] + 0.5 * beliefs[:, 2]
        Y = (np.sqrt(3) / 2) * beliefs[:, 2]
        
        plt.figure(figsize=(10, 8))
        if show_triangle:
            plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=2, alpha=0.6)
            plt.text(-0.05, -0.05, 'State A (0)', ha='center', fontsize=12)
            plt.text(1.05, -0.05, 'State B (1)', ha='center', fontsize=12)
            plt.text(0.5, np.sqrt(3)/2 + 0.05, 'State C (2)', ha='center', fontsize=12)
            plt.axis('equal')
            plt.axis('off')

        if hidden_states is not None:
            colors = ['#E41A1C', '#4DAF4A', '#377EB8'] 
            cmap = ListedColormap(colors)
            plt.scatter(X, Y, s=30, c=hidden_states, cmap=cmap, alpha=0.6)
        else:
            plt.scatter(X, Y, s=30, c='blue', alpha=0.5)
            
        plt.title(f"{title}\n(p0={self.p0}, q0={self.q0}, r0={self.r0})")
        plt.tight_layout()
        plt.show()

# --- Datasets ---

class NemoDataset(Dataset):
    def __init__(self, process: NemoProcess, num_samples, tokenizer, max_length=1024):
        self.process = process
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.generate_sequences(num_samples, max_length)
        
    def generate_sequences(self, num_samples, size):
        sequences = []
        for _ in tqdm(range(num_samples), desc="Gen Nemo"):
            hidden, obs = self.process.generate_sequence(size)
            beliefs = self.process.optimal(obs)
            encoding = self.tokenizer(stringify(obs), truncation=True, return_tensors='pt')
            
            sample = {
                "hidden": hidden.tolist(),
                "raw_obs": stringify(obs),
                "beliefs": beliefs,
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            }
            sequences.append(sample)
        return sequences

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class TwoStepEraseDataset(Dataset):
    def __init__(self, process: TwoStepEraseProcess, num_samples, tokenizer, max_length=1024):
        self.process = process
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.generate_sequences(num_samples, max_length)
        
    def generate_sequences(self, num_samples, size):
        sequences = []
        for _ in tqdm(range(num_samples), desc="Gen TwoStep"):
            hidden, obs = self.process.generate_sequence(size)
            beliefs = self.process.optimal(obs)
            encoding = self.tokenizer(stringify(obs), truncation=True, return_tensors='pt')
            
            sample = {
                "hidden": hidden.tolist(),
                "raw_obs": stringify(obs),
                "beliefs": beliefs,
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            }
            sequences.append(sample)
        return sequences

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
    
    
class RRXOROptimal:
    def __init__(self, alpha=1, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.num_states = 5
        self.alpha = alpha
        
        self.T0 = torch.zeros((5, 5), device=device, dtype=dtype)
        self.T1 = torch.zeros((5,5), device=device, dtype=dtype)
        # S_S -> S_0 (0), S_S -> S_1 (1)
        self.T0[0, 1] = 0.5; self.T1[0, 2] = 0.5
        
        # S_0 -> S_F (0), S_0 -> S_T (1)
        self.T0[1, 4] = 0.5; self.T1[1, 3] = 0.5
        
        # S_1 -> S_T (0), S_1 -> S_F (1)
        self.T0[2, 3] = 0.5; self.T1[2, 4] = 0.5
        
        # S_T -> S_S (1), S_F -> S_S (0)
        self.T1[3, 0] = alpha;       self.T0[4, 0] = alpha
        self.T1[4, 0] = 1.0 - alpha; self.T0[3, 0] = 1.0 - alpha
        
        self.next_state_0 = torch.argmax(self.T0, dim=1)
        self.next_state_1 = torch.argmax(self.T1, dim=1)
        sum_t0 = self.T0.sum(dim=1)
        sum_t1 = self.T1.sum(dim=1)
        denom = sum_t0 + sum_t1
        denom[denom == 0] = 1.0
        self.p_token0 = sum_t0 /denom
        self.T_stack = torch.stack([self.T0, self.T1])
        
    def generate_sequence(self, length, batch_size = 1):
        current_state = torch.randint(0, self.num_states, (batch_size,), device=self.device)
        states_seq = torch.zeros((batch_size, length), dtype=torch.long, device=self.device)
        obs_seq = torch.zeros((batch_size, length), dtype = torch.long, device = self.device)
        for t in range(length):
            states_seq[:, t] = current_state
            probs_0 = self.p_token0[current_state]
            rand_vals = torch.rand(batch_size, device=self.device)
            tokens = (rand_vals >= probs_0).long()
            obs_seq[:, t] = tokens
            next_s_0 = self.next_state_0[current_state]
            next_s_1 = self.next_state_1[current_state]
            current_state = torch.where(tokens == 0, next_s_0, next_s_1)
        
        return states_seq, obs_seq
    
    def optimal(self, obs_seq):
        batch_size, length = obs_seq.shape
        b = torch.ones((batch_size, self.num_states), device=self.device, dtype=self.dtype)
        belief_states = torch.zeros((batch_size, length, self.num_states), device=self.device, dtype=self.dtype)
        
        for t in range(length): 
            tokens = obs_seq[:, t]
            current_Ts = self.T_stack.index_select(0, tokens)
            b_next = torch.bmm(b.unsqueeze(1), current_Ts).squeeze(1)
            norms = b_next.sum(dim=1, keepdim=True)
            mask = norms > 1e-9
            b = torch.where(mask, b_next / norms, torch.ones_like(b)/self.num_states)
            belief_states[:, t,:] = b
            
        return belief_states
    
    def project_symmetric(self, beliefs):
        P = np.array([
            [0,0],
            [-1,1],
            [1,1],
            [1,-1],
            [-1,-1]
        ], dtype=float)
        return np.dot(beliefs, P)
    
    def plot_belief_clusters(self, num_sequences=50, seq_len=1000):
        """
        Generates data on GPU, moves to CPU, clusters, and plots to verify logic.
        """
        print(f"Generating verification data: {num_sequences} seqs x {seq_len} len...")
        
        # 1. Generate Data (GPU Speed)
        with torch.inference_mode():
            _, obs = self.generate_sequence(seq_len, batch_size=num_sequences)
            beliefs_gpu = self.optimal(obs)
            
            # 2. Move to CPU/Numpy for Plotting
            # Flatten to [N_samples, 5]
            beliefs = beliefs_gpu.reshape(-1, self.num_states).cpu().numpy()

        # 3. Find Unique Beliefs (Verification Step)
        # Rounding needed to handle floating point noise
        unique_b = np.unique(np.round(beliefs, 6), axis=0)
        
        print(f"Total Samples: {len(beliefs)}")
        print(f"Unique Belief States Found: {len(unique_b)}")
        
        # 4. Clustering
        n_clusters = min(36, len(unique_b))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(beliefs)
        centroids = kmeans.cluster_centers_

        # 5. Projection
        projected = self.project_symmetric(centroids)
        
        # 6. Plotting Setup
        plt.figure(figsize=(10, 8))
        
        pure_map = {
            0: {'color': '#7f8c8d', 'label': '$S_S$ (Sync)', 'marker': 'D'}, 
            1: {'color': '#E41A1C', 'label': '$S_0$', 'marker': 'o'},        
            2: {'color': '#4DAF4A', 'label': '$S_1$', 'marker': 'o'},        
            3: {'color': '#377EB8', 'label': '$S_T$', 'marker': 's'},        
            4: {'color': '#FF7F00', 'label': '$S_F$', 'marker': 's'}         
        }

        # Identify Pure States (One-hot vectors)
        is_pure = np.zeros(n_clusters, dtype=bool)
        pure_indices = {}
        eye = np.eye(5)
        
        for i in range(5):
            # Check distance of each centroid to the pure state 'i'
            dists = np.linalg.norm(centroids - eye[i], axis=1)
            closest_idx = np.argmin(dists)
            if dists[closest_idx] < 0.01: # Threshold for float tolerance
                is_pure[closest_idx] = True
                pure_indices[closest_idx] = i 
        
        non_pure_mask = ~is_pure

        # 7. Draw "Messy" Clusters (Mixed Beliefs)
        if np.any(non_pure_mask):
            plt.scatter(
                projected[non_pure_mask, 0],
                projected[non_pure_mask, 1],
                c=cm.nipy_spectral(np.linspace(0, 1, np.sum(non_pure_mask))),
                s=100,
                alpha=0.6,
                edgecolors='none',
                label='Mixed Beliefs'
            )

        # 8. Draw Pure States (Distinct Markers)
        for c_idx, state_idx in pure_indices.items():
            props = pure_map[state_idx]
            plt.scatter(
                projected[c_idx, 0], 
                projected[c_idx, 1], 
                c=props['color'], 
                s=250, 
                marker=props['marker'],
                edgecolors='black',
                linewidth=1.5,
                zorder=10,
                label=props['label']
            )
            # Add text label above marker
            plt.text(
                projected[c_idx, 0], 
                projected[c_idx, 1] + 0.08, 
                props['label'], 
                fontsize=11, 
                fontweight='bold', 
                ha='center',
                zorder=11
            )

        plt.title(f"RRXOR Belief Geometry (GPU-Generated)\n{len(unique_b)} Unique States Found")
        plt.xlabel("Sym Projection 1")
        plt.ylabel("Sym Projection 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
class RRXORIterableDataset(IterableDataset):
    def __init__(self, process: RRXOROptimal, tokenizer, seq_len=1024, batch_size=32, num_clusters=36):
        self.process = process
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.internal_batch_size = batch_size
        
        self.token_id0 = tokenizer.encode("0", add_special_tokens=False)[0]
        self.tokeni_id1 = tokenizer.encode("1", add_special_tokens=False)[0]
        self.token_map = torch.tensor([self.token_id0, self.tokeni_id1], device=process.device)

    def generate_batch(self):
        hidden_states, obs_idx = self.process.generate_sequence(
            self.seq_len,
            batch_size=self.internal_batch_size
        )
        beliefs = self.process.optimal(obs_idx)
        input_ids = self.token_map[obs_idx]
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids" : input_ids,
            "attention_mask": attention_mask,
            "beliefs": beliefs,
            "hidden_states": hidden_states
        }
        
    def __iter__(self):
        while True: 
            batch_data = self.generate_batch()
            for i in range(self.internal_batch_size):
                yield {
                    "input_ids": batch_data["input_ids"][i],
                    "attention_mask": batch_data["attention_mask"][i],
                    "beliefs": batch_data["beliefs"][i]
                }
# if __name__ =='__main__': 
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     process = RRXOROptimal(device=device)
#     process.plot_belief_clusters()