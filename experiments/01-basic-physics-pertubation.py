import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.architecture import ContinuousTransformer, TokenTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 50
NUM_SAMPLES = 2000
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 1
THETA_RES = 1
VOCAB_SIZE = int(180 / THETA_RES) + 1
MAX_SEQ_LEN = 1024
OUTPUT_DIR = "experiments/outputs-01"


A = 1.0   # Amplitude
W = 0.1   # Frequency


class PendulumDataset(Dataset):
    def __init__(self, num_samples, seq_len, is_continuous=True):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.is_continuous = is_continuous
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        phi = np.random.uniform(0, 2 * np.pi)
        t = np.arange(self.seq_len + 1)
        
        theta_deg = 90 + 89.9 * np.sin(W * t + phi)
        v_theta_deg = np.abs(89.9 * W * np.cos(W * t + phi))
        
        state = np.zeros((self.seq_len + 1, 2), dtype=np.float32)
        state[theta_deg > 90, 1] = 1.0
        state[theta_deg <= 90, 0] = 1.0
        
        if self.is_continuous:
            angle_from_vertical = np.deg2rad(theta_deg - 90)
            x = A * np.sin(angle_from_vertical)
            y = A * (1 - np.cos(angle_from_vertical))
            coords = np.stack([x, y], axis=-1).astype(np.float32)
            
            return coords[:-1], coords[1:], v_theta_deg[:-1].astype(np.float32), state[:-1]
        else:
            tokens = np.floor(theta_deg / THETA_RES).astype(np.int64)
            return tokens[:-1], tokens[1:], v_theta_deg[:-1].astype(np.float32), state[:-1]

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def probe_and_plot(model, loader, model_type):
    model.eval()
    acts = []
    vels = []
    
    handle = model.blocks[2].register_forward_hook(get_activation('layer3'))
    
    with torch.no_grad():
        for inputs, targets, v_mag, state in loader:
            inputs = inputs.to(device)
            model(inputs)
            act = activations['layer3']
            acts.append(act.reshape(-1, EMBED_DIM).cpu().numpy())
            vels.append(v_mag.reshape(-1).numpy())
            
    handle.remove()
    
    X = np.concatenate(acts, axis=0)
    y = np.concatenate(vels, axis=0)
    
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    score = reg.score(X, y)
    
    plt.figure()
    plt.scatter(y, y_pred, alpha=0.1, color='blue', s=2)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Ground Truth Velocity')
    plt.ylabel('Probed Predicted Velocity')
    plt.title(f'Velocity Probing - {model_type} (R^2: {score:.3f})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'probe_velocity_{model_type}.png'))
    plt.close()

def plot_sanity_check():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cont_ds = PendulumDataset(1, SEQ_LEN, is_continuous=True)
    tok_ds = PendulumDataset(1, SEQ_LEN, is_continuous=False)
    
    _, _, v_cont, _ = cont_ds[0]
    _, _, v_tok, _ = tok_ds[0]
    
    t = np.arange(SEQ_LEN)
    plt.figure()
    plt.plot(t, v_cont, label='Continuous Velocity')
    plt.plot(t, v_tok, label='Token Velocity', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity Magnitude')
    plt.title('Sanity Check: Velocity vs Time')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sanity_check_velocity.png'))
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_sanity_check()
    
    # 1. Train Continuous Model
    print("Training Continuous Transformer...")
    cont_ds = PendulumDataset(NUM_SAMPLES, SEQ_LEN, is_continuous=True)
    cont_loader = DataLoader(cont_ds, batch_size=BATCH_SIZE, shuffle=True)
    cont_model = ContinuousTransformer(2, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MAX_SEQ_LEN).to(device)
    cont_opt = optim.Adam(cont_model.parameters(), lr=LR)
    cont_crit = nn.MSELoss()
    
    for ep in range(EPOCHS):
        cont_model.train()
        total_loss = 0
        for inputs, targets, _, _ in cont_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            cont_opt.zero_grad()
            out = cont_model(inputs)
            loss = cont_crit(out, targets)
            loss.backward()
            cont_opt.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}/{EPOCHS} Loss: {total_loss/len(cont_loader):.4f}")
        
    torch.save(cont_model.state_dict(), os.path.join(OUTPUT_DIR, "continuous_model.pt"))
    probe_and_plot(cont_model, cont_loader, "Continuous")
    
    # 2. Train Token Model
    print("\nTraining Token Transformer...")
    tok_ds = PendulumDataset(NUM_SAMPLES, SEQ_LEN, is_continuous=False)
    tok_loader = DataLoader(tok_ds, batch_size=BATCH_SIZE, shuffle=True)
    tok_model = TokenTransformer(VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MAX_SEQ_LEN).to(device)
    tok_opt = optim.Adam(tok_model.parameters(), lr=LR)
    tok_crit = nn.CrossEntropyLoss()
    
    for ep in range(EPOCHS):
        tok_model.train()
        total_loss = 0
        for inputs, targets, _, _ in tok_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            tok_opt.zero_grad()
            out = tok_model(inputs)
            loss = tok_crit(out.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            tok_opt.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}/{EPOCHS} Loss: {total_loss/len(tok_loader):.4f}")
        
    torch.save(tok_model.state_dict(), os.path.join(OUTPUT_DIR, "token_model.pt"))
    probe_and_plot(tok_model, tok_loader, "Token")
    print("Done. Saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
