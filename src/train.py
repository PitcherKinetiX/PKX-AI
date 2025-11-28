# train model

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os


# ==========================================
# 1. Dataset
# ==========================================
class PitchWindowDataset(Dataset):
    def __init__(self, windows):
        self.x = torch.tensor(windows, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


# ==========================================
# 2. LSTM AutoEncoder
# ==========================================
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        out, (_, _) = self.lstm(x)
        last = out[:, -1, :]   # 마지막 hidden
        z = self.linear(last)  # latent vector
        return z


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.linear = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h0 = torch.relu(self.linear(z))
        h0 = h0.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(h0)
        x_hat = self.output(out)
        return x_hat


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, latent_dim=32, seq_len=30):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# ==========================================
# 3. Train Function
# ==========================================
def train_lstm_ae():

    # ---------------------------
    # Load data
    # ---------------------------
    npz_path = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\data\processed\2d_data.npz"
    data = np.load(npz_path, allow_pickle=True)

    windows = data["windows"]        # (N, 30, 42)
    video_ids = data["video_ids"]    # (N,)
    filenames = data["filenames"]

    print("Loaded all_data.npz")
    print("Windows:", windows.shape)

    # ---------------------------
    # Split train/val (video-wise)
    # ---------------------------
    train_videos = [0, 1, 2, 3]      # 정상 4개 영상
    val_videos = [4]                 # 남은 1개 영상

    train_idx = np.isin(video_ids, train_videos)
    val_idx   = np.isin(video_ids, val_videos)

    train_w = windows[train_idx]
    val_w   = windows[val_idx]

    print("Train windows:", train_w.shape)
    print("Val windows:", val_w.shape)

    # ---------------------------
    # Dataset & DataLoader
    # ---------------------------
    train_ds = PitchWindowDataset(train_w)
    val_ds   = PitchWindowDataset(val_w)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)

    # ---------------------------
    # Model
    # ---------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = LSTMAutoencoder().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---------------------------
    # Training loop
    # ---------------------------
    best_val = float("inf")
    patience = 10
    no_improve = 0

    epochs = 500

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # ---------------------------
        # Early Stopping
        # ---------------------------
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_lstm_ae.pth")
            print("  → Saved best model")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("Training finished!")


if __name__ == "__main__":
    train_lstm_ae()
