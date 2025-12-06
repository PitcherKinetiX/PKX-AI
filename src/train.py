import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau  # [핵심] 스케줄러 import


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
# 2. Attention-based LSTM Encoder
# ==========================================
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.ln(x)
        out, _ = self.lstm(x)
        score = torch.softmax(self.attn(out), dim=1)
        pooled = torch.sum(score * out, dim=1)
        z = self.linear(pooled)
        return z


# ==========================================
# 3. Autoregressive LSTM Decoder
# ==========================================
class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.init_linear = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.init_linear(z)).unsqueeze(1)
        outputs = []
        for _ in range(self.seq_len):
            out, _ = self.lstm(h)
            frm = self.output_layer(out)
            outputs.append(frm)
            h = out
        return torch.cat(outputs, dim=1)


# ==========================================
# 4. Full AE
# ==========================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, latent_dim=32, seq_len=24):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# ==========================================
# 5. Train Function (with Scaler and Scheduler)
# ==========================================
def train_lstm_ae():
    # 경로 설정
    base_dir = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX"
    npz_path = os.path.join(base_dir, r"data\train\processed\2d_data.npz")
    model_save_dir = os.path.join(base_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    # 데이터 로드 및 스케일링 (이 부분은 이전과 동일)
    data = np.load(npz_path, allow_pickle=True)
    windows = data["windows"]

    print("[Preprocess] Applying StandardScaler...")
    N, T, F = windows.shape
    windows_flat = windows.reshape(-1, F)
    scaler = StandardScaler()
    windows_scaled_flat = scaler.fit_transform(windows_flat)

    scaler_path = os.path.join(model_save_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    windows = windows_scaled_flat.reshape(N, T, F)
    print("Scaled Data Shape:", windows.shape)

    # 데이터 분할 및 로더 생성 (이 부분은 이전과 동일)
    num = len(windows)
    idx = np.random.permutation(num)
    train_end = int(num * 0.9)
    train_w = windows[idx[:train_end]]
    val_w = windows[idx[train_end:]]

    train_loader = DataLoader(PitchWindowDataset(train_w), batch_size=64, shuffle=True)
    val_loader = DataLoader(PitchWindowDataset(val_w), batch_size=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 모델 초기화
    seq_len = train_w.shape[1]
    input_dim = train_w.shape[-1]
    model = LSTMAutoencoder(input_dim=input_dim, seq_len=seq_len).to(device)

    # 옵티마이저, 손실 함수 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # [핵심 추가] Learning Rate Scheduler 설정
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Epochs 증가 (200 -> 300)
    epochs = 300

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"Starting Training for {epochs} epochs...")

    # -----------------------
    # Training Loop
    # -----------------------
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # [핵심] Scheduler 업데이트
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Best Model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            final_path = os.path.join(model_save_dir, "lstm_ae_final.pth")
            torch.save(model.state_dict(), final_path)

        if (epoch + 1) % 10 == 0 or current_lr != optimizer.param_groups[0]['lr']:
            print(
                f"Epoch {epoch + 1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.6f}")

    print("Training completed!")
    print(f"Best Val Loss: {best_val_loss:.6f}")

    # -----------------------
    # Save Loss History & Plot
    # -----------------------
    np.save(os.path.join(model_save_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(model_save_dir, "val_losses.npy"), np.array(val_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (Scaled)")
    plt.title("Training Loss Curve (With Scheduler)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, "loss_curve.png"))
    plt.show()


if __name__ == "__main__":
    train_lstm_ae()