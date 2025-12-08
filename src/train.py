import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
    def __init__(self, input_dim=13, hidden_dim=256, latent_dim=64, seq_len=32):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# ==========================================
# 5. Train Function (Scaling & Training)
# ==========================================
def train_lstm_ae():
    # -----------------------------------------------------------
    # [설정] 경로 및 하이퍼파라미터
    # -----------------------------------------------------------
    base_dir = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX"
    npz_path = os.path.join(base_dir, r"data\train\processed\2d_data.npz")  # Windowing 완료된 데이터
    model_save_dir = os.path.join(base_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------------------------------------------
    # [Step 1] 데이터 로드
    # -----------------------------------------------------------
    print(f"Loading data from {npz_path}...")
    loaded = np.load(npz_path, allow_pickle=True)

    # 키(Key) 이름 자동 찾기 (windows, data, arr_0 등 대응)
    if 'windows' in loaded:
        windows = loaded['windows']
    elif 'data' in loaded:
        windows = loaded['data']
    else:
        key = loaded.files[0]
        windows = loaded[key]
        print(f"Detected key: {key}")

    print(f"Raw Data Shape: {windows.shape}")  # 예상: (N, 24, 13)

    # -----------------------------------------------------------
    # [Step 2] 스케일링 (StandardScaler Fit & Transform) - 핵심 수정
    # -----------------------------------------------------------
    print("[Preprocess] Fitting and Applying StandardScaler...")

    N, T, F = windows.shape

    # 1. 3차원(N, T, F) -> 2차원(N*T, F)로 펼치기 (Scaler 학습용)
    windows_flat = windows.reshape(-1, F)

    # 2. Scaler 정의 및 학습
    scaler = StandardScaler()
    windows_scaled_flat = scaler.fit_transform(windows_flat)

    # 3. Scaler 저장 (나중에 Inference 할 때 필수)
    scaler_path = os.path.join(model_save_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # 4. 다시 3차원(N, T, F)으로 복구
    windows = windows_scaled_flat.reshape(N, T, F)
    print(f"Scaled Data Shape: {windows.shape}")

    # -----------------------------------------------------------
    # [Step 3] 데이터셋 분할 및 로더 생성
    # -----------------------------------------------------------
    num = len(windows)
    idx = np.random.permutation(num)
    train_end = int(num * 0.9)

    train_w = windows[idx[:train_end]]
    val_w = windows[idx[train_end:]]

    train_loader = DataLoader(PitchWindowDataset(train_w), batch_size=64, shuffle=True)
    val_loader = DataLoader(PitchWindowDataset(val_w), batch_size=64)

    # -----------------------------------------------------------
    # [Step 4] 모델 초기화
    # -----------------------------------------------------------
    seq_len = train_w.shape[1]
    input_dim = train_w.shape[-1]

    model = LSTMAutoencoder(input_dim=input_dim, seq_len=seq_len, hidden_dim=256, latent_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 학습률 스케줄러 (Loss가 안 줄어들면 LR 감소)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    epochs = 500
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"Starting Training for {epochs} epochs...")

    # -----------------------------------------------------------
    # [Step 5] 학습 루프 (Training Loop)
    # -----------------------------------------------------------
    for epoch in range(epochs):
        # Train
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)

            # Forward
            recon = model(batch)
            loss = criterion(recon, batch)

            # Backward
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

        # Scheduler Step
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            final_path = os.path.join(model_save_dir, "lstm_ae_final.pth")
            torch.save(model.state_dict(), final_path)

        if (epoch + 1) % 10 == 0 or current_lr != optimizer.param_groups[0]['lr']:
            print(
                f"Epoch {epoch + 1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.6f}")

    print("Training completed!")
    print(f"Best Val Loss: {best_val_loss:.6f}")

    # -----------------------------------------------------------
    # [Step 6] 결과 저장 및 시각화
    # -----------------------------------------------------------
    np.save(os.path.join(model_save_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(model_save_dir, "val_losses.npy"), np.array(val_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Standardized)")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, "loss_curve.png"))
    plt.show()

    # -----------------------------------------------------------
    # [Step 7] Reconstruction Visualization (All 13 Features)
    # -----------------------------------------------------------
    print("Visualizing all 13 features for a random validation sample...")

    # 1. 테스트 데이터 샘플 하나 가져오기 (val_loader에서)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            sample_input = batch.to(device)  # (Batch, Seq, Feat)
            sample_recon = model(sample_input)  # (Batch, Seq, Feat)
            break  # 첫 배치만 사용

    # 2. 샘플 선택 (배치 내 0번째 데이터)
    sample_idx = 0
    input_np = sample_input[sample_idx].cpu().numpy()
    recon_np = sample_recon[sample_idx].cpu().numpy()

    # 3. Feature 이름 정의 (JointAngleExtractor 순서 참고)
    feature_names = [
        "0. L Elbow Ang", "1. R Elbow Ang", "2. L Shoulder Ang", "3. R Shoulder Ang",
        "4. L Hip Ang", "5. R Hip Ang", "6. L Knee Ang (Lead)", "7. R Knee Ang (Drive)",
        "8. Lead Knee Vel", "9. Pelvis Rot Vel", "10. Trunk Rot Vel",
        "11. Elbow Ext Vel", "12. Shoulder IR Vel"
    ]

    # 4. 4x4 서브플롯 생성
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()  # 1차원 배열로 펼침

    for i in range(13):
        ax = axes[i]

        # 원본 (파란색) vs 복원 (주황색 점선)
        ax.plot(input_np[:, i], label='Original', color='blue', linewidth=1.5)
        ax.plot(recon_np[:, i], label='Recon', color='orange', linestyle='--', linewidth=1.5)

        ax.set_title(feature_names[i], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.5)

        # 범례는 첫 번째 그래프에만 표시 (공간 절약)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    # 남는 빈칸(13, 14, 15번) 지우기
    for i in range(13, 16):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle(f"Reconstruction Check (Sample {sample_idx})", fontsize=16, y=1.02)
    plt.show()

if __name__ == "__main__":
    train_lstm_ae()