import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import torch.nn as nn


# ==========================================
# 모델 클래스 정의 (반드시 학습 때와 같아야 함!)
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
# 이상 탐지 실행 함수
# ==========================================
def calculate_anomaly_score():
    base_dir = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX"

    # 1. 설정 (학습 때와 동일하게!)
    # 저장된 모델은 hidden=128, latent=32로 저장되었음
    model_path = os.path.join(base_dir, "models/lstm_ae_final.pth")
    scaler_path = os.path.join(base_dir, "models/scaler.pkl")
    data_path = os.path.join(base_dir, "data/train/processed/2d_data.npz")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 모델 & 스케일러 로드
    # [주의] 학습 코드에서 저장할 때 썼던 사이즈(128, 32)로 초기화해야 함
    model = LSTMAutoencoder(input_dim=13, seq_len=32, hidden_dim=256, latent_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model Loaded Successfully!")

    scaler = joblib.load(scaler_path)

    # 3. 데이터 로드
    loaded = np.load(data_path, allow_pickle=True)
    if 'windows' in loaded:
        windows = loaded['windows']
    elif 'data' in loaded:
        windows = loaded['data']
    else:
        windows = loaded[loaded.files[0]]

    # 4. 스케일링 적용
    N, T, F = windows.shape
    windows_flat = windows.reshape(-1, F)
    windows_scaled = scaler.transform(windows_flat).reshape(N, T, F)  # transform만!

    # 5. 추론 (Reconstruction)
    input_tensor = torch.tensor(windows_scaled, dtype=torch.float32).to(device)

    batch_size = 64
    all_losses = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = input_tensor[i: i + batch_size]
            recon = model(batch)

            # MSE 계산 (Sample별 평균)
            # (Batch, Time, Feat) -> (Batch,)
            loss = torch.mean((batch - recon) ** 2, dim=(1, 2))
            all_losses.append(loss.cpu().numpy())

    all_losses = np.concatenate(all_losses)

    # 6. 결과 시각화
    print(f"Total Samples: {len(all_losses)}")
    print(f"Max Anomaly Score: {np.max(all_losses):.6f}")
    print(f"Avg Anomaly Score: {np.mean(all_losses):.6f}")

    plt.figure(figsize=(12, 6))
    plt.plot(all_losses, label='Anomaly Score (MSE)', color='crimson', alpha=0.7)

    # 임계값(Threshold) 예시: 평균 + 2*표준편차
    threshold = np.mean(all_losses) + 2 * np.std(all_losses)
    plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.4f})')

    plt.title("Anomaly Scores for All Windows")
    plt.xlabel("Window Index (Time)")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    calculate_anomaly_score()