import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
import joblib
import matplotlib.pyplot as plt
from train import LSTMAutoencoder, PitchWindowDataset


# ---------------------------------------------------------
# Fine-tuning Function
# ---------------------------------------------------------
def fine_tune_and_get_threshold(user_video_path_list):
    """
    args:
        user_video_path_list: 사용자 영상에서 추출된 keypoints npz 파일들의 리스트 (또는 통합된 npz 경로)
    """

    # ==========================================
    # 1. 설정 및 모델 로드
    # ==========================================
    model_path = config.MODEL_DIR
    scaler_path = config.SCALER_DIR
    save_dir = config.FINE_TUNE_DIR

    # 저장 경로 생성
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Scaler 로드 (절대 새로 fit 하지 말 것!)
    scaler = joblib.load(scaler_path)
    print("Pre-trained Scaler loaded.")

    # ==========================================
    # 2. 사용자 데이터 전처리 (+ Clipping 추가)
    # ==========================================
    # 실제로는 user_video_path_list에 있는 데이터를 읽어서 windows로 만드는 과정 필요
    # 예시: user_data.npz 가 있다고 가정
    user_data_path = os.path.join(config.VAL_PROCESSED_DIR, "2d_data.npz")

    if not os.path.exists(user_data_path):
        print("사용자 데이터 경로를 확인해주세요. 임시 랜덤 데이터로 진행합니다.")
        dummy_windows = np.random.randn(50, 32, 13)  # Seq_Len 32로 가정
        windows = dummy_windows
    else:
        data = np.load(user_data_path, allow_pickle=True)
        windows = data["windows"]  # shape: (N, Seq_Len, Features)

    print(f"User Data Shape (Original): {windows.shape}")

    # Scaling
    N, T, F = windows.shape
    windows_flat = windows.reshape(-1, F)
    windows_scaled_flat = scaler.transform(windows_flat)  # transform only!
    user_windows = windows_scaled_flat.reshape(N, T, F)

    # -----------------------------------------------------------
    # [수정됨] Clipping 적용 (학습 코드와 통일)
    # 설명: 사용자 데이터에도 Phase Wrapping 등으로 인한 이상치가 있을 수 있으므로
    #       학습 때와 똑같이 -5 ~ 5 범위로 잘라줍니다.
    # -----------------------------------------------------------
    print("[Preprocess] Applying Clipping to remove extreme outliers...")
    user_windows = np.clip(user_windows, -5, 5)
    print(f"Clipping done. Data Range: [{np.min(user_windows):.2f}, {np.max(user_windows):.2f}]")

    # DataLoader
    # 데이터가 적으므로 배치 사이즈는 작게 (예: 4~8)
    user_loader = DataLoader(PitchWindowDataset(user_windows), batch_size=4, shuffle=True)

    # ==========================================
    # 3. 모델 로드 및 Fine-tuning 설정
    # ==========================================
    # Hidden dim, latent dim 등은 학습시켰던 모델과 동일해야 함
    model = LSTMAutoencoder(input_dim=F, hidden_dim=256, latent_dim=64, seq_len=T).to(device)

    # 학습된 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Base Model loaded.")

    # Fine-tuning 핵심: 낮은 Learning Rate
    ft_lr = 1e-4  # 기존 1e-3 보다 낮게 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr)
    criterion = nn.MSELoss()

    # Epochs: 데이터 양에 따라 조절 (소량 데이터 기준 30~50)
    ft_epochs = 50

    print(">>> Start Fine-tuning for User...")

    model.train()
    loss_history = []

    for epoch in range(ft_epochs):
        total_loss = 0
        for batch in user_loader:
            batch = batch.to(device)

            recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient Clipping도 안전장치로 추천
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(user_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Fine-tuning Epoch {epoch + 1}/{ft_epochs} | Loss: {avg_loss:.6f}")

    # 모델 저장
    user_model_path = os.path.join(save_dir, "user_specific_ae.pth")
    torch.save(model.state_dict(), user_model_path)
    print(f"Fine-tuned User Model saved to: {user_model_path}")

    # ==========================================
    # 4. Anomaly Threshold 계산 (정상 범위 설정)
    # ==========================================
    print(">>> Calculating Anomaly Threshold...")
    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        # 전체 사용자 데이터에 대해 오차 계산
        user_tensor = torch.tensor(user_windows, dtype=torch.float32).to(device)
        recon = model(user_tensor)

        # 샘플별 MSE 계산 (Mean over Time and Features)
        # shape: (N, T, F) -> (N, )
        loss_per_sample = torch.mean((user_tensor - recon) ** 2, dim=[1, 2])
        reconstruction_errors = loss_per_sample.cpu().numpy()

    # Threshold 설정 로직
    mean_error = np.mean(reconstruction_errors)
    std_error = np.std(reconstruction_errors)

    # 2 Sigma (95%) 적용
    threshold = mean_error + 2 * std_error

    print(f"User Normal Loss Mean: {mean_error:.6f}")
    print(f"User Normal Loss Std:  {std_error:.6f}")
    print(f"Calculated Threshold (Mean + 2*Std): {threshold:.6f}")

    # Threshold 및 통계 저장
    stats = {
        "mean": mean_error,
        "std": std_error,
        "threshold": threshold
    }
    joblib.dump(stats, os.path.join(save_dir, "user_stats.pkl"))

    # 분포 시각화
    plt.figure(figsize=(8, 5))
    plt.hist(reconstruction_errors, bins=20, alpha=0.7, label="User Normal Errors")
    plt.axvline(threshold, color='r', linestyle='--', label=f"Threshold ({threshold:.4f})")
    plt.title("Reconstruction Error Distribution (Fine-tuned)")
    plt.xlabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, threshold


if __name__ == "__main__":
    # 사용자 데이터가 준비되면 호출
    fine_tune_and_get_threshold(user_video_path_list=[])