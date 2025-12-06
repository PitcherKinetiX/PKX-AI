import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
import joblib
import matplotlib.pyplot as plt
from train import LSTMAutoencoder, PitchWindowDataset


# 위에서 정의한 클래스들이 필요함 (LSTMAutoencoder, PitchWindowDataset 등)
# 실제 파일 분리 시에는 import 해서 쓰면 됨.

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Scaler 로드 (절대 새로 fit 하지 말 것!)
    scaler = joblib.load(scaler_path)
    print("Pre-trained Scaler loaded.")

    # ==========================================
    # 2. 사용자 데이터 전처리
    # ==========================================
    # 여기서는 예시로 로컬 경로의 npz를 로드한다고 가정
    # 실제로는 user_video_path_list에 있는 데이터를 읽어서 windows로 만드는 과정 필요
    # 예시: user_data.npz 가 있다고 가정
    user_data_path = os.path.join(config.VAL_PROCESSED_DIR, "2d_data.npz")

    if not os.path.exists(user_data_path):
        print("사용자 데이터 경로를 확인해주세요. 임시 랜덤 데이터로 진행합니다.")
        # (테스트용) 임시 데이터 생성
        dummy_windows = np.random.randn(50, 24, 13)
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

    # DataLoader
    # 데이터가 적으므로 배치 사이즈는 작게 (예: 4~8)
    user_loader = DataLoader(PitchWindowDataset(user_windows), batch_size=4, shuffle=True)

    # ==========================================
    # 3. 모델 로드 및 Fine-tuning 설정
    # ==========================================
    model = LSTMAutoencoder(input_dim=F, seq_len=T).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Base Model loaded.")

    # Fine-tuning 핵심: 낮은 Learning Rate
    ft_lr = 1e-4  # 기존 1e-3 보다 낮게 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr)
    criterion = nn.MSELoss()

    # Epochs: 데이터가 적으므로 너무 많이 돌리면 과적합(Overfitting)됨.
    # 5개 영상 기준 30~50 에폭 정도면 충분할 수 있음. Loss 떨어지는 것 보고 조절.
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
        # 전체 사용자 데이터에 대해 오차 계산 (Batch 없이 전체 통과 혹은 DataLoader 사용)
        user_tensor = torch.tensor(user_windows, dtype=torch.float32).to(device)
        recon = model(user_tensor)

        # 샘플별 MSE 계산 (Mean over Time and Features)
        # shape: (N, T, F) -> (N, )
        loss_per_sample = torch.mean((user_tensor - recon) ** 2, dim=[1, 2])
        reconstruction_errors = loss_per_sample.cpu().numpy()

    # Threshold 설정 로직
    # 방법 A: 평균 + 2 * 표준편차 (약 95% 신뢰구간)
    # 방법 B: 평균 + 3 * 표준편차 (약 99% 신뢰구간 - 더 보수적)
    # 방법 C: 최대값 (사용자 정상 데이터 중 가장 못 한 것보다 못하면 이상치) -> 이건 노이즈에 취약함

    mean_error = np.mean(reconstruction_errors)
    std_error = np.std(reconstruction_errors)

    # 젬민이의 선택: 이상치 탐지를 얼마나 엄격하게 할 것인가?
    # 투구폼은 조금만 달라도 이상하니까 2 sigma 추천. 너무 예민하면 3 sigma로 변경.
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
    plt.axvline(threshold, color='r', linestyle='--', label="Threshold")
    plt.title("Reconstruction Error Distribution (Fine-tuned)")
    plt.xlabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, threshold


if __name__ == "__main__":
    # 사용자 데이터가 준비되면 호출
    fine_tune_and_get_threshold(user_video_path_list=[])