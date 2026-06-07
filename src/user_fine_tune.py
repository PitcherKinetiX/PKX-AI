import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib

import config
from src.train import LSTMAutoencoder, PitchWindowDataset

WINDOW_SIZE = 32
INPUT_DIM = 13
HIDDEN_DIM = 256
LATENT_DIM = 64


# ---------------------------------------------------------
# Fine-tuning (순수 함수)
# ---------------------------------------------------------
def fine_tune(windows, base_model_path, device=None, epochs=50, lr=1e-4):
    """
    사용자 영상에서 추출·결합된 windows로 base 모델을 fine-tune 한다.

    Args:
        windows: np.ndarray (N, WINDOW_SIZE, INPUT_DIM) — 스케일링 전 raw 윈도우
        base_model_path: 시작 가중치 경로 (일반 모델 또는 기존 사용자 모델)
        device: "cuda"/"cpu" (None이면 자동 선택)
        epochs, lr: fine-tuning 하이퍼파라미터 (소량 데이터 기준 낮은 LR)

    Returns:
        (state_dict, stats)
        - state_dict: CPU 텐서로 변환된 모델 가중치 (저장/재로드 호환)
        - stats: {"mean", "std", "threshold"} reconstruction error 통계
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    windows = np.asarray(windows, dtype=np.float32)
    N, T, F = windows.shape

    # Scaler는 전역 학습본을 그대로 사용 (절대 새로 fit 하지 말 것)
    scaler = joblib.load(config.SCALER_DIR)
    windows_scaled = scaler.transform(windows.reshape(-1, F)).reshape(N, T, F)

    # 학습 코드와 동일한 Clipping (Phase wrapping 등 이상치 제거)
    windows_scaled = np.clip(windows_scaled, -5, 5)

    loader = DataLoader(PitchWindowDataset(windows_scaled), batch_size=4, shuffle=True)

    # 모델 로드 (학습 모델과 동일한 구조)
    model = LSTMAutoencoder(input_dim=F, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, seq_len=T).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Anomaly Threshold 계산 (정상 범위 설정)
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(windows_scaled, dtype=torch.float32).to(device)
        recon = model(tensor)
        se = (tensor - recon) ** 2                                   # (N, T, F)
        loss_per_sample = torch.mean(se, dim=[1, 2]).cpu().numpy()   # (N,) 윈도우별 평균오차
        feat_per_sample = torch.mean(se, dim=1).cpu().numpy()        # (N, F) 윈도우별·특징별 평균오차

    mean_error = float(np.mean(loss_per_sample))
    std_error = float(np.std(loss_per_sample))
    stats = {
        "mean": mean_error,
        "std": std_error,
        "threshold": mean_error + 2.0 * std_error,  # 2 Sigma (95%)
        # 특징별 정상 기준선(리포트 레벨 분류용) — 특징마다 오차 스케일이 달라 per-feature로 저장
        "feat_mean": np.mean(feat_per_sample, axis=0).astype(float).tolist(),  # (F,)
        "feat_std": np.std(feat_per_sample, axis=0).astype(float).tolist(),    # (F,)
    }

    # CPU state_dict 로 반환 (CUDA 환경 의존 제거)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state_dict, stats
