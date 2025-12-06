# cal_error.py
import os
import numpy as np
import joblib
import torch
import config

from src.train import LSTMAutoencoder, PitchWindowDataset

WINDOW_SIZE = 24
WINDOW_STRIDE = 2


def frame_level_error(window_feat_err, seq_len=128):
    frame_err = np.zeros((seq_len, 13))
    frame_cnt = np.zeros(seq_len)

    for i, win_err in enumerate(window_feat_err):
        start = i * WINDOW_STRIDE
        end = start + WINDOW_SIZE
        if end > seq_len:
            break

        frame_err[start:end] += win_err
        frame_cnt[start:end] += 1

    return frame_err / (frame_cnt[:, None] + 1e-6)



def analyze_user_video(file_id="v_1"):
    """
    scaling → scaler.pkl
    model inference → user_specific_ae.pth
    thresholding → user_stats.pkl
    """

    # -----------------------------
    # 1) Load windowed features
    # -----------------------------
    feature_path = os.path.join(config.VAL_PROCESSED_DIR, f"{file_id}_processed.npz")
    npz = np.load(feature_path)
    windows = npz["windows"]     # already shape: (num_windows, 24, 13)

    # -----------------------------
    # 2) Load kps
    # -----------------------------
    kps_path = os.path.join(config.VAL_KPS_DIR, f"{file_id}_raw_kps.npy")
    kps = np.load(kps_path)

    # -----------------------------
    # 3) Scaling — always use scaler.pkl
    # -----------------------------
    scaler = joblib.load(config.SCALER_DIR)  # ← 글로벌 scaler

    num_windows = windows.shape[0]
    windows_flat = windows.reshape(-1, 13)
    windows_scaled_flat = scaler.transform(windows_flat)
    windows_scaled = windows_scaled_flat.reshape(num_windows, WINDOW_SIZE, 13)

    # -----------------------------
    # 4) Dataset (train과 동일)
    # -----------------------------
    dataset = PitchWindowDataset(windows_scaled)
    X = dataset.x  # torch.Tensor shape (num_windows, 24, 13)

    # -----------------------------
    # 5) Load fine-tuned model
    # -----------------------------
    model_path = os.path.join(config.FINE_TUNE_DIR, "user_specific_ae.pth")
    model = LSTMAutoencoder(
        input_dim=13,
        hidden_dim=128,
        latent_dim=32
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # -----------------------------
    # 6) Reconstruction
    # -----------------------------
    with torch.no_grad():
        recon = model(X).numpy()  # (num_windows, 24, 13)

    # -----------------------------
    # 7) Window-level error
    # -----------------------------
    window_feat_err = (windows_scaled - recon)**2

    # -----------------------------
    # 8) Convert window → frame
    # -----------------------------
    frame_err = frame_level_error(window_feat_err)

    # -----------------------------
    # 9) Load user_stats.pkl (threshold)
    # -----------------------------
    stats_path = os.path.join(config.FINE_TUNE_DIR, "user_stats.pkl")
    user_stats = joblib.load(stats_path)
    threshold = user_stats["threshold"]

    return {
        "kps": kps,
        "frame_error": frame_err,
        "threshold": threshold,
        "stats": user_stats
    }
