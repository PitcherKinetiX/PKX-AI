# cal_user_error.py
import os
import numpy as np
import joblib
import torch
import config
from src.train import LSTMAutoencoder, PitchWindowDataset

WINDOW_SIZE = 32
WINDOW_STRIDE = 2


def cal_user_error(file_id="v_1"):

    npz = np.load(os.path.join(config.VAL_PROCESSED_DIR, f"{file_id}_processed.npz"))
    windows = npz["windows"]
    num_windows = windows.shape[0]

    kps = np.load(os.path.join(config.VAL_KPS_DIR, f"{file_id}_raw_kps.npy"))

    scaler = joblib.load(config.SCALER_DIR)
    win_scaled = scaler.transform(windows.reshape(-1, 13)).reshape(num_windows, WINDOW_SIZE, 13)

    X = PitchWindowDataset(win_scaled).x

    model = LSTMAutoencoder(13, 256, 64)
    model.load_state_dict(torch.load(
        os.path.join(config.FINE_TUNE_DIR, "user_specific_ae.pth"),
        map_location="cpu"
    ))
    model.eval()

    # ------------------------
    # (① 추가됨) User reconstruction 저장
    # ------------------------
    with torch.no_grad():
        recon_user = model(X).numpy()

    win_err = (win_scaled - recon_user) ** 2
    win_err_mean = win_err.mean(axis=(1, 2))
    feat_err_mean = win_err.mean(axis=(0, 1))

    crit_w = int(np.argmax(win_err_mean))
    crit_win_err = win_err[crit_w]
    crit_feat_idx = int(np.argmax(crit_win_err.mean(axis=0)))
    crit_feat_top3 = np.argsort(crit_win_err.mean(axis=0))[-3:][::-1]

    start = crit_w * WINDOW_STRIDE
    end = start + WINDOW_SIZE
    crit_kps = kps[start:end]

    return {
        "windows": windows,
        "kps": kps,

        "feature_error": feat_err_mean,
        "window_error": win_err_mean,
        "window_feat_error": win_err,

        # ------------------------
        # (① 추가됨)
        # ------------------------
        "recon_user": recon_user,
        "windows_scaled": win_scaled,

        "critical_window": crit_w,
        "critical_feature": crit_feat_idx,
        "critical_top3_features": crit_feat_top3,
        "critical_kps": crit_kps,
        "critical_range": (start, end),
    }
