# cal_gen_error.py
import os
import numpy as np
import torch
import joblib
import config

from src.train import LSTMAutoencoder, PitchWindowDataset

WINDOW_SIZE = 32


def load_general_ae():
    model = LSTMAutoencoder(input_dim=13, hidden_dim=256, latent_dim=64)
    model.load_state_dict(torch.load(
        config.MODEL_DIR,
        map_location="cpu"
    ))
    model.eval()
    return model


def cal_gen_error():

    npz = np.load(os.path.join(config.VAL_PROCESSED_DIR, "2d_data.npz"))
    windows = npz["windows"]
    num_windows = windows.shape[0]

    scaler = joblib.load(config.SCALER_DIR)
    windows_scaled = scaler.transform(
        windows.reshape(-1, 13)
    ).reshape(num_windows, WINDOW_SIZE, 13)

    X = PitchWindowDataset(windows_scaled).x

    gen_model = load_general_ae()

    # ------------------------
    # 기존 기능: General recon
    # ------------------------
    with torch.no_grad():
        recon_gen = gen_model(X).numpy()

    window_feat_err = (windows_scaled - recon_gen) ** 2
    window_err = window_feat_err.mean(axis=(1, 2))
    feature_err = window_feat_err.mean(axis=(0, 1))

    worst_window_idx = int(np.argmax(window_err))
    worst_feature_idx = int(np.argmax(feature_err))

    # ------------------------
    # 기존 기능: latent shift
    # ------------------------
    user_model = LSTMAutoencoder(input_dim=13, hidden_dim=256, latent_dim=64)
    user_model.load_state_dict(torch.load(
        os.path.join(config.FINE_TUNE_DIR, "user_specific_ae.pth"),
        map_location="cpu"
    ))
    user_model.eval()

    with torch.no_grad():
        Z_gen = gen_model.encoder(X).numpy()
        Z_user = user_model.encoder(X).numpy()

        # (추가됨) User recon
        recon_user = user_model(X).numpy()

    latent_shift = Z_user.mean(axis=0) - Z_gen.mean(axis=0)

    return {
        "window_error": window_err,
        "feature_error": feature_err,

        "worst_window_idx": worst_window_idx,
        "worst_feature_idx": worst_feature_idx,

        "latent_shift": latent_shift,
        "latent_shift_norm": float(np.linalg.norm(latent_shift)),

        # ------------------------
        # (추가됨)
        # ------------------------
        "original": windows_scaled,
        "recon_gen": recon_gen,
        "recon_user": recon_user,
    }
