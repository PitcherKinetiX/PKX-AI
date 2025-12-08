# cal_med_error.py
import os
import numpy as np
import config
from medical_config import MEDICAL_THRESHOLDS

WINDOW_SIZE = 32
WINDOW_STRIDE = 2

VEL_NAME = ["knee_ext_vel", "pelvis_rot_vel", "trunk_rot_vel",
            "elbow_ext_vel", "shoulder_ir_vel"]

EFFECTIVE_FPS = 64               # ★ 2초 가정 → 128 / 2 = 64fps
RAD2DEG = 180 / np.pi            # rad → deg 변환


def cal_med_error(file_id="v_1"):

    npz = np.load(os.path.join(config.VAL_PROCESSED_DIR, f"{file_id}_processed.npz"))
    windows = npz["windows"]
    num_windows = windows.shape[0]

    # velocity extraction
    # rad/frame 값
    vel = windows[:, :, 8:13]        # (num_windows, 32, 5)

    # rad/frame → deg/s 변환
    vel_deg_s = vel * EFFECTIVE_FPS * RAD2DEG

    # window-level peak (deg/s 단위)
    window_peak = vel_deg_s.max(axis=1)      # (num_windows, 5)

    # overall peak per feature
    peak_values = window_peak.max(axis=0)    # (5,)

    # ----------------------
    # danger ratios (peak / danger_threshold)
    # ----------------------
    danger_ratios = []
    for i, name in enumerate(VEL_NAME):
        _, _, danger_th = MEDICAL_THRESHOLDS[name]
        danger_ratios.append(peak_values[i] / danger_th)
    danger_ratios = np.array(danger_ratios)

    # ----------------------
    # danger matrix (window × feature)
    # ----------------------
    danger_mat = np.zeros((num_windows, 5))
    for i, name in enumerate(VEL_NAME):
        _, _, danger_th = MEDICAL_THRESHOLDS[name]
        danger_mat[:, i] = window_peak[:, i] / danger_th

    # Most dangerous window + feature
    flat_idx = np.argmax(danger_mat)
    med_window_idx, med_feat_idx = np.unravel_index(flat_idx, danger_mat.shape)

    med_peak_value = window_peak[med_window_idx, med_feat_idx]
    most_critical_feature = VEL_NAME[med_feat_idx]

    med_start = med_window_idx * WINDOW_STRIDE
    med_end = med_start + WINDOW_SIZE

    # ----------------------
    # Medical score per feature (100/60/20)
    # ----------------------
    medical_scores = []
    for ratio in danger_ratios:
        if ratio < 1:
            medical_scores.append(100)
        elif ratio < 2:
            medical_scores.append(60)
        else:
            medical_scores.append(20)

    medical_overall_score = float(np.mean(medical_scores))

    return {
        "window_peak": window_peak,
        "peak_values": peak_values,
        "danger_ratios": danger_ratios,
        "danger_matrix": danger_mat,

        "critical_med_window": med_window_idx,
        "critical_med_feature": med_feat_idx,
        "critical_med_name": most_critical_feature,
        "critical_med_peak_value": float(med_peak_value),
        "critical_med_range": (med_start, med_end),

        "medical_scores": medical_scores,
        "medical_overall_score": medical_overall_score,
    }
