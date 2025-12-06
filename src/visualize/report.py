# report.py
import numpy as np
import os
from cal_error import analyze_user_video
import config

FEATURE_LABELS = [
    "L_elbow_angle",
    "R_elbow_angle",
    "L_shoulder_angle",
    "R_shoulder_angle",
    "L_hip_angle",
    "R_hip_angle",
    "L_knee_angle",
    "R_knee_angle",
    "knee_ext_vel",
    "pelvis_rot_vel",
    "trunk_rot_vel",
    "elbow_ext_vel",
    "shoulder_ir_vel",
]


# ------------------------------------
# Reconstruction Error Classification
# ------------------------------------
def classify_feature_error(err, mean, std):
    if err < mean + 0.5 * std:
        return "정상"
    elif err < mean + 1.0 * std:
        return "양호"
    elif err < mean + 2.0 * std:
        return "주의"
    else:
        return "위험"


# ------------------------------------
# Medical Thresholds
# ------------------------------------
MEDICAL_THRESHOLDS = {
    "knee_ext_vel": (300, 500, 700),
    "pelvis_rot_vel": (400, 600, 900),
    "trunk_rot_vel": (600, 900, 1200),
    "elbow_ext_vel": (2000, 2400, 2700),
    "shoulder_ir_vel": (7000, 8500, 10000),
}

MEDICAL_DESCRIPTIONS = {
    "knee_ext_vel": "무릎 신전 속도가 과도하면 PFPS 및 전방 스트레스 증가와 연관됩니다.",
    "pelvis_rot_vel": "골반 회전 타이밍이 어긋나면 허리 회전 부하가 증가합니다.",
    "trunk_rot_vel": "상체 회전 속도가 과도하면 UCL 부하가 증가할 수 있습니다.",
    "elbow_ext_vel": "팔꿈치 신전 속도가 과도할 경우 UCL strain이 증가합니다.",
    "shoulder_ir_vel": "어깨 내회전 속도가 빠르면 SLAP 병변 위험이 증가할 수 있습니다.",
}


def classify_medical_velocity(name, peak_deg_s):
    normal, caution, danger = MEDICAL_THRESHOLDS[name]

    if peak_deg_s < caution:
        return "정상"
    elif peak_deg_s < danger:
        return "주의"
    else:
        return "위험"


# ============================================
# Report Generation
# ============================================
def generate_report(file_id="v_1"):
    result = analyze_user_video(file_id)
    frame_err = result["frame_error"]           # (128,13)

    # 1) feature 재구성 오차 (전체 평균)
    feature_total_err = frame_err.mean(axis=0)

    # user_stats 기반 scaling threshold
    import joblib
    stats = joblib.load(os.path.join(config.FINE_TUNE_DIR, "user_stats.pkl"))
    mean, std = stats["mean"], stats["std"]

    # -------------------------
    # 2) 실제 velocity peak 구하기
    # -------------------------
    # window 파일 불러오기
    processed_path = os.path.join(config.VAL_PROCESSED_DIR, f"{file_id}_processed.npz")
    npz = np.load(processed_path)
    windows = npz["windows"]      # (num_windows, 24, 13)

    # velocity는 feature index 8~12
    vel = windows[:, :, 8:13]     # (num_windows, 24, 5)

    # rad/frame → deg/s 변환
    FPS = 128
    vel_deg_s = vel * FPS * (180 / np.pi)   # (nwin, 24, 5)

    # peak velocity
    vel_names = ["knee_ext_vel", "pelvis_rot_vel", "trunk_rot_vel",
                 "elbow_ext_vel", "shoulder_ir_vel"]

    peak_vels = {
        vel_names[i]: float(vel_deg_s[:, :, i].max())
        for i in range(5)
    }

    # -------------------------
    # 3) Report Text 구성
    # -------------------------
    report = []
    report.append("=== Pitching Motion Analysis Report ===\n")

    # ----- 13개 feature reconstruction error -----
    report.append("[1] Feature Reconstruction Error Summary")
    for i, err in enumerate(feature_total_err):
        status = classify_feature_error(err, mean, std)
        name = FEATURE_LABELS[i]
        report.append(f"- {name}: {status} (err={err:.4f})")

    # ----- Medical kinetic chain evaluation -----
    report.append("\n[2] Medical Evaluation (Peak Velocity Analysis)")
    for name in vel_names:
        peak = peak_vels[name]
        med_status = classify_medical_velocity(name, peak)
        desc = MEDICAL_DESCRIPTIONS[name]
        report.append(f"- {name}: {med_status}  (peak={peak:.1f} deg/s)")
        report.append(f"  설명: {desc}")

    return "\n".join(report)


if __name__ == "__main__":
    print(generate_report("v_1"))
