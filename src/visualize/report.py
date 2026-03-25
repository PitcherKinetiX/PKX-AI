# report.py
import numpy as np
import os
import joblib
import config

from cal_user_error import cal_user_error
from cal_gen_error import cal_gen_error
from cal_med_error import cal_med_error

FEATURE_LABELS = [
    "L_elbow_angle","R_elbow_angle",
    "L_shoulder_angle","R_shoulder_angle",
    "L_hip_angle","R_hip_angle",
    "L_knee_angle","R_knee_angle",
    "knee_ext_vel","pelvis_rot_vel",
    "trunk_rot_vel","elbow_ext_vel","shoulder_ir_vel"
]

VEL_LABELS = FEATURE_LABELS[8:]


def classify_error_level(value, mean, std):
    if value < mean + 0.5 * std:
        return "정상"
    elif value < mean + 1.0 * std:
        return "양호"
    elif value < mean + 2.0 * std:
        return "주의"
    else:
        return "위험"


def assign_grade(score):
    if score >= 90: return "A+"
    elif score >= 80: return "A-"
    elif score >= 70: return "B+"
    elif score >= 60: return "B-"
    elif score >= 50: return "C+"
    elif score >= 40: return "C-"
    elif score >= 30: return "D+"
    elif score >= 20: return "D-"
    else: return "F"


# ============================================================
# REPORT MAIN
# ============================================================
def generate_report(file_id="v_1"):

    # --------------------------------------------------------
    # Load user baseline stats (mean, std)
    # --------------------------------------------------------
    stats = joblib.load(os.path.join(config.FINE_TUNE_DIR, "user_stats.pkl"))
    mean_raw = stats["mean"]
    std_raw = stats["std"]

    # numpy array인지 확인
    if np.isscalar(mean_raw):
        mean = np.ones(13) * float(mean_raw)
    else:
        mean = np.array(mean_raw)

    if np.isscalar(std_raw):
        std = np.ones(13) * (float(std_raw) + 1e-6)
    else:
        std = np.array(std_raw) + 1e-6

    # --------------------------------------------------------
    # 1) USER ERROR ANALYSIS
    # --------------------------------------------------------
    user_res = cal_user_error(file_id)
    feat_err = user_res["feature_error"]
    crit_w = user_res["critical_window"]
    crit_feat = user_res["critical_feature"]
    crit_top3 = user_res["critical_top3_features"]

    feat_levels = [
        classify_error_level(feat_err[i], mean[i], std[i])
        for i in range(13)
    ]

    # User Consistency Score
    UserScore = float(np.clip(100 * (1 - feat_err.mean()), 0, 100))

    # --------------------------------------------------------
    # 2) GENERAL MODEL ANALYSIS
    # --------------------------------------------------------
    gen_res = cal_gen_error()
    gen_feat_err = gen_res["feature_error"]
    gen_worst_feat = gen_res["worst_feature_idx"]
    latent_shift = gen_res["latent_shift_norm"]

    GeneralScore = float(100 * np.exp(-latent_shift))
    GeneralScore = np.clip(GeneralScore, 0, 100)

    # --------------------------------------------------------
    # 3) MEDICAL ANALYSIS
    # --------------------------------------------------------
    med_res = cal_med_error(file_id)

    peak_values = med_res["peak_values"]
    danger_ratios = med_res["danger_ratios"]
    score_v = np.array(med_res["medical_scores"])  # velocity 기반

    # timing score
    timing_order = [
        "knee_ext_vel","pelvis_rot_vel","trunk_rot_vel",
        "elbow_ext_vel","shoulder_ir_vel"
    ]

    peak_idx = {VEL_LABELS[i]: peak_values[i] for i in range(5)}

    score_t = 100
    for a, b in zip(timing_order, timing_order[1:]):
        if peak_idx[a] > peak_idx[b]:  # 타이밍 역전
            score_t -= 20
    score_t = max(0, score_t)

    # 최종 medical score
    MedicalScore = float(0.6 * score_v.mean() + 0.4 * score_t)

    most_critical_med_feature = med_res["critical_med_name"]

    # --------------------------------------------------------
    # 4) FINAL SCORE
    # --------------------------------------------------------
    FinalScore = (
        0.33 * UserScore +
        0.33 * GeneralScore +
        0.34 * MedicalScore
    )
    Grade = assign_grade(FinalScore)

    # --------------------------------------------------------
    # BUILD TEXT REPORT
    # --------------------------------------------------------
    report = []
    report.append("=== Pitching Motion Analysis Report ===\n")

    # USER PART
    report.append("[1] 사용자 투구 재구성 오차 분석 (User AE)")
    for i in range(13):
        report.append(
            f"- {FEATURE_LABELS[i]}: {feat_levels[i]} (err={feat_err[i]:.4f})"
        )

    report.append(f"\n• Critical Window: {crit_w}")
    report.append(f"• Critical Feature: {FEATURE_LABELS[crit_feat]}")
    report.append("• Top3 Error Features:")
    for idx in crit_top3:
        report.append(f"   - {FEATURE_LABELS[idx]}")
    report.append(f"→ User Consistency Score: {UserScore:.1f}\n")

    # GEN PART
    report.append("[2] General Model 비교")
    for i, err in enumerate(gen_feat_err):
        report.append(f"- {FEATURE_LABELS[i]}: err={err:.4f}")
    report.append(f"• Worst General Feature: {FEATURE_LABELS[gen_worst_feat]}")
    report.append(f"• Latent Shift Norm: {latent_shift:.4f}")
    report.append(f"→ General Similarity Score: {GeneralScore:.1f}\n")

    # MED PART
    report.append("[3] 의학적 투구 메커니즘 분석")
    for i, name in enumerate(VEL_LABELS):
        report.append(
            f"- {name}: peak={peak_values[i]:.1f}, "
            f"ratio={danger_ratios[i]:.2f}, score={score_v[i]}"
        )
    report.append(f"• Timing Score: {score_t}")
    report.append(f"• Most Critical Medical Feature: {most_critical_med_feature}")
    report.append(f"→ Medical Safety Score: {MedicalScore:.1f}\n")

    # FINAL
    report.append("[4] 최종 평가")
    report.append(f"- Final Score: {FinalScore:.1f} / 100")
    report.append(f"- Grade: {Grade}")

    return "\n".join(report)


# ============================================================
# REPORT JSON (구조화된 dict 반환)
# ============================================================
def generate_report_json(file_id="v_1") -> dict:

    # --------------------------------------------------------
    # Load user baseline stats (mean, std)
    # --------------------------------------------------------
    stats = joblib.load(os.path.join(config.FINE_TUNE_DIR, "user_stats.pkl"))
    mean_raw = stats["mean"]
    std_raw = stats["std"]

    if np.isscalar(mean_raw):
        mean = np.ones(13) * float(mean_raw)
    else:
        mean = np.array(mean_raw)

    if np.isscalar(std_raw):
        std = np.ones(13) * (float(std_raw) + 1e-6)
    else:
        std = np.array(std_raw) + 1e-6

    # --------------------------------------------------------
    # 1) USER ERROR ANALYSIS
    # --------------------------------------------------------
    user_res = cal_user_error(file_id)
    feat_err = user_res["feature_error"]
    crit_w = user_res["critical_window"]
    crit_feat = user_res["critical_feature"]
    crit_top3 = user_res["critical_top3_features"]

    feat_levels = [
        classify_error_level(feat_err[i], mean[i], std[i])
        for i in range(13)
    ]

    UserScore = float(np.clip(100 * (1 - feat_err.mean()), 0, 100))

    # --------------------------------------------------------
    # 2) GENERAL MODEL ANALYSIS
    # --------------------------------------------------------
    gen_res = cal_gen_error()
    gen_feat_err = gen_res["feature_error"]
    gen_worst_feat = gen_res["worst_feature_idx"]
    latent_shift = gen_res["latent_shift_norm"]

    GeneralScore = float(np.clip(100 * np.exp(-latent_shift), 0, 100))

    # --------------------------------------------------------
    # 3) MEDICAL ANALYSIS
    # --------------------------------------------------------
    med_res = cal_med_error(file_id)

    peak_values = med_res["peak_values"]
    danger_ratios = med_res["danger_ratios"]
    score_v = np.array(med_res["medical_scores"])

    timing_order = [
        "knee_ext_vel", "pelvis_rot_vel", "trunk_rot_vel",
        "elbow_ext_vel", "shoulder_ir_vel"
    ]
    peak_idx = {VEL_LABELS[i]: peak_values[i] for i in range(5)}

    score_t = 100
    for a, b in zip(timing_order, timing_order[1:]):
        if peak_idx[a] > peak_idx[b]:
            score_t -= 20
    score_t = max(0, score_t)

    MedicalScore = float(0.6 * score_v.mean() + 0.4 * score_t)

    # --------------------------------------------------------
    # 4) FINAL SCORE
    # --------------------------------------------------------
    FinalScore = 0.33 * UserScore + 0.33 * GeneralScore + 0.34 * MedicalScore
    Grade = assign_grade(FinalScore)

    # --------------------------------------------------------
    # BUILD JSON RESPONSE
    # --------------------------------------------------------

    # 13개 특징 전체
    features = []
    for i in range(13):
        features.append({
            "index": i,
            "name": FEATURE_LABELS[i],
            "type": "velocity" if i >= 8 else "angle",
            "userError": float(feat_err[i]),
            "generalError": float(gen_feat_err[i]),
            "level": feat_levels[i],
        })

    # 속도 특징 5개 의학 분석
    velocity_analysis = []
    for i in range(5):
        velocity_analysis.append({
            "index": i + 8,
            "name": VEL_LABELS[i],
            "peakValue": float(peak_values[i]),
            "dangerRatio": float(danger_ratios[i]),
            "medicalScore": int(score_v[i]),
        })

    return {
        "scores": {
            "userConsistencyScore": round(UserScore, 2),
            "generalSimilarityScore": round(GeneralScore, 2),
            "medicalSafetyScore": round(MedicalScore, 2),
            "finalScore": round(FinalScore, 2),
            "grade": Grade,
            "timingScore": float(score_t),
        },
        "features": features,
        "velocityAnalysis": velocity_analysis,
        "criticalAreas": {
            "userCriticalWindow": int(crit_w),
            "userCriticalFeature": FEATURE_LABELS[crit_feat],
            "userCriticalTop3": [FEATURE_LABELS[idx] for idx in crit_top3],
            "medCriticalFeature": med_res["critical_med_name"],
            "medCriticalWindow": int(med_res["critical_med_window"]),
        },
        "generalModel": {
            "worstFeature": FEATURE_LABELS[gen_worst_feat],
            "latentShiftNorm": round(float(latent_shift), 4),
        },
    }


if __name__ == "__main__":
    print(generate_report("v_1"))
