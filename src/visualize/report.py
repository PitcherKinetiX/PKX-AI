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


# 점수/레벨 기준을 "같은 영상에 대한 일반(general) 모델의 재구성오차"로 잡는다.
# 학습오차(stats.mean)는 모델이 과적합해 매우 작아지므로(일반화 갭), 새 영상은 항상 몇 배 크게 보여
# 일관성 점수가 비정상적으로 낮고 모든 항목이 '위험'으로 쏠렸다. 일반모델 오차는 고정·안정적인 기준.
LEVEL_NORMAL_RATIO = 0.95   # user < 0.95*general → 정상 (개인화 모델이 일반모델보다 더 잘 재구성)
LEVEL_GOOD_RATIO = 1.10     # < 1.10x → 양호
LEVEL_CAUTION_RATIO = 1.35  # < 1.35x → 주의, 이상 → 위험
CONSISTENCY_PENALTY = 0.5   # 일관성 점수: user==general → 50점, user가 general의 절반이면 75점


def classify_error_level(user_err, gen_err):
    r = float(user_err) / (float(gen_err) + 1e-6)
    if r < LEVEL_NORMAL_RATIO:
        return "정상"
    elif r < LEVEL_GOOD_RATIO:
        return "양호"
    elif r < LEVEL_CAUTION_RATIO:
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
def generate_report(file_id="v_1", user_model_path=None, user_stats_path=None):

    # 개인화 모델/통계 경로: 인자가 없으면 기존 고정 경로로 폴백
    if user_model_path is None:
        user_model_path = os.path.join(config.FINE_TUNE_DIR, "user_specific_ae.pth")
    if user_stats_path is None:
        user_stats_path = os.path.join(config.FINE_TUNE_DIR, "user_stats.pkl")

    # 통계는 하위호환용으로만 로드 (점수 산정엔 사용하지 않음 — 일반모델 오차 기준으로 대체)
    stats = joblib.load(user_stats_path)

    # --------------------------------------------------------
    # 1) USER ERROR ANALYSIS (+ 비교 기준이 되는 GENERAL 오차 먼저 계산)
    # --------------------------------------------------------
    user_res = cal_user_error(file_id, user_model_path)
    feat_err = user_res["feature_error"]
    crit_w = user_res["critical_window"]
    crit_feat = user_res["critical_feature"]
    crit_top3 = user_res["critical_top3_features"]

    gen_res = cal_gen_error(file_id, user_model_path)
    gen_feat_err = np.asarray(gen_res["feature_error"], dtype=float)
    gen_worst_feat = gen_res["worst_feature_idx"]
    latent_shift = gen_res["latent_shift_norm"]

    # 특징 레벨: 개인화 모델 오차를 같은 영상의 일반모델 오차와 비교
    feat_levels = [
        classify_error_level(feat_err[i], gen_feat_err[i])
        for i in range(13)
    ]

    # 투구 일관성: 개인화 모델이 일반모델 대비 얼마나 잘 재구성하는지 (학습오차 과적합 영향 없음)
    UserScore = float(np.clip(
        100.0 * (1.0 - CONSISTENCY_PENALTY * float(feat_err.mean()) / (float(gen_feat_err.mean()) + 1e-6)),
        0, 100))

    # --------------------------------------------------------
    # 2) GENERAL MODEL SCORE
    # --------------------------------------------------------
    GeneralScore = float(np.clip(100 * np.exp(-latent_shift), 0, 100))

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

    # peak_window_indices: 각 특징의 피크가 발생한 윈도우 인덱스 (속도값 아님)
    peak_win_idx = {VEL_LABELS[i]: int(med_res["peak_window_indices"][i]) for i in range(5)}

    score_t = 100
    for a, b in zip(timing_order, timing_order[1:]):
        if peak_win_idx[a] > peak_win_idx[b]:  # 타이밍 역전 (a가 b보다 늦게 피크)
            score_t -= 20
    score_t = max(0, score_t)

    # 최종 medical score
    # 최악 부위에 민감하도록 평균과 최솟값을 절반씩 반영 (위험 부위 1개라도 있으면 점수가 충분히 내려감)
    score_v_component = 0.5 * float(score_v.mean()) + 0.5 * float(score_v.min())
    MedicalScore = float(0.6 * score_v_component + 0.4 * score_t)

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
def generate_report_json(file_id="v_1", user_model_path=None, user_stats_path=None) -> dict:

    # 개인화 모델/통계 경로: 인자가 없으면 기존 고정 경로로 폴백
    if user_model_path is None:
        user_model_path = os.path.join(config.FINE_TUNE_DIR, "user_specific_ae.pth")
    if user_stats_path is None:
        user_stats_path = os.path.join(config.FINE_TUNE_DIR, "user_stats.pkl")

    # 통계는 진단/하위호환용으로만 로드 (점수 산정엔 사용하지 않음 — 일반모델 오차 기준으로 대체)
    stats = joblib.load(user_stats_path)
    mean_raw = stats.get("mean")
    std_raw = stats.get("std")

    # --------------------------------------------------------
    # 1) USER ERROR ANALYSIS (+ 비교 기준이 되는 GENERAL 오차 먼저 계산)
    # --------------------------------------------------------
    user_res = cal_user_error(file_id, user_model_path)
    feat_err = user_res["feature_error"]
    crit_w = user_res["critical_window"]
    crit_feat = user_res["critical_feature"]
    crit_top3 = user_res["critical_top3_features"]

    gen_res = cal_gen_error(file_id, user_model_path)
    gen_feat_err = np.asarray(gen_res["feature_error"], dtype=float)
    gen_worst_feat = gen_res["worst_feature_idx"]
    latent_shift = gen_res["latent_shift_norm"]

    # 특징 레벨: 개인화 모델 오차를 같은 영상의 일반모델 오차와 비교
    feat_levels = [
        classify_error_level(feat_err[i], gen_feat_err[i])
        for i in range(13)
    ]

    # 투구 일관성: 개인화 모델이 일반모델 대비 얼마나 잘 재구성하는지 (학습오차 과적합 영향 없음)
    UserScore = float(np.clip(
        100.0 * (1.0 - CONSISTENCY_PENALTY * float(feat_err.mean()) / (float(gen_feat_err.mean()) + 1e-6)),
        0, 100))

    # --------------------------------------------------------
    # 2) GENERAL MODEL SCORE
    # --------------------------------------------------------
    GeneralScore = float(np.clip(100 * np.exp(-latent_shift), 0, 100))

    # ===== [DIAG] 투구 일관성(UserScore) 원인 진단 로그 — 원인 확정 후 제거 =====
    try:
        _feat = np.asarray(feat_err, dtype=float)
        _gen = np.asarray(gen_feat_err, dtype=float)
        _recon = np.asarray(user_res.get("recon_user"))
        _wins = np.asarray(user_res.get("windows_scaled"))
        print("[DIAG][user_consistency] user_model_path =", user_model_path)
        print("[DIAG][user_consistency] user_stats_path =", user_stats_path,
              " stats(mean,std) =", mean_raw, std_raw)
        print("[DIAG][user_consistency] USER feat_err.mean =", float(_feat.mean()),
              " | per-feature =", np.round(_feat, 4).tolist())
        print("[DIAG][user_consistency] GEN  feat_err.mean =", float(_gen.mean()))
        if _recon.size:
            print("[DIAG][user_consistency] recon_user min/max/mean =",
                  float(_recon.min()), float(_recon.max()), float(_recon.mean()))
        if _wins.size:
            print("[DIAG][user_consistency] win_scaled min/max =",
                  float(_wins.min()), float(_wins.max()))
        print("[DIAG][user_consistency] UserScore =", float(UserScore),
              " GeneralScore =", float(GeneralScore))
    except Exception as _e:  # noqa: BLE001
        print("[DIAG][user_consistency] diag failed:", _e)
    # ===== [/DIAG] =====

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
    peak_win_idx = {VEL_LABELS[i]: int(med_res["peak_window_indices"][i]) for i in range(5)}

    score_t = 100
    for a, b in zip(timing_order, timing_order[1:]):
        if peak_win_idx[a] > peak_win_idx[b]:
            score_t -= 20
    score_t = max(0, score_t)

    # 최악 부위에 민감하도록 평균과 최솟값을 절반씩 반영 (위험 부위 1개라도 있으면 점수가 충분히 내려감)
    score_v_component = 0.5 * float(score_v.mean()) + 0.5 * float(score_v.min())
    MedicalScore = float(0.6 * score_v_component + 0.4 * score_t)

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
