"""
개인화 모델 핵심 로직 스모크 테스트 (GCS/백엔드 없이 로컬 파일만 사용).

검증 대상:
  1) src.user_fine_tune.fine_tune  — 일반 모델 베이스로 fine-tune → state_dict + stats
  2) report.generate_report_json   — 전달받은 사용자 모델/통계 경로로 리포트 생성
  3) cal_user_error / cal_gen_error — 고정 경로가 아닌 인자 경로에서 모델 로드

실행 (AI 서버 PC, 의존성 설치된 환경):
    python scripts/verify_personalized_model.py
"""
import os
import sys
import glob
import tempfile

import numpy as np
import torch
import joblib

# api/main.py 와 동일하게 sys.path 구성
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "visualize"))
sys.path.insert(0, os.path.join(_ROOT, "src", "extract"))
sys.path.insert(0, os.path.join(_ROOT, "src", "preprocess"))
sys.path.insert(0, _ROOT)

import config  # noqa: E402
from src.user_fine_tune import fine_tune  # noqa: E402
from report import generate_report_json  # noqa: E402


def _require(path, what):
    if not os.path.exists(path):
        print(f"[FAIL] {what} 없음: {path}")
        sys.exit(1)
    print(f"[ok] {what}: {path}")


def main():
    print("=" * 60)
    print("개인화 모델 스모크 테스트")
    print("=" * 60)

    _require(config.MODEL_DIR, "일반 모델(base)")
    _require(config.SCALER_DIR, "스케일러")

    # 1) val processed npz 들을 '사용자 학습 데이터'로 사용
    npz_files = sorted(glob.glob(os.path.join(config.VAL_PROCESSED_DIR, "v_*_processed.npz")))
    if not npz_files:
        print(f"[FAIL] 학습용 npz 없음: {config.VAL_PROCESSED_DIR}/v_*_processed.npz")
        sys.exit(1)

    use = npz_files[:5] if len(npz_files) >= 5 else npz_files
    windows = np.concatenate([np.load(f)["windows"] for f in use], axis=0)
    print(f"[ok] 학습 windows: {windows.shape}  (npz {len(use)}개)")

    # 2) fine-tune (빠른 검증을 위해 epochs 축소)
    print("\n>>> fine_tune 실행 (epochs=5)...")
    state_dict, stats = fine_tune(windows, config.MODEL_DIR, epochs=5)
    print(f"[ok] fine_tune 완료. stats={stats}")
    assert {"mean", "std", "threshold"} <= set(stats.keys()), "stats 키 누락"

    # 3) 임시 경로에 저장 후, 그 경로로 리포트 생성
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "user_specific_ae.pth")
        stats_path = os.path.join(tmp, "user_stats.pkl")
        torch.save(state_dict, model_path)
        joblib.dump(stats, stats_path)
        print(f"[ok] 임시 모델/통계 저장: {tmp}")

        # 리포트 대상 file_id: processed + raw_kps 둘 다 있는 것
        file_id = None
        for f in npz_files:
            fid = os.path.basename(f).replace("_processed.npz", "")
            if os.path.exists(os.path.join(config.VAL_KPS_DIR, f"{fid}_raw_kps.npy")):
                file_id = fid
                break
        if file_id is None:
            print("[FAIL] raw_kps 가 있는 file_id 를 찾지 못함")
            sys.exit(1)
        print(f"[ok] 리포트 대상 file_id: {file_id}")

        print("\n>>> generate_report_json (전달받은 사용자 모델 경로 사용)...")
        report = generate_report_json(
            file_id=file_id,
            user_model_path=model_path,
            user_stats_path=stats_path,
        )

    scores = report["scores"]
    assert "finalScore" in scores and "grade" in scores, "리포트 scores 누락"
    assert len(report["features"]) == 13, "features 13개가 아님"
    assert len(report["velocityAnalysis"]) == 5, "velocityAnalysis 5개가 아님"

    print("\n" + "=" * 60)
    print("[PASS] 개인화 모델 핵심 로직 정상")
    print(f"  finalScore={scores['finalScore']}  grade={scores['grade']}")
    print(f"  userConsistency={scores['userConsistencyScore']} "
          f"generalSimilarity={scores['generalSimilarityScore']} "
          f"medicalSafety={scores['medicalSafetyScore']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
