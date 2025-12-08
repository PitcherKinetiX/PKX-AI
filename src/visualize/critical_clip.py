# critical_clip.py
import cv2
import numpy as np
import os
import config

from skel import draw_skeleton, normalize_to_screen
from mapping_table import FEATURE_TO_JOINT
from cal_med_error import cal_med_error

WINDOW_SIZE = 32
WINDOW_STRIDE = 2
FRAME_SIZE = (1280, 720)
FPS = 30


def save_med_critical_clip(file_id="v_1", debug=True):
    """
    의료적 기준으로 가장 위험한 window skeleton 영상 생성.
    report.py에서 호출 가능하고,
    본 파일 단독 실행 시에도 바로 동작하도록 구현.
    """

    # -----------------------------------------------------------
    # 1) cal_med_error 기반 critical info 가져오기
    # -----------------------------------------------------------
    med_res = cal_med_error(file_id)
    med_window = med_res["critical_med_window"]
    med_feat_idx = med_res["critical_med_feature"]
    med_feat_name = med_res["critical_med_name"]
    peak_value = med_res["critical_med_peak_value"]
    ratio = med_res["danger_ratios"][med_feat_idx]
    start, end = med_res["critical_med_range"]

    # -----------------------------------------------------------
    # Debug 출력
    # -----------------------------------------------------------
    if debug:
        print("\n===== [MEDICAL CRITICAL DEBUG] =====")
        print(f"· File ID                       : {file_id}")
        print(f"· Critical Window Index         : {med_window}")
        print(f"· Frame Range                   : {start} ~ {end}")
        print(f"· Critical Feature Index        : {med_feat_idx}")
        print(f"· Critical Feature Name         : {med_feat_name}")
        print(f"· Peak Value (deg/s)            : {peak_value:.2f}")
        print(f"· Danger Ratio                  : {ratio:.3f}")
        print("====================================\n")

    # -----------------------------------------------------------
    # 2) raw keypoints load
    # -----------------------------------------------------------
    kps_path = os.path.join(config.VAL_KPS_DIR, f"{file_id}_raw_kps.npy")
    kps = np.load(kps_path)  # (128,17,2)

    # window 구간 32프레임
    kps_32 = kps[start:end]

    # -----------------------------------------------------------
    # 3) 위험 joint index 매핑
    # -----------------------------------------------------------
    joint_idx = FEATURE_TO_JOINT[med_feat_idx]

    # -----------------------------------------------------------
    # 4) VideoWriter 준비
    # -----------------------------------------------------------
    os.makedirs(config.VAL_INFER_DIR, exist_ok=True)

    out_path = os.path.join(
        config.VAL_INFER_DIR,
        f"{file_id}_med_critical_window.mp4"
    )

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        FRAME_SIZE
    )

    # -----------------------------------------------------------
    # 5) 32프레임 영상 생성
    # -----------------------------------------------------------
    for kp in kps_32:
        img = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)

        # skeleton
        img = draw_skeleton(img, kp)

        # 좌표 스크린 transform
        kps_screen = normalize_to_screen(
            kp,
            frame_w=FRAME_SIZE[0],
            frame_h=FRAME_SIZE[1],
            scale=120
        )

        # 특정 joint 표시
        x, y = kps_screen[joint_idx]
        cv2.circle(img, (int(x), int(y)), 14, (0, 0, 255), -1)       # 빨간 점
        cv2.circle(img, (int(x), int(y)), 18, (255, 255, 255), 2)   # 흰 테두리

        writer.write(img)

    writer.release()

    print(f"[Saved Medical Critical Clip] {out_path}")
    return out_path


# ============================================================
# Standalone 실행 가능하도록 구성
# ============================================================
if __name__ == "__main__":
    print("Running Medical Critical Clip Generator...\n")
    save_med_critical_clip("v_1", debug=True)
