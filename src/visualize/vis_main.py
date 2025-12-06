# visualize_main.py
import cv2
import numpy as np
import os
import config
from cal_error import analyze_user_video
from skel import draw_skeleton, draw_heat_joint, draw_feature_bar
from mapping_table import FEATURE_TO_JOINT


# ---- 고정 옵션 (config에 없음) ----
FRAME_SIZE = (1280, 720)
FPS = 30


def visualize(file_id="v_1"):

    # -----------------------------
    # 1) Run analysis
    # -----------------------------
    result = analyze_user_video(file_id)
    kps = result["kps"]                  # (128, 17, 2)
    frame_err = result["frame_error"]    # (128, 13)
    threshold = result["threshold"]      # fine-tune threshold

    # -----------------------------
    # 2) Output 경로 생성
    # -----------------------------
    os.makedirs(config.VAL_INFER_DIR, exist_ok=True)
    output_path = os.path.join(config.VAL_INFER_DIR, f"{file_id}_analysis.mp4")

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        FRAME_SIZE
    )

    # -----------------------------
    # 3) Frame Loop
    # -----------------------------
    for f in range(len(kps)):
        img = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)

        # skeleton
        img = draw_skeleton(img, kps[f])

        # feature-level bar
        img = draw_feature_bar(img, frame_err[f])

        # joint-level highlight (per feature error)
        for feat_idx, joint_idx in FEATURE_TO_JOINT.items():
            img = draw_heat_joint(img, kps[f], joint_idx, frame_err[f][feat_idx])

        # anomaly 표시 (optional)
        frame_mean_err = frame_err[f].mean()
        if frame_mean_err > threshold:
            cv2.putText(img, "ANOMALY", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(img)

    out.release()
    print("Visualization saved:", output_path)


if __name__ == "__main__":
    visualize("v_1")
