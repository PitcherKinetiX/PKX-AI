# vis_main.py

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import config
from cal_user_error import cal_user_error
from cal_gen_error import cal_gen_error
from skel import draw_skeleton_hybrid

FRAME_SIZE = (1280, 720)
FPS = 30

FEATURE_NAMES = [
    "L elbow chain", "R elbow chain",
    "L shoulder", "R shoulder",
    "L hip chain", "R hip chain",
    "L knee", "R knee",
    "Knee extension velocity", "Pelvis rotation",
    "Trunk rotation", "Elbow extension",
    "Shoulder IR",
]


def visualize(file_id="v_1"):

    # 1) User model error
    user_res = cal_user_error(file_id)
    kps = user_res["kps"]
    win_err = user_res["window_feat_error"]

    num_frames = kps.shape[0]
    frame_err = np.zeros((num_frames, 13))
    frame_cnt = np.zeros(num_frames)

    for w in range(win_err.shape[0]):
        s = w * 2
        e = s + 32
        if e > num_frames:
            break
        frame_err[s:e] += win_err[w]
        frame_cnt[s:e] += 1

    frame_err /= (frame_cnt[:, None] + 1e-6)

    # 2) General AE
    gen_res = cal_gen_error()

    # 3) Skeleton video (기존 유지)
    os.makedirs(config.VAL_INFER_DIR, exist_ok=True)
    out_path = os.path.join(config.VAL_INFER_DIR, f"{file_id}_analysis.mp4")

    out = cv2.VideoWriter(out_path,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, FRAME_SIZE)

    for f in range(num_frames):
        img = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
        img = draw_skeleton_hybrid(img, kps[f], frame_err[f], None)
        out.write(img)

    out.release()

    # 4) Frame-level error plot (기존 + 축 레이블 추가)
    plt.figure(figsize=(12, 4))
    plt.plot(frame_err.mean(axis=1))
    plt.xlabel("Frame Index")  # NEW
    plt.ylabel("Mean Reconstruction Error")  # NEW
    plt.title("Frame-level User Reconstruction Error")
    plt.grid()
    plt.savefig(os.path.join(config.VAL_INFER_DIR, f"{file_id}_consistency.png"))
    plt.close()

    # 5) Latent shift plot (기존)
    plt.figure(figsize=(14,4))
    plt.bar(range(len(gen_res["latent_shift"])), gen_res["latent_shift"])
    plt.grid()
    plt.savefig(os.path.join(config.VAL_INFER_DIR, f"{file_id}_latent_shift.png"))
    plt.close()

    # 6) Feature temporal error (기존)
    plt.figure(figsize=(14,6))
    for i in range(13):
        plt.plot(frame_err[:, i], label=FEATURE_NAMES[i])
    plt.legend(ncol=3)
    plt.savefig(os.path.join(config.VAL_INFER_DIR, f"{file_id}_feature_temporal.png"))
    plt.close()

    # ----------------------------------------------------
    # 7) Original vs General vs User 비교 그래프
    # ----------------------------------------------------
    crit_w = user_res["critical_window"]

    gen = gen_res["recon_gen"][crit_w]  # (32,13)
    usr = gen_res["recon_user"][crit_w]  # (32,13)

    fig, axes = plt.subplots(4, 4, figsize=(22, 14))
    axes = axes.flatten()

    for i in range(13):
        ax = axes[i]

        ax.plot(gen[:, i], label="General AE", color="orange", linestyle="--")
        ax.plot(usr[:, i], label="User AE", color="green", linestyle="-.")

        ax.set_title(f"{i}. {FEATURE_NAMES[i]}")
        ax.grid()
        ax.legend()

        # remove empty subplots
    for j in range(13, 16):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(config.VAL_INFER_DIR, f"{file_id}_compare_gen_user.png"))
    plt.close()

    print("[Saved All Plots]")


if __name__ == "__main__":
    visualize("v_1")
