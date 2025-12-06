# skel.py
import cv2
import numpy as np

SKELETON_PAIRS = [
    (5,7), (7,9),
    (6,8), (8,10),
    (11,13),(13,15),
    (12,14),(14,16),
    (5,6),
    (11,12),
    (5,11), (6,12)
]

# -----------------------
# Traffic-light color map
# -----------------------
def error_to_color(e, max_e):
    norm = np.clip(e / (max_e + 1e-6), 0, 1)
    g = int(255 * (1 - norm))  # high error → low green
    r = int(255 * norm)        # high error → high red
    return (0, g, r)           # BGR


def normalize_to_screen(kps, frame_w=1280, frame_h=720, scale=120):
    center = np.array([frame_w // 2, frame_h // 2])
    kps_screen = kps * scale + center
    return kps_screen.astype(int)


def draw_skeleton(img, kps_norm, color=(0,255,0)):
    kps = normalize_to_screen(kps_norm, img.shape[1], img.shape[0])

    for (i, j) in SKELETON_PAIRS:
        cv2.line(img, tuple(kps[i]), tuple(kps[j]), color, 2)

    for (x, y) in kps:
        cv2.circle(img, (x, y), 3, color, -1)

    return img


def draw_heat_joint(img, kps_norm, joint_idx, err_value):
    kps = normalize_to_screen(kps_norm, img.shape[1], img.shape[0])
    x, y = kps[joint_idx]

    # error scaling → traffic-light color
    color = error_to_color(err_value, max_e=0.05)

    cv2.circle(img, (x, y), 10, color, -1)
    return img


# -----------------------
# FEATURE BAR + LABELS
# -----------------------
FEATURE_LABELS = [
    "L_elbow_angle",
    "R_elbow_angle",
    "L_sh_angle",
    "R_sh_angle",
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


def draw_feature_bar(img, feature_err):
    h, w = img.shape[:2]
    bar_w = 200
    bar_h = h // 2
    x0 = w - bar_w - 20
    y0 = 40

    max_err = np.max(feature_err) + 1e-6
    cell_h = bar_h // len(feature_err)

    for i, e in enumerate(feature_err):
        color = error_to_color(e, max_err)

        # error bar
        cv2.rectangle(
            img,
            (x0, y0 + i * cell_h),
            (x0 + int(bar_w * (e / max_err)), y0 + (i + 1) * cell_h),
            color,
            -1
        )

        # label
        cv2.putText(
            img,
            FEATURE_LABELS[i],
            (x0 - 170, y0 + i * cell_h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1
        )

    return img
