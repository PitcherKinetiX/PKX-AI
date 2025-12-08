import cv2
import numpy as np

# ------------------------------
# COCO Skeleton 구조
# ------------------------------
SKELETON_PAIRS = [
    (5,7),(7,9),
    (6,8),(8,10),
    (11,13),(13,15),
    (12,14),(14,16),
    (5,6),
    (11,12),
    (5,11),(6,12)
]

# ------------------------------
# Feature → Joint 매핑 (heatmap용)
# ------------------------------
FEATURE_TO_JOINTS = {
    0: [7, 5, 9],          # L elbow chain
    1: [8, 6, 10],         # R elbow chain
    2: [5, 7, 11],         # L shoulder
    3: [6, 8, 12],         # R shoulder
    4: [11, 5],            # L hip chain
    5: [12, 6],            # R hip chain
    6: [11, 13, 15],       # L knee
    7: [12, 14, 16],       # R knee

    8: [13, 14],           # knee ext vel
    9: [11, 12],           # pelvis rotation
    10: [5, 6],            # trunk rotation
    11: [8, 10],           # elbow extension
    12: [9, 10],           # shoulder IR
}

# ------------------------------
# Keypoint normalization
# ------------------------------
def normalize_to_screen(kps, w=1280, h=720, scale=120):
    center = np.array([w // 2, h // 2])
    return (kps * scale + center).astype(int)

# ------------------------------
# Heatmap 색상: error → BGR
# ------------------------------
def error_to_color(norm):
    r = int(255 * norm)
    g = int(255 * (1 - abs(norm - 0.5) * 2))
    b = int(255 * (1 - norm))
    return np.array([b, g, r], dtype=float)

# ------------------------------
# Hybrid Skeleton Drawing (원 제거 버전)
# ------------------------------
def draw_skeleton_hybrid(img, kps_norm, feature_err, top1_feat_idx=None):

    kps = normalize_to_screen(kps_norm, img.shape[1], img.shape[0])

    # --------------------------------------
    # 1) Initialize Joint Heat Buffer
    # --------------------------------------
    J = 17
    joint_color_sum = np.zeros((J, 3), dtype=float)
    joint_count = np.zeros(J, dtype=float)

    max_e = feature_err.max() + 1e-6

    # --------------------------------------
    # 2) Feature Heat Blending
    # --------------------------------------
    for feat_idx, err in enumerate(feature_err):
        norm = err / max_e
        c = error_to_color(norm) / 255.0

        joints = FEATURE_TO_JOINTS.get(feat_idx, [])
        for j in joints:
            joint_color_sum[j] += c
            joint_count[j] += 1

    # 평균색 생성
    joint_colors = np.zeros_like(joint_color_sum)
    for j in range(J):
        if joint_count[j] > 0:
            joint_colors[j] = joint_color_sum[j] / joint_count[j]
        else:
            joint_colors[j] = np.array([1.0, 1.0, 1.0])  # 기본 white

    # --------------------------------------
    # 3) Draw Skeleton Lines (heat-based)
    # --------------------------------------
    for (i, j) in SKELETON_PAIRS:
        c = tuple(map(int, joint_colors[i] * 255))
        cv2.line(img, tuple(kps[i]), tuple(kps[j]), c, 4)

    # --------------------------------------
    # 4) Draw Joint Dots (heat-based)
    # --------------------------------------
    for j in range(J):
        c = tuple(map(int, joint_colors[j] * 255))
        cv2.circle(img, tuple(kps[j]), 7, c, -1)

    # --------------------------------------
    # ❌ Top1 Highlight 기능 제거됨
    # --------------------------------------

    return img
