# C:\Users\Yul\PycharmProjects\PitcherKinetiX\src\preprocess\process_motion.py

import os
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class MotionPreprocessor:
    def __init__(self, target_len=240, smoothing_window=11, smoothing_poly=3, scale_normalize=True):
        self.target_len = target_len
        self.smoothing_window = smoothing_window
        self.smoothing_poly = smoothing_poly
        self.scale_normalize = scale_normalize

        # COCO joint index
        self.L_HIP = 11
        self.R_HIP = 12
        self.L_SH  = 5
        self.R_SH  = 6


    # ------------------------------------------------
    # JSON Loader
    # ------------------------------------------------
    def load_json(self, file_path):

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frames = []

        for item in data:
            if "instances" not in item:
                continue
            if len(item["instances"]) == 0:
                continue

            inst = item["instances"][0]
            if "keypoints" not in inst:
                continue

            kps = np.array(inst["keypoints"], dtype=float)

            # (17,3) → score 제거
            if kps.shape[1] == 3:
                kps = kps[:, :2]

            frames.append(kps)

        if len(frames) == 0:
            raise ValueError(f"[ERROR] No keypoints in: {file_path}")

        return np.stack(frames, axis=0)  # (T,17,2)


    # ------------------------------------------------
    # 1) Interpolation
    # ------------------------------------------------
    def interpolate(self, keypoints):
        T = keypoints.shape[0]
        old_t = np.linspace(0, 1, T)
        new_t = np.linspace(0, 1, self.target_len)

        f = interp1d(old_t, keypoints, axis=0, kind='linear')
        out = f(new_t)

        return np.nan_to_num(out)


    # ------------------------------------------------
    # 2) Hip center normalization
    # ------------------------------------------------
    def hip_center_normalize(self, kps):
        hip_center = (kps[:, self.L_HIP] + kps[:, self.R_HIP]) / 2.0
        return kps - hip_center[:, None, :]


    # ------------------------------------------------
    # ★ 3) 안정화된 Shoulder scale normalization
    # ------------------------------------------------
    def scale_normalize_coords(self, kps):
        if not self.scale_normalize:
            return kps

        # shoulder width per frame
        sw = np.linalg.norm(
            kps[:, self.L_SH] - kps[:, self.R_SH], axis=1
        )

        # 너무 작은 값 보호
        eps = 1e-6
        sw = np.maximum(sw, eps)

        # 중앙값 기반 outlier clipping (폭발 방지 핵심)
        median_w = np.median(sw)

        # 허용 가능한 scaling range
        lower = median_w * 0.5
        upper = median_w * 2.0

        sw = np.clip(sw, lower, upper)

        # scale
        scale = sw[:, None, None]
        kps = kps / scale

        # 안정화: 너무 큰 좌표 제한
        kps = np.clip(kps, -50, 50)

        return kps


    # ------------------------------------------------
    # 4) Smoothing
    # ------------------------------------------------
    def smooth(self, kps):

        N = kps.shape[0]
        flat = kps.reshape(N, -1)

        # savgol window 자동 조정
        window = max(self.smoothing_window, self.smoothing_poly + 2)
        if window % 2 == 0:
            window += 1

        smoothed = savgol_filter(
            flat,
            window_length=window,
            polyorder=self.smoothing_poly,
            axis=0,
            mode="interp"
        )

        smoothed = np.nan_to_num(smoothed)
        return smoothed.reshape(N, 17, 2)


    # ------------------------------------------------
    # Full pipeline for one file
    # ------------------------------------------------
    def process_file(self, file_path):
        kps = self.load_json(file_path)
        kps = self.interpolate(kps)
        kps = self.hip_center_normalize(kps)
        kps = self.scale_normalize_coords(kps)   # ← 수정된 부분
        kps = self.smooth(kps)
        return kps
