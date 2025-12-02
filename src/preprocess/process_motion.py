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

        # COCO indices
        self.L_HIP = 11
        self.R_HIP = 12
        self.L_SH  = 5
        self.R_SH  = 6

    # ------------------------------------------------
    # Load raw json
    # ------------------------------------------------
    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frames = []
        for item in data:
            if "instances" not in item: continue
            if len(item["instances"]) == 0: continue

            inst = item["instances"][0]
            if "keypoints" not in inst: continue

            kps = np.array(inst["keypoints"], dtype=float)
            if kps.shape[1] == 3:
                kps = kps[:, :2]
            frames.append(kps)

        if len(frames) == 0:
            raise ValueError(f"[ERROR] No keypoints in {file_path}")

        return np.stack(frames, axis=0)  # (T,17,2)

    # ------------------------------------------------
    # Hip center normalization
    # ------------------------------------------------
    def hip_center_normalize(self, kps):
        hip_center = (kps[:, self.L_HIP] + kps[:, self.R_HIP]) / 2.0
        return kps - hip_center[:, None, :]

    # ------------------------------------------------
    # Scale normalization
    # ------------------------------------------------
    def scale_normalize_coords(self, kps):
        if not self.scale_normalize:
            return kps

        sw = np.linalg.norm(kps[:, self.L_SH] - kps[:, self.R_SH], axis=1)
        sw = np.maximum(sw, 1e-6)

        median_w = np.median(sw)
        sw = np.clip(sw, median_w * 0.5, median_w * 2.0)

        scale = sw[:, None, None]
        return np.clip(kps / scale, -50, 50)

    # ------------------------------------------------
    # Smoothing (coords only)
    # ------------------------------------------------
    def smooth(self, arr):
        """
        arr shape : (T, 17*2)
        """
        T = arr.shape[0]
        window = max(self.smoothing_window, self.smoothing_poly + 2)
        if window % 2 == 0:
            window += 1

        sm = savgol_filter(
            arr,
            window_length=window,
            polyorder=self.smoothing_poly,
            axis=0,
            mode="interp"
        )
        return np.nan_to_num(sm)

    # ------------------------------------------------
    # Interpolation (coords only)
    # ------------------------------------------------
    def interpolate(self, kps):
        T = kps.shape[0]
        old_t = np.linspace(0, 1, T)
        new_t = np.linspace(0, 1, self.target_len)

        f = interp1d(old_t, kps, axis=0, kind='linear')
        out = f(new_t)
        return np.nan_to_num(out)

    # ------------------------------------------------
    # Full preprocessing pipeline
    # ------------------------------------------------
    def process_file(self, file_path, save_raw_kps_path=None):

        # Load raw
        raw_kps = self.load_json(file_path)

        # Save raw optionally
        if save_raw_kps_path is not None:
            np.save(save_raw_kps_path, raw_kps)

        # (1) hip normalize
        kps = self.hip_center_normalize(raw_kps)

        # (2) scale normalize
        kps = self.scale_normalize_coords(kps)

        # (3) interpolation (240 frames)
        kps = self.interpolate(kps)

        # (4) smoothing (좌표 smoothing은 여기서 단 1회만)
        kps = self.smooth(kps.reshape(240, -1)).reshape(240, 17, 2)

        return kps
