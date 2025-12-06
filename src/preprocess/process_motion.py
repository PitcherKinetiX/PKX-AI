import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, make_interp_spline
from scipy.signal import butter, filtfilt


class MotionPreprocessor:
    def __init__(self, target_len=128, cutoff_freq=15, scale_normalize=True):
        self.target_len = target_len
        self.cutoff_freq = cutoff_freq
        self.scale_normalize = scale_normalize

        # COCO indices
        self.L_HIP = 11
        self.R_HIP = 12
        self.L_SH = 5
        self.R_SH = 6

    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = []
        for item in data:
            if "instances" in item and len(item["instances"]) > 0:
                kps = np.array(item["instances"][0]["keypoints"], dtype=float)
                if kps.shape[1] == 3: kps = kps[:, :2]
                frames.append(kps)

        if not frames:
            print(f"[Warning] No keypoints in {file_path}")
            return np.zeros((0, 17, 2))

        return np.stack(frames, axis=0)

    # ------------------------------------------------
    # 0. Missing Values & Linear Fill
    # ------------------------------------------------
    def fill_missing(self, kps):
        T, V, C = kps.shape
        kps = kps.copy()

        # (0,0) -> NaN
        for t in range(T):
            for v in range(V):
                if kps[t, v, 0] == 0 and kps[t, v, 1] == 0:
                    kps[t, v, :] = np.nan

        # Linear Interpolation
        for v in range(V):
            if np.isnan(kps[:, v, 0]).all():
                kps[:, v, :] = 0.0
                continue

            series_x = pd.Series(kps[:, v, 0])
            kps[:, v, 0] = series_x.interpolate(method='linear', limit_direction='both').to_numpy()
            series_y = pd.Series(kps[:, v, 1])
            kps[:, v, 1] = series_y.interpolate(method='linear', limit_direction='both').to_numpy()

        return np.nan_to_num(kps)

    # ------------------------------------------------
    # 1. Interpolation (Cubic Spline) - 부드러운 확장
    # ------------------------------------------------
    def interpolate(self, kps):
        T, V, C = kps.shape
        if T < 2: return kps

        old_t = np.linspace(0, 1, T)
        new_t = np.linspace(0, 1, self.target_len)
        new_kps = np.zeros((self.target_len, V, C))

        for v in range(V):
            for c in range(C):
                y = kps[:, v, c]
                if T < 4:
                    f = interp1d(old_t, y, kind='linear')
                    new_kps[:, v, c] = f(new_t)
                else:
                    # Cubic Spline으로 부드럽게 연결하여 각진 떨림 방지
                    spl = make_interp_spline(old_t, y, k=3)
                    new_kps[:, v, c] = spl(new_t)

        return new_kps

    # ------------------------------------------------
    # 2. Smoothing (Butterworth 15Hz) - 표준 필터
    # ------------------------------------------------
    def smooth(self, kps):
        T, V, C = kps.shape
        fs = self.target_len
        fc = self.cutoff_freq
        order = 4

        nyquist = 0.5 * fs
        normal_cutoff = fc / nyquist

        if normal_cutoff >= 1.0: normal_cutoff = 0.99
        if normal_cutoff <= 0.0: normal_cutoff = 0.01

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        kps_smooth = np.zeros_like(kps)

        for v in range(V):
            for c in range(C):
                signal = kps[:, v, c]
                if np.all(np.isnan(signal)) or np.all(signal == 0):
                    kps_smooth[:, v, c] = signal
                    continue

                padlen = min(20, T - 1)
                try:
                    kps_smooth[:, v, c] = filtfilt(b, a, signal, padlen=padlen)
                except Exception:
                    kps_smooth[:, v, c] = signal

        return kps_smooth

    # ------------------------------------------------
    # 3. Hip Normalization
    # ------------------------------------------------
    def hip_center_normalize(self, kps):
        hip_center = (kps[:, self.L_HIP] + kps[:, self.R_HIP]) / 2.0
        return kps - hip_center[:, None, :]

    # ------------------------------------------------
    # 4. Scale Normalization
    # ------------------------------------------------
    def scale_normalize_coords(self, kps):
        if not self.scale_normalize: return kps
        sw = np.linalg.norm(kps[:, self.L_SH] - kps[:, self.R_SH], axis=1)
        sw = np.maximum(sw, 1e-6)
        scale_val = np.percentile(sw, 95)
        if scale_val < 1e-6: scale_val = 1.0
        return np.clip(kps / scale_val, -500, 500)

    # ------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------
    def process_file(self, file_path, save_raw_kps_path=None, save_processed_kps_path=None):
        raw_kps = self.load_json(file_path)
        if raw_kps.shape[0] == 0:
            raise ValueError(f"Empty data in {file_path}")

        if save_raw_kps_path:
            np.save(save_raw_kps_path, raw_kps)

        # 1. 결측치 채우기 (기본)
        kps = self.fill_missing(raw_kps)

        # 2. Cubic Spline 보간 (부드러움 확보)
        kps = self.interpolate(kps)

        # 3. Butterworth 스무딩 (노이즈 제거)
        kps = self.smooth(kps)

        # 4. 정규화 (좌표계 통일)
        kps = self.hip_center_normalize(kps)
        kps = self.scale_normalize_coords(kps)

        if save_processed_kps_path:
            np.save(save_processed_kps_path, kps)

        return kps