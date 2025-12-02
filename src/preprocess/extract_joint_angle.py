# C:\Users\Yul\PycharmProjects\PitcherKinetiX\src\preprocess\joint_angle_extractor.py

import numpy as np
from scipy.signal import savgol_filter


class JointAngleExtractor:

    def __init__(self, angle_smooth_window=9, angle_smooth_poly=3):
        self.angle_smooth_window = angle_smooth_window
        self.angle_smooth_poly = angle_smooth_poly

        self.L_SH, self.R_SH = 5, 6
        self.L_EL, self.R_EL = 7, 8
        self.L_WR, self.R_WR = 9, 10
        self.L_HP, self.R_HP = 11, 12
        self.L_KN, self.R_KN = 13, 14
        self.L_AN, self.R_AN = 15, 16

    # -------------------------
    # basic smoothing
    # -------------------------
    def _smooth(self, arr):
        T = arr.shape[0]
        window = self.angle_smooth_window
        if window % 2 == 0:
            window += 1
        sm = savgol_filter(arr, window_length=window,
                           polyorder=self.angle_smooth_poly,
                           axis=0, mode="interp")
        return np.nan_to_num(sm)

    # -------------------------
    # joint angle calc
    # -------------------------
    def _angle(self, a, b, c):
        u = a - b
        v = c - b
        dot = np.sum(u * v, axis=-1)
        nu = np.linalg.norm(u, axis=-1)
        nv = np.linalg.norm(v, axis=-1)

        eps = 1e-6
        nu = np.maximum(nu, eps)
        nv = np.maximum(nv, eps)

        cosang = np.clip(dot / (nu * nv), -1.0, 1.0)
        return np.arccos(cosang)

    def _angle_2pts(self, p1, p2):
        v = p2 - p1
        return np.arctan2(v[..., 1], v[..., 0])

    def _velocity(self, arr):
        return np.gradient(arr, axis=0)

    # -------------------------
    # main extractor
    # -------------------------
    def extract(self, kps):
        T = kps.shape[0]

        # 1) base angles (8개)
        angles = np.stack([
            self._angle(kps[:, self.L_SH], kps[:, self.L_EL], kps[:, self.L_WR]),
            self._angle(kps[:, self.R_SH], kps[:, self.R_EL], kps[:, self.R_WR]),
            self._angle(kps[:, self.L_EL], kps[:, self.L_SH], kps[:, self.L_HP]),
            self._angle(kps[:, self.R_EL], kps[:, self.R_SH], kps[:, self.R_HP]),
            self._angle(kps[:, self.L_SH], kps[:, self.L_HP], kps[:, self.L_KN]),
            self._angle(kps[:, self.R_SH], kps[:, self.R_HP], kps[:, self.R_KN]),
            self._angle(kps[:, self.L_HP], kps[:, self.L_KN], kps[:, self.L_AN]),
            self._angle(kps[:, self.R_HP], kps[:, self.R_KN], kps[:, self.R_AN]),
        ], axis=1)  # shape (T, 8)

        # smoothing 1
        angles = self._smooth(angles)

        # ------------------------------
        # 2) KINETIC CHAIN VELOCITY (5개)
        # ------------------------------
        # knee extension velocity (우완 기준)
        knee_ext_vel = self._velocity(angles[:, 7])  # R_knee

        # pelvis rotation velocity
        pelvis_angle = self._angle_2pts(kps[:, self.R_HP], kps[:, self.L_HP])
        pelvis_rot_vel = self._velocity(pelvis_angle)

        # trunk rotation velocity
        trunk_angle = self._angle_2pts(kps[:, self.R_SH], kps[:, self.L_SH])
        trunk_rot_vel = self._velocity(trunk_angle)

        # elbow extension velocity
        elbow_ext_vel = self._velocity(angles[:, 1])  # R_elbow

        # shoulder IR proxy
        wrist_angle = self._angle_2pts(kps[:, self.R_EL], kps[:, self.R_WR])
        shoulder_ir_vel = self._velocity(wrist_angle)

        kinetic_vel = np.stack([
            knee_ext_vel,
            pelvis_rot_vel,
            trunk_rot_vel,
            elbow_ext_vel,
            shoulder_ir_vel
        ], axis=1)  # (T, 5)

        # smoothing 2
        kinetic_vel = self._smooth(kinetic_vel)

        # ------------------------------
        # 최종 feature: angle(8) + velocity(5)
        # ------------------------------
        final = np.concatenate([angles, kinetic_vel], axis=1)  # (T, 13)

        return final
