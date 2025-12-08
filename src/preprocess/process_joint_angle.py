import numpy as np
from scipy.signal import butter, filtfilt


class JointAngleExtractor:
    def __init__(self, fps=128, vel_cutoff=10):
        """
        :param fps: 데이터의 초당 프레임 수 (전처리 Target FPS, 예: 128)
        :param vel_cutoff: 속도 데이터 스무딩용 컷오프 주파수 (Hz)
        """
        self.fps = fps
        self.vel_cutoff = vel_cutoff

        # COCO Keypoint Indices
        self.L_SH, self.R_SH = 5, 6
        self.L_EL, self.R_EL = 7, 8
        self.L_WR, self.R_WR = 9, 10
        self.L_HP, self.R_HP = 11, 12
        self.L_KN, self.R_KN = 13, 14
        self.L_AN, self.R_AN = 15, 16

    # ---------------------------------------------------------
    # LOW-PASS FILTER
    # ---------------------------------------------------------
    def _smooth(self, arr, cutoff):
        if arr.shape[0] < 15:
            return arr

        fs = self.fps
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        normal_cutoff = np.clip(normal_cutoff, 0.01, 0.99)

        b, a = butter(4, normal_cutoff, btype='low', analog=False)

        filtered = np.zeros_like(arr)

        for i in range(arr.shape[1]):
            sig = arr[:, i]
            if np.isnan(sig).any():
                sig = np.nan_to_num(sig)

            try:
                filtered[:, i] = filtfilt(b, a, sig, axis=0)
            except:
                filtered[:, i] = sig

        return filtered

    # ---------------------------------------------------------
    # ANGLE & VELOCITY
    # ---------------------------------------------------------
    def _angle(self, a, b, c):
        """3-point angle 0~pi"""
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
        """-pi ~ pi angle"""
        v = p2 - p1
        return np.arctan2(v[..., 1], v[..., 0])

    def _velocity(self, arr):
        return np.gradient(arr, axis=0)

    # ---------------------------------------------------------
    # MAIN EXTRACTION
    # ---------------------------------------------------------
    def extract(self, kps):
        """
        :param kps: (T, 17, 2)
        :return: (T, 13)
        """

        # ---------------------------------------------------------
        # 1) 8 Joint Angles
        # ---------------------------------------------------------
        angles = np.stack([
            self._angle(kps[:, self.L_SH], kps[:, self.L_EL], kps[:, self.L_WR]),  # 0 left elbow
            self._angle(kps[:, self.R_SH], kps[:, self.R_EL], kps[:, self.R_WR]),  # 1 right elbow
            self._angle(kps[:, self.L_EL], kps[:, self.L_SH], kps[:, self.L_HP]),  # 2 left shoulder
            self._angle(kps[:, self.R_EL], kps[:, self.R_SH], kps[:, self.R_HP]),  # 3 right shoulder
            self._angle(kps[:, self.L_SH], kps[:, self.L_HP], kps[:, self.L_KN]),  # 4 left hip
            self._angle(kps[:, self.R_SH], kps[:, self.R_HP], kps[:, self.R_KN]),  # 5 right hip
            self._angle(kps[:, self.L_HP], kps[:, self.L_KN], kps[:, self.L_AN]),  # 6 left knee (Lead)
            self._angle(kps[:, self.R_HP], kps[:, self.R_KN], kps[:, self.R_AN]),  # 7 right knee (Drive)
        ], axis=1)

        # ---------------------------------------------------------
        # 2) Kinetic Chain Velocities (5)
        # ---------------------------------------------------------

        # (1) Lead knee extension vel
        knee_ext_vel = self._velocity(angles[:, 6])

        # (2) Pelvis rotation vel
        pelvis_angle = self._angle_2pts(kps[:, self.R_HP], kps[:, self.L_HP])
        pelvis_angle = np.unwrap(pelvis_angle)
        pelvis_rot_vel = self._velocity(pelvis_angle)

        # (3) Trunk rotation vel
        trunk_angle = self._angle_2pts(kps[:, self.R_SH], kps[:, self.L_SH])
        trunk_angle = np.unwrap(trunk_angle)
        trunk_rot_vel = self._velocity(trunk_angle)

        # (4) Elbow extension vel (R)
        elbow_ext_vel = self._velocity(angles[:, 1])

        # ---------------------------------------------------------
        # (5) Shoulder Internal Rotation Velocity  ← ★ 수정된 부분
        # ---------------------------------------------------------
        upperarm_angle = self._angle_2pts(kps[:, self.R_SH], kps[:, self.R_EL])
        upperarm_angle = np.unwrap(upperarm_angle)
        shoulder_ir_vel = self._velocity(upperarm_angle)

        # stack
        kinetic_vel = np.stack([
            knee_ext_vel,
            pelvis_rot_vel,
            trunk_rot_vel,
            elbow_ext_vel,
            shoulder_ir_vel
        ], axis=1)

        # FILTER VELOCITIES
        kinetic_vel = self._smooth(kinetic_vel, cutoff=self.vel_cutoff)

        # final output (T, 13)
        final = np.concatenate([angles, kinetic_vel], axis=1)

        return final
