# C:\Users\Yul\PycharmProjects\PitcherKinetiX\src\preprocess\joint_angle_extractor.py
import numpy as np

class JointAngleExtractor:

    def __init__(self):
        self.L_SH, self.R_SH = 5, 6
        self.L_EL, self.R_EL = 7, 8
        self.L_WR, self.R_WR = 9, 10
        self.L_HP, self.R_HP = 11, 12
        self.L_KN, self.R_KN = 13, 14
        self.L_AN, self.R_AN = 15, 16

    def _angle(self, a, b, c):
        u = a - b
        v = c - b
        dot = np.sum(u * v, axis=-1)
        nu = np.linalg.norm(u, axis=-1)
        nv = np.linalg.norm(v, axis=-1)
        eps = 1e-6
        nu[nu < eps] = eps
        nv[nv < eps] = eps
        cosang = dot / (nu * nv)
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.arccos(cosang)

    def extract(self, kps):
        N = kps.shape[0]
        angles = {
            "L_elbow": self._angle(kps[:, self.L_SH], kps[:, self.L_EL], kps[:, self.L_WR]),
            "R_elbow": self._angle(kps[:, self.R_SH], kps[:, self.R_EL], kps[:, self.R_WR]),
            "L_shoulder": self._angle(kps[:, self.L_EL], kps[:, self.L_SH], kps[:, self.L_HP]),
            "R_shoulder": self._angle(kps[:, self.R_EL], kps[:, self.R_SH], kps[:, self.R_HP]),
            "L_hip": self._angle(kps[:, self.L_SH], kps[:, self.L_HP], kps[:, self.L_KN]),
            "R_hip": self._angle(kps[:, self.R_SH], kps[:, self.R_HP], kps[:, self.R_KN]),
            "L_knee": self._angle(kps[:, self.L_HP], kps[:, self.L_KN], kps[:, self.L_AN]),
            "R_knee": self._angle(kps[:, self.R_HP], kps[:, self.R_KN], kps[:, self.R_AN]),
        }

        return np.stack(list(angles.values()), axis=1)
