import numpy as np
from scipy.signal import butter, filtfilt


class JointAngleExtractor:
    def __init__(self, fps=128, vel_cutoff=10):
        """
        :param fps: 데이터의 초당 프레임 수 (전처리에서 맞춘 target_len과 동일하게 설정, 예: 128)
        :param vel_cutoff: 속도 데이터 스무딩용 컷오프 주파수 (Hz)
                           - 좌표보다 속도가 노이즈가 심하므로 보통 좌표(15Hz)보다 낮게 잡음 (10Hz 추천)
        """
        self.fps = fps
        self.vel_cutoff = vel_cutoff

        # COCO Indices
        self.L_SH, self.R_SH = 5, 6
        self.L_EL, self.R_EL = 7, 8
        self.L_WR, self.R_WR = 9, 10
        self.L_HP, self.R_HP = 11, 12
        self.L_KN, self.R_KN = 13, 14
        self.L_AN, self.R_AN = 15, 16

    def _smooth(self, arr, cutoff):
        """
        Butterworth Low-pass Filter 적용
        """
        # 데이터가 너무 짧으면 필터링 불가
        if arr.shape[0] < 15:
            return arr

        fs = self.fps
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        # 안전장치
        if normal_cutoff >= 1.0: normal_cutoff = 0.99
        if normal_cutoff <= 0.0: normal_cutoff = 0.01

        # 4차 필터 설계
        b, a = butter(4, normal_cutoff, btype='low', analog=False)

        # 필터 적용 (arr: [Time, Features])
        # filtfilt는 axis=-1이 기본이므로 axis=0(시간축)으로 설정해야 함
        filtered = np.zeros_like(arr)

        # NaN이 있으면 filtfilt가 전체를 NaN으로 만드므로 처리 필요
        # (이미 전처리에서 NaN을 다 채웠다고 가정하지만 안전장치 추가)
        for i in range(arr.shape[1]):
            signal = arr[:, i]
            # 만약 NaN이 섞여있다면 0으로 대체해서라도 돌림 (혹은 보간)
            if np.isnan(signal).any():
                signal = np.nan_to_num(signal)

            try:
                filtered[:, i] = filtfilt(b, a, signal, axis=0)
            except:
                filtered[:, i] = signal  # 에러 시 원본 반환

        return filtered

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
        # gradient는 노이즈를 증폭시키므로, 계산 후 스무딩 필수
        return np.gradient(arr, axis=0)

    def extract(self, kps):
        # 1) Base Angles (8개)
        # 이미 전처리 단계(MotionPreprocessor)에서 좌표 스무딩(15Hz)을 거쳤으므로,
        # 각도 값은 이미 부드럽습니다. 여기서는 추가 스무딩을 안 하거나 아주 약하게만 합니다.
        angles = np.stack([
            self._angle(kps[:, self.L_SH], kps[:, self.L_EL], kps[:, self.L_WR]),
            self._angle(kps[:, self.R_SH], kps[:, self.R_EL], kps[:, self.R_WR]),
            self._angle(kps[:, self.L_EL], kps[:, self.L_SH], kps[:, self.L_HP]),
            self._angle(kps[:, self.R_EL], kps[:, self.R_SH], kps[:, self.R_HP]),
            self._angle(kps[:, self.L_SH], kps[:, self.L_HP], kps[:, self.L_KN]),
            self._angle(kps[:, self.R_SH], kps[:, self.R_HP], kps[:, self.R_KN]),
            self._angle(kps[:, self.L_HP], kps[:, self.L_KN], kps[:, self.L_AN]),
            self._angle(kps[:, self.R_HP], kps[:, self.R_KN], kps[:, self.R_AN]),
        ], axis=1)

        # [선택] 각도 스무딩: 원하면 주석 해제 (보통 생략해도 무방)
        # angles = self._smooth(angles, cutoff=15)

        # 2) Kinetic Chain Velocity
        # 우완 투수 기준 (R_leg는 킥킹/지지, L_leg는 브레이싱)
        # 여기서는 동작의 '속도'를 보므로 미분 노이즈 제거를 위해 필터링 필수

        knee_ext_vel = self._velocity(angles[:, 7])  # R_knee (뒷다리 펴짐/굽힘)

        pelvis_angle = self._angle_2pts(kps[:, self.R_HP], kps[:, self.L_HP])
        pelvis_rot_vel = self._velocity(pelvis_angle)

        trunk_angle = self._angle_2pts(kps[:, self.R_SH], kps[:, self.L_SH])
        trunk_rot_vel = self._velocity(trunk_angle)

        elbow_ext_vel = self._velocity(angles[:, 1])  # R_elbow (던지는 팔)

        wrist_angle = self._angle_2pts(kps[:, self.R_EL], kps[:, self.R_WR])
        shoulder_ir_vel = self._velocity(wrist_angle)

        kinetic_vel = np.stack([
            knee_ext_vel,
            pelvis_rot_vel,
            trunk_rot_vel,
            elbow_ext_vel,
            shoulder_ir_vel
        ], axis=1)

        # [중요] 속도 데이터는 10Hz 정도로 부드럽게 필터링
        kinetic_vel = self._smooth(kinetic_vel, cutoff=self.vel_cutoff)

        final = np.concatenate([angles, kinetic_vel], axis=1)  # (T, 13)
        return final