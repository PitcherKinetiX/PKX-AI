import os
import glob
import json
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class PitcherCropper:
    def __init__(self,
                 input_video_dir,
                 output_video_dir,
                 crop_w=900,
                 crop_h=900,
                 score_thr=0.5,
                 ratio_thr=1.5,
                 device='cuda'):

        self.INPUT_DIR = input_video_dir
        self.OUTPUT_DIR = output_video_dir

        self.CROP_WIDTH = crop_w
        self.CROP_HEIGHT = crop_h
        self.SCORE_THRESHOLD = score_thr
        self.RATIO_THRESHOLD = ratio_thr

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self.inferencer = MMPoseInferencer(
            pose2d='rtmpose-l',
            pose3d=None,
            device=device
        )

    # -------------------------
    # 사람 매칭
    # -------------------------
    def match_people(self, tracks, current_people):
        valid_people = [
            p for p in current_people
            if p['bbox_score'] >= self.SCORE_THRESHOLD
        ]
        if not valid_people:
            return

        if not tracks:
            for i, person in enumerate(valid_people):
                tracks[i] = {'data': [person], 'last_kps': person['keypoints']}
            return

        matched = set()
        for track_id, info in tracks.items():
            last_kps = np.array(info['last_kps'])[:, :2]
            valid = (last_kps[:, 0] > 1) & (last_kps[:, 1] > 1)
            center_prev = np.mean(last_kps[valid], axis=0) if np.sum(valid) > 0 else np.mean(last_kps, axis=0)

            best_idx, best_dist = -1, 1e9
            for idx, p in enumerate(valid_people):
                if idx in matched:
                    continue
                curr = np.array(p['keypoints'])[:, :2]
                valid_c = (curr[:, 0] > 1) & (curr[:, 1] > 1)
                center_curr = np.mean(curr[valid_c], axis=0) if np.sum(valid_c) > 0 else np.mean(curr, axis=0)
                dist = np.linalg.norm(center_curr - center_prev)

                if dist < best_dist:
                    best_idx, best_dist = idx, dist

            if best_idx != -1 and best_dist < 200:
                info['data'].append(valid_people[best_idx])
                info['last_kps'] = valid_people[best_idx]['keypoints']
                matched.add(best_idx)

    # -------------------------
    # 베스트 트랙 선택
    # -------------------------
    def track_variance_score(self, track_id, data):
        if len(data) < 15:
            return -1

        bboxes = np.array([p['bbox'] for p in data])
        if len(bboxes.shape) == 3:
            bboxes = bboxes[:, 0, :]

        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        ratio = np.mean(h / (w + 1e-6))

        if ratio < self.RATIO_THRESHOLD:
            return 0

        kps = np.array([p['keypoints'] for p in data])[:, :, :2]
        valid = (kps[:, :, 0] > 1) & (kps[:, :, 1] > 1)
        x = np.where(valid, kps[:, :, 0], np.nan)
        y = np.where(valid, kps[:, :, 1], np.nan)

        return np.nansum(np.nanstd(x, axis=0)) + np.nansum(np.nanstd(y, axis=0))

    # -------------------------
    # [수정됨] 1.5배 확장된 정사각형 bbox 생성 함수
    # -------------------------
    def make_square_bbox(self, bbox, frame_width, frame_height):
        """
        첫 프레임의 BBox 높이(H_raw)를 기준으로 1.5배 확장된 정사각형 크롭 박스를 계산합니다.
        """
        x1, y1, x2, y2 = bbox

        # 1. 원본 BBox 높이 계산
        H_raw = y2 - y1

        # 2. 크기 확장 (1.5배)
        side = H_raw * 1.5  # 새로운 정사각형의 한 변의 길이 (1.5x)

        # 3. 중심점 계산
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # 4. 최종 BBox 좌표 계산
        new_x1 = int(cx - side / 2)
        new_x2 = int(cx + side / 2)
        new_y1 = int(cy - side / 2)
        new_y2 = int(cy + side / 2)

        # 화면 경계 클램프 (화면 밖으로 나가지 않도록 조정)
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(frame_width, new_x2)
        new_y2 = min(frame_height, new_y2)

        # 최종 BBox가 너무 작아지는 것을 방지 (최소 100x100)
        if new_x2 - new_x1 < 100: new_x2 = new_x1 + 100
        if new_y2 - new_y1 < 100: new_y2 = new_y1 + 100

        return [new_x1, new_y1, new_x2, new_y2]

    # -------------------------
    # crop frame
    # -------------------------
    def crop_frame(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        target_ratio = self.CROP_WIDTH / self.CROP_HEIGHT
        crop_ratio = cw / ch

        if crop_ratio > target_ratio:
            new_w = self.CROP_WIDTH
            new_h = int(new_w / crop_ratio)
            resized = cv2.resize(crop, (new_w, new_h))
            pad = self.CROP_HEIGHT - new_h
            return cv2.copyMakeBorder(resized, pad // 2, pad - pad // 2, 0, 0,
                                      cv2.BORDER_CONSTANT)
        else:
            new_h = self.CROP_HEIGHT
            new_w = int(new_h * crop_ratio)
            resized = cv2.resize(crop, (new_w, new_h))
            pad = self.CROP_WIDTH - new_w
            return cv2.copyMakeBorder(resized, 0, 0, pad // 2, pad - pad // 2,
                                      cv2.BORDER_CONSTANT)

    # -------------------------
    # main
    # -------------------------
    def process(self):
        videos = []
        for ptn in ['*.mp4', '*.MP4', '*.avi', '*.mov']:
            videos += glob.glob(os.path.join(self.INPUT_DIR, ptn))
        videos = list(set(videos))

        for vid in videos:
            name = os.path.basename(vid)
            print(f"[Processing] {name}")

            gen = self.inferencer(inputs=vid, show=False, det_thr=self.SCORE_THRESHOLD)
            tracks = {}

            for i, out in enumerate(gen):
                preds = out['predictions'][0]
                people = preds if isinstance(preds, list) else preds.get('instances', [])
                self.match_people(tracks, people)

            # best track 찾기 (코드 동일)
            best_id, best_score = -1, -1
            for tid, info in tracks.items():
                score = self.track_variance_score(tid, info['data'])
                if score > best_score:
                    best_score = score
                    best_id = tid

            if best_id == -1:
                print(f"[Warning] No pitcher found in {name}")
                continue

            track = tracks[best_id]
            first_bbox = track['data'][0]['bbox']
            bbox = first_bbox[0] if isinstance(first_bbox[0], (list, np.ndarray)) else first_bbox

            cap = cv2.VideoCapture(vid)
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fixed_bbox = self.make_square_bbox(bbox, fw, fh)
            print(f"[BBox] {fixed_bbox}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(
                os.path.join(self.OUTPUT_DIR, name),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (self.CROP_WIDTH, self.CROP_HEIGHT)
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cropped = self.crop_frame(frame, fixed_bbox)
                out.write(cropped)

            out.release()
            cap.release()

        print("Cropping Complete.")