import os
import glob
import json
import numpy as np
from mmpose.apis import MMPoseInferencer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Pose2DInferencer:
    def __init__(self,
                 input_cropped_dir,
                 json_output_dir,
                 vis_output_dir,
                 device='cuda'):

        self.INPUT_DIR = input_cropped_dir
        self.JSON_OUTPUT_DIR = json_output_dir
        self.vis_output_dir = vis_output_dir

        os.makedirs(self.JSON_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.vis_output_dir, exist_ok=True)

        print("[INFO] Init Pose2DInferencer (Simple Best-Score Mode)")

        # 시각화가 잘 나온다면 이 설정은 이미 검증된 것임
        self.inferencer = MMPoseInferencer(
            pose2d='rtmpose-l',
            det_model='rtmdet-m',
            device=device
        )

    def process_2d_extraction(self):
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files += glob.glob(os.path.join(self.INPUT_DIR, ext))
        video_files = list(set(video_files))

        print(f"[INFO] Found {len(video_files)} videos.")

        for vid in video_files:
            fname = os.path.basename(vid)
            print(f"[2D Inference] Processing: {fname}")

            # det_thr=0.4: 잡다한 배경 인물 제외
            result_generator = self.inferencer(
                inputs=vid,
                show=False,
                vis_out_dir=self.vis_output_dir,
                det_thr=0.4
            )

            frames_data = []

            for frame_idx, res in enumerate(result_generator):

                # 1. 결과 데이터 파싱 (리스트/딕셔너리 호환)
                predictions = res.get("predictions", [])
                if not predictions:
                    continue

                raw_preds = predictions[0]
                inst_list = []

                if isinstance(raw_preds, list):
                    inst_list = raw_preds
                elif isinstance(raw_preds, dict):
                    inst_list = raw_preds.get("instances", [])

                if not inst_list:
                    continue

                # 2. [핵심] 그냥 가장 점수 높은 사람 1명만 선택
                # 시각화에 나오는 '그 사람'이 바로 점수가 제일 높은 사람입니다.
                best_person = max(inst_list, key=lambda x: x.get('bbox_score', 0))

                # 3. 저장
                frames_data.append({
                    "frame_id": frame_idx,
                    "instances": [{
                        "track_id": best_person.get("track_id", -1),  # ID 있으면 저장, 없으면 -1
                        "bbox": best_person.get("bbox", []),
                        "bbox_score": best_person.get("bbox_score", 0.0),
                        "keypoints": best_person.get("keypoints", []),
                        "keypoints_score": best_person.get("keypoints_score", [])
                    }]
                })

            # JSON 파일 쓰기
            if frames_data:
                json_path = os.path.join(self.JSON_OUTPUT_DIR, os.path.splitext(fname)[0] + ".json")
                with open(json_path, "w") as f:
                    json.dump(frames_data, f, cls=NumpyEncoder)
                print(f"[SAVED] {len(frames_data)} frames -> {json_path}")
            else:
                print(f"[WARNING] No data extracted for {fname}")

        return self.JSON_OUTPUT_DIR