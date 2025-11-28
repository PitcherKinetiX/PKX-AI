# src/extract/pose2d_inferencer.py

import os
import glob
import json
import numpy as np
from mmpose.apis import MMPoseInferencer


class NumpyEncoder(json.JSONEncoder):
    """ NumPy 데이터 타입을 JSON으로 직렬화하기 위한 인코더 """

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Pose2DInferencer:
    def __init__(self, base_dir, device='cuda'):
        self.BASE_DIR = base_dir
        # 입력: 크롭된 영상 폴더
        self.INPUT_DIR = os.path.join(base_dir, 'data', 'cropped_videos')
        # 출력: 2D JSON 저장
        self.JSON_OUTPUT_DIR = os.path.join(base_dir, 'data', 'json_2d')
        # 출력: 2D 시각화 영상 저장
        self.VIS_OUTPUT_DIR = os.path.join(base_dir, 'data', 'vis_check_2d')

        os.makedirs(self.JSON_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.VIS_OUTPUT_DIR, exist_ok=True)

        print("[INFO] Loading Pose2D Inferencer (RTMPose-L 2D Only)...")

        self.inferencer = MMPoseInferencer(
            pose2d='rtmpose-l',
            pose3d=None,  # 3D 기능 명시적 비활성화
            device=device
        )

    def process_2d_extraction(self):
        """ 크롭된 영상에서 2D 좌표를 추출하고 저장합니다. """

        search_patterns = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.mkv']
        video_files = []
        for pattern in search_patterns:
            video_files.extend(glob.glob(os.path.join(self.INPUT_DIR, pattern)))
        video_files = list(set(video_files))

        if not video_files:
            print(f"[WARNING] No cropped videos found in {self.INPUT_DIR}. Skipping 2D extraction.")
            return self.INPUT_DIR  # 다음 단계에서 사용할 입력 경로 (현재는 크롭 폴더)

        print(f"[INFO] Found {len(video_files)} cropped videos for 2D extraction.")

        for idx, video_path in enumerate(video_files):
            filename = os.path.basename(video_path)
            print(f"\n[{idx + 1}/{len(video_files)}] Processing 2D: {filename}")

            try:
                # MMPose의 기본 기능으로 2D 좌표와 시각화 영상을 한 번에 처리
                result_generator = self.inferencer(
                    inputs=video_path,
                    show=False,
                    vis_out_dir=self.VIS_OUTPUT_DIR,  # 2D 뼈대 영상 저장
                    det_thr=0.3  # 감지 임계값 (이미 크롭됨)
                )

                final_json_data = []

                for i, result in enumerate(result_generator):
                    if i % 30 == 0: print(f"   ... frame {i}")

                    predictions = result['predictions'][0]
                    # 크롭 영상은 투수 1명만 있다고 가정
                    if isinstance(predictions, dict):
                        instances = predictions.get('instances', [])
                    elif isinstance(predictions, list):
                        instances = predictions
                    else:
                        instances = []

                    if instances:
                        person = instances[0]

                        frame_data = {
                            "frame_id": i,
                            "instances": [{
                                "bbox": person.get('bbox', [0, 0, 0, 0]),
                                "bbox_score": person.get('bbox_score', 0.0),
                                "keypoints": person.get('keypoints', []),  # 2D 좌표
                                "keypoints_3d": []  # 3D 데이터 없음
                            }]
                        }
                        final_json_data.append(frame_data)

                # JSON 저장
                json_name = os.path.splitext(filename)[0] + '.json'
                json_path = os.path.join(self.JSON_OUTPUT_DIR, json_name)

                with open(json_path, 'w') as f:
                    json.dump(final_json_data, f, cls=NumpyEncoder)

                print(f"   [SUCCESS] Saved JSON: {json_path}")
                print(f"   [SUCCESS] Saved Video: {os.path.join(self.VIS_OUTPUT_DIR, filename)}")

            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")
                continue

        return self.JSON_OUTPUT_DIR  # 다음 단계에서 사용할 JSON 폴더 경로