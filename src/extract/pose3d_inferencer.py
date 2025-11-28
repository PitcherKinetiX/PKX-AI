# src/extract/pose3d_inferencer.py
# 3d 추후 개발

'''
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


class Pose3DInferencer:
    def __init__(self, base_dir, device='cuda'):
        self.BASE_DIR = base_dir
        self.INPUT_DIR = os.path.join(base_dir, 'data', 'enhanced_videos')
        self.JSON_OUTPUT_DIR = os.path.join(base_dir, 'data', 'json_3d')
        self.VIS_OUTPUT_DIR = os.path.join(base_dir, 'data', 'vis_check_3d')

        os.makedirs(self.JSON_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.VIS_OUTPUT_DIR, exist_ok=True)

        print("[INFO] Loading Pose3D Inferencer (RTMPose-L + Human3D)...")

        self.inferencer = MMPoseInferencer(
            pose2d='rtmpose-l',
            pose3d='human3d',
            device=device
        )

    def process_3d_extraction(self):
        """ 크롭된 영상에서 3D 좌표를 추출하고 저장합니다. """

        search_patterns = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.mkv']
        video_files = []
        for pattern in search_patterns:
            video_files.extend(glob.glob(os.path.join(self.INPUT_DIR, pattern)))
        video_files = list(set(video_files))

        if not video_files:
            print(f"[WARNING] No cropped videos found in {self.INPUT_DIR}. Skipping 3D extraction.")
            return

        print(f"[INFO] Found {len(video_files)} cropped videos for 3D conversion.")

        for idx, video_path in enumerate(video_files):
            filename = os.path.basename(video_path)
            print(f"\n[{idx + 1}/{len(video_files)}] Processing 3D: {filename}")

            try:
                result_generator = self.inferencer(
                    inputs=video_path,
                    show=False,
                    vis_out_dir=self.VIS_OUTPUT_DIR,
                    det_thr=0.3
                )

                final_json_data = []

                for i, result in enumerate(result_generator):
                    if i % 30 == 0: print(f"   ... frame {i}")

                    predictions = result['predictions'][0]
                    if isinstance(predictions, dict):
                        instances = predictions.get('instances', [])
                    elif isinstance(predictions, list):
                        instances = predictions
                    else:
                        instances = []

                    if instances:
                        person = instances[0]
                        kps_3d = person.get('keypoints_3d', [])
                        if kps_3d is None: kps_3d = []

                        frame_data = {
                            "frame_id": i,
                            "instances": [{
                                "bbox": person.get('bbox', [0, 0, 0, 0]),
                                "bbox_score": person.get('bbox_score', 0.0),
                                "keypoints": person.get('keypoints', []),
                                "keypoints_3d": kps_3d
                            }]
                        }
                        final_json_data.append(frame_data)

                json_name = os.path.splitext(filename)[0] + '.json'
                json_path = os.path.join(self.JSON_OUTPUT_DIR, json_name)

                with open(json_path, 'w') as f:
                    json.dump(final_json_data, f, cls=NumpyEncoder)

                print(f"   [SUCCESS] Saved JSON: {json_path}")
                print(f"   [SUCCESS] Saved Video: {os.path.join(self.VIS_OUTPUT_DIR, filename)}")

            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")
                continue
'''