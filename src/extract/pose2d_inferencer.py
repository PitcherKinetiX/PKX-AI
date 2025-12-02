# src/extract/pose2d_inferencer.py

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
        self.VIS_OUTPUT_DIR = vis_output_dir

        os.makedirs(self.JSON_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.VIS_OUTPUT_DIR, exist_ok=True)

        print("[INFO] Init Pose2D Inferencer (RTMPose-L)")

        self.inferencer = MMPoseInferencer(
            pose2d='rtmpose-l',
            pose3d=None,
            device=device
        )

    def process_2d_extraction(self):
        video_files = []
        for p in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files += glob.glob(os.path.join(self.INPUT_DIR, p))
        video_files = list(set(video_files))

        print(f"[INFO] Found {len(video_files)} cropped videos.")

        for vid in video_files:
            fname = os.path.basename(vid)
            print(f"[2D] {fname}")

            result_gen = self.inferencer(
                inputs=vid,
                show=False,
                vis_out_dir=self.VIS_OUTPUT_DIR,
                det_thr=0.3
            )

            frames = []
            for i, res in enumerate(result_gen):
                preds = res["predictions"][0]

                if isinstance(preds, dict):
                    inst = preds.get("instances", [])
                else:
                    inst = preds

                if inst:
                    p = inst[0]
                    frames.append({
                        "frame_id": i,
                        "instances": [{
                            "bbox": p.get("bbox", [0,0,0,0]),
                            "bbox_score": p.get("bbox_score", 0.0),
                            "keypoints": p.get("keypoints", []),
                            "keypoints_3d": []
                        }]
                    })

            json_path = os.path.join(self.JSON_OUTPUT_DIR, fname.replace(".mp4", ".json"))
            with open(json_path, "w") as f:
                json.dump(frames, f, cls=NumpyEncoder)

            print(f"[SAVED] json: {json_path}")

        return self.JSON_OUTPUT_DIR
