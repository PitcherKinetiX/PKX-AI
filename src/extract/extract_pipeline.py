# src/extract/pose_pipeline.py

import os
import torch
import cv2

# 같은 디렉토리에 있는 모듈 import
from crop_pitcher import PitcherCropper
from pose2d_inferencer import Pose2DInferencer  # [수정] 3D -> 2D Inferencer


class PosePipeline:
    def __init__(self, project_root_path):
        self.project_root = project_root_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run_pipeline(self):
        print("=" * 60)
        print("[PIPELINE START] PitcherKinetiX Analysis (2D Focus)")
        print(f"[INFO] Using Device: {self.device}")
        print("=" * 60)

        # --- PHASE 1: 투수 크롭 ---
        print("\n[PHASE 1] Starting Pitcher Cropper (Isolating the Pitcher)...")
        try:
            cropper = PitcherCropper(base_dir=self.project_root, device=self.device)
            cropper.process()
            crop_output_dir = os.path.join(self.project_root, 'data', 'cropped_videos')
            print("[PHASE 1 SUCCESS] Videos cropped and saved to data/cropped_videos.")

        except Exception as e:
            print(f"[PHASE 1 FAILURE] Critical error during cropping: {e}")
            print("[CRITICAL] Cannot proceed.")
            return

        # --- PHASE 2: 2D 좌표 추출 ---
        print("\n[PHASE 2] Starting 2D Inferencer (Extracting Keypoints)...")
        try:
            inferencer_2d = Pose2DInferencer(
                base_dir=self.project_root,
                device=self.device
            )
            # Pose2DInferencer는 크롭된 폴더(crop_output_dir)를 읽도록 기본 설정되어 있음
            json_output_dir = inferencer_2d.process_2d_extraction()
            print(f"[PHASE 2 SUCCESS] 2D Coordinates saved to {json_output_dir}.")

        except Exception as e:
            print(f"[PHASE 2 FAILURE] Critical error during 2D extraction: {e}")
            return

        print("\n[PIPELINE COMPLETE] Project data acquisition finished successfully!")
        print("=" * 60)


if __name__ == "__main__":
    import torch

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_path = os.path.dirname(os.path.dirname(current_file_dir))

    pipeline = PosePipeline(project_root_path)
    pipeline.run_pipeline()