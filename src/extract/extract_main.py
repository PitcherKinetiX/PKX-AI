# src/extract/extract_main.py

import os
import torch

from crop_pitcher import PitcherCropper
from pose2d_inferencer import Pose2DInferencer


class PosePipeline:
    def __init__(self, video_input_dir, cropped_output_dir, json_output_dir, vis_output_dir):
        self.video_input_dir = video_input_dir
        self.cropped_output_dir = cropped_output_dir
        self.json_output_dir = json_output_dir
        self.vis_output_dir = vis_output_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run_pipeline(self):
        print("="*60)
        print("[PIPELINE START]")
        print("="*60)

        # ----------------------------
        # PHASE 1: Crop Pitcher
        # ----------------------------
        print("[PHASE 1] Cropping Pitcher...")
        cropper = PitcherCropper(
            input_video_dir=self.video_input_dir,
            output_video_dir=self.cropped_output_dir,
            device=self.device
        )
        cropper.process()
        print("[PHASE 1 DONE]")

        # ----------------------------
        # PHASE 2: Pose 2D Extraction
        # ----------------------------
        print("[PHASE 2] 2D Pose Extraction...")
        inferencer = Pose2DInferencer(
            input_cropped_dir=self.cropped_output_dir,
            json_output_dir=self.json_output_dir,
            vis_output_dir=self.vis_output_dir,
            device=self.device
        )
        inferencer.process_2d_extraction()
        print("[PHASE 2 DONE]")

        print("[PIPELINE COMPLETE]")


if __name__ == "__main__":

    project_root = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\data\train"

    video_input_dir = os.path.join(project_root, "video_data")
    cropped_output_dir = os.path.join(project_root, "cropped_videos")
    json_output_dir = os.path.join(project_root, "json_2d")
    vis_output_dir = os.path.join(project_root, "check_2d")

    pipeline = PosePipeline(
        video_input_dir=video_input_dir,
        cropped_output_dir=cropped_output_dir,
        json_output_dir=json_output_dir,
        vis_output_dir=vis_output_dir
    )
    pipeline.run_pipeline()
