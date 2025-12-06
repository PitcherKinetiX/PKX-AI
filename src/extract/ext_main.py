import torch
import config  # config.py 임포트

from crop_pitcher import PitcherCropper
from pose2d import Pose2DInferencer

class PosePipeline:
    def __init__(self, video_input_dir, cropped_output_dir, json_output_dir, vis_output_dir):
        self.video_input_dir = video_input_dir
        self.cropped_output_dir = cropped_output_dir
        self.json_output_dir = json_output_dir
        self.vis_output_dir = vis_output_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run_pipeline(self):
        print("="*60)
        print(f"[PIPELINE START] Device: {self.device}")
        print("="*60)

        # ----------------------------
        # PHASE 1: Crop Pitcher
        # ----------------------------
        print(f"[PHASE 1] Cropping Pitcher from: {self.video_input_dir}")
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
        print(f"[PHASE 2] 2D Pose Extraction to: {self.json_output_dir}")
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
    # train or val
    MODE = "val"  # 'train' or 'val'

    if MODE == "train":
        video_input = config.VIDEO_DIR
        cropped_output = config.CROPPED_VIDEO_DIR
        json_output = config.JSON_2D_DIR
        vis_output = config.VIS_DIR
    elif MODE == "val":
        video_input = config.VAL_VIDEO_DIR
        cropped_output = config.VAL_CROPPED_VIDEO_DIR
        json_output = config.VAL_JSON_2D_DIR
        vis_output = config.VAL_VIS_DIR

    pipeline = PosePipeline(
        video_input_dir=video_input,
        cropped_output_dir=cropped_output,
        json_output_dir=json_output,
        vis_output_dir=vis_output
    )
    pipeline.run_pipeline()