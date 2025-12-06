import torch
import config  # config.py 임포트
from pose2d import Pose2DInferencer


def run_pose_estimation_only(mode="val"):
    """
    Crop 단계 없이 Pose Estimation만 수행하는 함수
    mode: 'train' or 'val'
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print(f"[START] 2D Pose Estimation Only")
    print(f"Mode: {mode.upper()} | Device: {device}")
    print("=" * 60)

    # 1. 경로 설정 (Config 참조)
    if mode == "train":
        # 입력: 이미 Crop이 완료된 영상 폴더
        input_cropped_dir = config.CROPPED_VIDEO_DIR
        json_output_dir = config.JSON_2D_DIR
        vis_output_dir = config.VIS_DIR
    else:
        # 입력: 이미 Crop이 완료된 영상 폴더 (Val)
        input_cropped_dir = config.VAL_CROPPED_VIDEO_DIR
        json_output_dir = config.VAL_JSON_2D_DIR
        vis_output_dir = config.VAL_VIS_DIR

    print(f"Input Directory (Cropped Videos): {input_cropped_dir}")
    print(f"Output Directory (JSON): {json_output_dir}")
    print("-" * 60)

    # 2. Pose2DInferencer 실행
    # (이미 pose2d.py를 최신 로직으로 수정했으므로 그대로 불러와서 쓰면 됩니다)
    inferencer = Pose2DInferencer(
        input_cropped_dir=input_cropped_dir,
        json_output_dir=json_output_dir,
        vis_output_dir=vis_output_dir,
        device=device
    )

    inferencer.process_2d_extraction()

    print("=" * 60)
    print("[COMPLETE] 2D Pose Extraction Finished.")
    print("=" * 60)


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 여기만 수정하세요 ('train' 또는 'val')
    # ---------------------------------------------------------
    MODE = "train"

    run_pose_estimation_only(mode=MODE)