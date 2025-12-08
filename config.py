# config.py
import os

# 프로젝트 루트 계산
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# train 데이터 경로

TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "train")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "lstm_ae_final.pth")
SCALER_DIR = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
FINE_TUNE_DIR = os.path.join(PROJECT_ROOT, "models", "fine_tune")

VIDEO_DIR = os.path.join(TRAIN_DATA_DIR, "video_data")        # 원본 영상
CROPPED_VIDEO_DIR = os.path.join(TRAIN_DATA_DIR, "cropped_videos")    # 투수만 crop한 영상
KPS_DIR = os.path.join(TRAIN_DATA_DIR, "kps")            # keypoint JSON
JSON_2D_DIR = os.path.join(TRAIN_DATA_DIR, "json_2d")           # 전처리 전 데이터
PROCESSED_DIR = os.path.join(TRAIN_DATA_DIR, "processed")            # 전처리 후 데이터
VIS_DIR = os.path.join(TRAIN_DATA_DIR, "check_2d")

# val 데이터 경로

VAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "val")

VAL_VIDEO_DIR = os.path.join(VAL_DATA_DIR, "val_video_data")        # 원본 영상
VAL_CROPPED_VIDEO_DIR = os.path.join(VAL_DATA_DIR, "val_cropped_videos")    # 투수만 crop한 영상
VAL_KPS_DIR = os.path.join(VAL_DATA_DIR, "val_kps")            # keypoint JSON
VAL_JSON_2D_DIR = os.path.join(VAL_DATA_DIR, "val_json_2d")           # 전처리 전 데이터
VAL_PROCESSED_DIR = os.path.join(VAL_DATA_DIR, "val_processed")            # 전처리 후 데이터
VAL_VIS_DIR = os.path.join(VAL_DATA_DIR, "val_check_2d")
VAL_INFER_DIR = os.path.join(VAL_DATA_DIR, "val_infer_2d")
