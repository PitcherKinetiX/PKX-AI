import os
import numpy as np
import config

from process_motion import MotionPreprocessor
from process_joint_angle import JointAngleExtractor
from sliding_window import SlidingWindowGenerator


def run_preprocessing(
        json_dir,
        output_dir,
        kps_dir,
        window_size=24,
        stride=2,
        verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(kps_dir, exist_ok=True)

    processor = MotionPreprocessor()
    angle_extractor = JointAngleExtractor()
    window_gen = SlidingWindowGenerator(window_size=window_size, stride=stride)

    processed_files = {}

    if not os.path.exists(json_dir):
        print(f"[ERROR] JSON directory not found: {json_dir}")
        return {}

    file_list = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

    if not file_list:
        print(f"[WARNING] No JSON files found in {json_dir}")
        return {}

    for vid, file in enumerate(file_list):
        json_path = os.path.join(json_dir, file)

        if verbose:
            print(f"[PREPROCESSING] {json_path}")

        base = os.path.splitext(file)[0]
        raw_save_path = os.path.join(kps_dir, f"{base}_raw_kps.npy")

        # 1) preprocess (Interpolate -> Smooth -> Norm)
        # 이제 process_motion.py의 process_file이 save_raw_kps_path를 받으므로 정상 작동합니다.
        kps = processor.process_file(
            file_path=json_path,
            save_raw_kps_path=None,
            save_processed_kps_path=raw_save_path
        )  # (240, 17, 2)

        # 2) joint angles & velocity extraction
        angles = angle_extractor.extract(kps)  # (240, 13)

        # 3) sliding window
        windows = window_gen.generate(angles)  # (N, window_size, 13)
        N = windows.shape[0]

        video_ids = np.full(N, vid, dtype=np.int32)
        filenames = np.array([file] * N)

        # 4) Save individual npz
        save_path = os.path.join(output_dir, f"{base}_processed.npz")

        np.savez_compressed(
            save_path,
            windows=windows,
            video_ids=video_ids,
            filenames=filenames
        )

        processed_files[file] = save_path

        if verbose:
            print(f"[SAVED] -> {save_path} (Windows: {windows.shape})")

    return processed_files


def combine_processed(output_dir, save_filename="2d_data.npz"):
    all_windows = []
    all_vids = []
    all_names = []

    processed_files = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith("_processed.npz")
    ])

    if len(processed_files) == 0:
        print("[ERROR] No processed npz found in folder to combine.")
        return None, None, None

    for f in processed_files:
        full_path = os.path.join(output_dir, f)
        data = np.load(full_path)

        all_windows.append(data["windows"])
        all_vids.append(data["video_ids"])
        all_names.append(data["filenames"])

    merged_windows = np.concatenate(all_windows, axis=0)
    merged_vids = np.concatenate(all_vids, axis=0)
    merged_names = np.concatenate(all_names, axis=0)

    print("\n[COMBINED] Final dataset")
    print(f"windows   : {merged_windows.shape}")
    print(f"video_ids : {merged_vids.shape}")
    print(f"filenames : {merged_names.shape}")

    save_path = os.path.join(output_dir, save_filename)
    np.savez_compressed(
        save_path,
        windows=merged_windows,
        video_ids=merged_vids,
        filenames=merged_names
    )
    print(f"[SAVED] merged dataset -> {save_path}")

    return merged_windows, merged_vids, merged_names


if __name__ == "__main__":
    # -------------------------------------------------------
    # [설정] 여기서 'train' 또는 'val'을 선택하세요.
    # -------------------------------------------------------
    MODE = "val"  # 'train' or 'val'

    if MODE == "train":
        target_json_dir = config.JSON_2D_DIR
        target_output_dir = config.PROCESSED_DIR
        target_kps_dir = config.KPS_DIR
    elif MODE == "val":
        target_json_dir = config.VAL_JSON_2D_DIR
        target_output_dir = config.VAL_PROCESSED_DIR
        target_kps_dir = config.VAL_KPS_DIR

    print(f"[START] Preprocessing Mode: {MODE}")
    run_preprocessing(
        json_dir=target_json_dir,
        output_dir=target_output_dir,
        kps_dir=target_kps_dir
    )
    combine_processed(output_dir=target_output_dir)