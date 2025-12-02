from process_motion import MotionPreprocessor
from extract_joint_angle import JointAngleExtractor
from sliding_window import SlidingWindowGenerator
import numpy as np
import os

# 기본 경로
DEFAULT_JSON_DIR = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\data\train\json_2d"
DEFAULT_OUTPUT_DIR = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\data\train\processed"
DEFAULT_KPS_DIR = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\data\train\kps"

def run_preprocessing(
        json_dir: str = DEFAULT_JSON_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        kps_dir: str = DEFAULT_KPS_DIR,
        window_size: int = 30,
        stride: int = 5,
        verbose: bool = True):

    os.makedirs(output_dir, exist_ok=True)

    processor = MotionPreprocessor()
    angle_extractor = JointAngleExtractor()
    window_gen = SlidingWindowGenerator(window_size=window_size, stride=stride)

    processed_files = {}

    file_list = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

    for vid, file in enumerate(file_list):
        json_path = os.path.join(json_dir, file)

        if verbose:
            print(f"[PREPROCESSING] {json_path}")

        base = os.path.splitext(file)[0]

        # raw kps 저장 경로
        raw_save_path = os.path.join(kps_dir, f"{base}_raw_kps.npy")

        # 1) preprocess
        kps = processor.process_file(
            file_path=json_path,
            save_raw_kps_path=raw_save_path
        )  # (240, 17, 2)

        # 2) joint angles only  (240, 13)
        angles = angle_extractor.extract(kps)

        # 3) sliding window (N, 30, 13)
        windows = window_gen.generate(angles)
        N = windows.shape[0]

        video_ids = np.full(N, vid, dtype=np.int32)
        filenames = np.array([file] * N)

        # 4) 저장
        save_path = os.path.join(output_dir, f"{base}_processed.npz")

        np.savez_compressed(
            save_path,
            windows=windows,
            video_ids=video_ids,
            filenames=filenames
        )

        processed_files[file] = save_path

        if verbose:
            print(f"[SAVED] → {save_path}")
            print(f"   raw_kps  : {raw_save_path}")
            print(f"   windows  : {windows.shape}")
            print(f"   video_ids: {video_ids.shape}")
            print(f"   filenames: {filenames.shape}\n")

    return processed_files


def combine_processed(output_dir: str, save_filename="2d_data.npz"):

    all_windows = []
    all_vids = []
    all_names = []

    processed_files = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith("_processed.npz")
    ])

    if len(processed_files) == 0:
        raise ValueError("[ERROR] No processed npz found in folder.")

    for f in processed_files:
        full_path = os.path.join(output_dir, f)
        data = np.load(full_path)

        windows = data["windows"]
        vids = data["video_ids"]
        names = data["filenames"]

        all_windows.append(windows)
        all_vids.append(vids)
        all_names.append(names)

        print(f"[LOADED] {f} -> {windows.shape}")

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
    run_preprocessing()
    combine_processed(DEFAULT_OUTPUT_DIR)
