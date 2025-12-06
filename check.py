import json
import numpy as np

# JSON 파일 불러오기
path = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\data\val\val_json_2d\v_1.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)


def get_keypoints(frame):
    """해당 프레임의 keypoints를 numpy array로 변환"""
    inst = frame["instances"][0]  # 사람 1명만 있다고 가정
    kps = np.array(inst["keypoints"]).reshape(-1, 2)  # (17, 2)
    return kps


# 이동량 저장 리스트
movement = []

for i in range(1, len(data)):
    prev_kp = get_keypoints(data[i - 1])
    curr_kp = get_keypoints(data[i])

    # 프레임 이동 거리 계산 (L2 distance)
    dist = np.linalg.norm(curr_kp - prev_kp, axis=1)
    avg_move = np.mean(dist)
    max_move = np.max(dist)

    movement.append({
        "frame": i,
        "avg_move": avg_move,
        "max_move": max_move
    })

# 튀는 프레임 자동 탐지 (평균 이동이 30px 이상일 경우)
threshold = 30
jump_frames = [m for m in movement if m["avg_move"] > threshold]

print("=== 좌표 튐 감지 결과 ===")
for m in jump_frames:
    print(f"Frame {m['frame']:2d} : 평균 이동 {m['avg_move']:.2f}px / 최대 {m['max_move']:.2f}px")

print("\n총 튐 발생 프레임 수:", len(jump_frames))
