import os
import os.path as osp


def build_pose_inferencer_kwargs(device):
    """
    MMPoseInferencer 생성 kwargs 구성.

    PKX_WEIGHTS_DIR(기본 /app/weights)에 로컬 가중치가 있으면 그 파일을 사용해
    런타임에 download.openmmlab.com 에서 가중치를 받지 않는다(오프라인 동작).
    파일이 없으면 기존처럼 별칭으로 폴백(온라인 다운로드).

    기대 파일:
      - {PKX_WEIGHTS_DIR}/rtmpose-l.pth        (포즈: rtmpose-l body7)
      - {PKX_WEIGHTS_DIR}/rtmdet-m-person.pth  (사람 디텍터: rtmdet_m coco-obj365-person)
    """
    weights_dir = os.environ.get("PKX_WEIGHTS_DIR", "/app/weights")
    pose_w = osp.join(weights_dir, "rtmpose-l.pth")
    det_w = osp.join(weights_dir, "rtmdet-m-person.pth")

    kwargs = dict(pose2d="rtmpose-l", device=device)

    # 포즈 가중치: 로컬 파일이 있으면 사용 (config는 별칭으로 패키지에서 해석 → 네트워크 불필요)
    if osp.exists(pose_w):
        kwargs["pose2d_weights"] = pose_w

    # 디텍터: 로컬 가중치가 있으면 mmpose 패키지에 동봉된 사람 디텍터 config + 로컬 가중치 사용
    if osp.exists(det_w):
        try:
            from mmengine.config.utils import MODULE2PACKAGE
            from mmengine.utils import get_installed_path
            mmpose_path = get_installed_path(MODULE2PACKAGE["mmpose"])
            det_cfg = osp.join(
                mmpose_path, ".mim",
                "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py",
            )
            kwargs["det_model"] = det_cfg
            kwargs["det_weights"] = det_w
        except Exception:
            kwargs["det_model"] = "rtmdet-m"
    else:
        # 로컬 디텍터 가중치 없음 → 기존 동작(별칭, 온라인 폴백)
        kwargs["det_model"] = "rtmdet-m"

    return kwargs
