# 포즈 파이프라인 사전학습 가중치 (오프라인용)

AI 서버가 `download.openmmlab.com` 에 접속할 수 없는 환경(현재 빌드/런타임 망)을 위해,
포즈/디텍터 가중치를 **이 폴더에 직접 넣어** 런타임 다운로드 없이 동작하게 한다.

[crop_pitcher.py](../src/extract/crop_pitcher.py) / [pose2d.py](../src/extract/pose2d.py) 는
[pose_weights.py](../src/extract/pose_weights.py) 를 통해 아래 파일이 있으면 자동으로 사용한다.
(파일이 없으면 기존처럼 별칭으로 온라인 다운로드를 시도 → openmmlab 막힌 망에서는 실패)

## 넣어야 할 파일 (정확한 이름으로 저장)

| 저장 이름                 | 원본(openmmlab) URL |
|---------------------------|----------------------|
| `rtmpose-l.pth`           | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth |
| `rtmdet-m-person.pth`     | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth |

## 받는 방법 (openmmlab 막힌 경우)

openmmlab 이 닿는 다른 망에서 위 두 파일을 받아 **위 표의 이름으로 변경**한 뒤 이 폴더에 둔다.
(예: 모바일 핫스팟 / VPN / Google Colab 등 — 받은 뒤 이 repo `weights/` 로 복사)

```bash
# openmmlab 이 닿는 환경에서:
curl -L -o rtmpose-l.pth \
  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth"
curl -L -o rtmdet-m-person.pth \
  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
```

## 적용

파일을 넣은 뒤 도커 이미지를 재빌드하면 `COPY . .` 단계에서 `/app/weights/` 로 들어가
런타임에 openmmlab 없이 동작한다. (가중치 파일은 용량이 커서 git 에는 커밋하지 않음 — `.gitignore` 처리)
