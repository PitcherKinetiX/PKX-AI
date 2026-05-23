FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 시스템 패키지 (OpenCV + mmcv 컴파일용)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch는 베이스 이미지에 포함 (CUDA 12.1 + cuDNN 8)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MMPose 스택 설치
RUN pip install --no-cache-dir openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0"
RUN mim install "mmdet>=3.0.0"
RUN mim install "mmpose>=1.0.0"

COPY . .

# 모델 가중치 사전 다운로드 (빌드 시 CPU 모드, 런타임엔 GPU 자동 사용)
RUN python -c "\
from mmpose.apis import MMPoseInferencer; \
MMPoseInferencer(pose2d='rtmpose-l', det_model='rtmdet-m', device='cpu'); \
print('Model weights downloaded.')"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
