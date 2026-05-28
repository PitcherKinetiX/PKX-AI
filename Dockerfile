FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

# 시스템 패키지 (OpenCV + mmcv 소스 컴파일용)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# PyTorch는 베이스 이미지에 포함 (CUDA 11.7 + cuDNN 8)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MMPose 스택 설치 (로컬 환경과 동일 버전)
RUN pip install --no-cache-dir openmim ninja
RUN mim install "mmengine==0.10.7"

# mmcv 2.1.0 소스 직접 컴파일 — 로컬과 동일 버전, CUDA ops 보장
RUN git clone -b v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git /tmp/mmcv && \
    cd /tmp/mmcv && \
    FORCE_CUDA=1 pip install --no-cache-dir -v . && \
    rm -rf /tmp/mmcv

RUN mim install "mmdet==3.3.0"
RUN mim install "mmpose==1.3.2"

# numpy 버전 고정
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
# recompile sklearn from source against the pinned numpy to avoid ABI mismatch
RUN pip install --no-cache-dir --force-reinstall --no-binary scikit-learn "scikit-learn==1.3.2"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
