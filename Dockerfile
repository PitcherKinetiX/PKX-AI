FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

# apt-get 설치 시 타임존(tzdata) 선택창 등으로 인해 빌드가 멈추는 것을 방지
ENV DEBIAN_FRONTEND=noninteractive

# 시스템 패키지 (OpenCV 및 기본 빌드 환경)
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# OpenMMLab 스택 설치를 위한 MIM 및 Ninja 설치
RUN pip install --no-cache-dir openmim ninja
RUN mim install "mmengine==0.10.7"

# 🔥 [수정] 무겁고 에러 잘 나는 소스 컴파일 대신, 베이스 이미지(PyTorch 2.0.1 + CUDA 11.7) 환경에 맞는 mmcv 미리 빌드된 파일 설치
RUN mim install "mmcv==2.1.0" --find-links https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

# MMPose 및 MMDet 설치
RUN mim install "mmdet==3.3.0"
RUN mim install "mmpose==1.3.2"

# NumPy 버전 고정 및 ABI 불일치 방지를 위한 Scikit-learn 소스 재컴파일
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
RUN pip install --no-cache-dir --force-reinstall --no-binary scikit-learn "scikit-learn==1.3.2"

# 현재 디렉토리의 모든 파일 복사
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]