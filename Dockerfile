FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

# apt-get 설치 시 타임존(tzdata) 선택창 등으로 인해 빌드가 멈추는 것을 방지
ENV DEBIAN_FRONTEND=noninteractive

# 시스템 패키지 (OpenCV 및 소스 컴파일을 위한 기본 빌드 환경)
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

# 1. pandas 및 mmengine 필수 기초 패키지 설치
RUN pip install --no-cache-dir "pandas" "mmengine==0.10.7"

# 🔥 [mmcv 수동 컴파일 환경 수정]
# GTX 1060에 100% 매칭되는 Pascal 아키텍처 전용 번호(6.1)를 주입합니다.
COPY mmcv-2.1.0 /app/mmcv-src
RUN export FORCE_CUDA="1" && \
    export TORCH_CUDA_ARCH_LIST="6.1" && \
    export CUDA_HOME=/usr/local/cuda && \
    cd /app/mmcv-src && \
    pip install --no-cache-dir -e .

# 2. mmdet 및 mmpose 설치 (이거는 공식 PyPI 글로벌 서버에서 안전하게 받아집니다)
RUN pip install --no-cache-dir "mmdet==3.3.0"
RUN pip install --no-cache-dir "mmpose==1.3.2"

# NumPy 버전 고정 및 ABI 불일치 방지를 위한 Scikit-learn 소스 재컴파일
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
RUN pip install --no-cache-dir --force-reinstall --no-binary scikit-learn "scikit-learn==1.3.2"

# 현재 디렉토리의 모든 파일 복사 (weights/ 폴더 포함)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]