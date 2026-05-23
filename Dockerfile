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
RUN mim install "mmengine==0.10.4"
RUN mim install "mmcv==2.2.0"
RUN mim install "mmdet==3.3.0"
RUN mim install "mmpose==1.3.2"

# mim install upgrades numpy to 2.x; force back to 1.x for PyTorch 2.2 compatibility
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
# recompile sklearn from source against the pinned numpy to avoid ABI mismatch
RUN pip install --no-cache-dir --force-reinstall --no-binary scikit-learn "scikit-learn==1.3.2"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
