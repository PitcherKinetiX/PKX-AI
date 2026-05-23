FROM python:3.11-slim

WORKDIR /app

# OpenCV and MMPose system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MMPose and its dependencies via openmim
RUN pip install --no-cache-dir openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0"
RUN mim install "mmdet>=3.0.0"
RUN mim install "mmpose>=1.0.0"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
