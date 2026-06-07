import sys
import os
import tempfile

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "src", "visualize"))
sys.path.insert(0, os.path.join(_project_root, "src", "extract"))
sys.path.insert(0, os.path.join(_project_root, "src", "preprocess"))
sys.path.insert(0, _project_root)

import logging
import traceback
import uuid
import threading
import socket
import urllib.request
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# urllib/torch.hub 등 블로킹 소켓에 stall 타임아웃 적용 (네트워크 정지 시 잡이 무한 대기하지 않고 예외 → FAILED 처리)
# uvicorn(asyncio)은 논블로킹 소켓을 쓰므로 영향 없음.
socket.setdefaulttimeout(300)
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeStartResponse,
    AnalyzeStatusResponse,
    TrainRequest,
    TrainStartResponse,
    TrainStatusResponse,
)
import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET_NAME", "pitcherkinetix1")

app = FastAPI(
    title="PKX-AI Analysis API",
    description="투구 동작 분석 AI API - 13개 생체역학 특징 전체 반환",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.error("422 Validation error | body: %s | errors: %s", await request.body(), exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def _download_video(file_id_path: str, video_url: str | None, local_path: str):
    if video_url:
        urllib.request.urlretrieve(video_url, local_path)
    else:
        from google.cloud import storage
        client = storage.Client()
        client.bucket(GCS_BUCKET).blob(file_id_path).download_to_filename(local_path)


def _download_url(url: str, local_path: str):
    """Signed GET URL → 로컬 파일로 다운로드."""
    urllib.request.urlretrieve(url, local_path)


def _upload_url(url: str, local_path: str, content_type: str = "application/octet-stream"):
    """Signed PUT URL로 로컬 파일을 업로드 (백엔드가 서명한 content_type과 일치해야 함)."""
    with open(local_path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req) as resp:
        return resp.status


def _filename_from_url(url: str, default: str = "video.mp4") -> str:
    """Signed URL 경로에서 원본 파일명(확장자 보존)을 추출."""
    name = os.path.basename(unquote(urlparse(url).path))
    return name or default


def _run_pipeline(video_dir: str):
    """Run crop → pose2d → preprocess for all videos in video_dir."""
    from ext_main import PosePipeline
    from pre_main import run_preprocessing

    cropped_dir = os.path.join(video_dir, "cropped")
    json_dir = os.path.join(video_dir, "json_2d")
    vis_dir = os.path.join(video_dir, "vis")

    PosePipeline(
        video_input_dir=video_dir,
        cropped_output_dir=cropped_dir,
        json_output_dir=json_dir,
        vis_output_dir=vis_dir,
    ).run_pipeline()

    run_preprocessing(
        json_dir=json_dir,
        output_dir=config.VAL_PROCESSED_DIR,
        kps_dir=config.VAL_KPS_DIR,
    )


def _analyze_sync(request: AnalyzeRequest):
    """전체 분석 파이프라인을 동기로 실행하고 리포트(dict)를 반환."""
    from report import generate_report_json

    # "videos/4/1779545132787_f53fceea_test.mp4" → "1779545132787_f53fceea_test"
    file_id = os.path.splitext(os.path.basename(request.fileId))[0]
    npz_path = os.path.join(config.VAL_PROCESSED_DIR, f"{file_id}_processed.npz")

    if not os.path.exists(npz_path):
        video_filename = os.path.basename(request.fileId)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_video = os.path.join(tmpdir, video_filename)
            try:
                _download_video(request.fileId, request.videoUrl, local_video)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"GCS 다운로드 실패: {e}")
            try:
                _run_pipeline(tmpdir)
            except Exception as e:
                log.error("Pipeline failed:\n%s", traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"전처리 파이프라인 실패: {e}")

    # 개인화 모델/통계 다운로드 (있으면). 없으면 generate_report_json이 기본 경로로 폴백.
    with tempfile.TemporaryDirectory() as model_dir:
        user_model_path = None
        user_stats_path = None
        if request.userModelUrl and request.userStatsUrl:
            try:
                user_model_path = os.path.join(model_dir, "user_specific_ae.pth")
                user_stats_path = os.path.join(model_dir, "user_stats.pkl")
                _download_url(request.userModelUrl, user_model_path)
                _download_url(request.userStatsUrl, user_stats_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"개인화 모델 다운로드 실패: {e}")

        try:
            return generate_report_json(
                file_id=file_id,
                user_model_path=user_model_path,
                user_stats_path=user_stats_path,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"분석 데이터를 찾을 수 없습니다: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {e}")


# ── 비동기 잡 관리 (Cloudflare 100초 제한 회피) ──────────────────────────
# GPU 1개라 max_workers=1 로 직렬 처리 (동시 실행 OOM도 예방)
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **fields):
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _run_job(job_id: str, request: AnalyzeRequest):
    _set_job(job_id, status="RUNNING")
    try:
        result = _analyze_sync(request)
        _set_job(job_id, status="DONE", result=result)
        log.info("Analysis job %s DONE (analysisId=%s)", job_id, request.analysisId)
    except HTTPException as e:
        _set_job(job_id, status="FAILED", error=str(e.detail))
        log.error("Analysis job %s FAILED (analysisId=%s): %s", job_id, request.analysisId, e.detail)
    except Exception as e:  # noqa: BLE001
        log.error("Analysis job %s crashed:\n%s", job_id, traceback.format_exc())
        _set_job(job_id, status="FAILED", error=str(e))


@app.post("/api/analyze/start", response_model=AnalyzeStartResponse)
def analyze_start(request: AnalyzeRequest):
    """분석을 백그라운드로 시작하고 jobId를 즉시 반환 (짧은 응답 → Cloudflare 524 회피)."""
    job_id = str(uuid.uuid4())
    _set_job(job_id, status="PENDING", result=None, error=None)
    _executor.submit(_run_job, job_id, request)
    log.info("Analysis job %s queued (analysisId=%s)", job_id, request.analysisId)
    return AnalyzeStartResponse(jobId=job_id, status="PENDING")


@app.get("/api/analyze/status/{job_id}", response_model=AnalyzeStatusResponse)
def analyze_status(job_id: str):
    """잡 상태/결과 조회. 백엔드가 DONE 될 때까지 폴링."""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return AnalyzeStatusResponse(
        jobId=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )


# 기존 동기 엔드포인트 — 하위호환용 유지 (백엔드 전환·검증 후 제거 가능)
@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    return _analyze_sync(request)


# ── 개인화 모델 학습 (분석과 동일한 단일 executor로 GPU 직렬 처리) ─────────────
def _extract_windows_from_videos(video_urls, work_dir):
    """여러 영상 URL → 다운로드 → 포즈 파이프라인 → 결합된 windows (N,32,13) 반환."""
    from ext_main import PosePipeline
    from pre_main import run_preprocessing
    import glob

    video_dir = os.path.join(work_dir, "videos")
    cropped_dir = os.path.join(work_dir, "cropped")
    json_dir = os.path.join(work_dir, "json_2d")
    vis_dir = os.path.join(work_dir, "vis")
    processed_dir = os.path.join(work_dir, "processed")
    kps_dir = os.path.join(work_dir, "kps")
    os.makedirs(video_dir, exist_ok=True)

    for i, url in enumerate(video_urls):
        fname = _filename_from_url(url, default=f"train_{i}.mp4")
        local = os.path.join(video_dir, f"{i}_{fname}")
        _download_url(url, local)

    PosePipeline(
        video_input_dir=video_dir,
        cropped_output_dir=cropped_dir,
        json_output_dir=json_dir,
        vis_output_dir=vis_dir,
    ).run_pipeline()

    run_preprocessing(json_dir=json_dir, output_dir=processed_dir, kps_dir=kps_dir)

    npz_files = sorted(glob.glob(os.path.join(processed_dir, "*_processed.npz")))
    if not npz_files:
        raise RuntimeError("학습 영상에서 windows를 추출하지 못했습니다.")

    windows = [np.load(f)["windows"] for f in npz_files]
    return np.concatenate(windows, axis=0)


def _run_train_job(job_id: str, request: TrainRequest):
    import torch
    import joblib
    from src.user_fine_tune import fine_tune

    _set_job(job_id, status="RUNNING", progress=10)
    try:
        with tempfile.TemporaryDirectory() as work_dir:
            # 1) 학습 영상 → windows (원본 영상은 temp에만 존재, 보관 안 함)
            windows = _extract_windows_from_videos(request.videoUrls, work_dir)
            _set_job(job_id, progress=50, sampleCount=int(windows.shape[0]))

            # 2) base 모델: 증분이면 기존 사용자 모델, 아니면 일반 모델
            if request.baseModelUrl:
                base_model_path = os.path.join(work_dir, "base.pth")
                _download_url(request.baseModelUrl, base_model_path)
            else:
                base_model_path = config.MODEL_DIR

            # 3) fine-tune
            state_dict, stats = fine_tune(windows, base_model_path)
            _set_job(job_id, progress=80)

            # 4) 결과 저장 후 Signed PUT URL로 업로드
            model_path = os.path.join(work_dir, "user_specific_ae.pth")
            stats_path = os.path.join(work_dir, "user_stats.pkl")
            torch.save(state_dict, model_path)
            joblib.dump(stats, stats_path)
            _upload_url(request.modelUploadUrl, model_path)
            _upload_url(request.statsUploadUrl, stats_path)

            accuracy = float(np.clip(100.0 * (1.0 - stats["mean"]), 0.0, 100.0))
            _set_job(job_id, status="DONE", progress=100,
                     accuracy=accuracy, sampleCount=int(windows.shape[0]))
            log.info("Train job %s DONE (userId=%s, samples=%d, acc=%.2f)",
                     job_id, request.userId, windows.shape[0], accuracy)
    except Exception as e:  # noqa: BLE001
        log.error("Train job %s crashed:\n%s", job_id, traceback.format_exc())
        _set_job(job_id, status="FAILED", error=str(e))


@app.post("/api/train/start", response_model=TrainStartResponse)
def train_start(request: TrainRequest):
    """개인화 모델 학습을 백그라운드로 시작하고 jobId를 즉시 반환."""
    job_id = str(uuid.uuid4())
    _set_job(job_id, status="PENDING", progress=0, accuracy=None, sampleCount=None, error=None)
    _executor.submit(_run_train_job, job_id, request)
    log.info("Train job %s queued (userId=%s, videos=%d, incremental=%s)",
             job_id, request.userId, len(request.videoUrls), request.incremental)
    return TrainStartResponse(jobId=job_id, status="PENDING")


@app.get("/api/train/status/{job_id}", response_model=TrainStatusResponse)
def train_status(job_id: str):
    """학습 잡 상태/결과 조회. 백엔드가 DONE 될 때까지 폴링."""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return TrainStatusResponse(
        jobId=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        accuracy=job.get("accuracy"),
        sampleCount=job.get("sampleCount"),
        error=job.get("error"),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
