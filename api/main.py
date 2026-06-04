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
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeStartResponse,
    AnalyzeStatusResponse,
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
        import urllib.request
        urllib.request.urlretrieve(video_url, local_path)
    else:
        from google.cloud import storage
        client = storage.Client()
        client.bucket(GCS_BUCKET).blob(file_id_path).download_to_filename(local_path)


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

    try:
        return generate_report_json(file_id=file_id)
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


@app.get("/health")
def health():
    return {"status": "ok"}
