import sys
import os
import tempfile

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "src", "visualize"))
sys.path.insert(0, os.path.join(_project_root, "src", "extract"))
sys.path.insert(0, os.path.join(_project_root, "src", "preprocess"))
sys.path.insert(0, _project_root)

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from api.schemas import AnalyzeRequest, AnalyzeResponse
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


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
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
                raise HTTPException(status_code=500, detail=f"전처리 파이프라인 실패: {e}")

    try:
        result = generate_report_json(file_id=file_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"분석 데이터를 찾을 수 없습니다: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {e}")

    return result


@app.get("/health")
def health():
    return {"status": "ok"}
