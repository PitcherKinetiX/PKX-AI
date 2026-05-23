import sys
import os

# report.py가 cal_user_error 등을 상대 import하므로 경로 추가
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "src", "visualize"))
sys.path.insert(0, _project_root)

from fastapi import FastAPI, HTTPException
from api.schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(
    title="PKX-AI Analysis API",
    description="투구 동작 분석 AI API - 13개 생체역학 특징 전체 반환",
    version="1.0.0",
)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    from report import generate_report_json

    # Strip S3 path prefix (e.g. "videos/4/foo.mp4" → "foo.mp4")
    file_id = os.path.basename(request.fileId)

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
