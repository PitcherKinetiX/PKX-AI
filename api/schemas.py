from pydantic import BaseModel
from typing import Optional


class AnalyzeRequest(BaseModel):
    fileId: str
    userId: int
    analysisId: int
    modelType: str = "GENERAL"
    videoUrl: Optional[str] = None  # GCS Signed URL (외부 서버용)
    userModelUrl: Optional[str] = None   # 개인화 모델 pth Signed GET URL
    userStatsUrl: Optional[str] = None   # 개인화 모델 stats(pkl) Signed GET URL


# ── 개인화 모델 학습 ──────────────────────────────────────────
class TrainRequest(BaseModel):
    userId: int
    videoUrls: list[str]                  # 학습 영상 Signed GET URL 목록
    baseModelUrl: Optional[str] = None    # 기존 사용자 모델 GET URL (증분 학습 시), None이면 일반 모델 베이스
    modelUploadUrl: str                   # 새 pth 업로드용 Signed PUT URL
    statsUploadUrl: str                   # 새 stats(pkl) 업로드용 Signed PUT URL
    incremental: bool = False


class TrainStartResponse(BaseModel):
    jobId: str
    status: str                           # 항상 "PENDING"


class TrainStatusResponse(BaseModel):
    jobId: str
    status: str                           # PENDING | RUNNING | DONE | FAILED
    progress: int = 0                     # 0~100
    accuracy: Optional[float] = None
    sampleCount: Optional[int] = None
    error: Optional[str] = None


class FeatureDetail(BaseModel):
    index: int
    name: str
    type: str           # "angle" | "velocity"
    userError: float
    generalError: float
    level: str          # "정상" | "양호" | "주의" | "위험"


class VelocityDetail(BaseModel):
    index: int
    name: str
    peakValue: float
    dangerRatio: float
    medicalScore: int


class Scores(BaseModel):
    userConsistencyScore: float
    generalSimilarityScore: float
    medicalSafetyScore: float
    finalScore: float
    grade: str
    timingScore: float


class CriticalAreas(BaseModel):
    userCriticalWindow: int
    userCriticalFeature: str
    userCriticalTop3: list[str]
    medCriticalFeature: str
    medCriticalWindow: int


class GeneralModel(BaseModel):
    worstFeature: str
    latentShiftNorm: float


class AnalyzeResponse(BaseModel):
    scores: Scores
    features: list[FeatureDetail]           # 13개
    velocityAnalysis: list[VelocityDetail]  # 5개
    criticalAreas: CriticalAreas
    generalModel: GeneralModel


class AnalyzeStartResponse(BaseModel):
    jobId: str
    status: str                             # 항상 "PENDING"


class AnalyzeStatusResponse(BaseModel):
    jobId: str
    status: str                             # PENDING | RUNNING | DONE | FAILED
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None
