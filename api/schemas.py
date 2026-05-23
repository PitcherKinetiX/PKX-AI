from pydantic import BaseModel
from typing import Optional


class AnalyzeRequest(BaseModel):
    fileId: str
    userId: int
    analysisId: int
    modelType: str = "GENERAL"
    videoUrl: Optional[str] = None  # GCS Signed URL (외부 서버용)


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
