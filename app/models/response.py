"""
응답 스키마
==========

API 응답에 사용되는 Pydantic 모델 정의
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# =========================================================================
# 데이터 파이프라인 응답
# =========================================================================

class DataUploadResponse(BaseModel):
    """데이터 업로드 응답"""
    success: bool
    file_id: str
    filename: str
    quarter: Optional[str]
    message: str
    uploaded_at: datetime


class ProcessStatusResponse(BaseModel):
    """처리 상태 응답"""
    job_id: str
    status: str = Field(..., description="pending, processing, completed, failed")
    progress: int = Field(..., ge=0, le=100, description="진행률 (%)")
    started_at: str
    completed_at: Optional[str]
    error: Optional[str]


class MetricsResponse(BaseModel):
    """지표 응답"""
    corp_code: str
    quarter: str
    metrics: Dict[str, Any] = Field(..., description="13개 지표 값")
    calculated_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "corp_code": "005930",
                "quarter": "2024Q3",
                "metrics": {
                    "ROA": 2.5,
                    "ROE": 3.8,
                    "매출액영업이익률": 12.3,
                    "부채비율": 45.2,
                    "자기자본비율": 68.9,
                    "자본잠식률": -15.0,
                    "단기차입금비율": 8.5,
                    "유동비율": 180.5,
                    "당좌비율": 145.2,
                    "유동부채비율": 32.1,
                    "CFO_자산비율": 8.9,
                    "CFO_매출액비율": 15.2,
                    "CFO증감률": 25.3
                },
                "calculated_at": "2024-03-15T10:30:00"
            }
        }


# =========================================================================
# 분석 응답
# =========================================================================

class PredictionItem(BaseModel):
    """개별 지표 예측"""
    metric: str
    current: float
    predicted: float
    change: float
    direction: str = Field(..., description="improving, declining, stable")
    confidence: str = Field(..., description="high, medium, low")


class AnalysisResponse(BaseModel):
    """분석 응답"""
    corp_code: str
    company_name: Optional[str]
    analysis_date: datetime
    summary: Optional[Dict[str, Any]]
    predictions: Optional[Dict[str, Any]]
    shap_analysis: Optional[Dict[str, Any]]
    llm_insights: Optional[Dict[str, Any]]
    cached: bool = False


class PredictionResponse(BaseModel):
    """예측값 응답"""
    corp_code: str
    predictions: Dict[str, PredictionItem]
    predicted_at: datetime
    model_version: str


class ReportResponse(BaseModel):
    """보고서 상태 응답"""
    corp_code: str
    exists: bool
    report_path: Optional[str]
    generated_at: Optional[datetime]
    file_size: Optional[int] = Field(None, description="파일 크기 (bytes)")


# =========================================================================
# 관리자 응답
# =========================================================================

class ModelPerformanceResponse(BaseModel):
    """모델 성능 응답"""
    model_version: str
    metrics: Dict[str, Dict[str, float]] = Field(
        ...,
        description="지표별 성능 {지표: {r2, mae, rmse}}"
    )
    overall: Dict[str, float] = Field(
        ...,
        description="전체 평균 성능"
    )
    last_trained: Optional[datetime]
    sample_size: int


class FeedbackItem(BaseModel):
    """피드백 항목"""
    metric: str
    predicted: float
    actual: float
    error: float
    error_rate: float


class QuarterlyFeedbackResponse(BaseModel):
    """분기별 피드백 응답"""
    quarter: str
    comparison: List[FeedbackItem]
    error_analysis: Dict[str, Any]
    recommendations: List[str]


class CacheStatusResponse(BaseModel):
    """캐시 상태 응답"""
    enabled: bool
    ttl: int
    cached_companies: int
    cache_size_mb: float
    last_updated: Optional[datetime]
    hit_rate: Optional[float]


# =========================================================================
# 공통 응답
# =========================================================================

class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str
    detail: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.now)


class SuccessResponse(BaseModel):
    """성공 응답"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None


# =========================================================================
# 재무건전성 점수 응답
# =========================================================================

class HealthQuarterItem(BaseModel):
    """분기별 건전성 점수"""
    period: str = Field(..., description="분기 (예: 20253)")
    score: Optional[float] = Field(None, description="건전성 점수 (0~100)")
    label: Optional[str] = Field(None, description="라벨 (위험/주의/안정)")
    type: str = Field(..., description="데이터 타입 (actual/predicted)")


class HealthScoreResponse(BaseModel):
    """재무건전성 점수 응답"""
    company_code: str = Field(..., description="기업코드")
    company_name: str = Field(..., description="기업명")
    quarters: List[HealthQuarterItem] = Field(
        ..., description="분기별 건전성 점수 (과거 4분기 + 예측 1분기)"
    )
    current_score: Optional[float] = Field(None, description="현재 분기 점수")
    predicted_score: Optional[float] = Field(None, description="예측 분기 점수")

    class Config:
        json_schema_extra = {
            "example": {
                "company_code": "005930",
                "company_name": "삼성전자",
                "quarters": [
                    {"period": "20244", "score": 85.0, "label": "안정", "type": "actual"},
                    {"period": "20251", "score": 88.0, "label": "안정", "type": "actual"},
                    {"period": "20252", "score": 82.0, "label": "안정", "type": "actual"},
                    {"period": "20253", "score": 90.0, "label": "안정", "type": "actual"},
                    {"period": "20254", "score": 87.0, "label": "안정", "type": "predicted"},
                ],
                "current_score": 90.0,
                "predicted_score": 87.0,
            }
        }


# =========================================================================
# 업종상대 신호등 응답
# =========================================================================

class SignalResponse(BaseModel):
    """업종상대 신호등 응답"""
    company_code: str = Field(..., description="기업코드")
    company_name: str = Field(..., description="기업명")
    industry: str = Field(..., description="업종명")
    period: str = Field(..., description="분기 (예: 20253)")
    signals: Dict[str, str] = Field(..., description="지표별 신호등 (green/yellow/red/grey)")

    class Config:
        json_schema_extra = {
            "example": {
                "company_code": "005930",
                "company_name": "삼성전자",
                "industry": "통신 및 방송장비 제조업",
                "period": "20253",
                "signals": {
                    "ROA": "green",
                    "ROE": "green",
                    "OpMargin": "grey",
                    "DbRatio": "yellow",
                    "EqRatio": "green",
                    "CapImpRatio": "green",
                    "STDebtRatio": "green",
                    "CurRatio": "yellow",
                    "QkRatio": "green",
                    "CurLibRatio": "yellow",
                    "CFO_AsRatio": "green",
                    "CFO_Sale": "grey",
                },
            }
        }


# =========================================================================
# AI 코멘트 응답
# =========================================================================

class AICommentResponse(BaseModel):
    """AI 코멘트 응답"""
    company_code: str = Field(..., description="기업코드")
    company_name: str = Field(..., description="기업명")
    industry: str = Field(..., description="업종명")
    period: str = Field(..., description="분기 (예: 20253)")
    ai_comment: str = Field(..., description="AI 종합 코멘트 (약 500자)")

    class Config:
        json_schema_extra = {
            "example": {
                "company_code": "005930",
                "company_name": "삼성전자",
                "industry": "통신 및 방송장비 제조업",
                "period": "20253",
                "ai_comment": "삼성전자는 2025년 3분기 기준 재무건전성 점수 90점으로 '안정' 등급을 유지하고 있습니다...",
            }
        }
