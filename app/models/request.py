"""
요청 스키마
==========

API 요청에 사용되는 Pydantic 모델 정의
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DataProcessRequest(BaseModel):
    """데이터 처리 요청"""
    file_id: str = Field(..., description="업로드된 파일 ID")
    quarter: Optional[str] = Field(None, description="분기 (예: 2024Q3)")
    options: Optional[dict] = Field(default_factory=dict, description="처리 옵션")

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "abc123-def456",
                "quarter": "2024Q3",
                "options": {}
            }
        }


class AnalysisRequest(BaseModel):
    """분석 요청"""
    corp_code: str = Field(..., description="기업 코드")
    include_shap: bool = Field(True, description="SHAP 분석 포함")
    include_llm: bool = Field(True, description="LLM 분석 포함")


class BatchAnalysisRequest(BaseModel):
    """일괄 분석 요청"""
    corp_codes: List[str] = Field(..., description="기업 코드 목록")
    include_predictions: bool = Field(True, description="예측값 포함")
    include_shap: bool = Field(False, description="SHAP 분석 포함")


class RetrainRequest(BaseModel):
    """모델 재학습 요청"""
    include_latest: bool = Field(True, description="최신 데이터 포함")
    metrics: Optional[List[str]] = Field(None, description="특정 지표만 재학습")
    dry_run: bool = Field(False, description="테스트 모드")


class CacheRefreshRequest(BaseModel):
    """캐시 갱신 요청"""
    corp_codes: Optional[List[str]] = Field(None, description="특정 기업만 (없으면 전체)")
    force: bool = Field(False, description="강제 갱신")
