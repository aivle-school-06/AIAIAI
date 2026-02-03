"""
Core 모듈
=========

ML 모델, 데이터 로딩, 예측 등 핵심 기능을 담당하는 모듈.
기존 backend/report 모듈의 기능을 서버용으로 재구성했습니다.

구성 요소:
- constants: 상수 및 설정값 (지표 목록, 등급 기준 등)
- interfaces: 인터페이스 정의 (Protocol)
- data_loader: 데이터 로딩 및 기업 정보 조회
- predictor: XGBoost 모델 예측 및 SHAP 분석
- llm_async: 비동기 LLM 의견 생성 (OpenAI API 병렬 처리)
"""
from app.core.constants import (
    ALL_TARGETS,
    TARGET_METRICS,
    METRIC_DIRECTION,
    MODEL_R2,
    GRADE_THRESHOLDS,
)
from app.core.data_loader import DataLoader
from app.core.predictor import Predictor

__all__ = [
    'ALL_TARGETS',
    'TARGET_METRICS',
    'METRIC_DIRECTION',
    'MODEL_R2',
    'GRADE_THRESHOLDS',
    'DataLoader',
    'Predictor',
]
