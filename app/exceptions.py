"""
커스텀 예외 클래스
=================

애플리케이션 전용 예외를 정의합니다.
FastAPI 예외 핸들러에서 일관된 응답으로 변환됩니다.
"""
from typing import Optional, Dict, Any


class AIServerException(Exception):
    """AI Server 기본 예외 클래스"""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """에러 응답 딕셔너리로 변환"""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# 데이터 관련 예외
# =============================================================================
class DataNotFoundError(AIServerException):
    """데이터를 찾을 수 없는 경우"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATA_NOT_FOUND",
            status_code=404,
            details=details,
        )


class CompanyNotFoundError(DataNotFoundError):
    """기업 코드가 존재하지 않는 경우"""

    def __init__(self, company_code: str):
        super().__init__(
            message=f"기업 코드 '{company_code}'를 찾을 수 없습니다.",
            details={"company_code": company_code},
        )


class DataLoadError(AIServerException):
    """데이터 로드 실패"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATA_LOAD_ERROR",
            status_code=500,
            details=details,
        )


# =============================================================================
# 모델 관련 예외
# =============================================================================
class ModelNotFoundError(AIServerException):
    """모델 파일이 없는 경우"""

    def __init__(self, metric: str):
        super().__init__(
            message=f"'{metric}' 지표의 모델 파일을 찾을 수 없습니다.",
            error_code="MODEL_NOT_FOUND",
            status_code=500,
            details={"metric": metric},
        )


class PredictionError(AIServerException):
    """예측 수행 중 오류"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            status_code=500,
            details=details,
        )


class FeatureExtractionError(AIServerException):
    """피처 추출 실패"""

    def __init__(self, company_code: str):
        super().__init__(
            message=f"'{company_code}' 기업의 피처 추출에 실패했습니다.",
            error_code="FEATURE_EXTRACTION_ERROR",
            status_code=500,
            details={"company_code": company_code},
        )


# =============================================================================
# 보고서 관련 예외
# =============================================================================
class ReportGenerationError(AIServerException):
    """보고서 생성 실패"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="REPORT_GENERATION_ERROR",
            status_code=500,
            details=details,
        )


class PDFGenerationError(AIServerException):
    """PDF 생성 실패"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PDF_GENERATION_ERROR",
            status_code=500,
            details=details,
        )


# =============================================================================
# LLM 관련 예외
# =============================================================================
class LLMError(AIServerException):
    """LLM API 호출 실패"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            status_code=503,
            details=details,
        )


# =============================================================================
# 캐시 관련 예외
# =============================================================================
class CacheError(AIServerException):
    """캐시 작업 실패"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=500,
            details=details,
        )


# =============================================================================
# 검증 관련 예외
# =============================================================================
class ValidationError(AIServerException):
    """입력값 검증 실패"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details,
        )
