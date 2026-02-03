"""
인터페이스 정의
==============

Protocol을 사용하여 느슨한 결합 구현.
테스트 시 Mock 객체로 쉽게 대체 가능.
"""
from typing import Protocol, Optional, Dict, List, Any
from abc import abstractmethod
import numpy as np
import pandas as pd


class IDataLoader(Protocol):
    """데이터 로더 인터페이스"""

    @abstractmethod
    def get_company_list(self) -> pd.DataFrame:
        """전체 기업 목록 조회"""
        ...

    @abstractmethod
    def get_company_data(self, company_code: str) -> Optional[Dict[str, Any]]:
        """특정 기업 데이터 조회"""
        ...

    @abstractmethod
    def get_industry_stats(
        self, industry_code: str, year: int, quarter: str
    ) -> Dict[str, Any]:
        """업종별 통계 조회"""
        ...

    @abstractmethod
    def get_features_for_prediction(self, company_code: str) -> Optional[np.ndarray]:
        """예측용 피처 추출"""
        ...

    @abstractmethod
    def calculate_percentile(
        self, company_code: str, metric: str
    ) -> Optional[float]:
        """업종 내 백분위 계산"""
        ...


class IPredictor(Protocol):
    """예측기 인터페이스"""

    @abstractmethod
    def predict(self, company_code: str) -> Dict[str, Any]:
        """기업 예측 수행"""
        ...

    @abstractmethod
    def get_shap_analysis(
        self, company_code: str, metric: str
    ) -> Optional[Dict[str, Any]]:
        """SHAP 분석 결과 반환"""
        ...


class IGradeCalculator(Protocol):
    """등급 계산기 인터페이스"""

    @abstractmethod
    def calculate_grades(self, company_code: str) -> Dict[str, Any]:
        """기업 등급 계산"""
        ...


class IReportGenerator(Protocol):
    """보고서 생성기 인터페이스"""

    @abstractmethod
    def generate(self, company_code: str) -> Dict[str, Any]:
        """보고서 데이터 생성"""
        ...


class IPDFGenerator(Protocol):
    """PDF 생성기 인터페이스"""

    @abstractmethod
    def generate(self, report_data: Dict[str, Any]) -> bytes:
        """PDF 파일 생성 (바이트 반환)"""
        ...


class ILLMOpinion(Protocol):
    """LLM 분석 인터페이스"""

    @abstractmethod
    def generate_opinion(
        self,
        company_name: str,
        metric: str,
        current_value: float,
        predicted_value: float,
        shap_factors: List[Dict],
    ) -> str:
        """LLM 기반 인사이트 생성"""
        ...
