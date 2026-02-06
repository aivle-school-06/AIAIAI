"""
의존성 주입 (Dependency Injection)
==================================

FastAPI의 Depends()를 활용한 의존성 관리.
싱글톤 패턴으로 리소스 효율적 관리.
"""
from functools import lru_cache
from typing import Optional
import logging

from app.config import Settings, get_settings
from app.core.data_loader import DataLoader
from app.core.predictor import Predictor
from app.core.sentiment_analyzer import SentimentAnalyzer
from app.services.news_service import NewsService
from app.services.dart_service import DartService

logger = logging.getLogger(__name__)


# =============================================================================
# 싱글톤 인스턴스 홀더
# =============================================================================
class Container:
    """의존성 컨테이너 - 싱글톤 인스턴스 관리"""

    _data_loader: Optional[DataLoader] = None
    _predictor: Optional[Predictor] = None
    _sentiment_analyzer: Optional[SentimentAnalyzer] = None
    _news_service: Optional[NewsService] = None
    _dart_service: Optional[DartService] = None
    _initialized: bool = False

    @classmethod
    def initialize(cls, settings: Settings) -> None:
        """
        컨테이너 초기화 (앱 시작 시 호출)

        무거운 리소스(모델, 데이터)를 미리 로드합니다.
        """
        if cls._initialized:
            logger.info("Container already initialized, skipping...")
            return

        logger.info("Initializing dependency container...")

        try:
            # 데이터 로더 초기화
            cls._data_loader = DataLoader(settings)
            logger.info("DataLoader initialized")

            # 예측기 초기화 (모델 로드)
            cls._predictor = Predictor(settings, cls._data_loader)
            logger.info("Predictor initialized")

            # 감성 분석기 초기화 (HuggingFace 모델 로드)
            cls._sentiment_analyzer = SentimentAnalyzer(settings)
            logger.info("SentimentAnalyzer initialized")

            # 뉴스 서비스 초기화
            cls._news_service = NewsService(settings, cls._sentiment_analyzer)
            logger.info("NewsService initialized")

            # DART 서비스 초기화
            cls._dart_service = DartService(settings)
            logger.info("DartService initialized")

            cls._initialized = True
            logger.info("Container initialization complete")

        except Exception as e:
            logger.error(f"Container initialization failed: {e}")
            raise

    @classmethod
    def shutdown(cls) -> None:
        """컨테이너 정리 (앱 종료 시 호출)"""
        logger.info("Shutting down container...")
        cls._data_loader = None
        cls._predictor = None
        cls._sentiment_analyzer = None
        cls._news_service = None
        cls._dart_service = None
        cls._initialized = False

    @classmethod
    def get_data_loader(cls) -> DataLoader:
        """DataLoader 인스턴스 반환"""
        if cls._data_loader is None:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        return cls._data_loader

    @classmethod
    def get_predictor(cls) -> Predictor:
        """Predictor 인스턴스 반환"""
        if cls._predictor is None:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        return cls._predictor

    @classmethod
    def get_sentiment_analyzer(cls) -> SentimentAnalyzer:
        """SentimentAnalyzer 인스턴스 반환"""
        if cls._sentiment_analyzer is None:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        return cls._sentiment_analyzer

    @classmethod
    def get_news_service(cls) -> NewsService:
        """NewsService 인스턴스 반환"""
        if cls._news_service is None:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        return cls._news_service

    @classmethod
    def get_dart_service(cls) -> DartService:
        """DartService 인스턴스 반환"""
        if cls._dart_service is None:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        return cls._dart_service


# =============================================================================
# FastAPI 의존성 함수
# =============================================================================
def get_data_loader() -> DataLoader:
    """
    DataLoader 의존성

    Usage:
        @router.get("/companies")
        def list_companies(loader: DataLoader = Depends(get_data_loader)):
            return loader.get_company_list()
    """
    return Container.get_data_loader()


def get_predictor() -> Predictor:
    """
    Predictor 의존성

    Usage:
        @router.get("/predict/{company_code}")
        def predict(code: str, predictor: Predictor = Depends(get_predictor)):
            return predictor.predict(code)
    """
    return Container.get_predictor()


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """
    SentimentAnalyzer 의존성

    Usage:
        @router.get("/sentiment")
        def analyze(analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)):
            return analyzer.predict(["text"])
    """
    return Container.get_sentiment_analyzer()


def get_news_service() -> NewsService:
    """
    NewsService 의존성

    Usage:
        @router.post("/news/{company_code}")
        def get_news(service: NewsService = Depends(get_news_service)):
            return service.analyze_news("company_name")
    """
    return Container.get_news_service()


def get_dart_service() -> DartService:
    """
    DartService 의존성

    Usage:
        @router.post("/report/{company_code}")
        def get_report(service: DartService = Depends(get_dart_service)):
            return service.analyze_report_async("code", "name")
    """
    return Container.get_dart_service()


# =============================================================================
# 설정 의존성
# =============================================================================
def get_current_settings() -> Settings:
    """현재 설정 반환"""
    return get_settings()
