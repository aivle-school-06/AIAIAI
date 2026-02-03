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

logger = logging.getLogger(__name__)


# =============================================================================
# 싱글톤 인스턴스 홀더
# =============================================================================
class Container:
    """의존성 컨테이너 - 싱글톤 인스턴스 관리"""

    _data_loader: Optional[DataLoader] = None
    _predictor: Optional[Predictor] = None
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


# =============================================================================
# 설정 의존성
# =============================================================================
def get_current_settings() -> Settings:
    """현재 설정 반환"""
    return get_settings()
