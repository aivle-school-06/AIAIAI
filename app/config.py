"""
환경 설정 모듈
=============

애플리케이션 전역 설정을 관리합니다.
Pydantic Settings를 사용하여 환경변수를 자동으로 로드합니다.

수정 시 주의사항:
- 새로운 설정 추가 시 .env.example도 업데이트
- 민감한 정보는 반드시 환경변수로 관리
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path
from functools import lru_cache


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # =========================================================================
    # 서버 설정
    # =========================================================================
    APP_NAME: str = "AI Financial Analysis Server"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, description="디버그 모드")
    HOST: str = Field(default="0.0.0.0", description="서버 호스트")
    PORT: int = Field(default=8000, description="서버 포트")

    # =========================================================================
    # API Keys
    # =========================================================================
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API 키")

    # =========================================================================
    # 캐시 설정
    # =========================================================================
    CACHE_ENABLED: bool = Field(default=True, description="캐시 사용 여부")
    CACHE_TTL: int = Field(default=3600, description="캐시 TTL (초)")

    # =========================================================================
    # ML 모델 설정
    # =========================================================================
    MODEL_VERSION: str = Field(default="v1", description="모델 버전")

    # =========================================================================
    # 백엔드 연동
    # =========================================================================
    BACKEND_URL: Optional[str] = Field(default=None, description="백엔드 서버 URL")
    BACKEND_API_KEY: Optional[str] = Field(default=None, description="백엔드 API 키")

    # =========================================================================
    # 경로 설정 (프로퍼티로 처리)
    # =========================================================================
    @property
    def BASE_DIR(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def DATA_RAW_PATH(self) -> Path:
        return self.BASE_DIR / "data" / "raw"

    @property
    def DATA_PROCESSED_PATH(self) -> Path:
        return self.BASE_DIR / "data" / "processed"

    @property
    def DATA_CACHE_PATH(self) -> Path:
        return self.BASE_DIR / "data" / "cache"

    @property
    def MODEL_PATH(self) -> Path:
        return self.BASE_DIR / "ml_models"

    @property
    def REPORT_PATH(self) -> Path:
        return self.BASE_DIR / "reports"

    @property
    def FONT_PATH(self) -> Path:
        return self.BASE_DIR / "app" / "assets" / "fonts"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    설정 싱글톤 인스턴스 반환

    @lru_cache()를 사용하여 한 번만 생성됩니다.
    """
    return Settings()


# 전역 설정 인스턴스
settings = get_settings()
