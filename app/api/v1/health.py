"""
헬스체크 API
============

서버 상태 및 의존성 확인을 위한 엔드포인트

사용처:
- 로드밸런서 헬스체크
- 모니터링 시스템
- 배포 파이프라인
"""
from fastapi import APIRouter, Depends
from datetime import datetime
from typing import Dict, Any

from app.config import settings

router = APIRouter()


@router.get("")
async def health_check() -> Dict[str, Any]:
    """
    기본 헬스체크

    서버가 정상 작동 중인지 확인합니다.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.APP_VERSION
    }


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    상세 헬스체크

    서버 상태 및 의존성 상태를 상세히 확인합니다.
    """
    checks = {
        "server": "healthy",
        "model": "unknown",
        "cache": "unknown",
        "openai": "unknown"
    }

    # 모델 체크
    try:
        model_path = settings.MODEL_PATH
        if model_path.exists() and any(model_path.glob("*.joblib")):
            checks["model"] = "healthy"
        else:
            checks["model"] = "no models found"
    except Exception as e:
        checks["model"] = f"error: {str(e)}"

    # 캐시 체크
    try:
        cache_path = settings.DATA_CACHE_PATH
        if cache_path.exists():
            checks["cache"] = "healthy"
        else:
            checks["cache"] = "path not found"
    except Exception as e:
        checks["cache"] = f"error: {str(e)}"

    # OpenAI API 체크
    if settings.OPENAI_API_KEY:
        checks["openai"] = "configured"
    else:
        checks["openai"] = "not configured"

    # 전체 상태 결정
    overall = "healthy" if all(
        v in ["healthy", "configured"] for v in checks.values()
    ) else "degraded"

    return {
        "status": overall,
        "timestamp": datetime.now().isoformat(),
        "version": settings.APP_VERSION,
        "checks": checks
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    준비 상태 체크 (Kubernetes readiness probe용)

    서버가 트래픽을 받을 준비가 되었는지 확인합니다.
    """
    # TODO: 실제 준비 상태 로직 추가
    # - 모델 로드 완료 여부
    # - DB 연결 상태 등
    return {
        "ready": True,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    생존 상태 체크 (Kubernetes liveness probe용)

    서버 프로세스가 살아있는지 확인합니다.
    """
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat()
    }
