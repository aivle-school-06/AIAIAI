"""
관리자 API
==========

모델 모니터링 및 관리를 위한 엔드포인트

기능:
1. 모델 성능 현황 조회
2. 분기별 예측 피드백 (실제 vs 예측)
3. 모델 재학습 트리거
4. 캐시 관리
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.config import settings
from app.services.monitoring_service import MonitoringService

router = APIRouter()


# =========================================================================
# 모델 성능 모니터링
# =========================================================================

@router.get("/model/performance")
async def get_model_performance():
    """
    모델 성능 현황 조회

    현재 배포된 모델의 성능 지표를 반환합니다.

    Returns:
        - 13개 지표별 R², MAE, RMSE
        - 전체 평균 성능
        - 마지막 학습 일시
    """
    try:
        service = MonitoringService()
        performance = await service.get_model_performance()

        return {
            "model_version": settings.MODEL_VERSION,
            "metrics": performance.get("metrics", {}),
            "overall": performance.get("overall", {}),
            "last_trained": performance.get("last_trained"),
            "sample_size": performance.get("sample_size")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/feedback/{quarter}")
async def get_quarterly_feedback(quarter: str):
    """
    분기별 모델 피드백

    특정 분기의 예측값과 실제값을 비교하여 피드백을 제공합니다.

    Args:
        quarter: 분기 (예: "2024Q3")

    Returns:
        - 지표별 예측 vs 실제 비교
        - 오차 분석
        - 개선 필요 지표
    """
    try:
        service = MonitoringService()
        feedback = await service.get_quarterly_feedback(quarter)

        return {
            "quarter": quarter,
            "comparison": feedback.get("comparison", []),
            "error_analysis": feedback.get("error_analysis", {}),
            "recommendations": feedback.get("recommendations", [])
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/history")
async def get_model_history():
    """
    모델 학습 이력 조회

    Returns:
        모델 버전별 학습 이력 및 성능 변화
    """
    try:
        service = MonitoringService()
        history = await service.get_model_history()

        return {
            "current_version": settings.MODEL_VERSION,
            "history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# 모델 재학습
# =========================================================================

@router.post("/model/retrain")
async def trigger_model_retrain(
    background_tasks: BackgroundTasks,
    include_latest: bool = True,
    dry_run: bool = False
):
    """
    모델 재학습 트리거

    새로운 데이터로 모델을 재학습합니다.

    Args:
        include_latest: 최신 분기 데이터 포함 여부
        dry_run: 테스트 모드 (실제 학습 안 함)

    Returns:
        작업 ID
    """
    if dry_run:
        return {
            "status": "dry_run",
            "message": "테스트 모드: 실제 학습이 수행되지 않았습니다.",
            "config": {
                "include_latest": include_latest
            }
        }

    job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 백그라운드 작업 등록
    background_tasks.add_task(
        _retrain_model_background,
        job_id,
        include_latest
    )

    return {
        "job_id": job_id,
        "status": "started",
        "message": "모델 재학습이 시작되었습니다."
    }


async def _retrain_model_background(job_id: str, include_latest: bool):
    """백그라운드 재학습 작업"""
    # TODO: 실제 재학습 로직 구현
    pass


@router.get("/model/retrain/status/{job_id}")
async def get_retrain_status(job_id: str):
    """
    재학습 상태 확인

    Args:
        job_id: 작업 ID

    Returns:
        재학습 진행 상태
    """
    # TODO: 실제 상태 조회 구현
    return {
        "job_id": job_id,
        "status": "unknown",
        "message": "상태 조회 기능 구현 예정"
    }


# =========================================================================
# 캐시 관리
# =========================================================================

@router.get("/cache/status")
async def get_cache_status():
    """
    캐시 상태 조회

    Returns:
        - 캐시된 기업 수
        - 캐시 용량
        - 마지막 갱신 시간
    """
    try:
        service = MonitoringService()
        status = await service.get_cache_status()

        return {
            "enabled": settings.CACHE_ENABLED,
            "ttl": settings.CACHE_TTL,
            "cached_companies": status.get("count", 0),
            "cache_size_mb": status.get("size_mb", 0),
            "last_updated": status.get("last_updated"),
            "hit_rate": status.get("hit_rate")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/refresh")
async def refresh_cache(
    background_tasks: BackgroundTasks,
    corp_code: Optional[str] = None
):
    """
    캐시 갱신

    Args:
        corp_code: 특정 기업만 갱신 (없으면 전체)

    Returns:
        갱신 결과
    """
    if corp_code:
        # 특정 기업만 갱신
        try:
            service = MonitoringService()
            await service.refresh_company_cache(corp_code)

            return {
                "status": "completed",
                "corp_code": corp_code,
                "message": f"{corp_code} 캐시가 갱신되었습니다."
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # 전체 갱신 (백그라운드)
        background_tasks.add_task(_refresh_all_cache)

        return {
            "status": "started",
            "message": "전체 캐시 갱신이 시작되었습니다."
        }


async def _refresh_all_cache():
    """백그라운드 전체 캐시 갱신"""
    # TODO: 구현
    pass


@router.delete("/cache/clear")
async def clear_cache(confirm: bool = False):
    """
    캐시 초기화

    Args:
        confirm: 확인 플래그 (True여야 실행)

    Returns:
        초기화 결과
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="캐시 초기화를 확인하려면 confirm=true를 전달하세요."
        )

    try:
        service = MonitoringService()
        await service.clear_cache()

        return {
            "status": "completed",
            "message": "캐시가 초기화되었습니다."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# 시스템 정보
# =========================================================================

@router.get("/system/info")
async def get_system_info():
    """
    시스템 정보 조회

    Returns:
        서버 설정 및 상태 정보
    """
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG,
        "model_version": settings.MODEL_VERSION,
        "paths": {
            "data": str(settings.DATA_RAW_PATH),
            "models": str(settings.MODEL_PATH),
            "reports": str(settings.REPORT_PATH)
        },
        "cache": {
            "enabled": settings.CACHE_ENABLED,
            "ttl": settings.CACHE_TTL
        }
    }
