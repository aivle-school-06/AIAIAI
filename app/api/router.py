"""
API 라우터 통합 모듈
====================

모든 API 라우터를 통합하여 관리합니다.

새로운 라우터 추가 시:
1. api/v1/ 에 새 라우터 파일 생성
2. 이 파일에서 import 및 include_router 추가
"""
from fastapi import APIRouter

from app.api.v1 import health, data, analysis, admin, news

# 메인 API 라우터
api_router = APIRouter()

# =========================================================================
# v1 라우터 등록
# =========================================================================

# 헬스체크
api_router.include_router(
    health.router,
    prefix="/v1/health",
    tags=["Health Check"]
)

# 데이터 파이프라인
api_router.include_router(
    data.router,
    prefix="/v1/data",
    tags=["Data Pipeline"]
)

# 분석 요청
api_router.include_router(
    analysis.router,
    prefix="/v1/analysis",
    tags=["Analysis"]
)

# 관리자 기능
api_router.include_router(
    admin.router,
    prefix="/v1/admin",
    tags=["Admin"]
)

# 뉴스 & 보고서 분석
api_router.include_router(
    news.router,
    prefix="/v1/news",
    tags=["News & Report"]
)
