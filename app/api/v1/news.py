"""
뉴스 분석 API
=============

기업 뉴스 검색 및 감성 분석 엔드포인트.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_news_service
from app.services.news_service import NewsService

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# 내부 설정 (하드코딩)
# =============================================================================
NEWS_CONFIG = {
    "max_results": 50,
    "similarity_threshold": 0.6,
    "enable_gpt_filter": True,
    "enable_summary": True,
}


# =============================================================================
# Request/Response 모델
# =============================================================================
class NewsAnalysisRequest(BaseModel):
    """뉴스 분석 요청"""
    company_name: str = Field(..., description="기업명 (검색 키워드)")


class NewsItem(BaseModel):
    """뉴스 아이템"""
    title: str = Field(..., description="뉴스 제목")
    summary: str = Field(..., description="본문 요약")
    score: float = Field(..., description="긍정 점수 (0~1)")
    date: str = Field(..., description="발행일")
    link: str = Field(default="", description="원문 링크")
    sentiment: str = Field(..., description="감성 라벨 (NEG/NEU/POS)")


class NewsAnalysisResponse(BaseModel):
    """뉴스 분석 응답"""
    company_name: str = Field(..., description="기업명")
    total_count: int = Field(..., description="분석된 뉴스 수")
    news: List[NewsItem] = Field(..., description="뉴스 리스트")
    average_score: Optional[float] = Field(None, description="평균 긍정 점수")
    analyzed_at: str = Field(..., description="분석 시간")


# =============================================================================
# 엔드포인트
# =============================================================================
@router.post(
    "/{company_code}",
    summary="기업 뉴스 분석",
    description="기업 관련 뉴스를 검색하고 감성 분석을 수행합니다.",
    response_model=NewsAnalysisResponse,
)
async def analyze_company_news(
    company_code: str,
    request: NewsAnalysisRequest,
    service: NewsService = Depends(get_news_service),
):
    """
    기업 뉴스 분석

    - company_code: 기업 코드 (path parameter)
    - company_name: 기업명 (request body) - 뉴스 검색에 사용

    파이프라인:
    1. 네이버 뉴스 API로 기업명 검색
    2. 제목 유사도 기반 중복 제거
    3. GPT로 동일 이벤트 기사 필터링 (선택)
    4. 본문 추출 및 요약 (선택)
    5. KR-FinBert-SC 모델로 감성 분석

    Returns:
        - news: [{title, summary, score, date}, ...]
        - average_score: 전체 뉴스의 평균 긍정 점수
    """
    logger.info(f"News analysis request - code: {company_code}, name: {request.company_name}")

    try:
        # 비동기 버전 사용 (본문 요약 병렬 처리)
        result = await service.analyze_news_async(
            company_name=request.company_name,
            max_results=NEWS_CONFIG["max_results"],
            similarity_threshold=NEWS_CONFIG["similarity_threshold"],
            enable_gpt_filter=NEWS_CONFIG["enable_gpt_filter"],
            enable_summary=NEWS_CONFIG["enable_summary"],
        )
        return result

    except Exception as e:
        logger.exception(f"News analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"뉴스 분석 중 오류 발생: {str(e)}"
        )


@router.post(
    "/{company_code}/quick",
    summary="기업 뉴스 빠른 분석",
    description="요약 없이 빠르게 뉴스 감성 분석만 수행합니다.",
    response_model=NewsAnalysisResponse,
)
async def analyze_company_news_quick(
    company_code: str,
    request: NewsAnalysisRequest,
    service: NewsService = Depends(get_news_service),
):
    """
    기업 뉴스 빠른 분석 (요약 없음)

    요약 생성을 건너뛰어 빠르게 감성 점수만 반환합니다.
    """
    logger.info(f"Quick news analysis - code: {company_code}, name: {request.company_name}")

    try:
        # 요약 비활성화 시 동기 버전 사용 (더 빠름)
        result = service.analyze_news(
            company_name=request.company_name,
            max_results=NEWS_CONFIG["max_results"],
            similarity_threshold=NEWS_CONFIG["similarity_threshold"],
            enable_gpt_filter=NEWS_CONFIG["enable_gpt_filter"],
            enable_summary=False,
        )
        return result

    except Exception as e:
        logger.exception(f"Quick news analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"뉴스 분석 중 오류 발생: {str(e)}"
        )
