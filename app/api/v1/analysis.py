"""
분석 API
========

기업 분석, 예측, 보고서 생성 관련 엔드포인트.
의존성 주입(Depends)을 통해 서비스 레이어와 연결됩니다.
"""
import time
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import FileResponse

from app.dependencies import get_data_loader, get_predictor
from app.core.data_loader import DataLoader
from app.core.predictor import Predictor
from app.services.analysis_service import AnalysisService
from app.services.health_score_service import HealthScoreService
from app.services.signal_service import SignalService
from app.models.response import HealthScoreResponse, SignalResponse
from app.core.constants import ALL_TARGETS
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# =============================================================================
# report 모듈 import (PDF 생성용)
# =============================================================================
# PDF 생성 함수 import (동기 버전 - 폴백용)
try:
    from app.report.pdf_generator import generate_pdf_report
    PDF_AVAILABLE = True
    logger.info("PDF generator module loaded successfully")
except ImportError as e:
    PDF_AVAILABLE = False
    logger.warning(f"PDF generator not available: {e}")

# 비동기 PDF 서비스 import
try:
    from app.services.pdf_service import generate_pdf_async
    ASYNC_PDF_AVAILABLE = True
    logger.info("Async PDF service loaded successfully")
except ImportError as e:
    ASYNC_PDF_AVAILABLE = False
    logger.warning(f"Async PDF service not available: {e}")


# =============================================================================
# 의존성 주입
# =============================================================================
def get_analysis_service(
    data_loader: DataLoader = Depends(get_data_loader),
    predictor: Predictor = Depends(get_predictor),
) -> AnalysisService:
    """AnalysisService 의존성 주입"""
    return AnalysisService(data_loader, predictor)


def get_health_score_service(
    data_loader: DataLoader = Depends(get_data_loader),
    predictor: Predictor = Depends(get_predictor),
) -> HealthScoreService:
    """HealthScoreService 의존성 주입"""
    return HealthScoreService(data_loader, predictor)


def get_signal_service(
    data_loader: DataLoader = Depends(get_data_loader),
) -> SignalService:
    """SignalService 의존성 주입"""
    return SignalService(data_loader)


# =============================================================================
# 기업 목록
# =============================================================================
@router.get(
    "/companies",
    summary="기업 목록 조회",
    description="분석 가능한 기업 목록을 조회합니다.",
)
async def list_companies(
    market: Optional[str] = Query(None, description="시장구분 (KOSPI/KOSDAQ)"),
    industry: Optional[str] = Query(None, description="업종명 필터"),
    search: Optional[str] = Query(None, description="검색어 (기업명/코드)"),
    limit: int = Query(100, ge=1, le=1000, description="페이지 크기"),
    offset: int = Query(0, ge=0, description="시작 위치"),
    service: AnalysisService = Depends(get_analysis_service),
):
    """기업 목록 조회 (필터, 검색, 페이지네이션 지원)"""
    return service.get_company_list(
        market=market,
        industry=industry,
        search=search,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/industries",
    summary="업종 목록 조회",
    description="전체 업종 목록을 조회합니다.",
)
async def list_industries(
    service: AnalysisService = Depends(get_analysis_service),
):
    """업종 목록"""
    return {"industries": service.get_industry_list()}


# =============================================================================
# 기업 분석
# =============================================================================
@router.get(
    "/{company_code}",
    summary="기업 종합 분석",
    description="특정 기업의 종합 분석 결과를 반환합니다.",
)
async def get_company_analysis(
    company_code: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    기업 종합 분석

    - 현재 재무 지표
    - 예측 결과
    - 리스크 시그널
    - 업종 비교
    """
    # 기업 코드 정규화 (괄호 포함 형식으로)
    if not company_code.startswith('['):
        company_code = f'[{company_code}]'

    return service.get_company_analysis(company_code)


@router.get(
    "/{company_code}/predict",
    summary="예측 결과 조회",
    description="13개 재무지표의 예측 결과를 반환합니다.",
)
async def get_prediction(
    company_code: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    """예측 결과만 반환"""
    if not company_code.startswith('['):
        company_code = f'[{company_code}]'

    return service.get_prediction(company_code)


@router.get(
    "/{company_code}/predict/{metric}",
    summary="특정 지표 예측",
    description="특정 지표의 예측 결과를 반환합니다.",
)
async def get_metric_prediction(
    company_code: str,
    metric: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    """특정 지표 예측 결과"""
    if metric not in ALL_TARGETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric. Available: {ALL_TARGETS}"
        )

    if not company_code.startswith('['):
        company_code = f'[{company_code}]'

    return service.get_metric_prediction(company_code, metric)


@router.get(
    "/{company_code}/shap/{metric}",
    summary="SHAP 분석",
    description="특정 지표의 SHAP 분석 결과를 반환합니다.",
)
async def get_shap_analysis(
    company_code: str,
    metric: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    """SHAP 분석 결과"""
    if metric not in ALL_TARGETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric. Available: {ALL_TARGETS}"
        )

    if not company_code.startswith('['):
        company_code = f'[{company_code}]'

    return service.get_shap_analysis(company_code, metric)


@router.get(
    "/{company_code}/health-score",
    summary="재무건전성 점수 조회",
    description="기업의 재무건전성 점수를 분기별로 반환합니다.",
    response_model=HealthScoreResponse,
)
async def get_health_score(
    company_code: str,
    service: HealthScoreService = Depends(get_health_score_service),
):
    """
    재무건전성 점수 조회

    - 5개 핵심 지표 기반 0~100점 산출
    - 과거 4분기 + 예측 1분기 제공
    - 라벨: 위험(<40), 주의(40~70), 안정(≥70)
    """
    if not company_code.startswith('['):
        company_code = f'[{company_code}]'

    return service.get_health_scores(company_code)


@router.get(
    "/{company_code}/signals/{period}",
    summary="업종상대 신호등 조회",
    description="기업의 12개 재무지표를 업종 평균 대비 신호등으로 반환합니다.",
    response_model=SignalResponse,
)
async def get_signals(
    company_code: str,
    period: str,
    service: SignalService = Depends(get_signal_service),
):
    """
    업종상대 신호등 조회

    - 12개 지표에 대해 업종상대 z-score 기반 신호등 제공
    - Green (상위 30%), Yellow (중위 30%), Red (하위 40%), Grey (결측)
    - period 형식: 20253 (년도 + 분기)
    """
    if not company_code.startswith('['):
        company_code = f'[{company_code}]'

    return service.get_signals(company_code, period)


@router.get(
    "/{company_code}/historical",
    summary="과거 데이터 조회",
    description="기업의 과거 분기 데이터를 반환합니다.",
)
async def get_historical_data(
    company_code: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    """과거 분기 데이터"""
    if not company_code.startswith('['):
        company_code = f'[{company_code}]'

    return service.get_historical_data(company_code)


# =============================================================================
# 보고서 생성 (PDF)
# =============================================================================
@router.get(
    "/{company_code}/report",
    summary="보고서 생성",
    description="PDF 또는 JSON 형식의 분석 보고서를 생성합니다.",
    responses={
        200: {
            "content": {
                "application/pdf": {},
                "application/json": {},
            },
            "description": "분석 보고서"
        }
    }
)
async def generate_report(
    company_code: str,
    format: str = Query("pdf", description="보고서 형식 (pdf/json)"),
    async_mode: bool = Query(True, description="비동기 모드 (LLM 병렬 처리)"),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    보고서 생성

    - format=pdf: PDF 파일 다운로드
    - format=json: JSON 데이터 반환
    - async_mode=true: LLM 호출을 병렬 처리하여 속도 향상 (기본값)
    - async_mode=false: 기존 순차 처리 방식
    """
    # 기업 코드 정규화
    if not company_code.startswith('['):
        company_code = f'[{company_code}]'

    # JSON 형식 요청
    if format == "json":
        return service.get_company_analysis(company_code)

    # PDF 형식 요청
    if not PDF_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PDF 생성 모듈을 사용할 수 없습니다."
        )

    try:
        start_time = time.time()
        output_dir = str(settings.REPORT_PATH)

        # 비동기 모드 (LLM 병렬 처리)
        if async_mode and ASYNC_PDF_AVAILABLE:
            logger.info(f"Generating PDF report (ASYNC) for {company_code}")
            pdf_path = await generate_pdf_async(company_code, output_dir)
        else:
            # 기존 동기 모드
            logger.info(f"Generating PDF report (SYNC) for {company_code}")
            pdf_path = generate_pdf_report(company_code, output_dir)

        elapsed = time.time() - start_time
        logger.info(f"PDF generation completed in {elapsed:.1f}s (async={async_mode})")

        if not pdf_path or not Path(pdf_path).exists():
            raise HTTPException(
                status_code=500,
                detail="PDF 생성에 실패했습니다."
            )

        # 파일명 추출 (한글 파일명 처리)
        filename = Path(pdf_path).name
        logger.info(f"PDF generated: {pdf_path}")

        # 파일명을 URL 인코딩 (한글 지원)
        from urllib.parse import quote
        encoded_filename = quote(filename)

        # PDF 파일 응답
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=filename,
            headers={
                # RFC 5987 형식으로 한글 파일명 지원
                "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
                # 생성 시간 정보 추가
                "X-Generation-Time": f"{elapsed:.1f}s",
                "X-Generation-Mode": "async" if (async_mode and ASYNC_PDF_AVAILABLE) else "sync",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"PDF generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF 생성 중 오류 발생: {str(e)}"
        )


# =============================================================================
# 일괄 분석
# =============================================================================
@router.post(
    "/batch",
    summary="일괄 예측",
    description="여러 기업의 예측을 일괄 수행합니다.",
)
async def batch_predict(
    company_codes: List[str],
    service: AnalysisService = Depends(get_analysis_service),
):
    """일괄 예측"""
    if len(company_codes) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 companies per batch"
        )

    # 기업 코드 정규화
    normalized_codes = [
        f'[{code}]' if not code.startswith('[') else code
        for code in company_codes
    ]

    return service.batch_predict(normalized_codes)
