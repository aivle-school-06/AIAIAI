"""
AI Financial Analysis Server
=============================

KOSPI/KOSDAQ 상장기업 재무지표 예측 및 분석 보고서 생성 API 서버

주요 기능:
1. 분기 데이터 처리 및 지표 계산
2. XGBoost 모델 기반 재무지표 예측
3. SHAP 분석을 통한 예측 근거 설명
4. PDF 보고서 생성
5. 모델 성능 모니터링

실행 방법:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from app.config import settings
from app.api.router import api_router
from app.dependencies import Container
from app.exceptions import AIServerException

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리

    시작 시:
    - 설정 로드
    - ML 모델 로드
    - 데이터 로드
    - 캐시 초기화

    종료 시:
    - 리소스 정리
    """
    # =========================================================================
    # Startup
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # 경로 생성
    settings.DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
    settings.DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    settings.DATA_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    settings.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    settings.REPORT_PATH.mkdir(parents=True, exist_ok=True)

    logger.info(f"Data path: {settings.DATA_RAW_PATH}")
    logger.info(f"Model path: {settings.MODEL_PATH}")
    logger.info(f"Report path: {settings.REPORT_PATH}")

    # 의존성 컨테이너 초기화 (모델 & 데이터 로드)
    try:
        start_time = time.time()
        Container.initialize(settings)
        elapsed = time.time() - start_time
        logger.info(f"Container initialized in {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Failed to initialize container: {e}")
        raise

    logger.info("Server started successfully!")

    yield

    # =========================================================================
    # Shutdown
    # =========================================================================
    logger.info("Shutting down server...")
    Container.shutdown()
    logger.info("Server shutdown complete")


# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## AI 재무 분석 서버

KOSPI/KOSDAQ 상장기업의 재무지표를 예측하고 분석 보고서를 생성합니다.

### 주요 기능

- **데이터 파이프라인**: 분기 데이터 처리 및 지표 계산
- **예측 분석**: XGBoost 모델 기반 13개 재무지표 예측
- **보고서 생성**: PDF 형태의 상세 분석 보고서
- **모델 모니터링**: 예측 성능 피드백 및 관리

### API 버전

현재 API 버전: **v1**
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# =============================================================================
# 전역 예외 핸들러
# =============================================================================
@app.exception_handler(AIServerException)
async def ai_server_exception_handler(request: Request, exc: AIServerException):
    """AI Server 커스텀 예외 핸들러"""
    logger.error(f"AIServerException: {exc.message} [{exc.error_code}]")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": "INTERNAL_ERROR",
            "message": "서버 내부 오류가 발생했습니다.",
            "details": {"exception": str(exc)} if settings.DEBUG else {},
        },
    )


# =============================================================================
# 미들웨어
# =============================================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    start_time = time.time()

    response = await call_next(request)

    elapsed = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} ({elapsed:.3f}s)"
    )

    return response


# =============================================================================
# CORS 설정
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 라우터 등록
# =============================================================================
app.include_router(api_router, prefix="/api")


# =============================================================================
# 루트 엔드포인트
# =============================================================================
@app.get("/", tags=["Root"])
async def root():
    """서버 정보 반환"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


# =============================================================================
# 직접 실행 시
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
