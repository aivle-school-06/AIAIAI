# =============================================================================
# AI Financial Analysis Server - Dockerfile
# =============================================================================
# Multi-stage build for optimized image size

# -----------------------------------------------------------------------------
# Stage 1: Build
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# 런타임 의존성만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# builder에서 설치된 패키지 복사
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 애플리케이션 코드 복사 (데이터/모델은 Azure Blob에서 다운로드)
COPY ./app ./app
COPY ./scripts ./scripts

# 디렉토리 생성 (데이터/모델은 런타임에 다운로드됨)
RUN mkdir -p /app/data/raw /app/data/processed /app/data/cache \
    /app/ml_models /app/reports

# 스크립트 실행 권한
RUN chmod +x /app/scripts/entrypoint.sh

# 환경변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# XGBoost/sklearn cgroup 이슈 해결
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# 포트 노출
EXPOSE 8000

# 헬스체크 - curl 사용 (Python import 시 cgroup 이슈 회피)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# entrypoint 스크립트로 실행
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
