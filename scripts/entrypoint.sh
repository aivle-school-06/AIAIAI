#!/bin/bash
# =============================================================================
# 컨테이너 시작 스크립트
# =============================================================================

set -e

# XGBoost/sklearn cgroup 이슈 해결 (반드시 Python 실행 전에 설정)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=============================================="
echo "AI Financial Analysis Server - Starting..."
echo "=============================================="

# 클라우드 스토리지에서 데이터/모델 다운로드
if [ -n "$AWS_STORAGE_BUCKET" ]; then
    echo "Downloading assets from AWS S3..."
    python /app/scripts/download_assets.py
elif [ -n "$AZURE_STORAGE_CONNECTION_STRING" ]; then
    echo "Downloading assets from Azure Blob Storage..."
    python /app/scripts/download_assets.py
else
    echo "No cloud storage configured, using local files"
fi

# 필수 파일 확인
if [ ! -f "/app/data/processed/03_dataset_final.csv" ]; then
    echo "ERROR: Data file not found!"
    exit 1
fi

if [ ! -f "/app/ml_models/ROA_model.joblib" ]; then
    echo "ERROR: Model files not found!"
    exit 1
fi

echo "All required files present. Starting server..."

# 서버 실행
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
