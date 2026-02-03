#!/bin/bash
# =============================================================================
# 컨테이너 시작 스크립트
# =============================================================================

set -e

echo "=============================================="
echo "AI Financial Analysis Server - Starting..."
echo "=============================================="

# Azure Blob에서 데이터/모델 다운로드 (환경변수가 설정된 경우)
if [ -n "$AZURE_STORAGE_CONNECTION_STRING" ]; then
    echo "Downloading assets from Azure Blob Storage..."
    python /app/scripts/download_assets.py
else
    echo "AZURE_STORAGE_CONNECTION_STRING not set, using local files"
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
