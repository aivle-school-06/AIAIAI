"""
Azure Blob Storage에 Best 모델 업로드
=====================================

튜닝된 Best 모델을 Azure Blob Storage에 업로드합니다.

사용법:
1. 환경변수 설정:
   export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"

2. 실행:
   python scripts/upload_best_models.py

또는 connection string을 직접 입력:
   python scripts/upload_best_models.py --connection-string "your_string"
"""
import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
BEST_MODEL_DIR = Path('/Users/guna_bb/Desktop/BigBig/models/XGBoost/outputs_best')
LOCAL_MODEL_DIR = BASE_DIR / "ml_models"

# 업로드할 모델 파일 목록
MODEL_FILES = [
    "ROA_model.joblib",
    "ROE_model.joblib",
    "매출액영업이익률_model.joblib",
    "부채비율_model.joblib",
    "자기자본비율_model.joblib",
    "자본잠식률_model.joblib",
    "단기차입금비율_model.joblib",
    "유동비율_model.joblib",
    "당좌비율_model.joblib",
    "유동부채비율_model.joblib",
    "CFO_자산비율_model.joblib",
    "CFO_매출액비율_model.joblib",
    "CFO증감률_model.joblib",
]

# Best 모델 정보 (튜닝 결과)
BEST_MODEL_INFO = {
    "ROA_model.joblib": {"type": "Tuned", "r2": 0.4187},
    "ROE_model.joblib": {"type": "Tuned", "r2": 0.3564},
    "매출액영업이익률_model.joblib": {"type": "Tuned", "r2": 0.6664},
    "부채비율_model.joblib": {"type": "Tuned", "r2": 0.8197},
    "자기자본비율_model.joblib": {"type": "Tuned", "r2": 0.9430},
    "자본잠식률_model.joblib": {"type": "Tuned", "r2": 0.9897},
    "단기차입금비율_model.joblib": {"type": "Base", "r2": 0.7897},  # Base가 더 좋음
    "유동비율_model.joblib": {"type": "Tuned", "r2": 0.8720},
    "당좌비율_model.joblib": {"type": "Tuned", "r2": 0.8723},
    "유동부채비율_model.joblib": {"type": "Tuned", "r2": 0.8015},
    "CFO_자산비율_model.joblib": {"type": "Tuned", "r2": 0.2987},
    "CFO_매출액비율_model.joblib": {"type": "Tuned", "r2": 0.5254},
    "CFO증감률_model.joblib": {"type": "Base", "r2": 0.2973},  # Base가 더 좋음
}


def upload_to_azure(connection_string: str, container_name: str = "ai-server-assets"):
    """Azure Blob Storage에 Best 모델 업로드"""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.error("azure-storage-blob 패키지가 필요합니다:")
        logger.error("  pip install azure-storage-blob")
        return False

    try:
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container = blob_service.get_container_client(container_name)

        logger.info("=" * 60)
        logger.info("Azure Blob Storage에 Best 모델 업로드")
        logger.info("=" * 60)
        logger.info(f"컨테이너: {container_name}")
        logger.info(f"소스 경로: {BEST_MODEL_DIR}")
        logger.info("")

        uploaded = 0
        failed = 0

        for filename in MODEL_FILES:
            local_path = BEST_MODEL_DIR / filename
            blob_path = f"models/{filename}"
            info = BEST_MODEL_INFO.get(filename, {})

            if not local_path.exists():
                # Best 폴더에 없으면 로컬 ml_models에서 시도
                local_path = LOCAL_MODEL_DIR / filename

            if not local_path.exists():
                logger.error(f"  ✗ {filename} - 파일 없음")
                failed += 1
                continue

            try:
                blob_client = container.get_blob_client(blob_path)
                with open(local_path, "rb") as f:
                    blob_client.upload_blob(f, overwrite=True)

                size_mb = local_path.stat().st_size / 1024 / 1024
                logger.info(f"  ✓ {filename} ({info.get('type', '?')}, R²={info.get('r2', '?'):.4f}, {size_mb:.1f}MB)")
                uploaded += 1

            except Exception as e:
                logger.error(f"  ✗ {filename} - {e}")
                failed += 1

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"완료: {uploaded}개 업로드, {failed}개 실패")
        logger.info("=" * 60)

        return failed == 0

    except Exception as e:
        logger.error(f"Azure 연결 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Best 모델을 Azure Blob Storage에 업로드")
    parser.add_argument(
        "--connection-string",
        help="Azure Storage 연결 문자열 (또는 AZURE_STORAGE_CONNECTION_STRING 환경변수)",
    )
    parser.add_argument(
        "--container",
        default="ai-server-assets",
        help="Blob 컨테이너 이름 (기본값: ai-server-assets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 업로드 없이 파일 목록만 확인",
    )

    args = parser.parse_args()

    connection_string = args.connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("Best 모델 파일 확인 (Dry Run)")
        logger.info("=" * 60)
        for filename in MODEL_FILES:
            local_path = BEST_MODEL_DIR / filename
            info = BEST_MODEL_INFO.get(filename, {})
            if local_path.exists():
                size_mb = local_path.stat().st_size / 1024 / 1024
                logger.info(f"  ✓ {filename} ({info.get('type')}, R²={info.get('r2'):.4f}, {size_mb:.1f}MB)")
            else:
                logger.warning(f"  ✗ {filename} - 파일 없음")
        return

    if not connection_string:
        logger.error("Azure Storage 연결 문자열이 필요합니다.")
        logger.error("")
        logger.error("방법 1: 환경변수 설정")
        logger.error("  export AZURE_STORAGE_CONNECTION_STRING='your_connection_string'")
        logger.error("")
        logger.error("방법 2: 인자로 전달")
        logger.error("  python upload_best_models.py --connection-string 'your_string'")
        sys.exit(1)

    success = upload_to_azure(connection_string, args.container)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
