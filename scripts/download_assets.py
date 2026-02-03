"""
Azure Blob Storage에서 데이터/모델 다운로드
==========================================

컨테이너 시작 시 실행되어 필요한 파일들을 다운로드합니다.

환경변수:
- AZURE_STORAGE_CONNECTION_STRING: Azure Storage 연결 문자열
- AZURE_STORAGE_CONTAINER: 컨테이너 이름 (기본값: ai-server-assets)
"""
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 기본 경로
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "ml_models"

# 다운로드할 파일 목록
ASSETS = {
    "data": [
        "03_dataset_final.csv",
        "03_feature_cols.json",
    ],
    "models": [
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
    ],
}


def download_from_azure():
    """Azure Blob Storage에서 다운로드"""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.error("azure-storage-blob 패키지가 필요합니다: pip install azure-storage-blob")
        return False

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER", "ai-server-assets")

    if not connection_string:
        logger.error("AZURE_STORAGE_CONNECTION_STRING 환경변수가 설정되지 않았습니다.")
        return False

    try:
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container = blob_service.get_container_client(container_name)

        # 데이터 파일 다운로드
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for filename in ASSETS["data"]:
            blob_path = f"data/{filename}"
            local_path = DATA_DIR / filename

            if local_path.exists():
                logger.info(f"이미 존재: {local_path}")
                continue

            logger.info(f"다운로드 중: {blob_path} -> {local_path}")
            blob_client = container.get_blob_client(blob_path)
            with open(local_path, "wb") as f:
                f.write(blob_client.download_blob().readall())

        # 모델 파일 다운로드
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for filename in ASSETS["models"]:
            blob_path = f"models/{filename}"
            local_path = MODEL_DIR / filename

            if local_path.exists():
                logger.info(f"이미 존재: {local_path}")
                continue

            logger.info(f"다운로드 중: {blob_path} -> {local_path}")
            blob_client = container.get_blob_client(blob_path)
            with open(local_path, "wb") as f:
                f.write(blob_client.download_blob().readall())

        logger.info("모든 파일 다운로드 완료!")
        return True

    except Exception as e:
        logger.error(f"다운로드 실패: {e}")
        return False


def check_local_files():
    """로컬 파일 존재 여부 확인"""
    missing = []

    for filename in ASSETS["data"]:
        if not (DATA_DIR / filename).exists():
            missing.append(f"data/processed/{filename}")

    for filename in ASSETS["models"]:
        if not (MODEL_DIR / filename).exists():
            missing.append(f"ml_models/{filename}")

    return missing


if __name__ == "__main__":
    missing = check_local_files()

    if not missing:
        logger.info("모든 필요 파일이 존재합니다.")
    else:
        logger.info(f"누락된 파일: {len(missing)}개")
        for f in missing:
            logger.info(f"  - {f}")

        logger.info("Azure Blob Storage에서 다운로드 시도...")
        download_from_azure()
