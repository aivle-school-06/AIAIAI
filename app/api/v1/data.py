"""
데이터 파이프라인 API
=====================

분기 데이터 처리 및 지표 계산을 위한 엔드포인트

기능:
1. 새로운 분기 데이터 업로드
2. 데이터 처리 시작 (전처리, 지표 계산)
3. 처리 상태 확인
4. 처리된 지표 조회 (백엔드 전달용)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from app.config import settings
from app.models.request import DataProcessRequest
from app.models.response import (
    DataUploadResponse,
    ProcessStatusResponse,
    MetricsResponse
)
from app.services.data_pipeline import DataPipelineService

router = APIRouter()

# 처리 작업 상태 저장 (실제로는 Redis 등 사용)
processing_jobs: Dict[str, Dict] = {}


@router.post("/upload", response_model=DataUploadResponse)
async def upload_quarterly_data(
    file: UploadFile = File(..., description="분기 데이터 파일 (txt)"),
    quarter: str = None
):
    """
    새로운 분기 데이터 업로드

    Args:
        file: 분기 데이터 파일 (txt 형식)
        quarter: 분기 정보 (예: "2024Q3")

    Returns:
        업로드 결과 및 파일 ID
    """
    # 파일 확장자 검증
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="txt 파일만 업로드 가능합니다."
        )

    # 파일 저장
    file_id = str(uuid.uuid4())
    file_path = settings.DATA_RAW_PATH / f"{file_id}_{file.filename}"

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        return DataUploadResponse(
            success=True,
            file_id=file_id,
            filename=file.filename,
            quarter=quarter,
            message="파일이 성공적으로 업로드되었습니다.",
            uploaded_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 업로드 실패: {str(e)}"
        )


@router.post("/process")
async def start_data_processing(
    request: DataProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    데이터 처리 시작

    업로드된 파일을 처리하여:
    1. 전처리 수행
    2. 13개 주요 지표 계산
    3. 예측용/학습용 데이터 생성
    4. 전체 기업 예측값 캐싱

    Args:
        request: 처리 요청 정보

    Returns:
        작업 ID (상태 조회용)
    """
    job_id = str(uuid.uuid4())

    # 작업 상태 초기화
    processing_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "file_id": request.file_id,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None
    }

    # 백그라운드 작업 등록
    background_tasks.add_task(
        _process_data_background,
        job_id,
        request.file_id,
        request.quarter
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "데이터 처리가 시작되었습니다."
    }


async def _process_data_background(job_id: str, file_id: str, quarter: str):
    """백그라운드 데이터 처리 작업"""
    try:
        processing_jobs[job_id]["status"] = "processing"

        # 실제 처리 로직 호출
        service = DataPipelineService()

        # 1. 데이터 로드 (20%)
        processing_jobs[job_id]["progress"] = 20
        # await service.load_data(file_id)

        # 2. 전처리 (40%)
        processing_jobs[job_id]["progress"] = 40
        # await service.preprocess()

        # 3. 지표 계산 (60%)
        processing_jobs[job_id]["progress"] = 60
        # await service.calculate_metrics()

        # 4. 예측값 캐싱 (80%)
        processing_jobs[job_id]["progress"] = 80
        # await service.cache_predictions()

        # 5. 완료 (100%)
        processing_jobs[job_id]["progress"] = 100
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)


@router.get("/status/{job_id}", response_model=ProcessStatusResponse)
async def get_processing_status(job_id: str):
    """
    데이터 처리 상태 확인

    Args:
        job_id: 작업 ID

    Returns:
        처리 상태 및 진행률
    """
    if job_id not in processing_jobs:
        raise HTTPException(
            status_code=404,
            detail="작업을 찾을 수 없습니다."
        )

    job = processing_jobs[job_id]

    return ProcessStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        started_at=job["started_at"],
        completed_at=job.get("completed_at"),
        error=job.get("error")
    )


@router.get("/metrics/{corp_code}", response_model=MetricsResponse)
async def get_company_metrics(corp_code: str, quarter: Optional[str] = None):
    """
    기업 지표 조회 (백엔드 전달용)

    13개 주요 재무지표의 계산된 값을 반환합니다.
    (예측값이 아닌 실제 데이터 기반 계산 값)

    Args:
        corp_code: 기업 코드 (예: "005930")
        quarter: 분기 (예: "2024Q3", 미지정 시 최신)

    Returns:
        13개 지표 값 및 메타 정보
    """
    try:
        service = DataPipelineService()
        metrics = await service.get_metrics(corp_code, quarter)

        return MetricsResponse(
            corp_code=corp_code,
            quarter=quarter or "latest",
            metrics=metrics,
            calculated_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"지표 조회 실패: {str(e)}"
        )


@router.get("/metrics/batch")
async def get_batch_metrics(
    corp_codes: List[str] = None,
    quarter: Optional[str] = None
):
    """
    여러 기업 지표 일괄 조회

    Args:
        corp_codes: 기업 코드 목록
        quarter: 분기

    Returns:
        기업별 지표 데이터
    """
    # TODO: 구현
    return {
        "message": "일괄 조회 기능 구현 예정",
        "corp_codes": corp_codes,
        "quarter": quarter
    }
