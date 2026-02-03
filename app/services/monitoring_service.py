"""
모니터링 서비스
==============

모델 성능 모니터링 및 관리를 담당하는 서비스

주요 기능:
1. 모델 성능 지표 조회
2. 분기별 예측 피드백
3. 캐시 관리
4. 모델 이력 관리
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging
import json

from app.config import settings

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    모니터링 서비스

    모델 성능을 모니터링하고 관리 기능을 제공합니다.
    """

    def __init__(self):
        self.model_path = settings.MODEL_PATH
        self.cache_path = settings.DATA_CACHE_PATH

    async def get_model_performance(self) -> Dict[str, Any]:
        """
        현재 모델 성능 조회

        Returns:
            지표별 R², MAE, RMSE 및 전체 평균
        """
        logger.info("Getting model performance...")

        # TODO: 실제 구현 - 모델 메타데이터에서 성능 조회
        # 학습 시 저장된 성능 지표 로드

        metrics_performance = {}
        target_metrics = [
            'ROA', 'ROE', '매출액영업이익률',
            '부채비율', '자기자본비율', '자본잠식률',
            '단기차입금비율',
            '유동비율', '당좌비율', '유동부채비율',
            'CFO_자산비율', 'CFO_매출액비율', 'CFO증감률'
        ]

        for metric in target_metrics:
            # TODO: 실제 성능 로드
            metrics_performance[metric] = {
                "r2": None,
                "mae": None,
                "rmse": None
            }

        return {
            "metrics": metrics_performance,
            "overall": {
                "avg_r2": None,
                "avg_mae": None,
                "avg_rmse": None
            },
            "last_trained": None,
            "sample_size": 0
        }

    async def get_quarterly_feedback(self, quarter: str) -> Dict[str, Any]:
        """
        분기별 예측 피드백

        예측값과 실제값을 비교하여 피드백을 제공합니다.

        Args:
            quarter: 분기 (예: "2024Q3")

        Returns:
            예측 vs 실제 비교 데이터
        """
        logger.info(f"Getting feedback for {quarter}")

        # TODO: 실제 구현
        # 1. 해당 분기의 예측값 조회 (캐시)
        # 2. 실제값 조회 (데이터)
        # 3. 비교 및 오차 분석

        return {
            "comparison": [],
            "error_analysis": {
                "mean_absolute_error": None,
                "mean_percentage_error": None,
                "worst_predictions": []
            },
            "recommendations": []
        }

    async def get_model_history(self) -> List[Dict[str, Any]]:
        """
        모델 학습 이력 조회

        Returns:
            버전별 학습 이력
        """
        # TODO: 실제 구현 - 모델 버전 이력 관리
        return []

    async def get_cache_status(self) -> Dict[str, Any]:
        """
        캐시 상태 조회

        Returns:
            캐시 통계 정보
        """
        cache_files = list(self.cache_path.glob("*.json"))

        total_size = sum(f.stat().st_size for f in cache_files) if cache_files else 0

        latest_update = None
        if cache_files:
            latest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
            latest_update = datetime.fromtimestamp(latest_file.stat().st_mtime)

        return {
            "count": len(cache_files),
            "size_mb": total_size / (1024 * 1024),
            "last_updated": latest_update,
            "hit_rate": None  # TODO: 캐시 히트율 추적
        }

    async def refresh_company_cache(self, corp_code: str) -> None:
        """
        특정 기업 캐시 갱신

        Args:
            corp_code: 기업 코드
        """
        logger.info(f"Refreshing cache for {corp_code}")

        # 1. 기존 캐시 삭제
        cache_file = self.cache_path / f"{corp_code}.json"
        if cache_file.exists():
            cache_file.unlink()

        # 2. 새 예측값 계산 및 캐싱
        # TODO: AnalysisService 호출

    async def refresh_all_cache(self) -> Dict[str, Any]:
        """
        전체 캐시 갱신

        Returns:
            갱신 결과
        """
        logger.info("Refreshing all cache...")

        # TODO: 전체 기업 목록 조회 후 순차 갱신
        return {
            "total": 0,
            "refreshed": 0,
            "failed": 0
        }

    async def clear_cache(self) -> None:
        """
        캐시 전체 삭제
        """
        logger.info("Clearing all cache...")

        for cache_file in self.cache_path.glob("*.json"):
            cache_file.unlink()

    async def log_prediction(
        self,
        corp_code: str,
        quarter: str,
        predictions: Dict[str, float]
    ) -> None:
        """
        예측 기록 저장 (나중에 피드백용)

        Args:
            corp_code: 기업 코드
            quarter: 예측 대상 분기
            predictions: 예측값
        """
        log_path = self.cache_path / "prediction_logs"
        log_path.mkdir(exist_ok=True)

        log_file = log_path / f"{quarter}_{corp_code}.json"
        log_data = {
            "corp_code": corp_code,
            "quarter": quarter,
            "predictions": predictions,
            "predicted_at": datetime.now().isoformat()
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    async def compare_with_actual(
        self,
        corp_code: str,
        quarter: str,
        actual_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        예측값과 실제값 비교

        Args:
            corp_code: 기업 코드
            quarter: 분기
            actual_values: 실제값

        Returns:
            비교 결과
        """
        # 예측 기록 로드
        log_path = self.cache_path / "prediction_logs" / f"{quarter}_{corp_code}.json"

        if not log_path.exists():
            raise ValueError(f"No prediction record for {corp_code} in {quarter}")

        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        predictions = log_data["predictions"]

        # 비교
        comparison = []
        for metric in predictions:
            pred = predictions.get(metric)
            actual = actual_values.get(metric)

            if pred is not None and actual is not None:
                error = actual - pred
                error_rate = abs(error / actual) * 100 if actual != 0 else 0

                comparison.append({
                    "metric": metric,
                    "predicted": pred,
                    "actual": actual,
                    "error": error,
                    "error_rate": error_rate
                })

        return {
            "corp_code": corp_code,
            "quarter": quarter,
            "comparison": comparison
        }
