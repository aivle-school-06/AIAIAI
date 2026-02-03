"""
데이터 로더
==========

CSV 데이터 로드 및 기업 정보 조회.
기존 backend/report/data_loader.py를 서버용으로 재구성.
"""
import json
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

import pandas as pd
import numpy as np

from app.config import settings, Settings
from app.core.constants import ALL_TARGETS, TARGET_METRICS, META_COLS, METRIC_DIRECTION
from app.exceptions import DataLoadError, CompanyNotFoundError

logger = logging.getLogger(__name__)


class DataLoader:
    """
    데이터 로드 및 기업 정보 조회

    Args:
        settings: 애플리케이션 설정
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: Optional[List[str]] = None
        self._load_data()

    def _load_data(self) -> None:
        """데이터 파일 로드"""
        try:
            # ai-server/data/processed/ 경로 사용
            data_path = settings.DATA_PROCESSED_PATH
            dataset_file = data_path / '03_dataset_final.csv'
            feature_file = data_path / '03_feature_cols.json'

            logger.info(f"Loading data from {dataset_file}")
            self.df = pd.read_csv(dataset_file, low_memory=False)
            logger.info(f"Loaded {len(self.df)} rows")

            with open(feature_file, 'r', encoding='utf-8') as f:
                self.feature_cols = json.load(f)
            logger.info(f"Loaded {len(self.feature_cols)} feature columns")

        except FileNotFoundError as e:
            raise DataLoadError(f"데이터 파일을 찾을 수 없습니다: {e}")
        except Exception as e:
            raise DataLoadError(f"데이터 로드 실패: {e}")

    def get_company_list(self) -> pd.DataFrame:
        """
        전체 기업 목록 조회

        Returns:
            DataFrame: 기업코드, 기업명, 시장구분, 업종명
        """
        # 각 기업의 최신 분기 데이터만 추출
        latest = (
            self.df
            .sort_values(['기업코드', '년도', '분기'])
            .drop_duplicates('기업코드', keep='last')
        )

        return latest[['기업코드', '기업명', '시장구분', '업종명']]

    def get_company_data(self, company_code: str) -> Dict[str, Any]:
        """
        특정 기업의 전체 데이터 조회

        Args:
            company_code: 기업 코드 (예: '[005930]')

        Returns:
            Dict: 기업 데이터

        Raises:
            CompanyNotFoundError: 기업이 존재하지 않는 경우
        """
        company_df = self.df[self.df['기업코드'] == company_code].copy()

        if company_df.empty:
            raise CompanyNotFoundError(company_code)

        latest = company_df.sort_values(['년도', '분기']).iloc[-1]

        return {
            'meta': self._extract_meta(latest),
            'current': self._extract_current_metrics(latest),
            'relative': self._extract_relative_metrics(latest),
            'trend': self._extract_trend_metrics(latest),
            'historical': self._extract_historical(company_df),
            'raw_data': latest,
        }

    def _extract_meta(self, row: pd.Series) -> Dict[str, Any]:
        """메타 정보 추출"""
        return {
            '기업코드': row['기업코드'],
            '기업명': row['기업명'],
            '시장구분': row['시장구분'],
            '업종코드': row['업종코드'],
            '업종명': row['업종명'],
            '년도': int(row['년도']),
            '분기': row['분기'],
            '기준일': f"{int(row['년도'])}년 {row['분기']}",
        }

    def _extract_current_metrics(self, row: pd.Series) -> Dict[str, Any]:
        """현재 재무 지표 추출"""
        result = {}

        for category, metrics in TARGET_METRICS.items():
            result[category] = {}
            for metric in metrics:
                value = row.get(metric)
                result[category][metric] = {
                    'value': float(value) if pd.notna(value) else None,
                    'is_missing': pd.isna(value),
                }

        return result

    def _extract_relative_metrics(self, row: pd.Series) -> Dict[str, Optional[float]]:
        """업종 상대 지표 추출"""
        result = {}

        for metric in ALL_TARGETS:
            rel_col = f'{metric}_업종상대'
            if rel_col in row.index:
                value = row.get(rel_col)
                result[metric] = float(value) if pd.notna(value) else None
            else:
                result[metric] = None

        return result

    def _extract_trend_metrics(self, row: pd.Series) -> Dict[str, Dict[str, Optional[float]]]:
        """추세 지표 추출"""
        result = {}

        for metric in ALL_TARGETS:
            result[metric] = {
                'yoy': self._safe_get(row, f'{metric}_YoY'),
                'mom': self._safe_get(row, f'{metric}_MOM'),
                'ma4': self._safe_get(row, f'{metric}_MA4'),
                'std4': self._safe_get(row, f'{metric}_STD4'),
                'lag1': self._safe_get(row, f'{metric}_lag1'),
                'lag2': self._safe_get(row, f'{metric}_lag2'),
                'lag3': self._safe_get(row, f'{metric}_lag3'),
            }

        return result

    def _extract_historical(
        self, company_df: pd.DataFrame, n_quarters: int = 5
    ) -> Dict[str, Any]:
        """최근 N분기 데이터 추출"""
        recent = company_df.sort_values(['년도', '분기']).tail(n_quarters)

        result = {
            'periods': [],
            'metrics': {metric: [] for metric in ALL_TARGETS}
        }

        for _, row in recent.iterrows():
            result['periods'].append(f"{int(row['년도'])} {row['분기']}")

            for metric in ALL_TARGETS:
                value = row.get(metric)
                result['metrics'][metric].append(
                    float(value) if pd.notna(value) else None
                )

        return result

    def _safe_get(self, row: pd.Series, col: str) -> Optional[float]:
        """안전하게 값 추출"""
        if col in row.index:
            value = row.get(col)
            return float(value) if pd.notna(value) else None
        return None

    def get_industry_stats(
        self, industry_code: str, year: int, quarter: str
    ) -> Dict[str, Any]:
        """업종별 통계 조회"""
        industry_df = self.df[
            (self.df['업종코드'] == industry_code) &
            (self.df['년도'] == year) &
            (self.df['분기'] == quarter)
        ]

        result = {}

        for metric in ALL_TARGETS:
            values = industry_df[metric].dropna()

            if len(values) > 0:
                result[metric] = {
                    'count': len(values),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q50': float(values.quantile(0.50)),
                    'q75': float(values.quantile(0.75)),
                }
            else:
                result[metric] = None

        return result

    def calculate_percentile(
        self, company_code: str, metric: str
    ) -> Optional[float]:
        """업종 내 백분위 계산"""
        company_data = self.get_company_data(company_code)
        meta = company_data['meta']

        industry_df = self.df[
            (self.df['업종코드'] == meta['업종코드']) &
            (self.df['년도'] == meta['년도']) &
            (self.df['분기'] == meta['분기'])
        ]

        values = industry_df[metric].dropna()
        company_value = company_data['raw_data'].get(metric)

        if pd.isna(company_value) or len(values) == 0:
            return None

        direction = METRIC_DIRECTION.get(metric, 'higher')

        if direction == 'higher':
            percentile = (values < company_value).mean() * 100
            return 100 - percentile
        else:
            percentile = (values > company_value).mean() * 100
            return 100 - percentile

    def get_features_for_prediction(self, company_code: str) -> Optional[np.ndarray]:
        """예측용 피처 추출"""
        company_data = self.get_company_data(company_code)
        row = company_data['raw_data']

        features = row[self.feature_cols].values.astype(np.float32)
        return features.reshape(1, -1)

    def get_industry_comparison(self, company_code: str) -> Dict[str, Any]:
        """업종 내 지표별 비교 데이터"""
        company_data = self.get_company_data(company_code)
        meta = company_data['meta']

        industry_stats = self.get_industry_stats(
            meta['업종코드'], meta['년도'], meta['분기']
        )

        result = {
            'company_name': meta['기업명'],
            'industry': meta['업종명'],
            'metrics': {}
        }

        for metric in ALL_TARGETS:
            company_value = company_data['raw_data'].get(metric)
            stats = industry_stats.get(metric)

            if pd.notna(company_value) and stats:
                percentile = self.calculate_percentile(company_code, metric)
                result['metrics'][metric] = {
                    'company': float(company_value),
                    'industry_mean': stats['mean'],
                    'industry_median': stats['q50'],
                    'industry_min': stats['min'],
                    'industry_max': stats['max'],
                    'percentile': percentile,
                    'count': stats['count'],
                }
            else:
                result['metrics'][metric] = None

        return result
