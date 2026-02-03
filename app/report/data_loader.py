"""
데이터 로드 및 기업 조회 모듈
============================

이 모듈은 CSV 파일에서 데이터를 로드하고,
특정 기업의 재무 정보를 조회하는 기능을 제공합니다.

주요 기능:
- 전체 기업 목록 조회
- 특정 기업의 현재 재무 지표 조회
- 업종 상대 지표 조회
- 시계열 추세 데이터 조회
- 업종 내 백분위 계산
- 예측용 피처 추출

사용 예시:
    loader = get_data_loader()
    company_data = loader.get_company_data('[005930]')
"""
import pandas as pd
import numpy as np
import json
from typing import Optional, Dict, List, Tuple

# 설정 파일에서 필요한 값들 import
from .config import (
    DATASET_FILE,           # 데이터 파일 경로
    FEATURE_COLS_FILE,      # 피처 목록 파일 경로
    MODEL_PATH,             # 모델 저장 경로
    META_COLS,              # 메타 컬럼 목록
    ALL_TARGETS,            # 전체 예측 타겟 (13개)
    TARGET_METRICS,         # 카테고리별 타겟
    METRIC_DIRECTION        # 지표별 방향성
)


class DataLoader:
    """
    데이터 로드 및 기업 정보 조회 클래스

    이 클래스는 싱글톤 패턴으로 사용됩니다.
    get_data_loader() 함수를 통해 인스턴스를 가져옵니다.
    """

    def __init__(self):
        """
        초기화: 데이터 파일 로드

        - CSV 파일에서 전체 데이터를 로드
        - JSON 파일에서 피처 목록을 로드
        """
        self.df: Optional[pd.DataFrame] = None      # 전체 데이터
        self.feature_cols: Optional[List[str]] = None  # 피처 목록 (376개)
        self._load_data()

    def _load_data(self):
        """
        데이터 파일 로드 (내부 메서드)

        CSV 파일: 약 68,000행 x 400열의 재무 데이터
        JSON 파일: 모델에 사용되는 376개 피처 목록
        """
        # CSV 파일 로드 (low_memory=False로 dtype 경고 방지)
        self.df = pd.read_csv(DATASET_FILE, low_memory=False)

        # 피처 목록 로드
        with open(FEATURE_COLS_FILE, 'r', encoding='utf-8') as f:
            self.feature_cols = json.load(f)

    def get_company_list(self) -> pd.DataFrame:
        """
        전체 기업 목록 조회

        각 기업의 최신 분기 데이터를 기준으로 목록 반환

        Returns:
            DataFrame: 기업코드, 기업명, 시장구분, 업종명 컬럼
        """
        # 각 기업의 가장 최신 데이터만 추출
        latest = self.df.groupby('기업코드').apply(
            lambda x: x.sort_values(['년도', '분기']).iloc[-1]
        ).reset_index(drop=True)

        return latest[['기업코드', '기업명', '시장구분', '업종명']]

    def get_company_data(self, company_code: str) -> Optional[Dict]:
        """
        특정 기업의 전체 데이터 조회

        Args:
            company_code: 기업 코드 (예: '[005930]')

        Returns:
            Dict: 기업 데이터 딕셔너리
                - meta: 기본 정보 (기업명, 업종 등)
                - current: 현재 재무 지표 (13개)
                - relative: 업종 상대 지표
                - trend: 추세 데이터 (YoY, MOM, MA, STD, lag)
                - historical: 최근 4분기 데이터
                - raw_data: 원본 데이터 (Series)

            None: 기업 코드가 존재하지 않는 경우
        """
        # 해당 기업 데이터 필터링
        company_df = self.df[self.df['기업코드'] == company_code].copy()

        if company_df.empty:
            return None

        # 최신 분기 데이터 (가장 마지막 행)
        latest = company_df.sort_values(['년도', '분기']).iloc[-1]

        # 각 카테고리별 데이터 추출
        return {
            'meta': self._extract_meta(latest),              # 기본 정보
            'current': self._extract_current_metrics(latest), # 현재 지표
            'relative': self._extract_relative_metrics(latest), # 업종 상대
            'trend': self._extract_trend_metrics(latest),     # 추세 데이터
            'historical': self._extract_historical(company_df), # 과거 데이터
            'raw_data': latest,                               # 원본 (다른 용도로 사용)
        }

    def _extract_meta(self, row: pd.Series) -> Dict:
        """
        메타 정보 추출 (내부 메서드)

        Args:
            row: 최신 분기 데이터 행

        Returns:
            Dict: 기업 기본 정보
        """
        return {
            '기업코드': row['기업코드'],
            '기업명': row['기업명'],
            '시장구분': row['시장구분'],
            '업종코드': row['업종코드'],
            '업종명': row['업종명'],
            '년도': int(row['년도']),
            '분기': row['분기'],
            '기준일': f"{int(row['년도'])}년 {row['분기']}",  # 표시용 문자열
        }

    def _extract_current_metrics(self, row: pd.Series) -> Dict:
        """
        현재 재무 지표 추출 (내부 메서드)

        13개 예측 타겟 지표의 현재 값을 카테고리별로 정리

        Args:
            row: 최신 분기 데이터 행

        Returns:
            Dict: 카테고리별 지표 값
                {
                    '수익성': {
                        'ROA': {'value': 5.2, 'is_missing': False},
                        ...
                    },
                    ...
                }
        """
        result = {}

        # 카테고리별로 순회 (수익성, 안정성, 차입금, 유동성, 현금흐름)
        for category, metrics in TARGET_METRICS.items():
            result[category] = {}

            for metric in metrics:
                value = row.get(metric)  # 해당 지표 값 가져오기
                result[category][metric] = {
                    'value': float(value) if pd.notna(value) else None,
                    'is_missing': pd.isna(value),  # 결측 여부
                }

        return result

    def _extract_relative_metrics(self, row: pd.Series) -> Dict:
        """
        업종 상대 지표 추출 (내부 메서드)

        각 지표의 업종 평균 대비 차이값 (_업종상대 컬럼)

        예: ROA_업종상대 = 해당 기업 ROA - 업종 평균 ROA

        Args:
            row: 최신 분기 데이터 행

        Returns:
            Dict: {지표명: 상대값} (예: {'ROA': 2.1, 'ROE': -0.5, ...})
        """
        result = {}

        for metric in ALL_TARGETS:
            rel_col = f'{metric}_업종상대'  # 컬럼명 생성

            if rel_col in row.index:
                value = row.get(rel_col)
                result[metric] = float(value) if pd.notna(value) else None
            else:
                result[metric] = None

        return result

    def _extract_trend_metrics(self, row: pd.Series) -> Dict:
        """
        추세 지표 추출 (내부 메서드)

        각 지표의 시계열 관련 파생 피처들:
        - YoY: 전년동기대비 변화율 (%)
        - MOM: 전분기대비 변화율 (%)
        - MA4: 최근 4분기 이동평균
        - STD4: 최근 4분기 표준편차 (변동성)
        - lag1~3: 과거 1~3분기 값

        Args:
            row: 최신 분기 데이터 행

        Returns:
            Dict: {지표명: {yoy, mom, ma4, std4, lag1, lag2, lag3}}
        """
        result = {}

        for metric in ALL_TARGETS:
            result[metric] = {
                'yoy': self._safe_get(row, f'{metric}_YoY'),    # 전년동기비
                'mom': self._safe_get(row, f'{metric}_MOM'),    # 전분기비
                'ma4': self._safe_get(row, f'{metric}_MA4'),    # 이동평균
                'std4': self._safe_get(row, f'{metric}_STD4'),  # 변동성
                'lag1': self._safe_get(row, f'{metric}_lag1'),  # 1분기 전
                'lag2': self._safe_get(row, f'{metric}_lag2'),  # 2분기 전
                'lag3': self._safe_get(row, f'{metric}_lag3'),  # 3분기 전
            }

        return result

    def _extract_historical(self, company_df: pd.DataFrame, n_quarters: int = 5) -> Dict:
        """
        최근 N분기 데이터 추출 (내부 메서드)

        시계열 차트 그리기 등에 사용 (기본 5분기 = 1년 추세)

        Args:
            company_df: 해당 기업의 전체 데이터
            n_quarters: 추출할 분기 수 (기본 5)

        Returns:
            Dict: {
                'periods': ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2', '2025 Q3'],
                'metrics': {
                    'ROA': [1.5, 1.6, 1.0, 2.3, 2.5],
                    ...
                }
            }
        """
        # 최근 N분기만 추출
        recent = company_df.sort_values(['년도', '분기']).tail(n_quarters)

        result = {
            'periods': [],  # 기간 라벨
            'metrics': {metric: [] for metric in ALL_TARGETS}  # 지표별 값 리스트
        }

        for _, row in recent.iterrows():
            # 기간 라벨 추가 (예: "2025 Q3")
            result['periods'].append(f"{int(row['년도'])} {row['분기']}")

            # 각 지표 값 추가
            for metric in ALL_TARGETS:
                value = row.get(metric)
                result['metrics'][metric].append(
                    float(value) if pd.notna(value) else None
                )

        return result

    def get_industry_metric_comparison(self, company_code: str) -> Dict:
        """
        업종 내 지표별 비교 데이터

        각 지표별로 해당 기업 값, 업종 평균, 업종 중앙값을 비교

        Args:
            company_code: 기업 코드

        Returns:
            Dict: {
                'company_name': '삼성전자',
                'industry': '통신 및 방송장비 제조업',
                'metrics': {
                    'ROA': {'company': 2.3, 'industry_mean': 1.5, 'industry_median': 1.2, 'percentile': 18},
                    ...
                }
            }
        """
        company_data = self.get_company_data(company_code)
        if not company_data:
            return {}

        meta = company_data['meta']
        industry_code = meta['업종코드']
        year = meta['년도']
        quarter = meta['분기']

        # 업종 통계 조회
        industry_stats = self.get_industry_stats(industry_code, year, quarter)

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

    def _safe_get(self, row: pd.Series, col: str) -> Optional[float]:
        """
        안전하게 값 추출 (내부 메서드)

        컬럼이 없거나 값이 NaN인 경우 None 반환

        Args:
            row: 데이터 행
            col: 컬럼명

        Returns:
            float 또는 None
        """
        if col in row.index:
            value = row.get(col)
            return float(value) if pd.notna(value) else None
        return None

    def get_industry_stats(self, industry_code: str, year: int, quarter: str) -> Dict:
        """
        업종별 통계 조회

        특정 업종, 특정 분기의 지표별 통계 (평균, 표준편차, 사분위수 등)

        Args:
            industry_code: 업종 코드
            year: 연도
            quarter: 분기 (예: 'Q3')

        Returns:
            Dict: {지표명: {count, mean, std, min, max, q25, q50, q75}}
        """
        # 해당 업종, 해당 분기 데이터 필터링
        industry_df = self.df[
            (self.df['업종코드'] == industry_code) &
            (self.df['년도'] == year) &
            (self.df['분기'] == quarter)
        ]

        result = {}

        for metric in ALL_TARGETS:
            values = industry_df[metric].dropna()  # 결측 제외

            if len(values) > 0:
                result[metric] = {
                    'count': len(values),                    # 데이터 수
                    'mean': float(values.mean()),           # 평균
                    'std': float(values.std()),             # 표준편차
                    'min': float(values.min()),             # 최소값
                    'max': float(values.max()),             # 최대값
                    'q25': float(values.quantile(0.25)),    # 1사분위수
                    'q50': float(values.quantile(0.50)),    # 중앙값
                    'q75': float(values.quantile(0.75)),    # 3사분위수
                }
            else:
                result[metric] = None

        return result

    def calculate_percentile(self, company_code: str, metric: str) -> Optional[float]:
        """
        기업의 업종 내 백분위 계산

        해당 기업이 같은 업종 내에서 상위 몇 %인지 계산
        (지표 방향성 고려: 높을수록 좋은 지표는 높은 값이 상위,
         낮을수록 좋은 지표는 낮은 값이 상위)

        Args:
            company_code: 기업 코드
            metric: 지표명

        Returns:
            float: 상위 백분위 (예: 18.5 = 상위 18.5%)
            None: 계산 불가 시
        """
        # 기업 데이터 조회
        company_data = self.get_company_data(company_code)
        if not company_data:
            return None

        meta = company_data['meta']
        industry_code = meta['업종코드']
        year = meta['년도']
        quarter = meta['분기']

        # 같은 업종, 같은 분기 데이터 필터링
        industry_df = self.df[
            (self.df['업종코드'] == industry_code) &
            (self.df['년도'] == year) &
            (self.df['분기'] == quarter)
        ]

        values = industry_df[metric].dropna()
        company_value = company_data['raw_data'].get(metric)

        # 값이 없으면 계산 불가
        if pd.isna(company_value) or len(values) == 0:
            return None

        # 방향성에 따라 백분위 계산
        direction = METRIC_DIRECTION.get(metric, 'higher')

        if direction == 'higher':
            # 높을수록 좋은 지표: 자기보다 낮은 값의 비율 = 하위 %
            # 상위 % = 100 - 하위 %
            percentile = (values < company_value).mean() * 100
            return 100 - percentile
        else:
            # 낮을수록 좋은 지표: 자기보다 높은 값의 비율 = 하위 %
            # 상위 % = 100 - 하위 %
            percentile = (values > company_value).mean() * 100
            return 100 - percentile

    def get_features_for_prediction(self, company_code: str) -> Optional[np.ndarray]:
        """
        예측용 피처 추출

        XGBoost 모델에 입력할 피처 배열 생성
        고정된 376개 피처를 순서대로 추출

        Args:
            company_code: 기업 코드

        Returns:
            np.ndarray: (1, 376) 형태의 피처 배열
            None: 추출 실패 시
        """
        company_data = self.get_company_data(company_code)
        if not company_data:
            return None

        row = company_data['raw_data']

        # 고정된 피처 목록 순서대로 값 추출
        features = row[self.feature_cols].values.astype(np.float32)

        # 모델 입력 형태로 변환 (1행 x 376열)
        return features.reshape(1, -1)


# =============================================================================
# 싱글톤 패턴
# =============================================================================
# 전역 인스턴스 (처음에는 None)
_data_loader: Optional[DataLoader] = None


def get_data_loader() -> DataLoader:
    """
    DataLoader 싱글톤 인스턴스 반환

    처음 호출 시 인스턴스 생성, 이후에는 기존 인스턴스 재사용
    (데이터를 매번 로드하지 않아 효율적)

    Returns:
        DataLoader: 데이터 로더 인스턴스
    """
    global _data_loader

    if _data_loader is None:
        _data_loader = DataLoader()

    return _data_loader
