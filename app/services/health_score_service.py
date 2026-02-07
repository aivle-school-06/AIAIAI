"""
재무건전성 점수 서비스
====================

기업의 재무건전성을 0~100점으로 평가하는 서비스.
5개 핵심 지표에 대해 고정 임계값 기준으로 점수 산출.
과거 4분기 + 예측 1분기 점수를 제공.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

import pandas as pd
import numpy as np

from app.core.data_loader import DataLoader
from app.core.predictor import Predictor
from app.exceptions import CompanyNotFoundError

logger = logging.getLogger(__name__)


# =============================================================================
# 점수 산출 기준 (고정 임계값)
# =============================================================================
# 전체 데이터(2016~2025) 분포 기반으로 설정
# 대략 상위 30% = 안정, 중위 30% = 주의, 하위 40% = 위험

HEALTH_THRESHOLDS = {
    'ROA': {
        'direction': 'up',      # 높을수록 좋음
        '안정': (1.5, None),    # > 1.5%
        '주의': (0, 1.5),       # 0% ~ 1.5%
        # 위험: < 0%
    },
    '부채비율': {
        'direction': 'down',    # 낮을수록 좋음
        '안정': (None, 50),     # < 50%
        '주의': (50, 100),      # 50% ~ 100%
        # 위험: > 100%
    },
    '유동비율': {
        'direction': 'up',      # 높을수록 좋음
        '안정': (200, None),    # > 200%
        '주의': (130, 200),     # 130% ~ 200%
        # 위험: < 130%
    },
    'CFO_자산비율': {
        'direction': 'up',      # 높을수록 좋음
        '안정': (2, None),      # > 2%
        '주의': (0, 2),         # 0% ~ 2%
        # 위험: < 0%
    },
    '단기차입금비율': {
        'direction': 'down',    # 낮을수록 좋음
        '안정': (None, 60),     # < 60%
        '주의': (60, 85),       # 60% ~ 85%
        # 위험: > 85%
    },
}

# 최소 유효 지표 수
MIN_VALID_INDICATORS = 3

# 건전성 지표 목록
HEALTH_INDICATORS = list(HEALTH_THRESHOLDS.keys())


class HealthScoreService:
    """
    재무건전성 점수 서비스

    - 5개 핵심 지표로 0~100점 산출
    - 과거 4분기 실적 + 예측 1분기 제공
    - 라벨: 위험(<40), 주의(40~70), 안정(≥70)
    """

    def __init__(self, data_loader: DataLoader, predictor: Predictor):
        self.data_loader = data_loader
        self.predictor = predictor

    def get_health_scores(self, company_code: str) -> Dict[str, Any]:
        """
        기업의 재무건전성 점수 조회

        Args:
            company_code: 기업 코드 (예: '[005930]')

        Returns:
            Dict: 분기별 건전성 점수
        """
        # 기업 데이터 조회
        company_df = self.data_loader.df[
            self.data_loader.df['기업코드'] == company_code
        ].copy()

        if company_df.empty:
            raise CompanyNotFoundError(company_code)

        # 최신 4분기 데이터 추출
        company_df = company_df.sort_values(['년도', '분기'])
        recent_quarters = company_df.tail(4)

        # 기업 정보
        latest = company_df.iloc[-1]
        company_name = latest['기업명']
        clean_code = company_code.strip('[]')

        # 분기별 점수 계산 (과거 + 현재)
        quarters = []
        for _, row in recent_quarters.iterrows():
            period = self._format_period(row['년도'], row['분기'])
            score_result = self._calculate_score(row)

            quarters.append({
                'period': period,
                'score': score_result['score'],
                'label': score_result['label'],
                'type': 'actual',
            })

        # 현재 분기 점수
        current_score = quarters[-1]['score'] if quarters else None

        # 예측 분기 점수 계산
        predicted_quarter = self._get_predicted_score(company_code, latest)
        if predicted_quarter:
            quarters.append(predicted_quarter)

        predicted_score = predicted_quarter['score'] if predicted_quarter else None

        return {
            'company_code': clean_code,
            'company_name': company_name,
            'quarters': quarters,
            'current_score': current_score,
            'predicted_score': predicted_score,
        }

    def _format_period(self, year: int, quarter: str) -> str:
        """
        분기를 'YYYYQ' 형식으로 변환

        Args:
            year: 년도 (예: 2025)
            quarter: 분기 (예: 'Q3' 또는 3)

        Returns:
            str: '20253' 형식
        """
        if isinstance(quarter, str):
            # 'Q3' -> '3'
            q_num = quarter.replace('Q', '').strip()
        else:
            q_num = str(int(quarter))

        return f"{year}{q_num}"

    def _get_next_period(self, year: int, quarter: str) -> str:
        """다음 분기 계산"""
        if isinstance(quarter, str):
            q_num = int(quarter.replace('Q', '').strip())
        else:
            q_num = int(quarter)

        if q_num >= 4:
            return f"{year + 1}1"
        else:
            return f"{year}{q_num + 1}"

    def _calculate_score(self, row: pd.Series) -> Dict[str, Any]:
        """
        단일 분기 점수 계산

        Args:
            row: 분기 데이터 행

        Returns:
            Dict: {'score': float, 'label': str, 'details': Dict}
        """
        scores = []
        details = {}

        for indicator in HEALTH_INDICATORS:
            value = row.get(indicator)
            ind_score = self._get_indicator_score(value, indicator)

            if ind_score is not None:
                scores.append(ind_score)
                details[indicator] = {
                    'value': float(value) if pd.notna(value) else None,
                    'score': ind_score,
                    'label': ['위험', '주의', '안정'][ind_score],
                }
            else:
                details[indicator] = {
                    'value': None,
                    'score': None,
                    'label': None,
                }

        # 총점 계산
        if len(scores) >= MIN_VALID_INDICATORS:
            max_score = len(scores) * 2
            total_score = round((sum(scores) / max_score) * 100, 1)
            label = self._get_label(total_score)
        else:
            total_score = None
            label = None

        return {
            'score': total_score,
            'label': label,
            'details': details,
        }

    def _get_indicator_score(
        self, value: Optional[float], indicator: str
    ) -> Optional[int]:
        """
        개별 지표 점수 (0/1/2)

        Args:
            value: 지표 값
            indicator: 지표명

        Returns:
            int: 0(위험), 1(주의), 2(안정) 또는 None
        """
        if pd.isna(value):
            return None

        config = HEALTH_THRESHOLDS[indicator]

        # 안정 (2점)
        low, high = config['안정']
        if (low is None or value > low) and (high is None or value < high):
            return 2

        # 주의 (1점)
        low, high = config['주의']
        if (low is None or value >= low) and (high is None or value <= high):
            return 1

        # 위험 (0점)
        return 0

    def _get_label(self, score: Optional[float]) -> Optional[str]:
        """점수를 라벨로 변환"""
        if score is None:
            return None
        elif score >= 70:
            return '안정'
        elif score >= 40:
            return '주의'
        else:
            return '위험'

    def _get_predicted_score(
        self, company_code: str, latest_row: pd.Series
    ) -> Optional[Dict]:
        """
        예측값 기반 점수 계산

        Args:
            company_code: 기업 코드
            latest_row: 최신 분기 데이터

        Returns:
            Dict: 예측 분기 점수 정보
        """
        try:
            # 예측 수행
            prediction_result = self.predictor.predict(company_code)
            predictions = prediction_result.get('predictions', {})

            # 예측값으로 가상 row 생성
            predicted_values = {}
            for indicator in HEALTH_INDICATORS:
                pred = predictions.get(indicator, {})
                if isinstance(pred, dict) and 'predicted' in pred:
                    predicted_values[indicator] = pred['predicted']
                else:
                    predicted_values[indicator] = None

            # 점수 계산
            scores = []
            for indicator in HEALTH_INDICATORS:
                value = predicted_values.get(indicator)
                ind_score = self._get_indicator_score(value, indicator)
                if ind_score is not None:
                    scores.append(ind_score)

            if len(scores) >= MIN_VALID_INDICATORS:
                max_score = len(scores) * 2
                total_score = round((sum(scores) / max_score) * 100, 1)
                label = self._get_label(total_score)
            else:
                total_score = None
                label = None

            # 다음 분기 계산
            next_period = self._get_next_period(
                latest_row['년도'], latest_row['분기']
            )

            return {
                'period': next_period,
                'score': total_score,
                'label': label,
                'type': 'predicted',
            }

        except Exception as e:
            logger.warning(f"예측 점수 계산 실패: {e}")
            return None
