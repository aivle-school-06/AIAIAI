"""
업종상대 신호등 서비스
======================

기업의 12개 재무지표를 업종 평균 대비 z-score 기준으로
Green/Yellow/Red/Grey 신호등으로 라벨링하는 서비스.
"""
import logging
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

from app.core.data_loader import DataLoader
from app.exceptions import CompanyNotFoundError

logger = logging.getLogger(__name__)


# =============================================================================
# 신호등 지표 설정
# =============================================================================
# 12개 지표 (CFO증감률 제외)
# 한글 컬럼명 -> 영문 API 필드명 매핑 (predict 엔드포인트와 동일)
METRIC_MAPPING = {
    'ROA': 'ROA',
    'ROE': 'ROE',
    '매출액영업이익률': 'OpMargin',
    '부채비율': 'DbRatio',
    '자기자본비율': 'EqRatio',
    '자본잠식률': 'CapImpRatio',
    '단기차입금비율': 'STDebtRatio',
    '유동비율': 'CurRatio',
    '당좌비율': 'QkRatio',
    '유동부채비율': 'CurLibRatio',
    'CFO_자산비율': 'CFO_AsRatio',
    'CFO_매출액비율': 'CFO_Sale',
}

# 데이터 컬럼명 리스트
SIGNAL_METRICS = list(METRIC_MAPPING.keys())

# 지표별 방향 (up: 높을수록 좋음, down: 낮을수록 좋음)
METRIC_DIRECTION = {
    'ROA': 'up',
    'ROE': 'up',
    '매출액영업이익률': 'up',
    '부채비율': 'down',
    '자기자본비율': 'up',
    '자본잠식률': 'down',  # 음수일수록 좋음 (잠식 없음)
    '단기차입금비율': 'down',
    '유동비율': 'up',
    '당좌비율': 'up',
    '유동부채비율': 'down',
    'CFO_자산비율': 'up',
    'CFO_매출액비율': 'up',
}

# 신호등 분포 (Green:Yellow:Red = 4:3:3)
# z-score 기준 백분위
# 높을수록 좋은 지표: 상위 40% = Green, 중위 30% = Yellow, 하위 30% = Red
# 낮을수록 좋은 지표: 하위 40% = Green, 중위 30% = Yellow, 상위 30% = Red
GREEN_PERCENTILE = 0.6   # 상위 40% (percentile >= 0.6)
YELLOW_PERCENTILE = 0.3  # 중위 30% (0.3 <= percentile < 0.6)
# Red: 하위 30% (percentile < 0.3)


class SignalService:
    """
    업종상대 신호등 서비스

    - 12개 지표에 대해 업종상대 z-score 기반 신호등 제공
    - Green (상위 30%), Yellow (중위 30%), Red (하위 40%), Grey (결측)
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def get_signals(
        self, company_code: str, period: str
    ) -> Dict[str, Any]:
        """
        기업의 업종상대 신호등 조회

        Args:
            company_code: 기업 코드 (예: '[005930]')
            period: 분기 (예: '20253')

        Returns:
            Dict: 지표별 신호등 정보
        """
        # period 파싱 (20253 -> 년도: 2025, 분기: 3)
        year = int(period[:4])
        quarter = int(period[4:])

        # 기업 데이터 조회
        df = self.data_loader.df
        company_df = df[
            (df['기업코드'] == company_code) &
            (df['년도'] == year) &
            (df['분기'].astype(str).str.contains(str(quarter)))
        ]

        if company_df.empty:
            raise CompanyNotFoundError(f"{company_code} ({period})")

        row = company_df.iloc[0]
        company_name = row['기업명']
        industry = row['업종명']
        clean_code = company_code.strip('[]')

        # 해당 분기 전체 데이터 (백분위 계산용)
        period_df = df[
            (df['년도'] == year) &
            (df['분기'].astype(str).str.contains(str(quarter)))
        ]

        # 각 지표별 신호등 계산 (영문 필드명으로 응답)
        signals = {}
        for metric_kr, metric_en in METRIC_MAPPING.items():
            signal = self._get_signal(row, metric_kr, period_df)
            signals[metric_en] = signal

        return {
            'company_code': clean_code,
            'company_name': company_name,
            'industry': industry,
            'period': period,
            'signals': signals,
        }

    def _get_signal(
        self,
        row: pd.Series,
        metric: str,
        period_df: pd.DataFrame,
    ) -> str:
        """
        개별 지표 신호등 계산

        Args:
            row: 기업 데이터 행
            metric: 지표명
            period_df: 해당 분기 전체 데이터

        Returns:
            str: 'green', 'yellow', 'red', 'grey'
        """
        # 업종상대 z-score
        z_col = f"{metric}_업종상대"
        z_score = row.get(z_col)

        if pd.isna(z_score):
            return 'grey'

        # 해당 분기 z-score 백분위 계산
        z_values = period_df[z_col].dropna()
        if len(z_values) == 0:
            percentile = 0.5
        else:
            percentile = (z_values < z_score).mean()

        # 지표 방향에 따른 신호등 결정
        direction = METRIC_DIRECTION.get(metric, 'up')
        return self._determine_signal(percentile, direction)

    def _determine_signal(
        self, percentile: float, direction: str
    ) -> str:
        """
        백분위와 방향에 따른 신호등 결정

        Args:
            percentile: z-score 백분위 (0~1)
            direction: 'up' (높을수록 좋음) 또는 'down' (낮을수록 좋음)

        Returns:
            str: 'green', 'yellow', 'red'
        """
        if direction == 'up':
            # 높을수록 좋음: 상위 40% = Green, 중위 30% = Yellow, 하위 30% = Red
            if percentile >= GREEN_PERCENTILE:  # >= 0.6 (상위 40%)
                return 'green'
            elif percentile >= YELLOW_PERCENTILE:  # >= 0.3 (중위 30%)
                return 'yellow'
            else:  # < 0.3 (하위 30%)
                return 'red'
        else:
            # 낮을수록 좋음: 하위 40% = Green, 중위 30% = Yellow, 상위 30% = Red
            if percentile <= (1 - GREEN_PERCENTILE):  # <= 0.4 (하위 40%)
                return 'green'
            elif percentile <= (1 - YELLOW_PERCENTILE):  # <= 0.7 (중위 30%)
                return 'yellow'
            else:  # > 0.7 (상위 30%)
                return 'red'
