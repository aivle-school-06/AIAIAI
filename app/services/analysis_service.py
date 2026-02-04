"""
분석 서비스
==========

예측, 보고서 생성, SHAP 분석 등 분석 관련 비즈니스 로직.
의존성 주입을 통해 DataLoader, Predictor를 받습니다.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.core.data_loader import DataLoader
from app.core.predictor import Predictor
from app.core.constants import ALL_TARGETS, TARGET_METRICS
from app.exceptions import CompanyNotFoundError

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    분석 서비스

    예측, SHAP 분석, 보고서 생성 등을 담당합니다.
    DataLoader와 Predictor는 의존성 주입으로 받습니다.
    """

    def __init__(self, data_loader: DataLoader, predictor: Predictor):
        self.data_loader = data_loader
        self.predictor = predictor

    def get_company_analysis(self, company_code: str) -> Dict[str, Any]:
        """
        기업 종합 분석

        Args:
            company_code: 기업 코드

        Returns:
            Dict: 종합 분석 결과
        """
        # 기업 데이터 조회
        company_data = self.data_loader.get_company_data(company_code)

        # 예측 수행
        prediction_result = self.predictor.predict(company_code)

        # 업종 비교
        industry_comparison = self.data_loader.get_industry_comparison(company_code)

        return {
            'company_code': company_code,
            'company_name': company_data['meta']['기업명'],
            'industry': company_data['meta']['업종명'],
            'base_period': company_data['meta']['기준일'],
            'current_metrics': company_data['current'],
            'predictions': prediction_result['predictions'],
            'summary': prediction_result['summary'],
            'risk_signals': prediction_result['risk_signals'],
            'industry_comparison': industry_comparison,
            'generated_at': datetime.now().isoformat(),
        }

    # 지표명 매핑 (한글 -> 영문 약어)
    METRIC_NAME_MAP = {
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
        'CFO증감률': 'CFO_GR',
    }

    def get_prediction(self, company_code: str) -> Dict[str, Any]:
        """
        예측값만 간소화하여 반환 (13개 지표)

        Args:
            company_code: 기업 코드

        Returns:
            Dict: 예측값 (지표명: 예측값)
        """
        full_result = self.predictor.predict(company_code)

        # 예측값만 추출 (SHAP, 변화량 등 제외) + 지표명 영문 변환
        predictions_simple = {}
        for metric, data in full_result['predictions'].items():
            # 지표명을 영문으로 변환
            eng_metric = self.METRIC_NAME_MAP.get(metric, metric)
            if isinstance(data, dict) and 'predicted' in data:
                predictions_simple[eng_metric] = data['predicted']
            elif isinstance(data, dict) and 'error' in data:
                predictions_simple[eng_metric] = None

        # company_code에서 대괄호 제거 (예: "[000020]" -> "000020")
        raw_code = full_result['company_code']
        clean_code = raw_code.strip('[]')

        # base_period 변환 (예: "2025년 Q3" -> 20253)
        raw_period = full_result['base_period']
        # "2025년 Q3" -> "2025", "3" -> 20253
        import re
        match = re.match(r'(\d{4})년\s*Q(\d)', raw_period)
        if match:
            base_period_int = int(match.group(1)) * 10 + int(match.group(2))
        else:
            base_period_int = raw_period  # 파싱 실패 시 원본 유지

        return {
            'company_code': clean_code,
            'company_name': full_result['company_name'],
            'base_period': base_period_int,
            'predictions': predictions_simple,
        }

    def get_metric_prediction(
        self, company_code: str, metric: str
    ) -> Dict[str, Any]:
        """
        특정 지표 예측 결과

        Args:
            company_code: 기업 코드
            metric: 지표명

        Returns:
            Dict: 해당 지표의 예측 결과
        """
        if metric not in ALL_TARGETS:
            raise ValueError(f"Invalid metric: {metric}")

        prediction_result = self.predictor.predict(company_code)
        metric_result = prediction_result['predictions'].get(metric)

        if not metric_result:
            raise ValueError(f"Prediction not available for {metric}")

        return {
            'company_code': company_code,
            'company_name': prediction_result['company_name'],
            'metric': metric,
            'prediction': metric_result,
        }

    def get_shap_analysis(
        self, company_code: str, metric: str
    ) -> Dict[str, Any]:
        """
        SHAP 분석 결과

        Args:
            company_code: 기업 코드
            metric: 지표명

        Returns:
            Dict: SHAP 분석 결과
        """
        company_data = self.data_loader.get_company_data(company_code)
        shap_result = self.predictor.get_shap_analysis(company_code, metric)

        return {
            'company_code': company_code,
            'company_name': company_data['meta']['기업명'],
            'metric': metric,
            'shap_analysis': shap_result,
        }

    def get_company_list(
        self,
        market: Optional[str] = None,
        industry: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        기업 목록 조회

        Args:
            market: 시장구분 필터 (KOSPI/KOSDAQ)
            industry: 업종 필터
            search: 검색어 (기업명/코드)
            limit: 페이지 크기
            offset: 시작 위치

        Returns:
            Dict: 기업 목록 및 페이지네이션 정보
        """
        df = self.data_loader.get_company_list()

        # 필터 적용
        if market:
            df = df[df['시장구분'] == market]

        if industry:
            df = df[df['업종명'].str.contains(industry, na=False)]

        if search:
            mask = (
                df['기업명'].str.contains(search, na=False) |
                df['기업코드'].str.contains(search, na=False)
            )
            df = df[mask]

        total = len(df)

        # 페이지네이션
        df = df.iloc[offset:offset + limit]

        return {
            'companies': df.to_dict('records'),
            'total': total,
            'limit': limit,
            'offset': offset,
        }

    def get_industry_list(self) -> List[str]:
        """업종 목록 반환"""
        df = self.data_loader.get_company_list()
        return sorted(df['업종명'].unique().tolist())

    def batch_predict(
        self, company_codes: List[str]
    ) -> Dict[str, Any]:
        """
        일괄 예측

        Args:
            company_codes: 기업 코드 리스트

        Returns:
            Dict: 기업별 예측 결과
        """
        results = {}
        errors = []

        for code in company_codes:
            try:
                results[code] = self.predictor.predict(code)
            except CompanyNotFoundError:
                errors.append({'code': code, 'error': 'Company not found'})
            except Exception as e:
                errors.append({'code': code, 'error': str(e)})

        return {
            'results': results,
            'success_count': len(results),
            'error_count': len(errors),
            'errors': errors,
        }

    def get_historical_data(self, company_code: str) -> Dict[str, Any]:
        """
        과거 데이터 조회

        Args:
            company_code: 기업 코드

        Returns:
            Dict: 과거 분기 데이터
        """
        company_data = self.data_loader.get_company_data(company_code)
        return {
            'company_code': company_code,
            'company_name': company_data['meta']['기업명'],
            'historical': company_data['historical'],
            'trend': company_data['trend'],
        }
