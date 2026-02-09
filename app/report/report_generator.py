"""
보고서 생성 메인 모듈
====================

이 모듈은 기업 재무 분석 보고서를 생성하는 메인 클래스입니다.
다른 모듈들(data_loader, grade_calculator, ai_predictor)을 조합하여
완성된 보고서 데이터를 생성합니다.

보고서 구성 (6개 섹션):
1. Summary (Executive Summary): 핵심 지표, 종합 등급, 강점/약점
2. Company Info (기업 개요): 기본 정보, 규모, 최근 실적
3. Financial Analysis (재무 분석): 13개 지표 상세 분석
4. Industry Comparison (업종 비교): 업종 내 위치, 상대 지표
5. Trend Analysis (추세 분석): YoY, MOM, 시계열 데이터
6. AI Prediction (AI 예측): XGBoost 예측 + SHAP 분석

추가 데이터:
- opinion_data: LLM 종합 의견 생성용 구조화된 데이터
- appendix: 재무제표 원본 데이터

사용 예시:
    from report import generate_report
    report = generate_report('[005930]')
    print(report['sections']['summary']['overall_grade'])  # 예: 'B+'
"""
from typing import Dict, Optional
from datetime import datetime
from .config import TARGET_METRICS, METRIC_FORMAT, METRIC_DESCRIPTION
from .data_loader import get_data_loader
from .grade_calculator import get_grade_calculator
from .ai_predictor import get_predictor


class ReportGenerator:
    """
    기업 재무 분석 보고서 생성 클래스

    이 클래스는 여러 모듈의 기능을 조합하여 완성된 보고서를 생성합니다:
    - DataLoader: 기업 데이터 조회
    - GradeCalculator: 등급 계산
    - AIPredictor: AI 예측

    싱글톤 패턴으로 사용됩니다.
    get_report_generator() 함수를 통해 인스턴스를 가져옵니다.
    """

    def __init__(self):
        """
        초기화

        필요한 모든 모듈의 싱글톤 인스턴스를 가져옵니다.
        (각 모듈이 내부적으로 데이터/모델을 로드)
        """
        self.data_loader = get_data_loader()           # 데이터 로더
        self.grade_calculator = get_grade_calculator() # 등급 계산기
        self.predictor = get_predictor()               # AI 예측기

    def generate(self, company_code: str) -> Dict:
        """
        전체 보고서 생성 (메인 함수)

        특정 기업에 대한 완전한 분석 보고서를 생성합니다.

        처리 흐름:
        1. 기업 데이터 조회 (data_loader)
        2. 등급 계산 (grade_calculator)
        3. AI 예측 수행 (predictor)
        4. 6개 섹션 생성
        5. LLM용 데이터 준비
        6. 부록 생성

        Args:
            company_code: 기업 코드 (예: '[005930]')

        Returns:
            Dict: 전체 보고서 데이터
                {
                    'meta': {                           # 기업 메타 정보
                        '기업코드': '[005930]',
                        '기업명': '삼성전자',
                        '시장구분': 'KOSPI',
                        '업종명': '전자부품',
                        ...
                    },
                    'sections': {                       # 6개 섹션
                        'summary': {...},               # Executive Summary
                        'company_info': {...},          # 기업 개요
                        'financial_analysis': {...},    # 재무 분석
                        'industry_comparison': {...},   # 업종 비교
                        'trend_analysis': {...},        # 추세 분석
                        'ai_prediction': {...},         # AI 예측
                    },
                    'opinion_data': {...},              # LLM 종합 의견용
                    'appendix': {...},                  # 재무제표 원본
                    'generated_at': '2025-03-15T14:30:00',  # 생성 시각
                }

            오류 시: {'error': '오류 메시지'}
        """
        # =================================================================
        # 1단계: 기업 데이터 조회
        # =================================================================
        company_data = self.data_loader.get_company_data(company_code)

        if not company_data:
            return {'error': f'기업 코드 {company_code}를 찾을 수 없습니다.'}

        # =================================================================
        # 2단계: 등급 계산
        # =================================================================
        grades = self.grade_calculator.calculate_all_grades(company_code)

        # =================================================================
        # 3단계: AI 예측
        # =================================================================
        predictions = self.predictor.predict(company_code)

        # =================================================================
        # 4단계: 각 섹션 생성
        # =================================================================
        sections = {
            # [1] Executive Summary - 핵심 요약
            'summary': self._generate_summary(company_data, grades, predictions),

            # [2] 기업 개요 - 기본 정보, 규모, 최근 실적
            'company_info': self._generate_company_info(company_data),

            # [3] 재무 분석 - 13개 지표 상세
            'financial_analysis': self._generate_financial_analysis(company_data, grades),

            # [4] 업종 비교 - 업종 내 위치, 상대 지표
            'industry_comparison': self._generate_industry_comparison(company_data, grades),

            # [5] 추세 분석 - YoY, MOM, 시계열
            'trend_analysis': self._generate_trend_analysis(company_data),

            # [6] AI 예측 - XGBoost + SHAP (핵심 차별점)
            'ai_prediction': self._generate_ai_prediction(company_data, predictions),
        }

        # =================================================================
        # 5단계: LLM 종합 의견용 데이터 준비
        # =================================================================
        # 구조화된 데이터로 LLM이 이해하기 쉬운 형태로 변환
        opinion_data = self._prepare_opinion_data(company_data, grades, predictions)

        # =================================================================
        # 6단계: 최종 보고서 반환
        # =================================================================
        return {
            'meta': company_data['meta'],
            'sections': sections,
            'opinion_data': opinion_data,  # LLM 입력용
            'appendix': self._generate_appendix(company_data),  # 재무제표 원본
            'generated_at': datetime.now().isoformat(),
        }

    # =========================================================================
    # 섹션 1: Executive Summary
    # =========================================================================
    def _generate_summary(self, company_data: Dict, grades: Dict, predictions: Dict) -> Dict:
        """
        [1] Executive Summary 섹션 생성

        보고서의 첫 페이지에 표시될 핵심 요약 정보

        포함 내용:
        - 종합 등급 및 점수
        - 주요 6개 지표 스냅샷 (ROA, ROE, 매출액영업이익률, 부채비율, 유동비율, CFO_자산비율)
        - 강점/약점 Top 3
        - 리스크 시그널
        - AI 예측 전망

        Args:
            company_data: 기업 데이터 (data_loader에서 조회)
            grades: 등급 정보 (grade_calculator에서 계산)
            predictions: AI 예측 결과 (predictor에서 예측)

        Returns:
            Dict: Summary 섹션 데이터
                {
                    'overall_grade': 'B+',
                    'overall_score': 3.85,
                    'snapshot': {                       # 주요 6개 지표
                        'ROA': {'value': 2.33, 'formatted': '2.33%', 'grade': 'A'},
                        ...
                    },
                    'strengths': [...],                 # 상위 3개 강점
                    'weaknesses': [...],                # 상위 3개 약점
                    'risk_signals': [...],              # 리스크 시그널
                    'prediction_outlook': 'mixed',      # AI 예측 전망
                }
        """
        # 등급 요약 가져오기
        grade_summary = self.grade_calculator.get_grade_summary(grades)

        # 주요 6개 지표 스냅샷
        # 가장 중요하다고 판단되는 대표 지표들
        key_metrics = ['ROA', 'ROE', '매출액영업이익률', '부채비율', '유동비율', 'CFO_자산비율']
        snapshot = {}

        for metric in key_metrics:
            # 해당 지표가 속한 카테고리 찾기
            for category, metrics in TARGET_METRICS.items():
                if metric in metrics:
                    value = company_data['current'][category][metric]['value']
                    snapshot[metric] = {
                        'value': value,
                        'formatted': self._format_value(metric, value),
                        'grade': grades['metrics'][metric]['grade'],
                    }
                    break

        # 리스크 시그널 (AI 예측에서 추출)
        risk_signals = predictions.get('risk_signals', [])

        return {
            'overall_grade': grade_summary['overall_grade'],
            'overall_score': grade_summary['overall_score'],
            'snapshot': snapshot,
            'strengths': grade_summary['strengths'][:3],    # 상위 3개만
            'weaknesses': grade_summary['weaknesses'][:3],  # 상위 3개만
            'risk_signals': risk_signals,
            'prediction_outlook': predictions.get('summary', {}).get('overall_outlook'),
        }

    # =========================================================================
    # 섹션 2: 기업 개요
    # =========================================================================
    def _generate_company_info(self, company_data: Dict) -> Dict:
        """
        [2] 기업 개요 섹션 생성

        기업의 기본 정보와 규모, 최근 실적 데이터

        포함 내용:
        - 기본 정보: 기업명, 시장구분, 업종 등
        - 규모 정보: 자산총계, 매출액, 자본총계, 영업이익, 당기순이익
        - 최근 실적: 최근 4분기 매출/이익 추이

        Args:
            company_data: 기업 데이터

        Returns:
            Dict: 기업 개요 섹션 데이터
                {
                    'basic_info': {...},                # 메타 정보
                    'scale': {                          # 규모 정보 (원 단위)
                        '자산총계': 12345678900000,
                        '매출액': 98765432100000,
                        ...
                    },
                    'recent_performance': {             # 최근 4분기 실적
                        'periods': ['2024 Q4', '2025 Q1', '2025 Q2', '2025 Q3'],
                        '매출액': [...],
                        '영업이익': [...],
                        '당기순이익': [...],
                    },
                }
        """
        meta = company_data['meta']
        raw = company_data['raw_data']

        # 규모 정보 (재무제표 주요 항목)
        # 접두사 BS_, PL_은 각각 재무상태표(Balance Sheet), 손익계산서(Profit & Loss)
        scale = {
            '자산총계': self._safe_get_raw(raw, 'BS_자산총계'),
            '매출액': self._safe_get_raw(raw, 'PL_매출액'),
            '자본총계': self._safe_get_raw(raw, 'BS_자본총계'),
            '영업이익': self._safe_get_raw(raw, 'PL_영업이익'),
            '당기순이익': self._safe_get_raw(raw, 'PL_당기순이익'),
        }

        # 최근 4분기 실적 추이 (시계열 차트용)
        historical = company_data['historical']
        recent_performance = {
            'periods': historical['periods'],
            # 해당 컬럼이 있는 경우에만 데이터 포함
            '매출액': historical['metrics'].get('PL_매출액', []) if 'PL_매출액' in company_data['raw_data'].index else [],
            '영업이익': historical['metrics'].get('PL_영업이익', []) if 'PL_영업이익' in company_data['raw_data'].index else [],
            '당기순이익': historical['metrics'].get('PL_당기순이익', []) if 'PL_당기순이익' in company_data['raw_data'].index else [],
        }

        return {
            'basic_info': meta,
            'scale': scale,
            'recent_performance': recent_performance,
        }

    # =========================================================================
    # 섹션 3: 재무 분석
    # =========================================================================
    def _generate_financial_analysis(self, company_data: Dict, grades: Dict) -> Dict:
        """
        [3] 재무 상태 분석 섹션 생성

        13개 지표에 대한 상세 분석 (카테고리별 구성)

        포함 내용 (카테고리별):
        - 카테고리 등급 및 점수
        - 각 지표의 현재값, 등급, 백분위, 설명

        5개 카테고리:
        - 수익성: ROA, ROE, 매출액영업이익률
        - 안정성: 부채비율, 자기자본비율, 자본잠식률
        - 차입금: 단기차입금비율
        - 유동성: 유동비율, 당좌비율, 유동부채비율
        - 현금흐름: CFO_자산비율, CFO_매출액비율, CFO증감률

        Args:
            company_data: 기업 데이터
            grades: 등급 정보

        Returns:
            Dict: 카테고리별 재무 분석 데이터
                {
                    '수익성': {
                        'grade': 'A',
                        'score': 4.67,
                        'metrics': {
                            'ROA': {
                                'value': 2.33,
                                'formatted': '2.33%',
                                'is_missing': False,
                                'grade': 'A',
                                'percentile': 18.5,
                                'description': '총자산이익률 - ...',
                            },
                            ...
                        },
                    },
                    ...
                }
        """
        result = {}

        # 5개 카테고리별로 처리
        for category, metrics in TARGET_METRICS.items():
            category_data = {
                'grade': grades['categories'][category]['grade'],
                'score': grades['categories'][category]['score'],
                'metrics': {},
            }

            # 카테고리에 속한 각 지표 처리
            for metric in metrics:
                # 현재값 및 결측 여부
                value = company_data['current'][category][metric]['value']
                is_missing = company_data['current'][category][metric]['is_missing']

                # 등급 정보
                grade_info = grades['metrics'][metric]

                category_data['metrics'][metric] = {
                    'value': value,
                    'formatted': self._format_value(metric, value),
                    'is_missing': is_missing,
                    'grade': grade_info['grade'],
                    'percentile': grade_info['percentile'],
                    'description': METRIC_DESCRIPTION.get(metric),  # 지표 설명
                }

            result[category] = category_data

        return result

    # =========================================================================
    # 섹션 4: 업종 비교
    # =========================================================================
    def _generate_industry_comparison(self, company_data: Dict, grades: Dict) -> Dict:
        """
        [4] 업종 비교 분석 섹션 생성

        동일 업종 내에서 해당 기업의 상대적 위치 분석

        포함 내용:
        - 업종 정보
        - 카테고리별 업종 내 평균 백분위
        - 업종 대비 상대 지표값
        - 업종 내 강점/약점

        Args:
            company_data: 기업 데이터
            grades: 등급 정보

        Returns:
            Dict: 업종 비교 데이터
                {
                    'industry': '전자부품',
                    'industry_code': 'E01',
                    'category_position': {              # 카테고리별 평균 백분위
                        '수익성': 22.5,                 # 상위 22.5%
                        '안정성': 45.3,
                        ...
                    },
                    'relative_values': {                # 업종 평균 대비 차이
                        'ROA': 1.5,                     # 업종 평균보다 1.5%p 높음
                        ...
                    },
                    'strengths': [...],                 # 업종 내 강점 (상위 25%)
                    'weaknesses': [...],                # 업종 내 약점 (하위 25%)
                }
        """
        meta = company_data['meta']
        relative = company_data['relative']  # _업종상대 컬럼들

        # 업종 통계 조회 (현재 미사용, 추후 활용 가능)
        industry_stats = self.data_loader.get_industry_stats(
            meta['업종코드'], meta['년도'], meta['분기']
        )

        # 카테고리별 업종 내 평균 백분위 계산
        # 해당 카테고리의 지표들 백분위 평균 = 카테고리 업종 내 위치
        category_position = {}
        for category, metrics in TARGET_METRICS.items():
            positions = []
            for metric in metrics:
                percentile = grades['metrics'][metric].get('percentile')
                if percentile is not None:
                    positions.append(percentile)

            if positions:
                avg_percentile = sum(positions) / len(positions)
            else:
                avg_percentile = None

            category_position[category] = avg_percentile

        # 업종 내 강점/약점 분석
        # 강점: 상위 25% 이내 (percentile ≤ 25)
        # 약점: 하위 25% 이내 (percentile ≥ 75)
        strengths = []
        weaknesses = []

        for metric, rel_value in relative.items():
            percentile = grades['metrics'].get(metric, {}).get('percentile')
            if percentile is None:
                continue

            if percentile <= 25:  # 강점
                strengths.append({
                    'metric': metric,
                    'percentile': percentile,
                    'relative': rel_value,  # 업종 평균 대비 차이
                })
            elif percentile >= 75:  # 약점
                weaknesses.append({
                    'metric': metric,
                    'percentile': percentile,
                    'relative': rel_value,
                })

        # 정렬: 강점은 백분위 낮은 순, 약점은 높은 순
        strengths.sort(key=lambda x: x['percentile'])
        weaknesses.sort(key=lambda x: x['percentile'], reverse=True)

        return {
            'industry': meta['업종명'],
            'industry_code': meta['업종코드'],
            'category_position': category_position,
            'relative_values': relative,
            'strengths': strengths[:5],
            'weaknesses': weaknesses[:5],
        }

    # =========================================================================
    # 섹션 5: 추세 분석
    # =========================================================================
    def _generate_trend_analysis(self, company_data: Dict) -> Dict:
        """
        [5] 추세 분석 섹션 생성

        각 지표의 시계열 변화 추이 분석

        포함 내용:
        - 각 지표의 YoY (전년동기대비), MOM (전분기대비) 변화율
        - 이동평균(MA4), 변동성(STD4)
        - 추세 방향 (improving/declining/mixed/unknown)
        - 최근 4분기 히스토리

        Args:
            company_data: 기업 데이터

        Returns:
            Dict: 추세 분석 데이터
                {
                    'periods': ['2024 Q4', '2025 Q1', '2025 Q2', '2025 Q3'],
                    'metrics': {
                        'ROA': {
                            'yoy': 15.5,                # 전년동기비 +15.5%
                            'mom': -2.3,                # 전분기비 -2.3%
                            'ma4': 2.1,                 # 4분기 이동평균
                            'std4': 0.5,                # 4분기 표준편차
                            'direction': 'mixed',       # 추세 방향
                            'historical': [1.5, 1.8, 2.1, 2.3],  # 4분기 값
                        },
                        ...
                    },
                }
        """
        trend = company_data['trend']
        historical = company_data['historical']

        result = {}

        # 모든 예측 대상 지표에 대해 처리
        for metric in TARGET_METRICS.keys():
            for m in TARGET_METRICS[metric]:
                metric_trend = trend.get(m, {})

                # 추세 방향 판단 로직
                # YoY와 MOM이 모두 양수 → improving (개선 중)
                # YoY와 MOM이 모두 음수 → declining (악화 중)
                # 그 외 → mixed (혼조) 또는 unknown (데이터 부족)
                yoy = metric_trend.get('yoy')
                mom = metric_trend.get('mom')

                if yoy is not None and mom is not None:
                    if yoy > 0 and mom > 0:
                        direction = 'improving'  # 단기/장기 모두 상승
                    elif yoy < 0 and mom < 0:
                        direction = 'declining'  # 단기/장기 모두 하락
                    else:
                        direction = 'mixed'      # 단기/장기 방향 상이
                else:
                    direction = 'unknown'        # 데이터 부족

                result[m] = {
                    'yoy': metric_trend.get('yoy'),      # 전년동기비 (%)
                    'mom': metric_trend.get('mom'),      # 전분기비 (%)
                    'ma4': metric_trend.get('ma4'),      # 4분기 이동평균
                    'std4': metric_trend.get('std4'),    # 4분기 변동성
                    'direction': direction,              # 추세 방향
                    'historical': historical['metrics'].get(m, []),  # 시계열 데이터
                }

        return {
            'periods': historical['periods'],
            'metrics': result,
        }

    # =========================================================================
    # 섹션 6: AI 예측 (핵심 차별점)
    # =========================================================================
    def _generate_ai_prediction(self, company_data: Dict, predictions: Dict) -> Dict:
        """
        [6] AI 예측 섹션 생성 (핵심 섹션)

        XGBoost 모델 예측 결과 + SHAP 분석 기반 설명

        이 섹션이 보고서의 핵심 차별점입니다:
        - 다음 분기 13개 지표 예측값
        - 예측 변화량 및 방향
        - SHAP 기반 예측 근거 (상승/하락 요인)
        - 모델 신뢰도

        Args:
            company_data: 기업 데이터
            predictions: AI 예측 결과

        Returns:
            Dict: AI 예측 데이터
                {
                    'base_period': '2025년 Q3',         # 현재 기준 분기
                    'by_category': {                    # 카테고리별 예측
                        '수익성': {
                            'ROA': {
                                'current': 2.33,
                                'predicted': 1.40,
                                'change': -0.93,
                                'change_pct': -39.9,
                                'direction': 'down',
                                'confidence': 'medium',
                                'r2': 0.4034,
                                'top_positive_factors': [...],  # 상승 요인 Top 5
                                'top_negative_factors': [...],  # 하락 요인 Top 5
                            },
                            ...
                        },
                        ...
                    },
                    'summary': {...},                   # 예측 요약
                    'risk_signals': [...],              # 리스크 시그널
                    'all_predictions': {...},           # 전체 예측 (flat)
                }

            오류 시: {'error': '오류 메시지'}
        """
        # 예측 오류 시 그대로 반환
        if 'error' in predictions:
            return predictions

        pred_data = predictions.get('predictions', {})
        summary = predictions.get('summary', {})
        risk_signals = predictions.get('risk_signals', [])

        # 카테고리별로 예측 데이터 정리
        by_category = {}

        for category, metrics in TARGET_METRICS.items():
            cat_predictions = {}

            for metric in metrics:
                if metric in pred_data:
                    p = pred_data[metric]

                    cat_predictions[metric] = {
                        # 현재값과 예측값
                        'current': p.get('current'),
                        'predicted': p.get('predicted'),

                        # 변화량
                        'change': p.get('change'),          # 절대 변화량
                        'change_pct': p.get('change_pct'),  # 변화율 (%)
                        'direction': p.get('direction'),    # 방향 (up/down/stable)

                        # 신뢰도 정보
                        'confidence': p.get('confidence'),  # high/medium/low
                        'r2': p.get('r2'),                  # 모델 R² 값

                        # SHAP 분석 결과 (예측 근거)
                        # 상위 5개 요인만 포함 (너무 많으면 혼란)
                        'top_positive_factors': p.get('shap_analysis', {}).get('positive_factors', [])[:5],
                        'top_negative_factors': p.get('shap_analysis', {}).get('negative_factors', [])[:5],
                    }

            by_category[category] = cat_predictions

        return {
            'base_period': predictions.get('base_period'),
            'by_category': by_category,
            'summary': summary,
            'risk_signals': risk_signals,
            'all_predictions': pred_data,  # flat 구조로도 제공
        }

    # =========================================================================
    # LLM 종합 의견용 데이터 준비
    # =========================================================================
    def _prepare_opinion_data(self, company_data: Dict, grades: Dict, predictions: Dict) -> Dict:
        """
        LLM 종합 의견 생성용 데이터 준비

        LLM이 이해하기 쉬운 구조화된 데이터로 변환합니다.
        이 데이터를 LLM에 전달하면 종합 의견을 생성할 수 있습니다.

        포함 내용:
        - 기업 기본 정보
        - 등급 요약
        - 현재 지표값
        - 업종 대비 상대값
        - 추세 요약
        - AI 예측 요약
        - 리스크 시그널

        Args:
            company_data: 기업 데이터
            grades: 등급 정보
            predictions: AI 예측 결과

        Returns:
            Dict: LLM 입력용 구조화된 데이터
        """
        meta = company_data['meta']
        grade_summary = self.grade_calculator.get_grade_summary(grades)
        pred_summary = predictions.get('summary', {})

        return {
            # 기업 기본 정보
            'company': {
                'name': meta['기업명'],
                'industry': meta['업종명'],
                'market': meta['시장구분'],
                'period': meta['기준일'],
            },

            # 등급 요약
            'grades': {
                'overall': grade_summary['overall_grade'],
                'score': grade_summary['overall_score'],
                'strengths': [s['metric'] for s in grade_summary['strengths']],
                'weaknesses': [w['metric'] for w in grade_summary['weaknesses']],
            },

            # 현재 지표값 (flat 구조)
            'current_metrics': self._flatten_current(company_data['current']),

            # 업종 대비 상대값
            'relative_metrics': company_data['relative'],

            # 추세 요약 (YoY, MOM만)
            'trend': self._summarize_trend(company_data['trend']),

            # AI 예측 요약
            'prediction': {
                'outlook': pred_summary.get('overall_outlook'),
                'improving': pred_summary.get('improving_metrics', []),
                'declining': pred_summary.get('declining_metrics', []),
                'category_outlook': pred_summary.get('category_outlook', {}),
            },

            # 리스크 시그널
            'risk_signals': predictions.get('risk_signals', []),

            # 업종 내 위치 (신규)
            'industry_position': self._get_industry_position(grades),

            # 시계열 분석 요약 (신규)
            'time_analysis': self._get_time_analysis(company_data, grades),
        }

    def _flatten_current(self, current: Dict) -> Dict:
        """
        현재값 평탄화 (내부 메서드)

        중첩된 카테고리 구조를 flat 딕셔너리로 변환

        Args:
            current: 카테고리별 지표값 (중첩 구조)

        Returns:
            Dict: {지표명: 값} 형태의 flat 딕셔너리
        """
        result = {}
        for category, metrics in current.items():
            for metric, data in metrics.items():
                result[metric] = data['value']
        return result

    def _summarize_trend(self, trend: Dict) -> Dict:
        """
        추세 요약 (내부 메서드)

        전체 추세 데이터에서 YoY, MOM만 추출

        Args:
            trend: 전체 추세 데이터

        Returns:
            Dict: {지표명: {yoy, mom}} 형태의 요약 데이터
        """
        result = {}
        for metric, data in trend.items():
            result[metric] = {
                'yoy': data.get('yoy'),
                'mom': data.get('mom'),
            }
        return result

    def _get_industry_position(self, grades: Dict) -> Dict:
        """
        업종 내 위치 정보 추출 (내부 메서드)

        등급 정보에서 업종 내 백분위를 추출하여
        종합 및 카테고리별 업종 내 위치 정보 생성

        Args:
            grades: 등급 정보 딕셔너리

        Returns:
            Dict: 업종 내 위치 정보
                - overall_percentile: 종합 백분위
                - category_percentiles: 카테고리별 백분위
        """
        # 종합 백분위: 모든 지표 백분위의 평균
        all_percentiles = []
        for metric, info in grades.get('metrics', {}).items():
            pct = info.get('percentile')
            if pct is not None:
                all_percentiles.append(pct)

        overall_pct = sum(all_percentiles) / len(all_percentiles) if all_percentiles else None

        # 카테고리별 백분위
        category_pcts = {}
        for category, cat_info in grades.get('categories', {}).items():
            cat_metrics = cat_info.get('metrics', {})
            cat_percentiles = []
            for metric, metric_info in cat_metrics.items():
                pct = metric_info.get('percentile')
                if pct is not None:
                    cat_percentiles.append(pct)

            if cat_percentiles:
                category_pcts[category] = sum(cat_percentiles) / len(cat_percentiles)
            else:
                category_pcts[category] = None

        return {
            'overall_percentile': overall_pct,
            'category_percentiles': category_pcts,
        }

    def _get_time_analysis(self, company_data: Dict, grades: Dict) -> Dict:
        """
        시계열 분석 요약 (내부 메서드)

        최근 4분기 데이터를 분석하여 주요 변화 추이 요약

        Args:
            company_data: 기업 데이터
            grades: 등급 정보

        Returns:
            Dict: 시계열 분석 요약
                - summary: 전체 추세 요약 문장
                - key_changes: 주요 변화 리스트
                - improving_categories: 개선 추세 카테고리
                - declining_categories: 악화 추세 카테고리
        """
        trend = company_data.get('trend', {})

        # 카테고리별 추세 분석
        from .config import TARGET_METRICS

        improving_categories = []
        declining_categories = []
        key_changes = []

        for category, metrics in TARGET_METRICS.items():
            improving_count = 0
            declining_count = 0

            for metric in metrics:
                t = trend.get(metric, {})
                yoy = t.get('yoy')
                mom = t.get('mom')

                if yoy is not None and mom is not None:
                    # 방향성 고려
                    from .config import METRIC_DIRECTION
                    direction = METRIC_DIRECTION.get(metric, 'higher')

                    if direction == 'higher':
                        # 높을수록 좋은 지표
                        is_improving = yoy > 0 and mom > 0
                        is_declining = yoy < 0 and mom < 0
                    else:
                        # 낮을수록 좋은 지표
                        is_improving = yoy < 0 and mom < 0
                        is_declining = yoy > 0 and mom > 0

                    if is_improving:
                        improving_count += 1
                    elif is_declining:
                        declining_count += 1
                        # 주요 악화 지표 기록
                        if abs(yoy) > 5:  # 5% 이상 변화
                            key_changes.append(f"{metric} 악화 (YoY {yoy:+.1f}%)")

            # 카테고리 분류
            if improving_count > declining_count:
                improving_categories.append(category)
            elif declining_count > improving_count:
                declining_categories.append(category)

        # 요약 문장 생성
        if len(improving_categories) > len(declining_categories):
            summary = f"전반적으로 개선 추세 ({', '.join(improving_categories[:2])} 등)"
        elif len(declining_categories) > len(improving_categories):
            summary = f"전반적으로 악화 추세 ({', '.join(declining_categories[:2])} 등 주의)"
        else:
            summary = "전반적으로 혼조세 (개선/악화 혼재)"

        return {
            'summary': summary,
            'key_changes': key_changes[:3],  # 상위 3개만
            'improving_categories': improving_categories,
            'declining_categories': declining_categories,
        }

    # =========================================================================
    # 부록: 재무제표 원본
    # =========================================================================
    def _generate_appendix(self, company_data: Dict) -> Dict:
        """
        [부록] 재무제표 원본 데이터

        상세 분석이 필요한 경우를 위해 원본 재무제표 데이터 제공

        포함 내용:
        - 재무상태표 (BS_*) 항목들
        - 손익계산서 (PL_*) 항목들
        - 현금흐름표 (CF_*) 항목들
        - 사용된 피처 수

        Args:
            company_data: 기업 데이터

        Returns:
            Dict: 재무제표 원본 데이터
        """
        raw = company_data['raw_data']

        # 재무제표 항목들 추출 (_lag 접미사가 붙은 파생 피처는 제외)
        # BS_: Balance Sheet (재무상태표)
        bs_items = {col: self._safe_get_raw(raw, col)
                    for col in raw.index if col.startswith('BS_') and '_lag' not in col}

        # PL_: Profit & Loss (손익계산서)
        pl_items = {col: self._safe_get_raw(raw, col)
                    for col in raw.index if col.startswith('PL_') and '_lag' not in col}

        # CF_: Cash Flow (현금흐름표)
        cf_items = {col: self._safe_get_raw(raw, col)
                    for col in raw.index if col.startswith('CF_') and '_lag' not in col}

        return {
            'balance_sheet': bs_items,      # 재무상태표
            'income_statement': pl_items,   # 손익계산서
            'cash_flow': cf_items,          # 현금흐름표
            'feature_count': len(self.data_loader.feature_cols),  # 376개
        }

    # =========================================================================
    # 유틸리티 메서드
    # =========================================================================
    def _format_value(self, metric: str, value: Optional[float]) -> str:
        """
        값 포맷팅 (내부 메서드)

        지표에 맞는 형식으로 값을 문자열로 변환
        (모든 지표가 % 단위이므로 소수점 2자리까지 표시)

        Args:
            metric: 지표명
            value: 지표값

        Returns:
            str: 포맷팅된 문자열 (예: '2.33%', 'N/A')
        """
        if value is None:
            return 'N/A'
        fmt = METRIC_FORMAT.get(metric, '{:.2f}')
        return fmt.format(value)

    def _safe_get_raw(self, raw, col: str):
        """
        안전하게 raw 데이터 추출 (내부 메서드)

        컬럼이 없거나 값이 NaN인 경우 None 반환

        Args:
            raw: 원본 데이터 (Series)
            col: 컬럼명

        Returns:
            float 또는 None
        """
        if col in raw.index:
            val = raw[col]
            return float(val) if pd.notna(val) else None
        return None


# pandas import (파일 하단에서 사용)
import pandas as pd

# =============================================================================
# 싱글톤 패턴
# =============================================================================
# 전역 인스턴스 (처음에는 None)
_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """
    ReportGenerator 싱글톤 인스턴스 반환

    처음 호출 시 인스턴스 생성, 이후에는 기존 인스턴스 재사용

    Returns:
        ReportGenerator: 보고서 생성기 인스턴스

    사용 예시:
        generator = get_report_generator()
        report = generator.generate('[005930]')
    """
    global _generator

    if _generator is None:
        _generator = ReportGenerator()

    return _generator


def generate_report(company_code: str) -> Dict:
    """
    보고서 생성 편의 함수

    ReportGenerator 인스턴스를 가져와서 보고서 생성을 수행하는
    간편한 래퍼 함수입니다.

    Args:
        company_code: 기업 코드 (예: '[005930]')

    Returns:
        Dict: 전체 보고서 데이터

    사용 예시:
        from report import generate_report
        report = generate_report('[005930]')
        print(report['sections']['summary']['overall_grade'])
    """
    generator = get_report_generator()
    return generator.generate(company_code)
