"""
AI 예측 모듈 (핵심 차별점)
=========================

이 모듈은 프로젝트의 핵심 기능인 AI 기반 재무 지표 예측을 담당합니다.

주요 기능:
1. XGBoost 모델을 사용한 13개 지표 예측
2. SHAP을 활용한 예측 근거 분석 (Explainable AI)
3. 예측 신뢰도 평가
4. 리스크 시그널 자동 감지

SHAP (SHapley Additive exPlanations):
- 각 피처가 예측에 얼마나 기여했는지 수치화
- 양수: 예측값 상승에 기여
- 음수: 예측값 하락에 기여
- 절대값이 클수록 영향력이 큼

사용 예시:
    predictor = get_predictor()
    result = predictor.predict('[005930]')
    print(result['predictions']['ROA']['shap_analysis'])
"""
import numpy as np
import pandas as pd
import joblib
import shap
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 설정 파일에서 필요한 값들 import
from .config import (
    MODEL_PATH,             # 모델 파일 경로
    ALL_TARGETS,            # 13개 예측 타겟
    MODEL_R2,               # 각 지표별 모델 R²
    MODEL_CONFIDENCE,       # 신뢰도 기준
    METRIC_DIRECTION,       # 지표 방향성 (higher/lower)
    METRIC_DESCRIPTION,     # 지표 설명
    TARGET_METRICS          # 카테고리별 타겟
)
from .data_loader import get_data_loader


class AIPredictor:
    """
    AI 예측 및 SHAP 분석 클래스

    XGBoost 모델로 다음 분기 재무 지표를 예측하고,
    SHAP을 통해 예측 근거를 설명합니다.
    """

    def __init__(self):
        """
        초기화: 모델 및 SHAP Explainer 로드

        - 13개 지표별 XGBoost 모델 로드
        - 각 모델에 대한 SHAP TreeExplainer 생성
        """
        self.models: Dict = {}           # {지표명: XGBoost 모델}
        self.explainers: Dict = {}       # {지표명: SHAP Explainer}
        self.feature_cols: List[str] = None  # 피처 목록 (376개)
        self._load_models()

    def _load_models(self):
        """
        모델 파일 로드 (내부 메서드)

        models/XGBoost/outputs/ 디렉토리에서 각 지표별 모델 로드
        """
        # 피처 목록 가져오기
        data_loader = get_data_loader()
        self.feature_cols = data_loader.feature_cols

        # 13개 지표별로 모델 로드
        for metric in ALL_TARGETS:
            model_file = MODEL_PATH / f'{metric}_model.joblib'

            if model_file.exists():
                # joblib으로 저장된 모델 로드
                self.models[metric] = joblib.load(model_file)

                # SHAP TreeExplainer 생성
                # TreeExplainer는 트리 기반 모델(XGBoost, RF 등)에 최적화됨
                self.explainers[metric] = shap.TreeExplainer(self.models[metric])

    def predict(self, company_code: str) -> Dict:
        """
        기업 예측 수행 (메인 함수)

        특정 기업에 대해 13개 지표를 예측하고,
        SHAP 분석, 신뢰도, 리스크 시그널을 포함한 결과 반환

        Args:
            company_code: 기업 코드 (예: '[005930]')

        Returns:
            Dict: 예측 결과
                - company_code: 기업 코드
                - company_name: 기업명
                - base_period: 기준 분기
                - predictions: 지표별 예측 결과
                - risk_signals: 리스크 시그널 목록
                - summary: 예측 요약

        예시:
            {
                'predictions': {
                    'ROA': {
                        'current': 2.33,
                        'predicted': 1.40,
                        'change': -0.93,
                        'shap_analysis': {...}
                    },
                    ...
                }
            }
        """
        # 데이터 로더에서 기업 데이터 조회
        data_loader = get_data_loader()
        company_data = data_loader.get_company_data(company_code)

        if not company_data:
            return {'error': f'기업 코드 {company_code}를 찾을 수 없습니다.'}

        # 예측용 피처 추출 (376개)
        features = data_loader.get_features_for_prediction(company_code)
        if features is None:
            return {'error': '피처 추출 실패'}

        # 13개 지표 각각 예측
        predictions = {}
        risk_signals = []

        for metric in ALL_TARGETS:
            if metric not in self.models:
                predictions[metric] = {'error': '모델 없음'}
                continue

            # 단일 지표 예측
            result = self._predict_single(
                metric=metric,
                features=features,
                current_value=company_data['current'],
                trend=company_data['trend'].get(metric, {})
            )
            predictions[metric] = result

            # 리스크 시그널 수집 (악화 예측 시)
            if result.get('risk_signal'):
                risk_signals.append(result['risk_signal'])

        # 전체 요약 생성
        summary = self._generate_summary(predictions, company_data)

        return {
            'company_code': company_code,
            'company_name': company_data['meta']['기업명'],
            'base_period': company_data['meta']['기준일'],
            'predictions': predictions,
            'risk_signals': risk_signals,
            'summary': summary,
        }

    def _predict_single(
        self,
        metric: str,
        features: np.ndarray,
        current_value: Dict,
        trend: Dict
    ) -> Dict:
        """
        단일 지표 예측 (내부 메서드)

        하나의 지표에 대해:
        1. XGBoost 모델로 예측
        2. SHAP으로 예측 근거 분석
        3. 변화량 계산
        4. 신뢰도 평가
        5. 리스크 체크

        Args:
            metric: 지표명 (예: 'ROA')
            features: 피처 배열 (1, 376)
            current_value: 현재 지표값 딕셔너리
            trend: 추세 데이터

        Returns:
            Dict: 예측 결과
                - current: 현재 값
                - predicted: 예측 값
                - change: 변화량
                - change_pct: 변화율 (%)
                - direction: 방향 (improving/declining/stable)
                - confidence: 신뢰도 (high/medium/low)
                - r2: 모델 R²
                - shap_analysis: SHAP 분석 결과
                - risk_signal: 리스크 시그널 (있을 경우)
        """
        model = self.models[metric]
        explainer = self.explainers[metric]

        # =========================================
        # 1. 예측 수행
        # =========================================
        # XGBoost 모델에 피처 입력 → 예측값 출력
        predicted = float(model.predict(features)[0])

        # =========================================
        # 2. 현재 값 찾기
        # =========================================
        # current_value는 카테고리별로 구조화되어 있으므로 해당 카테고리에서 찾기
        current = None
        for category, metrics in TARGET_METRICS.items():
            if metric in metrics:
                current = current_value[category][metric]['value']
                break

        # =========================================
        # 3. SHAP 분석
        # =========================================
        # SHAP 값 계산: 각 피처가 예측에 얼마나 기여했는지
        shap_values = explainer.shap_values(features)
        shap_analysis = self._analyze_shap(shap_values[0], metric)

        # =========================================
        # 4. 변화량 계산
        # =========================================
        if current is not None:
            change = predicted - current  # 절대 변화량
            # 변화율 (%) = 변화량 / |현재값| * 100
            change_pct = (change / abs(current) * 100) if current != 0 else 0
        else:
            change = None
            change_pct = None

        # =========================================
        # 5. 방향성 판단
        # =========================================
        # 지표 특성(높을수록 좋은지)과 변화 방향을 고려해 개선/악화 판단
        direction = self._determine_direction(change, metric)

        # =========================================
        # 6. 신뢰도 평가
        # =========================================
        confidence = self._get_confidence(metric)

        # =========================================
        # 7. 리스크 시그널 체크
        # =========================================
        risk_signal = self._check_risk(metric, current, predicted, trend)

        return {
            'current': current,
            'predicted': predicted,
            'change': change,
            'change_pct': change_pct,
            'direction': direction,
            'confidence': confidence,
            'r2': MODEL_R2.get(metric),
            'shap_analysis': shap_analysis,
            'risk_signal': risk_signal,
            'description': METRIC_DESCRIPTION.get(metric),
        }

    def _analyze_shap(self, shap_values: np.ndarray, metric: str) -> Dict:
        """
        SHAP 값 분석 (내부 메서드)

        376개 피처의 SHAP 값을 분석하여
        예측에 가장 큰 영향을 준 요인들을 추출

        SHAP 값 해석:
        - 양수 (+): 해당 피처가 예측값을 높이는 방향으로 기여
        - 음수 (-): 해당 피처가 예측값을 낮추는 방향으로 기여
        - 절대값: 영향력의 크기

        Args:
            shap_values: SHAP 값 배열 (376,)
            metric: 지표명

        Returns:
            Dict: SHAP 분석 결과
                - top_factors: 영향력 상위 10개 요인
                - positive_factors: 상승 기여 요인 (상위 5개)
                - negative_factors: 하락 기여 요인 (상위 5개)
                - base_value: 기준값 (평균 예측값)
        """
        # 피처명과 SHAP 값을 쌍으로 묶기
        feature_shap = list(zip(self.feature_cols, shap_values))

        # 절대값 기준 정렬 (영향력 크기순)
        sorted_shap = sorted(feature_shap, key=lambda x: abs(x[1]), reverse=True)

        # 결과 저장용 리스트
        top_factors = []
        positive_factors = []
        negative_factors = []

        # 상위 10개 요인 추출
        for feature, shap_val in sorted_shap[:10]:
            factor = {
                'feature': feature,                              # 피처명
                'shap_value': float(shap_val),                  # SHAP 값
                'impact': 'positive' if shap_val > 0 else 'negative',  # 방향
                'description': self._get_feature_description(feature),  # 설명
            }
            top_factors.append(factor)

            # 양수/음수 분류
            if shap_val > 0:
                positive_factors.append(factor)
            else:
                negative_factors.append(factor)

        # 상승 요인 상위 5개 (SHAP 값 큰 순)
        top_positive = sorted(
            [(f, v) for f, v in feature_shap if v > 0],
            key=lambda x: x[1], reverse=True
        )[:5]

        # 하락 요인 상위 5개 (SHAP 값 작은 순, 음수니까)
        top_negative = sorted(
            [(f, v) for f, v in feature_shap if v < 0],
            key=lambda x: x[1]  # 음수이므로 오름차순 = 절대값 큰 순
        )[:5]

        return {
            'top_factors': top_factors,
            'positive_factors': [
                {
                    'feature': f,
                    'shap_value': float(v),
                    'description': self._get_feature_description(f)
                }
                for f, v in top_positive
            ],
            'negative_factors': [
                {
                    'feature': f,
                    'shap_value': float(v),
                    'description': self._get_feature_description(f)
                }
                for f, v in top_negative
            ],
            # base_value: 모델의 평균 예측값 (SHAP 값의 기준점)
            'base_value': float(self.explainers[metric].expected_value),
        }

    def _get_feature_description(self, feature: str) -> str:
        """
        피처 설명 생성 (내부 메서드)

        피처 이름에서 의미를 추출하여 사람이 읽기 쉬운 설명 생성

        Args:
            feature: 피처명 (예: 'ROA_lag1', 'PL_매출액_YoY')

        Returns:
            str: 피처 설명 (예: 'ROA 전분기 값', '매출액 전년동기비 변화')
        """
        # 피처 이름 패턴에 따라 설명 생성
        if '_lag1' in feature:
            base = feature.replace('_lag1', '')
            return f'{base} 전분기 값'
        elif '_lag2' in feature:
            base = feature.replace('_lag2', '')
            return f'{base} 2분기 전 값'
        elif '_lag3' in feature:
            base = feature.replace('_lag3', '')
            return f'{base} 3분기 전 값'
        elif '_MA4' in feature:
            base = feature.replace('_MA4', '')
            return f'{base} 4분기 이동평균'
        elif '_STD4' in feature:
            base = feature.replace('_STD4', '')
            return f'{base} 4분기 변동성'
        elif '_YoY' in feature:
            base = feature.replace('_YoY', '')
            return f'{base} 전년동기비 변화'
        elif '_MOM' in feature:
            base = feature.replace('_MOM', '')
            return f'{base} 전분기비 변화'
        elif '_업종상대' in feature:
            base = feature.replace('_업종상대', '')
            return f'{base} 업종 대비'
        elif feature.startswith('BS_'):
            return f'재무상태표 - {feature[3:]}'
        elif feature.startswith('PL_'):
            return f'손익계산서 - {feature[3:]}'
        elif feature.startswith('CF_'):
            return f'현금흐름표 - {feature[3:]}'
        else:
            return feature  # 원본 이름 그대로

    def _determine_direction(self, change: Optional[float], metric: str) -> str:
        """
        변화 방향 판단 (내부 메서드)

        지표의 특성(높을수록 좋은지)과 변화량을 고려하여
        개선/악화/유지를 판단

        Args:
            change: 변화량 (예측값 - 현재값)
            metric: 지표명

        Returns:
            str: 'improving' (개선), 'declining' (악화), 'stable' (유지), 'unknown'
        """
        if change is None:
            return 'unknown'

        # 지표 방향성 가져오기
        metric_dir = METRIC_DIRECTION.get(metric, 'higher')

        # 변화가 거의 없으면 stable
        if abs(change) < 0.01:
            return 'stable'

        # 방향성에 따라 개선/악화 판단
        if metric_dir == 'higher':
            # 높을수록 좋은 지표: 상승 = 개선, 하락 = 악화
            return 'improving' if change > 0 else 'declining'
        else:
            # 낮을수록 좋은 지표: 하락 = 개선, 상승 = 악화
            return 'improving' if change < 0 else 'declining'

    def _get_confidence(self, metric: str) -> str:
        """
        신뢰도 등급 반환 (내부 메서드)

        모델의 R² 값에 따라 예측 신뢰도 등급 결정

        Args:
            metric: 지표명

        Returns:
            str: 'high', 'medium', 'low'
        """
        r2 = MODEL_R2.get(metric, 0)

        if r2 >= MODEL_CONFIDENCE['high']:
            return 'high'       # R² >= 0.7
        elif r2 >= MODEL_CONFIDENCE['medium']:
            return 'medium'     # R² >= 0.4
        else:
            return 'low'        # R² < 0.4

    def _check_risk(
        self,
        metric: str,
        current: Optional[float],
        predicted: float,
        trend: Dict
    ) -> Optional[Dict]:
        """
        리스크 시그널 체크 (내부 메서드)

        예측 결과가 악화를 나타내면 리스크 시그널 생성
        추세도 악화 중이면 심각도 높음

        Args:
            metric: 지표명
            current: 현재 값
            predicted: 예측 값
            trend: 추세 데이터 (yoy, mom 등)

        Returns:
            Dict: 리스크 시그널 (악화 예측 시)
                - metric: 지표명
                - severity: 심각도 (high/medium)
                - message: 리스크 메시지
            None: 리스크 없음
        """
        if current is None:
            return None

        direction = METRIC_DIRECTION.get(metric, 'higher')
        change = predicted - current

        # 악화 예측 여부 판단 (변화량 0.5 이상)
        is_worsening = (direction == 'higher' and change < -0.5) or \
                       (direction == 'lower' and change > 0.5)

        if not is_worsening:
            return None  # 악화 아니면 리스크 없음

        # 추세도 악화 중인지 확인 (YoY, MoM 모두 악화 방향)
        mom = trend.get('mom')
        yoy = trend.get('yoy')
        trend_worsening = False

        if mom is not None and yoy is not None:
            if direction == 'higher':
                # 높을수록 좋은 지표가 감소 추세면 악화
                trend_worsening = mom < 0 and yoy < 0
            else:
                # 낮을수록 좋은 지표가 증가 추세면 악화
                trend_worsening = mom > 0 and yoy > 0

        # 심각도 결정 (추세도 악화면 high)
        severity = 'high' if trend_worsening else 'medium'

        return {
            'metric': metric,
            'severity': severity,
            'current': current,
            'predicted': predicted,
            'change': change,
            'message': f'{metric} 악화 예측: {current:.2f} → {predicted:.2f}',
            'trend_worsening': trend_worsening,
        }

    def _generate_summary(self, predictions: Dict, company_data: Dict) -> Dict:
        """
        예측 요약 생성 (내부 메서드)

        13개 지표 예측 결과를 종합하여 요약 정보 생성

        Args:
            predictions: 지표별 예측 결과
            company_data: 기업 데이터

        Returns:
            Dict: 예측 요약
                - overall_outlook: 전체 전망 (positive/negative/neutral)
                - improving_count: 개선 예측 지표 수
                - declining_count: 악화 예측 지표 수
                - category_outlook: 카테고리별 전망
        """
        # 방향별 지표 분류
        improving = []
        declining = []
        stable = []

        for metric, result in predictions.items():
            if 'error' in result:
                continue

            direction = result.get('direction')
            if direction == 'improving':
                improving.append(metric)
            elif direction == 'declining':
                declining.append(metric)
            else:
                stable.append(metric)

        # 카테고리별 전망 계산
        category_outlook = {}
        for category, metrics in TARGET_METRICS.items():
            cat_improving = len([m for m in metrics if m in improving])
            cat_declining = len([m for m in metrics if m in declining])
            cat_total = len(metrics)

            # 카테고리 전망 결정
            if cat_improving > cat_declining:
                outlook = 'positive'
            elif cat_declining > cat_improving:
                outlook = 'negative'
            else:
                outlook = 'neutral'

            category_outlook[category] = {
                'outlook': outlook,
                'improving': cat_improving,
                'declining': cat_declining,
                'total': cat_total,
            }

        # 전체 전망 결정
        if len(improving) > len(declining):
            overall = 'positive'
        elif len(declining) > len(improving):
            overall = 'negative'
        else:
            overall = 'neutral'

        return {
            'overall_outlook': overall,
            'improving_count': len(improving),
            'declining_count': len(declining),
            'stable_count': len(stable),
            'improving_metrics': improving,
            'declining_metrics': declining,
            'stable_metrics': stable,
            'category_outlook': category_outlook,
        }


# =============================================================================
# 싱글톤 패턴
# =============================================================================
_predictor: Optional[AIPredictor] = None


def get_predictor() -> AIPredictor:
    """
    AIPredictor 싱글톤 인스턴스 반환

    처음 호출 시 모델 로드 (시간 소요),
    이후에는 기존 인스턴스 재사용

    Returns:
        AIPredictor: 예측기 인스턴스
    """
    global _predictor

    if _predictor is None:
        _predictor = AIPredictor()

    return _predictor
