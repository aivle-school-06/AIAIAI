"""
AI 예측기
=========

XGBoost 모델을 사용한 재무 지표 예측.
SHAP을 활용한 예측 근거 분석 (Explainable AI).
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import joblib
import shap

from app.config import settings, Settings
from app.core.data_loader import DataLoader
from app.core.constants import (
    ALL_TARGETS,
    TARGET_METRICS,
    MODEL_R2,
    MODEL_CONFIDENCE,
    METRIC_DIRECTION,
    METRIC_DESCRIPTION,
)
from app.exceptions import ModelNotFoundError, PredictionError, FeatureExtractionError

logger = logging.getLogger(__name__)


class Predictor:
    """
    AI 예측 및 SHAP 분석

    Args:
        settings: 애플리케이션 설정
        data_loader: 데이터 로더 인스턴스
    """

    def __init__(self, settings: Settings, data_loader: DataLoader):
        self.settings = settings
        self.data_loader = data_loader
        self.models: Dict[str, Any] = {}
        self.explainers: Dict[str, shap.TreeExplainer] = {}
        self._load_models()

    def _load_models(self) -> None:
        """모델 파일 로드"""
        # ai-server/ml_models/ 경로 사용
        model_path = settings.MODEL_PATH

        for metric in ALL_TARGETS:
            model_file = model_path / f'{metric}_model.joblib'

            if model_file.exists():
                try:
                    self.models[metric] = joblib.load(model_file)
                    self.explainers[metric] = shap.TreeExplainer(self.models[metric])
                    logger.debug(f"Loaded model for {metric}")
                except Exception as e:
                    logger.warning(f"Failed to load model for {metric}: {e}")
            else:
                logger.warning(f"Model file not found: {model_file}")

        logger.info(f"Loaded {len(self.models)}/{len(ALL_TARGETS)} models")

    def predict(self, company_code: str) -> Dict[str, Any]:
        """
        기업 예측 수행

        Args:
            company_code: 기업 코드

        Returns:
            Dict: 예측 결과
        """
        # 기업 데이터 조회
        company_data = self.data_loader.get_company_data(company_code)

        # 피처 추출
        features = self.data_loader.get_features_for_prediction(company_code)
        if features is None:
            raise FeatureExtractionError(company_code)

        # 13개 지표 예측
        predictions = {}
        risk_signals = []

        for metric in ALL_TARGETS:
            if metric not in self.models:
                predictions[metric] = {'error': 'Model not available'}
                continue

            try:
                result = self._predict_single(
                    metric=metric,
                    features=features,
                    current_value=company_data['current'],
                    trend=company_data['trend'].get(metric, {})
                )
                predictions[metric] = result

                if result.get('risk_signal'):
                    risk_signals.append(result['risk_signal'])

            except Exception as e:
                logger.error(f"Prediction failed for {metric}: {e}")
                predictions[metric] = {'error': str(e)}

        # 요약 생성
        summary = self._generate_summary(predictions)

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
        trend: Dict,
    ) -> Dict[str, Any]:
        """단일 지표 예측"""
        model = self.models[metric]
        explainer = self.explainers[metric]

        # 1. 예측 수행
        predicted = float(model.predict(features)[0])

        # 2. 현재 값 찾기
        current = None
        for category, metrics in TARGET_METRICS.items():
            if metric in metrics:
                current = current_value[category][metric]['value']
                break

        # 3. SHAP 분석
        shap_values = explainer.shap_values(features)
        shap_analysis = self._analyze_shap(shap_values[0], metric)

        # 4. 변화량 계산
        if current is not None:
            change = predicted - current
            change_pct = (change / abs(current) * 100) if current != 0 else 0
        else:
            change = None
            change_pct = None

        # 5. 방향성 판단
        direction = self._determine_direction(change, metric)

        # 6. 신뢰도 평가
        confidence = self._get_confidence(metric)

        # 7. 리스크 체크
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

    def _analyze_shap(self, shap_values: np.ndarray, metric: str) -> Dict[str, Any]:
        """SHAP 값 분석"""
        feature_cols = self.data_loader.feature_cols
        feature_shap = list(zip(feature_cols, shap_values))

        # 영향력 순 정렬
        sorted_shap = sorted(feature_shap, key=lambda x: abs(x[1]), reverse=True)

        top_factors = []
        for feature, shap_val in sorted_shap[:10]:
            top_factors.append({
                'feature': feature,
                'shap_value': float(shap_val),
                'impact': 'positive' if shap_val > 0 else 'negative',
                'description': self._get_feature_description(feature),
            })

        # 상승/하락 요인 분리
        positive = [(f, v) for f, v in feature_shap if v > 0]
        negative = [(f, v) for f, v in feature_shap if v < 0]

        top_positive = sorted(positive, key=lambda x: x[1], reverse=True)[:5]
        top_negative = sorted(negative, key=lambda x: x[1])[:5]

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
            'base_value': float(self.explainers[metric].expected_value),
        }

    def _get_feature_description(self, feature: str) -> str:
        """피처 설명 생성"""
        if '_lag1' in feature:
            return f'{feature.replace("_lag1", "")} 전분기 값'
        elif '_lag2' in feature:
            return f'{feature.replace("_lag2", "")} 2분기 전 값'
        elif '_lag3' in feature:
            return f'{feature.replace("_lag3", "")} 3분기 전 값'
        elif '_MA4' in feature:
            return f'{feature.replace("_MA4", "")} 4분기 이동평균'
        elif '_STD4' in feature:
            return f'{feature.replace("_STD4", "")} 4분기 변동성'
        elif '_YoY' in feature:
            return f'{feature.replace("_YoY", "")} 전년동기비'
        elif '_MOM' in feature:
            return f'{feature.replace("_MOM", "")} 전분기비'
        elif '_업종상대' in feature:
            return f'{feature.replace("_업종상대", "")} 업종 대비'
        elif feature.startswith('BS_'):
            return f'재무상태표 - {feature[3:]}'
        elif feature.startswith('PL_'):
            return f'손익계산서 - {feature[3:]}'
        elif feature.startswith('CF_'):
            return f'현금흐름표 - {feature[3:]}'
        return feature

    def _determine_direction(self, change: Optional[float], metric: str) -> str:
        """변화 방향 판단"""
        if change is None:
            return 'unknown'

        if abs(change) < 0.01:
            return 'stable'

        metric_dir = METRIC_DIRECTION.get(metric, 'higher')

        if metric_dir == 'higher':
            return 'improving' if change > 0 else 'declining'
        else:
            return 'improving' if change < 0 else 'declining'

    def _get_confidence(self, metric: str) -> str:
        """신뢰도 등급 반환"""
        r2 = MODEL_R2.get(metric, 0)

        if r2 >= MODEL_CONFIDENCE['high']:
            return 'high'
        elif r2 >= MODEL_CONFIDENCE['medium']:
            return 'medium'
        return 'low'

    def _check_risk(
        self,
        metric: str,
        current: Optional[float],
        predicted: float,
        trend: Dict,
    ) -> Optional[Dict[str, Any]]:
        """리스크 시그널 체크"""
        if current is None:
            return None

        direction = METRIC_DIRECTION.get(metric, 'higher')
        change = predicted - current

        # 악화 예측 여부
        is_worsening = (
            (direction == 'higher' and change < -0.5) or
            (direction == 'lower' and change > 0.5)
        )

        if not is_worsening:
            return None

        # 추세 악화 체크
        mom = trend.get('mom')
        yoy = trend.get('yoy')
        trend_worsening = False

        if mom is not None and yoy is not None:
            if direction == 'higher':
                trend_worsening = mom < 0 and yoy < 0
            else:
                trend_worsening = mom > 0 and yoy > 0

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

    def _generate_summary(self, predictions: Dict) -> Dict[str, Any]:
        """예측 요약 생성"""
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

        # 카테고리별 전망
        category_outlook = {}
        for category, metrics in TARGET_METRICS.items():
            cat_improving = len([m for m in metrics if m in improving])
            cat_declining = len([m for m in metrics if m in declining])

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
                'total': len(metrics),
            }

        # 전체 전망
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

    def get_shap_analysis(
        self, company_code: str, metric: str
    ) -> Optional[Dict[str, Any]]:
        """특정 지표의 SHAP 분석 결과"""
        if metric not in self.models:
            raise ModelNotFoundError(metric)

        features = self.data_loader.get_features_for_prediction(company_code)
        if features is None:
            raise FeatureExtractionError(company_code)

        shap_values = self.explainers[metric].shap_values(features)
        return self._analyze_shap(shap_values[0], metric)
