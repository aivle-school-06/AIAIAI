"""
LLM 종합 의견 생성 모듈 (XAI + LLM 통합)
=========================================

이 모듈은 GPT API와 SHAP 분석 결과를 활용하여
재무 분석 보고서의 종합 의견을 생성합니다.

핵심 차별점:
1. SHAP 기반 예측 근거를 LLM이 자연어로 해석
2. 전문가용/비전문가용 맞춤 의견 생성
3. 예측 요인에 대한 직관적 설명
4. 13개 지표 전체에 대한 카테고리별/지표별 XAI + LLM 분석

사용 예시:
    from report.llm_opinion import generate_opinion, generate_category_analysis

    # 종합 의견
    opinion = generate_opinion(opinion_data)

    # 카테고리별 XAI + LLM 분석 (13개 지표)
    category_analysis = generate_category_analysis(opinion_data, predictions)
    print(category_analysis['수익성']['summary'])
    print(category_analysis['수익성']['metrics']['ROA']['insight'])
"""
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

from .config import TARGET_METRICS, METRIC_DESCRIPTION, METRIC_DIRECTION, get_feature_description

# .env 파일 로드
load_dotenv()


class LLMOpinionGenerator:
    """
    LLM 기반 종합 의견 생성 클래스 (XAI 통합)

    OpenAI GPT API와 SHAP 분석 결과를 결합하여
    설명 가능한 AI 기반 재무 분석 의견을 생성합니다.
    """

    def __init__(self):
        """
        초기화: OpenAI 클라이언트 생성
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # 가성비 좋은 모델

    def generate(self, opinion_data: Dict) -> Dict:
        """
        종합 의견 생성 (메인 함수)

        Args:
            opinion_data: report_generator에서 준비한 LLM용 데이터
                - company: 기업 기본 정보
                - grades: 등급 요약
                - current_metrics: 현재 지표값
                - relative_metrics: 업종 대비 상대값
                - trend: 추세 요약
                - prediction: AI 예측 요약
                - risk_signals: 리스크 시그널
                - shap_analysis: SHAP 분석 결과 (선택)

        Returns:
            Dict: 생성된 의견
                - expert: 전문가용 의견
                - simple: 비전문가용 의견
                - key_points: 핵심 포인트 (3~5개)
                - risk_summary: 리스크 요약
                - xai_explanation: XAI 기반 예측 근거 설명
        """
        # 프롬프트 생성
        prompt = self._build_prompt(opinion_data)

        # GPT API 호출
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=3000,
            )

            # 응답 파싱
            content = response.choices[0].message.content
            return self._parse_response(content, opinion_data)

        except Exception as e:
            return {
                'error': str(e),
                'expert': '의견 생성 중 오류가 발생했습니다.',
                'simple': '의견 생성 중 오류가 발생했습니다.',
                'key_points': [],
                'risk_summary': None,
                'xai_explanation': None,
            }

    def _get_system_prompt(self) -> str:
        """
        시스템 프롬프트 반환
        """
        return """당신은 한국 상장기업 재무 분석 전문가이자 AI 모델 해석 전문가입니다.
주어진 재무 데이터와 AI 예측 결과(SHAP 분석 포함)를 분석하여 종합 의견을 작성해주세요.

응답 형식:

[전문가용]
(금융/투자 전문가를 위한 상세 분석. 전문 용어 사용 가능. 5~7문장)
- 현재 재무 상태 평가
- AI 예측 결과 해석
- SHAP 분석 기반 핵심 요인 설명

[비전문가용]
(일반인을 위한 쉬운 설명. 비유와 쉬운 표현 사용)
- 돈을 잘 버나요? → (수익성 설명)
- 빚이 많나요? → (안정성 설명)
- 현금은 충분한가요? → (현금흐름 설명)
- 앞으로 전망은? → (AI 예측 기반 전망)

[AI 예측 근거]
(SHAP 분석 결과를 바탕으로 AI가 왜 이렇게 예측했는지 설명)
• (주요 상승 요인과 그 의미)
• (주요 하락 요인과 그 의미)
• (예측의 신뢰도와 해석 시 주의점)

[핵심 포인트]
• (핵심 사항 1)
• (핵심 사항 2)
• (핵심 사항 3)

[주의 사항]
(리스크가 있다면 설명, 없으면 "특별한 주의 사항 없음")
"""

    def _build_prompt(self, data: Dict) -> str:
        """
        분석용 프롬프트 생성

        Args:
            data: opinion_data

        Returns:
            str: 완성된 프롬프트
        """
        company = data.get('company', {})
        grades = data.get('grades', {})
        current = data.get('current_metrics', {})
        relative = data.get('relative_metrics', {})
        trend = data.get('trend', {})
        prediction = data.get('prediction', {})
        risk_signals = data.get('risk_signals', [])
        shap_analysis = data.get('shap_analysis', {})

        # 프롬프트 구성
        prompt = f"""다음 기업의 재무 상태와 AI 예측 결과를 분석하고 종합 의견을 작성해주세요.

## 기업 정보
- 기업명: {company.get('name', 'N/A')}
- 업종: {company.get('industry', 'N/A')}
- 시장: {company.get('market', 'N/A')}
- 기준일: {company.get('period', 'N/A')}

## 종합 등급
- 등급: {grades.get('overall', 'N/A')}
- 점수: {grades.get('score', 'N/A')}
- 강점 지표: {', '.join(grades.get('strengths', [])) or '없음'}
- 약점 지표: {', '.join(grades.get('weaknesses', [])) or '없음'}

## 현재 주요 지표
"""
        # 현재 지표값 추가
        key_metrics = ['ROA', 'ROE', '매출액영업이익률', '부채비율', '유동비율', 'CFO_자산비율']
        for metric in key_metrics:
            value = current.get(metric)
            rel = relative.get(metric)
            if value is not None:
                rel_str = f" (업종 대비 {rel:+.1f}%p)" if rel is not None else ""
                prompt += f"- {metric}: {value:.2f}%{rel_str}\n"

        # 추세 정보 추가
        prompt += "\n## 추세 분석\n"
        for metric in key_metrics:
            t = trend.get(metric, {})
            yoy = t.get('yoy')
            mom = t.get('mom')
            if yoy is not None and mom is not None:
                direction = "▲ 개선" if yoy > 0 and mom > 0 else "▼ 악화" if yoy < 0 and mom < 0 else "→ 혼조"
                prompt += f"- {metric}: YoY {yoy:+.1f}%, QoQ {mom:+.1f}% ({direction})\n"

        # AI 예측 추가
        prompt += f"""
## AI 예측 (다음 분기)
- 전체 전망: {prediction.get('outlook', 'N/A')}
- 개선 예측 지표: {', '.join(prediction.get('improving', [])) or '없음'}
- 악화 예측 지표: {', '.join(prediction.get('declining', [])) or '없음'}
"""

        # SHAP 분석 결과 추가 (핵심 차별점)
        if shap_analysis:
            prompt += "\n## AI 예측 근거 (SHAP 분석)\n"
            prompt += "AI 모델이 예측할 때 가장 중요하게 본 요인들입니다.\n\n"

            for metric, analysis in shap_analysis.items():
                if not analysis:
                    continue

                prompt += f"### {metric} 예측 근거\n"

                # 상승 요인
                pos_factors = analysis.get('positive_factors', [])[:3]
                if pos_factors:
                    prompt += "상승 요인:\n"
                    for f in pos_factors:
                        feature = f.get('feature', '')
                        desc = f.get('description', feature)
                        shap_val = f.get('shap_value', 0)
                        prompt += f"  ↑ {desc} (기여도: {shap_val:+.3f})\n"

                # 하락 요인
                neg_factors = analysis.get('negative_factors', [])[:3]
                if neg_factors:
                    prompt += "하락 요인:\n"
                    for f in neg_factors:
                        feature = f.get('feature', '')
                        desc = f.get('description', feature)
                        shap_val = f.get('shap_value', 0)
                        prompt += f"  ↓ {desc} (기여도: {shap_val:+.3f})\n"

                prompt += "\n"

        # 리스크 시그널 추가
        if risk_signals:
            prompt += "\n## 리스크 시그널\n"
            for signal in risk_signals[:3]:
                severity = signal.get('severity', 'medium')
                severity_str = "⚠️ 높음" if severity == 'high' else "⚡ 중간"
                prompt += f"- {severity_str}: {signal.get('message', '')}\n"

        return prompt

    def _parse_response(self, content: str, opinion_data: Dict) -> Dict:
        """
        GPT 응답 파싱

        Args:
            content: GPT 응답 텍스트
            opinion_data: 원본 데이터 (폴백용)

        Returns:
            Dict: 구조화된 의견
        """
        result = {
            'expert': '',
            'simple': '',
            'key_points': [],
            'risk_summary': None,
            'xai_explanation': None,
            'raw_response': content,
        }

        # 섹션별 파싱
        sections = content.split('[')

        for section in sections:
            if section.startswith('전문가용]'):
                result['expert'] = section.replace('전문가용]', '').strip()
            elif section.startswith('비전문가용]'):
                result['simple'] = section.replace('비전문가용]', '').strip()
            elif section.startswith('AI 예측 근거]'):
                result['xai_explanation'] = section.replace('AI 예측 근거]', '').strip()
            elif section.startswith('핵심 포인트]'):
                points_text = section.replace('핵심 포인트]', '').strip()
                # • 또는 - 로 시작하는 줄 추출
                points = []
                for line in points_text.split('\n'):
                    line = line.strip()
                    if line.startswith('•') or line.startswith('-'):
                        points.append(line.lstrip('•-').strip())
                result['key_points'] = points
            elif section.startswith('주의 사항]'):
                result['risk_summary'] = section.replace('주의 사항]', '').strip()

        return result

    def generate_xai_summary(self, predictions: Dict) -> Dict:
        """
        XAI(SHAP) 기반 예측 요약 생성

        각 지표별 예측의 주요 요인을 정리하여
        LLM 프롬프트에 사용할 수 있는 형태로 변환

        Args:
            predictions: AI 예측 결과 (shap_analysis 포함)

        Returns:
            Dict: 지표별 SHAP 요약
        """
        xai_summary = {}

        for metric, pred in predictions.items():
            if 'error' in pred or 'shap_analysis' not in pred:
                continue

            shap = pred['shap_analysis']
            xai_summary[metric] = {
                'direction': pred.get('direction'),
                'change': pred.get('change'),
                'confidence': pred.get('confidence'),
                'positive_factors': shap.get('positive_factors', [])[:3],
                'negative_factors': shap.get('negative_factors', [])[:3],
            }

        return xai_summary

    def generate_category_analysis(self, opinion_data: Dict, predictions: Dict) -> Dict:
        """
        카테고리별 XAI + LLM 분석 생성

        5개 카테고리(수익성, 안정성, 차입금, 유동성, 현금흐름)별로
        해당 지표들의 XAI + LLM 분석을 생성합니다.

        Args:
            opinion_data: report_generator에서 준비한 LLM용 데이터
            predictions: AI 예측 결과 (all_predictions)

        Returns:
            Dict: 카테고리별 분석 결과
                {
                    '수익성': {
                        'summary': '카테고리 종합 분석',
                        'outlook': 'positive/negative/neutral',
                        'metrics': {
                            'ROA': {
                                'insight': '지표별 인사이트',
                                'factors': {'positive': [...], 'negative': [...]},
                            },
                            ...
                        }
                    },
                    ...
                }
        """
        result = {}

        for category, metrics in TARGET_METRICS.items():
            # 해당 카테고리 지표들의 예측 데이터 수집
            category_predictions = {}
            for metric in metrics:
                if metric in predictions:
                    category_predictions[metric] = predictions[metric]

            if not category_predictions:
                continue

            # 카테고리별 분석 생성
            category_result = self._generate_single_category_analysis(
                category, metrics, opinion_data, category_predictions
            )
            result[category] = category_result

        return result

    def _generate_single_category_analysis(
        self,
        category: str,
        metrics: List[str],
        opinion_data: Dict,
        predictions: Dict
    ) -> Dict:
        """
        단일 카테고리에 대한 XAI + LLM 분석 생성

        Args:
            category: 카테고리명 (수익성, 안정성 등)
            metrics: 해당 카테고리의 지표 목록
            opinion_data: LLM용 데이터
            predictions: 해당 카테고리 지표들의 예측

        Returns:
            Dict: 카테고리 분석 결과
        """
        prompt = self._build_category_prompt(category, metrics, opinion_data, predictions)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_category_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            content = response.choices[0].message.content
            return self._parse_category_response(content, category, metrics, predictions)

        except Exception as e:
            # 폴백: 기본 분석 결과 생성
            return self._generate_fallback_category_analysis(category, metrics, predictions)

    def _get_category_system_prompt(self) -> str:
        """카테고리별 분석용 시스템 프롬프트"""
        return """당신은 한국 상장기업 재무 분석 전문가이자 XAI(설명가능한 AI) 해석 전문가입니다.
주어진 카테고리의 재무지표들과 AI 예측 결과(SHAP 분석 포함)를 분석하여
전문적인 인사이트를 제공해주세요.

**중요**: SHAP 분석 결과는 AI 모델이 예측할 때 어떤 피쳐(변수)가 예측에 영향을 미쳤는지 보여줍니다.
- 양수(+) 기여도: 해당 피쳐가 예측값을 높이는 방향으로 작용
- 음수(-) 기여도: 해당 피쳐가 예측값을 낮추는 방향으로 작용
- SHAP 피쳐를 해석할 때는 반드시 "왜 이 피쳐가 예측에 영향을 미쳤는지" 비즈니스 관점에서 설명해주세요.

응답 형식:

[카테고리 종합]
(해당 카테고리 전체에 대한 종합 분석. 3~4문장. 현재 상태와 예측 전망, SHAP 분석에서 발견한 주요 패턴 포함)

[전망]
(positive/negative/neutral 중 하나)

[지표별 분석]

{{지표명1}}:
- 현황: (현재 상태에 대한 1문장 평가)
- XAI 해석: (SHAP 요인들이 왜 이 예측을 만들었는지 구체적으로 설명. 예: "전분기 ROA 상승 모멘텀이 긍정적 영향을 미쳤고, 부채비율 증가가 부정적 요인으로 작용했습니다.")
- 시사점: (비즈니스적 의미와 권고사항. 1~2문장)

{{지표명2}}:
- 현황: ...
- XAI 해석: ...
- 시사점: ...

(모든 지표에 대해 반복)

[핵심 메시지]
• (SHAP 분석에서 발견한 가장 중요한 인사이트)
• (해당 카테고리에 대한 실행 가능한 권고사항)
"""

    def _build_category_prompt(
        self,
        category: str,
        metrics: List[str],
        opinion_data: Dict,
        predictions: Dict
    ) -> str:
        """카테고리별 분석 프롬프트 생성"""
        company = opinion_data.get('company', {})
        current_metrics = opinion_data.get('current_metrics', {})
        relative_metrics = opinion_data.get('relative_metrics', {})
        trend = opinion_data.get('trend', {})

        prompt = f"""다음 기업의 '{category}' 카테고리 지표들을 분석해주세요.

## 기업 정보
- 기업명: {company.get('name', 'N/A')}
- 업종: {company.get('industry', 'N/A')}
- 기준일: {company.get('period', 'N/A')}

## {category} 지표 현황
"""
        for metric in metrics:
            value = current_metrics.get(metric)
            relative = relative_metrics.get(metric)
            direction = METRIC_DIRECTION.get(metric, 'higher')
            description = METRIC_DESCRIPTION.get(metric, '')

            if value is not None:
                rel_str = f" (업종 대비 {relative:+.1f}%p)" if relative is not None else ""
                dir_str = "높을수록 좋음" if direction == 'higher' else "낮을수록 좋음"
                prompt += f"\n### {metric}\n"
                prompt += f"- 설명: {description}\n"
                prompt += f"- 현재값: {value:.2f}%{rel_str}\n"
                prompt += f"- 방향성: {dir_str}\n"

                # 추세 정보
                t = trend.get(metric, {})
                yoy = t.get('yoy')
                mom = t.get('mom')
                if yoy is not None and mom is not None:
                    prompt += f"- 추세: YoY {yoy:+.1f}%, QoQ {mom:+.1f}%\n"

                # AI 예측 + SHAP
                pred = predictions.get(metric, {})
                if pred:
                    predicted = pred.get('predicted')
                    change = pred.get('change')
                    confidence = pred.get('confidence', 'medium')

                    if predicted is not None:
                        prompt += f"\n#### AI 예측\n"
                        prompt += f"- 예측값: {predicted:.2f}% (변화: {change:+.2f}%p)\n"
                        prompt += f"- 신뢰도: {confidence}\n"

                        # SHAP 요인 (피쳐 설명 포함)
                        shap = pred.get('shap_analysis', {})
                        pos_factors = shap.get('positive_factors', [])[:3]
                        neg_factors = shap.get('negative_factors', [])[:3]

                        if pos_factors:
                            prompt += "\n#### SHAP 상승 요인 (예측값을 높이는 방향)\n"
                            for f in pos_factors:
                                feature_name = f.get('feature', '')
                                # 피쳐 설명 가져오기
                                feature_desc = get_feature_description(feature_name)
                                shap_val = f.get('shap_value', 0)
                                feature_val = f.get('value', None)
                                val_str = f" (현재값: {feature_val:.2f})" if feature_val is not None else ""
                                prompt += f"  ↑ {feature_desc}{val_str}\n"
                                prompt += f"    - 피쳐명: {feature_name}\n"
                                prompt += f"    - SHAP 기여도: {shap_val:+.3f} (양수 = 예측값 상승 기여)\n"

                        if neg_factors:
                            prompt += "\n#### SHAP 하락 요인 (예측값을 낮추는 방향)\n"
                            for f in neg_factors:
                                feature_name = f.get('feature', '')
                                feature_desc = get_feature_description(feature_name)
                                shap_val = f.get('shap_value', 0)
                                feature_val = f.get('value', None)
                                val_str = f" (현재값: {feature_val:.2f})" if feature_val is not None else ""
                                prompt += f"  ↓ {feature_desc}{val_str}\n"
                                prompt += f"    - 피쳐명: {feature_name}\n"
                                prompt += f"    - SHAP 기여도: {shap_val:+.3f} (음수 = 예측값 하락 기여)\n"

        return prompt

    def _parse_category_response(
        self,
        content: str,
        category: str,
        metrics: List[str],
        predictions: Dict
    ) -> Dict:
        """카테고리 분석 응답 파싱"""
        result = {
            'summary': '',
            'outlook': 'neutral',
            'key_messages': [],
            'metrics': {},
            'raw_response': content,
        }

        # 섹션별 파싱
        sections = content.split('[')

        for section in sections:
            if section.startswith('카테고리 종합]'):
                result['summary'] = section.replace('카테고리 종합]', '').strip()
                # 다음 섹션 시작 전까지만
                if '\n\n' in result['summary']:
                    result['summary'] = result['summary'].split('\n\n')[0].strip()

            elif section.startswith('전망]'):
                outlook_text = section.replace('전망]', '').strip().lower()
                if 'positive' in outlook_text or '긍정' in outlook_text:
                    result['outlook'] = 'positive'
                elif 'negative' in outlook_text or '부정' in outlook_text:
                    result['outlook'] = 'negative'
                else:
                    result['outlook'] = 'neutral'

            elif section.startswith('지표별 분석]'):
                metrics_text = section.replace('지표별 분석]', '').strip()
                result['metrics'] = self._parse_metrics_analysis(metrics_text, metrics, predictions)

            elif section.startswith('핵심 메시지]'):
                messages_text = section.replace('핵심 메시지]', '').strip()
                messages = []
                for line in messages_text.split('\n'):
                    line = line.strip()
                    if line.startswith('•') or line.startswith('-'):
                        messages.append(line.lstrip('•-').strip())
                result['key_messages'] = messages[:3]

        return result

    def _parse_metrics_analysis(
        self,
        text: str,
        metrics: List[str],
        predictions: Dict
    ) -> Dict:
        """지표별 분석 텍스트 파싱"""
        result = {}

        for metric in metrics:
            # 해당 지표 섹션 찾기
            metric_start = text.find(f'{metric}:')
            if metric_start == -1:
                # SHAP 데이터만으로 폴백
                pred = predictions.get(metric, {})
                shap = pred.get('shap_analysis', {})
                result[metric] = {
                    'insight': '',
                    'current_status': '',
                    'prediction_analysis': '',
                    'implication': '',
                    'factors': {
                        'positive': shap.get('positive_factors', [])[:3],
                        'negative': shap.get('negative_factors', [])[:3],
                    }
                }
                continue

            # 다음 지표 시작점 찾기
            next_start = len(text)
            for other_metric in metrics:
                if other_metric != metric:
                    other_start = text.find(f'{other_metric}:', metric_start + 1)
                    if other_start > metric_start and other_start < next_start:
                        next_start = other_start

            metric_text = text[metric_start:next_start].strip()

            # 각 항목 파싱
            current_status = ''
            prediction_analysis = ''
            implication = ''

            for line in metric_text.split('\n'):
                line = line.strip()
                if line.startswith('- 현황:'):
                    current_status = line.replace('- 현황:', '').strip()
                elif line.startswith('- 예측:'):
                    prediction_analysis = line.replace('- 예측:', '').strip()
                elif line.startswith('- 시사점:'):
                    implication = line.replace('- 시사점:', '').strip()

            # SHAP 요인
            pred = predictions.get(metric, {})
            shap = pred.get('shap_analysis', {})

            result[metric] = {
                'insight': prediction_analysis or current_status,
                'current_status': current_status,
                'prediction_analysis': prediction_analysis,
                'implication': implication,
                'factors': {
                    'positive': shap.get('positive_factors', [])[:3],
                    'negative': shap.get('negative_factors', [])[:3],
                }
            }

        return result

    def _generate_fallback_category_analysis(
        self,
        category: str,
        metrics: List[str],
        predictions: Dict
    ) -> Dict:
        """폴백 카테고리 분석 생성 (API 오류 시)"""
        result = {
            'summary': f'{category} 분석 데이터',
            'outlook': 'neutral',
            'key_messages': [],
            'metrics': {},
        }

        improving = []
        declining = []

        for metric in metrics:
            pred = predictions.get(metric, {})
            shap = pred.get('shap_analysis', {})
            direction = pred.get('direction', 'stable')

            if direction == 'improving':
                improving.append(metric)
            elif direction == 'declining':
                declining.append(metric)

            result['metrics'][metric] = {
                'insight': f"{metric} 예측 분석",
                'current_status': '',
                'prediction_analysis': '',
                'implication': '',
                'factors': {
                    'positive': shap.get('positive_factors', [])[:3],
                    'negative': shap.get('negative_factors', [])[:3],
                }
            }

        # 전망 결정
        if len(improving) > len(declining):
            result['outlook'] = 'positive'
        elif len(declining) > len(improving):
            result['outlook'] = 'negative'

        return result


# =============================================================================
# 싱글톤 패턴
# =============================================================================
_generator: Optional[LLMOpinionGenerator] = None


def get_llm_generator() -> LLMOpinionGenerator:
    """
    LLMOpinionGenerator 싱글톤 인스턴스 반환

    Returns:
        LLMOpinionGenerator: LLM 의견 생성기 인스턴스
    """
    global _generator

    if _generator is None:
        _generator = LLMOpinionGenerator()

    return _generator


def generate_opinion(opinion_data: Dict) -> Dict:
    """
    종합 의견 생성 편의 함수

    Args:
        opinion_data: report_generator에서 준비한 LLM용 데이터

    Returns:
        Dict: 생성된 의견

    사용 예시:
        from report.llm_opinion import generate_opinion
        opinion = generate_opinion(report['opinion_data'])
        print(opinion['expert'])
        print(opinion['xai_explanation'])
    """
    generator = get_llm_generator()
    return generator.generate(opinion_data)


def generate_category_analysis(opinion_data: Dict, predictions: Dict) -> Dict:
    """
    카테고리별 XAI + LLM 분석 생성 편의 함수

    13개 지표를 5개 카테고리로 나누어 각각에 대한
    XAI + LLM 기반 분석을 생성합니다.

    Args:
        opinion_data: report_generator에서 준비한 LLM용 데이터
        predictions: AI 예측 결과 (all_predictions)

    Returns:
        Dict: 카테고리별 분석 결과

    사용 예시:
        from report.llm_opinion import generate_category_analysis
        analysis = generate_category_analysis(opinion_data, predictions)
        print(analysis['수익성']['summary'])
        print(analysis['수익성']['metrics']['ROA']['insight'])
    """
    generator = get_llm_generator()
    return generator.generate_category_analysis(opinion_data, predictions)


def generate_industry_comparison_analysis(
    company_name: str,
    industry: str,
    category_positions: Dict,
    metrics_comparison: Dict,
    strengths: List,
    weaknesses: List
) -> str:
    """
    업종 내 비교 분석 LLM 생성

    Args:
        company_name: 기업명
        industry: 업종명
        category_positions: 카테고리별 업종 내 순위 {카테고리: 백분위}
        metrics_comparison: 지표별 비교 데이터 {지표: {company, industry_mean, percentile}}
        strengths: 강점 지표 리스트
        weaknesses: 약점 지표 리스트

    Returns:
        str: LLM이 생성한 업종 비교 분석 텍스트
    """
    generator = get_llm_generator()

    prompt = f"""다음 기업의 업종 내 위치를 분석해주세요.

## 기업 정보
- 기업명: {company_name}
- 업종: {industry}

## 카테고리별 업종 내 순위
"""
    for cat, pos in category_positions.items():
        if pos is not None:
            rank_text = f"상위 {pos:.0f}%" if pos <= 50 else f"하위 {100-pos:.0f}%"
            prompt += f"- {cat}: {rank_text}\n"

    prompt += "\n## 주요 지표 비교 (기업 vs 업종평균)\n"
    for metric, info in metrics_comparison.items():
        if info:
            company_val = info.get('company')
            mean_val = info.get('industry_mean')
            pct = info.get('percentile')
            if company_val is not None and mean_val is not None:
                diff = company_val - mean_val
                prompt += f"- {metric}: {company_val:.1f}% (업종평균 {mean_val:.1f}%, 차이 {diff:+.1f}%p, 상위 {pct:.0f}%)\n"

    if strengths:
        prompt += f"\n## 강점 (업종 내 상위권)\n"
        for s in strengths[:3]:
            prompt += f"- {s['metric']}: 상위 {s['percentile']:.0f}%\n"

    if weaknesses:
        prompt += f"\n## 약점 (업종 내 하위권)\n"
        for w in weaknesses[:3]:
            prompt += f"- {w['metric']}: 하위 {100-w['percentile']:.0f}%\n"

    prompt += """
## 요청사항
위 데이터를 바탕으로 다음 내용을 4~5문장으로 분석해주세요:
1. 전반적인 업종 내 경쟁력 평가
2. 가장 두드러진 강점과 그 의미
3. 개선이 필요한 부분과 리스크
4. 동종업계 대비 종합적인 포지셔닝

간결하고 핵심적인 내용만 작성해주세요. 마크다운 형식 없이 일반 텍스트로 작성해주세요.
"""

    try:
        response = generator.client.chat.completions.create(
            model=generator.model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 한국 상장기업 재무 분석 전문가입니다. 업종 내 경쟁 포지션을 분석하여 간결하고 핵심적인 인사이트를 제공합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"업종 비교 분석 중 오류가 발생했습니다."


def generate_category_industry_analysis(
    company_name: str,
    industry: str,
    category: str,
    metrics_comparison: Dict,
    category_position: float
) -> str:
    """
    카테고리별 업종 비교 분석 LLM 생성

    Args:
        company_name: 기업명
        industry: 업종명
        category: 카테고리명 (수익성, 안정성 등)
        metrics_comparison: 해당 카테고리 지표별 비교 데이터
        category_position: 카테고리 업종 내 순위 (백분위)

    Returns:
        str: LLM이 생성한 카테고리별 업종 비교 분석 텍스트
    """
    generator = get_llm_generator()

    rank_text = f"상위 {category_position:.0f}%" if category_position <= 50 else f"하위 {100-category_position:.0f}%"

    prompt = f"""다음 기업의 '{category}' 카테고리 업종 내 위치를 분석해주세요.

## 기업 정보
- 기업명: {company_name}
- 업종: {industry}
- {category} 종합 순위: {rank_text}

## {category} 지표별 업종 비교
"""
    for metric, info in metrics_comparison.items():
        if info:
            company_val = info.get('company')
            mean_val = info.get('industry_mean')
            pct = info.get('percentile', 50)
            if company_val is not None and mean_val is not None:
                diff = company_val - mean_val
                status = "우수" if pct <= 30 else "양호" if pct <= 50 else "보통" if pct <= 70 else "미흡"
                prompt += f"- {metric}: {company_val:.1f}% (업종평균 {mean_val:.1f}%, 차이 {diff:+.1f}%p, {status})\n"

    prompt += f"""
## 요청사항
위 데이터를 바탕으로 {category} 카테고리의 업종 내 경쟁력을 2~3문장으로 분석해주세요:
1. 업종 평균 대비 현재 수준
2. 주목할 만한 지표와 시사점

간결하게 핵심만 작성하세요. 마크다운 없이 일반 텍스트로 작성해주세요.
"""

    try:
        response = generator.client.chat.completions.create(
            model=generator.model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 한국 상장기업 재무 분석 전문가입니다. 업종 내 경쟁 포지션을 분석하여 간결한 인사이트를 제공합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return ""


def generate_timeseries_analysis(
    company_name: str,
    industry: str,
    category: str,
    metrics_data: Dict,
    trend_data: Dict,
    relative_data: Dict
) -> str:
    """
    카테고리별 시계열 데이터 기반 LLM 분석 생성

    과거 4분기 데이터를 기반으로 해당 카테고리의
    재무 상황 변화를 분석합니다.

    Args:
        company_name: 기업명
        industry: 업종명
        category: 카테고리명 (수익성, 안정성 등)
        metrics_data: 지표별 시계열 데이터 {지표명: [q1, q2, q3, q4]}
        trend_data: 트렌드 데이터 {지표명: {yoy, mom, ma4, std4}}
        relative_data: 업종 상대 데이터 {지표명: 상대값}

    Returns:
        str: LLM이 생성한 시계열 분석 텍스트
    """
    generator = get_llm_generator()

    # 프롬프트 생성
    prompt = f"""다음 기업의 '{category}' 카테고리 지표들의 시계열 변화를 분석해주세요.

## 기업 정보
- 기업명: {company_name}
- 업종: {industry}

## {category} 지표 시계열 데이터 (최근 4분기)
"""
    metrics = TARGET_METRICS.get(category, [])

    for metric in metrics:
        values = metrics_data.get(metric, [])
        trend = trend_data.get(metric, {})
        relative = relative_data.get(metric)
        description = METRIC_DESCRIPTION.get(metric, '')
        direction = METRIC_DIRECTION.get(metric, 'higher')
        dir_str = "높을수록 좋음" if direction == 'higher' else "낮을수록 좋음"

        prompt += f"\n### {metric}\n"
        prompt += f"- 의미: {description}\n"
        prompt += f"- 방향성: {dir_str}\n"

        if values:
            values_str = ' → '.join([f'{v:.1f}%' if v is not None else 'N/A' for v in values])
            prompt += f"- 4분기 추이: {values_str}\n"

        if trend:
            ma4 = trend.get('ma4')
            std4 = trend.get('std4')
            yoy = trend.get('yoy')
            mom = trend.get('mom')
            if ma4 is not None:
                prompt += f"- 4분기 평균(MA4): {ma4:.1f}%\n"
            if std4 is not None:
                prompt += f"- 변동성(STD4): {std4:.2f}\n"
            if yoy is not None:
                prompt += f"- 전년동기비(YoY): {yoy:+.1f}%\n"
            if mom is not None:
                prompt += f"- 전분기비(MOM): {mom:+.1f}%\n"

        if relative is not None:
            prompt += f"- 업종 대비: {relative:+.1f}%p\n"

    prompt += """
## 요청사항
위 시계열 데이터를 바탕으로 다음 내용을 3~4문장으로 분석해주세요:
1. 최근 4분기 동안의 전반적인 추세 (개선/악화/유지)
2. 변동성이 큰 지표와 그 의미
3. 업종 대비 상대적 위치
4. 향후 주시해야 할 포인트

간결하고 핵심적인 내용만 작성해주세요. 마크다운 형식 없이 일반 텍스트로 작성해주세요.
"""

    try:
        response = generator.client.chat.completions.create(
            model=generator.model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 한국 상장기업 재무 분석 전문가입니다. 시계열 데이터를 분석하여 간결하고 핵심적인 인사이트를 제공합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"{category} 시계열 분석 중 오류가 발생했습니다."


# =============================================================================
# 비동기 LLM 분석 (AsyncIO 기반 병렬 처리)
# =============================================================================
import asyncio
from openai import AsyncOpenAI


class AsyncLLMOpinionGenerator:
    """
    비동기 LLM 의견 생성 클래스

    asyncio.gather()를 사용하여 여러 LLM 호출을 병렬 처리합니다.
    PDF 생성 시간을 약 2분 → 30-40초로 단축할 수 있습니다.
    """

    def __init__(self, max_concurrent: int = 10):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        self.async_client = AsyncOpenAI(api_key=api_key)
        self.sync_generator = get_llm_generator()  # 기존 동기 generator 재사용
        self.model = "gpt-4o-mini"
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """세마포어로 제한된 비동기 API 호출"""
        async with self.semaphore:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"오류: {str(e)}"

    async def generate_all_parallel(
        self,
        opinion_data: Dict,
        predictions: Dict,
        timeseries_params: Optional[Dict] = None,
        industry_params: Optional[Dict] = None
    ) -> Dict:
        """
        모든 LLM 분석을 병렬로 실행

        Args:
            opinion_data: report_generator에서 준비한 LLM용 데이터
            predictions: AI 예측 결과 (all_predictions)
            timeseries_params: 시계열 분석 파라미터 (선택)
                - company_name: 기업명
                - industry: 업종명
                - category_data: {카테고리: {metrics_data, trend_data, relative_data}}
            industry_params: 업종 비교 파라미터 (선택)
                - company_name, industry
                - category_positions, metrics_comparison
                - strengths, weaknesses
                - category_metrics_comparison

        Returns:
            Dict: 모든 분석 결과
                - opinion: 종합 의견
                - category_analysis: 카테고리별 분석
                - timeseries: {카테고리: 시계열 분석}
                - industry_comparison: 업종 비교 종합
                - category_industry: {카테고리: 카테고리별 업종 비교}
        """
        tasks = {}

        # 1. 종합 의견
        tasks['opinion'] = self._generate_opinion_async(opinion_data)

        # 2. 카테고리별 분석 (5개 카테고리 병렬)
        tasks['category_analysis'] = self._generate_category_analysis_async(opinion_data, predictions)

        # 3. 시계열 분석 (5개 카테고리 병렬)
        if timeseries_params:
            company_name = timeseries_params.get('company_name', '')
            industry = timeseries_params.get('industry', '')
            category_data = timeseries_params.get('category_data', {})

            for category, data in category_data.items():
                tasks[f'timeseries_{category}'] = self._generate_timeseries_async(
                    company_name, industry, category,
                    data.get('metrics_data', {}),
                    data.get('trend_data', {}),
                    data.get('relative_data', {})
                )

        # 4. 업종 비교 분석
        if industry_params:
            company_name = industry_params.get('company_name', '')
            industry = industry_params.get('industry', '')

            # 업종 비교 종합
            if 'category_positions' in industry_params:
                tasks['industry_comparison'] = self._generate_industry_comparison_async(
                    company_name, industry,
                    industry_params['category_positions'],
                    industry_params.get('metrics_comparison', {}),
                    industry_params.get('strengths', []),
                    industry_params.get('weaknesses', [])
                )

            # 카테고리별 업종 비교 (5개 카테고리 병렬)
            category_metrics = industry_params.get('category_metrics_comparison', {})
            category_positions = industry_params.get('category_positions', {})
            for category, metrics in category_metrics.items():
                position = category_positions.get(category)
                if metrics and position is not None:
                    tasks[f'category_industry_{category}'] = self._generate_category_industry_async(
                        company_name, industry, category, metrics, position
                    )

        # 모든 태스크 병렬 실행
        keys = list(tasks.keys())
        coros = list(tasks.values())

        results = await asyncio.gather(*coros, return_exceptions=True)

        output = {}
        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                output[key] = None
            else:
                output[key] = result

        return output

    async def _generate_opinion_async(self, opinion_data: Dict) -> Dict:
        """비동기 종합 의견 생성"""
        prompt = self.sync_generator._build_prompt(opinion_data)
        system_prompt = self.sync_generator._get_system_prompt()
        content = await self._call_api(system_prompt, prompt, max_tokens=3000)
        return self.sync_generator._parse_response(content, opinion_data)

    async def _generate_category_analysis_async(self, opinion_data: Dict, predictions: Dict) -> Dict:
        """비동기 카테고리별 분석 (내부적으로 5개 병렬 실행)"""
        tasks = []
        categories = []

        for category, metrics in TARGET_METRICS.items():
            category_predictions = {m: predictions[m] for m in metrics if m in predictions}
            if category_predictions:
                task = self._generate_single_category_async(category, metrics, opinion_data, category_predictions)
                tasks.append(task)
                categories.append(category)

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {cat: res for cat, res in zip(categories, results) if not isinstance(res, Exception)}

    async def _generate_single_category_async(
        self, category: str, metrics: List, opinion_data: Dict, predictions: Dict
    ) -> Dict:
        """단일 카테고리 비동기 분석"""
        prompt = self.sync_generator._build_category_prompt(category, metrics, opinion_data, predictions)
        system_prompt = self.sync_generator._get_category_system_prompt()
        content = await self._call_api(system_prompt, prompt, max_tokens=2000)
        return self.sync_generator._parse_category_response(content, category, metrics, predictions)

    async def _generate_timeseries_async(
        self, company_name: str, industry: str, category: str,
        metrics_data: Dict, trend_data: Dict, relative_data: Dict
    ) -> str:
        """비동기 시계열 분석"""
        system_prompt = "당신은 한국 상장기업 재무 분석 전문가입니다. 시계열 데이터를 분석하여 간결한 인사이트를 제공합니다."

        prompt = f"'{category}' 카테고리 지표들의 시계열 변화를 분석해주세요.\n\n"
        prompt += f"## 기업 정보\n- 기업명: {company_name}\n- 업종: {industry}\n\n"
        prompt += f"## {category} 지표 시계열 데이터\n"

        for metric in TARGET_METRICS.get(category, []):
            values = metrics_data.get(metric, [])
            trend = trend_data.get(metric, {})
            relative = relative_data.get(metric)
            desc = METRIC_DESCRIPTION.get(metric, '')
            direction = METRIC_DIRECTION.get(metric, 'higher')

            prompt += f"\n### {metric}\n- 의미: {desc}\n- 방향: {'높을수록 좋음' if direction == 'higher' else '낮을수록 좋음'}\n"
            if values:
                prompt += f"- 4분기 추이: {' → '.join([f'{v:.1f}%' if v else 'N/A' for v in values])}\n"
            if trend.get('ma4'):
                prompt += f"- MA4: {trend['ma4']:.1f}%\n"
            if trend.get('yoy'):
                prompt += f"- YoY: {trend['yoy']:+.1f}%\n"
            if relative:
                prompt += f"- 업종 대비: {relative:+.1f}%p\n"

        prompt += "\n3~4문장으로 분석해주세요. 마크다운 없이 일반 텍스트로."

        return await self._call_api(system_prompt, prompt, max_tokens=500)

    async def _generate_industry_comparison_async(
        self, company_name: str, industry: str, category_positions: Dict,
        metrics_comparison: Dict, strengths: List, weaknesses: List
    ) -> str:
        """비동기 업종 비교 종합 분석"""
        system_prompt = "당신은 한국 상장기업 재무 분석 전문가입니다. 업종 내 경쟁 포지션을 분석합니다."

        prompt = f"다음 기업의 업종 내 위치를 분석해주세요.\n\n"
        prompt += f"## 기업 정보\n- 기업명: {company_name}\n- 업종: {industry}\n\n"
        prompt += "## 카테고리별 순위\n"
        for cat, pos in category_positions.items():
            if pos:
                prompt += f"- {cat}: {'상위' if pos <= 50 else '하위'} {pos:.0f}%\n"

        prompt += "\n## 주요 지표 비교\n"
        for metric, info in list(metrics_comparison.items())[:6]:
            if info and info.get('company') is not None:
                prompt += f"- {metric}: {info['company']:.1f}% (업종평균 {info.get('industry_mean', 0):.1f}%)\n"

        if strengths:
            prompt += f"\n## 강점: {', '.join([s['metric'] for s in strengths[:3]])}\n"
        if weaknesses:
            prompt += f"\n## 약점: {', '.join([w['metric'] for w in weaknesses[:3]])}\n"

        prompt += "\n4~5문장으로 분석해주세요. 마크다운 없이."

        return await self._call_api(system_prompt, prompt, max_tokens=600)

    async def _generate_category_industry_async(
        self, company_name: str, industry: str, category: str,
        metrics_comparison: Dict, category_position: float
    ) -> str:
        """비동기 카테고리별 업종 비교"""
        system_prompt = "당신은 한국 상장기업 재무 분석 전문가입니다."

        rank = f"상위 {category_position:.0f}%" if category_position <= 50 else f"하위 {100-category_position:.0f}%"
        prompt = f"'{category}' 카테고리 업종 내 위치를 분석해주세요.\n\n"
        prompt += f"- 기업명: {company_name}\n- 업종: {industry}\n- {category} 순위: {rank}\n\n"

        for metric, info in metrics_comparison.items():
            if info and info.get('company') is not None:
                prompt += f"- {metric}: {info['company']:.1f}% (평균 {info.get('industry_mean', 0):.1f}%)\n"

        prompt += "\n2~3문장으로 분석해주세요. 마크다운 없이."

        return await self._call_api(system_prompt, prompt, max_tokens=300)


# 비동기 싱글톤
_async_generator: Optional[AsyncLLMOpinionGenerator] = None


def get_async_llm_generator() -> AsyncLLMOpinionGenerator:
    """AsyncLLMOpinionGenerator 싱글톤 인스턴스 반환"""
    global _async_generator
    if _async_generator is None:
        _async_generator = AsyncLLMOpinionGenerator()
    return _async_generator


async def generate_all_llm_parallel(
    opinion_data: Dict,
    predictions: Dict,
    timeseries_params: Optional[Dict] = None,
    industry_params: Optional[Dict] = None
) -> Dict:
    """모든 LLM 분석을 병렬로 실행하는 편의 함수"""
    generator = get_async_llm_generator()
    return await generator.generate_all_parallel(
        opinion_data, predictions, timeseries_params, industry_params
    )
