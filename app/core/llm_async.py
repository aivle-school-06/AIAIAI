"""
비동기 LLM 의견 생성 모듈
========================

기존 llm_opinion.py의 비동기 버전.
asyncio.gather()를 사용하여 여러 LLM 호출을 병렬 처리합니다.

성능 개선:
- 기존: 17회 순차 호출 → 약 2분
- 개선: 17회 병렬 호출 → 약 10-20초 (예상)
"""
import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache

from openai import AsyncOpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# 상수 정의 (기존 config.py에서 가져옴)
# =============================================================================
TARGET_METRICS = {
    '수익성': ['ROA', 'ROE', '매출액영업이익률'],
    '안정성': ['부채비율', '차입금의존도'],
    '차입금': ['순차입금비율', '이자보상배율'],
    '유동성': ['유동비율', '당좌비율'],
    '현금흐름': ['CFO_자산비율', 'FCF_자산비율', 'CAPEX_자산비율', '영업현금흐름비율'],
}

METRIC_DESCRIPTION = {
    'ROA': '자산 대비 순이익률로, 기업의 자산 활용 효율성을 나타냄',
    'ROE': '자기자본 대비 순이익률로, 주주 투자 수익률을 나타냄',
    '매출액영업이익률': '매출 대비 영업이익 비율로, 본업의 수익성을 나타냄',
    '부채비율': '자기자본 대비 부채 비율로, 재무 안정성을 나타냄',
    '차입금의존도': '총자산 대비 차입금 비율로, 외부 자금 의존도를 나타냄',
    '순차입금비율': '순차입금(차입금-현금) 대비 자기자본 비율',
    '이자보상배율': '영업이익 대비 이자비용 배율로, 이자 지급 능력을 나타냄',
    '유동비율': '유동부채 대비 유동자산 비율로, 단기 지급 능력을 나타냄',
    '당좌비율': '재고자산 제외 유동자산의 유동부채 대비 비율',
    'CFO_자산비율': '영업활동현금흐름의 자산 대비 비율',
    'FCF_자산비율': '잉여현금흐름의 자산 대비 비율',
    'CAPEX_자산비율': '자본적 지출의 자산 대비 비율',
    '영업현금흐름비율': '영업활동현금흐름의 영업이익 대비 비율',
}

METRIC_DIRECTION = {
    'ROA': 'higher',
    'ROE': 'higher',
    '매출액영업이익률': 'higher',
    '부채비율': 'lower',
    '차입금의존도': 'lower',
    '순차입금비율': 'lower',
    '이자보상배율': 'higher',
    '유동비율': 'higher',
    '당좌비율': 'higher',
    'CFO_자산비율': 'higher',
    'FCF_자산비율': 'higher',
    'CAPEX_자산비율': 'context',
    '영업현금흐름비율': 'higher',
}


class AsyncLLMGenerator:
    """
    비동기 LLM 의견 생성 클래스

    OpenAI AsyncClient를 사용하여 여러 LLM 호출을 병렬 처리합니다.
    """

    def __init__(self, max_concurrent: int = 10):
        """
        초기화

        Args:
            max_concurrent: 동시 실행 가능한 최대 API 호출 수
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        단일 API 호출 (세마포어로 동시 호출 수 제한)

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            max_tokens: 최대 토큰 수
            temperature: 창의성 정도

        Returns:
            str: GPT 응답 텍스트
        """
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
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
                logger.error(f"LLM API 호출 실패: {e}")
                return f"오류: {str(e)}"

    # =========================================================================
    # 종합 의견 생성
    # =========================================================================
    async def generate_opinion(self, opinion_data: Dict) -> Dict:
        """종합 의견 생성 (비동기)"""
        system_prompt = self._get_opinion_system_prompt()
        user_prompt = self._build_opinion_prompt(opinion_data)

        content = await self._call_api(system_prompt, user_prompt, max_tokens=3000)
        return self._parse_opinion_response(content, opinion_data)

    def _get_opinion_system_prompt(self) -> str:
        """종합 의견용 시스템 프롬프트"""
        return """당신은 한국 상장기업 재무 분석 전문가이자 AI 모델 해석 전문가입니다.
주어진 재무 데이터와 AI 예측 결과(SHAP 분석 포함)를 분석하여 종합 의견을 작성해주세요.

응답 형식:

[전문가용]
(금융/투자 전문가를 위한 상세 분석. 전문 용어 사용 가능. 5~7문장)

[비전문가용]
(일반인을 위한 쉬운 설명. 비유와 쉬운 표현 사용)
- 돈을 잘 버나요? → (수익성 설명)
- 빚이 많나요? → (안정성 설명)
- 현금은 충분한가요? → (현금흐름 설명)
- 앞으로 전망은? → (AI 예측 기반 전망)

[AI 예측 근거]
(SHAP 분석 결과를 바탕으로 AI가 왜 이렇게 예측했는지 설명)

[핵심 포인트]
• (핵심 사항 1)
• (핵심 사항 2)
• (핵심 사항 3)

[주의 사항]
(리스크가 있다면 설명)
"""

    def _build_opinion_prompt(self, data: Dict) -> str:
        """종합 의견용 프롬프트 생성"""
        company = data.get('company', {})
        grades = data.get('grades', {})
        current = data.get('current_metrics', {})
        relative = data.get('relative_metrics', {})
        trend = data.get('trend', {})
        prediction = data.get('prediction', {})
        risk_signals = data.get('risk_signals', [])
        shap_analysis = data.get('shap_analysis', {})

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
        key_metrics = ['ROA', 'ROE', '매출액영업이익률', '부채비율', '유동비율', 'CFO_자산비율']
        for metric in key_metrics:
            value = current.get(metric)
            rel = relative.get(metric)
            if value is not None:
                rel_str = f" (업종 대비 {rel:+.1f}%p)" if rel is not None else ""
                prompt += f"- {metric}: {value:.2f}%{rel_str}\n"

        prompt += "\n## 추세 분석\n"
        for metric in key_metrics:
            t = trend.get(metric, {})
            yoy = t.get('yoy')
            mom = t.get('mom')
            if yoy is not None and mom is not None:
                direction = "▲ 개선" if yoy > 0 and mom > 0 else "▼ 악화" if yoy < 0 and mom < 0 else "→ 혼조"
                prompt += f"- {metric}: YoY {yoy:+.1f}%, QoQ {mom:+.1f}% ({direction})\n"

        prompt += f"""
## AI 예측 (다음 분기)
- 전체 전망: {prediction.get('outlook', 'N/A')}
- 개선 예측 지표: {', '.join(prediction.get('improving', [])) or '없음'}
- 악화 예측 지표: {', '.join(prediction.get('declining', [])) or '없음'}
"""

        if shap_analysis:
            prompt += "\n## AI 예측 근거 (SHAP 분석)\n"
            for metric, analysis in list(shap_analysis.items())[:3]:
                if not analysis:
                    continue
                prompt += f"### {metric}\n"
                pos_factors = analysis.get('positive_factors', [])[:2]
                neg_factors = analysis.get('negative_factors', [])[:2]
                for f in pos_factors:
                    prompt += f"  ↑ {f.get('description', f.get('feature', ''))} (기여도: {f.get('shap_value', 0):+.3f})\n"
                for f in neg_factors:
                    prompt += f"  ↓ {f.get('description', f.get('feature', ''))} (기여도: {f.get('shap_value', 0):+.3f})\n"

        if risk_signals:
            prompt += "\n## 리스크 시그널\n"
            for signal in risk_signals[:3]:
                prompt += f"- {signal.get('message', '')}\n"

        return prompt

    def _parse_opinion_response(self, content: str, opinion_data: Dict) -> Dict:
        """종합 의견 응답 파싱"""
        result = {
            'expert': '',
            'simple': '',
            'key_points': [],
            'risk_summary': None,
            'xai_explanation': None,
            'raw_response': content,
        }

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
                points = []
                for line in points_text.split('\n'):
                    line = line.strip()
                    if line.startswith('•') or line.startswith('-'):
                        points.append(line.lstrip('•-').strip())
                result['key_points'] = points
            elif section.startswith('주의 사항]'):
                result['risk_summary'] = section.replace('주의 사항]', '').strip()

        return result

    # =========================================================================
    # 카테고리별 분석 (병렬 처리)
    # =========================================================================
    async def generate_category_analysis(
        self, opinion_data: Dict, predictions: Dict
    ) -> Dict:
        """
        카테고리별 XAI + LLM 분석 생성 (병렬 처리)

        5개 카테고리를 동시에 분석합니다.
        """
        tasks = []
        categories = []

        for category, metrics in TARGET_METRICS.items():
            category_predictions = {
                m: predictions[m] for m in metrics if m in predictions
            }
            if category_predictions:
                task = self._generate_single_category_analysis(
                    category, metrics, opinion_data, category_predictions
                )
                tasks.append(task)
                categories.append(category)

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for category, result in zip(categories, results):
            if isinstance(result, Exception):
                logger.error(f"카테고리 분석 실패 ({category}): {result}")
                output[category] = self._generate_fallback_category(category)
            else:
                output[category] = result

        return output

    async def _generate_single_category_analysis(
        self,
        category: str,
        metrics: List[str],
        opinion_data: Dict,
        predictions: Dict
    ) -> Dict:
        """단일 카테고리 분석 (비동기)"""
        system_prompt = self._get_category_system_prompt()
        user_prompt = self._build_category_prompt(category, metrics, opinion_data, predictions)

        content = await self._call_api(system_prompt, user_prompt, max_tokens=2000)
        return self._parse_category_response(content, category, metrics, predictions)

    def _get_category_system_prompt(self) -> str:
        """카테고리 분석용 시스템 프롬프트"""
        return """당신은 한국 상장기업 재무 분석 전문가입니다.
주어진 카테고리의 재무지표와 AI 예측 결과를 분석하여 인사이트를 제공해주세요.

응답 형식:

[카테고리 종합]
(해당 카테고리 전체에 대한 종합 분석. 3~4문장)

[전망]
(positive/negative/neutral 중 하나)

[지표별 분석]

{지표명1}:
- 현황: (현재 상태에 대한 1문장 평가)
- XAI 해석: (SHAP 요인 해석)
- 시사점: (비즈니스적 의미)

{지표명2}:
...

[핵심 메시지]
• (주요 인사이트)
• (실행 가능한 권고사항)
"""

    def _build_category_prompt(
        self,
        category: str,
        metrics: List[str],
        opinion_data: Dict,
        predictions: Dict
    ) -> str:
        """카테고리 분석용 프롬프트"""
        company = opinion_data.get('company', {})
        current_metrics = opinion_data.get('current_metrics', {})
        relative_metrics = opinion_data.get('relative_metrics', {})
        trend = opinion_data.get('trend', {})

        prompt = f"""'{category}' 카테고리 지표들을 분석해주세요.

## 기업 정보
- 기업명: {company.get('name', 'N/A')}
- 업종: {company.get('industry', 'N/A')}

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

                t = trend.get(metric, {})
                yoy = t.get('yoy')
                mom = t.get('mom')
                if yoy is not None and mom is not None:
                    prompt += f"- 추세: YoY {yoy:+.1f}%, QoQ {mom:+.1f}%\n"

                pred = predictions.get(metric, {})
                if pred and 'predicted' in pred:
                    predicted = pred.get('predicted')
                    change = pred.get('change')
                    prompt += f"- AI 예측: {predicted:.2f}% (변화: {change:+.2f}%p)\n"

                    shap = pred.get('shap_analysis', {})
                    pos = shap.get('positive_factors', [])[:2]
                    neg = shap.get('negative_factors', [])[:2]
                    if pos:
                        prompt += "  상승요인: " + ", ".join([f.get('feature', '') for f in pos]) + "\n"
                    if neg:
                        prompt += "  하락요인: " + ", ".join([f.get('feature', '') for f in neg]) + "\n"

        return prompt

    def _parse_category_response(
        self, content: str, category: str, metrics: List[str], predictions: Dict
    ) -> Dict:
        """카테고리 분석 응답 파싱"""
        result = {
            'summary': '',
            'outlook': 'neutral',
            'key_messages': [],
            'metrics': {},
            'raw_response': content,
        }

        sections = content.split('[')
        for section in sections:
            if section.startswith('카테고리 종합]'):
                result['summary'] = section.replace('카테고리 종합]', '').strip().split('\n\n')[0]
            elif section.startswith('전망]'):
                outlook_text = section.replace('전망]', '').strip().lower()
                if 'positive' in outlook_text or '긍정' in outlook_text:
                    result['outlook'] = 'positive'
                elif 'negative' in outlook_text or '부정' in outlook_text:
                    result['outlook'] = 'negative'
            elif section.startswith('지표별 분석]'):
                result['metrics'] = self._parse_metrics_text(section, metrics, predictions)
            elif section.startswith('핵심 메시지]'):
                messages = [
                    line.lstrip('•-').strip()
                    for line in section.replace('핵심 메시지]', '').strip().split('\n')
                    if line.strip().startswith(('•', '-'))
                ]
                result['key_messages'] = messages[:3]

        return result

    def _parse_metrics_text(self, text: str, metrics: List[str], predictions: Dict) -> Dict:
        """지표별 텍스트 파싱"""
        result = {}
        for metric in metrics:
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
        return result

    def _generate_fallback_category(self, category: str) -> Dict:
        """폴백 카테고리 결과"""
        return {
            'summary': f'{category} 분석 데이터',
            'outlook': 'neutral',
            'key_messages': [],
            'metrics': {},
        }

    # =========================================================================
    # 시계열 분석 (병렬 처리)
    # =========================================================================
    async def generate_all_timeseries_analysis(
        self,
        company_name: str,
        industry: str,
        all_metrics_data: Dict[str, Dict],
        all_trend_data: Dict[str, Dict],
        all_relative_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        모든 카테고리의 시계열 분석 (병렬 처리)

        Args:
            all_metrics_data: {카테고리: {지표: [값들]}}
            all_trend_data: {지표: {yoy, mom, ...}}
            all_relative_data: {지표: 상대값}

        Returns:
            Dict[str, str]: {카테고리: 분석 텍스트}
        """
        tasks = []
        categories = []

        for category, metrics in TARGET_METRICS.items():
            metrics_data = all_metrics_data.get(category, {})
            if not metrics_data:
                continue

            trend_data = {m: all_trend_data.get(m, {}) for m in metrics}
            relative_data = {m: all_relative_data.get(m) for m in metrics}

            task = self.generate_timeseries_analysis(
                company_name, industry, category,
                metrics_data, trend_data, relative_data
            )
            tasks.append(task)
            categories.append(category)

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for category, result in zip(categories, results):
            if isinstance(result, Exception):
                logger.error(f"시계열 분석 실패 ({category}): {result}")
                output[category] = f"{category} 시계열 분석 중 오류 발생"
            else:
                output[category] = result

        return output

    async def generate_timeseries_analysis(
        self,
        company_name: str,
        industry: str,
        category: str,
        metrics_data: Dict,
        trend_data: Dict,
        relative_data: Dict
    ) -> str:
        """단일 카테고리 시계열 분석 (비동기)"""
        system_prompt = "당신은 한국 상장기업 재무 분석 전문가입니다. 시계열 데이터를 분석하여 간결한 인사이트를 제공합니다."

        prompt = f"""'{category}' 카테고리 지표들의 시계열 변화를 분석해주세요.

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
                if trend.get('ma4') is not None:
                    prompt += f"- 4분기 평균(MA4): {trend['ma4']:.1f}%\n"
                if trend.get('yoy') is not None:
                    prompt += f"- 전년동기비(YoY): {trend['yoy']:+.1f}%\n"
                if trend.get('mom') is not None:
                    prompt += f"- 전분기비(MOM): {trend['mom']:+.1f}%\n"

            if relative is not None:
                prompt += f"- 업종 대비: {relative:+.1f}%p\n"

        prompt += """
## 요청사항
위 시계열 데이터를 바탕으로 다음 내용을 3~4문장으로 분석해주세요:
1. 최근 4분기 동안의 전반적인 추세
2. 변동성이 큰 지표와 의미
3. 업종 대비 위치
4. 주시해야 할 포인트

마크다운 없이 일반 텍스트로 작성해주세요.
"""
        return await self._call_api(system_prompt, prompt, max_tokens=500)

    # =========================================================================
    # 업종 비교 분석 (병렬 처리)
    # =========================================================================
    async def generate_industry_comparison_analysis(
        self,
        company_name: str,
        industry: str,
        category_positions: Dict,
        metrics_comparison: Dict,
        strengths: List,
        weaknesses: List
    ) -> str:
        """업종 비교 종합 분석 (비동기)"""
        system_prompt = "당신은 한국 상장기업 재무 분석 전문가입니다. 업종 내 경쟁 포지션을 분석하여 간결한 인사이트를 제공합니다."

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

        prompt += "\n## 주요 지표 비교\n"
        for metric, info in list(metrics_comparison.items())[:6]:
            if info:
                company_val = info.get('company')
                mean_val = info.get('industry_mean')
                pct = info.get('percentile')
                if company_val is not None and mean_val is not None:
                    diff = company_val - mean_val
                    prompt += f"- {metric}: {company_val:.1f}% (업종평균 {mean_val:.1f}%, 차이 {diff:+.1f}%p, 상위 {pct:.0f}%)\n"

        if strengths:
            prompt += "\n## 강점\n"
            for s in strengths[:3]:
                prompt += f"- {s['metric']}: 상위 {s['percentile']:.0f}%\n"

        if weaknesses:
            prompt += "\n## 약점\n"
            for w in weaknesses[:3]:
                prompt += f"- {w['metric']}: 하위 {100-w['percentile']:.0f}%\n"

        prompt += """
## 요청사항
4~5문장으로 분석해주세요:
1. 전반적인 업종 내 경쟁력
2. 가장 두드러진 강점
3. 개선 필요 부분
4. 종합적인 포지셔닝

마크다운 없이 일반 텍스트로 작성해주세요.
"""
        return await self._call_api(system_prompt, prompt, max_tokens=600)

    async def generate_all_category_industry_analysis(
        self,
        company_name: str,
        industry: str,
        category_metrics_comparison: Dict[str, Dict],
        category_positions: Dict[str, float]
    ) -> Dict[str, str]:
        """
        모든 카테고리의 업종 비교 분석 (병렬 처리)

        Args:
            category_metrics_comparison: {카테고리: {지표: 비교데이터}}
            category_positions: {카테고리: 백분위}

        Returns:
            Dict[str, str]: {카테고리: 분석 텍스트}
        """
        tasks = []
        categories = []

        for category in TARGET_METRICS.keys():
            metrics_comparison = category_metrics_comparison.get(category, {})
            position = category_positions.get(category)

            if not metrics_comparison or position is None:
                continue

            task = self.generate_category_industry_analysis(
                company_name, industry, category,
                metrics_comparison, position
            )
            tasks.append(task)
            categories.append(category)

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for category, result in zip(categories, results):
            if isinstance(result, Exception):
                logger.error(f"카테고리 업종 비교 실패 ({category}): {result}")
                output[category] = ""
            else:
                output[category] = result

        return output

    async def generate_category_industry_analysis(
        self,
        company_name: str,
        industry: str,
        category: str,
        metrics_comparison: Dict,
        category_position: float
    ) -> str:
        """단일 카테고리 업종 비교 분석 (비동기)"""
        system_prompt = "당신은 한국 상장기업 재무 분석 전문가입니다. 업종 내 경쟁 포지션을 분석하여 간결한 인사이트를 제공합니다."

        rank_text = f"상위 {category_position:.0f}%" if category_position <= 50 else f"하위 {100-category_position:.0f}%"

        prompt = f"""'{category}' 카테고리 업종 내 위치를 분석해주세요.

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
{category} 카테고리의 업종 내 경쟁력을 2~3문장으로 분석해주세요:
1. 업종 평균 대비 현재 수준
2. 주목할 만한 지표와 시사점

마크다운 없이 일반 텍스트로 작성해주세요.
"""
        return await self._call_api(system_prompt, prompt, max_tokens=300)

    # =========================================================================
    # 전체 분석 병렬 실행
    # =========================================================================
    async def generate_all_analysis(
        self,
        opinion_data: Dict,
        predictions: Dict,
        timeseries_params: Optional[Dict] = None,
        industry_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        모든 LLM 분석을 병렬로 실행

        기존 17회 순차 호출 → 1회 병렬 호출로 변경

        Args:
            opinion_data: 종합 의견용 데이터
            predictions: 예측 결과
            timeseries_params: 시계열 분석 파라미터 (선택)
            industry_params: 업종 비교 파라미터 (선택)

        Returns:
            Dict: 모든 분석 결과
        """
        tasks = {
            'opinion': self.generate_opinion(opinion_data),
            'category_analysis': self.generate_category_analysis(opinion_data, predictions),
        }

        if timeseries_params:
            tasks['timeseries'] = self.generate_all_timeseries_analysis(**timeseries_params)

        if industry_params:
            company_name = industry_params.get('company_name')
            industry = industry_params.get('industry')

            if 'category_positions' in industry_params:
                tasks['industry_comparison'] = self.generate_industry_comparison_analysis(
                    company_name=company_name,
                    industry=industry,
                    category_positions=industry_params['category_positions'],
                    metrics_comparison=industry_params['metrics_comparison'],
                    strengths=industry_params.get('strengths', []),
                    weaknesses=industry_params.get('weaknesses', [])
                )

            if 'category_metrics_comparison' in industry_params:
                tasks['category_industry'] = self.generate_all_category_industry_analysis(
                    company_name=company_name,
                    industry=industry,
                    category_metrics_comparison=industry_params['category_metrics_comparison'],
                    category_positions=industry_params.get('category_positions', {})
                )

        # 모든 태스크 병렬 실행
        keys = list(tasks.keys())
        coros = list(tasks.values())

        results = await asyncio.gather(*coros, return_exceptions=True)

        output = {}
        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                logger.error(f"분석 실패 ({key}): {result}")
                output[key] = None
            else:
                output[key] = result

        return output


# =============================================================================
# 싱글톤 인스턴스
# =============================================================================
_async_generator: Optional[AsyncLLMGenerator] = None


def get_async_llm_generator() -> AsyncLLMGenerator:
    """AsyncLLMGenerator 싱글톤 인스턴스 반환"""
    global _async_generator
    if _async_generator is None:
        _async_generator = AsyncLLMGenerator()
    return _async_generator


# =============================================================================
# 편의 함수 (동기 래퍼)
# =============================================================================
def generate_opinion_sync(opinion_data: Dict) -> Dict:
    """종합 의견 생성 (동기 래퍼)"""
    generator = get_async_llm_generator()
    return asyncio.run(generator.generate_opinion(opinion_data))


def generate_all_analysis_sync(
    opinion_data: Dict,
    predictions: Dict,
    timeseries_params: Optional[Dict] = None,
    industry_params: Optional[Dict] = None
) -> Dict[str, Any]:
    """모든 분석 병렬 실행 (동기 래퍼)"""
    generator = get_async_llm_generator()
    return asyncio.run(generator.generate_all_analysis(
        opinion_data, predictions, timeseries_params, industry_params
    ))
