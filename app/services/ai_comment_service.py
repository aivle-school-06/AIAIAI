"""
AI 코멘트 서비스
================

정형 데이터(재무지표, 예측, 건전성, 신호등)와
비정형 데이터(뉴스, 사업보고서)를 종합하여
약 500자 내외의 AI 코멘트를 생성하는 서비스.
"""
import logging
from typing import Dict, Any, Optional

from app.core.data_loader import DataLoader
from app.core.predictor import Predictor
from app.services.health_score_service import HealthScoreService
from app.services.signal_service import SignalService, METRIC_MAPPING
from app.services.news_service import NewsService
from app.services.dart_service import DartService
from app.core.llm_async import AsyncLLMGenerator
from app.exceptions import CompanyNotFoundError

logger = logging.getLogger(__name__)


# 12개 지표 (CFO증감률 제외)
COMMENT_METRICS = list(METRIC_MAPPING.keys())


class AICommentService:
    """
    AI 코멘트 서비스

    - 정형 + 비정형 데이터 수집
    - LLM으로 500자 내외 종합 코멘트 생성
    """

    def __init__(
        self,
        data_loader: DataLoader,
        predictor: Predictor,
        news_service: NewsService,
        dart_service: DartService,
    ):
        self.data_loader = data_loader
        self.predictor = predictor
        self.health_service = HealthScoreService(data_loader, predictor)
        self.signal_service = SignalService(data_loader)
        self.news_service = news_service
        self.dart_service = dart_service
        self.llm_generator = AsyncLLMGenerator()

    async def generate_comment(
        self, company_code: str, period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        AI 코멘트 생성

        Args:
            company_code: 기업 코드 (예: '[005930]')
            period: 분기 (예: '20253'), None이면 최신 분기

        Returns:
            Dict: AI 코멘트 및 관련 정보
        """
        clean_code = company_code.strip('[]')

        # 기업 정보 조회
        df = self.data_loader.df
        company_df = df[df['기업코드'] == company_code]

        if company_df.empty:
            raise CompanyNotFoundError(company_code)

        # 최신 분기 결정
        latest = company_df.sort_values(['년도', '분기']).iloc[-1]
        company_name = latest['기업명']
        industry = latest['업종명']

        if period is None:
            year = int(latest['년도'])
            quarter = str(latest['분기']).replace('Q', '').strip()
            period = f"{year}{quarter}"

        # 데이터 수집 (병렬 처리)
        data = await self._collect_data(company_code, company_name, period)

        # LLM으로 코멘트 생성
        ai_comment = await self._generate_with_llm(
            company_name, industry, period, data
        )

        return {
            'company_code': clean_code,
            'company_name': company_name,
            'industry': industry,
            'period': period,
            'ai_comment': ai_comment,
        }

    async def _collect_data(
        self, company_code: str, company_name: str, period: str
    ) -> Dict[str, Any]:
        """데이터 수집"""
        data = {}

        # 1. 건전성 점수
        try:
            health = self.health_service.get_health_scores(company_code)
            data['health'] = {
                'current_score': health.get('current_score'),
                'predicted_score': health.get('predicted_score'),
                'label': health['quarters'][-2]['label'] if len(health['quarters']) >= 2 else None,
            }
        except Exception as e:
            logger.warning(f"건전성 점수 조회 실패: {e}")
            data['health'] = None

        # 2. 신호등
        try:
            signals = self.signal_service.get_signals(company_code, period)
            data['signals'] = signals.get('signals', {})
            # 요약
            signal_counts = {'green': 0, 'yellow': 0, 'red': 0, 'grey': 0}
            for sig in data['signals'].values():
                signal_counts[sig] += 1
            data['signal_summary'] = signal_counts
        except Exception as e:
            logger.warning(f"신호등 조회 실패: {e}")
            data['signals'] = None
            data['signal_summary'] = None

        # 3. 예측값
        try:
            prediction = self.predictor.predict(company_code)
            predictions = prediction.get('predictions', {})
            # 12개 지표만 필터링
            data['predictions'] = {}
            for metric_kr, metric_en in METRIC_MAPPING.items():
                if metric_kr in predictions:
                    pred = predictions[metric_kr]
                    data['predictions'][metric_en] = {
                        'current': pred.get('current'),
                        'predicted': pred.get('predicted'),
                        'direction': pred.get('direction'),
                    }
        except Exception as e:
            logger.warning(f"예측값 조회 실패: {e}")
            data['predictions'] = None

        # 4. 현재 지표값
        try:
            df = self.data_loader.df
            year = int(period[:4])
            quarter = int(period[4:])
            row = df[
                (df['기업코드'] == company_code) &
                (df['년도'] == year) &
                (df['분기'].astype(str).str.contains(str(quarter)))
            ].iloc[0]

            data['current_metrics'] = {}
            for metric_kr, metric_en in METRIC_MAPPING.items():
                val = row.get(metric_kr)
                if val is not None and not (isinstance(val, float) and val != val):
                    data['current_metrics'][metric_en] = round(float(val), 2)
                else:
                    data['current_metrics'][metric_en] = None
        except Exception as e:
            logger.warning(f"현재 지표 조회 실패: {e}")
            data['current_metrics'] = None

        # 5. 뉴스 분석
        try:
            news_result = await self.news_service.analyze_news_async(
                company_name=company_name,
                max_results=10,
                similarity_threshold=0.6,
                enable_gpt_filter=True,
                enable_summary=False,
            )
            data['news'] = {
                'count': news_result.get('total_count', 0),
                'average_score': news_result.get('average_score'),
                'top_headlines': [n['title'] for n in news_result.get('news', [])[:3]],
            }
        except Exception as e:
            logger.warning(f"뉴스 분석 실패: {e}")
            data['news'] = None

        # 6. 사업보고서
        try:
            clean_code = company_code.strip('[]')
            report_result = await self.dart_service.analyze_report_async(
                corp_code=clean_code
            )
            if report_result and report_result.get('news'):
                report_item = report_result['news'][0]
                data['report'] = {
                    'summary': report_item.get('summary', '')[:500],
                    'score': report_result.get('average_score'),
                }
            else:
                data['report'] = None
        except Exception as e:
            logger.warning(f"사업보고서 분석 실패: {e}")
            data['report'] = None

        return data

    async def _generate_with_llm(
        self,
        company_name: str,
        industry: str,
        period: str,
        data: Dict[str, Any],
    ) -> str:
        """LLM으로 AI 코멘트 생성"""

        # 프롬프트 구성
        user_prompt = self._build_prompt(company_name, industry, period, data)
        system_prompt = """당신은 기업 재무 분석 전문가입니다.
주어진 데이터를 바탕으로 기업의 재무 상태에 대한 종합 코멘트를 작성합니다.
객관적이고 전문적인 톤을 유지하며, 핵심 정보를 간결하게 전달합니다."""

        try:
            response = await self.llm_generator._call_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=800,
                temperature=0.7,
            )
            # 줄바꿈을 <br>로 변환
            return response.strip().replace('\n', '<br>')
        except Exception as e:
            logger.error(f"LLM 코멘트 생성 실패: {e}")
            return self._generate_fallback_comment(company_name, period, data)

    def _build_prompt(
        self,
        company_name: str,
        industry: str,
        period: str,
        data: Dict[str, Any],
    ) -> str:
        """LLM 프롬프트 구성"""

        year = period[:4]
        quarter = period[4:]

        prompt = f"""당신은 기업 재무 분석 전문가입니다. 아래 데이터를 바탕으로 {company_name}({industry})의 {year}년 {quarter}분기 재무 상태에 대한 종합 코멘트를 작성해주세요.

## 작성 지침
- 500자 내외로 작성
- 객관적이고 전문적인 톤 유지
- 핵심 수치와 시사점 포함
- 긍정적/부정적 요소 균형있게 언급
- 마지막에 종합 의견 제시

## 데이터

"""
        # 건전성 점수
        if data.get('health'):
            h = data['health']
            prompt += f"### 재무건전성\n- 현재 점수: {h.get('current_score')}점 ({h.get('label')})\n- 예측 점수: {h.get('predicted_score')}점\n\n"

        # 신호등 요약
        if data.get('signal_summary'):
            s = data['signal_summary']
            prompt += f"### 업종상대 평가 (12개 지표)\n- Green(우수): {s['green']}개, Yellow(보통): {s['yellow']}개, Red(주의): {s['red']}개, Grey(결측): {s['grey']}개\n\n"

        # 주요 지표
        if data.get('current_metrics'):
            m = data['current_metrics']
            prompt += "### 주요 재무지표\n"
            key_metrics = ['ROA', 'ROE', 'DbRatio', 'CurRatio', 'CFO_AsRatio']
            for k in key_metrics:
                if m.get(k) is not None:
                    prompt += f"- {k}: {m[k]}%\n"
            prompt += "\n"

        # 예측
        if data.get('predictions'):
            prompt += "### 다음 분기 예측\n"
            for metric, pred in list(data['predictions'].items())[:5]:
                if pred.get('current') and pred.get('predicted'):
                    direction = '↑' if pred['direction'] == 'improving' else '↓' if pred['direction'] == 'declining' else '→'
                    prompt += f"- {metric}: {pred['current']:.1f}% → {pred['predicted']:.1f}% {direction}\n"
            prompt += "\n"

        # 뉴스
        if data.get('news'):
            n = data['news']
            sentiment = "긍정적" if n.get('average_score', 0) > 0.6 else "부정적" if n.get('average_score', 0) < 0.4 else "중립적"
            prompt += f"### 최근 뉴스\n- 분석 건수: {n['count']}건, 감성: {sentiment} ({n.get('average_score', 0):.2f})\n"
            if n.get('top_headlines'):
                prompt += f"- 주요 헤드라인: {n['top_headlines'][0][:50]}...\n"
            prompt += "\n"

        # 사업보고서
        if data.get('report'):
            r = data['report']
            prompt += f"### 사업보고서 요약\n{r.get('summary', '')[:300]}...\n\n"

        prompt += "## 종합 코멘트를 작성해주세요:"

        return prompt

    def _generate_fallback_comment(
        self,
        company_name: str,
        period: str,
        data: Dict[str, Any],
    ) -> str:
        """LLM 실패 시 기본 코멘트 생성"""
        year = period[:4]
        quarter = period[4:]

        comment = f"{company_name}의 {year}년 {quarter}분기 재무 분석 결과입니다. "

        if data.get('health'):
            h = data['health']
            comment += f"재무건전성 점수는 {h.get('current_score')}점으로 '{h.get('label')}' 등급입니다. "

        if data.get('signal_summary'):
            s = data['signal_summary']
            comment += f"업종 대비 12개 지표 중 {s['green']}개가 우수, {s['yellow']}개가 보통 수준입니다. "

        if data.get('news'):
            n = data['news']
            comment += f"최근 뉴스 {n['count']}건의 평균 감성 점수는 {n.get('average_score', 0):.2f}입니다. "

        comment += "상세 분석은 개별 지표를 확인해주세요."

        return comment
