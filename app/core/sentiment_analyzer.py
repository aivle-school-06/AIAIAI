"""
감성 분석기
==========

HuggingFace 모델을 사용한 금융 뉴스 감성 분석.
서버 시작 시 모델을 로딩하고 싱글톤으로 유지합니다.
"""
import logging
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import Settings

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    금융 뉴스 감성 분석기

    KR-FinBert-SC 모델을 사용하여 금융 텍스트의 감성을 분석합니다.
    - NEG (0): 부정
    - NEU (1): 중립
    - POS (2): 긍정
    """

    LABEL_MAP = {0: "NEG", 1: "NEU", 2: "POS"}

    def __init__(self, settings: Settings):
        """
        감성 분석기 초기화

        Args:
            settings: 앱 설정 (HF 모델명 포함)
        """
        self.settings = settings
        self.model_name = settings.HF_SENTIMENT_MODEL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading sentiment model: {self.model_name}")
        logger.info(f"Using device: {self.device}")

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )
        logger.info("Tokenizer loaded")

        # 모델 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        ).to(self.device)
        self.model.eval()
        logger.info("Model loaded and set to eval mode")

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        텍스트 감성 분석

        Args:
            texts: 분석할 텍스트 리스트

        Returns:
            List[Dict]: 각 텍스트의 감성 분석 결과
                - label_id: 라벨 ID (0, 1, 2)
                - label: 라벨명 (NEG, NEU, POS)
                - score: 예측 확률
                - probs: 전체 확률 분포 [neg, neu, pos]
        """
        if isinstance(texts, str):
            texts = [texts]

        # 토큰화
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        # 추론
        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy().tolist()
            probs = probs.cpu().numpy().tolist()

        # 결과 구성
        results = []
        for i, p in enumerate(preds):
            results.append({
                "label_id": int(p),
                "label": self.LABEL_MAP.get(p, str(p)),
                "score": float(probs[i][p]),
                "probs": probs[i]
            })

        return results

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        단일 텍스트 감성 분석

        Args:
            text: 분석할 텍스트

        Returns:
            Dict: 감성 분석 결과
        """
        return self.predict([text])[0]

    def get_positive_score(self, text: str) -> float:
        """
        긍정 점수만 반환

        Args:
            text: 분석할 텍스트

        Returns:
            float: 긍정 확률 (0~1)
        """
        result = self.predict_single(text)
        return result["probs"][2]  # POS index
