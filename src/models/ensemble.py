"""Ансамбль моделей классификации сообщений домовых чатов.

Объединяет:
1. BaselineClassifier (TF-IDF + LogisticRegression) — быстрый и точный на служебных сообщениях и лексике
2. TransformerPipeline (MultiTaskRuBertTiny) — глубокий контекст, семантика и скрытые интонации
3. Модуль разрешения сложных кейсов (hardcases):
   - обсценная лексика и обфусцированный мат
   - сарказм и ирония (смех при авариях и претензиях)
   - смешанная тональность (mixed)
   - экспрессия (CAPS LOCK, восклицания)

Строго соблюдает доменные инварианты разметки:
- is_service == 1 -> sentiment = 'neutral', aggression = 0
- sentiment != 'negative' -> aggression = 0
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

import re

from src.features.preprocessor import TextPreprocessor
from src.models.baseline import BaselineClassifier
from src.models.transformer import HAS_TORCH, REV_SENTIMENT_MAP, TransformerPipeline

SENTIMENT_CLASSES = ("negative", "neutral", "positive")

# Паттерны прямых личных оскорблений, угроз и явной агрессии для жесткого оверрайда
INSULT_OVERRIDE_PATTERNS = [
    re.compile(
        r"\b(?:конченн[а-яё]*|ублюд[а-яё]*|дебил[а-яё]*|придур[а-яё]*|идиот[а-яё]*|"
        r"тварь[а-яё]*|мраз[а-яё]*|гнид[а-яё]*|чмо[а-яё]*|недоносок[а-яё]*|упыр[а-яё]*|"
        r"отребь[а-яё]*|быдл[а-яё]*|сволоч[а-яё]*|руко[@оoж\*\.]+[пp][ыеа-яё]*|клоун[а-яё]*|маргинал[а-яё]*)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:голову|мозги)[^.!?\n]{0,25}включи|включи[^.!?\n]{0,25}(?:голову|мозги)|"
        r"рот\s+закрой|закрой\s+рот|заткни[а-яё]*|молчал[а-яё]*\s+бы|недержание\s+во\s+рту|метать\s+бисер|🐖)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:тебе\s+п[\.\*a-яё]*ц|слышишь\s+меня|морду\s+набью|руки\s+оторв[а-яё]*|посад[а-яё]*\s+этого)",
        re.IGNORECASE,
    ),
]


class EnsembleClassifier:
    """Ансамбль классификаторов с взвешенным голосованием и обработкой хардкейсов."""

    def __init__(
        self,
        baseline: BaselineClassifier | None = None,
        transformer: TransformerPipeline | None = None,
        preprocessor: TextPreprocessor | None = None,
        weight_service: float = 0.15,
        weight_sentiment: float = 0.45,
        weight_aggression: float = 0.50,
        aggression_threshold: float = 0.65,
        obscene_sent_boost: float = 0.20,
        obscene_agg_boost: float = 0.15,
    ) -> None:
        """Инициализация ансамбля.

        Args:
            baseline: Экземпляр BaselineClassifier.
            transformer: Экземпляр TransformerPipeline (опционально).
            preprocessor: Экземпляр TextPreprocessor.
            weight_service: Вес трансформера для задачи is_service (остальное - бейслайн).
            weight_sentiment: Вес трансформера для задачи sentiment.
            weight_aggression: Вес трансформера для задачи aggression.
            aggression_threshold: Порог вероятности для бинарной детекции агрессии.
            obscene_sent_boost: Добавка к вероятности негатива при наличии мата.
            obscene_agg_boost: Добавка к вероятности агрессии при наличии мата.
        """
        self.baseline = baseline
        self.transformer = transformer
        self.preprocessor = preprocessor or TextPreprocessor()
        self.weight_service = float(weight_service)
        self.weight_sentiment = float(weight_sentiment)
        self.weight_aggression = float(weight_aggression)
        self.aggression_threshold = float(aggression_threshold)
        self.obscene_sent_boost = float(obscene_sent_boost)
        self.obscene_agg_boost = float(obscene_agg_boost)

    def fit(
        self,
        texts: Sequence[str],
        is_service: Sequence[int],
        sentiment: Sequence[str],
        aggression: Sequence[int],
    ) -> EnsembleClassifier:
        """Обучение бейслайна внутри ансамбля."""
        if self.baseline is None:
            self.baseline = BaselineClassifier()
        self.baseline.fit(texts, is_service, sentiment, aggression)
        return self

    def _get_baseline_probas(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Получение вероятностей от бейслайна."""
        if self.baseline is None or not getattr(self.baseline, "is_fitted", False):
            raise RuntimeError("Бейслайн не обучен или отсутствует в ансамбле.")

        clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
        x_vec = self.baseline.vectorizer.transform(clean_texts)

        p_svc = self.baseline.clf_service.predict_proba(x_vec)
        p_sent = self.baseline.clf_sentiment.predict_proba(x_vec)
        p_agg = self.baseline.clf_aggression.predict_proba(x_vec)

        # Выравнивание классов тональности к порядку ['negative', 'neutral', 'positive']
        base_classes = list(self.baseline.clf_sentiment.classes_)
        if base_classes != list(SENTIMENT_CLASSES):
            idx_order = [base_classes.index(c) for c in SENTIMENT_CLASSES]
            p_sent = p_sent[:, idx_order]

        return {"service": p_svc, "sentiment": p_sent, "aggression": p_agg}

    def _get_transformer_probas(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        max_length: int = 128,
    ) -> dict[str, np.ndarray] | None:
        """Получение вероятностей от трансформера."""
        if not HAS_TORCH or self.transformer is None or self.transformer.model is None:
            return None

        import torch

        self.transformer.model.train(False)
        clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
        all_svc, all_sent, all_agg = [], [], []

        with torch.no_grad():
            for i in range(0, len(clean_texts), batch_size):
                batch = clean_texts[i:i + batch_size]
                enc = self.transformer.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.transformer.device)
                att_mask = enc["attention_mask"].to(self.transformer.device)
                out = self.transformer.model(input_ids=input_ids, attention_mask=att_mask)

                all_svc.append(torch.softmax(out["service_logits"], dim=-1).cpu().numpy())
                all_sent.append(torch.softmax(out["sentiment_logits"], dim=-1).cpu().numpy())
                all_agg.append(torch.softmax(out["aggression_logits"], dim=-1).cpu().numpy())

        return {
            "service": np.vstack(all_svc),
            "sentiment": np.vstack(all_sent),
            "aggression": np.vstack(all_agg),
        }

    def predict_proba(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Взвешенное усреднение вероятностей с применением эвристик хардкейсов."""
        base_probs = self._get_baseline_probas(texts)
        trans_probs = self._get_transformer_probas(texts)

        # 1. Базовое смешивание вероятностей
        if trans_probs is not None:
            w_svc = self.weight_service
            w_sent = self.weight_sentiment
            w_agg = self.weight_aggression

            p_svc = (1 - w_svc) * base_probs["service"][:, 1] + w_svc * trans_probs["service"][:, 1]
            p_sent = (1 - w_sent) * base_probs["sentiment"] + w_sent * trans_probs["sentiment"]
            p_agg = (1 - w_agg) * base_probs["aggression"][:, 1] + w_agg * trans_probs["aggression"][:, 1]
        else:
            p_svc = base_probs["service"][:, 1].copy()
            p_sent = base_probs["sentiment"].copy()
            p_agg = base_probs["aggression"][:, 1].copy()

        # 2. Анализ сложных паттернов (hardcases: мат, сарказм, экспрессия)
        df_texts = pd.DataFrame({"text": list(texts)})
        feats = self.preprocessor.extract_features(df_texts)

        has_obscene = feats["has_obscene"].to_numpy()
        has_emoji_laugh = feats["has_emoji_laugh"].to_numpy()
        has_excl = feats["has_excl_high"].to_numpy()
        has_quest = feats["has_quest_high"].to_numpy()

        # Хардкейс 1: Обсценная лексика -> усиление негатива и агрессии
        if self.obscene_sent_boost > 0:
            p_sent[has_obscene == 1, 0] += self.obscene_sent_boost
        if self.obscene_agg_boost > 0:
            p_agg[has_obscene == 1] += self.obscene_agg_boost

        # Хардкейс 2: Сарказм (смех при негативном контексте или претензии с вопросами/восклицаниями)
        sarcasm_mask = (has_emoji_laugh == 1) & (
            (base_probs["sentiment"][:, 0] > 0.25) | (has_excl == 1) | (has_quest == 1)
        )
        p_sent[sarcasm_mask, 0] += 0.15

        # Нормализация распределения вероятностей тональности
        sent_sum = np.sum(p_sent, axis=1, keepdims=True)
        p_sent = p_sent / np.maximum(sent_sum, 1e-9)

        return {
            "service": p_svc,
            "sentiment": p_sent,
            "aggression": np.clip(p_agg, 0.0, 1.0),
        }

    def predict(self, texts: Sequence[str]) -> pd.DataFrame:
        """Предсказание меток с гарантированным соблюдением инвариантов."""
        probs = self.predict_proba(texts)

        pred_svc = (probs["service"] >= 0.5).astype(int)
        sent_indices = np.argmax(probs["sentiment"], axis=1)
        pred_sent = np.array([REV_SENTIMENT_MAP[idx] for idx in sent_indices], dtype=object)
        pred_agg = (probs["aggression"] >= self.aggression_threshold).astype(int)

        # Hard-rule override: прямые личные оскорбления и явные угрозы
        # Принудительно переводят тональность в negative и выставляют aggression = 1
        for i, text in enumerate(texts):
            if pred_svc[i] == 0:
                t = str(text) if pd.notna(text) else ""
                if any(p.search(t) for p in INSULT_OVERRIDE_PATTERNS):
                    pred_sent[i] = "negative"
                    pred_agg[i] = 1

        # Соблюдение жестких инвариантов разметки
        pred_sent, pred_agg = self._apply_invariants(pred_svc, pred_sent, pred_agg)

        return pd.DataFrame({
            "is_service": pred_svc,
            "sentiment": pred_sent,
            "aggression": pred_agg,
        })

    def predict_messages(self, messages_df: pd.DataFrame) -> pd.DataFrame:
        """Инференс для DataFrame сообщений с колонками 'message_id' и 'text'."""
        if "message_id" not in messages_df.columns:
            raise ValueError("Входной DataFrame должен содержать колонку 'message_id'")
        if "text" not in messages_df.columns:
            raise ValueError("Входной DataFrame должен содержать колонку 'text'")

        preds = self.predict(messages_df["text"])
        preds.insert(0, "message_id", messages_df["message_id"].to_numpy())
        return preds

    @staticmethod
    def _apply_invariants(
        is_service: np.ndarray,
        sentiment: np.ndarray,
        aggression: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Принудительное приведение меток к непротиворечивому виду."""
        sent_out = sentiment.copy()
        agg_out = aggression.copy()

        # Инвариант 1: Служебные сообщения всегда нейтральны и без агрессии
        svc_mask = is_service == 1
        sent_out[svc_mask] = "neutral"
        agg_out[svc_mask] = 0

        # Инвариант 2: Агрессия возможна исключительно при негативе
        non_neg = sent_out != "negative"
        agg_out[non_neg] = 0

        return sent_out, agg_out

    def save(self, save_dir: str | Path) -> None:
        """Сохранение конфигурации и весов ансамбля."""
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        cfg = {
            "weight_service": self.weight_service,
            "weight_sentiment": self.weight_sentiment,
            "weight_aggression": self.weight_aggression,
            "aggression_threshold": self.aggression_threshold,
            "obscene_sent_boost": self.obscene_sent_boost,
            "obscene_agg_boost": self.obscene_agg_boost,
        }
        with open(path / "ensemble_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        if self.baseline is not None:
            self.baseline.save(path / "baseline.joblib")

        if self.transformer is not None:
            self.transformer.save_pretrained(path / "transformer")

    @classmethod
    def load(cls, save_dir: str | Path, device: str = "cpu") -> EnsembleClassifier:
        """Загрузка сохраненного ансамбля с диска."""
        path = Path(save_dir)
        cfg_path = path / "ensemble_config.json"
        cfg: dict[str, Any] = {}
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        base_path = path / "baseline.joblib"
        baseline = BaselineClassifier.load(base_path) if base_path.is_file() else None

        trans_dir = path / "transformer"
        transformer = None
        if trans_dir.is_dir() and HAS_TORCH:
            transformer = TransformerPipeline.from_pretrained(trans_dir, device=device)

        return cls(
            baseline=baseline,
            transformer=transformer,
            **cfg,
        )

    @classmethod
    def from_pretrained(
        cls,
        baseline_path: str | Path,
        transformer_dir: str | Path | None = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> EnsembleClassifier:
        """Загрузка обученного бейслайна и трансформера из отдельных путей."""
        base = BaselineClassifier.load(baseline_path)
        trans = None
        if transformer_dir is not None and HAS_TORCH:
            trans_path = Path(transformer_dir)
            if trans_path.is_dir():
                trans = TransformerPipeline.from_pretrained(trans_path, device=device)

        return cls(baseline=base, transformer=trans, **kwargs)
