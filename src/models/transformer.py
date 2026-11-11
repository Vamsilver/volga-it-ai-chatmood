"""Мультизадачная нейросетевая модель на базе rubert-tiny2.

Архитектура:
- Общий энкодер: cointegrated/rubert-tiny2 (29M параметров, hidden_size=312)
- Mean Pooling с учетом attention_mask и Dropout (p=0.2)
- Три независимые головы классификации:
  1. service_head: Linear(312, 2) -> is_service (0/1)
  2. sentiment_head: Linear(312, 3) -> sentiment (negative/neutral/positive)
  3. aggression_head: Linear(312, 2) -> aggression (0/1)
- Доменный постпроцессинг (гарантия соблюдения инвариантов разметки)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}
REV_SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class MultiTaskRuBertTiny(nn.Module if HAS_TORCH else object):
    """Мультизадачный трансформер для одновременной классификации трех признаков."""

    def __init__(
        self,
        model_name: str | Path | None = None,
        dropout: float = 0.2,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("Для использования модели требуется установить torch и transformers")
        super().__init__()
        default_local = Path("models/rubert_tiny2_multitask")
        if model_name is None:
            resolved_name = str(default_local) if default_local.is_dir() else "cointegrated/rubert-tiny2"
        else:
            resolved_name = str(model_name)
        self.model_name = resolved_name
        is_local = Path(resolved_name).exists()
        self.encoder = AutoModel.from_pretrained(resolved_name, local_files_only=is_local)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.service_head = nn.Linear(hidden_size, 2)
        self.sentiment_head = nn.Linear(hidden_size, 3)
        self.aggression_head = nn.Linear(hidden_size, 2)

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling эмбеддингов токенов с учетом маски внимания."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Прямой проход модели."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        return {
            "service_logits": self.service_head(pooled),
            "sentiment_logits": self.sentiment_head(pooled),
            "aggression_logits": self.aggression_head(pooled),
        }


class TransformerPipeline:
    """Обертка над нейросетью для инференса, обучения и сохранения весов."""

    def __init__(
        self,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if self.model is not None and HAS_TORCH:
            self.model.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cpu",
    ) -> TransformerPipeline:
        """Загрузка обученной модели и токенизатора из локальной директории."""
        if not HAS_TORCH:
            raise RuntimeError("Для загрузки требуется torch и transformers")

        path = Path(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(path)
        with open(path / "config_multitask.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = MultiTaskRuBertTiny(model_name=cfg.get("model_name", "cointegrated/rubert-tiny2"))
        weights_path = path / "model_weights.pt"
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.train(False)

        return cls(model=model, tokenizer=tokenizer, device=device)

    def save_pretrained(self, save_dir: str | Path) -> None:
        """Сохранение весов модели, токенизатора и метаданных."""
        if not HAS_TORCH or self.model is None:
            raise RuntimeError("Модель не инициализирована")

        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

        weights_path = path / "model_weights.pt"
        torch.save(self.model.state_dict(), weights_path)

        cfg = {"model_name": getattr(self.model, "model_name", "cointegrated/rubert-tiny2")}
        with open(path / "config_multitask.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def predict(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        max_length: int = 128,
    ) -> pd.DataFrame:
        """Батч-инференс с гарантированным применением доменных инвариантов."""
        if not HAS_TORCH or self.model is None or self.tokenizer is None:
            raise RuntimeError("Модель или токенизатор не готовы к инференсу")

        self.model.train(False)
        clean_texts = [str(t) if pd.notna(t) else "" for t in texts]

        all_svc = []
        all_sent = []
        all_agg = []

        with torch.no_grad():
            for i in range(0, len(clean_texts), batch_size):
                batch_texts = clean_texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                att_mask = encoded["attention_mask"].to(self.device)

                out = self.model(input_ids=input_ids, attention_mask=att_mask)

                pred_svc = torch.argmax(out["service_logits"], dim=-1).cpu().numpy()
                pred_sent_idx = torch.argmax(out["sentiment_logits"], dim=-1).cpu().numpy()
                pred_agg = torch.argmax(out["aggression_logits"], dim=-1).cpu().numpy()

                pred_sent = [REV_SENTIMENT_MAP[idx] for idx in pred_sent_idx]

                all_svc.extend(pred_svc)
                all_sent.extend(pred_sent)
                all_agg.extend(pred_agg)

        # Применение жестких доменных инвариантов
        pred_svc_arr = np.array(all_svc, dtype=int)
        pred_sent_arr = np.array(all_sent, dtype=object)
        pred_agg_arr = np.array(all_agg, dtype=int)

        # 1. Служебные сообщения всегда нейтральны и без агрессии
        svc_mask = pred_svc_arr == 1
        pred_sent_arr[svc_mask] = "neutral"
        pred_agg_arr[svc_mask] = 0

        # 2. Агрессия возможна только при негативной тональности
        non_neg_mask = pred_sent_arr != "negative"
        pred_agg_arr[non_neg_mask] = 0

        return pd.DataFrame({
            "is_service": pred_svc_arr,
            "sentiment": pred_sent_arr,
            "aggression": pred_agg_arr,
        })
