"""Скрипт обучения мультизадачного трансформера rubert-tiny2 для VolgaFinal.

Запуск из командной строки:
    python train.py --epochs 4 --batch-size 32 --lr 3e-5 --device cuda
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data.loader import load_dev, load_train
from src.models.transformer import MultiTaskRuBertTiny, SENTIMENT_MAP, TransformerPipeline


def set_seed(seed: int = 42) -> None:
    """Фиксация генераторов случайных чисел для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class ChatDataset(Dataset if HAS_TORCH else object):
    """Датасет для мультизадачного обучения на сообщениях чатов."""

    def __init__(
        self,
        texts: list[str],
        services: list[int],
        sentiments: list[str],
        aggressions: list[int],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.services = services
        self.sentiments = [SENTIMENT_MAP[s] for s in sentiments]
        self.aggressions = aggressions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "service_target": torch.tensor(self.services[idx], dtype=torch.long),
            "sentiment_target": torch.tensor(self.sentiments[idx], dtype=torch.long),
            "aggression_target": torch.tensor(self.aggressions[idx], dtype=torch.long),
        }


def evaluate_model(
    pipeline: TransformerPipeline,
    dev_df: pd.DataFrame,
) -> dict[str, float]:
    """Оценка качества модели на dev-выборке по правилам evaluate.py."""
    preds = pipeline.predict(dev_df["text"], batch_size=64)

    res_mask = dev_df["is_service"] == 0
    f1_sent = f1_score(
        dev_df.loc[res_mask, "sentiment"],
        preds.loc[res_mask, "sentiment"],
        labels=["negative", "neutral", "positive"],
        average="macro",
        zero_division=0,
    )
    f1_agg = f1_score(
        dev_df.loc[res_mask, "aggression"],
        preds.loc[res_mask, "aggression"],
        labels=[1],
        average="macro",
        zero_division=0,
    )
    f1_svc = f1_score(
        dev_df["is_service"],
        preds["is_service"],
        labels=[1],
        average="macro",
        zero_division=0,
    )

    q1 = 0.6 * f1_sent + 0.2 * f1_agg + 0.2 * f1_svc
    return {
        "q1": float(q1),
        "f1_sentiment": float(f1_sent),
        "f1_aggression": float(f1_agg),
        "f1_service": float(f1_svc),
    }


def train(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    output_dir: str | Path,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 3e-5,
    device_name: str = "cuda",
) -> None:
    """Основной цикл обучения модели."""
    if not HAS_TORCH:
        raise RuntimeError("Для обучения требуются torch и transformers")

    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"[*] Используемое устройство: {device}")

    model_name = "cointegrated/rubert-tiny2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MultiTaskRuBertTiny(model_name=model_name).to(device)

    train_ds = ChatDataset(
        texts=train_df["text"].tolist(),
        services=train_df["is_service"].tolist(),
        sentiments=train_df["sentiment"].tolist(),
        aggressions=train_df["aggression"].tolist(),
        tokenizer=tokenizer,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Веса функций потерь для борьбы с дисбалансом
    svc_weights = torch.tensor([1.0, 3.0], device=device)
    sent_weights = torch.tensor([1.4, 1.0, 1.8], device=device)
    agg_weights = torch.tensor([1.0, 6.0], device=device)

    crit_svc = nn.CrossEntropyLoss(weight=svc_weights)
    crit_sent = nn.CrossEntropyLoss(weight=sent_weights)
    crit_agg = nn.CrossEntropyLoss(weight=agg_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_q1 = 0.0
    pipeline = TransformerPipeline(model=model, tokenizer=tokenizer, device=str(device))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(input_ids, attention_mask)
            l_svc = crit_svc(out["service_logits"], batch["service_target"].to(device))
            l_sent = crit_sent(out["sentiment_logits"], batch["sentiment_target"].to(device))
            l_agg = crit_agg(out["aggression_logits"], batch["aggression_target"].to(device))

            loss = l_svc + l_sent + l_agg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        metrics = evaluate_model(pipeline, dev_df)
        print(f"Эпоха {epoch}/{epochs} | Loss: {avg_loss:.4f} | Dev Q1: {metrics['q1']:.4f} (Sent: {metrics['f1_sentiment']:.4f}, Agg: {metrics['f1_aggression']:.4f}, Svc: {metrics['f1_service']:.4f})")

        if metrics["q1"] > best_q1:
            best_q1 = metrics["q1"]
            pipeline.save_pretrained(output_dir)
            print(f" [✔] Сохранен новый лучший чекпоинт (Q1={best_q1:.4f}) в {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение трансформера для VolgaFinal")
    parser.add_argument("--epochs", type=int, default=4, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча")
    parser.add_argument("--lr", type=float, default=3e-5, help="Скорость обучения (learning rate)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Устройство")
    parser.add_argument("--output-dir", type=str, default="models/rubert_tiny2_multitask", help="Каталог сохранения")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    train_df = load_train("data")
    dev_df = load_dev("data")

    print(f"[*] Загружен train: {len(train_df)} строк, dev: {len(dev_df)} строк")
    train(
        train_df=train_df,
        dev_df=dev_df,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
