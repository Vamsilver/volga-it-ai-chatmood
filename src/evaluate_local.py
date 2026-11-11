"""Модуль локальной валидации и детальной оценки качества классификации на dev-выборке.

Воспроизводит логику официального evaluate.py жюри Волга-IT 2026:
- Критерий 1: Q1 = 0.6 * F1(sentiment) + 0.2 * F1(aggression) + 0.2 * F1(is_service)
- Критерий 3: Q3 = средний macro-F1 по сложным подвыборкам (sarcasm, typos, jargon, emoji, context)
- Формирование отчёта об ошибках (error analysis diff)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.data.loader import load_dev, read_csv_safe

SENTIMENTS = ("negative", "neutral", "positive")
HARD_TAGS = ("sarcasm", "obscene", "typos", "jargon", "emoji", "mixed", "context")
MIN_TAG_SUPPORT = 15

SCALE = {
    "c1": (0.40, 0.85),
    "c3": (0.30, 0.80),
}
POINTS = {"c1": 20.0, "c3": 8.0}


def _lin_scale(score: float, key: str) -> float:
    """Линейное масштабирование метрики в диапазон баллов регламента."""
    lo, hi = SCALE[key]
    return float(np.clip((score - lo) / (hi - lo), 0.0, 1.0))


def evaluate_predictions(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> dict[str, Any]:
    """Расчет всех метрик Критерия 1 и Критерия 3 регламента соревнования.

    Args:
        gold_df: DataFrame с колонками message_id, is_service, sentiment, aggression (и tags).
        pred_df: DataFrame предсказаний с аналогичными колонками.

    Returns:
        Словарь с метриками и баллами.
    """
    merged = gold_df.merge(pred_df, on="message_id", suffixes=("", "_p"))

    # Приведение типов
    y_svc_true = merged["is_service"].astype(int).to_numpy()
    y_svc_pred = merged["is_service_p"].astype(int).to_numpy()

    # Фильтр сообщений жителей для тональности и агрессии
    res_mask = y_svc_true == 0
    res_df = merged[res_mask]

    y_sent_true = res_df["sentiment"].str.strip().str.lower().to_numpy()
    y_sent_pred = res_df["sentiment_p"].str.strip().str.lower().to_numpy()

    y_agg_true = res_df["aggression"].astype(int).to_numpy()
    y_agg_pred = res_df["aggression_p"].astype(int).to_numpy()

    # Расчет метрик Критерия 1
    f1_sent = float(f1_score(y_sent_true, y_sent_pred, labels=list(SENTIMENTS), average="macro", zero_division=0))
    sent_per_class = f1_score(y_sent_true, y_sent_pred, labels=list(SENTIMENTS), average=None, zero_division=0)
    f1_agg = float(f1_score(y_agg_true, y_agg_pred, labels=[1], average="macro", zero_division=0))
    f1_svc = float(f1_score(y_svc_true, y_svc_pred, labels=[1], average="macro", zero_division=0))
    acc_sent = float((y_sent_true == y_sent_pred).mean())

    q1 = 0.6 * f1_sent + 0.2 * f1_agg + 0.2 * f1_svc
    pts_c1 = round(POINTS["c1"] * _lin_scale(q1, "c1"), 2)

    # Расчет метрик Критерия 3 (сложные теги)
    hard_metrics = _evaluate_hard_tags(res_df)
    q3 = hard_metrics["q3"]
    pts_c3 = round(POINTS["c3"] * _lin_scale(q3, "c3"), 2)

    # Матрица ошибок тональности
    conf_sent = confusion_matrix(y_sent_true, y_sent_pred, labels=list(SENTIMENTS))

    return {
        "q1": round(q1, 4),
        "points_c1": pts_c1,
        "f1_sentiment_macro": round(f1_sent, 4),
        "f1_sentiment_per_class": dict(zip(SENTIMENTS, np.round(sent_per_class, 4).tolist())),
        "accuracy_sentiment": round(acc_sent, 4),
        "f1_aggression": round(f1_agg, 4),
        "f1_service": round(f1_svc, 4),
        "q3": round(q3, 4),
        "points_c3": pts_c3,
        "hard_tags": hard_metrics["tags"],
        "confusion_matrix_sentiment": conf_sent,
    }


def _evaluate_hard_tags(res_df: pd.DataFrame) -> dict[str, Any]:
    """Оценка macro-F1 на подвыборках со сложными тегами."""
    per_tag = {}
    if "tags" not in res_df.columns:
        return {"q3": 0.0, "tags": {}}

    tags_series = res_df["tags"].fillna("").astype(str)

    for tag in HARD_TAGS:
        mask = tags_series.apply(lambda s: tag in s.split("|"))
        sub = res_df[mask]
        n_samples = len(sub)
        if n_samples < MIN_TAG_SUPPORT:
            continue

        active_labels = [s for s in SENTIMENTS if (sub["sentiment"] == s).any()]
        f1_tag = float(f1_score(sub["sentiment"], sub["sentiment_p"], labels=active_labels, average="macro", zero_division=0))
        acc_tag = float((sub["sentiment"] == sub["sentiment_p"]).mean())

        per_tag[tag] = {
            "n": n_samples,
            "f1_macro": round(f1_tag, 4),
            "accuracy": round(acc_tag, 4),
        }

    q3 = float(np.mean([v["f1_macro"] for v in per_tag.values()])) if per_tag else 0.0
    return {"q3": q3, "tags": per_tag}


def generate_diff_report(
    merged: pd.DataFrame,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Генерация таблицы сообщений с ошибками предсказания (Error Analysis Diff)."""
    errors = merged[
        (merged["is_service"] != merged["is_service_p"]) |
        (merged["sentiment"] != merged["sentiment_p"]) |
        (merged["aggression"] != merged["aggression_p"])
    ].copy()

    cols = ["message_id", "text", "is_service", "is_service_p", "sentiment", "sentiment_p", "aggression", "aggression_p"]
    if "tags" in merged.columns:
        cols.append("tags")

    diff_df = errors[[c for c in cols if c in errors.columns]]
    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        diff_df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")

    return diff_df


def print_evaluation_report(metrics: dict[str, Any]) -> None:
    """Красивый вывод результатов оценки в консоль."""
    print("=" * 65)
    print("          РЕЗУЛЬТАТЫ ОЦЕНКИ НА ВАЛИДАЦИОННОМ НАБОРЕ (DEV)          ")
    print("=" * 65)
    print(f"Критерий 1 (Q1 score):          {metrics['q1']:.4f}  | Баллы: {metrics['points_c1']:.2f} / 20.0")
    print(f"  - F1 Sentiment (macro):       {metrics['f1_sentiment_macro']:.4f} (Acc: {metrics['accuracy_sentiment']:.4f})")
    print(f"    * По классам: negative={metrics['f1_sentiment_per_class']['negative']:.3f}, neutral={metrics['f1_sentiment_per_class']['neutral']:.3f}, positive={metrics['f1_sentiment_per_class']['positive']:.3f}")
    print(f"  - F1 Aggression:              {metrics['f1_aggression']:.4f}")
    print(f"  - F1 Service (объявления):    {metrics['f1_service']:.4f}")
    print("-" * 65)
    print(f"Критерий 3 (Сложные теги Q3):   {metrics['q3']:.4f}  | Баллы: {metrics['points_c3']:.2f} / 8.0")
    for tag, d in metrics["hard_tags"].items():
        print(f"  - {tag:12s} (N={d['n']:2d}): F1={d['f1_macro']:.4f}, Acc={d['accuracy']:.4f}")
    print("=" * 65)


def evaluate_model_on_dev(
    model_type: str = "baseline",
    weights_path: str | Path | None = None,
    data_dir: str | Path = "data",
    output_diff: str | Path | None = None,
) -> dict[str, Any]:
    """Полный цикл загрузки dev, предсказания и оценки модели."""
    dev_df = load_dev(data_dir)

    if model_type == "baseline":
        from src.models.baseline import BaselineClassifier
        path = Path(weights_path) if weights_path else Path("models/baseline.joblib")
        model = BaselineClassifier.load(path)
        preds = model.predict_messages(dev_df)
    elif model_type == "transformer":
        from src.models.transformer import TransformerPipeline
        path = Path(weights_path) if weights_path else Path("models/rubert_tiny2_multitask")
        pipeline = TransformerPipeline.from_pretrained(path, device="cpu")
        preds = pipeline.predict(dev_df["text"])
        preds.insert(0, "message_id", dev_df["message_id"].to_numpy())
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    metrics = evaluate_predictions(dev_df, preds)
    print_evaluation_report(metrics)

    if output_diff:
        merged = dev_df.merge(preds, on="message_id", suffixes=("", "_p"))
        diff = generate_diff_report(merged, output_diff)
        print(f"[*] Отчёт об ошибках сохранён: {output_diff} (ошибок: {len(diff)} из {len(dev_df)})")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальная оценка моделей на dev-сете VolgaFinal")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "transformer"], help="Тип модели")
    parser.add_argument("--weights", type=str, default=None, help="Путь к весам модели")
    parser.add_argument("--pred-file", type=str, default=None, help="Путь к готовому CSV предсказаний")
    parser.add_argument("--data-dir", type=str, default="data", help="Каталог с данными")
    parser.add_argument("--diff", type=str, default=None, help="Путь для сохранения CSV с ошибками")
    args = parser.parse_args()

    if args.pred_file:
        gold = load_dev(args.data_dir)
        pred = read_csv_safe(args.pred_file)
        metrics = evaluate_predictions(gold, pred)
        print_evaluation_report(metrics)
    else:
        evaluate_model_on_dev(
            model_type=args.model,
            weights_path=args.weights,
            data_dir=args.data_dir,
            output_diff=args.diff,
        )


if __name__ == "__main__":
    main()
