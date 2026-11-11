"""Модуль расчета суточных агрегатов настроения по домовым чатам.

Генерирует эталонный файл chat_daily.csv:
- По строке на каждую пару (chat_id, date), где было хотя бы одно сообщение
- Поля: chat_id;date;n_messages;n_resident;mood_index;negative_share;aggression_share
- Доли и индекс настроения рассчитываются строго по сообщениям жителей (is_service == 0)
- Если сообщений жителей за день нет (n_resident == 0), доли и индекс остаются пустыми (NaN)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_DAILY_COLUMNS = [
    "chat_id",
    "date",
    "n_messages",
    "n_resident",
    "mood_index",
    "negative_share",
    "aggression_share",
]


def compute_daily_aggregates(
    messages_df: pd.DataFrame,
    labels_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Вычисление суточных агрегатов активности и настроения по каждому чату.

    Args:
        messages_df: DataFrame с колонками ['message_id', 'chat_id', 'timestamp'].
                     Если labels_df is None, должен также содержать ['is_service', 'sentiment', 'aggression'].
        labels_df: Опциональный DataFrame предсказаний с колонками
                   ['message_id', 'is_service', 'sentiment', 'aggression'].

    Returns:
        pd.DataFrame с агрегатами по парам (chat_id, date).
    """
    if labels_df is not None:
        merged = messages_df[["message_id", "chat_id", "timestamp"]].merge(
            labels_df[["message_id", "is_service", "sentiment", "aggression"]],
            on="message_id",
            how="left",
        )
    else:
        req_cols = ["message_id", "chat_id", "timestamp", "is_service", "sentiment", "aggression"]
        missing = [c for c in req_cols if c not in messages_df.columns]
        if missing:
            raise ValueError(f"Входной DataFrame не содержит необходимых колонок: {missing}")
        merged = messages_df[req_cols].copy()

    # Извлечение календарной даты по московскому времени (первые 10 символов YYYY-MM-DD)
    merged["date"] = merged["timestamp"].astype(str).str.slice(0, 10)

    # Приведение типов меток
    merged["is_service"] = pd.to_numeric(merged["is_service"], errors="coerce").fillna(0).astype(int)
    merged["aggression"] = pd.to_numeric(merged["aggression"], errors="coerce").fillna(0).astype(int)
    merged["sentiment"] = merged["sentiment"].astype(str).str.strip().str.lower()

    # Маскирование: метрики считаются ТОЛЬКО по репликам жителей
    res_mask = merged["is_service"] == 0
    merged["is_res"] = res_mask.astype(int)
    merged["is_neg"] = (res_mask & (merged["sentiment"] == "negative")).astype(int)
    merged["is_pos"] = (res_mask & (merged["sentiment"] == "positive")).astype(int)
    merged["is_agg"] = (res_mask & (merged["aggression"] == 1)).astype(int)

    # Агрегация по чату и календарному дню
    grouped = merged.groupby(["chat_id", "date"], sort=True).agg(
        n_messages=("message_id", "size"),
        n_resident=("is_res", "sum"),
        n_neg=("is_neg", "sum"),
        n_pos=("is_pos", "sum"),
        n_agg=("is_agg", "sum"),
    ).reset_index()

    # Расчет долей и индексов
    nr = grouped["n_resident"].where(grouped["n_resident"] > 0)
    grouped["mood_index"] = (grouped["n_pos"] - grouped["n_neg"]) / nr
    grouped["negative_share"] = grouped["n_neg"] / nr
    grouped["aggression_share"] = grouped["n_agg"] / nr

    return grouped


def export_chat_daily_csv(
    daily_df: pd.DataFrame,
    output_path: str | Path,
    round_decimals: int = 4,
) -> Path:
    """Экспорт суточных агрегатов в эталонный файл chat_daily.csv.

    Формат:
    chat_id;date;n_messages;n_resident;mood_index;negative_share;aggression_share
    Разделитель ';', UTF-8, пустые значения для дней с n_resident=0.

    Args:
        daily_df: DataFrame, полученный из compute_daily_aggregates.
        output_path: Путь к выходному файлу.
        round_decimals: Округление вещественных показателей.

    Returns:
        Path к сохраненному файлу.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_df = daily_df[REQUIRED_DAILY_COLUMNS].copy()

    # Округление вещественных колонок
    float_cols = ["mood_index", "negative_share", "aggression_share"]
    for c in float_cols:
        export_df[c] = export_df[c].round(round_decimals)

    # Запись с разделителем ';'
    export_df.to_csv(
        path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )
    return path
