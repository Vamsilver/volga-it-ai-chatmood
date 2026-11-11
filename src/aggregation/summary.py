"""Модуль вычисления сводных агрегатов активности и настроения по домовым чатам.

Генерирует эталонный файл chat_summary.csv:
- По одной строке на каждый chat_id
- Поля: chat_id;n_resident;mood_index;negative_share;aggression_share;trend;peak_days
- Метрики mood_index, negative_share, aggression_share рассчитываются кумулятивно за весь период
- trend: взвешенная линейная регрессия недельного настроения (improving / worsening / stable)
- peak_days: до 3 дней максимального конфликта (n_resident >= 5, n_neg + 2*n_agg > 0), через '|'

Строго соблюдает официальный эталон жюри chatmood_metrics.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import chatmood_metrics as cm

REQUIRED_SUMMARY_COLUMNS = [
    "chat_id",
    "n_resident",
    "mood_index",
    "negative_share",
    "aggression_share",
    "trend",
    "peak_days",
]


def _ensure_daily_counts(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Проверка и восстановление счетчиков (n_neg, n_pos, n_agg), если они отсутствуют."""
    df = daily_df.copy()
    if "n_neg" in df.columns and "n_pos" in df.columns and "n_agg" in df.columns:
        return df

    nr = df["n_resident"].fillna(0).to_numpy()
    neg_s = df["negative_share"].fillna(0).to_numpy()
    agg_s = df["aggression_share"].fillna(0).to_numpy()
    mood_i = df["mood_index"].fillna(0).to_numpy()

    df["n_neg"] = np.round(neg_s * nr).astype(int)
    df["n_agg"] = np.round(agg_s * nr).astype(int)
    df["n_pos"] = np.round(mood_i * nr + df["n_neg"]).astype(int)
    return df


def compute_chat_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Вычисление сводных показателей по чатам за весь период наблюдения.

    Args:
        daily_df: DataFrame суточных агрегатов (из compute_daily_aggregates или chat_daily.csv).

    Returns:
        DataFrame со столбцами:
        chat_id, n_resident, mood_index, negative_share, aggression_share, trend, peak_days.
    """
    if daily_df.empty:
        return pd.DataFrame(columns=REQUIRED_SUMMARY_COLUMNS)

    df_ready = _ensure_daily_counts(daily_df)
    summary_df = cm.chat_summary(df_ready)

    # Приведение типов и порядка столбцов к регламенту
    summary_df = summary_df[REQUIRED_SUMMARY_COLUMNS].copy()
    summary_df["chat_id"] = summary_df["chat_id"].astype(str)
    summary_df["n_resident"] = summary_df["n_resident"].astype(int)
    summary_df["trend"] = summary_df["trend"].astype(str)
    summary_df["peak_days"] = summary_df["peak_days"].fillna("").astype(str)

    return summary_df


def export_chat_summary_csv(
    summary_df: pd.DataFrame,
    output_path: str | Path,
    round_decimals: int | None = None,
) -> Path:
    """Экспорт сводной таблицы в CSV-файл chat_summary.csv.

    Формат:
    chat_id;n_resident;mood_index;negative_share;aggression_share;trend;peak_days
    Разделитель ';', кодировка UTF-8 с BOM, пустые значения для NaN.

    Args:
        summary_df: DataFrame со сводными показателями по чатам.
        output_path: Путь для сохранения CSV-файла.
        round_decimals: Округление вещественных показателей (по умолчанию без округления).

    Returns:
        Path к сохраненному файлу.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_df = summary_df[REQUIRED_SUMMARY_COLUMNS].copy()

    if round_decimals is not None:
        float_cols = ["mood_index", "negative_share", "aggression_share"]
        for c in float_cols:
            export_df[c] = export_df[c].round(round_decimals)

    export_df.to_csv(
        path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )
    return path
