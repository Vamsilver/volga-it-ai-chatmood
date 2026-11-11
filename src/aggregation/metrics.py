"""Модуль математического расчета трендов настроения и дней пикового конфликта.

Реализует официальный алгоритмический эталон жюри chatmood_metrics.py:
1. trend:
   - Группировка дней активности по календарным ISO-неделям (пн-вс)
   - Учет недель с активностью жителей n_resident >= 5 (TREND_MIN_WEEK)
   - Если число таких недель < 3 -> 'stable'
   - Иначе -> взвешенная по n_resident линейная регрессия недельного индекса настроения на порядковый номер недели
   - Порог наклона TREND_SLOPE = 0.05:
     * slope > +0.05 -> 'improving'
     * slope < -0.05 -> 'worsening'
     * иначе -> 'stable'
2. peak_days:
   - Отбор дней с n_resident >= 5 (PEAK_MIN_DAY)
   - Индекс конфликта: conflict = n_negative + 2 * n_aggression
   - Исключение дней с conflict <= 0
   - Ранжирование по conflict (убывание) и дате (возрастание)
   - Топ-3 дня (PEAK_TOP = 3), отсортированные хронологически через '|'
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Константы регламента соревнования
TREND_SLOPE: float = 0.05
TREND_MIN_WEEK: int = 5
PEAK_MIN_DAY: int = 5
PEAK_TOP: int = 3
TRENDS: tuple[str, str, str] = ("improving", "worsening", "stable")


def calculate_conflict_score(
    n_negative: int | Sequence[int] | np.ndarray,
    n_aggression: int | Sequence[int] | np.ndarray,
) -> int | np.ndarray:
    """Вычисление интегрального индекса конфликта дня: conflict = n_neg + 2 * n_agg."""
    return np.asarray(n_negative) + 2 * np.asarray(n_aggression)


def calculate_mood_index(
    n_positive: int | Sequence[int] | np.ndarray,
    n_negative: int | Sequence[int] | np.ndarray,
    n_resident: int | Sequence[int] | np.ndarray,
) -> float | np.ndarray:
    """Вычисление индекса настроения: (n_pos - n_neg) / n_resident."""
    pos = np.asarray(n_positive, dtype=float)
    neg = np.asarray(n_negative, dtype=float)
    res = np.asarray(n_resident, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(res > 0, (pos - neg) / res, np.nan)


def _prepare_daily_counts(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Обеспечивает наличие счетчиков (n_resident, n_neg, n_pos, n_agg) в DataFrame."""
    if daily_df.empty:
        return daily_df

    df = daily_df.copy()
    nr = df["n_resident"].fillna(0).to_numpy() if "n_resident" in df.columns else np.zeros(len(df))

    if "n_neg" not in df.columns:
        neg_s = df.get("negative_share", pd.Series(0, index=df.index)).fillna(0).to_numpy()
        df["n_neg"] = np.round(neg_s * nr).astype(int)

    if "n_agg" not in df.columns:
        agg_s = df.get("aggression_share", pd.Series(0, index=df.index)).fillna(0).to_numpy()
        df["n_agg"] = np.round(agg_s * nr).astype(int)

    if "n_pos" not in df.columns:
        mood_i = df.get("mood_index", pd.Series(0, index=df.index)).fillna(0).to_numpy()
        df["n_pos"] = np.round(mood_i * nr + df["n_neg"]).astype(int)

    return df


def calculate_trend(chat_daily_df: pd.DataFrame) -> str:
    """Вычисление тренда настроения по суточным данным чата.

    Args:
        chat_daily_df: DataFrame суточных записей конкретного чата с колонками
                       ['date', 'n_resident', 'n_neg', 'n_pos'] (или долями).

    Returns:
        Одно из значений: 'improving', 'worsening', 'stable'.
    """
    if chat_daily_df.empty or "date" not in chat_daily_df.columns:
        return "stable"

    df = _prepare_daily_counts(chat_daily_df)
    dt = pd.to_datetime(df["date"])
    df["week"] = (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")

    w = df.groupby("week")[["n_resident", "n_neg", "n_pos"]].sum()
    w = w[w["n_resident"] >= TREND_MIN_WEEK]

    if len(w) < 3:
        return "stable"

    y = ((w["n_pos"] - w["n_neg"]) / w["n_resident"]).to_numpy(dtype=float)
    t = np.arange(len(w), dtype=float)
    wt = w["n_resident"].to_numpy(dtype=float)

    tm = np.average(t, weights=wt)
    ym = np.average(y, weights=wt)
    denom = np.sum(wt * (t - tm) ** 2)

    if denom <= 1e-12:
        return "stable"

    slope = np.sum(wt * (t - tm) * (y - ym)) / denom

    if slope > TREND_SLOPE:
        return "improving"
    if slope < -TREND_SLOPE:
        return "worsening"
    return "stable"


def calculate_peak_days(chat_daily_df: pd.DataFrame) -> list[str]:
    """Определение до 3 дней пикового конфликта для чата.

    Args:
        chat_daily_df: DataFrame суточных записей конкретного чата с колонками
                       ['date', 'n_resident', 'n_neg', 'n_agg'].

    Returns:
        Хронологически отсортированный список дат пиков ['YYYY-MM-DD', ...].
    """
    if chat_daily_df.empty or "date" not in chat_daily_df.columns:
        return []

    df = _prepare_daily_counts(chat_daily_df)
    valid_days = df[df["n_resident"] >= PEAK_MIN_DAY].copy()

    if valid_days.empty:
        return []

    valid_days["conflict"] = valid_days["n_neg"] + 2 * valid_days["n_agg"]
    conflicts = valid_days[valid_days["conflict"] > 0]

    if conflicts.empty:
        return []

    ranked = conflicts.sort_values(["conflict", "date"], ascending=[False, True])
    top_dates = ranked["date"].head(PEAK_TOP).tolist()
    return sorted(top_dates)
