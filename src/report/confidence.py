"""Модуль расчета доверительных интервалов и байесовского сглаживания для отчётности.

Реализует статистические методы для оценки неопределенности на днях с малым
объёмом сообщений жителей (Рекомендация п. 4 ТЗ):
1. Интервал оценок Вильсона (Wilson score interval) для биномиальных пропорций
   (доля негатива negative_share, доля агрессии aggression_share).
2. Байесовское сглаживание Дирихле (Dirichlet-Bayes shrinkage interval, N0=3.0)
   для разностного индекса социального настроения (mood_index).
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import norm


def get_z_score(confidence: float = 0.95) -> float:
    """Квантиль стандартного нормального распределения для заданного уровня доверия."""
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"Уровень доверия должен быть в диапазоне (0, 1), получено {confidence}")
    return float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))


def wilson_score_interval(
    k: float | int,
    n: int | float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Расчёт двустороннего 95% (или 1-alpha) интервала Вильсона для биномиальной пропорции.

    Параметры:
        k: количество целевых событий (или доля * n)
        n: общий объём испытаний (сообщений жителей n_resident)
        confidence: уровень доверия (по умолчанию 0.95)

    Возвращает:
        (ci_lower, ci_upper) в диапазоне [0.0, 1.0] или (nan, nan) при n <= 0.
    """
    if n is None or math.isnan(n) or n <= 0:
        return (float("nan"), float("nan"))

    if k is None or math.isnan(k):
        return (float("nan"), float("nan"))

    k_val = max(0.0, min(float(k), float(n)))
    n_val = float(n)

    z = get_z_score(confidence)
    z2 = z * z
    denom = n_val + z2

    center = (k_val + 0.5 * z2) / denom
    spread = (z / denom) * math.sqrt((k_val * (n_val - k_val) / n_val) + (0.25 * z2))

    ci_lower = max(0.0, center - spread)
    ci_upper = min(1.0, center + spread)
    return (float(ci_lower), float(ci_upper))


def dirichlet_bayes_mood_ci(
    n_resident: int | float,
    mood_index: float | None = None,
    negative_share: float | None = None,
    n_pos: int | None = None,
    n_neg: int | None = None,
    confidence: float = 0.95,
    alpha_pos: float = 0.5,
    alpha_neu: float = 1.5,
    alpha_neg: float = 1.0,
) -> tuple[float, float, float]:
    """Байесовское сглаживание Дирихле и доверительный интервал для индекса настроения (mood_index).

    Моделирует вектор исходов (pos, neu, neg) как мультиномиальное распределение
    с сопряженным априорным распределением Дирихле (alpha_pos=0.5, alpha_neu=1.5, alpha_neg=1.0,
    суммарный априорный псевдо-объём N0 = 3.0, отражающий типовой баланс домовых чатов).

    Параметры:
        n_resident: число сообщений жителей в этот день
        mood_index: исходный сырой индекс настроения (p_pos - p_neg)
        negative_share: исходная сырая доля негатива
        n_pos: точное число позитивных сообщений (если доступно)
        n_neg: точное число негативных сообщений (если доступно)
        confidence: уровень доверия
        alpha_pos, alpha_neu, alpha_neg: гиперпараметры априорного распределения

    Возвращает:
        (ci_lower, ci_upper, smoothed_mood) в диапазоне [-1.0, 1.0] или (nan, nan, nan) при n <= 0.
    """
    if n_resident is None or math.isnan(n_resident) or n_resident <= 0:
        return (float("nan"), float("nan"), float("nan"))

    n = float(n_resident)

    # Восстановление счетчиков классов, если не переданы явно
    if n_neg is None:
        neg_sh = 0.0 if (negative_share is None or math.isnan(negative_share)) else max(0.0, min(1.0, float(negative_share)))
        k_neg = round(neg_sh * n)
    else:
        k_neg = max(0, min(int(n_neg), int(n)))

    if n_pos is None:
        m_idx = 0.0 if (mood_index is None or math.isnan(mood_index)) else max(-1.0, min(1.0, float(mood_index)))
        pos_sh = max(0.0, min(1.0, m_idx + (k_neg / n)))
        k_pos = round(pos_sh * n)
    else:
        k_pos = max(0, min(int(n_pos), int(n)))

    # Гарантируем непротиворечивость суммы счетчиков
    if k_pos + k_neg > n:
        total = k_pos + k_neg
        k_pos = round(k_pos * n / total)
        k_neg = int(n - k_pos)

    k_neu = max(0.0, n - k_pos - k_neg)

    n0 = alpha_pos + alpha_neu + alpha_neg
    denom = n + n0

    p_pos_tilde = (k_pos + alpha_pos) / denom
    p_neg_tilde = (k_neg + alpha_neg) / denom

    smoothed_mood = p_pos_tilde - p_neg_tilde

    # Дисперсия разности двух зависимых мультиномиальных долей:
    # Var(p_pos - p_neg) = (p_pos + p_neg - (p_pos - p_neg)^2) / denom
    var_mu = max(0.0, (p_pos_tilde + p_neg_tilde - smoothed_mood * smoothed_mood) / denom)
    se = math.sqrt(var_mu)

    z = get_z_score(confidence)
    margin = z * se

    ci_lower = max(-1.0, smoothed_mood - margin)
    ci_upper = min(1.0, smoothed_mood + margin)

    return (float(ci_lower), float(ci_upper), float(smoothed_mood))


def enrich_daily_with_confidence(
    daily_df: pd.DataFrame,
    low_volume_threshold: int = 5,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Обогащает таблицу chat_daily статистическими интервалами неопределенности.

    ВНИМАНИЕ: Возвращает копию DataFrame с новыми колонками для отчёта и дашборда.
    Исходные сырые колонки (mood_index, negative_share, aggression_share) сохраняются
    нетронутыми в строгом соответствии с эталоном chatmood_metrics.py.
    """
    if daily_df.empty:
        return daily_df.copy()

    df = daily_df.copy()

    # Проверка наличия обязательных колонок
    if "n_resident" not in df.columns:
        return df

    is_low_volume = []
    mood_ci_lower_list = []
    mood_ci_upper_list = []
    mood_smoothed_list = []
    neg_ci_lower_list = []
    neg_ci_upper_list = []
    agg_ci_lower_list = []
    agg_ci_upper_list = []
    warning_badge_list = []

    for _, row in df.iterrows():
        n_res = row.get("n_resident", float("nan"))
        m_idx = row.get("mood_index", float("nan"))
        neg_sh = row.get("negative_share", float("nan"))
        agg_sh = row.get("aggression_share", float("nan"))

        if pd.isna(n_res) or n_res <= 0:
            is_low = False
            m_low, m_high, m_smooth = float("nan"), float("nan"), float("nan")
            n_low, n_high = float("nan"), float("nan")
            a_low, a_high = float("nan"), float("nan")
            badge = ""
        else:
            n_val = int(n_res)
            is_low = n_val < low_volume_threshold

            m_low, m_high, m_smooth = dirichlet_bayes_mood_ci(
                n_resident=n_val,
                mood_index=m_idx,
                negative_share=neg_sh,
                confidence=confidence,
            )

            k_neg = round(neg_sh * n_val) if not pd.isna(neg_sh) else 0
            n_low, n_high = wilson_score_interval(k=k_neg, n=n_val, confidence=confidence)

            k_agg = round(agg_sh * n_val) if not pd.isna(agg_sh) else 0
            a_low, a_high = wilson_score_interval(k=k_agg, n=n_val, confidence=confidence)

            badge = "⚠️ Малый объём данных (n < 5)" if is_low else ""

        is_low_volume.append(is_low)
        mood_ci_lower_list.append(m_low)
        mood_ci_upper_list.append(m_high)
        mood_smoothed_list.append(m_smooth)
        neg_ci_lower_list.append(n_low)
        neg_ci_upper_list.append(n_high)
        agg_ci_lower_list.append(a_low)
        agg_ci_upper_list.append(a_high)
        warning_badge_list.append(badge)

    df["is_low_volume"] = is_low_volume
    df["mood_ci_lower"] = mood_ci_lower_list
    df["mood_ci_upper"] = mood_ci_upper_list
    df["mood_smoothed"] = mood_smoothed_list
    df["neg_ci_lower"] = neg_ci_lower_list
    df["neg_ci_upper"] = neg_ci_upper_list
    df["agg_ci_lower"] = agg_ci_lower_list
    df["agg_ci_upper"] = agg_ci_upper_list
    df["warning_badge"] = warning_badge_list

    return df


def calculate_confidence_kpi(
    daily_df: pd.DataFrame,
    low_volume_threshold: int = 5,
) -> dict[str, Any]:
    """Расчет агрегированной статистики по надежности выборки и дням с малым объёмом."""
    if daily_df.empty or "n_resident" not in daily_df.columns:
        return {
            "total_days": 0,
            "active_days": 0,
            "low_volume_days": 0,
            "low_volume_share": 0.0,
            "zero_volume_days": 0,
            "reliable_days": 0,
        }

    total_days = len(daily_df)
    n_res = daily_df["n_resident"].fillna(0)
    zero_days = int((n_res == 0).sum())
    low_days = int(((n_res > 0) & (n_res < low_volume_threshold)).sum())
    reliable_days = int((n_res >= low_volume_threshold).sum())
    active_days = total_days - zero_days

    return {
        "total_days": total_days,
        "active_days": active_days,
        "low_volume_days": low_days,
        "low_volume_share": (low_days / max(total_days, 1)) * 100.0,
        "zero_volume_days": zero_days,
        "reliable_days": reliable_days,
    }
