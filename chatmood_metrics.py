"""Общие определения агрегатов и метрик финала Волга-IT 2026 («настроение домовых чатов»).

Этот модуль — часть задания: участники могут импортировать его, чтобы считать агрегаты ровно
так же, как жюри, а evaluate_final.py использует его для расчёта эталона.

Зависимости: pandas, numpy, scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENTIMENTS = ("negative", "neutral", "positive")
TRENDS = ("improving", "worsening", "stable")

# Порог наклона недельного индекса настроения (изменение за неделю), выше которого тренд не «stable»
TREND_SLOPE = 0.05
# Минимум сообщений жителей в неделе, чтобы неделя участвовала в оценке тренда
TREND_MIN_WEEK = 5
# Минимум сообщений жителей в дне, чтобы день мог быть «пиком конфликта»
PEAK_MIN_DAY = 5
PEAK_TOP = 3


def daily_aggregates(messages: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты по чату и дню.

    messages: message_id, chat_id, timestamp (МСК, 'YYYY-MM-DD HH:MM:SS')
    labels:   message_id, is_service (0/1), sentiment (negative/neutral/positive), aggression (0/1)

    Возвращает: chat_id, date, n_messages, n_resident, mood_index, negative_share, aggression_share
    Доли и индекс считаются только по сообщениям жителей (is_service = 0); если их нет — NaN.
    """
    m = messages[["message_id", "chat_id", "timestamp"]].merge(
        labels[["message_id", "is_service", "sentiment", "aggression"]], on="message_id", how="left"
    )
    m["date"] = m["timestamp"].str.slice(0, 10)
    m["is_service"] = m["is_service"].fillna(0).astype(int)
    m["aggression"] = m["aggression"].fillna(0).astype(int)
    res = m["is_service"] == 0
    m["r"] = res.astype(int)
    m["neg"] = (res & (m["sentiment"] == "negative")).astype(int)
    m["pos"] = (res & (m["sentiment"] == "positive")).astype(int)
    m["agg"] = (res & (m["aggression"] == 1)).astype(int)
    g = m.groupby(["chat_id", "date"]).agg(
        n_messages=("message_id", "size"), n_resident=("r", "sum"), n_neg=("neg", "sum"), n_pos=("pos", "sum"),
        n_agg=("agg", "sum"),
    ).reset_index()
    return _finish(g)


def _finish(g: pd.DataFrame) -> pd.DataFrame:
    nr = g["n_resident"].where(g["n_resident"] > 0)
    g["mood_index"] = (g["n_pos"] - g["n_neg"]) / nr
    g["negative_share"] = g["n_neg"] / nr
    g["aggression_share"] = g["n_agg"] / nr
    return g


def chat_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Сводка по чату из дневных агрегатов (нужны колонки n_neg, n_pos, n_agg).

    Возвращает: chat_id, n_resident, mood_index, negative_share, aggression_share, trend, peak_days
    """
    rows = []
    for chat, d in daily.groupby("chat_id"):
        tot = d[["n_messages", "n_resident", "n_neg", "n_pos", "n_agg"]].sum()
        nr = tot["n_resident"]
        rows.append({
            "chat_id": chat,
            "n_resident": int(nr),
            "mood_index": (tot["n_pos"] - tot["n_neg"]) / nr if nr else np.nan,
            "negative_share": tot["n_neg"] / nr if nr else np.nan,
            "aggression_share": tot["n_agg"] / nr if nr else np.nan,
            "trend": trend_of(d),
            "peak_days": "|".join(peak_days_of(d)),
        })
    return pd.DataFrame(rows)


def trend_of(d: pd.DataFrame) -> str:
    """Тренд настроения: взвешенная (по n_resident) линейная регрессия недельного mood_index
    на номер недели. Недели — ISO (пн–вс), учитываются недели с n_resident >= TREND_MIN_WEEK.
    Меньше трёх таких недель → 'stable'."""
    x = d.copy()
    dt = pd.to_datetime(x["date"])
    x["week"] = (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    w = x.groupby("week")[["n_resident", "n_neg", "n_pos"]].sum()
    w = w[w["n_resident"] >= TREND_MIN_WEEK]
    if len(w) < 3:
        return "stable"
    y = ((w["n_pos"] - w["n_neg"]) / w["n_resident"]).to_numpy()
    t = np.arange(len(w), dtype=float)
    wt = w["n_resident"].to_numpy(dtype=float)
    tm = np.average(t, weights=wt)
    ym = np.average(y, weights=wt)
    slope = np.sum(wt * (t - tm) * (y - ym)) / np.sum(wt * (t - tm) ** 2)
    if slope > TREND_SLOPE:
        return "improving"
    if slope < -TREND_SLOPE:
        return "worsening"
    return "stable"


def peak_days_of(d: pd.DataFrame) -> list[str]:
    """До трёх дней-пиков конфликта: дни с n_resident >= PEAK_MIN_DAY, ранжированные по
    conflict = n_neg + 2 * n_agg (по убыванию, при равенстве — более ранняя дата). Дни с conflict = 0
    пиками не считаются."""
    x = d[d["n_resident"] >= PEAK_MIN_DAY].copy()
    x["conflict"] = x["n_neg"] + 2 * x["n_agg"]
    x = x[x["conflict"] > 0].sort_values(["conflict", "date"], ascending=[False, True])
    return sorted(x["date"].head(PEAK_TOP).tolist())


def gold_aggregates(messages: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily_aggregates(messages, labels)
    return daily, chat_summary(daily)
