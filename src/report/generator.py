"""Генератор итогового аналитического HTML-отчёта по настроениям домовых чатов."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import jinja2
import pandas as pd

from src.report.charts import (
    build_aggression_volume_figure,
    build_hourly_profile_figure,
    build_mood_dynamics_figure,
    build_mood_heatmap_figure,
    build_sentiment_breakdown_figure,
)

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "вода": ["вода", "горяч", "холодн", "напор", "давлен", "кипяток", "кран", "протеч"],
    "лифт": ["лифт", "застрял", "грузов", "кнопк", "шахт"],
    "уборка / мусор": ["мусор", "гряз", "уборк", "вонь", "таракан", "контейнер", "баки"],
    "шум / соседи": ["шум", "перфоратор", "сосед", "крик", "музык", "дрель", "ноч", "спать"],
    "парковка / авто": ["парковк", "машин", "авто", "место", "газон", "шлагбаум"],
    "отопление": ["отоплен", "батаре", "холод", "мерзн", "тепло"],
    "УК / диспетчер": ["ук", "управляющ", "диспетчер", "плат", "квитанц", "тариф", "заявк"],
}


def load_report_datasets(
    pred_dir: str | Path,
    messages_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загрузка результатов работы пайплайна и объединение с исходными сообщениями."""
    p_dir = Path(pred_dir)
    summary_df = pd.read_csv(p_dir / "chat_summary.csv", sep=";")
    daily_df = pd.read_csv(p_dir / "chat_daily.csv", sep=";")
    preds_df = pd.read_csv(p_dir / "messages_pred.csv", sep=";")

    if messages_path and Path(messages_path).is_file():
        msgs_raw = pd.read_csv(messages_path, sep=";", encoding="utf-8")
        merged = msgs_raw.merge(preds_df, on="message_id", how="left")
    else:
        merged = preds_df

    return summary_df, daily_df, merged


def calculate_report_kpi(
    summary_df: pd.DataFrame,
    messages_df: pd.DataFrame,
) -> dict[str, Any]:
    """Расчет сводных показателей качества и социального климата для шапки отчёта."""
    total_msgs = len(messages_df)
    res_mask = messages_df["is_service"] == 0 if "is_service" in messages_df.columns else pd.Series(True, index=messages_df.index)
    res_df = messages_df[res_mask]
    n_res = len(res_df)
    n_service = total_msgs - n_res

    neg_count = (res_df["sentiment"] == "negative").sum() if "sentiment" in res_df.columns else 0
    agg_count = (res_df["aggression"] == 1).sum() if "aggression" in res_df.columns else 0
    pos_count = (res_df["sentiment"] == "positive").sum() if "sentiment" in res_df.columns else 0

    global_mood = (pos_count - neg_count) / max(n_res, 1)

    return {
        "total_messages": total_msgs,
        "resident_msgs": n_res,
        "service_msgs": n_service,
        "service_share": (n_service / max(total_msgs, 1)) * 100,
        "global_neg_share": (neg_count / max(n_res, 1)) * 100,
        "global_agg_share": (agg_count / max(n_res, 1)) * 100,
        "global_mood": global_mood,
        "total_negative": neg_count,
        "total_aggression": agg_count,
        "total_chats": len(summary_df),
        "worsening_count": (summary_df["trend"] == "worsening").sum(),
        "stable_count": (summary_df["trend"] == "stable").sum(),
        "improving_count": (summary_df["trend"] == "improving").sum(),
    }


def _extract_day_quotes(day_msgs: pd.DataFrame) -> list[dict[str, Any]]:
    """Извлечение цитат наиболее острых реплик жителей за день."""
    quotes = []
    agg_subset = day_msgs[day_msgs["aggression"] == 1]
    candidates = agg_subset if not agg_subset.empty else day_msgs[day_msgs["sentiment"] == "negative"]

    for _, q_row in candidates.head(3).iterrows():
        raw_text = str(q_row.get("text", "")).replace("\n", " ").strip()
        text_clean = (raw_text[:147] + "...") if len(raw_text) > 150 else raw_text
        quotes.append({
            "text": text_clean,
            "sentiment": q_row.get("sentiment", "negative"),
            "aggression": int(q_row.get("aggression", 0)),
        })
    return quotes


def _detect_topics(day_msgs: pd.DataFrame) -> list[str]:
    """Определение ключевых ЖКХ-тем в потоке сообщений дня."""
    if "text" not in day_msgs.columns:
        return ["Конфликт в чате"]
    corpus = " ".join(day_msgs["text"].dropna().astype(str).tolist()).lower()
    topics = [t for t, words in TOPIC_KEYWORDS.items() if any(w in corpus for w in words)]
    return topics if topics else ["Общее недовольство"]


def extract_incident_cards(
    summary_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    messages_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Глубокий разбор пиковых дней (Deep Dive) и формирование карточек инцидентов."""
    incidents = []
    for _, s_row in summary_df.iterrows():
        chat_id = s_row["chat_id"]
        peak_str = str(s_row.get("peak_days", ""))
        peak_days = [p.strip() for p in peak_str.split("|") if p.strip() and p.strip() != "nan"]

        chat_peaks = []
        for p_date in peak_days:
            sub_d = daily_df[(daily_df["chat_id"] == chat_id) & (daily_df["date"] == p_date)]
            n_res = sub_d["n_resident"].values[0] if not sub_d.empty else 0
            agg_sh = sub_d["aggression_share"].values[0] if not sub_d.empty else 0.0

            sub_m = messages_df[
                (messages_df.get("chat_id", "") == chat_id)
                & (messages_df.get("timestamp", "").astype(str).str.startswith(p_date))
                & (messages_df.get("is_service", 0) == 0)
            ]

            chat_peaks.append({
                "date": p_date,
                "n_resident": int(n_res),
                "aggression_share": round(float(agg_sh) * 100, 1),
                "topics": _detect_topics(sub_m),
                "quotes": _extract_day_quotes(sub_m),
            })

        is_worse = s_row["trend"] == "worsening" or s_row["aggression_share"] > 0.15
        is_mid = s_row["trend"] == "stable" or s_row["aggression_share"] > 0.08
        risk_class = "risk-high" if is_worse else ("risk-mid" if is_mid else "risk-low")
        risk_label = "Критический риск" if is_worse else ("Повышенное внимание" if is_mid else "Штатно")

        trend_labels = {"worsening": "Ухудшение 🔴", "improving": "Улучшение 🟢", "stable": "Стабильно ⚪"}

        incidents.append({
            "chat_id": chat_id,
            "trend": s_row["trend"],
            "trend_ru": trend_labels.get(s_row["trend"], "Стабильно ⚪"),
            "mood_index": float(s_row["mood_index"]),
            "negative_share": float(s_row["negative_share"]) * 100,
            "aggression_share": float(s_row["aggression_share"]) * 100,
            "n_resident": int(s_row["n_resident"]),
            "risk_class": risk_class,
            "risk_label": risk_label,
            "peaks_formatted": " · ".join([p["date"] for p in chat_peaks]) if chat_peaks else "—",
            "peaks": chat_peaks,
        })
    return incidents


def generate_html_report(
    pred_dir: str | Path,
    output_path: str | Path = "report.html",
    messages_path: str | Path | None = None,
    template_path: str | Path | None = None,
) -> Path:
    """Главная функция построения интерактивного HTML-отчёта."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    summary_df, daily_df, messages_df = load_report_datasets(pred_dir, messages_path)
    kpi = calculate_report_kpi(summary_df, messages_df)
    incident_data = extract_incident_cards(summary_df, daily_df, messages_df)

    # Построение графиков
    fig_mood = build_mood_dynamics_figure(daily_df)
    fig_agg = build_aggression_volume_figure(daily_df)
    fig_heat = build_mood_heatmap_figure(daily_df)
    fig_sent = build_sentiment_breakdown_figure(summary_df)
    fig_hour = build_hourly_profile_figure(messages_df)

    t_path = Path(template_path) if template_path else Path(__file__).parent / "template.html"
    template = jinja2.Template(t_path.read_text(encoding="utf-8"))

    rendered_html = template.render(
        kpi=kpi,
        incident_data=incident_data,
        html_mood=fig_mood.to_html(full_html=False, include_plotlyjs=True),
        html_agg=fig_agg.to_html(full_html=False, include_plotlyjs=False),
        html_heat=fig_heat.to_html(full_html=False, include_plotlyjs=False),
        html_sent=fig_sent.to_html(full_html=False, include_plotlyjs=False),
        html_hour=fig_hour.to_html(full_html=False, include_plotlyjs=False),
    )

    out_file.write_text(rendered_html, encoding="utf-8")
    return out_file
