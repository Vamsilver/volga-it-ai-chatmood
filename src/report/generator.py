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
from src.report.confidence import calculate_confidence_kpi

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "вода / водоснабжение": [
        "вода", "горяч", "холодн", "напор", "давлен", "кипяток", "кран",
        "гвс", "хвс", "водоснабжен", "пересох", "шланг", "колонк", "подмес",
    ],
    "лифт": [
        "лифт", "застрял", "грузов", "кнопк", "шахт", "лебедк", "пассажирск",
        "трос", "кабин", "застряли", "грузовой",
    ],
    "уборка / мусор / пухто": [
        "мусор", "гряз", "уборк", "вонь", "таракан", "контейнер", "баки",
        "пухто", "свалк", "крыс", "помойк", "клининг", "пакет", "антисанитар",
    ],
    "шум / тишина / соседи": [
        "шум", "перфоратор", "сосед", "крик", "музык", "дрель", "ноч",
        "спать", "топот", "дебош", "тишин", "сверл", "басы", "лай",
    ],
    "парковка / авто / проезд": [
        "парковк", "машин", "авто", "место", "газон", "поребрик", "блокиров",
        "проезд", "парковат", "эвакуатор", "колес", "тротуар",
    ],
    "отопление / теплоноситель": [
        "отоплен", "батаре", "холод", "мерзн", "тепло", "теплоносител",
        "гребенк", "развоздуш", "стояк холодн", "ледян", "замерза",
    ],
    "УК / диспетчер / заявки": [
        "ук", "управляющ", "диспетчер", "плат", "квитанц", "тариф", "заявк",
        "еирц", "адс", "квадо", "бухгалтер", "перерасчет", "жск", "тсн", "уо",
    ],
    "кровля / фасад / протечки": [
        "протеч", "затоп", "залил", "капает", "потолок", "крыш", "кровл",
        "фанов", "стояк", "стык", "герметиз", "фасад", "балкон", "козырек", "отмостк", "шов",
    ],
    "электрика / освещение": [
        "свет", "электрич", "лампочк", "лампа", "щиток", "проводк", "автомат",
        "фонар", "освещен", "плафон", "выбило", "напряжен", "электроэнерг",
    ],
    "безопасность / домофон / шлагбаум": [
        "домофон", "калитка", "замок", "дверь", "магнит", "взлом", "камер",
        "видеонаблюден", "охрана", "ключ", "шлагбаум", "код", "таблетк", "доводчик",
    ],
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

    raw_found = False
    candidates: list[Path] = []
    if messages_path:
        candidates.append(Path(messages_path))
    candidates.extend([
        Path("data/dev_messages.csv"),
        Path("data/test_messages.csv"),
        p_dir / "dev_messages.csv",
        p_dir / "messages.csv",
    ])

    merged = preds_df.copy()
    for cand in candidates:
        if cand.is_file():
            try:
                msgs_raw = pd.read_csv(cand, sep=";", encoding="utf-8")
                merged = msgs_raw.merge(preds_df, on="message_id", how="left")
                raw_found = True
                break
            except Exception:
                continue

    # Безопасная инициализация отсутствующих текстовых/идентификационных колонок
    for col in ["text", "chat_id", "timestamp"]:
        if col not in merged.columns:
            merged[col] = ""

    return summary_df, daily_df, merged


def calculate_report_kpi(
    summary_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
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

    conf_kpi = calculate_confidence_kpi(daily_df) if daily_df is not None else {
        "low_volume_days": 0,
        "low_volume_share": 0.0,
        "total_days": 0,
        "reliable_days": 0,
    }

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
        "low_volume_days": conf_kpi["low_volume_days"],
        "low_volume_share": conf_kpi["low_volume_share"],
        "total_days": conf_kpi["total_days"],
        "reliable_days": conf_kpi["reliable_days"],
    }


def _extract_day_quotes(day_msgs: pd.DataFrame, max_quotes: int = 3) -> list[dict[str, Any]]:
    """Извлечение цитат наиболее острых реплик жителей за день."""
    quotes: list[dict[str, Any]] = []
    if day_msgs.empty or "text" not in day_msgs.columns:
        return quotes

    sentiment_col = day_msgs.get("sentiment", pd.Series(dtype=object))
    aggression_col = day_msgs.get("aggression", pd.Series(dtype=int))
    conflict_sub = day_msgs[(sentiment_col == "negative") | (aggression_col == 1)].copy()
    if conflict_sub.empty:
        return quotes

    def _quote_score(row: pd.Series) -> float:
        text = str(row.get("text", "")).strip()
        score = 10.0 if row.get("aggression", 0) == 1 else 3.0
        # Бонус за информативность длины: предпочтение развернутым описаниям аварии перед односложными "+1"
        score += min(len(text) / 20.0, 5.0)
        # Бонус за эмоциональную пунктуацию и маркеры возмущения
        if any(mark in text for mark in ["!", "?", "😡", "🤬", "руко", "авария", "срочно"]):
            score += 2.0
        return score

    conflict_sub["_score"] = conflict_sub.apply(_quote_score, axis=1)
    top_candidates = conflict_sub.sort_values(by="_score", ascending=False).head(max_quotes)

    for _, q_row in top_candidates.iterrows():
        raw_text = str(q_row.get("text", "")).replace("\n", " ").strip()
        # Сохранение verbatim-цитаты до 280 символов без преждевременного усечения
        text_clean = (raw_text[:277] + "...") if len(raw_text) > 280 else raw_text
        quotes.append({
            "text": text_clean,
            "sentiment": q_row.get("sentiment", "negative"),
            "aggression": int(q_row.get("aggression", 0)),
        })
    return quotes


def _detect_topics(day_msgs: pd.DataFrame) -> list[str]:
    """Определение ключевых ЖКХ-тем в потоке сообщений дня."""
    if day_msgs.empty or "text" not in day_msgs.columns:
        return ["Конфликт в чате"]

    # Фильтруем конфликтное подмножество: только негативные или агрессивные сообщения
    sentiment_col = day_msgs.get("sentiment", pd.Series(dtype=object))
    aggression_col = day_msgs.get("aggression", pd.Series(dtype=int))
    conflict_sub = day_msgs[(sentiment_col == "negative") | (aggression_col == 1)]
    target_df = conflict_sub if not conflict_sub.empty else day_msgs

    corpus = " ".join(target_df["text"].dropna().astype(str).tolist()).lower()
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
    kpi = calculate_report_kpi(summary_df, messages_df, daily_df=daily_df)
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
