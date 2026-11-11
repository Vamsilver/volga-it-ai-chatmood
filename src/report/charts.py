"""Модуль построения интерактивных визуализаций для HTML-отчёта.

Использует Plotly для генерации графиков, встраиваемых в итоговый HTML-отчёт.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


from src.report.confidence import enrich_daily_with_confidence


def _build_chat_scatter_trace(chat_id: str, group: pd.DataFrame, color: str) -> go.Scatter:
    """Построение интерактивной линии настроения для одного чата с ДИ."""
    hover_template = (
        f"<b>Чат {chat_id}</b><br>"
        + "Дата: %{x|%Y-%m-%d}<br>"
        + "Индекс настроения: <b>%{y:.2f}</b><br>"
        + "95% ДИ настроения: [%{customdata[2]:.2f} ... %{customdata[3]:.2f}]<br>"
        + "Сглаженный индекс: %{customdata[4]:.2f}<br>"
        + "Сообщений жителей: %{customdata[0]}<br>"
        + "Негатив: %{customdata[1]:.1%} (ДИ: [%{customdata[5]:.1%} ... %{customdata[6]:.1%}])"
        + "%{customdata[7]}"
        + "<extra></extra>"
    )

    badge_col = [
        "<br><span style='color:#f59e0b;font-weight:bold;'>⚠️ Малый объём данных (n < 5)</span>"
        if lv else ""
        for lv in group["is_low_volume"]
    ]

    customdata = list(zip(
        group["n_resident"],
        group["negative_share"],
        group["mood_ci_lower"],
        group["mood_ci_upper"],
        group["mood_smoothed"],
        group["neg_ci_lower"],
        group["neg_ci_upper"],
        badge_col,
    ))

    marker_symbols = ["circle-open" if lv else "circle" for lv in group["is_low_volume"]]
    marker_sizes = [9 if lv else 7 for lv in group["is_low_volume"]]

    return go.Scatter(
        x=group["date"],
        y=group["mood_index"],
        mode="lines+markers",
        name=f"Чат {chat_id}",
        line=dict(color=color, width=2.5),
        marker=dict(
            size=marker_sizes,
            symbol=marker_symbols,
            line=dict(width=2, color=color),
        ),
        customdata=customdata,
        hovertemplate=hover_template,
    )


def build_mood_dynamics_figure(daily_df: pd.DataFrame) -> go.Figure:
    """Интерактивный график динамики индекса настроения (mood_index) по чатам с ДИ."""
    fig = go.Figure()

    if daily_df.empty or "mood_index" not in daily_df.columns:
        fig.add_annotation(text="Нет данных для отображения динамики", showarrow=False)
        return fig

    clean_df = daily_df.dropna(subset=["mood_index"]).copy()
    if "mood_ci_lower" not in clean_df.columns or "is_low_volume" not in clean_df.columns:
        clean_df = enrich_daily_with_confidence(clean_df)

    clean_df["date"] = pd.to_datetime(clean_df["date"])
    clean_df = clean_df.sort_values(by=["chat_id", "date"])

    # Добавляем эталонную нулевую линию
    min_date = clean_df["date"].min()
    max_date = clean_df["date"].max()

    fig.add_shape(
        type="line",
        x0=min_date,
        x1=max_date,
        y0=0.0,
        y1=0.0,
        line=dict(color="#94a3b8", width=1.5, dash="dash"),
    )

    # Цветовая палитра
    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#9333ea", "#0891b2"]

    for idx, (chat_id, group) in enumerate(clean_df.groupby("chat_id")):
        color = colors[idx % len(colors)]
        fig.add_trace(_build_chat_scatter_trace(str(chat_id), group, color))

        low_group = group[group["is_low_volume"]]
        if not low_group.empty:
            # Опорная верхняя граница ДИ для Plotly fill='tonexty'
            fig.add_trace(
                go.Scatter(
                    x=low_group["date"],
                    y=low_group["mood_ci_upper"],
                    mode="lines",
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"95% ДИ Верх (Чат {chat_id})",
                )
            )
            # Полупрозрачная лента неопределенности ДИ Дирихле для дней с n < 5
            fig.add_trace(
                go.Scatter(
                    x=low_group["date"],
                    y=low_group["mood_ci_lower"],
                    mode="lines",
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                    fill="tonexty",
                    fillcolor="rgba(59, 130, 246, 0.15)",
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"95% ДИ Неопределенность (Чат {chat_id})",
                )
            )

    fig.update_layout(
        title=dict(
            text="<b>Динамика индекса настроения жителей по дням</b><br><span style='font-size:12px;color:#64748b;'>Сплошные маркеры: n ≥ 5 · Полые маркеры (◯): малый объём (n &lt; 5, повышена дисперсия) · Тултип: 95% ДИ Вильсона и Дирихле</span>",
            font=dict(size=16),
        ),
        xaxis=dict(title="Дата", gridcolor="#f1f5f9"),
        yaxis=dict(
            title="Индекс настроения (от -1.0 до +1.0)",
            range=[-1.05, 1.05],
            gridcolor="#f1f5f9",
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=75, b=50),
        height=440,
    )
    return fig


def build_aggression_volume_figure(daily_df: pd.DataFrame) -> go.Figure:
    """График связи объема сообщений и доли агрессии во времени."""
    fig = go.Figure()

    if daily_df.empty:
        fig.add_annotation(text="Нет данных для агрессии", showarrow=False)
        return fig

    agg_df = daily_df.copy()
    agg_df["date"] = pd.to_datetime(agg_df["date"])
    by_day = (
        agg_df.groupby("date")
        .agg(
            total_resident=("n_resident", "sum"),
            mean_aggression=("aggression_share", "mean"),
        )
        .reset_index()
        .sort_values(by="date")
    )

    # Столбцы: объем сообщений
    fig.add_trace(
        go.Bar(
            x=by_day["date"],
            y=by_day["total_resident"],
            name="Сообщений жителей (все чаты)",
            marker=dict(color="#cbd5e1"),
            yaxis="y",
            hovertemplate="Дата: %{x|%Y-%m-%d}<br>Сообщений: %{y}<extra></extra>",
        )
    )

    # Линия: агрессия
    fig.add_trace(
        go.Scatter(
            x=by_day["date"],
            y=by_day["mean_aggression"] * 100,
            name="Доля агрессии (%)",
            mode="lines+markers",
            line=dict(color="#e11d48", width=3),
            marker=dict(size=6, color="#e11d48"),
            yaxis="y2",
            hovertemplate="Дата: %{x|%Y-%m-%d}<br>Агрессия: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Суммарный объём сообщений и средняя доля агрессии</b>", font=dict(size=16)),
        xaxis=dict(title="Дата", gridcolor="#f1f5f9"),
        yaxis=dict(title="Число сообщений", gridcolor="#f1f5f9"),
        yaxis2=dict(
            title="Доля агрессии (%)",
            overlaying="y",
            side="right",
            range=[0, max(30, (by_day["mean_aggression"].max() * 100) * 1.2)],
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=60, b=50),
        height=380,
    )
    return fig


def build_mood_heatmap_figure(daily_df: pd.DataFrame) -> go.Figure:
    """Тепловая карта (Heatmap): Чат × Дата с цветовым кодированием настроения."""
    if daily_df.empty or "mood_index" not in daily_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных для тепловой карты", showarrow=False)
        return fig

    pivot = daily_df.pivot(index="chat_id", columns="date", values="mood_index")
    pivot = pivot.sort_index(ascending=False)

    chats = [f"Чат {c}" for c in pivot.index]
    dates = [str(d) for d in pivot.columns]
    z_values = pivot.values

    # RdYlGn: темно-красный (-1.0) -> желтый (0.0) -> зеленый (+1.0)
    colorscale = [
        [0.0, "#b91c1c"],
        [0.3, "#f87171"],
        [0.5, "#facc15"],
        [0.7, "#4ade80"],
        [1.0, "#15803d"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=dates,
            y=chats,
            colorscale=colorscale,
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(title="Индекс настроения", tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0]),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>Дата: %{x}<br>Индекс: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Тепловая карта настроения (Чат × Дата)</b>", font=dict(size=16)),
        xaxis=dict(title="Дата", tickangle=-45),
        yaxis=dict(title="Домовой чат"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=80, r=40, t=60, b=80),
        height=320,
    )
    return fig


def build_sentiment_breakdown_figure(summary_df: pd.DataFrame) -> go.Figure:
    """Распределение долей настроений по чатам (stacked bar)."""
    fig = go.Figure()

    if summary_df.empty:
        fig.add_annotation(text="Нет данных по тональности", showarrow=False)
        return fig

    # Оценка долей positive, neutral, negative
    # mood_index = p_pos - p_neg => p_pos = mood_index + p_neg
    # p_neg = negative_share
    # p_neu = 1 - p_pos - p_neg
    chats = [f"Чат {c}" for c in summary_df["chat_id"]]
    neg_shares = summary_df["negative_share"].fillna(0) * 100
    pos_shares = (summary_df["mood_index"].fillna(0) + summary_df["negative_share"].fillna(0)).clip(lower=0) * 100
    neu_shares = (100 - neg_shares - pos_shares).clip(lower=0)

    fig.add_trace(
        go.Bar(
            name="Негативные (%)",
            x=chats,
            y=neg_shares,
            marker_color="#ef4444",
            hovertemplate="Негатив: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Нейтральные (%)",
            x=chats,
            y=neu_shares,
            marker_color="#94a3b8",
            hovertemplate="Нейтрал: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Позитивные (%)",
            x=chats,
            y=pos_shares,
            marker_color="#22c55e",
            hovertemplate="Позитив: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="stack",
        title=dict(text="<b>Структура тональности сообщений по чатам</b>", font=dict(size=16)),
        xaxis=dict(title="Домовой чат"),
        yaxis=dict(title="Доля реплик жителей (%)", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=50, r=30, t=60, b=50),
        height=360,
    )
    return fig


def build_hourly_profile_figure(messages_df: pd.DataFrame | None) -> go.Figure:
    """Почасовой профиль активности и агрессии (ЖКХ-аналитика)."""
    fig = go.Figure()

    if messages_df is None or messages_df.empty or "timestamp" not in messages_df.columns:
        fig.add_annotation(text="Исходные таймстампы сообщений не переданы", showarrow=False)
        return fig

    df = messages_df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["dt"])

    if df.empty:
        fig.add_annotation(text="Нет валидных временных меток", showarrow=False)
        return fig

    df["hour"] = df["dt"].dt.hour

    is_res = (df["is_service"] == 0) if "is_service" in df.columns else pd.Series(True, index=df.index)
    res_df = df[is_res]

    hourly = res_df.groupby("hour").agg(
        total_msgs=("message_id", "count"),
        agg_msgs=("aggression", lambda s: (s == 1).sum() if "aggression" in res_df.columns else 0),
        neg_msgs=("sentiment", lambda s: (s == "negative").sum() if "sentiment" in res_df.columns else 0),
    ).reindex(range(24), fill_value=0).reset_index()

    hourly["agg_share"] = (hourly["agg_msgs"] / hourly["total_msgs"].replace(0, 1)) * 100

    fig.add_trace(
        go.Bar(
            x=hourly["hour"],
            y=hourly["total_msgs"],
            name="Сообщений в час",
            marker_color="#93c5fd",
            hovertemplate="Час %{x}:00<br>Сообщений: %{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["agg_share"],
            name="Доля агрессии (%)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#be123c", width=2.5),
            marker=dict(size=6),
            hovertemplate="Час %{x}:00<br>Агрессия: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Суточный профиль активности и конфликтности (по часам)</b>", font=dict(size=16)),
        xaxis=dict(title="Час суток (0:00 - 23:00)", dtick=1),
        yaxis=dict(title="Количество сообщений жителей"),
        yaxis2=dict(title="Доля агрессии (%)", overlaying="y", side="right"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=60, b=50),
        height=360,
    )
    return fig
