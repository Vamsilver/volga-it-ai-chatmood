#!/usr/bin/env python3
"""Автоматическая проверка решений финала Волга-IT 2026 «ИИ и анализ данных»
(анализ тональности и настроения домовых чатов).

Считает автоматическую часть критериев 1, 2, 3 и 6 регламента (п. 5.2). Экспертные части
(2–5, 7) жюри выставляет по rubric.md, им помогает LLM-ассистент из jury_prompts.md.

Запуск на готовых ответах участника:
    python evaluate_final.py --gold-messages data/test_messages.csv --gold-labels data/test_labels.csv \
        --pred-dir submissions/ivanov/out --team ivanov --json results/ivanov.json --details results/ivanov_diff.csv

Запуск с прогоном решения (замер времени и памяти для критерия 6):
    python evaluate_final.py --gold-messages data/test_messages.csv --gold-labels data/test_labels.csv \
        --pred-dir work/ivanov --team ivanov \
        --cmd "python predict.py --input {input} --out {out}" --cwd submissions/ivanov

В --cmd подставляются {input} (CSV сообщений) и {out} (= --pred-dir).
Зависимости: pandas, numpy, scikit-learn, scipy; psutil — необязательно (для пиковой памяти).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chatmood_metrics as cm  # noqa: E402

# ---------------------------------------------------------------------------
# Шкалы. L — уровень «бесполезного» решения (0 баллов), H — уровень, при котором ставится максимум
# (около согласованности двух независимых разметчиков). Между ними — линейно.
# ---------------------------------------------------------------------------
SCALE = {
    "c1": (0.40, 0.85),   # Q1 = 0.6·F1macro(sentiment) + 0.2·F1(aggression) + 0.2·F1(is_service)
    "c2": (0.30, 0.85),   # Q2 — см. aggregation_quality()
    "c3": (0.30, 0.80),   # Q3 — средний macro-F1 тональности по сложным подвыборкам
}
POINTS = {"c1": 20, "c2_auto": 14, "c3_auto": 8, "c6": 5}
HARD_TAGS = ("sarcasm", "obscene", "typos", "jargon", "emoji", "mixed", "context")
MIN_TAG_SUPPORT = 15  # тег учитывается, если в тесте не меньше стольких сообщений жителей с ним
# Критерий 6: секунд на 1000 сообщений → баллы (на референсном ПК жюри)
SPEED_STEPS = [(10, 5), (30, 4), (60, 3), (120, 2), (300, 1)]
RAM_LIMIT_GB = 8.0  # превышение пиковой памяти — минус 1 балл


def lin(q: float, key: str) -> float:
    lo, hi = SCALE[key]
    return float(np.clip((q - lo) / (hi - lo), 0.0, 1.0))


def read_csv(path: Path, required: list[str], problems: list[str], name: str) -> pd.DataFrame | None:
    if not path.exists():
        problems.append(f"нет файла {name}")
        return None
    try:
        df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as e:  # noqa: BLE001
        problems.append(f"{name}: не читается как CSV с разделителем «;» ({e})")
        return None
    df.columns = [c.strip() for c in df.columns]
    miss = [c for c in required if c not in df.columns]
    if miss:
        problems.append(f"{name}: нет колонок {miss}")
        return None
    return df


# ---------------------------------------------------------------------------
# Критерий 1: классификация сообщений
# ---------------------------------------------------------------------------
def load_predictions(path: Path, gold: pd.DataFrame, problems: list[str]) -> pd.DataFrame:
    df = read_csv(path, ["message_id", "is_service", "sentiment", "aggression"], problems, "messages_pred.csv")
    base = gold[["message_id"]].copy()
    if df is None:
        base["is_service"], base["sentiment"], base["aggression"] = -1, "missing", -1
        return base
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    bad_sent = ~df["sentiment"].isin(cm.SENTIMENTS)
    if bad_sent.any():
        problems.append(f"messages_pred.csv: {int(bad_sent.sum())} строк с sentiment не из {cm.SENTIMENTS}")
    for c in ("is_service", "aggression"):
        v = pd.to_numeric(df[c].str.strip(), errors="coerce")
        bad = ~v.isin([0, 1])
        if bad.any():
            problems.append(f"messages_pred.csv: {int(bad.sum())} строк с {c} не 0/1")
        df[c] = v.where(~bad, -1).astype(int)
    dup = df["message_id"].duplicated()
    if dup.any():
        problems.append(f"messages_pred.csv: {int(dup.sum())} повторных message_id (берётся первый)")
        df = df[~dup]
    extra = ~df["message_id"].isin(gold["message_id"])
    if extra.any():
        problems.append(f"messages_pred.csv: {int(extra.sum())} message_id нет в тесте (игнорируются)")
    out = base.merge(df[["message_id", "is_service", "sentiment", "aggression"]], on="message_id", how="left")
    miss = out["sentiment"].isna()
    if miss.any():
        problems.append(f"messages_pred.csv: нет предсказаний для {int(miss.sum())} сообщений (считаются ошибкой)")
    out["sentiment"] = out["sentiment"].fillna("missing")
    out[["is_service", "aggression"]] = out[["is_service", "aggression"]].fillna(-1).astype(int)
    return out


def classification_quality(gold: pd.DataFrame, pred: pd.DataFrame) -> dict:
    m = gold.merge(pred, on="message_id", suffixes=("", "_p"))
    res = m[m["is_service"] == 0]
    f1_sent = f1_score(res["sentiment"], res["sentiment_p"], labels=list(cm.SENTIMENTS), average="macro", zero_division=0)
    f1_agg = f1_score(res["aggression"], res["aggression_p"], labels=[1], average="macro", zero_division=0)
    f1_svc = f1_score(m["is_service"], m["is_service_p"], labels=[1], average="macro", zero_division=0)
    per_class = f1_score(res["sentiment"], res["sentiment_p"], labels=list(cm.SENTIMENTS), average=None, zero_division=0)
    q = 0.6 * f1_sent + 0.2 * f1_agg + 0.2 * f1_svc
    return {
        "f1_sentiment_macro": round(f1_sent, 4),
        "f1_sentiment_per_class": dict(zip(cm.SENTIMENTS, np.round(per_class, 4).tolist())),
        "f1_aggression": round(f1_agg, 4),
        "f1_service": round(f1_svc, 4),
        "accuracy_sentiment": round(float((res["sentiment"] == res["sentiment_p"]).mean()), 4),
        "Q1": round(q, 4),
        "points": round(POINTS["c1"] * lin(q, "c1"), 2),
    }


# ---------------------------------------------------------------------------
# Критерий 3 (авто): сложные подвыборки
# ---------------------------------------------------------------------------
def hard_cases_quality(gold: pd.DataFrame, pred: pd.DataFrame) -> dict:
    m = gold.merge(pred, on="message_id", suffixes=("", "_p"))
    res = m[m["is_service"] == 0]
    per_tag = {}
    for t in HARD_TAGS:
        sub = res[res["tags"].str.split("|").apply(lambda x: t in x)]
        if len(sub) < MIN_TAG_SUPPORT:
            continue
        labs = [s for s in cm.SENTIMENTS if (sub["sentiment"] == s).any()]
        per_tag[t] = {
            "n": int(len(sub)),
            "f1_macro": round(f1_score(sub["sentiment"], sub["sentiment_p"], labels=labs, average="macro", zero_division=0), 4),
            "accuracy": round(float((sub["sentiment"] == sub["sentiment_p"]).mean()), 4),
        }
    q = float(np.mean([v["f1_macro"] for v in per_tag.values()])) if per_tag else 0.0
    return {"per_tag": per_tag, "Q3": round(q, 4), "points_auto": round(POINTS["c3_auto"] * lin(q, "c3"), 2)}


# ---------------------------------------------------------------------------
# Критерий 2 (авто): агрегаты
# ---------------------------------------------------------------------------
def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".").str.strip(), errors="coerce")


def aggregation_quality(gold_msgs: pd.DataFrame, gold: pd.DataFrame, pred_dir: Path, problems: list[str]) -> dict:
    gd, gs = cm.gold_aggregates(gold_msgs, gold)
    daily = read_csv(pred_dir / "chat_daily.csv",
                     ["chat_id", "date", "n_messages", "n_resident", "mood_index", "negative_share", "aggression_share"],
                     problems, "chat_daily.csv")
    summ = read_csv(pred_dir / "chat_summary.csv",
                    ["chat_id", "n_resident", "mood_index", "negative_share", "aggression_share", "trend", "peak_days"],
                    problems, "chat_summary.csv")
    parts = {}
    # --- дневная динамика ---------------------------------------------------
    if daily is not None:
        for c in ("n_messages", "n_resident", "mood_index", "negative_share", "aggression_share"):
            daily[c] = _to_float(daily[c])
        daily["date"] = daily["date"].str.slice(0, 10)
        g = gd[gd["n_resident"] >= 3][["chat_id", "date", "mood_index", "negative_share", "n_resident"]]
        j = g.merge(daily, on=["chat_id", "date"], how="left", suffixes=("", "_p"))
        cover = float(j["mood_index_p"].notna().mean())
        if cover < 1:
            problems.append(f"chat_daily.csv: нет строк для {int(j['mood_index_p'].isna().sum())} из {len(j)} дней эталона")
        mae = float(np.mean(np.abs(j["mood_index"] - j["mood_index_p"].fillna(0))))
        jj = j.dropna(subset=["negative_share_p"])
        r = float(pearsonr(jj["negative_share"], jj["negative_share_p"])[0]) if len(jj) > 3 and jj["negative_share_p"].std() > 0 else 0.0
        parts["daily_mood_mae"] = round(mae, 4)
        parts["daily_negative_corr"] = round(r, 4)
        parts["daily_coverage"] = round(cover, 4)
        s_mae = float(np.clip(1 - mae / 0.4, 0, 1))
        s_corr = float(np.clip(r, 0, 1))
    else:
        s_mae = s_corr = 0.0
    # --- сводка по чатам ------------------------------------------------------
    if summ is not None:
        for c in ("n_resident", "mood_index", "negative_share", "aggression_share"):
            summ[c] = _to_float(summ[c])
        summ["trend"] = summ["trend"].str.strip().str.lower()
        j = gs.merge(summ, on="chat_id", how="left", suffixes=("", "_p"))
        big = j[j["n_resident"] >= 10].dropna(subset=["mood_index_p"])
        rho = float(spearmanr(big["mood_index"], big["mood_index_p"])[0]) if len(big) >= 4 else 0.0
        rho = 0.0 if np.isnan(rho) else rho
        trend_acc = float((j["trend"] == j["trend_p"]).mean())
        prec = []
        for row in j.itertuples():
            g_days = [d for d in str(row.peak_days).split("|") if d]
            p_days = [d.strip()[:10] for d in str(row.peak_days_p if isinstance(row.peak_days_p, str) else "").split("|") if d.strip()]
            if not g_days:
                continue
            gdt = pd.to_datetime(g_days)
            hit = 0
            for d in p_days[: cm.PEAK_TOP]:
                try:
                    dt = pd.to_datetime(d)
                except Exception:  # noqa: BLE001
                    continue
                hit += int(any(abs((dt - x).days) <= 1 for x in gdt))
            prec.append(hit / len(g_days))
        peak = float(np.mean(prec)) if prec else 0.0
        parts.update({"chat_rank_spearman": round(rho, 4), "trend_accuracy": round(trend_acc, 4), "peak_days_hit": round(peak, 4)})
        s_rank, s_trend, s_peak = float(np.clip(rho, 0, 1)), trend_acc, peak
    else:
        s_rank = s_trend = s_peak = 0.0
    q = 0.30 * s_mae + 0.20 * s_corr + 0.20 * s_rank + 0.15 * s_trend + 0.15 * s_peak
    parts["Q2"] = round(q, 4)
    parts["points_auto"] = round(POINTS["c2_auto"] * lin(q, "c2"), 2)
    return parts


# ---------------------------------------------------------------------------
# Критерий 6: время и память
# ---------------------------------------------------------------------------
def run_solution(cmd: str, cwd: str | None, inp: Path, out: Path, n_msgs: int, timeout: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    full = cmd.format(input=str(inp.resolve()), out=str(out.resolve()))
    t0 = time.time()
    peak = None
    try:
        import psutil  # noqa: WPS433
        p = subprocess.Popen(full, shell=True, cwd=cwd)
        proc = psutil.Process(p.pid)
        peak = 0
        while p.poll() is None:
            try:
                rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
                peak = max(peak, rss)
            except psutil.Error:
                pass
            if time.time() - t0 > timeout:
                p.kill()
                break
            time.sleep(0.2)
        rc = p.returncode
    except ImportError:
        rc = subprocess.call(full, shell=True, cwd=cwd, timeout=timeout)
    dt = time.time() - t0
    per_k = dt / max(n_msgs, 1) * 1000
    pts = next((p for lim, p in SPEED_STEPS if per_k <= lim), 0)
    ram_gb = peak / 2**30 if peak else None
    if ram_gb and ram_gb > RAM_LIMIT_GB:
        pts = max(0, pts - 1)
    if rc != 0:
        pts = 0
    return {"command": full, "return_code": rc, "seconds": round(dt, 1), "sec_per_1000": round(per_k, 2),
            "peak_ram_gb": round(ram_gb, 2) if ram_gb else None, "points": pts}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Проверка решений финала Волга-IT 2026 (тональность чатов)")
    ap.add_argument("--gold-messages", required=True, type=Path)
    ap.add_argument("--gold-labels", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path, help="каталог с messages_pred.csv, chat_daily.csv, chat_summary.csv")
    ap.add_argument("--team", default="")
    ap.add_argument("--cmd", default=None, help="команда запуска решения с {input} и {out}")
    ap.add_argument("--cwd", default=None, help="каталог решения для --cmd")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--details", type=Path, default=None, help="CSV с расхождениями эталона и ответа (для апелляций)")
    a = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    msgs = pd.read_csv(a.gold_messages, sep=";", dtype=str, keep_default_na=False)
    gold = pd.read_csv(a.gold_labels, sep=";", dtype={"message_id": str, "sentiment": str, "tags": str}, keep_default_na=False)
    gold["is_service"] = gold["is_service"].astype(int)
    gold["aggression"] = gold["aggression"].astype(int)

    res: dict = {"team": a.team, "problems": []}
    if a.cmd:
        res["c6_speed"] = run_solution(a.cmd, a.cwd, a.gold_messages, a.pred_dir, len(msgs), a.timeout)
    pred = load_predictions(a.pred_dir / "messages_pred.csv", gold, res["problems"])
    res["c1_classification"] = classification_quality(gold, pred)
    res["c2_aggregation"] = aggregation_quality(msgs, gold, a.pred_dir, res["problems"])
    res["c3_hard_cases"] = hard_cases_quality(gold, pred)
    auto = res["c1_classification"]["points"] + res["c2_aggregation"]["points_auto"] + res["c3_hard_cases"]["points_auto"]
    if "c6_speed" in res:
        auto += res["c6_speed"]["points"]
    res["auto_points"] = round(auto, 2)
    res["auto_points_max"] = POINTS["c1"] + POINTS["c2_auto"] + POINTS["c3_auto"] + (POINTS["c6"] if a.cmd else 0)

    c1, c2, c3 = res["c1_classification"], res["c2_aggregation"], res["c3_hard_cases"]
    print(f"== {a.team or a.pred_dir} ==")
    print(f"К1 классификация: Q1={c1['Q1']}  (F1 тон. {c1['f1_sentiment_macro']}, агрессия {c1['f1_aggression']}, "
          f"служебные {c1['f1_service']})  → {c1['points']} / {POINTS['c1']}")
    agg_str = ", ".join(f"{k}={v}" for k, v in c2.items() if k not in ("Q2", "points_auto"))
    print(f"К2 агрегация (авто): Q2={c2['Q2']}  {{{agg_str}}}  → {c2['points_auto']} / {POINTS['c2_auto']}  (+ эксперт 0–6)")
    tags_str = ", ".join(f"{t}={v['f1_macro']}" for t, v in c3["per_tag"].items())
    print(f"К3 сложные случаи (авто): Q3={c3['Q3']}  {{{tags_str}}}  → {c3['points_auto']} / {POINTS['c3_auto']}  (+ эксперт 0–7)")
    if "c6_speed" in res:
        s = res["c6_speed"]
        print(f"К6 скорость: {s['seconds']} с, {s['sec_per_1000']} с/1000 сообщ., RAM {s['peak_ram_gb']} ГБ, код {s['return_code']} → {s['points']} / 5")
    print(f"Итого автоматически: {res['auto_points']} / {res['auto_points_max']}")
    if res["problems"]:
        print("Замечания к формату:")
        for p in res["problems"]:
            print("  -", p)
    if a.details:
        m = msgs[["message_id", "chat_id", "timestamp", "text"]].merge(gold, on="message_id").merge(
            pred, on="message_id", suffixes=("_gold", "_pred"))
        diff = (m.is_service_gold != m.is_service_pred) | (m.sentiment_gold != m.sentiment_pred) | (m.aggression_gold != m.aggression_pred)
        a.details.parent.mkdir(parents=True, exist_ok=True)
        m[diff].to_csv(a.details, sep=";", index=False, encoding="utf-8-sig")
        print(f"расхождений: {int(diff.sum())} → {a.details}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
