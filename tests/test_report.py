"""Unit-тесты для модуля построения отчётов и визуализации (Milestone 4)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.report.charts import (
    build_aggression_volume_figure,
    build_hourly_profile_figure,
    build_mood_dynamics_figure,
    build_mood_heatmap_figure,
    build_sentiment_breakdown_figure,
)
from src.report.confidence import (
    calculate_confidence_kpi,
    dirichlet_bayes_mood_ci,
    enrich_daily_with_confidence,
    wilson_score_interval,
)
from src.report.generator import (
    calculate_report_kpi,
    extract_incident_cards,
    generate_html_report,
    load_report_datasets,
)


class TestReportCharts(unittest.TestCase):
    """Тестирование корректности построения графиков Plotly."""

    def setUp(self) -> None:
        self.daily_df = pd.DataFrame({
            "chat_id": ["c1", "c1", "c2", "c2"],
            "date": ["2026-06-15", "2026-06-16", "2026-06-15", "2026-06-16"],
            "n_messages": [10, 15, 8, 12],
            "n_resident": [9, 14, 7, 11],
            "mood_index": [-0.2, 0.1, 0.3, -0.4],
            "negative_share": [0.3, 0.1, 0.1, 0.5],
            "aggression_share": [0.1, 0.0, 0.0, 0.2],
        })

        self.summary_df = pd.DataFrame({
            "chat_id": ["c1", "c2"],
            "n_resident": [23, 18],
            "mood_index": [-0.05, -0.05],
            "negative_share": [0.2, 0.3],
            "aggression_share": [0.05, 0.1],
            "trend": ["stable", "worsening"],
            "peak_days": ["2026-06-15", "2026-06-16"],
        })

        self.messages_df = pd.DataFrame({
            "message_id": ["m1", "m2", "m3"],
            "chat_id": ["c1", "c1", "c2"],
            "timestamp": ["2026-06-15 10:15:00", "2026-06-15 14:20:00", "2026-06-16 21:05:00"],
            "is_service": [0, 1, 0],
            "sentiment": ["negative", "neutral", "positive"],
            "aggression": [1, 0, 0],
            "text": ["Отключили воду!", "УК напоминает", "Спасибо"],
        })

    def test_build_mood_dynamics_figure(self) -> None:
        fig = build_mood_dynamics_figure(self.daily_df)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_build_aggression_volume_figure(self) -> None:
        fig = build_aggression_volume_figure(self.daily_df)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)  # bar + scatter

    def test_build_mood_heatmap_figure(self) -> None:
        fig = build_mood_heatmap_figure(self.daily_df)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)

    def test_build_sentiment_breakdown_figure(self) -> None:
        fig = build_sentiment_breakdown_figure(self.summary_df)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 3)  # Negative, Neutral, Positive

    def test_build_hourly_profile_figure(self) -> None:
        fig = build_hourly_profile_figure(self.messages_df)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)

    def test_build_mood_dynamics_figure_low_volume_distinction(self) -> None:
        df = pd.DataFrame({
            "chat_id": ["c1", "c1"],
            "date": ["2026-06-15", "2026-06-16"],
            "n_resident": [2, 10],
            "mood_index": [-0.5, 0.2],
            "negative_share": [0.5, 0.1],
            "aggression_share": [0.0, 0.0],
        })
        fig = build_mood_dynamics_figure(df)
        self.assertIsNotNone(fig)
        trace = fig.data[0]
        self.assertEqual(trace.marker.symbol[0], "circle-open")
        self.assertEqual(trace.marker.symbol[1], "circle")
        self.assertIn("95% ДИ", trace.hovertemplate)


class TestReportGenerator(unittest.TestCase):
    """Тестирование функций генератора отчёта и формирования HTML."""

    def setUp(self) -> None:
        self.pred_dir = Path("dev_out")
        self.messages_path = Path("data/dev_messages.csv")

    def test_load_report_datasets(self) -> None:
        summary_df, daily_df, merged = load_report_datasets(self.pred_dir, self.messages_path)
        self.assertFalse(summary_df.empty)
        self.assertFalse(daily_df.empty)
        self.assertFalse(merged.empty)
        self.assertIn("chat_id", summary_df.columns)
        self.assertIn("mood_index", daily_df.columns)

    def test_calculate_report_kpi(self) -> None:
        summary_df, _, merged = load_report_datasets(self.pred_dir, self.messages_path)
        kpi = calculate_report_kpi(summary_df, merged)
        self.assertIn("total_messages", kpi)
        self.assertIn("global_mood", kpi)
        self.assertIn("service_share", kpi)
        self.assertGreater(kpi["total_messages"], 0)

    def test_calculate_report_kpi_with_daily(self) -> None:
        summary_df, daily_df, merged = load_report_datasets(self.pred_dir, self.messages_path)
        kpi = calculate_report_kpi(summary_df, merged, daily_df=daily_df)
        self.assertIn("low_volume_days", kpi)
        self.assertIn("low_volume_share", kpi)
        self.assertGreater(kpi["low_volume_days"], 0)

    def test_extract_incident_cards(self) -> None:
        summary_df, daily_df, merged = load_report_datasets(self.pred_dir, self.messages_path)
        incidents = extract_incident_cards(summary_df, daily_df, merged)
        self.assertEqual(len(incidents), len(summary_df))
        for inc in incidents:
            self.assertIn("chat_id", inc)
            self.assertIn("risk_class", inc)
            self.assertIn("peaks", inc)

    def test_generate_html_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_report.html"
            res_path = generate_html_report(
                pred_dir=self.pred_dir,
                output_path=out_file,
                messages_path=self.messages_path,
            )
            self.assertTrue(res_path.is_file())
            content = res_path.read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("Волга-IT 2026", content)
            self.assertIn("Сравнительный рейтинг", content)
            self.assertIn("Тепловая карта", content)
            self.assertIn("Детальный разбор инцидентов", content)
            self.assertIn("ДИ Вильсона", content)
            self.assertIn("Дирихле", content)


class TestReportConfidence(unittest.TestCase):
    """Тестирование математической корректности доверительных интервалов."""

    def test_wilson_score_interval_bounds(self) -> None:
        low, high = wilson_score_interval(k=0, n=10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLessEqual(high, 1.0)

        low_1, high_1 = wilson_score_interval(k=10, n=10)
        self.assertGreater(low_1, 0.0)
        self.assertEqual(high_1, 1.0)

    def test_wilson_score_interval_single_observation(self) -> None:
        low, high = wilson_score_interval(k=1, n=1)
        self.assertAlmostEqual(low, 0.2065, places=3)
        self.assertAlmostEqual(high, 1.0, places=3)

    def test_wilson_score_interval_zero_n(self) -> None:
        import math
        low, high = wilson_score_interval(k=0, n=0)
        self.assertTrue(math.isnan(low))
        self.assertTrue(math.isnan(high))

    def test_dirichlet_bayes_mood_ci_bounds(self) -> None:
        low, high, smoothed = dirichlet_bayes_mood_ci(n_resident=1, mood_index=-1.0, negative_share=1.0)
        self.assertGreaterEqual(low, -1.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(smoothed, 0.0)
        self.assertAlmostEqual(smoothed, -0.375, places=3)

    def test_dirichlet_bayes_mood_ci_convergence(self) -> None:
        low, high, smoothed = dirichlet_bayes_mood_ci(n_resident=1000, mood_index=0.5, negative_share=0.1)
        self.assertAlmostEqual(smoothed, 0.5, places=1)
        self.assertLess(high - low, 0.15)

    def test_enrich_daily_with_confidence(self) -> None:
        df = pd.DataFrame({
            "chat_id": ["c1", "c1", "c1"],
            "date": ["2026-06-15", "2026-06-16", "2026-06-17"],
            "n_resident": [0, 2, 10],
            "mood_index": [None, -0.5, 0.2],
            "negative_share": [None, 0.5, 0.1],
            "aggression_share": [None, 0.0, 0.0],
        })
        enriched = enrich_daily_with_confidence(df)
        self.assertIn("is_low_volume", enriched.columns)
        self.assertIn("mood_ci_lower", enriched.columns)
        self.assertIn("mood_ci_upper", enriched.columns)
        self.assertIn("mood_smoothed", enriched.columns)
        self.assertIn("warning_badge", enriched.columns)

        self.assertFalse(enriched.loc[0, "is_low_volume"])
        self.assertTrue(enriched.loc[1, "is_low_volume"])
        self.assertIn("⚠️", enriched.loc[1, "warning_badge"])
        self.assertFalse(enriched.loc[2, "is_low_volume"])
        self.assertEqual(enriched.loc[2, "warning_badge"], "")

    def test_calculate_confidence_kpi(self) -> None:
        df = pd.DataFrame({
            "chat_id": ["c1", "c1", "c1", "c1"],
            "n_resident": [0, 1, 4, 10],
        })
        kpi = calculate_confidence_kpi(df)
        self.assertEqual(kpi["total_days"], 4)
        self.assertEqual(kpi["zero_volume_days"], 1)
        self.assertEqual(kpi["low_volume_days"], 2)
        self.assertEqual(kpi["reliable_days"], 1)
        self.assertEqual(kpi["low_volume_share"], 50.0)


if __name__ == "__main__":
    unittest.main()
