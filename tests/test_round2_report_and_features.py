"""Unit-тесты второго раунда (Milestone 10, Requirement R6 & Features Coverage).

Охватывает:
1. 10 тем ЖКХ и информативное ранжирование цитат (до 280 символов)
2. ДИ Вильсона и сглаживание Дирихле с лентой неопределенности Plotly (fill='tonexty')
3. Офлайн-инференс predict.py с кодом возврата 0 через subprocess
4. Lookback контекста диалогов и сохранение позитива при эмодзи смеха
5. Нормализация расширенного словаря ЖКХ (45+ терминов)
6. Обработка смешанной тональности (союз 'но...')
7. Высокоскоростная сквозная генерация автономного отчёта (< 0.5с)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
import pandas as pd

from src.features.preprocessor import TextPreprocessor, enrich_chat_context
from src.models.ensemble import EnsembleClassifier, MIXED_NEGATIVE_PATTERNS
from src.report.charts import build_mood_dynamics_figure
from src.report.confidence import (
    dirichlet_bayes_mood_ci,
    enrich_daily_with_confidence,
    wilson_score_interval,
)
from src.report.generator import (
    TOPIC_KEYWORDS,
    _detect_topics,
    _extract_day_quotes,
    generate_html_report,
    load_report_datasets,
)


class TestRound2ReportAndFeatures(unittest.TestCase):
    """Набор из 14 специализированных тестов требований Round 2 (R1–R6)."""

    def test_01_thematic_categories_ten_distinct_topics(self) -> None:
        """R6: Проверка наличия всех 10 утвержденных категорий ЖКХ в TOPIC_KEYWORDS."""
        expected_topics = [
            "вода / водоснабжение",
            "лифт",
            "уборка / мусор / пухто",
            "шум / тишина / соседи",
            "парковка / авто / проезд",
            "отопление / теплоноситель",
            "УК / диспетчер / заявки",
            "кровля / фасад / протечки",
            "электрика / освещение",
            "безопасность / домофон / шлагбаум",
        ]
        self.assertEqual(len(TOPIC_KEYWORDS), 10)
        for topic in expected_topics:
            self.assertIn(topic, TOPIC_KEYWORDS)
            self.assertGreaterEqual(len(TOPIC_KEYWORDS[topic]), 5)

    def test_02_thematic_detection_conflict_subset_filtering(self) -> None:
        """R6: _detect_topics фильтрует только конфликтный корпус, игнорируя позитив."""
        day_msgs = pd.DataFrame({
            "text": [
                "Спасибо за чистую воду и горячие батареи!",  # позитив
                "Прекрасная работа сантехников, все отлично",  # позитив
                "В 3 подъезде снова сломался лифт и застряли люди!",  # жалоба
            ],
            "sentiment": ["positive", "positive", "negative"],
            "aggression": [0, 0, 1],
        })
        topics = _detect_topics(day_msgs)
        # Должен определиться "лифт", но не темы воды или отопления
        self.assertIn("лифт", topics)
        self.assertNotIn("вода / водоснабжение", topics)
        self.assertNotIn("отопление / теплоноситель", topics)

    def test_03_extract_day_quotes_severity_and_length_ranking(self) -> None:
        """R6: Цитаты ранжируются по остроте конфликта и длине вместо наивного head(3)."""
        day_msgs = pd.DataFrame({
            "text": [
                "+1",  # короткая жалоба
                "да",   # односложная жалоба
                "тоже", # короткая жалоба
                "У нас затопило щиток на 4 этаже, искрит проводка, срочно вызовите аварийку!",  # развернутая жалоба
            ],
            "sentiment": ["negative", "negative", "negative", "negative"],
            "aggression": [0, 0, 0, 1],
        })
        quotes = _extract_day_quotes(day_msgs, max_quotes=1)
        self.assertEqual(len(quotes), 1)
        self.assertIn("затопило щиток", quotes[0]["text"])
        self.assertEqual(quotes[0]["aggression"], 1)

    def test_04_extract_day_quotes_verbatim_limit_280(self) -> None:
        """R6: Цитаты сохраняют verbatim-длину до 280 символов без усечения на 150."""
        long_complaint = (
            "Уважаемые соседи, на 9 этаже в третьем подъезде сорвало кран с кипятком! "
            "Вода хлещет прямо в шахту лифта и заливает электрощиты на 8 и 7 этажах. "
            "Диспетчер аварийной службы сбрасывает звонки уже более получаса, срочно помогите перекрыть стояк!"
        )
        self.assertGreater(len(long_complaint), 180)
        self.assertLessEqual(len(long_complaint), 280)

        day_msgs = pd.DataFrame({
            "text": [long_complaint],
            "sentiment": ["negative"],
            "aggression": [1],
        })
        quotes = _extract_day_quotes(day_msgs, max_quotes=1)
        self.assertEqual(len(quotes), 1)
        # Текст не должен быть обрезан на 150 символах
        self.assertEqual(quotes[0]["text"], long_complaint)
        self.assertFalse(quotes[0]["text"].endswith("..."))

    def test_05_load_report_datasets_safe_fallback(self) -> None:
        """R6: load_report_datasets корректно работает даже без явного файла сообщений."""
        summary_df, daily_df, merged = load_report_datasets("dev_out", messages_path=None)
        self.assertFalse(summary_df.empty)
        self.assertFalse(daily_df.empty)
        self.assertFalse(merged.empty)
        self.assertIn("text", merged.columns)
        self.assertIn("chat_id", merged.columns)
        self.assertIn("timestamp", merged.columns)

    def test_06_wilson_confidence_interval_bounds(self) -> None:
        """R6: Интервал Вильсона строго ограничен [0, 1] и корректен на малых выборках."""
        # Экстремальный случай n=1, k=1
        ci_low, ci_high = wilson_score_interval(k=1, n=1)
        self.assertGreater(ci_low, 0.0)
        self.assertLessEqual(ci_high, 1.0)
        self.assertAlmostEqual(ci_high, 1.0, places=2)

        # n=1, k=0
        ci_low0, ci_high0 = wilson_score_interval(k=0, n=1)
        self.assertEqual(ci_low0, 0.0)
        self.assertLess(ci_high0, 1.0)

        # Некорректные данные
        ci_nan_low, ci_nan_high = wilson_score_interval(k=0, n=0)
        self.assertTrue(pd.isna(ci_nan_low))
        self.assertTrue(pd.isna(ci_nan_high))

    def test_07_dirichlet_bayes_mood_ci_shrinkage(self) -> None:
        """R6: Байесовское сглаживание Дирихле сжимает волатильность при n=1."""
        ci_low, ci_high, smoothed = dirichlet_bayes_mood_ci(n_resident=1, mood_index=-1.0, negative_share=1.0)
        # Сырой mood_index был -1.0, но сглаженный индекс при N0=3.0 сдвигается к априорному среднему
        self.assertGreater(smoothed, -1.0)
        self.assertLess(smoothed, 0.0)
        self.assertGreaterEqual(ci_low, -1.0)
        self.assertLessEqual(ci_high, 1.0)

    def test_08_report_uncertainty_ribbon_trace(self) -> None:
        """R6: build_mood_dynamics_figure добавляет ленту неопределенности fill='tonexty' на n < 5."""
        df = pd.DataFrame({
            "chat_id": ["chat_test", "chat_test", "chat_test"],
            "date": ["2026-06-01", "2026-06-02", "2026-06-03"],
            "n_resident": [2, 3, 10],  # Первые 2 дня - малый объём n < 5
            "mood_index": [-0.5, 0.0, 0.5],
            "negative_share": [0.5, 0.2, 0.1],
            "aggression_share": [0.0, 0.0, 0.0],
        })
        fig = build_mood_dynamics_figure(df)
        self.assertIsNotNone(fig)

        # Проверяем наличие ленты неопределенности
        has_ribbon = any(
            getattr(trace, "fill", None) == "tonexty"
            and getattr(trace, "fillcolor", None) == "rgba(59, 130, 246, 0.15)"
            for trace in fig.data
        )
        self.assertTrue(has_ribbon, "Не найдена полупрозрачная лента Plotly fill='tonexty' с rgba(59, 130, 246, 0.15)")

    def test_09_offline_predict_cli_exit_code_zero(self) -> None:
        """R1: CLI-скрипт predict.py обязан завершаться строго с кодом 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                "predict.py",
                "--input", "data/dev_messages.csv",
                "--out", tmpdir,
                "--device", "cpu",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"predict.py завершился с ошибкой: {res.stderr}")
            self.assertTrue((Path(tmpdir) / "messages_pred.csv").is_file())
            self.assertTrue((Path(tmpdir) / "chat_daily.csv").is_file())
            self.assertTrue((Path(tmpdir) / "chat_summary.csv").is_file())

    def test_10_laugh_emoji_preserves_positive_sentiment(self) -> None:
        """R2: Смеющиеся эмодзи 😅🤣 в доброжелательном контексте сохраняют позитивный тон."""
        prep = TextPreprocessor()
        text = "Мажорка какая😅 а ещё говорит коммуналка дорогая🤣🤣🤣"
        feats = prep.extract_features(pd.DataFrame({"text": [text]}))
        self.assertEqual(feats.iloc[0]["has_emoji_laugh"], 1)
        self.assertEqual(feats.iloc[0]["has_obscene"], 0)

    def test_11_context_lookback_enrichment(self) -> None:
        """R2: Поддержка окна lookback=3 и обогащение коротких реплик согласия."""
        df = pd.DataFrame({
            "message_id": ["m1", "m2", "m3"],
            "chat_id": ["c1", "c1", "c1"],
            "timestamp": ["2026-06-15 12:00:00", "2026-06-15 12:01:00", "2026-06-15 12:02:00"],
            "text": [
                "В подъезде ужасно пахнет канализацией!",
                "Да, согласен",
                "Тоже",
            ],
        })
        enriched = enrich_chat_context(df, lookback_window=3)
        self.assertEqual(len(enriched), 3)
        self.assertIn("Контекст: В подъезде ужасно пахнет канализацией!", enriched.iloc[1])
        self.assertIn("Контекст:", enriched.iloc[2])

    def test_12_jkh_vocabulary_normalization(self) -> None:
        """R3: Расширенный словарь ЖКХ (45+ терминов) нормализует аббревиатуры."""
        prep = TextPreprocessor()
        terms_to_test = [
            ("ЕИРЦ прислал платежку", "расчетный центр"),
            ("собрание ТСН", "товарищество собственников недвижимости"),
            ("заменили стояк", "трубопровод"),
            ("пухто переполнено", "мусорный контейнер"),
            ("поребрик сломали", "бордюр"),
            ("фановая труба протекла", "канализационная труба"),
            ("перерасчет СОИ", "содержание общего имущества"),
        ]
        for raw, expected_sub in terms_to_test:
            cleaned = prep.clean_text(raw)
            self.assertIn(expected_sub, cleaned, f"Термин '{raw}' не был нормализован в '{expected_sub}'")

    def test_13_mixed_tone_conjunction_handling(self) -> None:
        """R4: Обработка смешанной тональности с союзом 'но...'."""
        examples = [
            "Спасибо за уборку, но трактор сломал забор!",
            "Убрали снег во дворе, но трактор сломал забор!",
            "Лифт сделали, но он не работает и скрипит!",
        ]
        for text in examples:
            matches_mixed = any(pat.search(text) for pat in MIXED_NEGATIVE_PATTERNS)
            self.assertTrue(matches_mixed, f"Паттерн смешанной тональности не среагировал на '{text}'")

    def test_14_generate_report_end_to_end_fast(self) -> None:
        """R6: Генерация полного автономного HTML-отчёта укладывается в норматив < 0.5с."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "speed_report.html"
            t0 = time.perf_counter()
            res_path = generate_html_report(
                pred_dir="dev_out",
                output_path=out_file,
                messages_path="data/dev_messages.csv",
            )
            elapsed = time.perf_counter() - t0
            self.assertTrue(res_path.is_file())
            self.assertGreater(res_path.stat().st_size, 1_000_000)
            self.assertLess(elapsed, 0.5, f"Генерация отчёта заняла {elapsed:.3f}с (лимит 0.5с)")


if __name__ == "__main__":
    unittest.main()
