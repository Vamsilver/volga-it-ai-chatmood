"""Adversarial Empirical Stress-Testing Harness for Challenger 2: Offline Integrity and Benchmark.

Focus areas:
1. Report rendering, offline integrity, script tags, embedded CSS, JSON validity.
2. Inference latency and throughput benchmark (<= 10.0 s / 1000 messages).
"""

from __future__ import annotations

import re
import tempfile
import time
import unittest
from pathlib import Path

from generate_report import generate_html_report
from predict import run_pipeline


class TestReportRenderingOfflineIntegrity(unittest.TestCase):
    """Report rendering test and offline integrity audit."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dev_out = Path("dev_out")
        cls.messages_path = Path("data/dev_messages.csv")
        cls.report_path = Path("report.html")

    def test_report_generation_runs_cleanly(self) -> None:
        """Verify generate_html_report generates report.html without throwing exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_report = Path(tmpdir) / "test_report.html"
            out = generate_html_report(
                pred_dir=self.dev_out,
                output_path=tmp_report,
                messages_path=self.messages_path,
            )
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1_000_000)

    def test_report_html_offline_integrity(self) -> None:
        """Verify report.html is 100% offline-compatible: zero external CDN links or unpkg/cdnjs dependencies."""
        content = self.report_path.read_text(encoding="utf-8")

        # 1. No unrendered Jinja tags outside bundled javascript
        html_without_scripts = re.sub(r"<script.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        jinja_matches = re.findall(r"(\{\{[^}]*\}\}|\{%.*?%\}|\{#.*?#\})", html_without_scripts)
        self.assertEqual(len(jinja_matches), 0, f"Found unrendered Jinja tags in HTML: {jinja_matches[:5]}")

        # 2. Complete HTML structure
        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("<html lang=\"ru\">", content)
        self.assertIn("</html>", content)

        # 3. Offline audit: No external scripts (<script src="http...>)
        external_scripts = re.findall(r'<script[^>]+src=["\'](https?:|//)[^"\']+["\']', content, re.IGNORECASE)
        self.assertEqual(
            len(external_scripts), 0,
            f"VIOLATION OF OFFLINE INVARIANT: Found external script sources: {external_scripts}"
        )

        # 4. No external stylesheets (<link rel="stylesheet" href="http...>)
        external_styles = re.findall(r'<link[^>]+href=["\'](https?:|//)[^"\']+["\']', content, re.IGNORECASE)
        self.assertEqual(
            len(external_styles), 0,
            f"VIOLATION OF OFFLINE INVARIANT: Found external style links: {external_styles}"
        )

        # 5. Check script tags matching and integrity
        open_scripts = len(re.findall(r"<script\b", content, re.IGNORECASE))
        close_scripts = len(re.findall(r"</script>", content, re.IGNORECASE))
        self.assertEqual(open_scripts, close_scripts, f"Mismatched script tags: {open_scripts} open vs {close_scripts} close")

        # 6. Embedded Plotly check
        self.assertIn("Plotly.newPlot", content, "Plotly.newPlot call missing in report.html")

        # 7. Section presence
        self.assertIn("Сравнительный рейтинг напряжённости", content)
        self.assertIn("1. Динамика настроения жителей во времени", content)
        self.assertIn("2. Тепловая карта настроений", content)
        self.assertIn("3. Анализ структуры обращений", content)
        self.assertIn("4. Детальный разбор инцидентов в дни-пики", content)
        self.assertIn("Методологическая справка и формулы регламента", content)
        self.assertIn("Вильсона", content)
        self.assertIn("Дирихле", content)


class TestInferenceBenchmark(unittest.TestCase):
    """Benchmark performance test: inference latency strictly <= 10.0 s / 1000 messages."""

    def test_benchmark_inference_speed(self) -> None:
        input_csv = Path("data/dev_messages.csv")
        self.assertTrue(input_csv.is_file())

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "bench_out"

            # Run full end-to-end pipeline
            t_start = time.perf_counter()
            metrics = run_pipeline(
                input_path=input_csv,
                output_dir=out_dir,
                models_dir="models",
                device="cpu",
                batch_size=64,
            )
            t_total = time.perf_counter() - t_start

            n_msgs = int(metrics["n_messages"])
            sec_per_1000 = (t_total / n_msgs) * 1000.0

            print(f"\n[BENCHMARK RESULT] Total messages: {n_msgs}")
            print(f"[BENCHMARK RESULT] Total pipeline time: {t_total:.3f} s")
            print(f"[BENCHMARK RESULT] Normalized latency: {sec_per_1000:.3f} s / 1000 messages")
            print(f"[BENCHMARK RESULT] Throughput: {n_msgs / t_total:.1f} msgs/sec")

            # Must satisfy Criterion 6: <= 10.0 s per 1000 messages
            self.assertLessEqual(
                sec_per_1000, 10.0,
                f"Performance regression! {sec_per_1000:.2f} s / 1000 messages exceeds 10.0 s limit"
            )


if __name__ == "__main__":
    unittest.main()
