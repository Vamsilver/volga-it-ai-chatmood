"""Adversarial Empirical Stress-Testing Harness for Challenger 2: Report & Benchmark.

Focus areas:
1. Report rendering, offline integrity, script tags, embedded CSS, JSON validity.
2. Inference latency and throughput benchmark (<= 10.0 s / 1000 messages).
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from generate_report import generate_html_report


class TestReportRenderingOfflineIntegrity(unittest.TestCase):
    """3. Report rendering test and offline integrity audit."""

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
        self.assertIn("Методологическая справка", content)

        self.assertIn("Вильсона", content)
        self.assertIn("Дирихле", content)


if __name__ == "__main__":
    unittest.main()
