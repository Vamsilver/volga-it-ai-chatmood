"""Adversarial Empirical Stress-Testing Harness for Challenger 2.

Focus areas:
1. Statistical boundary cases for confidence intervals (n=0, n=1, n=10000, bounds, Dirichlet, Wilson).
2. Schema & invariant preservation (dev_out/chat_daily.csv & dev_out/chat_summary.csv).
3. Report rendering, offline integrity, script tags, embedded CSS, JSON validity.
4. Inference latency and throughput benchmark (<= 10.0 s / 1000 messages).
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import chatmood_metrics as cm
from generate_report import generate_html_report
from predict import run_pipeline
from src.report.confidence import (
    calculate_confidence_kpi,
    dirichlet_bayes_mood_ci,
    enrich_daily_with_confidence,
    get_z_score,
    wilson_score_interval,
)


class TestStatisticalBoundaries(unittest.TestCase):
    """1. Stress-test of statistical boundary conditions and confidence intervals."""

    def test_z_score_bounds(self) -> None:
        """Verify z-score calculation and input validations."""
        z95 = get_z_score(0.95)
        self.assertAlmostEqual(z95, 1.95996, places=4)
        z99 = get_z_score(0.99)
        self.assertAlmostEqual(z99, 2.57583, places=4)

        with self.assertRaises(ValueError):
            get_z_score(0.0)
        with self.assertRaises(ValueError):
            get_z_score(1.0)
        with self.assertRaises(ValueError):
            get_z_score(-0.5)
        with self.assertRaises(ValueError):
            get_z_score(1.5)

    def test_wilson_boundary_zero_and_negative(self) -> None:
        """n=0, n<0, NaNs, None must return (nan, nan) without crashing or division by zero."""
        cases = [
            (0, 0),
            (0, -1),
            (-5, 0),
            (float("nan"), 0),
            (0, float("nan")),
            (None, 0),
            (0, None),
            (float("nan"), float("nan")),
            (-10, -5),
        ]
        for k, n in cases:
            low, high = wilson_score_interval(k=k, n=n)  # type: ignore
            self.assertTrue(math.isnan(low), f"Failed low isnan for k={k}, n={n}: got {low}")
            self.assertTrue(math.isnan(high), f"Failed high isnan for k={k}, n={n}: got {high}")

    def test_wilson_boundary_clamping(self) -> None:
        """k out of [0, n] bounds must be gracefully clamped."""
        # k > n
        low, high = wilson_score_interval(k=15, n=10)
        self.assertAlmostEqual(high, 1.0, places=5)
        self.assertGreater(low, 0.0)
        self.assertLessEqual(low, 1.0)

        # k < 0
        low, high = wilson_score_interval(k=-5, n=10)
        self.assertAlmostEqual(low, 0.0, places=5)
        self.assertGreaterEqual(high, 0.0)
        self.assertLess(high, 1.0)

    def test_wilson_boundary_n_one(self) -> None:
        """n=1 message: verify Wilson bounds remain strictly in [0.0, 1.0]."""
        # 1 target event out of 1 trial (e.g. 1 negative message out of 1 resident msg)
        low_1, high_1 = wilson_score_interval(k=1, n=1, confidence=0.95)
        self.assertGreaterEqual(low_1, 0.0)
        self.assertLessEqual(high_1, 1.0)
        self.assertAlmostEqual(low_1, 0.2065, places=3)
        self.assertAlmostEqual(high_1, 1.0000, places=3)

        # 0 target events out of 1 trial
        low_0, high_0 = wilson_score_interval(k=0, n=1, confidence=0.95)
        self.assertGreaterEqual(low_0, 0.0)
        self.assertLessEqual(high_0, 1.0)
        self.assertAlmostEqual(low_0, 0.0000, places=3)
        self.assertAlmostEqual(high_0, 0.7935, places=3)

    def test_wilson_asymptotic_convergence_n_10000(self) -> None:
        """n=10,000 messages: verify asymptotic convergence and narrow interval."""
        # Case 1: p = 0.5 (k = 5000)
        low, high = wilson_score_interval(k=5000, n=10000, confidence=0.95)
        width = high - low
        center = (low + high) / 2.0
        self.assertAlmostEqual(center, 0.5, places=3)
        self.assertLess(width, 0.021)  # ~1.96 * sqrt(0.25 / 10000) * 2 ≈ 0.0196

        # Case 2: p = 0.0 (k = 0)
        low_0, high_0 = wilson_score_interval(k=0, n=10000, confidence=0.95)
        self.assertEqual(low_0, 0.0)
        self.assertLess(high_0, 0.0005)

        # Case 3: p = 1.0 (k = 10000)
        low_1, high_1 = wilson_score_interval(k=10000, n=10000, confidence=0.95)
        self.assertEqual(high_1, 1.0)
        self.assertGreater(low_1, 0.9995)

    def test_dirichlet_boundary_zero_and_negative(self) -> None:
        """n=0, n<0, NaNs, None must return (nan, nan, nan) without division by zero."""
        cases = [0, -1, -100, float("nan"), None]
        for n in cases:
            low, high, smooth = dirichlet_bayes_mood_ci(n_resident=n)  # type: ignore
            self.assertTrue(math.isnan(low), f"Dirichlet low wasn't nan for n={n}")
            self.assertTrue(math.isnan(high), f"Dirichlet high wasn't nan for n={n}")
            self.assertTrue(math.isnan(smooth), f"Dirichlet smooth wasn't nan for n={n}")

    def test_dirichlet_boundary_n_one_extreme_cases(self) -> None:
        """n=1: test bounds strictly in [-1.0, 1.0] for all extreme configurations."""
        # Case A: 1 positive message (mood_index = +1.0, neg_share = 0.0)
        low_pos, high_pos, smooth_pos = dirichlet_bayes_mood_ci(
            n_resident=1, mood_index=1.0, negative_share=0.0, confidence=0.95
        )
        self.assertGreaterEqual(low_pos, -1.0)
        self.assertLessEqual(high_pos, 1.0)
        self.assertGreater(smooth_pos, 0.0)
        self.assertLessEqual(smooth_pos, 1.0)
        self.assertTrue(low_pos <= smooth_pos <= high_pos)
        # Prior alpha=(0.5, 1.5, 1.0), N0=3.0 -> denom=4.0
        # k_pos=1, k_neg=0 -> p_pos=1.5/4=0.375, p_neg=1.0/4=0.25 -> smooth = 0.125
        self.assertAlmostEqual(smooth_pos, 0.125, places=3)

        # Case B: 1 negative message (mood_index = -1.0, neg_share = 1.0)
        low_neg, high_neg, smooth_neg = dirichlet_bayes_mood_ci(
            n_resident=1, mood_index=-1.0, negative_share=1.0, confidence=0.95
        )
        self.assertGreaterEqual(low_neg, -1.0)
        self.assertLessEqual(high_neg, 1.0)
        self.assertLess(smooth_neg, 0.0)
        self.assertGreaterEqual(smooth_neg, -1.0)
        self.assertTrue(low_neg <= smooth_neg <= high_neg)
        # k_pos=0, k_neg=1 -> p_pos=0.5/4=0.125, p_neg=2.0/4=0.5 -> smooth = -0.375
        self.assertAlmostEqual(smooth_neg, -0.375, places=3)

        # Case C: 1 neutral message (mood_index = 0.0, neg_share = 0.0)
        low_neu, high_neu, smooth_neu = dirichlet_bayes_mood_ci(
            n_resident=1, mood_index=0.0, negative_share=0.0, confidence=0.95
        )
        self.assertGreaterEqual(low_neu, -1.0)
        self.assertLessEqual(high_neu, 1.0)
        self.assertTrue(low_neu <= smooth_neu <= high_neu)
        # k_pos=0, k_neg=0 -> p_pos=0.5/4=0.125, p_neg=1.0/4=0.25 -> smooth = -0.125
        self.assertAlmostEqual(smooth_neu, -0.125, places=3)

    def test_dirichlet_asymptotic_convergence_n_10000(self) -> None:
        """n=10,000 messages: verify asymptotic convergence to sample mean."""
        test_points = [
            (1.0, 0.0),    # pure positive
            (0.5, 0.1),    # mostly positive
            (0.0, 0.2),    # neutral
            (-0.4, 0.5),   # negative
            (-1.0, 1.0),   # pure negative
        ]
        for target_mood, neg_sh in test_points:
            low, high, smooth = dirichlet_bayes_mood_ci(
                n_resident=10000,
                mood_index=target_mood,
                negative_share=neg_sh,
                confidence=0.95,
            )
            # Must stay in bounds
            self.assertGreaterEqual(low, -1.0)
            self.assertLessEqual(high, 1.0)
            self.assertTrue(low <= smooth <= high)

            # Shrinkage towards prior must be minimal at n=10000 (< 0.002)
            self.assertAlmostEqual(smooth, target_mood, delta=0.002)

            # CI width should be narrow (< 0.04)
            width = high - low
            self.assertLess(width, 0.04, f"Width {width} too large for n=10000 at mood={target_mood}")

    def test_enrich_daily_adversarial_dataframe(self) -> None:
        """Stress-test enrich_daily_with_confidence on hostile input DataFrames."""
        # 1. Empty DataFrame
        empty_res = enrich_daily_with_confidence(pd.DataFrame())
        self.assertTrue(empty_res.empty)

        # 2. Hostile DataFrame with zeros, negatives, NaNs, floats, edge values
        df = pd.DataFrame({
            "chat_id": ["c1", "c1", "c2", "c2", "c3", "c3", "c4"],
            "date": ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-06", "2026-06-07"],
            "n_messages": [0, 1, 4, 5, 100, 10000, 2],
            "n_resident": [0, 1, 4, 5, 95, 9800, float("nan")],
            "mood_index": [float("nan"), -1.0, 0.5, 0.0, 0.15, -0.05, float("nan")],
            "negative_share": [float("nan"), 1.0, 0.25, 0.2, 0.1, 0.2, float("nan")],
            "aggression_share": [float("nan"), 0.0, 0.0, 0.0, 0.02, 0.01, float("nan")],
        })
        enriched = enrich_daily_with_confidence(df, low_volume_threshold=5)

        # Invariant: original columns must not be modified
        self.assertEqual(len(enriched), 7)
        self.assertTrue(math.isnan(enriched.loc[0, "mood_index"]))
        self.assertEqual(enriched.loc[1, "mood_index"], -1.0)
        self.assertEqual(enriched.loc[3, "mood_index"], 0.0)

        # New columns present
        expected_cols = [
            "is_low_volume", "mood_ci_lower", "mood_ci_upper", "mood_smoothed",
            "neg_ci_lower", "neg_ci_upper", "agg_ci_lower", "agg_ci_upper", "warning_badge"
        ]
        for col in expected_cols:
            self.assertIn(col, enriched.columns)

        # n_resident == 0 row
        self.assertFalse(enriched.loc[0, "is_low_volume"])
        self.assertTrue(math.isnan(enriched.loc[0, "mood_ci_lower"]))

        # n_resident == 1 row (low volume)
        self.assertTrue(enriched.loc[1, "is_low_volume"])
        self.assertIn("⚠️", enriched.loc[1, "warning_badge"])
        self.assertGreaterEqual(enriched.loc[1, "mood_ci_lower"], -1.0)
        self.assertLessEqual(enriched.loc[1, "mood_ci_upper"], 1.0)

        # n_resident == 5 row (boundary reliable)
        self.assertFalse(enriched.loc[3, "is_low_volume"])
        self.assertEqual(enriched.loc[3, "warning_badge"], "")

        # NaN resident row
        self.assertFalse(enriched.loc[6, "is_low_volume"])
        self.assertTrue(math.isnan(enriched.loc[6, "mood_ci_lower"]))

    def test_calculate_confidence_kpi_adversarial(self) -> None:
        """Test calculate_confidence_kpi with various edge cases."""
        # Empty
        kpi_empty = calculate_confidence_kpi(pd.DataFrame())
        self.assertEqual(kpi_empty["total_days"], 0)
        self.assertEqual(kpi_empty["low_volume_share"], 0.0)

        # Missing n_resident
        kpi_no_col = calculate_confidence_kpi(pd.DataFrame({"chat_id": ["c1"]}))
        self.assertEqual(kpi_no_col["total_days"], 0)

        # Mixed counts
        df = pd.DataFrame({"n_resident": [0, 0, 1, 2, 4, 5, 20]})
        kpi = calculate_confidence_kpi(df, low_volume_threshold=5)
        self.assertEqual(kpi["total_days"], 7)
        self.assertEqual(kpi["zero_volume_days"], 2)
        self.assertEqual(kpi["active_days"], 5)
        self.assertEqual(kpi["low_volume_days"], 3)  # [1, 2, 4]
        self.assertEqual(kpi["reliable_days"], 2)    # [5, 20]
        self.assertAlmostEqual(kpi["low_volume_share"], 300.0 / 7, places=3)


class TestDailySummaryInvariants(unittest.TestCase):
    """2. Invariant stress-test: schema, data types, exact raw formula adherence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dev_out = Path("dev_out")
        cls.messages_path = Path("data/dev_messages.csv")
        cls.daily_path = cls.dev_out / "chat_daily.csv"
        cls.summary_path = cls.dev_out / "chat_summary.csv"
        cls.preds_path = cls.dev_out / "messages_pred.csv"

    def test_files_exist(self) -> None:
        self.assertTrue(self.daily_path.is_file(), f"Missing {self.daily_path}")
        self.assertTrue(self.summary_path.is_file(), f"Missing {self.summary_path}")
        self.assertTrue(self.preds_path.is_file(), f"Missing {self.preds_path}")

    def test_chat_daily_csv_schema_and_types(self) -> None:
        """Verify chat_daily.csv adheres strictly to 7-column schema without smoothed values."""
        # Read raw header
        with open(self.daily_path, "r", encoding="utf-8-sig") as f:
            header_line = f.readline().strip()
        columns = header_line.split(";")

        expected_columns = [
            "chat_id", "date", "n_messages", "n_resident",
            "mood_index", "negative_share", "aggression_share"
        ]
        self.assertEqual(columns, expected_columns, f"Schema mismatch! Got {columns}, expected {expected_columns}")

        # Invariant: NO smoothed or confidence interval columns in chat_daily.csv
        forbidden_cols = ["mood_ci_lower", "mood_ci_upper", "mood_smoothed", "is_low_volume"]
        for fc in forbidden_cols:
            self.assertNotIn(fc, columns, f"Forbidden column {fc} found in chat_daily.csv! Schema violated.")

        # Read into DataFrame
        df = pd.read_csv(self.daily_path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        self.assertGreater(len(df), 0, "chat_daily.csv is empty")

        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        for idx, row in df.iterrows():
            chat_id = row["chat_id"].strip()
            self.assertTrue(len(chat_id) > 0, f"Row {idx}: empty chat_id")

            date_val = row["date"].strip()
            self.assertTrue(date_pattern.match(date_val), f"Row {idx}: invalid date format {date_val}")

            n_msg = int(row["n_messages"])
            n_res = int(row["n_resident"])
            self.assertGreater(n_msg, 0, f"Row {idx}: n_messages must be > 0")
            self.assertGreaterEqual(n_res, 0, f"Row {idx}: n_resident must be >= 0")
            self.assertLessEqual(n_res, n_msg, f"Row {idx}: n_resident ({n_res}) > n_messages ({n_msg})")

            # Check shares
            if n_res == 0:
                self.assertEqual(row["mood_index"], "", f"Row {idx}: mood_index must be empty for n_res=0")
                self.assertEqual(row["negative_share"], "", f"Row {idx}: negative_share must be empty for n_res=0")
                self.assertEqual(row["aggression_share"], "", f"Row {idx}: aggression_share must be empty for n_res=0")
            else:
                m_val = float(row["mood_index"])
                neg_val = float(row["negative_share"])
                agg_val = float(row["aggression_share"])
                self.assertTrue(-1.0001 <= m_val <= 1.0001, f"Row {idx}: mood_index {m_val} out of [-1, 1]")
                self.assertTrue(-0.0001 <= neg_val <= 1.0001, f"Row {idx}: negative_share {neg_val} out of [0, 1]")
                self.assertTrue(-0.0001 <= agg_val <= 1.0001, f"Row {idx}: aggression_share {agg_val} out of [0, 1]")

    def test_chat_daily_csv_exact_raw_formula_preservation(self) -> None:
        """Empirically recompute daily aggregates using jury's chatmood_metrics.py and compare cell-by-cell."""
        msgs_df = pd.read_csv(self.messages_path, sep=";", encoding="utf-8-sig")
        preds_df = pd.read_csv(self.preds_path, sep=";", encoding="utf-8-sig")

        # Ground truth daily aggregates from predict output using chatmood_metrics formula
        gold_daily = cm.daily_aggregates(msgs_df, preds_df)

        actual_daily = pd.read_csv(self.daily_path, sep=";", encoding="utf-8-sig")

        # Sort both by chat_id and date
        gold_sorted = gold_daily.sort_values(["chat_id", "date"]).reset_index(drop=True)
        actual_sorted = actual_daily.sort_values(["chat_id", "date"]).reset_index(drop=True)

        self.assertEqual(len(gold_sorted), len(actual_sorted), "Row count mismatch between recomputed and CSV")

        # Cell-by-cell comparison
        for i in range(len(gold_sorted)):
            g_row = gold_sorted.iloc[i]
            a_row = actual_sorted.iloc[i]

            self.assertEqual(g_row["chat_id"], a_row["chat_id"])
            self.assertEqual(g_row["date"], a_row["date"])
            self.assertEqual(int(g_row["n_messages"]), int(a_row["n_messages"]))
            self.assertEqual(int(g_row["n_resident"]), int(a_row["n_resident"]))

            if g_row["n_resident"] == 0:
                self.assertTrue(pd.isna(a_row["mood_index"]))
                self.assertTrue(pd.isna(a_row["negative_share"]))
                self.assertTrue(pd.isna(a_row["aggression_share"]))
            else:
                self.assertAlmostEqual(
                    float(g_row["mood_index"]),
                    float(a_row["mood_index"]),
                    places=5,
                    msg=f"mood_index discrepancy at {g_row['chat_id']} {g_row['date']}"
                )
                self.assertAlmostEqual(
                    float(g_row["negative_share"]),
                    float(a_row["negative_share"]),
                    places=5,
                    msg=f"negative_share discrepancy at {g_row['chat_id']} {g_row['date']}"
                )
                self.assertAlmostEqual(
                    float(g_row["aggression_share"]),
                    float(a_row["aggression_share"]),
                    places=5,
                    msg=f"aggression_share discrepancy at {g_row['chat_id']} {g_row['date']}"
                )

        # Confirm that on low volume days, chat_daily.csv values are strictly the RAW values, NOT Dirichlet-smoothed
        low_vol_rows = actual_sorted[(actual_sorted["n_resident"] > 0) & (actual_sorted["n_resident"] < 5)]
        self.assertGreater(len(low_vol_rows), 0, "No low volume rows found in dataset for test")
        for _, row in low_vol_rows.iterrows():
            raw_mood = float(row["mood_index"])
            _, _, smoothed = dirichlet_bayes_mood_ci(
                n_resident=int(row["n_resident"]),
                mood_index=raw_mood,
                negative_share=float(row["negative_share"]),
            )
            # If raw mood differs from prior (-0.1667), smoothed must differ from raw
            if abs(raw_mood - (-0.1667)) > 0.1:
                self.assertNotEqual(
                    round(raw_mood, 4),
                    round(smoothed, 4),
                    f"Warning: raw_mood unexpectedly identical to smoothed at {row['chat_id']} {row['date']}"
                )

    def test_chat_summary_csv_schema_and_invariants(self) -> None:
        """Verify chat_summary.csv schema, data types, trends, and peak days format."""
        with open(self.summary_path, "r", encoding="utf-8-sig") as f:
            header_line = f.readline().strip()
        columns = header_line.split(";")

        expected_columns = [
            "chat_id", "n_resident", "mood_index",
            "negative_share", "aggression_share", "trend", "peak_days"
        ]
        self.assertEqual(columns, expected_columns, f"Summary schema mismatch: {columns}")

        df = pd.read_csv(self.summary_path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        self.assertGreater(len(df), 0)

        for idx, row in df.iterrows():
            self.assertTrue(len(row["chat_id"].strip()) > 0)
            self.assertGreaterEqual(int(row["n_resident"]), 0)
            self.assertIn(row["trend"].strip(), cm.TRENDS)

            peaks = [p.strip() for p in row["peak_days"].split("|") if p.strip()]
            self.assertLessEqual(len(peaks), cm.PEAK_TOP)
            date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            for p in peaks:
                self.assertTrue(date_pattern.match(p), f"Invalid peak day format {p}")

if __name__ == "__main__":
    unittest.main()


