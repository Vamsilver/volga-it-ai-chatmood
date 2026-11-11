"""Модульные тесты для единой точки входа predict.py."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from predict import load_input_messages, parse_args, run_pipeline
from src.aggregation.daily import REQUIRED_DAILY_COLUMNS
from src.aggregation.summary import REQUIRED_SUMMARY_COLUMNS


class TestPredictPipeline(unittest.TestCase):
    """Тестирование точки входа predict.py и генерации выходных файлов."""

    def setUp(self):
        """Создание временного входного файла сообщений."""
        self.texts = [
            "Уважаемые жители, завтра отключение ГВС с 10:00 до 17:00",
            "Спасибо за отличную работу управляющей компании!",
            "Опять лифт сломался, сколько можно терпеть?! Руко@опые!",
        ]
        self.sample_df = pd.DataFrame({
            "message_id": ["m1", "m2", "m3"],
            "chat_id": ["c1", "c1", "c1"],
            "timestamp": ["2026-06-15 10:00:00", "2026-06-15 11:00:00", "2026-06-15 12:00:00"],
            "user_id": ["u1", "u2", "u3"],
            "text": self.texts,
        })

    def test_parse_args_defaults(self):
        """Проверка парсинга параметров командной строки."""
        args = parse_args(["--input", "in.csv", "--out", "out_dir"])
        self.assertEqual(args.input, "in.csv")
        self.assertEqual(args.out, "out_dir")
        self.assertEqual(args.models_dir, "models")
        self.assertEqual(args.batch_size, 64)

    def test_load_input_messages(self):
        """Проверка загрузки входного CSV и валидации колонок."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "messages.csv"
            self.sample_df.to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig")

            loaded_df = load_input_messages(csv_path)
            self.assertEqual(len(loaded_df), 3)
            self.assertIn("message_id", loaded_df.columns)
            self.assertIn("text", loaded_df.columns)

    def test_run_pipeline_end_to_end(self):
        """Сквозное тестирование генерации всех трех файлов."""
        baseline_path = Path("models/baseline.joblib")
        if not baseline_path.is_file():
            self.skipTest("models/baseline.joblib отсутствует на диске")

        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "input.csv"
            out_dir = Path(tmp_dir) / "output"
            self.sample_df.to_csv(in_file, sep=";", index=False, encoding="utf-8-sig")

            res = run_pipeline(
                input_path=in_file,
                output_dir=out_dir,
                models_dir="models",
                device="cpu",
            )
            self.assertEqual(res["n_messages"], 3)

            # 1. messages_pred.csv
            f_msgs = out_dir / "messages_pred.csv"
            self.assertTrue(f_msgs.is_file())
            df_msgs = pd.read_csv(f_msgs, sep=";")
            self.assertListEqual(
                list(df_msgs.columns),
                ["message_id", "is_service", "sentiment", "aggression"],
            )
            self.assertEqual(len(df_msgs), 3)

            # 2. chat_daily.csv
            f_daily = out_dir / "chat_daily.csv"
            self.assertTrue(f_daily.is_file())
            df_daily = pd.read_csv(f_daily, sep=";")
            self.assertListEqual(list(df_daily.columns), REQUIRED_DAILY_COLUMNS)

            # 3. chat_summary.csv
            f_summary = out_dir / "chat_summary.csv"
            self.assertTrue(f_summary.is_file())
            df_summary = pd.read_csv(f_summary, sep=";")
            self.assertListEqual(list(df_summary.columns), REQUIRED_SUMMARY_COLUMNS)


    def test_pipeline_reaction_context_propagation(self):
        """Проверка распространения контекста негатива на реплику-реакцию в пайплайне."""
        baseline_path = Path("models/baseline.joblib")
        if not baseline_path.is_file():
            self.skipTest("models/baseline.joblib отсутствует на диске")

        dialog_df = pd.DataFrame({
            "message_id": ["m10", "m11"],
            "chat_id": ["c99", "c99"],
            "timestamp": ["2026-06-15 10:00:00", "2026-06-15 10:01:00"],
            "user_id": ["u1", "u2"],
            "text": [
                "Лифт сломался, ужас какой-то!",
                "И на 9 тоже",
            ],
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "input.csv"
            out_dir = Path(tmp_dir) / "output"
            dialog_df.to_csv(in_file, sep=";", index=False, encoding="utf-8-sig")

            run_pipeline(
                input_path=in_file,
                output_dir=out_dir,
                models_dir="models",
                device="cpu",
            )
            df_preds = pd.read_csv(out_dir / "messages_pred.csv", sep=";")
            self.assertEqual(len(df_preds), 2)
            self.assertEqual(df_preds.iloc[1]["sentiment"], "negative")


if __name__ == "__main__":
    unittest.main()
