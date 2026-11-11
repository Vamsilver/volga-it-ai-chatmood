"""Модульные тесты для src/data/loader.py."""
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.loader import (
    read_csv_safe,
    validate_schema,
    normalize_timestamps,
    load_messages,
    load_labeled_data,
    load_train,
    load_dev,
    load_unlabeled,
    get_chat_group_splits,
)


class TestDataLoader(unittest.TestCase):
    """Тестирование функций загрузки, валидации и нормализации данных."""

    def test_normalize_timestamps_conversion(self):
        """Проверка конвертации МСК (UTC+3) -> UTC (-3 часа)."""
        df = pd.DataFrame({
            "message_id": ["m1"],
            "chat_id": ["c1"],
            "timestamp": ["2026-06-15 09:12:44"],
            "user_id": ["u1"],
            "text": ["Проверка времени"],
        })
        res = normalize_timestamps(df, to_utc=True)
        self.assertIn("date_msk", res.columns)
        self.assertIn("timestamp_utc", res.columns)
        self.assertEqual(res["date_msk"].iloc[0], "2026-06-15")

        utc_ts = res["timestamp_utc"].iloc[0]
        # 09:12:44 МСК должно стать 06:12:44 UTC
        self.assertEqual(utc_ts.hour, 6)
        self.assertEqual(utc_ts.minute, 12)
        self.assertEqual(utc_ts.second, 44)
        self.assertEqual(str(utc_ts.tz), "UTC")

    def test_validate_schema_missing_columns(self):
        """Проверка генерации исключения при отсутствии колонок."""
        bad_df = pd.DataFrame({"message_id": ["m1"], "text": ["Текст"]})
        with self.assertRaises(ValueError):
            validate_schema(bad_df, is_labeled=False)

    def test_validate_schema_bad_labels(self):
        """Проверка генерации исключения при недопустимых значениях меток."""
        bad_labels_df = pd.DataFrame({
            "message_id": ["m1"],
            "chat_id": ["c1"],
            "timestamp": ["2026-06-15 10:00:00"],
            "user_id": ["u1"],
            "text": ["Текст"],
            "is_service": ["2"],  # Недопустимо
            "sentiment": ["negative"],
            "aggression": ["0"],
        })
        with self.assertRaises(ValueError):
            validate_schema(bad_labels_df, is_labeled=True)

    def test_load_train_integrity(self):
        """Проверка загрузки реального train.csv."""
        df = load_train(data_dir="data")
        self.assertEqual(len(df), 3990)
        self.assertIn("is_service", df.columns)
        self.assertIn("sentiment", df.columns)
        self.assertIn("aggression", df.columns)
        self.assertIn("timestamp_utc", df.columns)
        self.assertIn("date_msk", df.columns)
        self.assertFalse(df["message_id"].isnull().any())

    def test_load_dev_integrity(self):
        """Проверка загрузки реального dev (messages + labels)."""
        df = load_dev(data_dir="data")
        self.assertEqual(len(df), 1091)
        self.assertEqual(df["chat_id"].nunique(), 4)
        self.assertIn("tags", df.columns)
        self.assertIn("timestamp_utc", df.columns)

    def test_load_unlabeled_integrity(self):
        """Проверка загрузки неразмеченного датасета."""
        df = load_unlabeled(data_dir="data")
        self.assertEqual(len(df), 20037)
        self.assertEqual(df["chat_id"].nunique(), 105)

    def test_group_kfold_zero_chat_leakage(self):
        """Проверка отсутствия пересечения чатов между train и val фолдами."""
        df = load_train(data_dir="data")
        splits = list(get_chat_group_splits(df, n_splits=5, seed=42))
        self.assertEqual(len(splits), 5)

        for fold, (train_idx, val_idx) in enumerate(splits):
            train_chats = set(df.iloc[train_idx]["chat_id"])
            val_chats = set(df.iloc[val_idx]["chat_id"])
            overlap = train_chats & val_chats
            self.assertEqual(
                len(overlap), 0, f"Фолд {fold} имеет утечку чатов: {overlap}"
            )


if __name__ == "__main__":
    unittest.main()
