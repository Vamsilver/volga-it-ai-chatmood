"""Модуль загрузки, валидации и нормализации данных домовых чатов.

Обеспечивает строгое соблюдение регламента соревнования:
- Разделитель CSV: ';'
- Кодировка: UTF-8 / UTF-8-sig
- Валидация обязательных полей и допустимых значений меток
- Нормализация временных меток (МСК -> UTC) с сохранением даты МСК
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Обязательные поля для входных сообщений
REQUIRED_MESSAGE_COLUMNS = [
    "message_id",
    "chat_id",
    "timestamp",
    "user_id",
    "text",
]

# Обязательные поля для разметки
REQUIRED_LABEL_COLUMNS = [
    "is_service",
    "sentiment",
    "aggression",
]

# Допустимые значения меток согласно регламенту
VALID_SENTIMENTS = {"negative", "neutral", "positive"}
VALID_BINARY_LABELS = {0, 1}


def read_csv_safe(filepath: str | Path) -> pd.DataFrame:
    """Чтение CSV файла с разделителем ';' и корректной кодировкой UTF-8.

    Args:
        filepath: Путь к CSV файлу.

    Returns:
        pd.DataFrame с очищенными именами колонок.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если файл пуст или некорректно структурирован.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    try:
        df = pd.read_csv(
            path,
            sep=";",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            sep=";",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )

    df.columns = [col.strip() for col in df.columns]
    return df


def validate_schema(
    df: pd.DataFrame,
    is_labeled: bool = False,
    allow_missing_text: bool = False,
) -> None:
    """Валидация схемы колонок и допустимых значений в DataFrame.

    Args:
        df: DataFrame для проверки.
        is_labeled: Флаг проверки колонок разметки.
        allow_missing_text: Разрешать ли пустой текст сообщения.

    Raises:
        ValueError: При отсутствии обязательных колонок или некорректных значениях.
    """
    missing_msg_cols = [c for c in REQUIRED_MESSAGE_COLUMNS if c not in df.columns]
    if missing_msg_cols:
        raise ValueError(f"Отсутствуют обязательные колонки сообщений: {missing_msg_cols}")

    if not allow_missing_text and (df["text"].str.strip() == "").all():
        raise ValueError("Все тексты сообщений в выборке пустые")

    if is_labeled:
        missing_lbl_cols = [c for c in REQUIRED_LABEL_COLUMNS if c not in df.columns]
        if missing_lbl_cols:
            raise ValueError(f"Отсутствуют обязательные колонки разметки: {missing_lbl_cols}")

        _validate_labels(df)


def _validate_labels(df: pd.DataFrame) -> None:
    """Валидация типов и диапазонов целевых меток."""
    # Валидация is_service
    svc_vals = set(pd.to_numeric(df["is_service"], errors="coerce").dropna().astype(int))
    if not svc_vals.issubset(VALID_BINARY_LABELS):
        raise ValueError(f"Недопустимые значения is_service: {svc_vals - VALID_BINARY_LABELS}")

    # Валидация sentiment
    sent_vals = set(df["sentiment"].str.strip().str.lower())
    if not sent_vals.issubset(VALID_SENTIMENTS):
        raise ValueError(f"Недопустимые значения sentiment: {sent_vals - VALID_SENTIMENTS}")

    # Валидация aggression
    agg_vals = set(pd.to_numeric(df["aggression"], errors="coerce").dropna().astype(int))
    if not agg_vals.issubset(VALID_BINARY_LABELS):
        raise ValueError(f"Недопустимые значения aggression: {agg_vals - VALID_BINARY_LABELS}")


def normalize_timestamps(df: pd.DataFrame, to_utc: bool = True) -> pd.DataFrame:
    """Нормализация временных меток из московского времени (МСК, UTC+3) в UTC.

    Добавляет:
    - 'date_msk': Строковая дата в формате YYYY-MM-DD по московскому времени
      (необходима для агрегации в chatmood_metrics.py).
    - 'timestamp_utc': tz-aware UTC datetime (если to_utc=True).

    Args:
        df: Исходный DataFrame с колонкой 'timestamp'.
        to_utc: Флаг генерации UTC метки.

    Returns:
        DataFrame с добавленными колонками нормализованного времени.
    """
    out = df.copy()
    raw_ts = out["timestamp"].astype(str).str.strip()

    # Сохраняем дату по московскому времени (первые 10 символов YYYY-MM-DD)
    out["date_msk"] = raw_ts.str.slice(0, 10)

    if to_utc:
        # Московское время (UTC+3) без летнего времени
        dt_msk = pd.to_datetime(raw_ts, errors="coerce")
        # Локализуем как UTC+3 (Europe/Moscow) и конвертируем в UTC
        dt_msk_aware = dt_msk.dt.tz_localize("Europe/Moscow", ambiguous="NaT", nonexistent="shift_forward")
        out["timestamp_utc"] = dt_msk_aware.dt.tz_convert("UTC")

    return out


def load_messages(
    filepath: str | Path,
    normalize_time: bool = True,
) -> pd.DataFrame:
    """Загрузка сырых сообщений (формат predict.py и dev_messages.csv).

    Args:
        filepath: Путь к CSV с сообщениями.
        normalize_time: Выполнять ли нормализацию времени.

    Returns:
        Валидированный DataFrame сообщений.
    """
    df = read_csv_safe(filepath)
    validate_schema(df, is_labeled=False)
    if normalize_time:
        df = normalize_timestamps(df, to_utc=True)
    return df


def load_labeled_data(
    messages_path: str | Path,
    labels_path: str | Path | None = None,
    normalize_time: bool = True,
) -> pd.DataFrame:
    """Загрузка размеченных данных (train.csv или dev_messages.csv + dev_labels.csv).

    Args:
        messages_path: Путь к CSV сообщений (или объединенному train.csv).
        labels_path: Путь к CSV меток (если они в отдельном файле).
        normalize_time: Выполнять ли нормализацию времени.

    Returns:
        Валидированный размеченный DataFrame с приведенными типами колонок.
    """
    df_msgs = read_csv_safe(messages_path)

    if labels_path is not None:
        df_lbls = read_csv_safe(labels_path)
        if "message_id" not in df_lbls.columns:
            raise ValueError(f"Файл меток {labels_path} не содержит колонку message_id")
        merged = pd.merge(df_msgs, df_lbls, on="message_id", how="inner")
    else:
        merged = df_msgs

    validate_schema(merged, is_labeled=True)

    # Приведение типов числовых меток
    merged["is_service"] = pd.to_numeric(merged["is_service"], errors="coerce").astype(int)
    merged["aggression"] = pd.to_numeric(merged["aggression"], errors="coerce").astype(int)
    merged["sentiment"] = merged["sentiment"].str.strip().str.lower()

    if normalize_time:
        merged = normalize_timestamps(merged, to_utc=True)

    return merged


def load_train(data_dir: str | Path = "data") -> pd.DataFrame:
    """Загрузка обучающего набора train.csv."""
    path = Path(data_dir) / "train.csv"
    return load_labeled_data(path, labels_path=None, normalize_time=True)


def load_dev(data_dir: str | Path = "data") -> pd.DataFrame:
    """Загрузка валидационного набора dev_messages.csv + dev_labels.csv."""
    dpath = Path(data_dir)
    return load_labeled_data(
        messages_path=dpath / "dev_messages.csv",
        labels_path=dpath / "dev_labels.csv",
        normalize_time=True,
    )


def load_unlabeled(data_dir: str | Path = "data") -> pd.DataFrame:
    """Загрузка неразмеченного набора unlabeled.csv."""
    path = Path(data_dir) / "unlabeled.csv"
    return load_messages(path, normalize_time=True)


def get_chat_group_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Генерация фолдов GroupKFold по chat_id для предотвращения утечки контекста.

    Args:
        df: DataFrame с колонкой 'chat_id'.
        n_splits: Количество фолдов.
        seed: Фиксированный random seed (используется для shuffle чатов).

    Yields:
        Кортежи (train_indices, val_indices) для каждого фолда.
    """
    if "chat_id" not in df.columns:
        raise ValueError("DataFrame должен содержать колонку chat_id для группового сплита")

    gkf = GroupKFold(n_splits=n_splits)
    unique_chats = df["chat_id"].unique()
    rng = np.random.RandomState(seed)
    shuffled_chats = rng.permutation(unique_chats)

    # Отображение перемешанных чатов на группы
    chat_to_group = {chat: i % n_splits for i, chat in enumerate(shuffled_chats)}
    groups = df["chat_id"].map(chat_to_group).to_numpy()

    return gkf.split(df, groups=groups)
