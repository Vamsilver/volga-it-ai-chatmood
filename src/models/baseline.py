"""Базовая модель классификации сообщений на основе TF-IDF и Logistic Regression.

Обеспечивает быстрое построение бейслайна для трех целевых задач:
1. is_service (бинарная классификация: объявление/рассылка/реклама vs реплика жителя)
2. sentiment (мультиклассовая: negative, neutral, positive)
3. aggression (бинарная: агрессия/хамство vs норма)

Соблюдает доменные инварианты разметки:
- is_service == 1 -> sentiment = 'neutral', aggression = 0
- sentiment != 'negative' -> aggression = 0
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion

from src.data.loader import get_chat_group_splits


class BaselineClassifier:
    """Бейслайн классификатор сообщений домовых чатов."""

    def __init__(
        self,
        random_state: int = 42,
        c_service: float = 1.0,
        c_sentiment: float = 1.0,
        c_aggression: float = 0.5,
        min_df: int = 2,
    ) -> None:
        self.random_state = random_state
        self.c_service = c_service
        self.c_sentiment = c_sentiment
        self.c_aggression = c_aggression
        self.min_df = min_df

        # Комбинация word и char TF-IDF для охвата морфологии, опечаток и ЖКХ-жаргона
        self.vectorizer = FeatureUnion([
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=self.min_df,
                    max_features=12000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b|\[\w+\]",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=self.min_df,
                    max_features=18000,
                    sublinear_tf=True,
                ),
            ),
        ])

        # Классификаторы для каждой целевой задачи
        self.clf_service = LogisticRegression(
            C=self.c_service,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.clf_sentiment = LogisticRegression(
            C=self.c_sentiment,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.clf_aggression = LogisticRegression(
            C=self.c_aggression,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )

        self.is_fitted = False

    def fit(
        self,
        texts: Sequence[str],
        is_service: Sequence[int],
        sentiment: Sequence[str],
        aggression: Sequence[int],
    ) -> BaselineClassifier:
        """Обучение трех классификаторов на общих признаках TF-IDF.

        Args:
            texts: Тексты сообщений.
            is_service: Метки служебных сообщений (0/1).
            sentiment: Тональность ('negative', 'neutral', 'positive').
            aggression: Метки агрессии (0/1).

        Returns:
            Обученный экземпляр self.
        """
        clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
        X = self.vectorizer.fit_transform(clean_texts)

        # 1. Обучение детектора служебных объявлений
        y_svc = np.asarray(is_service, dtype=int)
        self.clf_service.fit(X, y_svc)

        # 2. Обучение классификатора тональности
        y_sent = np.asarray(sentiment, dtype=str)
        self.clf_sentiment.fit(X, y_sent)

        # 3. Обучение детектора агрессии
        y_agg = np.asarray(aggression, dtype=int)
        self.clf_aggression.fit(X, y_agg)

        self.is_fitted = True
        return self

    def predict(self, texts: Sequence[str]) -> pd.DataFrame:
        """Предсказание меток с гарантированным соблюдением доменных инвариантов.

        Args:
            texts: Тексты сообщений.

        Returns:
            DataFrame с колонками ['is_service', 'sentiment', 'aggression'].
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() перед predict().")

        clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
        X = self.vectorizer.transform(clean_texts)

        pred_svc = self.clf_service.predict(X).astype(int)
        pred_sent = self.clf_sentiment.predict(X).astype(str)
        pred_agg = self.clf_aggression.predict(X).astype(int)

        # Применение жестких доменных инвариантов
        pred_sent, pred_agg = self._apply_invariants(pred_svc, pred_sent, pred_agg)

        return pd.DataFrame({
            "is_service": pred_svc,
            "sentiment": pred_sent,
            "aggression": pred_agg,
        })

    @staticmethod
    def _apply_invariants(
        is_service: np.ndarray,
        sentiment: np.ndarray,
        aggression: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Принудительная постобработка для исключения логических противоречий."""
        sent_out = sentiment.copy()
        agg_out = aggression.copy()

        # Инвариант 1: Служебные сообщения всегда нейтральны и без агрессии
        svc_mask = is_service == 1
        sent_out[svc_mask] = "neutral"
        agg_out[svc_mask] = 0

        # Инвариант 2: Агрессия существует исключительно при негативной тональности
        non_neg_mask = sent_out != "negative"
        agg_out[non_neg_mask] = 0

        return sent_out, agg_out

    def predict_messages(self, messages_df: pd.DataFrame) -> pd.DataFrame:
        """Предсказание для DataFrame сообщений в эталонном формате messages_pred.csv.

        Args:
            messages_df: DataFrame с колонками 'message_id' и 'text'.

        Returns:
            DataFrame с колонками: message_id, is_service, sentiment, aggression.
        """
        if "message_id" not in messages_df.columns:
            raise ValueError("Входной DataFrame должен содержать колонку 'message_id'")
        if "text" not in messages_df.columns:
            raise ValueError("Входной DataFrame должен содержать колонку 'text'")

        preds = self.predict(messages_df["text"])
        preds.insert(0, "message_id", messages_df["message_id"].to_numpy())
        return preds

    def save(self, filepath: str | Path) -> None:
        """Сохранение обученной модели на диск."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: str | Path) -> BaselineClassifier:
        """Загрузка обученной модели с диска."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Загруженный объект не является {cls.__name__}")
        return model


def cross_validate_baseline(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    min_df: int = 2,
) -> dict[str, float]:
    """Групповая кросс-валидация бейслайна по chat_id без утечки данных."""
    splits = get_chat_group_splits(train_df, n_splits=n_splits, seed=random_state)

    f1_sent_scores = []
    f1_agg_scores = []
    f1_svc_scores = []
    q1_scores = []

    for train_idx, val_idx in splits:
        tr = train_df.iloc[train_idx]
        val = train_df.iloc[val_idx]

        model = BaselineClassifier(random_state=random_state, min_df=min_df)
        model.fit(tr["text"], tr["is_service"], tr["sentiment"], tr["aggression"])

        preds = model.predict(val["text"])

        # Оценка согласно evaluate.py
        val_res_mask = (val["is_service"].to_numpy() == 0)
        y_val_sent = val["sentiment"].to_numpy()[val_res_mask]
        y_pred_sent = preds["sentiment"].to_numpy()[val_res_mask]

        y_val_agg = val["aggression"].to_numpy()[val_res_mask]
        y_pred_agg = preds["aggression"].to_numpy()[val_res_mask]

        y_val_svc = val["is_service"].to_numpy()
        y_pred_svc = preds["is_service"].to_numpy()

        f1_sent = f1_score(
            y_val_sent,
            y_pred_sent,
            labels=["negative", "neutral", "positive"],
            average="macro",
            zero_division=0,
        )
        f1_agg = f1_score(
            y_val_agg,
            y_pred_agg,
            labels=[1],
            average="macro",
            zero_division=0,
        )
        f1_svc = f1_score(
            y_val_svc,
            y_pred_svc,
            labels=[1],
            average="macro",
            zero_division=0,
        )

        q1 = 0.6 * f1_sent + 0.2 * f1_agg + 0.2 * f1_svc

        f1_sent_scores.append(f1_sent)
        f1_agg_scores.append(f1_agg)
        f1_svc_scores.append(f1_svc)
        q1_scores.append(q1)

    return {
        "f1_sentiment_mean": float(np.mean(f1_sent_scores)),
        "f1_sentiment_std": float(np.std(f1_sent_scores)),
        "f1_aggression_mean": float(np.mean(f1_agg_scores)),
        "f1_aggression_std": float(np.std(f1_agg_scores)),
        "f1_service_mean": float(np.mean(f1_svc_scores)),
        "f1_service_std": float(np.std(f1_svc_scores)),
        "q1_mean": float(np.mean(q1_scores)),
        "q1_std": float(np.std(q1_scores)),
    }
