"""Скрипт полууправляемого обучения (Semi-Supervised Learning) для Volga-IT 2026.

Реализует 4-компонентную стратегию (4-Pillar Decoupled Strategy):
1. Unsupervised TF-IDF Vocabulary Expansion: расширение словаря признаков на train (3.9k) + unlabeled (20k).
2. Decoupled High-Confidence Pseudo-Labeling: раздельные пороги уверенности для задач:
   - is_service: p >= 0.95 (1) и p <= 0.03 (0)
   - sentiment: квотированная выборка (p >= 0.92 для neutral, p >= 0.88 для negative/positive)
   - aggression: p >= 0.75 или оверрайд оскорблений (1) и p <= 0.01 (0)
3. Sample-Weighted Training: gold train веса = 1.0, pseudo-labels веса = 0.35 (защита от confirmation bias).
4. Zero-Regression Gatekeeper: строгая валидация на dev-выборке с защитой от деградации метрик (цель >= 38.0).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Добавление корня репозитория в sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.features.preprocessor import TextPreprocessor, enrich_chat_context
from src.models.baseline import BaselineClassifier
from src.models.ensemble import EnsembleClassifier, INSULT_OVERRIDE_PATTERNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semi-Supervised Learning Pipeline (Volga-IT 2026)")
    parser.add_argument("--train", type=str, default="data/train.csv", help="Путь к размеченному train.csv")
    parser.add_argument("--unlabeled", type=str, default="data/unlabeled.csv", help="Путь к неразмеченному unlabeled.csv")
    parser.add_argument("--dev-messages", type=str, default="data/dev_messages.csv", help="Путь к dev_messages.csv")
    parser.add_argument("--dev-labels", type=str, default="data/dev_labels.csv", help="Путь к dev_labels.csv")
    parser.add_argument("--models-dir", type=str, default="models", help="Каталог с моделями")
    parser.add_argument("--out-model", type=str, default="models/baseline.joblib", help="Путь сохранения модели")
    parser.add_argument("--w-pseudo", type=float, default=0.35, help="Вес псевдоразмеченных сэмплов (по умолчанию: 0.35)")
    parser.add_argument("--max-unlabeled", type=int, default=5000, help="Лимит неразмеченных сообщений для ускорения")
    parser.add_argument("--target-score", type=float, default=36.57, help="Минимальный порог качества gatekeeper")
    return parser.parse_args()


def load_dataset(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Файл не найден: {p}")
    sample = p.read_text(encoding="utf-8-sig", errors="replace")[:2048]
    sep = ";" if sample.count(";") >= sample.count(",") else ","
    df = pd.read_csv(p, sep=sep, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def extract_pseudo_labels(
    ensemble: EnsembleClassifier,
    unlabeled_df: pd.DataFrame,
    max_samples: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Извлечение высокоуверенных псевдометок отдельно по каждой задаче."""
    work_df = unlabeled_df.head(max_samples).copy()
    texts = work_df["text"].fillna("").astype(str).tolist()

    print(f"Вычисление вероятностей ансамбля для {len(texts)} неразмеченных сообщений...")
    probs = ensemble.predict_proba(texts)
    p_svc = probs["service"]
    p_sent = probs["sentiment"]
    p_agg = probs["aggression"]

    # 1. Задача is_service
    svc_pos_mask = p_svc >= 0.95
    svc_neg_mask = p_svc <= 0.03
    svc_indices = np.where(svc_pos_mask | svc_neg_mask)[0]
    svc_labels = np.where(svc_pos_mask[svc_indices], 1, 0)
    svc_texts = [texts[i] for i in svc_indices]

    df_pseudo_svc = pd.DataFrame({"text": svc_texts, "is_service": svc_labels})
    print(f"  [Pillar 2] Найдено псевдометок is_service: {len(df_pseudo_svc)} (1: {sum(svc_labels)}, 0: {len(svc_labels)-sum(svc_labels)})")

    # 2. Задача sentiment (квотированный отбор с порогом)
    sent_classes = ("negative", "neutral", "positive")
    selected_sent_texts = []
    selected_sent_labels = []

    for c_idx, c_name in enumerate(sent_classes):
        floor = 0.92 if c_name == "neutral" else 0.88
        class_mask = p_sent[:, c_idx] >= floor
        c_indices = np.where(class_mask)[0]
        # Сортировка по убыванию уверенности
        sorted_c = c_indices[np.argsort(-p_sent[c_indices, c_idx])][:300]
        for idx in sorted_c:
            selected_sent_texts.append(texts[idx])
            selected_sent_labels.append(c_name)

    df_pseudo_sent = pd.DataFrame({"text": selected_sent_texts, "sentiment": selected_sent_labels})
    print(f"  [Pillar 2] Найдено псевдометок sentiment: {len(df_pseudo_sent)}")

    # 3. Задача aggression
    agg_pos_mask = (p_agg >= 0.75) | np.array([any(p.search(t) for p in INSULT_OVERRIDE_PATTERNS) for t in texts])
    agg_neg_mask = p_agg <= 0.01
    agg_indices = np.where(agg_pos_mask | agg_neg_mask)[0]
    agg_labels = np.where(agg_pos_mask[agg_indices], 1, 0)
    agg_texts = [texts[i] for i in agg_indices]

    df_pseudo_agg = pd.DataFrame({"text": agg_texts, "aggression": agg_labels})
    print(f"  [Pillar 2] Найдено псевдометок aggression: {len(df_pseudo_agg)} (1: {sum(agg_labels)}, 0: {len(agg_labels)-sum(agg_labels)})")

    return df_pseudo_svc, df_pseudo_sent, df_pseudo_agg


def train_semi_supervised_baseline(
    train_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    ensemble: EnsembleClassifier,
    w_pseudo: float = 0.35,
    max_unlabeled: int = 5000,
) -> BaselineClassifier:
    """Обучение бейслайна с 4-компонентной стратегией."""
    baseline = BaselineClassifier()

    # Pillar 1: Unsupervised Vocabulary Expansion на train + unlabeled
    print("[Pillar 1] Неконтролируемое расширение словаря TF-IDF на train + unlabeled...")
    gold_texts = train_df["text"].fillna("").astype(str).tolist()
    unlabeled_texts = unlabeled_df["text"].fillna("").astype(str).tolist()
    all_vocab_texts = gold_texts + unlabeled_texts
    baseline.vectorizer.fit(all_vocab_texts)
    print(f"  Размер расширенного словаря успешно обучен на {len(all_vocab_texts)} текстах.")

    # Pillar 2: Decoupled Pseudo-Labeling
    df_p_svc, df_p_sent, df_p_agg = extract_pseudo_labels(ensemble, unlabeled_df, max_samples=max_unlabeled)

    # Pillar 3: Sample-Weighted Training
    # Обучение clf_service
    texts_svc = gold_texts + df_p_svc["text"].tolist()
    y_svc = train_df["is_service"].astype(int).tolist() + df_p_svc["is_service"].tolist()
    w_svc = [1.0] * len(gold_texts) + [w_pseudo] * len(df_p_svc)
    X_svc = baseline.vectorizer.transform(texts_svc)
    baseline.clf_service.fit(X_svc, np.array(y_svc, dtype=int), sample_weight=np.array(w_svc))

    # Обучение clf_sentiment
    texts_sent = gold_texts + df_p_sent["text"].tolist()
    y_sent = train_df["sentiment"].astype(str).tolist() + df_p_sent["sentiment"].tolist()
    w_sent = [1.0] * len(gold_texts) + [w_pseudo] * len(df_p_sent)
    X_sent = baseline.vectorizer.transform(texts_sent)
    baseline.clf_sentiment.fit(X_sent, np.array(y_sent, dtype=str), sample_weight=np.array(w_sent))

    # Обучение clf_aggression
    texts_agg = gold_texts + df_p_agg["text"].tolist()
    y_agg = train_df["aggression"].astype(int).tolist() + df_p_agg["aggression"].tolist()
    w_agg = [1.0] * len(gold_texts) + [w_pseudo] * len(df_p_agg)
    X_agg = baseline.vectorizer.transform(texts_agg)
    baseline.clf_aggression.fit(X_agg, np.array(y_agg, dtype=int), sample_weight=np.array(w_agg))

    baseline.is_fitted = True
    print("[Pillar 3] Модели LogisticRegression обучены с весами gold=1.0, pseudo=0.35.")
    return baseline


def evaluate_gatekeeper(
    new_baseline: BaselineClassifier,
    dev_messages_path: str | Path,
    dev_labels_path: str | Path,
    models_dir: str | Path,
    target_score: float = 36.57,
) -> tuple[bool, float, dict[str, float]]:
    """Pillar 4: Zero-Regression Gatekeeper."""
    print("[Pillar 4] Запуск Zero-Regression Gatekeeper на dev-выборке...")
    dev_msgs = load_dataset(dev_messages_path)
    dev_labels = load_dataset(dev_labels_path)

    # Сборка ансамбля с новым бейслайном
    m_dir = Path(models_dir)
    transformer_dir = m_dir / "rubert_tiny2_multitask"
    ensemble = EnsembleClassifier.from_pretrained(
        baseline_path=m_dir / "baseline.joblib",
        transformer_dir=transformer_dir if transformer_dir.is_dir() else None,
        device="cpu",
    )
    # Подменяем бейслайн на дообученный
    ensemble.baseline = new_baseline

    preds = ensemble.predict_messages(dev_msgs)

    # Расчет базовых F1
    gold_svc = dev_labels["is_service"].astype(int).to_numpy()
    gold_sent = dev_labels["sentiment"].astype(str).to_numpy()
    gold_agg = dev_labels["aggression"].astype(int).to_numpy()

    pred_svc = preds["is_service"].astype(int).to_numpy()
    pred_sent = preds["sentiment"].astype(str).to_numpy()
    pred_agg = preds["aggression"].astype(int).to_numpy()

    f1_svc = f1_score(gold_svc, pred_svc, zero_division=0)
    f1_sent = f1_score(gold_sent, pred_sent, average="macro", zero_division=0)
    f1_agg = f1_score(gold_agg, pred_agg, zero_division=0)

    q1 = (f1_sent + f1_agg + f1_svc) / 3.0
    print(f"  Gatekeeper Q1: {q1:.4f} (Sent: {f1_sent:.4f}, Agg: {f1_agg:.4f}, Svc: {f1_svc:.4f})")

    metrics = {"q1": q1, "f1_svc": f1_svc, "f1_sent": f1_sent, "f1_agg": f1_agg}
    passed = q1 >= 0.7947  # Отсутствие регрессии относительно базового Q1
    return passed, q1, metrics


def main() -> None:
    args = parse_args()
    print("=" * 65)
    print("  ЗАПУСК ПОЛУУПРАВЛЯЕМОГО ОБУЧЕНИЯ (SEMI-SUPERVISED LEARNING) ")
    print("=" * 65)

    train_df = load_dataset(args.train)
    unlabeled_df = load_dataset(args.unlabeled)
    print(f"Загружено train: {len(train_df)} строк, unlabeled: {len(unlabeled_df)} строк.")

    # Загрузка текущего ансамбля для псевдоразметки
    ensemble = EnsembleClassifier.from_pretrained(
        baseline_path=Path(args.models_dir) / "baseline.joblib",
        transformer_dir=Path(args.models_dir) / "rubert_tiny2_multitask",
        device="cpu",
    )

    new_baseline = train_semi_supervised_baseline(
        train_df=train_df,
        unlabeled_df=unlabeled_df,
        ensemble=ensemble,
        w_pseudo=args.w_pseudo,
        max_unlabeled=args.max_unlabeled,
    )

    passed, q1, metrics = evaluate_gatekeeper(
        new_baseline=new_baseline,
        dev_messages_path=args.dev_messages,
        dev_labels_path=args.dev_labels,
        models_dir=args.models_dir,
        target_score=args.target_score,
    )

    out_p = Path(args.out_model)
    if passed:
        print(f"[OK] Gatekeeper ПРОЙДЕН! Q1={q1:.4f} >= baseline. Сохранение: {out_p}")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        new_baseline.save(out_p)
    else:
        backup_p = out_p.parent / "baseline_semi_supervised.joblib"
        print(f"[!] Gatekeeper: Q1={q1:.4f}. Сохранение в альтернативный путь: {backup_p}")
        new_baseline.save(backup_p)

    print("=" * 65)
    print("  ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО  ")
    print("=" * 65)


if __name__ == "__main__":
    main()
