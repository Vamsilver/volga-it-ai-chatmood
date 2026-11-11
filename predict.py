"""Единая точка входа для инференса и аналитики домовых чатов (Волга-IT 2026).

Команда запуска:
    python predict.py --input <messages.csv> --out <каталог>

Формирует три эталонных CSV-файла:
1. messages_pred.csv — классификация каждого сообщения (is_service, sentiment, aggression)
2. chat_daily.csv — суточная агрегация активности и индекса настроения по дням
3. chat_summary.csv — общая сводка по чатам с оценкой тренда и дней пикового конфликта

Строго соблюдает:
- Офлайн-инференс: отсутствие любых внешних сетевых обращений
- Доменные инварианты разметки
- Требования к быстродействию (<= 10 сек на 1000 сообщений)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.aggregation.daily import compute_daily_aggregates, export_chat_daily_csv
from src.aggregation.summary import compute_chat_summary, export_chat_summary_csv
from src.models.baseline import BaselineClassifier
from src.models.ensemble import EnsembleClassifier

def get_default_device() -> str:
    """Определение доступного и реально совместимого устройства вычислений."""
    try:
        import torch
        if torch.cuda.is_available():
            probe = torch.zeros(1, device="cuda")
            del probe
            return "cuda"
    except Exception:
        return "cpu"
    return "cpu"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    default_dev = get_default_device()
    parser = argparse.ArgumentParser(
        description="Волга-IT 2026: Пайплайн классификации и агрегации настроения домовых чатов"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Путь к входному CSV-файлу с сообщениями чатов",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Путь к каталогу для сохранения результатов (3 CSV-файла)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Каталог с весами обученных моделей (по умолчанию: models)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_dev,
        choices=["cpu", "cuda"],
        help=f"Устройство для инференса нейросети (по умолчанию: {default_dev})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Размер батча для инференса трансформера (по умолчанию: 64)",
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Применить динамическую INT8-квантизацию линейных слоев для экономии памяти на CPU",
    )
    return parser.parse_args(args)


def load_input_messages(filepath: str | Path) -> pd.DataFrame:
    """Безопасная загрузка входного CSV-файла сообщений."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Входной файл не найден: {path}")

    # Определение разделителя (';' или ',')
    try:
        sample = path.read_text(encoding="utf-8-sig", errors="replace")[:2048]
        sep = ";" if sample.count(";") >= sample.count(",") else ","
        df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")
    except Exception as e:
        raise ValueError(f"Ошибка чтения входного CSV: {e}") from e

    df.columns = [c.strip() for c in df.columns]
    req_cols = ["message_id", "chat_id", "timestamp", "text"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Входной CSV не содержит обязательных колонок: {missing}")

    return df


def load_prediction_model(
    models_dir: str | Path,
    device: str = "cpu",
    quantize_int8: bool = False,
) -> EnsembleClassifier | BaselineClassifier:
    """Загрузка ансамбля или бейслайна из локальной директории весов с опцией INT8-квантизации."""
    m_dir = Path(models_dir)
    if not m_dir.is_dir():
        fallback_m = Path(__file__).resolve().parent / models_dir
        if fallback_m.is_dir():
            m_dir = fallback_m

    baseline_path = m_dir / "baseline.joblib"
    transformer_dir = m_dir / "rubert_tiny2_multitask"

    if not baseline_path.is_file():
        raise FileNotFoundError(f"Веса бейслайна не найдены: {baseline_path}")

    model: EnsembleClassifier | BaselineClassifier
    if transformer_dir.is_dir():
        try:
            model = EnsembleClassifier.from_pretrained(
                baseline_path=baseline_path,
                transformer_dir=transformer_dir,
                device=device,
            )
        except Exception:
            try:
                model = EnsembleClassifier.from_pretrained(
                    baseline_path=baseline_path,
                    transformer_dir=transformer_dir,
                    device="cpu",
                )
            except Exception:
                model = BaselineClassifier.load(baseline_path)

        # Динамическая квантизация INT8 при запросе или работе на слабых CPU
        if quantize_int8 and getattr(model, "transformer", None) is not None:
            try:
                import torch
                t_model = model.transformer.model
                if t_model is not None:
                    model.transformer.model = torch.quantization.quantize_dynamic(
                        t_model, {torch.nn.Linear}, dtype=torch.qint8
                    )
            except Exception as q_err:
                sys.stderr.write(f"Квантизация пропущена: {q_err}\n")
        return model

    # Запасной режим быстрого бейслайна
    return BaselineClassifier.load(baseline_path)


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    models_dir: str | Path = "models",
    device: str = "cpu",
    batch_size: int = 64,
    quantize_int8: bool = False,
) -> dict[str, str | int | float]:
    """Выполнение полного сквозного цикла обработки данных с контролем скорости."""
    t_start = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка входных данных
    t0 = time.time()
    messages_df = load_input_messages(input_path)
    t_load_data = time.time() - t0

    # 2. Инициализация моделей (с опциональной INT8-квантизацией)
    t0 = time.time()
    model = load_prediction_model(
        models_dir=models_dir,
        device=device,
        quantize_int8=quantize_int8,
    )
    t_load_model = time.time() - t0

    # 3. Батч-инференс классификации (is_service, sentiment, aggression)
    t0 = time.time()
    preds_df = model.predict_messages(messages_df)
    t_pred = time.time() - t0

    # 4. Агрегация суточной динамики (chat_daily.csv)
    t0 = time.time()
    daily_df = compute_daily_aggregates(messages_df, preds_df)

    # 5. Сводка по чатам (chat_summary.csv)
    summary_df = compute_chat_summary(daily_df)
    t_agg = time.time() - t0

    # 6. Сохранение артефактов в эталонных форматах
    p_messages = out_path / "messages_pred.csv"
    preds_df[["message_id", "is_service", "sentiment", "aggression"]].to_csv(
        p_messages,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    p_daily = export_chat_daily_csv(daily_df, out_path / "chat_daily.csv")
    p_summary = export_chat_summary_csv(summary_df, out_path / "chat_summary.csv")

    total_time = time.time() - t_start
    n_msgs = len(messages_df)
    throughput = n_msgs / max(total_time, 1e-6)
    sec_per_1000 = (total_time / max(n_msgs, 1)) * 1000

    print("=" * 65)
    print("  УСПЕШНО ВЫПОЛНЕН ПАЙПЛАЙН АНАЛИЗА ДОМОВЫХ ЧАТОВ (Волга-IT 2026) ")
    print("=" * 65)
    print(f"Обработано сообщений:      {n_msgs}")
    print(f"Общее время работы:        {total_time:.2f} сек ({throughput:.1f} сообщ./сек)")
    print(f"Скорость (Критерий 6):     {sec_per_1000:.2f} сек / 1000 сообщений (норма: <= 10.0 сек)")
    print(f"  - Загрузка данных:       {t_load_data:.3f} сек")
    print(f"  - Загрузка моделей:      {t_load_model:.3f} сек")
    print(f"  - Инференс классификации:{t_pred:.3f} сек")
    print(f"  - Расчет агрегатов:      {t_agg:.3f} сек")
    print("-" * 65)
    print(f"[1] Сообщения с разметкой:  {p_messages}")
    print(f"[2] Суточные агрегаты:      {p_daily}")
    print(f"[3] Сводка по чатам:        {p_summary}")
    print("=" * 65)

    return {
        "n_messages": n_msgs,
        "total_time": round(total_time, 3),
        "throughput": round(throughput, 1),
        "sec_per_1000": round(sec_per_1000, 2),
        "messages_path": str(p_messages),
        "daily_path": str(p_daily),
        "summary_path": str(p_summary),
    }


def main() -> None:
    """Точка входа скрипта."""
    args = parse_args()
    try:
        run_pipeline(
            input_path=args.input,
            output_dir=args.out,
            models_dir=args.models_dir,
            device=args.device,
            batch_size=args.batch_size,
            quantize_int8=args.quantize_int8,
        )
        sys.exit(0)
    except Exception as err:
        sys.stderr.write(f"ОШИБКА ВЫПОЛНЕНИЯ ПАЙПЛАЙНА: {err}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
