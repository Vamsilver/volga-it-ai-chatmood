"""CLI точка входа для автономной генерации HTML-отчёта по домовым чатам.

Использование:
    python generate_report.py --pred-dir dev_out/ --out report.html
    python generate_report.py --pred-dir dev_out/ --messages data/dev_messages.csv --out report.html
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.report.generator import generate_html_report


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация автономного аналитического HTML-отчёта по домовым чатам",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="dev_out",
        help="Каталог с CSV-файлами прогнозов (chat_summary.csv, chat_daily.csv, messages_pred.csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="report.html",
        help="Путь к результирующему HTML-файлу",
    )
    parser.add_argument(
        "--messages",
        type=str,
        default="data/dev_messages.csv",
        help="Опциональный путь к исходному файлу сообщений для детализации цитат инцидентов",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    t_start = time.time()
    msgs_path = args.messages if Path(args.messages).is_file() else None

    try:
        out_path = generate_html_report(
            pred_dir=args.pred_dir,
            output_path=args.out,
            messages_path=msgs_path,
        )
        elapsed = time.time() - t_start
        print("=" * 65)
        print("  УСПЕШНО СГЕНЕРИРОВАН АНАЛИТИЧЕСКИЙ ОТЧЁТ (Волга-IT 2026)")
        print("=" * 65)
        print(f"Каталог данных:      {args.pred_dir}")
        print(f"Итоговый файл отчёта: {out_path.resolve()}")
        print(f"Время генерации:     {elapsed:.2f} сек")
        print("=" * 65)
    except Exception as err:
        sys.stderr.write(f"ОШИБКА ГЕНЕРАЦИИ ОТЧЁТА: {err}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
