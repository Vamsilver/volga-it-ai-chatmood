# Анализ настроения домовых чатов — VolgaIT 2026 Final

Система анализа настроения жителей в чатах МКД по выгрузке CSV-сообщений.

## Установка

```bash
pip install -r requirements.txt
```

## Данные

Поместить в директорию `data/`:
- `train.csv`, `dev_messages.csv`, `dev_labels.csv`, `unlabeled.csv`
- `labeling_guide.md`, `chatmood_metrics.py`, `evaluate.py`

## Обучение модели

```bash
python train.py --seed 42
```

Обученные веса сохраняются в `models/`.

## Запуск предсказания

```bash
python predict.py --input <messages.csv> --out <output_dir>
```

Пример на dev:
```bash
python predict.py --input data/dev_messages.csv --out dev_out/
```

## Оценка качества на dev

```bash
python evaluate.py --gold-messages data/dev_messages.csv --gold-labels data/dev_labels.csv --pred-dir dev_out/
```

## Генерация отчёта

```bash
python generate_report.py --pred-dir dev_out/ --out dev_out/report.html
```

## Выходные файлы

| Файл | Описание |
|------|----------|
| `messages_pred.csv` | `message_id;is_service;sentiment;aggression` |
| `chat_daily.csv` | Агрегаты по чату и дате |
| `chat_summary.csv` | Итоговый профиль каждого чата |
| `report.html` | Визуальный отчёт |

## Время работы

*Будет заполнено после замера на финальном железе*

## Используемые источники

- Предобученная модель: `cointegrated/rubert-tiny2`
- Эталонный расчёт агрегатов: `chatmood_metrics.py`
