# MASTER_PLAN — VolgaFinal: Анализ настроения домовых чатов
> Дисциплина: «Искусственный интеллект и анализ данных» | Куратор: АИС Город
> Статус: 🟡 В работе | Обновлено: 2026-11-11

---

## Milestone 0 — Инфраструктура и анализ данных

- [ ] [id: 0.1] [role: analyst] [files: data/, notebooks/01_eda.ipynb] Разведочный анализ данных (EDA): train.csv, dev_messages.csv, dev_labels.csv, unlabeled.csv — распределения меток, длины сообщений, частотный анализ
- [ ] [id: 0.2] [role: coder] [files: src/data/loader.py] Реализовать DataLoader: чтение CSV (разделитель `;`, UTF-8), валидация схемы, нормализация timestamp (МСК→UTC)
- [ ] [id: 0.3] [role: analyst] [depends: 0.1] [files: notebooks/02_labeling_guide.ipynb] Изучить labeling_guide.md, зафиксировать правила разметки и граничные случаи в .context/specs/labeling.md

## Milestone 1 — Классификатор сообщений (is_service / sentiment / aggression)

- [ ] [id: 1.1] [role: coder] [depends: 0.2] [files: src/models/baseline.py] Baseline: TF-IDF + LogReg для трёх задач (is_service, sentiment, aggression) — быстрый старт для метрик
- [ ] [id: 1.2] [role: coder] [depends: 1.1] [files: src/models/transformer.py] Дообучить предобученную RU-модель (rubert-tiny2 или cointegrated/rubert-tiny2) с multi-task head
- [ ] [id: 1.3] [role: coder] [depends: 0.2] [files: src/features/preprocessor.py] Препроцессор текста: нормализация опечаток, жаргон ЖКХ, эмодзи, мат (с обфускацией), маркеры [имя][телефон]...
- [ ] [id: 1.4] [role: qa] [depends: 1.2] [files: src/evaluate_local.py] Локальная оценка на dev: macro-F1 по sentiment, F1 по aggression, F1 по is_service; вызов evaluate.py
- [ ] [id: 1.5] [role: coder] [depends: 1.2, 1.3] [files: src/models/ensemble.py] Ансамблирование: Vote/Averaging baseline + transformer для хардкейсов (сарказм, мат, mixed)

## Milestone 2 — Агрегация по чатам

- [ ] [id: 2.1] [role: coder] [depends: 1.1] [files: src/aggregation/daily.py] Агрегация daily: chat_daily.csv (chat_id, date, n_messages, n_resident, mood_index, negative_share, aggression_share)
- [ ] [id: 2.2] [role: coder] [depends: 2.1] [files: src/aggregation/summary.py] Агрегация summary: chat_summary.csv (chat_id, n_resident, mood_index, negative_share, aggression_share, trend, peak_days)
- [ ] [id: 2.3] [role: coder] [depends: 2.1, 2.2] [files: src/aggregation/metrics.py] Реализация по эталону chatmood_metrics.py: trend (ISO-недели, взвешенная линейная регрессия), peak_days (топ-3 дня)

## Milestone 3 — Точка входа и pipeline

- [ ] [id: 3.1] [role: coder] [depends: 1.5, 2.3] [files: predict.py] Единая команда: `python predict.py --input <messages.csv> --out <dir>` → messages_pred.csv + chat_daily.csv + chat_summary.csv
- [ ] [id: 3.2] [role: qa] [depends: 3.1] [files: dev_out/] Прогон на dev: python predict.py --input data/dev_messages.csv --out dev_out/; python evaluate.py ...
- [ ] [id: 3.3] [role: coder] [depends: 3.1] [files: predict.py] Оптимизация скорости: target ≤10 сек на 1000 сообщений на GTX 1050 Ti (батч-инференс, квантизация int8)

## Milestone 4 — Отчёт и визуализации

- [ ] [id: 4.1] [role: analyst] [depends: 2.3] [files: notebooks/03_report.ipynb, report.html] Генерация HTML-отчёта: динамика настроения по чатам, дни-пики, сравнение чатов, тепловая карта
- [ ] [id: 4.2] [role: coder] [depends: 4.1] [files: src/report/generator.py] Скрипт авто-генерации отчёта (без запуска Jupyter): `python generate_report.py --pred-dir dev_out/ --out report.html`

## Milestone 5 — Документация и финальная сдача

- [ ] [id: 5.1] [role: docs] [depends: 3.2, 4.2] [files: README.md] Финальный README.md: установка, команда обучения, точная команда запуска, время работы на нашем ПК
- [ ] [id: 5.2] [role: docs] [depends: 5.1] [files: README.md] Пояснительная записка (до 3 страниц): предобработка, выбор модели, специфика рус. языка и ЖКХ, что не сработало
- [ ] [id: 5.3] [role: qa] [depends: 3.1, 5.1] [files: dev_out/] Финальная проверка: reproduce с нуля по README.md; evaluate.py даёт корректные числа
