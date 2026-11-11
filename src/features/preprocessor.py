"""Модуль специализированной предобработки текстов сообщений домовых чатов.

Реализует специфику русского языка и доменной лексики ЖКХ (Критерий 3 регламента):
1. Обработка маркеров анонимизации: [имя], [телефон], [адрес], [ссылка]...
2. Замена эмодзи на семантические токены ([EMOJI_POS], [EMOJI_NEG], [EMOJI_LAUGH])
3. Расшифровка аббревиатур и сленга ЖКХ (ГВС, ХВС, ОДН, МОП, УК, парадная)
4. Детекция и нормализация обсценной лексики и обфусцированного мата (руко@опые, б***)
5. Нормализация пунктуации, опечаток склейки и признаков экспрессии (CAPS LOCK, !!!, ???)
"""
from __future__ import annotations

import re
from typing import Sequence
import pandas as pd

# Словарь аббревиатур и сленга ЖКХ
JKH_VOCABULARY = {
    r"\bгвс\b": "горячее водоснабжение",
    r"\bхвс\b": "холодное водоснабжение",
    r"\bодн\b": "общедомовые нужды",
    r"\bмоп\b": "места общего пользования",
    r"\bипу\b": "индивидуальный счетчик",
    r"\bодпу\b": "общедомовой счетчик",
    r"\bук\b": "управляющая компания",
    r"\bуправляйк[а-яё]*\b": "управляющая компания",
    r"\bтсж\b": "товарищество собственников",
    r"\bгжи\b": "жилищная инспекция",
    r"\bрсо\b": "ресурсоснабжающая организация",
}

# Семантические группы эмодзи
EMOJI_MAP = {
    # Позитивные
    r"[👍🙏😊❤️👌👏🎉💐🌸🤝😀😁😃]": " [EMOJI_POS] ",
    # Негативные / раздражение
    r"[😡🤬🤦🤮👿💩😠💢👎💔🤡😒😱]": " [EMOJI_NEG] ",
    # Смех / сарказм / насмешка
    r"[🤣😂😅😆]": " [EMOJI_LAUGH] ",
    # Информационные / объявления
    r"[📍🔹🏡📢ℹ️🔔⚠️]": " [EMOJI_INFO] ",
}

# Паттерны обфусцированного и открытого мата
OBSCENE_PATTERNS = [
    r"руко[@оoж\*\.]+[пp][ыеа-яё]*",
    r"\b[пp][иеe][пp]ец\b",
    r"\b[нn]аср[уеёя][тм]\b",
    r"\bср[её]м\b",
    r"\b[хx][уy][еeёя][\w]*\b",
    r"\b[бb][лl][яa][\w]*\b",
    r"\b[пp][иe][зz][дd][\w]*\b",
    r"[хx][\.\*@#]{1,4}",
    r"[бb][\.\*@#]{1,4}",
    r"[пp][\.\*@#]{1,4}[еe]ц",
]

# Паттерны прямых оскорблений, хамства, перехода на личности и угроз
AGGRESSION_PATTERNS = [
    r"\b(?:конченн[а-яё]*|ублюд[а-яё]*|дебил[а-яё]*|придур[а-яё]*|идиот[а-яё]*|тварь[а-яё]*|мраз[а-яё]*|гнид[а-яё]*|чмо[а-яё]*|недоносок[а-яё]*|упыр[а-яё]*|отребь[а-яё]*|быдл[а-яё]*|сволоч[а-яё]*|сук[а-яё]*)\b",
    r"(?:(?:голову|мозги)[^.!?\n]{0,25}включи|включи[^.!?\n]{0,25}(?:голову|мозги)|рот\s+закрой|закрой\s+рот|заткни[а-яё]*|молчал[а-яё]*\s+бы|недержание\s+во\s+рту)",
    r"(?:тебе\s+п[\.\*a-яё]*ц|слышишь\s+меня|морду\s+набью|руки\s+оторв[а-яё]*|посад[а-яё]*\s+этого)",
]

# Паттерны сарказма и ироничных псевдо-благодарностей
SARCASM_PATTERNS = [
    r"спасибо\s+за[^.!?\n]{1,50}(?:нет|не\s+работает|сломал|ледян|грязн|холодн|отключ)",
    r"зато\s+[^.!?\n]{1,50}[😡🤬🤦🤮👿💩😠💢👎💔]",
    r"наш\s+любимый\s+[^.!?\n]{1,30}\)+",
    r"наша\s+ук\s+нас\s+береж[а-яё]+",
    r"сэкономили\)+",
    r"ну\s+естественно\s*[\U0001f609\)]+",
]


class TextPreprocessor:
    """Текстовый процессор для нормализации специфики домовых чатов."""

    def __init__(
        self,
        normalize_jkh: bool = True,
        map_emojis: bool = True,
        normalize_obscene: bool = True,
        separate_markers: bool = True,
    ) -> None:
        self.normalize_jkh = normalize_jkh
        self.map_emojis = map_emojis
        self.normalize_obscene = normalize_obscene
        self.separate_markers = separate_markers

        # Предкомпиляция регулярных выражений
        self.jkh_compiled = [(re.compile(p, re.IGNORECASE), repl) for p, repl in JKH_VOCABULARY.items()]
        self.emoji_compiled = [(re.compile(p), repl) for p, repl in EMOJI_MAP.items()]
        self.obscene_compiled = [re.compile(p, re.IGNORECASE) for p in OBSCENE_PATTERNS]
        self.aggression_compiled = [re.compile(p, re.IGNORECASE) for p in AGGRESSION_PATTERNS]
        self.sarcasm_compiled = [re.compile(p, re.IGNORECASE) for p in SARCASM_PATTERNS]
        self.marker_regex = re.compile(r"(\[(?:имя|телефон|адрес|квартира|грз|email|ссылка|ник|номер)\])", re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """Очистка и нормализация одного сообщения."""
        if not text or pd.isna(text):
            return ""

        t = str(text)

        # 1. Отделение прилипшей пунктуации после слов (опечатки склейки: "воды.Завтра" -> "воды. Завтра")
        t = re.sub(r"([а-яёА-ЯЁa-zA-Z])([.,!?])([а-яёА-ЯЁa-zA-Z])", r"\1\2 \3", t)

        # 2. Нормализация маркеров анонимизации (выделение пробелами)
        if self.separate_markers:
            t = self.marker_regex.sub(r" \1 ", t)

        # 3. Замена эмодзи на семантические токены
        if self.map_emojis:
            for pattern, repl in self.emoji_compiled:
                t = pattern.sub(repl, t)

        # 4. Нормализация обсценной лексики
        if self.normalize_obscene:
            for pattern in self.obscene_compiled:
                t = pattern.sub(" [OBSCENE] ", t)

        # 5. Экспрессивная пунктуация
        t = re.sub(r"!{2,}", " !!! [EXCL_HIGH] ", t)
        t = re.sub(r"\?{2,}", " ??? [QUEST_HIGH] ", t)

        # 6. Расшифровка аббревиатур ЖКХ
        if self.normalize_jkh:
            for pattern, repl in self.jkh_compiled:
                t = pattern.sub(f" {repl} ", t)

        # 7. Схлопывание лишних пробелов
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def transform_series(self, texts: pd.Series | Sequence[str]) -> pd.Series:
        """Трансформация последовательности сообщений."""
        s = pd.Series(texts) if not isinstance(texts, pd.Series) else texts
        return s.apply(self.clean_text)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение интерпретируемых признаков для классификации и ансамблирования.

        Returns DataFrame со статистическими признаками текста:
        - len_chars, len_words
        - caps_ratio
        - has_address, has_phone, has_link
        - has_obscene, has_emoji_pos, has_emoji_neg, has_emoji_laugh
        """
        raw_texts = df["text"].fillna("").astype(str)
        cleaned = self.transform_series(raw_texts)

        feats = pd.DataFrame(index=df.index)
        feats["len_chars"] = raw_texts.str.len()
        feats["len_words"] = raw_texts.str.split().apply(len)

        # Доля заглавных букв (CAPS LOCK)
        feats["caps_ratio"] = raw_texts.apply(
            lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)
        ).round(3)

        # Маркеры анонимизации (критично для is_service)
        feats["has_address"] = raw_texts.str.contains(r"\[адрес\]", case=False).astype(int)
        feats["has_phone"] = raw_texts.str.contains(r"\[телефон\]", case=False).astype(int)
        feats["has_link"] = raw_texts.str.contains(r"\[ссылка\]", case=False).astype(int)

        # Эмоциональные токены
        feats["has_obscene"] = cleaned.str.contains(r"\[OBSCENE\]").astype(int)
        feats["has_emoji_pos"] = cleaned.str.contains(r"\[EMOJI_POS\]").astype(int)
        feats["has_emoji_neg"] = cleaned.str.contains(r"\[EMOJI_NEG\]").astype(int)
        feats["has_emoji_laugh"] = cleaned.str.contains(r"\[EMOJI_LAUGH\]").astype(int)

        # Сигналы агрессии (оскорбления, переход на личности, угрозы или мат)
        has_aggr = feats["has_obscene"].copy()
        for pattern in self.aggression_compiled:
            has_aggr = has_aggr | raw_texts.apply(lambda s: 1 if pattern.search(s) else 0)
        feats["has_aggression_signal"] = has_aggr.astype(int)

        # Сигналы сарказма (псевдо-благодарности, ироничные паттерны)
        has_sarcasm = pd.Series(0, index=df.index, dtype=int)
        for pattern in self.sarcasm_compiled:
            has_sarcasm = has_sarcasm | raw_texts.apply(lambda s: 1 if pattern.search(s) else 0)
        feats["has_sarcasm_signal"] = has_sarcasm.astype(int)

        # Экспрессия
        feats["has_excl_high"] = cleaned.str.contains(r"\[EXCL_HIGH\]").astype(int)
        feats["has_quest_high"] = cleaned.str.contains(r"\[QUEST_HIGH\]").astype(int)

        return feats


REACTION_PATTERNS = re.compile(
    r"(?:^(?:и\s+у\s+меня|и\s+на\s+|и\s+в\s+|мы\s+уже|ваши\s+предложения|в\s+отпуске\s+наверное|думаю[,\s]+нет|я\s+подала\s+тоже|горячая-холодная|[🤐\U0001f910]+)|"
    r"\b(?:тоже|также|так\s+же|аналогично|однозначно|согласен|согласна|согласны|точно|плюсую|\+1|поддерживаю|жмуркино|куда\s+обращаться)\b|"
    r"^да[.,!\s]+(?:первая|вторая|третья|в\s+явных|именно|участковый))",
    re.IGNORECASE,
)
REACTION_POS_EXCLUSIONS = re.compile(
    r"спасибо|благодар|класс|отличн|здравствуй|добр[ый|ое]|поздравл|супер|[👍🙏👌👏🎉🌸❤️]",
    re.IGNORECASE,
)


def enrich_chat_context(
    df: pd.DataFrame,
    max_time_diff_sec: float = 300.0,
    max_words: int = 5,
) -> pd.Series:
    """Обогащение коротких реплик-реакций контекстом предыдущего сообщения из того же чата.

    Если сообщение является короткой репликой-согласием/реакцией (до max_words слов),
    пришло в течение max_time_diff_sec секунд после предыдущего сообщения того же chat_id,
    и не содержит собственных позитивных маркеров — к нему подклеивается контекст предыдущего сообщения.

    Args:
        df: DataFrame с колонками text, и опционально chat_id, timestamp.
        max_time_diff_sec: Максимальная разница во времени между репликами (по умолчанию 300 сек).
        max_words: Максимальная длина реплики-реакции для обогащения (по умолчанию 5 слов).

    Returns:
        pd.Series обогащенных текстов в исходном порядке строк df.
    """
    if "text" not in df.columns or "chat_id" not in df.columns or "timestamp" not in df.columns:
        return df["text"] if "text" in df.columns else pd.Series(dtype=object)

    work_df = df[["chat_id", "timestamp", "text"]].copy()
    work_df["_orig_idx"] = work_df.index
    work_df = work_df.sort_values(["chat_id", "timestamp"]).reset_index(drop=True)
    work_df["prev_text"] = work_df.groupby("chat_id")["text"].shift(1)

    timestamps = pd.to_datetime(work_df["timestamp"], errors="coerce")
    prev_timestamps = work_df.groupby("chat_id")["timestamp"].shift(1)
    prev_timestamps = pd.to_datetime(prev_timestamps, errors="coerce")
    work_df["time_diff_sec"] = (timestamps - prev_timestamps).dt.total_seconds()

    enriched = []
    for idx, row in work_df.iterrows():
        curr = str(row["text"]).strip() if pd.notna(row["text"]) else ""
        prev = str(row["prev_text"]).strip() if pd.notna(row["prev_text"]) else ""
        dt = row["time_diff_sec"]

        if (
            prev
            and pd.notna(dt)
            and dt < max_time_diff_sec
            and len(curr.split()) <= max_words
            and REACTION_PATTERNS.search(curr)
            and not REACTION_POS_EXCLUSIONS.search(curr)
        ):
            enriched.append(f"{curr} [Контекст: {prev}]")
        else:
            enriched.append(curr)

    work_df["enriched_text"] = enriched
    work_df = work_df.sort_values("_orig_idx").reset_index(drop=True)
    return work_df["enriched_text"]

