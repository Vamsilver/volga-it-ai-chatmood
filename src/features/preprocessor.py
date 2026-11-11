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
    r"[😡🤬🤦🤮👿💩😠💢👎💔]": " [EMOJI_NEG] ",
    # Смех / сарказм / насмешка
    r"[🤣😂😅😆]": " [EMOJI_LAUGH] ",
    # Информационные / объявления
    r"[📍🔹🏡📢ℹ️🔔⚠️]": " [EMOJI_INFO] ",
}

# Паттерны обфусцированного и открытого мата
OBSCENE_PATTERNS = [
    r"руко[@оo][пp][ые]",
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

        # Экспрессия
        feats["has_excl_high"] = cleaned.str.contains(r"\[EXCL_HIGH\]").astype(int)
        feats["has_quest_high"] = cleaned.str.contains(r"\[QUEST_HIGH\]").astype(int)

        return feats
