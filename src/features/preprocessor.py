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
import numpy as np
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
    r"\bуправляшк[а-яё]*\b": "управляющая компания",
    r"\bтсж\b": "товарищество собственников",
    r"\bгжи\b": "жилищная инспекция",
    r"\bрсо\b": "ресурсоснабжающая организация",
    r"\bосс\b": "общее собрание собственников",
    r"\bадс\b": "аварийно диспетчерская служба",
    r"\bквадо\b": "система заявок",
    r"\bпарадн[а-яё]+\b": "подъезд",
    r"\bразвоздуш[а-яё]+\b": "спустить воздух",
    r"\bнарк[а-яё]*\b": "наркоманы",
    r"\bеирц\b": "расчетный центр",
    r"\bтсн\b": "товарищество собственников недвижимости",
    r"\bжск\b": "жилищный кооператив",
    r"\bсои\b": "содержание общего имущества",
    r"\bпдд\b": "правила движения",
    r"\bуо\b": "управляющая организация",
    r"\bмчс\b": "служба спасения",
    r"\bстояк[а-яё]*\b": "трубопровод",
    r"\bфанов[а-яё]+\s+труб[а-яё]*\b": "канализационная труба",
    r"\bфанов[а-яё]+\b": "канализация",
    r"\bтеплоносител[а-яё]*\b": "горячая вода отопления",
    r"\bгреб[её]нк[а-яё]*\b": "распределительный коллектор",
    r"\bприбор[а-яё]*\s+учет[а-яё]*\b": "счетчик",
    r"\bпромывк[а-яё]*\b": "очистка системы",
    r"\bстяжк[а-яё]*\b": "выравнивание пола",
    r"\bкровл[а-яё]*\b": "крыша",
    r"\bпухт[оа][а-яё]*\b": "мусорный контейнер",
    r"\bпоребрик[а-яё]*\b": "бордюр",
    r"\bукашк[а-яё]*\b": "управляющая компания",
    r"\bкомм?\.?\s*услуг[а-яё]*\b": "коммунальные услуги",
    r"\bнарик[а-яё]*\b": "наркоманы",
    r"\bлифт[её]р[а-яё]*\b": "мастер по лифтам",
    r"\bкозыр[её]к[а-яё]*\b": "навес подъезда",
    r"\bпандус[а-яё]*\b": "съезд для колясок",
    r"\bдоводчик[а-яё]*\b": "механизм двери",
    r"\bщитов[а-яё]+\b": "электрощитовая",
    r"\bзавоздуш[а-яё]+\b": "воздух в трубах",
}

# Семантические группы эмодзи
EMOJI_MAP = {
    # Позитивные
    r"[👍🙏😊❤️👌👏🎉💐🌸🤝😀😁😃🫶🥰😍🥳✨]": " [EMOJI_POS] ",
    # Негативные / раздражение / шок / отвращение
    r"[😡🤬🤦🤮👿💩😠💢👎💔🤡😒😱😳🤐😩😫😭🤢]": " [EMOJI_NEG] ",
    # Смех / сарказм / насмешка
    r"[🤣😂😅😆🙃😏]": " [EMOJI_LAUGH] ",
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


# Словарь типичных опечаток в проблемных и эмоциональных словах
TYPO_NORMALIZATION_MAP = {
    r"\bжмть\b": "жить",
    r"\bпострадалв\b": "пострадала",
    r"\bотключино\b": "отключено",
    r"\bотчетатся\b": "отчитаться",
    r"\bпроходчт\b": "проходят",
    r"\bичпугаются\b": "испугаются",
    r"\bсломалсо\b": "сломался",
    r"\bлдифт\b": "лифт",
    r"\bотопленее\b": "отопление",
    r"\bхолондая\b": "холодная",
    r"\bнапора\s+ли\b": "не пора ли",
    r"\bдобролись\b": "добрались",
    r"\bнавнодушн[а-яё]*\b": "равнодушного",
    r"\bпоаравтлиа\b": "поправила",
    r"\bполучпет\b": "получает",
    r"\bпочтовыц\b": "почтовый",
    r"\bсабака\b": "собака",
    r"\bмноготсобак\b": "много собак",
    r"\bуде\s+не\s+надо\b": "уже не надо",
    r"\bсбростлась\b": "сбросилась",
}


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
        self.typo_compiled = [(re.compile(p, re.IGNORECASE), repl) for p, repl in TYPO_NORMALIZATION_MAP.items()]
        self.marker_regex = re.compile(r"(\[(?:имя|телефон|адрес|квартира|грз|email|ссылка|ник|номер)\])", re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """Очистка и нормализация одного сообщения."""
        if not text or pd.isna(text):
            return ""

        t = str(text)

        # 0. Нормализация типичных опечаток в корнях проблемных слов
        for pattern, repl in self.typo_compiled:
            t = pattern.sub(repl, t)

        # 1. Отделение прилипшей пунктуации после слов и цифр (опечатки склейки: "воды.Завтра" -> "воды. Завтра", "2парадная ,аналогично")
        t = re.sub(r"\s+([.,!?])", r"\1", t)
        t = re.sub(r"([.,!?])([а-яёА-ЯЁa-zA-Z0-9])", r"\1 \2", t)

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
        feats["is_context_reaction"] = raw_texts.apply(is_context_reaction).astype(int)

        return feats


REACTION_PATTERNS = re.compile(
    r"(?:^(?:и\s+у\s+меня|и\s+у\s+нас|и\s+на\s+|и\s+в\s+|мы\s+тоже|я\s+тоже|у\s+нас\s+тоже|у\s+меня\s+тоже|мы\s+уже|я\s+с\s+вами|мы\s+с\s+вами|мы\s+уже\s+предложили|[🤐\U0001f910]+)|"
    r"^(?:тоже|также|так\s+же|аналогично|однозначно|согласен|согласна|согласны|соглашусь|точно|факт|плюсую|поддерживаю|полностью\s+соглас[а-яё]*)\b|"
    r"\b(?:аналогично|однозначно|плюсую|поддерживаю\s+сообщение|я\s+с\s+вами|мы\s+с\s+вами)\b|"
    r"(?:\+1|\+\s*1)(?!\d)|"
    r"^да[а!]*[.,!\s]+(?:именно|точно|согласен|конечно|так|правда)|"
    r"^да{2,}!*)",
    re.IGNORECASE,
)
REACTION_POS_EXCLUSIONS = re.compile(
    r"спасибо|благодар|класс|отличн|здравствуй|добр[ый|ое]|поздравл|супер|[👍🙏👌👏🎉🌸❤️]",
    re.IGNORECASE,
)


def is_context_reaction(text: str, max_words: int = 12) -> bool:
    """Проверка, является ли текст короткой репликой-реакцией/согласием."""
    if not text or pd.isna(text):
        return False
    t = str(text).strip()
    if "?" in t:
        return False
    t = re.sub(r"\s+([.,!?])", r"\1", t)
    t = re.sub(r"([.,!?])([а-яёА-ЯЁa-zA-Z0-9])", r"\1 \2", t)
    words = t.split()
    if len(words) > max_words:
        return False
    if REACTION_POS_EXCLUSIONS.search(t):
        return False
    if re.search(r"[😁😃😄😀🫶🥰😍🥳✨👍🙏👌👏🎉🌸❤️]", t):
        return False
    return bool(REACTION_PATTERNS.search(t))


def enrich_chat_context(
    df: pd.DataFrame,
    max_time_diff_sec: float = 300.0,
    max_words: int = 12,
    return_reaction_flags: bool = False,
    lookback_window: int = 3,
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """Обогащение коротких реплик-реакций контекстом предыдущего сообщения из того же чата.

    Если сообщение является короткой репликой-согласием/реакцией (до max_words слов),
    пришло в течение max_time_diff_sec секунд после предыдущего сообщения того же chat_id,
    и не содержит собственных позитивных маркеров — к нему подклеивается контекст предыдущего сообщения.

    Args:
        df: DataFrame с колонками text, и опционально chat_id, timestamp.
        max_time_diff_sec: Максимальная разница во времени между репликами (по умолчанию 300 сек).
        max_words: Максимальная длина реплики-реакции для обогащения (по умолчанию 12 слов).
        return_reaction_flags: Если True, возвращает кортеж (enriched_text, is_reaction).
        lookback_window: Глубина окна поиска предыдущих сообщений (по умолчанию 3).

    Returns:
        pd.Series обогащенных текстов в исходном порядке строк df (или кортеж Series).
    """
    if "text" not in df.columns or "chat_id" not in df.columns or "timestamp" not in df.columns:
        res = df["text"] if "text" in df.columns else pd.Series(dtype=object)
        if return_reaction_flags:
            return res, pd.Series(False, index=res.index)
        return res

    work_df = df[["chat_id", "timestamp", "text"]].copy()
    work_df["_orig_idx"] = np.arange(len(work_df))
    work_df["ts"] = pd.to_datetime(work_df["timestamp"], errors="coerce")
    work_df = work_df.sort_values(["chat_id", "ts"]).reset_index(drop=True)

    enriched = []
    is_reaction = []
    for i in range(len(work_df)):
        curr = str(work_df.loc[i, "text"]).strip() if pd.notna(work_df.loc[i, "text"]) else ""
        curr_ts = work_df.loc[i, "ts"]
        curr_chat = work_df.loc[i, "chat_id"]

        is_react = is_context_reaction(curr, max_words=max_words)
        found_prev = None

        if is_react and pd.notna(curr_ts):
            for step in range(1, lookback_window + 1):
                prev_i = i - step
                if prev_i < 0 or work_df.loc[prev_i, "chat_id"] != curr_chat:
                    break
                prev_ts = work_df.loc[prev_i, "ts"]
                if pd.isna(prev_ts):
                    break
                dt = (curr_ts - prev_ts).total_seconds()
                if dt < 0 or dt > max_time_diff_sec:
                    break
                prev_cand = str(work_df.loc[prev_i, "text"]).strip() if pd.notna(work_df.loc[prev_i, "text"]) else ""
                if prev_cand:
                    found_prev = prev_cand
                    break

        if found_prev:
            is_reaction.append(True)
            enriched.append(f"{curr} [Контекст: {found_prev}]")
        else:
            is_reaction.append(is_react)
            enriched.append(curr)

    work_df["enriched_text"] = enriched
    work_df["is_reaction"] = is_reaction
    work_df = work_df.sort_values("_orig_idx").reset_index(drop=True)
    if return_reaction_flags:
        return work_df["enriched_text"], work_df["is_reaction"]
    return work_df["enriched_text"]

