"""Adversarial Verification Suite by Challenger (Round 2).

Empirically challenges and tests:
1. Sarcasm with laughing emojis (😅🤣) and preservation of pure positive sentiment.
2. Context lookback (chains of 2-3 messages), short agreement replies, long replies (boundary: 12 vs 13 words), rhetorical / interrogative questions ('?').
3. Exhaustive verification of all 45 terms in JKH_VOCABULARY.
4. Aggression detection on insult variations, veiled hostility, future-tense profanity, and ZERO false positives on realistic complaints.
5. Mixed tone ('но...') handling and boundary with neutral explanations.
6. Metric thresholds verification against dev set (Q1 >= 0.7947, Q2 >= 0.7972, Q3 >= 0.6983, Total auto >= 36.57).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.features.preprocessor import (
    JKH_VOCABULARY,
    TextPreprocessor,
    enrich_chat_context,
    is_context_reaction,
)
from src.models.ensemble import (
    INSULT_OVERRIDE_PATTERNS,
    MIXED_NEGATIVE_PATTERNS,
    SARCASM_STRONG_PATTERNS,
    EnsembleClassifier,
)


class TestAdversarialChallengerR2(unittest.TestCase):
    """Empirical adversarial stress testing for VolgaFinal R2."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.preprocessor = TextPreprocessor()
        cls.baseline_path = Path("models/baseline.joblib")
        cls.transformer_dir = Path("models/rubert_tiny2_multitask")
        if cls.baseline_path.is_file():
            cls.ensemble = EnsembleClassifier.from_pretrained(
                baseline_path=cls.baseline_path,
                transformer_dir=cls.transformer_dir if cls.transformer_dir.is_dir() else None,
                device="cpu",
                aggression_threshold=0.78,
            )
        else:
            cls.ensemble = None

    # =========================================================================
    # 1. SARCASM & LAUGHING EMOJIS (😅🤣) - POSITIVE SENTIMENT PRESERVATION
    # =========================================================================

    def test_01_pure_positive_with_laughing_emojis_not_flipped_to_negative(self) -> None:
        """Adversarial: Pure positive sentiment with 😅/🤣 MUST NEVER be flipped to negative."""
        pure_positive_with_laughs = [
            "Спасибо огромное сантехникам, повеселили с утра😅 Все починили отлично!",
            "Ой какая красота у нас во дворе, просто супер 🤣🤣",
            "Спасибо за праздник для детей! Было очень весело 😅🎉",
            "Хаха, классная новость, наконец-то дали горячую воду! 😅",
            "Посмеялись от души, спасибо соседям за отличный вечер! 🤣",
            "Спасибо дворнику за чистые дорожки, прямо приятно идти 😅👍",
            "Какая прелесть, спасибо нашему совету дома за цветы у подъезда! 😅💐",
            "Прекрасно починили лифт, теперь летает быстро 😅",
            "Отличная работа, спасибо за оперативность 😅",
        ]
        if self.ensemble is None:
            self.skipTest("Ensemble model not found.")

        df = pd.DataFrame({
            "message_id": [f"pos_laugh_{i}" for i in range(len(pure_positive_with_laughs))],
            "text": pure_positive_with_laughs,
        })
        preds = self.ensemble.predict_messages(df)

        for i, row in preds.iterrows():
            orig_text = pure_positive_with_laughs[i]
            # Sentiment MUST NOT be negative!
            self.assertNotEqual(
                row["sentiment"],
                "negative",
                f"Adversarial failure: Pure positive message '{orig_text}' was wrongly classified as 'negative'!",
            )
            self.assertEqual(
                row["sentiment"],
                "positive",
                f"Pure positive message '{orig_text}' should be recognized as 'positive', got '{row['sentiment']}'.",
            )
            # Aggression must be 0
            self.assertEqual(
                row["aggression"],
                0,
                f"Adversarial failure: Pure positive message '{orig_text}' flagged as aggressive!",
            )

    def test_02_sarcasm_with_laughing_emojis_correctly_classified_as_negative(self) -> None:
        """Adversarial: Sarcastic complaints with laughing emojis MUST be classified as negative."""
        sarcastic_with_laughs = [
            "Спасибо за ледяную воду из горячего крана 🤣🤣🤣",
            "Наша УК нас бережет 😅🤣",
            "Красота, примите меры наконец к этой грязи 😅",
            "Опять лифт отключен, куда тогда уходят денежки 🤣🤣",
            "В явных лидерах по грязи в подъезде 😅🤣",
            "Не царское это дело, снег убирать во дворе 🤣🤣",
            "Вы решили поюморить над жильцами? 😅",
            "В отпуске наверное наш сантехник 😅🤣",
            "Зато сэкономили)))) 🤣🤣",
        ]
        if self.ensemble is None:
            self.skipTest("Ensemble model not found.")

        df = pd.DataFrame({
            "message_id": [f"sarc_laugh_{i}" for i in range(len(sarcastic_with_laughs))],
            "text": sarcastic_with_laughs,
        })
        preds = self.ensemble.predict_messages(df)

        for i, row in preds.iterrows():
            orig_text = sarcastic_with_laughs[i]
            self.assertEqual(
                row["sentiment"],
                "negative",
                f"Adversarial failure: Sarcastic message '{orig_text}' was NOT detected as 'negative'!",
            )

    # =========================================================================
    # 2. CONTEXT LOOKBACK (CHAINS OF 2-3 MSGS, WORD BOUNDARIES, QUESTIONS)
    # =========================================================================

    def test_03_context_chain_lookback_propagation(self) -> None:
        """Adversarial: Context lookback across sequential chains of 2-3 short agreement messages."""
        df_dialog = pd.DataFrame({
            "message_id": ["m_root", "m_agree1", "m_agree2", "m_agree3"],
            "chat_id": ["chat_adv", "chat_adv", "chat_adv", "chat_adv"],
            "timestamp": [
                "2026-07-01 10:00:00",
                "2026-07-01 10:00:30",  # dt = 30s
                "2026-07-01 10:01:10",  # dt = 40s
                "2026-07-01 10:02:00",  # dt = 50s
            ],
            "text": [
                "В подъезде ужасно пахнет канализацией!",
                "Полностью согласен, дышать нечем",
                "+1",
                "Аналогично",
            ],
        })
        if self.ensemble is None:
            self.skipTest("Ensemble model not found.")

        preds = self.ensemble.predict_messages(df_dialog)
        for i, row in preds.iterrows():
            self.assertEqual(
                row["sentiment"],
                "negative",
                f"Adversarial failure: Chain message index {i} '{df_dialog.loc[i, 'text']}' was not resolved as negative!",
            )

    def test_04_context_reaction_word_count_boundary(self) -> None:
        """Adversarial: is_context_reaction must accept <= 12 words and reject > 12 words."""
        # 12 words: eligible
        text_12_words = "Я тоже полностью согласен с этим безобразием и поддерживаю вызов аварийной службы"
        words_12 = text_12_words.split()
        self.assertEqual(len(words_12), 12)
        self.assertTrue(
            is_context_reaction(text_12_words, max_words=12),
            "Failed on exact 12-word reaction clause!",
        )

        # 13 words: must be rejected
        text_13_words = "Я тоже полностью согласен с этим безобразием и поддерживаю вызов аварийной службы срочно"
        words_13 = text_13_words.split()
        self.assertEqual(len(words_13), 13)
        self.assertFalse(
            is_context_reaction(text_13_words, max_words=12),
            "Failed: 13-word clause should NOT be marked as context reaction!",
        )

    def test_05_context_reaction_rejects_questions(self) -> None:
        """Adversarial: Replies with '?' must NOT be treated as short agreement reactions."""
        question_replies = [
            "Тоже не работает?",
            "У вас тоже нет воды?",
            "Согласны?",
            "Да, точно?",
            "И у меня тоже?",
            "Аналогично?",
        ]
        for q in question_replies:
            self.assertFalse(
                is_context_reaction(q),
                f"Adversarial failure: Question '{q}' was falsely identified as context reaction!",
            )

    def test_06_standalone_emoji_context_stripping(self) -> None:
        """Adversarial: Standalone emojis (e.g. 🤐🤐) preserve negative sentiment even after context tag."""
        df_emoji_reaction = pd.DataFrame({
            "message_id": ["m_root_bad", "m_zipped"],
            "chat_id": ["chat_emoji", "chat_emoji"],
            "timestamp": ["2026-07-01 10:00:00", "2026-07-01 10:01:00"],
            "text": [
                "Опять из крана течет ржавая жижа вместо воды",
                "🤐🤐",
            ],
        })
        if self.ensemble is None:
            self.skipTest("Ensemble model not found.")

        preds = self.ensemble.predict_messages(df_emoji_reaction)
        self.assertEqual(preds.iloc[1]["sentiment"], "negative")

    # =========================================================================
    # 3. EXHAUSTIVE HOUSING SLANG NORMALIZATION (ALL 45 TERMS IN JKH_VOCABULARY)
    # =========================================================================

    def test_07_all_45_jkh_vocabulary_terms_normalization(self) -> None:
        """Adversarial: Verify that JKH_VOCABULARY contains >= 45 terms and EVERY single term is normalized."""
        self.assertGreaterEqual(
            len(JKH_VOCABULARY),
            45,
            f"Expected JKH_VOCABULARY to contain at least 45 terms, found {len(JKH_VOCABULARY)}",
        )

        test_sentences = {
            r"\bгвс\b": ("отключили ГВС в доме", "горячее водоснабжение"),
            r"\bхвс\b": ("пропал напор ХВС", "холодное водоснабжение"),
            r"\bодн\b": ("высокие счета за ОДН", "общедомовые нужды"),
            r"\bмоп\b": ("не моют МОП на этаже", "места общего пользования"),
            r"\bипу\b": ("нужно поверить ИПУ", "индивидуальный счетчик"),
            r"\bодпу\b": ("сломался ОДПУ тепла", "общедомовой счетчик"),
            r"\bук\b": ("заявление передано в УК", "управляющая компания"),
            r"\bуправляйк[а-яё]*\b": ("наша управляйка бездействует", "управляющая компания"),
            r"\bуправляшк[а-яё]*\b": ("звонили в управляшку", "управляющая компания"),
            r"\bтсж\b": ("председатель ТСЖ ответил", "товарищество собственников"),
            r"\bгжи\b": ("направим жалобу в ГЖИ", "жилищная инспекция"),
            r"\bрсо\b": ("авария на сетях РСО", "ресурсоснабжающая организация"),
            r"\bосс\b": ("проведем очное ОСС", "общее собрание собственников"),
            r"\bадс\b": ("номер дежурного АДС", "аварийно диспетчерская служба"),
            r"\bквадо\b": ("создана заявка в квадо", "система заявок"),
            r"\bпарадн[а-яё]+\b": ("в нашей парадной грязно", "подъезд"),
            r"\bразвоздуш[а-яё]+\b": ("надо развоздушить батарею", "спустить воздух"),
            r"\bнарк[а-яё]*\b": ("в подвале сидят нарки", "наркоманы"),
            r"\bеирц\b": ("ошибка в квитанции ЕИРЦ", "расчетный центр"),
            r"\bтсн\b": ("зарегистрировано новое ТСН", "товарищество собственников недвижимости"),
            r"\bжск\b": ("бухгалтерия ЖСК закрыта", "жилищный кооператив"),
            r"\bсои\b": ("норматив на СОИ вырос", "содержание общего имущества"),
            r"\bпдд\b": ("машины нарушают ПДД во дворе", "правила движения"),
            r"\bуо\b": ("обязанности УО по договору", "управляющая организация"),
            r"\bмчс\b": ("вызвали наряд МЧС", "служба спасения"),
            r"\bстояк[а-яё]*\b": ("капает общедомовой стояк", "трубопровод"),
            r"\bфанов[а-яё]+\s+труб[а-яё]*\b": ("засорилась фановая труба", "канализационная труба"),
            r"\bфанов[а-яё]+\b": ("пробита фановая в подвале", "канализация"),
            r"\bтеплоносител[а-яё]*\b": ("слишком холодный теплоноситель", "горячая вода отопления"),
            r"\bгреб[её]нк[а-яё]*\b": ("протекла гребенка отопления", "распределительный коллектор"),
            r"\bприбор[а-яё]*\s+учет[а-яё]*\b": ("опломбирован прибор учета воды", "счетчик"),
            r"\bпромывк[а-яё]*\b": ("назначена промывка теплотрассы", "очистка системы"),
            r"\bстяжк[а-яё]*\b": ("шумит стяжка у соседей", "выравнивание пола"),
            r"\bкровл[а-яё]*\b": ("протекает кровля над квартирой", "крыша"),
            r"\bпухт[оа][а-яё]*\b": ("переполнена мусорная пухта", "мусорный контейнер"),
            r"\bпоребрик[а-яё]*\b": ("трактор сбил поребрик", "бордюр"),
            r"\bукашк[а-яё]*\b": ("опять укашка прислала отписку", "управляющая компания"),
            r"\bкомм?\.?\s*услуг[а-яё]*\b": ("оплата за комм. услуги завышена", "коммунальные услуги"),
            r"\bнарик[а-яё]*\b": ("на лестнице опять нарики", "наркоманы"),
            r"\bлифт[её]р[а-яё]*\b": ("приехал дежурный лифтер", "мастер по лифтам"),
            r"\bкозыр[её]к[а-яё]*\b": ("над входом оторвался козырек", "навес подъезда"),
            r"\bпандус[а-яё]*\b": ("сломан пандус для колясок", "съезд для колясок"),
            r"\bдоводчик[а-яё]*\b": ("не работает доводчик на двери", "механизм двери"),
            r"\bщитов[а-яё]+\b": ("заперта этажная щитовая", "электрощитовая"),
            r"\bзавоздуш[а-яё]+\b": ("вновь завоздушена система", "воздух в трубах"),
        }

        # Check coverage of test sentences
        for pattern_key in JKH_VOCABULARY:
            self.assertIn(
                pattern_key,
                test_sentences,
                f"Missing test sentence for vocabulary pattern: {pattern_key}",
            )

        # Run normalization on each sentence and assert replacement
        for pattern_key, (sample_text, expected_target) in test_sentences.items():
            cleaned = self.preprocessor.clean_text(sample_text)
            self.assertIn(
                expected_target,
                cleaned,
                f"Term '{sample_text}' (key: {pattern_key}) failed to normalize to '{expected_target}'. Got: '{cleaned}'",
            )

    # =========================================================================
    # 4. AGGRESSION DETECTION ON INSULTS, PROFANITY & ZERO FP ON COMPLAINTS
    # =========================================================================

    def test_08_aggression_detected_on_insults_and_future_tense_profanity(self) -> None:
        """Adversarial: Insults, behavioral aggression, and future-tense profanity MUST trigger aggression == 1."""
        aggressive_cases = [
            ("ты конченный дебил, рот закрой", "дебил / рот закрой"),
            ("ублюдки рукожопые все сломали", "ублюдки / рукожопые"),
            ("закрой рот и не позорься", "закрой рот"),
            ("включи мозги хоть раз", "включи мозги"),
            ("вы колясочник?", "колясочник (veiled insult)"),
            ("сделано чисто по глупости", "по глупости"),
            ("какие родители, такие дети", "какие родители"),
            ("чистота у вас дома такая же видимо", "чистота у вас дома"),
            ("нормальный человек до такого бы не додумался", "нормальный человек не додумался"),
            ("опять на детской площадке насрут собаки", "насрут"),
            ("придут и все засрут в подъезде", "засрут"),
            ("обосрут все стены своими граффити", "обосрут"),
            ("мы сами все засрем если не уберут", "засрем"),
        ]
        if self.ensemble is None:
            self.skipTest("Ensemble model not found.")

        df = pd.DataFrame({
            "message_id": [f"agg_{i}" for i in range(len(aggressive_cases))],
            "text": [text for text, _ in aggressive_cases],
        })
        preds = self.ensemble.predict_messages(df)

        for i, row in preds.iterrows():
            text, label = aggressive_cases[i]
            self.assertEqual(
                row["aggression"],
                1,
                f"Adversarial failure: Hostile text '{text}' ({label}) was NOT flagged as aggression!",
            )
            self.assertEqual(
                row["sentiment"],
                "negative",
                f"Adversarial failure: Hostile text '{text}' ({label}) was NOT negative!",
            )

    def test_09_zero_false_positives_on_realistic_angry_complaints(self) -> None:
        """Adversarial: Realistic harsh complaints without personal insults MUST HAVE aggression == 0."""
        standard_complaints = [
            "В подъезде ужасно грязно, когда наконец помоют полы?!",
            "Опять отключили горячую воду без предупреждения, сколько можно терпеть?!",
            "Лифт не работает уже третий день, почините немедленно!",
            "Батареи ледяные, дома холод собачий!",
            "Мусор не вывозят уже неделю, контейнеры переполнены!",
            "За что мы платим такие деньги по квитанциям, если двор не убирают?!",
            "Почему во дворе до сих пор не горит фонарь?",
            "Вчера застряли в лифте на два часа, безобразие!",
            "Прошу срочно прислать сантехника, капает стояк.",
            "У нас на первом этаже забился мусоропровод, стоит невыносимая вонь.",
            "Двор завален сугробами, скорая помощь не может проехать к подъезду!",
            "Сколько можно кормить нас завтраками, где отопление?!",
        ]
        if self.ensemble is None:
            self.skipTest("Ensemble model not found.")

        df = pd.DataFrame({
            "message_id": [f"complaint_{i}" for i in range(len(standard_complaints))],
            "text": standard_complaints,
        })
        preds = self.ensemble.predict_messages(df)

        for i, row in preds.iterrows():
            text = standard_complaints[i]
            self.assertEqual(
                row["aggression"],
                0,
                f"Adversarial FALSE POSITIVE: Standard complaint '{text}' was wrongly flagged as aggression == 1!",
            )

    # =========================================================================
    # 5. MIXED TONE ("НО ...") HANDLING
    # =========================================================================

    def test_10_mixed_tone_sentiment_negative(self) -> None:
        """Adversarial: Polite or concessive opening followed by 'но...' complaint MUST be negative."""
        mixed_examples = [
            "Спасибо за ответ, но проблема осталась не решена",
            "Спасибо за уборку, но трактор сломал забор!",
            "Вода есть, но ржавая и вонючая",
            "Конечно хорошо, но толку нет никакого",
            "Убрали снег во дворе, однако ничего не изменилось",
            "Сделали ремонт в парадной, но стало только хуже",
        ]
        if self.ensemble is None:
            self.skipTest("Ensemble model not found.")

        df = pd.DataFrame({
            "message_id": [f"mixed_{i}" for i in range(len(mixed_examples))],
            "text": mixed_examples,
        })
        preds = self.ensemble.predict_messages(df)

        for i, row in preds.iterrows():
            text = mixed_examples[i]
            self.assertEqual(
                row["sentiment"],
                "negative",
                f"Adversarial failure: Mixed tone sentence '{text}' was NOT classified as negative!",
            )

    # =========================================================================
    # 6. DEV SET METRIC QUALITY THRESHOLDS VERIFICATION
    # =========================================================================

    def test_11_dev_metrics_meet_criteria(self) -> None:
        """Empirical evaluation: Verify that dev metrics exceed baseline thresholds."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "evaluate.py",
            "--gold-messages", "data/dev_messages.csv",
            "--gold-labels", "data/dev_labels.csv",
            "--pred-dir", "dev_out/",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(res.returncode, 0, f"evaluate.py failed: {res.stderr}")
        out = res.stdout

        # Parse Q1, Q2, Q3, Total
        m_q1 = re.search(r"Q1=([0-9.]+)", out)
        m_q2 = re.search(r"Q2=([0-9.]+)", out)
        m_q3 = re.search(r"Q3=([0-9.]+)", out)
        m_total = re.search(r"Итого автоматически:\s*([0-9.]+)\s*/\s*42", out)

        self.assertIsNotNone(m_q1, f"Failed to parse Q1 from output:\n{out}")
        self.assertIsNotNone(m_q2, f"Failed to parse Q2 from output:\n{out}")
        self.assertIsNotNone(m_q3, f"Failed to parse Q3 from output:\n{out}")
        self.assertIsNotNone(m_total, f"Failed to parse Total score from output:\n{out}")

        q1 = float(m_q1.group(1))
        q2 = float(m_q2.group(1))
        q3 = float(m_q3.group(1))
        total = float(m_total.group(1))

        # Check required thresholds:
        # Q1 >= 0.7947, Q2 >= 0.7972, Q3 >= 0.6983, Total auto >= 36.57
        self.assertGreaterEqual(q1, 0.7947, f"Q1 {q1} < 0.7947")
        self.assertGreaterEqual(q2, 0.7972, f"Q2 {q2} < 0.7972")
        self.assertGreaterEqual(q3, 0.6983, f"Q3 {q3} < 0.6983")
        self.assertGreaterEqual(total, 36.57, f"Total score {total} < 36.57")

        # Check target goal ~38.18
        self.assertGreaterEqual(total, 38.00, f"Expected total score >= 38.0, got {total}")


if __name__ == "__main__":
    unittest.main()
