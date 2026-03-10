import os
import re
import logging
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from difflib import get_close_matches

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

# =========================
# CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
DOCS_DIR = Path("docs")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================
# STATE
# =========================
DOC_CACHE = {"signature": None, "chunks": []}
CHAT_MEMORY: Dict[int, deque] = defaultdict(lambda: deque(maxlen=8))

DOC_TITLES = {
    "kolektyvni_dohovory.txt": "Закон «Про колективні договори і угоди»",
    "kzpp.txt": "КЗпП України",
    "vidpustky.txt": "Закон «Про відпустки»",
    "oplata_pratsi.txt": "Закон «Про оплату праці»",
    "okhorona_pratsi.txt": "Закон «Про охорону праці»",
    "voiennyi_stan_trudovi_vidnosyny.txt": "Закон про трудові відносини під час воєнного стану",
    "hirnychyi_zakon.txt": "Гірничий закон України",
    "prestyzhnist_shahtarskoi_pratsi.txt": "Закон «Про підвищення престижності шахтарської праці»",
}

STOPWORDS = {
    "і", "й", "та", "або", "але", "що", "це", "як", "до", "за", "на", "у", "в", "із",
    "зі", "для", "про", "при", "від", "по", "не", "так", "чи", "якщо", "то", "а",
    "я", "ти", "ми", "ви", "вони", "він", "вона", "мене", "мені", "мій", "моє", "моя",
    "мої", "твій", "твоя", "ваш", "ваша", "ваші",
    "с", "со", "и", "или", "но", "как", "это", "от", "по", "меня", "мне", "мой",
    "моя", "мои", "можуть", "можно", "може", "є", "бути", "быть",
    "хто", "что", "де", "где", "які", "какие", "скільки", "сколько",
    "ответь", "відповідай", "коротко", "кратко", "стисло", "пожалуйста", "будь", "ласка",
    "скажи", "скажіть", "подскажи", "підкажи"
}

SHORT_REQUEST_MARKERS = [
    "коротко", "кратко", "стисло", "в двух словах", "в двох словах",
    "без деталей", "без подробностей", "короче", "коротка відповідь",
    "коротка", "короткий ответ"
]

INTENT_PHRASES = {
    "weekend_work": [
        "робота у вихідні дні",
        "залучення до роботи у вихідні дні",
        "компенсація за роботу у вихідний день",
        "вихідний день",
    ],
    "vacation": [
        "щорічна основна відпустка",
        "тривалість щорічної основної відпустки",
        "24 календарних дні",
        "щорічні відпустки",
    ],
    "vacation_problem": [
        "перенесення щорічної відпустки",
        "ненадання відпустки",
        "відмова у відпустці",
        "порушення строків повідомлення про відпустку",
    ],
    "salary": [
        "заробітна плата",
        "оплата праці",
        "строки виплати заробітної плати",
        "доплата",
        "надбавка",
        "премія",
    ],
    "salary_delay": [
        "затримка зарплати",
        "не виплачують зарплату",
        "борг по зарплаті",
    ],
    "safety": [
        "охорона праці",
        "безпечні умови праці",
        "засоби індивідуального захисту",
        "спецодяг",
    ],
    "safety_violation": [
        "небезпечні умови",
        "порушення охорони праці",
        "відмова від небезпечної роботи",
        "травма на виробництві",
        "нещасний випадок",
    ],
    "dismissal": [
        "звільнення працівника",
        "розірвання трудового договору",
    ],
    "boss_pressure": [
        "тиск начальства",
        "погрози звільненням",
        "примус до звільнення",
        "незаконні вимоги",
        "цькування",
    ],
    "collective_agreement": [
        "колективний договір",
        "укладення колективного договору",
        "профспілка",
    ],
    "war_time": [
        "воєнний стан",
        "призупинення трудового договору",
        "особливості трудових відносин",
    ],
    "miner_benefits": [
        "пільги шахтарям",
        "компенсації працівникам гірничого підприємства",
        "гарантії шахтарям",
        "підземні роботи",
    ],
    "sick_leave": [
        "тимчасова непрацездатність",
        "лікарняний",
    ],
    "explanatory_note": [
        "пояснювальна записка",
        "объяснительная",
        "пояснительная",
        "причина відсутності на роботі",
        "причина отсутствия на работе",
    ],
    "late_absence": [
        "запізнення",
        "відсутність на роботі",
        "неявка",
        "не вийшов на зміну",
    ],
    "complaint": [
        "скарга",
        "жалоба",
        "куди скаржитись",
        "куда жаловаться",
        "написати скаргу",
    ],
}

INTENT_KEYWORDS = {
    "weekend_work": [
        "выходной", "выходные", "вихідний", "вихідні",
        "суббота", "воскресенье", "неділя",
        "заставить выйти", "выйти на работу",
    ],
    "vacation": [
        "отпуск", "отпуска", "відпустка", "відпустки",
    ],
    "vacation_problem": [
        "не дает отпуск", "не даёт отпуск", "не дают отпуск",
        "відмовляють у відпустці", "не пускают в отпуск",
        "не отпускают", "не дають відпустку",
    ],
    "salary": [
        "зарплата", "зарплату", "заробітна", "оплата",
        "премия", "надбавка", "доплата",
    ],
    "salary_delay": [
        "не платят", "задержка", "затримка",
        "борг", "должны зарплату", "не выплатили", "не виплатили",
    ],
    "safety": [
        "охрана", "охорона", "безопасность", "безпека",
        "спецодежда", "спецодяг", "каска",
    ],
    "safety_violation": [
        "опасно", "небезпечно", "травма",
        "нещасний", "несчастный", "угроза жизни", "загроза життю",
        "без каски", "без спецодягу",
    ],
    "dismissal": [
        "уволить", "увольнение", "звільнення", "сокращение",
    ],
    "boss_pressure": [
        "начальник", "начальство", "бригадир", "мастер", "майстер",
        "давит", "тиснуть", "угрожает", "погрожує",
        "заставляет", "змушує", "кричит", "кричить",
    ],
    "collective_agreement": [
        "колдоговор", "коллективный договор", "колективний договір",
        "профсоюз", "профспілка",
    ],
    "war_time": [
        "военное", "военный", "воєнний",
        "мобилизация", "мобілізація", "призупинити", "призупинення",
    ],
    "miner_benefits": [
        "шахтер", "шахтар", "льготы", "пільги", "уголь", "вугілля",
    ],
    "sick_leave": [
        "больничный", "лікарняний", "непрацездатність",
    ],
    "explanatory_note": [
        "объяснительная", "пояснительная", "пояснювальна",
        "что писать", "що писати", "как написать", "записка",
    ],
    "late_absence": [
        "проспал", "опоздал", "запізнився", "не вышел",
        "не вийшов", "прогулял", "пропустил", "пропустив", "неявка",
    ],
    "complaint": [
        "скарга", "жалоба", "скаржитись", "жаловаться",
        "інспекція", "инспекция", "куди звернутись", "куда обратиться",
    ],
}

KNOWN_TERMS = sorted(set(
    [x for values in INTENT_KEYWORDS.values() for x in values] +
    [x for values in INTENT_PHRASES.values() for x in values]
))

PRACTICAL_TEMPLATES = [
    {
        "match": ["не вышел", "не вийшов", "прогул", "неявка", "объяснительная", "пояснювальна", "пояснительная"],
        "text": (
            "Можна написати так:\n\n"
            "«Я, ПІБ, був відсутній на роботі __.__.____ з причини ________. "
            "Після того як зміг, повідомив керівника / вийшов на зв’язок. "
            "Прошу врахувати мої пояснення».\n\n"
            "Пиши спокійно, коротко і без зайвих подробиць. Якщо є довідка чи інше підтвердження — додай."
        ),
    },
    {
        "match": ["опоздал", "запізнився", "проспал", "будильник"],
        "text": (
            "Для запізнення можна так:\n\n"
            "«Я, ПІБ, запізнився на роботу __.__.____ з причини ________. "
            "На роботу прибув о __:__. Прошу врахувати мої пояснення».\n\n"
            "Краще писати коротко і по факту."
        ),
    },
    {
        "match": ["не платят", "задержка зарплаты", "затримка зарплати", "борг по зарплаті"],
        "text": (
            "Якщо затримують зарплату, починай збирати докази: місяці, суми, розрахункові листки, скріни. "
            "Скаргу можна почати так:\n\n"
            "«Прошу провести перевірку щодо порушення строків виплати заробітної плати на підприємстві _____. "
            "Заробітна плата за ______ не виплачена / виплачена із затримкою на ___ днів».\n\n"
            "Куди йти: профспілка, Держпраці, а якщо борг серйозний — ще й прокуратура."
        ),
    },
    {
        "match": ["без спецодежды", "без спецодягу", "опасно", "небезпечно", "без каски"],
        "text": (
            "Якщо умови реально небезпечні, не мовчи. Напиши коротко:\n\n"
            "«Повідомляю про небезпечні умови праці: ________. "
            "Прошу усунути порушення та забезпечити безпечні умови праці».\n\n"
            "Якщо хочеш, я можу одразу зібрати тобі готовий текст під твою ситуацію."
        ),
    },
]


def read_txt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_words(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ']{3,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def detect_short_request(question: str) -> bool:
    q = question.lower()
    return any(marker in q for marker in SHORT_REQUEST_MARKERS)


def detect_response_language(text: str) -> str:
    t = text.lower()
    ua_markers = ["і", "ї", "є", "ґ", "що", "це", "мені", "тобі", "робити", "відпустка", "працівник"]
    ru_markers = ["что", "это", "мне", "тебе", "работа", "отпуск", "объяснительная", "если"]

    ua_score = sum(1 for m in ua_markers if m in t)
    ru_score = sum(1 for m in ru_markers if m in t)

    return "uk" if ua_score >= ru_score else "ru"


def trim_text(text: str, max_len: int = 500) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    short = text[:max_len]
    if " " in short:
        short = short.rsplit(" ", 1)[0]
    return short + "..."


def clean_document_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = [line.strip() for line in text.splitlines()]

    cleaned = []
    started = False

    start_patterns = [
        r"^ЗАКОН УКРАЇНИ$",
        r"^Кодекс законів про працю України",
        r"^Гірничий закон України",
        r"^Про підвищення престижності шахтарської праці",
        r"^Стаття\s+1\b",
        r"^Розділ\s+I\b",
        r"^Глава\s+I\b",
        r"^Цей Закон",
        r"^Цей Кодекс",
        r"^Закон України",
    ]

    for line in lines:
        if not started:
            for pattern in start_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    started = True
                    break

        if not started:
            continue

        if re.match(r"^№\s*[\d/\-]+", line):
            continue
        if re.match(r"^\{.*\}$", line):
            continue
        if re.match(r"^(ВВР|Із змінами|Затверджується|Документ|Редакція|Чинний)", line, re.IGNORECASE):
            continue
        if re.match(r"^(htm|pdf|doc|zip)$", line, re.IGNORECASE):
            continue
        if re.match(r"^(Розділ документа|Картка документа|Інформація|Зміст документа)$", line, re.IGNORECASE):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_by_articles(text: str) -> List[dict]:
    text = clean_document_text(text)
    if not text:
        return []

    matches = list(re.finditer(r"(?im)^Стаття\s+\d+[^\n]*", text))
    chunks = []

    if not matches:
        step = 1500
        for i in range(0, len(text), step):
            part = text[i:i + step].strip()
            if part:
                chunks.append({"heading": "Фрагмент документа", "text": part})
        return chunks

    first_start = matches[0].start()
    preamble = text[:first_start].strip()
    if len(preamble) > 120:
        chunks.append({"heading": "Преамбула", "text": preamble})

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        first_line = chunk_text.splitlines()[0].strip()
        chunks.append({"heading": first_line, "text": chunk_text})

    return chunks


def get_docs_signature() -> Tuple:
    if not DOCS_DIR.exists():
        return tuple()

    signature = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        stat = path.stat()
        signature.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def load_document_chunks() -> List[dict]:
    signature = get_docs_signature()
    if DOC_CACHE["signature"] == signature:
        return DOC_CACHE["chunks"]

    chunks = []

    if not DOCS_DIR.exists():
        DOC_CACHE["signature"] = signature
        DOC_CACHE["chunks"] = []
        return []

    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = read_txt_file(path)
        if not text.strip():
            continue

        source_title = DOC_TITLES.get(path.name, path.stem)

        for chunk in split_by_articles(text):
            chunk_text = chunk["text"]
            heading = chunk["heading"]

            chunks.append({
                "source": path.name,
                "source_title": source_title,
                "heading": heading,
                "text": chunk_text,
                "text_lower": chunk_text.lower(),
                "words": set(normalize_words(chunk_text)),
                "heading_words": set(normalize_words(heading)),
            })

    DOC_CACHE["signature"] = signature
    DOC_CACHE["chunks"] = chunks
    return chunks


def fuzzy_matches_from_question(question: str) -> List[str]:
    words = re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ']{4,}", question.lower())
    found = []

    for word in words:
        matches = get_close_matches(word, KNOWN_TERMS, n=3, cutoff=0.78)
        found.extend(matches)

    result = []
    seen = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def detect_intents(question: str) -> List[str]:
    q = question.lower()
    fuzzy_found = fuzzy_matches_from_question(question)

    intents = []

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(k in q for k in keywords):
            intents.append(intent)
            continue

        if any(any(k in f for k in keywords) or any(f in k for k in keywords) for f in fuzzy_found):
            intents.append(intent)

    if ("выход" in q or "вихід" in q) and ("работ" in q or "робот" in q or "выйт" in q or "вийт" in q):
        if "weekend_work" not in intents:
            intents.append("weekend_work")

    if ("объясн" in q or "поясн" in q or "пояснюв" in q) and "explanatory_note" not in intents:
        intents.append("explanatory_note")

    if any(w in q for w in ["проспал", "опоздал", "запізнився", "не вышел", "не вийшов", "неявка"]):
        if "late_absence" not in intents:
            intents.append("late_absence")

    return intents


def is_practical_question(question: str) -> bool:
    intents = set(detect_intents(question))
    practical_intents = {
        "explanatory_note",
        "late_absence",
        "boss_pressure",
        "complaint",
    }
    return bool(intents & practical_intents)


def expand_question(question: str) -> str:
    intents = detect_intents(question)
    extra_terms = []
    fuzzy_terms = fuzzy_matches_from_question(question)

    extra_terms.extend(fuzzy_terms)

    for intent in intents:
        extra_terms.extend(INTENT_PHRASES.get(intent, []))

    q = question.lower()
    if "не выш" in q or "не вийш" in q or "прогул" in q or "неявка" in q:
        extra_terms.extend(["відсутність на роботі", "пояснювальна записка", "причина відсутності"])

    if "не дает отпуск" in q or "не дає відпустку" in q or "не отпускают" in q:
        extra_terms.extend(["перенесення щорічної відпустки", "порушення строків повідомлення про відпустку"])

    result = []
    seen = set()
    for item in extra_terms:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return (question + " " + " ".join(result)).strip()


def topic_bonus(question: str, chunk: dict) -> int:
    intents = detect_intents(question)
    source = chunk["source"].lower()
    heading = chunk["heading"].lower()
    text = chunk["text_lower"]

    bonus = 0

    if "vacation" in intents:
        if "vidpust" in source or "kzpp" in source:
            bonus += 10
        if "відпуст" in heading or "відпуст" in text:
            bonus += 12

    if "vacation_problem" in intents:
        if "vidpust" in source or "kzpp" in source:
            bonus += 12

    if "salary" in intents or "salary_delay" in intents:
        if "oplata" in source or "kzpp" in source:
            bonus += 10

    if "safety" in intents or "safety_violation" in intents:
        if "okhorona" in source or "kzpp" in source or "hirnychyi" in source:
            bonus += 10

    if "war_time" in intents:
        if "voiennyi" in source:
            bonus += 14

    if "collective_agreement" in intents:
        if "kolektyvni" in source:
            bonus += 14

    if "miner_benefits" in intents:
        if "hirnychyi" in source or "prestyzhnist" in source:
            bonus += 14

    if "weekend_work" in intents:
        if "kzpp" in source:
            bonus += 16
        if "вихід" in heading or "вихід" in text:
            bonus += 20
        if "робота у вихідні" in text:
            bonus += 20

    if "dismissal" in intents and "kzpp" in source:
        bonus += 10

    if "sick_leave" in intents and "kzpp" in source:
        bonus += 8

    return bonus


def find_relevant_chunks(question: str, chunks: List[dict], top_n: int = 4) -> Tuple[List[dict], int]:
    expanded_question = expand_question(question)
    question_words = set(normalize_words(expanded_question))
    q_lower = expanded_question.lower()

    if not question_words:
        return [], 0

    scored = []

    for chunk in chunks:
        overlap = question_words & chunk["words"]
        heading_overlap = question_words & chunk["heading_words"]

        score = len(overlap) * 3 + len(heading_overlap) * 6
        score += topic_bonus(expanded_question, chunk)

        for word in question_words:
            if len(word) >= 6 and word in chunk["text_lower"]:
                score += 1

        if ("объясн" in q_lower or "поясн" in q_lower) and ("пояснюв" in chunk["text_lower"] or "поясн" in chunk["text_lower"]):
            score += 6

        if ("выходн" in q_lower or "вихідн" in q_lower) and "вихід" in chunk["text_lower"]:
            score += 8

        if ("отпуск" in q_lower or "відпуст" in q_lower) and "24 календарних дні" in chunk["text_lower"]:
            score += 12

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    result = []
    seen = set()
    for score, chunk in scored:
        key = (chunk["source"], chunk["heading"])
        if key in seen:
            continue
        seen.add(key)
        result.append((score, chunk))
        if len(result) >= top_n:
            break

    if not result:
        return [], 0

    max_score = result[0][0]
    return [item[1] for item in result], max_score


def docs_are_relevant(question: str, relevant_chunks: List[dict], max_score: int) -> bool:
    intents = detect_intents(question)

    if not relevant_chunks:
        return False

    if max_score >= 18:
        return True

    if intents and max_score >= 12:
        doc_heavy_intents = {
            "weekend_work", "vacation", "vacation_problem", "salary",
            "salary_delay", "safety", "safety_violation", "dismissal",
            "collective_agreement", "war_time", "miner_benefits", "sick_leave"
        }
        if any(intent in doc_heavy_intents for intent in intents):
            return True

    return False


def build_context_for_llm(relevant_chunks: List[dict]) -> str:
    parts = []
    for i, chunk in enumerate(relevant_chunks, start=1):
        parts.append(
            f"[Фрагмент {i}]\n"
            f"Документ: {chunk['source_title']}\n"
            f"Файл: {chunk['source']}\n"
            f"Стаття/розділ: {chunk['heading']}\n"
            f"Текст: {trim_text(chunk['text'], 1100)}"
        )
    return "\n\n".join(parts)


def build_sources_list(relevant_chunks: List[dict]) -> str:
    items = []
    seen = set()

    for chunk in relevant_chunks:
        item = f"{chunk['source_title']}, {chunk['heading']}"
        if item not in seen:
            seen.add(item)
            items.append(item)

    return "; ".join(items[:3])


def get_practical_template(question: str) -> Optional[str]:
    q = question.lower()
    for item in PRACTICAL_TEMPLATES:
        if any(key in q for key in item["match"]):
            return item["text"]
    return None


def build_fallback_answer(question: str, relevant_chunks: List[dict], short_mode: bool, docs_relevant: bool) -> str:
    practical = get_practical_template(question)
    if practical:
        return practical

    if docs_relevant and relevant_chunks:
        first = relevant_chunks[0]
        if short_mode:
            return (
                f"Коротко: {trim_text(first['text'], 320)}\n\n"
                f"Основа: {first['source_title']}, {first['heading']}"
            )
        return (
            f"Я знайшов найближчу норму:\n{trim_text(first['text'], 600)}\n\n"
            f"Основа: {first['source_title']}, {first['heading']}"
        )

    return (
        "Розкажи трохи простіше, що саме сталося. "
        "Я або підкажу по закону, або дам короткий практичний шаблон."
    )


def get_history_text(chat_id: int) -> str:
    history = CHAT_MEMORY[chat_id]
    if not history:
        return ""

    lines = []
    for role, content in list(history)[-4:]:
        prefix = "Користувач" if role == "user" else "Бот"
        lines.append(f"{prefix}: {trim_text(content, 260)}")
    return "\n".join(lines)


def explain_with_llm(chat_id: int, question: str, relevant_chunks: List[dict], docs_relevant: bool) -> str:
    short_mode = detect_short_request(question)
    response_lang = detect_response_language(question)

    if not client:
        return build_fallback_answer(question, relevant_chunks, short_mode, docs_relevant)

    practical_template = get_practical_template(question)
    context_text = build_context_for_llm(relevant_chunks) if relevant_chunks else "Немає релевантних фрагментів."
    history_text = get_history_text(chat_id)
    sources_text = build_sources_list(relevant_chunks) if relevant_chunks else ""

    lang_rule = (
        "Відповідай тільки українською мовою. Не змішуй українську з російською."
        if response_lang == "uk"
        else "Отвечай только на русском языке. Не смешивай русский с украинским."
    )

    if short_mode:
        style_block = (
            "Користувач просить коротку відповідь. Відповідь максимум 3-4 речення. Без нумерації, заголовків і води."
            if response_lang == "uk"
            else "Пользователь просит короткий ответ. Ответ максимум 3-4 предложения. Без нумерации, заголовков и воды."
        )
    else:
        style_block = (
            "Відповідай як живий помічник у чаті. Спочатку скажи головне простими словами. Потім, якщо треба, дай одну конкретну пораду або короткий шаблон. Не розтягуй відповідь."
            if response_lang == "uk"
            else "Отвечай как живой помощник в чате. Сначала скажи главное простыми словами. Потом, если нужно, дай один конкретный совет или короткий шаблон. Не растягивай ответ."
        )

    doc_mode_block = (
        "Якщо документи реально підходять, спирайся на них і в кінці додай один рядок 'Основа: ...'. Якщо документи підходять слабко, все одно допоможи по-людськи, але не вигадуй статті."
        if response_lang == "uk"
        else "Если документы реально подходят, опирайся на них и в конце добавь одну строку 'Основа: ...'. Если документы подходят слабо, все равно помоги по-человечески, но не придумывай статьи."
    )

    instructions = (
        "Ти помічник з трудових питань для працівників шахти. "
        "Ти добре розумієш звичайну мову, суржик, помилки й криві формулювання. "
        "Твоє завдання — відповідати коротко, спокійно, зрозуміло і по-людськи. "
        "Не пиши як юрист і не пиши як лекцію. "
        "Не моралізуй. "
        "Не підштовхуй до брехні. Краще допоможи написати нейтрально, коротко і без зайвого проти себе. "
        "Якщо людина питає практичну річ, можна дати короткий шаблон. "
        + lang_rule + " " + doc_mode_block + " " + style_block
        if response_lang == "uk"
        else
        "Ты помощник по трудовым вопросам для работников шахты. "
        "Ты хорошо понимаешь обычную речь, суржик, ошибки и кривые формулировки. "
        "Твоя задача — отвечать коротко, спокойно, понятно и по-человечески. "
        "Не пиши как юрист и не пиши как лекцию. "
        "Не морализируй. "
        "Не подталкивай ко лжи. Лучше помоги написать нейтрально, коротко и без лишнего против себя. "
        "Если человек спрашивает практическую вещь, можно дать короткий шаблон. "
        + lang_rule + " " + doc_mode_block + " " + style_block
    )

    prompt = (
        f"Останні репліки діалогу:\n{history_text or 'Немає попереднього контексту.'}\n\n"
        f"Нове питання користувача:\n{question}\n\n"
        f"Чи є релевантні документи: {'так' if docs_relevant else 'ні або слабко'}\n\n"
        f"Фрагменти документів:\n{context_text}\n\n"
        f"Якщо доречно, готовий практичний шаблон:\n{practical_template or 'Немає окремого шаблону.'}\n\n"
        f"Можливі джерела для рядка 'Основа':\n{sources_text or 'Не вказуй основу, якщо реально не спирався на документи.'}\n\n"
        "Відповідай коротко і по-людськи."
        if response_lang == "uk"
        else
        f"Последние реплики диалога:\n{history_text or 'Нет предыдущего контекста.'}\n\n"
        f"Новый вопрос пользователя:\n{question}\n\n"
        f"Есть ли релевантные документы: {'да' if docs_relevant else 'нет или слабо'}\n\n"
        f"Фрагменты документов:\n{context_text}\n\n"
        f"Если уместно, готовый практический шаблон:\n{practical_template or 'Нет отдельного шаблона.'}\n\n"
        f"Возможные источники для строки 'Основа':\n{sources_text or 'Не указывай основу, если реально не опирался на документы.'}\n\n"
        "Отвечай коротко и по-человечески."
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=prompt,
        )
        answer = (response.output_text or "").strip()

        if not answer:
            return build_fallback_answer(question, relevant_chunks, short_mode, docs_relevant)

        if docs_relevant and sources_text and "основа:" not in answer.lower() and not is_practical_question(question):
            answer = f"{answer}\n\nОснова: {sources_text}"

        return answer

    except Exception as e:
        logging.exception("LLM error")
        return (
            "LLM-відповідь тимчасово не спрацювала.\n\n"
            f"Помилка: {e}\n\n"
            + build_fallback_answer(question, relevant_chunks, short_mode, docs_relevant)
        )


def answer_from_documents(chat_id: int, question: str) -> str:
    chunks = load_document_chunks()

    if not chunks:
        return (
            "Я не бачу документів у папці docs.\n\n"
            "Поклади туди txt-файли із законами та колдоговором."
        )

    relevant_chunks, max_score = find_relevant_chunks(question, chunks)

    if not relevant_chunks:
        broader_question = expand_question(question)
        relevant_chunks, max_score = find_relevant_chunks(broader_question, chunks)

    docs_relevant = docs_are_relevant(question, relevant_chunks, max_score)

    if is_practical_question(question) and max_score < 24:
        docs_relevant = False

    return explain_with_llm(chat_id, question, relevant_chunks, docs_relevant)


def split_long_message(text: str, limit: int = 3800) -> List[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]

    parts = []
    while len(text) > limit:
        chunk = text[:limit]
        split_at = chunk.rfind("\n")
        if split_at < 1200:
            split_at = chunk.rfind(" ")
        if split_at < 1200:
            split_at = limit

        parts.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        parts.append(text)

    return parts


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привіт. Пиши як звичайній людині.\n\n"
        "Я можу:\n"
        "- підказати по законах;\n"
        "- пояснити, що робити в простих робочих ситуаціях;\n"
        "- допомогти з коротким текстом пояснювальної чи скарги."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Приклади:\n"
        "- можуть мене викликати у вихідний?\n"
        "- якщо не дають відпустку, що робити?\n"
        "- я не вийшов на роботу, що писати в пояснювальній?\n"
        "- затримують зарплату, куди скаржитись?\n"
        "- якщо не видали спецодяг, що робити?"
    )


async def reload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    DOC_CACHE["signature"] = None
    DOC_CACHE["chunks"] = []
    load_document_chunks()
    await update.message.reply_text("Документи перезавантажено.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    CHAT_MEMORY[chat_id].append(("user", user_text))
    answer = answer_from_documents(chat_id, user_text)
    CHAT_MEMORY[chat_id].append(("assistant", answer))

    for part in split_long_message(answer):
        await update.message.reply_text(part)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("Не найден TELEGRAM_BOT_TOKEN в .env")

    if not OPENAI_API_KEY:
        logging.warning("OPENAI_API_KEY не найден. Бот будет работать слабее, без LLM.")

    load_document_chunks()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reload", reload_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Бот запущен. Модель: %s", OPENAI_MODEL)
    app.run_polling()


if __name__ == "__main__":
    main()