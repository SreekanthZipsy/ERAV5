"""Small multi-lane corpus used by the demo (web, code, indic, agentic, stem, eval)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


LANES = ("web", "code", "indic", "agentic", "stem", "reasoning")


@dataclass
class Document:
    doc_id: str
    lane: str
    text: str
    split: str  # train | eval | validation
    quality: float
    source: str
    # Agentic: mark observation spans with <<OBS>> ... <</OBS>>
    # Prompt/response for SFT-style: <<PROMPT>> ... <</PROMPT>> <<ANSWER>> ... <</ANSWER>>


def build_demo_corpus() -> list[Document]:
    docs: list[Document] = []

    web = [
        "The capital of India is New Delhi and it sits on the Yamuna river.",
        "Common sense says humans usually have two index fingers, one on each hand.",
        "Wikipedia lists twin primes such as seventeen and nineteen.",
        "A monsoon brings seasonal rain across the Indian subcontinent every year.",
        "General knowledge includes geography history science and culture facts.",
    ]
    for i, t in enumerate(web):
        docs.append(Document(f"web_{i}", "web", t, "train", 0.7, "fineweb_demo"))

    code = [
        "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n",
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n",
        "SELECT user_id, COUNT(*) FROM events GROUP BY user_id HAVING COUNT(*) > 10;\n",
    ]
    for i, t in enumerate(code):
        docs.append(Document(f"code_{i}", "code", t, "train", 0.85, "stack_demo"))

    indic = [
        "भारत की राजधानी नई दिल्ली है और यह यमुना नदी के किनारे बसी है।",
        "नमस्ते मेरा नाम अरjun है और मैं हिंदी में बात कर सकता हूँ।",
        "தமிழ் ஒரு செம்மொழி மற்றும் இந்தியாவின் முக்கிய மொழிகளில் ஒன்று.",
        "বাংলা ভাষায় কথা বলা মানুষের সংখ্যা বিশ্বে অনেক বেশি।",
        "Verified Indic gold: संविधान भारत का सर्वोच्च कानून है।",
    ]
    for i, t in enumerate(indic):
        q = 0.95 if "Verified" in t or "संविधान" in t else 0.75
        src = "sangraha_verified" if q > 0.9 else "setu_demo"
        docs.append(Document(f"indic_{i}", "indic", t, "train", q, src))

    agentic = [
        (
            "<<PROMPT>>Install nginx and confirm it listens on port 80.<</PROMPT>>"
            "I will update packages then install nginx."
            "<<OBS>>Reading package lists... Done</OBS>>"
            "sudo apt-get install -y nginx"
            "<<OBS>>Setting up nginx... Listening on 0.0.0.0:80</OBS>>"
            "<<ANSWER>>nginx is installed and listening on port 80.<</ANSWER>>"
        ),
        (
            "<<PROMPT>>Fix the failing unit test in add.<</PROMPT>>"
            "I will open the file and run the test."
            "<<OBS>>AssertionError: add(2,2) != 5</OBS>>"
            "return a + b  # corrected"
            "<<OBS>>1 passed in 0.01s</OBS>>"
            "<<ANSWER>>Patched add and tests pass.<</ANSWER>>"
        ),
        (
            "<<PROMPT>>Book a flight under airline policy.<</PROMPT>>"
            "Checking policy constraints then calling book_flight."
            "<<OBS>>tool:book_flight status=ok confirmation=A1</OBS>>"
            "<<ANSWER>>Booked confirmation A1 per policy.<</ANSWER>>"
        ),
    ]
    for i, t in enumerate(agentic):
        docs.append(Document(f"agent_{i}", "agentic", t, "train", 0.9, "terminal_trace_demo"))

    stem = [
        "How many integers from 1 to 1000 are divisible by 3 or 5? Answer 467 by inclusion exclusion.",
        "Forty three divided by seventeen is approximately two point five three.",
        "The derivative of x squared is two x and the integral of two x is x squared plus C.",
        "AIME style: find the remainder when 2^10 is divided by 7.",
    ]
    for i, t in enumerate(stem):
        docs.append(Document(f"stem_{i}", "stem", t, "train", 0.8, "openr1_demo"))

    reasoning = [
        "<<PROMPT>>Think low: 43 / 17?<</PROMPT>><<ANSWER>>About 2.5</ANSWER>>",
        "<<PROMPT>>Think medium: integers 1..1000 divisible by 3 or 5?<</PROMPT>"
        "<<ANSWER>>floor(1000/3)+floor(1000/5)-floor(1000/15)=333+200-66=467</ANSWER>>",
        "<<PROMPT>>Think high: plan multi-step search then recover on failure.<</PROMPT>"
        "<<ANSWER>>Plan search, call tool, on miss try alternate source, summarize.</ANSWER>>",
    ]
    for i, t in enumerate(reasoning):
        docs.append(Document(f"reason_{i}", "reasoning", t, "train", 0.88, "effort_traces_demo"))

    # Evaluation / validation firewall fodder — must NEVER enter loss-bearing train batches.
    eval_docs = [
        Document("eval_swe_1", "code", "SECRET_EVAL_PATCH: fix issue #999 in private bench", "eval", 1.0, "swe_bench_eval"),
        Document("eval_indic_1", "indic", "EVAL ONLY: हिन्दी मूल्यांकन प्रश्न गुप्त है", "eval", 1.0, "indic_eval"),
        Document("val_mmlu_1", "web", "VALIDATION: What is the capital of France? Paris.", "validation", 1.0, "mmlu_val"),
    ]
    docs.extend(eval_docs)
    return docs


def write_corpus(path: Path) -> list[Document]:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    docs = build_demo_corpus()
    out = path / "documents.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    return docs
