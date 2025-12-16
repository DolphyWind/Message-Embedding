from typing import Iterable
import spacy
from collections import Counter

spacy.require_gpu()
_nlp = spacy.load("tr_core_news_trf")
KEEP_POS: set[str] = {"NOUN", "PROPN", "ADJ"}


def group_to_query(
    group: Iterable[str],
    max_tokens: int = 10
) -> str:
    text: str = " ".join(group)
    doc = _nlp(text)

    tokens: list[str] = [
        tok.lemma_.lower()
        for tok in doc
        if tok.pos_ in KEEP_POS
        and not tok.is_stop
        and tok.is_alpha
        and len(tok) > 2
    ]

    if not tokens:
        return ""

    counts: Counter[str] = Counter(tokens)
    selected: list[str] = [
        tok for tok, _ in counts.most_common(max_tokens)
    ]

    return " ".join(selected)
