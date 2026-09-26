"""Dense concepts for the far-field experiments that need large n.

The named concepts in ``experiments/concepts.py`` have 7-26 items, enough for
ordering nulls but not for a structure function over a decade of lags
(``e08_wiener_spiral``) or a prior-vs-spacing regression with landmarks
(``e07_prior_cartogram``). Years and integers are dense, ordered, have a known
and lumpy prior, and GPT-2 represents them (Modell, Rubin-Delanchy & Whiteley,
arXiv:2505.18235: GPT-2 small's years are isometric to log(2019 - year)).

Only items that are a *single* token after a leading space are kept (the prompt
templates put the item last, and the prior is read as the next-token mass on
that one token); :func:`single_token_items` does the filtering with the model's
own tokenizer and returns each kept item's integer value, so gaps are handled
by the estimators (``roughness.structure_function(positions=...)``).

Landmarks are pre-registered here, before any run: years that are heavily
mentioned, and round or culturally special numbers that are *not* digit-count
boundaries (so a bulge there cannot be Cacioli's tokenisation effect at 9|10 and
99|100).
"""

from __future__ import annotations

YEARS = [str(y) for y in range(1700, 2021)]
NUMBERS = [str(n) for n in range(0, 200)]

YEAR_TEMPLATES = [
    "In {}",
    "In the year {}",
    "Back in {}",
    "It was {}",
    "Since {}",
    "By {}",
    "Around {}",
    "In the summer of {}",
    "The census of {}",
    "He was born in {}",
    "They married in {}",
    "It happened in {}",
]

NUMBER_TEMPLATES = [
    "{}",
    "number {}",
    "I counted {}",
    "There were {}",
    "It costs {}",
    "page {}",
    "exactly {}",
    "about {}",
    "The total is {}",
    "She scored {}",
    "We need {}",
    "Room {}",
]

# Contexts that end right before the item, for the model's own prior over items
# (next-token mass on " <item>"). Written with a trailing "{}" so they go
# through the same cache machinery with an empty item.
YEAR_PRIOR_CONTEXTS = ["In{}", "In the year{}", "Back in{}", "It was{}", "Since{}", "By{}"]
NUMBER_PRIOR_CONTEXTS = ["There were{}", "I counted{}", "It costs{}", "page{}", "exactly{}", "about{}"]

YEAR_LANDMARKS = [1776, 1789, 1812, 1848, 1865, 1914, 1918, 1929, 1939, 1941, 1945, 1969, 1984, 1989, 2000, 2001, 2008]
NUMBER_LANDMARKS = [12, 15, 20, 24, 25, 30, 40, 50, 60, 75, 144, 150]

DENSE = {
    "years": (YEARS, YEAR_TEMPLATES, YEAR_PRIOR_CONTEXTS, YEAR_LANDMARKS),
    "numbers": (NUMBERS, NUMBER_TEMPLATES, NUMBER_PRIOR_CONTEXTS, NUMBER_LANDMARKS),
}


def single_token_items(model_name: str, items: list[str]) -> tuple[list[str], list[int], list[int]]:
    """Items whose ``" " + item`` is one token for ``model_name``'s tokenizer.
    Returns ``(kept_items, their integer values, their token ids)``."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    kept, values, ids = [], [], []
    for it in items:
        enc = tok.encode(" " + it)
        if len(enc) == 1:
            kept.append(it)
            values.append(int(it))
            ids.append(enc[0])
    return kept, values, ids
