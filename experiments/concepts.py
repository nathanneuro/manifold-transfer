"""Shared concept lists and prompt templates for the GPT-2 → DistilGPT2 runs.

Templates are the unit of resampling for the small-n topology checks
(``bootstrap_topology``): a concept with seven items has seven points, and the
only honest variance estimate at that n comes from many templates per item.
Keep the set wide and varied in register; five templates was too few to say
whether the weekday verdict was a property of the model or of the prompts.
"""

from __future__ import annotations

TEMPLATES = [
    "{}",
    "the {}",
    "today {}",
    "word: {}",
    "it is {}",
    "It was {}",
    "On {}",
    "I think {}",
    "She said {}",
    "We arrived on {}",
    "The answer is {}",
    "Next comes {}",
    "After that, {}",
    "Everyone remembers {}",
    "Write down {}",
    "He whispered: {}",
    "In the book, {}",
    "Nothing beats {}",
    "Then came {}",
    "My favourite is {}",
    "Look, {}",
    "Finally, {}",
    "They chose {}",
    "Consider {}",
]

# concept -> (ordered items, chart topology)
CONCEPTS: dict[str, tuple[list[str], str]] = {
    "digits": ("zero one two three four five six seven eight nine".split(), "interval"),
    "teens": (
        "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        "interval",
    ),
    "ranks": (
        "first second third fourth fifth sixth seventh eighth ninth tenth".split(),
        "interval",
    ),
    "letters": (list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "interval"),
    "weekdays": (
        "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
        "circle",
    ),
    "months": (
        "January February March April May June July August September October November December".split(),
        "circle",
    ),
}

# Pre-registered read-out depths as fractions of each model's depth. Two
# depths, fixed before looking: the weekday circle in GPT-2 is reported
# mid-network, and the final layer is what the v1 audit read. Any verdict
# chosen across them is corrected for the number of depths (bonferroni).
DEPTHS: dict[str, float] = {"mid": 0.5, "final": 1.0}


def instances(items: list[str]) -> list[str]:
    """Items-major, templates-minor — the layout ``bootstrap_topology`` expects,
    and identical for both models so rows match by index."""
    return [tmpl.format(item) for item in items for tmpl in TEMPLATES]
