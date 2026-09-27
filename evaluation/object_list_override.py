"""Eval-only oracle object lists for the scene decoder.

``experiment.eval_object_list_override`` replaces the autoregressively decoded
object-category tokens with a fixed list, so the pose / track / state heads run
on a known set of objects. Two sources:

- ``gt_text``: categories named in the GT caption (upper bound of a
  text -> object-list front end);
- ``gt_objects``: the GT object list itself (upper bound of perfect category
  prediction).

The ground plane is always appended last, matching the training order.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from dataset_process.object_taxonomy import CATEGORY_ALIASES, canonical_category, is_static_ground
from dataset_process.omomo.omomo_io import CANONICAL_OBJECT_NAMES as OMOMO_OBJECT_NAMES

MODES = (None, "gt_text", "gt_objects")
# Handled inside run.eval_model's object loop rather than here:
#   no_early_stop  -- mask <|eoobj|> and ground until one real object is emitted;
#   pred_text      -- this module's gt_text extraction on the model's own caption;
#   multilabel     -- every first-step category above a probability threshold;
#   set_head       -- every category the trained set head scores above the
#                     threshold, forced in canonical order (needs eval_feed_soobj);
#   pred_text_verified -- caption categories the object head also gives
#                     p >= eval_caption_verify_threshold (verified_caption_categories)
#                     are forced first, then decoding continues as no_early_stop.
DECODE_MODES = ("no_early_stop", "pred_text", "multilabel", "set_head", "pred_text_verified")

# Caption spellings of single-word categories that no source alias table
# carries (the captions split them).
TEXT_ALIASES = {
    "soccer_ball": "soccerball",
    "large_box": "box",
    "small_box": "box",
    "plastic_box": "box",
    "floor_lamp": "floor_lamp",
    "small_table": "side_table",
    "large_table": "table",
    "clothes_stand": "clothes_rack",
    "trash_bin": "trash_can",
}

_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "several": 2, "both": 2}


def _text_names(category: str) -> set[str]:
    """Surface names a caption may use for ``category``: the category itself,
    its source aliases (OMOMO's "largebox", "floorlamp", ...) and the joined
    or split spelling of compounds ("floor_lamp" <-> "floorlamp")."""
    names = {category}
    for aliases in (CATEGORY_ALIASES, OMOMO_OBJECT_NAMES, TEXT_ALIASES):
        names.update(src for src in aliases if canonical_category(aliases[src]) == category)
    for name in list(names):
        names.add(name.replace("_", ""))
    return {n for n in names if n}


def _phrase_pattern(category: str) -> re.Pattern:
    words = [w for w in category.split("_") if w]
    head = words[-1]
    # Plural forms of the head noun: mugs, boxes, knives, shelves, trays/berries.
    forms = {head, head + "s", head + "es"}
    if head.endswith("fe"):
        forms.add(head[:-2] + "ves")
    elif head.endswith("f"):
        forms.add(head[:-1] + "ves")
    elif head.endswith("y"):
        forms.add(head[:-1] + "ies")
    head_alt = "(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + ")"
    body = r"[\s\-]+".join([re.escape(w) for w in words[:-1]] + [head_alt])
    # Optional count word before the phrase ("two mugs").
    return re.compile(rf"(?:\b(\w+)\s+)?\b{body}\b")


def _category_patterns(category: str) -> list[re.Pattern]:
    return [_phrase_pattern(name) for name in sorted(_text_names(category), key=len, reverse=True)]


def text_object_categories(text: str, categories: Iterable[str]) -> list[str]:
    """Categories named in ``text``, in order of first mention.

    Longer names are matched first and consume their span, so "table lamp"
    does not also yield "table". A preceding count word ("two mugs") repeats
    the category that many times.
    """
    text = text.lower()
    taken = [False] * len(text)
    hits: list[tuple[int, str, int]] = []
    for category in sorted(set(categories), key=lambda c: (-len(c), c)):
        if is_static_ground(category):
            continue
        matches = [m for pattern in _category_patterns(category) for m in pattern.finditer(text)]
        for m in matches:
            start = m.start(0) if m.group(1) is None else m.start(0) + len(m.group(1)) + 1
            if any(taken[start:m.end()]):
                continue
            for i in range(start, m.end()):
                taken[i] = True
            count = _NUMBER_WORDS.get(m.group(1) or "", 1)
            hits.append((start, category, count))
    hits.sort()
    out: list[str] = []
    for _, category, count in hits:
        # One instance per category unless a count word says otherwise.
        have = out.count(category)
        out.extend([category] * max(0, count - have))
    return out


def override_categories(
    mode: str | None,
    *,
    descriptions: Sequence[str] | str | None,
    gt_object_names: Iterable[str],
    decodable_categories: Iterable[str],
    max_objects: int,
) -> list[str] | None:
    """Forced category list (ground last) or ``None`` when ``mode`` is off."""
    if mode is None:
        return None
    if mode not in MODES:
        raise ValueError(f"eval_object_list_override must be one of {MODES}, got {mode!r}")
    decodable = set(decodable_categories)
    if mode == "gt_objects":
        cats = [canonical_category(n) for n in gt_object_names]
        cats = [c for c in cats if not is_static_ground(c)]
    else:
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        cats = text_object_categories(" ".join(descriptions or []), decodable)
    cats = [c for c in cats if c in decodable][: max(0, max_objects - 1)]
    ground = next((c for c in decodable if is_static_ground(c)), None)
    if ground is not None:
        cats.append(ground)
    return cats


def verified_caption_categories(
    caption: str,
    first_step_probs: dict[str, float],
    decodable_categories: Iterable[str],
    threshold: float,
    max_objects: int,
) -> list[str]:
    """Non-ground categories the caption names (mention order) that the object
    head's first-step distribution also supports (``p >= threshold``).

    The caption names the right object more often than the object token does,
    but about half of the time it names one that is not in the scene; the object
    head assigns those a very low probability, so a small threshold removes most
    of them. The result is only a forced prefix: decoding continues afterwards,
    so objects the caption never mentions can still be emitted.
    """
    decodable = set(decodable_categories)
    named = text_object_categories(caption or "", decodable)
    kept = [c for c in named if not is_static_ground(c) and first_step_probs.get(c, 0.0) >= threshold]
    return kept[: max(0, max_objects - 1)]
