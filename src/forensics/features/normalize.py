"""Text sanitization shared by every consumer of raw text (stylometric features,
statistical scoring-model features, and the encoder's tokenizer). Neutralizes two
of the RAID surface attacks that measured as the worst regressions
(`homoglyph`: 72.0% accuracy, `zero_width_space`: 52.7%, see README) by mapping
the input back toward the clean-text distribution every model was actually
trained on -- no retraining needed, since clean training text is unaffected by
either transform (NFKC and the confusables map are no-ops on plain ASCII, and
there are no zero-width characters to strip in clean text).

Confirmed by inspecting the actual cached RAID attack samples directly rather
than assuming: `homoglyph` substitutes Cyrillic/Greek look-alikes for Latin
letters (e.g. Cyrillic 'е' U+0435 for Latin 'e', Greek 'Α' U+0391 for Latin
'A'); `zero_width_space` inserts U+200B between every character. Before this
fix, stripping U+200B only happened inside `stylometric_features` -- the
statistical detectors and the encoder both scored the raw, un-stripped text,
which is why those two components (not the stylometric ones) took the damage.
"""
from __future__ import annotations

import ftfy

# Cyrillic/Greek characters visually confusable with Latin letters, mapped back
# to their Latin look-alike. Deliberately limited to the strong, unambiguous
# visual matches (not a full Unicode confusables table) to avoid mangling
# genuine Cyrillic/Greek text beyond recognition.
_CONFUSABLES = {
    # Cyrillic lowercase -> Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ј": "j", "ԁ": "d", "ѵ": "v",
    # Cyrillic uppercase -> Latin
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "Ј": "J",
    "І": "I", "Ԁ": "D",
    # Greek uppercase -> Latin
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Greek lowercase -> Latin
    "ο": "o", "ν": "v", "υ": "u", "ι": "i",
}
_CONFUSABLES_TABLE = str.maketrans(_CONFUSABLES)

# Zero-width / invisible characters seen in obfuscation attacks: zero-width
# space, zero-width non-joiner, zero-width joiner, BOM/zero-width no-break
# space, word joiner.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿⁠"), None)


def normalize_text(s: str) -> str:
    s = ftfy.fix_text(s, normalization="NFKC")
    s = s.translate(_ZERO_WIDTH)
    s = s.translate(_CONFUSABLES_TABLE)
    return s
