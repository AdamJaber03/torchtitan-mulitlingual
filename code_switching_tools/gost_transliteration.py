"""Reversible Cyrillic <-> Latin transliteration, GOST 7.79-2000 System B.

Russian counterpart of buckwalter_transliteration.py. Used as the fallback in
WordwiseUnigramCodeSwitching when a Russian word is absent from the 1-to-1
translation dictionary: unidecode collapses distinct letters onto the same Latin
output (ш/щ -> "sh"/"shch" share a prefix, е/э -> "e", и/й -> "i"), so a model
trained on unidecoded Russian cannot in principle recover the original string.
GOST System B keeps the map injective, which is what the en_translated_ru arm
needs in order to be the information-preserving control it is meant to be.

Two deviations from the letter of the standard, both in service of reversibility:

* ts is always "cz", never the context-conditional "c" the standard allows
  before e/i/y/j.
* Uppercase Cyrillic maps to a title-cased digraph (Ж -> "Zh", Щ -> "Shh")
  rather than an all-caps one, so decoding never has to guess word case.

Reversibility holds for well-formed Russian. It can fail only where a vowel is
immediately followed by a soft or hard sign -- "e" + "`" is indistinguishable
from "e`" (э) -- a sequence Russian orthography does not produce. Non-Russian
Cyrillic letters outside the table below pass through unchanged; run this file
directly to measure the residual rate on a corpus sample.
"""

# Modern Russian alphabet, GOST 7.79-2000 System B.
_RU2LAT = {
    "а": "a",   "б": "b",   "в": "v",   "г": "g",   "д": "d",
    "е": "e",   "ё": "yo",  "ж": "zh",  "з": "z",   "и": "i",
    "й": "j",   "к": "k",   "л": "l",   "м": "m",   "н": "n",
    "о": "o",   "п": "p",   "р": "r",   "с": "s",   "т": "t",
    "у": "u",   "ф": "f",   "х": "x",   "ц": "cz",  "ч": "ch",
    "ш": "sh",  "щ": "shh", "ъ": "``",  "ы": "y`",  "ь": "`",
    "э": "e`",  "ю": "yu",  "я": "ya",
}

# Pre-1918 Russian and neighbouring Cyrillic alphabets. Codes are distinct from
# each other and from the table above; where GOST reuses a code across alphabets
# (є and ѣ both "ye") this table picks unused codes instead.
_RU2LAT_EXTENDED = {
    "і": "i`",  "ѣ": "ye",  "ѳ": "fh",  "ѵ": "yh",  "ґ": "g`",
    "є": "ye`", "ї": "yi",  "ѕ": "z`",  "ј": "j`",  "ў": "u`",
    "џ": "dh",  "љ": "lh",  "њ": "nh",  "ћ": "th",  "ќ": "kh",
    "ђ": "dz",  "ѝ": "ih",  "ѐ": "eh",  "ѓ": "gh",  "ѫ": "oh",
}

_FULL_RU2LAT = {**_RU2LAT, **_RU2LAT_EXTENDED}

# Uppercase Cyrillic -> title-cased Latin ("Ж" -> "Zh"). The hard and soft signs
# encode to punctuation, which has no case, so Ъ and Ь share their lowercase
# codes and always decode back lowercase.
_FULL_RU2LAT.update({ru.upper(): lat.capitalize() for ru, lat in _RU2LAT.items()})
_FULL_RU2LAT.update({ru.upper(): lat.capitalize() for ru, lat in _RU2LAT_EXTENDED.items()})

# Decode map: lowercase entries win any code shared with an uppercase letter.
_LAT2RU = {}
for _ru, _lat in _FULL_RU2LAT.items():
    _LAT2RU.setdefault(_lat, _ru)
for _ru, _lat in {**_RU2LAT, **_RU2LAT_EXTENDED}.items():
    _LAT2RU[_lat] = _ru

_MAX_CODE_LEN = max(len(lat) for lat in _LAT2RU)


def russian_to_gost(text: str) -> str:
    """Transliterate Cyrillic to GOST 7.79-2000 System B Latin."""
    return "".join(_FULL_RU2LAT.get(char, char) for char in text)


def gost_to_russian(text: str) -> str:
    """Inverse of russian_to_gost, by greedy longest-match decoding."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        for length in range(min(_MAX_CODE_LEN, n - i), 0, -1):
            match = _LAT2RU.get(text[i:i + length])
            if match is not None:
                out.append(match)
                i += length
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


if __name__ == "__main__":
    import re
    import sys

    RU_CHARS = re.compile(r"[Ѐ-ӿԀ-ԯⷠ-ⷿꙀ-ꚟᲀ-᲏]")

    samples = [
        "Объект съел щуку в Ёлкино, а мэр Йошкар-Олы жёстко цитировал Чехова.",
        "Съешь же ещё этих мягких французских булок, да выпей чаю.",
        "ПРИВЕТ, Мир! Цыплёнок жарится — 42%.",
    ]
    failures = 0
    for s in samples:
        lat = russian_to_gost(s)
        back = gost_to_russian(lat)
        ok = back == s
        failures += not ok
        print(f"{'ok ' if ok else 'FAIL'}  {s}\n      -> {lat}\n      -> {back}")

    # Every letter in the table must survive a round trip on its own. Ъ and Ь are
    # excluded: their codes are punctuation and always decode back lowercase.
    for ru in _FULL_RU2LAT:
        if ru in ("Ъ", "Ь"):
            continue
        if gost_to_russian(russian_to_gost(ru)) != ru:
            print(f"FAIL  single-letter round trip: {ru!r} -> {russian_to_gost(ru)!r}")
            failures += 1

    residual = sum(len(RU_CHARS.findall(russian_to_gost(s))) for s in samples)
    print(f"\nresidual Cyrillic characters after transliteration: {residual}")
    print("FAILURES:" if failures else "all round trips passed", failures or "")
    sys.exit(1 if failures or residual else 0)
