"""
Amharic text cleaning utilities for ASR dataset construction.

Implements the rules from the dataset spec:
  - Verbatim text, Ge'ez script only
  - Strip Latin letters, digits, emojis, annotation markers
  - Convert plain integers (0-9999) to Amharic words when possible,
    otherwise DROP the segment (safer than leaving garbage)
  - Remove [ሙዚቃ], (inaudible), >> markers, HTML entities, etc.
  - Keep natural Ge'ez punctuation (። ፣ ፤ ፥ ፦ ፧ ፨) by default
  - Character audit: list every unique char, flag anything outside Ethiopic + basic punct
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# ---------------------------------------------------------------------------
# Ethiopic Unicode blocks (Ge'ez script)
# ---------------------------------------------------------------------------
# U+1200–U+137F  : Ethiopic core (letters, punctuation)
# U+1380–U+139F  : Ethiopic Supplement
# U+2D80–U+2DDF  : Ethiopic Extended
# U+AB00–U+AB2F  : Ethiopic Extended-A
ETHIOPIC_RANGES = (
    (0x1200, 0x139F),
    (0x2D80, 0x2DDF),
    (0xAB00, 0xAB2F),
)


def is_ethiopic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ETHIOPIC_RANGES)


# Natural Ge'ez punctuation we WANT to keep
ETHIOPIC_PUNCT = set("።፣፤፥፦፧፨")  # ።=full stop  ፣=comma  ፤=semicolon  ፥=colon  ፦=preface colon  ፧=questionmark  ፨=paragraph
# ASCII punctuation we tolerate as separators (will be normalized to spaces)
ASCII_PUNCT_SEPS = set(".,;:!?-—()[]{}\"'")
# Whitespace
WHITESPACE = set(" \t\n\r\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000")

# Allowed chars = Ethiopic letters + Ethiopic punctuation + ASCII whitespace
def is_allowed(ch: str) -> bool:
    return is_ethiopic(ch) or ch in ETHIOPIC_PUNCT or ch in WHITESPACE


# ---------------------------------------------------------------------------
# Annotation / noise patterns to strip
# ---------------------------------------------------------------------------
NOISE_PATTERNS = [
    # WebVTT timing / cue settings (e.g. "<00:00:01.500>")
    re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>"),
    # Generic angle-bracket tags <i>, </i>, <b>, <c.something>, etc.
    re.compile(r"</?[a-zA-Z][^>]*>"),
    # Annotation markers: [ሙዚቃ], [Music], (inaudible), (laughs), [Applause]
    re.compile(r"[\[\(]\s*[^\]\)]{0,40}\s*[\]\)]"),
    # Speaker indicators: ">>" or "Name:" at start of cue
    re.compile(r"^>\s*>"),
    re.compile(r"^[A-ZÀ-Ÿ][A-Za-zÀ-ÿ\s]{0,30}:\s*"),
    # Music note symbols, common VTT artifacts
    re.compile(r"♪|♫|♬|♩"),
    # HTML entities
    re.compile(r"&[a-zA-Z]+;|&#\d+;"),
    # Multiple consecutive non-text markers
    re.compile(r"\.{2,}"),
    # Multiple dashes
    re.compile(r"-{2,}"),
]


# ---------------------------------------------------------------------------
# Amharic number → words (covers 0-9999, the common case)
# Drops the segment if a number has Latin digits we can't convert.
# ---------------------------------------------------------------------------
_AMHARIC_ONES = ["ዜሮ", "አንድ", "ሁለት", "ሦስት", "አራት", "አምስት",
                 "ስድስት", "ሰባት", "ስምንት", "ዘጠኝ"]
_AMHARIC_TEENS = ["አስር", "አስራአንድ", "አስራሁለት", "አስራሦስት",
                  "አስራአራት", "አስራአምስት", "አስራስድስት",
                  "አስራሰባት", "አስራስምንት", "አስራዘጠኝ"]
_AMHARIC_TENS = ["", "አስር", "ሀያ", "ሰላሳ", "አርባ", "አምሳ",
                 "ስልሳ", "ሰባ", "ሰማንያ", "ዘጠና"]


def _amharic_0_to_99(n: int) -> str:
    if n < 10:
        return _AMHARIC_ONES[n]
    if n < 20:
        return _AMHARIC_TEENS[n - 10]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _AMHARIC_TENS[tens]
    return _AMHARIC_TENS[tens] + " " + _AMHARIC_ONES[ones]


def _amharic_0_to_999(n: int) -> str:
    if n < 100:
        return _amharic_0_to_99(n)
    hundreds, rest = divmod(n, 100)
    head = "መቶ" if hundreds == 1 else _AMHARIC_ONES[hundreds] + " መቶ"
    if rest == 0:
        return head
    return head + " " + _amharic_0_to_99(rest)


def _amharic_0_to_9999(n: int) -> str:
    if n < 1000:
        return _amharic_0_to_999(n)
    thousands, rest = divmod(n, 1000)
    head = "ሺህ" if thousands == 1 else _amharic_0_to_99(thousands) + " ሺህ"
    if rest == 0:
        return head
    return head + " " + _amharic_0_to_999(rest)


def number_to_amharic_words(n: int) -> str | None:
    """Return Amharic word form, or None if out of supported range."""
    if not (0 <= n <= 9999):
        return None
    return _amharic_0_to_9999(n)


# Replace ASCII digit runs with Amharic word form (or mark segment for drop)
_DIGIT_RUN = re.compile(r"\d+")


def replace_digits_with_words(text: str) -> tuple[str, bool]:
    """
    Replace every run of ASCII digits with its Amharic word form.
    Returns (cleaned_text, ok). ok=False means an unconvertible number was
    found and the segment should be dropped.
    """
    ok = True

    def repl(m: re.Match) -> str:
        nonlocal ok
        n = int(m.group())
        words = number_to_amharic_words(n)
        if words is None:
            ok = False
            return ""
        return " " + words + " "

    return _DIGIT_RUN.sub(repl, text), ok


# ---------------------------------------------------------------------------
# Main cleaner
# ---------------------------------------------------------------------------
def clean_amharic_text(
    raw: str,
    *,
    keep_punctuation: bool = True,
    drop_on_latin: bool = True,
) -> tuple[str | None, str]:
    """
    Clean a raw subtitle line according to the spec.

    Returns (cleaned_text_or_None, reason_if_dropped).
    - cleaned_text is None when the line should be dropped.
    - reason_if_dropped is one of:
        "empty"            - ended up empty after cleaning
        "no_ethiopic"      - contains no Ethiopic characters at all
        "latin_present"    - contains Latin letters and drop_on_latin=True
        "unconvertible_number" - had a number outside 0-9999 range
        "too_many_noise"   - majority of chars were noise/annotation markers
    """
    if raw is None:
        return None, "empty"

    text = unicodedata.normalize("NFC", raw)

    # Strip noise patterns
    for pat in NOISE_PATTERNS:
        text = pat.sub(" ", text)

    # Replace digits with words
    text, ok = replace_digits_with_words(text)
    if not ok:
        return None, "unconvertible_number"

    # Strip emojis and symbols (broad catch-all for what's left)
    cleaned_chars: list[str] = []
    for ch in text:
        if is_allowed(ch):
            cleaned_chars.append(ch)
        elif ch in ASCII_PUNCT_SEPS:
            cleaned_chars.append(" ")
        elif unicodedata.category(ch).startswith(("S", "C")):
            # Symbol or control char -> drop
            cleaned_chars.append(" ")
        elif ch in WHITESPACE:
            cleaned_chars.append(" ")
        else:
            # Latin letters, etc. -> keep for now so we can flag if needed
            cleaned_chars.append(ch)

    text = "".join(cleaned_chars)

    # Latin letter check (after noise removal)
    latin_letters = re.findall(r"[A-Za-z]", text)
    if latin_letters and drop_on_latin:
        return None, "latin_present"

    # Whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()

    # Punctuation policy
    if not keep_punctuation:
        for p in ETHIOPIC_PUNCT:
            text = text.replace(p, " ")
        text = re.sub(r"\s+", " ", text).strip()

    # Empty?
    if not text:
        return None, "empty"

    # No Ethiopic at all?
    if not any(is_ethiopic(c) for c in text):
        return None, "no_ethiopic"

    # Too many noise chars compared to Ethiopic?
    ethiopic_count = sum(1 for c in text if is_ethiopic(c))
    total_non_space = sum(1 for c in text if not c.isspace())
    if total_non_space > 0 and ethiopic_count / max(1, total_non_space) < 0.5:
        return None, "too_many_noise"

    return text, ""


# ---------------------------------------------------------------------------
# Character audit
# ---------------------------------------------------------------------------
def character_audit(texts: Iterable[str]) -> dict[str, dict]:
    """
    Build a full character audit.

    Returns: { char: { "count": int, "category": str, "allowed": bool } }
    Categories: ethiopic, ethiopic_punct, ascii_punct, whitespace, latin, digit, other
    """
    audit: dict[str, dict] = {}
    for text in texts:
        for ch in text:
            if ch in audit:
                audit[ch]["count"] += 1
                continue
            if is_ethiopic(ch):
                cat = "ethiopic"
                allowed = True
            elif ch in ETHIOPIC_PUNCT:
                cat = "ethiopic_punct"
                allowed = True
            elif ch in WHITESPACE:
                cat = "whitespace"
                allowed = True
            elif ch.isdigit():
                cat = "digit"
                allowed = False
            elif ch.isascii() and ch.isalpha():
                cat = "latin"
                allowed = False
            elif ch in ASCII_PUNCT_SEPS:
                cat = "ascii_punct"
                allowed = False
            else:
                cat = "other"
                allowed = False
            audit[ch] = {"count": 1, "category": cat, "allowed": allowed}
    return audit


def format_audit_report(audit: dict[str, dict]) -> str:
    """Pretty-print the character audit as a human-readable report."""
    rows = sorted(audit.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    lines = ["# Character Audit Report", ""]
    lines.append(f"{'Char':<8} {'Count':>10}  {'Category':<14}  Allowed")
    lines.append("-" * 50)
    total = sum(v["count"] for v in audit.values())
    for ch, v in rows:
        display = ch if not ch.isspace() else f"U+{ord(ch):04X}"
        lines.append(f"{display:<8} {v['count']:>10}  {v['category']:<14}  {'YES' if v['allowed'] else 'NO'}")
    lines.append("-" * 50)
    lines.append(f"Total characters: {total}")
    flagged = [(ch, v) for ch, v in audit.items() if not v["allowed"]]
    lines.append(f"Flagged unique characters: {len(flagged)}")
    if flagged:
        lines.append("These characters should be reviewed or stripped before training:")
        for ch, v in sorted(flagged, key=lambda kv: -kv[1]["count"]):
            display = ch if not ch.isspace() else f"U+{ord(ch):04X}"
            lines.append(f"  - {display!r} (count={v['count']}, category={v['category']})")
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test
    samples = [
        "ሰላም ይህ ሶፊ ሲኒማ ቻናል ነው",                       # clean
        "Hello ሰላም world",                                # mixed Latin
        "በ 2024 ዓ.ም የተደረገ ግምጫ",                          # digits
        "[ሙዚቃ] ሰላም እንዴት ናችሁ",                          # annotation
        ">> አንድ ሰው እያለ ነበር",                            # speaker marker
        "ቁጥሩ 999999 ነው",                                 # too-large number
        "በ1996 ዓ.ም. የተወለደው ልጅ",                          # 4-digit year (ok)
        "በ 1,234 ብር ተሸጠ",                                # comma number
        "♪ ♫ music ♪",                                    # music only -> drop
        "ሰላም። እንኳን ደህና መጡ።",                            # punctuation
    ]
    for s in samples:
        out, reason = clean_amharic_text(s)
        print(f"{s!r:60s} -> {out!r:50s} reason={reason!r}")
