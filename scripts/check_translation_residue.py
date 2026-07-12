#!/usr/bin/env python3
"""Detect Japanese machine-translation residue in English pages.

Counts Japanese sentence-ending punctuation (。) and kana characters in
the visible body text of knowledge/en/**/*.html, after removing regions
that legitimately contain Japanese:

- <script>, <style>, <code>, <pre> blocks (code demos, JSON payloads)
- elements explicitly marked lang="ja"
- HTML comments

A healthy translated page has zero 。 and at most a handful of kana
(e.g. a bilingual glossary column). Pages exceeding the thresholds are
almost certainly unfinished machine translations (audit finding N-02).

Usage:
    python3 scripts/check_translation_residue.py            # report
    python3 scripts/check_translation_residue.py --check    # exit 1 on hit

Thresholds: fail when 。 count > MAX_KUTEN or kana count > MAX_KANA.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EN_ROOT = os.path.join(REPO_ROOT, "knowledge", "en")

MAX_KUTEN = 0   # any 。 in EN body prose is residue
MAX_KANA = 40   # generous allowance for bilingual glossary tables

# Pages that legitimately contain Japanese body text (reviewed by hand).
ALLOWLIST = {
    # NLP chapter teaching Japanese text processing: MeCab example
    # sentence, particle examples (は/の), katakana spelling variants,
    # and a Japanese book citation.
    "knowledge/en/ML/nlp-introduction/chapter1-nlp-basics.html",
}

STRIP_BLOCKS = re.compile(
    r"<(script|style|code|pre)\b[^>]*>.*?</\1>|<!--.*?-->",
    re.DOTALL | re.IGNORECASE,
)
# crude but effective: drop any tag that carries lang="ja" together with
# its immediate text content (glossary cells, locale labels)
STRIP_JA_TAGGED = re.compile(
    r'<([a-zA-Z0-9]+)\b[^>]*lang="ja"[^>]*>.*?</\1>', re.DOTALL
)
TAGS = re.compile(r"<[^>]+>")
KANA = re.compile(r"[ぁ-ゟァ-ヿ]")


def body_text(html: str) -> str:
    html = STRIP_BLOCKS.sub(" ", html)
    html = STRIP_JA_TAGGED.sub(" ", html)
    return TAGS.sub(" ", html)


def scan() -> list[tuple[str, int, int]]:
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(EN_ROOT):
        for name in sorted(filenames):
            if not name.endswith(".html"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as fh:
                text = body_text(fh.read())
            kuten = text.count("。")  # 。
            kana = len(KANA.findall(text))
            if kuten > MAX_KUTEN or kana > MAX_KANA:
                rel = os.path.relpath(path, REPO_ROOT)
                if rel in ALLOWLIST:
                    continue
                offenders.append((rel, kuten, kana))
    return offenders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when residue is found")
    args = parser.parse_args()

    offenders = scan()
    if offenders:
        print(f"Japanese residue found in {len(offenders)} EN page(s) "
              f"(thresholds: 。>{MAX_KUTEN}, kana>{MAX_KANA}):")
        for rel, kuten, kana in sorted(offenders, key=lambda t: -t[1]):
            print(f"  {rel}  (。={kuten}, kana={kana})")
        return 1 if args.check else 0
    print("ok    no Japanese residue in knowledge/en")
    return 0


if __name__ == "__main__":
    sys.exit(main())
