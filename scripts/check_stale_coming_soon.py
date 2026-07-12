#!/usr/bin/env python3
"""Detect "coming soon" chapter-nav labels whose target chapter exists.

When a series grows (e.g. 3 chapters -> 5), the old chapters' next/prev
navigation sometimes keeps its <span class="coming-soon"> placeholder
even though the target file now exists — a dead end for learners
(audit finding N-03). This script flags exactly those cases.

A coming-soon span is checked only when its label looks like chapter
navigation (次の章/前の章/Next/Previous/第N章/Chapter N/arrows); header
links such as 知識ベース (準備中) and index-page topic cards are out of
scope. The target chapter number is taken from the label when explicit,
otherwise inferred from the direction arrow.

Usage:
    python3 scripts/check_stale_coming_soon.py            # report
    python3 scripts/check_stale_coming_soon.py --check    # exit 1 on hit
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

SPAN_RE = re.compile(r'<span class="coming-soon">([^<]*)</span>')
NAV_HINT = re.compile(
    r"(次の章|前の章|Next Chapter|Previous|←|→|第\s*\d+\s*章|Chapter\s*\d+)",
    re.IGNORECASE)
EXCLUDE = re.compile(r"知識ベース|Knowledge Base", re.IGNORECASE)
EXPLICIT_N = re.compile(r"(?:第\s*(\d+)\s*章|Chapter\s*(\d+))", re.IGNORECASE)
PREV_HINT = re.compile(r"^←|Previous|前の章", re.IGNORECASE)
# index-page chapter cards that clearly point at the series' own chapter;
# related-series labels (e.g. 機械学習入門シリーズ (準備中)) are out of scope
CARD_LABEL = re.compile(
    r"^(第\s*\d+\s*章を読む|学習を開始|👉\s*Chapter\s*\d+\s*を読む"
    r"|第\s*\d+\s*章\s*[:：]|Read Chapter\s*\d+|Start Learning)")


def scan() -> list[tuple[str, str, str]]:
    stale = []
    pattern = os.path.join(REPO_ROOT, "knowledge", "*", "*", "*",
                           "chapter*.html")
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        m = re.match(r"chapter-?(\d+)", base)
        if not m:
            continue
        n = int(m.group(1))
        d = os.path.dirname(path)
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
        if "coming-soon" not in html:
            continue
        for sp in SPAN_RE.finditer(html):
            label = sp.group(1).strip()
            if EXCLUDE.search(label) or not NAV_HINT.search(label):
                continue
            em = EXPLICIT_N.search(label)
            if em:
                target_n = int(em.group(1) or em.group(2))
                if target_n == n:
                    target_n = n - 1 if PREV_HINT.search(label) else n + 1
            else:
                target_n = n - 1 if PREV_HINT.search(label) else n + 1
            cands = (glob.glob(os.path.join(d, f"chapter-{target_n}.html"))
                     + glob.glob(os.path.join(d, f"chapter{target_n}-*.html")))
            if cands:
                rel = os.path.relpath(path, REPO_ROOT)
                stale.append((rel, label, os.path.basename(cands[0])))
    pattern = os.path.join(REPO_ROOT, "knowledge", "*", "*", "*",
                           "index.html")
    for path in sorted(glob.glob(pattern)):
        d = os.path.dirname(path)
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
        if "coming-soon" not in html:
            continue
        for sp in SPAN_RE.finditer(html):
            label = sp.group(1).strip()
            if not CARD_LABEL.search(label):
                continue
            em = EXPLICIT_N.search(label)
            if em:
                target_n = int(em.group(1) or em.group(2))
            else:
                ctx = html[max(0, sp.start() - 600):sp.start()]
                ms = EXPLICIT_N.findall(ctx)
                if not ms:
                    continue
                a, b = ms[-1]
                target_n = int(a or b)
            cands = (glob.glob(os.path.join(d, f"chapter-{target_n}.html"))
                     + glob.glob(os.path.join(d, f"chapter{target_n}-*.html")))
            if cands:
                rel = os.path.relpath(path, REPO_ROOT)
                stale.append((rel, label, os.path.basename(cands[0])))
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when stale navs are found")
    args = parser.parse_args()

    stale = scan()
    if stale:
        print(f"{len(stale)} coming-soon nav(s) point at chapters that "
              f"already exist:")
        for rel, label, target in stale:
            print(f"  {rel}  [{label}] -> {target}")
        return 1 if args.check else 0
    print("ok    no stale coming-soon chapter navs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
