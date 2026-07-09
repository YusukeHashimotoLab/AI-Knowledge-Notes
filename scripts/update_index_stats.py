#!/usr/bin/env python3
"""Verify or regenerate the display statistics on dojo tops and landing pages.

The knowledge base shows three layers of manually-written numbers that drift
whenever series are added or removed:

  1. Dojo header stat lines      (knowledge/<loc>/<DOJO>/index.html, .stats)
  2. Dojo category "(N Series)"  labels in category headers
  3. Landing page totals          (hero chips, domain-card badges, prose/meta)

This script recomputes everything from the filesystem (a series = a directory
directly under a dojo that contains index.html; chapters = chapter*.html files
inside it) and either reports drift (--check, default; exit 1 on mismatch) or
rewrites the numbers in place (--write).

Per-series chapter badges on dojo cards are CHECKED (reported) but never
rewritten, because their formats vary and mismatches sometimes need editorial
judgement (e.g. a series intentionally listing only a study track).

Usage:
  python3 scripts/update_index_stats.py            # check, exit 1 on drift
  python3 scripts/update_index_stats.py --write    # fix in place
  python3 scripts/update_index_stats.py --quiet    # check, print only drift
"""

from __future__ import annotations

import argparse
import os
import re
import sys

KNOWLEDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge")
LOCALES = ("jp", "en")
DOJOS = ("FM", "MI", "ML", "MS", "PI")

# Canonical chapter files only: chapter-3.html / chapter3-topic.html.
# Deliberately excludes supplements like chapter-3-enhancements.html so that
# displayed chapter counts mean "numbered chapters", not "chapter-ish files".
CHAPTER_RE = re.compile(r"^chapter-\d+\.html$|^chapter\d+-[a-z0-9-]+\.html$")


def count_dojo(locale: str, dojo: str):
    """Return (n_series, n_chapters, {slug: n_chapters}) for one dojo."""
    base = os.path.join(KNOWLEDGE, locale, dojo)
    per_series = {}
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "index.html")):
            continue
        n = 0
        for root, _dirs, files in os.walk(d):
            n += sum(1 for f in files if CHAPTER_RE.match(f))
        per_series[name] = n
    return len(per_series), sum(per_series.values()), per_series


class Patcher:
    """Collects regex substitutions for one file and applies/report them."""

    def __init__(self, path: str):
        self.path = path
        with open(path, encoding="utf-8") as fh:
            self.text = fh.read()
        self.original = self.text
        self.drift = []  # human-readable mismatch descriptions

    def sub(self, pattern: str, repl: str, label: str, count: int = 0, flags=0):
        """Replace pattern; record drift if the replacement changes anything."""
        new, n = re.subn(pattern, repl, self.text, count=count, flags=flags)
        if n and new != self.text:
            self.drift.append(label)
            self.text = new

    def changed(self) -> bool:
        return self.text != self.original

    def write(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(self.text)


def rel(path: str) -> str:
    return os.path.relpath(path, os.path.join(KNOWLEDGE, ".."))


def patch_dojo_header(p: Patcher, locale: str, n_series: int, n_chapters: int):
    if locale == "jp":
        p.sub(
            r"📚 \d+シリーズ \| 📖 \d+章",
            f"📚 {n_series}シリーズ | 📖 {n_chapters}章",
            f"header stats → {n_series}シリーズ/{n_chapters}章",
            count=1,
        )
    else:
        p.sub(
            r"📚 \d+ Series \| 📖 \d+ Chapters",
            f"📚 {n_series} Series | 📖 {n_chapters} Chapters",
            f"header stats → {n_series} Series/{n_chapters} Chapters",
            count=1,
        )


def patch_category_counts(p: Patcher, locale: str):
    """Set every category header's (N Series/Nシリーズ) to its real card count."""
    header_re = re.compile(r'<div class="category-header">([^<]*)</div>')
    headers = list(header_re.finditer(p.text))
    # Work on a copy assembled piecewise so spans stay valid.
    out = []
    prev_end = 0
    for i, m in enumerate(headers):
        seg_end = headers[i + 1].start() if i + 1 < len(headers) else len(p.text)
        block = p.text[m.end():seg_end]
        n_cards = block.count('<div class="series-item">')
        label = m.group(1)
        if locale == "jp":
            new_label, n = re.subn(r"（\d+シリーズ）", f"（{n_cards}シリーズ）", label)
        else:
            new_label, n = re.subn(r"\(\d+ Series\)", f"({n_cards} Series)", label)
        out.append(p.text[prev_end:m.start()])
        if n and new_label != label:
            p.drift.append(f"category '{label.strip()}' → {n_cards}")
            out.append(f'<div class="category-header">{new_label}</div>')
        else:
            out.append(p.text[m.start():m.end()])
        prev_end = m.end()
    out.append(p.text[prev_end:])
    p.text = "".join(out)


def fix_series_badges(p: Patcher, per_series: dict):
    """Set each card's chapters-badge number to the series' real chapter count.

    Cards are located by splitting on the series-item opener; each segment then
    holds one card's full markup (nested divs and all), so the slug href and the
    badge are guaranteed to be in the same segment. Example counts ("・35例",
    ", 30 Examples") are left untouched — they can't be verified cheaply.
    """
    opener = '<div class="series-item">'
    parts = p.text.split(opener)
    badge_re = re.compile(r'(class="badge chapters-badge">\s*)(\d+)')
    for i in range(1, len(parts)):
        seg = parts[i]
        href = re.search(r'href="\.?/?([a-z0-9][a-z0-9-]*)/index\.html"', seg)
        if not href or href.group(1) not in per_series:
            continue
        real = per_series[href.group(1)]
        m = badge_re.search(seg)
        if m and int(m.group(2)) != real:
            p.drift.append(f"badge {href.group(1)}: {m.group(2)} → {real}")
            parts[i] = badge_re.sub(lambda mm: f"{mm.group(1)}{real}", seg, count=1)
    p.text = opener.join(parts)


def patch_landing(p: Patcher, locale: str, totals, per_dojo):
    n_series, n_chapters = totals
    if locale == "jp":
        p.sub(r"<b>\d+</b> シリーズ", f"<b>{n_series}</b> シリーズ", f"chip → {n_series}シリーズ", count=1)
        p.sub(r"<b>\d+</b> 章", f"<b>{n_chapters}</b> 章", f"chip → {n_chapters}章", count=1)
        p.sub(r"\d+シリーズ\d+章", f"{n_series}シリーズ{n_chapters}章",
              f"prose totals → {n_series}/{n_chapters}")
        p.sub(r"\d+シリーズ・\d+章", f"{n_series}シリーズ・{n_chapters}章",
              f"feature totals → {n_series}/{n_chapters}")
    else:
        p.sub(r"<b>\d+</b> Series", f"<b>{n_series}</b> Series", f"chip → {n_series} Series", count=1)
        p.sub(r"<b>\d+</b> Chapters", f"<b>{n_chapters}</b> Chapters", f"chip → {n_chapters} Chapters", count=1)
        p.sub(r"\d+ series, and \d+ chapters", f"{n_series} series, and {n_chapters} chapters",
              f"meta totals → {n_series}/{n_chapters}")
        p.sub(r"\d+ series and \d+ chapters", f"{n_series} series and {n_chapters} chapters",
              f"prose totals → {n_series}/{n_chapters}")
        p.sub(r"\d+ series, \d+ chapters", f"{n_series} series, {n_chapters} chapters",
              f"jsonld totals → {n_series}/{n_chapters}")
    # Domain-card badges: identify each card by its dojo link.
    for dojo in DOJOS:
        ds, dc, _ = per_dojo[dojo]
        # Badges appear immediately before the card's dojo link.
        if locale == "jp":
            pattern = (r'(<span class="stat-badge stat-series">)\d+シリーズ(</span>\s*'
                       r'<span class="stat-badge stat-chapters">)\d+章'
                       r'(</span>\s*</div>\s*<a href="\./' + dojo + r'/index\.html")')
            repl = rf"\g<1>{ds}シリーズ\g<2>{dc}章\g<3>"
        else:
            pattern = (r'(<span class="stat-badge stat-series">)\d+ Series(</span>\s*'
                       r'<span class="stat-badge stat-chapters">)\d+ Chapters'
                       r'(</span>\s*</div>\s*<a href="\./' + dojo + r'/index\.html")')
            repl = rf"\g<1>{ds} Series\g<2>{dc} Chapters\g<3>"
        p.sub(pattern, repl, f"landing {dojo} badges → {ds}/{dc}", count=1, flags=re.S)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="fix files in place")
    ap.add_argument("--quiet", action="store_true", help="print only drift/mismatches")
    args = ap.parse_args()

    exit_code = 0
    for locale in LOCALES:
        per_dojo = {}
        for dojo in DOJOS:
            per_dojo[dojo] = count_dojo(locale, dojo)
            idx = os.path.join(KNOWLEDGE, locale, dojo, "index.html")
            p = Patcher(idx)
            n_series, n_chapters, per_series = per_dojo[dojo]
            patch_dojo_header(p, locale, n_series, n_chapters)
            patch_category_counts(p, locale)
            fix_series_badges(p, per_series)
            if p.changed():
                exit_code = 1
                for d in p.drift:
                    print(f"DRIFT {rel(idx)}: {d}")
                if args.write:
                    p.write()
                    print(f"FIXED {rel(idx)}")
            elif not args.quiet:
                print(f"ok    {rel(idx)} ({n_series} series / {n_chapters} chapters)")

        totals = (sum(v[0] for v in per_dojo.values()), sum(v[1] for v in per_dojo.values()))
        landing = os.path.join(KNOWLEDGE, locale, "index.html")
        p = Patcher(landing)
        patch_landing(p, locale, totals, per_dojo)
        if p.changed():
            exit_code = 1
            for d in p.drift:
                print(f"DRIFT {rel(landing)}: {d}")
            if args.write:
                p.write()
                print(f"FIXED {rel(landing)}")
        elif not args.quiet:
            print(f"ok    {rel(landing)} (totals {totals[0]} series / {totals[1]} chapters)")

    if args.write:
        return 0
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
