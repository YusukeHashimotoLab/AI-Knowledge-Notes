#!/usr/bin/env python3
"""Generate the client-side search index for the knowledge base.

Walks knowledge/<locale> and emits assets/search-index.json per locale — a lean
array the search page (knowledge/<locale>/search.html) fetches and filters in
the browser. Indexed per page: title, dojo, series, section headings, and the
meta description. Body text is deliberately NOT indexed to keep the payload
small enough for client-side use.

Record shape (short keys to save bytes):
  {"u": "ML/optimization-introduction/chapter1-….html",  # URL relative to locale root
   "t": "Chapter 1: Optimization Fundamentals",           # page title (h1 preferred)
   "d": "ML",                                             # dojo code
   "s": "optimization-introduction",                      # series slug ('' for dojo tops)
   "h": "1.1 What is… | 1.2 Convexity…",                  # h2/h3 headings, ' | ' joined
   "x": "meta description text"}

Usage:
  python3 scripts/build_search_index.py          # write both locales
  python3 scripts/build_search_index.py --check  # exit 1 if on-disk index is stale
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

KNOWLEDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge")
LOCALES = ("jp", "en")
DOJOS = ("FM", "MI", "ML", "MS", "PI")

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def text(html_fragment: str) -> str:
    return WS_RE.sub(" ", TAG_RE.sub("", html_fragment)).strip()


def extract(path: str):
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    m = re.search(r"<h1[^>]*>(.*?)</h1>", s, re.S)
    if not m:
        m = re.search(r"<title>(.*?)</title>", s, re.S)
    title = text(m.group(1))[:120] if m else ""
    heads = [text(h)[:80] for h in re.findall(r"<h[23][^>]*>(.*?)</h[23]>", s, re.S)]
    heads = [h for h in heads if h][:20]
    m = re.search(r'<meta[^>]*name="description"[^>]*content="([^"]*)"', s) or \
        re.search(r'<meta[^>]*content="([^"]*)"[^>]*name="description"', s)
    desc = text(m.group(1))[:200] if m else ""
    return title, " | ".join(heads), desc


def build(locale: str):
    records = []
    base = os.path.join(KNOWLEDGE, locale)
    for dojo in DOJOS:
        droot = os.path.join(base, dojo)
        for root, dirs, files in os.walk(droot):
            dirs.sort()
            for f in sorted(files):
                if not f.endswith(".html"):
                    continue
                path = os.path.join(root, f)
                rel = os.path.relpath(path, base)
                parts = rel.split(os.sep)
                series = parts[1] if len(parts) == 3 else ""
                title, heads, desc = extract(path)
                if not title:
                    continue
                records.append({"u": rel.replace(os.sep, "/"), "t": title,
                                "d": dojo, "s": series, "h": heads, "x": desc})
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the on-disk index matches; exit 1 if stale")
    args = ap.parse_args()

    stale = False
    for locale in LOCALES:
        records = build(locale)
        payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
        out = os.path.join(KNOWLEDGE, locale, "assets", "search-index.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if args.check:
            try:
                current = open(out, encoding="utf-8").read()
            except FileNotFoundError:
                current = ""
            if current != payload:
                print(f"STALE {locale}/assets/search-index.json "
                      f"(regenerate with: python3 scripts/build_search_index.py)")
                stale = True
            else:
                print(f"ok    {locale}/assets/search-index.json "
                      f"({len(records)} pages, {len(payload)//1024}KB)")
        else:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(payload)
            print(f"wrote {locale}/assets/search-index.json "
                  f"({len(records)} pages, {len(payload)//1024}KB)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
