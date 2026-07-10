#!/usr/bin/env python3
"""
add_head_metadata.py — Idempotent inserter of SEO head metadata (canonical
URL, hreflang alternates, Open Graph tags) across every tracked HTML page in
the static site (knowledge/, private/, endowed/).

Modeled on scripts/normalize_cdn_loading.py's conventions: pure regex-based
head editing (no BeautifulSoup — that would reformat all ~1,600 files),
per-category counters, dry-run-by-default with an explicit --write flag.

Target file set mirrors scripts/build_sitemap.py: `git ls-files '*.html'`,
excluding any path with an 'archive' path segment and excluding
knowledge/{jp,en}/search.html (noindex, excluded from the sitemap too).

Insertion point: a single stable point, immediately after the closing
</title> tag, matching indentation already used for hand-authored canonical/
hreflang/OG blocks elsewhere in this repo (e.g. knowledge/jp/index.html).

Tags added (idempotent — each *kind* is only added if no tag of that kind
already exists anywhere in the file; existing tags are left untouched):

  1. canonical:
       <link rel="canonical" href="BASE+url-encoded-relative-path">
     Skipped (counted) if any `rel="canonical"` link is already present.

  2. hreflang — ONLY for files under knowledge/(jp|en)/... whose counterpart
     file (knowledge/<other-locale>/<same-rest-of-path>) exists on disk:
       <link rel="alternate" hreflang="ja" href="...knowledge/jp/X">
       <link rel="alternate" hreflang="en" href="...knowledge/en/X">
       <link rel="alternate" hreflang="x-default" href="...knowledge/en/X">
     Skipped (counted, "no counterpart") if the sibling-locale file is
     missing. Skipped (counted, "existing") if any hreflang link already
     exists in the file. Not attempted at all for non-knowledge/{jp,en}
     pages (counted "not applicable").

  3. Open Graph:
       og:title       - <title> text, HTML-attribute-escaped
       og:type        - "article" if the filename starts with "chapter",
                         else "website"
       og:url         - same value as the canonical href
       og:site_name   - "AI Terakoya" (all pages, for simplicity)
       og:locale      - "ja_JP" if the path has a "jp" segment or
                         <html lang="ja">, else "en_US"
       og:description - only if the page already has
                         <meta name="description" content="...">; reuses
                         that content verbatim (no invented descriptions)
     Skipped (counted) if any `property="og:...` meta already exists.

Files with no <head> or no <title> are skipped and counted/reported.

Usage:
    python3 scripts/add_head_metadata.py             # dry-run summary
    python3 scripts/add_head_metadata.py --write      # apply changes
    python3 scripts/add_head_metadata.py --check      # exit 1 if changes pending
"""
from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import sys
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # wp/
BASE_URL = "https://yusukehashimotolab.github.io/AI-Knowledge-Notes/"
LOCALES = ("jp", "en")

# --------------------------------------------------------------------------- #
# File collection (mirrors scripts/build_sitemap.py)
# --------------------------------------------------------------------------- #
def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout


def tracked_html_files() -> list[str]:
    out = _git(["ls-files", "*.html"])
    return sorted(line for line in out.splitlines() if line)


def is_excluded(relpath: str) -> bool:
    parts = relpath.split("/")
    if "archive" in parts:
        return True
    if os.path.basename(relpath) == "search.html":
        return True
    return False


def url_for(relpath: str) -> str:
    return BASE_URL + "/".join(quote(seg) for seg in relpath.split("/"))


# --------------------------------------------------------------------------- #
# Regexes
# --------------------------------------------------------------------------- #
HEAD_RE = re.compile(r"<head\b", re.I)
TITLE_RE = re.compile(r"([ \t]*)<title\b[^>]*>(.*?)</title>", re.S | re.I)
CANONICAL_RE = re.compile(r'rel=["\']canonical["\']', re.I)
HREFLANG_RE = re.compile(r"\bhreflang=", re.I)
OG_RE = re.compile(r'property=["\']og:', re.I)
DESC_TAG_RE = re.compile(r'<meta\b[^>]*\bname=["\']description["\'][^>]*>', re.I)
CONTENT_ATTR_RE = re.compile(r'content=(["\'])(.*?)\1', re.S)
HTML_LANG_JA_RE = re.compile(r'<html\b[^>]*\blang=["\']ja["\']', re.I)


def esc_attr(text: str) -> str:
    """Normalize then HTML-attribute-escape (avoids double-escaping titles
    that already contain entities like '&amp;' while also fixing raw '&')."""
    return html.escape(html.unescape(text), quote=True)


# --------------------------------------------------------------------------- #
# Core per-file transform
# --------------------------------------------------------------------------- #
def process(relpath: str, text: str, counts: dict) -> tuple[str, bool]:
    if not HEAD_RE.search(text):
        counts["skipped_no_head"] += 1
        return text, False

    m = TITLE_RE.search(text)
    if not m:
        counts["skipped_no_title"] += 1
        return text, False

    indent = m.group(1)
    title_text_raw = m.group(2)
    insert_pos = m.end()

    parts = relpath.split("/")
    canonical_url = url_for(relpath)
    pieces: list[str] = []

    # --- 1. canonical --------------------------------------------------- #
    if CANONICAL_RE.search(text):
        counts["canonical_skipped_existing"] += 1
    else:
        pieces.append(f'<link rel="canonical" href="{canonical_url}">')
        counts["canonical_added"] += 1

    # --- 2. hreflang ------------------------------------------------------ #
    is_kb_locale_page = (
        len(parts) >= 3 and parts[0] == "knowledge" and parts[1] in LOCALES
    )
    if is_kb_locale_page:
        if HREFLANG_RE.search(text):
            counts["hreflang_skipped_existing"] += 1
        else:
            other_locale = "en" if parts[1] == "jp" else "jp"
            other_relpath = "/".join(["knowledge", other_locale] + parts[2:])
            if os.path.isfile(os.path.join(ROOT, other_relpath)):
                jp_relpath = "/".join(["knowledge", "jp"] + parts[2:])
                en_relpath = "/".join(["knowledge", "en"] + parts[2:])
                jp_url = url_for(jp_relpath)
                en_url = url_for(en_relpath)
                pieces.append(f'<link rel="alternate" hreflang="ja" href="{jp_url}">')
                pieces.append(f'<link rel="alternate" hreflang="en" href="{en_url}">')
                pieces.append(f'<link rel="alternate" hreflang="x-default" href="{en_url}">')
                counts["hreflang_added"] += 1
            else:
                counts["hreflang_skipped_no_counterpart"] += 1
    else:
        counts["hreflang_not_applicable"] += 1

    # --- 3. Open Graph ------------------------------------------------- #
    if OG_RE.search(text):
        counts["og_skipped_existing"] += 1
    else:
        og_title = esc_attr(title_text_raw)
        og_type = "article" if os.path.basename(relpath).startswith("chapter") else "website"
        is_jp = ("jp" in parts) or bool(HTML_LANG_JA_RE.search(text))
        og_locale = "ja_JP" if is_jp else "en_US"

        pieces.append(f'<meta property="og:title" content="{og_title}">')
        pieces.append(f'<meta property="og:type" content="{og_type}">')
        pieces.append(f'<meta property="og:url" content="{canonical_url}">')
        pieces.append('<meta property="og:site_name" content="AI Terakoya">')
        pieces.append(f'<meta property="og:locale" content="{og_locale}">')

        dm = DESC_TAG_RE.search(text)
        if dm:
            cm = CONTENT_ATTR_RE.search(dm.group(0))
            if cm:
                pieces.append(f'<meta property="og:description" content="{cm.group(2)}">')
                counts["og_description_included"] += 1
        counts["og_added"] += 1

    if not pieces:
        return text, False

    block = "\n" + "\n".join(indent + p for p in pieces)
    new_text = text[:insert_pos] + block + text[insert_pos:]
    return new_text, True


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
COUNTER_KEYS = [
    "canonical_added", "canonical_skipped_existing",
    "hreflang_added", "hreflang_skipped_existing",
    "hreflang_skipped_no_counterpart", "hreflang_not_applicable",
    "og_added", "og_description_included", "og_skipped_existing",
    "skipped_no_head", "skipped_no_title",
]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="apply changes to disk")
    ap.add_argument("--check", action="store_true",
                     help="exit 1 if any file would change; writes nothing")
    ap.add_argument("paths", nargs="*", help="limit to given files (default: all tracked *.html)")
    args = ap.parse_args(argv)

    if args.write and args.check:
        print("error: --write and --check are mutually exclusive", file=sys.stderr)
        return 2

    all_files = tracked_html_files()
    if args.paths:
        wanted = {os.path.relpath(os.path.abspath(p), ROOT) for p in args.paths}
        files = [f for f in all_files if f in wanted]
    else:
        files = [f for f in all_files if not is_excluded(f)]

    counts = {k: 0 for k in COUNTER_KEYS}
    files_modified: list[str] = []

    for relpath in files:
        abspath = os.path.join(ROOT, relpath)
        with open(abspath, encoding="utf-8") as fh:
            orig = fh.read()
        new_text, changed = process(relpath, orig, counts)
        if changed:
            files_modified.append(relpath)
            if args.write:
                with open(abspath, "w", encoding="utf-8") as fh:
                    fh.write(new_text)

    mode = "WRITE" if args.write else ("CHECK" if args.check else "DRY RUN")
    print("=" * 60)
    print(f"add_head_metadata ({mode})")
    print("=" * 60)
    print(f"Files scanned : {len(files)}")
    print(f"Files modified: {len(files_modified)}")
    print("-" * 60)
    for k in COUNTER_KEYS:
        print(f"{k:<36} {counts[k]:>8}")
    print("-" * 60)

    if args.check:
        if files_modified:
            print(f"STALE: {len(files_modified)} file(s) would change:")
            for f in files_modified[:30]:
                print(f"  {f}")
            if len(files_modified) > 30:
                print(f"  ... ({len(files_modified) - 30} more)")
            return 1
        print("ok: no pending changes")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
