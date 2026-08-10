#!/usr/bin/env python3
"""Generate wp/sitemap.xml for the AI Terakoya knowledge base.

Source of truth for "what pages exist" is `git ls-files '*.html'` (not a
filesystem walk) — this is the simplest robust way to cover every tracked
HTML page across knowledge/{jp,en}, profile/{jp,en}, and endowed/{jp,en}/**
in one shot without having to special-case each tree's layout.

Excluded:
  - knowledge/<locale>/search.html   (the search pages are noindex)
  - any path with an 'archive' path segment
  - anything under private/  (legacy path of the profile/ tree; those files are
    now noindex meta-refresh redirect stubs kept only so old URLs keep working)

Each URL is https://yusukehashimotolab.github.io/AI-Knowledge-Notes/<path>,
with every path segment individually percent-encoded via urllib.parse.quote
(defensive: no non-ASCII/space filenames exist in this repo today, but the
knowledge base is actively authored and that could change).

<lastmod> source: a single `git log -z --format=C|%cI --name-only` walk over
the *entire* history builds a path -> latest-commit-date map in one process
(~2s for this repo's ~280 commits), instead of one `git log -1` subprocess
per file (~35s measured for 1600 files). Only each file's first (i.e.
newest, since git log defaults to newest-first) appearance is kept.

--check and shallow clones: CI checkouts here use the default `actions/
checkout` depth (fetch-depth: 1, i.e. shallow — no lastmod history
available). Rather than force a full-history checkout just for this check
(slower clone on every push) or skip the sitemap check in CI entirely,
--check auto-detects a shallow repository via
`git rev-parse --is-shallow-repository` and, when shallow, degrades to
comparing only the <loc> URL set between the on-disk sitemap and a freshly
generated one — ignoring <lastmod>, which is expected to be unreliable
there (git log on a depth-1 clone only sees the single fetched commit, so
every file would appear "last modified" on that commit's date). On a full
(unshallow) clone or local run, --check instead compares the file
byte-for-byte, same convention as build_search_index.py --check.

Usage:
  python3 scripts/build_sitemap.py          # write sitemap.xml
  python3 scripts/build_sitemap.py --check  # exit 1 if on-disk file is stale
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
from urllib.parse import quote
from xml.sax.saxutils import escape

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SITEMAP_PATH = os.path.join(REPO_ROOT, "sitemap.xml")
BASE_URL = "https://yusukehashimotolab.github.io/AI-Knowledge-Notes/"
LOCALES = ("jp", "en")

HEADER_RE = re.compile(r"^C\|(.+)$")


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout


def tracked_html_files() -> list[str]:
    """All tracked *.html paths (repo-relative, POSIX separators), sorted."""
    out = _git(["ls-files", "*.html"])
    return sorted(line for line in out.splitlines() if line)


def is_excluded(relpath: str) -> bool:
    parts = relpath.split("/")
    if "archive" in parts:
        return True
    if relpath == "404.html":  # noindex error page
        return True
    # private/ is the legacy path of the profile/ tree; every file there is a
    # noindex meta-refresh redirect stub, so it must never enter the sitemap.
    if parts[0] == "private":
        return True
    if len(parts) == 3 and parts[0] == "knowledge" and parts[1] in LOCALES \
            and parts[2] == "search.html":
        return True
    return False


def lastmod_map() -> dict[str, str]:
    """path -> ISO-8601 commit date of its most recent commit (one git-log walk)."""
    raw = subprocess.run(
        ["git", "log", "-z", "--format=C|%cI", "--name-only"],
        cwd=REPO_ROOT, check=True, capture_output=True,
    ).stdout
    mapping: dict[str, str] = {}
    current_date = None
    for field in raw.split(b"\x00"):
        text = field.decode("utf-8")
        if text.startswith("\n"):
            text = text[1:]
        if not text:
            continue
        m = HEADER_RE.match(text)
        if m:
            current_date = m.group(1)
            continue
        if text not in mapping:
            mapping[text] = current_date
    return mapping


def url_for(relpath: str) -> str:
    segments = relpath.split("/")
    return BASE_URL + "/".join(quote(seg) for seg in segments)


def is_shallow_repo() -> bool:
    out = _git(["rev-parse", "--is-shallow-repository"]).strip()
    return out == "true"


def build_entries() -> list[tuple[str, str | None]]:
    lastmods = lastmod_map()
    entries = []
    for relpath in tracked_html_files():
        if is_excluded(relpath):
            continue
        entries.append((url_for(relpath), lastmods.get(relpath)))
    return entries


def render(entries: list[tuple[str, str | None]]) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for url, lastmod in entries:
        lines.append("  <url>")
        lines.append(f"    <loc>{escape(url)}</loc>")
        if lastmod:
            lines.append(f"    <lastmod>{escape(lastmod)}</lastmod>")
        lines.append("  </url>")
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


def extract_locs(xml_text: str) -> set[str]:
    return set(re.findall(r"<loc>(.*?)</loc>", xml_text))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                     help="verify the on-disk sitemap matches; exit 1 if stale")
    args = ap.parse_args()

    entries = build_entries()
    payload = render(entries)

    if not args.check:
        with open(SITEMAP_PATH, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"wrote sitemap.xml ({len(entries)} urls)")
        return 0

    try:
        current = open(SITEMAP_PATH, encoding="utf-8").read()
    except FileNotFoundError:
        current = ""

    if is_shallow_repo():
        cur_urls, new_urls = extract_locs(current), extract_locs(payload)
        if cur_urls != new_urls:
            added = sorted(new_urls - cur_urls)
            removed = sorted(cur_urls - new_urls)
            print("STALE sitemap.xml (shallow clone: URL-set comparison only, "
                  "lastmod not checked)")
            for u in added[:20]:
                print(f"  + {u}")
            for u in removed[:20]:
                print(f"  - {u}")
            print("regenerate with: python3 scripts/build_sitemap.py")
            return 1
        print(f"ok    sitemap.xml ({len(entries)} urls, shallow clone: "
              f"URL-set comparison only)")
        return 0

    if current != payload:
        diff = list(difflib.unified_diff(
            current.splitlines(), payload.splitlines(),
            fromfile="sitemap.xml (on disk)", tofile="sitemap.xml (regenerated)",
            lineterm="", n=1,
        ))
        print("STALE sitemap.xml (regenerate with: python3 scripts/build_sitemap.py)")
        for line in diff[:20]:
            print(f"  {line}")
        if len(diff) > 20:
            print(f"  ... ({len(diff) - 20} more diff lines)")
        return 1

    print(f"ok    sitemap.xml ({len(entries)} urls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
