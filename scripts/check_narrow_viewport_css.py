#!/usr/bin/env python3
"""Check that every copy of the narrow-viewport CSS block and of its script agrees.

knowledge-base.css (en and jp) carries section 19, "Narrow-viewport overflow
guards": one ``@media screen and (max-width: 768px)`` block. Knowledge pages that
link neither knowledge-base.css nor dojo.css keep all their CSS in an inline
``<style>`` element, so each of them carries a copy of that block (marked by a
comment containing the same title). A change made to one copy and not the
others would leave some pages behind; this script finds that.

Checked:
  - both locale sheets contain the section, and its text is identical in both;
  - every inline-style page carries exactly one copy in a ``<style>`` element in
    its ``<head>``, with the same rules as the sheet in the same order
    (indentation and line breaks aside), except that the rules in SHEET_TO_INLINE
    are replaced by their inline counterparts and INLINE_EXTRA is appended;
  - no page that links knowledge-base.css or dojo.css carries a copy;
  - right after the media block, the section (and every copy) has one rule
    for keyboard focus, applying at every width; it must be the same in the
    sheets and in every copy.

Rules are compared after removing comments and normalising whitespace, in
order: selector lists and declarations must match exactly.

The block makes wide content scroll inside its own box; keyboard-scrollers.js
gives each such box a Tab stop. Also checked:
  - knowledge/en/assets/js/keyboard-scrollers.js is the canonical copy (edit
    that one); knowledge/jp/assets/js/ holds an identical copy;
  - every knowledge page loads its own locale's copy exactly once, with
    ``<script defer src="...">`` right before ``</body>`` and a relative src
    that reaches the file;
  - tools/convert_md_to_html.py emits that tag in its page template.

Usage:
    python3 scripts/check_narrow_viewport_css.py            # report
    python3 scripts/check_narrow_viewport_css.py --check    # exit 1 on drift
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SHEETS = ("knowledge/en/assets/css/knowledge-base.css",
          "knowledge/jp/assets/css/knowledge-base.css")
TITLE = "Narrow-viewport overflow guards"
MEDIA = "@media screen and (max-width: 768px)"

# Rules of the sheet's block that the inline copies state differently: they
# depend on values the sheet sets elsewhere and the inline-style pages set
# themselves. {sheet rule: inline rule}
SHEET_TO_INLINE: dict[str, str] = {
    # the table margins: section 7 of the sheet vs. the table rule most
    # inline-style pages carry (margin-bottom: var(--spacing-md), no top margin)
    ".table-scroll { margin-top: 1.5rem; margin-bottom: var(--spacing-md) }":
        ".table-scroll { margin-top: 0; margin-bottom: var(--spacing-md, 1.5rem) }",
}

# Rules only the inline copies have: the sheet provides them elsewhere (the
# sheet sizes .video-container itself, section 17; inline-style pages do not).
INLINE_EXTRA = [
    ".video-container iframe { max-width: 100%; height: auto; aspect-ratio: 16 / 9 }",
]

JS_NAME = "keyboard-scrollers.js"
JS_CANONICAL = "knowledge/en/assets/js/" + JS_NAME
JS_COPIES = ("knowledge/jp/assets/js/" + JS_NAME,)
CONVERTER = "tools/convert_md_to_html.py"
# the converter's footer template line, followed by the end of the body
CONVERTER_TAG = '<script defer src="{js_src}"></script>\n</body>'

COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
SCRIPT_TAG_RE = re.compile(r"<script\b([^>]*)>\s*</script>", re.I)
SRC_RE = re.compile(r"""\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.I)
DEFER_RE = re.compile(r"(?:^|\s)defer(?:\s|=|$)", re.I)
STYLE_RE = re.compile(r"<style\b[^>]*>(.*?)</style>", re.S | re.I)
SHARED_LINK_RE = re.compile(r"<link\b[^>]*\bhref=[\"'][^\"']*(?:knowledge-base|dojo)\.css[\"']", re.I)


def normalise_rule(selectors: str, body: str) -> str:
    sels = ", ".join(" ".join(s.split()) for s in selectors.split(","))
    decls = []
    for d in body.split(";"):
        d = " ".join(d.split())
        if not d:
            continue
        prop, _, value = d.partition(":")
        decls.append(f"{prop.strip()}: {value.strip()}")
    return f"{sels} {{ {'; '.join(decls)} }}"


def parse_rules(css: str) -> list[str]:
    """Flat list of normalised rules in a block body (no nested at-rules)."""
    css = COMMENT_RE.sub(" ", css)
    rules = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        rules.append(normalise_rule(m.group(1), m.group(2)))
    return rules


def media_block(text: str, start: int) -> tuple[int, int] | None:
    """(start, end) of the first MEDIA block at or after ``start``."""
    i = text.find(MEDIA, start)
    if i < 0:
        return None
    j = text.find("{", i)
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return i, k + 1
        k += 1
    return None


def trailing_rule(text: str, end: int) -> tuple[list[str], int]:
    """The one rule right after a media block ending at ``end`` (comments and
    white space before it allowed): ([normalised rule], end), or ([], end)."""
    masked = COMMENT_RE.sub(lambda m: " " * len(m.group(0)), text)
    m = re.match(r"\s*([^{}@]+)\{([^{}]*)\}", masked[end:])
    if not m:
        return [], end
    return [normalise_rule(m.group(1), m.group(2))], end + m.end()


def sheet_section(text: str) -> tuple[str, list[str], list[str]] | None:
    """Section 19 text (from its banner to the end of the focus rule after its
    media block), the media block's rules and the focus rule."""
    t = text.find(TITLE)
    if t < 0:
        return None
    banner = text.rfind("/*", 0, t)
    blk = media_block(text, t)
    if blk is None:
        return None
    s, e = blk
    inner = text[text.find("{", s) + 1:e - 1]
    focus, end = trailing_rule(text, e)
    return text[banner:end], parse_rules(inner), focus


def inline_copies(head: str) -> list[tuple[list[str], list[str]]]:
    """(media block rules, focus rule) of every copy in the head's <style> elements."""
    copies = []
    for m in STYLE_RE.finditer(head):
        css = m.group(1)
        pos = 0
        while True:
            t = css.find(TITLE, pos)
            if t < 0:
                break
            blk = media_block(css, t)
            if blk is None:
                copies.append(([], []))
                break
            s, e = blk
            focus, end = trailing_rule(css, e)
            copies.append((parse_rules(css[css.find("{", s) + 1:e - 1]), focus))
            pos = end
    return copies


def knowledge_pages(root: str) -> list[str]:
    out = []
    for loc in ("en", "jp"):
        out += glob.glob(os.path.join(root, "knowledge", loc, "**", "*.html"),
                         recursive=True)
    return sorted(out)


def script_problems(rel: str, html: str) -> list[str]:
    """Problems with the page's keyboard-scrollers.js tag ([] when it is right)."""
    masked = HTML_COMMENT_RE.sub(lambda m: " " * len(m.group(0)), html)
    tags = []
    for m in SCRIPT_TAG_RE.finditer(masked):
        src = SRC_RE.search(m.group(1))
        src = next((g for g in src.groups() if g is not None), "") if src else ""
        if src.rsplit("/", 1)[-1] == JS_NAME:
            tags.append((m, src))
    if len(tags) != 1:
        return [f"{rel}: loads {JS_NAME} {len(tags)} times (expected 1)"]
    m, src = tags[0]
    out = []
    locale_js = "/".join(rel.split("/")[:2]) + "/assets/js/" + JS_NAME
    target = os.path.normpath(os.path.join(os.path.dirname(rel), src)).replace(os.sep, "/")
    if "://" in src or src.startswith("/") or target != locale_js:
        out.append(f"{rel}: {JS_NAME} src {src!r} does not reach {locale_js}")
    if not DEFER_RE.search(m.group(1)):
        out.append(f"{rel}: {JS_NAME} is loaded without defer")
    if not re.match(r"\s*</body\s*>", masked[m.end():], re.I):
        out.append(f"{rel}: {JS_NAME} is not loaded right before </body>")
    return out


def check(root: str = REPO_ROOT) -> tuple[list[str], dict]:
    problems: list[str] = []
    sections = {}
    for rel in SHEETS:
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8") as fh:
                sec = sheet_section(fh.read())
        except OSError as exc:
            problems.append(f"{rel}: cannot read ({exc})")
            continue
        if sec is None:
            problems.append(f"{rel}: section '{TITLE}' with {MEDIA} not found")
            continue
        sections[rel] = sec
    stats = {"inline_pages": 0, "inline_ok": 0, "linked_pages": 0, "script_pages": 0, "script_ok": 0}
    try:
        with open(os.path.join(root, JS_CANONICAL), encoding="utf-8") as fh:
            js = fh.read()
        for rel in JS_COPIES:
            try:
                with open(os.path.join(root, rel), encoding="utf-8") as fh:
                    if fh.read() != js:
                        problems.append(f"{rel}: differs from {JS_CANONICAL} (the canonical copy)")
            except OSError as exc:
                problems.append(f"{rel}: cannot read ({exc})")
    except OSError as exc:
        problems.append(f"{JS_CANONICAL}: cannot read ({exc})")
    try:
        with open(os.path.join(root, CONVERTER), encoding="utf-8") as fh:
            if CONVERTER_TAG not in fh.read():
                problems.append(f"{CONVERTER}: page template does not end the body with {CONVERTER_TAG!r}")
    except OSError as exc:
        problems.append(f"{CONVERTER}: cannot read ({exc})")
    for path in knowledge_pages(root):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        with open(path, encoding="utf-8") as fh:
            sp = script_problems(rel, fh.read())
        stats["script_pages"] += 1
        if sp:
            problems += sp
        else:
            stats["script_ok"] += 1
    if len(sections) != len(SHEETS):
        return problems, stats
    (r0, (text0, rules0, focus0)), (r1, (text1, _, _)) = sections.items()
    if text0 != text1:
        problems.append(f"{r1}: section '{TITLE}' differs from {r0}")
    if len(focus0) != 1 or ":focus-visible" not in focus0[0]:
        problems.append(f"{r0}: no :focus-visible rule right after the {MEDIA} block of section '{TITLE}'")
    mapping = {parse_rules(k)[0]: parse_rules(v)[0] for k, v in SHEET_TO_INLINE.items()}
    missing = [r for r in mapping if r not in rules0]
    if missing:
        problems.append(f"{r0}: SHEET_TO_INLINE rules not in the sheet: {missing}")
    expected = [mapping.get(r, r) for r in rules0] + [parse_rules(x)[0] for x in INLINE_EXTRA]
    for path in knowledge_pages(root):
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
        head = html.split("</head>", 1)[0]
        copies = inline_copies(head)
        if SHARED_LINK_RE.search(head):
            stats["linked_pages"] += 1
            if copies:
                problems.append(f"{rel}: links the shared sheet but carries an inline copy")
            continue
        if not STYLE_RE.search(head):
            continue
        stats["inline_pages"] += 1
        if len(copies) != 1:
            problems.append(f"{rel}: {len(copies)} inline copies (expected 1)")
            continue
        rules, focus = copies[0]
        if rules != expected:
            extra = [r for r in rules if r not in expected]
            lack = [r for r in expected if r not in rules]
            detail = f"missing {lack}, unexpected {extra}" if (extra or lack) else "rule order differs"
            problems.append(f"{rel}: inline copy differs from the sheet: {detail}")
            continue
        if focus != focus0:
            problems.append(f"{rel}: the focus rule after the copy differs from the sheet's: {focus} vs {focus0}")
            continue
        stats["inline_ok"] += 1
    return problems, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 on drift")
    parser.add_argument("--root", default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    problems, stats = check(args.root)
    print(f"shared sheets: {len(SHEETS)}; inline-style pages: {stats['inline_pages']} "
          f"({stats['inline_ok']} in step); pages linking a shared sheet: {stats['linked_pages']}; "
          f"pages loading {JS_NAME}: {stats['script_ok']} of {stats['script_pages']} correctly")
    for p in problems:
        print("  DRIFT", p)
    print(f"Drift: {len(problems)}")
    return 1 if (problems and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
