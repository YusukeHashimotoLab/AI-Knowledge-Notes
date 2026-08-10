#!/usr/bin/env python3
"""
normalize_cdn_loading.py — Idempotent normalizer for CDN resource loading across
the static site (wp/knowledge, wp/profile, wp/endowed).

Transformations (all idempotent — a second run reports 0 modifications):
  1. Pin exact library versions on cdn.jsdelivr.net URLs
       mathjax@3            -> mathjax@3.2.2
       mermaid@10           -> mermaid@10.9.3   (latest 10.x on the CDN)
       mermaid (unpinned)   -> mermaid@10.9.3
     (prism 1.29.0, katex 0.16.x and cdnjs mermaid 10.6.1 are already exact.)
  2. Add Subresource Integrity (integrity="sha384-..." crossorigin="anonymous")
     to every <script src>/<link href> that points at a known final CDN URL.
     Module `import` URLs (mermaid.esm.min.mjs) cannot take an HTML integrity
     attribute and are version-pinned only.
  3. MathJax config ordering: when an inline `(window.)?MathJax = {...}` config
     block appears AFTER the async loader tag, swap them so config precedes the
     loader (byte-preserving swap). Combined blocks (containing mermaid) skipped.
  4. defer on Mermaid: add `defer` only to pages whose Mermaid loader has NO
     dependent inline `mermaid.initialize` / `import mermaid` call. Pages that
     call initialize inline are left synchronous (adding defer would break them).
  5. Remove dead loads (CONSERVATIVE — when in doubt, keep):
       - Mermaid loader + pure init block removed from pages with no
         `class="mermaid"` diagram. Reverted if any mermaid JS usage would remain.
       - MathJax loader + pure config block removed from pages with no math.
         Math detection strips <pre>/<code>, then looks for \( \) \[ \] $$
         \begin{ , a math/tex script, class="MathJax"/mjx-, or single-$ math.
  6. viewport: insert
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
     right after <meta charset ...> on <head> pages that lack a viewport meta.

Usage:
    python3 scripts/normalize_cdn_loading.py            # apply to knowledge/profile/endowed
    python3 scripts/normalize_cdn_loading.py --dry-run  # report only, write nothing
    python3 scripts/normalize_cdn_loading.py path ...   # limit to given files/dirs
"""
from __future__ import annotations
import os
import re
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # wp/
DEFAULT_DIRS = ["knowledge", "profile", "endowed"]

# --- Subresource Integrity map: final CDN URL -> sha384 hash -----------------
# Computed with: curl -sL <url> | openssl dgst -sha384 -binary | openssl base64 -A
SRI = {
    "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js":
        "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei",
    "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js":
        "sha384-KKWa9jJ1MZvssLeOoXG6FiOAZfAgmzsIIfw8BXwI9+kYm0lPCbC6yTQPBC00F1/L",
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js":
        "sha384-M5jmNxKC9EVnuqeMwRHvFuYUE8Hhp0TgBruj/GZRkYtiMrCRgH7yvv5KY+Owi7TW",
    "https://cdn.jsdelivr.net/npm/mermaid@10.9.3/dist/mermaid.min.js":
        "sha384-R63zfMfSwJF4xCR11wXii+QUsbiBIdiDzDbtxia72oGWfkT7WHJfmD/I/eeHPJyT",
    "https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.6.1/mermaid.min.js":
        "sha384-+NGfjU8KzpDLXRHduEqW+ZiJr2rIg+cidUVk7B51R5xK7cHwMKQfrdFwGdrq1Bcz",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js":
        "sha384-06z5D//U/xpvxZHuUz92xBvq3DqBBFi7Up53HRrbV7Jlv7Yvh/MZ7oenfUe9iCEt",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js":
        "sha384-WJdEkJKrbsqw0evQ4GB6mlsKe5cGTxBOw4KAEIa52ZLB7DDpliGkwdme/HMa5n1m",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js":
        "sha384-9WmlN8ABpoFSSHvBGGjhvB3E/D8UkNB9HpLJjBQFC2VSQsM1odiQDv4NbEo+7l15",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js":
        "sha384-RhrmFFMb0ZCHImjFMpR/UE3VEtIVTCtNrtKQqXCzqXZNJala02N3UbVhi+qzw3CY",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js":
        "sha384-MXybTpajaBV0AkcBaCPT4KIvo0FzoCiWXgcihYsw4FUkEz0Pv3JGV6tk2G8vJtDc",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css":
        "sha384-wFjoQjtV1y5jVHbt0p35Ui8aV8GVpEZkyF99OXWqP/eNJDU93D3Ugxkoyh6Y2I4A",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css":
        "sha384-rCCjoCPCsizaAAYVoz1Q0CmCTvnctK0JkfCSjx7IIxexTBg+uCKtFYycedUjMyA2",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-okaidia.min.css":
        "sha384-qTzu9jz8wpyzFe5KLoZfw0CS5iY+kCoZlBd5ByJ3f0NUT9dgCIU19M1IQKj594Ei",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js":
        "sha384-Uq05+JLko69eOiPr39ta9bh7kld5PKZoU+fF7g0EXTAriEollhZ+DrN8Q/Oi8J2Q",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.js":
        "sha384-6QJu8apxMmB9TiPVWzYKF5pRgKcz7snO0/QU+MrWmgBLECQjoa6erxX2VQ5t41Jd",
    "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.css":
        "sha384-nUkTNLI8COlMCRJ0FHIdX76If83145OTCLUx4gQyfnO0gGeO/sD9czGEUBxtkcUv",
    "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css":
        "sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV",
    "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js":
        "sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8",
    "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js":
        "sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05",
    "https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css":
        "sha384-vKruj+a13U8yHIkAyGgK1J3ArTLzrFGBbBc0tDp4ad/EyewESeXE/Iv67Aj8gKZ0",
    "https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.js":
        "sha384-PwRUT/YqbnEjkZO0zZxNqcxACrXe+j766U2amXcgMg5457rve2Y7I6ZJSm2A0mS4",
    "https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/contrib/auto-render.min.js":
        "sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05",
    "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css":
        "sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn",
    "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js":
        "sha384-cpW21h6RZv/phavutF+AuVYrr+dA8xD9zs6FwLpaCct6O9ctzYFfFr4dgmgccOTx",
    "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js":
        "sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05",
}

# --------------------------------------------------------------------------- #
# Transformation 1: version pinning
# --------------------------------------------------------------------------- #
def pin_versions(s: str) -> tuple[str, int]:
    n = 0
    s, c = re.subn(r'(https://cdn\.jsdelivr\.net/npm/mathjax@)3(/es5/)', r'\g<1>3.2.2\g<2>', s); n += c
    s, c = re.subn(r'(https://cdn\.jsdelivr\.net/npm/mermaid@)10(/dist/)', r'\g<1>10.9.3\g<2>', s); n += c
    s, c = re.subn(r'(https://cdn\.jsdelivr\.net/npm/mermaid)(/dist/)', r'\g<1>@10.9.3\g<2>', s); n += c
    return s, n


# --------------------------------------------------------------------------- #
# Math / diagram content detection (conservative)
# --------------------------------------------------------------------------- #
PRE_RE = re.compile(r'<pre\b.*?</pre>', re.I | re.S)
CODE_RE = re.compile(r'<code\b.*?</code>', re.I | re.S)
MATH_MARKERS = [r'\(', r'\)', r'\[', r'\]', '$$', r'\begin{']
DOLLAR_RE = re.compile(r'\$[^$\n]{1,80}\$')
MATHISH_RE = re.compile(r'[\\^_=]|\\[a-zA-Z]+|[a-zA-Z]\^|[a-zA-Z]_')
MERMAID_CLASS_RE = re.compile(r'class="[^"]*\bmermaid\b', re.I)


def _strip_code(s: str) -> str:
    return CODE_RE.sub('', PRE_RE.sub('', s))


def math_needed(s: str) -> bool:
    if re.search(r'type="math/tex"', s) or 'class="MathJax"' in s or 'mjx-' in s:
        return True
    stripped = _strip_code(s)
    if any(m in stripped for m in MATH_MARKERS):
        return True
    for frag in DOLLAR_RE.findall(stripped):
        if MATHISH_RE.search(frag):
            return True
    return False


def diagram_needed(s: str) -> bool:
    return bool(MERMAID_CLASS_RE.search(s))


# --------------------------------------------------------------------------- #
# Tag regexes
# --------------------------------------------------------------------------- #
MERMAID_SRC_TAG = re.compile(
    r'[ \t]*<script\b[^>]*\bsrc="[^"]*mermaid[^"]*\.min\.js"[^>]*>\s*</script>\n?', re.I)
MERMAID_MODULE = re.compile(
    r'[ \t]*<script\b[^>]*\btype="module"[^>]*>(?:(?!</script>).)*?import\s+mermaid'
    r'(?:(?!</script>).)*?</script>\n?', re.I | re.S)
MERMAID_INIT_BLOCK = re.compile(
    r'[ \t]*<script>\s*(?://[^\n]*\n\s*)?mermaid\.initialize\s*\([^<]*?\)\s*;?\s*</script>\n?',
    re.I | re.S)
MERMAID_JS_USAGE = re.compile(r'mermaid\.(?:initialize|init|run|render)\b|import\s+mermaid', re.I)

MATHJAX_LOADER_TAG = re.compile(
    r'[ \t]*<script\b[^>]*\bsrc="[^"]*(?:tex-mml-chtml|tex-svg|/mathjax[@/])[^"]*"[^>]*>\s*</script>\n?',
    re.I)
# Inline <script> ... </script> with NO attributes (exact `<script>` open tag)
INLINE_SCRIPT = re.compile(r'[ \t]*<script>((?:(?!</script>).)*?)</script>\n?', re.S)
MATHJAX_CONFIG_BODY = re.compile(r'(?:window\.)?MathJax\s*=\s*\{')

SCRIPT_SRC_TAG = re.compile(r'<script\b[^>]*\bsrc="[^"]*"[^>]*>', re.I)
LINK_TAG = re.compile(r'<link\b[^>]*>', re.I)


# --------------------------------------------------------------------------- #
# Transformation 5a: remove dead mermaid
# --------------------------------------------------------------------------- #
def remove_dead_mermaid(s: str) -> tuple[str, int]:
    has_src = MERMAID_SRC_TAG.search(s) or MERMAID_MODULE.search(s)
    if not has_src or diagram_needed(s):
        return s, 0
    new = MERMAID_SRC_TAG.sub('', s)
    # remove pure module mermaid scripts (import + optional initialize only)
    def _mod(m):
        body = m.group(0)
        # allow only mermaid import + initialize/run inside
        inner = re.sub(r'<[^>]+>', '', body)
        if MATHJAX_CONFIG_BODY.search(inner) or 'renderMathInElement' in inner:
            return body  # keep — combined
        return ''
    new = MERMAID_MODULE.sub(_mod, new)
    # remove pure inline mermaid.initialize blocks
    def _init(m):
        body = m.group(0)
        if MATHJAX_CONFIG_BODY.search(body) or 'renderMathInElement' in body:
            return body  # combined -> keep
        return ''
    new = MERMAID_INIT_BLOCK.sub(_init, new)
    # safety: if any mermaid JS usage would remain, revert entirely
    if MERMAID_JS_USAGE.search(new) or MERMAID_SRC_TAG.search(new):
        return s, 0
    if new == s:
        return s, 0
    return new, 1


# --------------------------------------------------------------------------- #
# Transformation 5b: remove dead mathjax
# --------------------------------------------------------------------------- #
def remove_dead_mathjax(s: str) -> tuple[str, int]:
    if not MATHJAX_LOADER_TAG.search(s) or math_needed(s):
        return s, 0
    new = MATHJAX_LOADER_TAG.sub('', s)
    # remove pure inline MathJax config blocks (config-only)
    def _cfg(m):
        body = m.group(1)
        if not MATHJAX_CONFIG_BODY.search(body):
            return m.group(0)
        if 'mermaid' in body or 'renderMathInElement' in body:
            return m.group(0)  # combined -> keep
        return ''
    new = INLINE_SCRIPT.sub(_cfg, new)
    if new == s:
        return s, 0
    return new, 1


# --------------------------------------------------------------------------- #
# Transformation 3: MathJax config-before-loader ordering
# --------------------------------------------------------------------------- #
LOADER_ONLY_TAG = re.compile(
    r'<script\b[^>]*\bsrc="[^"]*(?:tex-mml-chtml|tex-svg|/mathjax[@/])[^"]*"[^>]*>\s*</script>',
    re.I)
# Tag-only (no surrounding whitespace) inline config block
CONFIG_TAG = re.compile(r'<script>(?:(?!</script>).)*?</script>', re.S)


def reorder_mathjax(s: str) -> tuple[str, int]:
    lm = LOADER_ONLY_TAG.search(s)
    if not lm:
        return s, 0
    ls, le = lm.span()
    # find inline config block located AFTER the loader
    cfg = None
    for m in CONFIG_TAG.finditer(s):
        body = m.group(0)
        if not MATHJAX_CONFIG_BODY.search(body):
            continue
        if 'mermaid' in body:  # combined -> unsafe to move before mermaid usage
            continue
        if m.start() >= le:
            cfg = m
            break
        else:
            return s, 0  # earliest config already before loader -> ok
    if cfg is None:
        return s, 0
    cs, ce = cfg.span()
    # swap the two tags in place, preserving all surrounding whitespace:
    #   prefix + configTag + (middle+indentC) + loaderTag + suffix
    new = s[:ls] + s[cs:ce] + s[le:cs] + s[ls:le] + s[ce:]
    return new, 1


# --------------------------------------------------------------------------- #
# Transformation 2: SRI injection
# --------------------------------------------------------------------------- #
def _inject(tag: str) -> str:
    if 'integrity=' in tag:
        return tag
    m = re.search(r'(?:src|href)="([^"]+)"', tag)
    if not m or m.group(1) not in SRI:
        return tag
    h = SRI[m.group(1)]
    inner = tag[:-1].rstrip()          # drop trailing '>'
    selfclose = inner.endswith('/')
    if selfclose:
        inner = inner[:-1].rstrip()
    add = f' integrity="{h}" crossorigin="anonymous"'
    return inner + add + (' />' if selfclose else '>')


def add_sri(s: str) -> tuple[str, int]:
    n = 0

    def repl(m):
        nonlocal n
        out = _inject(m.group(0))
        if out != m.group(0):
            n += 1
        return out

    s = SCRIPT_SRC_TAG.sub(repl, s)
    s = LINK_TAG.sub(repl, s)
    return s, n


# --------------------------------------------------------------------------- #
# Transformation 4: defer on Mermaid loaders with no inline init dependency
# --------------------------------------------------------------------------- #
DEFER_MERMAID_TAG = re.compile(
    r'<script\b(?![^>]*\b(?:defer|async)\b)([^>]*\bsrc="[^"]*mermaid[^"]*\.min\.js"[^>]*)>', re.I)


def defer_mermaid(s: str) -> tuple[str, int]:
    if MERMAID_JS_USAGE.search(s):
        return s, 0  # inline dependency -> keep synchronous
    n = 0

    def repl(m):
        nonlocal n
        n += 1
        return '<script defer' + m.group(1) + '>'

    s = DEFER_MERMAID_TAG.sub(repl, s)
    return s, n


# --------------------------------------------------------------------------- #
# Transformation 6: viewport meta
# --------------------------------------------------------------------------- #
CHARSET_META = re.compile(r'([ \t]*)(<meta\b[^>]*\bcharset=[^>]*>)', re.I)
VIEWPORT = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'


def add_viewport(s: str) -> tuple[str, int]:
    if '<head' not in s.lower() or 'name="viewport"' in s:
        return s, 0
    m = CHARSET_META.search(s)
    if not m:
        return s, 0
    indent = m.group(1)
    insert = m.group(0) + '\n' + indent + VIEWPORT
    new = s[:m.start()] + insert + s[m.end():]
    return new, 1


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
STAGES = [
    ("pin_versions", pin_versions),
    ("remove_dead_mermaid", remove_dead_mermaid),
    ("remove_dead_mathjax", remove_dead_mathjax),
    ("reorder_mathjax", reorder_mathjax),
    ("add_sri", add_sri),
    ("defer_mermaid", defer_mermaid),
    ("add_viewport", add_viewport),
]


def collect_files(args):
    files = []
    targets = args or [os.path.join(ROOT, d) for d in DEFAULT_DIRS]
    for t in targets:
        if os.path.isdir(t):
            files += glob.glob(os.path.join(t, '**', '*.html'), recursive=True)
        elif t.endswith('.html') and os.path.isfile(t):
            files.append(t)
    return sorted(set(files))


def main(argv):
    dry = '--dry-run' in argv
    paths = [a for a in argv if not a.startswith('--')]
    files = collect_files(paths)

    counts = {name: 0 for name, _ in STAGES}
    files_modified = 0
    edits = {}  # file -> number of stage-hits (for "most edited" ranking)

    for f in files:
        with open(f, encoding='utf-8') as fh:
            orig = fh.read()
        s = orig
        hits = 0
        for name, fn in STAGES:
            s, c = fn(s)
            if c:
                counts[name] += c
                hits += 1
        if s != orig:
            files_modified += 1
            edits[f] = hits
            if not dry:
                with open(f, 'w', encoding='utf-8') as fh:
                    fh.write(s)

    # summary
    print("=" * 60)
    print(f"CDN loading normalization {'(DRY RUN)' if dry else ''}")
    print("=" * 60)
    print(f"Files scanned : {len(files)}")
    print(f"Files modified: {files_modified}")
    print("-" * 60)
    print(f"{'transformation':<24} {'occurrences':>12}")
    print("-" * 60)
    for name, _ in STAGES:
        print(f"{name:<24} {counts[name]:>12}")
    print("-" * 60)
    top = sorted(edits.items(), key=lambda kv: kv[1], reverse=True)[:20]
    if top:
        print("Top modified files (by #stages touched):")
        for f, h in top:
            print(f"  {h}  {os.path.relpath(f, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
