#!/usr/bin/env python3
"""Visual-regression harness for the AI Terakoya static site.

WHAT IT IS FOR
--------------
A pixel-level safety net for refactors that are supposed to be *visually
invisible* — above all the planned inline-``<style>`` -> shared-stylesheet
migration. It renders the same set of pages from two sources (two git refs, or
the working tree against a ref), screenshots them with headless Chromium, and
proves either "byte-for-pixel identical" or "here is exactly which pages moved,
by how much, and what it looks like".

QUICK START
-----------
Run everything from ``wp/`` (this repo's root is the web root).

    # one-off setup (Python 3.11+; keep the venv outside the repo)
    python3 -m venv /tmp/vr-venv
    /tmp/vr-venv/bin/pip install playwright pillow numpy
    /tmp/vr-venv/bin/python -m playwright install chromium

    # determinism self-check: same ref on both sides must yield 0 diffs
    /tmp/vr-venv/bin/python scripts/visual_regression.py \
        --base HEAD --head HEAD --out /tmp/vr-selfcheck

    # working tree vs. last commit (the everyday refactor check)
    /tmp/vr-venv/bin/python scripts/visual_regression.py \
        --base HEAD --head worktree --out /tmp/vr-report

    # two commits, wider sample, more parallelism
    /tmp/vr-venv/bin/python scripts/visual_regression.py \
        --base main --head refactor/shared-css \
        --pages all --workers 8 --out /tmp/vr-full

Then open ``<out>/index.html``.

CLI
---
``--base REF``          git ref for the "before" side (required).
``--head REF|worktree`` git ref, or the literal ``worktree`` (default) to use
                        the current working tree, uncommitted edits included.
``--pages SPEC``        one of:
                        * a manifest file — one relative page path per line,
                          ``#`` comments allowed (the default,
                          ``scripts/visual_regression_pages.txt``);
                        * a single ``.html`` path;
                        * a glob over tracked pages, with real shell semantics —
                          ``*``/``?`` stop at ``/`` and only ``**`` crosses
                          directories, so ``knowledge/*/index.html`` is the two
                          locale landings while ``knowledge/en/ML/**/index.html``
                          is all 41 ML series indexes;
                        * ``all`` — every tracked non-archive ``*.html`` (1,646).
``--out DIR``           report directory (created; PNGs + ``index.html``).
``--workers N``         concurrent browser contexts (default 4).
``--threshold F``       max tolerated fraction of differing pixels per page
                        (default 0.0005 = 0.05 %). Above it, the page FAILS.
``--pixel-delta N``     per-channel 0-255 delta below which two pixels are
                        considered equal (default 8). Absorbs subpixel AA
                        noise without hiding a real colour change.
``--viewport WxH``      default 1280x2000. Screenshots are always ``full_page``.
``--retries N``         re-shoot + re-diff a mismatching page N more times
                        (default 1). A page is only reported as a diff if the
                        mismatch reproduces on the final attempt.
``--page-timeout SEC``  hard budget for one page on one side (default 180). A
                        page that blows it is reported as an ERROR instead of
                        stalling the run.
``--no-mask-embeds``    stop masking cross-origin ``<iframe>``s. They are masked
                        by default; see determinism point 8 below.
``--block-hosts H,H``   abort all requests to these hosts on both sides. Rehearses
                        a CDN outage so you can confirm the DEGRADED guard fires:
                        ``--block-hosts cdn.jsdelivr.net,cdnjs.cloudflare.com
                        --fail-on-degraded`` must exit 1, not 0.
``--fail-on-degraded``  exit non-zero when a page could not be fully rendered
                        (CDN unreachable, MathJax/Mermaid never ran). Off by
                        default for offline local use; **CI passes it**.
``--list``              print the resolved page list and exit.

EXIT CODES
----------
``0`` all pages within threshold.
``1`` at least one page exceeded the threshold, or an unrecoverable error, or
      (with ``--fail-on-degraded``) at least one page rendered degraded.

HOW DETERMINISM IS ENFORCED
---------------------------
These pages pull MathJax, Mermaid, KaTeX and Prism from public CDNs, so a naive
screenshot differs run to run. Every mitigation below is applied identically to
both sides:

1. **Fixed rendering geometry** — ``device_scale_factor=1``, fixed viewport,
   ``prefers-reduced-motion: reduce``, fixed ``--force-color-profile=srgb`` and
   ``--font-render-hinting=none`` Chromium flags, timezone/locale pinned.
2. **Animation kill switch** — an init-script stylesheet forces
   ``animation``/``transition`` to ``none``, ``caret-color: transparent``,
   ``scroll-behavior: auto`` and hides scrollbars, on every frame and iframe.
3. **Deterministic PRNG** — ``Math.random`` is replaced by a seeded LCG so
   Mermaid's generated element ids are stable between the two sides (they are
   not pixels, but stability keeps DOM-signature settling honest).
4. **Staged readiness barrier**, in order, each with its own timeout:
   ``load`` -> ``networkidle`` -> ``document.fonts.ready`` ->
   ``MathJax.startup.promise`` + ``MathJax.typesetPromise()`` -> KaTeX
   ``.katex`` nodes present -> every ``.mermaid`` div carries
   ``data-processed`` -> Prism ``.token`` spans present.
5. **DOM/geometry settling** — after the barrier, ``scrollHeight`` plus a cheap
   DOM signature must stay unchanged across consecutive polls before the shutter
   opens. This is what actually catches late Mermaid reflows.
6. **Flake quarantine** — a mismatch is re-shot (``--retries``); a diff is only
   believed if it reproduces.
7. **Bounded everything** — every wait, including the MathJax ``evaluate`` (which
   has no implicit Playwright timeout), is raced against a timer, and each page
   gets a hard ``--page-timeout``. 12 pages in this corpus load MathJax twice
   (cdnjs *and* jsdelivr); on those, ``MathJax.startup.promise`` can stay pending
   forever, which without these bounds hangs the whole run.
8. **Cross-origin iframes masked** — a third party's iframe content is invisible
   to every readiness signal we have and keeps changing after our DOM settles
   (this was caught for real: a Google Maps embed's info card had loaded on one
   side and not the other, a reproducible 0.088 % "diff" with no code change).
   Such iframes are painted over with a flat box, so their *position and size*
   are still compared — a CSS change that moves or resizes an embed still fails —
   while the third party's own pixels are excluded. 31 iframes in this corpus:
   28 YouTube, 3 Google Maps. ``--no-mask-embeds`` opts out.

KNOWN CORPUS QUIRKS THAT AFFECT THIS TOOL
-----------------------------------------
* ``private/**`` are ``<meta http-equiv="refresh" content="0; ...">`` redirect
  stubs. The screenshot races the redirect, so they are inherently
  non-deterministic and are excluded from the sample manifest.
* 12 pages load MathJax from both cdnjs and jsdelivr (see
  ``scripts/normalize_cdn_loading.py``). They render, but slowly and with a
  pending startup promise; the MathJax barrier gives up on its budget and the
  DOM-settling loop then decides when to shoot.
* Pages with YouTube/Google Maps ``<iframe>`` embeds fire third-party requests
  that fail routinely. Those are recorded as informational notes, not as
  degradation, because they never move a pixel of our own layout. The iframes
  themselves are masked (point 8 above).

OFFLINE / VACUOUS-PASS PROTECTION
---------------------------------
If a CDN is unreachable, *both* sides fail the same way and a pixel comparison
would pass while proving nothing. So each render records what the page *asked
for* (parsed from the served HTML: ``$$`` math, ``.mermaid`` divs, Prism
classes) against what actually appeared in the DOM (``mjx-container``/``.katex``
counts, ``data-processed`` marks, ``.token`` spans) plus every failed network
request. Any shortfall marks the page ``DEGRADED``; the report shows a banner
and the run warns loudly. Pass ``--fail-on-degraded`` (CI does) to turn that
into a hard failure rather than a green tick that means nothing.

Do not take that on trust — rehearse the outage::

    python scripts/visual_regression.py --base HEAD --head HEAD \
        --pages knowledge/en/MI/gnn-introduction/chapter-2.html \
        --block-hosts cdn.jsdelivr.net,cdnjs.cloudflare.com \
        --fail-on-degraded --out /tmp/vr-outage

The pixels will match perfectly (both sides broke identically) and the run must
still exit 1, reporting "math never typeset" and "mermaid 0/5 diagrams rendered".

DEPENDENCIES
------------
``playwright``, ``pillow``, ``numpy`` (plus ``playwright install chromium``).
Deliberately nothing heavier — no image-diff service, no Node toolchain. These
are *not* in ``requirements.txt``: this tool is opt-in and only CI's
visual-regression job and refactor authors need the browser download.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import functools
import html
import http.server
import io
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    from PIL import Image, ImageDraw

    # The tallest pages in this corpus render past 100 Mpx as full-page PNGs.
    # These files are produced by our own browser, so Pillow's decompression-bomb
    # guard is pure noise here.
    Image.MAX_IMAGE_PIXELS = None
except ImportError as exc:  # pragma: no cover - dependency guard
    sys.exit(
        f"missing dependency: {exc.name}\n"
        "  pip install playwright pillow numpy && python -m playwright install chromium"
    )

try:
    from playwright.async_api import async_playwright, Error as PWError
except ImportError:  # pragma: no cover - dependency guard
    sys.exit(
        "missing dependency: playwright\n"
        "  pip install playwright pillow numpy && python -m playwright install chromium"
    )


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

DEFAULT_MANIFEST = "scripts/visual_regression_pages.txt"

# Chromium's single-shot capture tops out well below this; taller pages are
# captured as viewport-sized strips and stitched by Pillow.
STITCH_ABOVE_PX = 15000
STITCH_CHUNK_PX = 4000

# Readiness-barrier budgets (ms). Generous but bounded: a page that blows
# through one of these is reported DEGRADED, never silently screenshotted.
T_NETWORK_IDLE = 15000
T_FONTS = 5000
T_MATHJAX_LOAD = 20000
T_MATHJAX_TYPESET = 30000
T_KATEX = 15000
T_MERMAID = 25000
T_PRISM = 10000
SETTLE_POLL_MS = 150
SETTLE_STABLE_POLLS = 3
SETTLE_MAX_MS = 8000

CHROMIUM_ARGS = [
    "--force-color-profile=srgb",
    "--font-render-hinting=none",
    "--disable-lcd-text",
    "--disable-skia-runtime-opts",
    "--disable-partial-raster",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-features=PaintHolding,LazyImageLoading,LazyFrameLoading",
    "--hide-scrollbars",
    "--deterministic-mode",
]

# Applied before any page script runs, in every frame.
INIT_SCRIPT = r"""
(() => {
  // Seeded LCG so Mermaid's id generation is identical on both sides.
  let _s = 0x2545F491;
  Math.random = function () {
    _s = (_s * 1103515245 + 12345) & 0x7fffffff;
    return _s / 0x7fffffff;
  };

  const CSS = `
    *, *::before, *::after {
      animation-delay: 0s !important;
      animation-duration: 0s !important;
      animation-iteration-count: 1 !important;
      animation-play-state: paused !important;
      transition-delay: 0s !important;
      transition-duration: 0s !important;
      caret-color: transparent !important;
      scroll-behavior: auto !important;
    }
    html { scroll-behavior: auto !important; }
    ::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }
    /* Blinking cursors / marquees / spinners are the classic flake sources. */
    marquee { -webkit-animation: none !important; animation: none !important; }
  `;
  const install = () => {
    if (!document.documentElement) return;
    if (document.getElementById('__vr_determinism__')) return;
    const st = document.createElement('style');
    st.id = '__vr_determinism__';
    st.textContent = CSS;
    (document.head || document.documentElement).appendChild(st);
  };
  install();
  document.addEventListener('DOMContentLoaded', install);
})();
"""

# Reads the rendered DOM and reports what each renderer actually produced.
PROBE_SCRIPT = r"""
() => {
  const q = (s) => document.querySelectorAll(s).length;
  const mermaidAll = document.querySelectorAll('.mermaid').length;
  const mermaidDone = document.querySelectorAll('.mermaid[data-processed]').length;
  const mermaidSvg = document.querySelectorAll('.mermaid svg').length;
  return {
    mathjax_nodes: q('mjx-container') + q('.MathJax') + q('svg[data-mml-node]'),
    katex_nodes: q('.katex'),
    mermaid_total: mermaidAll,
    mermaid_processed: mermaidDone,
    mermaid_svg: mermaidSvg,
    prism_tokens: q('.token'),
    scroll_height: document.documentElement.scrollHeight,
    scroll_width: document.documentElement.scrollWidth,
    img_total: q('img'),
    img_broken: Array.from(document.images).filter(
      (i) => i.complete && i.naturalWidth === 0).length,
    // Cheap structural signature; changes if anything reflows or is injected.
    dom_sig: document.body ? (document.body.innerHTML.length + ':' +
             document.querySelectorAll('*').length) : '0:0',
  };
}
"""


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #


@dataclass
class Expectation:
    """What a page's *source* says it needs, parsed from the served HTML."""

    mathjax: bool = False
    katex: bool = False
    math_delims: int = 0
    mermaid_divs: int = 0
    prism: bool = False
    exists: bool = True


#: A failed request only means "this render is degraded" for hosts the *layout*
#: depends on: our own server, and the renderer CDNs. Third-party embeds (YouTube
#: and friends) fire tracking/ad requests that fail routinely on both sides and
#: never move a pixel — those are recorded but not treated as degradation.
LAYOUT_CRITICAL_HOSTS = (
    "127.0.0.1", "localhost",
    "cdn.jsdelivr.net", "cdnjs.cloudflare.com", "unpkg.com",
)


def _host_of(url: str) -> str:
    m = re.match(r"[a-z]+://([^/:]+)", url, re.I)
    return (m.group(1) if m else "").lower()


def _layout_critical(url: str) -> bool:
    return _host_of(url) in LAYOUT_CRITICAL_HOSTS


@dataclass
class Shot:
    path: Path | None = None
    width: int = 0
    height: int = 0
    seconds: float = 0.0
    probe: dict = field(default_factory=dict)
    expect: Expectation = field(default_factory=Expectation)
    failed_requests: list[str] = field(default_factory=list)
    third_party_failures: list[str] = field(default_factory=list)
    barrier_notes: list[str] = field(default_factory=list)
    console_errors: int = 0
    masked_embeds: int = 0
    error: str | None = None

    @property
    def degraded_reasons(self) -> list[str]:
        r: list[str] = []
        p, e = self.probe, self.expect
        if not p:
            return ["no render probe"]
        if (e.mathjax or e.katex) and e.math_delims > 0:
            if p.get("mathjax_nodes", 0) == 0 and p.get("katex_nodes", 0) == 0:
                r.append(
                    f"math never typeset ({e.math_delims} $$ delimiters in source, "
                    "0 mjx-container/.katex nodes)"
                )
        if e.mermaid_divs > 0:
            done = p.get("mermaid_processed", 0)
            if done < e.mermaid_divs:
                r.append(f"mermaid {done}/{e.mermaid_divs} diagrams rendered")
            elif p.get("mermaid_svg", 0) == 0:
                r.append("mermaid marked processed but produced no <svg>")
        if e.prism and p.get("prism_tokens", 0) == 0:
            r.append("Prism highlighted nothing (0 .token spans)")
        if p.get("img_broken", 0):
            r.append(f"{p['img_broken']} broken <img>")
        for f in self.failed_requests[:3]:
            r.append(f"layout-critical request failed: {f}")
        return r


@dataclass
class PageResult:
    page: str
    verdict: str = "PENDING"  # MATCH | DIFF | DEGRADED | ERROR | NEW | REMOVED
    diff_pixels: int = 0
    total_pixels: int = 0
    diff_ratio: float = 0.0
    max_delta: int = 0
    mean_delta: float = 0.0
    dims_base: tuple[int, int] = (0, 0)
    dims_head: tuple[int, int] = (0, 0)
    dims_mismatch: bool = False
    seconds: float = 0.0
    attempts: int = 1
    reproduced: bool = False
    degraded_base: list[str] = field(default_factory=list)
    degraded_head: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    art_base: str | None = None
    art_head: str | None = None
    art_diff: str | None = None
    art_side: str | None = None
    error: str | None = None

    @property
    def is_failure(self) -> bool:
        return self.verdict in ("DIFF", "ERROR")

    @property
    def is_degraded(self) -> bool:
        return bool(self.degraded_base or self.degraded_head)


# --------------------------------------------------------------------------- #
# git worktrees + local servers
# --------------------------------------------------------------------------- #


def run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler minus the logging, plus a no-cache policy."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: D102 - silence
        pass

    def end_headers(self):
        # Local HTML/CSS must never be served from the browser cache: the two
        # sides live on different ports so they cannot collide, but a stale
        # 304 on a re-shoot would defeat the retry logic.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class Source:
    """One side of the comparison: a document root served over HTTP."""

    def __init__(self, label: str, ref: str, root: Path, temporary: bool):
        self.label = label
        self.ref = ref
        self.root = root
        self.temporary = temporary
        self._srv: _Server | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    def serve(self) -> None:
        handler = functools.partial(_QuietHandler, directory=str(self.root))
        self._srv = _Server(("127.0.0.1", 0), handler)
        self.port = self._srv.server_address[1]
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    def url(self, page: str) -> str:
        return f"http://127.0.0.1:{self.port}/{page}"

    def stop(self) -> None:
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None

    def read(self, page: str) -> str | None:
        p = self.root / page
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None


def make_source(
    label: str, ref: str, repo: Path, tmp_dir: Path, worktrees: list[Path]
) -> Source:
    """Materialise `ref` as a document root. ``worktree``/``.`` = live repo."""
    if ref.lower() in ("worktree", "workdir", "working-tree", "."):
        return Source(label, "worktree", repo, temporary=False)
    dest = tmp_dir / f"vr-{label}-{re.sub(r'[^A-Za-z0-9._-]', '_', ref)}"
    if dest.exists():
        try:
            run_git(repo, "worktree", "remove", "--force", str(dest))
        except RuntimeError:
            shutil.rmtree(dest, ignore_errors=True)
    # A previous run killed mid-flight (SIGTERM, CI timeout) leaves the path
    # registered but absent; prune makes reuse of the same tmp-dir self-healing.
    try:
        run_git(repo, "worktree", "prune")
    except RuntimeError:
        pass
    run_git(repo, "worktree", "add", "--detach", "--force", "--quiet", str(dest), ref)
    worktrees.append(dest)
    return Source(label, ref, dest, temporary=True)


def cleanup_worktrees(repo: Path, worktrees: Sequence[Path]) -> None:
    for wt in worktrees:
        try:
            run_git(repo, "worktree", "remove", "--force", str(wt))
        except RuntimeError:
            shutil.rmtree(wt, ignore_errors=True)
    try:
        run_git(repo, "worktree", "prune")
    except RuntimeError:
        pass


# --------------------------------------------------------------------------- #
# page-list resolution
# --------------------------------------------------------------------------- #


def tracked_html(repo: Path) -> list[str]:
    out = run_git(repo, "ls-files", "*.html")
    return sorted(p for p in out.split() if "/archive/" not in p)


def resolve_pages(spec: str, repo: Path) -> list[str]:
    if spec == "all":
        return tracked_html(repo)
    cand = (repo / spec) if not os.path.isabs(spec) else Path(spec)
    if cand.is_file() and cand.suffix.lower() in (".html", ".htm"):
        # a single page, not a manifest
        return [cand.resolve().relative_to(repo).as_posix()]
    if cand.is_file():
        pages: list[str] = []
        for raw in cand.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                pages.append(line)
        # de-duplicate, preserve manifest order
        seen: set[str] = set()
        return [p for p in pages if not (p in seen or seen.add(p))]
    if any(ch in spec for ch in "*?["):
        rx = _glob_regex(spec)
        matched = [p for p in tracked_html(repo) if rx.match(p)]
        if not matched:
            raise SystemExit(f"--pages: glob matched no tracked page: {spec!r}")
        return sorted(matched)
    raise SystemExit(f"--pages: not a manifest file, glob or 'all': {spec!r}")


def _glob_regex(spec: str) -> re.Pattern[str]:
    """Shell-glob semantics with proper path awareness.

    ``fnmatch`` lets a single ``*`` swallow ``/``, which makes
    ``knowledge/*/index.html`` silently match six levels deep. Here ``*`` and
    ``?`` stop at a path separator and only ``**`` crosses directories.
    """
    out, i, n = ["(?s:"], 0, len(spec)
    while i < n:
        c = spec[i]
        if c == "*":
            if spec[i:i + 3] == "**/":
                out.append("(?:[^/]+/)*")
                i += 3
                continue
            if spec[i:i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = spec.find("]", i + 1)
            if j < 0:
                out.append(re.escape(c))
            else:
                body = spec[i + 1:j].replace("\\", "\\\\")
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = j + 1
                continue
        else:
            out.append(re.escape(c))
        i += 1
    out.append(r")\Z")
    return re.compile("".join(out))


#: Regions where math delimiters are *not* math: MathJax's own config block, code
#: samples, Mermaid sources. MathJax skips these (skipHtmlTags / ignoreHtmlClass),
#: so counting them would flag perfectly healthy pages as DEGRADED.
_NON_MATH_REGIONS = re.compile(
    r"<script\b.*?</script>|<style\b.*?</style>|<pre\b.*?</pre>|<code\b.*?</code>"
    r"|<textarea\b.*?</textarea>|<!--.*?-->"
    r'|<div[^>]*class="[^"]*\bmermaid\b[^"]*"[^>]*>.*?</div>',
    re.S | re.I,
)


def parse_expectation(source: Source, page: str) -> Expectation:
    text = source.read(page)
    if text is None:
        return Expectation(exists=False)
    lower = text.lower()
    body = _NON_MATH_REGIONS.sub(" ", text)
    display = body.count("$$") // 2 + len(re.findall(r"\\\[", body))
    inline = len(re.findall(r"(?<![\\$])\$(?!\$)[^$\n]{1,200}\$(?!\$)", body)) \
        + len(re.findall(r"\\\(", body))
    # Require real display math, or enough inline hits that a stray '$' in prose
    # cannot be the explanation.
    delims = display if display else (inline if inline >= 3 else 0)
    return Expectation(
        mathjax="mathjax" in lower,
        katex="katex" in lower,
        math_delims=delims,
        mermaid_divs=len(re.findall(r'class="[^"]*\bmermaid\b[^"]*"', text)),
        prism="prism" in lower,
        exists=True,
    )


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #


async def _await_barrier(page, expect: Expectation, notes: list[str]) -> None:
    """Block until every renderer this page depends on has finished."""

    async def soft(coro, label: str):
        try:
            await coro
        except (PWError, asyncio.TimeoutError) as exc:
            notes.append(f"{label}: {type(exc).__name__}")

    await soft(page.wait_for_load_state("networkidle", timeout=T_NETWORK_IDLE),
               "networkidle")
    await soft(
        page.wait_for_function("() => document.fonts && document.fonts.status === 'loaded'",
                              timeout=T_FONTS),
        "fonts.ready",
    )

    if expect.mathjax and expect.math_delims:
        try:
            await page.wait_for_function(
                "() => !!(window.MathJax && window.MathJax.startup "
                "&& window.MathJax.startup.promise)",
                timeout=T_MATHJAX_LOAD,
            )
            # page.evaluate has NO implicit timeout, and 12 pages in this corpus
            # load MathJax twice (cdnjs *and* jsdelivr). The second load can
            # leave startup.promise permanently pending, which would hang the
            # whole run. Race it in JS *and* bound it from Python.
            await asyncio.wait_for(
                page.evaluate(
                    """async (budget) => {
                        const s = window.MathJax.startup;
                        const bail = new Promise((r) => setTimeout(r, budget));
                        const work = (async () => {
                            if (s && s.promise) await s.promise;
                            if (window.MathJax.typesetPromise) {
                                try { await window.MathJax.typesetPromise(); }
                                catch (e) {}
                            }
                        })();
                        // Escape hatch for the double-loaded-MathJax pages: once
                        // the typeset node count has stopped growing, the work is
                        // observably done even if the promise never resolves.
                        const settled = new Promise((resolve) => {
                            let last = -1, same = 0;
                            const iv = setInterval(() => {
                                const n = document.querySelectorAll(
                                    'mjx-container, svg[data-mml-node]').length;
                                same = (n > 0 && n === last) ? same + 1 : 0;
                                last = n;
                                if (same >= 3) { clearInterval(iv); resolve('settled'); }
                            }, 150);
                            setTimeout(() => clearInterval(iv), budget);
                        });
                        await Promise.race([work, settled, bail]);
                    }""",
                    T_MATHJAX_TYPESET,
                ),
                timeout=(T_MATHJAX_TYPESET + 5000) / 1000,
            )
            await page.wait_for_function(
                "() => document.querySelectorAll('mjx-container, svg[data-mml-node]')"
                ".length > 0",
                timeout=T_MATHJAX_TYPESET,
            )
        except (PWError, asyncio.TimeoutError) as exc:
            notes.append(f"mathjax: {type(exc).__name__}")

    if expect.katex and expect.math_delims:
        await soft(
            page.wait_for_function(
                "() => document.querySelectorAll('.katex').length > 0",
                timeout=T_KATEX,
            ),
            "katex",
        )

    if expect.mermaid_divs:
        await soft(
            page.wait_for_function(
                "(n) => document.querySelectorAll('.mermaid[data-processed]').length >= n"
                " && document.querySelectorAll('.mermaid svg').length >= n",
                arg=expect.mermaid_divs,
                timeout=T_MERMAID,
            ),
            "mermaid",
        )

    if expect.prism:
        await soft(
            page.wait_for_function(
                "() => !document.querySelector('code[class*=\"language-\"]')"
                " || document.querySelectorAll('.token').length > 0",
                timeout=T_PRISM,
            ),
            "prism",
        )

    # Geometry/DOM settling: the shutter only opens once nothing moves.
    stable, last, waited = 0, None, 0
    while stable < SETTLE_STABLE_POLLS and waited < SETTLE_MAX_MS:
        sig = await page.evaluate(
            "() => document.documentElement.scrollHeight + '|' +"
            " (document.body ? document.body.innerHTML.length : 0) + '|' +"
            " document.querySelectorAll('*').length"
        )
        stable = stable + 1 if sig == last else 0
        last = sig
        await page.wait_for_timeout(SETTLE_POLL_MS)
        waited += SETTLE_POLL_MS
    if stable < SETTLE_STABLE_POLLS:
        notes.append("dom-never-settled")


MASK_EMBEDS_SCRIPT = r"""
() => {
  // Cross-origin <iframe>s are outside every readiness signal we have: their
  // content keeps mutating (Google Maps info cards, YouTube player chrome) long
  // after our own DOM has settled, and we cannot observe it. Tag them so the
  // screenshot paints a flat box over each one. The box preserves the iframe's
  // position and size, so a CSS change that moves or resizes an embed is still
  // caught — only the third party's own pixels are excluded.
  let n = 0;
  for (const f of document.querySelectorAll('iframe')) {
    let host = '';
    try { host = new URL(f.getAttribute('src') || '', location.href).host; }
    catch (e) { host = ''; }
    if (host && host !== location.host) { f.setAttribute('data-vr-masked', ''); n++; }
  }
  return n;
}
"""


async def _screenshot(page, width: int, height: int, dest: Path, mask) -> None:
    """full_page screenshot, stitching strips for pages taller than Chromium
    will capture in one shot."""
    total = await page.evaluate("() => document.documentElement.scrollHeight")
    common = dict(animations="disabled", caret="hide", scale="css",
                  mask=mask, mask_color="#FF00FF")
    if total <= STITCH_ABOVE_PX:
        await page.screenshot(path=str(dest), full_page=True, **common)
        return
    strips: list[Image.Image] = []
    y = 0
    while y < total:
        h = min(STITCH_CHUNK_PX, total - y)
        buf = await page.screenshot(
            full_page=True,
            clip={"x": 0, "y": y, "width": width, "height": h}, **common,
        )
        strips.append(Image.open(io.BytesIO(buf)).convert("RGB"))
        y += h
    out = Image.new("RGB", (max(s.width for s in strips), sum(s.height for s in strips)))
    off = 0
    for s in strips:
        out.paste(s, (0, off))
        off += s.height
    out.save(dest)


async def capture(
    context, source: Source, page_path: str, dest: Path, viewport: tuple[int, int],
    budget_s: float = 180.0, mask_embeds: bool = True,
    block_hosts: tuple[str, ...] = ()
) -> Shot:
    shot = Shot(expect=parse_expectation(source, page_path))
    if not shot.expect.exists:
        shot.error = "missing"
        return shot
    t0 = time.perf_counter()
    pg = await context.new_page()
    failed: list[str] = []
    noise: list[str] = []
    errors = [0]

    def _note_failure(desc: str, url: str) -> None:
        entry = f"{desc} {url[:160]}"
        (failed if _layout_critical(url) else noise).append(entry)

    pg.on("requestfailed",
          lambda r: _note_failure(str(r.failure or "failed"), r.url))
    pg.on("response",
          lambda r: _note_failure(f"HTTP {r.status}", r.url) if r.status >= 400
          else None)
    pg.on("console", lambda m: errors.__setitem__(0, errors[0] + 1)
          if m.type == "error" else None)
    if block_hosts:
        # Simulated outage, used to verify that the DEGRADED guard really fires
        # rather than trusting that it would. Applied to both sides equally.
        async def _abort(route):
            if _host_of(route.request.url) in block_hosts:
                await route.abort()
            else:
                await route.continue_()
        await pg.route("**/*", _abort)

    async def _work() -> None:
        await pg.goto(source.url(page_path), wait_until="load", timeout=45000)
        mask = []
        if mask_embeds:
            shot.masked_embeds = await pg.evaluate(MASK_EMBEDS_SCRIPT) or 0
            if shot.masked_embeds:
                mask = [pg.locator("iframe[data-vr-masked]")]
        await _await_barrier(pg, shot.expect, shot.barrier_notes)
        shot.probe = await pg.evaluate(PROBE_SCRIPT) or {}
        await _screenshot(pg, viewport[0], viewport[1], dest, mask)
        with Image.open(dest) as im:
            shot.width, shot.height = im.size
        shot.path = dest

    try:
        # Hard per-side budget. Without it a single pathological page (e.g. one
        # that loads MathJax twice and never resolves startup.promise) stalls the
        # entire run instead of reporting itself as a problem.
        await asyncio.wait_for(_work(), timeout=budget_s)
    except asyncio.TimeoutError:
        shot.error = f"page budget of {budget_s:.0f}s exceeded"
        shot.barrier_notes.append("page-budget-exceeded")
    except Exception as exc:  # noqa: BLE001 - any failure is a reportable error
        shot.error = f"{type(exc).__name__}: {exc}".strip()[:400]
    finally:
        shot.failed_requests = failed[:20]
        shot.third_party_failures = noise[:20]
        shot.console_errors = errors[0]
        shot.seconds = time.perf_counter() - t0
        await pg.close()
    return shot


# --------------------------------------------------------------------------- #
# diffing
# --------------------------------------------------------------------------- #


def _load(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"), dtype=np.int16)


def _pad_to(a: np.ndarray, h: int, w: int) -> np.ndarray:
    if a.shape[0] == h and a.shape[1] == w:
        return a
    out = np.full((h, w, 3), 255, dtype=np.int16)
    out[: a.shape[0], : a.shape[1]] = a
    return out


def diff_images(base_png: Path, head_png: Path, pixel_delta: int):
    """Return (diff_px, total_px, ratio, max_delta, mean_delta, mask, arrays)."""
    a, b = _load(base_png), _load(head_png)
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    ap, bp = _pad_to(a, h, w), _pad_to(b, h, w)
    delta = np.abs(ap - bp).max(axis=2)  # worst channel per pixel
    mask = delta > pixel_delta
    diff_px = int(mask.sum())
    total = h * w
    return (
        diff_px,
        total,
        diff_px / total if total else 0.0,
        int(delta.max()) if total else 0,
        float(delta.mean()) if total else 0.0,
        mask,
        ap.astype(np.uint8),
        bp.astype(np.uint8),
    )


def _label(img: Image.Image, text: str) -> Image.Image:
    bar = 22
    out = Image.new("RGB", (img.width, img.height + bar), (24, 26, 32))
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    try:
        d.text((6, 6), text, fill=(235, 238, 245))
    except Exception:  # pragma: no cover - fontless environments
        pass
    return out


def write_diff_art(
    out_dir: Path, slug: str, base_arr: np.ndarray, head_arr: np.ndarray,
    mask: np.ndarray, max_side_height: int = 4200
) -> tuple[str, str]:
    """Write a red-highlight diff PNG and a labelled base|head|diff triptych."""
    # Highlight: desaturated head with differing pixels blown out to magenta.
    grey = head_arr.mean(axis=2).astype(np.uint8)
    hl = np.stack([grey, grey, grey], axis=2)
    hl = (hl * 0.45).astype(np.uint8)
    hl[mask] = np.array([255, 0, 128], dtype=np.uint8)
    diff_img = Image.fromarray(hl, "RGB")
    diff_name = f"{slug}.diff.png"
    diff_img.save(out_dir / diff_name)

    cols = [
        _label(Image.fromarray(base_arr, "RGB"), "BASE"),
        _label(Image.fromarray(head_arr, "RGB"), "HEAD"),
        _label(diff_img, "DIFF (magenta = changed)"),
    ]
    gut = 8
    W = sum(c.width for c in cols) + gut * (len(cols) - 1)
    H = max(c.height for c in cols)
    side = Image.new("RGB", (W, H), (12, 13, 16))
    x = 0
    for c in cols:
        side.paste(c, (x, 0))
        x += c.width + gut
    if side.height > max_side_height:  # keep the artifact a sane size
        scale = max_side_height / side.height
        side = side.resize(
            (max(1, int(side.width * scale)), max_side_height), Image.LANCZOS
        )
    side_name = f"{slug}.side.png"
    side.save(out_dir / side_name)
    return diff_name, side_name


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def slugify(page: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", page).strip("_")[:150]


async def compare_page(
    context, base: Source, head: Source, page: str, args, out_dir: Path,
    shots_dir: Path, viewport: tuple[int, int]
) -> PageResult:
    res = PageResult(page=page)
    slug = slugify(page)
    t0 = time.perf_counter()

    for attempt in range(1, args.retries + 2):
        res.attempts = attempt
        bp = shots_dir / f"{slug}.base.png"
        hp = shots_dir / f"{slug}.head.png"
        sb = await capture(context, base, page, bp, viewport, args.page_timeout,
                           not args.no_mask_embeds, args.block_hosts)
        sh = await capture(context, head, page, hp, viewport, args.page_timeout,
                           not args.no_mask_embeds, args.block_hosts)

        res.degraded_base = sb.degraded_reasons
        res.degraded_head = sh.degraded_reasons
        res.notes = [f"base barrier: {n}" for n in sb.barrier_notes] + \
                    [f"head barrier: {n}" for n in sh.barrier_notes]
        if sb.masked_embeds or sh.masked_embeds:
            res.notes.append(
                f"{max(sb.masked_embeds, sh.masked_embeds)} cross-origin iframe(s) "
                "masked out of the comparison (position and size still compared)"
            )
        tp = len(sb.third_party_failures) + len(sh.third_party_failures)
        if tp:
            hosts = sorted({_host_of(u.split(" ", 1)[-1])
                            for u in sb.third_party_failures + sh.third_party_failures}
                           - {""})
            res.notes.append(
                f"{tp} third-party embed request(s) failed on "
                f"{', '.join(hosts[:5])} — informational, not counted as degradation"
            )
        res.dims_base = (sb.width, sb.height)
        res.dims_head = (sh.width, sh.height)

        if sb.error == "missing" and sh.error == "missing":
            res.verdict, res.error = "ERROR", "page absent from both sources"
            break
        if sb.error == "missing":
            res.verdict = "NEW"
            res.notes.append("page does not exist in base — nothing to compare")
            break
        if sh.error == "missing":
            res.verdict = "REMOVED"
            res.notes.append("page does not exist in head — nothing to compare")
            break
        if sb.error or sh.error:
            res.verdict = "ERROR"
            res.error = f"base: {sb.error or 'ok'} | head: {sh.error or 'ok'}"
            if attempt <= args.retries:
                continue
            break

        (res.diff_pixels, res.total_pixels, res.diff_ratio, res.max_delta,
         res.mean_delta, mask, arr_b, arr_h) = diff_images(bp, hp, args.pixel_delta)
        res.dims_mismatch = res.dims_base != res.dims_head

        over = res.diff_ratio > args.threshold or res.dims_mismatch
        if not over:
            res.verdict = "DEGRADED" if res.is_degraded else "MATCH"
            for p in (bp, hp):
                p.unlink(missing_ok=True)  # matches keep no artwork
            break

        # A mismatch must reproduce before we believe it.
        if attempt <= args.retries:
            continue
        res.verdict = "DIFF"
        res.reproduced = attempt > 1
        if res.dims_mismatch:
            res.notes.append(
                f"full-page size differs: base {res.dims_base[0]}x{res.dims_base[1]}"
                f" vs head {res.dims_head[0]}x{res.dims_head[1]}"
            )
        res.art_base, res.art_head = bp.name, hp.name
        res.art_diff, res.art_side = write_diff_art(shots_dir, slug, arr_b, arr_h, mask)

    res.seconds = time.perf_counter() - t0
    return res


async def run(args) -> int:
    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"--repo is not a git repository: {repo}")

    pages = resolve_pages(args.pages, repo)
    if args.list:
        print("\n".join(pages))
        print(f"\n{len(pages)} page(s)", file=sys.stderr)
        return 0
    if not pages:
        raise SystemExit("no pages resolved from --pages")

    vw, vh = args.viewport
    out_dir = Path(args.out).resolve()
    shots_dir = out_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(args.tmp_dir or tempfile.gettempdir()).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    worktrees: list[Path] = []
    base = head = None
    results: list[PageResult] = []
    started = time.time()
    t_wall = time.perf_counter()

    try:
        base = make_source("base", args.base, repo, tmp_dir, worktrees)
        head = make_source("head", args.head, repo, tmp_dir, worktrees)
        base.serve()
        head.serve()
        print(f"base  {args.base:<28} -> {base.root}  :{base.port}")
        print(f"head  {args.head:<28} -> {head.root}  :{head.port}")
        print(f"pages {len(pages)}  workers {args.workers}  "
              f"viewport {vw}x{vh}  threshold {args.threshold:.6f}  "
              f"pixel-delta {args.pixel_delta}")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=CHROMIUM_ARGS)
            queue: asyncio.Queue[str] = asyncio.Queue()
            for p in pages:
                queue.put_nowait(p)
            done = [0]
            lock = asyncio.Lock()

            async def worker(_i: int):
                ctx = await browser.new_context(
                    viewport={"width": vw, "height": vh},
                    device_scale_factor=1,
                    is_mobile=False,
                    has_touch=False,
                    reduced_motion="reduce",
                    color_scheme="light",
                    forced_colors="none",
                    locale="en-US",
                    timezone_id="UTC",
                    java_script_enabled=True,
                    bypass_csp=False,
                )
                await ctx.add_init_script(INIT_SCRIPT)
                try:
                    while True:
                        try:
                            page = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        r = await compare_page(ctx, base, head, page, args,
                                               out_dir, shots_dir, (vw, vh))
                        async with lock:
                            results.append(r)
                            done[0] += 1
                            flag = {"MATCH": "ok", "DIFF": "DIFF", "ERROR": "ERR",
                                    "DEGRADED": "degr", "NEW": "new",
                                    "REMOVED": "gone"}.get(r.verdict, r.verdict)
                            extra = ""
                            if r.verdict in ("DIFF", "MATCH", "DEGRADED"):
                                extra = f" {r.diff_ratio*100:.4f}% ({r.diff_pixels}px)"
                            print(f"[{done[0]:>4}/{len(pages)}] {flag:<5}"
                                  f"{r.seconds:6.1f}s {page}{extra}", flush=True)
                finally:
                    await ctx.close()

            await asyncio.gather(*(worker(i) for i in range(max(1, args.workers))))
            await browser.close()
    finally:
        for s in (base, head):
            if s:
                s.stop()
        if not args.keep_worktrees:
            cleanup_worktrees(repo, worktrees)

    wall = time.perf_counter() - t_wall
    order = {"DIFF": 0, "ERROR": 1, "DEGRADED": 2, "NEW": 3, "REMOVED": 4, "MATCH": 5}
    results.sort(key=lambda r: (order.get(r.verdict, 9), -r.diff_ratio, r.page))

    summary = {
        "base": args.base, "head": args.head, "pages": len(pages),
        "workers": args.workers, "viewport": f"{vw}x{vh}",
        "threshold": args.threshold, "pixel_delta": args.pixel_delta,
        "retries": args.retries,
        "started": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(started)),
        "wall_seconds": round(wall, 1),
        "per_page_seconds": round(wall / max(1, len(pages)), 2),
        "capture_seconds_total": round(sum(r.seconds for r in results), 1),
        "counts": {k: sum(1 for r in results if r.verdict == k)
                   for k in ("MATCH", "DIFF", "DEGRADED", "ERROR", "NEW", "REMOVED")},
        "degraded_pages": sum(1 for r in results if r.is_degraded),
    }
    write_report(out_dir, summary, results)
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary,
                    "results": [asdict(r) for r in results]}, indent=1, default=str),
        encoding="utf-8",
    )

    c = summary["counts"]
    print("\n" + "=" * 72)
    print(f"pages {len(pages)}   match {c['MATCH']}   DIFF {c['DIFF']}   "
          f"degraded {c['DEGRADED']}   error {c['ERROR']}   "
          f"new {c['NEW']}   removed {c['REMOVED']}")
    print(f"wall {wall:.1f}s   mean {summary['per_page_seconds']:.2f}s/page "
          f"(both sides)   est. 1646 pages @ {args.workers}w: "
          f"{summary['per_page_seconds']*1646/60:.0f} min")
    print(f"report {out_dir / 'index.html'}")
    if summary["degraded_pages"]:
        print(f"\n!! WARNING: {summary['degraded_pages']} page(s) rendered DEGRADED "
              "(CDN unreachable or a renderer never ran).")
        print("   A pixel match on those pages proves nothing. See the report banner.")
    print("=" * 72)

    failures = [r for r in results if r.is_failure]
    if failures:
        print(f"\nFAILED: {len(failures)} page(s) over threshold / errored:")
        for r in failures[:40]:
            print(f"  {r.verdict:<6} {r.diff_ratio*100:8.4f}%  {r.page}"
                  f"{'  ' + r.error if r.error else ''}")
        return 1
    if summary["degraded_pages"] and args.fail_on_degraded:
        return 1
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

REPORT_CSS = """
:root{--bg:#0f1115;--fg:#e7e9ee;--mut:#98a0b0;--card:#171a21;--line:#262b36;
--ok:#3fb950;--bad:#f85149;--warn:#d29922;--info:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:34px 0 10px;
border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--mut);margin:0 0 18px}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:10px 14px;min-width:112px}
.kpi b{display:block;font-size:22px;line-height:1.2}
.kpi span{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.kpi.bad b{color:var(--bad)}.kpi.ok b{color:var(--ok)}.kpi.warn b{color:var(--warn)}
.banner{border-radius:8px;padding:12px 14px;margin:16px 0;border:1px solid}
.banner.bad{background:#2d1214;border-color:#5c1f22;color:#ffb4b0}
.banner.warn{background:#2b2410;border-color:#5c4a12;color:#ffdf9e}
.banner.ok{background:#10251a;border-color:#1c4a2e;color:#9ce6b0}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);
vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;
letter-spacing:.05em;position:sticky;top:0;background:var(--bg)}
td.num{text-align:right;font-variant-numeric:tabular-nums;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.v{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
font-weight:700;letter-spacing:.03em}
.v.MATCH{background:#10251a;color:var(--ok)}
.v.DIFF{background:#2d1214;color:var(--bad)}
.v.DEGRADED{background:#2b2410;color:var(--warn)}
.v.ERROR{background:#2d1214;color:var(--bad)}
.v.NEW,.v.REMOVED{background:#161d2b;color:var(--info)}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;
margin:10px 0;padding:0 12px}
details[open]{padding-bottom:12px}
summary{cursor:pointer;padding:11px 0;font-weight:600}
.imgwrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px;
background:#000;margin-top:8px}
.imgwrap img{display:block;max-width:none}
.note{color:var(--mut);font-size:12px;margin:6px 0 0}
a{color:var(--info)}
.legend{color:var(--mut);font-size:12px;margin-top:6px}
"""


def write_report(out_dir: Path, s: dict, results: list[PageResult]) -> None:
    c = s["counts"]
    esc = html.escape

    def kpi(v, label, cls=""):
        return f'<div class="kpi {cls}"><b>{v}</b><span>{label}</span></div>'

    banners = []
    if c["DIFF"] or c["ERROR"]:
        banners.append(
            f'<div class="banner bad"><b>FAIL</b> — {c["DIFF"]} page(s) exceeded the '
            f'{s["threshold"]*100:.4f}% differing-pixel threshold'
            + (f', {c["ERROR"]} errored' if c["ERROR"] else "")
            + ". This refactor is <b>not</b> pixel-neutral.</div>"
        )
    elif s["degraded_pages"]:
        banners.append(
            '<div class="banner warn"><b>PASS, BUT NOT TRUSTWORTHY</b> — '
            f'{s["degraded_pages"]} page(s) rendered degraded, so their pixel match '
            "may be vacuous (both sides failing identically). Fix connectivity to the "
            "MathJax/Mermaid/Prism CDNs and re-run before trusting this result.</div>"
        )
    else:
        banners.append(
            f'<div class="banner ok"><b>PASS</b> — all {s["pages"]} page(s) render '
            "within tolerance, with every renderer verified to have produced output."
            "</div>"
        )
    if s["degraded_pages"] and (c["DIFF"] or c["ERROR"]):
        banners.append(
            f'<div class="banner warn">{s["degraded_pages"]} page(s) also rendered '
            "degraded — see the Degraded column.</div>"
        )

    rows = []
    for r in results:
        links = []
        for label, art in (("base", r.art_base), ("head", r.art_head),
                           ("diff", r.art_diff)):
            if art:
                links.append(f'<a href="shots/{esc(art)}">{label}</a>')
        degr = "; ".join(r.degraded_base + r.degraded_head)
        rows.append(
            "<tr>"
            f'<td><span class="v {r.verdict}">{r.verdict}</span></td>'
            f'<td><code>{esc(r.page)}</code>'
            + (f'<div class="note">{esc(r.error)}</div>' if r.error else "")
            + (f'<div class="note">{esc("; ".join(r.notes))}</div>' if r.notes else "")
            + "</td>"
            f'<td class="num">{r.diff_ratio*100:.4f}%</td>'
            f'<td class="num">{r.diff_pixels:,}</td>'
            f'<td class="num">{r.max_delta}</td>'
            f'<td class="num">{r.mean_delta:.3f}</td>'
            f'<td class="mono">{r.dims_base[0]}x{r.dims_base[1]}'
            + (f' / {r.dims_head[0]}x{r.dims_head[1]}' if r.dims_mismatch else "")
            + "</td>"
            f'<td class="num">{r.seconds:.1f}</td>'
            f'<td class="num">{r.attempts}</td>'
            f'<td class="note">{esc(degr)}</td>'
            f"<td>{' '.join(links)}</td>"
            "</tr>"
        )

    galleries = []
    for r in results:
        if r.verdict != "DIFF" or not r.art_side:
            continue
        galleries.append(
            f"<details open><summary>{esc(r.page)} — "
            f"{r.diff_ratio*100:.4f}% ({r.diff_pixels:,} px, max delta "
            f"{r.max_delta})"
            + (" — reproduced on retry" if r.reproduced else "")
            + "</summary>"
            f'<div class="imgwrap"><img src="shots/{esc(r.art_side)}" '
            f'alt="side-by-side for {esc(r.page)}" loading="lazy"></div>'
            f'<p class="note">Full resolution: '
            f'<a href="shots/{esc(r.art_base)}">base</a> · '
            f'<a href="shots/{esc(r.art_head)}">head</a> · '
            f'<a href="shots/{esc(r.art_diff)}">diff mask</a>. '
            "Triptych is downscaled for size; magenta marks changed pixels.</p>"
            "</details>"
        )
    if not galleries:
        galleries.append('<p class="note">No mismatching pages — no diff images '
                         "were produced.</p>")

    meta = " · ".join([
        f"base <code>{esc(s['base'])}</code>",
        f"head <code>{esc(s['head'])}</code>",
        f"viewport {s['viewport']}",
        f"threshold {s['threshold']*100:.4f}%",
        f"pixel-delta {s['pixel_delta']}",
        f"retries {s['retries']}",
        f"workers {s['workers']}",
        s["started"],
    ])

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual regression: {esc(s['base'])} vs {esc(s['head'])}</title>
<style>{REPORT_CSS}</style></head><body><div class="wrap">
<h1>Visual regression report</h1>
<p class="sub">{meta}</p>
{''.join(banners)}
<div class="kpis">
{kpi(s['pages'], 'pages')}
{kpi(c['MATCH'], 'match', 'ok' if c['MATCH'] else '')}
{kpi(c['DIFF'], 'diffs', 'bad' if c['DIFF'] else '')}
{kpi(c['DEGRADED'], 'degraded', 'warn' if c['DEGRADED'] else '')}
{kpi(c['ERROR'], 'errors', 'bad' if c['ERROR'] else '')}
{kpi(c['NEW'] + c['REMOVED'], 'new/removed')}
{kpi(f"{s['wall_seconds']:.0f}s", 'wall clock')}
{kpi(f"{s['per_page_seconds']:.2f}s", 'per page')}
</div>
<h2>Mismatches</h2>
{''.join(galleries)}
<h2>All pages ({len(results)}), diffs first</h2>
<div style="overflow-x:auto"><table>
<thead><tr><th>Verdict</th><th>Page</th><th>Diff %</th><th>Diff px</th>
<th>Max &Delta;</th><th>Mean &Delta;</th><th>Dims (W&times;H)</th><th>Sec</th>
<th>Tries</th><th>Degraded</th><th>Art</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="legend">Diff % is the fraction of pixels whose worst RGB channel
differs by more than {s['pixel_delta']}/255. A page fails when that fraction
exceeds {s['threshold']*100:.4f}% or when the two full-page screenshots have
different dimensions. Matching pages keep no screenshots (report size).
Machine-readable copy of everything here: <code>results.json</code>.</p>
</div></body></html>
"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def _viewport(v: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)[xX×](\d+)", v.strip())
    if not m:
        raise argparse.ArgumentTypeError("expected WxH, e.g. 1280x2000")
    return int(m.group(1)), int(m.group(2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visual_regression.py",
        description="Pixel-compare page rendering between two git refs "
                    "(or the working tree and a ref).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(__doc__ or "").split("QUICK START")[-1][:1600],
    )
    p.add_argument("--base", required=True, metavar="REF",
                   help="git ref for the 'before' side")
    p.add_argument("--head", default="worktree", metavar="REF|worktree",
                   help="git ref, or 'worktree' for the live working tree (default)")
    p.add_argument("--pages", default=DEFAULT_MANIFEST, metavar="FILE|GLOB|all",
                   help=f"manifest file, glob, or 'all' (default: {DEFAULT_MANIFEST})")
    p.add_argument("--out", default="vr-report", metavar="DIR",
                   help="report directory (default: vr-report)")
    p.add_argument("--workers", type=int, default=4, metavar="N",
                   help="concurrent browser contexts (default: 4)")
    p.add_argument("--threshold", type=float, default=0.0005, metavar="F",
                   help="max differing-pixel fraction per page (default: 0.0005)")
    p.add_argument("--pixel-delta", type=int, default=8, metavar="N",
                   help="per-channel delta treated as equal (default: 8)")
    p.add_argument("--viewport", type=_viewport, default=(1280, 2000), metavar="WxH",
                   help="viewport, full_page always on (default: 1280x2000)")
    p.add_argument("--page-timeout", type=float, default=180.0, metavar="SEC",
                   help="hard budget for rendering one page on one side; the "
                        "page is reported as an ERROR rather than stalling the "
                        "run (default: 180)")
    p.add_argument("--retries", type=int, default=1, metavar="N",
                   help="re-shoot a mismatch N more times before believing it "
                        "(default: 1)")
    p.add_argument("--block-hosts", default="", metavar="HOST[,HOST]",
                   type=lambda v: tuple(h.strip().lower() for h in v.split(",")
                                        if h.strip()),
                   help="abort every request to these hosts on BOTH sides. Use it "
                        "to rehearse a CDN outage and confirm the DEGRADED guard "
                        "fires, e.g. --block-hosts cdn.jsdelivr.net,"
                        "cdnjs.cloudflare.com --fail-on-degraded")
    p.add_argument("--no-mask-embeds", action="store_true",
                   help="do NOT paint over cross-origin <iframe>s (YouTube, "
                        "Google Maps). They are masked by default because their "
                        "content is outside every readiness signal and flakes; "
                        "the mask still preserves the iframe's box geometry")
    p.add_argument("--fail-on-degraded", action="store_true",
                   help="exit 1 when any page rendered degraded (CI uses this)")
    p.add_argument("--repo", default=".", metavar="DIR",
                   help="repository / web root (default: cwd, i.e. run from wp/)")
    p.add_argument("--tmp-dir", default=None, metavar="DIR",
                   help="where git worktrees are created (default: system temp)")
    p.add_argument("--keep-worktrees", action="store_true",
                   help="do not remove the temporary worktrees (debugging)")
    p.add_argument("--list", action="store_true",
                   help="print the resolved page list and exit")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.retries < 0:
        raise SystemExit("--retries must be >= 0")
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
