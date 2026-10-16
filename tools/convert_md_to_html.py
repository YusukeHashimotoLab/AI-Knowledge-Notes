#!/usr/bin/env python3
"""
Convert Markdown files to HTML for the bilingual knowledge base.
Supports all Dojos (FM, MI, ML, MS, PI) in knowledge/en/ and knowledge/jp/.

This script converts Markdown files with YAML frontmatter to production-ready HTML
with MathJax support for equations, Mermaid for diagrams, and responsive styling.
The locale (en/jp) is inferred from the target path, or set with --lang.

Usage:
    python3 tools/convert_md_to_html.py knowledge/en/ML/some-series/
    python3 tools/convert_md_to_html.py knowledge/jp/MI/some-series/chapter-1.md
    python3 tools/convert_md_to_html.py ML --lang jp   # whole Dojo
    python3 tools/convert_md_to_html.py --lang jp      # everything in one locale
"""

import argparse
import os
import re
import sys
import yaml
import logging
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import markdown
from markdown.extensions import Extension
from markdown import util
from markdown.inlinepatterns import InlineProcessor
from markdown.preprocessors import Preprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# knowledge/ root anchored to this file so the script works from any cwd
KNOWLEDGE_ROOT = Path(__file__).resolve().parent.parent / "knowledge"

# Locale-specific strings and defaults
LOCALES = {
    "en": {
        "html_lang": "en",
        "label_reading_time": "Reading Time",
        "label_difficulty": "Difficulty",
        "label_code_examples": "Code Examples",
        "label_exercises": "Exercises",
        "unit_examples": "",
        "unit_exercises": "",
        "label_created_by": "Created by",
        "label_supervised": "Supervised by",
        "supervisor": "Dr. Yusuke Hashimoto (Tohoku University)",
        "label_version": "Version",
        "label_created": "Created",
        "label_license": "License",
        "nav_prev": "← Previous Chapter",
        "nav_index": "Back to Series Index",
        "nav_next": "Next Chapter →",
        "default_reading_time": "20-25 minutes",
        "default_difficulty": "Beginner",
        "default_created_at": "2025-01-01",
    },
    "jp": {
        "html_lang": "ja",
        "label_reading_time": "読了時間",
        "label_difficulty": "難易度",
        "label_code_examples": "コード例",
        "label_exercises": "演習問題",
        "unit_examples": "個",
        "unit_exercises": "問",
        "label_created_by": "作成者",
        "label_supervised": "監修",
        "supervisor": "Dr. Yusuke Hashimoto（東北大学）",
        "label_version": "バージョン",
        "label_created": "作成日",
        "label_license": "ライセンス",
        "nav_prev": "← 前の章",
        "nav_index": "シリーズ目次に戻る",
        "nav_next": "次の章 →",
        "default_reading_time": "20-25分",
        "default_difficulty": "初級",
        "default_created_at": "2025-10-17",
    },
}

# Supported Dojos
DOJOS = ["FM", "MI", "ML", "MS", "PI"]

# HTML template header
HTML_HEADER_TEMPLATE = '''<!DOCTYPE html>
<html lang="{html_lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - AI Terakoya</title>

    <link href="{css_href}" rel="stylesheet"/>
    <style>
        /* page-specific residual: the rules the shared sheet does not provide.
           Everything else that used to live here is now in knowledge-base.css;
           re-declaring it would only risk drifting out of sync with the sheet. */
        details {{ background-color: var(--color-bg-alt); margin-bottom: var(--spacing-md) }}
        summary {{ user-select: none; margin: calc(-1 * var(--spacing-md)); border-radius: var(--border-radius) }}
        summary:hover {{
            /* accent wash; the literal is the FM/MI/ML purple and is the fallback
               for engines without color-mix() (Chrome <111 / Safari <16.2). */
            background-color: rgba(123, 44, 191, 0.1);
            background-color: color-mix(in srgb, var(--color-accent) 10%, transparent);
        }}
        details[open] summary {{ margin-bottom: var(--spacing-md); border-bottom: 1px solid var(--color-border) }}
        .learning-objectives {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: var(--spacing-lg);
            border-radius: var(--border-radius);
            border-left: 4px solid var(--color-accent);
            margin-bottom: var(--spacing-xl);
        }}
        .learning-objectives h2 {{ margin-top: 0; border-bottom: none }}
        @media (max-width: 768px) {{
            th {{ padding: var(--spacing-xs) }}
            td {{ padding: var(--spacing-xs) }}
        }}
    </style>

    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>

    <!-- MathJax for LaTeX equation rendering -->
    <script>
        MathJax = {{
            tex: {{
                // The HTML must carry a DOUBLE backslash before ( ) [ ] so the
                // JS string is backslash-paren; a single one collapses to a bare
                // paren and turns every prose parenthesis into a math delimiter
                // (non-raw Python template: 4 backslashes -> 2 in HTML -> 1 in JS).
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                ignoreHtmlClass: 'mermaid'
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
</head>
<body class="{dojo_class}">
    <header>
        <div class="header-content">
            <h1>{chapter_title}</h1>
            <p class="subtitle">{subtitle}</p>
            <div class="meta">
                <span class="meta-item">📖 {label_reading_time}: {reading_time}</span>
                <span class="meta-item">📊 {label_difficulty}: {difficulty}</span>
                <span class="meta-item">💻 {label_code_examples}: {code_examples}{unit_examples}</span>
                <span class="meta-item">📝 {label_exercises}: {exercises}{unit_exercises}</span>
            </div>
        </div>
    </header>

    <main class="container">
'''

HTML_FOOTER_TEMPLATE = '''
    </main>

    <footer>
        <p><strong>{label_created_by}</strong>: AI Terakoya Content Team</p>
        <p><strong>{label_supervised}</strong>: {supervisor}</p>
        <p><strong>{label_version}</strong>: {version} | <strong>{label_created}</strong>: {created_at}</p>
        <p><strong>{label_license}</strong>: Creative Commons BY 4.0</p>
        <p>© 2025 AI Terakoya. All rights reserved.</p>
    </footer>
<script defer src="{js_src}"></script>
</body>
</html>
'''


def shared_css_href(series_path: Path) -> str:
    """Relative href from a series directory to the locale's shared stylesheet.

    Series directories always sit exactly one level under their Dojo directory
    (``knowledge/<lang>/<DOJO>/<series>``, see ``find_series_directories``), so
    the sheet at ``knowledge/<lang>/assets/css/knowledge-base.css`` is always two
    hops up. Deriving it from ``series_path`` rather than from ``KNOWLEDGE_ROOT``
    keeps the href correct even when the tree is converted somewhere else (a
    staging copy, a test fixture) instead of pointing back at the real repo.
    """
    lang_root = series_path.parent.parent          # knowledge/<lang>
    return os.path.relpath(
        lang_root / 'assets' / 'css' / 'knowledge-base.css', series_path
    ).replace(os.sep, '/')


def shared_js_href(series_path: Path) -> str:
    """Relative src from a series directory to the locale's keyboard-scrollers.js.

    The script (``knowledge/<lang>/assets/js/keyboard-scrollers.js``) gives every
    box that scrolls on its own (wide tables, code, display math) a Tab stop so
    keyboard users can scroll it; every knowledge page loads it with
    ``<script defer>`` right before ``</body>``. Derived from ``series_path`` for
    the same reason as ``shared_css_href``.
    """
    lang_root = series_path.parent.parent          # knowledge/<lang>
    return os.path.relpath(
        lang_root / 'assets' / 'js' / 'keyboard-scrollers.js', series_path
    ).replace(os.sep, '/')


def dojo_body_class(series_path: Path) -> str:
    """Body class that selects the Dojo's accent palette, e.g. ``dojo-fm``.

    This is the shared sheet's explicit, generator-intended accent hook. The
    other hook it offers keys off a ``<link rel="canonical">`` that this template
    does not emit, so without this class a generated page would silently fall
    back to the default accent - invisible for FM/MI/ML but wrong for MS and PI.
    """
    return 'dojo-' + series_path.parent.name.lower()


class MathUnderscorePreprocessor(Preprocessor):
    r"""Escape ``_`` inside ``$...$`` / ``$$...$$`` as the previous converter did.

    This is the previous converter's MathPreprocessor, kept so that a ``$...$``
    span that MathInlineProcessor decides is *not* math (dollar amounts around
    prose, links or tags) reaches Markdown exactly as before, underscores
    escaped. For real math the ``\_`` it adds is turned back into ``_`` by
    MathInlineProcessor's legacy escapes, so the TeX is unchanged.

    Unlike before it runs at priority 10, after fenced code and raw HTML blocks
    have been set aside by Markdown, and it skips single-line code spans, so
    those are never altered. It works line by line and cannot tell an indented
    code block (or code in a block quote, or a code span spread over lines)
    from a list paragraph, so there it behaves exactly as before: "_" inside a
    "$...$" gets a backslash. Use fenced code for code that contains "$".
    """

    # Fenced code block delimiter: three or more backticks or tildes, optionally
    # followed by an info string (``` ```python ```). A closing fence is a run of
    # the same character, at least as long, with nothing after it.
    FENCE_RE = re.compile(r'^(?P<fence>`{3,}|~{3,})(?P<info>.*)$')

    # Math spans inside a prose line. The ``$$...$$`` alternative comes first so
    # that mid-line display math is matched as a whole rather than as two
    # adjacent ``$`` delimiters.
    INLINE_MATH_RE = re.compile(r'(\$\$[^$]+\$\$|\$[^$]+\$)')

    @staticmethod
    def _escape_underscores(text: str) -> str:
        """Escape underscores so Markdown does not read them as emphasis."""
        return text.replace('_', r'\_')

    CODE_SPAN_RE = re.compile(r'(`+)(?:.+?)(?<!`)\1(?!`)')

    @classmethod
    def _protect_inline_math(cls, text: str) -> str:
        """Escape underscores inside inline ``$...$`` spans, skipping code spans."""
        out, pos = [], 0
        for code in cls.CODE_SPAN_RE.finditer(text):
            out.append(cls._protect_math_spans(text[pos:code.start()]))
            out.append(code.group(0))
            pos = code.end()
        out.append(cls._protect_math_spans(text[pos:]))
        return ''.join(out)

    @classmethod
    def _protect_math_spans(cls, text: str) -> str:
        parts = cls.INLINE_MATH_RE.split(text)
        protected_parts = []
        for part in parts:
            if part.startswith('$') and part.endswith('$') and len(part) > 2:
                protected_parts.append(cls._escape_underscores(part))
            else:
                protected_parts.append(part)
        return ''.join(protected_parts)

    def run(self, lines: List[str]) -> List[str]:
        """
        Process lines to protect LaTeX math notation from Markdown parsing.

        Args:
            lines: Input Markdown lines

        Returns:
            Processed lines with protected math notation
        """
        new_lines = []
        in_display_math = False
        fence = None  # active fence marker while inside a fenced code block

        for line in lines:
            stripped = line.strip()
            fence_match = self.FENCE_RE.match(stripped)

            # --- Fenced code blocks: emit verbatim, never math-process ------
            if fence is not None:
                if (fence_match
                        and fence_match.group('fence')[0] == fence[0]
                        and len(fence_match.group('fence')) >= len(fence)
                        and not fence_match.group('info').strip()):
                    fence = None  # closing fence
                new_lines.append(line)
                continue
            if fence_match:
                fence = fence_match.group('fence')  # opening fence
                new_lines.append(line)
                continue

            # --- Inside a multi-line display block --------------------------
            if in_display_math:
                end = line.find('$$')
                if not stripped:
                    # Blank line: the block cannot continue past a paragraph
                    # break, so treat it as closed (guards against a truncated
                    # equation swallowing the rest of the document).
                    in_display_math = False
                    new_lines.append(line)
                elif end == -1:
                    new_lines.append(self._escape_underscores(line))
                else:
                    # Closing delimiter: math up to it, normal text after it.
                    in_display_math = False
                    new_lines.append(
                        self._escape_underscores(line[:end])
                        + '$$'
                        + self._protect_inline_math(line[end + 2:])
                    )
                continue

            # --- Display math starting at the beginning of the line ---------
            if stripped.startswith('$$'):
                open_at = line.find('$$')
                body_at = open_at + 2
                close_at = line.find('$$', body_at)
                head = line[:body_at]  # indentation + opening '$$'
                if close_at >= body_at:
                    # Self-contained equation: protect the body, keep any trailing
                    # text as prose, and do NOT flip the state.
                    new_lines.append(
                        head
                        + self._escape_underscores(line[body_at:close_at])
                        + '$$'
                        + self._protect_inline_math(line[close_at + 2:])
                    )
                else:
                    # Unmatched '$$': opening delimiter of a multi-line block. Any
                    # content on the same line is already math.
                    in_display_math = True
                    new_lines.append(
                        head + self._escape_underscores(line[body_at:])
                    )
                continue

            # --- Plain prose: protect inline math spans only ----------------
            new_lines.append(self._protect_inline_math(line))

        return new_lines


class MathInlineProcessor(InlineProcessor):
    r"""Keep LaTeX math away from Markdown's inline processing.

    Registered just after code spans (backtick, 190) and before backslash escapes
    (escape, 180), so it only ever sees text that Markdown treats as prose:
    fenced and indented code, code spans and raw HTML blocks never reach it.
    Each math span is stored in ``md.htmlStash`` and put back after rendering,
    so emphasis, links, attr_list and the like cannot touch it.

    Recognised spans (newlines allowed, as long as they stay in one paragraph):
      * ``$$...$$`` and ``$...$`` (``\$`` is a literal dollar, never a delimiter);
      * ``\\[...\\]`` and ``\\(...\\)`` - the Markdown-escaped spelling of the TeX
        delimiters, emitted as ``\[...\]`` and ``\(...\)``;
      * ``\begin{env}...\end{env}`` (or ``\\begin``) outside any delimiter.

    Inside a span four backslash escapes keep the meaning Markdown gave them,
    because existing sources rely on it: ``\\`` is one backslash (``\\{`` gives
    ``\{``, ``\\\\`` gives ``\\``), and ``\*``, ``\_`` and ``\|`` give ``*``, ``_``
    and ``|``. Every other backslash is TeX and is kept as written (``\{``,
    ``\!``, ``\,``), and a bare ``*`` or ``_`` is just a character. In ``$...$``
    MathUnderscorePreprocessor has already turned every ``_`` into ``\_``, so a
    bare ``_`` comes out as ``_`` and a written ``\_`` stays ``\_`` (as in
    ``\text{avg\_dev}``), exactly as before.

    Not math, and left to Markdown as before: a candidate that would swallow a
    code span or anything else Markdown has already set aside, one that
    overlaps an inline HTML tag or autolink (``<a href="$x$">``), a ``$...$``
    that runs out of a link label (``[price $5](url) and $6``), and a ``$...$``
    with no TeX command that contains ``[`` or ``<`` or other Markdown
    structure (``**`` outside braces, an escaped list dash, a run of words) -
    the prose between two dollar amounts. So math with no TeX command that uses
    ``[`` or ``<`` (``$[A^*][B]$``, ``$a<b$``) is processed as before; put a TeX
    command in it (``\lbrack``, ``\lt``, ``\mathrm{..}``) to protect it.
    Single-backslash ``\(`` and ``\[`` are not delimiters in Markdown text, as
    before (raw HTML, where they do work, is never touched).

    Not handled (same as the previous converter): a ``|`` inside math in a table
    cell still splits the cell, so write it as ``\|`` there; and lines inside a
    multi-line equation that look like Markdown blocks (``- ``, ``# ``, ``1. ``)
    are parsed as blocks before this processor runs.
    """

    MATH_RE = (
        r'(?P<dd>(?<!\\)\$\$(?P<dd_body>(?:\\.|[^$\\])+?)\$\$)'
        r'|(?P<bsdisp>\\\\\[(?P<bsdisp_body>[\s\S]+?)\\\\\])'
        r'|(?P<bsinl>\\\\\((?P<bsinl_body>[\s\S]+?)\\\\\))'
        r'|(?P<env>(?<!\\)\\{1,2}begin\{(?P<envname>[A-Za-z]+\*?)\}'
        r'[\s\S]+?\\{1,2}end\{(?P=envname)\})'
        r'|(?P<d>(?<!\\)\$(?P<d_body>(?:\\.|[^$\\])+?)\$)'
    )
    LEGACY_ESCAPE_RE = re.compile(r'\\([\\*_|])')
    TEX_COMMAND_RE = re.compile(r'\\[A-Za-z]')
    PROSE_SPAN_RE = re.compile(
        r'(?<![{^])\*\*(?!\})|\\- |[A-Za-z]{3,} [A-Za-z]{3,} [A-Za-z]{3,} [A-Za-z]{3,}'
        r'|[぀-ヿ一-鿿]{6,}'
    )
    # In a $...$ span with no TeX command, "[" or "<" means Markdown link or
    # HTML syntax (inline, reference or shortcut links, autolinks, tags) lies
    # between two dollar amounts, so the span is left to Markdown.
    MARKDOWN_SYNTAX_RE = re.compile(r'[\[<]')
    # An HTML tag as the HTML grammar has it (name, then attributes, then '>'),
    # so an inequality such as "a<b, c>d" is not a tag.
    HTML_TAG_RE = re.compile(
        r'<(?:/[A-Za-z][A-Za-z0-9-]*\s*'
        r'|[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z_:][-A-Za-z0-9_:.]*'
        r'(?:\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s"\'=<>`]+))?)*\s*/?)>'
        r'|<!--[\s\S]*?-->'
        # Markdown autolinks are not HTML but must not be swallowed either.
        r'|<(?:[A-Za-z][A-Za-z0-9+.-]*:[^<>\s]*|(?:mailto:)?[^<>\s!]+@[^<>\s@]+)>'
    )

    def __init__(self, md=None):
        super().__init__(self.MATH_RE, md)

    @staticmethod
    def escape_html(tex: str) -> str:
        """Escape for HTML text and attribute values alike (``alt``/``title``).

        An existing entity such as ``&lt;`` is left alone, as Markdown leaves it.
        """
        tex = re.sub(r'&(?!#?[A-Za-z0-9]+;)', '&amp;', tex)
        return tex.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

    def _touches_html_tag(self, data: str, start: int, end: int) -> bool:
        """True when the span and an inline HTML tag overlap at all.

        Covers ``<a href="$x$">`` (the span is in an attribute value),
        ``$5 <a href="/x">link</a> $6`` (it would swallow real tags) and
        ``$5 <span title='$6 tax'>`` (it would end inside a tag).
        """
        for tag in self.HTML_TAG_RE.finditer(data):
            if tag.start() >= end:
                break
            if tag.end() > start:
                return True
        return False

    def _crosses_link_label(self, data: str, start: int, body_at: int, end: int) -> bool:
        """True when the span runs out of a link label opened before it.

        "[price $5](url) and $6", "[price $5] [ref] and $6" and a shortcut
        reference "[price $5] and $6" (with "[price $5]: /url" defined): a ``[``
        opened earlier in the paragraph closes inside the span and what follows
        that ``]`` is link syntax, or the label is a defined reference. A
        half-open interval in a link label, "[$(0,1]$ intro](i.html)", is math.
        """
        opens = []
        i = 0
        while i < start:
            ch = data[i]
            if ch == '\\':
                i += 2
                continue
            if ch == '[':
                opens.append(i)
            elif ch == ']' and opens:
                opens.pop()
            i += 1
        if not opens:
            return False
        depth, i = 0, body_at
        while i < end:
            ch = data[i]
            if ch == '\\':
                i += 2
                continue
            depth += (ch == '[') - (ch == ']')
            if depth < 0:
                after = data[i + 1:]
                if re.match(r'\(|\s?\[', after):
                    return True
                label = re.sub(r'\s+', ' ', data[opens[-1] + 1:i]).strip().lower()
                return label in self.md.references
            i += 1
        return False

    def handleMatch(self, m, data):
        if util.STX in m.group(0) or self._touches_html_tag(data, m.start(0), m.end(0)):
            # The candidate swallows something Markdown already set aside (a code
            # span), sits in an attribute value or spans inline HTML: not math.
            return None, None, None
        if m.group('d') is not None:
            body = m.group('d_body')
            if self._crosses_link_label(data, m.start(0), m.start('d_body'), m.end(0)):
                return None, None, None  # "[cost $5](url) ... $6": link text, not math
            if not self.TEX_COMMAND_RE.search(body) and (
                    self.PROSE_SPAN_RE.search(body) or self.MARKDOWN_SYNTAX_RE.search(body)):
                return None, None, None  # dollar amounts around prose, a link or a tag
            tex = m.group('d')
        elif m.group('bsdisp') is not None:
            tex = '\\[' + m.group('bsdisp_body') + '\\]'
        elif m.group('bsinl') is not None:
            tex = '\\(' + m.group('bsinl_body') + '\\)'
        else:
            tex = m.group(0)
        tex = self.LEGACY_ESCAPE_RE.sub(r'\1', tex)
        return self.md.htmlStash.store(self.escape_html(tex)), m.start(0), m.end(0)


class MermaidPreprocessor(Preprocessor):
    """Preprocessor to convert Mermaid code blocks to div.mermaid."""

    def run(self, lines: List[str]) -> List[str]:
        """
        Convert Mermaid code blocks to HTML div elements.

        Args:
            lines: Input Markdown lines

        Returns:
            Processed lines with Mermaid blocks as div elements
        """
        new_lines = []
        in_mermaid = False
        mermaid_content = []

        for line in lines:
            if line.strip() == '```mermaid':
                in_mermaid = True
                mermaid_content = []
            elif in_mermaid and line.strip() == '```':
                # End of mermaid block - convert to div
                new_lines.append('<div class="mermaid">')
                new_lines.extend(mermaid_content)
                new_lines.append('</div>')
                in_mermaid = False
            elif in_mermaid:
                mermaid_content.append(line)
            else:
                new_lines.append(line)

        return new_lines


class MathExtension(Extension):
    """Extension that keeps LaTeX math away from Markdown processing."""

    def extendMarkdown(self, md):
        """Register MathInlineProcessor between code spans (190) and escapes (180).

        MathUnderscorePreprocessor runs last among the preprocessors (10), after
        fenced code (25) and raw HTML blocks (20) are set aside.
        """
        md.preprocessors.register(MathUnderscorePreprocessor(md), 'math_underscore', 10)
        md.inlinePatterns.register(MathInlineProcessor(md), 'math', 185)


class MermaidExtension(Extension):
    """Extension to add Mermaid preprocessing."""

    def extendMarkdown(self, md):
        """Register the MermaidPreprocessor."""
        md.preprocessors.register(MermaidPreprocessor(md), 'mermaid', 175)


def extract_frontmatter(content: str) -> Tuple[Dict, str]:
    """
    Extract YAML frontmatter from Markdown content.

    Args:
        content: Raw Markdown content with optional frontmatter

    Returns:
        Tuple of (frontmatter_dict, body_content)
    """
    match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if match:
        try:
            frontmatter = yaml.safe_load(match.group(1))
            body = content[match.end():]
            return frontmatter, body
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML frontmatter: {e}")
            return {}, content
    return {}, content


TABLE_SCROLL_OPEN = '<div class="table-scroll">'
TABLE_SCROLL_CLOSE = '</div>'


class _TableTagLocator(HTMLParser):
    """The token stream ``wrap_tables`` needs, with source offsets.

    ``tags`` lists, in source order, every ``<table>``/``</table>`` and
    ``<div>``/``</div>`` tag the HTML tokenizer sees, whitespace-only text and
    comments (which a wrapper may contain around its table), and an ``other``
    entry for anything else. ``html.parser`` reads ``<script>``/``<style>`` (and
    ``<textarea>``/``<title>``) content as text and skips comments, so a
    ``<table>`` written there, inside an attribute value, or escaped as
    ``&lt;table&gt;`` (fenced code, code spans) is not reported as a tag.
    """

    _WS = ' \t\n\f\r'                               # HTML's ASCII whitespace

    def __init__(self, text: str):
        super().__init__(convert_charrefs=False)
        self._text = text
        self._line_starts = [0] + [m.end() for m in re.finditer('\n', text)]
        # (kind, start, end, classes); kind: table-start, table-end, div-start,
        # div-end, space, comment, other
        self.tags: List[Tuple[str, int, int, Tuple[str, ...]]] = []
        self.feed(text)
        self.close()

    def _offset(self) -> int:
        line, col = self.getpos()                    # start of the token being handled
        return self._line_starts[line - 1] + col

    def handle_starttag(self, tag, attrs):
        start = self._offset()
        end = start + len(self.get_starttag_text())
        if tag in ('table', 'div'):
            classes = ()
            if tag == 'div':
                value = next((v for k, v in attrs if k == 'class'), None) or ''
                classes = tuple(value.split())
            self.tags.append((tag + '-start', start, end, classes))
        else:
            self.tags.append(('other', start, end, ()))

    def handle_startendtag(self, tag, attrs):        # "<table/>": HTML ignores the slash
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        start = self._offset()
        end = self._text.index('>', start) + 1
        self.tags.append((tag + '-end' if tag in ('table', 'div') else 'other', start, end, ()))

    def handle_data(self, data):
        self.tags.append(('space' if not data.strip(self._WS) else 'other', 0, 0, ()))

    def handle_comment(self, data):
        self.tags.append(('comment', 0, 0, ()))

    def handle_entityref(self, name):
        self.tags.append(('other', 0, 0, ()))

    handle_charref = handle_decl = handle_pi = unknown_decl = handle_entityref


def _in_wrapper(tags, i: int, j: int) -> bool:
    """True when the table whose start tag is ``tags[i]`` and end tag ``tags[j]``
    sits alone in a ``div`` whose class list contains ``table-scroll``: only
    whitespace and comments between that div's start tag and ``<table``, and
    between ``</table>`` and the next ``</div>``."""
    skip = ('space', 'comment')
    k = i - 1
    while k >= 0 and tags[k][0] in skip:
        k -= 1
    if k < 0 or tags[k][0] != 'div-start' or 'table-scroll' not in tags[k][3]:
        return False
    k = j + 1
    while k < len(tags) and tags[k][0] in skip:
        k += 1
    return k < len(tags) and tags[k][0] == 'div-end'


def wrap_tables(html: str) -> str:
    """Wrap every outermost ``<table>`` in ``<div class="table-scroll">``.

    The wrapper is what lets a wide table scroll sideways on a narrow screen while
    the table itself keeps ``display: table`` (knowledge-base.css section 19 gives
    ``.table-scroll`` its ``overflow-x: auto`` below 768px only; above that the div
    has no declarations at all). Exactly the two strings are inserted, directly
    before the ``<table`` start tag and directly after the matching ``</table>``;
    no other byte changes. Tables nested in another table are left alone, as is a
    table whose closing tag is missing. A table that already sits alone in a
    ``table-scroll`` div is skipped, whatever the div's formatting (line breaks,
    indentation or comments around the table, other attributes, more classes,
    either quote style), so running this twice adds nothing.

    The same function produced the wrappers in the existing pages, so a page whose
    body was generated by this converter still equals its regenerated body.
    """
    tags = _TableTagLocator(html).tags
    spans = []
    depth = 0
    first = 0
    for idx, (kind, s, e, _) in enumerate(tags):
        if kind == 'table-start':
            if depth == 0:
                first = idx
            depth += 1
        elif kind == 'table-end' and depth:
            depth -= 1
            if depth == 0 and not _in_wrapper(tags, first, idx):
                spans.append((tags[first][1], e))
    out = []
    pos = 0
    for s, e in spans:
        out += [html[pos:s], TABLE_SCROLL_OPEN, html[s:e], TABLE_SCROLL_CLOSE]
        pos = e
    out.append(html[pos:])
    return ''.join(out)


def convert_markdown_to_html(md_content: str) -> str:
    """
    Convert Markdown content to HTML using Python-Markdown library.

    Tables (Markdown tables and raw-HTML ones alike) come out wrapped in
    ``<div class="table-scroll">``; see ``wrap_tables``.

    Args:
        md_content: Markdown content (without frontmatter)

    Returns:
        Converted HTML content
    """
    # Configure Markdown processor with extensions
    md = markdown.Markdown(
        extensions=[
            MathExtension(),    # Math block protection (MUST be first - priority 200)
            MermaidExtension(), # Custom Mermaid preprocessing (priority 175)
            'tables',           # GitHub-flavored tables support
            'fenced_code',      # Fenced code blocks with language
            'sane_lists',       # Better list handling
            'attr_list',        # Add attributes to elements
        ],
        extension_configs={
            'fenced_code': {
                'lang_prefix': 'language-'
            }
        }
    )

    # Convert Markdown to HTML
    html = md.convert(md_content)

    return wrap_tables(html)


def create_navigation(chapter_num: int, series_path: Path, current_file: str, loc: Dict) -> str:
    """
    Create navigation links for chapter.

    Args:
        chapter_num: Current chapter number
        series_path: Path to series directory
        current_file: Name of current Markdown file

    Returns:
        Navigation HTML
    """
    nav_html = '<div class="navigation">\n'

    # Get all chapter HTML files in the series (sorted)
    chapter_html_files = sorted([f.name for f in series_path.glob("chapter*.html")])

    # Find current file index
    current_html = current_file.replace('.md', '.html')
    try:
        current_idx = chapter_html_files.index(current_html)
    except ValueError:
        # If current file not in list yet (being generated), estimate position
        current_idx = chapter_num - 1

    # Previous chapter
    if current_idx > 0 and len(chapter_html_files) > current_idx:
        prev_file = chapter_html_files[current_idx - 1]
        nav_html += f'    <a href="{prev_file}" class="nav-button">{loc["nav_prev"]}</a>\n'

    # Index
    nav_html += f'    <a href="index.html" class="nav-button">{loc["nav_index"]}</a>\n'

    # Next chapter (estimate next file name)
    # Try to find next chapter MD file
    next_chapter_files = sorted(series_path.glob(f"chapter*{chapter_num+1}*.md"))
    if next_chapter_files:
        next_html = next_chapter_files[0].name.replace('.md', '.html')
        nav_html += f'    <a href="{next_html}" class="nav-button">{loc["nav_next"]}</a>\n'

    nav_html += '</div>'
    return nav_html


def convert_chapter(series_path: Path, chapter_file: str, loc: Dict) -> bool:
    """
    Convert a single chapter Markdown file to HTML.

    Args:
        series_path: Path to series directory
        chapter_file: Name of chapter Markdown file

    Returns:
        True if successful, False otherwise
    """
    md_path = series_path / chapter_file
    html_path = series_path / chapter_file.replace('.md', '.html')

    logger.info(f"Converting {md_path} to {html_path}...")

    try:
        # Read Markdown
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Extract frontmatter
        frontmatter, body = extract_frontmatter(md_content)

        # Convert body to HTML
        body_html = convert_markdown_to_html(body)

        # Extract chapter number from filename (supports both patterns)
        # Pattern 1: chapter-1.md, chapter-2.md
        # Pattern 2: chapter1-introduction.md, chapter2-fundamentals.md
        chapter_match = re.match(r'chapter-?(\d+)', chapter_file)
        chapter_num = int(chapter_match.group(1)) if chapter_match else 1

        # Create navigation
        nav_html = create_navigation(chapter_num, series_path, chapter_file, loc)

        # Build complete HTML
        html = HTML_HEADER_TEMPLATE.format(
            css_href=shared_css_href(series_path),
            dojo_class=dojo_body_class(series_path),
            title=frontmatter.get('title', 'Chapter'),
            chapter_title=frontmatter.get('chapter_title', frontmatter.get('title', 'Chapter')),
            subtitle=frontmatter.get('subtitle', ''),
            reading_time=frontmatter.get('reading_time', loc['default_reading_time']),
            difficulty=frontmatter.get('difficulty', loc['default_difficulty']),
            code_examples=frontmatter.get('code_examples', 0),
            exercises=frontmatter.get('exercises', 0),
            **{k: v for k, v in loc.items() if k.startswith(('html_', 'label_', 'unit_'))}
        )

        html += body_html
        html += nav_html
        html += HTML_FOOTER_TEMPLATE.format(
            js_src=shared_js_href(series_path),
            version=frontmatter.get('version', '1.0'),
            created_at=frontmatter.get('created_at', loc['default_created_at']),
            supervisor=loc['supervisor'],
            **{k: v for k, v in loc.items() if k.startswith('label_')}
        )

        # Write HTML atomically (write to temp, then rename)
        temp_path = html_path.with_suffix('.html.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(html)

        # Rename temp to final
        temp_path.replace(html_path)

        logger.info(f"✓ Created {html_path}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to convert {chapter_file}: {e}")
        return False


def find_series_directories(dojo: str, lang: str) -> List[Path]:
    """
    Find all series directories within a Dojo.

    Args:
        dojo: Dojo name (FM, MI, ML, MS, PI)
        lang: Locale key (en or jp)

    Returns:
        List of series directory paths
    """
    dojo_path = KNOWLEDGE_ROOT / lang / dojo
    if not dojo_path.exists():
        logger.warning(f"Dojo directory not found: {dojo_path}")
        return []

    # Find all directories that contain chapter*.md files
    series_dirs = []
    for item in dojo_path.iterdir():
        if item.is_dir() and list(item.glob("chapter*.md")):
            series_dirs.append(item)

    return sorted(series_dirs)


def convert_series(series_path: Path, loc: Dict) -> Tuple[int, int]:
    """
    Convert all chapters in a series.

    Args:
        series_path: Path to series directory

    Returns:
        Tuple of (successful_count, total_count)
    """
    logger.info(f"\nProcessing series: {series_path.name}")
    logger.info("-" * 60)

    # Find all chapter*.md files
    chapter_files = sorted(series_path.glob("chapter*.md"))

    if not chapter_files:
        logger.warning(f"No chapter files found in {series_path.name}")
        return 0, 0

    success_count = 0
    for chapter_path in chapter_files:
        if convert_chapter(series_path, chapter_path.name, loc):
            success_count += 1

    return success_count, len(chapter_files)


def infer_lang(target: Optional[str]) -> Optional[str]:
    """Infer the locale from a target path (looks for en/jp under knowledge/)."""
    if not target:
        return None
    parts = Path(target).resolve().parts
    for i, part in enumerate(parts):
        if part == "knowledge" and i + 1 < len(parts) and parts[i + 1] in LOCALES:
            return parts[i + 1]
    # Fallback: a bare en/jp path component anywhere
    for part in Path(target).parts:
        if part in LOCALES:
            return part
    return None


def main(target: Optional[str] = None, lang: Optional[str] = None):
    """
    Main conversion function.

    Args:
        target: Optional target (Dojo name, series path, or file path)
        lang: Locale key (en or jp); inferred from target path when omitted
    """
    lang = lang or infer_lang(target) or "en"
    loc = LOCALES[lang]
    logger.info(f"Starting Markdown to HTML conversion ({lang} knowledge base)...")
    logger.info("=" * 60)

    total_success = 0
    total_files = 0

    if target:
        target_path = Path(target)

        # Check if target is a specific file
        if target_path.suffix == '.md' and target_path.exists():
            series_path = target_path.parent
            if convert_chapter(series_path, target_path.name, loc):
                total_success = 1
            total_files = 1
        # Check if target is a series directory
        elif target_path.is_dir() and list(target_path.glob("chapter*.md")):
            success, total = convert_series(target_path, loc)
            total_success += success
            total_files += total
        # Check if target is a Dojo
        elif target.upper() in DOJOS:
            series_dirs = find_series_directories(target.upper(), lang)
            for series_path in series_dirs:
                success, total = convert_series(series_path, loc)
                total_success += success
                total_files += total
        else:
            logger.error(f"Invalid target: {target}")
            logger.error("Usage: python convert_md_to_html.py [FM|MI|ML|MS|PI|series_path|file_path] [--lang en|jp]")
            sys.exit(1)
    else:
        # Process all Dojos in the selected locale
        for dojo in DOJOS:
            series_dirs = find_series_directories(dojo, lang)
            for series_path in series_dirs:
                success, total = convert_series(series_path, loc)
                total_success += success
                total_files += total

    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Conversion complete!")
    logger.info(f"Successfully converted: {total_success}/{total_files} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert knowledge-base Markdown to HTML (en/jp)")
    parser.add_argument("target", nargs="?", help="Dojo name, series directory, or .md file")
    parser.add_argument("--lang", choices=sorted(LOCALES), default=None,
                        help="Locale (inferred from target path when omitted; defaults to en)")
    args = parser.parse_args()
    main(args.target, args.lang)
