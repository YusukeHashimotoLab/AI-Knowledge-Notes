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
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import markdown
from markdown.extensions import Extension
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


def dojo_body_class(series_path: Path) -> str:
    """Body class that selects the Dojo's accent palette, e.g. ``dojo-fm``.

    This is the shared sheet's explicit, generator-intended accent hook. The
    other hook it offers keys off a ``<link rel="canonical">`` that this template
    does not emit, so without this class a generated page would silently fall
    back to the default accent - invisible for FM/MI/ML but wrong for MS and PI.
    """
    return 'dojo-' + series_path.parent.name.lower()


class MathPreprocessor(Preprocessor):
    r"""Preprocessor to protect math blocks from Markdown emphasis processing.

    Underscores inside LaTeX math must not be seen by Markdown's emphasis
    parser, so they are escaped to ``\_`` before the block/inline parsers run.
    Two rules keep that escaping confined to actual math:

    1. A line that opens ``$$`` *and* closes it again on the same line is a
       *self-contained* display equation, e.g. ``$$ S = QK^T $$`` or
       ``$$\oint_C f(z) dz = 0$$ where $f(z)$ is analytic``. Its equation body is
       protected, any trailing prose is treated as prose, and the "inside display
       math" state is NOT flipped - the previous implementation toggled the flag
       once for such a line and never back, so every following line (prose *and*
       code) was treated as math and had its underscores escaped, corrupting
       Python examples in the output. Only a line with a single, unmatched ``$$``
       opens a multi-line display block.
    2. Fenced code blocks are skipped entirely. This preprocessor is registered
       at priority 200 while ``fenced_code`` registers its own preprocessor at
       priority 25, so the fences are still raw Markdown lines here and would
       otherwise be math-processed.
    3. A blank line ends an open display block. Markdown would split a ``$$``
       block containing a blank line into separate paragraphs (which MathJax
       cannot render anyway), so this costs nothing for valid content and keeps a
       truncated equation in the source - e.g. ``$$ P(y_t | y_{`` with the rest
       of the line lost - from putting the remainder of the document into math
       mode.
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

    @classmethod
    def _protect_inline_math(cls, text: str) -> str:
        """Escape underscores inside inline ``$...$`` / ``$$...$$`` spans only."""
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
    """Extension to protect math blocks from Markdown emphasis."""

    def extendMarkdown(self, md):
        """Register the MathPreprocessor with high priority."""
        md.preprocessors.register(MathPreprocessor(md), 'math', 200)


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


def convert_markdown_to_html(md_content: str) -> str:
    """
    Convert Markdown content to HTML using Python-Markdown library.

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

    return html


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
