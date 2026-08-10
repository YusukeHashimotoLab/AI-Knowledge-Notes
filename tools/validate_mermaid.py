#!/usr/bin/env python3
"""
Mermaid Diagram Validation Tool

このスクリプトは、変換されたHTMLファイル内のMermaidダイアグラムを検証し、
構文エラーや不完全なブロックを検出します。

使用方法:
    python validate_mermaid.py              # 全HTMLファイルを検証
    python validate_mermaid.py <directory>  # 指定ディレクトリのみ検証
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
import sys


# --- Contamination detection -------------------------------------------------
#
# The block extractor uses a non-greedy `<div class="mermaid">(.*?)</div>`
# regex. That is correct for well-formed pages, but it has a blind spot: if a
# block is closed with something that is NOT `</div>` -- e.g. a bogus
# type-named tag such as `</flowchart>` / `</graph>` / `</mermaid>` /
# `</timeline>` / `</sequenceDiagram>`, or a leftover markdown fence -- the
# regex simply keeps scanning and re-closes on a LATER `</div>`, swallowing all
# the page markup in between into the "diagram" body. Because only the FIRST
# line of the body is type-checked, such a block passed validation silently
# (29 real pages were affected before this was found by hand).
#
# The rules below make that bug class loud: a Mermaid body is pure Mermaid
# text, so any closing HTML tag, markdown fence, or block-level HTML open tag
# inside it means the div swallowed page content.
#
# Calibration against the live corpus (1,952 mermaid divs) -- the only
# tag-like tokens that legitimately occur inside diagram bodies are:
#   <br/> (3,855)  <br> (66)  '< br>' (23)  <sub>/</sub> (20)  <sup>/</sup> (2)
#   <-->  (38, a mermaid bidirectional edge, not a tag)
#   <sos>/</sos> (3, a literal seq2seq label token, see below)
# and zero markdown fences. Inline formatting tags are therefore allowlisted.

# Inline formatting / line-break tags that may legitimately appear in a node
# label. Never reported.
INLINE_ALLOWED_TAGS = {
    'br', 'b', 'i', 'u', 's', 'em', 'strong', 'sub', 'sup',
    'small', 'span', 'code', 'kbd', 'mark', 'wbr',
}

# Block-level / structural / document HTML tags. An OPEN tag from this set, or
# a CLOSING tag from this set, inside a Mermaid body means page markup was
# swallowed -> ERROR.
BLOCK_LEVEL_TAGS = {
    'html', 'head', 'body', 'div', 'p',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'dl', 'dt', 'dd',
    'table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th', 'caption', 'colgroup',
    'section', 'article', 'header', 'footer', 'nav', 'aside', 'main',
    'blockquote', 'pre', 'figure', 'figcaption', 'hr',
    'form', 'fieldset', 'legend', 'label', 'input', 'button', 'select',
    'option', 'textarea',
    'script', 'style', 'link', 'meta', 'title', 'template', 'noscript',
    'iframe', 'img', 'picture', 'source', 'video', 'audio', 'canvas', 'svg',
    'details', 'summary', 'dialog', 'a',
}

# Mermaid diagram type names. Doubles as the source for the bogus type-named
# closing tags (`</flowchart>`, `</sequenceDiagram>`, ...) plus the `</mermaid>`
# variant, so the two can never drift apart.
MERMAID_DIAGRAM_TYPES = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram',
                         'stateDiagram-v2', 'stateDiagram', 'erDiagram', 'gantt',
                         'pie', 'timeline', 'journey', 'gitGraph', 'mindmap',
                         'quadrantChart', 'xychart-beta', 'sankey-beta',
                         'requirementDiagram', 'C4Context', 'block-beta']
BOGUS_CLOSER_TAGS = {t.lower() for t in MERMAID_DIAGRAM_TYPES} | {'mermaid'}

# `</name>` anywhere in the body. Deliberately strict about the `</` being
# adjacent to the name: mermaid labels are full of `<` comparisons and `-->`
# arrows, and a loose pattern would start matching them.
_CLOSING_TAG_RE = re.compile(r'</([A-Za-z][A-Za-z0-9-]*)\s*>')

# `<name ...>` for block-level tags only. Both anchors matter for zero false
# positives, because mermaid labels legitimately contain `<` comparisons next to
# `-->` arrows:
#   * no whitespace is allowed after `<`, so `A[ρ < a] --> B` is not an `<a>` tag
#   * the lookahead demands real tag syntax after the name, so `A[x <p] --> B`
#     is not a `<p>` tag while `<p>`, `<p class="x">` and `<br/>` still match
# Longest-first alternation keeps `<h1>` from being read as an `<h...>` prefix,
# and `<param>`-style names never match a shorter tag name.
_BLOCK_OPEN_TAG_RE = re.compile(
    r'<(' + '|'.join(sorted(BLOCK_LEVEL_TAGS, key=len, reverse=True)) + r')(?=[\s/>])[^>]*>',
    re.IGNORECASE,
)

# Markdown code fences (``` / ~~~) left behind by the markdown -> HTML step.
_FENCE_RE = re.compile(r'```|~~~')


class MermaidValidator:
    """Mermaid diagram validator for HTML files."""

    def __init__(self, base_dir: Path = None):
        """
        Initialize validator.

        Args:
            base_dir: Base directory to search for HTML files (default: current directory)
        """
        # Resolve base_dir to an absolute path so that relative_to() comparisons
        # in print_report() have a consistent absolute reference. Scanned file
        # paths are also resolved (see validate_all) to match this.
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        self.errors = []
        self.warnings = []
        self.total_diagrams = 0

    def extract_mermaid_blocks(self, html_content: str, file_path: Path) -> List[Tuple[int, str]]:
        """
        Extract all Mermaid blocks from HTML content.

        Args:
            html_content: HTML file content
            file_path: Path to the HTML file (for error reporting)

        Returns:
            List of (line_number, mermaid_content) tuples
        """
        # NOTE: The previous implementation was line-based and assumed every
        # Mermaid block spanned multiple lines with the opening <div ...> tag,
        # the diagram body, and the closing </div> each on their own line. Many
        # of our generated pages emit the whole block inline on a single line
        # (e.g. `<div class="mermaid">flowchart TD; A-->B</div>`) or place the
        # diagram text on the same line as the opening tag. The old logic never
        # captured inline content and never detected a same-line </div>, so it
        # silently swallowed following markup (<h3>, <pre>, ...) as if it were
        # the diagram, producing false "No valid diagram type" errors.
        #
        # Mermaid diagram bodies never contain a nested <div>, so the first
        # </div> after an opening tag reliably closes the block. We therefore
        # scan with a DOTALL regex that handles inline, single-line, and
        # multi-line blocks uniformly, and derive the 1-based start line from
        # the match offset for error reporting.
        blocks = []
        pattern = re.compile(r'<div class="mermaid">(.*?)</div>', re.DOTALL)
        matched = 0
        for match in pattern.finditer(html_content):
            matched += 1
            start_line = html_content.count('\n', 0, match.start()) + 1
            blocks.append((start_line, match.group(1).strip()))

        # Any opening tag without a matching </div> is an unclosed block.
        open_count = html_content.count('<div class="mermaid">')
        if open_count > matched:
            # Report the first unclosed opening tag position.
            idx = -1
            for _ in range(matched + 1):
                idx = html_content.find('<div class="mermaid">', idx + 1)
            start_line = html_content.count('\n', 0, idx) + 1 if idx >= 0 else 0
            self.errors.append({
                'file': file_path,
                'line': start_line,
                'type': 'unclosed_block',
                'message': 'Mermaid block is not closed (missing </div>)'
            })

        return blocks

    def detect_contamination(self, content: str) -> List[Dict]:
        """
        Detect page markup that leaked into a Mermaid diagram body.

        A Mermaid body is pure diagram text. Anything below means the
        ``<div class="mermaid">`` was closed by something other than ``</div>``
        (a bogus type-named tag, a stray markdown fence) and the non-greedy
        extractor re-closed on a later ``</div>``, absorbing page markup:

        * a markdown code fence (``` or ~~~)                  -> error
        * a block-level HTML open tag (<p, <div, <h3, <ul...) -> error
        * a closing tag for a block-level HTML element        -> error
        * a closing tag named after a Mermaid diagram type
          (</flowchart>, </graph>, </mermaid>, </timeline>,
          </sequenceDiagram>)                                -> error
        * any other closing tag that is not inline formatting -> warning

        Inline formatting tags (``<br/>``, ``<br>``, ``<sub>``, ``<sup>``,
        ``<b>``, ``<i>``, ...) are allowlisted because Mermaid node labels use
        them legitimately; ``INLINE_ALLOWED_TAGS`` documents the full set.

        Each rule reports at most once per block: a swallowed body holds dozens
        of tags that all resolve to the same reported line, and one actionable
        pointer per diagram beats a wall of duplicates.

        The last rule is a warning rather than an error on purpose: three live
        pages carry a literal ``</sos>`` seq2seq label token that an HTML tidy
        pass balanced into the body. It is junk, but it is pre-existing junk,
        so it must not turn the corpus red - it surfaces as a warning instead.

        Args:
            content: Mermaid diagram body as extracted from the div

        Returns:
            List of issue dicts ({'severity', 'message'}), possibly empty
        """
        issues = []
        if not content:
            return issues

        def _line_of(offset: int) -> int:
            """1-based line number *within the diagram body*."""
            return content.count('\n', 0, offset) + 1

        # 1. Markdown fences never belong in a rendered Mermaid body.
        for m in _FENCE_RE.finditer(content):
            issues.append({
                'severity': 'error',
                'message': (
                    f'Markdown code fence "{m.group(0)}" inside Mermaid body '
                    f'(body line {_line_of(m.start())}) - the block was not '
                    f'closed properly and swallowed page content'
                ),
            })
            break  # one report per block is enough

        # 2. Block-level HTML open tags.
        for m in _BLOCK_OPEN_TAG_RE.finditer(content):
            issues.append({
                'severity': 'error',
                'message': (
                    f'Block-level HTML tag "{m.group(0)[:40]}" inside Mermaid '
                    f'body (body line {_line_of(m.start())}) - the block '
                    f'swallowed page markup'
                ),
            })
            break

        # 3. Closing tags. A swallowed block typically contains dozens of them
        # (</li>, </td>, </p>, ...), all at the same reported line, so emit at
        # most one error and one warning per block. A bogus type-named closer is
        # the most diagnostic finding, so it wins over a generic block closer.
        bogus_hit = None
        block_hit = None
        unknown_hit = None
        for m in _CLOSING_TAG_RE.finditer(content):
            name = m.group(1)
            lowered = name.lower()
            if lowered in INLINE_ALLOWED_TAGS:
                continue
            hit = (name, _line_of(m.start()))
            if lowered in BOGUS_CLOSER_TAGS:
                bogus_hit = bogus_hit or hit
            elif lowered in BLOCK_LEVEL_TAGS:
                block_hit = block_hit or hit
            else:
                unknown_hit = unknown_hit or hit

        if bogus_hit:
            name, body_line = bogus_hit
            issues.append({
                'severity': 'error',
                'message': (
                    f'Bogus type-named closing tag "</{name}>" inside Mermaid '
                    f'body (body line {body_line}) - the block must be closed '
                    f'with </div>; the extractor re-closed on a later </div> '
                    f'and swallowed page content'
                ),
            })
        elif block_hit:
            name, body_line = block_hit
            issues.append({
                'severity': 'error',
                'message': (
                    f'Closing HTML tag "</{name}>" inside Mermaid body '
                    f'(body line {body_line}) - the block swallowed page markup'
                ),
            })

        if unknown_hit:
            name, body_line = unknown_hit
            issues.append({
                'severity': 'warning',
                'message': (
                    f'Unexpected closing tag "</{name}>" inside Mermaid body '
                    f'(body line {body_line}) - Mermaid bodies should contain '
                    f'only inline formatting tags '
                    f'({", ".join(sorted(INLINE_ALLOWED_TAGS))})'
                ),
            })

        return issues

    def validate_mermaid_syntax(self, content: str, file_path: Path, line_num: int) -> Dict:
        """
        Validate Mermaid diagram syntax.

        Args:
            content: Mermaid diagram content
            file_path: Path to the HTML file
            line_num: Line number where the diagram starts

        Returns:
            Dictionary with validation results
        """
        issues = []

        # Check for diagram type.
        # NOTE: 'xychart-beta' (and other newer types below) were missing from
        # the original list, so valid xychart diagrams were reported as errors.
        # This list must track the diagram types actually renderable by the
        # mermaid version pinned in the pages (mermaid@10.9.x).
        diagram_types = MERMAID_DIAGRAM_TYPES

        # Contamination checks run FIRST: a swallowed block usually still has a
        # valid-looking first line, so the type check alone cannot see it.
        issues.extend(self.detect_contamination(content))

        first_line = content.strip().split('\n')[0] if content.strip() else ''
        has_diagram_type = any(first_line.startswith(dtype) for dtype in diagram_types)

        if not has_diagram_type:
            issues.append({
                'severity': 'error',
                'message': f'No valid diagram type found (first line: "{first_line[:50]}...")'
            })

        # Check for common syntax issues
        lines = content.strip().split('\n')

        # Validate graph/flowchart direction for graph diagrams
        if first_line.startswith('graph') or first_line.startswith('flowchart'):
            parts = first_line.split()
            # Inline single-line blocks look like `flowchart TD; A-->B`; strip a
            # trailing ';' (and anything after it) so the direction token is
            # compared cleanly instead of e.g. 'TD;'.
            direction = parts[1].split(';')[0] if len(parts) >= 2 else ''
            if len(parts) < 2:
                issues.append({
                    'severity': 'warning',
                    'message': f'Graph/flowchart missing direction (TD, LR, etc.)'
                })
            elif direction not in ['TD', 'TB', 'BT', 'RL', 'LR']:
                issues.append({
                    'severity': 'warning',
                    'message': f'Unknown graph direction: {direction}'
                })

        # Check for style syntax
        style_lines = [l for l in lines if l.strip().startswith('style ')]
        for style_line in style_lines:
            # Basic style syntax check: style <node> <properties>
            parts = style_line.strip().split(maxsplit=2)
            if len(parts) < 3:
                issues.append({
                    'severity': 'warning',
                    'message': f'Incomplete style definition: "{style_line.strip()}"'
                })

        # Check for empty content
        if not content.strip():
            issues.append({
                'severity': 'error',
                'message': 'Mermaid block is empty'
            })

        return {
            'file': file_path,
            'line': line_num,
            'diagram_type': first_line.split()[0] if has_diagram_type else 'unknown',
            'issues': issues
        }

    def validate_file(self, html_file: Path) -> None:
        """
        Validate all Mermaid diagrams in a single HTML file.

        Args:
            html_file: Path to the HTML file
        """
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append({
                'file': html_file,
                'line': 0,
                'type': 'file_error',
                'message': f'Could not read file: {e}'
            })
            return

        # Extract Mermaid blocks
        blocks = self.extract_mermaid_blocks(content, html_file)
        self.total_diagrams += len(blocks)

        # Validate each block
        for line_num, mermaid_content in blocks:
            result = self.validate_mermaid_syntax(mermaid_content, html_file, line_num)

            # Categorize issues
            for issue in result['issues']:
                if issue['severity'] == 'error':
                    self.errors.append({
                        'file': html_file,
                        'line': line_num,
                        'type': 'syntax_error',
                        'diagram_type': result['diagram_type'],
                        'message': issue['message']
                    })
                else:
                    self.warnings.append({
                        'file': html_file,
                        'line': line_num,
                        'type': 'syntax_warning',
                        'diagram_type': result['diagram_type'],
                        'message': issue['message']
                    })

    def validate_all(self, target_dir: Path = None) -> None:
        """
        Validate all HTML files in the target directory.

        Args:
            target_dir: Directory to search (default: all series directories)
        """
        if target_dir:
            # Resolve to absolute paths so reported file paths share the same
            # absolute base as self.base_dir (avoids ValueError in relative_to).
            target_dir = Path(target_dir).resolve()
            html_files = [p.resolve() for p in target_dir.glob('**/*.html')]
        else:
            # Search all series directories
            html_files = []
            for series_dir in self.base_dir.glob('*-introduction'):
                html_files.extend(p.resolve() for p in series_dir.glob('*.html'))

        if not html_files:
            print(f"⚠ No HTML files found in {target_dir or self.base_dir}")
            return

        print(f"🔍 Validating Mermaid diagrams in {len(html_files)} HTML files...")
        print("=" * 70)

        for html_file in sorted(html_files):
            self.validate_file(html_file)

    def _display_path(self, file_path: Path) -> str:
        """
        Return a path for display relative to base_dir.

        Uses Path.relative_to() when file_path is inside base_dir, and falls
        back to os.path.relpath() otherwise (e.g. when the scan target lives on
        a different branch of the tree than base_dir, or on a different drive on
        Windows where relpath can still raise ValueError — then show absolute).
        """
        try:
            return str(Path(file_path).relative_to(self.base_dir))
        except ValueError:
            try:
                return os.path.relpath(str(file_path), str(self.base_dir))
            except ValueError:
                return str(file_path)

    def print_report(self) -> None:
        """Print validation report."""
        print(f"\n📊 Validation Report")
        print("=" * 70)
        print(f"Total Mermaid diagrams found: {self.total_diagrams}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print()

        # Print errors
        if self.errors:
            print("❌ ERRORS:")
            print("-" * 70)
            for error in self.errors:
                rel_path = self._display_path(error['file'])
                print(f"\n  File: {rel_path}:{error['line']}")
                print(f"  Type: {error['type']}")
                if 'diagram_type' in error:
                    print(f"  Diagram: {error['diagram_type']}")
                print(f"  Message: {error['message']}")
            print()

        # Print warnings
        if self.warnings:
            print("⚠️  WARNINGS:")
            print("-" * 70)
            for warning in self.warnings:
                rel_path = self._display_path(warning['file'])
                print(f"\n  File: {rel_path}:{warning['line']}")
                print(f"  Type: {warning['type']}")
                if 'diagram_type' in warning:
                    print(f"  Diagram: {warning['diagram_type']}")
                print(f"  Message: {warning['message']}")
            print()

        # Summary
        if not self.errors and not self.warnings:
            print("✅ All Mermaid diagrams passed validation!")
        elif not self.errors:
            print(f"✅ No critical errors found (but {len(self.warnings)} warnings)")
        else:
            print(f"❌ Found {len(self.errors)} error(s) that need attention")

        print("=" * 70)


def main():
    """Main entry point."""
    base_dir = Path.cwd()

    # Parse command-line arguments
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.exists():
            print(f"❌ Error: Directory '{target}' does not exist")
            sys.exit(1)
        validator = MermaidValidator(base_dir)
        validator.validate_all(target)
    else:
        validator = MermaidValidator(base_dir)
        validator.validate_all()

    # Print report
    validator.print_report()

    # Exit with error code if errors were found
    sys.exit(1 if validator.errors else 0)


if __name__ == '__main__':
    main()
