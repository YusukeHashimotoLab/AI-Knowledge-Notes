#!/usr/bin/env python3
"""
Convert Markdown files to HTML for English knowledge base.
Supports all Dojos (FM, MI, ML, MS, PI) in the knowledge/en/ directory.

This script converts Markdown files with YAML frontmatter to production-ready HTML
with MathJax support for equations, Mermaid for diagrams, and responsive styling.
"""

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

# Base path for English knowledge base
BASE_PATH = Path("knowledge/en")

# Supported Dojos
DOJOS = ["FM", "MI", "ML", "MS", "PI"]

# HTML template header
HTML_HEADER_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - AI Terakoya</title>

    <style>
        :root {{
            --color-primary: #2c3e50;
            --color-primary-dark: #1a252f;
            --color-accent: #7b2cbf;
            --color-accent-light: #9d4edd;
            --color-text: #2d3748;
            --color-text-light: #4a5568;
            --color-bg: #ffffff;
            --color-bg-alt: #f7fafc;
            --color-border: #e2e8f0;
            --color-code-bg: #f8f9fa;
            --color-link: #3182ce;
            --color-link-hover: #2c5aa0;

            --spacing-xs: 0.5rem;
            --spacing-sm: 1rem;
            --spacing-md: 1.5rem;
            --spacing-lg: 2rem;
            --spacing-xl: 3rem;

            --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;

            --border-radius: 8px;
            --box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: var(--font-body);
            line-height: 1.7;
            color: var(--color-text);
            background-color: var(--color-bg);
            font-size: 16px;
        }}

        header {{
            background: linear-gradient(135deg, var(--color-accent) 0%, var(--color-accent-light) 100%);
            color: white;
            padding: var(--spacing-xl) var(--spacing-md);
            margin-bottom: var(--spacing-xl);
            box-shadow: var(--box-shadow);
        }}

        .header-content {{
            max-width: 900px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: var(--spacing-sm);
            line-height: 1.2;
        }}

        .subtitle {{
            font-size: 1.1rem;
            opacity: 0.95;
            font-weight: 400;
            margin-bottom: var(--spacing-md);
        }}

        .meta {{
            display: flex;
            flex-wrap: wrap;
            gap: var(--spacing-md);
            font-size: 0.9rem;
            opacity: 0.9;
        }}

        .meta-item {{
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 0 var(--spacing-md) var(--spacing-xl);
        }}

        h2 {{
            font-size: 1.75rem;
            color: var(--color-primary);
            margin-top: var(--spacing-xl);
            margin-bottom: var(--spacing-md);
            padding-bottom: var(--spacing-xs);
            border-bottom: 3px solid var(--color-accent);
        }}

        h3 {{
            font-size: 1.4rem;
            color: var(--color-primary);
            margin-top: var(--spacing-lg);
            margin-bottom: var(--spacing-sm);
        }}

        h4 {{
            font-size: 1.1rem;
            color: var(--color-primary-dark);
            margin-top: var(--spacing-md);
            margin-bottom: var(--spacing-sm);
        }}

        p {{
            margin-bottom: var(--spacing-md);
            color: var(--color-text);
        }}

        a {{
            color: var(--color-link);
            text-decoration: none;
            transition: color 0.2s;
        }}

        a:hover {{
            color: var(--color-link-hover);
            text-decoration: underline;
        }}

        ul, ol {{
            margin-left: var(--spacing-lg);
            margin-bottom: var(--spacing-md);
        }}

        li {{
            margin-bottom: var(--spacing-xs);
            color: var(--color-text);
        }}

        pre {{
            background-color: var(--color-code-bg);
            border: 1px solid var(--color-border);
            border-radius: var(--border-radius);
            padding: var(--spacing-md);
            overflow-x: auto;
            margin-bottom: var(--spacing-md);
            font-family: var(--font-mono);
            font-size: 0.9rem;
            line-height: 1.5;
        }}

        code {{
            font-family: var(--font-mono);
            font-size: 0.9em;
            background-color: var(--color-code-bg);
            padding: 0.2em 0.4em;
            border-radius: 3px;
        }}

        pre code {{
            background-color: transparent;
            padding: 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: var(--spacing-md);
            font-size: 0.95rem;
        }}

        th, td {{
            border: 1px solid var(--color-border);
            padding: var(--spacing-sm);
            text-align: left;
        }}

        th {{
            background-color: var(--color-bg-alt);
            font-weight: 600;
            color: var(--color-primary);
        }}

        blockquote {{
            border-left: 4px solid var(--color-accent);
            padding-left: var(--spacing-md);
            margin: var(--spacing-md) 0;
            color: var(--color-text-light);
            font-style: italic;
            background-color: var(--color-bg-alt);
            padding: var(--spacing-md);
            border-radius: var(--border-radius);
        }}

        .mermaid {{
            text-align: center;
            margin: var(--spacing-lg) 0;
            background-color: var(--color-bg-alt);
            padding: var(--spacing-md);
            border-radius: var(--border-radius);
        }}

        details {{
            background-color: var(--color-bg-alt);
            border: 1px solid var(--color-border);
            border-radius: var(--border-radius);
            padding: var(--spacing-md);
            margin-bottom: var(--spacing-md);
        }}

        summary {{
            cursor: pointer;
            font-weight: 600;
            color: var(--color-primary);
            user-select: none;
            padding: var(--spacing-xs);
            margin: calc(-1 * var(--spacing-md));
            padding: var(--spacing-md);
            border-radius: var(--border-radius);
        }}

        summary:hover {{
            background-color: rgba(123, 44, 191, 0.1);
        }}

        details[open] summary {{
            margin-bottom: var(--spacing-md);
            border-bottom: 1px solid var(--color-border);
        }}

        .learning-objectives {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: var(--spacing-lg);
            border-radius: var(--border-radius);
            border-left: 4px solid var(--color-accent);
            margin-bottom: var(--spacing-xl);
        }}

        .learning-objectives h2 {{
            margin-top: 0;
            border-bottom: none;
        }}

        .navigation {{
            display: flex;
            justify-content: space-between;
            gap: var(--spacing-md);
            margin: var(--spacing-xl) 0;
            padding-top: var(--spacing-lg);
            border-top: 2px solid var(--color-border);
        }}

        .nav-button {{
            flex: 1;
            padding: var(--spacing-md);
            background: linear-gradient(135deg, var(--color-accent) 0%, var(--color-accent-light) 100%);
            color: white;
            border-radius: var(--border-radius);
            text-align: center;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: var(--box-shadow);
        }}

        .nav-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            text-decoration: none;
        }}

        footer {{
            margin-top: var(--spacing-xl);
            padding: var(--spacing-lg) var(--spacing-md);
            background-color: var(--color-bg-alt);
            border-top: 1px solid var(--color-border);
            text-align: center;
            font-size: 0.9rem;
            color: var(--color-text-light);
        }}

        @media (max-width: 768px) {{
            h1 {{
                font-size: 1.5rem;
            }}

            h2 {{
                font-size: 1.4rem;
            }}

            h3 {{
                font-size: 1.2rem;
            }}

            .meta {{
                font-size: 0.85rem;
            }}

            .navigation {{
                flex-direction: column;
            }}

            table {{
                font-size: 0.85rem;
            }}

            th, td {{
                padding: var(--spacing-xs);
            }}
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
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
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
<body>
    <header>
        <div class="header-content">
            <h1>{chapter_title}</h1>
            <p class="subtitle">{subtitle}</p>
            <div class="meta">
                <span class="meta-item">📖 Reading Time: {reading_time}</span>
                <span class="meta-item">📊 Difficulty: {difficulty}</span>
                <span class="meta-item">💻 Code Examples: {code_examples}</span>
                <span class="meta-item">📝 Exercises: {exercises}</span>
            </div>
        </div>
    </header>

    <main class="container">
'''

HTML_FOOTER_TEMPLATE = '''
    </main>

    <footer>
        <p><strong>Created by</strong>: AI Terakoya Content Team</p>
        <p><strong>Supervised by</strong>: Dr. Yusuke Hashimoto (Tohoku University)</p>
        <p><strong>Version</strong>: {version} | <strong>Created</strong>: {created_at}</p>
        <p><strong>License</strong>: Creative Commons BY 4.0</p>
        <p>© 2025 AI Terakoya. All rights reserved.</p>
    </footer>
</body>
</html>
'''


class MathPreprocessor(Preprocessor):
    """Preprocessor to protect math blocks from Markdown emphasis processing."""

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

        for line in lines:
            # Check for display math delimiters ($$...$$)
            if line.strip().startswith('$$'):
                in_display_math = not in_display_math
                new_lines.append(line)
            elif in_display_math:
                # Protect underscores in math mode by escaping them
                # This prevents Markdown from treating _ as emphasis
                protected_line = line.replace('_', r'\_')
                new_lines.append(protected_line)
            else:
                # Also protect inline math $...$
                # Use regex to find and protect inline math
                parts = re.split(r'(\$[^$]+\$)', line)
                protected_parts = []
                for part in parts:
                    if part.startswith('$') and part.endswith('$') and len(part) > 2:
                        # This is inline math - protect underscores
                        protected_parts.append(part.replace('_', r'\_'))
                    else:
                        protected_parts.append(part)
                new_lines.append(''.join(protected_parts))

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


def create_navigation(chapter_num: int, series_path: Path, current_file: str) -> str:
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
        nav_html += f'    <a href="{prev_file}" class="nav-button">← Previous Chapter</a>\n'

    # Index
    nav_html += '    <a href="index.html" class="nav-button">Back to Series Index</a>\n'

    # Next chapter (estimate next file name)
    # Try to find next chapter MD file
    next_chapter_files = sorted(series_path.glob(f"chapter*{chapter_num+1}*.md"))
    if next_chapter_files:
        next_html = next_chapter_files[0].name.replace('.md', '.html')
        nav_html += f'    <a href="{next_html}" class="nav-button">Next Chapter →</a>\n'

    nav_html += '</div>'
    return nav_html


def convert_chapter(series_path: Path, chapter_file: str) -> bool:
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
        nav_html = create_navigation(chapter_num, series_path, chapter_file)

        # Build complete HTML
        html = HTML_HEADER_TEMPLATE.format(
            title=frontmatter.get('title', 'Chapter'),
            chapter_title=frontmatter.get('chapter_title', frontmatter.get('title', 'Chapter')),
            subtitle=frontmatter.get('subtitle', ''),
            reading_time=frontmatter.get('reading_time', '20-25 minutes'),
            difficulty=frontmatter.get('difficulty', 'Beginner'),
            code_examples=frontmatter.get('code_examples', 0),
            exercises=frontmatter.get('exercises', 0)
        )

        html += body_html
        html += nav_html
        html += HTML_FOOTER_TEMPLATE.format(
            version=frontmatter.get('version', '1.0'),
            created_at=frontmatter.get('created_at', '2025-01-01')
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


def find_series_directories(dojo: str) -> List[Path]:
    """
    Find all series directories within a Dojo.

    Args:
        dojo: Dojo name (FM, MI, ML, MS, PI)

    Returns:
        List of series directory paths
    """
    dojo_path = BASE_PATH / dojo
    if not dojo_path.exists():
        logger.warning(f"Dojo directory not found: {dojo_path}")
        return []

    # Find all directories that contain chapter*.md files
    series_dirs = []
    for item in dojo_path.iterdir():
        if item.is_dir() and list(item.glob("chapter*.md")):
            series_dirs.append(item)

    return sorted(series_dirs)


def convert_series(series_path: Path) -> Tuple[int, int]:
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
        if convert_chapter(series_path, chapter_path.name):
            success_count += 1

    return success_count, len(chapter_files)


def main(target: Optional[str] = None):
    """
    Main conversion function.

    Args:
        target: Optional target (Dojo name, series path, or file path)
    """
    logger.info("Starting Markdown to HTML conversion for English knowledge base...")
    logger.info("=" * 60)

    total_success = 0
    total_files = 0

    if target:
        target_path = Path(target)

        # Check if target is a specific file
        if target_path.suffix == '.md' and target_path.exists():
            series_path = target_path.parent
            if convert_chapter(series_path, target_path.name):
                total_success = 1
            total_files = 1
        # Check if target is a series directory
        elif target_path.is_dir() and list(target_path.glob("chapter*.md")):
            success, total = convert_series(target_path)
            total_success += success
            total_files += total
        # Check if target is a Dojo
        elif target.upper() in DOJOS:
            series_dirs = find_series_directories(target.upper())
            for series_path in series_dirs:
                success, total = convert_series(series_path)
                total_success += success
                total_files += total
        else:
            logger.error(f"Invalid target: {target}")
            logger.error("Usage: python convert_md_to_html_en.py [FM|MI|ML|MS|PI|series_path|file_path]")
            sys.exit(1)
    else:
        # Process all Dojos
        for dojo in DOJOS:
            series_dirs = find_series_directories(dojo)
            for series_path in series_dirs:
                success, total = convert_series(series_path)
                total_success += success
                total_files += total

    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Conversion complete!")
    logger.info(f"Successfully converted: {total_success}/{total_files} files")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
