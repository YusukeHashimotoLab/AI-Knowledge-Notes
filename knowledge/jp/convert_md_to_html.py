#!/usr/bin/env python3
"""
Convert Markdown files to HTML using the GNN template structure.
Converts chapters from the 4 new series (chemoinformatics, bioinformatics,
active-learning, data-driven-materials).
"""

import os
import re
import yaml
from pathlib import Path
import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor

# Series to convert
SERIES = [
    "mi-introduction",
    "pi-introduction",
    "nm-introduction",
    "chemoinformatics-introduction",
    "bioinformatics-introduction",
    "active-learning-introduction",
    "data-driven-materials-introduction",
    "transformer-introduction",
    "gnn-introduction",
    "bayesian-optimization-introduction",
    "computational-materials-basics-introduction",
    "reinforcement-learning-introduction",
    "robotic-lab-automation-introduction",
    "experimental-data-analysis-introduction",
    "high-throughput-computing-introduction",
    "materials-databases-introduction",
    "materials-applications-introduction",
    "battery-mi-application",
    "catalyst-mi-application",
    "materials-property-mapping-introduction"
]

BASE_PATH = Path("/wp/knowledge/jp/MI")

# HTML template header
HTML_HEADER_TEMPLATE = '''<!DOCTYPE html>
<html lang="ja">
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
                <span class="meta-item">📖 読了時間: {reading_time}</span>
                <span class="meta-item">📊 難易度: {difficulty}</span>
                <span class="meta-item">💻 コード例: {code_examples}個</span>
                <span class="meta-item">📝 演習問題: {exercises}問</span>
            </div>
        </div>
    </header>

    <main class="container">
'''

HTML_FOOTER_TEMPLATE = '''
    </main>

    <footer>
        <p><strong>作成者</strong>: AI Terakoya Content Team</p>
        <p><strong>監修</strong>: Dr. Yusuke Hashimoto（東北大学）</p>
        <p><strong>バージョン</strong>: {version} | <strong>作成日</strong>: {created_at}</p>
        <p><strong>ライセンス</strong>: Creative Commons BY 4.0</p>
        <p>© 2025 AI Terakoya. All rights reserved.</p>
    </footer>
</body>
</html>
'''


class MathPreprocessor(Preprocessor):
    """Preprocessor to protect math blocks from Markdown emphasis processing."""

    def run(self, lines):
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
                import re
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

    def run(self, lines):
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
        md.preprocessors.register(MathPreprocessor(md), 'math', 200)


class MermaidExtension(Extension):
    """Extension to add Mermaid preprocessing."""

    def extendMarkdown(self, md):
        md.preprocessors.register(MermaidPreprocessor(md), 'mermaid', 175)


def extract_frontmatter(content):
    """Extract YAML frontmatter from Markdown content."""
    match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if match:
        frontmatter = yaml.safe_load(match.group(1))
        body = content[match.end():]
        return frontmatter, body
    return {}, content


def convert_markdown_to_html(md_content):
    """Convert Markdown content to HTML using Python-Markdown library."""

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


def create_navigation(chapter_num, series_path, current_file):
    """Create navigation links for chapter."""
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
        nav_html += f'    <a href="{prev_file}" class="nav-button">← 前の章</a>\n'

    # Index
    nav_html += '    <a href="index.html" class="nav-button">シリーズ目次に戻る</a>\n'

    # Next chapter (estimate next file name)
    # Try to find next chapter MD file
    next_chapter_files = sorted(series_path.glob(f"chapter*{chapter_num+1}*.md"))
    if next_chapter_files:
        next_html = next_chapter_files[0].name.replace('.md', '.html')
        nav_html += f'    <a href="{next_html}" class="nav-button">次の章 →</a>\n'

    nav_html += '</div>'
    return nav_html


def convert_chapter(series_path, chapter_file):
    """Convert a single chapter Markdown file to HTML."""
    md_path = series_path / chapter_file
    html_path = series_path / chapter_file.replace('.md', '.html')

    print(f"Converting {md_path} to {html_path}...")

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
        reading_time=frontmatter.get('reading_time', '20-25分'),
        difficulty=frontmatter.get('difficulty', '初級'),
        code_examples=frontmatter.get('code_examples', 0),
        exercises=frontmatter.get('exercises', 0)
    )

    html += body_html
    html += nav_html
    html += HTML_FOOTER_TEMPLATE.format(
        version=frontmatter.get('version', '1.0'),
        created_at=frontmatter.get('created_at', '2025-10-17')
    )

    # Write HTML
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Created {html_path}")


def main():
    """Main conversion function."""
    print("Starting Markdown to HTML conversion...")
    print("=" * 60)

    for series in SERIES:
        series_path = BASE_PATH / series
        print(f"\nProcessing series: {series}")
        print("-" * 60)

        if not series_path.exists():
            print(f"⚠ Series directory not found: {series_path}")
            continue

        # Find all chapter*.md files in the series directory
        chapter_files = sorted(series_path.glob("chapter*.md"))

        if not chapter_files:
            print(f"⚠ No chapter files found in {series}")
            continue

        # Convert each chapter file
        for chapter_path in chapter_files:
            chapter_file = chapter_path.name
            convert_chapter(series_path, chapter_file)

    print("\n" + "=" * 60)
    print("✓ Conversion complete!")
    print(f"Total series processed: {len(SERIES)}")


if __name__ == "__main__":
    main()
