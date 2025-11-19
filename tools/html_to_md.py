#!/usr/bin/env python3
"""
Convert HTML files back to Markdown with YAML frontmatter.
Extracts metadata from HTML and converts content to clean Markdown format.

This script reverses the HTML generation process, allowing editing of HTML files
and extracting them back to Markdown source format.
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: beautifulsoup4 is required. Install with: pip install beautifulsoup4")
    sys.exit(1)

try:
    import html2text
except ImportError:
    print("Error: html2text is required. Install with: pip install html2text")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base path for English knowledge base
BASE_PATH = Path("knowledge/en")


def extract_metadata_from_html(soup: BeautifulSoup) -> Dict:
    """
    Extract metadata from HTML header and footer.

    Args:
        soup: BeautifulSoup object of HTML content

    Returns:
        Dictionary of metadata
    """
    metadata = {}

    # Extract title from h1
    h1 = soup.find('h1')
    if h1:
        metadata['chapter_title'] = h1.get_text().strip()
        metadata['title'] = h1.get_text().strip()

    # Extract subtitle
    subtitle = soup.find('p', class_='subtitle')
    if subtitle:
        metadata['subtitle'] = subtitle.get_text().strip()

    # Extract meta items
    meta_items = soup.find_all('span', class_='meta-item')
    for item in meta_items:
        text = item.get_text().strip()

        # Parse different meta types
        if '📖' in text or 'Reading Time:' in text or 'Reading time:' in text:
            # Extract reading time
            time_match = re.search(r':\s*(.+)', text)
            if time_match:
                metadata['reading_time'] = time_match.group(1).strip()

        elif '📊' in text or 'Difficulty:' in text:
            # Extract difficulty
            diff_match = re.search(r':\s*(.+)', text)
            if diff_match:
                metadata['difficulty'] = diff_match.group(1).strip()

        elif '💻' in text or 'Code Examples:' in text or 'Code examples:' in text:
            # Extract code examples count
            code_match = re.search(r':\s*(\d+)', text)
            if code_match:
                metadata['code_examples'] = int(code_match.group(1))

        elif '📝' in text or 'Exercises:' in text:
            # Extract exercises count
            ex_match = re.search(r':\s*(\d+)', text)
            if ex_match:
                metadata['exercises'] = int(ex_match.group(1))

    # Extract footer information
    footer = soup.find('footer')
    if footer:
        footer_text = footer.get_text()

        # Extract version
        version_match = re.search(r'Version[:\s]+([^\s|]+)', footer_text)
        if version_match:
            metadata['version'] = version_match.group(1).strip()

        # Extract created date
        created_match = re.search(r'Created[:\s]+([^\s]+)', footer_text)
        if created_match:
            metadata['created_at'] = created_match.group(1).strip()

    return metadata


def convert_html_to_markdown(html_content: str) -> Tuple[Dict, str]:
    """
    Convert HTML content to Markdown with extracted frontmatter.

    Args:
        html_content: Raw HTML content

    Returns:
        Tuple of (metadata_dict, markdown_content)
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract metadata
    metadata = extract_metadata_from_html(soup)

    # Remove header, footer, and navigation
    for tag in soup.find_all(['header', 'footer']):
        tag.decompose()

    for tag in soup.find_all('div', class_='navigation'):
        tag.decompose()

    # Get main content
    main = soup.find('main', class_='container')
    if not main:
        # Fallback: use body content
        main = soup.find('body')

    if not main:
        logger.error("Could not find main content in HTML")
        return metadata, ""

    # Convert Mermaid divs back to code blocks
    for mermaid_div in main.find_all('div', class_='mermaid'):
        mermaid_text = mermaid_div.get_text().strip()
        code_block = soup.new_tag('pre')
        code_block.string = f"```mermaid\n{mermaid_text}\n```"
        mermaid_div.replace_with(code_block)

    # Configure html2text converter
    h = html2text.HTML2Text()
    h.body_width = 0  # Don't wrap lines
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.skip_internal_links = False
    h.inline_links = True
    h.protect_links = True
    h.unicode_snob = True
    h.escape_snob = False

    # Convert to Markdown
    markdown_content = h.handle(str(main))

    # Post-process Markdown
    # Fix Mermaid blocks
    markdown_content = re.sub(
        r'```\n```mermaid\n(.*?)\n```\n```',
        r'```mermaid\n\1\n```',
        markdown_content,
        flags=re.DOTALL
    )

    # Clean up excessive newlines
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)

    # Fix escaped underscores in math (if any remain)
    # This is a heuristic - may need adjustment
    markdown_content = re.sub(r'\\\\_', '_', markdown_content)

    return metadata, markdown_content.strip()


def create_frontmatter_yaml(metadata: Dict) -> str:
    """
    Create YAML frontmatter from metadata dictionary.

    Args:
        metadata: Metadata dictionary

    Returns:
        YAML frontmatter string
    """
    yaml_lines = ["---"]

    # Add fields in specific order
    field_order = [
        'title',
        'chapter_title',
        'subtitle',
        'reading_time',
        'difficulty',
        'code_examples',
        'exercises',
        'version',
        'created_at'
    ]

    for field in field_order:
        if field in metadata:
            value = metadata[field]
            # Quote strings with special characters
            if isinstance(value, str) and (':' in value or '#' in value or value.startswith('&')):
                yaml_lines.append(f'{field}: "{value}"')
            else:
                yaml_lines.append(f'{field}: {value}')

    yaml_lines.append("---")
    return '\n'.join(yaml_lines)


def convert_html_file(html_path: Path, output_dir: Optional[Path] = None, backup: bool = True) -> bool:
    """
    Convert a single HTML file to Markdown.

    Args:
        html_path: Path to HTML file
        output_dir: Optional output directory (default: same as HTML file)
        backup: Whether to create backup if .md file exists

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Converting {html_path} to Markdown...")

    try:
        # Read HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Convert to Markdown
        metadata, markdown_content = convert_html_to_markdown(html_content)

        # Create frontmatter
        frontmatter = create_frontmatter_yaml(metadata)

        # Combine frontmatter and content
        full_markdown = f"{frontmatter}\n\n{markdown_content}\n"

        # Determine output path
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            md_path = output_dir / html_path.with_suffix('.md').name
        else:
            md_path = html_path.with_suffix('.md')

        # Backup existing file if needed
        if backup and md_path.exists():
            backup_path = md_path.with_suffix('.md.bak')
            logger.info(f"Backing up existing {md_path} to {backup_path}")
            md_path.rename(backup_path)

        # Write Markdown atomically
        temp_path = md_path.with_suffix('.md.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(full_markdown)

        # Rename temp to final
        temp_path.replace(md_path)

        logger.info(f"✓ Created {md_path}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to convert {html_path}: {e}")
        return False


def convert_directory(directory: Path, output_dir: Optional[Path] = None, backup: bool = True) -> Tuple[int, int]:
    """
    Convert all HTML files in a directory.

    Args:
        directory: Directory containing HTML files
        output_dir: Optional output directory
        backup: Whether to create backups

    Returns:
        Tuple of (successful_count, total_count)
    """
    logger.info(f"\nProcessing directory: {directory}")
    logger.info("-" * 60)

    # Find all HTML chapter files
    html_files = sorted(directory.glob("chapter*.html"))

    if not html_files:
        logger.warning(f"No chapter HTML files found in {directory}")
        return 0, 0

    success_count = 0
    for html_file in html_files:
        if convert_html_file(html_file, output_dir, backup):
            success_count += 1

    return success_count, len(html_files)


def main():
    """Main conversion function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert HTML files to Markdown with YAML frontmatter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert single file
  python html_to_md.py knowledge/en/ML/transformer-introduction/chapter-1.html

  # Convert all chapters in a series
  python html_to_md.py knowledge/en/ML/transformer-introduction/

  # Convert to different output directory
  python html_to_md.py knowledge/en/ML/transformer-introduction/ --output-dir markdown_output/

  # Convert without creating backups
  python html_to_md.py knowledge/en/ML/transformer-introduction/ --no-backup
        '''
    )

    parser.add_argument(
        'path',
        type=str,
        help='HTML file or directory to convert'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for Markdown files (default: same as input)'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create .bak backup files'
    )

    args = parser.parse_args()

    logger.info("Starting HTML to Markdown conversion...")
    logger.info("=" * 60)

    path = Path(args.path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    backup = not args.no_backup

    total_success = 0
    total_files = 0

    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    # Check if path is a file or directory
    if path.is_file() and path.suffix == '.html':
        if convert_html_file(path, output_dir, backup):
            total_success = 1
        total_files = 1
    elif path.is_dir():
        success, total = convert_directory(path, output_dir, backup)
        total_success += success
        total_files += total
    else:
        logger.error(f"Invalid path: {path} (must be .html file or directory)")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Conversion complete!")
    logger.info(f"Successfully converted: {total_success}/{total_files} files")


if __name__ == "__main__":
    main()
