#!/usr/bin/env python3
"""
Batch convert all HTML files in knowledge directory to Markdown.
Extends html_to_md.py to handle all HTML files (not just chapter*.html).
"""

import os
import sys
from pathlib import Path

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from html_to_md import convert_html_file
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def batch_convert(knowledge_dir: Path, no_backup: bool = False) -> tuple:
    """
    Convert all HTML files in knowledge directory to Markdown.

    Args:
        knowledge_dir: Path to knowledge directory
        no_backup: If True, don't create backup files

    Returns:
        Tuple of (success_count, total_count, failed_files)
    """
    # Find all HTML files
    html_files = list(knowledge_dir.rglob("*.html"))

    logger.info(f"Found {len(html_files)} HTML files to convert")

    success_count = 0
    failed_files = []

    for i, html_file in enumerate(html_files, 1):
        logger.info(f"[{i}/{len(html_files)}] Processing: {html_file.relative_to(knowledge_dir)}")

        try:
            if convert_html_file(html_file, backup=not no_backup):
                success_count += 1
            else:
                failed_files.append(str(html_file))
        except Exception as e:
            logger.error(f"Error converting {html_file}: {e}")
            failed_files.append(str(html_file))

    return success_count, len(html_files), failed_files


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch convert all HTML files in knowledge directory to Markdown'
    )

    parser.add_argument(
        'knowledge_dir',
        type=str,
        help='Path to knowledge directory'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create .bak backup files'
    )

    args = parser.parse_args()

    knowledge_dir = Path(args.knowledge_dir)

    if not knowledge_dir.exists():
        logger.error(f"Directory does not exist: {knowledge_dir}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Starting batch HTML to Markdown conversion")
    logger.info("=" * 60)

    success, total, failed = batch_convert(knowledge_dir, args.no_backup)

    logger.info("=" * 60)
    logger.info(f"Conversion complete!")
    logger.info(f"Successfully converted: {success}/{total} files")

    if failed:
        logger.warning(f"Failed files ({len(failed)}):")
        for f in failed:
            logger.warning(f"  - {f}")


if __name__ == "__main__":
    main()
