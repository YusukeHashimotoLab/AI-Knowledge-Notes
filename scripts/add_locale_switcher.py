#!/usr/bin/env python3
"""
AI Terakoya Locale Switcher Implementation
============================================

Production-ready script to add language switchers to all English knowledge base files.

Features:
- Automatically detects corresponding Japanese files
- Adds elegant locale switcher to breadcrumb navigation
- Extracts sync dates from git history or file mtime
- Safe atomic writes with backup support
- Comprehensive error handling and validation
- Progress reporting with tqdm

Author: AI Terakoya Development Team
Date: 2025-11-16
Version: 1.0.0
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    from bs4 import BeautifulSoup
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Install with: pip install beautifulsoup4 tqdm")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class LocaleSwitcherConfig:
    """Configuration for locale switcher behavior."""
    dry_run: bool = False
    no_backup: bool = False
    force: bool = False
    sync_date: Optional[str] = None
    verbose: bool = False


class LocaleSwitcher:
    """Handles adding locale switchers to HTML files."""

    SWITCHER_CLASS = "locale-switcher"
    SWITCHER_TEMPLATE = """<div class="locale-switcher">
    <span class="current-locale">🌐 EN</span>
    <span class="locale-separator">|</span>
    {jp_link}
    <span class="locale-meta">Last sync: {sync_date}</span>
</div>
"""

    SWITCHER_LINK_ACTIVE = '<a href="{jp_path}" class="locale-link">日本語</a>'
    SWITCHER_LINK_DISABLED = '<span class="locale-link disabled">日本語 (準備中)</span>'

    def __init__(self, config: LocaleSwitcherConfig, base_path: Path):
        """
        Initialize LocaleSwitcher.

        Args:
            config: Configuration object
            base_path: Base path to the wp directory
        """
        self.config = config
        self.base_path = base_path
        self.knowledge_en = base_path / "knowledge" / "en"
        self.knowledge_jp = base_path / "knowledge" / "jp"

        if config.verbose:
            logger.setLevel(logging.DEBUG)

    def get_jp_path(self, en_file: Path) -> Tuple[Optional[Path], Optional[str]]:
        """
        Get corresponding Japanese file path and relative URL.

        Args:
            en_file: English HTML file path

        Returns:
            Tuple of (absolute JP path if exists, relative URL for HTML link)
        """
        try:
            # Get relative path from knowledge/en/
            rel_path = en_file.relative_to(self.knowledge_en)

            # Construct Japanese path
            jp_file = self.knowledge_jp / rel_path

            # Calculate relative URL from en file to jp file
            # Example: ../../jp/ML/transformer-introduction/chapter1-self-attention.html
            en_parts = list(en_file.relative_to(self.knowledge_en).parts)
            depth = len(en_parts) - 1  # Number of directories deep

            # Build relative path: ../../jp/...
            rel_url = "../" * depth + "../jp/" + str(rel_path)

            # Check if Japanese file exists
            if jp_file.exists():
                logger.debug(f"Found JP file: {jp_file}")
                return jp_file, rel_url
            else:
                logger.debug(f"JP file not found: {jp_file}")
                return None, rel_url

        except ValueError as e:
            logger.error(f"Invalid path structure for {en_file}: {e}")
            return None, None

    def get_sync_date(self, file_path: Path) -> str:
        """
        Get last sync date from git history or file mtime.

        Args:
            file_path: Path to file

        Returns:
            Date string in YYYY-MM-DD format
        """
        # Use custom date if provided
        if self.config.sync_date:
            return self.config.sync_date

        # Try git log first
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ai", str(file_path)],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse git date: "2025-11-16 12:34:56 +0900"
                date_str = result.stdout.strip().split()[0]
                return date_str
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            logger.debug("Git not available, using file mtime")

        # Fallback to file modification time
        mtime = file_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')

    def has_locale_switcher(self, soup: BeautifulSoup) -> bool:
        """
        Check if HTML already has a locale switcher.

        Args:
            soup: BeautifulSoup object

        Returns:
            True if locale switcher exists
        """
        return soup.find(class_=self.SWITCHER_CLASS) is not None

    def create_switcher_html(self, jp_exists: bool, jp_url: str, sync_date: str) -> str:
        """
        Create locale switcher HTML.

        Args:
            jp_exists: Whether Japanese file exists
            jp_url: Relative URL to Japanese file
            sync_date: Sync date string

        Returns:
            HTML string for locale switcher
        """
        if jp_exists:
            jp_link = self.SWITCHER_LINK_ACTIVE.format(jp_path=jp_url)
        else:
            jp_link = self.SWITCHER_LINK_DISABLED

        return self.SWITCHER_TEMPLATE.format(
            jp_link=jp_link,
            sync_date=sync_date
        )

    def insert_switcher(self, soup: BeautifulSoup, switcher_html: str) -> bool:
        """
        Insert locale switcher into HTML.

        Args:
            soup: BeautifulSoup object
            switcher_html: HTML string to insert

        Returns:
            True if successfully inserted
        """
        # Find breadcrumb nav
        breadcrumb = soup.find('nav', class_='breadcrumb')

        if breadcrumb:
            # Insert after breadcrumb
            switcher_soup = BeautifulSoup(switcher_html, 'html.parser')
            breadcrumb.insert_after(switcher_soup)
            logger.debug("Inserted switcher after breadcrumb")
            return True

        # Fallback: insert after body tag
        body = soup.find('body')
        if body:
            switcher_soup = BeautifulSoup(switcher_html, 'html.parser')
            body.insert(0, switcher_soup)
            logger.debug("Inserted switcher after body tag")
            return True

        logger.warning("Could not find insertion point for switcher")
        return False

    def validate_html(self, html_content: str) -> bool:
        """
        Validate HTML structure.

        Args:
            html_content: HTML string

        Returns:
            True if valid HTML structure
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Check for essential tags
            return soup.find('html') is not None and soup.find('body') is not None
        except Exception as e:
            logger.error(f"HTML validation failed: {e}")
            return False

    def process_file(self, file_path: Path) -> bool:
        """
        Process a single HTML file to add locale switcher.

        Args:
            file_path: Path to HTML file

        Returns:
            True if successfully processed
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Validate input HTML
            if not self.validate_html(original_content):
                logger.error(f"Invalid HTML structure in {file_path}")
                return False

            # Parse HTML
            soup = BeautifulSoup(original_content, 'html.parser')

            # Check if switcher already exists
            if self.has_locale_switcher(soup):
                if not self.config.force:
                    logger.info(f"Skipping {file_path.name} - switcher already exists")
                    return False
                else:
                    # Remove existing switcher
                    existing = soup.find(class_=self.SWITCHER_CLASS)
                    if existing:
                        existing.decompose()
                    logger.debug(f"Removed existing switcher from {file_path.name}")

            # Get Japanese file info
            jp_file, jp_url = self.get_jp_path(file_path)

            if jp_url is None:
                logger.error(f"Could not determine JP path for {file_path}")
                return False

            # Get sync date
            sync_date = self.get_sync_date(file_path)

            # Create switcher HTML
            switcher_html = self.create_switcher_html(
                jp_exists=(jp_file is not None),
                jp_url=jp_url,
                sync_date=sync_date
            )

            # Insert switcher
            if not self.insert_switcher(soup, switcher_html):
                logger.error(f"Failed to insert switcher into {file_path}")
                return False

            # Generate new HTML
            new_content = str(soup)

            # Validate output HTML
            if not self.validate_html(new_content):
                logger.error(f"Generated invalid HTML for {file_path}")
                return False

            # Dry run - just report
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would update {file_path.name}")
                logger.debug(f"  JP file exists: {jp_file is not None}")
                logger.debug(f"  Sync date: {sync_date}")
                return True

            # Create backup
            if not self.config.no_backup:
                backup_path = file_path.with_suffix('.html.bak')
                shutil.copy2(file_path, backup_path)
                logger.debug(f"Created backup: {backup_path}")

            # Atomic write
            temp_path = file_path.with_suffix('.html.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            # Move temp file to original
            shutil.move(str(temp_path), str(file_path))

            logger.info(f"✓ Updated {file_path.name}")
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=self.config.verbose)
            return False

    def process_directory(self, target_path: Optional[Path] = None) -> dict:
        """
        Process all HTML files in directory tree.

        Args:
            target_path: Starting directory (default: knowledge/en/)

        Returns:
            Dictionary with processing statistics
        """
        if target_path is None:
            target_path = self.knowledge_en

        # Find all HTML files
        html_files = list(target_path.rglob('*.html'))

        if not html_files:
            logger.warning(f"No HTML files found in {target_path}")
            return {'total': 0, 'success': 0, 'skipped': 0, 'failed': 0}

        logger.info(f"Found {len(html_files)} HTML files")

        stats = {'total': len(html_files), 'success': 0, 'skipped': 0, 'failed': 0}

        # Process files with progress bar
        for file_path in tqdm(html_files, desc="Processing files", disable=self.config.verbose):
            result = self.process_file(file_path)
            if result:
                stats['success'] += 1
            elif self.has_locale_switcher(BeautifulSoup(file_path.read_text(encoding='utf-8'), 'html.parser')):
                stats['skipped'] += 1
            else:
                stats['failed'] += 1

        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Add locale switchers to AI Terakoya English knowledge base',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on all files
  python3 add_locale_switcher.py --dry-run

  # Process specific directory
  python3 add_locale_switcher.py knowledge/en/ML/

  # Force overwrite existing switchers
  python3 add_locale_switcher.py --force

  # Set custom sync date
  python3 add_locale_switcher.py --sync-date 2025-11-16

  # Verbose output without backups
  python3 add_locale_switcher.py --verbose --no-backup
"""
    )

    parser.add_argument(
        'path',
        type=str,
        nargs='?',
        help='Target path (default: knowledge/en/)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help="Don't create .bak files"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing switchers'
    )
    parser.add_argument(
        '--sync-date',
        type=str,
        metavar='YYYY-MM-DD',
        help='Set custom sync date'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed logging'
    )

    args = parser.parse_args()

    # Validate sync date format
    if args.sync_date:
        try:
            datetime.strptime(args.sync_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {args.sync_date}. Use YYYY-MM-DD")
            sys.exit(1)

    # Determine base path (script is in wp/scripts/)
    script_dir = Path(__file__).parent
    base_path = script_dir.parent

    # Validate base path structure
    if not (base_path / "knowledge" / "en").exists():
        logger.error(f"Invalid base path: {base_path}")
        logger.error("Expected structure: <base>/knowledge/en/")
        sys.exit(1)

    # Create config
    config = LocaleSwitcherConfig(
        dry_run=args.dry_run,
        no_backup=args.no_backup,
        force=args.force,
        sync_date=args.sync_date,
        verbose=args.verbose
    )

    # Create switcher
    switcher = LocaleSwitcher(config, base_path)

    # Determine target path
    if args.path:
        target = Path(args.path)
        if not target.is_absolute():
            target = base_path / target
        if not target.exists():
            logger.error(f"Target path does not exist: {target}")
            sys.exit(1)
    else:
        target = None

    # Process files
    logger.info("Starting locale switcher addition...")
    if config.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")

    stats = switcher.process_directory(target)

    # Report results
    print("\n" + "="*60)
    print("LOCALE SWITCHER ADDITION SUMMARY")
    print("="*60)
    print(f"Total files found:     {stats['total']}")
    print(f"Successfully updated:  {stats['success']}")
    print(f"Skipped (exists):      {stats['skipped']}")
    print(f"Failed:                {stats['failed']}")
    print("="*60)

    if config.dry_run:
        print("\nThis was a DRY RUN. No files were modified.")
        print("Run without --dry-run to apply changes.")

    # Exit with appropriate code
    sys.exit(0 if stats['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
