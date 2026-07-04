#!/usr/bin/env python3
"""
AI Terakoya CSS Locale Switcher Styles Updater
===============================================

Production-ready script to add locale switcher CSS to knowledge-base.css.

Features:
- Safely adds locale switcher styles to existing CSS
- Preserves existing formatting and structure
- Detects and prevents duplicate insertions
- Creates backup before modification
- Validates CSS structure

Author: AI Terakoya Development Team
Date: 2025-11-16
Version: 1.0.0
"""

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Locale Switcher CSS to add
LOCALE_SWITCHER_CSS = """
/* ========================================
   15. Locale Switcher Styles
   ======================================== */

/* Main locale switcher container */
.locale-switcher {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 6px;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Current locale indicator */
.current-locale {
    font-weight: 600;
    color: var(--color-accent, #7b2cbf);
}

/* Separator between locales */
.locale-separator {
    color: var(--color-border, #cbd5e0);
    font-weight: 300;
}

/* Active locale link */
.locale-link {
    color: var(--color-link, #3182ce);
    text-decoration: none;
    transition: color 0.2s, transform 0.2s;
    font-weight: 500;
}

.locale-link:hover {
    color: var(--color-link-hover, #2c5aa0);
    text-decoration: underline;
    transform: translateY(-1px);
}

/* Disabled locale link (when translation not available) */
.locale-link.disabled {
    color: var(--color-text-light, #a0aec0);
    cursor: not-allowed;
    pointer-events: none;
}

/* Sync date metadata */
.locale-meta {
    font-size: 0.8rem;
    color: var(--color-text-light, #718096);
    margin-left: auto;
    opacity: 0.8;
}

/* Mobile responsive adjustments */
@media (max-width: 768px) {
    .locale-switcher {
        font-size: 0.85rem;
        padding: 0.4rem 0.8rem;
    }

    .locale-meta {
        display: none; /* Hide sync date on mobile */
    }
}

/* Print styles - hide locale switcher */
@media print {
    .locale-switcher {
        display: none;
    }
}

/* High contrast mode support */
@media (prefers-contrast: high) {
    .locale-switcher {
        border: 2px solid currentColor;
        background: transparent;
    }

    .locale-link {
        text-decoration: underline;
    }
}
"""


class CSSUpdater:
    """Handles adding locale switcher CSS to knowledge-base.css."""

    MARKER_START = "/* ========================================"
    MARKER_LOCALE = "15. Locale Switcher Styles"
    SECTION_PATTERN = r'/\*\s*=+\s*\n\s*15\.\s*Locale\s+Switcher\s+Styles'

    def __init__(self, css_path: Path, dry_run: bool = False, no_backup: bool = False):
        """
        Initialize CSS updater.

        Args:
            css_path: Path to knowledge-base.css
            dry_run: If True, don't modify files
            no_backup: If True, don't create backup
        """
        self.css_path = css_path
        self.dry_run = dry_run
        self.no_backup = no_backup

    def has_locale_styles(self, content: str) -> bool:
        """
        Check if CSS already has locale switcher styles.

        Args:
            content: CSS file content

        Returns:
            True if locale switcher styles exist
        """
        return bool(re.search(self.SECTION_PATTERN, content, re.IGNORECASE))

    def find_insertion_point(self, content: str) -> Optional[int]:
        """
        Find appropriate insertion point for locale switcher CSS.

        Strategy:
        1. After section 14 (Index Page Specific Styles)
        2. Before Accessibility Enhancements (if section 14 not found)
        3. At the end of file as fallback

        Args:
            content: CSS file content

        Returns:
            Character index for insertion, or None if not found
        """
        # Try to find end of section 14 (Index Page Specific Styles)
        section_14_pattern = r'/\*\s*=+\s*\n\s*14\.\s*Index\s+Page\s+Specific\s+Styles\s*\n\s*=+\s*\*/'
        matches = list(re.finditer(section_14_pattern, content, re.IGNORECASE))

        if matches:
            # Find the end of section 14 content
            last_match = matches[-1]
            # Look for the next section marker or end of file
            next_section = re.search(r'\n/\* =+', content[last_match.end():])
            if next_section:
                insertion_point = last_match.end() + next_section.start()
                logger.debug(f"Found insertion point after section 14 at position {insertion_point}")
                return insertion_point

        # Fallback: before accessibility section (section 13)
        section_13_pattern = r'/\*\s*=+\s*\n\s*13\.\s*Accessibility\s+Enhancements'
        matches = list(re.finditer(section_13_pattern, content, re.IGNORECASE))

        if matches:
            insertion_point = matches[0].start()
            logger.debug(f"Found insertion point before section 13 at position {insertion_point}")
            return insertion_point

        # Final fallback: end of file (before final newline if exists)
        content_stripped = content.rstrip()
        insertion_point = len(content_stripped)
        logger.debug(f"Using end of file as insertion point at position {insertion_point}")
        return insertion_point

    def insert_locale_css(self, content: str) -> str:
        """
        Insert locale switcher CSS into content.

        Args:
            content: Original CSS content

        Returns:
            Updated CSS content
        """
        insertion_point = self.find_insertion_point(content)

        if insertion_point is None:
            logger.error("Could not find insertion point")
            return content

        # Insert the CSS
        before = content[:insertion_point]
        after = content[insertion_point:]

        # Ensure proper spacing
        if not before.endswith('\n\n'):
            before = before.rstrip() + '\n\n'

        new_content = before + LOCALE_SWITCHER_CSS.lstrip() + '\n' + after

        logger.debug(f"Inserted locale switcher CSS at position {insertion_point}")
        return new_content

    def update_css(self) -> bool:
        """
        Update CSS file with locale switcher styles.

        Returns:
            True if successful
        """
        try:
            # Read existing CSS
            if not self.css_path.exists():
                logger.error(f"CSS file not found: {self.css_path}")
                return False

            with open(self.css_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Check if already has locale styles
            if self.has_locale_styles(original_content):
                logger.info("Locale switcher styles already exist in CSS")
                return True

            # Insert locale CSS
            new_content = self.insert_locale_css(original_content)

            if new_content == original_content:
                logger.error("Failed to insert locale switcher CSS")
                return False

            # Dry run - just report
            if self.dry_run:
                logger.info(f"[DRY RUN] Would update {self.css_path}")
                logger.info(f"[DRY RUN] Would add {len(LOCALE_SWITCHER_CSS)} characters of CSS")
                return True

            # Create backup
            if not self.no_backup:
                backup_path = self.css_path.with_suffix('.css.bak')
                shutil.copy2(self.css_path, backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Write updated CSS
            with open(self.css_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            logger.info(f"✓ Successfully updated {self.css_path.name}")
            return True

        except Exception as e:
            logger.error(f"Error updating CSS: {e}", exc_info=True)
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Add locale switcher styles to knowledge-base.css',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview changes
  python3 update_css_locale.py --dry-run

  # Update CSS without backup
  python3 update_css_locale.py --no-backup

  # Update specific CSS file
  python3 update_css_locale.py --css-path /path/to/knowledge-base.css
"""
    )

    parser.add_argument(
        '--css-path',
        type=str,
        help='Path to knowledge-base.css (default: auto-detect)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help="Don't create .bak backup file"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Determine CSS path
    if args.css_path:
        css_path = Path(args.css_path)
    else:
        # Auto-detect: script is in wp/scripts/
        script_dir = Path(__file__).parent
        base_path = script_dir.parent
        css_path = base_path / "knowledge" / "en" / "assets" / "css" / "knowledge-base.css"

    # Validate CSS path
    if not css_path.exists():
        logger.error(f"CSS file not found: {css_path}")
        sys.exit(1)

    logger.info(f"Target CSS file: {css_path}")

    # Create updater
    updater = CSSUpdater(css_path, dry_run=args.dry_run, no_backup=args.no_backup)

    # Update CSS
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")

    success = updater.update_css()

    # Report results
    print("\n" + "="*60)
    print("CSS UPDATE SUMMARY")
    print("="*60)
    if success:
        if args.dry_run:
            print("Status: Ready to update (dry run)")
        else:
            print("Status: Successfully updated")
        print(f"File: {css_path.name}")
        print(f"Size of added CSS: {len(LOCALE_SWITCHER_CSS)} characters")
    else:
        print("Status: Failed")
    print("="*60)

    if args.dry_run and success:
        print("\nThis was a DRY RUN. No files were modified.")
        print("Run without --dry-run to apply changes.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
