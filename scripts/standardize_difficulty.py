#!/usr/bin/env python3
"""
Standardize exercise difficulty labels across all English HTML files.

This script finds and standardizes non-standard difficulty labels in exercise headings
to the approved format: (Easy), (Medium), (Hard).

Usage:
    python standardize_difficulty.py [--dry-run] [--verbose]
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ReplacementPattern:
    """Pattern for difficulty label replacement."""
    pattern: re.Pattern
    replacement: str
    description: str


class DifficultyStandardizer:
    """Standardize exercise difficulty labels in HTML files."""

    # Approved standard formats (don't change these)
    STANDARD_FORMATS = [
        r'\(Difficulty:\s*easy\)',
        r'\(Difficulty:\s*medium\)',
        r'\(Difficulty:\s*hard\)',
        r'\(Easy\)',
        r'\(Medium\)',
        r'\(Hard\)',
    ]

    # Non-standard patterns to replace
    REPLACEMENT_PATTERNS = [
        # English variations
        ReplacementPattern(
            pattern=re.compile(r'\(Basic\)', re.IGNORECASE),
            replacement='(Easy)',
            description='(Basic) → (Easy)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(Intermediate\)', re.IGNORECASE),
            replacement='(Medium)',
            description='(Intermediate) → (Medium)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(Advanced\)', re.IGNORECASE),
            replacement='(Hard)',
            description='(Advanced) → (Hard)'
        ),

        # Star ratings
        ReplacementPattern(
            pattern=re.compile(r'\(Difficulty:\s*★☆☆\)', re.IGNORECASE),
            replacement='(Easy)',
            description='(Difficulty: ★☆☆) → (Easy)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(Difficulty:\s*★★☆\)', re.IGNORECASE),
            replacement='(Medium)',
            description='(Difficulty: ★★☆) → (Medium)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(Difficulty:\s*★★★\)', re.IGNORECASE),
            replacement='(Hard)',
            description='(Difficulty: ★★★) → (Hard)'
        ),

        # Japanese labels (in English files - should not exist but handle anyway)
        ReplacementPattern(
            pattern=re.compile(r'\(難易度:\s*基礎\)'),
            replacement='(Easy)',
            description='(難易度: 基礎) → (Easy)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(難易度:\s*中級\)'),
            replacement='(Medium)',
            description='(難易度: 中級) → (Medium)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(難易度:\s*上級\)'),
            replacement='(Hard)',
            description='(難易度: 上級) → (Hard)'
        ),

        # Additional common variations
        ReplacementPattern(
            pattern=re.compile(r'\(Beginner\)', re.IGNORECASE),
            replacement='(Easy)',
            description='(Beginner) → (Easy)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(Expert\)', re.IGNORECASE),
            replacement='(Hard)',
            description='(Expert) → (Hard)'
        ),

        # Verbose difficulty formats
        ReplacementPattern(
            pattern=re.compile(r'\(Difficulty:\s*Basic\)', re.IGNORECASE),
            replacement='(Easy)',
            description='(Difficulty: Basic) → (Easy)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(Difficulty:\s*Intermediate\)', re.IGNORECASE),
            replacement='(Medium)',
            description='(Difficulty: Intermediate) → (Medium)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(Difficulty:\s*Advanced\)', re.IGNORECASE),
            replacement='(Hard)',
            description='(Difficulty: Advanced) → (Hard)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(Difficulty:\s*Beginner\)', re.IGNORECASE),
            replacement='(Easy)',
            description='(Difficulty: Beginner) → (Easy)'
        ),
        ReplacementPattern(
            pattern=re.compile(r'\(Difficulty:\s*Expert\)', re.IGNORECASE),
            replacement='(Hard)',
            description='(Difficulty: Expert) → (Hard)'
        ),
    ]

    def __init__(self, base_path: Path, dry_run: bool = False, verbose: bool = False):
        """
        Initialize the standardizer.

        Args:
            base_path: Base path to the knowledge directory
            dry_run: If True, don't modify files
            verbose: If True, show detailed progress
        """
        self.base_path = base_path
        self.dry_run = dry_run
        self.verbose = verbose

        # Statistics
        self.files_scanned = 0
        self.files_modified = 0
        self.total_replacements = 0
        self.replacements_by_pattern: Dict[str, int] = defaultdict(int)
        self.files_with_changes: List[Tuple[Path, int]] = []

    def is_standard_format(self, text: str) -> bool:
        """Check if difficulty label is already in standard format."""
        for pattern in self.STANDARD_FORMATS:
            if re.search(pattern, text):
                return True
        return False

    def standardize_content(self, content: str, file_path: Path) -> Tuple[str, int]:
        """
        Standardize difficulty labels in content.

        Args:
            content: File content
            file_path: Path to the file (for logging)

        Returns:
            Tuple of (modified_content, num_replacements)
        """
        modified_content = content
        num_replacements = 0

        # Find all exercise headings (h3 and h4)
        heading_pattern = re.compile(
            r'(<h[34][^>]*>)(.*?)(</h[34]>)',
            re.IGNORECASE | re.DOTALL
        )

        def replace_in_heading(match: re.Match) -> str:
            """Replace non-standard difficulty labels in heading."""
            nonlocal num_replacements

            open_tag = match.group(1)
            heading_text = match.group(2)
            close_tag = match.group(3)

            # Only process if it looks like an exercise heading
            if not re.search(r'exercise\s+\d+', heading_text, re.IGNORECASE):
                return match.group(0)

            original_text = heading_text

            # Try each replacement pattern
            for pattern_info in self.REPLACEMENT_PATTERNS:
                if pattern_info.pattern.search(heading_text):
                    heading_text = pattern_info.pattern.sub(
                        pattern_info.replacement,
                        heading_text
                    )

            # Check if any changes were made
            if heading_text != original_text:
                num_replacements += 1

                if self.verbose:
                    print(f"  {file_path.name}:")
                    print(f"    Before: {original_text.strip()}")
                    print(f"    After:  {heading_text.strip()}")

                # Track which pattern was used
                for pattern_info in self.REPLACEMENT_PATTERNS:
                    if pattern_info.pattern.search(original_text):
                        self.replacements_by_pattern[pattern_info.description] += 1

            return f"{open_tag}{heading_text}{close_tag}"

        # Replace all headings
        modified_content = heading_pattern.sub(replace_in_heading, modified_content)

        return modified_content, num_replacements

    def process_file(self, file_path: Path) -> None:
        """
        Process a single HTML file.

        Args:
            file_path: Path to the HTML file
        """
        self.files_scanned += 1

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Standardize content
            modified_content, num_replacements = self.standardize_content(
                content, file_path
            )

            # Write back if changes were made
            if num_replacements > 0:
                self.files_modified += 1
                self.total_replacements += num_replacements
                self.files_with_changes.append((file_path, num_replacements))

                if not self.dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)

                    if self.verbose:
                        print(f"✓ Modified: {file_path.relative_to(self.base_path)}")
                else:
                    if self.verbose:
                        print(f"[DRY RUN] Would modify: {file_path.relative_to(self.base_path)}")

        except Exception as e:
            print(f"✗ Error processing {file_path.relative_to(self.base_path)}: {e}")

    def process_directory(self, dojo: str) -> None:
        """
        Process all HTML files in a dojo directory.

        Args:
            dojo: Dojo name (FM, ML, MS, MI, PI)
        """
        dojo_path = self.base_path / dojo

        if not dojo_path.exists():
            print(f"⚠ Skipping {dojo}: directory not found")
            return

        print(f"\nProcessing {dojo} dojo...")

        # Find all HTML files
        html_files = sorted(dojo_path.glob('**/*.html'))

        if not html_files:
            print(f"  No HTML files found in {dojo}")
            return

        for html_file in html_files:
            self.process_file(html_file)

    def run(self) -> None:
        """Run the standardization process."""
        print("=" * 80)
        print("Exercise Difficulty Label Standardizer")
        print("=" * 80)

        if self.dry_run:
            print("\n⚠ DRY RUN MODE - No files will be modified\n")

        # Process all dojos
        dojos = ['FM', 'ML', 'MS', 'MI', 'PI']

        for dojo in dojos:
            self.process_directory(dojo)

        # Print summary
        self.print_summary()

    def print_summary(self) -> None:
        """Print processing summary."""
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        print(f"\nFiles scanned: {self.files_scanned}")
        print(f"Files modified: {self.files_modified}")
        print(f"Total replacements: {self.total_replacements}")

        if self.replacements_by_pattern:
            print("\nReplacements by pattern:")
            for pattern, count in sorted(
                self.replacements_by_pattern.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  {pattern}: {count}")

        if self.files_with_changes and not self.verbose:
            print(f"\nFiles with changes ({len(self.files_with_changes)}):")
            for file_path, num_changes in self.files_with_changes:
                rel_path = file_path.relative_to(self.base_path)
                print(f"  {rel_path} ({num_changes} changes)")

        if self.dry_run:
            print("\n⚠ DRY RUN MODE - No files were actually modified")
            print("Run without --dry-run to apply changes")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Standardize exercise difficulty labels in HTML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview changes
  python standardize_difficulty.py --dry-run --verbose

  # Apply changes
  python standardize_difficulty.py

  # Verbose output showing all changes
  python standardize_difficulty.py --verbose
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed progress and changes'
    )

    args = parser.parse_args()

    # Determine base path
    script_dir = Path(__file__).parent
    base_path = script_dir.parent / 'knowledge' / 'en'

    if not base_path.exists():
        print(f"Error: Knowledge directory not found at {base_path}")
        return 1

    # Run standardization
    standardizer = DifficultyStandardizer(
        base_path=base_path,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    standardizer.run()

    return 0


if __name__ == '__main__':
    exit(main())
