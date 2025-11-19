#!/usr/bin/env python3
"""
Fix checkbox lists in HTML files.

This script finds checkbox items (✅, ☑, etc.) that are incorrectly placed
in <p> tags without proper line breaks, and converts them to proper <ul><li> lists.
"""

import re
from pathlib import Path
from typing import List, Tuple

# Base directory
BASE_DIR = Path(__file__).parent.parent / "knowledge" / "en"


def find_checkbox_p_tags(content: str) -> List[Tuple[str, str]]:
    """
    Find <p> tags containing multiple checkbox items without proper formatting.

    Returns list of (original_text, fixed_text) tuples.
    """
    fixes = []

    # Pattern: <p>✅ text✅ text✅ text</p> (multiple checkboxes in one p tag)
    pattern = r'<p>((?:✅|☑|✓)[^\n<]+(?:\n(?:✅|☑|✓)[^\n<]+)+)</p>'

    for match in re.finditer(pattern, content, re.MULTILINE):
        original = match.group(0)
        checkbox_text = match.group(1)

        # Split by checkbox symbols
        items = re.split(r'(?=✅|☑|✓)', checkbox_text)
        items = [item.strip() for item in items if item.strip()]

        # Create ul/li structure
        li_items = '\n'.join(f'<li>{item}</li>' for item in items)
        fixed = f'<ul>\n{li_items}\n</ul>'

        fixes.append((original, fixed))

    return fixes


def fix_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, int]:
    """
    Fix checkbox lists in a single file.

    Returns:
        Tuple of (was_modified, num_fixes)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        fixes = find_checkbox_p_tags(content)

        if not fixes:
            return False, 0

        # Apply fixes
        new_content = content
        for original, fixed in fixes:
            new_content = new_content.replace(original, fixed, 1)

        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

        return True, len(fixes)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Fix checkbox lists in HTML files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without modifying files')
    args = parser.parse_args()

    print("=" * 70)
    print("Checkbox List Fix Script")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be modified]\n")

    # Find all HTML files
    html_files = list(BASE_DIR.rglob("*.html"))
    print(f"Scanning {len(html_files)} HTML files...\n")

    # Process files
    modified_count = 0
    total_fixes = 0
    modified_files = []

    for html_file in html_files:
        was_modified, num_fixes = fix_file(html_file, dry_run=args.dry_run)

        if was_modified:
            modified_count += 1
            total_fixes += num_fixes
            rel_path = html_file.relative_to(BASE_DIR.parent.parent)
            modified_files.append((rel_path, num_fixes))

            mode = "[DRY RUN]" if args.dry_run else "[MODIFIED]"
            print(f"{mode} {rel_path}: {num_fixes} fixes")

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Files scanned: {len(html_files)}")
    print(f"  Files modified: {modified_count}")
    print(f"  Total fixes: {total_fixes}")
    print("=" * 70)

    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to apply changes.")

    return 0


if __name__ == "__main__":
    exit(main())
