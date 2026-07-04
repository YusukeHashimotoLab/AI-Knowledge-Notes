#!/usr/bin/env python3
"""
Add DOCTYPE declaration to all HTML files missing it.
"""

import re
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).parent.parent

def add_doctype(file_path: Path) -> Tuple[bool, str]:
    """Add DOCTYPE to HTML file if missing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if DOCTYPE already exists
        if content.strip().startswith('<!DOCTYPE') or content.strip().startswith('<!doctype'):
            return False, "DOCTYPE already exists"

        # Add DOCTYPE as first line
        new_content = '<!DOCTYPE html>\n' + content

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True, "DOCTYPE added"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main execution."""
    print("=" * 70)
    print("Adding DOCTYPE Declarations")
    print("=" * 70)

    # Find all HTML files
    html_files = list(BASE_DIR.glob("knowledge/**/*.html"))

    files_processed = 0
    files_updated = 0

    for html_file in html_files:
        files_processed += 1
        success, message = add_doctype(html_file)

        if success:
            files_updated += 1
            if files_updated <= 10:  # Show first 10
                print(f"  ✅ {html_file.relative_to(BASE_DIR)}")

    if files_updated > 10:
        print(f"\n  ... and {files_updated - 10} more files")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Files processed: {files_processed}")
    print(f"Files updated:   {files_updated}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
