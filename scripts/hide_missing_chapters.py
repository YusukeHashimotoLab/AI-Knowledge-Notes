#!/usr/bin/env python3
"""
Hide missing chapter links from series index.html files.

Based on the analysis, the following series have missing chapters:
- FM/equilibrium-thermodynamics: chapter-2,3,4,5.html
- MS/materials-chemistry-introduction: chapter-1,2,3,4,5.html
- MI/mi-journals-conferences-introduction: chapter-4.html
- MS/electrical-magnetic-testing-introduction: chapter-5.html
- MS/materials-thermodynamics-introduction: chapter-6.html
- MS/mechanical-testing-introduction: chapter-5.html
- MS/polymer-materials-introduction: chapter-5.html
- MS/synthesis-processes-introduction: chapter-5.html
- MS/thin-film-nano-introduction: chapter-5.html
- PI/digital-twin-introduction: chapter-2,5.html
- PI/food-process-ai-introduction: chapter-2,5.html
"""

import os
import sys
from pathlib import Path
import re

# Series with missing chapters
MISSING_CHAPTERS = {
    'FM/equilibrium-thermodynamics': ['chapter-2.html', 'chapter-3.html', 'chapter-4.html', 'chapter-5.html'],
    'MS/materials-chemistry-introduction': ['chapter-1.html', 'chapter-2.html', 'chapter-3.html', 'chapter-4.html', 'chapter-5.html'],
    'MI/mi-journals-conferences-introduction': ['chapter-4.html'],
    'MS/electrical-magnetic-testing-introduction': ['chapter-5.html'],
    'MS/materials-thermodynamics-introduction': ['chapter-6.html'],
    'MS/mechanical-testing-introduction': ['chapter-5.html'],
    'MS/polymer-materials-introduction': ['chapter-5.html'],
    'MS/synthesis-processes-introduction': ['chapter-5.html'],
    'MS/thin-film-nano-introduction': ['chapter-5.html'],
    'PI/digital-twin-introduction': ['chapter-2.html', 'chapter-5.html'],
    'PI/food-process-ai-introduction': ['chapter-2.html', 'chapter-5.html'],
}

def hide_missing_chapters_in_file(file_path, missing_chapters):
    """Hide missing chapter links in an index.html file."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    fixes_applied = 0

    for chapter in missing_chapters:
        # Pattern 1: Remove entire chapter card sections
        # Match from <div class="chapter-card"> to closing </div>
        pattern = rf'<div class="chapter-card"[^>]*>.*?href="{chapter}".*?</div>\s*</div>'
        content, count = re.subn(pattern, '', content, flags=re.DOTALL)
        fixes_applied += count

        # Pattern 2: Remove simple list item links
        pattern = rf'<li[^>]*>\s*<a[^>]*href="{chapter}"[^>]*>.*?</a>\s*</li>'
        content, count = re.subn(pattern, '', content, flags=re.DOTALL)
        fixes_applied += count

        # Pattern 3: Remove standalone links
        pattern = rf'<a[^>]*href="{chapter}"[^>]*>.*?</a>'
        content, count = re.subn(pattern, '', content, flags=re.DOTALL)
        fixes_applied += count

        # Pattern 4: Remove navigation button sections
        # <a class="nav-button" href="chapter-X.html">
        pattern = rf'<a class="nav-button"[^>]*href="{chapter}"[^>]*>.*?</a>'
        content, count = re.subn(pattern, '', content, flags=re.DOTALL)
        fixes_applied += count

    if content == original_content:
        return False, "No changes needed"

    # Create backup
    backup_path = str(file_path) + '.bak'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)

    # Write fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return True, f"Hid {len(missing_chapters)} missing chapters ({fixes_applied} fixes)"

def main():
    base_dir = Path("knowledge/en")

    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        sys.exit(1)

    print("Hiding missing chapter links from series index files...\n")

    files_fixed = 0
    total_chapters_hidden = 0
    total_fixes = 0

    for series_path, missing_chapters in MISSING_CHAPTERS.items():
        index_path = base_dir / series_path / "index.html"

        if not index_path.exists():
            print(f"⚠️  Index not found: {series_path}/index.html")
            continue

        fixed, message = hide_missing_chapters_in_file(index_path, missing_chapters)
        if fixed:
            files_fixed += 1
            print(f"✓ Fixed: {series_path}/index.html")
            print(f"  {message}")
            total_chapters_hidden += len(missing_chapters)

            # Extract number of fixes
            if "(" in message and " fixes)" in message:
                num = int(message.split("(")[1].split(" fixes)")[0])
                total_fixes += num

    print("\n" + "="*60)
    print(f"Summary:")
    print(f"  Series updated: {files_fixed}/{len(MISSING_CHAPTERS)}")
    print(f"  Chapters hidden: {total_chapters_hidden}")
    print(f"  Total fixes applied: {total_fixes}")
    print("="*60)

if __name__ == "__main__":
    main()
