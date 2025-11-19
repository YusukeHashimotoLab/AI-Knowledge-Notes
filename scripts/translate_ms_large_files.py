#!/usr/bin/env python3
"""
Translate MS HTML files using a chunked approach for large files
This script will process files and output translation prompts for Claude Code
"""

import os
import re
import sys
from pathlib import Path

# All empty files to translate
EMPTY_FILES = [
    "3d-printing-introduction/chapter-2.html",
    "3d-printing-introduction/chapter-3.html",
    "3d-printing-introduction/chapter-5.html",
    "advanced-materials-systems-introduction/chapter-1.html",
    "advanced-materials-systems-introduction/chapter-2.html",
    "advanced-materials-systems-introduction/chapter-3.html",
    "advanced-materials-systems-introduction/chapter-4.html",
    "advanced-materials-systems-introduction/chapter-5.html",
    "electrical-magnetic-testing-introduction/chapter-1.html",
    "electrical-magnetic-testing-introduction/chapter-2.html",
    "electrical-magnetic-testing-introduction/chapter-3.html",
    "electrical-magnetic-testing-introduction/chapter-4.html",
    "electrical-magnetic-testing-introduction/chapter-5.html",
    "materials-microstructure-introduction/chapter-1.html",
    "materials-microstructure-introduction/chapter-2.html",
    "materials-microstructure-introduction/chapter-3.html",
    "materials-microstructure-introduction/chapter-4.html",
    "materials-microstructure-introduction/chapter-5.html",
    "materials-science-introduction/chapter-1.html",
    "materials-science-introduction/chapter-2.html",
    "materials-science-introduction/chapter-3.html",
    "materials-science-introduction/chapter-4.html",
    "materials-science-introduction/chapter-5.html",
    "materials-thermodynamics-introduction/chapter-1.html",
    "materials-thermodynamics-introduction/chapter-2.html",
    "materials-thermodynamics-introduction/chapter-3.html",
    "materials-thermodynamics-introduction/chapter-4.html",
    "materials-thermodynamics-introduction/chapter-5.html",
    "mechanical-testing-introduction/chapter-1.html",
    "mechanical-testing-introduction/chapter-2.html",
    "mechanical-testing-introduction/chapter-3.html",
    "mechanical-testing-introduction/chapter-4.html",
    "polymer-materials-introduction/chapter-1.html",
    "polymer-materials-introduction/chapter-2.html",
    "polymer-materials-introduction/chapter-3.html",
    "polymer-materials-introduction/chapter-4.html",
    "processing-introduction/chapter-1.html",
    "processing-introduction/chapter-2.html",
    "processing-introduction/chapter-3.html",
    "processing-introduction/chapter-4.html",
    "processing-introduction/chapter-5.html",
    "processing-introduction/chapter-6.html",
    "spectroscopy-introduction/chapter-1.html",
    "spectroscopy-introduction/chapter-2.html",
    "spectroscopy-introduction/chapter-3.html",
    "spectroscopy-introduction/chapter-4.html",
    "spectroscopy-introduction/chapter-5.html",
    "synthesis-processes-introduction/chapter-1.html",
    "synthesis-processes-introduction/chapter-2.html",
    "synthesis-processes-introduction/chapter-3.html",
    "synthesis-processes-introduction/chapter-4.html",
    "thin-film-nano-introduction/chapter-1.html",
    "thin-film-nano-introduction/chapter-2.html",
    "thin-film-nano-introduction/chapter-3.html",
    "thin-film-nano-introduction/chapter-4.html",
    "xrd-analysis-introduction/chapter-1.html",
    "xrd-analysis-introduction/chapter-2.html",
    "xrd-analysis-introduction/chapter-3.html",
    "xrd-analysis-introduction/chapter-4.html",
    "xrd-analysis-introduction/chapter-5.html",
]

BASE_DIR = Path(__file__).parent.parent / "knowledge"
EN_BASE = BASE_DIR / "en" / "MS"
JP_BASE = BASE_DIR / "jp" / "MS"

def count_japanese_chars(text):
    """Count Japanese characters"""
    hiragana = len(re.findall(r'[あ-ん]', text))
    katakana = len(re.findall(r'[ア-ン]', text))
    kanji = len(re.findall(r'[一-龯]', text))
    return hiragana + katakana + kanji

def check_file_status():
    """Check status of all files"""
    print("Checking file status...\n")

    for rel_path in EMPTY_FILES:
        en_file = EN_BASE / rel_path
        jp_file = JP_BASE / rel_path

        category = rel_path.split('/')[0]
        filename = rel_path.split('/')[1]

        # Check Japanese source
        if not jp_file.exists():
            print(f"❌ {category}/{filename}: Japanese source missing")
            continue

        # Check English file size
        en_size = en_file.stat().st_size if en_file.exists() else 0
        jp_size = jp_file.stat().st_size

        if en_size < 1000:
            print(f"📝 {category}/{filename}: Empty (JP: {jp_size:,} bytes)")
        else:
            jp_count = count_japanese_chars(en_file.read_text(encoding='utf-8'))
            if jp_count > 0:
                print(f"⚠️  {category}/{filename}: Has content but {jp_count} JP chars")
            else:
                print(f"✅ {category}/{filename}: Already translated")

def group_by_category():
    """Group files by category"""
    categories = {}
    for rel_path in EMPTY_FILES:
        category = rel_path.split('/')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(rel_path)

    return categories

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--status':
        check_file_status()
        return

    categories = group_by_category()

    print("="*80)
    print("MS TRANSLATION BATCH PLAN")
    print("="*80)
    print(f"\nTotal files: {len(EMPTY_FILES)}")
    print(f"Categories: {len(categories)}\n")

    for category, files in sorted(categories.items()):
        print(f"{category}: {len(files)} files")
        for f in files:
            print(f"  - {f.split('/')[1]}")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
