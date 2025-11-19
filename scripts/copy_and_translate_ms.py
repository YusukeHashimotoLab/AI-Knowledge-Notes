#!/usr/bin/env python3
"""
Copy Japanese HTML files to English and mark for translation
Since files are too large for API translation, we'll copy them and create a translation checklist
"""

import os
import re
import sys
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).parent.parent / "knowledge"
EN_BASE = BASE_DIR / "en" / "MS"
JP_BASE = BASE_DIR / "jp" / "MS"

# Files with existing Japanese sources
FILES_TO_COPY = [
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
    "polymer-materials-introduction/chapter-1.html",
    "polymer-materials-introduction/chapter-2.html",
    "polymer-materials-introduction/chapter-3.html",
    "polymer-materials-introduction/chapter-4.html",
    "processing-introduction/chapter-1.html",
    "processing-introduction/chapter-2.html",
    "processing-introduction/chapter-3.html",
    "processing-introduction/chapter-4.html",
    "processing-introduction/chapter-5.html",
    "spectroscopy-introduction/chapter-1.html",
    "spectroscopy-introduction/chapter-2.html",
    "spectroscopy-introduction/chapter-3.html",
    "spectroscopy-introduction/chapter-4.html",
    "spectroscopy-introduction/chapter-5.html",
    "xrd-analysis-introduction/chapter-1.html",
    "xrd-analysis-introduction/chapter-2.html",
    "xrd-analysis-introduction/chapter-3.html",
    "xrd-analysis-introduction/chapter-4.html",
    "xrd-analysis-introduction/chapter-5.html",
]

def perform_basic_translation(content):
    """
    Perform basic Japanese to English translations for common terms
    This is a simple find-replace for common materials science terms
    """

    translations = {
        # Common headers and navigation
        "ホーム": "Home",
        "知識": "Knowledge",
        "材料科学": "Materials Science",
        "基礎": "Fundamentals",
        "応用": "Applications",
        "次へ": "Next",
        "前へ": "Previous",
        "目次": "Table of Contents",
        "章": "Chapter",
        "節": "Section",
        "まとめ": "Summary",
        "演習問題": "Exercises",
        "参考文献": "References",
        "キーワード": "Keywords",

        # Common materials science terms
        "結晶構造": "Crystal Structure",
        "格子定数": "Lattice Constant",
        "原子": "Atom",
        "分子": "Molecule",
        "電子": "Electron",
        "陽子": "Proton",
        "中性子": "Neutron",
        "イオン": "Ion",
        "化学結合": "Chemical Bond",
        "金属結合": "Metallic Bond",
        "共有結合": "Covalent Bond",
        "イオン結合": "Ionic Bond",
        "機械的性質": "Mechanical Properties",
        "電気的性質": "Electrical Properties",
        "磁気的性質": "Magnetic Properties",
        "熱的性質": "Thermal Properties",
        "光学的性質": "Optical Properties",

        # Action verbs
        "解説": "Explanation",
        "について": "About",
        "とは": "What is",
        "学習": "Learning",
        "理解": "Understanding",
    }

    result = content
    for jp, en in translations.items():
        result = result.replace(jp, en)

    return result

def copy_with_basic_translation(jp_file, en_file):
    """Copy file and apply basic translations"""
    try:
        # Read Japanese content
        with open(jp_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply basic translations
        translated = perform_basic_translation(content)

        # Write to English file
        with open(en_file, 'w', encoding='utf-8') as f:
            f.write(translated)

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main function"""
    print("="*80)
    print("MS HTML File Copy and Basic Translation")
    print("="*80)
    print(f"\nProcessing {len(FILES_TO_COPY)} files...\n")

    success = 0
    failed = 0

    for idx, rel_path in enumerate(FILES_TO_COPY, 1):
        jp_file = JP_BASE / rel_path
        en_file = EN_BASE / rel_path

        category = rel_path.split('/')[0]
        filename = rel_path.split('/')[1]

        print(f"[{idx}/{len(FILES_TO_COPY)}] {category}/{filename}")

        if not jp_file.exists():
            print(f"  ❌ Japanese source not found")
            failed += 1
            continue

        if copy_with_basic_translation(jp_file, en_file):
            print(f"  ✅ Copied and basic translation applied")
            success += 1
        else:
            print(f"  ❌ Failed to copy")
            failed += 1

    print("\n" + "="*80)
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Total: {len(FILES_TO_COPY)}")
    print("="*80)

    print("\nNote: These files have basic term translations applied.")
    print("Full translation of paragraph content still needs to be done manually")
    print("or through additional translation services.")

if __name__ == '__main__':
    main()
