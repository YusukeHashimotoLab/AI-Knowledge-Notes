#!/usr/bin/env python3
"""
Automated Translation Script for Materials Properties Series
Translates Japanese HTML files to English while preserving structure

Usage:
    python3 translate_materials_properties.py

Requirements:
    pip install deep-translator beautifulsoup4 lxml

Or use Google Translate (no API key needed):
    pip install googletrans==4.0.0-rc1
"""

import os
import re
from pathlib import Path

# Comprehensive manual translations for common UI elements
MANUAL_TRANSLATIONS = {
    # HTML lang attribute
    'lang="ja"': 'lang="en"',

    # Page titles
    '第1章：固体電子論の基礎 - バンド理論入門 - MS Terakoya': 'Chapter 1: Fundamentals of Solid-State Electronic Theory - Introduction to Band Theory - MS Terakoya',
    '第2章：結晶場理論と電子状態 - MS Terakoya - 材料物性論入門': 'Chapter 2: Crystal Field Theory and Electronic States - MS Terakoya - Introduction to Materials Properties',
    '第3章：第一原理計算入門（DFT基礎） - MS Terakoya - 材料物性論入門': 'Chapter 3: Introduction to First-Principles Calculations (DFT Fundamentals) - MS Terakoya - Introduction to Materials Properties',
    '第4章：電気的・磁気的性質 | MS Terakoya - 材料物性論入門': 'Chapter 4: Electrical and Magnetic Properties | MS Terakoya - Introduction to Materials Properties',
    '第5章：光学的・熱的性質 | MS Terakoya - 材料物性論入門': 'Chapter 5: Optical and Thermal Properties | MS Terakoya - Introduction to Materials Properties',
    '第6章：実践：物性計算ワークフロー | MS Terakoya - 材料物性論入門': 'Chapter 6: Practical Materials Property Calculation Workflow | MS Terakoya - Introduction to Materials Properties',
    '材料物性論入門シリーズ - MS Terakoya': 'Introduction to Materials Properties Series - MS Terakoya',

    # Common headings
    '第1章：固体電子論の基礎': 'Chapter 1: Fundamentals of Solid-State Electronic Theory',
    '第2章：結晶場理論と電子状態': 'Chapter 2: Crystal Field Theory and Electronic States',
    '第3章：第一原理計算入門（DFT基礎）': 'Chapter 3: Introduction to First-Principles Calculations (DFT Fundamentals)',
    '第4章：電気的・磁気的性質': 'Chapter 4: Electrical and Magnetic Properties',
    '第5章：光学的・熱的性質': 'Chapter 5: Optical and Thermal Properties',
    '第6章：実践：物性計算ワークフロー': 'Chapter 6: Practical Materials Property Calculation Workflow',
    '材料物性論入門シリーズ': 'Introduction to Materials Properties Series',

    # Breadcrumbs
    'AI寺子屋トップ': 'AI Terakoya Top',
    '材料科学': 'Materials Science',

    # Meta/common UI
    '読了時間': 'Reading time',
    '難易度': 'Difficulty',
    '中級': 'Intermediate',
    '上級': 'Advanced',
    '中級〜上級': 'Intermediate-Advanced',
    'コード例': 'code examples',
    '学習目標': 'Learning Objectives',
    '学習を開始': 'Start Learning',
    '全6章構成': '6 chapters total',
    '学習時間': 'Study time',

    # Learning levels
    'この章で学ぶこと': 'What You Will Learn in This Chapter',
    '学習目標（3レベル）': 'Learning Objectives (3 Levels)',
    '基本レベル': 'Basic Level',
    '基本理解': 'Basic Understanding',
    '中級レベル': 'Intermediate Level',
    '上級レベル': 'Advanced Level',
    '実践スキル': 'Practical Skills',
    '応用力': 'Applied Capabilities',

    # Common phrases
    'まとめ': 'Summary',
    '演習問題': 'Exercises',
    '参考文献': 'References',
    '次のステップ': 'Next Steps',
    'よくある質問': 'FAQ',
    '免責事項': 'Disclaimer',
    '作成者': 'Author',
    'バージョン': 'Version',
    '作成日': 'Created',
    'ライセンス': 'License',

    # Navigation
    '← シリーズ目次': '← Series Table of Contents',
    '学習を開始 →': 'Start Learning →',
    '第1章から学習を開始': 'Start from Chapter 1',

    # Exercise difficulty
    '難易度：★☆☆': 'Difficulty: ★☆☆',
    '難易度：★★☆': 'Difficulty: ★★☆',
    '難易度：★★★': 'Difficulty: ★★★',
    '解答を見る': 'Show Answer',
    '正解': 'Correct Answer',
    '解説': 'Explanation',
    '補足': 'Additional Notes',
    'ヒント': 'Hint',
    '問題': 'Problem',
    '推奨解答': 'Recommended Answer',
    '解答例': 'Sample Answer',

    # Time estimates
    '30-35分': '30-35 minutes',
    '25-30分': '25-30 minutes',
    '35-40分': '35-40 minutes',
    '40-45分': '40-45 minutes',
    '180-220分': '180-220 minutes',
}


def apply_manual_translations(content):
    """Apply manual translations"""
    for jp, en in MANUAL_TRANSLATIONS.items():
        content = content.replace(jp, en)
    return content


def count_japanese_chars(text):
    """Count Japanese characters (hiragana, katakana, kanji)"""
    japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]'
    return len(re.findall(japanese_pattern, text))


def count_total_chars(text):
    """Count total non-whitespace characters"""
    return len(re.sub(r'\s', '', text))


def get_japanese_percentage(text):
    """Calculate percentage of Japanese characters"""
    jp_chars = count_japanese_chars(text)
    total_chars = count_total_chars(text)
    return (jp_chars / total_chars * 100) if total_chars > 0 else 0


def translate_file_basic(input_path, output_path):
    """
    Basic translation: Apply manual translations only
    This will translate common UI elements but NOT the main content
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply manual translations
    content = apply_manual_translations(content)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    jp_percentage = get_japanese_percentage(content)
    return jp_percentage


def main():
    """Main translation function"""
    base_dir = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge"
    jp_dir = f"{base_dir}/jp/MS/materials-properties-introduction"
    en_dir = f"{base_dir}/en/MS/materials-properties-introduction"

    files = [
        "index.html",
        "chapter-1.html",
        "chapter-2.html",
        "chapter-3.html",
        "chapter-4.html",
        "chapter-5.html",
        "chapter-6.html",
    ]

    print("=" * 70)
    print("Materials Properties Translation Script")
    print("=" * 70)
    print()
    print("MODE: Basic Manual Translation Only")
    print("This will translate common UI elements and structure.")
    print("Main content (paragraphs, detailed text) will require manual translation")
    print("or use of translation API (DeepL, Google Translate).")
    print()
    print("=" * 70)
    print()

    results = []
    for filename in files:
        input_path = f"{jp_dir}/{filename}"
        output_path = f"{en_dir}/{filename}"

        if not os.path.exists(input_path):
            print(f"⚠️  {filename}: SOURCE FILE NOT FOUND")
            continue

        print(f"Processing: {filename}...", end=" ")
        jp_percentage = translate_file_basic(input_path, output_path)

        status = "✅ GOOD" if jp_percentage < 1.0 else f"⚠️  {jp_percentage:.2f}% JP remaining"
        print(status)

        results.append({
            'file': filename,
            'jp_percentage': jp_percentage,
            'output': output_path
        })

    print()
    print("=" * 70)
    print("Translation Summary")
    print("=" * 70)
    print()

    for result in results:
        status_icon = "✅" if result['jp_percentage'] < 1.0 else "⚠️"
        print(f"{status_icon} {result['file']:20s}: {result['jp_percentage']:6.2f}% Japanese")

    print()
    avg_jp = sum(r['jp_percentage'] for r in results) / len(results) if results else 0
    print(f"Average Japanese content: {avg_jp:.2f}%")
    print()

    if avg_jp >= 1.0:
        print("⚠️  WARNING: Files still contain >1% Japanese content")
        print("    Manual translation or translation API required for full conversion")
        print()
        print("Next steps:")
        print("1. Use DeepL API for professional translation")
        print("2. Manual translation of remaining content")
        print("3. Review and quality check")
    else:
        print("✅ SUCCESS: All files have <1% Japanese content!")

    print("=" * 70)


if __name__ == "__main__":
    main()
