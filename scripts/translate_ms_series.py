#!/usr/bin/env python3
"""
MS Series Translation Script
Translates Japanese MS series HTML files to English while preserving structure
"""

import os
import re
from pathlib import Path

# Translation mappings for common terms and phrases
TRANSLATIONS = {
    # Meta and titles
    "lang=\"ja\"": "lang=\"en\"",
    "シリーズ - MS Terakoya": "Series - MS Terakoya",
    "入門シリーズ": "Introduction Series",
    "結晶学入門シリーズ": "Introduction to Crystallography Series",
    "材料物性入門シリーズ": "Introduction to Materials Properties Series",
    "金属材料入門シリーズ": "Introduction to Metallic Materials Series",
    "セラミックス材料入門シリーズ": "Introduction to Ceramic Materials Series",
    "複合材料入門シリーズ": "Introduction to Composite Materials Series",
    "電子顕微鏡入門シリーズ": "Introduction to Electron Microscopy Series",

    # Navigation and breadcrumbs
    "AI寺子屋トップ": "AI Terakoya Home",
    "材料科学": "Materials Science",

    # Common UI elements
    "学習を開始 →": "Start Learning →",
    "第1章から学習を開始 →": "Start with Chapter 1 →",
    "全5章構成": "5 Chapters",
    "全6章構成": "6 Chapters",
    "学習時間:": "Study Time:",
    "コード例:": "Code Examples:",
    "難易度:": "Level:",
    "初級": "Beginner",
    "中級": "Intermediate",
    "上級": "Advanced",
    "初級〜中級": "Beginner-Intermediate",

    # Chapter labels
    "第1章": "Chapter 1",
    "第2章": "Chapter 2",
    "第3章": "Chapter 3",
    "第4章": "Chapter 4",
    "第5章": "Chapter 5",
    "第6章": "Chapter 6",

    # Section headers
    "シリーズ概要": "Series Overview",
    "学習の流れ": "Learning Path",
    "シリーズ構成": "Series Structure",
    "学習目標": "Learning Objectives",
    "推奨学習パターン": "Recommended Learning Patterns",
    "前提知識": "Prerequisites",
    "使用するPythonライブラリ": "Python Libraries Used",
    "FAQ - よくある質問": "FAQ - Frequently Asked Questions",
    "学習のポイント": "Key Learning Points",
    "次のステップ": "Next Steps",
    "免責事項": "Disclaimer",

    # Learning patterns
    "パターン1: 初学者向け - 順序通り学習": "Pattern 1: For Beginners - Sequential Learning",
    "パターン2: 中級者向け - 集中学習": "Pattern 2: For Intermediate - Intensive Learning",
    "パターン3: 実践重視 - コーディング中心": "Pattern 3: Practice-Focused - Coding-Centric",

    # Time units
    "分": "min",
    "時間": "hours",
    "日間": "days",
    "日目:": "Day:",

    # Common phrases
    "を学びます": "will be covered",
    "を理解": "understand",
    "を習得": "master",
    "ができる": "able to",
    "について": "regarding",
    "に必要": "necessary for",
    "の基礎": "fundamentals of",
    "の応用": "applications of",

    # Technical terms (preserve some, translate others)
    "コード例": "code examples",
    "実践": "practice",
    "計算": "calculation",
    "解析": "analysis",
    "可視化": "visualization",

    # Common descriptions
    "このシリーズを完了することで、以下のスキルと知識を習得できます：": "Upon completing this series, you will acquire the following skills and knowledge:",
    "本シリーズは": "This series",
    "入門コースです": "is an introductory course",

    # Footer elements
    "作成者": "Author",
    "バージョン": "Version",
    "作成日": "Created",
    "ライセンス": "License",

    # Table headers
    "分野": "Field",
    "必要レベル": "Required Level",
    "説明": "Description",

    # Prerequisites
    "高校〜大学初年次": "High School to College Freshman",
    "高校レベル": "High School Level",
    "入門〜初級": "Introductory to Beginner",
    "入門レベル（推奨）": "Introductory Level (Recommended)",
    "必須ではない": "Not required",

    # Subjects
    "化学": "Chemistry",
    "数学": "Mathematics",
    "物理学": "Physics",
    "材料科学": "Materials Science",

    # Common technical terms
    "原子": "atom",
    "分子": "molecule",
    "化学結合": "chemical bond",
    "周期表": "periodic table",
    "基本知識": "basic knowledge",
    "基本文法": "basic syntax",
    "基礎知識": "fundamental knowledge",
}

def translate_text(text, preserve_technical=True):
    """
    Translate Japanese text to English
    Args:
        text: Japanese text to translate
        preserve_technical: If True, preserve technical terms and code
    """
    # Apply translation mappings
    for jp, en in TRANSLATIONS.items():
        text = text.replace(jp, en)

    return text

def translate_html_file(source_path, target_path):
    """
    Translate an HTML file from Japanese to English
    """
    print(f"Translating: {source_path} -> {target_path}")

    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Translate content
    translated = translate_text(content)

    # Write translated content
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(translated)

    print(f"✓ Completed: {target_path}")

def translate_series(series_name, source_base, target_base):
    """
    Translate an entire series
    """
    source_dir = Path(source_base) / series_name
    target_dir = Path(target_base) / series_name

    print(f"\n{'='*60}")
    print(f"Translating series: {series_name}")
    print(f"{'='*60}")

    # Find all HTML files
    html_files = sorted(source_dir.glob("*.html"))

    for html_file in html_files:
        target_file = target_dir / html_file.name
        translate_html_file(str(html_file), str(target_file))

    print(f"✓ Series completed: {series_name} ({len(html_files)} files)")

def main():
    """Main translation workflow"""
    source_base = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS"
    target_base = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS"

    series_list = [
        "crystallography-introduction",
        "materials-properties-introduction",
        "metallic-materials-introduction",
        "ceramic-materials-introduction",
        "composite-materials-introduction",
        "electron-microscopy-introduction"
    ]

    print("MS Series Translation Tool")
    print("="*60)
    print(f"Source: {source_base}")
    print(f"Target: {target_base}")
    print(f"Series count: {len(series_list)}")
    print("="*60)

    for series in series_list:
        translate_series(series, source_base, target_base)

    print("\n" + "="*60)
    print("✓ All translations completed!")
    print("="*60)

if __name__ == "__main__":
    main()
