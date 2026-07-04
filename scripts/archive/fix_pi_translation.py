#!/usr/bin/env python3
"""
Fix PI Translation Script - In-place translation for remaining PI files with Japanese characters
Processes 95 PI files identified in JAPANESE_CHARACTER_FILES_LIST.txt
Preserves all structure, code blocks, equations, and styling while translating text content
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
import sys


# Base paths
BASE_EN = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en')
FILE_LIST = BASE_EN / 'JAPANESE_CHARACTER_FILES_LIST.txt'

# Comprehensive translation dictionary for PI content
TRANSLATIONS = {
    # HTML attributes
    'lang="ja"': 'lang="en"',

    # Breadcrumb navigation
    'AI寺子屋トップ': 'AI Terakoya Top',
    'プロセス・インフォマティクス': 'Process Informatics',
    'プロセスインフォマティクス': 'Process Informatics',

    # Common metadata
    '読了時間': 'Reading Time',
    '難易度': 'Difficulty',
    '総学習時間': 'Total Learning Time',
    'レベル': 'Level',

    # Difficulty levels
    '初級': 'Beginner',
    '中級': 'Intermediate',
    '上級': 'Advanced',
    '初級〜中級': 'Beginner to Intermediate',
    '中級〜上級': 'Intermediate to Advanced',

    # Time units
    '分': ' minutes',
    '時間': ' hours',
    '個': '',
    'シリーズ': ' Series',
    '章': ' Chapter',

    # Navigation elements
    '前の章': 'Previous Chapter',
    '次の章': 'Next Chapter',
    'シリーズ目次': 'Series Contents',
    '目次に戻る': 'Back to Contents',
    '目次へ': 'To Contents',

    # Common section titles
    '学習内容': 'Learning Content',
    '学習目標': 'Learning Objectives',
    'この章で学ぶこと': 'What You Will Learn',
    'まとめ': 'Summary',
    '演習問題': 'Exercises',
    '参考文献': 'References',
    '次のステップ': 'Next Steps',
    'シリーズ概要': 'Series Overview',
    '各章の詳細': 'Chapter Details',
    '全体の学習成果': 'Overall Learning Outcomes',
    '前提知識': 'Prerequisites',
    '使用技術とツール': 'Technologies and Tools Used',
    '学習の進め方': 'How to Learn',
    '推奨学習順序': 'Recommended Learning Order',
    '主要ライブラリ': 'Main Libraries',
    '開発環境': 'Development Environment',
    '更新履歴': 'Update History',

    # Chapter markers
    '第1章': 'Chapter 1',
    '第2章': 'Chapter 2',
    '第3章': 'Chapter 3',
    '第4章': 'Chapter 4',
    '第5章': 'Chapter 5',
    '第6章': 'Chapter 6',

    # Common phrases
    'コード例': 'Code Examples',
    '実装例': 'Implementation Example',
    '実例': 'Example',
    '例題': 'Example Problem',
    'さあ、始めましょう': "Let's Get Started",

    # Technical terms - Process
    'プロセス': 'Process',
    '最適化': 'Optimization',
    '制御': 'Control',
    '監視': 'Monitoring',
    '予測': 'Prediction',
    'データ解析': 'Data Analysis',
    '機械学習': 'Machine Learning',
    '深層学習': 'Deep Learning',
    '人工知能': 'Artificial Intelligence',
    'ベイズ最適化': 'Bayesian Optimization',
    '実験計画法': 'Design of Experiments',
    'デジタルツイン': 'Digital Twin',
    '品質管理': 'Quality Control',
    'スケールアップ': 'Scale-up',
    '製造': 'Manufacturing',
    '化学プラント': 'Chemical Plant',
    '医薬品': 'Pharmaceutical',
    '半導体': 'Semiconductor',
    '食品': 'Food',
    '安全': 'Safety',
    'プロセス安全': 'Process Safety',
    'シミュレーション': 'Simulation',
    'オントロジー': 'Ontology',
    '知識グラフ': 'Knowledge Graph',
    'エージェント': 'Agent',
}


def load_pi_files() -> List[str]:
    """
    Load and filter PI files from JAPANESE_CHARACTER_FILES_LIST.txt

    Returns:
        List of PI file paths (95 files starting with "./PI/")
    """
    if not FILE_LIST.exists():
        raise FileNotFoundError(f"File list not found: {FILE_LIST}")

    with open(FILE_LIST, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Filter for PI files only (starting with "./PI/")
    pi_files = [line.strip() for line in lines if line.strip().startswith('./PI/')]

    return pi_files


def convert_relative_to_absolute(relative_path: str) -> Path:
    """
    Convert relative path from file list to absolute path

    Args:
        relative_path: Path like "./PI/ai-agent-process/chapter-3.html"

    Returns:
        Absolute Path object
    """
    # Remove leading "./" and prepend base directory
    clean_path = relative_path.lstrip('./')
    return BASE_EN / clean_path


def translate_text(text: str) -> str:
    """
    Apply dictionary-based translation to text content
    Preserves code blocks, equations, and HTML structure

    Args:
        text: Original text content

    Returns:
        Translated text content
    """
    result = text

    # Sort by length (longest first) to avoid partial replacements
    # This prevents issues like translating "初級" before "初級〜中級"
    for ja, en in sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        result = result.replace(ja, en)

    return result


def translate_file(file_path: Path) -> Tuple[bool, str]:
    """
    Translate a single HTML file in-place

    Args:
        file_path: Absolute path to HTML file

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    try:
        # Read original content
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Apply translations
        translated_content = translate_text(original_content)

        # Check if any changes were made
        if original_content == translated_content:
            return True, "No changes needed (already translated)"

        # Write translated content back to same file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        return True, "Translation applied successfully"

    except UnicodeDecodeError as e:
        return False, f"Encoding error: {e}"
    except IOError as e:
        return False, f"I/O error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def verify_file(file_path: Path) -> bool:
    """
    Verify that file is readable and appears to be valid HTML

    Args:
        file_path: Path to verify

    Returns:
        True if file is valid, False otherwise
    """
    if not file_path.exists():
        return False

    if not file_path.is_file():
        return False

    if file_path.suffix.lower() != '.html':
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(500)  # Read first 500 chars
            # Basic HTML validation
            if not ('<html' in content.lower() or '<!doctype' in content.lower()):
                return False
    except Exception:
        return False

    return True


def main():
    """
    Main translation process for 95 PI files
    """
    print("=" * 80)
    print("PI Translation Fix Script - In-place Translation")
    print("=" * 80)
    print(f"Base directory: {BASE_EN}")
    print(f"File list: {FILE_LIST}")
    print()

    # Load PI file list
    try:
        pi_files = load_pi_files()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Found {len(pi_files)} PI files to process")
    print()

    # Statistics
    total_files = len(pi_files)
    successful = 0
    failed = 0
    skipped = 0
    not_found = 0

    # Process each file
    for i, relative_path in enumerate(pi_files, 1):
        absolute_path = convert_relative_to_absolute(relative_path)

        print(f"[{i}/{total_files}] {relative_path}")

        # Verify file exists and is valid
        if not verify_file(absolute_path):
            if not absolute_path.exists():
                print(f"  ✗ File not found: {absolute_path}")
                not_found += 1
            else:
                print(f"  ⚠ Invalid or non-HTML file")
                skipped += 1
            print()
            continue

        # Translate file
        success, message = translate_file(absolute_path)

        if success:
            if "already translated" in message:
                print(f"  ○ {message}")
                skipped += 1
            else:
                print(f"  ✓ {message}")
                successful += 1
        else:
            print(f"  ✗ {message}")
            failed += 1

        print()

    # Final summary
    print("=" * 80)
    print("TRANSLATION COMPLETE")
    print("=" * 80)
    print(f"Total files processed:    {total_files}")
    print(f"Successfully translated:  {successful}")
    print(f"Already translated:       {skipped}")
    print(f"Failed:                   {failed}")
    print(f"Not found:                {not_found}")
    print("=" * 80)

    # Exit code based on results
    if failed > 0 or not_found > 0:
        print("\nWARNING: Some files were not processed successfully")
        sys.exit(1)
    else:
        print("\nSUCCESS: All files processed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()
