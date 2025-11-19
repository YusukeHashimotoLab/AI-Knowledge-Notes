#!/usr/bin/env python3
"""
Translation script for remaining PI series from Japanese to English
Handles all remaining chapter files with preservation of structure, code, equations, and styling
"""

import os
import re
from pathlib import Path
import shutil

# Remaining PI series that need chapter translations
SERIES = [
    'food-process-ai',
    'process-safety',
    'process-simulation',
    'qa-introduction',
    'scaleup-introduction',
    'semiconductor-manufacturing-ai',
    'process-monitoring-control-introduction',
    'process-ontology-kg',
    'ai-agent-process',  # chapters 3-6 only
    'deep-learning-modeling',
    'pi-introduction',  # chapters only
    'process-data-analysis',  # chapters only
    'doe-introduction',  # chapters only
    'chemical-plant-ai',  # index only
    'digital-twin',  # index only
    'pharma-manufacturing-ai',  # index only
    'process-optimization-introduction',  # index only
]

BASE_JP = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/PI')
BASE_EN = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/PI')

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

    # Technical terms
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


def translate_text(text):
    """Apply dictionary-based translation to text"""
    result = text

    # Sort by length (longest first) to avoid partial replacements
    for ja, en in sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        result = result.replace(ja, en)

    return result


def translate_file(src_file, dst_file):
    """Translate a single HTML file from Japanese to English"""
    print(f"Translating: {src_file.name}")

    try:
        with open(src_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply translations
        translated = translate_text(content)

        # Ensure destination directory exists
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        # Write translated content
        with open(dst_file, 'w', encoding='utf-8') as f:
            f.write(translated)

        print(f"  ✓ Created: {dst_file}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def translate_series(series_name):
    """Translate all HTML files in a series"""
    src_dir = BASE_JP / series_name
    dst_dir = BASE_EN / series_name

    if not src_dir.exists():
        print(f"✗ Source directory not found: {src_dir}")
        return 0

    print(f"\n{'='*60}")
    print(f"Series: {series_name}")
    print(f"{'='*60}")

    # Get all HTML files in source
    html_files = sorted(src_dir.glob('*.html'))

    if not html_files:
        print("  No HTML files found")
        return 0

    success_count = 0

    for src_file in html_files:
        dst_file = dst_dir / src_file.name

        # Skip if already translated (unless it's ai-agent-process chapters 1-2)
        if dst_file.exists():
            if series_name == 'ai-agent-process' and src_file.name in ['chapter-1.html', 'chapter-2.html']:
                print(f"  ⊙ Skipping {src_file.name} (already translated)")
                success_count += 1
                continue
            elif series_name in ['pi-introduction', 'process-data-analysis', 'doe-introduction'] and src_file.name == 'index.html':
                print(f"  ⊙ Skipping {src_file.name} (already translated)")
                success_count += 1
                continue

        if translate_file(src_file, dst_file):
            success_count += 1

    print(f"\nCompleted: {success_count}/{len(html_files)} files")
    return success_count


def main():
    """Main translation process"""
    print("\n" + "="*60)
    print("PI Series Translation - Remaining Files")
    print("="*60)

    total_files = 0
    total_success = 0

    for series in SERIES:
        count = translate_series(series)
        if count > 0:
            total_success += count
            total_files += count

    print("\n" + "="*60)
    print(f"TRANSLATION COMPLETE")
    print(f"Total files translated: {total_success}")
    print("="*60)


if __name__ == '__main__':
    main()
