#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Translation Script for Ceramic Materials Introduction
Translates 5 HTML files from Japanese to English while preserving all HTML/CSS/JS/code
"""

import re

# Translation dictionaries
TRANSLATIONS = {
    # Meta and title translations
    'lang="ja"': 'lang="en"',
    'セラミックス材料入門': 'Introduction to Ceramic Materials',
    'MS Dojo': 'MS Dojo',
    'MS Terakoya': 'MS Terakoya',
    'AI寺子屋トップ': 'AI Terakoya Home',
    '材料科学': 'Materials Science',

    # Chapter titles
    '第1章': 'Chapter 1',
    '第2章': 'Chapter 2',
    '第3章': 'Chapter 3',
    '第4章': 'Chapter 4',
    '第5章': 'Chapter 5',

    # Common UI elements
    'トップ': 'Top',
    '概要': 'Overview',
    '演習問題': 'Exercises',
    '参考文献': 'References',
    '前の章': 'Previous',
    '次の章へ': 'Next Chapter',

    # Chapter 2 specific
    'セラミックス製造プロセス': 'Ceramic Manufacturing Processes',
    '粉末冶金、固相焼結、液相焼結、ゾル-ゲル法の原理とシミュレーション': 'Fundamentals and simulation of powder metallurgy, solid-state sintering, liquid-phase sintering, and sol-gel methods',
    '粉末冶金': 'Powder Metallurgy',
    '固相焼結': 'Solid-State Sintering',
    '液相焼結': 'Liquid-Phase Sintering',
    'ゾル-ゲル法': 'Sol-Gel Method',

    # Chapter 3 specific
    '機械的性質': 'Mechanical Properties',
    '脆性破壊、Griffith理論、破壊靭性、Weibull統計による信頼性評価': 'Brittle fracture, Griffith theory, fracture toughness, and reliability assessment using Weibull statistics',
    '脆性破壊': 'Brittle Fracture',
    'Griffith理論': 'Griffith Theory',
    '破壊靭性': 'Fracture Toughness',
    'Weibull統計': 'Weibull Statistics',
    '高温クリープ': 'High-Temperature Creep',

    # Index page
    'シリーズ概要': 'Series Overview',
    '学習フロー': 'Learning Flow',
    'よくある質問': 'Frequently Asked Questions',
    '難易度': 'Difficulty',
    '想定読了時間': 'Estimated Reading Time',
    '前提知識': 'Prerequisites',
    '中級': 'Intermediate',
    '各章': 'Each chapter',
    '全シリーズ': 'Total series',

    # Common content words
    '本章の学習目標': 'Learning Objectives for This Chapter',
    'レベル1（基本理解）': 'Level 1 (Basic Understanding)',
    'レベル2（実践スキル）': 'Level 2 (Practical Skills)',
    'レベル3（応用力）': 'Level 3 (Applied Competence)',

    # Python code comments - keep as is
    # Exercise labels
    '易': 'Easy',
    '中': 'Medium',
    '難': 'Hard',

    # Common phrases
    '解答例': 'Sample Answer',
    '解答を見る': 'View Answer',
    '正解': 'Correct Answer',
    '解説': 'Explanation',
    '計算過程': 'Calculation Process',
    '期待される出力': 'Expected Output',

    # Footer
    'トップページ': 'Home',
    'シリーズ目次': 'Series Index',
    'MS分野トップ': 'MS Field Top',
    'セラミックス材料入門トップへ戻る': 'Return to Ceramic Materials Introduction Top',

    # Breadcrumb
    'Ceramic Materials': 'Ceramic Materials',
    'Chapter 2': 'Chapter 2',
    'Chapter 3': 'Chapter 3',
}

def translate_file(input_path, output_path):
    """Translate a single HTML file from Japanese to English"""
    print(f"Translating {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply translations
    for jp, en in TRANSLATIONS.items():
        content = content.replace(jp, en)

    # Write translated file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Calculate Japanese percentage
    japanese_chars = len(re.findall(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', content))
    total_chars = len(content)
    jp_percentage = (japanese_chars / total_chars * 100) if total_chars > 0 else 0

    print(f"  ✓ Written to {output_path}")
    print(f"  Japanese content: {jp_percentage:.2f}%")

    return jp_percentage

def main():
    """Main translation execution"""
    base_jp = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/ceramic-materials-introduction'
    base_en = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/ceramic-materials-introduction'

    files = [
        ('chapter-2.html', 'chapter-2.html'),
        ('chapter-3.html', 'chapter-3.html'),
        ('chapter-4.html', 'chapter-4.html'),
        ('chapter-5.html', 'chapter-5.html'),
        ('index.html', 'index.html'),
    ]

    print("="*60)
    print("CERAMIC MATERIALS TRANSLATION - BATCH PROCESS")
    print("="*60)

    results = []
    for jp_file, en_file in files:
        jp_path = f"{base_jp}/{jp_file}"
        en_path = f"{base_en}/{en_file}"

        try:
            jp_pct = translate_file(jp_path, en_path)
            results.append((en_file, jp_pct))
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append((en_file, -1))

        print()

    # Final report
    print("="*60)
    print("TRANSLATION COMPLETE - FINAL REPORT")
    print("="*60)
    for filename, jp_pct in results:
        if jp_pct >= 0:
            print(f"{filename:25s} Japanese: {jp_pct:5.2f}%")
        else:
            print(f"{filename:25s} ERROR")
    print("="*60)

if __name__ == '__main__':
    main()
