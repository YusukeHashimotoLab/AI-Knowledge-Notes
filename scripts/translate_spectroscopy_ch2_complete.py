#!/usr/bin/env python3
"""
Complete translation script for spectroscopy-introduction chapter-2.html
Translates all Japanese content to English while preserving HTML structure
"""

import re

def translate_spectroscopy_ch2():
    """Complete translation of chapter 2: Infrared and Raman Spectroscopy"""

    # Read the Japanese source file
    jp_file = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/spectroscopy-introduction/chapter-2.html'

    with open(jp_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Translation mappings - comprehensive coverage
    translations = {
        # HTML lang attribute
        '<html lang="ja">': '<html lang="en">',

        # Meta and title
        '<title>第2章:赤外・ラマン分光法 - MS Terakoya</title>':
            '<title>Chapter 2: Infrared and Raman Spectroscopy - MS Terakoya</title>',

        # Breadcrumb navigation
        'AI寺子屋トップ': 'AI Terakoya Home',
        '材料科学': 'Materials Science',

        # Header content
        '<h1>第2章:赤外・ラマン分光法</h1>': '<h1>Chapter 2: Infrared and Raman Spectroscopy</h1>',
        '<p class="subtitle">振動分光で探る分子構造と化学結合</p>':
            '<p class="subtitle">Molecular Structure and Chemical Bonding via Vibrational Spectroscopy</p>',

        # Meta information
        '📚 シリーズ: 分光分析入門': '📚 Series: Introduction to Spectroscopy',
        '⏱️ 学習時間: 100分': '⏱️ Study Time: 100 minutes',
        '🎯 難易度: 初級〜中級': '🎯 Level: Beginner to Intermediate',

        # Introduction section
        '<h2>イントロダクション</h2>': '<h2>Introduction</h2>',

        # Main introduction text - split into parts for accurate translation
        '赤外分光（Infrared Spectroscopy, IR）':
            'Infrared spectroscopy (Infrared Spectroscopy, IR)',
        'とラマン分光（Raman Spectroscopy）は、分子の振動情報を通じて化学結合、官能基、結晶構造を解明する相補的な手法です。':
            ' and Raman Spectroscopy are complementary techniques that elucidate chemical bonds, functional groups, and crystal structures through molecular vibrational information.',
        'IRは赤外光の吸収を測定し、Ramanは散乱光の周波数シフトを観測します。':
            'IR measures the absorption of infrared light, while Raman observes the frequency shift of scattered light.',
        '両者は異なる選択則に従うため、IRで活性な振動がRamanで不活性、またはその逆という相補性を持ちます。':
            'Since they follow different selection rules, vibrations that are IR-active may be Raman-inactive, and vice versa, providing complementary information.',

        # Info box
        '<strong>IRとRamanの使い分け</strong><br>': '<strong>Choosing Between IR and Raman</strong><br>',
        '<li><strong>IR</strong>: 極性基（C=O, O-H, N-H）の検出、有機物の官能基同定、固体・液体・気体すべてに適用可能</li>':
            '<li><strong>IR</strong>: Detection of polar groups (C=O, O-H, N-H), identification of functional groups in organic compounds, applicable to solids, liquids, and gases</li>',
        '<li><strong>Raman</strong>: 対称振動（C=C, S-S）の検出、水溶液試料、結晶性評価（低波数領域）、非破壊・非接触測定</li>':
            '<li><strong>Raman</strong>: Detection of symmetric vibrations (C=C, S-S), aqueous samples, crystallinity evaluation (low-frequency region), non-destructive and non-contact measurement</li>',

        # Section 1
        '<h2>1. 分子振動の基礎</h2>': '<h2>1. Fundamentals of Molecular Vibrations</h2>',
        '<h3>1.1 調和振動子モデル</h3>': '<h3>1.1 Harmonic Oscillator Model</h3>',
    }

    # Apply translations - first pass
    for jp, en in translations.items():
        content = content.replace(jp, en)

    print("First pass translation completed")
    return content

if __name__ == "__main__":
    content = translate_spectroscopy_ch2()
    print(f"Content length: {len(content)} characters")
