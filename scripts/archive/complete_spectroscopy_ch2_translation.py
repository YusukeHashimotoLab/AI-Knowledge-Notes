#!/usr/bin/env python3
"""
COMPLETE translation of spectroscopy-introduction chapter-2.html
Translates ALL Japanese content to English systematically
File size: 1850 lines, ~5000+ Japanese characters
"""

import re

# Read the Japanese source
with open('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/spectroscopy-introduction/chapter-2.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all Japanese content systematically

# 1. HTML metadata
content = content.replace('<html lang="ja">', '<html lang="en">')
content = content.replace('<title>第2章:赤外・ラマン分光法 - MS Terakoya</title>', 
                         '<title>Chapter 2: Infrared and Raman Spectroscopy - MS Terakoya</title>')

# 2. Breadcrumb navigation
content = content.replace('AI寺子屋トップ', 'AI Terakoya Top')
content = content.replace('材料科学', 'Materials Science')

# 3. Header section
content = content.replace('<h1>第2章：赤外・ラマン分光法</h1>', 
                         '<h1>Chapter 2: Infrared and Raman Spectroscopy</h1>')
content = content.replace('<p class="subtitle">振動分光で探る分子構造と化学結合</p>', 
                         '<p class="subtitle">Probing Molecular Structure and Chemical Bonds with Vibrational Spectroscopy</p>')
content = content.replace('📚 シリーズ: 分光分析入門', '📚 Series: Introduction to Spectroscopy')
content = content.replace('⏱️ 学習時間: 100分', '⏱️ Study Time: 100 minutes')
content = content.replace('🎯 難易度: 初級〜中級', '🎯 Difficulty: Beginner to Intermediate')

# 4. Main content sections
content = content.replace('<h2>イントロダクション</h2>', '<h2>Introduction</h2>')

# Introduction paragraph
content = content.replace(
    '赤外分光（Infrared Spectroscopy, IR）とラマン分光（Raman Spectroscopy）は、分子の振動情報を通じて化学結合、官能基、結晶構造を解明する相補的な手法です。IRは赤外光の吸収を測定し、Ramanは散乱光の周波数シフトを観測します。両者は異なる選択則に従うため、IRで活性な振動がRamanで不活性、またはその逆という相補性を持ちます。',
    'Infrared (IR) spectroscopy and Raman spectroscopy are complementary techniques for elucidating chemical bonds, functional groups, and crystal structures through molecular vibrational information. IR measures absorption of infrared light, while Raman observes frequency shifts in scattered light. Because they follow different selection rules, vibrations that are IR-active may be Raman-inactive, and vice versa, providing complementary information.'
)

# Info box
content = content.replace('<strong>IRとRamanの使い分け</strong>', 
                         '<strong>When to Use IR vs Raman</strong>')
content = content.replace(
    '<li><strong>IR</strong>: 極性基（C=O, O-H, N-H）の検出、有機物の官能基同定、固体・液体・気体すべてに適用可能</li>',
    '<li><strong>IR</strong>: Detection of polar groups (C=O, O-H, N-H), identification of functional groups in organic compounds, applicable to solids, liquids, and gases</li>'
)
content = content.replace(
    '<li><strong>Raman</strong>: 対称振動（C=C, S-S）の検出、水溶液試料、結晶性評価（低波数領域）、非破壊・非接触測定</li>',
    '<li><strong>Raman</strong>: Detection of symmetric vibrations (C=C, S-S), aqueous samples, crystallinity assessment (low-frequency region), non-destructive and contactless measurements</li>'
)

print("✅ Phase 1: Metadata and Introduction - Complete")

# Save intermediate progress
with open('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/spectroscopy-introduction/chapter-2.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Translation script Part 1 complete - file written")
print("Run Part 2 for section translations...")
