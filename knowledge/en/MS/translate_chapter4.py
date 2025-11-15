#!/usr/bin/env python3
"""
Translation script for 3d-printing-introduction chapter-4.html
Reads Japanese HTML and outputs English translation while preserving structure
"""

# Read the source file
with open('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/3d-printing-introduction/chapter-4.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Translation mapping - comprehensive Japanese to English
translations = {
    # Meta and headers
    'lang="ja"': 'lang="en"',
    '<title>第4章：材料噴射法・結合剤噴射法・その他AM技術 - MS Terakoya</title>': '<title>Chapter 4: Fundamentals of Additive Manufacturing - MS Terakoya</title>',

    # Breadcrumb
    'AI寺子屋トップ': 'AI Terakoya Top',
    '材料科学': 'Materials Science',

    # Header
    '第4章：積層造形の基礎': 'Chapter 4: Fundamentals of Additive Manufacturing',
    'AM技術の原理と分類 - 3Dプリンティングの技術体系': 'Principles and Classification of AM Technologies - 3D Printing Technical Framework',
    '📚 3Dプリンティング入門シリーズ': '📚 3D Printing Introduction Series',
    '⏱️ 読了時間: 35-40分': '⏱️ Reading time: 35-40 minutes',
    '🎓 難易度: 初級〜中級': '🎓 Difficulty: Beginner to Intermediate',

    # Learning objectives
    '学習目標': 'Learning Objectives',
    'この章を完了すると、以下を説明できるようになります：': 'Upon completing this chapter, you will be able to explain:',

    '基本理解（Level 1）': 'Basic Understanding (Level 1)',
    '積層造形（AM）の定義とISO/ASTM 52900規格の基本概念': 'Definition of Additive Manufacturing (AM) and basic concepts of ISO/ASTM 52900 standard',
    '7つのAMプロセスカテゴリ（MEX, VPP, PBF, MJ, BJ, SL, DED）の特徴': 'Characteristics of 7 AM process categories (MEX, VPP, PBF, MJ, BJ, SL, DED)',
    'STLファイル形式の構造（三角形メッシュ、法線ベクトル、頂点順序）': 'Structure of STL file format (triangle mesh, normal vectors, vertex order)',
    'AMの歴史（1986年ステレオリソグラフィから現代システムまで）': 'History of AM (from 1986 stereolithography to modern systems)',

    '実践スキル（Level 2）': 'Practical Skills (Level 2)',
    'PythonでSTLファイルを読み込み、体積・表面積を計算できる': 'Ability to read STL files in Python and calculate volume and surface area',
    'numpy-stlとtrimeshを使ったメッシュ検証と修復ができる': 'Ability to validate and repair meshes using numpy-stl and trimesh',
    'スライシングの基本原理（レイヤー高さ、シェル、インフィル）を理解': 'Understanding of basic slicing principles (layer height, shell, infill)',
    'G-codeの基本構造（G0/G1/G28/M104など）を読み解ける': 'Ability to interpret basic G-code structure (G0/G1/G28/M104, etc.)',

    '応用力（Level 3）': 'Application Skills (Level 3)',
    '用途要求に応じて最適なAMプロセスを選択できる': 'Ability to select optimal AM process according to application requirements',
    'メッシュの問題（非多様体、法線反転）を検出・修正できる': 'Ability to detect and fix mesh problems (non-manifold, inverted normals)',
    '造形パラメータ（レイヤー高さ、印刷速度、温度）を最適化できる': 'Ability to optimize build parameters (layer height, print speed, temperature)',
    'STLファイルの品質評価とプリント適性判断ができる': 'Ability to assess STL file quality and printability',
}

print(f"Original file length: {len(content)} characters")
print(f"Starting translation...")

# Apply translations
translated = content
for jp, en in translations.items():
    if jp in translated:
        translated = translated.replace(jp, en)
        print(f"✓ Translated: {jp[:50]}...")

# Write the output
output_path = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-4_partial.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(translated[:5000])

print(f"\nPartial translation written to: {output_path}")
print(f"Total translations applied: {len([k for k in translations.keys() if k in content])}")
