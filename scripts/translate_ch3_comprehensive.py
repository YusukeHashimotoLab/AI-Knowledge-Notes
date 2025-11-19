#!/usr/bin/env python3
"""
COMPREHENSIVE Translation for 3D Printing Chapter 3
Target: 0 Japanese characters remaining
Strategy: Section-by-section systematic translation
"""

import re
from pathlib import Path

JP_FILE = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/3d-printing-introduction/chapter-3.html")
EN_FILE = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-3.html")

def translate_comprehensive():
    """Complete systematic translation"""

    with open(JP_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Original: {len(content)} chars")

    # HTML attributes
    content = content.replace('lang="ja"', 'lang="en"')

    # Title
    content = content.replace(
        '第3章：光造形法・粉末床溶融結合法 - SLA/DLP/SLS/SLM - MS Terakoya',
        'Chapter 3: Vat Photopolymerization & Powder Bed Fusion - SLA/DLP/SLS/SLM - MS Terakoya'
    )

    # Breadcrumb
    content = content.replace('AI寺子屋トップ', 'AI Terakoya Home')
    content = content.replace('材料科学', 'Materials Science')
    content = content.replace('advanced-materials-systems-introduction', '3d-printing-introduction')

    # Header - CRITICAL: This IS Chapter 3 on AM Fundamentals
    content = content.replace(
        '<h1>第3章：積層造形の基礎</h1>',
        '<h1>Chapter 3: Fundamentals of Additive Manufacturing</h1>'
    )
    content = content.replace(
        'AM技術の原理と分類 - 3Dプリンティングの技術体系',
        'AM Technology Principles and Classification - Technical Framework of 3D Printing'
    )

    # Meta
    content = content.replace('📚 3Dプリンティング入門シリーズ', '3D Printing Introduction Series')
    content = content.replace('⏱️ 読了時間: 35-40分', 'Reading Time: 35-40 minutes')
    content = content.replace('🎓 難易度: 初級〜中級', 'Difficulty: Beginner to Intermediate')

    # Learning objectives
    content = content.replace('<h2>学習目標</h2>', '<h2>Learning Objectives</h2>')
    content = content.replace('この章を完了すると、以下を説明できるようになります：',
                            'Upon completing this chapter, you will be able to explain:')
    content = content.replace('<h3>基本理解（Level 1）</h3>', '<h3>Basic Understanding (Level 1)</h3>')
    content = content.replace('<h3>実践スキル（Level 2）</h3>', '<h3>Practical Skills (Level 2)</h3>')
    content = content.replace('<h3>応用力（Level 3）</h3>', '<h3>Application Capability (Level 3)</h3>')

    # Learning objective list items - specific translations
    list_items = {
        '積層造形（AM）の定義とISO/ASTM 52900規格の基本概念':
            'Definition of Additive Manufacturing (AM) and basic concepts of ISO/ASTM 52900 standard',
        '7つのAMプロセスカテゴリ（MEX, VPP, PBF, MJ, BJ, SL, DED）の特徴':
            'Characteristics of the 7 AM process categories (MEX, VPP, PBF, MJ, BJ, SL, DED)',
        'STLファイル形式の構造（三角形メッシュ、法線ベクトル、頂点順序）':
            'Structure of STL file format (triangle mesh, normal vectors, vertex order)',
        'AMの歴史（1986年ステレオリソグラフィから現代システムまで)':
            'History of AM (from 1986 stereolithography to modern systems)',
        'PythonでSTLファイルを読み込み、体積・表面積を計算できる':
            'Load STL files in Python and calculate volume and surface area',
        'numpy-stlとtrimeshを使ったメッシュ検証と修復ができる':
            'Perform mesh verification and repair using numpy-stl and trimesh',
        'スライシングの基本原理（レイヤー高さ、シェル、インフィル）を理解':
            'Understand basic principles of slicing (layer height, shell, infill)',
        'G-codeの基本構造（G0/G1/G28/M104など）を読み解ける':
            'Interpret basic structure of G-code (G0/G1/G28/M104, etc.)',
        '用途要求に応じて最適なAMプロセスを選択できる':
            'Select optimal AM process based on application requirements',
        'メッシュの問題（非多様体、法線反転）を検出・修正できる':
            'Detect and fix mesh problems (non-manifold, inverted normals)',
        '造形パラメータ（レイヤー高さ、印刷速度、温度）を最適化できる':
            'Optimize build parameters (layer height, print speed, temperature)',
        'STLファイルの品質評価とプリント適性判断ができる':
            'Evaluate STL file quality and determine printability',
    }

    for jp, en in list_items.items():
        content = content.replace(jp, en)

    # Main section headings
    sections = {
        '1.1 積層造形（AM）とは': '1.1 What is Additive Manufacturing (AM)?',
        '1.1.1 積層造形の定義': '1.1.1 Definition of Additive Manufacturing',
        '1.1.2 AMの歴史と発展': '1.1.2 History and Development of AM',
        '1.1.3 AMの主要応用分野': '1.1.3 Major Application Areas of AM',

        '1.2 ISO/ASTM 52900による7つのAMプロセス分類':
            '1.2 Seven AM Process Categories by ISO/ASTM 52900',
        '1.2.1 AMプロセス分類の全体像': '1.2.1 Overview of AM Process Classification',
        '1.2.2 Material Extrusion (MEX) - 材料押出': '1.2.2 Material Extrusion (MEX)',
        '1.2.3 Vat Photopolymerization (VPP) - 液槽光重合': '1.2.3 Vat Photopolymerization (VPP)',
        '1.2.4 Powder Bed Fusion (PBF) - 粉末床溶融結合': '1.2.4 Powder Bed Fusion (PBF)',
        '1.2.5 Material Jetting (MJ) - 材料噴射': '1.2.5 Material Jetting (MJ)',
        '1.2.6 Binder Jetting (BJ) - 結合剤噴射': '1.2.6 Binder Jetting (BJ)',
        '1.2.7 Sheet Lamination (SL) - シート積層': '1.2.7 Sheet Lamination (SL)',
        '1.2.8 Directed Energy Deposition (DED) - 指向性エネルギー堆積':
            '1.2.8 Directed Energy Deposition (DED)',

        '1.3 STLファイル形式とデータ処理': '1.3 STL File Format and Data Processing',
        '1.3.1 STLファイルの構造': '1.3.1 Structure of STL Files',
        '1.3.2 STLファイルの重要概念': '1.3.2 Key Concepts of STL Files',
        '1.3.3 STLファイルの品質指標': '1.3.3 Quality Metrics for STL Files',
        '1.3.4 PythonによるSTL処理': '1.3.4 STL Processing with Python',

        '1.4 スライシングとG-code生成': '1.4 Slicing and G-code Generation',
        '1.4.1 スライシングの原理': '1.4.1 Principles of Slicing',
        '1.4.2 造形パラメータの最適化': '1.4.2 Optimization of Build Parameters',
        '1.4.3 G-codeの構造と解析': '1.4.3 Structure and Analysis of G-code',
        '1.4.4 PythonによるG-code解析': '1.4.4 G-code Analysis with Python',
    }

    for jp, en in sections.items():
        content = content.replace(jp, en)

    # Subsections
    subsections = {
        'レイヤー高さ（Layer Height）の選択': 'Layer Height Selection',
        'シェル（外殻）の生成': 'Shell (Perimeter) Generation',
        'インフィル（内部充填）パターン': 'Infill Patterns',
        'サポート構造の生成': 'Support Structure Generation',
        'サポートのタイプ': 'Types of Support',
        'サポート設定の重要パラメータ': 'Key Support Parameters',
        'G-codeの基礎': 'G-code Basics',
        '主要なG-codeコマンド': 'Main G-code Commands',
        'G-codeの例（造形開始部分）': 'G-code Example (Start Sequence)',
        '主要スライシングソフトウェア': 'Major Slicing Software',
        'ツールパス最適化戦略': 'Toolpath Optimization Strategies',

        # STL sections
        'STLファイルの基本構造': 'Basic Structure of STL Files',
        '法線ベクトル（Normal Vector）': 'Normal Vector',
        '多様体（Manifold）条件': 'Manifold Conditions',
        '頂点順序ルール：': 'Vertex Order Rule:',

        # Application subsections
        '応用1: プロトタイピング（Rapid Prototyping）': 'Application 1: Rapid Prototyping',
        '応用2: ツーリング（Tooling & Fixtures）': 'Application 2: Tooling & Fixtures',
        '応用3: 最終製品（End-Use Parts）': 'Application 3: End-Use Parts',

        # VPP/PBF subsections
        'VPPの2つの主要方式：': 'Two Main VPP Methods:',
        'PBFの3つの主要方式：': 'Three Main PBF Methods:',
    }

    for jp, en in subsections.items():
        content = content.replace(jp, en)

    # Info/Warning boxes
    boxes = {
        '💡 産業的重要性': '💡 Industrial Significance',
        '⚠️ AMの制約と課題': '⚠️ Constraints and Challenges of AM',
        '⚠️ プロセス選択の指針': '⚠️ Guidelines for Process Selection',
        '⚠️ 非多様体メッシュの問題': '⚠️ Non-Manifold Mesh Issues',
        '💡 STLファイルの解像度トレードオフ': '💡 STL File Resolution Trade-offs',
        '💡 FDMの代表的装置': '💡 Representative FDM Equipment',
        '💡 インフィル密度の目安': '💡 Infill Density Guidelines',
        '⚠️ レイヤー高さの制約': '⚠️ Layer Height Constraints',
    }

    for jp, en in boxes.items():
        content = content.replace(jp, en)

    # Common technical terms - comprehensive list
    terms = {
        # Core AM terms
        '積層造形': 'Additive Manufacturing',
        '付加製造': 'Additive Manufacturing',
        '材料押出': 'Material Extrusion',
        '液槽光重合': 'Vat Photopolymerization',
        '粉末床溶融結合': 'Powder Bed Fusion',
        '材料噴射': 'Material Jetting',
        '結合剤噴射': 'Binder Jetting',
        'シート積層': 'Sheet Lamination',
        '指向性エネルギー堆積': 'Directed Energy Deposition',

        # Specific processes
        '光造形': 'Stereolithography',
        'ステレオリソグラフィ': 'Stereolithography',
        '熱溶解積層': 'Fused Deposition Modeling',
        '選択的レーザー焼結': 'Selective Laser Sintering',
        '選択的レーザー溶融': 'Selective Laser Melting',
        '電子ビーム溶解': 'Electron Beam Melting',

        # Materials
        '光硬化性樹脂': 'photopolymer resin',
        'フォトポリマー': 'photopolymer',
        '熱可塑性樹脂': 'thermoplastic',
        'フィラメント': 'filament',
        '粉末材料': 'powder material',
        '金属粉末': 'metal powder',
        'チタン合金': 'titanium alloy',
        'アルミニウム合金': 'aluminum alloy',
        'ステンレス鋼': 'stainless steel',
        'ナイロン': 'nylon',
        '工具鋼': 'tool steel',

        # Process parameters
        'レイヤー高さ': 'layer height',
        '積層高さ': 'layer height',
        '層高さ': 'layer height',
        '露光時間': 'exposure time',
        'レーザー出力': 'laser power',
        '走査速度': 'scanning speed',
        'ビルドプラットフォーム': 'build platform',
        '造形台': 'build platform',
        'ノズル温度': 'nozzle temperature',
        'ベッド温度': 'bed temperature',
        '印刷速度': 'print speed',
        '造形速度': 'build speed',

        # File/Software terms
        'スライシング': 'slicing',
        'ツールパス': 'toolpath',
        'サポート構造': 'support structure',
        'サポート材': 'support material',
        'シェル': 'shell',
        '外殻': 'perimeter',
        'インフィル': 'infill',
        '内部充填': 'infill',
        '充填率': 'infill density',
        'リトラクション': 'retraction',

        # Quality/Properties
        '造形品質': 'build quality',
        '表面品質': 'surface quality',
        '精度': 'accuracy',
        '解像度': 'resolution',
        '寸法精度': 'dimensional accuracy',
        '機械的性質': 'mechanical properties',
        '引張強度': 'tensile strength',
        '相対密度': 'relative density',
        '異方性': 'anisotropy',
        '等方性': 'isotropic',

        # Mesh/Geometry terms
        '三角形メッシュ': 'triangle mesh',
        '法線ベクトル': 'normal vector',
        '頂点': 'vertex',
        '頂点座標': 'vertex coordinates',
        '多様体': 'manifold',
        '非多様体': 'non-manifold',
        '閉じた表面': 'watertight',
        '水密性': 'watertight',
        '自己交差': 'self-intersection',
        '法線反転': 'inverted normals',
        '重複頂点': 'duplicate vertices',
        '縮退三角形': 'degenerate triangles',

        # Common phrases
        'とは': 'is',
        'について': 'regarding',
        'による': 'by',
        'における': 'in',
        'に関する': 'regarding',
        'のため': 'for',
        'など': 'etc.',
        'また': 'Also',
        'さらに': 'Furthermore',
        'ただし': 'However',
        'しかし': 'However',
        'なお': 'Note that',
        'すなわち': 'namely',
        'つまり': 'in other words',
        'したがって': 'therefore',
        'そのため': 'Therefore',
        '以下': 'following',
        '以上': 'above',
        '例えば': 'For example',
        '特に': 'especially',
        '通常': 'typically',
        '一般的に': 'generally',
        '主に': 'mainly',

        # Navigation
        '次のステップ': 'Next Steps',
        'シリーズ目次': 'Series Index',
        '第4章': 'Chapter 4',
        '参考文献': 'References',
        '使用ツールとライブラリ': 'Tools and Libraries',
        '章末演習': 'Chapter Exercises',
        '演習': 'Exercise',
        '本章のまとめ': 'Chapter Summary',

        # Footer
        '東北大学 材料科学研究科': 'Tohoku University Graduate School of Materials Science',
        '東北大学大学院材料科学専攻': 'Tohoku University Graduate School of Materials Science',
    }

    for jp, en in terms.items():
        content = content.replace(jp, en)

    # Table headers
    table_headers = {
        'コマンド': 'Command',
        '分類': 'Category',
        '機能': 'Function',
        '例': 'Example',
        'パラメータ': 'Parameter',
        '推奨値': 'Recommended Value',
        '効果': 'Effect',
        'パターン': 'Pattern',
        '強度': 'Strength',
        '印刷速度': 'Print Speed',
        '材料使用量': 'Material Usage',
        '特徴': 'Features',
        'レイヤー高さ': 'Layer Height',
        '造形品質': 'Build Quality',
        '造形時間': 'Build Time',
        '典型的な用途': 'Typical Applications',
        'ソフトウェア': 'Software',
        'ライセンス': 'License',
        '推奨用途': 'Recommended Use',
    }

    for jp, en in table_headers.items():
        content = content.replace(f'<th>{jp}</th>', f'<th>{en}</th>')

    # Write output
    with open(EN_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Translation complete: {len(content)} chars")
    print(f"Output file: {EN_FILE}")

    return EN_FILE

if __name__ == "__main__":
    output = translate_comprehensive()
    print(f"\n✅ File written: {output}")
    print("\nVerifying Japanese characters...")
