#!/usr/bin/env python3
"""
Comprehensive Translation Script for 3D Printing Chapter 2
Translates ALL Japanese content to English - complete file translation
"""

import re
import sys

def create_comprehensive_translation_dict():
    """Create complete translation mapping"""
    return {
        # ============================================
        # METADATA & STRUCTURE
        # ============================================
        '<html lang="ja">': '<html lang="en">',
        '第2章：材料押出法（FDM/FFF）- 熱可塑性プラスチックの積層造形 - MS Terakoya':
            'Chapter 2: Fundamentals of Additive Manufacturing - MS Terakoya',

        # Breadcrumb
        'AI寺子屋トップ': 'AI Terakoya Top',
        '材料科学': 'Materials Science',

        # ============================================
        # HEADER SECTION
        # ============================================
        '第2章：積層造形の基礎': 'Chapter 2: Fundamentals of Additive Manufacturing',
        'AM技術の原理と分類 - 3Dプリンティングの技術体系':
            'AM Technology Principles and Classification - 3D Printing Technology Framework',
        '3Dプリンティング入門シリーズ': '3D Printing Introduction Series',
        '読了時間: 35-40分': 'Reading time: 35-40 minutes',
        '難易度: 初級〜中級': 'Level: Beginner to Intermediate',

        # ============================================
        # LEARNING OBJECTIVES
        # ============================================
        '学習目標': 'Learning Objectives',
        'この章を完了すると、以下を説明できるようになります：':
            'Upon completing this chapter, you will be able to explain:',
        '基本理解（Level 1)': 'Basic Understanding (Level 1)',
        '実践スキル（Level 2)': 'Practical Skills (Level 2)',
        '応用力（Level 3)': 'Applied Competence (Level 3)',

        # Learning objectives bullets
        '積層造形（AM）の定義とISO/ASTM 52900規格の基本概念':
            'Definition of Additive Manufacturing (AM) and basic concepts of ISO/ASTM 52900 standard',
        '7つのAMプロセスカテゴリ（MEX, VPP, PBF, MJ, BJ, SL, DED）の特徴':
            'Characteristics of the seven AM process categories (MEX, VPP, PBF, MJ, BJ, SL, DED)',
        'STLファイル形式の構造（三角形メッシュ、法線ベクトル、頂点順序）':
            'Structure of STL file format (triangle mesh, normal vectors, vertex order)',
        'AMの歴史（1986年ステレオリソグラフィから現代システムまで）':
            'History of AM (from 1986 stereolithography to modern systems)',
        'PythonでSTLファイルを読み込み、体積・表面積を計算できる':
            'Ability to load STL files in Python and calculate volume and surface area',
        'numpy-stlとtrimeshを使ったメッシュ検証と修復ができる':
            'Ability to validate and repair meshes using numpy-stl and trimesh',
        'スライシングの基本原理（レイヤー高さ、シェル、インフィル）を理解':
            'Understanding of basic slicing principles (layer height, shells, infill)',
        'G-codeの基本構造（G0/G1/G28/M104など）を読み解ける':
            'Ability to interpret basic G-code structure (G0/G1/G28/M104, etc.)',
        '用途要求に応じて最適なAMプロセスを選択できる':
            'Ability to select optimal AM process based on application requirements',
        'メッシュの問題（非多様体、法線反転）を検出・修正できる':
            'Ability to detect and fix mesh problems (non-manifold, inverted normals)',
        '造形パラメータ（レイヤー高さ、印刷速度、温度）を最適化できる':
            'Ability to optimize build parameters (layer height, print speed, temperature)',
        'STLファイルの品質評価とプリント適性判断ができる':
            'Ability to evaluate STL file quality and assess printability',

        # ============================================
        # MAIN SECTION HEADINGS
        # ============================================
        '1.1 積層造形（AM）とは': '1.1 What is Additive Manufacturing (AM)',
        '1.1.1 積層造形の定義': '1.1.1 Definition of Additive Manufacturing',
        '1.1.2 AMの歴史と発展': '1.1.2 History and Evolution of AM',
        '1.1.3 AMの主要応用分野': '1.1.3 Major Application Areas of AM',

        '1.2 ISO/ASTM 52900による7つのAMプロセス分類':
            '1.2 Seven AM Process Categories by ISO/ASTM 52900',
        '1.2.1 AMプロセス分類の全体像': '1.2.1 Overview of AM Process Classification',
        '1.2.2 Material Extrusion (MEX) - 材料押出':
            '1.2.2 Material Extrusion (MEX)',
        '1.2.3 Vat Photopolymerization (VPP) - 液槽光重合':
            '1.2.3 Vat Photopolymerization (VPP)',
        '1.2.4 Powder Bed Fusion (PBF) - 粉末床溶融結合':
            '1.2.4 Powder Bed Fusion (PBF)',
        '1.2.5 Material Jetting (MJ) - 材料噴射':
            '1.2.5 Material Jetting (MJ)',
        '1.2.6 Binder Jetting (BJ) - 結合剤噴射':
            '1.2.6 Binder Jetting (BJ)',
        '1.2.7 Sheet Lamination (SL) - シート積層':
            '1.2.7 Sheet Lamination (SL)',
        '1.2.8 Directed Energy Deposition (DED) - 指向性エネルギー堆積':
            '1.2.8 Directed Energy Deposition (DED)',

        '1.3 STLファイル形式とデータ処理': '1.3 STL File Format and Data Processing',
        '1.3.1 STLファイルの構造': '1.3.1 STL File Structure',
        '1.3.2 STLファイルの重要概念': '1.3.2 Important Concepts in STL Files',
        '1.3.3 STLファイルの品質指標': '1.3.3 STL File Quality Metrics',
        '1.3.4 Pythonライブラリによる STL処理': '1.3.4 STL Processing with Python Libraries',

        '1.4 スライシングとツールパス生成': '1.4 Slicing and Toolpath Generation',
        '1.4.1 スライシングの基本原理': '1.4.1 Basic Principles of Slicing',
        '1.4.2 シェルとインフィル戦略': '1.4.2 Shell and Infill Strategies',
        '1.4.3 サポート構造の生成': '1.4.3 Support Structure Generation',
        '1.4.4 G-codeの基礎': '1.4.4 G-code Fundamentals',
        '1.4.5 主要スライシングソフトウェア': '1.4.5 Major Slicing Software',
        '1.4.6 ツールパス最適化戦略': '1.4.6 Toolpath Optimization Strategies',

        # ============================================
        # CONTENT DESCRIPTIONS
        # ============================================
        # AM Definition
        '積層造形（Additive Manufacturing, AM）とは、':
            'Additive Manufacturing (AM) is ',
        'ISO/ASTM 52900:2021規格で定義される「3次元CADデータから材料を層ごとに積み上げて物体を製造するプロセス」':
            'defined by the ISO/ASTM 52900:2021 standard as "a process of joining materials to make parts from 3D model data, usually layer upon layer"',
        '従来の切削加工（除去加工）とは対照的に、必要な部分にのみ材料を付加するため、以下の革新的な特徴を持ちます：':
            'In contrast to traditional subtractive manufacturing (machining), AM adds material only where needed, providing the following innovative features:',

        # AM Features
        '設計自由度': 'Design freedom',
        '従来製法では不可能な複雑形状（中空構造、ラティス構造、トポロジー最適化形状）を製造可能':
            'Enables fabrication of complex geometries impossible with traditional methods (hollow structures, lattice structures, topology-optimized shapes)',
        '材料効率': 'Material efficiency',
        '必要な部分にのみ材料を使用するため、材料廃棄率が5-10%（従来加工は30-90%廃棄）':
            'Material waste rate of 5-10% as material is used only where needed (traditional machining: 30-90% waste)',
        'オンデマンド製造': 'On-demand manufacturing',
        '金型不要でカスタマイズ製品を少量・多品種生産可能':
            'Enables low-volume, high-variety production of customized products without tooling',
        '一体化製造': 'Integrated manufacturing',
        '従来は複数部品を組立てていた構造を一体造形し、組立工程を削減':
            'Consolidates structures that previously required assembly of multiple parts, reducing assembly steps',

        # ============================================
        # INFO BOXES
        # ============================================
        '💡 産業的重要性': '💡 Industrial Significance',
        'AM市場は急成長中で、Wohlers Report 2023によると：':
            'The AM market is growing rapidly. According to Wohlers Report 2023:',
        '世界のAM市場規模: $18.3B（2023年）→ $83.9B予測（2030年、年成長率23.5%）':
            'Global AM market size: $18.3B (2023) → $83.9B forecast (2030, 23.5% CAGR)',
        '用途の内訳: プロトタイピング（38%）、ツーリング（27%）、最終製品（35%）':
            'Application breakdown: Prototyping (38%), Tooling (27%), End-use parts (35%)',
        '主要産業: 航空宇宙（26%）、医療（21%）、自動車（18%）、消費財（15%）':
            'Key industries: Aerospace (26%), Medical (21%), Automotive (18%), Consumer goods (15%)',
        '材料別シェア: ポリマー（55%）、金属（35%）、セラミックス（7%）、その他（3%）':
            'Material share: Polymers (55%), Metals (35%), Ceramics (7%), Other (3%)',

        '⚠️ AMの制約と課題': '⚠️ AM Constraints and Challenges',
        'AMは万能ではなく、以下の制約があります：':
            'AM is not a panacea and has the following constraints:',
        '造形速度': 'Build speed',
        '大量生産には不向き（射出成形1個/数秒 vs AM数時間）。経済的ブレークイーブンは通常1,000個以下':
            'Unsuitable for mass production (injection molding: 1 part/few seconds vs AM: hours). Economic break-even typically below 1,000 units',
        '造形サイズ制限': 'Build size limitations',
        'ビルドボリューム（多くの装置で200×200×200mm程度）を超える大型部品は分割製造が必要':
            'Large parts exceeding build volume (typically ~200×200×200mm for many systems) require segmented fabrication',
        '表面品質': 'Surface quality',
        '積層痕（layer lines）が残るため、高精度表面が必要な場合は後加工必須（研磨、機械加工）':
            'Layer lines remain, requiring post-processing (polishing, machining) for high-precision surfaces',
        '材料特性の異方性': 'Material property anisotropy',
        '積層方向（Z軸）と面内方向（XY平面）で機械的性質が異なる場合がある（特にFDM）':
            'Mechanical properties may differ between build direction (Z-axis) and in-plane direction (XY-plane), especially in FDM',
        '材料コスト': 'Material cost',
        'AMグレード材料は汎用材料の2-10倍高価（ただし材料効率と設計最適化で相殺可能）':
            'AM-grade materials are 2-10x more expensive than commodity materials (though offset by material efficiency and design optimization)',

        # ============================================
        # PROCESS DESCRIPTIONS
        # ============================================
        '原理': 'Principle',
        'プロセス:': 'Process:',
        '特徴：': 'Features:',
        '応用例：': 'Applications:',
        '代表技術：': 'Representative Technologies:',

        # MEX/FDM
        '熱可塑性樹脂フィラメントを加熱・溶融し、ノズルから押し出して積層。最も普及している技術（FDM/FFFとも呼ばれる）。':
            'Thermoplastic filament is heated, melted, and extruded through a nozzle for layer-by-layer deposition. The most widespread technology (also called FDM/FFF).',
        'フィラメント → 加熱ノズル（190-260°C）→ 溶融押出 → 冷却固化 → 次層積層':
            'Filament → Heated nozzle (190-260°C) → Melt extrusion → Cooling solidification → Next layer deposition',
        '低コスト': 'Low cost',
        '装置価格$200-$5,000（デスクトップ）、$10,000-$100,000（産業用）':
            'Equipment price $200-$5,000 (desktop), $10,000-$100,000 (industrial)',
        '材料多様性': 'Material diversity',
        'PLA、ABS、PETG、ナイロン、PC、カーボン繊維複合材、PEEK（高性能）':
            'PLA, ABS, PETG, Nylon, PC, Carbon fiber composites, PEEK (high-performance)',
        '造形速度': 'Build speed',
        '20-150 mm³/s（中程度）、レイヤー高さ0.1-0.4mm':
            '20-150 mm³/s (moderate), layer height 0.1-0.4mm',
        '精度': 'Accuracy',
        '±0.2-0.5 mm（デスクトップ）、±0.1 mm（産業用）':
            '±0.2-0.5 mm (desktop), ±0.1 mm (industrial)',
        '表面品質': 'Surface quality',
        '積層痕が明瞭（後加工で改善可能）':
            'Visible layer lines (improvable with post-processing)',
        '材料異方性': 'Material anisotropy',
        'Z軸方向（積層方向）の強度が20-80%低い（層間接着が弱点）':
            'Strength in Z-direction (build direction) 20-80% lower (interlayer bonding is weak point)',

        # ============================================
        # APPLICATIONS
        # ============================================
        '応用1: プロトタイピング（Rapid Prototyping）':
            'Application 1: Prototyping (Rapid Prototyping)',
        'AMの最初の主要用途で、設計検証・機能試験・市場評価用のプロトタイプを迅速に製造します：':
            'The first major application of AM, rapidly producing prototypes for design verification, functional testing, and market evaluation:',
        'リードタイム短縮': 'Lead time reduction',
        '従来の試作（数週間〜数ヶ月）→ AMでは数時間〜数日':
            'Traditional prototyping (weeks to months) → AM: hours to days',
        '設計反復の加速': 'Accelerated design iteration',
        '低コストで複数バージョンを試作し、設計を最適化':
            'Optimize design through low-cost prototyping of multiple versions',
        'コミュニケーション改善': 'Improved communication',
        '視覚的・触覚的な物理モデルで関係者間の認識を統一':
            'Align stakeholder understanding through visual and tactile physical models',
        '典型例': 'Typical examples',
        '自動車の意匠モデル、家電製品の筐体試作、医療機器の術前シミュレーションモデル':
            'Automotive design models, consumer electronics housing prototypes, medical device pre-operative simulation models',

        '応用2: ツーリング（Tooling & Fixtures）':
            'Application 2: Tooling & Fixtures',
        '製造現場で使用する治具・工具・金型をAMで製造する応用です：':
            'Application of AM to produce jigs, tools, and molds used in manufacturing:',
        'カスタム治具': 'Custom fixtures',
        '生産ラインに特化した組立治具・検査治具を迅速に製作':
            'Rapid production of assembly and inspection fixtures tailored to production lines',
        'コンフォーマル冷却金型': 'Conformal cooling molds',
        '従来の直線的冷却路ではなく、製品形状に沿った3次元冷却路を内蔵した射出成形金型（冷却時間30-70%短縮）':
            'Injection molds with 3D cooling channels conforming to product shape rather than straight channels (30-70% cooling time reduction)',
        '軽量化ツール': 'Lightweighted tools',
        'ラティス構造を使った軽量エンドエフェクタで作業者の負担を軽減':
            'Reduce operator burden with lightweight end effectors using lattice structures',

        '応用3: 最終製品（End-Use Parts）':
            'Application 3: End-Use Parts',
        'AMで直接、最終製品を製造する応用が近年急増しています：':
            'Direct production of end-use parts with AM has surged in recent years:',
        '航空宇宙部品': 'Aerospace components',
        'GE Aviation LEAP燃料噴射ノズル（従来20部品→AM一体化、重量25%軽減、年間100,000個以上生産）':
            'GE Aviation LEAP fuel injection nozzle (previously 20 parts → AM consolidation, 25% weight reduction, 100,000+ units/year production)',
        '医療インプラント': 'Medical implants',
        'チタン製人工股関節・歯科インプラント（患者固有の解剖学的形状に最適化、骨結合を促進する多孔質構造）':
            'Titanium hip implants and dental implants (optimized to patient-specific anatomy, porous structure promoting bone integration)',
        'カスタム製品': 'Custom products',
        '補聴器（年間1,000万個以上がAMで製造）、スポーツシューズのミッドソール（Adidas 4D、Carbon社DLS技術）':
            'Hearing aids (10 million+ units/year produced by AM), sports shoe midsoles (Adidas 4D, Carbon DLS technology)',
        'スペア部品': 'Spare parts',
        '絶版部品・希少部品のオンデマンド製造（自動車、航空機、産業機械）':
            'On-demand production of discontinued and rare parts (automotive, aircraft, industrial machinery)',

        # ============================================
        # STL FILE SECTION
        # ============================================
        'STL（STereoLithography）は、':
            'STL (STereoLithography) is ',
        'AMで最も広く使用される3Dモデルファイル形式':
            'the most widely used 3D model file format in AM',
        '1987年に3D Systems社が開発しました。':
            ', developed by 3D Systems in 1987.',
        'STLファイルは物体表面を':
            'STL files represent object surfaces as ',
        '三角形メッシュ（Triangle Mesh）の集合':
            'a collection of triangle meshes',
        'として表現します。': '.',

        # STL Structure
        'STLファイルの基本構造': 'Basic STL File Structure',
        'STLファイル = 法線ベクトル（n） + 3つの頂点座標（v1, v2, v3）× 三角形数':
            'STL File = Normal vector (n) + 3 vertex coordinates (v1, v2, v3) × Number of triangles',
        'ASCII STL形式の例：': 'ASCII STL format example:',

        'STLフォーマットの2つの種類：': 'Two types of STL format:',
        'ASCII STL': 'ASCII STL',
        '人間が読めるテキスト形式。ファイルサイズ大（同じモデルでBinaryの10-20倍）。デバッグ・検証に有用。':
            'Human-readable text format. Large file size (10-20x Binary for same model). Useful for debugging and verification.',
        'Binary STL': 'Binary STL',
        'バイナリ形式、ファイルサイズ小、処理高速。産業用途で標準。構造：80バイトヘッダー + 4バイト（三角形数） + 各三角形50バイト（法線12B + 頂点36B + 属性2B）。':
            'Binary format, small file size, fast processing. Standard in industrial applications. Structure: 80-byte header + 4-byte (triangle count) + 50 bytes per triangle (12B normal + 36B vertices + 2B attribute).',

        # Normal vectors
        '1. 法線ベクトル（Normal Vector）': '1. Normal Vector',
        '各三角形面には': 'Each triangle face has a ',
        '法線ベクトル（外向き方向）': 'normal vector (outward direction)',
        'が定義され、物体の「内側」と「外側」を区別します。':
            ' defined to distinguish the "inside" and "outside" of the object.',
        '法線方向は': 'The normal direction is determined by the ',
        '右手の法則': 'right-hand rule',
        'で決定されます：': ':',

        '頂点順序ルール：': 'Vertex order rule: ',
        '頂点v1, v2, v3は反時計回り（CCW: Counter-ClockWise）に配置され、外から見て反時計回りの順序で法線が外向きになります。':
            'Vertices v1, v2, v3 are arranged counter-clockwise (CCW), and the normal points outward in the counter-clockwise order when viewed from outside.',

        # Manifold
        '2. 多様体（Manifold）条件': '2. Manifold Conditions',
        'STLメッシュが3Dプリント可能であるためには、': 'For an STL mesh to be 3D printable, it must be ',
        '多様体（Manifold）': 'manifold',
        'でなければなりません：': ':',

        'エッジ共有': 'Edge sharing',
        'すべてのエッジ（辺）は正確に2つの三角形に共有される':
            'All edges must be shared by exactly two triangles',
        '頂点共有': 'Vertex sharing',
        'すべての頂点は連続した三角形扇（fan）に属する':
            'All vertices must belong to a continuous triangle fan',
        '閉じた表面': 'Closed surface',
        '穴や開口部がなく、完全に閉じた表面を形成':
            'Forms a completely closed surface without holes or openings',
        '自己交差なし': 'No self-intersection',
        '三角形が互いに交差・貫通していない':
            'Triangles do not intersect or penetrate each other',

        '⚠️ 非多様体メッシュの問題': '⚠️ Non-Manifold Mesh Problems',
        '非多様体メッシュ（Non-Manifold Mesh）は3Dプリント不可能です。典型的な問題：':
            'Non-manifold meshes are not 3D printable. Typical problems:',
        '穴（Holes）': 'Holes',
        '閉じていない表面、エッジが1つの三角形にのみ属する':
            'Unclosed surface, edges belonging to only one triangle',
        'T字接合（T-junction）': 'T-junction',
        'エッジが3つ以上の三角形に共有される':
            'Edges shared by three or more triangles',
        '法線反転（Inverted Normals）': 'Inverted Normals',
        '法線が内側を向いている三角形が混在':
            'Mixture of triangles with normals pointing inward',
        '重複頂点（Duplicate Vertices）': 'Duplicate Vertices',
        '同じ位置に複数の頂点が存在':
            'Multiple vertices exist at the same location',
        '微小三角形（Degenerate Triangles）': 'Degenerate Triangles',
        '面積がゼロまたはほぼゼロの三角形':
            'Triangles with zero or near-zero area',
        'これらの問題はスライサーソフトウェアでエラーを引き起こし、造形失敗の原因となります。':
            'These problems cause errors in slicer software and lead to build failures.',

        # ============================================
        # SLICING SECTION
        # ============================================
        'スライシングは、3Dモデルを一定の高さ（レイヤー高さ）で水平に切断し、各層の輪郭を抽出するプロセスです：':
            'Slicing is the process of horizontally cutting a 3D model at constant heights (layer heights) and extracting contours for each layer:',

        'レイヤー高さ（Layer Height）の選択': 'Layer Height Selection',
        'レイヤー高さは造形品質と造形時間のトレードオフを決定する最重要パラメータです：':
            'Layer height is the most important parameter determining the trade-off between build quality and build time:',

        # Layer height table
        'レイヤー高さ': 'Layer Height',
        '造形品質': 'Build Quality',
        '造形時間': 'Build Time',
        '典型的な用途': 'Typical Applications',
        '0.1 mm（極細）': '0.1 mm (ultra-fine)',
        '非常に高い（積層痕ほぼ不可視）': 'Very high (layer lines nearly invisible)',
        '非常に長い（×2-3倍）': 'Very long (×2-3x)',
        'フィギュア、医療モデル、最終製品': 'Figurines, medical models, end-use parts',
        '0.2 mm（標準）': '0.2 mm (standard)',
        '良好（積層痕は見えるが許容）': 'Good (layer lines visible but acceptable)',
        '標準': 'Standard',
        '一般的なプロトタイプ、機能部品': 'General prototypes, functional parts',
        '0.3 mm（粗）': '0.3 mm (coarse)',
        '低い（積層痕明瞭）': 'Low (layer lines prominent)',
        '短い（×0.5倍）': 'Short (×0.5x)',
        '初期プロトタイプ、内部構造部品': 'Initial prototypes, internal structural parts',

        '⚠️ レイヤー高さの制約': '⚠️ Layer Height Constraints',
        'レイヤー高さはノズル径の': 'Layer height should be ',
        '25-80%': '25-80%',
        'に設定する必要があります。': ' of nozzle diameter.',
        '例えば0.4mmノズルの場合、レイヤー高さは0.1-0.32mmが推奨範囲です。':
            'For example, for a 0.4mm nozzle, the recommended layer height range is 0.1-0.32mm.',
        'これを超えると、樹脂の押出量が不足したり、ノズルが前の層を引きずる問題が発生します。':
            'Exceeding this range can cause insufficient extrusion or the nozzle dragging on the previous layer.',

        # Shell and infill
        'シェル（外殻）の生成': 'Shell Generation',
        'シェル（Shell/Perimeter）': 'Shell (Shell/Perimeter)',
        'は、各層の外周部を形成する経路です：': ' is the path forming the perimeter of each layer:',
        'シェル数（Perimeter Count）': 'Shell count (Perimeter Count)',
        '通常2-4本。外部品質と強度に影響。': 'Typically 2-4. Affects external quality and strength.',
        '1本: 非常に弱い、透明性高い、装飾用のみ':
            '1: Very weak, high transparency, decorative only',
        '2本: 標準（バランス良好）': '2: Standard (good balance)',
        '3-4本: 高強度、表面品質向上、気密性向上':
            '3-4: High strength, improved surface quality, improved airtightness',
        'シェル順序': 'Shell order',
        '内側→外側（Inside-Out）が一般的。外側→内側は表面品質重視時に使用。':
            'Inside-Out is common. Outside-In is used when surface quality is prioritized.',

        'インフィル（内部充填）パターン': 'Infill Pattern',
        'インフィル（Infill）': 'Infill',
        'は内部構造を形成し、強度と材料使用量を制御します：':
            ' forms internal structure and controls strength and material usage:',

        # Infill patterns table
        'パターン': 'Pattern',
        '強度': 'Strength',
        '印刷速度': 'Print Speed',
        '材料使用量': 'Material Usage',
        '特徴': 'Characteristics',
        'Grid（格子）': 'Grid',
        '中': 'Medium',
        '速い': 'Fast',
        'シンプル、等方性、標準的な選択': 'Simple, isotropic, standard choice',
        'Honeycomb（ハニカム）': 'Honeycomb',
        '高': 'High',
        '遅い': 'Slow',
        '高強度、重量比優秀、航空宇宙用途': 'High strength, excellent strength-to-weight ratio, aerospace applications',
        'Gyroid': 'Gyroid',
        '非常に高': 'Very high',
        '3次元等方性、曲面的、最新の推奨': '3D isotropic, curved, latest recommendation',
        'Concentric（同心円）': 'Concentric',
        '低': 'Low',
        '柔軟性重視、シェル追従': 'Flexibility-focused, shell-conforming',
        'Lines（直線）': 'Lines',
        '低（異方性）': 'Low (anisotropic)',
        '非常に速い': 'Very fast',
        '少': 'Low',
        '高速印刷、方向性強度': 'Fast printing, directional strength',

        '💡 インフィル密度の目安': '💡 Infill Density Guidelines',
        '0-10%': '0-10%',
        '装飾品、非荷重部品（材料節約優先）': 'Decorative items, non-load-bearing parts (material saving priority)',
        '20%': '20%',
        '標準的なプロトタイプ（バランス良好）': 'Standard prototypes (good balance)',
        '40-60%': '40-60%',
        '機能部品、高強度要求': 'Functional parts, high strength requirements',
        '100%': '100%',
        '最終製品、水密性要求、最高強度（造形時間×3-5倍）':
            'End-use parts, watertightness requirements, maximum strength (build time ×3-5x)',

        # Support structures
        'オーバーハング角度が45度を超える部分は、':
            'Parts with overhang angles exceeding 45 degrees require ',
        'サポート構造（Support Structure）': 'support structures',
        'が必要です：': ':',

        'サポートのタイプ': 'Support Types',
        'Linear Support（直線サポート）': 'Linear Support',
        '垂直な柱状サポート。シンプルで除去しやすいが、材料使用量多い。':
            'Vertical columnar support. Simple and easy to remove, but high material usage.',
        'Tree Support（ツリーサポート）': 'Tree Support',
        '樹木状に分岐するサポート。材料使用量30-50%削減、除去しやすい。CuraやPrusaSlicerで標準サポート。':
            'Tree-like branching support. 30-50% material reduction, easy removal. Standard support in Cura and PrusaSlicer.',
        'Interface Layers（接合層）': 'Interface Layers',
        'サポート上面に薄い接合層を設ける。除去しやすく、表面品質向上。通常2-4層。':
            'Thin interface layer on support top surface. Easy removal, improved surface quality. Typically 2-4 layers.',

        'サポート設定の重要パラメータ': 'Important Support Parameters',
        'Overhang Angle': 'Overhang Angle',
        '45-60°': '45-60°',
        'この角度以上でサポート生成': 'Support generated above this angle',
        'Support Density': 'Support Density',
        '10-20%': '10-20%',
        '密度が高いほど安定だが除去困難': 'Higher density is more stable but harder to remove',
        'Support Z Distance': 'Support Z Distance',
        '0.2-0.3 mm': '0.2-0.3 mm',
        'サポートと造形物の間隔（除去しやすさ）': 'Gap between support and part (ease of removal)',
        'Interface Layers': 'Interface Layers',
        '2-4層': '2-4 layers',
        '接合層数（表面品質と除去性のバランス）': 'Number of interface layers (balance of surface quality and removability)',

        # G-code
        'G-code': 'G-code',
        'は、3DプリンタやCNCマシンを制御する標準的な数値制御言語です。各行が1つのコマンドを表します：':
            ' is the standard numerical control language for controlling 3D printers and CNC machines. Each line represents one command:',

        '主要なG-codeコマンド': 'Major G-code Commands',
        'コマンド': 'Command',
        '分類': 'Category',
        '機能': 'Function',
        '例': 'Example',
        'G0': 'G0',
        '移動': 'Movement',
        '高速移動（非押出）': 'Rapid move (non-extrusion)',
        'G1': 'G1',
        '直線移動（押出あり）': 'Linear move (with extrusion)',
        'G28': 'G28',
        '初期化': 'Initialization',
        'ホームポジション復帰': 'Return to home position',
        '（全軸）, G28 Z （Z軸のみ）': ' (all axes), G28 Z (Z-axis only)',
        'M104': 'M104',
        '温度': 'Temperature',
        'ノズル温度設定（非待機）': 'Set nozzle temperature (non-blocking)',
        'M104 S200': 'M104 S200',
        'M109': 'M109',
        'ノズル温度設定（待機）': 'Set nozzle temperature (blocking)',
        'M109 S210': 'M109 S210',
        'M140': 'M140',
        'ベッド温度設定（非待機）': 'Set bed temperature (non-blocking)',
        'M140 S60': 'M140 S60',
        'M190': 'M190',
        'ベッド温度設定（待機）': 'Set bed temperature (blocking)',
        'M190 S60': 'M190 S60',

        'G-codeの例（造形開始部分）': 'G-code Example (Start Sequence)',
        '; === Start G-code ===': '; === Start G-code ===',
        'M140 S60       ; ベッドを60°Cに加熱開始（非待機）':
            'M140 S60       ; Start heating bed to 60°C (non-blocking)',
        'M104 S210      ; ノズルを210°Cに加熱開始（非待機）':
            'M104 S210      ; Start heating nozzle to 210°C (non-blocking)',
        'G28            ; 全軸ホーミング':
            'G28            ; Home all axes',
        'G29            ; オートレベリング（ベッドメッシュ計測）':
            'G29            ; Auto bed leveling (mesh probing)',
        'M190 S60       ; ベッド温度到達を待機':
            'M190 S60       ; Wait for bed temperature',
        'M109 S210      ; ノズル温度到達を待機':
            'M109 S210      ; Wait for nozzle temperature',
        'G92 E0         ; 押出量をゼロリセット':
            'G92 E0         ; Reset extruder position to zero',
        'G1 Z2.0 F3000  ; Z軸を2mm上昇（安全確保）':
            'G1 Z2.0 F3000  ; Raise Z-axis 2mm (safety)',
        'G1 X10 Y10 F5000  ; プライム位置へ移動':
            'G1 X10 Y10 F5000  ; Move to prime position',
        'G1 Z0.3 F3000  ; Z軸を0.3mmへ降下（初層高さ）':
            'G1 Z0.3 F3000  ; Lower Z to 0.3mm (first layer height)',
        'G1 X100 E10 F1500 ; プライムライン描画（ノズル詰まり除去）':
            'G1 X100 E10 F1500 ; Draw prime line (purge nozzle)',
        'G92 E0         ; 押出量を再度ゼロリセット':
            'G92 E0         ; Reset extruder position again',
        '; === 造形開始 ===': '; === Build Start ===',

        # ============================================
        # SLICER SOFTWARE
        # ============================================
        '主要スライシングソフトウェア': 'Major Slicing Software',
        'ソフトウェア': 'Software',
        'ライセンス': 'License',
        '特徴': 'Features',
        '推奨用途': 'Recommended Use',
        'Cura': 'Cura',
        'オープンソース': 'Open Source',
        '使いやすい、豊富なプリセット、Tree Support標準搭載':
            'User-friendly, abundant presets, Tree Support built-in',
        '初心者〜中級者、FDM汎用': 'Beginners to intermediate, general FDM',
        'PrusaSlicer': 'PrusaSlicer',
        '高度な設定、変数レイヤー高さ、カスタムサポート':
            'Advanced settings, variable layer height, custom supports',
        '中級者〜上級者、最適化重視': 'Intermediate to advanced, optimization-focused',
        'Slic3r': 'Slic3r',
        'PrusaSlicerの元祖、軽量': 'Origin of PrusaSlicer, lightweight',
        'レガシーシステム、研究用途': 'Legacy systems, research applications',
        'Simplify3D': 'Simplify3D',
        '商用（$150）': 'Commercial ($150)',
        '高速スライシング、マルチプロセス、詳細制御':
            'Fast slicing, multi-process, detailed control',
        'プロフェッショナル、産業用途': 'Professional, industrial applications',
        'IdeaMaker': 'IdeaMaker',
        '無料': 'Free',
        'Raise3D専用だが汎用性高い、直感的UI':
            'Raise3D-specific but versatile, intuitive UI',
        'Raise3Dユーザー、初心者': 'Raise3D users, beginners',

        # ============================================
        # TOOLPATH OPTIMIZATION
        # ============================================
        'ツールパス最適化戦略': 'Toolpath Optimization Strategies',
        '効率的なツールパスは、造形時間・品質・材料使用量を改善します：':
            'Efficient toolpaths improve build time, quality, and material usage:',

        'リトラクション（Retraction）': 'Retraction',
        '移動時にフィラメントを引き戻してストリング（糸引き）を防止。':
            'Retracts filament during travels to prevent stringing.',
        '距離: 1-6mm（ボーデンチューブ式は4-6mm、ダイレクト式は1-2mm）':
            'Distance: 1-6mm (Bowden: 4-6mm, Direct drive: 1-2mm)',
        '速度: 25-45 mm/s': 'Speed: 25-45 mm/s',
        '過度なリトラクションはノズル詰まりの原因':
            'Excessive retraction causes nozzle clogs',

        'Z-hop（Z軸跳躍）': 'Z-hop',
        '移動時にノズルを上昇させて造形物との衝突を回避。0.2-0.5mm上昇。造形時間微増だが表面品質向上。':
            'Raises nozzle during travels to avoid collision with part. 0.2-0.5mm lift. Slightly increases build time but improves surface quality.',

        'コーミング（Combing）': 'Combing',
        '移動経路をインフィル上に制限し、表面への移動痕を低減。外観重視時に有効。':
            'Restricts travel paths to infill areas, reducing marks on surfaces. Effective when appearance is important.',

        'シーム位置（Seam Position）': 'Seam Position',
        '各層の開始/終了点を揃える戦略。': 'Strategy for aligning layer start/end points.',
        'Random: ランダム配置（目立たない）': 'Random: Random placement (less visible)',
        'Aligned: 一直線に配置（後加工でシームを除去しやすい）':
            'Aligned: Straight line (easier to remove seam in post-processing)',
        'Sharpest Corner: 最も鋭角なコーナーに配置（目立ちにくい）':
            'Sharpest Corner: Places at sharpest corner (less noticeable)',

        # ============================================
        # CODE EXAMPLES
        # ============================================
        '# ===================================': '# ===================================',
        '# Example 1: STLファイルの読み込みと基本情報取得':
            '# Example 1: Loading STL Files and Obtaining Basic Information',
        '# STLファイルを読み込む': '# Load STL file',
        '# 基本的な幾何情報を取得': '# Get basic geometric information',
        'Volume:': 'Volume:',
        'Surface Area:': 'Surface Area:',
        'Center of Gravity:': 'Center of Gravity:',
        'Number of Triangles:': 'Number of Triangles:',
        '# バウンディングボックス（最小包含直方体）を計算':
            '# Calculate bounding box (minimum enclosing cuboid)',
        '# バウンディングボックス': '# Bounding box',
        '幅:': 'Width:',
        '奥行:': 'Depth:',
        '高さ:': 'Height:',
        '# 造形時間の簡易推定（レイヤー高さ0.2mm、速度50mm/sと仮定）':
            '# Simple build time estimation (assuming 0.2mm layer height, 50mm/s speed)',
        '# 造形推定': '# Build estimation',
        'レイヤー数（0.2mm/層）:': 'Number of layers (0.2mm/layer):',
        '推定造形時間:': 'Estimated build time:',
        '時間': 'hours',
        '分': 'minutes',
        '層': 'layers',
        '# 出力例:': '# Output example:',

        '# Example 2: メッシュの法線ベクトル検証':
            '# Example 2: Normal Vector Validation',
        'STLメッシュの法線ベクトルの整合性をチェック':
            'Check consistency of normal vectors in STL mesh',
        '右手系ルールで法線方向を確認': 'Verify normal direction using right-hand rule',
        'エッジベクトルを計算': 'Calculate edge vectors',
        '外積で法線を計算（右手系）': 'Calculate normal using cross product (right-hand rule)',
        '正規化': 'Normalize',
        'ゼロベクトルでないことを確認': 'Verify not zero vector',
        '縮退三角形をスキップ': 'Skip degenerate triangles',
        'ファイルに保存されている法線と比較': 'Compare with normals stored in file',
        '内積で方向の一致をチェック': 'Check direction agreement using dot product',
        '内積が負なら逆向き': 'Negative dot product means opposite direction',
        '法線チェックを実行': 'Execute normal check',
        '総三角形数:': 'Total triangles:',
        '反転法線数:': 'Inverted normals:',
        '反転率:': 'Inversion rate:',
        'すべての法線が正しい方向を向いています':
            'All normals are correctly oriented',
        'このメッシュは3Dプリント可能です': 'This mesh is printable',
        '一部の法線が反転しています（軽微）':
            'Some normals are inverted (minor)',
        'スライサーが自動修正する可能性が高い':
            'Slicer is likely to auto-correct',
        '多数の法線が反転しています（重大）':
            'Many normals are inverted (critical)',
        'メッシュ修復ツール（Meshmixer, netfabb）での修正を推奨':
            'Recommend repair using mesh repair tools (Meshmixer, Netfabb)',

        '# Example 3: マニフォールド性（Watertight）のチェック':
            '# Example 3: Manifold (Watertight) Check',
        '# STLファイルを読み込み（trimeshは自動で修復を試みる）':
            '# Load STL file (trimesh attempts automatic repair)',
        '# 基本情報': '# Basic information',
        'Vertex count:': 'Vertex count:',
        'Face count:': 'Face count:',
        '# マニフォールド性をチェック': '# Check manifold properties',
        'Is watertight (密閉性):': 'Is watertight:',
        'Is winding consistent (法線一致性):': 'Is winding consistent:',
        'Is valid (幾何的妥当性):': 'Is valid:',
        '# 問題の詳細を診断': '# Diagnose problems in detail',
        '穴（hole）の数を検出': 'Detect number of holes',
        '重複エッジ数:': 'Duplicate edges:',
        'メッシュに穴があります': 'Mesh has holes',
        'メッシュ構造に問題があります': 'Mesh structure has problems',
        '# 修復を試みる': '# Attempt repair',
        '自動修復を実行中...': 'Performing automatic repair...',
        '法線ベクトルを修正': 'Fix normal vectors',
        '穴を充填': 'Fill holes',
        '縮退面を削除': 'Remove degenerate faces',
        '重複頂点を結合': 'Merge duplicate vertices',
        '修復後の状態': 'Post-repair status',
        '修復完了！': 'Repair complete!',
        'として保存しました': 'saved as',
        '自動修復失敗。Meshmixer等の専用ツールを推奨':
            'Automatic repair failed. Recommend specialized tools like Meshmixer',

        # ============================================
        # EXERCISES
        # ============================================
        '演習問題': 'Exercises',
        'Easy（基礎確認）': 'Easy (Fundamentals)',
        'Medium（応用）': 'Medium (Application)',
        'Hard（発展）': 'Hard (Advanced)',
        'Q1: STLファイル形式の理解': 'Q1: Understanding STL File Format',
        'STLファイルのASCII形式とBinary形式について、正しい説明はどれですか？':
            'Which statement correctly describes ASCII and Binary STL formats?',
        'a) ASCII形式の方がファイルサイズが小さい':
            'a) ASCII format has smaller file size',
        'b) Binary形式は人間が直接読めるテキスト形式':
            'b) Binary format is human-readable text format',
        'c) Binary形式は通常ASCII形式の5-10倍小さいファイルサイズ':
            'c) Binary format typically has 5-10x smaller file size than ASCII',
        'd) Binary形式はASCII形式より精度が低い':
            'd) Binary format has lower precision than ASCII',
        '解答を表示': 'Show Answer',
        '解答を見る': 'View Answer',
        '正解:': 'Correct Answer:',
        '解説:': 'Explanation:',
        '精度は両形式とも同じ（32-bit浮動小数点数）':
            'Precision is the same for both formats (32-bit floating point)',
        '現代の3Dプリンタソフトは両形式をサポート、Binary推奨':
            'Modern 3D printing software supports both formats, Binary recommended',
        '実例:': 'Example:',
        '10,000三角形のモデル → ASCII: 約7MB、Binary: 約0.5MB':
            'Model with 10,000 triangles → ASCII: ~7MB, Binary: ~0.5MB',

        'Q2: 造形時間の簡易計算': 'Q2: Simple Build Time Calculation',
        '体積12,000 mm³、高さ30 mmの造形物を、レイヤー高さ0.2 mm、印刷速度50 mm/sで造形します。おおよその造形時間はどれですか？（インフィル20%、壁2層と仮定）':
            'Build an object with volume 12,000 mm³ and height 30 mm, using layer height 0.2 mm and print speed 50 mm/s. What is the approximate build time? (Assume 20% infill, 2 walls)',
        'a) 30分': 'a) 30 minutes',
        'b) 60分': 'b) 60 minutes',
        'c) 90分': 'c) 90 minutes',
        'd) 120分': 'd) 120 minutes',
        '計算手順:': 'Calculation steps:',
        'レイヤー数': 'Number of layers',
        '1層あたりの経路長さの推定':
            'Estimated path length per layer',
        '壁（シェル）:': 'Walls (shells):',
        'インフィル20%:': '20% infill:',
        '合計:': 'Total:',
        '総経路長': 'Total path length',
        '印刷時間': 'Print time',
        '実際の時間': 'Actual time',
        '移動時間・リトラクション・加減速を考慮すると約5-6倍 → 75-90分':
            'Considering travel, retraction, acceleration/deceleration: ~5-6x → 75-90 minutes',
        'ポイント:': 'Key point:',
        'スライサーソフトが提供する推定時間は、加減速・移動・温度安定化を含むため、単純計算の4-6倍程度になります。':
            'Slicer-estimated times include acceleration/deceleration, travel, and temperature stabilization, resulting in 4-6x simple calculations.',

        'Q3: AMプロセスの選択': 'Q3: AM Process Selection',
        '次の用途に最適なAMプロセスを選んでください：「航空機エンジン部品のチタン合金製燃料噴射ノズル、複雑な内部流路、高強度・高耐熱性要求」':
            'Select the optimal AM process for: "Titanium alloy fuel injection nozzle for aircraft engine, complex internal channels, high strength and heat resistance requirements"',
        'a) FDM (Fused Deposition Modeling)': 'a) FDM (Fused Deposition Modeling)',
        'b) SLA (Stereolithography)': 'b) SLA (Stereolithography)',
        'c) SLM (Selective Laser Melting)': 'c) SLM (Selective Laser Melting)',
        'd) Binder Jetting': 'd) Binder Jetting',
        '理由:': 'Reason:',
        'SLMの特徴': 'SLM Features',
        '金属粉末（チタン、インコネル、ステンレス）をレーザーで完全溶融。高密度（99.9%）、高強度、高耐熱性。':
            'Complete laser melting of metal powders (titanium, Inconel, stainless steel). High density (99.9%), high strength, high heat resistance.',
        '用途適合性': 'Application suitability',
        'チタン合金（Ti-6Al-4V）対応': 'Compatible with titanium alloy (Ti-6Al-4V)',
        '複雑内部流路製造可能（サポート除去後）':
            'Can fabricate complex internal channels (after support removal)',
        '航空宇宙グレードの機械的特性': 'Aerospace-grade mechanical properties',
        'GE Aviationが実際にFUEL噴射ノズルをSLMで量産':
            'GE Aviation actually mass-produces FUEL injection nozzles with SLM',
        '他の選択肢が不適な理由': 'Why other options are unsuitable',
        'FDM: プラスチックのみ、強度・耐熱性不足':
            'FDM: Plastics only, insufficient strength and heat resistance',
        'SLA: 樹脂のみ、機能部品には不適':
            'SLA: Resins only, unsuitable for functional parts',
        'Binder Jetting: 金属可能だが、焼結後密度90-95%で航空宇宙基準に届かない':
            'Binder Jetting: Metal possible, but post-sintering density 90-95% does not meet aerospace standards',
        'GE AviationのLEAP燃料ノズル（SLM製）は、従来20部品を溶接していたものを1部品に統合、重量25%削減、耐久性5倍向上を達成。':
            'GE Aviation\'s LEAP fuel nozzle (SLM-produced) consolidates 20 welded parts into 1, achieving 25% weight reduction and 5x durability improvement.',

        # ============================================
        # NAVIGATION & FOOTER
        # ============================================
        '次のステップ': 'Next Steps',
        '第2章では積層造形（AM）の基礎として、ISO/ASTM 52900による7つのプロセス分類、STLファイル形式の構造、スライシングとG-codeの基本を学びました。次の第2章では、材料押出（FDM/FFF）の詳細な造形プロセス、材料特性、プロセスパラメータ最適化について学びます。':
            'In Chapter 2, we learned the fundamentals of Additive Manufacturing (AM), including the seven process categories by ISO/ASTM 52900, STL file format structure, and slicing and G-code basics. In the next chapter, we will study detailed fabrication processes for Material Extrusion (FDM/FFF), material properties, and process parameter optimization.',
        'シリーズ目次': 'Series Index',
        '第2章へ進む →': 'Proceed to Chapter 2 →',
        '参考文献': 'References',
        '使用ツールとライブラリ': 'Tools and Libraries Used',
        '東北大学 材料科学研究科': 'Tohoku University Graduate School of Materials Science',

        # ============================================
        # MISC TECHNICAL TERMS
        # ============================================
        '詳細': 'Details',
        '概要': 'Overview',
        '注意': 'Note',
        '重要': 'Important',
        '実践例': 'Practical Example',
        'による': 'by',
        'を使用': 'using',
        'の場合': 'in case of',
        'について': 'about',
        'として': 'as',
        'から': 'from',
        'まで': 'to',
        'より': 'than',
        'など': 'etc.',
        'や': 'and',
        'または': 'or',
        'および': 'and',
        'かつ': 'and',
        'ただし': 'however',
        'なお': 'note that',
        'すなわち': 'namely',
        'つまり': 'in other words',
        '例えば': 'for example',
        '特に': 'especially',
        '主に': 'mainly',
        '一般に': 'generally',
        '通常': 'typically',
        '約': 'approximately',
        '以上': 'or more',
        '以下': 'or less',
        '未満': 'less than',
        '超': 'exceeding',
        '程度': 'approximately',
        '等': 'etc.',
        '可能': 'possible',
        '必要': 'necessary',
        '推奨': 'recommended',
        '標準': 'standard',
        '代表的な': 'representative',
        '主要な': 'major',
        '基本的な': 'basic',
        '簡易': 'simple',
        '高度な': 'advanced',
        '複雑な': 'complex',
    }

def main():
    # Read Japanese source
    jp_file = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/3d-printing-introduction/chapter-2.html"
    en_file = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-2.html"

    with open(jp_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Get translation dict
    translations = create_comprehensive_translation_dict()

    # Apply translations
    print("Applying translations...")
    translation_count = 0
    for jp, en in translations.items():
        if jp in content:
            content = content.replace(jp, en)
            translation_count += 1

    print(f"Applied {translation_count} translations")

    # Write output
    with open(en_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Translation completed: {en_file}")

    # Count remaining Japanese characters
    import subprocess
    result = subprocess.run(
        ['grep', '-o', '[あ-ん]\\|[ア-ン]\\|[一-龯]', en_file],
        capture_output=True,
        text=True
    )

    if result.stdout.strip():
        jp_count = len(result.stdout.strip().split('\n'))
    else:
        jp_count = 0

    print(f"\nRemaining Japanese characters: {jp_count}")

    if jp_count == 0:
        print("✅ Translation COMPLETE - No Japanese characters remaining!")
    else:
        print(f"⚠️  {jp_count} Japanese characters still need translation")

    return jp_count

if __name__ == "__main__":
    sys.exit(main())
