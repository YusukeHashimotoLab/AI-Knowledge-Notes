#!/usr/bin/env python3
"""
COMPLETE Translation Script for 3D Printing Chapter 2
Translates ALL Japanese text to English - ZERO Japanese characters remaining
Target: 7,298 Japanese characters → 0
"""

import re
from pathlib import Path

def main():
    """Execute complete translation"""

    jp_file = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/3d-printing-introduction/chapter-2.html"
    en_file = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-2.html"

    print("=" * 80)
    print("COMPREHENSIVE TRANSLATION: 3D Printing Chapter 2")
    print("=" * 80)

    # Read Japanese source
    with open(jp_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply ALL translations
    content = translate_metadata(content)
    content = translate_header(content)
    content = translate_learning_objectives(content)
    content = translate_section_1_1(content)
    content = translate_section_1_2(content)
    content = translate_section_1_3(content)
    content = translate_section_1_4(content)
    content = translate_exercises(content)
    content = translate_tables(content)
    content = translate_mermaid_diagrams(content)
    content = translate_code_comments(content)
    content = translate_info_boxes(content)
    content = translate_references(content)
    content = translate_navigation(content)
    content = final_cleanup(content)

    # Write translated content
    with open(en_file, 'w', encoding='utf-8') as f:
        f.write(content)

    # Verify completion
    import subprocess
    result = subprocess.run(
        ['grep', '-o', '[あ-ん]\\|[ア-ン]\\|[一-龯]', en_file],
        capture_output=True,
        text=True
    )
    remaining = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

    print(f"\n✅ Translation written to: {en_file}")
    print(f"📊 Remaining Japanese characters: {remaining}")

    if remaining == 0:
        print("\n🎉 SUCCESS: Translation complete - ZERO Japanese characters!")
    else:
        print(f"\n⚠️  WARNING: {remaining} Japanese characters still remain")
        print("Additional translation pass required...")

    return remaining

def translate_metadata(content):
    """Translate HTML metadata"""
    replacements = {
        '<html lang="ja">': '<html lang="en">',
        '第2章：材料押出法（FDM/FFF）- 熱可塑性プラスチックの積層造形 - MS Terakoya':
            'Chapter 2: Fundamentals of Additive Manufacturing - MS Terakoya',
    }
    return apply_replacements(content, replacements)

def translate_header(content):
    """Translate header section"""
    replacements = {
        'AI寺子屋トップ': 'AI Terakoya Top',
        '材料科学': 'Materials Science',
        '第2章：積層造形の基礎': 'Chapter 2: Fundamentals of Additive Manufacturing',
        'AM技術の原理と分類 - 3Dプリンティングの技術体系':
            'AM Technology Principles and Classification - 3D Printing Technology Framework',
        '3Dプリンティング入門シリーズ': '3D Printing Introduction Series',
        '読了時間: 35-40分': 'Reading time: 35-40 minutes',
        '難易度: 初級〜中級': 'Level: Beginner to Intermediate',
    }
    return apply_replacements(content, replacements)

def translate_learning_objectives(content):
    """Translate learning objectives section"""
    replacements = {
        '学習目標': 'Learning Objectives',
        'この章を完了すると、以下を説明できるようになります：':
            'Upon completing this chapter, you will be able to explain:',
        '基本理解（Level 1）': 'Basic Understanding (Level 1)',
        '実践スキル（Level 2）': 'Practical Skills (Level 2)',
        '応用力（Level 3）': 'Applied Competence (Level 3)',

        # Level 1 objectives
        '積層造形（AM）の定義とISO/ASTM 52900規格の基本概念':
            'Definition of Additive Manufacturing (AM) and basic concepts of ISO/ASTM 52900 standard',
        '7つのAMプロセスカテゴリ（MEX, VPP, PBF, MJ, BJ, SL, DED）の特徴':
            'Characteristics of seven AM process categories (MEX, VPP, PBF, MJ, BJ, SL, DED)',
        'STLファイル形式の構造（三角形メッシュ、法線ベクトル、頂点順序）':
            'Structure of STL file format (triangle mesh, normal vectors, vertex order)',
        'AMの歴史（1986年ステレオリソグラフィから現代システムまで）':
            'History of AM (from 1986 stereolithography to modern systems)',

        # Level 2 objectives
        'PythonでSTLファイルを読み込み、体積・表面積を計算できる':
            'Ability to load STL files in Python and calculate volume and surface area',
        'numpy-stlとtrimeshを使ったメッシュ検証と修復ができる':
            'Ability to validate and repair meshes using numpy-stl and trimesh',
        'スライシングの基本原理（レイヤー高さ、シェル、インフィル）を理解':
            'Understanding basic slicing principles (layer height, shells, infill)',
        'G-codeの基本構造（G0/G1/G28/M104など）を読み解ける':
            'Ability to interpret basic G-code structure (G0/G1/G28/M104, etc.)',

        # Level 3 objectives
        '用途要求に応じて最適なAMプロセスを選択できる':
            'Ability to select optimal AM process based on application requirements',
        'メッシュの問題（非多様体、法線反転）を検出・修正できる':
            'Ability to detect and fix mesh problems (non-manifold, inverted normals)',
        '造形パラメータ（レイヤー高さ、印刷速度、温度）を最適化できる':
            'Ability to optimize build parameters (layer height, print speed, temperature)',
        'STLファイルの品質評価とプリント適性判断ができる':
            'Ability to evaluate STL file quality and assess printability',
    }
    return apply_replacements(content, replacements)

def translate_section_1_1(content):
    """Translate Section 1.1: What is AM"""
    replacements = {
        '1.1 積層造形（AM）とは': '1.1 What is Additive Manufacturing (AM)',
        '1.1.1 積層造形の定義': '1.1.1 Definition of Additive Manufacturing',
        '積層造形（Additive Manufacturing, AM）とは、':
            'Additive Manufacturing (AM) is ',
        'ISO/ASTM 52900:2021規格で定義される「3次元CADデータから材料を層ごとに積み上げて物体を製造するプロセス」です。従来の切削加工（除去加工）とは対照的に、必要な部分にのみ材料を付加するため、以下の革新的な特徴を持ちます：':
            'defined by the ISO/ASTM 52900:2021 standard as "a process of joining materials to make objects from 3D model data, usually layer upon layer". In contrast to traditional subtractive manufacturing (machining), AM adds material only where needed, providing the following innovative features:',

        # Features
        '設計自由度': 'Design freedom',
        '従来製法では不可能な複雑形状（中空構造、ラティス構造、トポロジー最適化形状）を製造可能':
            'Enables fabrication of complex geometries impossible with traditional methods (hollow structures, lattice structures, topology-optimized shapes)',
        '材料効率': 'Material efficiency',
        '必要な部分にのみ材料を使用するため、材料廃棄率が5-10%（従来加工は30-90%廃棄）':
            'Material wastage rate of 5-10% as material is used only where needed (traditional machining: 30-90% waste)',
        'オンデマンド製造': 'On-demand manufacturing',
        '金型不要でカスタマイズ製品を少量・多品種生産可能':
            'Enables low-volume, high-variety production of customized products without tooling',
        '一体化製造': 'Integrated manufacturing',
        '従来は複数部品を組立てていた構造を一体造形し、組立工程を削減':
            'Consolidates structures that previously required assembly of multiple parts, reducing assembly steps',

        # Info box
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

        # History
        '1.1.2 AMの歴史と発展': '1.1.2 History and Evolution of AM',
        '積層造形技術は約40年の歴史を持ち、以下のマイルストーンを経て現在に至ります：':
            'Additive manufacturing technology has approximately 40 years of history, reaching the present through the following milestones:',

        # Timeline events
        '1986年: ステレオリソグラフィ（SLA）発明':
            '1986: Stereolithography (SLA) Invention',
        'Chuck Hull博士（3D Systems社創業者）が光硬化樹脂を層状に硬化させる最初のAM技術を発明（US Patent 4,575,330）。「3Dプリンティング」という言葉もこの時期に誕生。':
            'Dr. Chuck Hull (founder of 3D Systems) invented the first AM technology to cure photopolymer resin layer by layer (US Patent 4,575,330). The term "3D printing" was also coined during this period.',

        '1988年: 選択的レーザー焼結（SLS）登場':
            '1988: Selective Laser Sintering (SLS) Introduced',
        'Carl Deckard博士（テキサス大学）がレーザーで粉末材料を焼結する技術を開発。金属やセラミックスへの応用可能性を開く。':
            'Dr. Carl Deckard (University of Texas) developed laser sintering technology for powder materials, opening possibilities for metal and ceramic applications.',

        '1992年: 熱溶解積層（FDM）特許':
            '1992: Fused Deposition Modeling (FDM) Patent',
        'Stratasys社がFDM技術を商用化。現在最も普及している3Dプリンティング方式の基礎を確立。':
            'Stratasys commercialized FDM technology, establishing the foundation of the currently most widespread 3D printing method.',

        '2005年: RepRapプロジェクト':
            '2005: RepRap Project',
        'Adrian Bowyer教授がオープンソース3Dプリンタ「RepRap」を発表。特許切れと相まって低価格化・民主化が進展。':
            'Professor Adrian Bowyer introduced the open-source 3D printer "RepRap". Combined with patent expiration, this led to price reduction and democratization.',

        '2012年以降: 金属AMの産業普及':
            '2012 onwards: Industrial Adoption of Metal AM',
        '電子ビーム溶解（EBM）、選択的レーザー溶融（SLM）が航空宇宙・医療分野で実用化。GE AviationがFUEL噴射ノズルを量産開始。':
            'Electron Beam Melting (EBM) and Selective Laser Melting (SLM) were implemented in aerospace and medical fields. GE Aviation began mass production of fuel injection nozzles.',

        '2023年現在: 大型化・高速化の時代':
            '2023 Present: Era of Large-scale & High-speed Systems',
        'バインダージェット、連続繊維複合材AM、マルチマテリアルAMなど新技術が産業実装段階へ。':
            'New technologies such as binder jetting, continuous fiber composite AM, and multi-material AM are entering industrial implementation stage.',

        # Applications
        '1.1.3 AMの主要応用分野': '1.1.3 Major Application Areas of AM',

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
            'Injection molds with 3D cooling channels conforming to product shape, not straight channels (30-70% cooling time reduction)',
        '軽量化ツール': 'Lightweighted tools',
        'ラティス構造を使った軽量エンドエフェクタで作業者の負担を軽減':
            'Reduce operator burden with lightweight end effectors using lattice structures',
        'BMWの組立ライン用治具（年間100,000個以上をAMで製造）、GolfのTaylorMadeドライバー金型':
            'BMW assembly line fixtures (100,000+ units manufactured with AM annually), TaylorMade golf driver molds',

        '応用3: 最終製品（End-Use Parts）':
            'Application 3: End-Use Parts',
        'AMで直接、最終製品を製造する応用が近年急増しています：':
            'Direct production of end-use parts with AM has surged in recent years:',
        '航空宇宙部品': 'Aerospace components',
        '医療インプラント': 'Medical implants',
        'カスタム製品': 'Custom products',
        'スペア部品': 'Spare parts',
        '絶版部品・希少部品のオンデマンド製造（自動車、航空機、産業機械）':
            'On-demand production of discontinued and rare parts (automotive, aircraft, industrial machinery)',

        # Constraints box
        '⚠️ AMの制約と課題': '⚠️ AM Constraints and Challenges',
        '造形速度': 'Build speed',
        '一般に従来加工より遅い（数時間〜数日）。大量生産には不向き':
            'Generally slower than traditional manufacturing (hours to days). Not suitable for mass production',
        '材料コスト': 'Material cost',
        'AMグレード材料は汎用材料の2-10倍高価（ただし材料効率と設計最適化で相殺可能）':
            'AM-grade materials cost 2-10 times more than general-purpose materials (however, this can be offset by material efficiency and design optimization)',
        '機械的性質': 'Mechanical properties',
        '層間接着や残留応力により、方向依存性（異方性）が発生':
            'Directional dependence (anisotropy) occurs due to layer bonding and residual stress',
        '造形サイズ制約': 'Build size constraints',
        '装置のビルドボリュームに制限（デスクトップ機: 200-300mm、産業機: 400-800mm）':
            'Limited by equipment build volume (desktop: 200-300mm, industrial: 400-800mm)',
        '後処理の必要性': 'Post-processing requirements',
        'サポート除去、表面仕上げ、熱処理など、追加工程が必要':
            'Additional steps required such as support removal, surface finishing, heat treatment',
    }
    return apply_replacements(content, replacements)

def translate_section_1_2(content):
    """Translate Section 1.2: Seven AM Processes"""
    replacements = {
        '1.2 ISO/ASTM 52900による7つのAMプロセス分類':
            '1.2 Seven AM Process Categories by ISO/ASTM 52900',
        '1.2.1 AMプロセス分類の全体像': '1.2.1 Overview of AM Process Classification',
        'ISO/ASTM 52900:2021規格では、すべてのAM技術を<strong>エネルギー源と材料供給方法に基づいて7つのプロセスカテゴリ</strong>に分類しています。各プロセスには固有の長所・短所があり、用途に応じて最適な技術を選択する必要があります。':
            'The ISO/ASTM 52900:2021 standard <strong>categorizes all AM technologies into seven process categories based on energy source and material supply method</strong>. Each process has unique advantages and disadvantages, and it is necessary to select the optimal technology according to the application.',

        # Process names
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

        # MEX Process
        '原理': 'Principle',
        '熱可塑性樹脂フィラメントを加熱ノズルで溶融し、層状に押出して積層。':
            'Thermoplastic filament is melted through heated nozzle and extruded layer by layer.',
        'プロセス:': 'Process:',
        '加熱ノズル（180-260°C） → 溶融押出 → 冷却固化 → 次層積層':
            'Heated nozzle (180-260°C) → melt extrusion → cooling solidification → next layer deposition',

        '主要材料': 'Main Materials',
        '対応材料': 'Compatible Materials',
        '特徴：': 'Features:',
        '低コスト': 'Low cost',
        '装置価格$200-$10,000、材料費$20-50/kg（他技術の1/5-1/10）':
            'Equipment cost $200-$10,000, material cost $20-50/kg (1/5-1/10 of other technologies)',
        '材料多様性': 'Material diversity',
        'PLA、ABS、PETG、ナイロン、ポリカーボネート、TPU（軟質）、複合材（カーボン・ガラス繊維）':
            'PLA, ABS, PETG, nylon, polycarbonate, TPU (flexible), composites (carbon/glass fiber)',
        '操作簡便性': 'Ease of operation',
        '教育現場や個人ユーザーでも扱いやすい（デスクトップ機の普及）':
            'Easy to handle in educational settings and for individual users (widespread desktop systems)',

        '短所：': 'Disadvantages:',
        '積層痕': 'Layer lines',
        'レイヤー高さ（50-300 μm）による縞模様が可視（表面粗さRa 6-15 μm）':
            'Visible striping from layer height (50-300 μm) (surface roughness Ra 6-15 μm)',
        '異方性': 'Anisotropy',
        '層間接着強度がZ方向で20-50%低下（引張強度: XY方向50 MPa、Z方向30 MPa）':
            'Interlayer bonding strength 20-50% lower in Z direction (tensile strength: XY 50 MPa, Z 30 MPa)',
        'サポート必要': 'Supports required',
        'オーバーハング角45°以上で必要、除去に手間':
            'Required for overhang angles >45°, removal is labor-intensive',

        '応用例：': 'Applications:',
        'プロトタイピング（最も一般的な用途、低コスト・高速）':
            'Prototyping (most common use, low cost & fast)',
        '治具・工具（製造現場で使用、軽量・カスタマイズ容易）':
            'Jigs & tools (used in manufacturing, lightweight & easy to customize)',
        '教育用モデル（学校・大学で広く使用、安全・低コスト）':
            'Educational models (widely used in schools and universities, safe & low cost)',
        '最終製品（カスタム補聴器、義肢装具、建築模型）':
            'End-use parts (custom hearing aids, prosthetics, architectural models)',

        '💡 FDMの代表的装置': '💡 Representative FDM Systems',
        'デュアルヘッド、ビルドボリューム330×240×300mm、$6,000':
            'Dual head, build volume 330×240×300mm, $6,000',
        'オープンソース系、高い信頼性、$1,200':
            'Open source platform, high reliability, $1,200',
        '産業用、ULTEM 9085対応、$250,000':
            'Industrial grade, ULTEM 9085 compatible, $250,000',
        '連続カーボン繊維複合材対応、$100,000':
            'Continuous carbon fiber composite compatible, $100,000',

        # VPP Process
        '液状の光硬化性樹脂（フォトポリマー）に紫外線（UV）レーザーorプロジェクターで光を照射し、選択的に硬化させて積層。':
            'UV laser or projector irradiates liquid photopolymer resin to selectively cure and build layers.',
        'UV照射 → 光重合反応 → 固化 → ビルドプラットフォーム上昇 → 次層照射':
            'UV irradiation → photopolymerization reaction → solidification → build platform rise → next layer irradiation',

        '方式の分類': 'Process Variants',
        'UV レーザー（355 nm）をガルバノミラーで走査し、点描的に硬化。高精度だが低速。':
            'UV laser (355 nm) scanned with galvanometer mirrors, curing point by point. High accuracy but slow.',
        'プロジェクターで面全体を一括露光。高速だが解像度はプロジェクター画素数に依存（Full HD: 1920×1080）。':
            'Entire layer exposed at once with projector. Fast but resolution depends on projector pixels (Full HD: 1920×1080).',
        'LCDマスクusing、DLP類似だが低コスト化（$200-$1,000のデスクトップ機多数）。':
            'Uses LCD mask, similar to DLP but lower cost (many desktop models at $200-$1,000).',

        '長所：': 'Advantages:',
        '高精度': 'High precision',
        'XY解像度25-100 μm、Z解像度10-50 μm（全AM技術中で最高レベル）':
            'XY resolution 25-100 μm, Z resolution 10-50 μm (highest level among all AM technologies)',
        '表面品質': 'Surface quality',
        '滑らかな表面（Ra < 5 μm）、積層痕がほぼ見えない':
            'Smooth surface (Ra < 5 μm), layer lines nearly invisible',
        '複雑形状対応': 'Complex geometry capability',
        '微細なディテールや中空構造も高精度に造形':
            'Fabricates fine details and hollow structures with high precision',

        '材料制約': 'Material limitations',
        '光硬化性樹脂のみ（機械的性質はFDMより劣る場合が多い）':
            'Photopolymer resins only (mechanical properties often inferior to FDM)',
        '後処理必須': 'Post-processing required',
        '洗浄（IPAetc.）→ 二次硬化（UV照射）→ サポート除去':
            'Cleaning (IPA etc.) → post-curing (UV irradiation) → support removal',
        '材料コスト高': 'High material cost',
        '樹脂価格$100-400/L（FDMフィラメントの5-10倍）':
            'Resin price $100-400/L (5-10 times FDM filament)',

        'ジュエリー鋳造用ワックスモデル（高精度・複雑形状）':
            'Wax models for jewelry casting (high accuracy & complex geometry)',
        '歯科用モデル（義歯、クラウン、ブリッジ）':
            'Dental models (dentures, crowns, bridges)',
        'フィギュア・模型（ディテール表現が必要）':
            'Figures & models (detailed representation required)',
        '医療用モデル（術前計画、解剖学習）':
            'Medical models (surgical planning, anatomical learning)',

        # PBF Process
        '粉末材料を薄く敷き詰め、レーザーor電子ビームで選択的に溶融・焼結し、冷却固化させて積層。金属・ポリマー・セラミックスに対応。':
            'Thin layer of powder material is spread, selectively melted or sintered by laser or electron beam, then cooled and solidified to build layers. Compatible with metals, polymers, and ceramics.',
        '粉末敷設 → レーザー/電子ビーム走査 → 溶融・焼結 → 固化 → 次層粉末敷設':
            'Powder spreading → laser/electron beam scanning → melting/sintering → solidification → next layer powder spreading',

        'ポリマー粉末（PA12ナイロンetc.）をレーザー焼結。サポート不要（周囲粉末が支持）。':
            'Laser sintering of polymer powder (PA12 nylon etc.). No support required (surrounding powder provides support).',
        '金属粉末（Ti-6Al-4V、AlSi10Mg、Inconel 718etc.）を完全溶融。高密度部品（相対密度>99%）製造可能。':
            'Complete melting of metal powder (Ti-6Al-4V, AlSi10Mg, Inconel 718 etc.). High-density parts (relative density >99%) can be manufactured.',
        '電子ビームで金属粉末を溶融。高温予熱（650-1000°C）により残留応力が小さく、造形速度が高速。':
            'Metal powder melted by electron beam. High-temperature preheating (650-1000°C) results in lower residual stress and faster build speed.',

        '高強度': 'High strength',
        '溶融・再凝固により鍛造材に匹敵する機械的性質（引張強度500-1200 MPa）':
            'Mechanical properties comparable to forged materials due to melting and re-solidification (tensile strength 500-1200 MPa)',
        'サポート不要（粉末が支持）でオーバーハング造形可能':
            'Overhang fabrication possible without support (powder provides support)',
        '金属・セラミックス対応': 'Metal & ceramic compatibility',
        '高融点材料（チタン、インコネル、タングステン）も造形可能':
            'Can fabricate high melting point materials (titanium, Inconel, tungsten)',

        '装置コスト超高': 'Very high equipment cost',
        'SLM/EBM装置$300,000-$1,500,000':
            'SLM/EBM equipment $300,000-$1,500,000',
        '粉末取扱い': 'Powder handling',
        '微細金属粉末は爆発性・有毒性があり、不活性ガス雰囲気が必要':
            'Fine metal powder is explosive and toxic, requires inert gas atmosphere',
        '表面粗さ': 'Surface roughness',
        '粉末粒径（15-45 μm）により、表面粗さRa 5-20 μm':
            'Surface roughness Ra 5-20 μm due to powder particle size (15-45 μm)',

        '航空宇宙部品（軽量化ブラケット、燃料ノズル）':
            'Aerospace parts (lightweighted brackets, fuel nozzles)',
        '医療インプラント（整形外科、歯科）':
            'Medical implants (orthopedic, dental)',
        '自動車エンジン部品（ターボハウジング、シリンダーヘッド）':
            'Automotive engine parts (turbo housing, cylinder heads)',
        '工業用エンドパーツ（金型、熱交換器）':
            'Industrial end-use parts (molds, heat exchangers)',
    }
    return apply_replacements(content, replacements)

def translate_section_1_3(content):
    """Translate Section 1.3: STL File Format"""
    replacements = {
        '1.3 STLファイル形式とデータ処理': '1.3 STL File Format and Data Processing',
        '1.3.1 STLファイルの構造': '1.3.1 STL File Structure',
        'STL（STereoLithography）ファイルは、3Dモデルを<strong>三角形メッシュ（Triangle Mesh）</strong>で表現する最も普及した3Dプリンティング用フォーマットです。':
            'The STL (STereoLithography) file is the most widespread 3D printing format, representing 3D models as <strong>triangle meshes</strong>.',

        '1.3.2 STLファイルの重要概念': '1.3.2 Important STL Concepts',
        '法線ベクトル（Normal Vector）': 'Normal Vector',
        '各三角形は外向き法線ベクトルを持つ': 'Each triangle has an outward-pointing normal vector',
        '頂点順序（Vertex Order）': 'Vertex Order',
        '右手系（Right-hand rule）で反時計回り': 'Counter-clockwise following right-hand rule',
        '多様体条件（Manifold Condition）': 'Manifold Condition',
        '各エッジは正確に2つの三角形で共有される': 'Each edge is shared by exactly two triangles',

        '1.3.3 STLファイルの品質指標': '1.3.3 STL Quality Metrics',
        '三角形数': 'Triangle count',
        '解像度とファイルサイズのトレードオフ': 'Trade-off between resolution and file size',
        'アスペクト比': 'Aspect ratio',
        '細長い三角形は避けるべき': 'Elongated triangles should be avoided',
        'メッシュの閉じ性': 'Mesh closure',
        '穴や隙間がないこと': 'No holes or gaps',

        '1.3.4 Pythonライブラリによる STL処理': '1.3.4 STL Processing with Python Libraries',
        '# STLファイルを読み込み': '# Load STL file',
        '# 基本的な幾何情報': '# Basic geometric information',
        '# バウンディングボックス': '# Bounding box',
        '# 造形時間の簡易推定': '# Simple build time estimation',
        '# === STLファイル基本情報 ===': '# === STL File Basic Information ===',
        '# === バウンディングボックス ===': '# === Bounding Box ===',
        '# === 造形推定 ===': '# === Build Estimation ===',

        '💡 STLファイルの解像度トレードオフ': '💡 STL Resolution Trade-offs',
        '低解像度（1,000三角形）': 'Low resolution (1,000 triangles)',
        'ファイルサイズ小、曲面がカクカク': 'Small file size, faceted curves',
        '中解像度（10,000-50,000三角形）': 'Medium resolution (10,000-50,000 triangles)',
        '実用的なバランス、多くの用途で推奨': 'Practical balance, recommended for most uses',
        '高解像度（100,000+三角形）': 'High resolution (100,000+ triangles)',
        'ファイルサイズ大、スライシング処理が重い': 'Large file size, heavy slicing processing',

        '⚠️ 非多様体メッシュの問題': '⚠️ Non-Manifold Mesh Issues',
        '共有エッジ数≠2': 'Shared edge count ≠ 2',
        'T字交差、エッジの重複、孤立頂点': 'T-junctions, duplicate edges, isolated vertices',
        'スライサーでエラー発生': 'Causes slicer errors',
        '修復ツール（Meshmixer, netfabb）で修正': 'Fix with repair tools (Meshmixer, Netfabb)',
    }
    return apply_replacements(content, replacements)

def translate_section_1_4(content):
    """Translate Section 1.4: Slicing"""
    replacements = {
        '1.4 スライシングとツールパス生成': '1.4 Slicing and Toolpath Generation',
        '1.4.1 スライシングの基本原理': '1.4.1 Basic Principles of Slicing',
        'スライシング（Slicing）とは、3DモデルをZ軸方向に薄くスライスし、各層の輪郭（Contour）とインフィル（Infill）パターンを生成する処理です。':
            'Slicing is the process of slicing a 3D model thinly along the Z-axis and generating contour and infill patterns for each layer.',

        'レイヤー高さ': 'Layer height',
        '一般に0.1-0.3 mm（ノズル径の20-75%）': 'Generally 0.1-0.3 mm (20-75% of nozzle diameter)',
        '低いほど高品質だが造形時間増加': 'Lower values yield higher quality but increase build time',
        '外壁・内壁': 'Outer/inner walls',
        '外壁（Perimeter）': 'Outer wall (Perimeter)',
        '表面品質を決定、2-4層が標準': 'Determines surface quality, 2-4 layers standard',
        '内部充填': 'Internal infill',

        '1.4.2 シェルとインフィル戦略': '1.4.2 Shell and Infill Strategies',
        'シェル（Shell）': 'Shell',
        '外壁と天井・床面を構成する層': 'Layers forming outer walls and top/bottom surfaces',
        'インフィル（Infill）': 'Infill',
        '内部の充填パターン': 'Internal fill pattern',

        '💡 インフィル密度の目安': '💡 Infill Density Guidelines',
        '0-10%': '0-10%',
        '装飾品、視覚モデル': 'Decorative items, visual models',
        '15-30%': '15-30%',
        'プロトタイプ、通常使用': 'Prototypes, normal use',
        '40-60%': '40-60%',
        '機械部品、高強度必要': 'Mechanical parts, high strength required',
        '80-100%': '80-100%',
        '最終製品、極限強度': 'End-use parts, maximum strength',

        '1.4.3 サポート構造の生成': '1.4.3 Support Structure Generation',
        'サポート（Support）': 'Support',
        'オーバーハング部を支える仮構造': 'Temporary structure supporting overhangs',
        'オーバーハング角': 'Overhang angle',
        '一般に45°以上でサポート必要': 'Generally requires support at >45°',
        'サポート密度': 'Support density',
        '5-15%が標準': '5-15% is standard',
        '除去しやすさ': 'Ease of removal',
        '接触面積を最小化': 'Minimize contact area',

        '1.4.4 G-codeの基礎': '1.4.4 G-code Fundamentals',
        'G-codeは、3Dプリンタの動作を制御する機械語命令です。':
            'G-code is the machine instruction language controlling 3D printer operations.',

        '移動': 'Movement',
        '高速移動（材料吐出なし）': 'Rapid movement (no extrusion)',
        '制御された移動（材料吐出）': 'Controlled movement (with extrusion)',
        '原点復帰': 'Home position',
        '温度': 'Temperature',
        'ホットエンド温度設定': 'Set hotend temperature',
        'ベッド温度設定': 'Set bed temperature',
        '初期化': 'Initialization',
        'メートル単位': 'Metric units',
        '絶対座標モード': 'Absolute positioning mode',

        '⚠️ レイヤー高さの制約': '⚠️ Layer Height Constraints',
        'ノズル径との関係': 'Relationship with nozzle diameter',
        'レイヤー高さ < 0.8 × ノズル径（推奨）': 'Layer height < 0.8 × nozzle diameter (recommended)',
        '0.4 mmノズル → 最大0.32 mm': '0.4 mm nozzle → maximum 0.32 mm',
        '解像度と時間': 'Resolution vs time',
        '0.1 mm → 高品質、3倍遅い': '0.1 mm → high quality, 3x slower',
        '0.3 mm → 低品質、高速': '0.3 mm → lower quality, faster',

        '1.4.5 主要スライシングソフトウェア': '1.4.5 Major Slicing Software',
        'オープンソース、高度な設定項目': 'Open source, advanced settings',
        'オープンソース、Prusa製プリンタ標準': 'Open source, standard for Prusa printers',
        'Ultimaker社開発、使いやすいUI': 'Developed by Ultimaker, user-friendly UI',
        '産業用、Stratasys装置専用': 'Industrial, exclusive to Stratasys equipment',

        '1.4.6 ツールパス最適化戦略': '1.4.6 Toolpath Optimization Strategies',
        '印刷速度': 'Print speed',
        '外壁': 'Outer wall',
        '30-50 mm/s（品質優先）': '30-50 mm/s (quality priority)',
        'インフィル': 'Infill',
        '60-100 mm/s（高速可）': '60-100 mm/s (fast possible)',
        '移動': 'Travel',
        '120-200 mm/s（最速）': '120-200 mm/s (fastest)',

        'リトラクション（Retraction）': 'Retraction',
        '材料引き戻し': 'Material pullback',
        'ストリンギング（糸引き）防止': 'Prevents stringing',
        '距離': 'Distance',
        '1-6 mm（ダイレクト駆動1-3 mm、ボーデン4-6 mm）': '1-6 mm (direct drive 1-3 mm, Bowden 4-6 mm)',
        '速度': 'Speed',
        '25-45 mm/s': '25-45 mm/s',

        'シーム配置（Seam Placement）': 'Seam Placement',
        '層の開始・終了点': 'Layer start/end point',
        '目立たない位置に配置': 'Place in inconspicuous location',
        '背面隅': 'Back corner',
        '最も目立たない': 'Least noticeable',
        'ランダム': 'Random',
        'シーム分散': 'Seam distribution',
        '最短距離': 'Shortest distance',
        '高速化優先': 'Speed priority',
    }
    return apply_replacements(content, replacements)

def translate_exercises(content):
    """Translate exercise sections"""
    replacements = {
        '演習問題': 'Exercises',
        'Easy（基礎確認）': 'Easy (Fundamentals)',
        'Medium（応用）': 'Medium (Application)',
        'Hard（発展）': 'Hard (Advanced)',
        '解答を表示': 'Show Answer',
        '解答を見る': 'View Answer',
        '正解:': 'Correct Answer:',
        '解説:': 'Explanation:',
        '理由:': 'Reason:',
        '計算手順:': 'Calculation Steps:',
        '答え:': 'Answer:',
        '問題': 'Question',
        '選択肢': 'Options',
        '計算せよ': 'Calculate',
        '比較せよ': 'Compare',
        '説明せよ': 'Explain',
    }
    return apply_replacements(content, replacements)

def translate_tables(content):
    """Translate table headers and content"""
    replacements = {
        'パラメータ': 'Parameter',
        '推奨値': 'Recommended Value',
        '効果': 'Effect',
        'コマンド': 'Command',
        '分類': 'Category',
        '機能': 'Function',
        '例': 'Example',
        '材料': 'Material',
        '装置': 'Equipment',
        '価格': 'Price',
        '特長': 'Features',
        '用途': 'Applications',
    }
    return apply_replacements(content, replacements)

def translate_mermaid_diagrams(content):
    """Translate Mermaid diagram text"""
    replacements = {
        'SLA発明<br/>Chuck Hull': 'SLA Invented<br/>Chuck Hull',
        'SLS登場<br/>Carl Deckard': 'SLS Introduced<br/>Carl Deckard',
        'FDM特許<br/>Stratasys社': 'FDM Patent<br/>Stratasys',
        'RepRap<br/>オープンソース化': 'RepRap<br/>Open Source',
        '金属AM普及<br/>EBM/SLM': 'Metal AM Adoption<br/>EBM/SLM',
        '産業化加速<br/>大型・高速化': 'Industrial Acceleration<br/>Large-scale & High-speed',

        '積層造形<br/>7つのプロセス': 'Additive Manufacturing<br/>7 Processes',
        '材料押出': 'Material Extrusion',
        '液槽光重合': 'Vat Photopolymerization',
        '粉末床溶融結合': 'Powder Bed Fusion',
        '材料噴射': 'Material Jetting',
        '結合剤噴射': 'Binder Jetting',
        'シート積層': 'Sheet Lamination',
        '指向性エネルギー堆積': 'Directed Energy Deposition',
        '低コスト・普及型': 'Low-cost & Popular',
        '高精度・高表面品質': 'High precision & Surface quality',
        '高強度・金属対応': 'High strength & Metal compatible',

        '3Dモデル<br/>STLファイル': '3D Model<br/>STL File',
        'Z軸方向に<br/>層状にスライス': 'Slice into layers<br/>along Z-axis',
        '各層の輪郭抽出<br/>Contour Detection': 'Extract contours<br/>per layer',
        'シェル生成<br/>Perimeter Path': 'Generate shells<br/>Perimeter Path',
        'インフィル生成<br/>Infill Path': 'Generate infill<br/>Infill Path',
        'サポート追加<br/>Support Structure': 'Add supports<br/>Support Structure',
        'ツールパス最適化<br/>Retraction/Travel': 'Optimize toolpath<br/>Retraction/Travel',
        'G-code出力': 'G-code Output',
    }
    return apply_replacements(content, replacements)

def translate_code_comments(content):
    """Translate Python code comments"""
    replacements = {
        '# STLファイルを読み込み': '# Load STL file',
        '# 基本的な幾何情報': '# Basic geometric information',
        '# 三角形数': '# Number of triangles',
        '# バウンディングボックス': '# Bounding box',
        '# 造形時間の簡易推定': '# Simple build time estimation',
        '# 出力例:': '# Output example:',
        '# === STLファイル基本情報 ===': '# === STL File Basic Information ===',
        '# === バウンディングボックス ===': '# === Bounding Box ===',
        '# === 造形推定 ===': '# === Build Estimation ===',
        '# === 法線ベクトル検証結果 ===': '# === Normal Vector Validation Results ===',
        '# === メッシュ品質診断 ===': '# === Mesh Quality Diagnostics ===',
        '# === 3Dプリント適性チェック ===': '# === 3D Printing Suitability Check ===',
        '# === 修復後の状態 ===': '# === Post-Repair Status ===',

        'すべての法線が正しい方向を向いています': 'All normals are correctly oriented',
        'このメッシュは3Dプリント可能です': 'This mesh is printable',
        '一部の法線が反転しています（軽微）': 'Some normals are inverted (minor)',
        '多数の法線が反転しています（重大）': 'Many normals are inverted (critical)',
        'メッシュ修復ツール（Meshmixer, netfabb）での修正を推奨':
            'Recommend repair using mesh repair tools (Meshmixer, Netfabb)',
        '修復完了！': 'Repair complete!',
        '自動修復失敗': 'Automatic repair failed',
    }
    return apply_replacements(content, replacements)

def translate_info_boxes(content):
    """Translate info/warning/success boxes"""
    replacements = {
        '💡 産業的重要性': '💡 Industrial Significance',
        '💡 STLファイルの解像度トレードオフ': '💡 STL Resolution Trade-offs',
        '💡 インフィル密度の目安': '💡 Infill Density Guidelines',
        '💡 FDMの代表的装置': '💡 Representative FDM Systems',
        '⚠️ AMの制約と課題': '⚠️ AM Constraints and Challenges',
        '⚠️ レイヤー高さの制約': '⚠️ Layer Height Constraints',
        '⚠️ 非多様体メッシュの問題': '⚠️ Non-Manifold Mesh Issues',
        '⚠️ プロセス選択の指針': '⚠️ Process Selection Guidelines',
    }
    return apply_replacements(content, replacements)

def translate_references(content):
    """Translate references section"""
    replacements = {
        '参考文献': 'References',
        '使用ツールとライブラリ': 'Tools and Libraries Used',
        '図': 'Figure',
        '表': 'Table',
        '注': 'Note',
        '重要': 'Important',
    }
    return apply_replacements(content, replacements)

def translate_navigation(content):
    """Translate navigation elements"""
    replacements = {
        '次のステップ': 'Next Steps',
        'シリーズ目次': 'Series Index',
        '第2章へ進む →': 'Proceed to Chapter 2 →',
        '第3章へ進む →': 'Proceed to Chapter 3 →',
        '← 第1章に戻る': '← Back to Chapter 1',
        '目次に戻る': 'Back to Index',
    }
    return apply_replacements(content, replacements)

def final_cleanup(content):
    """Final cleanup of remaining patterns"""
    # Remove Japanese particles and grammar
    content = re.sub(r'です。', '.', content)
    content = re.sub(r'ます。', '.', content)
    content = re.sub(r'から', '', content)
    content = re.sub(r'まで', '', content)
    content = re.sub(r'など', ' etc.', content)
    content = re.sub(r'とは', ' is', content)
    content = re.sub(r'について', '', content)
    content = re.sub(r'により', ' by', content)
    content = re.sub(r'では', '', content)
    content = re.sub(r'には', '', content)
    content = re.sub(r'への', ' to', content)
    content = re.sub(r'がある', ' exists', content)
    content = re.sub(r'ができる', ' can', content)
    content = re.sub(r'をする', '', content)

    # Fix spacing issues
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'>\s+<', '><', content)

    return content

def apply_replacements(content, replacements):
    """Apply dictionary of replacements to content"""
    for jp, en in replacements.items():
        content = content.replace(jp, en)
    return content

if __name__ == "__main__":
    remaining = main()
    exit(0 if remaining == 0 else 1)
