#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive translation for 3D Printing Introduction Chapter 4
Handles all Japanese content systematically
"""

import re

def translate_3dprint_chapter4():
    source = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/3d-printing-introduction/chapter-4.html'
    target = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-4.html'

    with open(source, 'r', encoding='utf-8') as f:
        content = f.read()

    # Comprehensive translation dictionary
    translations = {
        # HTML attributes
        'lang="ja"': 'lang="en"',

        # Title and meta
        '第4章：材料噴射法・結合剤噴射法・その他AM技術': 'Chapter 4: Fundamentals of Additive Manufacturing',

        # Breadcrumb
        'AI寺子屋トップ': 'AI Terakoya Top',
        '材料科学': 'Materials Science',

        # Header
        '第4章：積層造形の基礎': 'Chapter 4: Fundamentals of Additive Manufacturing',
        'AM技術の原理と分類 - 3Dプリンティングの技術体系': 'Principles and Classification of AM Technologies - 3D Printing Technical Framework',
        '📚 3Dプリンティング入門シリーズ': '📚 3D Printing Introduction Series',
        '⏱️ 読了時間: 35-40分': '⏱️ Reading time: 35-40 minutes',
        '🎓 難易度: 初級〜中級': '🎓 Difficulty: Beginner to Intermediate',

        # Main sections
        '学習目標': 'Learning Objectives',
        'この章を完了すると、以下を説明できるようになります：': 'Upon completing this chapter, you will be able to explain:',

        '基本理解（Level 1）': 'Basic Understanding (Level 1)',
        '実践スキル（Level 2)': 'Practical Skills (Level 2)',
        '応用力（Level 3）': 'Application Skills (Level 3)',

        # Level 1 objectives
        '積層造形（AM）の定義とISO/ASTM 52900規格の基本概念': 'Definition of Additive Manufacturing (AM) and basic concepts of ISO/ASTM 52900 standard',
        '7つのAMプロセスカテゴリ（MEX, VPP, PBF, MJ, BJ, SL, DED）の特徴': 'Characteristics of 7 AM process categories (MEX, VPP, PBF, MJ, BJ, SL, DED)',
        'STLファイル形式の構造（三角形メッシュ、法線ベクトル、頂点順序）': 'Structure of STL file format (triangle mesh, normal vectors, vertex order)',
        'AMの歴史（1986年ステレオリソグラフィから現代システムまで）': 'History of AM (from 1986 stereolithography to modern systems)',

        # Level 2 objectives
        'PythonでSTLファイルを読み込み、体積・表面積を計算できる': 'Ability to read STL files in Python and calculate volume and surface area',
        'numpy-stlとtrimeshを使ったメッシュ検証と修復ができる': 'Ability to validate and repair meshes using numpy-stl and trimesh',
        'スライシングの基本原理（レイヤー高さ、シェル、インフィル）を理解': 'Understanding of basic slicing principles (layer height, shell, infill)',
        'G-codeの基本構造（G0/G1/G28/M104など）を読み解ける': 'Ability to interpret basic G-code structure (G0/G1/G28/M104, etc.)',

        # Level 3 objectives
        '用途要求に応じて最適なAMプロセスを選択できる': 'Ability to select optimal AM process according to application requirements',
        'メッシュの問題（非多様体、法線反転）を検出・修正できる': 'Ability to detect and fix mesh problems (non-manifold, inverted normals)',
        '造形パラメータ（レイヤー高さ、印刷速度、温度）を最適化できる': 'Ability to optimize build parameters (layer height, print speed, temperature)',
        'STLファイルの品質評価とプリント適性判断ができる': 'Ability to assess STL file quality and printability',

        # Section headers
        '1.1 積層造形（AM）とは': '1.1 What is Additive Manufacturing (AM)?',
        '1.1.1 積層造形の定義': '1.1.1 Definition of Additive Manufacturing',
        '1.1.2 AMの歴史と発展': '1.1.2 History and Evolution of AM',
        '1.1.3 AMの主要応用分野': '1.1.3 Major Application Areas of AM',
        '1.2 ISO/ASTM 52900による7つのAMプロセス分類': '1.2 Seven AM Process Categories by ISO/ASTM 52900',
        '1.2.1 AMプロセス分類の全体像': '1.2.1 Overview of AM Process Classification',
        '1.2.2 Material Extrusion (MEX) - 材料押出': '1.2.2 Material Extrusion (MEX)',
        '1.2.3 Vat Photopolymerization (VPP) - 液槽光重合': '1.2.3 Vat Photopolymerization (VPP)',
        '1.2.4 Powder Bed Fusion (PBF) - 粉末床溶融結合': '1.2.4 Powder Bed Fusion (PBF)',
        '1.2.5 Material Jetting (MJ) - 材料噴射': '1.2.5 Material Jetting (MJ)',
        '1.2.6 Binder Jetting (BJ) - 結合剤噴射': '1.2.6 Binder Jetting (BJ)',
        '1.2.7 Sheet Lamination (SL) - シート積層': '1.2.7 Sheet Lamination (SL)',
        '1.2.8 Directed Energy Deposition (DED) - 指向性エネルギー堆積': '1.2.8 Directed Energy Deposition (DED)',
        '1.3 STLファイル形式とデータ処理': '1.3 STL File Format and Data Processing',
        '1.3.1 STLファイルの構造': '1.3.1 Structure of STL Files',
        '1.3.2 STLファイルの重要概念': '1.3.2 Important Concepts of STL Files',
        '1.3.3 STLファイルの品質指標': '1.3.3 STL File Quality Metrics',
        '1.3.4 Pythonライブラリによる STL処理': '1.3.4 STL Processing with Python Libraries',
        '1.4 スライシングとツールパス生成': '1.4 Slicing and Toolpath Generation',
        '1.4.1 スライシングの基本原理': '1.4.1 Basic Principles of Slicing',
        '1.4.2 シェルとインフィル戦略': '1.4.2 Shell and Infill Strategies',
        '1.4.3 サポート構造の生成': '1.4.3 Support Structure Generation',
        '1.4.4 G-codeの基礎': '1.4.4 G-code Fundamentals',
        '1.4.5 主要スライシングソフトウェア': '1.4.5 Major Slicing Software',
        '1.4.6 ツールパス最適化戦略': '1.4.6 Toolpath Optimization Strategies',

        # Content paragraphs - key terms
        '積層造形（Additive Manufacturing, AM）とは、': 'Additive Manufacturing (AM) is ',
        '<strong>ISO/ASTM 52900:2021規格で定義される「3次元CADデータから材料を層ごとに積み上げて物体を製造するプロセス」</strong>です。': '<strong>defined by ISO/ASTM 52900:2021 standard as "the process of manufacturing objects by adding material layer by layer from 3D CAD data"</strong>.',
        '従来の切削加工（除去加工）とは対照的に、必要な部分にのみ材料を付加するため、以下の革新的な特徴を持ちます：': 'In contrast to traditional subtractive machining (material removal), AM adds material only where needed, providing the following innovative characteristics:',

        # Key features
        '設計自由度': 'Design Freedom',
        '従来製法では不可能な複雑形状（中空構造、ラティス構造、トポロジー最適化形状）を製造可能': 'Capability to manufacture complex geometries impossible with traditional methods (hollow structures, lattice structures, topology-optimized shapes)',
        '材料効率': 'Material Efficiency',
        '必要な部分にのみ材料を使用するため、材料廃棄率が5-10%（従来加工は30-90%廃棄）': 'Material waste rate of 5-10% by using material only where needed (traditional machining: 30-90% waste)',
        'オンデマンド製造': 'On-Demand Manufacturing',
        '金型不要でカスタマイズ製品を少量・多品種生産可能': 'Capability for low-volume, high-variety production of customized products without tooling',
        '一体化製造': 'Integrated Manufacturing',
        '従来は複数部品を組立てていた構造を一体造形し、組立工程を削減': 'One-piece fabrication of structures that traditionally required assembly of multiple parts, reducing assembly steps',

        # Info boxes
        '💡 産業的重要性': '💡 Industrial Importance',
        'AM市場は急成長中で、Wohlers Report 2023によると：': 'The AM market is growing rapidly. According to Wohlers Report 2023:',
        '世界のAM市場規模: $18.3B（2023年）→ $83.9B予測（2030年、年成長率23.5%）': 'Global AM market size: $18.3B (2023) → $83.9B projected (2030, 23.5% CAGR)',
        '用途の内訳: プロトタイピング（38%）、ツーリング（27%）、最終製品（35%）': 'Application breakdown: Prototyping (38%), Tooling (27%), End-use parts (35%)',
        '主要産業: 航空宇宙（26%）、医療（21%）、自動車（18%）、消費財（15%）': 'Major industries: Aerospace (26%), Medical (21%), Automotive (18%), Consumer goods (15%)',
        '材料別シェア: ポリマー（55%）、金属（35%）、セラミックス（7%）、その他（3%）': 'Material share: Polymers (55%), Metals (35%), Ceramics (7%), Others (3%)',

        # History
        '積層造形技術は約40年の歴史を持ち、以下のマイルストーンを経て現在に至ります：': 'Additive manufacturing technology has approximately 40 years of history, evolving through the following milestones:',
        'SLA発明': 'SLA Invented',
        'SLS登場': 'SLS Introduced',
        'FDM特許': 'FDM Patent',
        'Stratasys社': 'Stratasys Inc.',
        'オープンソース化': 'Open Source',
        '金属AM普及': 'Metal AM Adoption',
        '産業化加速': 'Industrial Acceleration',
        '大型・高速化': 'Large-scale & High-speed',

        # Applications
        '応用1: プロトタイピング（Rapid Prototyping）': 'Application 1: Rapid Prototyping',
        'AMの最初の主要用途で、設計検証・機能試験・市場評価用のプロトタイプを迅速に製造します：': 'The first major application of AM, for rapid manufacturing of prototypes for design verification, functional testing, and market evaluation:',
        'リードタイム短縮': 'Lead Time Reduction',
        '従来の試作（数週間〜数ヶ月）→ AMでは数時間〜数日': 'Traditional prototyping (weeks to months) → AM in hours to days',
        '設計反復の加速': 'Accelerated Design Iteration',
        '低コストで複数バージョンを試作し、設計を最適化': 'Prototype multiple versions at low cost to optimize design',
        'コミュニケーション改善': 'Improved Communication',
        '視覚的・触覚的な物理モデルで関係者間の認識を統一': 'Unify understanding among stakeholders with visual and tactile physical models',
        '典型例': 'Typical Examples',
        '自動車の意匠モデル、家電製品の筐体試作、医療機器の術前シミュレーションモデル': 'Automotive design models, consumer electronics housing prototypes, pre-surgical simulation models for medical devices',

        '応用2: ツーリング（Tooling & Fixtures）': 'Application 2: Tooling & Fixtures',
        '製造現場で使用する治具・工具・金型をAMで製造する応用です：': 'Application of manufacturing jigs, tools, and molds used in production facilities with AM:',
        'カスタム治具': 'Custom Fixtures',
        '生産ラインに特化した組立治具・検査治具を迅速に製作': 'Rapid fabrication of assembly and inspection fixtures specialized for production lines',
        'コンフォーマル冷却金型': 'Conformal Cooling Molds',
        '従来の直線的冷却路ではなく、製品形状に沿った3次元冷却路を内蔵した射出成形金型（冷却時間30-70%短縮）': 'Injection molds with 3D cooling channels conforming to product shape, not traditional straight channels (30-70% cooling time reduction)',
        '軽量化ツール': 'Lightweight Tools',
        'ラティス構造を使った軽量エンドエフェクタで作業者の負担を軽減': 'Reduce worker burden with lightweight end-effectors using lattice structures',
        'BMWの組立ライン用治具（年間100,000個以上をAMで製造）、GolfのTaylorMadeドライバー金型': 'BMW assembly line fixtures (over 100,000 units manufactured annually with AM), TaylorMade golf driver molds',

        '応用3: 最終製品（End-Use Parts）': 'Application 3: End-Use Parts',
        'AMで直接、最終製品を製造する応用が近年急増しています：': 'Direct manufacturing of end-use products with AM has been rapidly increasing in recent years:',
        '航空宇宙部品': 'Aerospace Components',
        'GE Aviation LEAP燃料噴射ノズル（従来20部品→AM一体化、重量25%軽減、年間100,000個以上生産）': 'GE Aviation LEAP fuel injection nozzles (20 parts consolidated into one AM part, 25% weight reduction, over 100,000 units produced annually)',
        '医療インプラント': 'Medical Implants',
        'チタン製人工股関節・歯科インプラント（患者固有の解剖学的形状に最適化、骨結合を促進する多孔質構造）': 'Titanium hip replacements and dental implants (optimized for patient-specific anatomy, porous structures promoting bone integration)',
        'カスタム製品': 'Custom Products',
        '補聴器（年間1,000万個以上がAMで製造）、スポーツシューズのミッドソール（Adidas 4D、Carbon社DLS技術）': 'Hearing aids (over 10 million units manufactured annually with AM), sports shoe midsoles (Adidas 4D, Carbon DLS technology)',
        'スペア部品': 'Spare Parts',
        '絶版部品・希少部品のオンデマンド製造（自動車、航空機、産業機械）': 'On-demand manufacturing of discontinued and rare parts (automotive, aircraft, industrial machinery)',

        # Warning boxes
        '⚠️ AMの制約と課題': '⚠️ AM Constraints and Challenges',
        'AMは万能ではなく、以下の制約があります：': 'AM is not universal and has the following constraints:',
        '造形速度': 'Build Speed',
        '大量生産には不向き（射出成形1個/数秒 vs AM数時間）。経済的ブレークイーブンは通常1,000個以下': 'Not suitable for mass production (injection molding 1 piece/seconds vs AM hours). Economic break-even typically below 1,000 units',
        '造形サイズ制限': 'Build Size Limitations',
        'ビルドボリューム（多くの装置で200×200×200mm程度）を超える大型部品は分割製造が必要': 'Large parts exceeding build volume (typically around 200×200×200mm for many machines) require segmented manufacturing',
        '表面品質': 'Surface Quality',
        '積層痕（layer lines）が残るため、高精度表面が必要な場合は後加工必須（研磨、機械加工）': 'Layer lines remain, requiring post-processing (polishing, machining) when high-precision surfaces are needed',
        '材料特性の異方性': 'Material Property Anisotropy',
        '積層方向（Z軸）と面内方向（XY平面）で機械的性質が異なる場合がある（特にFDM）': 'Mechanical properties may differ between build direction (Z-axis) and in-plane direction (XY-plane), especially in FDM',
        '材料コスト': 'Material Cost',
        'AMグレード材料は汎用材料の2-10倍高価（ただし材料効率と設計最適化で相殺可能）': 'AM-grade materials are 2-10 times more expensive than generic materials (can be offset by material efficiency and design optimization)',

        # Process descriptions
        '原理': 'Principle',
        '熱可塑性樹脂フィラメントを加熱・溶融し、ノズルから押し出して積層。最も普及している技術（FDM/FFFとも呼ばれる）。': 'Thermoplastic filament is heated and melted, then extruded through a nozzle for layer-by-layer deposition. The most widespread technology (also called FDM/FFF).',
        '特徴：': 'Characteristics:',
        '低コスト': 'Low Cost',
        '装置価格$200-$5,000（デスクトップ）、$10,000-$100,000（産業用）': 'Equipment price $200-$5,000 (desktop), $10,000-$100,000 (industrial)',
        '材料多様性': 'Material Diversity',
        'PLA、ABS、PETG、ナイロン、PC、カーボン繊維複合材、PEEK（高性能）': 'PLA, ABS, PETG, nylon, PC, carbon fiber composites, PEEK (high-performance)',
        '20-150 mm³/s（中程度）、レイヤー高さ0.1-0.4mm': '20-150 mm³/s (moderate), layer height 0.1-0.4mm',
        '精度': 'Accuracy',
        '±0.2-0.5 mm（デスクトップ）、±0.1 mm（産業用）': '±0.2-0.5 mm (desktop), ±0.1 mm (industrial)',
        '積層痕が明瞭（後加工で改善可能）': 'Layer lines are visible (improvable with post-processing)',
        '材料異方性': 'Material Anisotropy',
        'Z軸方向（積層方向）の強度が20-80%低い（層間接着が弱点）': 'Z-axis (build direction) strength is 20-80% lower (interlayer adhesion is weakness)',

        '応用例：': 'Applications:',
        'プロトタイピング（最も一般的な用途、低コスト・高速）': 'Prototyping (most common application, low cost and fast)',
        '治具・工具（製造現場で使用、軽量・カスタマイズ容易）': 'Jigs and tools (used in manufacturing, lightweight and easily customizable)',
        '教育用モデル（学校・大学で広く使用、安全・低コスト）': 'Educational models (widely used in schools and universities, safe and low cost)',
        '最終製品（カスタム補聴器、義肢装具、建築模型）': 'End-use parts (custom hearing aids, prosthetics, architectural models)',

        # Equipment
        '💡 FDMの代表的装置': '💡 Representative FDM Equipment',
        'デュアルヘッド、ビルドボリューム330×240×300mm、$6,000': 'Dual head, build volume 330×240×300mm, $6,000',
        'オープンソース系、高い信頼性、$1,200': 'Open source based, high reliability, $1,200',
        '産業用、ULTEM 9085対応、$250,000': 'Industrial, ULTEM 9085 compatible, $250,000',
        '連続カーボン繊維複合材対応、$100,000': 'Continuous carbon fiber composite compatible, $100,000',

        # VPP
        '液状の光硬化性樹脂（フォトポリマー）に紫外線（UV）レーザーまたはプロジェクターで光を照射し、選択的に硬化させて積層。':
            'Liquid photopolymer resin is selectively cured layer by layer using ultraviolet (UV) laser or projector light.',
        'VPPの2つの主要方式：': 'Two main VPP methods:',
        '<strong>SLA（Stereolithography）</strong>: UV レーザー（355 nm）をガルバノミラーで走査し、点描的に硬化。高精度だが低速。':
            '<strong>SLA (Stereolithography)</strong>: UV laser (355 nm) scanned with galvanometer mirrors, pointwise curing. High precision but slow.',
        '<strong>DLP（Digital Light Processing）</strong>: プロジェクターで面全体を一括露光。高速だが解像度はプロジェクター画素数に依存（Full HD: 1920×1080）。':
            '<strong>DLP (Digital Light Processing)</strong>: Entire layer exposed at once with projector. Fast but resolution depends on projector pixel count (Full HD: 1920×1080).',
        '<strong>LCD-MSLA（Masked SLA）</strong>: LCDマスクを使用、DLP類似だが低コスト化（$200-$1,000のデスクトップ機多数）。':
            '<strong>LCD-MSLA (Masked SLA)</strong>: Uses LCD mask, similar to DLP but lower cost (many desktop machines $200-$1,000).',

        '高精度': 'High Precision',
        'XY解像度25-100 μm、Z解像度10-50 μm（全AM技術中で最高レベル）': 'XY resolution 25-100 μm, Z resolution 10-50 μm (highest level among all AM technologies)',
        '滑らかな表面（Ra < 5 μm）、積層痕がほぼ見えない': 'Smooth surface (Ra < 5 μm), layer lines nearly invisible',
        'SLA（10-50 mm³/s）、DLP/LCD（100-500 mm³/s、面積依存）': 'SLA (10-50 mm³/s), DLP/LCD (100-500 mm³/s, area dependent)',
        '材料制約': 'Material Constraints',
        '光硬化性樹脂のみ（機械的性質はFDMより劣る場合が多い）': 'Photopolymer resin only (mechanical properties often inferior to FDM)',
        '後処理必須': 'Post-processing Required',
        '洗浄（IPA等）→ 二次硬化（UV照射）→ サポート除去': 'Cleaning (IPA etc.) → Secondary curing (UV exposure) → Support removal',

        '歯科用途（歯列矯正モデル、サージカルガイド、義歯、年間数百万個生産）': 'Dental applications (orthodontic models, surgical guides, dentures, millions produced annually)',
        'ジュエリー鋳造用ワックスモデル（高精度・複雑形状）': 'Wax models for jewelry casting (high precision, complex shapes)',
        '医療モデル（術前計画、解剖学モデル、患者説明用）': 'Medical models (surgical planning, anatomical models, patient education)',
        'マスターモデル（シリコン型取り用、デザイン検証）': 'Master models (for silicone molding, design verification)',

        # PBF
        '粉末材料を薄く敷き詰め、レーザーまたは電子ビームで選択的に溶融・焼結し、冷却固化させて積層。金属・ポリマー・セラミックスに対応。':
            'Powder material is spread in thin layers, selectively melted or sintered with laser or electron beam, then cooled and solidified. Compatible with metals, polymers, and ceramics.',
        'PBFの3つの主要方式：': 'Three main PBF methods:',
        '<strong>SLS（Selective Laser Sintering）</strong>: ポリマー粉末（PA12ナイロン等）をレーザー焼結。サポート不要（周囲粉末が支持）。':
            '<strong>SLS (Selective Laser Sintering)</strong>: Laser sintering of polymer powder (PA12 nylon etc.). No support needed (surrounding powder provides support).',
        '<strong>SLM（Selective Laser Melting）</strong>: 金属粉末（Ti-6Al-4V、AlSi10Mg、Inconel 718等）を完全溶融。高密度部品（相対密度>99%）製造可能。':
            '<strong>SLM (Selective Laser Melting)</strong>: Complete melting of metal powder (Ti-6Al-4V, AlSi10Mg, Inconel 718 etc.). Can produce high-density parts (relative density >99%).',
        '<strong>EBM（Electron Beam Melting）</strong>: 電子ビームで金属粉末を溶融。高温予熱（650-1000°C）により残留応力が小さく、造形速度が速い。':
            '<strong>EBM (Electron Beam Melting)</strong>: Melting metal powder with electron beam. High-temperature preheating (650-1000°C) results in low residual stress and faster build speed.',

        '高強度': 'High Strength',
        '溶融・再凝固により鍛造材に匹敵する機械的性質（引張強度500-1200 MPa）': 'Mechanical properties comparable to forged materials through melting and re-solidification (tensile strength 500-1200 MPa)',
        '複雑形状対応': 'Complex Geometry Capability',
        'サポート不要（粉末が支持）でオーバーハング造形可能': 'Can build overhangs without support (powder provides support)',
        'Ti合金、Al合金、ステンレス鋼、Ni超合金、Co-Cr合金、ナイロン': 'Ti alloys, Al alloys, stainless steel, Ni superalloys, Co-Cr alloys, nylon',
        '高コスト': 'High Cost',
        '装置価格$200,000-$1,500,000、材料費$50-$500/kg': 'Equipment price $200,000-$1,500,000, material cost $50-$500/kg',
        '後処理': 'Post-processing',
        'サポート除去、熱処理（応力除去）、表面仕上げ（ブラスト、研磨）': 'Support removal, heat treatment (stress relief), surface finishing (blasting, polishing)',

        '航空宇宙部品（軽量化、一体化、GE LEAP燃料ノズル等）': 'Aerospace components (weight reduction, integration, GE LEAP fuel nozzles etc.)',
        '金型（コンフォーマル冷却、複雑形状、H13工具鋼）': 'Molds (conformal cooling, complex shapes, H13 tool steel)',
        '自動車部品（軽量化ブラケット、カスタムエンジン部品）': 'Automotive parts (lightweight brackets, custom engine components)',

        # Other processes
        'インクジェットプリンタと同様に、液滴状の材料（光硬化性樹脂またはワックス）をヘッドから噴射し、UV照射で即座に硬化させて積層。':
            'Similar to inkjet printers, droplets of material (photopolymer resin or wax) are jetted from heads and immediately cured with UV exposure for layer-by-layer build.',
        '超高精度': 'Ultra-high Precision',
        'XY解像度42-85 μm、Z解像度16-32 μm': 'XY resolution 42-85 μm, Z resolution 16-32 μm',
        'マルチマテリアル': 'Multi-material',
        '同一造形で複数材料・複数色を使い分け可能': 'Can use multiple materials and colors within single build',
        'フルカラー造形': 'Full-color Build',
        'CMYK樹脂の組合せで1,000万色以上の表現': 'Over 10 million colors expressible through CMYK resin combinations',
        '極めて滑らか（積層痕ほぼなし）': 'Extremely smooth (virtually no layer lines)',
        '装置$50,000-$300,000、材料費$200-$600/kg': 'Equipment $50,000-$300,000, material cost $200-$600/kg',
        '光硬化性樹脂のみ、機械的性質は中程度': 'Photopolymer resin only, moderate mechanical properties',

        '医療解剖モデル（軟組織・硬組織を異なる材料で再現）、フルカラー建築模型、デザイン検証モデル':
            'Medical anatomical models (soft/hard tissue reproduced with different materials), full-color architectural models, design verification models',

        # Binder Jetting
        '粉末床に液状バインダー（接着剤）をインクジェット方式で噴射し、粉末粒子を結合。造形後に焼結または含浸処理で強度向上。':
            'Liquid binder (adhesive) is jetted inkjet-style onto powder bed to bond powder particles. Strength improved through sintering or infiltration after build.',
        '高速造形': 'High-speed Build',
        'レーザー走査不要で面全体を一括処理、造形速度100-500 mm³/s': 'No laser scanning needed, entire layer processed at once, build speed 100-500 mm³/s',
        '金属粉末、セラミックス、砂型（鋳造用）、フルカラー（石膏）': 'Metal powder, ceramics, sand molds (for casting), full-color (gypsum)',
        'サポート不要': 'No Support Needed',
        '周囲粉末が支持、除去後リサイクル可能': 'Surrounding powder provides support, recyclable after removal',
        '低密度問題': 'Low Density Issue',
        '焼結前は脆弱（グリーン密度50-60%）、焼結後も相対密度90-98%': 'Fragile before sintering (green density 50-60%), relative density 90-98% after sintering',
        '脱脂 → 焼結（金属：1200-1400°C）→ 含浸（銅・青銅）': 'Debinding → Sintering (metal: 1200-1400°C) → Infiltration (copper/bronze)',

        '砂型鋳造用型（エンジンブロック等の大型鋳物）、金属部品（Desktop Metal、HP Metal Jet）、フルカラー像（記念品、教育モデル）':
            'Sand molds for casting (large castings like engine blocks), metal parts (Desktop Metal, HP Metal Jet), full-color figures (souvenirs, educational models)',

        # Sheet Lamination
        'シート状材料（紙、金属箔、プラスチックフィルム）を積層し、接着または溶接で結合。各層をレーザーまたはブレードで輪郭切断。':
            'Sheet materials (paper, metal foil, plastic film) are laminated and bonded by adhesive or welding. Each layer contour-cut with laser or blade.',
        '代表技術：': 'Representative Technologies:',
        '<strong>LOM（Laminated Object Manufacturing）</strong>: 紙・プラスチックシート、接着剤で積層、レーザー切断':
            '<strong>LOM (Laminated Object Manufacturing)</strong>: Paper/plastic sheets, laminated with adhesive, laser cut',
        '<strong>UAM（Ultrasonic Additive Manufacturing）</strong>: 金属箔を超音波溶接、CNC切削で輪郭加工':
            '<strong>UAM (Ultrasonic Additive Manufacturing)</strong>: Metal foil ultrasonically welded, contour machined with CNC',
        '大型造形可能、材料費安価、精度中程度、用途限定的（主に視覚モデル、金属では埋込センサー等）':
            'Large-scale build possible, low material cost, moderate accuracy, limited applications (mainly visual models, embedded sensors in metal)',

        # DED
        '金属粉末またはワイヤーを供給しながら、レーザー・電子ビーム・アークで溶融し、基板上に堆積。大型部品や既存部品の補修に使用。':
            'Metal powder or wire fed and melted with laser, electron beam, or arc, then deposited on substrate. Used for large parts and repair of existing parts.',
        '高速堆積': 'High-speed Deposition',
        '堆積速度1-5 kg/h（PBFの10-50倍）': 'Deposition rate 1-5 kg/h (10-50 times PBF)',
        '大型対応': 'Large-scale Capability',
        'ビルドボリューム制限が少ない（多軸ロボットアーム使用）': 'Minimal build volume constraints (using multi-axis robot arms)',
        '補修・コーティング': 'Repair & Coating',
        '既存部品の摩耗部分修復、表面硬化層形成': 'Repair worn parts of existing components, form surface hardened layers',
        '低精度': 'Low Precision',
        '精度±0.5-2 mm、後加工（機械加工）必須': 'Accuracy ±0.5-2 mm, post-processing (machining) required',
        'タービンブレード補修、大型航空宇宙部品、工具の耐摩耗コーティング': 'Turbine blade repair, large aerospace parts, wear-resistant tool coatings',

        # Process selection warning
        '⚠️ プロセス選択の指針': '⚠️ Process Selection Guidelines',
        '最適なAMプロセスは用途要求により異なります：': 'The optimal AM process varies by application requirements:',
        '精度最優先': 'Precision Priority',
        'VPP（SLA/DLP）またはMJ': 'VPP (SLA/DLP) or MJ',
        'MEX（FDM/FFF）': 'MEX (FDM/FFF)',
        '金属高強度部品': 'Metal High-strength Parts',
        'PBF（SLM/EBM）': 'PBF (SLM/EBM)',
        '大量生産（砂型）': 'Mass Production (Sand molds)',
        'BJ': 'BJ',
        '大型・高速堆積': 'Large-scale & High-speed Deposition',
        'DED': 'DED',

        # STL section
        'STL（STereoLithography）は、<strong>AMで最も広く使用される3Dモデルファイル形式</strong>で、1987年に3D Systems社が開発しました。':
            'STL (STereoLithography) is <strong>the most widely used 3D model file format in AM</strong>, developed by 3D Systems in 1987.',
        'STLファイルは物体表面を<strong>三角形メッシュ（Triangle Mesh）の集合</strong>として表現します。':
            'STL files represent object surfaces as <strong>a collection of triangle meshes</strong>.',

        'STLファイルの基本構造': 'Basic Structure of STL Files',
        'STLファイル = 法線ベクトル（n） + 3つの頂点座標（v1, v2, v3）× 三角形数':
            'STL file = Normal vector (n) + 3 vertex coordinates (v1, v2, v3) × Number of triangles',

        'STLフォーマットの2つの種類：': 'Two types of STL format:',
        '<strong>ASCII STL</strong>: 人間が読めるテキスト形式。ファイルサイズ大（同じモデルでBinaryの10-20倍）。デバッグ・検証に有用。':
            '<strong>ASCII STL</strong>: Human-readable text format. Large file size (10-20 times Binary for same model). Useful for debugging and verification.',
        '<strong>Binary STL</strong>: バイナリ形式、ファイルサイズ小、処理高速。産業用途で標準。構造：80バイトヘッダー + 4バイト（三角形数） + 各三角形50バイト（法線12B + 頂点36B + 属性2B）。':
            '<strong>Binary STL</strong>: Binary format, small file size, fast processing. Standard for industrial use. Structure: 80-byte header + 4 bytes (triangle count) + 50 bytes per triangle (normal 12B + vertices 36B + attributes 2B).',

        # Normal vectors
        '1. 法線ベクトル（Normal Vector）': '1. Normal Vector',
        '各三角形面には<strong>法線ベクトル（外向き方向）</strong>が定義され、物体の「内側」と「外側」を区別します。':
            'Each triangular face has a <strong>normal vector (outward direction)</strong> defined to distinguish between "inside" and "outside" of the object.',
        '法線方向は<strong>右手の法則</strong>で決定されます：': 'Normal direction is determined by the <strong>right-hand rule</strong>:',

        '頂点順序ルール：': 'Vertex Ordering Rule:',
        '頂点v1, v2, v3は反時計回り（CCW: Counter-ClockWise）に配置され、外から見て反時計回りの順序で法線が外向きになります。':
            'Vertices v1, v2, v3 are arranged counter-clockwise (CCW), so that the normal points outward when viewed from outside.',

        # Manifold
        '2. 多様体（Manifold）条件': '2. Manifold Conditions',
        'STLメッシュが3Dプリント可能であるためには、<strong>多様体（Manifold）</strong>でなければなりません：':
            'For an STL mesh to be 3D printable, it must be <strong>manifold</strong>:',
        'エッジ共有': 'Edge Sharing',
        'すべてのエッジ（辺）は正確に2つの三角形に共有される': 'Every edge is shared by exactly two triangles',
        '頂点共有': 'Vertex Sharing',
        'すべての頂点は連続した三角形扇（fan）に属する': 'Every vertex belongs to a continuous triangle fan',
        '閉じた表面': 'Closed Surface',
        '穴や開口部がなく、完全に閉じた表面を形成': 'Forms a completely closed surface without holes or openings',
        '自己交差なし': 'No Self-intersection',
        '三角形が互いに交差・貫通していない': 'Triangles do not intersect or penetrate each other',

        # Non-manifold problems
        '⚠️ 非多様体メッシュの問題': '⚠️ Non-Manifold Mesh Problems',
        '非多様体メッシュ（Non-Manifold Mesh）は3Dプリント不可能です。典型的な問題：': 'Non-manifold meshes are not 3D printable. Typical problems:',
        '穴（Holes）': 'Holes',
        '閉じていない表面、エッジが1つの三角形にのみ属する': 'Open surface, edges belonging to only one triangle',
        'T字接合（T-junction）': 'T-junction',
        'エッジが3つ以上の三角形に共有される': 'Edges shared by three or more triangles',
        '法線反転（Inverted Normals）': 'Inverted Normals',
        '法線が内側を向いている三角形が混在': 'Triangles with inward-facing normals mixed in',
        '重複頂点（Duplicate Vertices）': 'Duplicate Vertices',
        '同じ位置に複数の頂点が存在': 'Multiple vertices at the same position',
        '微小三角形（Degenerate Triangles）': 'Degenerate Triangles',
        '面積がゼロまたはほぼゼロの三角形': 'Triangles with zero or near-zero area',
        'これらの問題はスライサーソフトウェアでエラーを引き起こし、造形失敗の原因となります。':
            'These problems cause errors in slicer software and lead to build failures.',

        # Quality metrics
        'STLメッシュの品質は以下の指標で評価されます：': 'STL mesh quality is evaluated by the following metrics:',
        '三角形数（Triangle Count）': 'Triangle Count',
        '通常10,000-500,000個。過少（粗いモデル）または過多（ファイルサイズ大・処理遅延）は避ける。':
            'Typically 10,000-500,000. Avoid too few (coarse model) or too many (large file size, processing delays).',
        'エッジ長の一様性': 'Edge Length Uniformity',
        '極端に大小の三角形が混在すると造形品質低下。理想的には0.1-1.0 mm範囲。':
            'Quality degrades with extreme variation in triangle sizes. Ideally in 0.1-1.0 mm range.',
        'アスペクト比（Aspect Ratio）': 'Aspect Ratio',
        '細長い三角形（高アスペクト比）は数値誤差の原因。理想的にはアスペクト比 < 10。':
            'Elongated triangles (high aspect ratio) cause numerical errors. Ideally aspect ratio < 10.',
        '法線の一貫性': 'Normal Consistency',
        'すべての法線が外向き統一。反転法線が混在すると内外判定エラー。':
            'All normals consistently outward. Mixed inverted normals cause inside/outside determination errors.',

        # Python libraries
        'PythonでSTLファイルを扱うための主要ライブラリ：': 'Major Python libraries for handling STL files:',
        '<strong>numpy-stl</strong>: 高速STL読込・書込、体積・表面積計算、法線ベクトル操作。シンプルで軽量。':
            '<strong>numpy-stl</strong>: Fast STL read/write, volume and surface area calculation, normal vector operations. Simple and lightweight.',
        '<strong>trimesh</strong>: 包括的な3Dメッシュ処理ライブラリ。メッシュ修復、ブーリアン演算、レイキャスト、衝突検出。多機能だが依存関係多い。':
            '<strong>trimesh</strong>: Comprehensive 3D mesh processing library. Mesh repair, Boolean operations, ray casting, collision detection. Feature-rich but many dependencies.',
        '<strong>PyMesh</strong>: 高度なメッシュ処理（リメッシュ、サブディビジョン、フィーチャー抽出）。インストールやや複雑。':
            '<strong>PyMesh</strong>: Advanced mesh processing (remeshing, subdivision, feature extraction). Somewhat complex installation.',

        # Slicing section
        'STLファイルを3Dプリンタが理解できる指令（G-code）に変換するプロセスを<strong>スライシング（Slicing）</strong>といいます。':
            'The process of converting STL files into commands (G-code) that 3D printers can understand is called <strong>slicing</strong>.',
        'このセクションでは、スライシングの基本原理、ツールパス戦略、そしてG-codeの基礎を学びます。':
            'In this section, we learn the basic principles of slicing, toolpath strategies, and G-code fundamentals.',

        'スライシングは、3Dモデルを一定の高さ（レイヤー高さ）で水平に切断し、各層の輪郭を抽出するプロセスです：':
            'Slicing is the process of horizontally cutting a 3D model at constant height (layer height) and extracting the contour of each layer:',

        # Flowchart nodes
        '3Dモデル': '3D Model',
        'STLファイル': 'STL File',
        'Z軸方向に': 'In Z-axis direction',
        '層状にスライス': 'Layer-wise slicing',
        '各層の輪郭抽出': 'Contour extraction for each layer',
        'Contour Detection': 'Contour Detection',
        'シェル生成': 'Shell generation',
        'Perimeter Path': 'Perimeter Path',
        'インフィル生成': 'Infill generation',
        'Infill Path': 'Infill Path',
        'サポート追加': 'Add support',
        'Support Structure': 'Support Structure',
        'ツールパス最適化': 'Toolpath optimization',
        'Retraction/Travel': 'Retraction/Travel',
        'G-code出力': 'G-code output',

        # Layer height
        'レイヤー高さ（Layer Height）の選択': 'Layer Height Selection',
        'レイヤー高さは造形品質と造形時間のトレードオフを決定する最重要パラメータです：':
            'Layer height is the most important parameter determining the tradeoff between build quality and build time:',

        # Table headers
        'レイヤー高さ': 'Layer Height',
        '造形品質': 'Build Quality',
        '造形時間': 'Build Time',
        '典型的な用途': 'Typical Applications',

        # Table rows
        '0.1 mm（極細）': '0.1 mm (Ultra-fine)',
        '非常に高い（積層痕ほぼ不可視）': 'Very high (layer lines nearly invisible)',
        '非常に長い（×2-3倍）': 'Very long (×2-3 times)',
        'フィギュア、医療モデル、最終製品': 'Figurines, medical models, end-use parts',

        '0.2 mm（標準）': '0.2 mm (Standard)',
        '良好（積層痕は見えるが許容）': 'Good (layer lines visible but acceptable)',
        '標準': 'Standard',
        '一般的なプロトタイプ、機能部品': 'General prototypes, functional parts',

        '0.3 mm（粗）': '0.3 mm (Coarse)',
        '低い（積層痕明瞭）': 'Low (layer lines obvious)',
        '短い（×0.5倍）': 'Short (×0.5 times)',
        '初期プロトタイプ、内部構造部品': 'Initial prototypes, internal structure parts',

        # Layer height constraint warning
        '⚠️ レイヤー高さの制約': '⚠️ Layer Height Constraints',
        'レイヤー高さはノズル径の<strong>25-80%</strong>に設定する必要があります。':
            'Layer height must be set to <strong>25-80%</strong> of nozzle diameter.',
        '例えば0.4mmノズルの場合、レイヤー高さは0.1-0.32mmが推奨範囲です。':
            'For example, with a 0.4mm nozzle, layer height of 0.1-0.32mm is the recommended range.',
        'これを超えると、樹脂の押出量が不足したり、ノズルが前の層を引きずる問題が発生します。':
            'Exceeding this causes insufficient resin extrusion or the nozzle dragging previous layers.',

        # Shell and infill
        'シェル（外殻）の生成': 'Shell (Perimeter) Generation',
        '<strong>シェル（Shell/Perimeter）</strong>は、各層の外周部を形成する経路です：':
            '<strong>Shell/Perimeter</strong> is the path forming the outer periphery of each layer:',
        'シェル数（Perimeter Count）': 'Perimeter Count',
        '通常2-4本。外部品質と強度に影響。': 'Typically 2-4. Affects external quality and strength.',
        '1本: 非常に弱い、透明性高い、装飾用のみ': '1: Very weak, high transparency, decorative only',
        '2本: 標準（バランス良好）': '2: Standard (good balance)',
        '3-4本: 高強度、表面品質向上、気密性向上': '3-4: High strength, improved surface quality, improved air-tightness',
        'シェル順序': 'Shell Order',
        '内側→外側（Inside-Out）が一般的。外側→内側は表面品質重視時に使用。':
            'Inside-out is common. Outside-in is used when surface quality is prioritized.',

        'インフィル（内部充填）パターン': 'Infill (Internal Fill) Patterns',
        '<strong>インフィル（Infill）</strong>は内部構造を形成し、強度と材料使用量を制御します：':
            '<strong>Infill</strong> forms internal structure and controls strength and material usage:',

        # Infill table headers
        'パターン': 'Pattern',
        '強度': 'Strength',
        '印刷速度': 'Print Speed',
        '材料使用量': 'Material Usage',
        '特徴': 'Characteristics',

        # Infill patterns
        'Grid（格子）': 'Grid',
        '中': 'Medium',
        '速い': 'Fast',
        'シンプル、等方性、標準的な選択': 'Simple, isotropic, standard choice',

        'Honeycomb（ハニカム）': 'Honeycomb',
        '高': 'High',
        '遅い': 'Slow',
        '高強度、重量比優秀、航空宇宙用途': 'High strength, excellent strength-to-weight ratio, aerospace applications',

        'Gyroid': 'Gyroid',
        '非常に高': 'Very High',
        '3次元等方性、曲面的、最新の推奨': '3D isotropic, curved, latest recommendation',

        'Concentric（同心円）': 'Concentric',
        '低': 'Low',
        '少': 'Low',
        '柔軟性重視、シェル追従': 'Flexibility focused, follows shell',

        'Lines（直線）': 'Lines',
        '低（異方性）': 'Low (anisotropic)',
        '非常に速い': 'Very fast',
        '高速印刷、方向性強度': 'High-speed printing, directional strength',

        # Infill density guidelines
        '💡 インフィル密度の目安': '💡 Infill Density Guidelines',
        '0-10%': '0-10%',
        '装飾品、非荷重部品（材料節約優先）': 'Decorative items, non-load bearing parts (material saving priority)',
        '20%': '20%',
        '標準的なプロトタイプ（バランス良好）': 'Standard prototypes (good balance)',
        '40-60%': '40-60%',
        '機能部品、高強度要求': 'Functional parts, high strength requirements',
        '100%': '100%',
        '最終製品、水密性要求、最高強度（造形時間×3-5倍）': 'End-use parts, watertight requirements, maximum strength (build time ×3-5 times)',

        # Support structures
        'オーバーハング角度が45度を超える部分は、<strong>サポート構造（Support Structure）</strong>が必要です：':
            'Parts with overhang angles exceeding 45 degrees require <strong>support structures</strong>:',
        'サポートのタイプ': 'Support Types',
        'Linear Support（直線サポート）': 'Linear Support',
        '垂直な柱状サポート。シンプルで除去しやすいが、材料使用量多い。': 'Vertical columnar support. Simple and easy to remove, but high material usage.',
        'Tree Support（ツリーサポート）': 'Tree Support',
        '樹木状に分岐するサポート。材料使用量30-50%削減、除去しやすい。CuraやPrusaSlicerで標準サポート。':
            'Tree-like branching support. 30-50% material reduction, easy to remove. Standard in Cura and PrusaSlicer.',
        'Interface Layers（接合層）': 'Interface Layers',
        'サポート上面に薄い接合層を設ける。除去しやすく、表面品質向上。通常2-4層。':
            'Thin interface layer on support top surface. Easy removal, improved surface quality. Typically 2-4 layers.',

        'サポート設定の重要パラメータ': 'Important Support Parameters',
        'Overhang Angle': 'Overhang Angle',
        '45-60°': '45-60°',
        'この角度以上でサポート生成': 'Generate support above this angle',
        'Support Density': 'Support Density',
        '10-20%': '10-20%',
        '密度が高いほど安定だが除去困難': 'Higher density is more stable but harder to remove',
        'Support Z Distance': 'Support Z Distance',
        '0.2-0.3 mm': '0.2-0.3 mm',
        'サポートと造形物の間隔（除去しやすさ）': 'Gap between support and part (ease of removal)',
        'Interface Layers': 'Interface Layers',
        '2-4層': '2-4 layers',
        '接合層数（表面品質と除去性のバランス）': 'Number of interface layers (balance between surface quality and removability)',

        # G-code
        '<strong>G-code</strong>は、3DプリンタやCNCマシンを制御する標準的な数値制御言語です。各行が1つのコマンドを表します：':
            '<strong>G-code</strong> is the standard numerical control language for controlling 3D printers and CNC machines. Each line represents one command:',
        '主要なG-codeコマンド': 'Major G-code Commands',

        # G-code table
        'コマンド': 'Command',
        '分類': 'Category',
        '機能': 'Function',
        '例': 'Example',

        '移動': 'Movement',
        '高速移動（非押出）': 'Rapid movement (no extrusion)',
        '直線移動（押出あり）': 'Linear movement (with extrusion)',
        '初期化': 'Initialization',
        'ホームポジション復帰': 'Return to home position',
        '（全軸）': '(all axes)',
        '（Z軸のみ）': '(Z-axis only)',
        '温度': 'Temperature',
        'ノズル温度設定（非待機）': 'Nozzle temperature setting (non-blocking)',
        'ノズル温度設定（待機）': 'Nozzle temperature setting (blocking)',
        'ベッド温度設定（非待機）': 'Bed temperature setting (non-blocking)',
        'ベッド温度設定（待機）': 'Bed temperature setting (blocking)',

        # G-code example
        'G-codeの例（造形開始部分）': 'G-code Example (Build Start Section)',
        '; === Start G-code ===': '; === Start G-code ===',
        '; ベッドを60°Cに加熱開始（非待機）': '; Start bed heating to 60°C (non-blocking)',
        '; ノズルを210°Cに加熱開始（非待機）': '; Start nozzle heating to 210°C (non-blocking)',
        '; 全軸ホーミング': '; Home all axes',
        '; オートレベリング（ベッドメッシュ計測）': '; Auto-leveling (bed mesh measurement)',
        '; ベッド温度到達を待機': '; Wait for bed temperature',
        '; ノズル温度到達を待機': '; Wait for nozzle temperature',
        '; 押出量をゼロリセット': '; Reset extrusion to zero',
        '; Z軸を2mm上昇（安全確保）': '; Raise Z-axis 2mm (safety)',
        '; プライム位置へ移動': '; Move to prime position',
        '; Z軸を0.3mmへ降下（初層高さ）': '; Lower Z-axis to 0.3mm (first layer height)',
        '; プライムライン描画（ノズル詰まり除去）': '; Draw prime line (clear nozzle)',
        '; 押出量を再度ゼロリセット': '; Reset extrusion again to zero',
        '; === 造形開始 ===': '; === Build start ===',

        # Slicer software table
        'ソフトウェア': 'Software',
        'ライセンス': 'License',
        '推奨用途': 'Recommended Use',

        'オープンソース': 'Open Source',
        '使いやすい、豊富なプリセット、Tree Support標準搭載': 'Easy to use, abundant presets, Tree Support built-in',
        '初心者〜中級者、FDM汎用': 'Beginners to intermediate, general FDM',

        '高度な設定、変数レイヤー高さ、カスタムサポート': 'Advanced settings, variable layer height, custom support',
        '中級者〜上級者、最適化重視': 'Intermediate to advanced, optimization focused',

        'PrusaSlicerの元祖、軽量': 'Original PrusaSlicer, lightweight',
        'レガシーシステム、研究用途': 'Legacy systems, research applications',

        '商用（$150）': 'Commercial ($150)',
        '高速スライシング、マルチプロセス、詳細制御': 'Fast slicing, multi-process, detailed control',
        'プロフェッショナル、産業用途': 'Professional, industrial applications',

        '無料': 'Free',
        'Raise3D専用だが汎用性高い、直感的UI': 'Raise3D specific but versatile, intuitive UI',
        'Raise3Dユーザー、初心者': 'Raise3D users, beginners',

        # Toolpath optimization
        '効率的なツールパスは、造形時間・品質・材料使用量を改善します：':
            'Efficient toolpaths improve build time, quality, and material usage:',
        'リトラクション（Retraction）': 'Retraction',
        '移動時にフィラメントを引き戻してストリング（糸引き）を防止。':
            'Pull back filament during travel to prevent stringing.',
        '距離: 1-6mm（ボーデンチューブ式は4-6mm、ダイレクト式は1-2mm）': 'Distance: 1-6mm (Bowden 4-6mm, direct 1-2mm)',
        '速度: 25-45 mm/s': 'Speed: 25-45 mm/s',
        '過度なリトラクションはノズル詰まりの原因': 'Excessive retraction causes nozzle clogging',

        'Z-hop（Z軸跳躍）': 'Z-hop',
        '移動時にノズルを上昇させて造形物との衝突を回避。0.2-0.5mm上昇。造形時間微増だが表面品質向上。':
            'Raise nozzle during travel to avoid collision with build. 0.2-0.5mm lift. Slight time increase but improved surface quality.',

        'コーミング（Combing）': 'Combing',
        '移動経路をインフィル上に制限し、表面への移動痕を低減。外観重視時に有効。':
            'Restrict travel paths to infill, reducing travel marks on surface. Effective when appearance matters.',

        'シーム位置（Seam Position）': 'Seam Position',
        '各層の開始/終了点を揃える戦略。': 'Strategy for aligning layer start/end points.',
        'Random: ランダム配置（目立たない）': 'Random: Random placement (inconspicuous)',
        'Aligned: 一直線に配置（後加工でシームを除去しやすい）': 'Aligned: Aligned in line (easy to remove seam in post-processing)',
        'Sharpest Corner: 最も鋭角なコーナーに配置（目立ちにくい）': 'Sharpest Corner: Place at sharpest corner (less noticeable)',

        # Example titles
        'Example 1: STLファイルの読み込みと基本情報取得': 'Example 1: Loading STL Files and Getting Basic Information',
        'Example 2: メッシュの法線ベクトル検証': 'Example 2: Mesh Normal Vector Verification',
        'Example 3: マニフォールド性のチェック': 'Example 3: Manifold Check',
        'Example 4: 基本的なスライシングアルゴリズム': 'Example 4: Basic Slicing Algorithm',

        # Comment translations in code
        '# STLファイルを読み込む': '# Load STL file',
        '# 基本的な幾何情報を取得': '# Get basic geometric information',
        '=== STLファイル基本情報 ===': '=== STL File Basic Information ===',
        'Volume:': 'Volume:',
        'Surface Area:': 'Surface Area:',
        'Center of Gravity:': 'Center of Gravity:',
        'Number of Triangles:': 'Number of Triangles:',
        '# バウンディングボックス（最小包含直方体）を計算': '# Calculate bounding box (minimum enclosing box)',
        '=== バウンディングボックス ===': '=== Bounding Box ===',
        '幅:': 'Width:',
        '奥行:': 'Depth:',
        '高さ:': 'Height:',
        '# 造形時間の簡易推定（レイヤー高さ0.2mm、速度50mm/sと仮定）': '# Simple build time estimation (assuming 0.2mm layer height, 50mm/s speed)',
        '# 簡易計算: 表面積に基づく推定': '# Simple calculation: estimate based on surface area',
        '=== 造形推定 ===': '=== Build Estimation ===',
        'レイヤー数（0.2mm/層）:': 'Number of layers (0.2mm/layer):',
        '層': 'layers',
        '推定造形時間:': 'Estimated build time:',
        '分': 'minutes',
        '時間': 'hours',
        '# 出力例:': '# Output example:',

        # Example 2 comments
        '"""STLメッシュの法線ベクトルの整合性をチェック': '"""Check consistency of normal vectors in STL mesh',
        'Args:': 'Args:',
        'mesh_data: numpy-stlのMeshオブジェクト': 'mesh_data: numpy-stl Mesh object',
        'Returns:': 'Returns:',
        'tuple: (flipped_count, total_count, percentage)': 'tuple: (flipped_count, total_count, percentage)',
        '"""': '"""',
        '# 右手系ルールで法線方向を確認': '# Check normal direction with right-hand rule',
        '# エッジベクトルを計算': '# Calculate edge vectors',
        '# 外積で法線を計算（右手系）': '# Calculate normal with cross product (right-hand)',
        '# 正規化': '# Normalize',
        '# ゼロベクトルでないことを確認': '# Confirm not zero vector',
        '# 縮退三角形をスキップ': '# Skip degenerate triangles',
        '# ファイルに保存されている法線と比較': '# Compare with stored normal in file',
        '# 内積で方向の一致をチェック': '# Check direction match with dot product',
        '# 内積が負なら逆向き': '# If dot product negative, opposite direction',
        '# STLファイルを読み込み': '# Load STL file',
        '# 法線チェックを実行': '# Execute normal check',
        '=== 法線ベクトル検証結果 ===': '=== Normal Vector Verification Results ===',
        '総三角形数:': 'Total triangles:',
        '反転法線数:': 'Flipped normals:',
        '反転率:': 'Flip rate:',
        '✅ すべての法線が正しい方向を向いています': '✅ All normals point in correct direction',
        '   このメッシュは3Dプリント可能です': '   This mesh is 3D printable',
        '⚠️ 一部の法線が反転しています（軽微）': '⚠️ Some normals are flipped (minor)',
        '   スライサーが自動修正する可能性が高い': '   Slicer likely to auto-correct',
        '❌ 多数の法線が反転しています（重大）': '❌ Many normals are flipped (critical)',
        '   メッシュ修復ツール（Meshmixer, netfabb）での修正を推奨': '   Recommend repair with mesh repair tools (Meshmixer, netfabb)',

        # Example 3 comments
        '# ===================================': '# ===================================',
        '# Example 3: マニフォールド性（Watertight）のチェック': '# Example 3: Manifold (Watertight) Check',
        '# STLファイルを読み込み（trimeshは自動で修復を試みる）': '# Load STL file (trimesh attempts automatic repair)',
        '=== メッシュ品質診断 ===': '=== Mesh Quality Diagnosis ===',
        '# 基本情報': '# Basic information',
        'Vertex count:': 'Vertex count:',
        'Face count:': 'Face count:',
        '# マニフォールド性をチェック': '# Check manifold property',
        '=== 3Dプリント適性チェック ===': '=== 3D Print Suitability Check ===',
        'Is watertight (密閉性):': 'Is watertight:',
        'Is winding consistent (法線一致性):': 'Is winding consistent:',
        'Is valid (幾何的妥当性):': 'Is valid:',
        '# 問題の詳細を診断': '# Diagnose problem details',
        '# 穴（hole）の数を検出': '# Detect number of holes',
        '⚠️ 問題検出:': '⚠️ Problem detected:',
        '   - メッシュに穴があります': '   - Mesh has holes',
        '   - 重複エッジ数:': '   - Duplicate edges:',
        '⚠️ メッシュ構造に問題があります': '⚠️ Mesh structure has problems',
        '# 修復を試みる': '# Attempt repair',
        '🔧 自動修復を実行中...': '🔧 Executing automatic repair...',
        '# 法線を修正': '# Fix normals',
        '   ✓ 法線ベクトルを修正': '   ✓ Fixed normal vectors',
        '# 穴を埋める': '# Fill holes',
        '   ✓ 穴を充填': '   ✓ Filled holes',
        '# 縮退三角形を削除': '# Remove degenerate faces',
        '   ✓ 縮退面を削除': '   ✓ Removed degenerate faces',
        '# 重複頂点を結合': '# Merge duplicate vertices',
        '   ✓ 重複頂点を結合': '   ✓ Merged duplicate vertices',
        '# 修復後の状態を確認': '# Check post-repair status',
        '=== 修復後の状態 ===': '=== Post-repair Status ===',
        '# 修復したメッシュを保存': '# Save repaired mesh',
        '✅ 修復完了！ model_repaired.stl として保存しました': '✅ Repair complete! Saved as model_repaired.stl',
        '❌ 自動修復失敗。Meshmixer等の専用ツールを推奨': '❌ Automatic repair failed. Recommend dedicated tools like Meshmixer',
        '✅ このメッシュは3Dプリント可能です': '✅ This mesh is 3D printable',

        # Exercise section
        '演習問題': 'Exercises',
    }

    # Apply all translations
    translated = content
    count = 0
    for jp, en in translations.items():
        if jp in translated:
            translated = translated.replace(jp, en)
            count += 1

    # Calculate Japanese percentage
    jp_chars_after = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', translated))
    jp_percent = (jp_chars_after / len(translated)) * 100 if len(translated) > 0 else 0

    # Write output
    with open(target, 'w', encoding='utf-8') as f:
        f.write(translated)

    print(f"Translation complete!")
    print(f"Translations applied: {count}")
    print(f"Japanese characters remaining: {jp_chars_after}")
    print(f"Japanese percentage: {jp_percent:.2f}%")

    return jp_chars_after, jp_percent

if __name__ == '__main__':
    jp_count, jp_pct = translate_3dprint_chapter4()
    print(f"\n✓ Final count: {jp_count} Japanese characters ({jp_pct:.2f}%)")
