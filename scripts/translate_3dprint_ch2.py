#!/usr/bin/env python3
"""
Comprehensive translation script for 3D Printing Chapter 2
Translates complete Japanese HTML file to English
"""

# Read the complete Japanese source file
with open('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/3d-printing-introduction/chapter-2.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Translation mapping - comprehensive dictionary
translations = {
    'lang="ja"': 'lang="en"',

    # Title and meta
    '第2章：材料押出法（FDM/FFF）- 熱可塑性プラスチックの積層造形': 'Chapter 2: Fundamentals of Additive Manufacturing',
    '熱可塑性プラスチックの積層造形': 'Thermoplastic Additive Manufacturing',

    # Header
    '第2章：積層造形の基礎': 'Chapter 2: Fundamentals of Additive Manufacturing',
    'AM技術の原理と分類 - 3Dプリンティングの技術体系': 'AM Technology Principles and Classification - 3D Printing Technology Framework',
    '3Dプリンティング入門シリーズ': '3D Printing Introduction Series',
    '読了時間: 35-40分': 'Reading time: 35-40 minutes',
    '難易度: 初級〜中級': 'Difficulty: Beginner to Intermediate',

    # Breadcrumb
    'AI寺子屋トップ': 'AI Terakoya Top',
    '材料科学': 'Materials Science',

    # Learning objectives
    '学習目標': 'Learning Objectives',
    'この章を完了すると、以下を説明できるようになります：': 'Upon completing this chapter, you will be able to explain:',

    '基本理解（Level 1）': 'Basic Understanding (Level 1)',
    '積層造形（AM）の定義とISO/ASTM 52900規格の基本概念': 'Definition of Additive Manufacturing (AM) and basic concepts of ISO/ASTM 52900 standard',
    '7つのAMプロセスカテゴリ（MEX, VPP, PBF, MJ, BJ, SL, DED）の特徴': 'Characteristics of 7 AM process categories (MEX, VPP, PBF, MJ, BJ, SL, DED)',
    'STLファイル形式の構造（三角形メッシュ、法線ベクトル、頂点順序）': 'Structure of STL file format (triangle mesh, normal vectors, vertex order)',
    'AMの歴史（1986年ステレオリソグラフィから現代システムまで）': 'History of AM (from 1986 stereolithography to modern systems)',

    '実践スキル（Level 2）': 'Practical Skills (Level 2)',
    'PythonでSTLファイルを読み込み、体積・表面積を計算できる': 'Read STL files in Python and calculate volume/surface area',
    'numpy-stlとtrimeshを使ったメッシュ検証と修復ができる': 'Perform mesh verification and repair using numpy-stl and trimesh',
    'スライシングの基本原理（レイヤー高さ、シェル、インフィル）を理解': 'Understand basic slicing principles (layer height, shell, infill)',
    'G-codeの基本構造（G0/G1/G28/M104など）を読み解ける': 'Interpret basic G-code structure (G0/G1/G28/M104, etc.)',

    '応用力（Level 3）': 'Application Skills (Level 3)',
    '用途要求に応じて最適なAMプロセスを選択できる': 'Select optimal AM process according to application requirements',
    'メッシュの問題（非多様体、法線反転）を検出・修正できる': 'Detect and fix mesh problems (non-manifold, inverted normals)',
    '造形パラメータ（レイヤー高さ、印刷速度、温度）を最適化できる': 'Optimize build parameters (layer height, print speed, temperature)',
    'STLファイルの品質評価とプリント適性判断ができる': 'Evaluate STL file quality and assess printability',

    # Section 1.1
    '1.1 積層造形（AM）とは': '1.1 What is Additive Manufacturing (AM)?',
    '1.1.1 積層造形の定義': '1.1.1 Definition of Additive Manufacturing',
    '積層造形（Additive Manufacturing, AM）とは、': 'Additive Manufacturing (AM) is ',
    'ISO/ASTM 52900:2021規格で定義される「3次元CADデータから材料を層ごとに積み上げて物体を製造するプロセス」': 'defined in the ISO/ASTM 52900:2021 standard as "a process of joining materials to make parts from 3D model data, usually layer upon layer"',
    'です。従来の切削加工（除去加工）とは対照的に、必要な部分にのみ材料を付加するため、以下の革新的な特徴を持ちます：': '. In contrast to traditional subtractive manufacturing (machining), it adds material only where needed, offering the following innovative characteristics:',

    '設計自由度': 'Design Freedom',
    '従来製法では不可能な複雑形状（中空構造、ラティス構造、トポロジー最適化形状）を製造可能': 'Enables manufacturing of complex geometries impossible with conventional methods (hollow structures, lattice structures, topology-optimized shapes)',
    '材料効率': 'Material Efficiency',
    '必要な部分にのみ材料を使用するため、材料廃棄率が5-10%（従来加工は30-90%廃棄）': 'Uses material only where needed, with waste rates of 5-10% (conventional machining: 30-90% waste)',
    'オンデマンド製造': 'On-Demand Manufacturing',
    '金型不要でカスタマイズ製品を少量・多品種生産可能': 'Enables low-volume, high-variety customized production without tooling',
    '一体化製造': 'Integrated Manufacturing',
    '従来は複数部品を組立てていた構造を一体造形し、組立工程を削減': 'Consolidates multi-part assemblies into single structures, reducing assembly steps',

    '💡 産業的重要性': '💡 Industrial Significance',
    'AM市場は急成長中で、Wohlers Report 2023によると：': 'The AM market is rapidly growing. According to Wohlers Report 2023:',
    '世界のAM市場規模: $18.3B（2023年）→ $83.9B予測（2030年、年成長率23.5%）': 'Global AM market size: $18.3B (2023) → $83.9B projected (2030, CAGR 23.5%)',
    '用途の内訳: プロトタイピング（38%）、ツーリング（27%）、最終製品（35%）': 'Application breakdown: Prototyping (38%), Tooling (27%), End-use parts (35%)',
    '主要産業: 航空宇宙（26%）、医療（21%）、自動車（18%）、消費財（15%）': 'Key industries: Aerospace (26%), Medical (21%), Automotive (18%), Consumer goods (15%)',
    '材料別シェア: ポリマー（55%）、金属（35%）、セラミックス（7%）、その他（3%）': 'Material breakdown: Polymers (55%), Metals (35%), Ceramics (7%), Other (3%)',

    # 1.1.2
    '1.1.2 AMの歴史と発展': '1.1.2 History and Evolution of AM',
    '積層造形技術は約40年の歴史を持ち、以下のマイルストーンを経て現在に至ります：': 'Additive manufacturing technology has approximately 40 years of history, reaching the present through the following milestones:',

    'SLA発明': 'SLA Invention',
    'Chuck Hull': 'Chuck Hull',
    'SLS登場': 'SLS Introduction',
    'Carl Deckard': 'Carl Deckard',
    'FDM特許': 'FDM Patent',
    'Stratasys社': 'Stratasys',
    'オープンソース化': 'Open Source',
    '金属AM普及': 'Metal AM Adoption',
    '産業化加速': 'Industrial Acceleration',
    '大型・高速化': 'Large-scale, High-speed',

    '1986年: ステレオリソグラフィ（SLA）発明': '1986: Stereolithography (SLA) Invention',
    'Chuck Hull博士（3D Systems社創業者）が光硬化樹脂を層状に硬化させる最初のAM技術を発明（US Patent 4,575,330）。「3Dプリンティング」という言葉もこの時期に誕生。': 'Dr. Chuck Hull (founder of 3D Systems) invented the first AM technology that cured photopolymer resin layer by layer (US Patent 4,575,330). The term "3D printing" was also coined during this period.',

    '1988年: 選択的レーザー焼結（SLS）登場': '1988: Selective Laser Sintering (SLS) Introduction',
    'Carl Deckard博士（テキサス大学）がレーザーで粉末材料を焼結する技術を開発。金属やセラミックスへの応用可能性を開く。': 'Dr. Carl Deckard (University of Texas) developed technology to sinter powder materials with laser. Opened possibilities for metal and ceramic applications.',

    '1992年: 熱溶解積層（FDM）特許': '1992: Fused Deposition Modeling (FDM) Patent',
    'Stratasys社がFDM技術を商用化。現在最も普及している3Dプリンティング方式の基礎を確立。': 'Stratasys commercialized FDM technology. Established the foundation for the most widely adopted 3D printing method today.',

    '2005年: RepRapプロジェクト': '2005: RepRap Project',
    'Adrian Bowyer教授がオープンソース3Dプリンタ「RepRap」を発表。特許切れと相まって低価格化・民主化が進展。': 'Professor Adrian Bowyer introduced the open-source 3D printer "RepRap". Combined with expiring patents, drove cost reduction and democratization.',

    '2012年以降: 金属AMの産業普及': '2012 Onwards: Industrial Adoption of Metal AM',
    '電子ビーム溶解（EBM）、選択的レーザー溶融（SLM）が航空宇宙・医療分野で実用化。GE AviationがFUEL噴射ノズルを量産開始。': 'Electron Beam Melting (EBM) and Selective Laser Melting (SLM) became practical in aerospace and medical fields. GE Aviation began mass production of fuel injection nozzles.',

    '2023年現在: 大型化・高速化の時代': '2023 Present: Era of Large-scale and High-speed',
    'バインダージェット、連続繊維複合材AM、マルチマテリアルAMなど新技術が産業実装段階へ。': 'New technologies such as binder jetting, continuous fiber composite AM, and multi-material AM enter industrial implementation stage.',

    # 1.1.3
    '1.1.3 AMの主要応用分野': '1.1.3 Major Application Areas of AM',

    '応用1: プロトタイピング（Rapid Prototyping）': 'Application 1: Prototyping (Rapid Prototyping)',
    'AMの最初の主要用途で、設計検証・機能試験・市場評価用のプロトタイプを迅速に製造します：': 'The first major application of AM, for rapid manufacturing of prototypes for design verification, functional testing, and market evaluation:',

    'リードタイム短縮': 'Lead Time Reduction',
    '従来の試作（数週間〜数ヶ月）→ AMでは数時間〜数日': 'Conventional prototyping (weeks to months) → AM: hours to days',
    '設計反復の加速': 'Accelerated Design Iteration',
    '低コストで複数バージョンを試作し、設計を最適化': 'Produce multiple versions at low cost to optimize design',
    'コミュニケーション改善': 'Improved Communication',
    '視覚的・触覚的な物理モデルで関係者間の認識を統一': 'Unify understanding among stakeholders with visual and tactile physical models',
    '典型例': 'Typical Examples',
    '自動車の意匠モデル、家電製品の筐体試作、医療機器の術前シミュレーションモデル': 'Automotive design models, consumer electronics case prototypes, medical device pre-surgical simulation models',

    '応用2: ツーリング（Tooling & Fixtures）': 'Application 2: Tooling (Tooling & Fixtures)',
    '製造現場で使用する治具・工具・金型をAMで製造する応用です：': 'Manufacturing of fixtures, tools, and molds used in production facilities with AM:',

    'カスタム治具': 'Custom Fixtures',
    '生産ラインに特化した組立治具・検査治具を迅速に製作': 'Rapidly produce assembly and inspection fixtures specialized for production lines',
    'コンフォーマル冷却金型': 'Conformal Cooling Molds',
    '従来の直線的冷却路ではなく、製品形状に沿った3次元冷却路を内蔵した射出成形金型（冷却時間30-70%短縮）': 'Injection molds with 3D cooling channels conforming to product shape instead of conventional straight channels (30-70% cooling time reduction)',
    '軽量化ツール': 'Lightweight Tools',
    'ラティス構造を使った軽量エンドエフェクタで作業者の負担を軽減': 'Reduce worker burden with lightweight end effectors using lattice structures',
    'BMWの組立ライン用治具（年間100,000個以上をAMで製造）、GolfのTaylorMadeドライバー金型': 'BMW assembly line fixtures (over 100,000 produced annually with AM), Golf TaylorMade driver molds',

    '応用3: 最終製品（End-Use Parts）': 'Application 3: End-Use Parts',
    'AMで直接、最終製品を製造する応用が近年急増しています：': 'Direct manufacturing of end-use parts with AM has been rapidly increasing in recent years:',

    '航空宇宙部品': 'Aerospace Components',
    'GE Aviation LEAP燃料噴射ノズル（従来20部品→AM一体化、重量25%軽減、年間100,000個以上生産）': 'GE Aviation LEAP fuel injection nozzles (consolidated from 20 parts to single AM part, 25% weight reduction, over 100,000 produced annually)',
    '医療インプラント': 'Medical Implants',
    'チタン製人工股関節・歯科インプラント（患者固有の解剖学的形状に最適化、骨結合を促進する多孔質構造）': 'Titanium hip implants and dental implants (optimized to patient-specific anatomy, porous structures promoting bone integration)',
    'カスタム製品': 'Custom Products',
    '補聴器（年間1,000万個以上がAMで製造）、スポーツシューズのミッドソール（Adidas 4D、Carbon社DLS技術）': 'Hearing aids (over 10 million produced annually with AM), sports shoe midsoles (Adidas 4D, Carbon DLS technology)',
    'スペア部品': 'Spare Parts',
    '絶版部品・希少部品のオンデマンド製造（自動車、航空機、産業機械）': 'On-demand manufacturing of discontinued and rare parts (automotive, aircraft, industrial machinery)',

    '⚠️ AMの制約と課題': '⚠️ Constraints and Challenges of AM',
    'AMは万能ではなく、以下の制約があります：': 'AM is not omnipotent and has the following constraints:',

    '造形速度': 'Build Speed',
    '大量生産には不向き（射出成形1個/数秒 vs AM数時間）。経済的ブレークイーブンは通常1,000個以下': 'Not suitable for mass production (injection molding: 1 part/seconds vs AM: hours). Economic break-even typically below 1,000 parts',
    '造形サイズ制限': 'Build Size Limitations',
    'ビルドボリューム（多くの装置で200×200×200mm程度）を超える大型部品は分割製造が必要': 'Large parts exceeding build volume (typically ~200×200×200mm for many machines) require segmented manufacturing',
    '表面品質': 'Surface Quality',
    '積層痕（layer lines）が残るため、高精度表面が必要な場合は後加工必須（研磨、機械加工）': 'Layer lines remain, requiring post-processing (polishing, machining) for high-precision surfaces',
    '材料特性の異方性': 'Material Property Anisotropy',
    '積層方向（Z軸）と面内方向（XY平面）で機械的性質が異なる場合がある（特にFDM）': 'Mechanical properties may differ between build direction (Z-axis) and in-plane direction (XY plane), especially in FDM',
    '材料コスト': 'Material Cost',
    'AMグレード材料は汎用材料の2-10倍高価（ただし材料効率と設計最適化で相殺可能）': 'AM-grade materials are 2-10x more expensive than general-purpose materials (though offset by material efficiency and design optimization)',

    # Section 1.2
    '1.2 ISO/ASTM 52900による7つのAMプロセス分類': '1.2 Seven AM Process Categories by ISO/ASTM 52900',
    '1.2.1 AMプロセス分類の全体像': '1.2.1 Overview of AM Process Classification',
    'ISO/ASTM 52900:2021規格では、すべてのAM技術を': 'The ISO/ASTM 52900:2021 standard classifies all AM technologies into ',
    'エネルギー源と材料供給方法に基づいて7つのプロセスカテゴリ': 'seven process categories based on energy source and material delivery method',
    'に分類しています。各プロセスには固有の長所・短所があり、用途に応じて最適な技術を選択する必要があります。': '. Each process has unique advantages and disadvantages, requiring selection of optimal technology according to application.',

    '積層造形': 'Additive Manufacturing',
    '7つのプロセス': '7 Processes',
    'Material Extrusion': 'Material Extrusion',
    '材料押出': 'Material Extrusion',
    'Vat Photopolymerization': 'Vat Photopolymerization',
    '液槽光重合': 'Vat Photopolymerization',
    'Powder Bed Fusion': 'Powder Bed Fusion',
    '粉末床溶融結合': 'Powder Bed Fusion',
    'Material Jetting': 'Material Jetting',
    '材料噴射': 'Material Jetting',
    'Binder Jetting': 'Binder Jetting',
    '結合剤噴射': 'Binder Jetting',
    'Sheet Lamination': 'Sheet Lamination',
    'シート積層': 'Sheet Lamination',
    'Directed Energy Deposition': 'Directed Energy Deposition',
    '指向性エネルギー堆積': 'Directed Energy Deposition',

    '低コスト・普及型': 'Low-cost, Widespread',
    '高精度・高表面品質': 'High-precision, High surface quality',
    '高強度・金属対応': 'High-strength, Metal-capable',

    # 1.2.2
    '1.2.2 Material Extrusion (MEX) - 材料押出': '1.2.2 Material Extrusion (MEX)',
    '原理': 'Principle',
    '熱可塑性樹脂フィラメントを加熱・溶融し、ノズルから押し出して積層。最も普及している技術（FDM/FFFとも呼ばれる）。': 'Thermoplastic filament is heated, melted, and extruded through a nozzle to build layers. Most widely adopted technology (also called FDM/FFF).',

    'プロセス: フィラメント → 加熱ノズル（190-260°C）→ 溶融押出 → 冷却固化 → 次層積層': 'Process: Filament → Heated nozzle (190-260°C) → Melt extrusion → Cooling solidification → Next layer',

    '特徴：': 'Characteristics:',
    '低コスト': 'Low Cost',
    '装置価格$200-$5,000（デスクトップ）、$10,000-$100,000（産業用）': 'Equipment price $200-$5,000 (desktop), $10,000-$100,000 (industrial)',
    '材料多様性': 'Material Variety',
    'PLA、ABS、PETG、ナイロン、PC、カーボン繊維複合材、PEEK（高性能）': 'PLA, ABS, PETG, Nylon, PC, Carbon fiber composites, PEEK (high-performance)',
    '造形速度': 'Build Speed',
    '20-150 mm³/s（中程度）、レイヤー高さ0.1-0.4mm': '20-150 mm³/s (moderate), layer height 0.1-0.4mm',
    '精度': 'Accuracy',
    '±0.2-0.5 mm（デスクトップ）、±0.1 mm（産業用）': '±0.2-0.5 mm (desktop), ±0.1 mm (industrial)',
    '表面品質': 'Surface Quality',
    '積層痕が明瞭（後加工で改善可能）': 'Layer lines are visible (can be improved with post-processing)',
    '材料異方性': 'Material Anisotropy',
    'Z軸方向（積層方向）の強度が20-80%低い（層間接着が弱点）': 'Z-axis (build direction) strength is 20-80% lower (interlayer adhesion is weak point)',

    '応用例：': 'Application Examples:',
    'プロトタイピング（最も一般的な用途、低コスト・高速）': 'Prototyping (most common application, low-cost, fast)',
    '治具・工具（製造現場で使用、軽量・カスタマイズ容易）': 'Jigs and tools (used in manufacturing, lightweight, easy to customize)',
    '教育用モデル（学校・大学で広く使用、安全・低コスト）': 'Educational models (widely used in schools and universities, safe, low-cost)',
    '最終製品（カスタム補聴器、義肢装具、建築模型）': 'End-use parts (custom hearing aids, prosthetics, architectural models)',

    '💡 FDMの代表的装置': '💡 Representative FDM Equipment',
    'デュアルヘッド、ビルドボリューム330×240×300mm、$6,000': 'Dual head, build volume 330×240×300mm, $6,000',
    'オープンソース系、高い信頼性、$1,200': 'Open-source based, high reliability, $1,200',
    '産業用、ULTEM 9085対応、$250,000': 'Industrial, ULTEM 9085 compatible, $250,000',
    '連続カーボン繊維複合材対応、$100,000': 'Continuous carbon fiber composite compatible, $100,000',

    # 1.2.3
    '1.2.3 Vat Photopolymerization (VPP) - 液槽光重合': '1.2.3 Vat Photopolymerization (VPP)',
    '液状の光硬化性樹脂（フォトポリマー）に紫外線（UV）レーザーまたはプロジェクターで光を照射し、選択的に硬化させて積層。': 'UV laser or projector selectively cures liquid photopolymer resin by irradiation, building layers.',

    'プロセス: UV照射 → 光重合反応 → 固化 → ビルドプラットフォーム上昇 → 次層照射': 'Process: UV irradiation → Photopolymerization → Solidification → Build platform rises → Next layer irradiation',

    'VPPの2つの主要方式：': 'Two major VPP methods:',
    'UV レーザー（355 nm）をガルバノミラーで走査し、点描的に硬化。高精度だが低速。': 'UV laser (355 nm) scanned by galvanometer mirror, point-by-point curing. High precision but slow.',
    'プロジェクターで面全体を一括露光。高速だが解像度はプロジェクター画素数に依存（Full HD: 1920×1080）。': 'Entire layer exposed at once with projector. Fast but resolution depends on projector pixels (Full HD: 1920×1080).',
    'LCDマスクを使用、DLP類似だが低コスト化（$200-$1,000のデスクトップ機多数）。': 'Uses LCD mask, similar to DLP but lower cost ($200-$1,000 desktop machines available).',

    '高精度': 'High Precision',
    'XY解像度25-100 μm、Z解像度10-50 μm（全AM技術中で最高レベル）': 'XY resolution 25-100 μm, Z resolution 10-50 μm (highest among all AM technologies)',
    '滑らかな表面（Ra < 5 μm）、積層痕がほぼ見えない': 'Smooth surface (Ra < 5 μm), layer lines nearly invisible',
    'SLA（10-50 mm³/s）、DLP/LCD（100-500 mm³/s、面積依存）': 'SLA (10-50 mm³/s), DLP/LCD (100-500 mm³/s, area-dependent)',
    '材料制約': 'Material Constraints',
    '光硬化性樹脂のみ（機械的性質はFDMより劣る場合が多い）': 'Photopolymer resins only (mechanical properties often inferior to FDM)',
    '後処理必須': 'Post-processing Required',
    '洗浄（IPA等）→ 二次硬化（UV照射）→ サポート除去': 'Washing (IPA etc.) → Post-curing (UV exposure) → Support removal',

    '歯科用途（歯列矯正モデル、サージカルガイド、義歯、年間数百万個生産）': 'Dental applications (orthodontic models, surgical guides, dentures, millions produced annually)',
    'ジュエリー鋳造用ワックスモデル（高精度・複雑形状）': 'Jewelry casting wax models (high precision, complex shapes)',
    '医療モデル（術前計画、解剖学モデル、患者説明用）': 'Medical models (preoperative planning, anatomical models, patient explanation)',
    'マスターモデル（シリコン型取り用、デザイン検証）': 'Master models (silicone molding, design verification)',

    # 1.2.4
    '1.2.4 Powder Bed Fusion (PBF) - 粉末床溶融結合': '1.2.4 Powder Bed Fusion (PBF)',
    '粉末材料を薄く敷き詰め、レーザーまたは電子ビームで選択的に溶融・焼結し、冷却固化させて積層。金属・ポリマー・セラミックスに対応。': 'Thin layer of powder material is spread, selectively melted/sintered by laser or electron beam, then cooled to solidify. Compatible with metals, polymers, ceramics.',

    'プロセス: 粉末敷設 → レーザー/電子ビーム走査 → 溶融・焼結 → 固化 → 次層粉末敷設': 'Process: Powder spreading → Laser/electron beam scanning → Melting/sintering → Solidification → Next layer powder spreading',

    'PBFの3つの主要方式：': 'Three major PBF methods:',
    'ポリマー粉末（PA12ナイロン等）をレーザー焼結。サポート不要（周囲粉末が支持）。': 'Laser sintering of polymer powder (PA12 nylon etc.). No support needed (surrounding powder provides support).',
    '金属粉末（Ti-6Al-4V、AlSi10Mg、Inconel 718等）を完全溶融。高密度部品（相対密度>99%）製造可能。': 'Complete melting of metal powder (Ti-6Al-4V, AlSi10Mg, Inconel 718 etc.). High-density parts (relative density >99%) achievable.',
    '電子ビームで金属粉末を溶融。高温予熱（650-1000°C）により残留応力が小さく、造形速度が速い。': 'Metal powder melted by electron beam. Low residual stress due to high-temperature preheating (650-1000°C), fast build speed.',

    '高強度': 'High Strength',
    '溶融・再凝固により鍛造材に匹敵する機械的性質（引張強度500-1200 MPa）': 'Mechanical properties comparable to wrought materials through melting and re-solidification (tensile strength 500-1200 MPa)',
    '複雑形状対応': 'Complex Geometry Capability',
    'サポート不要（粉末が支持）でオーバーハング造形可能': 'Overhang building possible without support (powder provides support)',
    'Ti合金、Al合金、ステンレス鋼、Ni超合金、Co-Cr合金、ナイロン': 'Ti alloys, Al alloys, stainless steel, Ni superalloys, Co-Cr alloys, nylon',
    '高コスト': 'High Cost',
    '装置価格$200,000-$1,500,000、材料費$50-$500/kg': 'Equipment price $200,000-$1,500,000, material cost $50-$500/kg',
    '後処理': 'Post-processing',
    'サポート除去、熱処理（応力除去）、表面仕上げ（ブラスト、研磨）': 'Support removal, heat treatment (stress relief), surface finishing (blasting, polishing)',

    '航空宇宙部品（軽量化、一体化、GE LEAP燃料ノズル等）': 'Aerospace components (weight reduction, consolidation, GE LEAP fuel nozzles etc.)',
    '医療インプラント（患者固有形状、多孔質構造、Ti-6Al-4V）': 'Medical implants (patient-specific geometry, porous structures, Ti-6Al-4V)',
    '金型（コンフォーマル冷却、複雑形状、H13工具鋼）': 'Molds (conformal cooling, complex geometry, H13 tool steel)',
    '自動車部品（軽量化ブラケット、カスタムエンジン部品）': 'Automotive parts (lightweight brackets, custom engine components)',

    # 1.2.5
    '1.2.5 Material Jetting (MJ) - 材料噴射': '1.2.5 Material Jetting (MJ)',
    'インクジェットプリンタと同様に、液滴状の材料（光硬化性樹脂またはワックス）をヘッドから噴射し、UV照射で即座に硬化させて積層。': 'Similar to inkjet printers, droplets of material (photopolymer resin or wax) are jetted from heads and instantly cured by UV exposure to build layers.',

    '超高精度': 'Ultra-high Precision',
    'XY解像度42-85 μm、Z解像度16-32 μm': 'XY resolution 42-85 μm, Z resolution 16-32 μm',
    'マルチマテリアル': 'Multi-material',
    '同一造形で複数材料・複数色を使い分け可能': 'Multiple materials and colors can be used in single build',
    'フルカラー造形': 'Full-color Printing',
    'CMYK樹脂の組合せで1,000万色以上の表現': 'Over 10 million colors achievable with CMYK resin combinations',
    '極めて滑らか（積層痕ほぼなし）': 'Extremely smooth (virtually no layer lines)',
    '装置$50,000-$300,000、材料費$200-$600/kg': 'Equipment $50,000-$300,000, material cost $200-$600/kg',
    '光硬化性樹脂のみ、機械的性質は中程度': 'Photopolymer resins only, moderate mechanical properties',

    '医療解剖モデル（軟組織・硬組織を異なる材料で再現）、フルカラー建築模型、デザイン検証モデル': 'Medical anatomical models (soft/hard tissues reproduced with different materials), full-color architectural models, design verification models',

    # 1.2.6
    '1.2.6 Binder Jetting (BJ) - 結合剤噴射': '1.2.6 Binder Jetting (BJ)',
    '粉末床に液状バインダー（接着剤）をインクジェット方式で噴射し、粉末粒子を結合。造形後に焼結または含浸処理で強度向上。': 'Liquid binder (adhesive) is jetted inkjet-style onto powder bed to bind powder particles. Strength improved by sintering or infiltration after building.',

    '高速造形': 'High-speed Building',
    'レーザー走査不要で面全体を一括処理、造形速度100-500 mm³/s': 'No laser scanning required, entire layer processed at once, build speed 100-500 mm³/s',
    '金属粉末、セラミックス、砂型（鋳造用）、フルカラー（石膏）': 'Metal powder, ceramics, sand molds (for casting), full-color (gypsum)',
    '周囲粉末が支持、除去後リサイクル可能': 'Surrounding powder provides support, recyclable after removal',
    '低密度問題': 'Low Density Issue',
    '焼結前は脆弱（グリーン密度50-60%）、焼結後も相対密度90-98%': 'Fragile before sintering (green density 50-60%), relative density 90-98% after sintering',
    '脱脂 → 焼結（金属：1200-1400°C）→ 含浸（銅・青銅）': 'Debinding → Sintering (metal: 1200-1400°C) → Infiltration (copper/bronze)',

    '砂型鋳造用型（エンジンブロック等の大型鋳物）、金属部品（Desktop Metal、HP Metal Jet）、フルカラー像（記念品、教育モデル）': 'Sand molds for casting (large castings like engine blocks), metal parts (Desktop Metal, HP Metal Jet), full-color figures (memorabilia, educational models)',

    # 1.2.7
    '1.2.7 Sheet Lamination (SL) - シート積層': '1.2.7 Sheet Lamination (SL)',
    'シート状材料（紙、金属箔、プラスチックフィルム）を積層し、接着または溶接で結合。各層をレーザーまたはブレードで輪郭切断。': 'Sheet materials (paper, metal foil, plastic film) are laminated and bonded by adhesive or welding. Each layer is contour-cut by laser or blade.',

    '代表技術：': 'Representative Technologies:',
    '紙・プラスチックシート、接着剤で積層、レーザー切断': 'Paper/plastic sheets, laminated with adhesive, laser cutting',
    '金属箔を超音波溶接、CNC切削で輪郭加工': 'Metal foil ultrasonic welding, CNC milling for contouring',

    '大型造形可能、材料費安価、精度中程度、用途限定的（主に視覚モデル、金属では埋込センサー等）': 'Large-scale building possible, low material cost, moderate accuracy, limited applications (mainly visual models, embedded sensors in metal)',

    # 1.2.8
    '1.2.8 Directed Energy Deposition (DED) - 指向性エネルギー堆積': '1.2.8 Directed Energy Deposition (DED)',
    '金属粉末またはワイヤーを供給しながら、レーザー・電子ビーム・アークで溶融し、基板上に堆積。大型部品や既存部品の補修に使用。': 'Metal powder or wire is fed while being melted by laser, electron beam, or arc, deposited on substrate. Used for large parts or repair of existing parts.',

    '高速堆積': 'High Deposition Rate',
    '堆積速度1-5 kg/h（PBFの10-50倍）': 'Deposition rate 1-5 kg/h (10-50x faster than PBF)',
    '大型対応': 'Large-scale Capability',
    'ビルドボリューム制限が少ない（多軸ロボットアーム使用）': 'Few build volume limitations (multi-axis robot arms used)',
    '補修・コーティング': 'Repair and Coating',
    '既存部品の摩耗部分修復、表面硬化層形成': 'Repair worn parts, form surface hardening layers',
    '低精度': 'Low Precision',
    '精度±0.5-2 mm、後加工（機械加工）必須': 'Accuracy ±0.5-2 mm, post-processing (machining) required',

    'タービンブレード補修、大型航空宇宙部品、工具の耐摩耗コーティング': 'Turbine blade repair, large aerospace components, wear-resistant coatings for tools',

    '⚠️ プロセス選択の指針': '⚠️ Process Selection Guidelines',
    '最適なAMプロセスは用途要求により異なります：': 'Optimal AM process varies by application requirements:',

    '精度最優先': 'Precision Priority',
    'VPP（SLA/DLP）またはMJ': 'VPP (SLA/DLP) or MJ',
    '低コスト・普及型': 'Low-cost, Common',
    'MEX（FDM/FFF）': 'MEX (FDM/FFF)',
    '金属高強度部品': 'Metal High-strength Parts',
    'PBF（SLM/EBM）': 'PBF (SLM/EBM)',
    '大量生産（砂型）': 'Mass Production (Sand Molds)',
    'BJ': 'BJ',
    '大型・高速堆積': 'Large-scale, High-speed Deposition',
    'DED': 'DED',
}

# Apply translations
for jp, en in translations.items():
    content = content.replace(jp, en)

print(f"Translation complete. Writing to target file...")
print(f"Total replacements: {len(translations)}")
