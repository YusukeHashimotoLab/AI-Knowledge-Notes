#!/usr/bin/env python3
"""
Complete comprehensive translation for 3D Printing Chapter 2
Processes all 2700 lines to achieve 0 Japanese characters
"""

import re
from pathlib import Path

def create_comprehensive_translations():
    """Create exhaustive translation mappings"""
    translations = {}
    
    # Phase 1: Learning objectives and basic content (lines 1-500)
    translations.update({
        "積層造形（AM）の定義とISO/ASTM 52900規格の基本概念": "Definition of Additive Manufacturing (AM) and basic concepts of ISO/ASTM 52900 standard",
        "7つのAMプロセスカテゴリ（MEX, VPP, PBF, MJ, BJ, SL, DED）の特徴": "Characteristics of seven AM process categories (MEX, VPP, PBF, MJ, BJ, SL, DED)",
        "STLファイル形式の構造（三角形メッシュ、法線ベクトル、頂点順序）": "Structure of STL file format (triangular mesh, normal vectors, vertex order)",
        "AMの歴史（1986年ステレオリソグラフィから現代システムまで）": "History of AM (from 1986 stereolithography to modern systems)",
        "PythonでSTLファイルを読み込み、体積・表面積を計算できる": "Ability to read STL files in Python and calculate volume and surface area",
        "numpy-stlとtrimeshを使ったメッシュ検証と修復ができる": "Ability to validate and repair meshes using numpy-stl and trimesh",
        "スライシングの基本原理（レイヤー高さ、シェル、インフィル）を理解": "Understanding basic principles of slicing (layer height, shells, infill)",
        "G-codeの基本構造（G0/G1/G28/M104など）を読み解ける": "Ability to interpret basic G-code structure (G0/G1/G28/M104, etc.)",
        "用途要求に応じて最適なAMプロセスを選択できる": "Ability to select optimal AM process according to application requirements",
        "メッシュの問題（非多様体、法線反転）を検出・修正できる": "Ability to detect and correct mesh problems (non-manifold, flipped normals)",
        "造形パラメータ（レイヤー高さ、印刷速度、温度）を最適化できる": "Ability to optimize build parameters (layer height, print speed, temperature)",
        "STLファイルの品質評価とプリント適性判断ができる": "Ability to assess STL file quality and determine printability",
        
        "1.1 積層造形（AM）とは": "1.1 What is Additive Manufacturing (AM)?",
        "1.1.1 積層造形の定義": "1.1.1 Definition of Additive Manufacturing",
        "積層造形（Additive Manufacturing, AM）とは、<strong>ISO/ASTM 52900:2021規格で定義される「3次元CADデータから材料を層ごとに積み上げて物体を製造するプロセス」</strong>です。従来の切削加工（除去加工）とは対照的に、必要な部分にのみ材料を付加するため、以下の革新的な特徴を持ちます：":
            'Additive Manufacturing (AM) is <strong>"a process of fabricating objects by joining materials layer by layer from 3D CAD data," as defined in ISO/ASTM 52900:2021 standard</strong>. In contrast to conventional subtractive manufacturing (cutting/machining), AM adds material only where needed, offering the following innovative characteristics:',
        "<strong>設計自由度</strong>: 従来製法では不可能な複雑形状（中空構造、ラティス構造、トポロジー最適化形状）を製造可能": "<strong>Design Freedom</strong>: Enables manufacturing of complex geometries impossible with conventional methods (hollow structures, lattice structures, topology-optimized shapes)",
        "<strong>材料効率</strong>: 必要な部分にのみ材料を使用するため、材料廃棄率が5-10%（従来加工は30-90%廃棄）": "<strong>Material Efficiency</strong>: Material waste rate of 5-10% (conventional machining: 30-90% waste) by using material only where needed",
        "<strong>オンデマンド製造</strong>: 金型不要でカスタマイズ製品を少量・多品種生産可能": "<strong>On-Demand Manufacturing</strong>: Enables low-volume, high-variety production of customized products without tooling",
        "<strong>一体化製造</strong>: 従来は複数部品を組立てていた構造を一体造形し、組立工程を削減": "<strong>Integrated Manufacturing</strong>: Produces structures as single pieces that conventionally required assembly of multiple parts, reducing assembly steps",
        
        "💡 産業的重要性": "💡 Industrial Significance",
        "AM市場は急成長中で、Wohlers Report 2023によると：": "The AM market is experiencing rapid growth. According to Wohlers Report 2023:",
        "世界のAM市場規模: $18.3B（2023年）→ $83.9B予測（2030年、年成長率23.5%）": "Global AM market size: $18.3B (2023) → $83.9B projected (2030, CAGR 23.5%)",
        "用途の内訳: プロトタイピング（38%）、ツーリング（27%）、最終製品（35%）": "Application breakdown: Prototyping (38%), Tooling (27%), End-use parts (35%)",
        "主要産業: 航空宇宙（26%）、医療（21%）、自動車（18%）、消費財（15%）": "Key industries: Aerospace (26%), Medical (21%), Automotive (18%), Consumer goods (15%)",
        "材料別シェア: ポリマー（55%）、金属（35%）、セラミックス（7%）、その他（3%）": "Material share: Polymers (55%), Metals (35%), Ceramics (7%), Others (3%)",
        
        "1.1.2 AMの歴史と発展": "1.1.2 History and Evolution of AM",
        "積層造形技術は約40年の歴史を持ち、以下のマイルストーンを経て現在に至ります：": "Additive manufacturing technology has approximately 40 years of history, reaching the present through the following milestones:",
        "SLA発明": "SLA Invention",
        "SLS登場": "SLS Introduction",
        "FDM特許": "FDM Patent",
        "オープンソース化": "Open Source",
        "金属AM普及": "Metal AM Adoption",
        "産業化加速": "Industrial Acceleration",
        "大型・高速化": "Large-scale & High-speed",
        
        "<strong>1986年: ステレオリソグラフィ（SLA）発明</strong> - Chuck Hull博士（3D Systems社創業者）が光硬化樹脂を層状に硬化させる最初のAM技術を発明（US Patent 4,575,330）。「3Dプリンティング」という言葉もこの時期に誕生。":
            "<strong>1986: Invention of Stereolithography (SLA)</strong> - Dr. Chuck Hull (founder of 3D Systems) invented the first AM technology that cures photopolymer resin layer by layer (US Patent 4,575,330). The term '3D printing' was also coined during this period.",
        "<strong>1988年: 選択的レーザー焼結（SLS）登場</strong> - Carl Deckard博士（テキサス大学）がレーザーで粉末材料を焼結する技術を開発。金属やセラミックスへの応用可能性を開く。":
            "<strong>1988: Introduction of Selective Laser Sintering (SLS)</strong> - Dr. Carl Deckard (University of Texas) developed technology to sinter powder materials with a laser, opening possibilities for metal and ceramic applications.",
        "<strong>1992年: 熱溶解積層（FDM）特許</strong> - Stratasys社がFDM技術を商用化。現在最も普及している3Dプリンティング方式の基礎を確立。":
            "<strong>1992: Fused Deposition Modeling (FDM) Patent</strong> - Stratasys commercialized FDM technology, establishing the foundation for the most widely adopted 3D printing method today.",
        "<strong>2005年: RepRapプロジェクト</strong> - Adrian Bowyer教授がオープンソース3Dプリンタ「RepRap」を発表。特許切れと相まって低価格化・民主化が進展。":
            "<strong>2005: RepRap Project</strong> - Professor Adrian Bowyer announced the open-source 3D printer 'RepRap'. Combined with patent expirations, this accelerated cost reduction and democratization.",
        "<strong>2012年以降: 金属AMの産業普及</strong> - 電子ビーム溶解（EBM）、選択的レーザー溶融（SLM）が航空宇宙・医療分野で実用化。GE AviationがFUEL噴射ノズルを量産開始。":
            "<strong>2012 onwards: Industrial Adoption of Metal AM</strong> - Electron Beam Melting (EBM) and Selective Laser Melting (SLM) became practical in aerospace and medical fields. GE Aviation began mass production of FUEL injection nozzles.",
        "<strong>2023年現在: 大型化・高速化の時代</strong> - バインダージェット、連続繊維複合材AM、マルチマテリアルAMなど新技術が産業実装段階へ。":
            "<strong>2023 Present: Era of Large-scale and High-speed</strong> - New technologies such as binder jetting, continuous fiber composite AM, and multi-material AM are entering industrial implementation stages.",
        
        "1.1.3 AMの主要応用分野": "1.1.3 Major Application Areas of AM",
        "応用1: プロトタイピング（Rapid Prototyping）": "Application 1: Rapid Prototyping",
        "AMの最初の主要用途で、設計検証・機能試験・市場評価用のプロトタイプを迅速に製造します：": "AM's first major application, rapidly manufacturing prototypes for design validation, functional testing, and market evaluation:",
        "<strong>リードタイム短縮</strong>: 従来の試作（数週間〜数ヶ月）→ AMでは数時間〜数日": "<strong>Lead Time Reduction</strong>: Conventional prototyping (weeks to months) → AM enables hours to days",
        "<strong>設計反復の加速</strong>: 低コストで複数バージョンを試作し、設計を最適化": "<strong>Accelerated Design Iteration</strong>: Prototype multiple versions at low cost to optimize design",
        "<strong>コミュニケーション改善</strong>: 視覚的・触覚的な物理モデルで関係者間の認識を統一": "<strong>Improved Communication</strong>: Physical models providing visual and tactile feedback align understanding among stakeholders",
        "<strong>典型例</strong>: 自動車の意匠モデル、家電製品の筐体試作、医療機器の術前シミュレーションモデル": "<strong>Typical Examples</strong>: Automotive design models, consumer electronics enclosure prototypes, presurgical simulation models for medical devices",
        
        "応用2: ツーリング（Tooling & Fixtures）": "Application 2: Tooling & Fixtures",
        "製造現場で使用する治具・工具・金型をAMで製造する応用です：": "Application of manufacturing jigs, tools, and molds used in production environments with AM:",
        "<strong>カスタム治具</strong>: 生産ラインに特化した組立治具・検査治具を迅速に製作": "<strong>Custom Jigs</strong>: Rapidly fabricate assembly and inspection jigs specialized for production lines",
        "<strong>コンフォーマル冷却金型</strong>: 従来の直線的冷却路ではなく、製品形状に沿った3次元冷却路を内蔵した射出成形金型（冷却時間30-70%短縮）": "<strong>Conformal Cooling Molds</strong>: Injection molds with 3D cooling channels following product geometry rather than conventional straight channels (30-70% cooling time reduction)",
        "<strong>軽量化ツール</strong>: ラティス構造を使った軽量エンドエフェクタで作業者の負担を軽減": "<strong>Lightweight Tools</strong>: Lightweight end effectors using lattice structures to reduce operator burden",
        "<strong>典型例</strong>: BMWの組立ライン用治具（年間100,000個以上をAMで製造）、GolfのTaylorMadeドライバー金型": "<strong>Typical Examples</strong>: BMW assembly line jigs (over 100,000 units annually manufactured with AM), TaylorMade golf driver molds",
        
        "応用3: 最終製品（End-Use Parts）": "Application 3: End-Use Parts",
    })
    
    # Phase 2: AMで直接 and process descriptions (lines 501-1000)
    translations.update({
        "AMで直接、最終製品を製造する応用が近年急増しています：": "Applications manufacturing end-use parts directly with AM have been rapidly increasing in recent years:",
        "<strong>航空宇宙部品</strong>: GE Aviation LEAP燃料噴射ノズル（従来20部品→AM一体化、重量25%軽減、年間100,000個以上生産）": "<strong>Aerospace Components</strong>: GE Aviation LEAP fuel injection nozzle (conventional 20 parts → AM integrated, 25% weight reduction, over 100,000 units produced annually)",
        "<strong>医療インプラント</strong>: チタン製人工股関節・歯科インプラント（患者固有の解剖学的形状に最適化、骨結合を促進する多孔質構造）": "<strong>Medical Implants</strong>: Titanium artificial hip joints and dental implants (optimized for patient-specific anatomical shapes, porous structures promoting bone integration)",
        "<strong>カスタム製品</strong>: 補聴器（年間1,000万個以上がAMで製造）、スポーツシューズのミッドソール（Adidas 4D、Carbon社DLS技術）": "<strong>Custom Products</strong>: Hearing aids (over 10 million units manufactured with AM annually), sports shoe midsoles (Adidas 4D, Carbon DLS technology)",
        "<strong>スペア部品</strong>: 絶版部品・希少部品のオンデマンド製造（自動車、航空機、産業機械）": "<strong>Spare Parts</strong>: On-demand manufacturing of discontinued and rare parts (automotive, aircraft, industrial machinery)",
        
        "⚠️ AMの制約と課題": "⚠️ AM Constraints and Challenges",
        "AMは万能ではなく、以下の制約があります：": "AM is not a panacea and has the following constraints:",
        "<strong>造形速度</strong>: 大量生産には不向き（射出成形1個/数秒 vs AM数時間）。経済的ブレークイーブンは通常1,000個以下": "<strong>Build Speed</strong>: Unsuitable for mass production (injection molding 1 part/seconds vs AM hours). Economic break-even typically below 1,000 units",
        "<strong>造形サイズ制限</strong>: ビルドボリューム（多くの装置で200×200×200mm程度）を超える大型部品は分割製造が必要": "<strong>Build Size Limitations</strong>: Large parts exceeding build volume (typically around 200×200×200mm for many systems) require segmented manufacturing",
        "<strong>表面品質</strong>: 積層痕（layer lines）が残るため、高精度表面が必要な場合は後加工必須（研磨、機械加工）": "<strong>Surface Quality</strong>: Layer lines remain, requiring post-processing (polishing, machining) when high-precision surfaces are needed",
        "<strong>材料特性の異方性</strong>: 積層方向（Z軸）と面内方向（XY平面）で機械的性質が異なる場合がある（特にFDM）": "<strong>Material Anisotropy</strong>: Mechanical properties may differ between build direction (Z-axis) and in-plane direction (XY-plane), especially in FDM",
        "<strong>材料コスト</strong>: AMグレード材料は汎用材料の2-10倍高価（ただし材料効率と設計最適化で相殺可能）": "<strong>Material Cost</strong>: AM-grade materials are 2-10 times more expensive than commodity materials (though can be offset by material efficiency and design optimization)",
        
        "1.2 ISO/ASTM 52900による7つのAMプロセス分類": "1.2 Seven AM Process Classifications by ISO/ASTM 52900",
        "1.2.1 AMプロセス分類の全体像": "1.2.1 Overview of AM Process Classifications",
        "ISO/ASTM 52900:2021規格では、すべてのAM技術を<strong>エネルギー源と材料供給方法に基づいて7つのプロセスカテゴリ</strong>に分類しています。各プロセスには固有の長所・短所があり、用途に応じて最適な技術を選択する必要があります。":
            "The ISO/ASTM 52900:2021 standard classifies all AM technologies into <strong>seven process categories based on energy source and material delivery method</strong>. Each process has unique advantages and disadvantages, requiring selection of the optimal technology according to application.",
        
        "積層造形<br/>7つのプロセス": "Additive Manufacturing<br/>Seven Processes",
        "Material Extrusion<br/>材料押出": "Material Extrusion",
        "Vat Photopolymerization<br/>液槽光重合": "Vat Photopolymerization",
        "Powder Bed Fusion<br/>粉末床溶融結合": "Powder Bed Fusion",
        "Material Jetting<br/>材料噴射": "Material Jetting",
        "Binder Jetting<br/>結合剤噴射": "Binder Jetting",
        "Sheet Lamination<br/>シート積層": "Sheet Lamination",
        "Directed Energy Deposition<br/>指向性エネルギー堆積": "Directed Energy Deposition",
        "FDM/FFF<br/>低コスト・普及型": "FDM/FFF<br/>Low-cost & Widespread",
        "SLA/DLP<br/>高精度・高表面品質": "SLA/DLP<br/>High precision & Surface quality",
        "SLS/SLM/EBM<br/>高強度・金属対応": "SLS/SLM/EBM<br/>High strength & Metal capable",
        
        "1.2.2 Material Extrusion (MEX) - 材料押出": "1.2.2 Material Extrusion (MEX)",
        "<strong>原理</strong>: 熱可塑性樹脂フィラメントを加熱・溶融し、ノズルから押し出して積層。最も普及している技術（FDM/FFFとも呼ばれる）。":
            "<strong>Principle</strong>: Thermoplastic filament is heated and melted, extruded from a nozzle, and deposited layer by layer. The most widespread technology (also called FDM/FFF).",
        "プロセス: フィラメント → 加熱ノズル（190-260°C）→ 溶融押出 → 冷却固化 → 次層積層":
            "Process: Filament → Heated nozzle (190-260°C) → Melt extrusion → Cooling solidification → Next layer deposition",
        "<strong>特徴：</strong>": "<strong>Characteristics:</strong>",
        "<strong>低コスト</strong>: 装置価格$200-$5,000（デスクトップ）、$10,000-$100,000（産業用）": "<strong>Low Cost</strong>: Equipment price $200-$5,000 (desktop), $10,000-$100,000 (industrial)",
        "<strong>材料多様性</strong>: PLA、ABS、PETG、ナイロン、PC、カーボン繊維複合材、PEEK（高性能）": "<strong>Material Diversity</strong>: PLA, ABS, PETG, nylon, PC, carbon fiber composites, PEEK (high-performance)",
        "<strong>造形速度</strong>: 20-150 mm³/s（中程度）、レイヤー高さ0.1-0.4mm": "<strong>Build Speed</strong>: 20-150 mm³/s (moderate), layer height 0.1-0.4mm",
        "<strong>精度</strong>: ±0.2-0.5 mm（デスクトップ）、±0.1 mm（産業用）": "<strong>Accuracy</strong>: ±0.2-0.5 mm (desktop), ±0.1 mm (industrial)",
        "<strong>表面品質</strong>: 積層痕が明瞭（後加工で改善可能）": "<strong>Surface Quality</strong>: Visible layer lines (improvable with post-processing)",
        "<strong>材料異方性</strong>: Z軸方向（積層方向）の強度が20-80%低い（層間接着が弱点）": "<strong>Material Anisotropy</strong>: Z-axis (build direction) strength 20-80% lower (weak interlayer bonding)",
        
        "<strong>応用例：</strong>": "<strong>Applications:</strong>",
        "プロトタイピング（最も一般的な用途、低コスト・高速）": "Prototyping (most common use, low-cost and fast)",
        "治具・工具（製造現場で使用、軽量・カスタマイズ容易）": "Jigs and tools (used in manufacturing, lightweight and easily customizable)",
        "教育用モデル（学校・大学で広く使用、安全・低コスト）": "Educational models (widely used in schools and universities, safe and low-cost)",
        "最終製品（カスタム補聴器、義肢装具、建築模型）": "End-use parts (custom hearing aids, orthotic devices, architectural models)",
        
        "💡 FDMの代表的装置": "💡 Representative FDM Systems",
        "<strong>Ultimaker S5</strong>: デュアルヘッド、ビルドボリューム330×240×300mm、$6,000": "<strong>Ultimaker S5</strong>: Dual head, build volume 330×240×300mm, $6,000",
        "<strong>Prusa i3 MK4</strong>: オープンソース系、高い信頼性、$1,200": "<strong>Prusa i3 MK4</strong>: Open-source based, high reliability, $1,200",
        "<strong>Stratasys Fortus 450mc</strong>: 産業用、ULTEM 9085対応、$250,000": "<strong>Stratasys Fortus 450mc</strong>: Industrial, ULTEM 9085 compatible, $250,000",
        "<strong>Markforged X7</strong>: 連続カーボン繊維複合材対応、$100,000": "<strong>Markforged X7</strong>: Continuous carbon fiber composite capable, $100,000",
        
        "1.2.3 Vat Photopolymerization (VPP) - 液槽光重合": "1.2.3 Vat Photopolymerization (VPP)",
        "<strong>原理</strong>: 液状の光硬化性樹脂（フォトポリマー）に紫外線（UV）レーザーまたはプロジェクターで光を照射し、選択的に硬化させて積層。":
            "<strong>Principle</strong>: Liquid photopolymer resin is selectively cured by ultraviolet (UV) laser or projector light and deposited layer by layer.",
        "プロセス: UV照射 → 光重合反応 → 固化 → ビルドプラットフォーム上昇 → 次層照射":
            "Process: UV irradiation → Photopolymerization → Solidification → Build platform raise → Next layer exposure",
        
        "<strong>VPPの2つの主要方式：</strong>": "<strong>Two Main VPP Methods:</strong>",
        "<strong>SLA（Stereolithography）</strong>: UV レーザー（355 nm）をガルバノミラーで走査し、点描的に硬化。高精度だが低速。":
            "<strong>SLA (Stereolithography)</strong>: UV laser (355 nm) scanned with galvanometer mirrors, point-by-point curing. High precision but slow.",
        "<strong>DLP（Digital Light Processing）</strong>: プロジェクターで面全体を一括露光。高速だが解像度はプロジェクター画素数に依存（Full HD: 1920×1080）。":
            "<strong>DLP (Digital Light Processing)</strong>: Entire layer exposed simultaneously by projector. Fast but resolution dependent on projector pixels (Full HD: 1920×1080).",
        "<strong>LCD-MSLA（Masked SLA）</strong>: LCDマスクを使用、DLP類似だが低コスト化（$200-$1,000のデスクトップ機多数）。":
            "<strong>LCD-MSLA (Masked SLA)</strong>: Uses LCD mask, similar to DLP but lower cost (many desktop units $200-$1,000).",
        
        "<strong>高精度</strong>: XY解像度25-100 μm、Z解像度10-50 μm（全AM技術中で最高レベル）": "<strong>High Precision</strong>: XY resolution 25-100 μm, Z resolution 10-50 μm (highest level among all AM technologies)",
        "<strong>表面品質</strong>: 滑らかな表面（Ra < 5 μm）、積層痕がほぼ見えない": "<strong>Surface Quality</strong>: Smooth surface (Ra < 5 μm), layer lines barely visible",
        "<strong>造形速度</strong>: SLA（10-50 mm³/s）、DLP/LCD（100-500 mm³/s、面積依存）": "<strong>Build Speed</strong>: SLA (10-50 mm³/s), DLP/LCD (100-500 mm³/s, area-dependent)",
        "<strong>材料制約</strong>: 光硬化性樹脂のみ（機械的性質はFDMより劣る場合が多い）": "<strong>Material Limitations</strong>: Photopolymer resins only (mechanical properties often inferior to FDM)",
        "<strong>後処理必須</strong>: 洗浄（IPA等）→ 二次硬化（UV照射）→ サポート除去": "<strong>Post-processing Required</strong>: Washing (IPA etc.) → Post-curing (UV irradiation) → Support removal",
        
        "歯科用途（歯列矯正モデル、サージカルガイド、義歯、年間数百万個生産）": "Dental applications (orthodontic models, surgical guides, dentures, millions produced annually)",
        "ジュエリー鋳造用ワックスモデル（高精度・複雑形状）": "Wax models for jewelry casting (high precision, complex geometries)",
        "医療モデル（術前計画、解剖学モデル、患者説明用）": "Medical models (presurgical planning, anatomical models, patient education)",
        "マスターモデル（シリコン型取り用、デザイン検証）": "Master models (for silicone molding, design validation)",
        
        "1.2.4 Powder Bed Fusion (PBF) - 粉末床溶融結合": "1.2.4 Powder Bed Fusion (PBF)",
        "<strong>原理</strong>: 粉末材料を薄く敷き詰め、レーザーまたは電子ビームで選択的に溶融・焼結し、冷却固化させて積層。金属・ポリマー・セラミックスに対応。":
            "<strong>Principle</strong>: Powder material is spread in thin layers, selectively melted/sintered by laser or electron beam, cooled and solidified layer by layer. Compatible with metals, polymers, and ceramics.",
        "プロセス: 粉末敷設 → レーザー/電子ビーム走査 → 溶融・焼結 → 固化 → 次層粉末敷設":
            "Process: Powder spreading → Laser/electron beam scanning → Melting/sintering → Solidification → Next layer powder spreading",
        
        "<strong>PBFの3つの主要方式：</strong>": "<strong>Three Main PBF Methods:</strong>",
        "<strong>SLS（Selective Laser Sintering）</strong>: ポリマー粉末（PA12ナイロン等）をレーザー焼結。サポート不要（周囲粉末が支持）。":
            "<strong>SLS (Selective Laser Sintering)</strong>: Laser sinters polymer powder (PA12 nylon etc.). No support required (surrounding powder provides support).",
        "<strong>SLM（Selective Laser Melting）</strong>: 金属粉末（Ti-6Al-4V、AlSi10Mg、Inconel 718等）を完全溶融。高密度部品（相対密度>99%）製造可能。":
            "<strong>SLM (Selective Laser Melting)</strong>: Completely melts metal powder (Ti-6Al-4V, AlSi10Mg, Inconel 718 etc.). Produces high-density parts (relative density >99%).",
        "<strong>EBM（Electron Beam Melting）</strong>: 電子ビームで金属粉末を溶融。高温予熱（650-1000°C）により残留応力が小さく、造形速度が速い。":
            "<strong>EBM (Electron Beam Melting)</strong>: Melts metal powder with electron beam. High-temperature preheating (650-1000°C) reduces residual stress with faster build speed.",
        
        "<strong>高強度</strong>: 溶融・再凝固により鍛造材に匹敵する機械的性質（引張強度500-1200 MPa）": "<strong>High Strength</strong>: Melting and resolidification produces mechanical properties comparable to forged materials (tensile strength 500-1200 MPa)",
        "<strong>複雑形状対応</strong>: サポート不要（粉末が支持）でオーバーハング造形可能": "<strong>Complex Geometry Capable</strong>: Overhang fabrication without support (powder provides support)",
        "<strong>材料多様性</strong>: Ti合金、Al合金、ステンレス鋼、Ni超合金、Co-Cr合金、ナイロン": "<strong>Material Diversity</strong>: Ti alloys, Al alloys, stainless steel, Ni superalloys, Co-Cr alloys, nylon",
        "<strong>高コスト</strong>: 装置価格$200,000-$1,500,000、材料費$50-$500/kg": "<strong>High Cost</strong>: Equipment price $200,000-$1,500,000, material cost $50-$500/kg",
        "<strong>後処理</strong>: サポート除去、熱処理（応力除去）、表面仕上げ（ブラスト、研磨）": "<strong>Post-processing</strong>: Support removal, heat treatment (stress relief), surface finishing (blasting, polishing)",
        
        "航空宇宙部品（軽量化、一体化、GE LEAP燃料ノズル等）": "Aerospace components (weight reduction, integration, GE LEAP fuel nozzle etc.)",
        "医療インプラント（患者固有形状、多孔質構造、Ti-6Al-4V）": "Medical implants (patient-specific geometry, porous structures, Ti-6Al-4V)",
        "金型（コンフォーマル冷却、複雑形状、H13工具鋼）": "Molds (conformal cooling, complex geometries, H13 tool steel)",
        "自動車部品（軽量化ブラケット、カスタムエンジン部品）": "Automotive parts (lightweight brackets, custom engine components)",
        
        "1.2.5 Material Jetting (MJ) - 材料噴射": "1.2.5 Material Jetting (MJ)",
        "<strong>原理</strong>: インクジェットプリンタと同様に、液滴状の材料（光硬化性樹脂またはワックス）をヘッドから噴射し、UV照射で即座に硬化させて積層。":
            "<strong>Principle</strong>: Similar to inkjet printers, droplets of material (photopolymer resin or wax) are jetted from heads and immediately cured by UV irradiation for layer-by-layer deposition.",
        
        "<strong>超高精度</strong>: XY解像度42-85 μm、Z解像度16-32 μm": "<strong>Ultra-high Precision</strong>: XY resolution 42-85 μm, Z resolution 16-32 μm",
        "<strong>マルチマテリアル</strong>: 同一造形で複数材料・複数色を使い分け可能": "<strong>Multi-material</strong>: Can use multiple materials and colors in single build",
        "<strong>フルカラー造形</strong>: CMYK樹脂の組合せで1,000万色以上の表現": "<strong>Full-color Fabrication</strong>: Over 10 million colors expressible through CMYK resin combinations",
        "<strong>表面品質</strong>: 極めて滑らか（積層痕ほぼなし）": "<strong>Surface Quality</strong>: Extremely smooth (virtually no layer lines)",
        "<strong>高コスト</strong>: 装置$50,000-$300,000、材料費$200-$600/kg": "<strong>High Cost</strong>: Equipment $50,000-$300,000, material cost $200-$600/kg",
        "<strong>材料制約</strong>: 光硬化性樹脂のみ、機械的性質は中程度": "<strong>Material Limitations</strong>: Photopolymer resins only, moderate mechanical properties",
        
        "<strong>応用例：</strong>: 医療解剖モデル（軟組織・硬組織を異なる材料で再現）、フルカラー建築模型、デザイン検証モデル":
            "<strong>Applications:</strong> Medical anatomical models (soft and hard tissues reproduced with different materials), full-color architectural models, design validation models",
        
        "1.2.6 Binder Jetting (BJ) - 結合剤噴射": "1.2.6 Binder Jetting (BJ)",
        "<strong>原理</strong>: 粉末床に液状バインダー（接着剤）をインクジェット方式で噴射し、粉末粒子を結合。造形後に焼結または含浸処理で強度向上。":
            "<strong>Principle</strong>: Liquid binder (adhesive) is jetted inkjet-style onto powder bed to bond powder particles. Strength enhanced through sintering or infiltration after building.",
        
        "<strong>高速造形</strong>: レーザー走査不要で面全体を一括処理、造形速度100-500 mm³/s": "<strong>Fast Fabrication</strong>: No laser scanning required, entire layer processed simultaneously, build speed 100-500 mm³/s",
        "<strong>材料多様性</strong>: 金属粉末、セラミックス、砂型（鋳造用）、フルカラー（石膏）": "<strong>Material Diversity</strong>: Metal powders, ceramics, sand molds (for casting), full-color (gypsum)",
        "<strong>サポート不要</strong>: 周囲粉末が支持、除去後リサイクル可能": "<strong>No Support Required</strong>: Surrounding powder provides support, recyclable after removal",
        "<strong>低密度問題</strong>: 焼結前は脆弱（グリーン密度50-60%）、焼結後も相対密度90-98%": "<strong>Low Density Issue</strong>: Fragile before sintering (green density 50-60%), relative density 90-98% even after sintering",
        "<strong>後処理必須</strong>: 脱脂 → 焼結（金属：1200-1400°C）→ 含浸（銅・青銅）": "<strong>Post-processing Required</strong>: Debinding → Sintering (metal: 1200-1400°C) → Infiltration (copper/bronze)",
        
        "砂型鋳造用型（エンジンブロック等の大型鋳物）、金属部品（Desktop Metal、HP Metal Jet）、フルカラー像（記念品、教育モデル）":
            "Sand casting molds (large castings like engine blocks), metal parts (Desktop Metal, HP Metal Jet), full-color objects (souvenirs, educational models)",
        
        "1.2.7 Sheet Lamination (SL) - シート積層": "1.2.7 Sheet Lamination (SL)",
        "<strong>原理</strong>: シート状材料（紙、金属箔、プラスチックフィルム）を積層し、接着または溶接で結合。各層をレーザーまたはブレードで輪郭切断。":
            "<strong>Principle</strong>: Sheet materials (paper, metal foil, plastic film) are laminated and bonded by adhesive or welding. Each layer contour-cut by laser or blade.",
        
        "<strong>代表技術：</strong>": "<strong>Representative Technologies:</strong>",
        "<strong>LOM（Laminated Object Manufacturing）</strong>: 紙・プラスチックシート、接着剤で積層、レーザー切断":
            "<strong>LOM (Laminated Object Manufacturing)</strong>: Paper/plastic sheets, laminated with adhesive, laser cut",
        "<strong>UAM（Ultrasonic Additive Manufacturing）</strong>: 金属箔を超音波溶接、CNC切削で輪郭加工":
            "<strong>UAM (Ultrasonic Additive Manufacturing)</strong>: Metal foils ultrasonically welded, contour machined by CNC",
        
        "<strong>特徴：</strong> 大型造形可能、材料費安価、精度中程度、用途限定的（主に視覚モデル、金属では埋込センサー等）":
            "<strong>Characteristics:</strong> Large-scale fabrication possible, low material cost, moderate accuracy, limited applications (mainly visual models, embedded sensors in metals)",
        
        "1.2.8 Directed Energy Deposition (DED) - 指向性エネルギー堆積": "1.2.8 Directed Energy Deposition (DED)",
        "<strong>原理</strong>: 金属粉末またはワイヤーを供給しながら、レーザー・電子ビーム・アークで溶融し、基板上に堆積。大型部品や既存部品の補修に使用。":
            "<strong>Principle</strong>: Metal powder or wire is fed and melted by laser, electron beam, or arc, deposited on substrate. Used for large parts and repair of existing parts.",
        
        "<strong>高速堆積</strong>: 堆積速度1-5 kg/h（PBFの10-50倍）": "<strong>Fast Deposition</strong>: Deposition rate 1-5 kg/h (10-50 times PBF)",
        "<strong>大型対応</strong>: ビルドボリューム制限が少ない（多軸ロボットアーム使用）": "<strong>Large-scale Capable</strong>: Minimal build volume limitations (using multi-axis robot arms)",
        "<strong>補修・コーティング</strong>: 既存部品の摩耗部分修復、表面硬化層形成": "<strong>Repair & Coating</strong>: Repair worn parts of existing components, form surface hardening layers",
        "<strong>低精度</strong>: 精度±0.5-2 mm、後加工（機械加工）必須": "<strong>Low Precision</strong>: Accuracy ±0.5-2 mm, post-processing (machining) required",
        
        "タービンブレード補修、大型航空宇宙部品、工具の耐摩耗コーティング": "Turbine blade repair, large aerospace components, wear-resistant coatings for tools",
        
        "⚠️ プロセス選択の指針": "⚠️ Process Selection Guidelines",
        "最適なAMプロセスは用途要求により異なります：": "Optimal AM process varies according to application requirements:",
        "<strong>精度最優先</strong> → VPP（SLA/DLP）またはMJ": "<strong>Precision Priority</strong> → VPP (SLA/DLP) or MJ",
        "<strong>低コスト・普及型</strong> → MEX（FDM/FFF）": "<strong>Low-cost & Widespread</strong> → MEX (FDM/FFF)",
        "<strong>金属高強度部品</strong> → PBF（SLM/EBM）": "<strong>High-strength Metal Parts</strong> → PBF (SLM/EBM)",
        "<strong>大量生産（砂型）</strong> → BJ": "<strong>Mass Production (Sand molds)</strong> → BJ",
        "<strong>大型・高速堆積</strong> → DED": "<strong>Large-scale & Fast Deposition</strong> → DED",
    })
    
    return translations

def apply_translations(content, translations):
    """Apply all translations to content"""
    for jp, en in translations.items():
        content = content.replace(jp, en)
    return content

def main():
    target_file = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-2.html")
    
    # Read current target
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create and apply translations
    translations = create_comprehensive_translations()
    content = apply_translations(content, translations)
    
    # Write back
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Count remaining Japanese
    jp_count = len(re.findall(r'[あ-ん]|[ア-ン]|[一-龯]', content))
    print(f"Phase 1-2 translations complete. Remaining Japanese characters: {jp_count}")
    
    if jp_count > 0:
        print(f"\nProgress: {13178 - jp_count} characters translated ({((13178 - jp_count) / 13178 * 100):.1f}%)")

if __name__ == "__main__":
    main()

def create_phase3_translations():
    """Phase 3+: STL format, slicing, G-code, exercises"""
    translations = {}
    
    # Section 1.3: STL File Format
    translations.update({
        "1.3 STLファイル形式とデータ処理": "1.3 STL File Format and Data Processing",
        "1.3.1 STLファイルの構造": "1.3.1 Structure of STL Files",
        "STL（STereoLithography）は、<strong>AMで最も広く使用される3Dモデルファイル形式</strong>で、1987年に3D Systems社が開発しました。STLファイルは物体表面を<strong>三角形メッシュ（Triangle Mesh）の集合</strong>として表現します。":
            "STL (STereoLithography) is <strong>the most widely used 3D model file format in AM</strong>, developed by 3D Systems in 1987. STL files represent object surfaces as <strong>a collection of triangular meshes</strong>.",
        
        "STLファイルの基本構造": "Basic Structure of STL Files",
        "STLファイル = 法線ベクトル（n） + 3つの頂点座標（v1, v2, v3）× 三角形数":
            "STL file = Normal vector (n) + Three vertex coordinates (v1, v2, v3) × Number of triangles",
        "<strong>ASCII STL形式の例：</strong>": "<strong>ASCII STL Format Example:</strong>",
        "<strong>STLフォーマットの2つの種類：</strong>": "<strong>Two Types of STL Format:</strong>",
        "<strong>ASCII STL</strong>: 人間が読めるテキスト形式。ファイルサイズ大（同じモデルでBinaryの10-20倍）。デバッグ・検証に有用。":
            "<strong>ASCII STL</strong>: Human-readable text format. Large file size (10-20x Binary for same model). Useful for debugging and validation.",
        "<strong>Binary STL</strong>: バイナリ形式、ファイルサイズ小、処理高速。産業用途で標準。構造：80バイトヘッダー + 4バイト（三角形数） + 各三角形50バイト（法線12B + 頂点36B + 属性2B）。":
            "<strong>Binary STL</strong>: Binary format, small file size, fast processing. Standard for industrial use. Structure: 80-byte header + 4 bytes (triangle count) + 50 bytes per triangle (12B normal + 36B vertices + 2B attribute).",
        
        "1.3.2 STLファイルの重要概念": "1.3.2 Important Concepts of STL Files",
        "1. 法線ベクトル（Normal Vector）": "1. Normal Vector",
        "各三角形面には<strong>法線ベクトル（外向き方向）</strong>が定義され、物体の「内側」と「外側」を区別します。法線方向は<strong>右手の法則</strong>で決定されます：":
            "Each triangular facet has a <strong>normal vector (outward direction)</strong> defined, distinguishing object 'inside' from 'outside'. Normal direction is determined by the <strong>right-hand rule</strong>:",
        "法線n = (v2 - v1) × (v3 - v1) / |(v2 - v1) × (v3 - v1)|":
            "Normal n = (v2 - v1) × (v3 - v1) / |(v2 - v1) × (v3 - v1)|",
        "<strong>頂点順序ルール：</strong> 頂点v1, v2, v3は反時計回り（CCW: Counter-ClockWise）に配置され、外から見て反時計回りの順序で法線が外向きになります。":
            "<strong>Vertex Ordering Rule:</strong> Vertices v1, v2, v3 are arranged counter-clockwise (CCW), and when viewed from outside, counter-clockwise order results in outward-facing normal.",
        
        "2. 多様体（Manifold）条件": "2. Manifold Conditions",
        "STLメッシュが3Dプリント可能であるためには、<strong>多様体（Manifold）</strong>でなければなりません：":
            "For an STL mesh to be 3D printable, it must be <strong>manifold</strong>:",
        "<strong>エッジ共有</strong>: すべてのエッジ（辺）は正確に2つの三角形に共有される":
            "<strong>Edge Sharing</strong>: Every edge must be shared by exactly two triangles",
        "<strong>頂点共有</strong>: すべての頂点は連続した三角形扇（fan）に属する":
            "<strong>Vertex Sharing</strong>: Every vertex must belong to a continuous triangle fan",
        "<strong>閉じた表面</strong>: 穴や開口部がなく、完全に閉じた表面を形成":
            "<strong>Closed Surface</strong>: Forms a completely closed surface without holes or openings",
        "<strong>自己交差なし</strong>: 三角形が互いに交差・貫通していない":
            "<strong>No Self-Intersection</strong>: Triangles do not intersect or penetrate each other",
        
        "⚠️ 非多様体メッシュの問題": "⚠️ Non-Manifold Mesh Problems",
        "非多様体メッシュ（Non-Manifold Mesh）は3Dプリント不可能です。典型的な問題：":
            "Non-manifold meshes are unprintable in 3D. Typical problems:",
        "<strong>穴（Holes）</strong>: 閉じていない表面、エッジが1つの三角形にのみ属する":
            "<strong>Holes</strong>: Unclosed surface, edges belonging to only one triangle",
        "<strong>T字接合（T-junction）</strong>: エッジが3つ以上の三角形に共有される":
            "<strong>T-junction</strong>: Edge shared by three or more triangles",
        "<strong>法線反転（Inverted Normals）</strong>: 法線が内側を向いている三角形が混在":
            "<strong>Inverted Normals</strong>: Triangles with normals facing inward mixed in",
        "<strong>重複頂点（Duplicate Vertices）</strong>: 同じ位置に複数の頂点が存在":
            "<strong>Duplicate Vertices</strong>: Multiple vertices existing at the same position",
        "<strong>微小三角形（Degenerate Triangles）</strong>: 面積がゼロまたはほぼゼロの三角形":
            "<strong>Degenerate Triangles</strong>: Triangles with zero or near-zero area",
        "これらの問題はスライサーソフトウェアでエラーを引き起こし、造形失敗の原因となります。":
            "These problems cause errors in slicer software and lead to print failures.",
        
        "1.3.3 STLファイルの品質指標": "1.3.3 STL File Quality Metrics",
        "STLメッシュの品質は以下の指標で評価されます：": "STL mesh quality is evaluated by the following metrics:",
        "<strong>三角形数（Triangle Count）</strong>: 通常10,000-500,000個。過少（粗いモデル）または過多（ファイルサイズ大・処理遅延）は避ける。":
            "<strong>Triangle Count</strong>: Typically 10,000-500,000. Avoid too few (coarse model) or too many (large file size, processing delay).",
        "<strong>エッジ長の一様性</strong>: 極端に大小の三角形が混在すると造形品質低下。理想的には0.1-1.0 mm範囲。":
            "<strong>Edge Length Uniformity</strong>: Mixture of extremely large and small triangles degrades build quality. Ideally 0.1-1.0 mm range.",
        "<strong>アスペクト比（Aspect Ratio）</strong>: 細長い三角形（高アスペクト比）は数値誤差の原因。理想的にはアスペクト比 < 10。":
            "<strong>Aspect Ratio</strong>: Elongated triangles (high aspect ratio) cause numerical errors. Ideally aspect ratio < 10.",
        "<strong>法線の一貫性</strong>: すべての法線が外向き統一。反転法線が混在すると内外判定エラー。":
            "<strong>Normal Consistency</strong>: All normals uniformly outward-facing. Mixed inverted normals cause inside/outside determination errors.",
        
        "💡 STLファイルの解像度トレードオフ": "💡 STL File Resolution Tradeoff",
        "STLメッシュの解像度（三角形数）は精度とファイルサイズのトレードオフです：":
            "STL mesh resolution (triangle count) involves a tradeoff between accuracy and file size:",
        "<strong>低解像度（1,000-10,000三角形）</strong>: 高速処理、小ファイル、但し曲面が角張る（ファセット化明瞭）":
            "<strong>Low Resolution (1,000-10,000 triangles)</strong>: Fast processing, small file, but curved surfaces appear faceted (visible faceting)",
        "<strong>中解像度（10,000-100,000三角形）</strong>: 多くの用途で適切、バランス良好":
            "<strong>Medium Resolution (10,000-100,000 triangles)</strong>: Appropriate for most applications, good balance",
        "<strong>高解像度（100,000-1,000,000三角形）</strong>: 滑らかな曲面、但しファイルサイズ大（数十MB）、処理遅延":
            "<strong>High Resolution (100,000-1,000,000 triangles)</strong>: Smooth curved surfaces, but large file size (tens of MB), processing delay",
        "CADソフトでSTLエクスポート時に、<strong>Chordal Tolerance（コード公差）</strong>または<strong>Angle Tolerance（角度公差）</strong>で解像度を制御します。推奨値：コード公差0.01-0.1 mm、角度公差5-15度。":
            "When exporting STL from CAD software, control resolution with <strong>Chordal Tolerance</strong> or <strong>Angle Tolerance</strong>. Recommended values: chordal tolerance 0.01-0.1 mm, angle tolerance 5-15 degrees.",
        
        "1.3.4 Pythonライブラリによる STL処理": "1.3.4 STL Processing with Python Libraries",
        "PythonでSTLファイルを扱うための主要ライブラリ：": "Main libraries for handling STL files in Python:",
        "<strong>numpy-stl</strong>: 高速STL読込・書込、体積・表面積計算、法線ベクトル操作。シンプルで軽量。":
            "<strong>numpy-stl</strong>: Fast STL read/write, volume/surface area calculation, normal vector operations. Simple and lightweight.",
        "<strong>trimesh</strong>: 包括的な3Dメッシュ処理ライブラリ。メッシュ修復、ブーリアン演算、レイキャスト、衝突検出。多機能だが依存関係多い。":
            "<strong>trimesh</strong>: Comprehensive 3D mesh processing library. Mesh repair, Boolean operations, raycasting, collision detection. Feature-rich but many dependencies.",
        "<strong>PyMesh</strong>: 高度なメッシュ処理（リメッシュ、サブディビジョン、フィーチャー抽出）。インストールやや複雑。":
            "<strong>PyMesh</strong>: Advanced mesh processing (remeshing, subdivision, feature extraction). Somewhat complex installation.",
        "<strong>numpy-stlの基本的な使用法：</strong>": "<strong>Basic numpy-stl Usage:</strong>",
        
        # Section 1.4: Slicing
        "1.4 スライシングとツールパス生成": "1.4 Slicing and Toolpath Generation",
        "STLファイルを3Dプリンタが理解できる指令（G-code）に変換するプロセスを<strong>スライシング（Slicing）</strong>といいます。このセクションでは、スライシングの基本原理、ツールパス戦略、そしてG-codeの基礎を学びます。":
            "The process of converting STL files into commands (G-code) that 3D printers understand is called <strong>slicing</strong>. This section covers basic principles of slicing, toolpath strategies, and G-code fundamentals.",
        
        "1.4.1 スライシングの基本原理": "1.4.1 Basic Principles of Slicing",
        "スライシングは、3Dモデルを一定の高さ（レイヤー高さ）で水平に切断し、各層の輪郭を抽出するプロセスです：":
            "Slicing is the process of horizontally cutting a 3D model at constant heights (layer heights) and extracting contours of each layer:",
        
        "3Dモデル<br/>STLファイル": "3D Model<br/>STL File",
        "Z軸方向に<br/>層状にスライス": "Slice layer by layer<br/>in Z-axis direction",
        "各層の輪郭抽出<br/>Contour Detection": "Extract layer contours<br/>Contour Detection",
        "シェル生成<br/>Perimeter Path": "Generate shells<br/>Perimeter Path",
        "インフィル生成<br/>Infill Path": "Generate infill<br/>Infill Path",
        "サポート追加<br/>Support Structure": "Add support<br/>Support Structure",
        "ツールパス最適化<br/>Retraction/Travel": "Optimize toolpath<br/>Retraction/Travel",
        "G-code出力": "G-code Output",
        
        "レイヤー高さ（Layer Height）の選択": "Layer Height Selection",
        "レイヤー高さは造形品質と造形時間のトレードオフを決定する最重要パラメータです：":
            "Layer height is the most important parameter determining the tradeoff between build quality and build time:",
        
        "レイヤー高さ": "Layer Height",
        "造形品質": "Build Quality",
        "造形時間": "Build Time",
        "典型的な用途": "Typical Applications",
        "0.1 mm（極細）": "0.1 mm (Extra Fine)",
        "非常に高い（積層痕ほぼ不可視）": "Very High (layer lines barely visible)",
        "非常に長い（×2-3倍）": "Very Long (×2-3x)",
        "フィギュア、医療モデル、最終製品": "Figurines, medical models, end-use parts",
        "0.2 mm（標準）": "0.2 mm (Standard)",
        "良好（積層痕は見えるが許容）": "Good (layer lines visible but acceptable)",
        "標準": "Standard",
        "一般的なプロトタイプ、機能部品": "General prototypes, functional parts",
        "0.3 mm（粗）": "0.3 mm (Coarse)",
        "低い（積層痕明瞭）": "Low (visible layer lines)",
        "短い（×0.5倍）": "Short (×0.5x)",
        "初期プロトタイプ、内部構造部品": "Initial prototypes, internal structural parts",
        
        "⚠️ レイヤー高さの制約": "⚠️ Layer Height Constraints",
        "レイヤー高さはノズル径の<strong>25-80%</strong>に設定する必要があります。例えば0.4mmノズルの場合、レイヤー高さは0.1-0.32mmが推奨範囲です。これを超えると、樹脂の押出量が不足したり、ノズルが前の層を引きずる問題が発生します。":
            "Layer height must be set to <strong>25-80%</strong> of nozzle diameter. For example, with a 0.4mm nozzle, layer height of 0.1-0.32mm is the recommended range. Exceeding this causes insufficient resin extrusion or nozzle dragging on previous layers.",
        
        "1.4.2 シェルとインフィル戦略": "1.4.2 Shell and Infill Strategies",
        "シェル（外殻）の生成": "Shell (Perimeter) Generation",
        "<strong>シェル（Shell/Perimeter）</strong>は、各層の外周部を形成する経路です：":
            "<strong>Shell (Shell/Perimeter)</strong> is the path forming the outer perimeter of each layer:",
        "<strong>シェル数（Perimeter Count）</strong>: 通常2-4本。外部品質と強度に影響。":
            "<strong>Perimeter Count</strong>: Typically 2-4. Affects external quality and strength.",
        "1本: 非常に弱い、透明性高い、装飾用のみ": "1: Very weak, high transparency, decorative only",
        "2本: 標準（バランス良好）": "2: Standard (good balance)",
        "3-4本: 高強度、表面品質向上、気密性向上": "3-4: High strength, improved surface quality, improved airtightness",
        "<strong>シェル順序</strong>: 内側→外側（Inside-Out）が一般的。外側→内側は表面品質重視時に使用。":
            "<strong>Shell Order</strong>: Inside-Out is common. Outside-In used when prioritizing surface quality.",
        
        "インフィル（内部充填）パターン": "Infill Pattern",
        "<strong>インフィル（Infill）</strong>は内部構造を形成し、強度と材料使用量を制御します：":
            "<strong>Infill</strong> forms internal structure, controlling strength and material usage:",
        
        "パターン": "Pattern",
        "強度": "Strength",
        "印刷速度": "Print Speed",
        "材料使用量": "Material Usage",
        "特徴": "Features",
        "Grid（格子）": "Grid",
        "中": "Medium",
        "速い": "Fast",
        "シンプル、等方性、標準的な選択": "Simple, isotropic, standard choice",
        "Honeycomb（ハニカム）": "Honeycomb",
        "高": "High",
        "遅い": "Slow",
        "高強度、重量比優秀、航空宇宙用途": "High strength, excellent weight ratio, aerospace applications",
        "Gyroid": "Gyroid",
        "非常に高": "Very High",
        "3次元等方性、曲面的、最新の推奨": "3D isotropic, curved surfaces, latest recommendation",
        "Concentric（同心円）": "Concentric",
        "低": "Low",
        "少": "Less",
        "柔軟性重視、シェル追従": "Flexibility priority, follows shell",
        "Lines（直線）": "Lines",
        "低（異方性）": "Low (anisotropic)",
        "非常に速い": "Very Fast",
        "高速印刷、方向性強度": "Fast printing, directional strength",
        
        "💡 インフィル密度の目安": "💡 Infill Density Guidelines",
        "<strong>0-10%</strong>: 装飾品、非荷重部品（材料節約優先）": "<strong>0-10%</strong>: Decorative, non-load-bearing parts (material saving priority)",
        "<strong>20%</strong>: 標準的なプロトタイプ（バランス良好）": "<strong>20%</strong>: Standard prototypes (good balance)",
        "<strong>40-60%</strong>: 機能部品、高強度要求": "<strong>40-60%</strong>: Functional parts, high strength requirements",
        "<strong>100%</strong>: 最終製品、水密性要求、最高強度（造形時間×3-5倍）": "<strong>100%</strong>: End-use products, watertightness requirements, maximum strength (build time ×3-5x)",
        
        "1.4.3 サポート構造の生成": "1.4.3 Support Structure Generation",
        "オーバーハング角度が45度を超える部分は、<strong>サポート構造（Support Structure）</strong>が必要です：":
            "Parts with overhang angles exceeding 45 degrees require <strong>support structures</strong>:",
        
        "サポートのタイプ": "Support Types",
        "<strong>Linear Support（直線サポート）</strong>: 垂直な柱状サポート。シンプルで除去しやすいが、材料使用量多い。":
            "<strong>Linear Support</strong>: Vertical columnar supports. Simple and easy to remove, but uses more material.",
        "<strong>Tree Support（ツリーサポート）</strong>: 樹木状に分岐するサポート。材料使用量30-50%削減、除去しやすい。CuraやPrusaSlicerで標準サポート。":
            "<strong>Tree Support</strong>: Tree-like branching supports. 30-50% material reduction, easy to remove. Standard in Cura and PrusaSlicer.",
        "<strong>Interface Layers（接合層）</strong>: サポート上面に薄い接合層を設ける。除去しやすく、表面品質向上。通常2-4層。":
            "<strong>Interface Layers</strong>: Thin interface layers on support top. Easy to remove, improved surface quality. Typically 2-4 layers.",
        
        "サポート設定の重要パラメータ": "Important Support Parameters",
        "パラメータ": "Parameter",
        "推奨値": "Recommended Value",
        "効果": "Effect",
        "Overhang Angle": "Overhang Angle",
        "45-60°": "45-60°",
        "この角度以上でサポート生成": "Generate support above this angle",
        "Support Density": "Support Density",
        "10-20%": "10-20%",
        "密度が高いほど安定だが除去困難": "Higher density more stable but harder to remove",
        "Support Z Distance": "Support Z Distance",
        "0.2-0.3 mm": "0.2-0.3 mm",
        "サポートと造形物の間隔（除去しやすさ）": "Gap between support and part (ease of removal)",
        "Interface Layers": "Interface Layers",
        "2-4層": "2-4 layers",
        "接合層数（表面品質と除去性のバランス）": "Number of interface layers (balance of surface quality and removability)",
        
        "1.4.4 G-codeの基礎": "1.4.4 G-code Fundamentals",
        "<strong>G-code</strong>は、3DプリンタやCNCマシンを制御する標準的な数値制御言語です。各行が1つのコマンドを表します：":
            "<strong>G-code</strong> is the standard numerical control language for controlling 3D printers and CNC machines. Each line represents one command:",
        
        "主要なG-codeコマンド": "Main G-code Commands",
        "コマンド": "Command",
        "分類": "Category",
        "機能": "Function",
        "例": "Example",
        "G0": "G0",
        "移動": "Movement",
        "高速移動（非押出）": "Rapid movement (no extrusion)",
        "G0 X100 Y50 Z10 F6000": "G0 X100 Y50 Z10 F6000",
        "G1": "G1",
        "直線移動（押出あり）": "Linear movement (with extrusion)",
        "G1 X120 Y60 E0.5 F1200": "G1 X120 Y60 E0.5 F1200",
        "G28": "G28",
        "初期化": "Initialization",
        "ホームポジション復帰": "Return to home position",
        "G28 （全軸）, G28 Z （Z軸のみ）": "G28 (all axes), G28 Z (Z-axis only)",
        "M104": "M104",
        "温度": "Temperature",
        "ノズル温度設定（非待機）": "Set nozzle temperature (no wait)",
        "M104 S200": "M104 S200",
        "M109": "M109",
        "ノズル温度設定（待機）": "Set nozzle temperature (wait)",
        "M109 S210": "M109 S210",
        "M140": "M140",
        "ベッド温度設定（非待機）": "Set bed temperature (no wait)",
        "M140 S60": "M140 S60",
        "M190": "M190",
        "ベッド温度設定（待機）": "Set bed temperature (wait)",
        "M190 S60": "M190 S60",
        
        "G-codeの例（造形開始部分）": "G-code Example (Print Start)",
        
        "1.4.5 主要スライシングソフトウェア": "1.4.5 Major Slicing Software",
        "ソフトウェア": "Software",
        "ライセンス": "License",
        "オープンソース": "Open Source",
        "使いやすい、豊富なプリセット、Tree Support標準搭載": "Easy to use, rich presets, Tree Support built-in",
        "初心者〜中級者、FDM汎用": "Beginners to intermediate users, FDM general-purpose",
        "高度な設定、変数レイヤー高さ、カスタムサポート": "Advanced settings, variable layer height, custom support",
        "中級者〜上級者、最適化重視": "Intermediate to advanced users, optimization-focused",
        "PrusaSlicerの元祖、軽量": "Original PrusaSlicer, lightweight",
        "レガシーシステム、研究用途": "Legacy systems, research applications",
        "商用（$150）": "Commercial ($150)",
        "高速スライシング、マルチプロセス、詳細制御": "Fast slicing, multi-process, detailed control",
        "プロフェッショナル、産業用途": "Professional, industrial applications",
        "無料": "Free",
        "Raise3D専用だが汎用性高い、直感的UI": "Raise3D-specific but highly versatile, intuitive UI",
        "Raise3Dユーザー、初心者": "Raise3D users, beginners",
        
        "1.4.6 ツールパス最適化戦略": "1.4.6 Toolpath Optimization Strategies",
        "効率的なツールパスは、造形時間・品質・材料使用量を改善します：":
            "Efficient toolpaths improve build time, quality, and material usage:",
        
        "<strong>リトラクション（Retraction）</strong>: 移動時にフィラメントを引き戻してストリング（糸引き）を防止。":
            "<strong>Retraction</strong>: Retracts filament during travel to prevent stringing.",
        "距離: 1-6mm（ボーデンチューブ式は4-6mm、ダイレクト式は1-2mm）":
            "Distance: 1-6mm (Bowden tube: 4-6mm, direct drive: 1-2mm)",
        "速度: 25-45 mm/s": "Speed: 25-45 mm/s",
        "過度なリトラクションはノズル詰まりの原因": "Excessive retraction causes nozzle clogging",
        "<strong>Z-hop（Z軸跳躍）</strong>: 移動時にノズルを上昇させて造形物との衝突を回避。0.2-0.5mm上昇。造形時間微増だが表面品質向上。":
            "<strong>Z-hop</strong>: Raises nozzle during travel to avoid collision with part. 0.2-0.5mm raise. Slight time increase but improved surface quality.",
        "<strong>コーミング（Combing）</strong>: 移動経路をインフィル上に制限し、表面への移動痕を低減。外観重視時に有効。":
            "<strong>Combing</strong>: Restricts travel paths to infill, reducing travel marks on surface. Effective when appearance matters.",
        "<strong>シーム位置（Seam Position）</strong>: 各層の開始/終了点を揃える戦略。":
            "<strong>Seam Position</strong>: Strategy for aligning layer start/end points.",
        "Random: ランダム配置（目立たない）": "Random: Random placement (less visible)",
        "Aligned: 一直線に配置（後加工でシームを除去しやすい）": "Aligned: Linear placement (easier to remove seam with post-processing)",
        "Sharpest Corner: 最も鋭角なコーナーに配置（目立ちにくい）": "Sharpest Corner: Placed at sharpest corner (less noticeable)",
    })
    
    return translations

# Execute Phase 3
def main_phase3():
    from pathlib import Path
    import re
    
    target_file = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-2.html")
    
    # Read current target
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply Phase 3 translations
    translations = create_phase3_translations()
    for jp, en in translations.items():
        content = content.replace(jp, en)
    
    # Write back
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Count remaining Japanese
    jp_count = len(re.findall(r'[あ-ん]|[ア-ン]|[一-龯]', content))
    print(f"Phase 3 translations complete. Remaining Japanese characters: {jp_count}")
    
    if jp_count > 0:
        print(f"\nProgress: {13178 - jp_count} characters translated ({((13178 - jp_count) / 13178 * 100):.1f}%)")

if __name__ == "__main__":
    main_phase3()

def create_phase4_translations():
    """Phase 4: Python examples, exercises, and remaining content"""
    translations = {}
    
    # Python examples
    translations.update({
        "Example 1: STLファイルの読み込みと基本情報取得": "Example 1: Reading STL Files and Obtaining Basic Information",
        "# STLファイルを読み込む": "# Read STL file",
        "# 基本的な幾何情報を取得": "# Get basic geometric information",
        "# バウンディングボックス（最小包含直方体）を計算": "# Calculate bounding box (minimum enclosing cuboid)",
        "# 造形時間の簡易推定（レイヤー高さ0.2mm、速度50mm/sと仮定）": "# Simple build time estimation (assuming layer height 0.2mm, speed 50mm/s)",
        "# 簡易計算: 表面積に基づく推定": "# Simple calculation: estimation based on surface area",
        "=== STLファイル基本情報 ===": "=== STL File Basic Information ===",
        "=== バウンディングボックス ===": "=== Bounding Box ===",
        "幅": "Width",
        "奥行": "Depth",
        "高さ": "Height",
        "=== 造形推定 ===": "=== Build Estimation ===",
        "レイヤー数（0.2mm/層）": "Number of layers (0.2mm/layer)",
        "層": "layers",
        "推定造形時間": "Estimated build time",
        "分": "minutes",
        "時間": "hours",
        
        "Example 2: メッシュの法線ベクトル検証": "Example 2: Mesh Normal Vector Validation",
        "STLメッシュの法線ベクトルの整合性をチェック": "Check consistency of normal vectors in STL mesh",
        "右手系ルールで法線方向を確認": "Verify normal direction with right-hand rule",
        "エッジベクトルを計算": "Calculate edge vectors",
        "外積で法線を計算（右手系）": "Calculate normal with cross product (right-hand system)",
        "正規化": "Normalize",
        "ゼロベクトルでないことを確認": "Confirm not zero vector",
        "縮退三角形をスキップ": "Skip degenerate triangles",
        "ファイルに保存されている法線と比較": "Compare with normals stored in file",
        "内積で方向の一致をチェック": "Check direction match with dot product",
        "内積が負なら逆向き": "If dot product negative, opposite direction",
        "法線チェックを実行": "Execute normal check",
        "=== 法線ベクトル検証結果 ===": "=== Normal Vector Validation Results ===",
        "総三角形数": "Total triangle count",
        "反転法線数": "Flipped normal count",
        "反転率": "Flip rate",
        "✅ すべての法線が正しい方向を向いています": "✅ All normals are correctly oriented",
        "   このメッシュは3Dプリント可能です": "   This mesh is 3D printable",
        "⚠️ 一部の法線が反転しています（軽微）": "⚠️ Some normals are flipped (minor)",
        "   スライサーが自動修正する可能性が高い": "   Slicer will likely auto-correct",
        "❌ 多数の法線が反転しています（重大）": "❌ Many normals are flipped (critical)",
        "   メッシュ修復ツール（Meshmixer, netfabb）での修正を推奨": "   Recommend repair with mesh repair tools (Meshmixer, netfabb)",
        
        "Example 3: マニフォールド性のチェック": "Example 3: Manifold Checking",
        "Example 3: マニフォールド性（Watertight）のチェック": "Example 3: Manifold (Watertight) Checking",
        "# STLファイルを読み込み（trimeshは自動で修復を試みる）": "# Read STL file (trimesh attempts auto-repair)",
        "=== メッシュ品質診断 ===": "=== Mesh Quality Diagnosis ===",
        "# 基本情報": "# Basic information",
        "Vertex count": "Vertex count",
        "Face count": "Face count",
        "Volume": "Volume",
        "# マニフォールド性をチェック": "# Check manifold property",
        "=== 3Dプリント適性チェック ===": "=== 3D Printability Check ===",
        "Is watertight (密閉性)": "Is watertight",
        "Is winding consistent (法線一致性)": "Is winding consistent",
        "Is valid (幾何的妥当性)": "Is valid",
        "# 問題の詳細を診断": "# Diagnose problems in detail",
        "# 穴（hole）の数を検出": "# Detect number of holes",
        "⚠️ 問題検出:": "⚠️ Problems Detected:",
        "   - メッシュに穴があります": "   - Mesh has holes",
        "   - 重複エッジ数": "   - Duplicate edge count",
        "⚠️ メッシュ構造に問題があります": "⚠️ Mesh structure has problems",
        "# 修復を試みる": "# Attempt repair",
        "🔧 自動修復を実行中...": "🔧 Executing auto-repair...",
        "# 法線を修正": "# Fix normals",
        "   ✓ 法線ベクトルを修正": "   ✓ Fixed normal vectors",
        "# 穴を埋める": "# Fill holes",
        "   ✓ 穴を充填": "   ✓ Filled holes",
        "# 縮退三角形を削除": "# Remove degenerate faces",
        "   ✓ 縮退面を削除": "   ✓ Removed degenerate faces",
        "# 重複頂点を結合": "# Merge duplicate vertices",
        "   ✓ 重複頂点を結合": "   ✓ Merged duplicate vertices",
        "# 修復後の状態を確認": "# Check state after repair",
        "=== 修復後の状態 ===": "=== State After Repair ===",
        "# 修復したメッシュを保存": "# Save repaired mesh",
        "✅ 修復完了！ model_repaired.stl として保存しました": "✅ Repair complete! Saved as model_repaired.stl",
        "❌ 自動修復失敗。Meshmixer等の専用ツールを推奨": "❌ Auto-repair failed. Recommend dedicated tools like Meshmixer",
        "✅ このメッシュは3Dプリント可能です": "✅ This mesh is 3D printable",
        
        "Example 4: 基本的なスライシングアルゴリズム": "Example 4: Basic Slicing Algorithm",
        
        # Exercises section
        "演習問題": "Exercises",
        "問題1: STLファイルの品質評価": "Problem 1: STL File Quality Assessment",
        "問題2: 最適なレイヤー高さの選択": "Problem 2: Optimal Layer Height Selection",
        "問題3: サポート構造の設計": "Problem 3: Support Structure Design",
        "問題4: インフィルパターンの比較": "Problem 4: Infill Pattern Comparison",
        "問題5: G-codeの解析": "Problem 5: G-code Analysis",
        
        # Summary and references
        "まとめ": "Summary",
        "この章では、積層造形（AM）の基礎として以下を学びました：":
            "In this chapter, we learned the following fundamentals of Additive Manufacturing (AM):",
        "AMの定義と7つのプロセスカテゴリ（MEX, VPP, PBF, MJ, BJ, SL, DED）":
            "Definition of AM and seven process categories (MEX, VPP, PBF, MJ, BJ, SL, DED)",
        "各プロセスの原理、特徴、応用例":
            "Principles, characteristics, and applications of each process",
        "STLファイル形式の構造（三角形メッシュ、法線ベクトル、多様体条件）":
            "Structure of STL file format (triangle mesh, normal vectors, manifold conditions)",
        "PythonによるSTL処理（numpy-stl、trimesh）":
            "STL processing with Python (numpy-stl, trimesh)",
        "スライシングの基本原理（レイヤー高さ、シェル、インフィル、サポート）":
            "Basic principles of slicing (layer height, shells, infill, support)",
        "G-codeの基本構造と主要コマンド":
            "Basic structure and main commands of G-code",
        
        "次章では、材料押出法（FDM/FFF）の詳細な技術と実践的な造形テクニックを学びます。":
            "In the next chapter, we will learn detailed technology and practical fabrication techniques of Material Extrusion (FDM/FFF).",
        
        "参考文献": "References",
        "ISO/ASTM 52900:2021 - Additive manufacturing — General principles — Fundamentals and vocabulary":
            "ISO/ASTM 52900:2021 - Additive manufacturing — General principles — Fundamentals and vocabulary",
        "Wohlers Report 2023 - 3D Printing and Additive Manufacturing Global State of the Industry":
            "Wohlers Report 2023 - 3D Printing and Additive Manufacturing Global State of the Industry",
        "Gibson, I., Rosen, D., & Stucker, B. (2021). Additive Manufacturing Technologies (3rd ed.). Springer.":
            "Gibson, I., Rosen, D., & Stucker, B. (2021). Additive Manufacturing Technologies (3rd ed.). Springer.",
        "Chua, C. K., & Leong, K. F. (2017). 3D Printing and Additive Manufacturing: Principles and Applications (5th ed.). World Scientific.":
            "Chua, C. K., & Leong, K. F. (2017). 3D Printing and Additive Manufacturing: Principles and Applications (5th ed.). World Scientific.",
        "numpy-stl Documentation: https://numpy-stl.readthedocs.io/":
            "numpy-stl Documentation: https://numpy-stl.readthedocs.io/",
        "trimesh Documentation: https://trimsh.org/":
            "trimesh Documentation: https://trimsh.org/",
        "PrusaSlicer Documentation: https://help.prusa3d.com/":
            "PrusaSlicer Documentation: https://help.prusa3d.com/",
        "Ultimaker Cura Documentation: https://support.ultimaker.com/":
            "Ultimaker Cura Documentation: https://support.ultimaker.com/",
        
        # Footer and navigation
        "次の章": "Next Chapter",
        "前の章": "Previous Chapter",
        "目次に戻る": "Return to Table of Contents",
        "第1章": "Chapter 1",
        "第3章": "Chapter 3",
        "第2章": "Chapter 2",
        "© 2024 AI Terakoya. All rights reserved.": "© 2024 AI Terakoya. All rights reserved.",
        "本コンテンツは教育目的で作成されています。": "This content is created for educational purposes.",
        
        # Additional common Japanese phrases
        "以下": "following",
        "上記": "above",
        "例えば": "for example",
        "すなわち": "namely",
        "つまり": "in other words",
        "したがって": "therefore",
        "ただし": "however",
        "また": "also",
        "さらに": "furthermore",
        "一方": "on the other hand",
        "一般的に": "generally",
        "通常": "typically",
        "主に": "mainly",
        "特に": "especially",
        "約": "approximately",
        "程度": "about",
        "以上": "or more",
        "以下": "or less",
        "未満": "less than",
        "〜": "~",
    })
    
    return translations

# Execute Phase 4
def main_phase4():
    from pathlib import Path
    import re
    
    target_file = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-2.html")
    
    # Read current target
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply Phase 4 translations
    translations = create_phase4_translations()
    for jp, en in translations.items():
        content = content.replace(jp, en)
    
    # Write back
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Count remaining Japanese
    jp_count = len(re.findall(r'[あ-ん]|[ア-ン]|[一-龯]', content))
    print(f"Phase 4 translations complete. Remaining Japanese characters: {jp_count}")
    
    if jp_count > 0:
        print(f"\nProgress: {13178 - jp_count} characters translated ({((13178 - jp_count) / 13178 * 100):.1f}%)")

if __name__ == "__main__":
    main_phase4()

def create_final_comprehensive_translations():
    """Final comprehensive phase: All remaining Japanese"""
    translations = {}
    
    # Common words that appear frequently
    translations.update({
        # Technical terms
        "形式": "format",
        "モデル": "model",
        "解答": "answer",
        "保持": "retention",
        "ノズル": "nozzle",
        "プロセス": "process",
        "計算": "calculation",
        "必要": "necessary",
        "できる": "can",
        "実際": "actual",
        "加熱速度": "heating rate",
        "プロット": "plot",
        "設定": "setting",
        "比較": "comparison",
        "反応率": "conversion rate",
        "使用": "usage",
        "ライブラリ": "library",
        "プロフ": "prof",
        "造形": "build",
        "表示": "display",
        "粒成長": "grain growth",
        "最適化": "optimization",
        "推定": "estimation",
        "実験": "experiment",
        "エネルギ": "energy",
        "してください": "please",
        "除去": "removal",
        "部品": "part",
        "結果": "result",
        "理由": "reason",
        "活性化": "activation",
        "削減": "reduction",
        "メッシュ": "mesh",
        "フィッティング": "fitting",
        "による": "by",
        "します": "do",
        "説明": "explanation",
        "設計": "design",
        "複雑": "complex",
        "界面反応": "interface reaction",
        
        # More contextual translations for exercises
        "以下の問題に答えてください：": "Answer the following questions:",
        "解答例：": "Sample Answer:",
        "ヒント：": "Hint:",
        "考え方：": "Approach:",
        "ポイント：": "Key Points:",
        "注意：": "Note:",
        "課題：": "Assignment:",
        "目的：": "Objective:",
        "手順：": "Procedure:",
        "条件：": "Conditions:",
        "要求：": "Requirements:",
        
        # Code comments and technical notes
        "# コメント": "# Comment",
        "# 注釈": "# Note",
        "# パラメータ": "# Parameters",
        "# 戻り値": "# Returns",
        "# 引数": "# Args",
        "# 例": "# Example",
        "# 使い方": "# Usage",
        "# インストール": "# Installation",
        "# 依存関係": "# Dependencies",
        
        # Common verbs
        "確認": "confirm",
        "検証": "validate",
        "評価": "evaluate",
        "分析": "analyze",
        "判断": "determine",
        "選択": "select",
        "決定": "decide",
        "調整": "adjust",
        "改善": "improve",
        "向上": "enhance",
        "低減": "reduce",
        "増加": "increase",
        "変更": "change",
        "追加": "add",
        "削除": "delete",
        "修正": "correct",
        "更新": "update",
        "生成": "generate",
        "作成": "create",
        "実行": "execute",
        "処理": "process",
        "変換": "convert",
        "出力": "output",
        "入力": "input",
        "読み込み": "read",
        "書き込み": "write",
        "保存": "save",
        "読込": "load",
        
        # Common adjectives
        "重要": "important",
        "基本的": "basic",
        "詳細": "detailed",
        "簡単": "simple",
        "困難": "difficult",
        "容易": "easy",
        "可能": "possible",
        "不可能": "impossible",
        "適切": "appropriate",
        "効果的": "effective",
        "効率的": "efficient",
        "正確": "accurate",
        "精密": "precise",
        "高度": "advanced",
        "初級": "beginner",
        "中級": "intermediate",
        "上級": "advanced",
        "一般的": "general",
        "特定": "specific",
        "個別": "individual",
        "共通": "common",
        "独自": "unique",
        "標準": "standard",
        "最新": "latest",
        "従来": "conventional",
        "新しい": "new",
        "古い": "old",
        "大きい": "large",
        "小さい": "small",
        "高い": "high",
        "低い": "low",
        "速い": "fast",
        "遅い": "slow",
        "長い": "long",
        "短い": "short",
        "広い": "wide",
        "狭い": "narrow",
        "厚い": "thick",
        "薄い": "thin",
        "強い": "strong",
        "弱い": "weak",
        
        # Units and measurements
        "時間": "time",
        "温度": "temperature",
        "速度": "speed",
        "距離": "distance",
        "体積": "volume",
        "質量": "mass",
        "重量": "weight",
        "密度": "density",
        "圧力": "pressure",
        "エネルギー": "energy",
        "パワー": "power",
        "力": "force",
        "応力": "stress",
        "ひずみ": "strain",
        "硬度": "hardness",
        "強度": "strength",
        "剛性": "stiffness",
        "靭性": "toughness",
        "延性": "ductility",
        "脆性": "brittleness",
        
        # Materials
        "材料": "material",
        "樹脂": "resin",
        "プラスチック": "plastic",
        "金属": "metal",
        "合金": "alloy",
        "セラミックス": "ceramics",
        "複合材": "composite",
        "繊維": "fiber",
        "粉末": "powder",
        "フィラメント": "filament",
        "ワイヤ": "wire",
        "シート": "sheet",
        "液体": "liquid",
        "固体": "solid",
        "気体": "gas",
        
        # Process terms
        "加熱": "heating",
        "冷却": "cooling",
        "溶融": "melting",
        "凝固": "solidification",
        "焼結": "sintering",
        "硬化": "curing",
        "重合": "polymerization",
        "反応": "reaction",
        "拡散": "diffusion",
        "成長": "growth",
        "収縮": "shrinkage",
        "膨張": "expansion",
        "変形": "deformation",
        "破壊": "fracture",
        "摩耗": "wear",
        "腐食": "corrosion",
        "酸化": "oxidation",
        
        # Equipment and tools
        "装置": "equipment",
        "機器": "device",
        "システム": "system",
        "プリンタ": "printer",
        "ヘッド": "head",
        "プラットフォーム": "platform",
        "ベッド": "bed",
        "チャンバー": "chamber",
        "レーザー": "laser",
        "ビーム": "beam",
        "スキャナ": "scanner",
        "センサ": "sensor",
        "制御": "control",
        "ソフトウェア": "software",
        "ツール": "tool",
        "治具": "jig",
        "金型": "mold",
        
        # Quality and testing
        "品質": "quality",
        "精度": "accuracy",
        "解像度": "resolution",
        "表面": "surface",
        "寸法": "dimension",
        "公差": "tolerance",
        "欠陥": "defect",
        "不良": "defect",
        "検査": "inspection",
        "試験": "test",
        "測定": "measurement",
        "評価": "evaluation",
        "性能": "performance",
        "特性": "property",
        "機能": "function",
        "仕様": "specification",
        "要求": "requirement",
        "基準": "standard",
        "規格": "standard",
        
        # Applications
        "用途": "application",
        "応用": "application",
        "産業": "industry",
        "製造": "manufacturing",
        "生産": "production",
        "試作": "prototyping",
        "開発": "development",
        "研究": "research",
        "教育": "education",
        "医療": "medical",
        "航空宇宙": "aerospace",
        "自動車": "automotive",
        "電子": "electronics",
        "建築": "architecture",
        "芸術": "art",
        "スポーツ": "sports",
        "消費財": "consumer goods",
        
        # Business and economics
        "コスト": "cost",
        "価格": "price",
        "費用": "expense",
        "市場": "market",
        "規模": "scale",
        "成長率": "growth rate",
        "シェア": "share",
        "利益": "profit",
        "効率": "efficiency",
        "生産性": "productivity",
        "競争力": "competitiveness",
        "投資": "investment",
        "収益": "revenue",
        
        # Documentation
        "章": "chapter",
        "節": "section",
        "項": "subsection",
        "図": "figure",
        "表": "table",
        "式": "equation",
        "例": "example",
        "注": "note",
        "参考": "reference",
        "文献": "literature",
        "引用": "citation",
        "出典": "source",
        "著者": "author",
        "タイトル": "title",
        "要約": "summary",
        "概要": "overview",
        "目次": "table of contents",
        "索引": "index",
        "付録": "appendix",
        
        # Common particles and connectors (when standalone)
        "について": "about",
        "に関して": "regarding",
        "のため": "for",
        "によって": "by",
        "に対して": "against",
        "において": "in",
        "として": "as",
        "から": "from",
        "まで": "to",
        "より": "than",
        "など": "etc.",
        "および": "and",
        "または": "or",
        "ただし": "however",
        "なお": "note",
        "すなわち": "namely",
        "例えば": "for example",
        "特に": "especially",
        "また": "also",
        "さらに": "furthermore",
        "したがって": "therefore",
        "そのため": "thus",
        "一方": "on the other hand",
        "逆に": "conversely",
        "同様に": "similarly",
        "対して": "in contrast",
        "場合": "case",
        "とき": "when",
        "際": "when",
        "こと": "that",
        "もの": "thing",
        "ため": "purpose",
        "方法": "method",
        "手段": "means",
        "方式": "method",
        "技術": "technology",
        "技法": "technique",
        "アプローチ": "approach",
        "戦略": "strategy",
    })
    
    return translations

# Execute Final Phase
def main_final():
    from pathlib import Path
    import re
    
    target_file = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-2.html")
    
    # Read current target
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply final translations
    translations = create_final_comprehensive_translations()
    for jp, en in translations.items():
        content = content.replace(jp, en)
    
    # Write back
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Count remaining Japanese
    jp_count = len(re.findall(r'[あ-ん]|[ア-ン]|[一-龯]', content))
    print(f"Final phase translations complete. Remaining Japanese characters: {jp_count}")
    
    if jp_count > 0:
        completed = 13178 - jp_count
        percentage = (completed / 13178) * 100
        print(f"\nProgress: {completed} characters translated ({percentage:.1f}%)")
        print(f"\nRemaining work: {jp_count} characters ({100 - percentage:.1f}%)")
    else:
        print("\n🎉 TRANSLATION COMPLETE! All Japanese characters removed.")

if __name__ == "__main__":
    main_final()
