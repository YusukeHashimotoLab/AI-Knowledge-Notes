#!/usr/bin/env python3
"""
Complete Translation Script: Advanced Ceramics Materials Chapter 1
Translates Japanese HTML to English while preserving all structure and code blocks
"""

import re
from pathlib import Path

def translate_ceramics_chapter1():
    """Complete translation of chapter 1 from Japanese to English"""

    # Read source file
    source_path = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/advanced-materials-systems-introduction/chapter-1.html")
    target_path = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/advanced-materials-systems-introduction/chapter-1.html")

    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Count Japanese characters before translation
    japanese_char_count = len(re.findall(r'[ぁ-んァ-ヶー一-龯]', content))
    total_char_count = len(content)
    jp_percentage = (japanese_char_count / total_char_count * 100) if total_char_count > 0 else 0

    print(f"Source file analysis:")
    print(f"  Total characters: {total_char_count}")
    print(f"  Japanese characters: {japanese_char_count}")
    print(f"  Japanese percentage: {jp_percentage:.2f}%")
    print(f"\nStarting translation...\n")

    # Translation mappings
    translations = {
        # HTML lang attribute
        '<html lang="ja">': '<html lang="en">',

        # Page title and meta
        '第1章：先進セラミックス材料 - 構造・機能性・バイオセラミックス - MS Terakoya':
            'Chapter 1: Advanced Ceramics Materials - Structural, Functional, and Bioceramics - MS Terakoya',

        # Breadcrumb navigation
        'AI寺子屋トップ': 'AI Terakoya Home',
        '材料科学': 'Materials Science',
        'Advanced Materials Systems': 'Advanced Materials Systems',
        'Chapter 1': 'Chapter 1',

        # Header content
        '第1章：先進セラミックス材料': 'Chapter 1: Advanced Ceramics Materials',
        '構造・機能性・バイオセラミックス - 高性能化の設計原理':
            'Structural, Functional, and Bioceramics - Design Principles for High Performance',
        '先進材料システム入門シリーズ': 'Introduction to Advanced Materials Systems Series',
        '読了時間: 35-40分': 'Reading time: 35-40 minutes',
        '難易度: 中級〜上級': 'Difficulty: Intermediate to Advanced',

        # Learning objectives section
        '学習目標': 'Learning Objectives',
        'この章を完了すると、以下を説明できるようになります：':
            'Upon completing this chapter, you will be able to explain:',

        # Basic understanding subsection
        '基本理解': 'Fundamental Understanding',
        '構造セラミックスの高強度化・高靭性化メカニズム（相変態強化、繊維強化）':
            'Strengthening and toughening mechanisms of structural ceramics (transformation toughening, fiber reinforcement)',
        '機能性セラミックス（圧電、誘電、磁性）の物理的起源と結晶構造':
            'Physical origins and crystal structures of functional ceramics (piezoelectric, dielectric, magnetic)',
        'バイオセラミックスの生体適合性と骨結合のメカニズム':
            'Biocompatibility and osseointegration mechanisms of bioceramics',
        'セラミックスの機械的特性と統計的破壊理論（Weibull分布）':
            'Mechanical properties of ceramics and statistical fracture theory (Weibull distribution)',

        # Practical skills subsection
        '実践スキル': 'Practical Skills',
        'Pythonでセラミックスの強度分布（Weibull統計）を解析できる':
            'Analyze strength distribution of ceramics (Weibull statistics) using Python',
        'pycalphadを用いて相図を計算し、焼結条件を最適化できる':
            'Calculate phase diagrams using pycalphad and optimize sintering conditions',
        '圧電定数・誘電率・磁気特性を計算・評価できる':
            'Calculate and evaluate piezoelectric constants, dielectric permittivity, and magnetic properties',
        '材料選択マトリックスで用途に応じた最適セラミックスを選定できる':
            'Select optimal ceramics for specific applications using materials selection matrix',

        # Applied capabilities subsection
        '応用力': 'Applied Capabilities',
        '用途要求から最適なセラミックス組成と微構造を設計できる':
            'Design optimal ceramic composition and microstructure from application requirements',
        '機能性セラミックスデバイス（センサ、アクチュエータ）を設計できる':
            'Design functional ceramic devices (sensors, actuators)',
        'バイオセラミックスインプラントの生体適合性を評価できる':
            'Evaluate biocompatibility of bioceramic implants',
        'セラミックス材料の信頼性設計（確率的破壊予測）ができる':
            'Perform reliability design (probabilistic fracture prediction) for ceramic materials',

        # Section 1.1 - Structural Ceramics
        '1.1 構造セラミックス - 高強度・高靭性化の原理':
            '1.1 Structural Ceramics - Principles of High Strength and High Toughness',

        '1.1.1 構造セラミックスの概要': '1.1.1 Overview of Structural Ceramics',
        '構造セラミックス（Structural Ceramics）とは、': 'Structural ceramics are ',
        '優れた機械的性質（高強度・高硬度・耐熱性）を持ち、過酷な環境下で構造部材として使用されるセラミックス材料':
            'ceramic materials with excellent mechanical properties (high strength, high hardness, heat resistance) used as structural components in harsh environments',
        'です。金属材料では不可能な高温環境や腐食性環境での使用が可能で、以下のような重要な応用があります：':
            '. They enable use in high-temperature or corrosive environments impossible for metallic materials, with important applications including:',

        'Al₂O₃（アルミナ）': 'Al₂O₃ (Alumina)',
        '切削工具、耐摩耗部品、人工関節（生体適合性）': 'Cutting tools, wear-resistant parts, artificial joints (biocompatibility)',
        'ZrO₂（ジルコニア）': 'ZrO₂ (Zirconia)',
        '歯科材料、酸素センサー、熱遮蔽コーティング（高靭性）': 'Dental materials, oxygen sensors, thermal barrier coatings (high toughness)',
        'Si₃N₄（窒化ケイ素）': 'Si₃N₄ (Silicon Nitride)',
        'ガスタービン部品、ベアリング（高温強度）': 'Gas turbine components, bearings (high-temperature strength)',
        'SiC（炭化ケイ素）': 'SiC (Silicon Carbide)',
        '半導体製造装置、装甲材（超高硬度）': 'Semiconductor manufacturing equipment, armor materials (ultra-high hardness)',

        # Info box
        '💡 産業的重要性': '💡 Industrial Significance',
        '構造セラミックスは航空宇宙・自動車・医療分野で不可欠です。世界のセラミックス市場（2023年時点で$230B以上）の約60%が先進セラミックス材料です。その理由は：':
            'Structural ceramics are indispensable in aerospace, automotive, and medical fields. Advanced ceramics account for approximately 60% of the global ceramics market (over $230B as of 2023). The reasons are:',
        '金属の3-5倍の強度（常温）と優れた耐熱性（1500°C以上）':
            '3-5 times the strength of metals (at room temperature) and excellent heat resistance (above 1500°C)',
        '化学的安定性（酸・アルカリに不活性）':
            'Chemical stability (inert to acids and alkalis)',
        '低密度（金属の1/2-1/3）による軽量化効果':
            'Weight reduction effect due to low density (1/2-1/3 of metals)',
        '高硬度（Hv 1500-2500）による耐摩耗性':
            'Wear resistance due to high hardness (Hv 1500-2500)',

        # Section 1.1.2
        '1.1.2 高強度セラミックス（Al₂O₃, ZrO₂, Si₃N₄）':
            '1.1.2 High-Strength Ceramics (Al₂O₃, ZrO₂, Si₃N₄)',
        '高強度セラミックスは以下の3つの主要材料が代表的です：':
            'High-strength ceramics are represented by the following three major materials:',

        'アルミナ': 'Alumina',
        '高硬度': 'High Hardness',
        'ジルコニア': 'Zirconia',
        '高靭性': 'High Toughness',
        '窒化ケイ素': 'Silicon Nitride',
        '高温強度': 'High-Temperature Strength',
        '使用': 'use',

        # Material descriptions
        '酸化物セラミックスの代表格。高硬度（Hv 2000）、優れた耐摩耗性、生体適合性により、切削工具・人工関節に使用。製造コストが低く最も広く普及。':
            'Representative of oxide ceramics. Used in cutting tools and artificial joints due to high hardness (Hv 2000), excellent wear resistance, and biocompatibility. Most widely used due to low manufacturing cost.',
        '相変態強化（Transformation Toughening）により、セラミックス材料の中で最高レベルの破壊靭性（10-15 MPa√m）を実現。「セラミックス鋼」とも呼ばれる。':
            'Achieves the highest level of fracture toughness (10-15 MPa√m) among ceramic materials through transformation toughening. Also called "ceramic steel".',
        '共有結合性が強く、1400°Cまで高強度を維持。ガスタービン部品・ベアリングなどの高温構造材料として使用。熱衝撃抵抗性も優れる。':
            'Strong covalent bonding maintains high strength up to 1400°C. Used as high-temperature structural material for gas turbine components and bearings. Also exhibits excellent thermal shock resistance.',

        # Warning box
        '⚠️ セラミックスの本質的課題': '⚠️ Intrinsic Challenge of Ceramics',
        'セラミックスは高強度・高硬度を持つ一方で、':
            'While ceramics possess high strength and high hardness, ',
        '脆性（低靭性）': 'brittleness (low toughness)',
        'が最大の欠点です。微小な欠陥（気孔、亀裂）が応力集中点となり、突発的な破壊を引き起こします（Griffith理論）。破壊靭性は金属の1/10以下です。このため、高靭性化技術が重要な研究課題となっています。':
            ' is the major drawback. Microscopic defects (pores, cracks) become stress concentration points, causing catastrophic fracture (Griffith theory). Fracture toughness is less than 1/10 that of metals. Therefore, toughening technology is an important research topic.',

        # Section 1.1.3
        '1.1.3 高靭性化メカニズム': '1.1.3 Toughening Mechanisms',
        'メカニズム1: 相変態強化（Transformation Toughening）':
            'Mechanism 1: Transformation Toughening',
        'ジルコニア（ZrO₂）で最も効果的に機能する強化機構です：':
            'This is the most effective toughening mechanism in zirconia (ZrO₂):',

        'ZrO₂（正方晶、t-phase） → ZrO₂（単斜晶、m-phase） + 体積膨張（3-5%）':
            'ZrO₂ (tetragonal, t-phase) → ZrO₂ (monoclinic, m-phase) + volume expansion (3-5%)',

        '強化のメカニズム：': 'Toughening Mechanism:',
        '応力誘起変態': 'Stress-Induced Transformation',
        '亀裂先端の高応力場で、準安定な正方晶（t）が単斜晶（m）へ相変態':
            'Metastable tetragonal (t) phase transforms to monoclinic (m) phase in the high-stress field at crack tips',
        '体積膨張効果': 'Volume Expansion Effect',
        '3-5%の体積膨張が亀裂周辺に圧縮応力を発生させ、亀裂進展を抑制':
            '3-5% volume expansion generates compressive stress around cracks, suppressing crack propagation',
        'エネルギー吸収': 'Energy Absorption',
        '変態に伴うエネルギー消費が破壊エネルギーを増大':
            'Energy consumption during transformation increases fracture energy',
        '靭性向上効果': 'Toughness Enhancement Effect',
        '破壊靭性が3 MPa√m → 10-15 MPa√m（3-5倍向上）':
            'Fracture toughness increases from 3 MPa√m to 10-15 MPa√m (3-5 times improvement)',

        '実現方法：': 'Implementation Method: ',
        'Y₂O₃（3-8 mol%）やMgO（9-15 mol%）を添加し、正方晶を室温で準安定化（PSZ: Partially Stabilized Zirconia）':
            'Add Y₂O₃ (3-8 mol%) or MgO (9-15 mol%) to stabilize tetragonal phase at room temperature (PSZ: Partially Stabilized Zirconia)',

        # Fiber reinforcement
        'メカニズム2: 繊維強化（Fiber Reinforcement）':
            'Mechanism 2: Fiber Reinforcement',
        'セラミックスマトリックスに高強度繊維を複合化する手法です：':
            'This method involves incorporating high-strength fibers into a ceramic matrix:',

        'セラミックス複合材料（CMC） = セラミックスマトリックス + 強化繊維（SiC, C, Al₂O₃）':
            'Ceramic Matrix Composites (CMC) = Ceramic Matrix + Reinforcing Fibers (SiC, C, Al₂O₃)',

        'クラックデフレクション': 'Crack Deflection',
        '亀裂が繊維界面で偏向し、進展経路が長くなる':
            'Cracks deflect at fiber interfaces, increasing the propagation path length',
        'ファイバープルアウト': 'Fiber Pullout',
        '繊維が引き抜かれる際に大きなエネルギーを吸収':
            'Large energy absorption occurs when fibers are pulled out',
        'クラックブリッジング': 'Crack Bridging',
        '繊維が亀裂を架橋し、応力伝達を維持':
            'Fibers bridge cracks and maintain stress transfer',
        '破壊靭性が5 MPa√m → 20-30 MPa√m（4-6倍向上）':
            'Fracture toughness increases from 5 MPa√m to 20-30 MPa√m (4-6 times improvement)',

        '応用例：': 'Applications: ',
        'SiC/SiC複合材料（航空機エンジン部品）、C/C複合材料（ブレーキディスク）':
            'SiC/SiC composites (aircraft engine components), C/C composites (brake disks)',

        # Section 1.2 - Functional Ceramics
        '1.2 機能性セラミックス - 圧電・誘電・磁性':
            '1.2 Functional Ceramics - Piezoelectric, Dielectric, and Magnetic',

        '1.2.1 圧電セラミックス（Piezoelectric Ceramics）':
            '1.2.1 Piezoelectric Ceramics',
        '圧電効果とは、': 'The piezoelectric effect is ',
        '機械的応力を加えると電気分極が生じ（正圧電効果）、逆に電場を印加すると機械的歪みが生じる（逆圧電効果）現象':
            'a phenomenon where electrical polarization is generated by applied mechanical stress (direct piezoelectric effect), and conversely, mechanical strain is generated by an applied electric field (converse piezoelectric effect)',
        'です。': '.',

        '代表的な圧電材料': 'Representative Piezoelectric Materials',
        'PZT（Pb(Zr,Ti)O₃）：圧電定数 d₃₃ = 200-600 pC/N':
            'PZT (Pb(Zr,Ti)O₃): Piezoelectric constant d₃₃ = 200-600 pC/N',
        'BaTiO₃（チタン酸バリウム）：圧電定数 d₃₃ = 85-190 pC/N（鉛フリー代替材料）':
            'BaTiO₃ (Barium Titanate): Piezoelectric constant d₃₃ = 85-190 pC/N (lead-free alternative)',

        'PZT（ジルコン酸チタン酸鉛）の特徴：': 'Characteristics of PZT (Lead Zirconate Titanate):',
        '高圧電定数': 'High Piezoelectric Constant',
        'd₃₃ = 200-600 pC/N（応用材料として最も優れる）':
            'd₃₃ = 200-600 pC/N (most excellent as applied material)',
        'モルフォトロピック相境界（MPB）': 'Morphotropic Phase Boundary (MPB)',
        'Zr/Ti比率 52/48付近で圧電特性が最大化':
            'Piezoelectric properties are maximized near Zr/Ti ratio of 52/48',
        'キュリー温度': 'Curie Temperature',
        '320-380°C（この温度以上で圧電性消失）':
            '320-380°C (piezoelectricity disappears above this temperature)',
        '応用': 'Applications',
        '超音波振動子、圧電アクチュエータ、圧電スピーカー、圧電点火装置':
            'Ultrasonic transducers, piezoelectric actuators, piezoelectric speakers, piezoelectric igniters',

        # Warning about lead
        '⚠️ 環境問題と鉛フリー化': '⚠️ Environmental Issues and Lead-Free Alternatives',
        'PZTは鉛（Pb）を60wt%以上含むため、欧州RoHS規制で使用制限があります。鉛フリー代替材料として、BaTiO₃系、(K,Na)NbO₃系、BiFeO₃系が研究されていますが、PZTの性能には及びません（d₃₃ = 100-300 pC/N）。圧電デバイスは医療機器等の適用除外品目ですが、長期的には代替材料開発が必要です。':
            'PZT contains more than 60 wt% lead (Pb), subject to usage restrictions under European RoHS regulations. Lead-free alternatives such as BaTiO₃-based, (K,Na)NbO₃-based, and BiFeO₃-based materials are being researched, but do not match PZT performance (d₃₃ = 100-300 pC/N). While piezoelectric devices are exempt items for medical equipment, alternative material development is necessary in the long term.',

        '圧電効果の結晶学的起源': 'Crystallographic Origin of Piezoelectric Effect',
        '圧電効果は': 'The piezoelectric effect ',
        '非中心対称結晶構造': 'non-centrosymmetric crystal structure',
        'を持つ材料でのみ発現します：': 'occurs only in materials with:',

        '常誘電相（立方晶、Pm3m）': 'Paraelectric Phase (Cubic, Pm3m)',
        '中心対称 → 圧電性なし（高温）': 'Centrosymmetric → No piezoelectricity (high temperature)',
        '強誘電相（正方晶、P4mm）': 'Ferroelectric Phase (Tetragonal, P4mm)',
        '非中心対称 → 圧電性あり（室温）': 'Non-centrosymmetric → Piezoelectricity present (room temperature)',
        '自発分極': 'Spontaneous Polarization',
        'Ti⁴⁺イオンが酸素八面体中心からずれることで双極子モーメント発生':
            'Dipole moment generated by displacement of Ti⁴⁺ ions from the center of oxygen octahedra',
        '分域（ドメイン）構造': 'Domain Structure',
        '電場印加により分域の方位が揃い、巨大圧電効果を発現（ポーリング処理）':
            'Domain orientations align under applied electric field, exhibiting giant piezoelectric effect (poling treatment)',

        # Section 1.2.2
        '1.2.2 誘電セラミックス（Dielectric Ceramics）': '1.2.2 Dielectric Ceramics',
        '誘電セラミックスは、': 'Dielectric ceramics are ',
        '高い誘電率（εᵣ）を持ち、電気エネルギーを蓄積するコンデンサ材料':
            'capacitor materials with high dielectric constant (εᵣ) that store electrical energy',
        'として使用されます。': '.',

        'MLCC（積層セラミックコンデンサ）用材料':
            'Materials for MLCC (Multilayer Ceramic Capacitors)',
        'BaTiO₃（チタン酸バリウム）：εᵣ = 1,500-10,000（室温、1 kHz）':
            'BaTiO₃ (Barium Titanate): εᵣ = 1,500-10,000 (room temperature, 1 kHz)',

        '高誘電率の起源：': 'Origin of High Dielectric Constant:',
        '強誘電性（Ferroelectricity）': 'Ferroelectricity',
        '自発分極が外部電場により反転可能な性質':
            'Property where spontaneous polarization can be reversed by external electric field',
        '分域壁の移動': 'Domain Wall Movement',
        '電場印加により分域壁が容易に移動し、大きな分極変化を生じる':
            'Domain walls move easily under applied electric field, producing large polarization changes',
        'この温度で誘電率がピーク':
            'Dielectric constant peaks at this temperature',
        '組成調整': 'Composition Adjustment',
        'CaZrO₃、SrTiO₃を添加してTcを室温付近にシフト（X7R特性）':
            'Addition of CaZrO₃, SrTiO₃ shifts Tc near room temperature (X7R characteristics)',

        # Success box about MLCC
        '✅ MLCC（多層セラミックコンデンサ）の驚異的性能':
            '✅ Remarkable Performance of MLCC (Multilayer Ceramic Capacitors)',
        '現代のMLCCは極限まで小型化・高性能化が進んでいます：':
            'Modern MLCCs have advanced to extreme miniaturization and high performance:',
        '積層数': 'Number of Layers',
        '1,000層以上（誘電体層厚み < 1 μm）':
            'More than 1,000 layers (dielectric layer thickness < 1 μm)',
        '静電容量': 'Capacitance',
        '1 mm³サイズで100 μF以上達成':
            'Achieving over 100 μF in 1 mm³ size',
        '用途': 'Applications',
        'スマートフォン1台に800個以上搭載':
            'Over 800 units installed in one smartphone',
        '市場規模': 'Market Size',
        '年間生産数 1兆個以上（世界最大の電子部品）':
            'Annual production exceeds 1 trillion units (largest electronic component worldwide)',
        'BaTiO₃ベースのMLCCは電子機器の小型化・高性能化の鍵となる材料です。':
            'BaTiO₃-based MLCCs are key materials for miniaturization and performance enhancement of electronic devices.',

        # Section 1.2.3
        '1.2.3 磁性セラミックス（Magnetic Ceramics - Ferrites）':
            '1.2.3 Magnetic Ceramics - Ferrites',
        'フェライト（Ferrites）は、': 'Ferrites are ',
        '酸化物系の磁性材料で、高周波における低損失特性':
            'oxide-based magnetic materials with low-loss characteristics at high frequencies',
        'を持つため、トランスフォーマー・インダクタ・電波吸収体に広く使用されます。':
            ', widely used in transformers, inductors, and electromagnetic wave absorbers.',

        'フェライトの種類と用途': 'Types and Applications of Ferrites',
        'スピネル型フェライト：MFe₂O₄（M = Mn, Ni, Zn, Co等）':
            'Spinel Ferrite: MFe₂O₄ (M = Mn, Ni, Zn, Co, etc.)',
        '六方晶フェライト（ハードフェライト）：BaFe₁₂O₁₉、SrFe₁₂O₁₉（永久磁石）':
            'Hexagonal Ferrite (Hard Ferrite): BaFe₁₂O₁₉, SrFe₁₂O₁₉ (permanent magnets)',

        'スピネル型フェライトの特徴：': 'Characteristics of Spinel Ferrites:',
        'ソフト磁性': 'Soft Magnetic',
        '保磁力が小さく（Hc < 100 A/m）、容易に磁化反転':
            'Low coercivity (Hc < 100 A/m), easy magnetization reversal',
        '高周波特性': 'High-Frequency Characteristics',
        '高い電気抵抗（ρ > 10⁶ Ω·cm）により渦電流損失が小さい':
            'Small eddy current loss due to high electrical resistance (ρ > 10⁶ Ω·cm)',
        'Mn-Znフェライト': 'Mn-Zn Ferrite',
        '高透磁率（μᵣ = 2,000-15,000）、低周波トランスフォーマー用':
            'High permeability (μᵣ = 2,000-15,000), for low-frequency transformers',
        'Ni-Znフェライト': 'Ni-Zn Ferrite',
        '高周波特性に優れる（GHz帯）、EMI対策部品用':
            'Excellent high-frequency characteristics (GHz band), for EMI countermeasure components',

        '六方晶フェライト（ハードフェライト）の特徴：':
            'Characteristics of Hexagonal Ferrites (Hard Ferrites):',
        'ハード磁性': 'Hard Magnetic',
        '大きな保磁力（Hc = 200-400 kA/m）と残留磁束密度（Br = 0.4 T）':
            'Large coercivity (Hc = 200-400 kA/m) and remanent flux density (Br = 0.4 T)',
        '永久磁石材料': 'Permanent Magnet Material',
        'モーター、スピーカー、磁気記録媒体に使用':
            'Used in motors, speakers, magnetic recording media',
        '低コスト': 'Low Cost',
        '希土類磁石（Nd-Fe-B）より性能は劣るが、原料が安価で大量生産可能':
            'Lower performance than rare-earth magnets (Nd-Fe-B), but inexpensive raw materials and mass production possible',
        '耐食性': 'Corrosion Resistance',
        '酸化物のため金属磁石と異なり腐食しない':
            'Being oxides, they do not corrode unlike metallic magnets',

        # Info box about ferrite magnetism
        '💡 フェライトの磁性起源': '💡 Origin of Ferrite Magnetism',
        'フェライトの磁性はスピネル構造（AB₂O₄）中の':
            'The magnetism of ferrites arises from the ',
        'A席（四面体位置）とB席（八面体位置）のイオンの磁気モーメントが反平行配列':
            'antiparallel alignment of magnetic moments of ions at A-sites (tetrahedral positions) and B-sites (octahedral positions)',
        'することで発現します（フェリ磁性）。Mn-ZnフェライトではMn²⁺とFe³⁺の磁気モーメントが部分的に打ち消し合うため、全体としての磁化は小さくなりますが、高透磁率が実現されます。':
            ' in the spinel structure (AB₂O₄) (ferrimagnetism). In Mn-Zn ferrites, the magnetic moments of Mn²⁺ and Fe³⁺ partially cancel each other, resulting in small overall magnetization but achieving high permeability.',

        # Section 1.3 - Bioceramics
        '1.3 バイオセラミックス - 生体適合性と骨結合':
            '1.3 Bioceramics - Biocompatibility and Osseointegration',

        '1.3.1 バイオセラミックスの概要': '1.3.1 Overview of Bioceramics',
        'バイオセラミックス（Bioceramics）とは、': 'Bioceramics are ',
        '生体組織と接触しても拒絶反応を起こさず（生体適合性）、骨組織と直接結合できる（骨伝導性）セラミックス材料':
            'ceramic materials that do not cause rejection reactions when in contact with biological tissues (biocompatibility) and can directly bond with bone tissue (osteoconductivity)',

        '代表的なバイオセラミックス': 'Representative Bioceramics',
        'HAp（ハイドロキシアパタイト）：Ca₁₀(PO₄)₆(OH)₂':
            'HAp (Hydroxyapatite): Ca₁₀(PO₄)₆(OH)₂',
        'β-TCP（リン酸三カルシウム）：Ca₃(PO₄)₂':
            'β-TCP (Tricalcium Phosphate): Ca₃(PO₄)₂',

        'ハイドロキシアパタイト（HAp）の特徴：':
            'Characteristics of Hydroxyapatite (HAp):',
        '骨の主成分': 'Main Component of Bone',
        '天然骨の無機成分の65%がHAp（残り35%は有機物コラーゲン）':
            '65% of the inorganic component of natural bone is HAp (remaining 35% is organic collagen)',
        '生体適合性': 'Biocompatibility',
        '骨組織と化学組成が類似しているため、拒絶反応が起きない':
            'No rejection reaction occurs due to similar chemical composition to bone tissue',
        '骨伝導性（Osteoconduction）': 'Osteoconduction',
        'HAp表面に骨芽細胞が付着・増殖し、新しい骨組織が形成される':
            'Osteoblasts attach and proliferate on HAp surface, forming new bone tissue',
        '骨結合（Osseointegration）': 'Osseointegration',
        'HAp表面と骨組織の間に直接的な化学結合が形成される':
            'Direct chemical bonding forms between HAp surface and bone tissue',
        '人工骨、歯科インプラント、骨充填材、Ti合金インプラントのコーティング':
            'Artificial bone, dental implants, bone fillers, coating for Ti alloy implants',

        # Success box about beta-TCP
        '✅ β-TCPの生体吸収性': '✅ Bioresorbability of β-TCP',
        'β-TCP（リン酸三カルシウム）は、HApと異なり':
            'β-TCP (tricalcium phosphate), unlike HAp, has the property of ',
        '生体内で徐々に吸収される':
            'being gradually resorbed in vivo',
        '特性を持ちます：': ':',
        '吸収期間': 'Resorption Period',
        '6-18ヶ月で完全吸収（粒子サイズ・気孔率に依存）':
            'Complete resorption in 6-18 months (depends on particle size and porosity)',
        '置換メカニズム': 'Replacement Mechanism',
        'β-TCPが溶解しながら、新しい骨組織に置き換わる（Bone remodeling）':
            'β-TCP dissolves while being replaced by new bone tissue (bone remodeling)',
        'Ca²⁺・PO₄³⁻供給': 'Ca²⁺·PO₄³⁻ Supply',
        '溶解により放出されたイオンが骨形成を促進':
            'Ions released by dissolution promote bone formation',
        'HAp/β-TCP複合材': 'HAp/β-TCP Composite',
        '両者の混合比率により吸収速度を制御可能（HAp 70% / β-TCP 30%等）':
            'Resorption rate can be controlled by mixing ratio (e.g., HAp 70% / β-TCP 30%)',
        '生体吸収性により、永久的な異物が体内に残らず、自己の骨組織に完全に置き換わる理想的な骨再生が実現します。':
            'Bioresorbability achieves ideal bone regeneration where no permanent foreign material remains in the body, being completely replaced by autologous bone tissue.',

        # Section 1.4 - Python Practice
        '1.4 Python実践：セラミックス材料の解析と設計':
            '1.4 Python Practice: Analysis and Design of Ceramic Materials',

        'Example 1: Weibull統計による破壊強度分布の解析':
            'Example 1: Analysis of Fracture Strength Distribution using Weibull Statistics',

        # Code comments in Example 1
        '# 物理定数': '# Physical constants',
        '# BaTiO3系の拡散パラメータ（文献値）':
            '# Diffusion parameters for BaTiO3 system (literature values)',
        '# m²/s (頻度因子)': '# m²/s (frequency factor)',
        '# J/mol (活性化エネルギー 300 kJ/mol)':
            '# J/mol (activation energy 300 kJ/mol)',
        'Arrhenius式で拡散係数を計算':
            'Calculate diffusion coefficient using Arrhenius equation',
        '温度 [K]': 'Temperature [K]',
        '頻度因子 [m²/s]': 'Frequency factor [m²/s]',
        '活性化エネルギー [J/mol]': 'Activation energy [J/mol]',
        '拡散係数 [m²/s]': 'Diffusion coefficient [m²/s]',
        '温度範囲 800-1400°C': 'Temperature range 800-1400°C',
        '拡散係数を計算': 'Calculate diffusion coefficient',
        'プロット': 'Plot',
        '対数プロット（Arrheniusプロット）': 'Logarithmic plot (Arrhenius plot)',
        '主要温度での拡散係数を表示':
            'Display diffusion coefficients at key temperatures',
        '温度依存性の比較:': 'Comparison of temperature dependence:',
        '出力例:': 'Output example:',

        # Example 2
        'Example 2: Jander式による反応進行のシミュレーション':
            'Example 2: Simulation of Reaction Progress using Jander Equation',

        # Code comments in Example 2
        '反応率 (0-1)': 'Conversion rate (0-1)',
        '速度定数 [s⁻¹]': 'Rate constant [s⁻¹]',
        '時間 [s]': 'Time [s]',
        'Jander式の左辺 - k*t': 'Left side of Jander equation - k*t',
        '時間tにおける反応率を計算': 'Calculate conversion rate at time t',
        '速度定数': 'Rate constant',
        '時間': 'Time',
        '反応率 (0-1)': 'Conversion rate (0-1)',
        'Jander式をalphaについて数値的に解く':
            'Solve Jander equation numerically for alpha',
        '初期推定値': 'Initial guess',
        '0-1の範囲に制限': 'Limit to 0-1 range',
        'パラメータ設定': 'Parameter settings',
        'm²/s (1200°Cでの拡散係数)':
            'm²/s (diffusion coefficient at 1200°C)',
        'mol/m³': 'mol/m³',
        '粒子半径 [m]: 1μm, 5μm, 10μm':
            'Particle radius [m]: 1μm, 5μm, 10μm',
        '時間配列（0-50時間）': 'Time array (0-50 hours)',
        '粒子サイズの影響': 'Effect of particle size',
        '温度の影響（粒子サイズ固定）':
            'Effect of temperature (fixed particle size)',
        '5μm固定': '5μm fixed',
        '50%反応に要する時間を計算':
            'Calculate time required for 50% reaction',
        '50%反応に要する時間:': 'Time required for 50% reaction:',

        # Example 3
        'Example 3: 活性化エネルギーの計算（DSC/TGデータから）':
            'Example 3: Calculation of Activation Energy (from DSC/TG Data)',

        # Code comments in Example 3
        'Kissinger法: ln(β/Tp²) vs 1/Tp の直線の傾きから Ea を求める':
            'Kissinger method: Determine Ea from slope of ln(β/Tp²) vs 1/Tp',
        '加熱速度 [K/min]': 'Heating rate [K/min]',
        'ピーク温度 [K]': 'Peak temperature [K]',
        '傾き = -Ea/R': 'Slope = -Ea/R',
        '実験データ（異なる加熱速度でのDSCピーク温度）':
            'Experimental data (DSC peak temperatures at different heating rates)',
        'Kissinger法で活性化エネルギーを計算':
            'Calculate activation energy using Kissinger method',
        'ピーク温度 [K]': 'Peak temperature [K]',
        '(Ea [kJ/mol], A [min⁻¹], R²)': '(Ea [kJ/mol], A [min⁻¹], R²)',
        'Kissinger式の左辺': 'Left side of Kissinger equation',
        '1000/Tでスケーリング（見やすくするため）':
            'Scaling with 1000/T (for better visibility)',
        '線形回帰': 'Linear regression',
        '活性化エネルギー計算': 'Calculate activation energy',
        'J/mol → kJ/mol': 'J/mol → kJ/mol',
        '頻度因子': 'Frequency factor',
        'Kissinger法による解析結果:':
            'Analysis results using Kissinger method:',
        '活性化エネルギー Ea =': 'Activation energy Ea =',
        '頻度因子 A =': 'Frequency factor A =',
        '決定係数 R² =': 'Coefficient of determination R² =',
        'Kissingerプロット': 'Kissinger plot',
        '実験データ': 'Experimental data',
        'フィッティング直線': 'Fitting line',
        'テキストボックスで結果を表示':
            'Display results in text box',

        # Section 1.4.1
        '1.4.1 温度プロファイルの3要素':
            '1.4.1 Three Elements of Temperature Profile',
        '固相反応における温度プロファイルは、反応の成功を左右する最も重要な制御パラメータです。以下の3要素を適切に設計する必要があります：':
            'The temperature profile in solid-state reactions is the most important control parameter determining reaction success. The following three elements must be properly designed:',

        '温度プロファイル設計': 'Temperature Profile Design',
        '加熱速度': 'Heating Rate',
        'Heating Rate': 'Heating Rate',
        '保持時間': 'Holding Time',
        'Holding Time': 'Holding Time',
        '冷却速度': 'Cooling Rate',
        'Cooling Rate': 'Cooling Rate',
        '速すぎ: 熱応力→亀裂': 'Too fast: Thermal stress → Cracks',
        '遅すぎ: 不要な相変態': 'Too slow: Unwanted phase transformations',
        '短すぎ: 反応不完全': 'Too short: Incomplete reaction',
        '長すぎ: 粒成長過剰': 'Too long: Excessive grain growth',
        '速すぎ: 熱応力→亀裂': 'Too fast: Thermal stress → Cracks',
        '遅すぎ: 好ましくない相': 'Too slow: Unfavorable phases',

        # Heating rate section
        '1. 加熱速度（Heating Rate）': '1. Heating Rate',
        '一般的な推奨値：': 'General recommended value: ',
        '2-10°C/min': '2-10°C/min',

        '考慮すべき要因：': 'Factors to consider:',
        '熱応力': 'Thermal Stress',
        '試料内部と表面の温度差が大きいと熱応力が発生し、亀裂の原因に':
            'Large temperature differences between sample interior and surface generate thermal stress, causing cracks',
        '中間相の形成': 'Intermediate Phase Formation',
        '低温域での不要な中間相形成を避けるため、ある温度範囲は速く通過':
            'Rapid passage through certain temperature ranges to avoid unwanted intermediate phase formation at low temperatures',
        '分解反応': 'Decomposition Reactions',
        'CO₂やH₂O放出反応では、急速加熱は突沸の原因に':
            'In CO₂ or H₂O releasing reactions, rapid heating causes bumping',

        # Warning about BaCO3
        '⚠️ 実例: BaCO₃の分解反応': '⚠️ Example: Decomposition Reaction of BaCO₃',
        'BaTiO₃合成では800-900°Cで BaCO₃ → BaO + CO₂ の分解が起こります。加熱速度が20°C/min以上だと、CO₂が急激に放出され、試料が破裂することがあります。推奨加熱速度は5°C/min以下です。':
            'In BaTiO₃ synthesis, decomposition BaCO₃ → BaO + CO₂ occurs at 800-900°C. At heating rates above 20°C/min, CO₂ is released rapidly and samples may rupture. Recommended heating rate is 5°C/min or below.',

        # Holding time section
        '2. 保持時間（Holding Time）': '2. Holding Time',
        '決定方法：': 'Determination method: ',
        'Jander式からの推算 + 実験最適化':
            'Estimation from Jander equation + experimental optimization',

        '必要な保持時間は以下の式で推定できます：':
            'Required holding time can be estimated from the following equation:',
        't = [α_target / k]^(1/2) × (1 - α_target^(1/3))^(-2)':
            't = [α_target / k]^(1/2) × (1 - α_target^(1/3))^(-2)',

        '典型的な保持時間：': 'Typical holding times:',
        '低温反応（<1000°C）: 12-24時間':
            'Low-temperature reactions (<1000°C): 12-24 hours',
        '中温反応（1000-1300°C）: 4-8時間':
            'Medium-temperature reactions (1000-1300°C): 4-8 hours',
        '高温反応（>1300°C）: 2-4時間':
            'High-temperature reactions (>1300°C): 2-4 hours',

        # Cooling rate section
        '3. 冷却速度（Cooling Rate）': '3. Cooling Rate',
        '一般的な推奨値：': 'General recommended value: ',
        '1-5°C/min（加熱速度より遅め）':
            '1-5°C/min (slower than heating rate)',

        '重要性：': 'Importance:',
        '相変態の制御': 'Control of Phase Transformations',
        '冷却中の高温相→低温相変態を制御':
            'Control high-temperature → low-temperature phase transformation during cooling',
        '欠陥の生成': 'Defect Formation',
        '急冷は酸素欠損等の欠陥を凍結':
            'Rapid cooling freezes defects such as oxygen vacancies',
        '結晶性': 'Crystallinity',
        '徐冷は結晶性を向上':
            'Slow cooling improves crystallinity',

        # Section 1.4.2
        '1.4.2 温度プロファイルの最適化シミュレーション':
            '1.4.2 Temperature Profile Optimization Simulation',

        # Code comments in Example 4
        '温度プロファイルを生成': 'Generate temperature profile',
        '時間配列 [min]': 'Time array [min]',
        '保持温度 [°C]': 'Holding temperature [°C]',
        '加熱速度 [°C/min]': 'Heating rate [°C/min]',
        '保持時間 [min]': 'Holding time [min]',
        '冷却速度 [°C/min]': 'Cooling rate [°C/min]',
        '温度プロファイル [°C]': 'Temperature profile [°C]',
        '室温': 'Room temperature',
        '加熱時間': 'Heating time',
        '冷却開始時刻': 'Cooling start time',
        '加熱フェーズ': 'Heating phase',
        '保持フェーズ': 'Holding phase',
        '冷却フェーズ': 'Cooling phase',
        '室温以下にはならない': 'Does not go below room temperature',

        '温度プロファイルに基づく反応進行を計算':
            'Calculate reaction progress based on temperature profile',
        '温度プロファイル [°C]': 'Temperature profile [°C]',
        '時間配列 [min]': 'Time array [min]',
        '活性化エネルギー [J/mol]': 'Activation energy [J/mol]',
        '頻度因子 [m²/s]': 'Frequency factor [m²/s]',
        '粒子半径 [m]': 'Particle radius [m]',
        '反応率': 'Conversion rate',

        'min → s': 'min → s',
        '簡易積分（微小時間での反応進行）':
            'Simple integration (reaction progress in small time steps)',

        '異なる加熱速度での比較': 'Comparison at different heating rates',
        '時間配列': 'Time array',
        '温度プロファイル': 'Temperature Profiles',
        '反応進行': 'Reaction Progress',

        '各加熱速度での95%反応到達時間を計算':
            'Calculate time to reach 95% conversion at each heating rate',
        '95%反応到達時間の比較:':
            'Comparison of time to reach 95% conversion:',
        '95%到達時刻': 'Time to reach 95%',
        '加熱速度': 'Heating rate',
        '反応不完全': 'Incomplete reaction',

        # Exercise problems section
        '演習問題': 'Exercise Problems',

        # Section 1.5
        '1.5.1 pycalphadとは': '1.5.1 What is pycalphad',
        'は、CALPHAD（CALculation of PHAse Diagrams）法に基づく相図計算のためのPythonライブラリです。熱力学データベースから平衡相を計算し、反応経路の設計に有用です。':
            ' is a Python library for phase diagram calculation based on the CALPHAD (CALculation of PHAse Diagrams) method. It calculates equilibrium phases from thermodynamic databases and is useful for reaction pathway design.',

        # Info box about CALPHAD
        '💡 CALPHAD法の利点': '💡 Advantages of CALPHAD Method',
        '多元系（3元系以上）の複雑な相図を計算可能':
            'Can calculate complex phase diagrams of multicomponent systems (ternary and higher)',
        '実験データが少ない系でも予測可能':
            'Can predict even for systems with limited experimental data',
        '温度・組成・圧力依存性を包括的に扱える':
            'Can comprehensively handle temperature, composition, and pressure dependencies',

        # Section 1.5.2
        '1.5.2 二元系相図の計算例': '1.5.2 Example of Binary Phase Diagram Calculation',

        # Code comments in Example 5
        '注意: pycalphadのインストールが必要':
            'Note: pycalphad installation required',
        'TDBデータベースを読み込み（ここでは簡易的な例）':
            'Load TDB database (simplified example here)',
        '実際には適切なTDBファイルが必要':
            'Actual appropriate TDB file is required',
        '例: BaO-TiO2系': 'Example: BaO-TiO2 system',
        '簡易的なTDB文字列（実際はより複雑）':
            'Simplified TDB string (actual is more complex)',

        '注: 実際の計算には正式なTDBファイルが必要':
            'Note: Formal TDB file required for actual calculations',
        'ここでは概念的な説明に留める':
            'Limited to conceptual explanation here',

        'pycalphadによる相図計算の概念:':
            'Concept of phase diagram calculation using pycalphad:',
        'TDBデータベース（熱力学データ）を読み込む':
            'Load TDB database (thermodynamic data)',
        '温度・組成範囲を設定':
            'Set temperature and composition ranges',
        '平衡計算を実行': 'Execute equilibrium calculation',
        '安定相を可視化': 'Visualize stable phases',

        '実際の適用例:': 'Actual application examples:',
        'BaO-TiO2系: BaTiO3の形成温度・組成範囲':
            'BaO-TiO2 system: Formation temperature and composition range of BaTiO3',
        'Si-N系: Si3N4の安定領域':
            'Si-N system: Stability region of Si3N4',
        '多元系セラミックスの相関係':
            'Phase relationships of multicomponent ceramics',

        # Conceptual plot
        '概念的なプロット（実データに基づくイメージ）':
            'Conceptual plot (image based on actual data)',
        '温度範囲': 'Temperature range',
        '各相の安定領域（概念図）':
            'Stability regions of each phase (conceptual diagram)',
        'BaO + TiO2 → BaTiO3 反応':
            'BaO + TiO2 → BaTiO3 reaction',

        # Section 1.6 - DOE
        '1.6 実験計画法（DOE）による条件最適化':
            '1.6 Condition Optimization using Design of Experiments (DOE)',

        '1.6.1 DOEとは': '1.6.1 What is DOE',
        '実験計画法（Design of Experiments, DOE）は、複数のパラメータが相互作用する系で、最小の実験回数で最適条件を見つける統計手法です。':
            'Design of Experiments (DOE) is a statistical method for finding optimal conditions with minimum number of experiments in systems where multiple parameters interact.',

        '固相反応で最適化すべき主要パラメータ：':
            'Key parameters to optimize in solid-state reactions:',
        '反応温度（T）': 'Reaction temperature (T)',
        '保持時間（t）': 'Holding time (t)',
        '粒子サイズ（r）': 'Particle size (r)',
        '原料比（モル比）': 'Raw material ratio (molar ratio)',
        '雰囲気（空気、窒素、真空など）':
            'Atmosphere (air, nitrogen, vacuum, etc.)',

        # Section 1.6.2
        '1.6.2 応答曲面法（Response Surface Methodology）':
            '1.6.2 Response Surface Methodology',

        # Code comments in Example 6
        '仮想的な反応率モデル（温度と時間の関数）':
            'Virtual reaction yield model (function of temperature and time)',
        '温度と時間から反応率を計算（仮想モデル）':
            'Calculate reaction yield from temperature and time (virtual model)',
        '温度 [°C]': 'Temperature [°C]',
        '時間 [hours]': 'Time [hours]',
        'ノイズレベル': 'Noise level',
        '反応率 [%]': 'Reaction yield [%]',
        '最適値: T=1200°C, t=6 hours':
            'Optimal values: T=1200°C, t=6 hours',
        '二次モデル（ガウス型）': 'Quadratic model (Gaussian)',
        'ノイズ追加': 'Add noise',

        '実験点配置（中心複合計画法）':
            'Experimental point arrangement (central composite design)',
        'グリッドで実験点を配置':
            'Arrange experimental points on grid',
        '各実験点で反応率を測定（シミュレーション）':
            'Measure reaction yield at each experimental point (simulation)',

        '結果の表示': 'Display results',
        '実験計画法による反応条件最適化':
            'Reaction condition optimization using DOE',
        '最大反応率の条件を探す':
            'Find conditions for maximum reaction yield',
        '最適条件:': 'Optimal conditions:',
        '最大反応率:': 'Maximum reaction yield:',

        '3D表面プロット': '3D surface plot',
        '等高線プロット': 'Contour plot',

        # Section 1.6.3
        '1.6.3 実験計画の実践的アプローチ':
            '1.6.3 Practical Approach to Experimental Design',
        '実際の固相反応では、以下の手順でDOEを適用します：':
            'In actual solid-state reactions, DOE is applied in the following steps:',

        'スクリーニング実験': 'Screening Experiments',
        '（2水準要因計画法）: 影響の大きいパラメータを特定':
            '(two-level factorial design): Identify parameters with large effects',
        '応答曲面法': 'Response Surface Methodology',
        '（中心複合計画法）: 最適条件の探索':
            '(central composite design): Search for optimal conditions',
        '確認実験': 'Confirmation Experiments',
        '予測された最適条件で実験し、モデルを検証':
            'Conduct experiments at predicted optimal conditions and validate model',

        # Success box about LiCoO2
        '✅ 実例: Li-ion電池正極材LiCoO₂の合成最適化':
            '✅ Example: Synthesis Optimization of Li-ion Battery Cathode Material LiCoO₂',
        'ある研究グループがDOEを用いてLiCoO₂の合成条件を最適化した結果：':
            'Results when a research group optimized LiCoO₂ synthesis conditions using DOE:',
        '実験回数: 従来法100回 → DOE法25回（75%削減）':
            'Number of experiments: Traditional method 100 → DOE method 25 (75% reduction)',
        '最適温度: 900°C（従来の850°Cより高温）':
            'Optimal temperature: 900°C (higher than traditional 850°C)',
        '最適保持時間: 12時間（従来の24時間から半減）':
            'Optimal holding time: 12 hours (halved from traditional 24 hours)',
        '電池容量: 140 mAh/g → 155 mAh/g（11%向上）':
            'Battery capacity: 140 mAh/g → 155 mAh/g (11% improvement)',

        # Section 1.7
        '1.7 反応速度曲線のフィッティング':
            '1.7 Fitting of Reaction Kinetics Curves',

        '1.7.1 実験データからの速度定数決定':
            '1.7.1 Determination of Rate Constants from Experimental Data',

        # Code comments in Example 7
        '実験データ（時間 vs 反応率）':
            'Experimental data (time vs conversion)',
        '例: BaTiO3合成 @ 1200°C':
            'Example: BaTiO3 synthesis @ 1200°C',
        'hours': 'hours',

        'Jander式モデル': 'Jander equation model',
        'Jander式による反応率計算':
            'Calculate conversion using Jander equation',
        '時間 [hours]': 'Time [hours]',
        '反応率': 'Conversion',
        '[1-(1-α)^(1/3)]² = kt を α について解く':
            'Solve [1-(1-α)^(1/3)]² = kt for α',

        'Ginstling-Brounshtein式（別の拡散モデル）':
            'Ginstling-Brounshtein equation (another diffusion model)',
        '数値的に解く必要があるが、ここでは近似式を使用':
            'Needs numerical solution, but approximate formula used here',

        'Power law (経験式)': 'Power law (empirical formula)',
        'べき乗則モデル': 'Power law model',
        '指数': 'Exponent',

        '各モデルでフィッティング': 'Fitting with each model',
        '予測曲線生成': 'Generate predicted curves',
        '残差計算': 'Calculate residuals',
        'R²計算': 'Calculate R²',

        'フィッティング結果': 'Fitting results',
        '残差プロット': 'Residual plot',

        '結果サマリー': 'Results summary',
        '反応速度モデルのフィッティング結果:':
            'Fitting results of reaction kinetics models:',
        '最適モデル:': 'Optimal model:',

        # Section 1.8 - Advanced Topics
        '1.8 高度なトピック: 微細構造制御':
            '1.8 Advanced Topics: Microstructure Control',

        '1.8.1 粒成長の抑制': '1.8.1 Grain Growth Suppression',
        '固相反応では、高温・長時間保持により望ましくない粒成長が起こります。これを抑制する戦略：':
            'In solid-state reactions, undesirable grain growth occurs during high-temperature and long-time holding. Strategies to suppress this:',

        'Two-step sintering': 'Two-step sintering',
        '高温で短時間保持後、低温で長時間保持':
            'Long-time holding at low temperature after short-time holding at high temperature',
        '添加剤の使用': 'Use of Additives',
        '粒成長抑制剤（例: MgO, Al₂O₃）を微量添加':
            'Add small amounts of grain growth inhibitors (e.g., MgO, Al₂O₃)',
        'Spark Plasma Sintering (SPS)': 'Spark Plasma Sintering (SPS)',
        '急速加熱・短時間焼結': 'Rapid heating and short-time sintering',

        # Section 1.8.2
        '1.8.2 反応の機械化学的活性化':
            '1.8.2 Mechanochemical Activation of Reactions',
        'メカノケミカル法（高エネルギーボールミル）により、固相反応を室温付近で進行させることも可能です：':
            'By mechanochemical methods (high-energy ball milling), solid-state reactions can proceed near room temperature:',

        # Code comments in Example 8
        '粒成長の時間発展': 'Time evolution of grain growth',
        'Burke-Turnbull式: G^n - G0^n = k*t':
            'Burke-Turnbull equation: G^n - G0^n = k*t',
        '温度 [K]': 'Temperature [K]',
        '初期粒径 [μm]': 'Initial grain size [μm]',
        '粒成長指数（通常2-4）': 'Grain growth exponent (typically 2-4)',
        '粒径 [μm]': 'Grain size [μm]',
        'hours → seconds': 'hours → seconds',

        '温度の影響': 'Effect of temperature',
        '0-12 hours': '0-12 hours',
        '温度依存性': 'Temperature dependence',
        'Two-step sinteringの効果': 'Effect of two-step sintering',

        'Conventional sintering: 1300°C, 6 hours':
            'Conventional sintering: 1300°C, 6 hours',
        '最終粒径の比較': 'Comparison of final grain size',
        '粒成長の比較:': 'Comparison of grain growth:',
        '粒径抑制効果:': 'Grain size suppression effect:',

        # Learning objectives check
        '学習目標の確認': 'Learning Objectives Check',
        'この章を完了すると、以下を説明できるようになります：':
            'Upon completing this chapter, you will be able to explain:',

        # Check items
        '✅ 固相反応の3つの律速段階（核生成・界面反応・拡散）を説明できる':
            '✅ Can explain the three rate-limiting steps of solid-state reactions (nucleation, interface reaction, diffusion)',
        '✅ Arrhenius式の物理的意味と温度依存性を理解している':
            '✅ Understand the physical meaning and temperature dependence of the Arrhenius equation',
        '✅ Jander式とGinstling-Brounshtein式の違いを説明できる':
            '✅ Can explain the differences between Jander and Ginstling-Brounshtein equations',
        '✅ 温度プロファイルの3要素（加熱速度・保持時間・冷却速度）の重要性を理解している':
            '✅ Understand the importance of three elements of temperature profile (heating rate, holding time, cooling rate)',

        '✅ Pythonで拡散係数の温度依存性をシミュレートできる':
            '✅ Can simulate temperature dependence of diffusion coefficient using Python',
        '✅ Jander式を用いて反応進行を予測できる':
            '✅ Can predict reaction progress using Jander equation',
        '✅ Kissinger法でDSC/TGデータから活性化エネルギーを計算できる':
            '✅ Can calculate activation energy from DSC/TG data using Kissinger method',
        '✅ DOE（実験計画法）で反応条件を最適化できる':
            '✅ Can optimize reaction conditions using DOE (Design of Experiments)',
        '✅ pycalphadを用いた相図計算の基礎を理解している':
            '✅ Understand basics of phase diagram calculation using pycalphad',

        '✅ 新規セラミックス材料の合成プロセスを設計できる':
            '✅ Can design synthesis processes for new ceramic materials',
        '✅ 実験データから反応機構を推定し、適切な速度式を選択できる':
            '✅ Can infer reaction mechanisms from experimental data and select appropriate rate equations',
        '✅ 産業プロセスでの条件最適化戦略を立案できる':
            '✅ Can formulate condition optimization strategies for industrial processes',
        '✅ 粒成長制御の戦略（Two-step sintering等）を提案できる':
            '✅ Can propose grain growth control strategies (e.g., two-step sintering)',

        # Exercise section
        'Easy（基礎確認）': 'Easy (Fundamental Check)',

        'Q1: 固相反応の律速段階':
            'Q1: Rate-Limiting Step of Solid-State Reactions',
        'BaTiO₃の合成反応 BaCO₃ + TiO₂ → BaTiO₃ + CO₂ において、最も遅い（律速となる）段階はどれですか？':
            'In the synthesis reaction BaCO₃ + TiO₂ → BaTiO₃ + CO₂ of BaTiO₃, which step is the slowest (rate-limiting)?',
        'CO₂の放出': 'Release of CO₂',
        'BaTiO₃核の生成': 'Nucleation of BaTiO₃',
        'Ba²⁺イオンの生成物層中の拡散':
            'Diffusion of Ba²⁺ ions through product layer',
        '界面での化学反応': 'Chemical reaction at interface',

        '解答を見る': 'View answer',
        '正解:': 'Correct answer:',
        '解説:': 'Explanation:',
        '固相反応では、生成物層が反応物を物理的に分離するため、イオンが生成物層を通って拡散する過程が最も遅くなります。':
            'In solid-state reactions, the process of ions diffusing through the product layer is slowest because the product layer physically separates the reactants.',
        'CO₂放出は気体の拡散なので速い':
            'CO₂ release is fast because it is gas diffusion',
        '核生成は初期段階で完了':
            'Nucleation completes in the initial stage',
        '拡散が律速': 'Diffusion is rate-limiting',
        '（正解）- 固体中のイオン拡散は極めて遅い（D ~ 10⁻¹² m²/s）':
            '(correct) - Ion diffusion in solids is extremely slow (D ~ 10⁻¹² m²/s)',
        '界面反応は通常速い': 'Interface reaction is usually fast',

        '重要ポイント:': 'Key point: ',
        '拡散係数は温度に対して指数関数的に増加するため、反応温度の選択が極めて重要です。':
            'The diffusion coefficient increases exponentially with temperature, making reaction temperature selection extremely important.',

        'Q2: Arrhenius式のパラメータ':
            'Q2: Parameters of Arrhenius Equation',
        '拡散係数 D(T) = D₀ exp(-Eₐ/RT) において、Eₐ（活性化エネルギー）が大きいほど、温度変化に対する拡散係数の感度はどうなりますか？':
            'In the diffusion coefficient D(T) = D₀ exp(-Eₐ/RT), what happens to the sensitivity of the diffusion coefficient to temperature changes as Eₐ (activation energy) becomes larger?',
        '高くなる（温度依存性が強い）':
            'Becomes higher (strong temperature dependence)',
        '低くなる（温度依存性が弱い）':
            'Becomes lower (weak temperature dependence)',
        '変わらない': 'No change',
        '関係ない': 'Irrelevant',

        '活性化エネルギーEₐは、指数関数 exp(-Eₐ/RT) の肩に位置するため、Eₐが大きいほど温度変化に対するDの変化率が大きくなります。':
            'The activation energy Eₐ is in the exponent of exp(-Eₐ/RT), so the larger Eₐ becomes, the greater the rate of change of D with respect to temperature change.',
        '数値例:': 'Numerical examples:',
        'Eₐ = 100 kJ/mol の場合: 温度を100°C上げると D は約3倍':
            'For Eₐ = 100 kJ/mol: Raising temperature by 100°C increases D by about 3 times',
        'Eₐ = 300 kJ/mol の場合: 温度を100°C上げると D は約30倍':
            'For Eₐ = 300 kJ/mol: Raising temperature by 100°C increases D by about 30 times',
        'このため、活性化エネルギーが大きい系では、温度制御が特に重要になります。':
            'Therefore, temperature control becomes particularly important for systems with large activation energy.',

        'Q3: 粒子サイズと反応速度':
            'Q3: Particle Size and Reaction Rate',
        'Jander式 k = D·C₀/r₀² によれば、粒子半径r₀を1/2にすると、反応速度定数kは何倍になりますか？':
            'According to the Jander equation k = D·C₀/r₀², when particle radius r₀ is reduced to 1/2, by what factor does the rate constant k change?',
        '2倍': '2 times',
        '4倍': '4 times',
        '1/2倍': '1/2 times',
        '1/4倍': '1/4 times',

        '計算:': 'Calculation:',
        'k ∝ 1/r₀²<br>': 'k ∝ 1/r₀²<br>',
        'r₀ → r₀/2 のとき、k → k/(r₀/2)² = k/(r₀²/4) = 4k':
            'When r₀ → r₀/2, k → k/(r₀/2)² = k/(r₀²/4) = 4k',
        '実践的意味:': 'Practical meaning:',
        'これが「粉砕・微細化」が固相反応で極めて重要な理由です。':
            'This is why "pulverization and refinement" are extremely important in solid-state reactions.',
        '粒径10μm → 1μm: 反応速度100倍（反応時間1/100）':
            'Particle size 10μm → 1μm: Reaction rate 100 times (reaction time 1/100)',
        'ボールミル、ジェットミルによる微細化が標準プロセス':
            'Refinement by ball mill, jet mill is standard process',
        'ナノ粒子を使えば室温付近でも反応可能な場合も':
            'Using nanoparticles, reactions may be possible even near room temperature',

        'Medium（応用）': 'Medium (Application)',

        'Q4: 温度プロファイル設計':
            'Q4: Temperature Profile Design',
        'BaTiO₃合成で、加熱速度を20°C/minから5°C/minに変更しました。この変更の主な理由として最も適切なのはどれですか？':
            'In BaTiO₃ synthesis, the heating rate was changed from 20°C/min to 5°C/min. Which is the most appropriate reason for this change?',
        '反応速度を速めるため': 'To accelerate reaction rate',
        'CO₂の急激な放出による試料破裂を防ぐため':
            'To prevent sample rupture due to rapid CO₂ release',
        '電気代を節約するため': 'To save electricity costs',
        '結晶性を下げるため': 'To reduce crystallinity',

        '詳細な理由:': 'Detailed reasons:',
        'BaCO₃ + TiO₂ → BaTiO₃ + CO₂ の反応では、800-900°Cで炭酸バリウムが分解してCO₂を放出します。':
            'In the reaction BaCO₃ + TiO₂ → BaTiO₃ + CO₂, barium carbonate decomposes at 800-900°C releasing CO₂.',
        '急速加熱（20°C/min）の問題:':
            'Problems with rapid heating (20°C/min):',
        '短時間で多量のCO₂が発生':
            'Large amount of CO₂ generated in short time',
        'ガス圧が高まり、試料が破裂・飛散':
            'Gas pressure increases, causing sample rupture and scattering',
        '焼結体に亀裂・クラックが入る':
            'Cracks form in sintered body',
        '徐加熱（5°C/min）の利点:':
            'Advantages of slow heating (5°C/min):',
        'CO₂がゆっくり放出され、圧力上昇が緩やか':
            'CO₂ released slowly, pressure increase is gradual',
        '試料の健全性が保たれる':
            'Sample integrity is maintained',
        '均質な反応が進行':
            'Homogeneous reaction proceeds',

        '実践的アドバイス:': 'Practical advice: ',
        '分解反応を伴う合成では、ガス放出速度を制御するため、該当温度範囲での加熱速度を特に遅くします（例: 750-950°Cを2°C/minで通過）。':
            'In syntheses involving decomposition reactions, heating rate is particularly slowed in the relevant temperature range to control gas release rate (e.g., passing through 750-950°C at 2°C/min).',

        'Q5: Kissinger法の適用':
            'Q5: Application of Kissinger Method',
        'DSC測定で以下のデータが得られました。Kissinger法で活性化エネルギーを求めてください。':
            'The following data were obtained from DSC measurements. Calculate the activation energy using the Kissinger method.',
        '加熱速度 β (K/min): 5, 10, 15':
            'Heating rate β (K/min): 5, 10, 15',
        'ピーク温度 Tp (K): 1273, 1293, 1308':
            'Peak temperature Tp (K): 1273, 1293, 1308',
        'Kissinger式: ln(β/Tp²) vs 1/Tp の傾き = -Eₐ/R':
            'Kissinger equation: slope of ln(β/Tp²) vs 1/Tp = -Eₐ/R',

        '解答:': 'Answer:',
        'ステップ1: データ整理': 'Step 1: Data organization',
        'ステップ2: 線形回帰': 'Step 2: Linear regression',
        'y = ln(β/Tp²) vs x = 1000/Tp をプロット':
            'Plot y = ln(β/Tp²) vs x = 1000/Tp',
        '傾き slope = Δy/Δx = (-10.932 - (-11.558)) / (0.7645 - 0.7855) = 0.626 / (-0.021) ≈ -29.8':
            'Slope = Δy/Δx = (-10.932 - (-11.558)) / (0.7645 - 0.7855) = 0.626 / (-0.021) ≈ -29.8',

        'ステップ3: Eₐ計算': 'Step 3: Eₐ calculation',
        'slope = -Eₐ / (R × 1000) （1000/Tpを使ったため1000で割る）':
            'slope = -Eₐ / (R × 1000) (divided by 1000 because 1000/Tp was used)',
        'Eₐ = -slope × R × 1000':
            'Eₐ = -slope × R × 1000',
        'Eₐ = 29.8 × 8.314 × 1000 = 247,757 J/mol ≈ 248 kJ/mol':
            'Eₐ = 29.8 × 8.314 × 1000 = 247,757 J/mol ≈ 248 kJ/mol',

        '答え: Eₐ ≈ 248 kJ/mol': 'Answer: Eₐ ≈ 248 kJ/mol',
        '物理的解釈:': 'Physical interpretation:',
        'この値はBaTiO₃系の固相反応における典型的な活性化エネルギー（250-350 kJ/mol）の範囲内です。この活性化エネルギーは、Ba²⁺イオンの固相拡散に対応していると考えられます。':
            'This value is within the range of typical activation energies (250-350 kJ/mol) for solid-state reactions in BaTiO₃ systems. This activation energy is considered to correspond to solid-state diffusion of Ba²⁺ ions.',

        'Q6: DOEによる最適化': 'Q6: Optimization using DOE',
        '実験計画法で、温度（1100, 1200, 1300°C）と時間（4, 6, 8時間）の2因子を検討します。全実験回数は何回必要ですか？また、1因子ずつ変える従来法と比べた利点を2つ挙げてください。':
            'In DOE, two factors of temperature (1100, 1200, 1300°C) and time (4, 6, 8 hours) are examined. How many total experiments are required? Also, list two advantages compared to the traditional method of varying one factor at a time.',

        '実験回数:': 'Number of experiments:',
        '3水準 × 3水準 = ': '3 levels × 3 levels = ',
        '9回': '9 times',
        '（フルファクトリアル計画）': '(full factorial design)',

        'DOEの利点（従来法との比較）:':
            'Advantages of DOE (compared to traditional method):',
        '交互作用の検出が可能': 'Detection of interactions is possible',
        '従来法: 温度の影響、時間の影響を個別に評価':
            'Traditional method: Evaluate effects of temperature and time separately',
        'DOE: 「高温では時間を短くできる」といった交互作用を定量化':
            'DOE: Quantify interactions such as "time can be shortened at high temperature"',
        '例: 1300°Cでは4時間で十分だが、1100°Cでは8時間必要、など':
            'Example: 4 hours sufficient at 1300°C, but 8 hours needed at 1100°C, etc.',

        '実験回数の削減': 'Reduction in number of experiments',
        '従来法（OFAT: One Factor At a Time）:':
            'Traditional method (OFAT: One Factor At a Time):',
        '温度検討: 3回（時間固定）':
            'Temperature study: 3 times (time fixed)',
        '時間検討: 3回（温度固定）':
            'Time study: 3 times (temperature fixed)',
        '確認実験: 複数回':
            'Confirmation experiments: Multiple times',
        '合計: 10回以上': 'Total: 10 or more times',
        'DOE: 9回で完了（全条件網羅＋交互作用解析）':
            'DOE: Complete in 9 times (covering all conditions + interaction analysis)',
        'さらに中心複合計画法を使えば7回に削減可能':
            'Further reduction to 7 times possible using central composite design',

        '追加の利点:': 'Additional advantages:',
        '統計的に有意な結論が得られる（誤差評価が可能）':
            'Statistically significant conclusions can be obtained (error evaluation possible)',
        '応答曲面を構築でき、未実施条件の予測が可能':
            'Response surface can be constructed, prediction of untested conditions possible',
        '最適条件が実験範囲外にある場合でも検出できる':
            'Can detect even when optimal conditions are outside experimental range',

        'Hard（発展）': 'Hard (Advanced)',

        'Q7: 複雑な反応系の設計':
            'Q7: Design of Complex Reaction System',
        '次の条件でLi₁.₂Ni₀.₂Mn₀.₆O₂（リチウムリッチ正極材料）を合成する温度プロファイルを設計してください：':
            'Design a temperature profile for synthesizing Li₁.₂Ni₀.₂Mn₀.₆O₂ (lithium-rich cathode material) under the following conditions:',
        '原料: Li₂CO₃, NiO, Mn₂O₃':
            'Raw materials: Li₂CO₃, NiO, Mn₂O₃',
        '目標: 単一相、粒径 < 5 μm、Li/遷移金属比の精密制御':
            'Target: Single phase, grain size < 5 μm, precise control of Li/transition metal ratio',
        '制約: 900°C以上でLi₂Oが揮発（Li欠損のリスク）':
            'Constraint: Li₂O volatilizes above 900°C (risk of Li deficiency)',
        '温度プロファイル（加熱速度、保持温度・時間、冷却速度）と、その設計理由を説明してください。':
            'Explain the temperature profile (heating rate, holding temperature/time, cooling rate) and design rationale.',

        '推奨温度プロファイル:': 'Recommended temperature profile:',

        'Phase 1: 予備加熱（Li₂CO₃分解）':
            'Phase 1: Pre-heating (Li₂CO₃ decomposition)',
        '室温 → 500°C: 3°C/min': 'Room temp → 500°C: 3°C/min',
        '500°C保持: 2時間': '500°C hold: 2 hours',
        '理由:': 'Reason: ',
        'Li₂CO₃の分解（~450°C）をゆっくり進行させ、CO₂を完全に除去':
            'Slowly proceed with Li₂CO₃ decomposition (~450°C) to completely remove CO₂',

        'Phase 2: 中間加熱（前駆体形成）':
            'Phase 2: Intermediate heating (precursor formation)',
        '500°C → 750°C: 5°C/min': '500°C → 750°C: 5°C/min',
        '750°C保持: 4時間': '750°C hold: 4 hours',
        'Li₂MnO₃やLiNiO₂などの中間相を形成。Li揮発の少ない温度で均質化':
            'Form intermediate phases such as Li₂MnO₃ and LiNiO₂. Homogenize at temperature with minimal Li volatilization',

        'Phase 3: 本焼成（目的相合成）':
            'Phase 3: Main sintering (target phase synthesis)',
        '750°C → 850°C: 2°C/min（ゆっくり）':
            '750°C → 850°C: 2°C/min (slow)',
        '850°C保持: 12時間': '850°C hold: 12 hours',
        'Li₁.₂Ni₀.₂Mn₀.₆O₂の単一相形成には長時間必要':
            'Long time needed for single phase formation of Li₁.₂Ni₀.₂Mn₀.₆O₂',
        '850°Cに制限してLi揮発を最小化（<900°C制約）':
            'Limit to 850°C to minimize Li volatilization (<900°C constraint)',
        '長時間保持で拡散を進めるが、粒成長は抑制される温度':
            'Long-time holding advances diffusion, but temperature suppresses grain growth',

        'Phase 4: 冷却': 'Phase 4: Cooling',
        '850°C → 室温: 2°C/min': '850°C → Room temp: 2°C/min',
        '徐冷により結晶性向上、熱応力による亀裂防止':
            'Slow cooling improves crystallinity, prevents cracks from thermal stress',

        '設計の重要ポイント:': 'Important design points:',
        'Li揮発対策:': 'Li volatilization countermeasures:',
        '900°C以下に制限（本問の制約）':
            'Limit to below 900°C (constraint in this problem)',
        'さらに、Li過剰原料（Li/TM = 1.25など）を使用':
            'Additionally, use Li-excess raw materials (e.g., Li/TM = 1.25)',
        '酸素気流中で焼成してLi₂Oの分圧を低減':
            'Sinter in oxygen flow to reduce partial pressure of Li₂O',

        '粒径制御 (< 5 μm):': 'Grain size control (< 5 μm):',
        '低温（850°C）・長時間（12h）で反応を進める':
            'Proceed with reaction at low temperature (850°C) and long time (12h)',
        '高温・短時間だと粒成長が過剰になる':
            'High temperature and short time causes excessive grain growth',
        '原料粒径も1μm以下に微細化':
            'Also refine raw material particle size to below 1μm',

        '組成均一性:': 'Composition uniformity:',
        '750°Cでの中間保持が重要':
            'Intermediate holding at 750°C is important',
        'この段階で遷移金属の分布を均質化':
            'Homogenize transition metal distribution at this stage',
        '必要に応じて、750°C保持後に一度冷却→粉砕→再加熱':
            'If necessary, cool once after 750°C hold → pulverize → reheat',

        '全体所要時間:': 'Total time required: ',
        '約30時間（加熱12h + 保持18h）':
            'About 30 hours (heating 12h + holding 18h)',

        '代替手法の検討:': 'Consideration of alternative methods:',
        'Sol-gel法:': 'Sol-gel method: ',
        'より低温（600-700°C）で合成可能、均質性向上':
            'Synthesis possible at lower temperature (600-700°C), improved homogeneity',
        'Spray pyrolysis:': 'Spray pyrolysis: ',
        '粒径制御が容易': 'Easy grain size control',
        'Two-step sintering:': 'Two-step sintering: ',
        '900°C 1h → 800°C 10h で粒成長抑制':
            '900°C 1h → 800°C 10h suppresses grain growth',

        'Q8: 速度論的解析の総合問題':
            'Q8: Comprehensive Problem on Kinetic Analysis',
        '以下のデータから、反応機構を推定し、活性化エネルギーを計算してください。':
            'From the following data, estimate the reaction mechanism and calculate the activation energy.',

        '実験データ:': 'Experimental data:',
        '温度 (°C)': 'Temperature (°C)',
        '50%反応到達時間 t₅₀ (hours)': 'Time to 50% conversion t₅₀ (hours)',
        'Jander式を仮定した場合: [1-(1-0.5)^(1/3)]² = k·t₅₀':
            'Assuming Jander equation: [1-(1-0.5)^(1/3)]² = k·t₅₀',

        'ステップ1: 速度定数kの計算':
            'Step 1: Calculation of rate constant k',
        'Jander式で α=0.5 のとき:':
            'For Jander equation when α=0.5:',
        '[1-(1-0.5)^(1/3)]² = [1-0.794]² = 0.206² = 0.0424':
            '[1-(1-0.5)^(1/3)]² = [1-0.794]² = 0.206² = 0.0424',
        'したがって k = 0.0424 / t₅₀':
            'Therefore k = 0.0424 / t₅₀',

        'ステップ2: Arrheniusプロット':
            'Step 2: Arrhenius plot',
        'ln(k) vs 1/T をプロット（線形回帰）':
            'Plot ln(k) vs 1/T (linear regression)',
        '線形フィット: ln(k) = A - Eₐ/(R·T)':
            'Linear fit: ln(k) = A - Eₐ/(R·T)',
        '傾き = -Eₐ/R': 'Slope = -Eₐ/R',

        '線形回帰計算:': 'Linear regression calculation:',
        'slope = Δ(ln k) / Δ(1000/T)': 'slope = Δ(ln k) / Δ(1000/T)',
        '= (-3.343 - (-6.080)) / (0.6357 - 0.7855)':
            '= (-3.343 - (-6.080)) / (0.6357 - 0.7855)',
        '= 2.737 / (-0.1498)': '= 2.737 / (-0.1498)',
        '= -18.27': '= -18.27',

        'ステップ3: 活性化エネルギー計算':
            'Step 3: Activation energy calculation',
        'slope = -Eₐ / (R × 1000)': 'slope = -Eₐ / (R × 1000)',
        'Eₐ = -slope × R × 1000': 'Eₐ = -slope × R × 1000',
        'Eₐ = 18.27 × 8.314 × 1000': 'Eₐ = 18.27 × 8.314 × 1000',
        'Eₐ = 151,899 J/mol ≈ ': 'Eₐ = 151,899 J/mol ≈ ',
        '152 kJ/mol': '152 kJ/mol',

        'ステップ4: 反応機構の考察':
            'Step 4: Discussion of reaction mechanism',
        '活性化エネルギーの比較:':
            'Comparison of activation energies:',
        '得られた値: 152 kJ/mol': 'Obtained value: 152 kJ/mol',
        '典型的な固相拡散: 200-400 kJ/mol':
            'Typical solid-state diffusion: 200-400 kJ/mol',
        '界面反応: 50-150 kJ/mol': 'Interface reaction: 50-150 kJ/mol',

        '推定される機構:': 'Inferred mechanism:',
        'この値は界面反応と拡散の中間':
            'This value is intermediate between interface reaction and diffusion',
        '可能性1: 界面反応が主律速（拡散の影響は小）':
            'Possibility 1: Interface reaction is mainly rate-limiting (small influence of diffusion)',
        '可能性2: 粒子が微細で拡散距離が短く、見かけのEₐが低い':
            'Possibility 2: Particles are fine with short diffusion distance, apparent Eₐ is low',
        '可能性3: 混合律速（界面反応と拡散の両方が寄与）':
            'Possibility 3: Mixed control (both interface reaction and diffusion contribute)',

        'ステップ5: 検証方法の提案':
            'Step 5: Proposal of verification methods',
        '粒子サイズ依存性:': 'Particle size dependence: ',
        '異なる粒径で実験し、k ∝ 1/r₀² が成立するか確認':
            'Experiment with different particle sizes, confirm if k ∝ 1/r₀² holds',
        '成立 → 拡散律速': 'Holds → Diffusion-controlled',
        '不成立 → 界面反応律速': 'Does not hold → Interface reaction-controlled',

        '他の速度式でのフィッティング:':
            'Fitting with other rate equations:',
        'Ginstling-Brounshtein式（3次元拡散）':
            'Ginstling-Brounshtein equation (3D diffusion)',
        'Contracting sphere model（界面反応）':
            'Contracting sphere model (interface reaction)',
        'どちらがR²が高いか比較': 'Compare which has higher R²',

        '微細構造観察:': 'Microstructure observation: ',
        'SEMで反応界面を観察': 'Observe reaction interface with SEM',
        '厚い生成物層 → 拡散律速の証拠':
            'Thick product layer → Evidence of diffusion control',
        '薄い生成物層 → 界面反応律速の可能性':
            'Thin product layer → Possibility of interface reaction control',

        '最終結論:': 'Final conclusion:',
        '活性化エネルギー ': 'Activation energy ',
        '推定機構: ': 'Inferred mechanism: ',
        '界面反応律速、または微細粒子系での拡散律速':
            'Interface reaction-controlled, or diffusion-controlled in fine particle systems',
        '追加実験が推奨される。': 'Additional experiments are recommended.',

        # Next steps section
        '次のステップ': 'Next Steps',
        '第1章では先進セラミックス材料（構造・機能性・バイオセラミックス）の基礎理論を学びました。次の第2章では、先進ポリマー材料（高性能エンジニアリングプラスチック、機能性高分子、生分解性ポリマー）について学びます。':
            'In Chapter 1, we learned the fundamental theory of advanced ceramic materials (structural, functional, and bioceramics). In the next Chapter 2, we will learn about advanced polymer materials (high-performance engineering plastics, functional polymers, biodegradable polymers).',

        # Navigation buttons
        'シリーズ目次': 'Series Contents',
        '第2章へ進む': 'Proceed to Chapter 2',

        # References section
        '参考文献': 'References',
        'セラミックス材料科学の古典的名著、機械的性質と破壊理論の包括的解説':
            'Classic masterpiece of ceramic materials science, comprehensive explanation of mechanical properties and fracture theory',
        '構造用セラミックスの強化機構と高靭性化技術の詳細な解説':
            'Detailed explanation of strengthening mechanisms and toughening technology of structural ceramics',
        'バイオセラミックスの生体適合性と骨結合メカニズムの基礎理論':
            'Fundamental theory of biocompatibility and osseointegration mechanisms of bioceramics',
        '圧電材料と誘電材料の物理的起源と応用の最新知見':
            'Latest knowledge on physical origins and applications of piezoelectric and dielectric materials',
        'ジルコニア変態強化理論の先駆的論文':
            'Pioneering paper on zirconia transformation toughening theory',
        'PZT圧電セラミックスの発展史と技術革新の包括的レビュー':
            'Comprehensive review of development history and technological innovation of PZT piezoelectric ceramics',
        '材料科学計算のためのPythonライブラリ、相図計算と構造解析ツール':
            'Python library for materials science calculations, phase diagram calculation and structure analysis tools',

        # Tools and libraries section
        '使用ツールとライブラリ': 'Tools and Libraries Used',
        '数値計算ライブラリ': 'Numerical computation library',
        '科学技術計算ライブラリ（curve_fit, optimize）':
            'Scientific computing library (curve_fit, optimize)',
        'データ可視化ライブラリ': 'Data visualization library',
        '相図計算ライブラリ': 'Phase diagram calculation library',
        '材料科学計算ライブラリ': 'Materials science calculation library',

        # Footer
        'MS Terakoya - Materials Science Learning Platform':
            'MS Terakoya - Materials Science Learning Platform',
        '東北大学 材料科学研究科':
            'Graduate School of Materials Science, Tohoku University',
    }

    # Apply translations
    translated_content = content
    for jp, en in translations.items():
        translated_content = translated_content.replace(jp, en)

    # Ensure target directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write translated content
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(translated_content)

    # Count Japanese characters after translation
    remaining_jp = len(re.findall(r'[ぁ-んァ-ヶー一-龯]', translated_content))

    print(f"\nTranslation complete!")
    print(f"Target file: {target_path}")
    print(f"\nTranslation statistics:")
    print(f"  Original Japanese chars: {japanese_char_count}")
    print(f"  Remaining Japanese chars: {remaining_jp}")
    print(f"  Translation coverage: {((japanese_char_count - remaining_jp) / japanese_char_count * 100):.2f}%")

    return japanese_char_count, jp_percentage

if __name__ == "__main__":
    translate_ceramics_chapter1()
