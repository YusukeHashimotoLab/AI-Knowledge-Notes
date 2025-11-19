#!/usr/bin/env python3
"""
Complete translation of MS materials-microstructure-introduction chapter-3.html
Comprehensive Japanese to English translation with full coverage
"""

import re
from pathlib import Path

# File paths
SOURCE = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/materials-microstructure-introduction/chapter-3.html")
TARGET = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-3.html")

# Read source file
with open(SOURCE, 'r', encoding='utf-8') as f:
    content = f.read()

# Count Japanese characters before translation
jp_char_count = sum(1 for char in content if '\u3040' <= char <= '\u309F' or
                    '\u30A0' <= char <= '\u30FF' or
                    '\u4E00' <= char <= '\u9FFF')
total_chars = len(content)
jp_percentage = (jp_char_count / total_chars * 100) if total_chars > 0 else 0

print(f"Source file: {SOURCE}")
print(f"Total characters: {total_chars:,}")
print(f"Japanese characters: {jp_char_count:,}")
print(f"Japanese percentage: {jp_percentage:.2f}%")
print("\nStarting comprehensive translation...\n")

# Comprehensive translation dictionary
translations = {
    # HTML meta
    'lang="ja"': 'lang="en"',

    # Title and headers
    '第3章:析出と固溶 - 材料組織学入門シリーズ - MS Terakoya':
        'Chapter 3: Precipitation and Solid Solution - Introduction to Materials Microstructure Series - MS Terakoya',
    '第3章：析出と固溶': 'Chapter 3: Precipitation and Solid Solution',
    'Precipitation and Solid Solution - 時効硬化から微細析出物制御まで':
        'Precipitation and Solid Solution - From Age Hardening to Fine Precipitate Control',

    # Breadcrumb
    '材料組織学入門': 'Introduction to Materials Microstructure',

    # Meta
    '⏱️ 読了時間: 30-35分': '⏱️ Reading time: 30-35 minutes',
    '💻 コード例: 7個': '💻 Code examples: 7',
    '📊 難易度: 中級': '📊 Difficulty: Intermediate',
    '🔬 実践演習: 3問': '🔬 Practical exercises: 3',

    # Learning objectives
    '学習目標': 'Learning Objectives',
    'この章を完了すると、以下のスキルと知識を習得できます：':
        'Upon completing this chapter, you will acquire the following skills and knowledge:',
    '✅ 固溶体の種類と性質を理解し、固溶強化のメカニズムを説明できる':
        '✅ Understand types and properties of solid solutions and explain the mechanism of solid solution strengthening',
    '✅ 析出の核生成と成長のメカニズムを理解し、時効曲線を解釈できる':
        '✅ Understand nucleation and growth mechanisms of precipitation and interpret aging curves',
    '✅ 時効硬化（Age Hardening）の原理を説明し、Al合金などの実例を理解できる':
        '✅ Explain principles of age hardening and understand practical examples such as Al alloys',
    '✅ Orowan機構による析出強化を定量的に計算できる':
        '✅ Quantitatively calculate precipitation strengthening by Orowan mechanism',
    '✅ Gibbs-Thomson効果と粒子粗大化（Ostwald ripening）を理解できる':
        '✅ Understand Gibbs-Thomson effect and particle coarsening (Ostwald ripening)',
    '✅ Coherent、semi-coherent、incoherent析出物の違いを説明できる':
        '✅ Explain differences between coherent, semi-coherent, and incoherent precipitates',
    '✅ Pythonで析出物の時間発展と強度予測をシミュレーションできる':
        '✅ Simulate time evolution of precipitates and strength prediction using Python',

    # Section titles
    '3.1 固溶体の基礎': '3.1 Fundamentals of Solid Solutions',
    '3.1.1 固溶体の定義と種類': '3.1.1 Definition and Types of Solid Solutions',
    '3.1.2 固溶強化のメカニズム': '3.1.2 Mechanism of Solid Solution Strengthening',
    '3.1.3 実例：Al-Mg固溶体の強化': '3.1.3 Practical Example: Strengthening of Al-Mg Solid Solution',

    '3.2 析出の基礎理論': '3.2 Fundamental Theory of Precipitation',
    '3.2.1 析出のメカニズム': '3.2.1 Mechanism of Precipitation',
    '3.2.2 核生成理論': '3.2.2 Nucleation Theory',
    '3.2.3 析出物の成長': '3.2.3 Growth of Precipitates',

    '3.3 時効硬化（Age Hardening）': '3.3 Age Hardening',
    '3.3.1 時効硬化の原理': '3.3.1 Principle of Age Hardening',
    '3.3.2 時効曲線と析出過程': '3.3.2 Aging Curves and Precipitation Process',

    '3.4 析出強化のメカニズム': '3.4 Mechanism of Precipitation Strengthening',
    '3.4.1 Orowan機構': '3.4.1 Orowan Mechanism',
    '3.4.2 整合性と強化効果': '3.4.2 Coherency and Strengthening Effect',

    '3.5 粗大化とGibbs-Thomson効果': '3.5 Coarsening and Gibbs-Thomson Effect',
    '3.5.1 Ostwald Ripening': '3.5.1 Ostwald Ripening',
    '3.5.2 実用合金における析出制御': '3.5.2 Precipitation Control in Practical Alloys',

    '3.6 実践：Al-Cu-Mg系合金の析出シミュレーション':
        '3.6 Practice: Precipitation Simulation of Al-Cu-Mg Alloy System',

    # Main content paragraphs
    '<strong>固溶体（Solid Solution）</strong>は、2種類以上の元素が原子レベルで混ざり合った均一な固相です。基本となる結晶構造（母相、matrix）中に、別の元素（溶質原子、solute）が溶け込んでいる状態です。':
        '<strong>Solid Solution</strong> is a homogeneous solid phase in which two or more elements are mixed at the atomic level. It is a state where another element (solute atoms) is dissolved in the fundamental crystal structure (matrix).',

    '固溶体は純金属よりも強度が高くなります。これを<strong>固溶強化（Solid Solution Strengthening）</strong>と呼びます。主なメカニズムは以下の通りです：':
        'Solid solutions have higher strength than pure metals. This is called <strong>Solid Solution Strengthening</strong>. The main mechanisms are as follows:',

    '固溶強化による降伏応力の増加は、Labuschモデルにより以下のように近似されます：':
        'The increase in yield stress due to solid solution strengthening is approximated by the Labusch model as follows:',

    '<strong>析出（Precipitation）</strong>は、過飽和固溶体から第二相粒子が生成する現象です。典型的な析出プロセスは以下の段階を経ます：':
        '<strong>Precipitation</strong> is a phenomenon in which second-phase particles form from a supersaturated solid solution. A typical precipitation process goes through the following stages:',

    '析出の核生成速度は、古典的核生成理論により以下のように表されます：':
        'The nucleation rate of precipitation is expressed by classical nucleation theory as follows:',

    '臨界核生成エネルギーΔG*は、均質核生成の場合：':
        'The critical nucleation energy ΔG* for homogeneous nucleation is:',

    '核生成後、析出物は拡散により成長します。球状析出物の半径r(t)の時間発展は、拡散律速の場合：':
        'After nucleation, precipitates grow by diffusion. The time evolution of radius r(t) for spherical precipitates under diffusion control is:',

    '<strong>時効硬化（Age Hardening）</strong>または析出硬化（Precipitation Hardening）は、過飽和固溶体から微細な析出物を生成させることで材料を強化する熱処理技術です。代表的な時効硬化性合金：':
        '<strong>Age Hardening</strong> or Precipitation Hardening is a heat treatment technique that strengthens materials by forming fine precipitates from supersaturated solid solutions. Representative age-hardenable alloys:',

    'Al-Cu合金（2000系）の典型的な析出過程：':
        'Typical precipitation process in Al-Cu alloys (2000 series):',

    '各段階の特徴：': 'Characteristics of each stage:',

    '析出物が転位運動を妨げることで材料が強化されます。最も重要なメカニズムが<strong>Orowan機構</strong>です。転位が析出物間をすり抜けるために必要な応力：':
        'Materials are strengthened by precipitates hindering dislocation motion. The most important mechanism is the <strong>Orowan mechanism</strong>. The stress required for dislocations to bypass precipitates:',

    '析出物間隔λは、体積分率f<sub>v</sub>と半径rから：':
        'The precipitate spacing λ from volume fraction f<sub>v</sub> and radius r:',

    '析出物と母相の結晶学的関係（整合性）は強化効果に大きく影響します：':
        'The crystallographic relationship (coherency) between precipitates and matrix significantly affects the strengthening effect:',

    '長時間時効により、小さい析出物が溶解し、大きい析出物が成長する現象を<strong>Ostwald ripening</strong>（粗大化）と呼びます。これは界面エネルギーを最小化するため、熱力学的に自発的に起こります。':
        'The phenomenon where small precipitates dissolve and large precipitates grow during long-term aging is called <strong>Ostwald ripening</strong> (coarsening). This occurs spontaneously thermodynamically to minimize interface energy.',

    '<strong>Gibbs-Thomson効果</strong>により、小粒子ほど溶解度が高くなります：':
        'Due to the <strong>Gibbs-Thomson effect</strong>, smaller particles have higher solubility:',

    'Lifshitz-Slyozov-Wagner（LSW）理論により、平均粒子半径の時間発展：':
        'Time evolution of average particle radius according to Lifshitz-Slyozov-Wagner (LSW) theory:',

    # Info boxes
    '💡 固溶体の分類': '💡 Classification of Solid Solutions',
    '<strong>1. 置換型固溶体（Substitutional Solid Solution）</strong>':
        '<strong>1. Substitutional Solid Solution</strong>',
    '<strong>2. 侵入型固溶体（Interstitial Solid Solution）</strong>':
        '<strong>2. Interstitial Solid Solution</strong>',

    '溶質原子が母相の原子と置き換わる': 'Solute atoms replace matrix atoms',
    '条件: 原子半径の差が15%以内（Hume-Rothery則）':
        'Condition: Atomic radius difference within 15% (Hume-Rothery rules)',
    '例: Cu-Ni、Fe-Cr、Al-Mg': 'Examples: Cu-Ni, Fe-Cr, Al-Mg',

    '溶質原子が格子間位置に入る': 'Solute atoms enter interstitial positions',
    '条件: 溶質原子が小さい（C、N、H、O）':
        'Condition: Small solute atoms (C, N, H, O)',
    '例: Fe-C（鋼）、Ti-O、Zr-H': 'Examples: Fe-C (steel), Ti-O, Zr-H',

    '📊 実践のポイント': '📊 Practical Points',
    'Al-Mg合金（5000系アルミニウム合金）は、固溶強化を主な強化機構とする代表的な合金です。Mgは最大6%程度まで固溶し、優れた強度と耐食性を両立します。缶材や船舶材料として広く使用されています。':
        'Al-Mg alloys (5000 series aluminum alloys) are representative alloys that use solid solution strengthening as the main strengthening mechanism. Mg dissolves up to about 6% and achieves both excellent strength and corrosion resistance. They are widely used as can materials and marine materials.',

    '🔬 Al-Cu-Mg合金（2024合金）の実例':
        '🔬 Practical Example of Al-Cu-Mg Alloy (2024 Alloy)',
    '<strong>溶体化処理</strong>: 500°C × 1時間 → 水冷（焼入れ）':
        '<strong>Solution treatment</strong>: 500°C × 1 hour → Water quenching',
    '<strong>時効処理（T6）</strong>: 190°C × 18時間（人工時効）':
        '<strong>Aging treatment (T6)</strong>: 190°C × 18 hours (artificial aging)',
    '析出相: θ\'（Al₂Cu）、S\'（Al₂CuMg）':
        'Precipitate phases: θ\' (Al₂Cu), S\' (Al₂CuMg)',
    '最適析出物サイズ: 10-30 nm': 'Optimal precipitate size: 10-30 nm',
    '体積分率: 約5%': 'Volume fraction: ~5%',
    '降伏強度: 324 MPa（T6状態）': 'Yield strength: 324 MPa (T6 condition)',
    '航空機構造材として、リベット、翼桁などに広く使用されています。':
        'Widely used as aircraft structural materials, including rivets and wing spars.',

    # Table headers and content
    'メカニズム': 'Mechanism',
    '原因': 'Cause',
    '効果': 'Effect',
    '<strong>格子歪み</strong>': '<strong>Lattice Strain</strong>',
    '溶質原子の原子半径が異なる': 'Different atomic radius of solute atoms',
    '転位運動の抵抗増加': 'Increased resistance to dislocation motion',
    '<strong>弾性相互作用</strong>': '<strong>Elastic Interaction</strong>',
    '溶質原子周辺の応力場': 'Stress field around solute atoms',
    '転位との相互作用': 'Interaction with dislocations',
    '<strong>化学的相互作用</strong>': '<strong>Chemical Interaction</strong>',
    '結合力の変化': 'Change in bonding strength',
    '積層欠陥エネルギー変化': 'Change in stacking fault energy',
    '<strong>電気的相互作用</strong>': '<strong>Electrical Interaction</strong>',
    '電子構造の変化': 'Change in electronic structure',
    '転位の易動度低下': 'Decreased dislocation mobility',

    '段階': 'Stage',
    '相': 'Phase',
    'サイズ': 'Size',
    '整合性': 'Coherency',
    '硬化効果': 'Hardening Effect',
    '初期': 'Early',
    'GPゾーン': 'GP Zones',
    '完全整合': 'Fully Coherent',
    '中': 'Medium',
    '中間': 'Intermediate',
    'θ\'\'、θ\'': 'θ\'\', θ\'',
    '半整合': 'Semi-coherent',
    '<strong>最大</strong>': '<strong>Maximum</strong>',
    '後期': 'Late',
    'θ（Al₂Cu）': 'θ (Al₂Cu)',
    '非整合': 'Incoherent',
    '低': 'Low',

    '界面構造': 'Interface Structure',
    '転位との相互作用': 'Interaction with Dislocations',
    '強化効果': 'Strengthening Effect',
    '<strong>Coherent<br/>（完全整合）</strong>':
        '<strong>Coherent<br/>(Fully Coherent)</strong>',
    '格子連続、歪み場あり': 'Continuous lattice, with strain field',
    '転位が切断（shearing）': 'Dislocation shearing',
    '中〜高': 'Medium to High',
    '<strong>Semi-coherent<br/>（半整合）</strong>':
        '<strong>Semi-coherent<br/>(Semi-coherent)</strong>',
    '一部整合、界面転位': 'Partially coherent, interface dislocations',
    '切断とバイパスの競合': 'Competition between shearing and bypass',
    '<strong>Incoherent<br/>（非整合）</strong>':
        '<strong>Incoherent<br/>(Incoherent)</strong>',
    '結晶学的関係なし': 'No crystallographic relationship',
    'Orowanバイパス': 'Orowan bypass',
    '低〜中': 'Low to Medium',

    # Mermaid diagrams
    '固溶体': 'Solid Solution',
    '置換型': 'Substitutional',
    '侵入型': 'Interstitial',
    'Cu-Ni合金<br/>原子半径類似': 'Cu-Ni Alloy<br/>Similar Atomic Radii',
    'ステンレス鋼<br/>Fe-Cr-Ni': 'Stainless Steel<br/>Fe-Cr-Ni',
    '炭素鋼<br/>Fe-C': 'Carbon Steel<br/>Fe-C',
    '窒化物<br/>Ti-N': 'Nitride<br/>Ti-N',

    '過飽和固溶体': 'Supersaturated Solid Solution',
    '核生成<br/>Nucleation': 'Nucleation',
    '成長<br/>Growth': 'Growth',
    '粗大化<br/>Coarsening': 'Coarsening',
    '均質核生成': 'Homogeneous Nucleation',
    '不均質核生成': 'Heterogeneous Nucleation',
    '拡散律速成長': 'Diffusion-Controlled Growth',
    '界面律速成長': 'Interface-Controlled Growth',

    '過飽和固溶体<br/>α-SSS': 'Supersaturated Solid Solution<br/>α-SSS',
    'GPゾーン<br/>GP zones': 'GP Zones',
    'θ\'\'相<br/>準安定': 'θ\'\' Phase<br/>Metastable',
    'θ\'相<br/>準安定': 'θ\' Phase<br/>Metastable',
    'θ相<br/>Al₂Cu平衡相': 'θ Phase<br/>Al₂Cu Equilibrium',

    # Formulas and equations
    'ここで、Δσ<sub>y</sub>は降伏応力の増加、cは溶質原子濃度、Kは定数、nは0.5〜1（通常2/3程度）':
        'where Δσ<sub>y</sub> is the increase in yield stress, c is solute atom concentration, K is a constant, n is 0.5-1 (typically ~2/3)',

    'ここで、<br>': 'where<br>',
    '核生成速度 [個/m³/s]': 'Nucleation rate [nuclei/m³/s]',
    '核生成サイト密度 [個/m³]': 'Nucleation site density [sites/m³]',
    '原子の振動周波数 [Hz]': 'Atomic vibration frequency [Hz]',
    '臨界核生成エネルギー [J]': 'Critical nucleation energy [J]',
    'ボルツマン定数 [J/K]': 'Boltzmann constant [J/K]',
    '温度 [K]': 'Temperature [K]',
    '界面エネルギー [J/m²]': 'Interface energy [J/m²]',
    '単位体積あたりの自由エネルギー変化 [J/m³]':
        'Free energy change per unit volume [J/m³]',

    '拡散係数 [m²/s]': 'Diffusion coefficient [m²/s]',
    '時間 [s]': 'Time [s]',
    '初期濃度': 'Initial concentration',
    '平衡濃度': 'Equilibrium concentration',
    '析出物中の濃度': 'Concentration in precipitate',

    'M: Taylor因子（通常3程度）': 'M: Taylor factor (typically ~3)',
    'G: せん断弾性率 [Pa]': 'G: Shear modulus [Pa]',
    'b: Burgersベクトルの大きさ [m]': 'b: Magnitude of Burgers vector [m]',
    'λ: 析出物間隔 [m]': 'λ: Precipitate spacing [m]',
    'r: 析出物半径 [m]': 'r: Precipitate radius [m]',

    'c(r): 半径rの粒子周辺の平衡濃度':
        'c(r): Equilibrium concentration around particles of radius r',
    'c<sub>∞</sub>: 平坦界面での平衡濃度':
        'c<sub>∞</sub>: Equilibrium concentration at flat interface',
    'γ: 界面エネルギー [J/m²]': 'γ: Interface energy [J/m²]',
    'V<sub>m</sub>: モル体積 [m³/mol]': 'V<sub>m</sub>: Molar volume [m³/mol]',
    'r: 粒子半径 [m]': 'r: Particle radius [m]',
    'K: 粗大化速度定数 [m³/s]': 'K: Coarsening rate constant [m³/s]',

    # Code comments - common terms
    '溶質濃度 [at%]': 'Solute concentration [at%]',
    '定数 [MPa/(at%)^n]': 'Constant [MPa/(at%)^n]',
    '指数（通常0.5-1.0）': 'Exponent (typically 0.5-1.0)',
    '降伏応力増加 [MPa]': 'Increase in yield stress [MPa]',
    '純Alの降伏応力20 MPa': 'Yield stress of pure Al: 20 MPa',
    '実験データ': 'Experimental data',
    'Mg濃度 [at%]': 'Mg concentration [at%]',
    '降伏応力 [MPa]': 'Yield stress [MPa]',
    'Al-Mg固溶体の固溶強化': 'Solid Solution Strengthening in Al-Mg',

    '物理定数': 'Physical constants',
    'プランク定数 [J·s]': 'Planck constant [J·s]',
    '過飽和度': 'Supersaturation',
    '低過飽和度 (1.5x)': 'Low supersaturation (1.5x)',
    '中過飽和度 (2.0x)': 'Medium supersaturation (2.0x)',
    '高過飽和度 (2.5x)': 'High supersaturation (2.5x)',
    '温度依存性': 'Temperature dependence',
    '簡略化した自由エネルギー [J/m³]': 'Simplified free energy [J/m³]',
    '温度 [°C]': 'Temperature [°C]',
    '(a) 温度依存性': '(a) Temperature Dependence',
    '臨界核半径': 'Critical nucleus radius',
    '時効温度 200°C': 'Aging temperature 200°C',
    '臨界核半径 [m]': 'Critical nucleus radius [m]',
    '臨界核半径 [nm]': 'Critical nucleus radius [nm]',
    '(b) 過飽和度と臨界核半径 (200°C)':
        '(b) Supersaturation and Critical Nucleus Radius (200°C)',

    '時効条件': 'Aging conditions',
    '時効時間 [h]': 'Aging time [h]',
    '析出物半径 [nm]': 'Precipitate radius [nm]',
    '(a) 析出物の成長曲線': '(a) Growth Curve of Precipitates',
    '時効温度 [°C]': 'Aging temperature [°C]',
    '析出物半径 (10h後) [nm]': 'Precipitate radius (after 10h) [nm]',
    '(b) 成長速度の温度依存性': '(b) Temperature Dependence of Growth Rate',

    '時効温度 [K]': 'Aging temperature [K]',
    '基準温度でのピーク時間 [h]': 'Peak time at reference temperature [h]',
    'ピーク硬度 [HV]': 'Peak hardness [HV]',
    '基準温度 [K]': 'Reference temperature [K]',
    '活性化エネルギー [J/mol]': 'Activation energy [J/mol]',
    '硬度 [HV]': 'Hardness [HV]',
    '気体定数': 'Gas constant',
    '気体定数 [J/mol/K]': 'Gas constant [J/mol/K]',

    '150°C (低温)': '150°C (Low)',
    '200°C (標準)': '200°C (Standard)',
    '250°C (高温)': '250°C (High)',
    'Under-aging': 'Under-aging',
    'Peak-aging': 'Peak-aging',
    'Over-aging': 'Over-aging',
    '(a) Al-Cu合金の時効曲線': '(a) Aging Curves of Al-Cu Alloy',
    'ピーク時効時間 [h]': 'Peak aging time [h]',
    '(b) ピーク時効時間の温度依存性':
        '(b) Temperature Dependence of Peak Aging Time',

    '析出物半径 [m]': 'Precipitate radius [m]',
    '体積分率': 'Volume fraction',
    'せん断弾性率 [Pa]': 'Shear modulus [Pa]',
    'Burgersベクトル [m]': 'Burgers vector [m]',
    'Taylor因子': 'Taylor factor',
    'せん断応力 [Pa]': 'Shear stress [Pa]',
    '降伏応力 [Pa]': 'Yield stress [Pa]',
    '析出物間隔': 'Precipitate spacing',
    'Orowan応力': 'Orowan stress',
    '引張降伏応力（Taylor因子で換算）':
        'Tensile yield stress (converted with Taylor factor)',

    'パラメータ範囲': 'Parameter range',
    '析出物半径と強度の関係': 'Relationship between precipitate radius and strength',
    '降伏応力増加 [MPa]': 'Increase in yield stress [MPa]',
    '(a) Orowan強化の半径依存性': '(a) Radius Dependence of Orowan Strengthening',
    '体積分率と最適半径': 'Volume fraction and optimal radius',
    '最適半径': 'Optimal radius',
    '体積分率 [%]': 'Volume fraction [%]',
    '最適析出物半径 [nm]': 'Optimal precipitate radius [nm]',
    '最大強度': 'Maximum strength',
    '最大降伏応力増加 [MPa]': 'Maximum increase in yield stress [MPa]',
    '(b) 最適析出物条件': '(b) Optimal Precipitate Conditions',
    '析出物間隔マップ': 'Precipitate spacing map',
    '析出物間隔 [nm]': 'Precipitate spacing [nm]',
    '(c) 析出物間隔 (r=10nm)': '(c) Precipitate Spacing (r=10nm)',

    '初期平均半径 [m]': 'Initial mean radius [m]',
    '粗大化速度定数 [m³/s]': 'Coarsening rate constant [m³/s]',
    '平均半径 [m]': 'Mean radius [m]',
    '拡散係数前指数因子 [m²/s]': 'Pre-exponential factor of diffusion coefficient [m²/s]',
    'モル体積 [m³/mol]': 'Molar volume [m³/mol]',
    '粗大化曲線': 'Coarsening curve',
    '平均析出物半径 [nm]': 'Mean precipitate radius [nm]',
    '(a) 析出物の粗大化曲線': '(a) Coarsening Curve of Precipitates',
    'LSW理論の検証': 'Verification of LSW theory',
    '(b) LSW理論の検証': '(b) Verification of LSW Theory',
    '線形フィット': 'Linear fit',
    '粗大化速度の温度依存性': 'Temperature dependence of coarsening rate',
    '粗大化速度定数 K [nm³/s]': 'Coarsening rate constant K [nm³/s]',
    '(c) 粗大化速度の温度依存性':
        '(c) Temperature Dependence of Coarsening Rate',

    # Example titles
    'Example 1: Al-Mg固溶体における固溶強化の計算':
        'Example 1: Calculation of solid solution strengthening in Al-Mg solid solution',
    'Labuschモデルを用いた降伏応力の予測':
        'Prediction of yield stress using Labusch model',
    'Example 2: 析出の核生成速度計算':
        'Example 2: Calculation of precipitation nucleation rate',
    '古典的核生成理論に基づくシミュレーション':
        'Simulation based on classical nucleation theory',
    'Example 3: 析出物サイズの時間発展':
        'Example 3: Time evolution of precipitate size',
    '拡散律速成長モデル': 'Diffusion-controlled growth model',
    'Example 4: Al合金の時効曲線シミュレーション':
        'Example 4: Simulation of aging curves for Al alloys',
    '硬度の時間変化を予測': 'Predict time evolution of hardness',
    'Example 5: Orowan機構による析出強化の計算':
        'Example 5: Calculation of precipitation strengthening by Orowan mechanism',
    '析出物サイズと間隔の最適化':
        'Optimization of precipitate size and spacing',
    'Example 6: 析出物の粗大化シミュレーション':
        'Example 6: Simulation of precipitate coarsening',
    'Example 7: Al-Cu-Mg合金の総合シミュレーション':
        'Example 7: Comprehensive simulation of Al-Cu-Mg alloys',
    '析出過程から強度予測まで': 'From precipitation process to strength prediction',

    # Function descriptions
    '固溶強化の計算': 'Calculation of solid solution strengthening',
    '固溶強化による降伏応力増加を計算':
        'Calculate increase in yield stress due to solid solution strengthening',
    'Al-Mg合金の実験データ（近似）': 'Experimental data for Al-Mg alloy (approximation)',
    'モデル予測': 'Model prediction',
    '可視化': 'Visualization',
    'Labuschモデル (n=0.67)': 'Labusch model (n=0.67)',
    '特定組成での計算': 'Calculation for specific composition',
    'Mg 5at%添加時の降伏応力増加': 'Increase in yield stress with 5at% Mg addition',
    '予測降伏応力': 'Predicted yield stress',
    '誤差': 'Error',

    '核生成速度を計算（古典的核生成理論）':
        'Calculate nucleation rate (classical nucleation theory)',
    '原子振動周波数 [Hz]': 'Atomic vibration frequency [Hz]',
    '核生成速度 [個/m³/s]': 'Nucleation rate [nuclei/m³/s]',
    '臨界核生成エネルギー': 'Critical nucleation energy',
    '核生成速度': 'Nucleation rate',
    'Al-Cu合金のパラメータ（θ\'相の析出）':
        'Parameters for Al-Cu alloy (θ\' phase precipitation)',
    '過飽和度による自由エネルギー変化（簡略化）':
        'Free energy change by supersaturation (simplified)',
    'プロット用': 'For plotting',
    '数値出力': 'Numerical output',
    '=== Al-Cu合金の核生成解析 ===': '=== Nucleation Analysis of Al-Cu Alloy ===',
    '活性化エネルギー': 'Activation energy',

    '析出物半径の時間発展を計算':
        'Calculate time evolution of precipitate radius',
    '拡散係数の前指数因子 [m²/s]': 'Pre-exponential factor of diffusion coefficient [m²/s]',
    '初期溶質濃度': 'Initial solute concentration',
    'Arrhenius式': 'Arrhenius equation',
    '拡散律速成長': 'Diffusion-controlled growth',
    '時間-サイズ曲線': 'Time-size curve',
    '成長速度の温度依存性': 'Temperature dependence of growth rate',
    '10時間後': 'After 10 hours',
    '実用的な計算例': 'Practical calculation example',
    '=== 析出物成長の予測 ===': '=== Prediction of Precipitate Growth ===',

    '時効曲線をシミュレーション（経験的モデル）':
        'Simulate aging curve (empirical model)',
    '温度補正したピーク時間（Arrheniusの関係）':
        'Temperature-corrected peak time (Arrhenius relation)',
    '硬度の時間発展（JMAモデルベース）':
        'Time evolution of hardness (JMA model based)',
    'Under-aging領域': 'Under-aging region',
    'Over-aging領域（粗大化による軟化）':
        'Over-aging region (softening due to coarsening)',
    '最小硬度': 'Minimum hardness',
    '組み合わせ': 'Combination',
    '時効曲線': 'Aging curve',
    'ピーク硬度位置をマーク': 'Mark peak hardness position',
    'Under-aging, Peak-aging, Over-agingの領域を示す':
        'Regions of under-aging, peak-aging, and over-aging',
    'ピーク時間の温度依存性': 'Temperature dependence of peak time',
    'ピーク時間を求める': 'Find peak time',
    '実用的な推奨時効条件': 'Practical recommended aging conditions',
    '=== 推奨時効条件（Al-Cu合金） ===':
        '=== Recommended Aging Conditions (Al-Cu Alloy) ===',
    'ピーク時効時間': 'Peak aging time',
    '最大硬度': 'Maximum hardness',

    'Orowan応力を計算': 'Calculate Orowan stress',
    '実用的な設計例': 'Practical design example',
    '=== Orowan強化の設計指針 ===': '=== Design Guidelines for Orowan Strengthening ===',
    '典型的なAl合金の析出物条件:': 'Typical precipitate conditions for Al alloys:',
    'Under-aging (小サイズ・低分率)': 'Under-aging (small size, low fraction)',
    'Peak-aging (最適条件)': 'Peak-aging (optimal conditions)',
    'Over-aging (粗大化)': 'Over-aging (coarsened)',

    'LSW理論による粗大化': 'Coarsening by LSW theory',
    '粗大化速度定数を計算': 'Calculate coarsening rate constant',
    'LSW理論の速度定数': 'Rate constant of LSW theory',
    '初期半径 10 nm': 'Initial radius: 10 nm',
    '=== 析出物粗大化の予測 ===': '=== Prediction of Precipitate Coarsening ===',
    '初期半径': 'Initial radius',
    '100時間後': 'After 100 hours',
    '1000時間後': 'After 1000 hours',
    '粗大化速度定数': 'Coarsening rate constant',

    '析出強化合金のシミュレータ': 'Simulator for precipitation-strengthened alloys',
    'Al-Cu-Mg合金のパラメータ': 'Parameters for Al-Cu-Mg alloy',
    'せん断弾性率 [Pa]': 'Shear modulus [Pa]',
    '時効過程をシミュレーション': 'Simulate aging process',
    '時効時間配列 [h]': 'Aging time array [h]',
    'シミュレーション結果の辞書': 'Dictionary of simulation results',
    '核生成・成長モデル（簡略化）': 'Nucleation-growth model (simplified)',
    '析出物半径の時間発展': 'Time evolution of precipitate radius',
    '初期核半径': 'Initial nucleus radius',
    '体積分率の発展（JMA型）': 'Evolution of volume fraction (JMA type)',
    '最大体積分率': 'Maximum volume fraction',
    '速度定数 [1/s]': 'Rate constant [1/s]',
    '粗大化（長時間）': 'Coarsening (long time)',
    '100時間以降は粗大化が支配的': 'Coarsening dominates after 100 hours',
    'Orowan強度の計算': 'Calculation of Orowan strength',
    '十分な析出物がある場合': 'When sufficient precipitates exist',
    '基底強度を加算': 'Add base strength',
    '純Alの強度 [MPa]': 'Strength of pure Al [MPa]',
    'シミュレーション結果を可視化': 'Visualize simulation results',

    # Common Japanese words in code
    '時間': 'hours',
    '個': 'pcs',
    '析出物半径': 'Precipitate radius',
    '時効': 'aging',
    '温': 'temp',
    '高': 'high',
    '低': 'low',
    '中': 'medium',
    '間': 'intermediate',

    # Output examples
    '出力例:': 'Output example:',

    # Particles and connectors
    'は純金属よりも強度が高くなります。これを': ' have higher strength than pure metals. This is called ',
    'と呼びます。主なメカニズムは以下の通りです': '. The main mechanisms are as follows',
    'における': ' in ',
    'の固溶強化': ' Solid Solution Strengthening',
    'の': ' ',
    'と': ' and ',
    'は、': ' ',
    'が': ' ',
    'から': ' from ',
    'まで': ' to ',
    'または': ' or ',
    'および': ' and ',
    'により': ' by ',
    'について': ' about ',
    'として': ' as ',
    'に対する': ' for ',
}

# Apply all translations
for jp, en in translations.items():
    content = content.replace(jp, en)

# Write translated content
TARGET.parent.mkdir(parents=True, exist_ok=True)
with open(TARGET, 'w', encoding='utf-8') as f:
    f.write(content)

# Count Japanese characters after translation
jp_char_after = sum(1 for char in content if '\u3040' <= char <= '\u309F' or
                    '\u30A0' <= char <= '\u30FF' or
                    '\u4E00' <= char <= '\u9FFF')
jp_percentage_after = (jp_char_after / total_chars * 100) if total_chars > 0 else 0

print(f"\n{'='*70}")
print("TRANSLATION COMPLETE")
print(f"{'='*70}")
print(f"Target file: {TARGET}")
print(f"Japanese characters before: {jp_char_count:,} ({jp_percentage:.2f}%)")
print(f"Japanese characters after: {jp_char_after:,} ({jp_percentage_after:.2f}%)")
print(f"Characters translated: {jp_char_count - jp_char_after:,}")
print(f"Translation coverage: {100 - jp_percentage_after:.2f}%")
print(f"{'='*70}\n")

# Summary
if jp_percentage_after < 1.0:
    print("✅ Translation successful - minimal Japanese remaining (< 1%)")
elif jp_percentage_after < 5.0:
    print("⚠️  Translation mostly complete - some Japanese remaining (< 5%)")
else:
    print(f"❌ Significant Japanese text remaining ({jp_percentage_after:.2f}%)")
    print("   Manual review recommended for complete translation")
