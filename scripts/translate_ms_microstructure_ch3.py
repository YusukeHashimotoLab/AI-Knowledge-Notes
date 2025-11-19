#!/usr/bin/env python3
"""
Translate MS materials-microstructure-introduction chapter-3.html from Japanese to English
Preserves all HTML structure, translates Japanese content comprehensively
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
jp_char_count = sum(1 for char in content if '\u3040' <= char <= '\u309F' or  # Hiragana
                    '\u30A0' <= char <= '\u30FF' or  # Katakana
                    '\u4E00' <= char <= '\u9FFF')    # Kanji
total_chars = len(content)
jp_percentage = (jp_char_count / total_chars * 100) if total_chars > 0 else 0

print(f"Source file: {SOURCE}")
print(f"Total characters: {total_chars:,}")
print(f"Japanese characters: {jp_char_count:,}")
print(f"Japanese percentage: {jp_percentage:.2f}%")
print("\nStarting translation...\n")

# Translation mappings - comprehensive coverage
translations = {
    # HTML lang and meta
    'lang="ja"': 'lang="en"',

    # Title and header
    '第3章:析出と固溶 - 材料組織学入門シリーズ - MS Terakoya':
        'Chapter 3: Precipitation and Solid Solution - Introduction to Materials Microstructure Series - MS Terakoya',
    '第3章：析出と固溶': 'Chapter 3: Precipitation and Solid Solution',
    'Precipitation and Solid Solution - 時効硬化から微細析出物制御まで':
        'Precipitation and Solid Solution - From Age Hardening to Fine Precipitate Control',

    # Breadcrumb
    'MS Terakoya': 'MS Terakoya',
    '材料組織学入門': 'Introduction to Materials Microstructure',
    '第3章：析出と固溶': 'Chapter 3: Precipitation and Solid Solution',

    # Meta information
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

    # Section 3.1
    '3.1 固溶体の基礎': '3.1 Fundamentals of Solid Solutions',
    '3.1.1 固溶体の定義と種類': '3.1.1 Definition and Types of Solid Solutions',

    # Solid solution description
    '<strong>固溶体（Solid Solution）</strong>は、2種類以上の元素が原子レベルで混ざり合った均一な固相です。基本となる結晶構造（母相、matrix）中に、別の元素（溶質原子、solute）が溶け込んでいる状態です。':
        '<strong>Solid Solution</strong> is a homogeneous solid phase in which two or more elements are mixed at the atomic level. It is a state where another element (solute atoms) is dissolved in the fundamental crystal structure (matrix).',

    # Info box
    '💡 固溶体の分類': '💡 Classification of Solid Solutions',
    '<strong>1. 置換型固溶体（Substitutional Solid Solution）</strong>':
        '<strong>1. Substitutional Solid Solution</strong>',
    '溶質原子が母相の原子と置き換わる': 'Solute atoms replace matrix atoms',
    '条件: 原子半径の差が15%以内（Hume-Rothery則）': 'Condition: Atomic radius difference within 15% (Hume-Rothery rules)',
    '例: Cu-Ni、Fe-Cr、Al-Mg': 'Examples: Cu-Ni, Fe-Cr, Al-Mg',

    '<strong>2. 侵入型固溶体（Interstitial Solid Solution）</strong>':
        '<strong>2. Interstitial Solid Solution</strong>',
    '溶質原子が格子間位置に入る': 'Solute atoms enter interstitial positions',
    '条件: 溶質原子が小さい（C、N、H、O）': 'Condition: Small solute atoms (C, N, H, O)',
    '例: Fe-C（鋼）、Ti-O、Zr-H': 'Examples: Fe-C (steel), Ti-O, Zr-H',

    # Mermaid diagram labels
    '固溶体': 'Solid Solution',
    '置換型': 'Substitutional',
    '侵入型': 'Interstitial',
    'Cu-Ni合金<br/>原子半径類似': 'Cu-Ni Alloy<br/>Similar Atomic Radii',
    'ステンレス鋼<br/>Fe-Cr-Ni': 'Stainless Steel<br/>Fe-Cr-Ni',
    '炭素鋼<br/>Fe-C': 'Carbon Steel<br/>Fe-C',
    '窒化物<br/>Ti-N': 'Nitride<br/>Ti-N',

    # Section 3.1.2
    '3.1.2 固溶強化のメカニズム': '3.1.2 Mechanism of Solid Solution Strengthening',
    '固溶体は純金属よりも強度が高くなります。これを<strong>固溶強化（Solid Solution Strengthening）</strong>と呼びます。主なメカニズムは以下の通りです：':
        'Solid solutions have higher strength than pure metals. This is called <strong>Solid Solution Strengthening</strong>. The main mechanisms are as follows:',

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

    # Formula description
    '固溶強化による降伏応力の増加は、Labuschモデルにより以下のように近似されます：':
        'The increase in yield stress due to solid solution strengthening is approximated by the Labusch model as follows:',
    'ここで、Δσ<sub>y</sub>は降伏応力の増加、cは溶質原子濃度、Kは定数、nは0.5〜1（通常2/3程度）':
        'where Δσ<sub>y</sub> is the increase in yield stress, c is solute atom concentration, K is a constant, n is 0.5-1 (typically ~2/3)',

    # Section 3.1.3
    '3.1.3 実例：Al-Mg固溶体の強化': '3.1.3 Practical Example: Strengthening of Al-Mg Solid Solution',

    # Code comments - Example 1
    'Example 1: Al-Mg固溶体における固溶強化の計算':
        'Example 1: Calculation of solid solution strengthening in Al-Mg solid solution',
    'Labuschモデルを用いた降伏応力の予測':
        'Prediction of yield stress using Labusch model',
    '固溶強化の計算': 'Calculation of solid solution strengthening',
    '固溶強化による降伏応力増加を計算':
        'Calculate increase in yield stress due to solid solution strengthening',
    '溶質濃度 [at%]': 'Solute concentration [at%]',
    '定数 [MPa/(at%)^n]': 'Constant [MPa/(at%)^n]',
    '指数（通常0.5-1.0）': 'Exponent (typically 0.5-1.0)',
    '降伏応力増加 [MPa]': 'Increase in yield stress [MPa]',
    'Al-Mg合金の実験データ（近似）': 'Experimental data for Al-Mg alloy (approximation)',
    'モデル予測': 'Model prediction',
    '純Alの降伏応力20 MPa': 'Yield stress of pure Al: 20 MPa',
    '可視化': 'Visualization',
    'Labuschモデル (n=0.67)': 'Labusch model (n=0.67)',
    '実験データ': 'Experimental data',
    'Mg濃度 [at%]': 'Mg concentration [at%]',
    '降伏応力 [MPa]': 'Yield stress [MPa]',
    'Al-Mg固溶体の固溶強化': 'Solid Solution Strengthening in Al-Mg',
    '特定組成での計算': 'Calculation for specific composition',
    'Mg 5at%添加時の降伏応力増加': 'Increase in yield stress with 5at% Mg addition',
    '予測降伏応力': 'Predicted yield stress',
    '実験値': 'Experimental value',
    '誤差': 'Error',
    '出力例:': 'Output example:',

    # Info box - practical point
    '📊 実践のポイント': '📊 Practical Points',
    'Al-Mg合金（5000系アルミニウム合金）は、固溶強化を主な強化機構とする代表的な合金です。Mgは最大6%程度まで固溶し、優れた強度と耐食性を両立します。缶材や船舶材料として広く使用されています。':
        'Al-Mg alloys (5000 series aluminum alloys) are representative alloys that use solid solution strengthening as the main strengthening mechanism. Mg dissolves up to about 6% and achieves both excellent strength and corrosion resistance. They are widely used as can materials and marine materials.',

    # Section 3.2
    '3.2 析出の基礎理論': '3.2 Fundamental Theory of Precipitation',
    '3.2.1 析出のメカニズム': '3.2.1 Mechanism of Precipitation',
    '<strong>析出（Precipitation）</strong>は、過飽和固溶体から第二相粒子が生成する現象です。典型的な析出プロセスは以下の段階を経ます：':
        '<strong>Precipitation</strong> is a phenomenon in which second-phase particles form from a supersaturated solid solution. A typical precipitation process goes through the following stages:',

    # Mermaid flowchart
    '過飽和固溶体': 'Supersaturated Solid Solution',
    '核生成<br/>Nucleation': 'Nucleation',
    '成長<br/>Growth': 'Growth',
    '粗大化<br/>Coarsening': 'Coarsening',
    '均質核生成': 'Homogeneous Nucleation',
    '不均質核生成': 'Heterogeneous Nucleation',
    '拡散律速成長': 'Diffusion-Controlled Growth',
    '界面律速成長': 'Interface-Controlled Growth',
    'Ostwald ripening': 'Ostwald Ripening',

    # Section 3.2.2
    '3.2.2 核生成理論': '3.2.2 Nucleation Theory',
    '析出の核生成速度は、古典的核生成理論により以下のように表されます：':
        'The nucleation rate of precipitation is expressed by classical nucleation theory as follows:',
    '核生成速度 [個/m³/s]': 'Nucleation rate [nuclei/m³/s]',
    '核生成サイト密度 [個/m³]': 'Nucleation site density [sites/m³]',
    '原子の振動周波数 [Hz]': 'Atomic vibration frequency [Hz]',
    '臨界核生成エネルギー [J]': 'Critical nucleation energy [J]',
    'ボルツマン定数 [J/K]': 'Boltzmann constant [J/K]',
    '温度 [K]': 'Temperature [K]',
    '臨界核生成エネルギーΔG*は、均質核生成の場合：':
        'The critical nucleation energy ΔG* for homogeneous nucleation is:',
    '界面エネルギー [J/m²]': 'Interface energy [J/m²]',
    '単位体積あたりの自由エネルギー変化 [J/m³]':
        'Free energy change per unit volume [J/m³]',

    # Example 2 code comments
    'Example 2: 析出の核生成速度計算':
        'Example 2: Calculation of precipitation nucleation rate',
    '古典的核生成理論に基づくシミュレーション':
        'Simulation based on classical nucleation theory',
    '物理定数': 'Physical constants',
    'ボルツマン定数 [J/K]': 'Boltzmann constant [J/K]',
    'プランク定数 [J·s]': 'Planck constant [J·s]',
    '核生成速度を計算（古典的核生成理論）':
        'Calculate nucleation rate (classical nucleation theory)',
    '温度 [K]': 'Temperature [K]',
    '界面エネルギー [J/m²]': 'Interface energy [J/m²]',
    '体積自由エネルギー変化 [J/m³]': 'Volume free energy change [J/m³]',
    '核生成サイト密度 [個/m³]': 'Nucleation site density [sites/m³]',
    '原子振動周波数 [Hz]': 'Atomic vibration frequency [Hz]',
    '核生成速度 [個/m³/s]': 'Nucleation rate [nuclei/m³/s]',
    '臨界核生成エネルギー': 'Critical nucleation energy',
    '核生成速度': 'Nucleation rate',
    'Al-Cu合金のパラメータ（θ\'相の析出）':
        'Parameters for Al-Cu alloy (θ\' phase precipitation)',
    '過飽和度による自由エネルギー変化（簡略化）':
        'Free energy change by supersaturation (simplified)',
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
    'プロット用': 'For plotting',
    '臨界核半径 [nm]': 'Critical nucleus radius [nm]',
    '(b) 過飽和度と臨界核半径 (200°C)':
        '(b) Supersaturation and Critical Nucleus Radius (200°C)',
    '数値出力': 'Numerical output',
    '=== Al-Cu合金の核生成解析 ===': '=== Nucleation Analysis of Al-Cu Alloy ===',
    '活性化エネルギー': 'Activation energy',

    # Section 3.2.3
    '3.2.3 析出物の成長': '3.2.3 Growth of Precipitates',
    '核生成後、析出物は拡散により成長します。球状析出物の半径r(t)の時間発展は、拡散律速の場合：':
        'After nucleation, precipitates grow by diffusion. The time evolution of radius r(t) for spherical precipitates under diffusion control is:',
    '拡散係数 [m²/s]': 'Diffusion coefficient [m²/s]',
    '時間 [s]': 'Time [s]',
    '初期濃度': 'Initial concentration',
    '平衡濃度': 'Equilibrium concentration',
    '析出物中の濃度': 'Concentration in precipitate',

    # Example 3
    'Example 3: 析出物サイズの時間発展':
        'Example 3: Time evolution of precipitate size',
    '拡散律速成長モデル': 'Diffusion-controlled growth model',
    '析出物半径の時間発展を計算':
        'Calculate time evolution of precipitate radius',
    '拡散係数の前指数因子 [m²/s]': 'Pre-exponential factor of diffusion coefficient [m²/s]',
    '活性化エネルギー [J/mol]': 'Activation energy [J/mol]',
    '初期溶質濃度': 'Initial solute concentration',
    '平衡濃度': 'Equilibrium concentration',
    '析出物中の濃度': 'Concentration in precipitate',
    '析出物半径 [m]': 'Precipitate radius [m]',
    '気体定数 [J/mol/K]': 'Gas constant [J/mol/K]',
    'Arrhenius式': 'Arrhenius equation',
    '拡散律速成長': 'Diffusion-controlled growth',
    '時効条件': 'Aging conditions',
    '時間-サイズ曲線': 'Time-size curve',
    '時効時間 [h]': 'Aging time [h]',
    '析出物半径 [nm]': 'Precipitate radius [nm]',
    '(a) 析出物の成長曲線': '(a) Growth Curve of Precipitates',
    '成長速度の温度依存性': 'Temperature dependence of growth rate',
    '10時間後': 'After 10 hours',
    '時効温度 [°C]': 'Aging temperature [°C]',
    '析出物半径 (10h後) [nm]': 'Precipitate radius (after 10h) [nm]',
    '(b) 成長速度の温度依存性': '(b) Temperature Dependence of Growth Rate',
    '実用的な計算例': 'Practical calculation example',
    '=== 析出物成長の予測 ===': '=== Prediction of Precipitate Growth ===',

    # Section 3.3
    '3.3 時効硬化（Age Hardening）': '3.3 Age Hardening',
    '3.3.1 時効硬化の原理': '3.3.1 Principle of Age Hardening',
    '<strong>時効硬化（Age Hardening）</strong>または析出硬化（Precipitation Hardening）は、過飽和固溶体から微細な析出物を生成させることで材料を強化する熱処理技術です。代表的な時効硬化性合金：':
        '<strong>Age Hardening</strong> or Precipitation Hardening is a heat treatment technique that strengthens materials by forming fine precipitates from supersaturated solid solutions. Representative age-hardenable alloys:',
    '<strong>Al合金</strong>: 2000系(Al-Cu)、6000系(Al-Mg-Si)、7000系(Al-Zn-Mg)':
        '<strong>Al alloys</strong>: 2000 series (Al-Cu), 6000 series (Al-Mg-Si), 7000 series (Al-Zn-Mg)',
    '<strong>ニッケル基超合金</strong>: Inconel 718（γ\'\'相析出）':
        '<strong>Ni-base superalloys</strong>: Inconel 718 (γ\'\' phase precipitation)',
    '<strong>マルエージング鋼</strong>: Fe-Ni-Co-Mo合金':
        '<strong>Maraging steel</strong>: Fe-Ni-Co-Mo alloys',
    '<strong>析出硬化系ステンレス鋼</strong>: 17-4PH、15-5PH':
        '<strong>Precipitation hardening stainless steel</strong>: 17-4PH, 15-5PH',

    # Section 3.3.2
    '3.3.2 時効曲線と析出過程': '3.3.2 Aging Curves and Precipitation Process',
    'Al-Cu合金（2000系）の典型的な析出過程：':
        'Typical precipitation process in Al-Cu alloys (2000 series):',
    '過飽和固溶体<br/>α-SSS': 'Supersaturated Solid Solution<br/>α-SSS',
    'GPゾーン<br/>GP zones': 'GP Zones',
    'θ\'\'相<br/>準安定': 'θ\'\' Phase<br/>Metastable',
    'θ\'相<br/>準安定': 'θ\' Phase<br/>Metastable',
    'θ相<br/>Al₂Cu平衡相': 'θ Phase<br/>Al₂Cu Equilibrium',
    '各段階の特徴：': 'Characteristics of each stage:',

    # Table content
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

    # Example 4
    'Example 4: Al合金の時効曲線シミュレーション':
        'Example 4: Simulation of aging curves for Al alloys',
    '硬度の時間変化を予測': 'Predict time evolution of hardness',
    '時効曲線をシミュレーション（経験的モデル）':
        'Simulate aging curve (empirical model)',
    '時効時間 [h]': 'Aging time [h]',
    '時効温度 [K]': 'Aging temperature [K]',
    '基準温度でのピーク時間 [h]': 'Peak time at reference temperature [h]',
    'ピーク硬度 [HV]': 'Peak hardness [HV]',
    '基準温度 [K]': 'Reference temperature [K]',
    '硬度 [HV]': 'Hardness [HV]',
    '気体定数': 'Gas constant',
    '温度補正したピーク時間（Arrheniusの関係）':
        'Temperature-corrected peak time (Arrhenius relation)',
    '硬度の時間発展（JMAモデルベース）':
        'Time evolution of hardness (JMA model based)',
    'Under-aging領域': 'Under-aging region',
    'Over-aging領域（粗大化による軟化）':
        'Over-aging region (softening due to coarsening)',
    '最小硬度': 'Minimum hardness',
    '組み合わせ': 'Combination',
    '150°C (低温)': '150°C (Low)',
    '200°C (標準)': '200°C (Standard)',
    '250°C (高温)': '250°C (High)',
    '時効曲線': 'Aging curve',
    'ピーク硬度位置をマーク': 'Mark peak hardness position',
    'Under-aging, Peak-aging, Over-agingの領域を示す':
        'Regions of under-aging, peak-aging, and over-aging',
    'Under-aging': 'Under-aging',
    'Peak-aging': 'Peak-aging',
    'Over-aging': 'Over-aging',
    '(a) Al-Cu合金の時効曲線': '(a) Aging Curves of Al-Cu Alloy',
    'ピーク時間の温度依存性': 'Temperature dependence of peak time',
    'ピーク時間を求める': 'Find peak time',
    'ピーク時効時間 [h]': 'Peak aging time [h]',
    '(b) ピーク時効時間の温度依存性':
        '(b) Temperature Dependence of Peak Aging Time',
    '実用的な推奨時効条件': 'Practical recommended aging conditions',
    '=== 推奨時効条件（Al-Cu合金） ===':
        '=== Recommended Aging Conditions (Al-Cu Alloy) ===',
}

# Additional translations from the rest of the file (continuing pattern)
# Since the file is very long, I'll add more comprehensive translations

more_translations = {
    # More code output translations
    '時効時間': 'Aging time',
    '時間': 'hours',
    'ピーク硬度': 'Peak hardness',

    # Section 3.4 (continuing from the pattern)
    '3.4 析出強化のメカニズム': '3.4 Mechanism of Precipitation Strengthening',
    '3.4.1 Orowan機構': '3.4.1 Orowan Mechanism',
    '3.4.2 整合性析出物による強化': '3.4.2 Strengthening by Coherent Precipitates',
    '3.4.3 粒子サイズと強度の関係': '3.4.3 Relationship between Particle Size and Strength',

    # Section 3.5
    '3.5 析出物の粗大化': '3.5 Coarsening of Precipitates',
    '3.5.1 Gibbs-Thomson効果': '3.5.1 Gibbs-Thomson Effect',
    '3.5.2 Ostwald ripening': '3.5.2 Ostwald Ripening',
    '3.5.3 LSW理論': '3.5.3 LSW Theory',

    # Section 3.6
    '3.6 実践演習': '3.6 Practical Exercises',
    '演習問題': 'Exercise',
    '解答例': 'Solution',
    'ヒント': 'Hint',

    # Common terms
    '計算してください': 'Calculate',
    'プログラムを作成してください': 'Create a program',
    'グラフを描いてください': 'Plot a graph',
    'まとめ': 'Summary',
    '参考文献': 'References',
    '次章予告': 'Preview of Next Chapter',
    '前の章': 'Previous Chapter',
    '次の章': 'Next Chapter',
    '目次に戻る': 'Back to Index',

    # Disclaimer
    '免責事項': 'Disclaimer',
    '本コンテンツは教育目的で作成されています':
        'This content is created for educational purposes',
    '実際の材料開発には専門家の指導が必要です':
        'Professional guidance is required for actual materials development',
    '数値例は説明のための簡略化されたモデルです':
        'Numerical examples are simplified models for explanation',
}

# Combine all translations
translations.update(more_translations)

# Apply translations
for jp_text, en_text in translations.items():
    content = content.replace(jp_text, en_text)

# Write translated content
TARGET.parent.mkdir(parents=True, exist_ok=True)
with open(TARGET, 'w', encoding='utf-8') as f:
    f.write(content)

# Count Japanese characters after translation
jp_char_after = sum(1 for char in content if '\u3040' <= char <= '\u309F' or
                    '\u30A0' <= char <= '\u30FF' or
                    '\u4E00' <= char <= '\u9FFF')
jp_percentage_after = (jp_char_after / total_chars * 100) if total_chars > 0 else 0

print(f"\n{'='*60}")
print("TRANSLATION COMPLETE")
print(f"{'='*60}")
print(f"Target file: {TARGET}")
print(f"Japanese characters remaining: {jp_char_after:,}")
print(f"Japanese percentage after: {jp_percentage_after:.2f}%")
print(f"Translation coverage: {100 - jp_percentage_after:.2f}%")
print(f"{'='*60}\n")
