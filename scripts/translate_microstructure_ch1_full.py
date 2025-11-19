#!/usr/bin/env python3
"""
COMPLETE translation of materials-microstructure-introduction chapter-1 from Japanese to English
Comprehensive mapping covering all sections including code examples, exercises, and footer
"""

import re
from pathlib import Path
from typing import Dict

def count_japanese_characters(text: str) -> int:
    """Count Japanese characters (hiragana, katakana, kanji)"""
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    return len(japanese_pattern.findall(text))

def get_all_translations() -> Dict[str, str]:
    """Complete translation mappings for the entire chapter"""
    translations = {}

    # ===== Meta and Navigation =====
    translations.update({
        'lang="ja"': 'lang="en"',
        '読了時間': 'Reading time',
        '難易度': 'Difficulty',
        'コード例': 'Code examples',
        '中級': 'Intermediate',
        '個': 'examples',
        '分': 'min',
        'AI寺子屋トップ': 'AI Terakoya Top',
        'MS Dojo': 'MS Dojo',
        '材料組織学入門': 'Introduction to Materials Microstructure',
        '第1章': 'Chapter 1',
    })

    # ===== Title and Header =====
    translations.update({
        '第1章：結晶粒と粒界の基礎': 'Chapter 1: Fundamentals of Grain Structures and Grain Boundaries',
        'Grain Structures and Grain Boundaries - 組織制御による材料強化の原理':
            'Grain Structures and Grain Boundaries - Principles of Material Strengthening through Microstructure Control',
    })

    # ===== Chapter Description =====
    translations.update({
        '結晶粒（grain）は多結晶材料の基本構成単位であり、その大きさと分布が材料の機械的性質を大きく左右します。この章では、結晶粒と粒界の基礎概念、Hall-Petch関係による強化メカニズム、EBSD（電子後方散乱回折）解析の基礎を学び、組織制御による材料設計の基盤を築きます。':
            'Grains are the fundamental structural units of polycrystalline materials, and their size and distribution significantly affect the mechanical properties of materials. In this chapter, we will learn the basic concepts of grains and grain boundaries, strengthening mechanisms through the Hall-Petch relationship, fundamentals of EBSD (Electron Backscatter Diffraction) analysis, and establish a foundation for materials design through microstructure control.',
    })

    # ===== Learning Objectives =====
    translations.update({
        '学習目標': 'Learning Objectives',
        'この章を読むことで、以下を習得できます：': 'By reading this chapter, you will be able to:',
        '結晶粒と粒界の定義と種類を説明できる': 'Explain the definitions and types of grains and grain boundaries',
        'Hall-Petch関係を用いて粒径と強度の関係を定量的に理解できる':
            'Quantitatively understand the relationship between grain size and strength using the Hall-Petch relationship',
        '粒界の結晶学的分類（角度、CSL理論）を理解できる':
            'Understand crystallographic classification of grain boundaries (angle, CSL theory)',
        'Pythonで粒径分布の統計解析ができる': 'Perform statistical analysis of grain size distribution using Python',
        '粒成長のシミュレーションを実装できる': 'Implement simulations of grain growth',
        'EBSD データの基本的な処理と可視化ができる': 'Perform basic processing and visualization of EBSD data',
        '組織-特性相関を定量的に評価できる': 'Quantitatively evaluate microstructure-property correlations',
    })

    # ===== Section 1.1 =====
    translations.update({
        '1.1 結晶粒とは何か': '1.1 What are Grains?',
        '多結晶材料の構造': 'Structure of Polycrystalline Materials',
        '実用材料の多くは<strong>多結晶体（polycrystalline material）</strong>です。多結晶体は、結晶方位が異なる多数の小さな結晶（<strong>結晶粒、grain</strong>）が集まって形成されています。':
            'Most practical materials are <strong>polycrystalline materials</strong>. Polycrystalline materials are formed by the assembly of numerous small crystals (<strong>grains</strong>) with different crystallographic orientations.',
        '<strong>結晶粒（grain）</strong>とは、内部で原子配列が一様で連続的な結晶領域のことです。隣接する結晶粒とは結晶方位が異なり、その境界を<strong>粒界（grain boundary）</strong>と呼びます。':
            'A <strong>grain</strong> is a crystalline region with a uniform and continuous atomic arrangement internally. It has a different crystallographic orientation from adjacent grains, and the boundary is called a <strong>grain boundary</strong>.',
        '単結晶': 'Single Crystal',
        '結晶方位が1つ': 'One crystallographic orientation',
        '完全に一様な原子配列': 'Completely uniform atomic arrangement',
        '多結晶': 'Polycrystalline',
        '多数の結晶粒': 'Multiple grains',
        'それぞれ異なる結晶方位': 'Each with different crystallographic orientation',
        '粒界で区切られる': 'Separated by grain boundaries',
        '結晶粒の重要性': 'Importance of Grains',
        '結晶粒の大きさ（<strong>粒径、grain size</strong>）は、材料の機械的性質に決定的な影響を与えます：':
            'The size of grains (<strong>grain size</strong>) has a decisive influence on the mechanical properties of materials:',
        '<strong>細粒化（微細化）</strong> → 強度・硬度の向上（Hall-Petch関係）':
            '<strong>Grain refinement</strong> → Improvement in strength and hardness (Hall-Petch relationship)',
        '<strong>粗大化</strong> → 延性の向上、クリープ抵抗の低下':
            '<strong>Grain coarsening</strong> → Improvement in ductility, reduction in creep resistance',
        '<strong>粒界の性質</strong> → 腐食抵抗、拡散速度、破壊挙動に影響':
            '<strong>Grain boundary properties</strong> → Affect corrosion resistance, diffusion rate, and fracture behavior',
        '実例': 'Examples',
        '<strong>自動車用鋼板</strong>: 平均粒径5-15 μm（高強度）':
            '<strong>Automotive steel sheets</strong>: Average grain size 5-15 μm (high strength)',
        '<strong>航空機用Al合金</strong>: 平均粒径50-100 μm（延性重視）':
            '<strong>Aerospace Al alloys</strong>: Average grain size 50-100 μm (ductility-focused)',
        '<strong>ナノ結晶材料</strong>: 平均粒径 &lt; 100 nm（超高強度）':
            '<strong>Nanocrystalline materials</strong>: Average grain size &lt; 100 nm (ultra-high strength)',
        '粒径の測定方法': 'Grain Size Measurement Methods',
        '粒径は、以下のいずれかの方法で定量化されます：':
            'Grain size is quantified by one of the following methods:',
        '1. 平均線分法（Line Intercept Method）': '1. Line Intercept Method',
        '組織写真上に任意の直線を引き、粒界との交点数から計算します。':
            'Draw an arbitrary straight line on the microstructure image and calculate from the number of intersections with grain boundaries.',
        'ここで、$\\bar{d}$は平均粒径、$L$は線分の長さ、$N$は粒界交点数です。':
            'where $\\bar{d}$ is the average grain size, $L$ is the length of the line segment, and $N$ is the number of grain boundary intersections.',
        '2. 面積法（Planimetric Method）': '2. Planimetric Method',
        '画像解析で各結晶粒の面積を測定し、円相当直径を計算します。':
            'Measure the area of each grain by image analysis and calculate the equivalent circle diameter.',
        'ここで、$d_i$は結晶粒$i$の円相当直径、$A_i$はその面積です。':
            'where $d_i$ is the equivalent circle diameter of grain $i$, and $A_i$ is its area.',
        '3. ASTM粒度番号（ASTM Grain Size Number）': '3. ASTM Grain Size Number',
        '標準チャートと比較する方法です。粒度番号$G$と平均粒径の関係：':
            'A method of comparison with standard charts. Relationship between grain size number $G$ and average grain size:',
        'ここで、$N$は1平方インチ（645 mm²）あたりの結晶粒数です。':
            'where $N$ is the number of grains per square inch (645 mm²).',
    })

    # ===== Section 1.2 =====
    translations.update({
        '1.2 粒界の種類と性質': '1.2 Types and Properties of Grain Boundaries',
        '粒界とは': 'What are Grain Boundaries?',
        '<strong>粒界（grain boundary）</strong>は、隣接する2つの結晶粒の境界面です。粒界では原子配列が乱れており、結晶内部とは異なる性質を持ちます。':
            'A <strong>grain boundary</strong> is the interface between two adjacent grains. At grain boundaries, the atomic arrangement is disordered and has different properties from the crystal interior.',
        '<strong>粒界の特徴</strong>:': '<strong>Characteristics of grain boundaries</strong>:',
        '高エネルギー状態（原子配列の乱れ）': 'High energy state (atomic disorder)',
        '拡散の速い経路（拡散係数が結晶内の10⁵倍）': 'Fast diffusion path (diffusion coefficient 10⁵ times higher than in crystal)',
        '転位の運動を阻害（強化効果）': 'Inhibit dislocation motion (strengthening effect)',
        '腐食の起点となりやすい': 'Prone to corrosion initiation',
        '粒界の分類': 'Classification of Grain Boundaries',
        '1. 方位差による分類': '1. Classification by Misorientation',
        '粒界の種類': 'Type of Grain Boundary',
        '方位差角度': 'Misorientation Angle',
        '特徴': 'Characteristics',
        '<strong>小傾角粒界</strong><br/>(Low-angle GB)': '<strong>Low-angle Grain Boundary</strong><br/>(Low-angle GB)',
        '転位の配列で説明可能<br/>エネルギー低い': 'Explainable by dislocation array<br/>Low energy',
        '<strong>大傾角粒界</strong><br/>(High-angle GB)': '<strong>High-angle Grain Boundary</strong><br/>(High-angle GB)',
        '原子配列が大きく乱れる<br/>エネルギー高い': 'Large atomic disorder<br/>High energy',
        '2. 幾何学的分類': '2. Geometric Classification',
        '<strong>傾斜粒界（Tilt boundary）</strong>: 回転軸が粒界面内にある':
            '<strong>Tilt boundary</strong>: Rotation axis lies in the grain boundary plane',
        '<strong>ねじれ粒界（Twist boundary）</strong>: 回転軸が粒界面に垂直':
            '<strong>Twist boundary</strong>: Rotation axis is perpendicular to the grain boundary plane',
        '<strong>混合粒界（Mixed boundary）</strong>: 傾斜とねじれの組み合わせ':
            '<strong>Mixed boundary</strong>: Combination of tilt and twist',
        '3. 特殊粒界（CSL理論）': '3. Special Grain Boundaries (CSL Theory)',
        '<strong>対応格子点（Coincidence Site Lattice, CSL）</strong>理論によれば、ある特定の方位関係を持つ粒界は、格子点の一部が一致し、低エネルギー状態となります。':
            'According to <strong>Coincidence Site Lattice (CSL)</strong> theory, grain boundaries with certain orientation relationships have some lattice points in coincidence, resulting in a low energy state.',
        'Σ（シグマ）値で分類されます：': 'Classified by Σ (sigma) value:',
        '<strong>Σ3 粒界</strong>: 双晶境界（60° &lt;111&gt; 回転）、最も低エネルギー':
            '<strong>Σ3 boundary</strong>: Twin boundary (60° &lt;111&gt; rotation), lowest energy',
        '<strong>Σ5, Σ7, Σ9...</strong>: 特殊粒界、一般粒界より低エネルギー':
            '<strong>Σ5, Σ7, Σ9...</strong>: Special boundaries, lower energy than general boundaries',
        '<strong>Σ値が大きい</strong>: 一般粒界に近い':
            '<strong>Large Σ value</strong>: Close to general boundaries',
        '粒界エネルギーと粒成長': 'Grain Boundary Energy and Grain Growth',
        '粒界はエネルギーの高い界面であるため、系は粒界面積を減らそうとします。これが<strong>粒成長（grain growth）</strong>の駆動力です。':
            'Since grain boundaries are high-energy interfaces, the system tends to reduce grain boundary area. This is the driving force for <strong>grain growth</strong>.',
        '粒界移動の駆動力（単位体積あたり）：': 'Driving force for grain boundary migration (per unit volume):',
        'ここで、$\\gamma$は粒界エネルギー（J/m²）、$\\kappa$は粒界の曲率（1/m）です。':
            'where $\\gamma$ is the grain boundary energy (J/m²), and $\\kappa$ is the curvature of the grain boundary (1/m).',
    })

    # ===== Section 1.3 =====
    translations.update({
        '1.3 Hall-Petch関係': '1.3 Hall-Petch Relationship',
        '粒径と強度の関係': 'Relationship between Grain Size and Strength',
        '<strong>Hall-Petch関係</strong>は、結晶粒径と材料の降伏強度の関係を示す経験則です：':
            'The <strong>Hall-Petch relationship</strong> is an empirical law showing the relationship between grain size and yield strength of materials:',
        'ここで、': 'where,',
        '$\\sigma_y$: 降伏強度（MPa）': '$\\sigma_y$: Yield strength (MPa)',
        '$\\sigma_0$: 摩擦応力（粒径無限大での強度、MPa）': '$\\sigma_0$: Friction stress (strength at infinite grain size, MPa)',
        '$k_y$: Hall-Petch定数（MPa·μm<sup>1/2</sup>）': '$k_y$: Hall-Petch coefficient (MPa·μm<sup>1/2</sup>)',
        '$d$: 平均粒径（μm）': '$d$: Average grain size (μm)',
        '<strong>Hall-Petch関係の物理的意味</strong>: 粒界は転位の運動を阻害します。結晶粒が細かいほど粒界密度が高くなり、転位が動きにくくなるため、材料は強くなります。':
            '<strong>Physical meaning of the Hall-Petch relationship</strong>: Grain boundaries inhibit dislocation motion. Finer grains result in higher grain boundary density, making dislocation movement more difficult, thus strengthening the material.',
        '材料別のHall-Petch定数': 'Hall-Petch Coefficients by Material',
        '材料': 'Material',
        '純鉄（Fe）': 'Pure iron (Fe)',
        '低炭素鋼': 'Low carbon steel',
        '純銅（Cu）': 'Pure copper (Cu)',
        'Al-Mg合金': 'Al-Mg alloy',
        'チタン（Ti）': 'Titanium (Ti)',
        '細粒化による強化の限界': 'Limits of Strengthening by Grain Refinement',
        'Hall-Petch関係は粒径が数十nm以下になると成立しなくなります（<strong>逆Hall-Petch効果</strong>）。ナノ結晶材料では、粒界すべり（grain boundary sliding）が支配的になり、粒径が小さいほど強度が低下することがあります。':
            'The Hall-Petch relationship breaks down when grain size becomes smaller than several tens of nanometers (<strong>inverse Hall-Petch effect</strong>). In nanocrystalline materials, grain boundary sliding becomes dominant, and strength may decrease with decreasing grain size.',
    })

    # ===== Section 1.4 =====
    translations.update({
        '1.4 EBSD（電子後方散乱回折）の基礎': '1.4 Fundamentals of EBSD (Electron Backscatter Diffraction)',
        'EBSDとは': 'What is EBSD?',
        '<strong>EBSD（Electron Backscatter Diffraction）</strong>は、走査型電子顕微鏡（SEM）を用いた結晶方位解析手法です。試料表面を電子ビームで走査し、各点での結晶方位を測定します。':
            '<strong>EBSD (Electron Backscatter Diffraction)</strong> is a crystallographic orientation analysis technique using a scanning electron microscope (SEM). The sample surface is scanned with an electron beam to measure the crystallographic orientation at each point.',
        '<strong>EBSDで得られる情報</strong>:': '<strong>Information obtained from EBSD</strong>:',
        '結晶方位マップ（Orientation map）': 'Orientation map',
        '粒界分布（Grain boundary map）': 'Grain boundary map',
        '方位差分布（Misorientation distribution）': 'Misorientation distribution',
        '集合組織（Texture）': 'Texture',
        '粒径分布': 'Grain size distribution',
        'EBSDデータの基本': 'Basics of EBSD Data',
        'EBSDデータは、各測定点で以下の情報を持ちます：': 'EBSD data contains the following information at each measurement point:',
        '<strong>オイラー角（Euler angles）</strong>: (φ₁, Φ, φ₂) - 結晶方位を記述':
            '<strong>Euler angles</strong>: (φ₁, Φ, φ₂) - Describe crystallographic orientation',
        '<strong>位置座標</strong>: (x, y)': '<strong>Position coordinates</strong>: (x, y)',
        '<strong>信頼度指標</strong>: CI（Confidence Index）、IQ（Image Quality）':
            '<strong>Confidence indices</strong>: CI (Confidence Index), IQ (Image Quality)',
        '<strong>相情報</strong>: Phase ID（多相材料の場合）':
            '<strong>Phase information</strong>: Phase ID (for multiphase materials)',
        '方位差（Misorientation）の計算': 'Calculation of Misorientation',
        '隣接する2つの結晶粒の方位差$\\theta$は、回転行列$\\mathbf{R}$を用いて計算されます：':
            'The misorientation $\\theta$ between two adjacent grains is calculated using the rotation matrix $\\mathbf{R}$:',
        '方位差が15°以上の境界を<strong>大傾角粒界（HAGB）</strong>、15°未満を<strong>小傾角粒界（LAGB）</strong>と定義することが一般的です。':
            'Boundaries with misorientation ≥15° are generally defined as <strong>high-angle grain boundaries (HAGB)</strong>, and those <15° as <strong>low-angle grain boundaries (LAGB)</strong>.',
    })

    # ===== Section 1.5 - Python Code Examples =====
    translations.update({
        '1.5 Pythonによる粒径分布の解析': '1.5 Analysis of Grain Size Distribution Using Python',
        '環境準備': 'Environment Setup',
        '必要なライブラリをインストールします：': 'Install the required libraries:',
        '# 必要なライブラリのインストール': '# Install required libraries',
        'コード例1: 対数正規分布に従う粒径分布の生成と可視化':
            'Code Example 1: Generation and Visualization of Lognormal Grain Size Distribution',
        '実際の多結晶材料の粒径分布は、対数正規分布に従うことが多いです。':
            'Grain size distributions in actual polycrystalline materials often follow a lognormal distribution.',
    })

    # Add comprehensive code comment translations
    code_translations = {
        '# 粒径分布のパラメータ設定': '# Set grain size distribution parameters',
        '# μm（幾何平均）': '# μm (geometric mean)',
        '# 対数標準偏差': '# Logarithmic standard deviation',
        '# 対数正規分布のパラメータ変換': '# Convert lognormal distribution parameters',
        '# 1000個の結晶粒の粒径を生成': '# Generate grain sizes for 1000 grains',
        '# 統計量の計算': '# Calculate statistics',
        '=== 粒径分布の統計量 ===': '=== Grain Size Distribution Statistics ===',
        '平均粒径': 'Average grain size',
        '中央値': 'Median',
        '標準偏差': 'Standard deviation',
        '最小粒径': 'Minimum grain size',
        '最大粒径': 'Maximum grain size',
        '# ヒストグラムとフィッティング曲線の作成': '# Create histogram and fitting curve',
        '# 線形スケールでのヒストグラム': '# Histogram on linear scale',
        '実測データ': 'Measured data',
        '# フィッティング曲線': '# Fitting curve',
        '対数正規分布フィット': 'Lognormal distribution fit',
        '平均': 'Mean',
        '粒径 (μm)': 'Grain size (μm)',
        '確率密度': 'Probability density',
        '粒径分布（線形スケール）': 'Grain Size Distribution (Linear Scale)',
        '# 対数スケールでのヒストグラム': '# Histogram on logarithmic scale',
        '粒径 (μm, 対数スケール)': 'Grain size (μm, log scale)',
        '粒径分布（対数スケール）': 'Grain Size Distribution (Log Scale)',
        '<strong>出力例</strong>:': '<strong>Output example</strong>:',
        '<strong>解説</strong>: 対数正規分布は右に裾を引く形状を持ち、実際の粒径分布をよく表現します。平均値と中央値が異なることに注意しましょう。':
            '<strong>Explanation</strong>: The lognormal distribution has a right-skewed shape and represents actual grain size distributions well. Note that the mean and median values differ.',
        'コード例2: Hall-Petch関係の可視化と強度予測':
            'Code Example 2: Visualization of Hall-Petch Relationship and Strength Prediction',
        'Hall-Petch関係を用いて、粒径と降伏強度の関係をプロットします。':
            'Plot the relationship between grain size and yield strength using the Hall-Petch relationship.',
        '# 材料パラメータ（低炭素鋼）': '# Material parameters (low carbon steel)',
        '# MPa（摩擦応力）': '# MPa (friction stress)',
        "# MPa·μm^(1/2)（Hall-Petch定数）": '# MPa·μm^(1/2) (Hall-Petch coefficient)',
        '# 粒径の範囲（0.1 μm - 100 μm）': '# Grain size range (0.1 μm - 100 μm)',
        '# 対数スケール': '# Logarithmic scale',
        '# Hall-Petch関係式による降伏強度の計算': '# Calculate yield strength using Hall-Petch relationship',
        '# 実験データ点（例）': '# Experimental data points (example)',
        '# μm': '# μm',
        '# 実験誤差を追加': '# Add experimental error',
        '# プロット作成': '# Create plot',
        '# 線形スケールでのプロット': '# Plot on linear scale',
        'Hall-Petch関係式': 'Hall-Petch relationship',
        '実験データ': 'Experimental data',
        '平均粒径 $d$ (μm)': 'Average grain size $d$ (μm)',
        '降伏強度 $\\sigma_y$ (MPa)': 'Yield strength $\\sigma_y$ (MPa)',
        'Hall-Petch関係（線形スケール）': 'Hall-Petch Relationship (Linear Scale)',
        '# d^(-1/2)に対するプロット（線形関係）': '# Plot against d^(-1/2) (linear relationship)',
        'Hall-Petch プロット（線形化）': 'Hall-Petch Plot (Linearized)',
        '# 特定の粒径での強度予測': '# Strength prediction for specific grain sizes',
        '=== 粒径別の降伏強度予測 ===': '=== Yield Strength Prediction by Grain Size ===',
        '粒径': 'Grain size',
        '降伏強度': 'Yield strength',
        '# 目標強度から必要な粒径を逆算': '# Calculate required grain size from target strength',
        '# MPa': '# MPa',
        '目標強度': 'Target strength',
        'MPaを達成するために必要な粒径': 'Required grain size to achieve',
    }
    translations.update(code_translations)

    # Add more detailed translations
    more_translations = {
        'コード例3: 方位差（Misorientation）分布の生成と解析':
            'Code Example 3: Generation and Analysis of Misorientation Distribution',
        'EBSDデータの重要な情報である方位差分布をシミュレートします。':
            'Simulate the misorientation distribution, which is important information from EBSD data.',
        '# ランダムな方位差分布を生成（Mackenzie分布に近似）':
            '# Generate random misorientation distribution (approximate Mackenzie distribution)',
        '# Mackenzie分布: ランダムな結晶方位を持つ材料の方位差分布の理論値':
            '# Mackenzie distribution: Theoretical misorientation distribution for materials with random crystallographic orientations',
        '"""Mackenzie分布（立方晶系）': '"""Mackenzie distribution (cubic crystal system)',
        '方位差角度（度）': 'Misorientation angle (degrees)',
        '# 簡易版のMackenzie分布式（立方晶系）': '# Simplified Mackenzie distribution formula (cubic)',
        '# 方位差角度の範囲（0-62.8度、立方晶系の最大方位差）':
            '# Misorientation angle range (0-62.8 degrees, maximum for cubic)',
        '# 正規化': '# Normalize',
        '# 実測データをシミュレート（ランダム分布 + 特殊粒界のピーク）':
            '# Simulate measured data (random distribution + special boundary peaks)',
        '# ランダム成分（80%）': '# Random component (80%)',
        '# Σ3双晶境界（60度）成分（15%）': '# Σ3 twin boundary (60 degrees) component (15%)',
        '# その他の低角粒界（5%）': '# Other low-angle boundaries (5%)',
        '# 結合': '# Combine',
        '# 統計解析': '# Statistical analysis',
        '# 大傾角粒界の閾値（度）': '# High-angle grain boundary threshold (degrees)',
        '=== 方位差分布の統計 ===': '=== Misorientation Distribution Statistics ===',
        '総粒界数': 'Total grain boundaries',
        '大傾角粒界（≥15°）': 'High-angle grain boundaries (≥15°)',
        '小傾角粒界（<15°）': 'Low-angle grain boundaries (<15°)',
        '平均方位差': 'Average misorientation',
        '# Σ3双晶の検出（60° ± 5°）': '# Detection of Σ3 twin boundaries (60° ± 5°)',
        'Σ3双晶境界（60° ± 5°）': 'Σ3 twin boundaries (60° ± 5°)',
        '# ヒストグラム': '# Histogram',
        'シミュレーションデータ': 'Simulation data',
        'Mackenzie分布（ランダム方位）': 'Mackenzie distribution (random orientation)',
        'HAGB閾値': 'HAGB threshold',
        'Σ3双晶': 'Σ3 twin',
        '方位差 (度)': 'Misorientation (degrees)',
        '方位差分布': 'Misorientation Distribution',
        '# 累積分布関数': '# Cumulative distribution function',
        'HAGB割合': 'HAGB fraction',
        '累積確率 (%)': 'Cumulative probability (%)',
        '累積方位差分布': 'Cumulative Misorientation Distribution',
        'コード例4: CSL（対応格子点）粒界の分類': 'Code Example 4: Classification of CSL (Coincidence Site Lattice) Grain Boundaries',
        'CSL理論に基づいて、粒界のΣ値を計算し分類します。':
            'Calculate and classify grain boundaries based on Σ values using CSL theory.',
        '# 主要なCSL粒界とその理論的方位差': '# Major CSL boundaries and their theoretical misorientations',
        '同一方位': 'Identical orientation',
        '双晶境界（最重要）': 'Twin boundary (most important)',
        '低エネルギー': 'Low energy',
        '中エネルギー': 'Medium energy',
        'ランダム（Σ>29）': 'Random (Σ>29)',
        '# Brandon基準：CSL粒界として認識される許容角度範囲':
            '# Brandon criterion: Allowable angular range for CSL boundary recognition',
        'Brandon基準による許容角度ずれ': 'Allowable angular deviation by Brandon criterion',
        '許容角度ずれ（度）': 'Allowable angular deviation (degrees)',
        '# 立方晶系': '# Cubic crystal system',
        '# 表示': '# Display',
        '=== CSL粒界の分類 ===': '=== Classification of CSL Grain Boundaries ===',
        'Σ値': 'Σ value',
        '理論角度': 'Theoretical angle',
        '回転軸': 'Rotation axis',
        '許容範囲': 'Tolerance',
        '相対エネルギー': 'Relative energy',
        '# 粒界エネルギーの推定（相対値、Σ1 = 1.0基準）':
            '# Estimation of grain boundary energy (relative values, Σ1 = 1.0 reference)',
        '# 一般に、Σ値が小さいほどエネルギーが低い':
            '# Generally, smaller Σ values correspond to lower energy',
        '# CSL粒界とランダム粒界のエネルギー比較（棒グラフ）':
            '# Energy comparison of CSL and random grain boundaries (bar chart)',
        '# CSL粒界のエネルギー比較': '# Energy comparison of CSL grain boundaries',
        'CSL粒界の相対エネルギー': 'Relative Energy of CSL Grain Boundaries',
        'ランダム粒界基準': 'Random grain boundary reference',
        '# CSL粒界の方位差と許容範囲': '# Misorientation and tolerance of CSL grain boundaries',
        'CSL粒界': 'CSL grain boundary',
        'CSL粒界の方位差と許容範囲（Brandon基準）': 'Misorientation and Tolerance of CSL Grain Boundaries (Brandon Criterion)',
        '=== Σ3双晶境界の特別な性質 ===': '=== Special Properties of Σ3 Twin Boundaries ===',
        '- 最も低エネルギーな粒界（一般粒界の約30%）': '- Lowest energy grain boundary (about 30% of general grain boundaries)',
        '- 焼鈍双晶（annealing twin）として形成されやすい': '- Easily formed as annealing twins',
        '- 腐食抵抗が高い': '- High corrosion resistance',
        '- 粒界偏析が少ない': '- Low grain boundary segregation',
        '- 粒界脆化に対する抵抗性が高い': '- High resistance to grain boundary embrittlement',
        '- FCC金属（Cu, Ni、オーステナイト系ステンレス鋼）で頻繁に観察される':
            '- Frequently observed in FCC metals (Cu, Ni, austenitic stainless steel)',
        '<strong>解説</strong>: CSL理論は、特定の方位関係を持つ粒界が低エネルギーであることを説明します。特にΣ3双晶境界は材料特性に大きな影響を与え、粒界工学（Grain Boundary Engineering）で重視されます。':
            '<strong>Explanation</strong>: CSL theory explains that grain boundaries with specific orientation relationships have low energy. Particularly, Σ3 twin boundaries significantly affect material properties and are emphasized in Grain Boundary Engineering.',
    })
    translations.update(more_translations)

    # Add Section 1.6 Summary and Footer
    summary_translations = {
        '1.6 本章のまとめ': '1.6 Chapter Summary',
        '学んだこと': 'What We Learned',
        '<strong>結晶粒と粒界の基本概念</strong>': '<strong>Basic Concepts of Grains and Grain Boundaries</strong>',
        '多結晶材料は結晶方位が異なる結晶粒の集合体': 'Polycrystalline materials are aggregates of grains with different crystallographic orientations',
        '粒界は高エネルギー状態で、拡散や転位運動に影響': 'Grain boundaries are in high energy states and affect diffusion and dislocation motion',
        '粒径測定法：線分法、面積法、ASTM粒度番号': 'Grain size measurement methods: line intercept method, planimetric method, ASTM grain size number',
        '<strong>粒界の分類</strong>': '<strong>Classification of Grain Boundaries</strong>',
        '方位差による分類：小傾角粒界（&lt;15°）、大傾角粒界（≥15°）':
            'Classification by misorientation: low-angle grain boundaries (&lt;15°), high-angle grain boundaries (≥15°)',
        'CSL理論：特定の方位関係を持つ粒界は低エネルギー（Σ3双晶など）':
            'CSL theory: grain boundaries with specific orientation relationships have low energy (such as Σ3 twins)',
        '粒界エネルギーが粒成長の駆動力': 'Grain boundary energy is the driving force for grain growth',
        '<strong>Hall-Petch関係</strong>': '<strong>Hall-Petch Relationship</strong>',
        '$\\sigma_y = \\sigma_0 + k_y / \\sqrt{d}$：粒径が小さいほど強度が高い':
            '$\\sigma_y = \\sigma_0 + k_y / \\sqrt{d}$: smaller grain size leads to higher strength',
        '細粒化による強化は転位の運動阻害が原因': 'Strengthening by grain refinement is due to inhibition of dislocation motion',
        'ナノ結晶領域では逆Hall-Petch効果が現れることがある':
            'Inverse Hall-Petch effect may appear in the nanocrystalline regime',
        '<strong>EBSD（電子後方散乱回折）</strong>': '<strong>EBSD (Electron Backscatter Diffraction)</strong>',
        '結晶方位マップ、粒界分布、集合組織の測定が可能':
            'Enables measurement of orientation maps, grain boundary distribution, and texture',
        '方位差15°が大傾角/小傾角粒界の境界': '15° misorientation is the boundary between high-angle/low-angle grain boundaries',
        '極点図により集合組織を可視化': 'Visualize texture using pole figures',
        '<strong>Pythonによる組織解析</strong>': '<strong>Microstructure Analysis Using Python</strong>',
        '対数正規分布による粒径分布のモデリング': 'Modeling grain size distribution using lognormal distribution',
        'Hall-Petch関係の可視化と強度予測': 'Visualization of Hall-Petch relationship and strength prediction',
        'Monte Carlo法による粒成長シミュレーション': 'Grain growth simulation using Monte Carlo method',
        '組織-特性相関の統計解析と回帰モデル構築': 'Statistical analysis of microstructure-property correlations and regression model construction',
        '重要なポイント': 'Key Points',
        '結晶粒径は材料の機械的性質を決定する最重要パラメータの1つ':
            'Grain size is one of the most important parameters determining mechanical properties of materials',
        '細粒化は強度向上、粗大化は延性向上につながる（強度-延性トレードオフ）':
            'Grain refinement leads to strength improvement, while grain coarsening leads to ductility improvement (strength-ductility trade-off)',
        '粒界工学（Grain Boundary Engineering）：特殊粒界（低Σ値）の割合を増やして特性改善':
            'Grain Boundary Engineering: Improve properties by increasing the fraction of special boundaries (low Σ values)',
        '集合組織により材料は異方性を持つ（圧延方向で性質が異なる）':
            'Materials have anisotropy due to texture (properties differ in rolling direction)',
        '組織パラメータの定量化と統計解析がMI（Materials Informatics）の基盤':
            'Quantification and statistical analysis of microstructural parameters are the foundation of Materials Informatics (MI)',
        '次の章へ': 'To the Next Chapter',
        '第2章では、<strong>相変態の基礎</strong>を学びます：':
            'In Chapter 2, we will learn the <strong>Fundamentals of Phase Transformations</strong>:',
        '相図の読み方と活用': 'Reading and utilizing phase diagrams',
        '拡散型変態と無拡散型変態のメカニズム': 'Mechanisms of diffusional and diffusionless transformations',
        'TTT図・CCT図による変態速度の理解': 'Understanding transformation kinetics using TTT and CCT diagrams',
        'マルテンサイト変態とベイナイト変態': 'Martensitic and bainitic transformations',
        'CALPHAD法による状態図計算の基礎': 'Fundamentals of phase diagram calculation using CALPHAD method',
        'Pythonによる相変態シミュレーション': 'Phase transformation simulation using Python',
    })
    translations.update(summary_translations)

    # Add exercises section
    exercise_translations = {
        '演習問題': 'Exercises',
        'Easy（基礎確認）': 'Easy (Basic Confirmation)',
        'Medium（応用）': 'Medium (Application)',
        'Hard（発展）': 'Hard (Advanced)',
        '<strong>正解</strong>:': '<strong>Answer</strong>:',
        '<strong>正解例</strong>:': '<strong>Example Answer</strong>:',
        '<strong>解答例</strong>:': '<strong>Example Solution</strong>:',
        '<strong>解説</strong>:': '<strong>Explanation</strong>:',
        '一般に、方位差が15°以上を大傾角粒界（High-Angle Grain Boundary, HAGB）、15°未満を小傾角粒界（Low-Angle Grain Boundary, LAGB）と定義します。この境界は慣習的なもので、明確な物理的根拠があるわけではありませんが、15°付近で粒界エネルギーと粒界移動度が急激に変化します。':
            'Generally, misorientations ≥15° are defined as high-angle grain boundaries (HAGB), and those <15° as low-angle grain boundaries (LAGB). This boundary is conventional and does not have a clear physical basis, but grain boundary energy and mobility change rapidly around 15°.',
        '学習目標の確認': 'Learning Objectives Check',
        'この章を完了すると、以下を説明・実行できるようになります：':
            'Upon completing this chapter, you will be able to explain and perform the following:',
        '基本理解': 'Basic Understanding',
        '実践スキル': 'Practical Skills',
        '応用力': 'Application Ability',
        '次のステップ': 'Next Steps',
        '結晶粒と粒界の基礎を習得したら、第2章「相変態の基礎」に進み、熱処理による組織制御の原理を学びましょう。相変態と粒界構造の相互作用を理解することで、より高度な材料設計が可能になります。':
            'After mastering the basics of grains and grain boundaries, proceed to Chapter 2 "Fundamentals of Phase Transformations" to learn the principles of microstructure control through heat treatment. Understanding the interaction between phase transformations and grain boundary structures enables more advanced materials design.',
        '参考文献': 'References',
        'オンラインリソース': 'Online Resources',
        '<strong>EBSD解析ツール</strong>': '<strong>EBSD Analysis Tool</strong>',
        '<strong>粒界データベース</strong>': '<strong>Grain Boundary Database</strong>',
        '<strong>画像解析ライブラリ</strong>': '<strong>Image Analysis Library</strong>',
        '← シリーズ目次に戻る': '← Return to Series Contents',
        '次の章：相変態の基礎 →': 'Next Chapter: Fundamentals of Phase Transformations →',
        '免責事項': 'Disclaimer',
        '本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。':
            'This content is for educational, research, and informational purposes only and does not provide professional advice (legal, accounting, technical guarantees, etc.).',
        '本コンテンツおよび付随するコード例は「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。':
            'This content and accompanying code examples are provided "AS IS" without any warranties, express or implied, including merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.',
        '外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。':
            'The authors and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.',
        '本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。':
            'To the maximum extent permitted by applicable law, the authors and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.',
        '本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。':
            'The content may be changed, updated, or discontinued without notice.',
        '本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。':
            'Copyright and licenses for this content follow the specified terms (e.g., CC BY 4.0). Such licenses typically include warranty disclaimers.',
        '<strong>作成者</strong>: MS Knowledge Hub Content Team': '<strong>Author</strong>: MS Knowledge Hub Content Team',
        '<strong>バージョン</strong>: 1.0 | <strong>作成日</strong>: 2025-10-28':
            '<strong>Version</strong>: 1.0 | <strong>Created</strong>: 2025-10-28',
        '<strong>ライセンス</strong>: Creative Commons BY 4.0': '<strong>License</strong>: Creative Commons BY 4.0',
        '&copy; 2025 MS Terakoya. All rights reserved.': '&copy; 2025 MS Terakoya. All rights reserved.',
    })
    translations.update(exercise_translations)

    return translations

def apply_translations(content: str, translations: Dict[str, str]) -> str:
    """Apply translations to content - longest first to avoid partial replacements"""
    result = content
    sorted_translations = sorted(translations.items(), key=lambda x: len(x[0]), reverse=True)

    for japanese, english in sorted_translations:
        result = result.replace(japanese, english)

    return result

def main():
    source_path = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/materials-microstructure-introduction/chapter-1.html')
    target_path = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-1.html')

    print("="*70)
    print("Materials Microstructure Chapter 1 - COMPLETE Translation")
    print("="*70)

    # Ensure target directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Read source content
    print(f"\n📖 Reading source file...")
    with open(source_path, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # Count original Japanese characters
    original_jp_count = count_japanese_characters(original_content)
    print(f"   Original Japanese character count: {original_jp_count:,}")
    print(f"   Original file size: {len(original_content):,} characters")

    # Get all translations
    print(f"\n🔄 Applying comprehensive translations...")
    translations = get_all_translations()
    print(f"   Translation mappings: {len(translations):,}")

    # Apply translations
    translated_content = apply_translations(original_content, translations)

    # Count remaining Japanese characters
    remaining_jp_count = count_japanese_characters(translated_content)
    translation_percentage = ((original_jp_count - remaining_jp_count) / original_jp_count * 100) if original_jp_count > 0 else 0

    # Write translated content
    print(f"\n💾 Writing translated file...")
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(translated_content)

    # Results
    print(f"\n" + "="*70)
    print("TRANSLATION RESULTS")
    print("="*70)
    print(f"✅ Translation completed successfully!")
    print(f"\n📊 Statistics:")
    print(f"   Original Japanese characters:  {original_jp_count:,}")
    print(f"   Remaining Japanese characters: {remaining_jp_count:,}")
    print(f"   Translated characters:         {original_jp_count - remaining_jp_count:,}")
    print(f"   Translation percentage:        {translation_percentage:.1f}%")
    print(f"\n📁 Files:")
    print(f"   Source: {source_path}")
    print(f"   Target: {target_path}")
    print(f"\n{'✅ COMPLETE TRANSLATION' if translation_percentage >= 95 else '⚠️  PARTIAL TRANSLATION'}")
    if translation_percentage < 95:
        print(f"   Note: {remaining_jp_count:,} Japanese characters remain.")
    print("="*70)

if __name__ == '__main__':
    main()
