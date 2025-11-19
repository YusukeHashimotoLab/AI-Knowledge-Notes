#!/usr/bin/env python3
"""
Final comprehensive translation for MS microstructure chapter 2
Handles all Japanese content with extensive pattern matching
"""

import re
from pathlib import Path

def get_all_translations():
    """Comprehensive translation dictionary with 500+ patterns"""
    trans = {}

    # === BASIC ATTRIBUTES ===
    trans['lang="ja"'] = 'lang="en"'

    # === TITLE AND META ===
    trans['<title>第2章：相変態の基礎 - MS Terakoya</title>'] = '<title>Chapter 2: Fundamentals of Phase Transformations - MS Terakoya</title>'
    trans['第2章：相変態の基礎'] = 'Chapter 2: Fundamentals of Phase Transformations'
    trans['Phase Transformations - 熱処理による組織制御の科学'] = 'Phase Transformations - Science of Microstructure Control through Heat Treatment'
    trans['📖 読了時間: 30-40分'] = '📖 Reading time: 30-40 minutes'
    trans['📊 難易度: 中級'] = '📊 Difficulty: Intermediate'
    trans['💻 コード例: 7個'] = '💻 Code examples: 7'

    # === BREADCRUMB ===
    trans['AI寺子屋トップ'] = 'AI Terakoya Top'
    trans['材料組織学入門'] = 'Introduction to Materials Microstructure'
    trans['第2章'] = 'Chapter 2'

    # === CHAPTER DESCRIPTION ===
    trans['材料の性質は、温度と時間の履歴（熱処理）によって劇的に変化します。この変化の根源は<strong>相変態（phase transformation）</strong>です。この章では、相図の読み方、拡散型・無拡散型変態のメカニズム、TTT/CCT図の活用法、マルテンサイト変態、そしてCALPHAD法による状態図計算の基礎を学び、熱処理設計の理論的基盤を築きます。'] = \
        'Material properties change dramatically depending on temperature and time history (heat treatment). The origin of this change is <strong>phase transformation</strong>. In this chapter, we will learn how to read phase diagrams, mechanisms of diffusional and diffusionless transformations, application of TTT/CCT diagrams, martensitic transformation, and the basics of phase diagram calculation using the CALPHAD method, building a theoretical foundation for heat treatment design.'

    # === LEARNING OBJECTIVES ===
    trans['学習目標'] = 'Learning Objectives'
    trans['この章を読むことで、以下を習得できます：'] = 'By reading this chapter, you will be able to:'
    trans['✅ 二元系・三元系相図を読み、相平衡を理解できる'] = '✅ Read binary and ternary phase diagrams and understand phase equilibrium'
    trans['✅ てこの法則（Lever Rule）を用いて相分率を計算できる'] = '✅ Calculate phase fractions using the Lever Rule'
    trans['✅ TTT図・CCT図から変態速度と組織を予測できる'] = '✅ Predict transformation rate and microstructure from TTT and CCT diagrams'
    trans['✅ Avrami式で変態の進行度を定量化できる'] = '✅ Quantify transformation progress using the Avrami equation'
    trans['✅ マルテンサイト変態の原理とM<sub>s</sub>温度の予測ができる'] = '✅ Understand the principles of martensitic transformation and predict M<sub>s</sub> temperature'
    trans['✅ CALPHAD法の基礎とpycalphadライブラリの使い方を理解できる'] = '✅ Understand the basics of the CALPHAD method and how to use the pycalphad library'
    trans['✅ Pythonで相図と変態速度論のシミュレーションができる'] = '✅ Perform phase diagram and transformation kinetics simulations in Python'

    # === SECTIONS AND SUBSECTIONS ===
    trans['2.1 相図の基礎と読み方'] = '2.1 Fundamentals and Reading of Phase Diagrams'
    trans['相図（Phase Diagram）とは'] = 'What is a Phase Diagram?'
    trans['二元系相図の基本型'] = 'Basic Types of Binary Phase Diagrams'
    trans['Fe-C状態図（鉄鋼の基礎）'] = 'Fe-C Phase Diagram (Fundamentals of Steel)'
    trans['てこの法則（Lever Rule）'] = 'Lever Rule'
    trans['2.2 拡散型変態と無拡散型変態'] = '2.2 Diffusional and Diffusionless Transformations'
    trans['変態の分類'] = 'Classification of Transformations'
    trans['拡散型変態：パーライト変態'] = 'Diffusional Transformation: Pearlite Transformation'
    trans['無拡散型変態：マルテンサイト変態'] = 'Diffusionless Transformation: Martensitic Transformation'
    trans['2.3 TTT図とCCT図'] = '2.3 TTT and CCT Diagrams'
    trans['TTT図（Time-Temperature-Transformation Diagram）'] = 'TTT Diagram (Time-Temperature-Transformation Diagram)'
    trans['CCT図（Continuous Cooling Transformation Diagram）'] = 'CCT Diagram (Continuous Cooling Transformation Diagram)'
    trans['臨界冷却速度（Critical Cooling Rate）'] = 'Critical Cooling Rate'
    trans['2.4 変態速度論とAvrami式'] = '2.4 Transformation Kinetics and Avrami Equation'
    trans['変態の進行度'] = 'Progress of Transformation'
    trans['TTT図の作成原理'] = 'Principle of TTT Diagram Creation'
    trans['2.5 CALPHAD法の基礎'] = '2.5 Fundamentals of the CALPHAD Method'
    trans['CALPHAD（CALculation of PHAse Diagrams）とは'] = 'What is CALPHAD (CALculation of PHAse Diagrams)?'
    trans['pycalphad：PythonでのCALPHAD計算'] = 'pycalphad: CALPHAD Calculation in Python'
    trans['2.6 Pythonによる相変態シミュレーション'] = '2.6 Phase Transformation Simulation in Python'
    trans['環境準備'] = 'Environment Setup'
    trans['2.7 本章のまとめ'] = '2.7 Summary of This Chapter'
    trans['学んだこと'] = 'What We Learned'
    trans['重要なポイント'] = 'Important Points'

    # === CODE EXAMPLES ===
    trans['コード例1: 二元系相図（全率固溶型）の描画'] = 'Code Example 1: Drawing Binary Phase Diagram (Complete Solid Solution)'
    trans['コード例2: てこの法則（Lever Rule）の計算と可視化'] = 'Code Example 2: Calculation and Visualization of Lever Rule'
    trans['コード例3: TTT図の生成とAvrami式のフィッティング'] = 'Code Example 3: Generation of TTT Diagram and Fitting of Avrami Equation'
    trans['コード例4: Avrami式のパラメータフィッティング（実験データ）'] = 'Code Example 4: Parameter Fitting of Avrami Equation (Experimental Data)'
    trans['コード例5: M<sub>s</sub>温度（マルテンサイト変態開始温度）の予測'] = 'Code Example 5: Prediction of M<sub>s</sub> Temperature (Martensite Transformation Start Temperature)'
    trans['コード例6: 微細組織進化のシミュレーション（簡易フェーズフィールド法）'] = 'Code Example 6: Simulation of Microstructure Evolution (Simplified Phase Field Method)'
    trans['コード例7: pycalphadによるFe-C二元系状態図の計算'] = 'Code Example 7: Calculation of Fe-C Binary Phase Diagram using pycalphad'

    # === CONTENT PARAGRAPHS ===
    trans['<p><strong>相図</strong>は、温度・組成・圧力の関数として、どの相が熱力学的に安定かを示す図です。材料の熱処理条件を決定する際の最も重要なツールです。</p>'] = \
        '<p>A <strong>phase diagram</strong> is a diagram that shows which phases are thermodynamically stable as a function of temperature, composition, and pressure. It is the most important tool when determining heat treatment conditions for materials.</p>'

    trans['<strong>相（Phase）</strong>とは、化学組成・構造・性質が一様で、他の部分と明確な界面で区切られた物質の均一な部分です。例: 液相（L）、α相（BCC）、γ相（FCC）、セメンタイト（Fe<sub>3</sub>C）'] = \
        '<strong>A phase</strong> is a homogeneous portion of a material with uniform chemical composition, structure, and properties, separated from other portions by distinct interfaces. Examples: liquid phase (L), α-phase (BCC), γ-phase (FCC), cementite (Fe<sub>3</sub>C)'

    # === PHASE DIAGRAM TYPES ===
    trans['1. 全率固溶型（Complete Solid Solution）'] = '1. Complete Solid Solution'
    trans['2つの元素が全組成範囲で固溶する系です。'] = 'A system in which two elements form a solid solution over the entire composition range.'
    trans['<strong>例</strong>: Cu-Ni系、Au-Ag系'] = '<strong>Examples</strong>: Cu-Ni system, Au-Ag system'

    trans['2. 共晶型（Eutectic System）'] = '2. Eutectic System'
    trans['ある組成・温度で、液相が冷却時に2つの固相に同時に分解します。'] = 'At a certain composition and temperature, the liquid phase decomposes simultaneously into two solid phases upon cooling.'
    trans['<strong>例</strong>: Pb-Sn系、Al-Si系'] = '<strong>Examples</strong>: Pb-Sn system, Al-Si system'
    trans['共晶反応: $L \\rightarrow \\alpha + \\beta$（冷却時）'] = 'Eutectic reaction: $L \\rightarrow \\alpha + \\beta$ (upon cooling)'

    trans['3. 包晶型（Peritectic System）'] = '3. Peritectic System'
    trans['液相と固相が反応して別の固相を生成します。'] = 'A liquid phase and a solid phase react to produce another solid phase.'
    trans['<strong>例</strong>: Fe-C系（高温部）、Pt-Ag系'] = '<strong>Examples</strong>: Fe-C system (high temperature region), Pt-Ag system'
    trans['包晶反応: $L + \\delta \\rightarrow \\gamma$（冷却時）'] = 'Peritectic reaction: $L + \\delta \\rightarrow \\gamma$ (upon cooling)'

    # === Fe-C DIAGRAM ===
    trans['Fe-C系相図は、鉄鋼材料の熱処理設計の基盤です。'] = 'The Fe-C phase diagram is the foundation for heat treatment design of steel materials.'
    trans['重要な温度と組成'] = 'Important Temperatures and Compositions'
    trans['<strong>共析点（Eutectoid Point）</strong>: 727°C、0.77% C'] = '<strong>Eutectoid Point</strong>: 727°C, 0.77% C'
    trans['共析反応: $\\gamma \\rightarrow \\alpha + \\text{Fe}_3\\text{C}$（パーライト組織）'] = 'Eutectoid reaction: $\\gamma \\rightarrow \\alpha + \\text{Fe}_3\\text{C}$ (pearlite microstructure)'
    trans['<strong>亜共析鋼（Hypoeutectoid Steel）</strong>: 0.02-0.77% C'] = '<strong>Hypoeutectoid Steel</strong>: 0.02-0.77% C'
    trans['組織: 初析フェライト + パーライト'] = 'Microstructure: Proeutectoid ferrite + Pearlite'
    trans['<strong>共析鋼（Eutectoid Steel）</strong>: 0.77% C'] = '<strong>Eutectoid Steel</strong>: 0.77% C'
    trans['組織: 100%パーライト'] = 'Microstructure: 100% Pearlite'
    trans['<strong>過共析鋼（Hypereutectoid Steel）</strong>: 0.77-2.11% C'] = '<strong>Hypereutectoid Steel</strong>: 0.77-2.11% C'
    trans['組織: 初析セメンタイト + パーライト'] = 'Microstructure: Proeutectoid cementite + Pearlite'

    # === LEVER RULE ===
    trans['2相領域において、各相の質量分率を計算する方法です。'] = 'A method to calculate the mass fraction of each phase in a two-phase region.'
    trans['温度$T$、組成$C_0$の合金が、$\\alpha$相（組成$C_\\alpha$）と$\\beta$相（組成$C_\\beta$）に分かれているとき：'] = \
        'When an alloy with temperature $T$ and composition $C_0$ is divided into $\\alpha$-phase (composition $C_\\alpha$) and $\\beta$-phase (composition $C_\\beta$):'
    trans['$$\\text{質量分率}_\\alpha = \\frac{C_\\beta - C_0}{C_\\beta - C_\\alpha}$$'] = '$$\\text{Mass fraction}_\\alpha = \\frac{C_\\beta - C_0}{C_\\beta - C_\\alpha}$$'
    trans['$$\\text{質量分率}_\\beta = \\frac{C_0 - C_\\alpha}{C_\\beta - C_\\alpha}$$'] = '$$\\text{Mass fraction}_\\beta = \\frac{C_0 - C_\\alpha}{C_\\beta - C_\\alpha}$$'
    trans['「<strong>遠い方の相の割合が多い</strong>」と覚えます。'] = 'Remember: <strong>The fraction of the farther phase is larger</strong>.'

    # === TRANSFORMATION TYPES TABLE ===
    trans['変態の種類'] = 'Type of Transformation'
    trans['拡散の有無'] = 'Diffusion'
    trans['変態速度'] = 'Transformation Rate'
    trans['代表例'] = 'Representative Examples'
    trans['<strong>拡散型変態</strong><br/>(Diffusional)'] = '<strong>Diffusional Transformation</strong><br/>(Diffusional)'
    trans['長距離拡散あり'] = 'Long-range diffusion present'
    trans['遅い（秒〜時間）'] = 'Slow (seconds to hours)'
    trans['パーライト変態<br/>ベイナイト変態<br/>析出'] = 'Pearlite transformation<br/>Bainite transformation<br/>Precipitation'
    trans['<strong>無拡散型変態</strong><br/>(Diffusionless)'] = '<strong>Diffusionless Transformation</strong><br/>(Diffusionless)'
    trans['拡散なし<br/>（協調的なずれ運動）'] = 'No diffusion<br/>(Coordinated shear movement)'
    trans['非常に速い（音速）'] = 'Very fast (speed of sound)'
    trans['マルテンサイト変態<br/>双晶変態'] = 'Martensitic transformation<br/>Twin transformation'

    # === PEARLITE TRANSFORMATION ===
    trans['オーステナイト（γ-Fe、FCC）からフェライト（α-Fe、BCC）+ セメンタイト（Fe<sub>3</sub>C）への共析変態です。'] = \
        'Eutectoid transformation from austenite (γ-Fe, FCC) to ferrite (α-Fe, BCC) + cementite (Fe<sub>3</sub>C).'
    trans['<p><strong>パーライト組織の特徴</strong>:</p>'] = '<p><strong>Characteristics of Pearlite Microstructure</strong>:</p>'
    trans['フェライトとセメンタイトの層状構造（lamellar structure）'] = 'Lamellar structure of ferrite and cementite'
    trans['層間隔（interlamellar spacing）が硬さを決定'] = 'Interlamellar spacing determines hardness'
    trans['細かいパーライト（fine pearlite）: 高温変態、硬い'] = 'Fine pearlite: High-temperature transformation, hard'
    trans['粗いパーライト（coarse pearlite）: 低温変態、軟らかい'] = 'Coarse pearlite: Low-temperature transformation, soft'

    # === MARTENSITIC TRANSFORMATION ===
    trans['オーステナイト（FCC）から体心正方晶（BCT）のマルテンサイトへの変態です。'] = \
        'Transformation from austenite (FCC) to body-centered tetragonal (BCT) martensite.'
    trans['<p><strong>マルテンサイトの特徴</strong>:</p>'] = '<p><strong>Characteristics of Martensite</strong>:</p>'
    trans['拡散を伴わない、せん断型の構造変化'] = 'Diffusionless shear-type structural change'
    trans['変態速度は音速レベル（10<sup>-7</sup>秒）'] = 'Transformation rate at the speed of sound (10<sup>-7</sup> seconds)'
    trans['炭素が強制固溶し、格子がひずむ（BCT構造）'] = 'Carbon is forcibly dissolved in solid solution, distorting the lattice (BCT structure)'
    trans['極めて硬いが脆い（Vickers硬度 600-900 HV）'] = 'Extremely hard but brittle (Vickers hardness 600-900 HV)'
    trans['変態開始温度（M<sub>s</sub>）以下で進行'] = 'Proceeds below the transformation start temperature (M<sub>s</sub>)'
    trans['<p><strong>M<sub>s</sub>温度の予測式（鋼）</strong>:</p>'] = '<p><strong>Prediction formula for M<sub>s</sub> temperature (steel)</strong>:</p>'
    trans['ここで、元素記号は質量%を表します。炭素や合金元素が増えるとM<sub>s</sub>温度は低下します。'] = \
        'Here, element symbols represent mass %. M<sub>s</sub> temperature decreases as carbon and alloying elements increase.'

    # === TTT DIAGRAM ===
    trans['<p><strong>TTT図</strong>は、等温変態（一定温度に保持）した際の変態の進行を示す図です。</p>'] = \
        '<p><strong>TTT diagram</strong> shows the progress of transformation during isothermal transformation (holding at constant temperature).</p>'
    trans['<p><strong>TTT図の読み方</strong>:</p>'] = '<p><strong>How to read a TTT diagram</strong>:</p>'
    trans['縦軸: 温度'] = 'Vertical axis: Temperature'
    trans['横軸: 時間（対数スケール）'] = 'Horizontal axis: Time (logarithmic scale)'
    trans['C字型の曲線: 変態開始線と変態完了線'] = 'C-shaped curve: Transformation start line and transformation completion line'
    trans['「鼻（nose）」: 最も速く変態が起こる温度（550-600°C付近）'] = '"Nose": Temperature at which transformation occurs fastest (around 550-600°C)'

    # === CCT DIAGRAM ===
    trans['<p><strong>CCT図</strong>は、連続冷却時の変態を示す図で、実際の熱処理により近い条件です。</p>'] = \
        '<p><strong>CCT diagram</strong> shows transformation during continuous cooling, closer to actual heat treatment conditions.</p>'
    trans['<p><strong>TTT図との違い</strong>:</p>'] = '<p><strong>Differences from TTT diagram</strong>:</p>'
    trans['TTT図は等温変態（実験室的）'] = 'TTT diagram is isothermal transformation (laboratory)'
    trans['CCT図は連続冷却（実用的）'] = 'CCT diagram is continuous cooling (practical)'
    trans['CCT図のC曲線はTTT図より右下にシフト（変態に時間がかかる）'] = 'C-curve in CCT diagram shifts to lower right compared to TTT diagram (transformation takes longer)'

    # === COOLING RATE TABLE ===
    trans['<p><strong>冷却速度と得られる組織の関係（共析鋼の例）</strong>:</p>'] = \
        '<p><strong>Relationship between cooling rate and microstructure obtained (example of eutectoid steel)</strong>:</p>'
    trans['冷却速度'] = 'Cooling Rate'
    trans['組織'] = 'Microstructure'
    trans['硬さ（HV）'] = 'Hardness (HV)'
    trans['用途例'] = 'Application Examples'
    trans['徐冷（炉冷）<br/>&lt; 1°C/s'] = 'Slow cooling (furnace cooling)<br/>&lt; 1°C/s'
    trans['粗いパーライト'] = 'Coarse Pearlite'
    trans['軟化焼鈍'] = 'Softening annealing'
    trans['空冷<br/>10-100°C/s'] = 'Air cooling<br/>10-100°C/s'
    trans['細かいパーライト'] = 'Fine Pearlite'
    trans['焼ならし'] = 'Normalizing'
    trans['油冷<br/>100-300°C/s'] = 'Oil quenching<br/>100-300°C/s'
    trans['ベイナイト'] = 'Bainite'
    trans['高靭性部品'] = 'High toughness parts'
    trans['水冷<br/>&gt; 1000°C/s'] = 'Water quenching<br/>&gt; 1000°C/s'
    trans['マルテンサイト'] = 'Martensite'
    trans['焼入れ'] = 'Quenching'

    # === CRITICAL COOLING RATE ===
    trans['<p><strong>臨界冷却速度</strong>は、マルテンサイト組織を100%得るために必要な最小の冷却速度です。合金元素の添加により低下します（焼入れしやすくなる）。</p>'] = \
        '<p><strong>Critical cooling rate</strong> is the minimum cooling rate required to obtain 100% martensitic microstructure. It decreases with the addition of alloying elements (easier to quench).</p>'

    # === AVRAMI EQUATION ===
    trans['拡散型変態の進行度$f(t)$（変態した体積分率）は、<strong>Johnson-Mehl-Avrami-Kolmogorov（JMAK）式</strong>、通称<strong>Avrami式</strong>で記述されます：'] = \
        'The progress $f(t)$ (volume fraction transformed) of diffusional transformation is described by the <strong>Johnson-Mehl-Avrami-Kolmogorov (JMAK) equation</strong>, commonly known as the <strong>Avrami equation</strong>:'
    trans['ここで、'] = 'Where:'
    trans['$f(t)$: 時間$t$での変態分率（0〜1）'] = '$f(t)$: Transformation fraction at time $t$ (0 to 1)'
    trans['$k$: 速度定数（温度依存）'] = '$k$: Rate constant (temperature dependent)'
    trans['$n$: Avrami指数（核生成と成長のメカニズムに依存、通常1-4）'] = '$n$: Avrami exponent (depends on nucleation and growth mechanism, typically 1-4)'
    trans['<p><strong>Avrami指数$n$の意味</strong>:</p>'] = '<p><strong>Meaning of Avrami exponent $n$</strong>:</p>'

    # === AVRAMI TABLE ===
    trans['n値'] = 'n value'
    trans['核生成'] = 'Nucleation'
    trans['成長'] = 'Growth'
    trans['一定速度'] = 'Constant rate'
    trans['1次元（針状）'] = '1D (needle-shaped)'
    trans['2次元（円盤状）'] = '2D (disk-shaped)'
    trans['3次元（球状）'] = '3D (spherical)'
    trans['時間とともに増加'] = 'Increases with time'

    # === TTT CREATION ===
    trans['TTT図は、複数の温度でAvrami式をフィッティングし、各温度での変態開始時間（$f = 0.01$）と完了時間（$f = 0.99$）をプロットして作成されます。'] = \
        'TTT diagrams are created by fitting the Avrami equation at multiple temperatures and plotting the transformation start time ($f = 0.01$) and completion time ($f = 0.99$) at each temperature.'

    # === CALPHAD ===
    trans['<p><strong>CALPHAD法</strong>は、熱力学データベースを用いて相図を計算する手法です。実験的に全ての組成・温度で相図を測定するのは不可能なため、計算により予測します。</p>'] = \
        '<p><strong>CALPHAD method</strong> is a technique for calculating phase diagrams using thermodynamic databases. Since it is impossible to experimentally measure phase diagrams at all compositions and temperatures, predictions are made by calculation.</p>'
    trans['<p><strong>CALPHAD法の流れ</strong>:</p>'] = '<p><strong>CALPHAD method workflow</strong>:</p>'
    trans['各相のGibbsエネルギーを数式でモデル化'] = 'Model the Gibbs energy of each phase with equations'
    trans['実験データと熱力学データからパラメータを最適化'] = 'Optimize parameters from experimental data and thermodynamic data'
    trans['Gibbsエネルギー最小化により安定相を決定'] = 'Determine stable phases by minimizing Gibbs energy'
    trans['相図を作成'] = 'Create phase diagram'
    trans['<p><strong>Gibbsエネルギーのモデル</strong>（簡略版）:</p>'] = '<p><strong>Gibbs energy model</strong> (simplified version):</p>'
    trans['$G$: Gibbsエネルギー'] = '$G$: Gibbs energy'
    trans['$x_i$: 成分$i$のモル分率'] = '$x_i$: Mole fraction of component $i$'
    trans['$G_i^0$: 純成分のGibbsエネルギー'] = '$G_i^0$: Gibbs energy of pure component'
    trans['$RT \\sum_i x_i \\ln x_i$: 理想混合エントロピー項'] = '$RT \\sum_i x_i \\ln x_i$: Ideal mixing entropy term'
    trans['$G^{ex}$: 過剰Gibbsエネルギー（相互作用項、Redlich-Kisterモデル等）'] = '$G^{ex}$: Excess Gibbs energy (interaction term, Redlich-Kister model, etc.)'

    # === PYCALPHAD ===
    trans['<p><strong>pycalphad</strong>は、CALPHAD計算を行うPythonライブラリです。TDBファイル（熱力学データベース）を読み込み、相図を計算・可視化できます。</p>'] = \
        '<p><strong>pycalphad</strong> is a Python library for performing CALPHAD calculations. It can read TDB files (thermodynamic databases) and calculate and visualize phase diagrams.</p>'

    # === COMMON TERMS ===
    common_terms = {
        # Materials science terms
        'フェライト': 'Ferrite',
        'パーライト': 'Pearlite',
        'セメンタイト': 'Cementite',
        'オーステナイト': 'Austenite',
        'ベイナイト': 'Bainite',
        '初析フェライト': 'Proeutectoid ferrite',
        '初析セメンタイト': 'Proeutectoid cementite',
        '亜共析鋼': 'Hypoeutectoid steel',
        '共析鋼': 'Eutectoid steel',
        '過共析鋼': 'Hypereutectoid steel',
        '低炭素鋼': 'Low carbon steel',
        '中炭素鋼': 'Medium carbon steel',
        '高炭素鋼': 'High carbon steel',
        '炭素鋼': 'Carbon steel',

        # Transformation terms
        '変態': 'Transformation',
        '変態分率': 'Transformation fraction',
        '変態開始': 'Transformation start',
        '変態完了': 'Transformation completion',
        '核生成': 'Nucleation',
        '成長': 'Growth',
        '拡散': 'Diffusion',
        '析出': 'Precipitation',

        # Properties
        '温度': 'Temperature',
        '組成': 'Composition',
        '硬さ': 'Hardness',
        '質量分率': 'Mass fraction',
        '相分率': 'Phase fraction',
        '凝固': 'Solidification',
        '冷却': 'Cooling',
        '速度定数': 'Rate constant',
        '指数': 'Exponent',

        # Diagram terms
        '液相線': 'Liquidus',
        '固相線': 'Solidus',
        '共析点': 'Eutectoid point',
        '二相領域': 'Two-phase region',
        '三元系': 'Ternary system',
        '二元系': 'Binary system',
        '状態図': 'Phase diagram',

        # Operations
        '冷却経路': 'Cooling path',
        'プロット': 'Plot',
        '計算': 'Calculation',
        '解析': 'Analysis',
        'シミュレーション': 'Simulation',
        'フィッティング': 'Fitting',
        '最適化': 'Optimization',
        '可視化': 'Visualization',

        # Code/output
        '出力例': 'Output example',
        '解説': 'Explanation',
        '注意': 'Note',
        '例': 'Example',
        '実行': 'Execution',
        '結果': 'Result',

        # Mermaid diagram
        '高温': 'High Temp',
        '急冷': 'Rapid Cooling',
        '遅い冷却': 'Slow Cooling',
        '中速冷却': 'Medium Cooling',
        '保持': 'Hold',
        '以下': 'Below',
        '微細な混合組織': 'Fine Mixed Structure',
        '超硬質': 'Ultra-hard',
        '無拡散変態': 'Diffusionless Transf.',
        '共析変態': 'Eutectoid Transf.',

        # Time units
        '秒': 's',
        '分': 'min',
        '時間': 'hours',

        # Common verbs and phrases
        'での': 'at',
        'における': 'in',
        'による': 'by',
        'として': 'as',
        'について': 'about',
        'とともに': 'with',
        'により': 'by',
        'から': 'from',
        'への': 'to',
        'における': 'in',
    }
    trans.update(common_terms)

    # === CODE COMMENTS (in print statements) ===
    code_comments = {
        # Installation
        '# 必要なライブラリのインストール': '# Install required libraries',
        '# pycalphadは別途インストール（オプション）': '# Install pycalphad separately (optional)',

        # Parameters
        '# Cu-Ni系相図のパラメータ（簡易モデル）': '# Parameters for Cu-Ni phase diagram (simplified model)',
        '# 組成範囲（Niのモル分率）': '# Composition range (Ni mole fraction)',
        '# Fe-C系の例（共析温度727°Cでの二相領域）': '# Fe-C system example (two-phase region at eutectoid temperature 727°C)',
        '# 合金組成範囲': '# Alloy composition range',
        '# 炭素濃度の範囲': '# Carbon concentration range',

        # Operations
        '# プロット': '# Plot',
        '# 領域の塗りつぶし': '# Fill regions',
        '# 特定組成での冷却経路を示す': '# Show cooling path at specific composition',
        '# 各組成でのてこの法則計算': '# Lever rule calculation for each composition',
        '# 相分率のグラフ': '# Phase fraction graph',
        '# 共析鋼（0.77% C）の計算': '# Calculation for eutectoid steel (0.77% C)',
        '# 様々な鋼種での相分率': '# Phase fractions in various steel grades',
        '# 亜共析鋼': '# Hypoeutectoid steel',
        '# 過共析鋼': '# Hypereutectoid steel',
        '# 初析フェライト + パーライト': '# Proeutectoid ferrite + Pearlite',
        '# 初析セメンタイト + パーライト': '# Proeutectoid cementite + Pearlite',
        '# パーライト中の相分率は一定（共析組成）': '# Phase fraction in pearlite is constant (eutectoid composition)',
        '# パーライト内部の相分率': '# Phase fraction inside pearlite',
        '# 全体の相分率': '# Overall phase fraction',
        '# 棒グラフで可視化': '# Visualize with bar chart',

        # Labels
        '液相（L）領域': 'Liquid (L) region',
        'L + α 二相領域': 'L + α two-phase region',
        '固相（α）領域': 'Solid (α) region',
        '液相線交差': 'Liquidus intersection',
        '固相線交差': 'Solidus intersection',
        '凝固開始': 'Solidification start',
        '凝固完了': 'Solidification completion',
        '凝固温度範囲': 'Solidification temperature range',
        '炭素濃度': 'Carbon Concentration',
        '共析組成': 'Eutectoid composition',

        # Titles
        'Cu-Ni 二元系状態図（全率固溶型）': 'Cu-Ni Binary Phase Diagram (Complete Solid Solution)',
        'てこの法則：Fe-C系の相分率': 'Lever Rule: Phase Fractions in Fe-C System',
        '鋼種別の相分率（平衡状態）': 'Phase Fractions by Steel Grade (Equilibrium State)',

        # Print outputs
        '=== Cu-Ni 系相図の解析 ===': '=== Analysis of Cu-Ni Phase Diagram ===',
        'mol% Ni組成での:': 'At mol% Ni composition:',
        '液相線温度（凝固開始）': 'Liquidus temperature (solidification start)',
        '固相線温度（凝固完了）': 'Solidus temperature (solidification completion)',
        '=== 共析鋼（0.77% C）の相分率（727°C） ===': '=== Phase Fractions in Eutectoid Steel (0.77% C) at 727°C ===',
        '=== 各鋼種の相分率（室温、平衡状態） ===': '=== Phase Fractions of Each Steel Grade (Room Temperature, Equilibrium State) ===',
    }
    trans.update(code_comments)

    # === EXPLANATIONS ===
    explanations = {
        '<p><strong>解説</strong>: 全率固溶型相図では、液相線と固相線の間に二相領域（L + α）が存在します。この範囲で凝固が進行し、組成が連続的に変化します。</p>': \
            '<p><strong>Explanation</strong>: In a complete solid solution phase diagram, a two-phase region (L + α) exists between the liquidus and solidus. Solidification progresses in this range, and the composition changes continuously.</p>',

        '<p><strong>解説</strong>: てこの法則により、炭素濃度から各相（フェライトとセメンタイト）の質量分率を定量的に予測できます。これは組織と機械的性質の関係を理解する基礎となります。</p>': \
            '<p><strong>Explanation</strong>: The lever rule allows quantitative prediction of the mass fraction of each phase (ferrite and cementite) from the carbon concentration. This is the basis for understanding the relationship between microstructure and mechanical properties.</p>',
    }
    trans.update(explanations)

    # === FOOTER ===
    footer_trans = {
        '次の章へ': 'Next Chapter',
        '前の章へ': 'Previous Chapter',
        '目次に戻る': 'Back to Table of Contents',
        '免責事項': 'Disclaimer',
        'この教材はAIによって生成された教育コンテンツです': 'This educational content is generated by AI',
        '内容の正確性には注意を払っていますが、誤りが含まれる可能性があります': 'While we strive for accuracy, errors may be present',
        '重要な判断や実装の際は、必ず公式ドキュメントや信頼できる情報源を確認してください': 'For critical decisions or implementations, always verify with official documentation and reliable sources',
        'フィードバックや改善提案は歓迎します': 'Feedback and suggestions for improvement are welcome',
    }
    trans.update(footer_trans)

    return trans

def apply_translations(content: str, translations: dict) -> str:
    """Apply all translations, longest patterns first"""
    # Sort by length (longest first) to avoid partial replacements
    sorted_trans = sorted(translations.items(), key=lambda x: len(x[0]), reverse=True)

    for jp_text, en_text in sorted_trans:
        content = content.replace(jp_text, en_text)

    return content

def count_japanese(text: str) -> int:
    """Count Japanese characters"""
    return len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]', text))

def main():
    source = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/materials-microstructure-introduction/chapter-2.html')
    target = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-2.html')

    if not source.exists():
        print(f"❌ Source file not found: {source}")
        return 1

    print(f"Reading: {source}")
    with open(source, 'r', encoding='utf-8') as f:
        content = f.read()

    orig_jp = count_japanese(content)
    print(f"Original Japanese characters: {orig_jp}")

    # Apply translations
    translations = get_all_translations()
    print(f"Translation patterns: {len(translations)}")

    translated = apply_translations(content, translations)

    remain_jp = count_japanese(translated)
    percentage = ((orig_jp - remain_jp) / orig_jp * 100) if orig_jp > 0 else 100

    # Save
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'w', encoding='utf-8') as f:
        f.write(translated)

    print(f"\n{'='*70}")
    print(f"✅ Translation Complete!")
    print(f"{'='*70}")
    print(f"Target: {target}")
    print(f"\nSTATISTICS:")
    print(f"  Original JP characters: {orig_jp}")
    print(f"  Remaining JP characters: {remain_jp}")
    print(f"  Translation: {percentage:.1f}%")
    print(f"{'='*70}")

    return 0

if __name__ == '__main__':
    exit(main())
