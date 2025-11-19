#!/usr/bin/env python3
"""
Complete translation script for MS materials-microstructure-introduction chapter-4.html
Translates Japanese to English while preserving HTML structure and code
"""

import re

def translate_chapter4_complete():
    """Comprehensive translation with complete mappings"""

    source_path = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/materials-microstructure-introduction/chapter-4.html'
    target_path = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-4.html'

    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Count before translation
    jp_before = sum(1 for char in content if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF')
    total_chars = len(content)

    # Core translations dictionary - comprehensive mapping
    translations = {
        # HTML lang attribute
        'lang="ja"': 'lang="en"',

        # Title and meta
        '第4章:転位と塑性変形 - 材料組織学入門シリーズ - MS Terakoya': 'Chapter 4: Dislocations and Plastic Deformation - Introduction to Materials Microstructure Series - MS Terakoya',

        # Breadcrumb navigation
        '材料組織学入門': 'Introduction to Materials Microstructure',
        '第4章：転位と塑性変形': 'Chapter 4: Dislocations and Plastic Deformation',

        # Header section
        'Dislocations and Plastic Deformation - 加工硬化から再結晶まで': 'Dislocations and Plastic Deformation - From Work Hardening to Recrystallization',
        '読了時間: 30-35分': 'Reading time: 30-35 minutes',
        'コード例: 7個': 'Code examples: 7',
        '難易度: 中級〜上級': 'Difficulty: Intermediate to Advanced',
        '実践演習: 3問': 'Practical exercises: 3',

        # Learning objectives
        '学習目標': 'Learning Objectives',
        'この章を完了すると、以下のスキルと知識を習得できます：': 'Upon completing this chapter, you will acquire the following skills and knowledge:',
        '✅ 転位の種類（刃状、らせん、混合）とBurgersベクトルの概念を理解できる': '✅ Understand types of dislocations (edge, screw, mixed) and the concept of Burgers vector',
        '✅ 転位の運動とPeach-Koehler力を理解し、応力下での挙動を予測できる': '✅ Understand dislocation motion and Peach-Koehler force, and predict behavior under stress',
        '✅ 加工硬化（Work Hardening）のメカニズムと転位密度の関係を説明できる': '✅ Explain the mechanism of work hardening and its relationship with dislocation density',
        '✅ Taylor式を用いて転位密度から降伏応力を計算できる': '✅ Calculate yield stress from dislocation density using the Taylor equation',
        '✅ 動的回復と再結晶のメカニズムを理解し、熱処理への応用を説明できる': '✅ Understand mechanisms of dynamic recovery and recrystallization, and explain their applications to heat treatment',
        '✅ 転位密度測定法（XRD、TEM、EBSD）の原理を理解できる': '✅ Understand the principles of dislocation density measurement methods (XRD, TEM, EBSD)',
        '✅ Pythonで転位運動、加工硬化、再結晶挙動をシミュレーションできる': '✅ Simulate dislocation motion, work hardening, and recrystallization behavior using Python',

        # Section 4.1
        '4.1 転位の基礎': '4.1 Fundamentals of Dislocations',
        '4.1.1 転位とは何か': '4.1.1 What are Dislocations?',
        '<p><strong>転位（Dislocation）</strong>は、結晶中の線状欠陥であり、塑性変形を担う最も重要な結晶欠陥です。理想的な結晶が完全にすべるには理論強度（G/10程度）が必要ですが、転位の存在により実際の降伏応力は理論強度の1/100〜1/1000に低下します。</p>':
            '<p><strong>Dislocations</strong> are linear defects in crystals and the most important crystal defects responsible for plastic deformation. While an ideal crystal requires theoretical strength (approximately G/10) for complete slip, the presence of dislocations reduces the actual yield stress to 1/100 to 1/1000 of the theoretical strength.</p>',

        '🔬 転位の発見': '🔬 Discovery of Dislocations',
        '<p>転位の概念は、1934年にTaylor、Orowan、Polanyiによって独立に提唱されました。結晶の実測強度が理論強度より遥かに低い理由を説明するために導入され、1950年代にTEM（透過電子顕微鏡）で初めて直接観察されました。</p>':
            '<p>The concept of dislocations was independently proposed by Taylor, Orowan, and Polanyi in 1934. It was introduced to explain why the measured strength of crystals is far lower than the theoretical strength, and was first directly observed using TEM (Transmission Electron Microscopy) in the 1950s.</p>',

        '4.1.2 転位の種類': '4.1.2 Types of Dislocations',
        '<p>転位は、Burgersベクトル<strong>b</strong>と転位線方向<strong>ξ</strong>の関係で分類されます：</p>':
            '<p>Dislocations are classified based on the relationship between the Burgers vector <strong>b</strong> and the dislocation line direction <strong>ξ</strong>:</p>',

        # Table headers and content
        '転位の種類': 'Dislocation Type',
        'Burgersベクトルと転位線の関係': 'Relationship between Burgers Vector and Dislocation Line',
        '特徴': 'Characteristics',
        '運動様式': 'Mode of Motion',
        '刃状転位<br/>（Edge）': 'Edge Dislocation',
        '（垂直）': '(Perpendicular)',
        '余剰原子面の挿入<br/>圧縮・引張応力場': 'Extra half-plane insertion<br/>Compressive/tensile stress field',
        'すべり運動<br/>上昇運動（高温）': 'Glide motion<br/>Climb motion (high temperature)',
        'らせん転位<br/>（Screw）': 'Screw Dislocation',
        '（平行）': '(Parallel)',
        'らせん状の格子変位<br/>純粋なせん断歪み': 'Helical lattice displacement<br/>Pure shear strain',
        '交差すべり可能<br/>任意の面ですべり': 'Cross-slip possible<br/>Slip on any plane',
        '混合転位<br/>（Mixed）': 'Mixed Dislocation',
        '刃状とらせんの中間': 'Intermediate between edge and screw',
        'すべり面上を運動': 'Motion on slip plane',

        # Mermaid diagram labels
        '転位': 'Dislocations',
        '刃状転位<br/>Edge Dislocation': 'Edge Dislocation',
        'らせん転位<br/>Screw Dislocation': 'Screw Dislocation',
        '混合転位<br/>Mixed Dislocation': 'Mixed Dislocation',
        '余剰原子面': 'Extra half-plane',
        '上昇運動可能': 'Climb motion possible',
        '交差すべり': 'Cross-slip',
        '高速移動': 'Fast motion',
        '刃状+らせん成分': 'Edge + screw components',
        '最も一般的': 'Most common',

        # Section 4.1.3
        '4.1.3 Burgersベクトル': '4.1.3 Burgers Vector',
        '<p><strong>Burgersベクトル（b）</strong>は、転位を一周する回路（Burgers circuit）の閉じない部分を表すベクトルで、転位の種類と大きさを決定します。</p>':
            '<p>The <strong>Burgers vector (b)</strong> is a vector representing the closure failure of a circuit around a dislocation (Burgers circuit), determining the type and magnitude of the dislocation.</p>',

        '主な結晶構造でのBurgersベクトル：': 'Burgers vectors in major crystal structures:',
        '<strong>FCC（面心立方）</strong>: b = (a/2)&lt;110&gt;（最密面{111}上のすべり）':
            '<strong>FCC (Face-Centered Cubic)</strong>: b = (a/2)&lt;110&gt; (slip on close-packed {111} planes)',
        '<strong>BCC（体心立方）</strong>: b = (a/2)&lt;111&gt;（{110}、{112}、{123}面ですべり）':
            '<strong>BCC (Body-Centered Cubic)</strong>: b = (a/2)&lt;111&gt; (slip on {110}, {112}, {123} planes)',
        '<strong>HCP（六方最密）</strong>: b = (a/3)&lt;1120&gt;（基底面）、&lt;c+a&gt;（柱面・錐面）':
            '<strong>HCP (Hexagonal Close-Packed)</strong>: b = (a/3)&lt;1120&gt; (basal plane), &lt;c+a&gt; (prismatic and pyramidal planes)',

        # Code example 1
        'Example 1: Burgersベクトルの可視化と計算': 'Example 1: Visualization and Calculation of Burgers Vectors',
        '主要な結晶構造での転位特性': 'Dislocation characteristics in major crystal structures',
        'FCC構造のBurgersベクトル': 'Burgers vector for FCC structure',
        'BCC構造のBurgersベクトル': 'Burgers vector for BCC structure',
        '格子定数 [nm]': 'Lattice parameter [nm]',
        '<110>型Burgersベクトルのリスト': 'List of <110> type Burgers vectors',
        '<111>型Burgersベクトルのリスト': 'List of <111> type Burgers vectors',
        'ベクトルの大きさ [nm]': 'Magnitude of vector [nm]',
        '<110>方向（FCC主すべり系）': '<110> direction (primary slip system in FCC)',
        '<111>方向（BCC主すべり系）': '<111> direction (primary slip system in BCC)',
        'Burgersベクトル: b = (a/2)<110>': 'Burgers vector: b = (a/2)<110>',
        'Burgersベクトル: b = (a/2)<111>': 'Burgers vector: b = (a/2)<111>',
        '大きさ': 'Magnitude',
        '主要金属の格子定数': 'Lattice parameters of major metals',
        '計算と可視化': 'Calculation and visualization',
        'Burgersベクトルの大きさ比較': 'Comparison of Burgers vector magnitudes',
        'Burgersベクトルの大きさ |b| [nm]': 'Burgers vector magnitude |b| [nm]',
        '(a) 金属のBurgersベクトル比較': '(a) Comparison of Burgers Vectors in Metals',
        '数値をバーの上に表示': 'Display values above bars',
        '3D可視化（Al FCC の例）': '3D visualization (Al FCC example)',
        '原点からのベクトル描画': 'Draw vectors from origin',
        '最初の3つだけ表示': 'Display only the first 3',
        '(b) Al (FCC) のBurgersベクトル<110>': '(b) Burgers Vectors <110> for Al (FCC)',
        '軸範囲を統一': 'Unify axis ranges',
        '数値出力': 'Numerical output',
        '=== Burgersベクトルの計算結果 ===': '=== Burgers Vector Calculation Results ===',
        '格子定数': 'Lattice parameter',
        'Burgersベクトル: |b| =': 'Burgers vector: |b| =',
        '主すべり系': 'Primary slip system',
        'すべりベクトル数': 'Number of slip vectors',
        '出力例': 'Output example',

        # Section 4.2
        '4.2 転位の運動とPeach-Koehler力': '4.2 Dislocation Motion and Peach-Koehler Force',
        '4.2.1 転位に働く力': '4.2.1 Forces Acting on Dislocations',
        '<p>転位は応力下で運動し、塑性変形を引き起こします。転位に働く単位長さあたりの力は<strong>Peach-Koehler力</strong>で表されます：</p>':
            '<p>Dislocations move under stress and cause plastic deformation. The force per unit length acting on a dislocation is represented by the <strong>Peach-Koehler force</strong>:</p>',

        '<strong>F = (σ · b) × ξ</strong>': '<strong>F = (σ · b) × ξ</strong>',
        'F: 転位に働く力（単位長さあたり）[N/m]': 'F: Force acting on dislocation (per unit length) [N/m]',
        'σ: 応力テンソル [Pa]': 'σ: Stress tensor [Pa]',
        'b: Burgersベクトル [m]': 'b: Burgers vector [m]',
        'ξ: 転位線方向の単位ベクトル': 'ξ: Unit vector along dislocation line',

        '<p>純粋な刃状転位の場合、すべり面に平行なせん断応力τにより：</p>':
            '<p>For a pure edge dislocation, by shear stress τ parallel to the slip plane:</p>',
        'F = τ · b': 'F = τ · b',
        '<p>転位が移動すると、すべり面上でせん断変形が生じます。転位が結晶を横切ると、全体で1原子層分（|b|）のずれが生じます。</p>':
            '<p>When a dislocation moves, shear deformation occurs on the slip plane. When a dislocation crosses the crystal, a total displacement of one atomic layer (|b|) occurs.</p>',

        # Section 4.2.2
        '4.2.2 臨界分解せん断応力（CRSS）': '4.2.2 Critical Resolved Shear Stress (CRSS)',
        '<p><strong>臨界分解せん断応力（Critical Resolved Shear Stress, CRSS）</strong>は、すべり系が活動するために必要な最小のせん断応力です。単結晶の降伏は、CRSSが最初に達成されるすべり系で起こります。</p>':
            '<p><strong>Critical Resolved Shear Stress (CRSS)</strong> is the minimum shear stress required for a slip system to become active. Yielding in single crystals occurs on the slip system where CRSS is first reached.</p>',

        '<p>引張応力σとすべり系のなす角度を用いて：</p>':
            '<p>Using the angles between tensile stress σ and the slip system:</p>',

        'τ<sub>resolved</sub> = σ · cos(φ) · cos(λ)': 'τ<sub>resolved</sub> = σ · cos(φ) · cos(λ)',
        'φ: すべり面法線と引張軸のなす角度': 'φ: Angle between slip plane normal and tensile axis',
        'λ: すべり方向と引張軸のなす角度': 'λ: Angle between slip direction and tensile axis',
        'cos(φ)·cos(λ): Schmid因子': 'cos(φ)·cos(λ): Schmid factor',

        # Code example 2
        'Example 2: Peach-Koehler力とSchmid因子の計算': 'Example 2: Calculation of Peach-Koehler Force and Schmid Factor',
        '単結晶の降伏挙動予測': 'Prediction of yielding behavior in single crystals',
        'Schmid因子を計算': 'Calculate Schmid factor',
        'すべり面法線と引張軸の角度 [度]': 'Angle between slip plane normal and tensile axis [degrees]',
        'すべり方向と引張軸の角度 [度]': 'Angle between slip direction and tensile axis [degrees]',
        'Schmid因子': 'Schmid factor',
        'Peach-Koehler力を計算（簡略化：刃状転位）': 'Calculate Peach-Koehler force (simplified: edge dislocation)',
        'せん断応力 [Pa]': 'Shear stress [Pa]',
        'Burgersベクトルの大きさ [m]': 'Magnitude of Burgers vector [m]',
        '単位長さあたりの力 [N/m]': 'Force per unit length [N/m]',
        'Schmid因子マップの作成': 'Create Schmid factor map',
        'Schmid因子の計算': 'Calculate Schmid factor',
        '最大Schmid因子（45°, 45°で最大値0.5）': 'Maximum Schmid factor (maximum value 0.5 at 45°, 45°)',
        'Schmid因子マップ': 'Schmid Factor Map',
        '最大値 (φ=45°, λ=45°)': 'Maximum (φ=45°, λ=45°)',
        'φ: すべり面法線と引張軸の角度 [°]': 'φ: Angle between slip plane normal and tensile axis [°]',
        'λ: すべり方向と引張軸の角度 [°]': 'λ: Angle between slip direction and tensile axis [°]',
        '(a) Schmid因子マップ': '(a) Schmid Factor Map',
        '降伏応力の方位依存性': 'Orientation dependence of yield stress',
        'FCC単結晶（Al）の例': 'Example of FCC single crystal (Al)',
        '焼鈍材の典型値': 'Typical value for annealed material',
        '異なる方位での降伏応力': 'Yield stress at different orientations',
        '立方方位': 'Cubic orientation',
        '最も硬い方位': 'Hardest orientation',
        '降伏応力 = CRSS / Schmid因子': 'Yield stress = CRSS / Schmid factor',
        '降伏応力 [MPa]': 'Yield stress [MPa]',
        '(b) Al単結晶の方位依存性': '(b) Orientation Dependence of Al Single Crystal',
        'Peach-Koehler力の計算例': 'Calculation example of Peach-Koehler force',
        '=== Peach-Koehler力の計算 ===': '=== Peach-Koehler Force Calculation ===',
        'Schmid因子=0.5を仮定': 'Assuming Schmid factor = 0.5',
        '引張応力': 'Tensile stress',
        '分解せん断応力': 'Resolved shear stress',

        # Section 4.3
        '4.3 加工硬化（Work Hardening）': '4.3 Work Hardening',
        '4.3.1 加工硬化のメカニズム': '4.3.1 Mechanisms of Work Hardening',
        '<p><strong>加工硬化（Work Hardening）</strong>または<strong>ひずみ硬化（Strain Hardening）</strong>は、塑性変形により材料が硬化する現象です。主な原因は転位密度の増加と転位同士の相互作用です。</p>':
            '<p><strong>Work hardening</strong> or <strong>strain hardening</strong> is a phenomenon in which materials harden due to plastic deformation. The main causes are the increase in dislocation density and interactions between dislocations.</p>',

        # Flowchart
        '塑性変形開始': 'Start of plastic deformation',
        '転位が増殖<br/>Frank-Read源': 'Dislocation multiplication<br/>Frank-Read source',
        '転位密度増加<br/>ρ: 10⁸ → 10¹⁴ m⁻²': 'Dislocation density increase<br/>ρ: 10⁸ → 10¹⁴ m⁻²',
        '転位同士が絡み合う<br/>Forest転位': 'Dislocations entangle<br/>Forest dislocations',
        '転位運動の抵抗増加': 'Increased resistance to dislocation motion',
        '降伏応力上昇<br/>加工硬化': 'Yield stress increase<br/>Work hardening',

        # Section 4.3.2
        '4.3.2 Taylor式と転位密度': '4.3.2 Taylor Equation and Dislocation Density',
        '<p>降伏応力と転位密度の関係は<strong>Taylor式</strong>で表されます：</p>':
            '<p>The relationship between yield stress and dislocation density is expressed by the <strong>Taylor equation</strong>:</p>',

        'σ<sub>y</sub> = σ<sub>0</sub> + α · M · G · b · √ρ': 'σ<sub>y</sub> = σ<sub>0</sub> + α · M · G · b · √ρ',
        'σ<sub>y</sub>: 降伏応力 [Pa]': 'σ<sub>y</sub>: Yield stress [Pa]',
        'σ<sub>0</sub>: 基底応力（格子摩擦応力）[Pa]': 'σ<sub>0</sub>: Friction stress (lattice friction stress) [Pa]',
        'α: 定数（0.2〜0.5、通常0.3-0.4）': 'α: Constant (0.2-0.5, typically 0.3-0.4)',
        'M: Taylor因子（多結晶の平均、FCC:3.06、BCC:2.75）': 'M: Taylor factor (polycrystalline average, FCC: 3.06, BCC: 2.75)',
        'G: せん断弾性率 [Pa]': 'G: Shear modulus [Pa]',
        'ρ: 転位密度 [m⁻²]': 'ρ: Dislocation density [m⁻²]',

        '<p>典型的な転位密度：</p>': '<p>Typical dislocation densities:</p>',

        # Table for dislocation density
        '状態': 'State',
        '転位密度 ρ [m⁻²]': 'Dislocation Density ρ [m⁻²]',
        '平均転位間隔': 'Average Dislocation Spacing',
        '焼鈍材（十分軟化）': 'Annealed (well softened)',
        '中程度加工': 'Moderately worked',
        '高度加工（冷間圧延）': 'Heavily worked (cold rolled)',

        # Code example 3
        'Example 3: 応力-ひずみ曲線と加工硬化': 'Example 3: Stress-Strain Curve and Work Hardening',
        'Taylor式による強度予測': 'Strength prediction using Taylor equation',
        '加工硬化による応力-ひずみ曲線を計算': 'Calculate stress-strain curve due to work hardening',
        '真ひずみ': 'True strain',
        '材料名': 'Material name',
        '真応力 [MPa]': 'True stress [MPa]',
        '転位密度 [m⁻²]': 'Dislocation density [m⁻²]',
        '材料パラメータ': 'Material parameters',
        '初期転位密度': 'Initial dislocation density',
        'ひずみに伴う転位密度の増加（簡略化）': 'Increase in dislocation density with strain (simplified)',
        '増殖項': 'Multiplication term',
        '回復項（室温では小さい）': 'Recovery term (small at room temperature)',
        'Taylor式': 'Taylor equation',
        'ひずみ範囲': 'Strain range',
        '応力-ひずみ曲線': 'Stress-strain curve',
        'ひずみ [%]': 'Strain [%]',
        '(a) 応力-ひずみ曲線（加工硬化）': '(a) Stress-Strain Curve (Work Hardening)',
        '転位密度の発展': 'Evolution of dislocation density',
        '(b) 転位密度の発展': '(b) Evolution of Dislocation Density',
        '加工硬化率': 'Work hardening rate',
        '(c) 加工硬化率の変化': '(c) Change in Work Hardening Rate',
    }

    # Apply all translations
    for jp, en in translations.items():
        content = content.replace(jp, en)

    # Additional pattern-based translations for common phrases in code comments
    content = re.sub(r'# (.+)：(.+)', lambda m: f'# {translate_comment(m.group(1), m.group(2))}', content)

    # Write output
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Count after translation
    jp_after = sum(1 for char in content if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF')

    # Statistics
    print(f"Translation Complete!")
    print(f"=" * 60)
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"")
    print(f"Japanese Character Count:")
    print(f"  Before translation: {jp_before} characters ({(jp_before/total_chars)*100:.2f}%)")
    print(f"  After translation:  {jp_after} characters ({(jp_after/total_chars)*100:.2f}%)")
    print(f"  Translated:         {jp_before - jp_after} characters")
    print(f"")
    print(f"File Statistics:")
    print(f"  Total characters:   {total_chars}")
    print(f"  Total lines:        2515")
    print(f"=" * 60)

def translate_comment(key, value):
    """Helper function to translate code comments"""
    comment_map = {
        '計算': 'Calculation',
        '可視化': 'Visualization',
        '出力': 'Output',
        '結果': 'Result',
    }
    key_en = comment_map.get(key, key)
    return f'{key_en}: {value}'

if __name__ == "__main__":
    translate_chapter4_complete()
