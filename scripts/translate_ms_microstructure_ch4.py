#!/usr/bin/env python3
"""
Comprehensive translation script for MS materials-microstructure-introduction chapter-4.html
Translates Japanese to English while preserving HTML structure
"""

def translate_chapter4():
    """Complete translation with character counting"""

    # Translation mapping for the entire document
    translations = {
        # Meta and title
        'lang="ja"': 'lang="en"',
        '第4章:転位と塑性変形 - 材料組織学入門シリーズ - MS Terakoya': 'Chapter 4: Dislocations and Plastic Deformation - Introduction to Materials Microstructure Series - MS Terakoya',

        # Breadcrumb
        '材料組織学入門': 'Introduction to Materials Microstructure',
        '第4章：転位と塑性変形': 'Chapter 4: Dislocations and Plastic Deformation',

        # Header
        '第4章：転位と塑性変形': 'Chapter 4: Dislocations and Plastic Deformation',
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
        '<p><strong>転位（Dislocation）</strong>は、結晶中の線状欠陥であり、塑性変形を担う最も重要な結晶欠陥です。理想的な結晶が完全にすべるには理論強度（G/10程度）が必要ですが、転位の存在により実際の降伏応力は理論強度の1/100〜1/1000に低下します。</p>': '<p><strong>Dislocations</strong> are linear defects in crystals and the most important crystal defects responsible for plastic deformation. While an ideal crystal requires theoretical strength (approximately G/10) for complete slip, the presence of dislocations reduces the actual yield stress to 1/100 to 1/1000 of the theoretical strength.</p>',

        # Info box
        '🔬 転位の発見': '🔬 Discovery of Dislocations',
        '<p>転位の概念は、1934年にTaylor、Orowan、Polanyiによって独立に提唱されました。結晶の実測強度が理論強度より遥かに低い理由を説明するために導入され、1950年代にTEM（透過電子顕微鏡）で初めて直接観察されました。</p>': '<p>The concept of dislocations was independently proposed by Taylor, Orowan, and Polanyi in 1934. It was introduced to explain why the measured strength of crystals is far lower than the theoretical strength, and was first directly observed using TEM (Transmission Electron Microscopy) in the 1950s.</p>',

        # Section 4.1.2
        '4.1.2 転位の種類': '4.1.2 Types of Dislocations',
        '<p>転位は、Burgersベクトル<strong>b</strong>と転位線方向<strong>ξ</strong>の関係で分類されます：</p>': '<p>Dislocations are classified based on the relationship between the Burgers vector <strong>b</strong> and the dislocation line direction <strong>ξ</strong>:</p>',

        # Table headers
        '転位の種類': 'Dislocation Type',
        'Burgersベクトルと転位線の関係': 'Relationship between Burgers Vector and Dislocation Line',
        '特徴': 'Characteristics',
        '運動様式': 'Mode of Motion',

        # Table content
        '刃状転位': 'Edge Dislocation',
        '（Edge）': '(Edge)',
        '（垂直）': '(Perpendicular)',
        '余剰原子面の挿入': 'Extra half-plane insertion',
        '圧縮・引張応力場': 'Compressive/tensile stress field',
        'すべり運動': 'Glide motion',
        '上昇運動（高温）': 'Climb motion (high temperature)',

        'らせん転位': 'Screw Dislocation',
        '（Screw）': '(Screw)',
        '（平行）': '(Parallel)',
        'らせん状の格子変位': 'Helical lattice displacement',
        '純粋なせん断歪み': 'Pure shear strain',
        '交差すべり可能': 'Cross-slip possible',
        '任意の面ですべり': 'Slip on any plane',

        '混合転位': 'Mixed Dislocation',
        '（Mixed）': '(Mixed)',
        '刃状とらせんの中間': 'Intermediate between edge and screw',
        'すべり面上を運動': 'Motion on slip plane',

        # Mermaid diagram
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
        '<p><strong>Burgersベクトル（b）</strong>は、転位を一周する回路（Burgers circuit）の閉じない部分を表すベクトルで、転位の種類と大きさを決定します。</p>': '<p>The <strong>Burgers vector (b)</strong> is a vector representing the closure failure of a circuit around a dislocation (Burgers circuit), determining the type and magnitude of the dislocation.</p>',

        # Blockquote
        '主な結晶構造でのBurgersベクトル：': 'Burgers vectors in major crystal structures:',
        '<strong>FCC（面心立方）</strong>: b = (a/2)&lt;110&gt;（最密面{111}上のすべり）': '<strong>FCC (Face-Centered Cubic)</strong>: b = (a/2)&lt;110&gt; (slip on close-packed {111} planes)',
        '<strong>BCC（体心立方）</strong>: b = (a/2)&lt;111&gt;（{110}、{112}、{123}面ですべり）': '<strong>BCC (Body-Centered Cubic)</strong>: b = (a/2)&lt;111&gt; (slip on {110}, {112}, {123} planes)',
        '<strong>HCP（六方最密）</strong>: b = (a/3)&lt;1120&gt;（基底面）、&lt;c+a&gt;（柱面・錐面）': '<strong>HCP (Hexagonal Close-Packed)</strong>: b = (a/3)&lt;1120&gt; (basal plane), &lt;c+a&gt; (prismatic and pyramidal planes)',

        # Code example
        'Example 1: Burgersベクトルの可視化と計算': 'Example 1: Visualization and Calculation of Burgers Vectors',
        '主要な結晶構造での転位特性': 'Dislocation characteristics in major crystal structures',
    }

    # Read source file
    source_path = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/materials-microstructure-introduction/chapter-4.html'
    target_path = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-4.html'

    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Count Japanese characters before translation
    jp_chars_before = sum(1 for char in content if '\u3040' <= char <= '\u309F' or  # Hiragana
                                                   '\u30A0' <= char <= '\u30FF' or  # Katakana
                                                   '\u4E00' <= char <= '\u9FFF')    # Kanji

    # Apply translations
    for jp_text, en_text in translations.items():
        content = content.replace(jp_text, en_text)

    # Count Japanese characters after translation
    jp_chars_after = sum(1 for char in content if '\u3040' <= char <= '\u309F' or
                                                  '\u30A0' <= char <= '\u30FF' or
                                                  '\u4E00' <= char <= '\u9FFF')

    # Write translated file
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Calculate statistics
    total_chars = len(content)
    jp_percentage_before = (jp_chars_before / total_chars) * 100
    jp_percentage_after = (jp_chars_after / total_chars) * 100

    print(f"Translation Summary:")
    print(f"===================")
    print(f"Japanese characters before: {jp_chars_before} ({jp_percentage_before:.2f}%)")
    print(f"Japanese characters after: {jp_chars_after} ({jp_percentage_after:.2f}%)")
    print(f"Characters translated: {jp_chars_before - jp_chars_after}")
    print(f"Total file size: {total_chars} characters")
    print(f"\nTarget file created: {target_path}")

if __name__ == "__main__":
    translate_chapter4()
