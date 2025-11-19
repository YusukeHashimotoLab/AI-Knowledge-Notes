#!/usr/bin/env python3
"""
Complete translation script for NM Introduction Chapter 1
Handles all Japanese content systematically while preserving structure
"""

import re
import sys

def translate_content():
    """Main translation function with comprehensive mappings"""

    source_file = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MI/nm-introduction/chapter1-introduction.html"
    target_file = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MI/nm-introduction/chapter1-introduction.html"

    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Comprehensive translation dictionary
    translations = {
        # HTML lang attribute
        'lang="ja"': 'lang="en"',

        # Page title and meta
        'Chapter 1: ナノ材料入門 - AI Terakoya': 'Chapter 1: Introduction to Nanomaterials - AI Terakoya',

        # Header content
        'Chapter 1: ナノ材料入門': 'Chapter 1: Introduction to Nanomaterials',
        'ナノスケールの世界とサイズ効果': 'The Nanoscale World and Size Effects',
        '📖 読了時間: 20-25分': '📖 Reading time: 20-25 minutes',
        '📊 難易度: 初級': '📊 Difficulty: Beginner',
        '💻 コード例: 0個': '💻 Code examples: 0',
        '📝 演習問題: 0問': '📝 Practice problems: 0',

        # Breadcrumb navigation
        'AI寺子屋トップ': 'AI Terakoya Home',
        'マテリアルズ・インフォマティクス': 'Materials Informatics',

        # Chapter description
        'ナノスケールで現れる独特の物性とサイズ効果を直感的に理解します。代表的なナノ材料の分類と歴史的背景を素早く掴みます。': 'Gain an intuitive understanding of the unique physical properties and size effects that emerge at the nanoscale. Quickly grasp the classification and historical background of representative nanomaterials.',
        '💡 補足': '💡 Supplement',
        '「小さくなるほど表面の振る舞いが支配的に」。量子閉じ込めは"音階が粗くなる"イメージで理解すると掴みやすいです。': '"The smaller it gets, the more surface behavior dominates." Quantum confinement is easier to grasp when understood as an image of "musical notes becoming coarser."',

        # Learning objectives
        '本章の学習目標': 'Learning Objectives for This Chapter',
        '本章を学習することで、以下のことができるようになります:': 'By studying this chapter, you will be able to:',
        '✅ ナノスケールのサイズ感覚を理解し、日常的なスケールと比較できる': '✅ Understand the sense of scale at the nanoscale and compare it with everyday scales',
        '✅ 表面積/体積比の増大がもたらす物性変化を定量的に説明できる': '✅ Quantitatively explain the physical property changes brought about by the increase in surface area-to-volume ratio',
        '✅ 量子効果と量子閉じ込め効果の基本原理を理解できる': '✅ Understand the basic principles of quantum effects and quantum confinement effects',
        '✅ ナノ材料を次元(0D/1D/2D/3D)に基づいて分類できる': '✅ Classify nanomaterials based on dimensionality (0D/1D/2D/3D)',
        '✅ ナノ材料の主要な応用分野とその特徴を説明できる': '✅ Explain the main application areas of nanomaterials and their characteristics',
        '✅ ナノ材料の安全性と倫理的課題について議論できる': '✅ Discuss the safety and ethical issues of nanomaterials',

        # Section 1.1
        '1.1 ナノ材料とは': '1.1 What are Nanomaterials?',
        'ナノスケールの定義': 'Definition of Nanoscale',
        'ナノ材料(Nanomaterials)を理解する第一歩は、「ナノ」というスケールを実感することです。': 'The first step in understanding nanomaterials is to get a sense of the "nano" scale.',
        'ナノメートル(nm) は、1メートルの10億分の1という極めて小さな長さの単位です:': 'A nanometer (nm) is an extremely small unit of length, one-billionth of a meter:',
        'この途方もなく小さなスケールを理解するために、身近なサイズと比較してみましょう:': "To understand this incredibly small scale, let's compare it with familiar sizes:",

        # Common terms and phrases
        '対象': 'Object',
        'サイズ': 'Size',
        'ナノメートル換算': 'Nanometer equivalent',
        '人間の身長': 'Human height',
        '約': 'approx.',
        '髪の毛の太さ': 'Hair thickness',
        '赤血球': 'Red blood cell',
        '細菌(大腸菌)': 'Bacteria (E. coli)',
        'ウイルス(インフルエンザ)': 'Virus (influenza)',
        'ナノ材料の典型的サイズ': 'Typical size of nanomaterials',
        'DNAの二重らせん直径': 'DNA double helix diameter',
        '水分子': 'Water molecule',
        '原子(炭素)': 'Atom (carbon)',

        'ナノ材料は、ウイルスと同じくらいか、それより小さいスケールの材料です。このスケールでは、数個から数千個の原子が集まって一つの構造を形成しています。': 'Nanomaterials are materials at a scale similar to or smaller than viruses. At this scale, structures are formed by the assembly of a few to several thousand atoms.',

        'ナノ材料の定義': 'Definition of Nanomaterials',
        '国際標準化機構(ISO)の技術仕様書ISO/TS 80004-1では、ナノ材料を以下のように定義しています:': 'The International Organization for Standardization (ISO) technical specification ISO/TS 80004-1 defines nanomaterials as follows:',
        'ナノ材料: 少なくとも一つの外部寸法、または内部構造がナノスケール(おおよそ1 nmから100 nm)にある材料': 'Nanomaterials: Materials with at least one external dimension or internal structure at the nanoscale (approximately 1 nm to 100 nm)',

        'この定義の重要なポイントは、「少なくとも一つの次元」という部分です。つまり、三次元すべてがナノサイズである必要はなく、一つの方向だけがナノサイズであっても、ナノ材料と呼ばれます。この考え方が、後述する次元別分類(0D、1D、2D、3D)につながります。': 'The important point of this definition is the phrase "at least one dimension". In other words, not all three dimensions need to be nano-sized; even if only one direction is nano-sized, it is called a nanomaterial. This concept leads to the dimensional classification (0D, 1D, 2D, 3D) discussed later.',

        'ナノ材料の主要な特徴は以下の4つです:': 'The four main characteristics of nanomaterials are:',
        '表面積/体積比の飛躍的増大: サイズが小さくなるほど、表面に存在する原子の割合が増加します': 'Dramatic increase in surface area-to-volume ratio: As size decreases, the proportion of atoms on the surface increases',
        '量子効果の発現: 粒子サイズが電子の波長と同程度になると、量子力学的効果が顕著になります': 'Emergence of quantum effects: When particle size becomes comparable to the wavelength of electrons, quantum mechanical effects become prominent',
        'サイズ依存的な物性: 同じ化学組成でも、サイズによって色、融点、触媒活性などが変化します': 'Size-dependent physical properties: Even with the same chemical composition, properties such as color, melting point, and catalytic activity change with size',
        '特異な光学特性: 金属ナノ粒子の局在表面プラズモン共鳴など、バルク材料にはない光学特性が現れます': 'Unique optical properties: Optical properties not found in bulk materials appear, such as localized surface plasmon resonance in metal nanoparticles',

        'なぜナノ材料が注目されるのか': 'Why are Nanomaterials Attracting Attention?',
        'バルク材料(通常サイズの材料)とナノ材料では、同じ化学組成でも全く異なる性質を示すことがあります。': 'Bulk materials (normal-sized materials) and nanomaterials can exhibit completely different properties even with the same chemical composition.',
        '代表的な例として、金(Au) のサイズ効果を見てみましょう:': "As a representative example, let's look at the size effect of gold (Au):",

        # Table headers
        '粒子サイズ': 'Particle size',
        '色': 'Color',
        '融点': 'Melting point',
        '特徴': 'Characteristics',
        'バルク(塊)': 'Bulk (mass)',
        '金色(黄金色)': 'Golden color (golden yellow)',
        '化学的に安定、触媒活性なし': 'Chemically stable, no catalytic activity',
        '青紫色': 'Blue-purple',
        '局在表面プラズモン共鳴': 'Localized surface plasmon resonance',
        '赤色': 'Red',
        '強い光吸収、バイオイメージング': 'Strong light absorption, bioimaging',
        '赤〜紫色': 'Red to purple',
        '高い触媒活性': 'High catalytic activity',
        '変化': 'Variable',
        '量子効果の発現': 'Emergence of quantum effects',

        '同じ金という元素でも、粒子サイズによってこれほど大きく性質が変わるのです。このサイズ依存性こそが、ナノ材料研究の魅力であり、様々な応用可能性を生み出す源泉となっています。': 'Even though it is the same element, gold, the properties change this dramatically depending on particle size. This size dependence is the charm of nanomaterials research and the source of various application possibilities.',

        # Section 1.2
        '1.2 サイズ効果と表面・界面効果': '1.2 Size Effects and Surface/Interface Effects',
        '表面積/体積比の増大': 'Increase in Surface Area-to-Volume Ratio',
        'ナノ材料の最も重要な特性の一つが、表面積/体積比の飛躍的増大です。': 'One of the most important properties of nanomaterials is the dramatic increase in surface area-to-volume ratio.',
        '簡単な例として、半径 $r$ の球形粒子を考えてみましょう。': "As a simple example, let's consider a spherical particle with radius $r$.",
        '表面積: $S = 4\\pi r^2$': 'Surface area: $S = 4\\pi r^2$',
        '体積: $V = \\frac{4}{3}\\pi r^3$': 'Volume: $V = \\frac{4}{3}\\pi r^3$',
        '表面積/体積比:': 'Surface area-to-volume ratio:',

        'この式から、粒子半径が小さくなるほど、表面積/体積比が増大することがわかります。つまり、サイズが1/10になれば、表面積/体積比は10倍になります。': 'From this equation, we can see that as the particle radius decreases, the surface area-to-volume ratio increases. In other words, if the size becomes 1/10, the surface area-to-volume ratio becomes 10 times larger.',
        '具体的な数値で比較してみましょう:': "Let's compare with specific numerical values:",

        '粒子直径': 'Particle diameter',
        '表面積/体積比': 'Surface area-to-volume ratio',
        '総原子数(Au)': 'Total atoms (Au)',
        '表面原子の割合': 'Percentage of surface atoms',

        '10 nmの金ナノ粒子では、全原子の約40%が表面に存在します。2 nmになると、なんと80%もの原子が表面にあります。': 'In 10 nm gold nanoparticles, about 40% of all atoms are on the surface. At 2 nm, an astonishing 80% of atoms are on the surface.',
        'この表面原子の増大が、以下のような劇的な物性変化をもたらします:': 'This increase in surface atoms brings about the following dramatic changes in physical properties:',
        '触媒活性の向上: 反応は主に表面で起こるため': 'Enhanced catalytic activity: Because reactions mainly occur on the surface',
        '反応性の増大: 表面原子は内部原子より不安定': 'Increased reactivity: Surface atoms are less stable than interior atoms',
        '融点の低下: 表面エネルギーの寄与が大きくなる': 'Decrease in melting point: Surface energy contribution becomes larger',
        '溶解度の変化: 表面積増大により溶解速度が上昇': 'Change in solubility: Dissolution rate increases due to increased surface area',

        '表面エネルギーの影響': 'Influence of Surface Energy',
        'ナノ粒子では、表面エネルギーが材料全体の性質に大きな影響を与えます。': 'In nanoparticles, surface energy has a significant impact on the overall properties of the material.',
        '代表的な現象が融点降下(Melting point depression) です。ナノ粒子は、バルク材料より低い温度で融解します。': 'A representative phenomenon is melting point depression. Nanoparticles melt at lower temperatures than bulk materials.',
        'この現象はGibbs-Thomson効果として知られ、以下の式で近似できます:': 'This phenomenon is known as the Gibbs-Thomson effect and can be approximated by the following equation:',
        'ここで:': 'where:',
        '$T_m(r)$: 半径 $r$ の粒子の融点': '$T_m(r)$: Melting point of particle with radius $r$',
        '$T_{m,\\text{bulk}}$: バルク材料の融点': '$T_{m,\\text{bulk}}$: Melting point of bulk material',
        '$\\gamma$: 表面エネルギー(表面張力)': '$\\gamma$: Surface energy (surface tension)',
        '$V_m$: モル体積': '$V_m$: Molar volume',
        '$\\Delta H_f$: 融解エンタルピー': '$\\Delta H_f$: Enthalpy of fusion',
        '$r$: 粒子半径': '$r$: Particle radius',

        '金ナノ粒子の融点の実験データ:': 'Experimental data on melting points of gold nanoparticles:',
        'バルクからの低下': 'Decrease from bulk',
        'バルク': 'Bulk',

        '2 nmの金ナノ粒子は、バルクの金より700°C以上も低い温度で融解します。この性質は、低温焼結材料や熱応答性材料の開発に利用されています。': '2 nm gold nanoparticles melt at temperatures more than 700°C lower than bulk gold. This property is utilized in the development of low-temperature sintering materials and thermo-responsive materials.',

        '触媒活性の向上': 'Enhancement of Catalytic Activity',
        '表面積/体積比の増大は、触媒活性の飛躍的向上につながります。': 'The increase in surface area-to-volume ratio leads to a dramatic enhancement of catalytic activity.',
        '白金(Pt)触媒を例に考えてみましょう:': "Let's consider platinum (Pt) catalyst as an example:",
        '用途: 燃料電池の電極触媒、自動車排ガス浄化触媒': 'Applications: Fuel cell electrode catalyst, automotive exhaust purification catalyst',
        '反応: 水素酸化反応(H₂ → 2H⁺ + 2e⁻)': 'Reaction: Hydrogen oxidation reaction (H₂ → 2H⁺ + 2e⁻)',
        '白金の粒子サイズと触媒活性の関係:': 'Relationship between platinum particle size and catalytic activity:',

        'Pt粒子サイズ': 'Pt particle size',
        '表面積(g当たり)': 'Surface area (per g)',
        '相対触媒活性': 'Relative catalytic activity',
        'コスト効率': 'Cost efficiency',
        'バルク板': 'Bulk plate',
        '粉末': 'Powder',
        'ナノ粒子': 'Nanoparticles',

        '3 nmの白金ナノ粒子は、バルクの白金板と比べて1,500倍の触媒活性を示します。これは、同じ質量の白金から1,500倍の性能を引き出せることを意味し、希少金属の使用量削減に大きく貢献しています。': '3 nm platinum nanoparticles show 1,500 times the catalytic activity compared to bulk platinum plates. This means that 1,500 times the performance can be extracted from the same mass of platinum, greatly contributing to the reduction in the use of rare metals.',
    }

    # Apply translations - do a first pass
    for jp_text, en_text in translations.items():
        content = content.replace(jp_text, en_text)

    # Write the result
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Translation complete!")
    print(f"Source: {source_file}")
    print(f"Target: {target_file}")

    # Count remaining Japanese characters for verification
    jp_pattern = re.compile(r'[ぁ-んァ-ヶー一-龯]+')
    jp_matches = jp_pattern.findall(content)
    jp_count = len(jp_matches)

    print(f"\nRemaining Japanese text segments: {jp_count}")
    if jp_count > 0 and jp_count < 50:
        print("\nFirst few remaining Japanese segments:")
        for i, match in enumerate(jp_matches[:10]):
            print(f"  {i+1}. {match}")

    return target_file

if __name__ == "__main__":
    translate_content()
