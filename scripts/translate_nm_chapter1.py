#!/usr/bin/env python3
"""
Translation script: JP → EN for NM Introduction Chapter 1
Translates Japanese HTML content to English while preserving all structure and formatting.
"""

import re
import html

def translate_nm_chapter1():
    """Main translation function for nanomaterial introduction chapter 1"""

    # Read the Japanese source file
    source_path = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MI/nm-introduction/chapter1-introduction.html"
    target_path = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MI/nm-introduction/chapter1-introduction.html"

    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Translation mappings
    translations = {
        # Meta and header
        'lang="ja"': 'lang="en"',
        'Chapter 1: ナノ材料入門 - AI Terakoya': 'Chapter 1: Introduction to Nanomaterials - AI Terakoya',
        'ナノスケールの世界とサイズ効果': 'The Nanoscale World and Size Effects',
        '読了時間: 20-25分': 'Reading time: 20-25 minutes',
        '難易度: 初級': 'Difficulty: Beginner',
        'コード例: 0個': 'Code examples: 0',
        '演習問題: 0問': 'Practice problems: 0',

        # Breadcrumb
        'AI寺子屋トップ': 'AI Terakoya Home',
        'マテリアルズ・インフォマティクス': 'Materials Informatics',

        # Main headings
        'Chapter 1: ナノ材料入門': 'Chapter 1: Introduction to Nanomaterials',
        'ナノスケールで現れる独特の物性とサイズ効果を直感的に理解します。代表的なナノ材料の分類と歴史的背景を素早く掴みます。': 'Gain an intuitive understanding of the unique physical properties and size effects that emerge at the nanoscale. Quickly grasp the classification and historical background of representative nanomaterials.',
        '💡 補足:': '💡 Supplement:',
        '「小さくなるほど表面の振る舞いが支配的に」。量子閉じ込めは"音階が粗くなる"イメージで理解すると掴みやすいです。': '"The smaller it gets, the more surface behavior dominates." Quantum confinement is easier to grasp when understood as an image of "musical notes becoming coarser."',

        # Learning objectives
        '本章の学習目標': 'Learning Objectives for This Chapter',
        '本章を学習することで、以下のことができるようになります:': 'By studying this chapter, you will be able to:',
        'ナノスケールのサイズ感覚を理解し、日常的なスケールと比較できる': 'Understand the sense of scale at the nanoscale and compare it with everyday scales',
        '表面積/体積比の増大がもたらす物性変化を定量的に説明できる': 'Quantitatively explain the physical property changes brought about by the increase in surface area-to-volume ratio',
        '量子効果と量子閉じ込め効果の基本原理を理解できる': 'Understand the basic principles of quantum effects and quantum confinement effects',
        'ナノ材料を次元(0D/1D/2D/3D)に基づいて分類できる': 'Classify nanomaterials based on dimensionality (0D/1D/2D/3D)',
        'ナノ材料の主要な応用分野とその特徴を説明できる': 'Explain the main application areas of nanomaterials and their characteristics',
        'ナノ材料の安全性と倫理的課題について議論できる': 'Discuss the safety and ethical issues of nanomaterials',

        # Section 1.1
        '1.1 ナノ材料とは': '1.1 What are Nanomaterials?',
        'ナノスケールの定義': 'Definition of Nanoscale',
        'ナノ材料(Nanomaterials)を理解する第一歩は、「ナノ」というスケールを実感することです。': 'The first step in understanding nanomaterials is to get a sense of the "nano" scale.',
        'ナノメートル(nm)': 'nanometer (nm)',
        'は、1メートルの10億分の1という極めて小さな長さの単位です:': 'is an extremely small unit of length, one-billionth of a meter:',
        'この途方もなく小さなスケールを理解するために、身近なサイズと比較してみましょう:': "To understand this incredibly small scale, let's compare it with familiar sizes:",

        # Table 1
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
        'ナノ材料': 'Nanomaterials',
        '少なくとも一つの外部寸法、または内部構造がナノスケール(おおよそ1 nmから100 nm)にある材料': 'Materials with at least one external dimension or internal structure at the nanoscale (approximately 1 nm to 100 nm)',

        'この定義の重要なポイントは、「少なくとも一つの次元」という部分です。つまり、三次元すべてがナノサイズである必要はなく、一つの方向だけがナノサイズであっても、ナノ材料と呼ばれます。この考え方が、後述する次元別分類(0D、1D、2D、3D)につながります。': 'The important point of this definition is the phrase "at least one dimension". In other words, not all three dimensions need to be nano-sized; even if only one direction is nano-sized, it is called a nanomaterial. This concept leads to the dimensional classification (0D, 1D, 2D, 3D) discussed later.',

        'ナノ材料の主要な特徴は以下の4つです:': 'The four main characteristics of nanomaterials are:',
        '表面積/体積比の飛躍的増大': 'Dramatic increase in surface area-to-volume ratio',
        'サイズが小さくなるほど、表面に存在する原子の割合が増加します': 'As size decreases, the proportion of atoms on the surface increases',
        '量子効果の発現': 'Emergence of quantum effects',
        '粒子サイズが電子の波長と同程度になると、量子力学的効果が顕著になります': 'When particle size becomes comparable to the wavelength of electrons, quantum mechanical effects become prominent',
        'サイズ依存的な物性': 'Size-dependent physical properties',
        '同じ化学組成でも、サイズによって色、融点、触媒活性などが変化します': 'Even with the same chemical composition, properties such as color, melting point, and catalytic activity change with size',
        '特異な光学特性': 'Unique optical properties',
        '金属ナノ粒子の局在表面プラズモン共鳴など、バルク材料にはない光学特性が現れます': 'Optical properties not found in bulk materials appear, such as localized surface plasmon resonance in metal nanoparticles',

        'なぜナノ材料が注目されるのか': 'Why are Nanomaterials Attracting Attention?',
        'バルク材料(通常サイズの材料)とナノ材料では、同じ化学組成でも全く異なる性質を示すことがあります。': 'Bulk materials (normal-sized materials) and nanomaterials can exhibit completely different properties even with the same chemical composition.',
        '代表的な例として、金(Au)のサイズ効果を見てみましょう:': 'As a representative example, let\'s look at the size effect of gold (Au):',

        # Gold table
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
        '簡単な例として、半径': 'As a simple example, consider a spherical particle with radius',
        'の球形粒子を考えてみましょう。': '.',
        '表面積': 'Surface area',
        '体積': 'Volume',
        '表面積/体積比': 'Surface area-to-volume ratio',

        'この式から、粒子半径が小さくなるほど、表面積/体積比が増大することがわかります。つまり、サイズが1/10になれば、表面積/体積比は10倍になります。': 'From this equation, we can see that as the particle radius decreases, the surface area-to-volume ratio increases. In other words, if the size becomes 1/10, the surface area-to-volume ratio becomes 10 times larger.',
        '具体的な数値で比較してみましょう:': 'Let\'s compare with specific numerical values:',

        # SA/V table
        '粒子直径': 'Particle diameter',
        '総原子数(Au)': 'Total atoms (Au)',
        '表面原子の割合': 'Percentage of surface atoms',

        '10 nmの金ナノ粒子では、全原子の約40%が表面に存在します。2 nmになると、なんと80%もの原子が表面にあります。': 'In 10 nm gold nanoparticles, about 40% of all atoms are on the surface. At 2 nm, an astonishing 80% of atoms are on the surface.',
        'この表面原子の増大が、以下のような劇的な物性変化をもたらします:': 'This increase in surface atoms brings about the following dramatic changes in physical properties:',
        '触媒活性の向上': 'Enhanced catalytic activity',
        '反応は主に表面で起こるため': 'Because reactions mainly occur on the surface',
        '反応性の増大': 'Increased reactivity',
        '表面原子は内部原子より不安定': 'Surface atoms are less stable than interior atoms',
        '融点の低下': 'Decrease in melting point',
        '表面エネルギーの寄与が大きくなる': 'Surface energy contribution becomes larger',
        '溶解度の変化': 'Change in solubility',
        '表面積増大により溶解速度が上昇': 'Dissolution rate increases due to increased surface area',

        '表面エネルギーの影響': 'Influence of Surface Energy',
        'ナノ粒子では、表面エネルギーが材料全体の性質に大きな影響を与えます。': 'In nanoparticles, surface energy has a significant impact on the overall properties of the material.',
        '代表的な現象が融点降下(Melting point depression)です。ナノ粒子は、バルク材料より低い温度で融解します。': 'A representative phenomenon is melting point depression. Nanoparticles melt at lower temperatures than bulk materials.',
        'この現象はGibbs-Thomson効果として知られ、以下の式で近似できます:': 'This phenomenon is known as the Gibbs-Thomson effect and can be approximated by the following equation:',
        'ここで:': 'where:',
        '半径': 'radius',
        'の粒子の融点': 'melting point of particle with',
        'バルク材料の融点': 'melting point of bulk material',
        '表面エネルギー(表面張力)': 'surface energy (surface tension)',
        'モル体積': 'molar volume',
        '融解エンタルピー': 'enthalpy of fusion',
        '粒子半径': 'particle radius',

        '金ナノ粒子の融点の実験データ:': 'Experimental data on melting points of gold nanoparticles:',
        'バルクからの低下': 'Decrease from bulk',
        'バルク': 'Bulk',

        '2 nmの金ナノ粒子は、バルクの金より700°C以上も低い温度で融解します。この性質は、低温焼結材料や熱応答性材料の開発に利用されています。': '2 nm gold nanoparticles melt at temperatures more than 700°C lower than bulk gold. This property is utilized in the development of low-temperature sintering materials and thermo-responsive materials.',

        '触媒活性の向上': 'Enhancement of Catalytic Activity',
        '表面積/体積比の増大は、触媒活性の飛躍的向上につながります。': 'The increase in surface area-to-volume ratio leads to a dramatic enhancement of catalytic activity.',
        '白金(Pt)触媒を例に考えてみましょう:': 'Let\'s consider platinum (Pt) catalyst as an example:',
        '用途': 'Applications',
        '燃料電池の電極触媒、自動車排ガス浄化触媒': 'Fuel cell electrode catalyst, automotive exhaust purification catalyst',
        '反応': 'Reaction',
        '水素酸化反応': 'Hydrogen oxidation reaction',
        '白金の粒子サイズと触媒活性の関係:': 'Relationship between platinum particle size and catalytic activity:',

        # Pt catalyst table
        'Pt粒子サイズ': 'Pt particle size',
        '表面積(g当たり)': 'Surface area (per g)',
        '相対触媒活性': 'Relative catalytic activity',
        'コスト効率': 'Cost efficiency',
        'バルク板': 'Bulk plate',
        '粉末': 'Powder',
        'ナノ粒子': 'Nanoparticles',

        '3 nmの白金ナノ粒子は、バルクの白金板と比べて1,500倍の触媒活性を示します。これは、同じ質量の白金から1,500倍の性能を引き出せることを意味し、希少金属の使用量削減に大きく貢献しています。': '3 nm platinum nanoparticles show 1,500 times the catalytic activity compared to bulk platinum plates. This means that 1,500 times the performance can be extracted from the same mass of platinum, greatly contributing to the reduction in the use of rare metals.',

        # Section 1.3
        '1.3 量子効果と量子閉じ込め': '1.3 Quantum Effects and Quantum Confinement',
        '量子効果の発現': 'Emergence of Quantum Effects',
        '粒子サイズがナノスケールになると、古典物理学では説明できない量子力学的効果が顕著になります。': 'When particle size becomes nanoscale, quantum mechanical effects that cannot be explained by classical physics become prominent.',
        '量子効果を理解する鍵は、de Broglie(ド・ブロイ)波長です。すべての粒子は波としての性質を持ち、その波長': 'The key to understanding quantum effects is the de Broglie wavelength. All particles have wave properties, and the wavelength',
        'は以下の式で与えられます:': 'is given by the following equation:',
        'プランク定数': 'Planck constant',
        '運動量(質量 × 速度)': 'momentum (mass × velocity)',
        '粒子の質量': 'particle mass',
        '粒子の速度': 'particle velocity',

        '室温(300 K)での電子のde Broglie波長を計算してみましょう:': 'Let\'s calculate the de Broglie wavelength of electrons at room temperature (300 K):',
        '電子の熱運動エネルギー': 'Thermal kinetic energy of electron',
        '電子の質量': 'Electron mass',
        '速度': 'Velocity',
        'de Broglie波長': 'de Broglie wavelength',

        '電子のde Broglie波長は約6 nm程度です。粒子サイズがこの波長と同程度か、それより小さくなると、電子は粒子の中に「閉じ込められた波」として振る舞い、量子効果が重要になります。': 'The de Broglie wavelength of electrons is about 6 nm. When particle size becomes comparable to or smaller than this wavelength, electrons behave as "confined waves" within the particle, and quantum effects become important.',

        '量子閉じ込め効果': 'Quantum Confinement Effect',
        '量子閉じ込め効果(Quantum confinement effect)とは、電子や正孔(ホール)が狭い空間に閉じ込められることで、そのエネルギー状態が離散的になる現象です。': 'The quantum confinement effect refers to the phenomenon where the energy states of electrons or holes (positive charge carriers) become discrete when they are confined in a narrow space.',
        '最も単純なモデルとして、1次元無限井戸型ポテンシャルを考えましょう。長さ': 'As the simplest model, let\'s consider a one-dimensional infinite potential well. The energy levels of a particle confined in a box of length',
        'の箱の中に閉じ込められた粒子のエネルギー準位は:': 'are:',
        '量子数': 'quantum number',
        '箱の長さ(粒子サイズ)': 'box length (particle size)',

        'この式から重要な結論が得られます:': 'From this equation, we obtain important conclusions:',
        'エネルギーは離散的': 'Energy is discrete',
        '連続的な値ではなく、特定の値($E_1, E_2, E_3, \\ldots$)のみ許される': 'Not continuous values, only specific values are allowed',
        '最低エネルギー(基底状態)が存在': 'A minimum energy (ground state) exists',
        'であり、ゼロではない': 'and is not zero',
        'エネルギーギャップはサイズに依存': 'Energy gap depends on size',
        '粒子サイズが小さくなるほど、エネルギーギャップが大きくなります。': 'As particle size decreases, the energy gap increases.',
        'これが、半導体ナノ粒子(量子ドット)でサイズによって色が変わる理由です。': 'This is why the color of semiconductor nanoparticles (quantum dots) changes with size.',

        '半導体量子ドットの発光色制御': 'Emission Color Control in Semiconductor Quantum Dots',
        '量子ドット(Quantum dots, QDs)は、半導体ナノ粒子で、サイズによってバンドギャップ(禁制帯幅)が変化し、発光色を制御できます。': 'Quantum dots (QDs) are semiconductor nanoparticles in which the band gap changes with size, allowing control of emission color.',
        'CdSe(セレン化カドミウム)量子ドットの例:': 'Example of CdSe (cadmium selenide) quantum dots:',

        # QD table
        'バンドギャップ': 'Band gap',
        '発光色': 'Emission color',
        '発光波長': 'Emission wavelength',
        '応用例': 'Application examples',
        '赤外': 'Infrared',
        'オレンジ': 'Orange',
        'ディスプレイ': 'Display',
        '黄緑色': 'Yellow-green',
        'バイオイメージング': 'Bioimaging',
        '緑色': 'Green',
        '青色': 'Blue',

        '粒子直径が10 nmから2 nmへ小さくなると、バンドギャップが1.85 eVから2.75 eVへ増大し、発光色が赤色から青色へ変化します。': 'As particle diameter decreases from 10 nm to 2 nm, the band gap increases from 1.85 eV to 2.75 eV, and the emission color changes from red to blue.',
        'これはBrus方程式(最も単純な近似形)で説明できます:': 'This can be explained by the Brus equation (in its simplest approximation):',

        '半径': 'radius',
        'の量子ドットのバンドギャップ': 'band gap of quantum dot with',
        'バルク半導体のバンドギャップ': 'band gap of bulk semiconductor',
        '電子と正孔の有効質量': 'effective masses of electron and hole',
        '電子の電荷': 'electron charge',
        '誘電率': 'dielectric constant',
        '第2項: 量子閉じ込めによるエネルギー増大': 'Second term: energy increase due to quantum confinement',
        '第3項: クーロン相互作用によるエネルギー減少': 'Third term: energy decrease due to Coulomb interaction',

        '量子ドットの主要な応用:': 'Major applications of quantum dots:',
        'QLED(量子ドットLEDディスプレイ)': 'QLED (quantum dot LED display)',
        'サムスン、ソニーなどが製品化、色再現性が従来比150%向上': 'Commercialized by Samsung, Sony, etc., with 150% improvement in color reproduction compared to conventional displays',
        'バイオイメージング': 'Bioimaging',
        '蛍光色素より明るく、光退色しにくい': 'Brighter than fluorescent dyes and more resistant to photobleaching',
        '太陽電池': 'Solar cells',
        '多接合型太陽電池で理論効率向上(Shockley-Queisser限界を超える可能性)': 'Theoretical efficiency improvement in multi-junction solar cells (potential to exceed the Shockley-Queisser limit)',
        '量子情報技術': 'Quantum information technology',
        '量子ビットの候補材料': 'Candidate material for qubits',

        '金属ナノ粒子の局在表面プラズモン共鳴': 'Localized Surface Plasmon Resonance in Metal Nanoparticles',
        '金属ナノ粒子では、局在表面プラズモン共鳴(Localized Surface Plasmon Resonance, LSPR)という特異な光学現象が現れます。': 'In metal nanoparticles, a unique optical phenomenon called localized surface plasmon resonance (LSPR) appears.',
        'プラズモンとは、金属中の自由電子の集団振動です。ナノ粒子では、光の電場によって電子雲が振動し、特定の波長で共鳴が起こります。': 'Plasmons are collective oscillations of free electrons in metals. In nanoparticles, the electron cloud oscillates due to the electric field of light, and resonance occurs at specific wavelengths.',
        '金ナノ粒子のLSPR:': 'LSPR of gold nanoparticles:',

        # LSPR table
        '粒子サイズ・形状': 'Particle size/shape',
        'LSPR波長': 'LSPR wavelength',
        '観察される色': 'Observed color',
        '応用': 'Applications',
        '球形': 'Spherical',
        'バイオセンシング': 'Biosensing',
        '赤紫色': 'Red-purple',
        '光熱療法': 'Photothermal therapy',
        'SERS基板': 'SERS substrate',
        'ナノロッド(縦横比3:1)': 'Nanorod (aspect ratio 3:1)',
        '青緑色': 'Blue-green',
        'イメージング': 'Imaging',
        'ナノシェル(Au/SiO₂)': 'Nanoshell (Au/SiO₂)',
        '透明(近赤外)': 'Transparent (near-infrared)',
        'がん温熱療法': 'Cancer thermal therapy',

        'LSPRの応用例:': 'Application examples of LSPR:',
        '抗体を金ナノ粒子に修飾し、標的分子結合でLSPR波長がシフト(検出限界: pMオーダー)': 'Antibodies are modified on gold nanoparticles, and the LSPR wavelength shifts upon target molecule binding (detection limit: pM order)',
        '表面増強ラマン散乱(SERS)': 'Surface-enhanced Raman scattering (SERS)',
        'ラマン信号が10⁶〜10¹⁴倍増強、単分子検出も可能': 'Raman signal enhanced 10⁶ to 10¹⁴ times, enabling single-molecule detection',
        '近赤外光(生体透過性が高い)で金ナノ粒子を加熱し、がん細胞を選択的に死滅': 'Gold nanoparticles are heated with near-infrared light (high biological transparency) to selectively kill cancer cells',
        'カラーフィルター': 'Color filters',
        'LSPR波長を制御したプラズモニックカラーフィルター': 'Plasmonic color filters with controlled LSPR wavelength',

        # Section 1.4
        '1.4 ナノ材料の分類': '1.4 Classification of Nanomaterials',
        'ナノ材料は、何次元がナノサイズかによって分類されます。': 'Nanomaterials are classified according to how many dimensions are at the nanoscale.',
        '次元別分類': 'Classification by Dimensionality',

        # Flowchart labels
        '0次元': '0-dimensional',
        '1次元': '1-dimensional',
        '2次元': '2-dimensional',
        '3次元': '3-dimensional',
        'ナノ粒子': 'Nanoparticles',
        '量子ドット': 'Quantum Dots',
        'フラーレン': 'Fullerenes',
        'カーボンナノチューブ': 'Carbon Nanotubes',
        'ナノワイヤー': 'Nanowires',
        'ナノファイバー': 'Nanofibers',
        'グラフェン': 'Graphene',
        '遷移金属ダイカルコゲナイド': 'Transition Metal Dichalcogenides',
        'ナノシート': 'Nanosheets',
        'ナノ多孔体': 'Nanoporous materials',
        'ナノコンポジット': 'Nanocomposites',
        'ナノ結晶材料': 'Nanocrystalline materials',

        '分類の基準': 'Classification Criteria',
        '0次元(0D)': '0-dimensional (0D)',
        '3次元すべてがナノサイズ(長さ、幅、高さすべて < 100 nm)': 'All three dimensions are nano-sized (length, width, height all < 100 nm)',
        '1次元(1D)': '1-dimensional (1D)',
        '2次元がナノサイズ、1次元は長い(直径 < 100 nm、長さは任意)': 'Two dimensions are nano-sized, one dimension is long (diameter < 100 nm, length arbitrary)',
        '2次元(2D)': '2-dimensional (2D)',
        '1次元がナノサイズ、2次元は広がりを持つ(厚さ < 100 nm、長さ・幅は任意)': 'One dimension is nano-sized, two dimensions have extension (thickness < 100 nm, length and width arbitrary)',
        '3次元(3D)': '3-dimensional (3D)',
        'バルク材料だがナノ構造を内部に持つ(ナノ細孔、ナノ結晶粒など)': 'Bulk material but has nanostructures inside (nanopores, nanocrystalline grains, etc.)',
    }

    # Apply translations
    for jp, en in translations.items():
        content = content.replace(jp, en)

    # Write the translated content
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Translation complete: {target_path}")
    return target_path

if __name__ == "__main__":
    translate_nm_chapter1()
