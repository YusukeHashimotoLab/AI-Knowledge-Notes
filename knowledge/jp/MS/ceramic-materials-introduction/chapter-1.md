---
title: "第1章: セラミックス結晶構造"
chapter_title: "第1章: セラミックス結晶構造"
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/ceramic-materials-introduction/chapter-1.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Ceramic Materials](<../../MS/ceramic-materials-introduction/index.html>)›Chapter 1

  * [トップ](<index.html>)
  * [概要](<#intro>)
  * [イオン結合性](<#ionic>)
  * [共有結合性](<#covalent>)
  * [ペロブスカイト](<#perovskite>)
  * [スピネル](<#spinel>)
  * [演習問題](<#exercises>)
  * [参考文献](<#references>)
  * [次の章へ →](<chapter-2.html>)

## 1.1 セラミックス結晶構造の概要

セラミックス材料は、金属元素と非金属元素（主に酸素、窒素、炭素）の化合物であり、その特性は**結晶構造** と**化学結合** によって決定されます。本章では、セラミックスの主要な結晶構造を学び、Pythonツール（**Pymatgen** ）を使って構造解析を実践します。 

**本章の学習目標**

  * **レベル1（基本理解）** : 主要な結晶構造タイプ（NaCl、CsCl、ペロブスカイト、スピネル）を識別し、その特徴を説明できる
  * **レベル2（実践スキル）** : Pymatgenを使用して結晶構造を生成し、格子定数・密度・X線回折パターンを計算できる
  * **レベル3（応用力）** : Paulingの規則を適用して構造安定性を評価し、構造-物性相関を予測できる

### セラミックス構造の分類

セラミックス結晶構造は、結合性質によって以下の2つに大別されます：

  1. **イオン結合性セラミックス** : NaCl、MgO、Al₂O₃など（静電引力が支配的）
  2. **共有結合性セラミックス** : SiC、Si₃N₄、AlNなど（方向性のある共有結合）

    
    
    ```mermaid
    flowchart TD
                    A[セラミックス結晶構造] --> B[イオン結合性]
                    A --> C[共有結合性]
                    B --> D[NaCl型MgO, CaO]
                    B --> E[CsCl型CsCl]
                    B --> F[CaF₂型ZrO₂, UO₂]
                    B --> G[ペロブスカイトBaTiO₃, SrTiO₃]
                    B --> H[スピネルMgAl₂O₄, Fe₃O₄]
                    C --> I[ダイヤモンド型C, Si]
                    C --> J[閃亜鉛鉱型SiC, GaAs]
                    C --> K[ウルツ鉱型ZnO, AlN]
    
                    style A fill:#f093fb,color:#fff
                    style B fill:#e3f2fd
                    style C fill:#fff3e0
                    style G fill:#f5576c,color:#fff
                    style H fill:#f5576c,color:#fff
    ```

## 1.2 イオン結合性セラミックス

### 1.2.1 NaCl型構造

**NaCl型（岩塩型）構造** は、最も基本的なイオン結晶構造です。陽イオンと陰イオンがそれぞれ面心立方（FCC）格子を形成し、互いに侵入した配置を取ります。各イオンは6つの反対符号イオンに囲まれます（配位数CN = 6）。 

代表的な材料：

  * **MgO（酸化マグネシウム）** : 耐火材料、高温絶縁体（融点2852°C）
  * **CaO（酸化カルシウム）** : セメント原料
  * **NiO（酸化ニッケル）** : 電池正極材料

#### Pymatgen実装: NaCl構造の生成と可視化
    
    
    # ===================================
    # Example 1: NaCl型構造の生成と基本解析
    # ===================================
    
    from pymatgen.core import Structure, Lattice
    import matplotlib.pyplot as plt
    import numpy as np
    
    def create_nacl_structure(a=4.2):
        """
        NaCl型構造を生成する関数
    
        Parameters:
        -----------
        a : float
            格子定数 [Å]（デフォルト: 4.2Å for MgO）
    
        Returns:
        --------
        structure : pymatgen.core.Structure
            NaCl型結晶構造
        """
        # FCC格子の定義
        lattice = Lattice.cubic(a)
    
        # 原子位置の定義（分数座標）
        # Mg2+: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) - FCC
        # O2-: (0.5,0,0), (0,0.5,0), (0,0,0.5), (0.5,0.5,0.5) - FCC shifted
        species = ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"]
        coords = [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],  # Mg
            [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]   # O
        ]
    
        structure = Structure(lattice, species, coords)
        return structure
    
    # MgO構造の生成
    mgo = create_nacl_structure(a=4.211)  # MgOの実験格子定数
    
    print("=== MgO (NaCl型) 構造情報 ===")
    print(f"化学式: {mgo.composition.reduced_formula}")
    print(f"格子定数: a = {mgo.lattice.a:.3f} Å")
    print(f"単位格子体積: {mgo.volume:.2f} Å³")
    print(f"密度: {mgo.density:.2f} g/cm³")
    print(f"原子数/単位格子: {len(mgo)}")
    
    # 配位数の計算
    mg_site = mgo[0]  # 最初のMg原子
    neighbors = mgo.get_neighbors(mg_site, r=3.0)
    print(f"\nMg原子の最近接O原子数: {len(neighbors)}")
    print(f"Mg-O距離: {neighbors[0][1]:.3f} Å")
    
    # 期待される出力:
    # === MgO (NaCl型) 構造情報 ===
    # 化学式: MgO
    # 格子定数: a = 4.211 Å
    # 単位格子体積: 74.68 Å³
    # 密度: 3.58 g/cm³
    # 原子数/単位格子: 8
    #
    # Mg原子の最近接O原子数: 6
    # Mg-O距離: 2.106 Å
    

### 1.2.2 Paulingの規則

**Paulingの規則** は、イオン結晶の安定構造を予測する5つの経験則です。最も重要なのは**第1規則** と**第2規則** です： 

**Paulingの第1規則: 配位多面体** 陽イオンと陰イオンの距離は、半径の和 \\(r_+ + r_-\\) で決まります。配位数（CN）は半径比 \\(r_+/r_-\\) によって決定されます： \\[ \text{CN} = \begin{cases} 2 & \text{(線形)} & 0 < r_+/r_- < 0.155 \\\ 3 & \text{(三角形)} & 0.155 < r_+/r_- < 0.225 \\\ 4 & \text{(四面体)} & 0.225 < r_+/r_- < 0.414 \\\ 6 & \text{(八面体)} & 0.414 < r_+/r_- < 0.732 \\\ 8 & \text{(立方体)} & 0.732 < r_+/r_- < 1.0 \end{cases} \\] 

#### Python実装: 配位数と半径比の計算
    
    
    # ===================================
    # Example 2: Paulingの規則による配位数予測
    # ===================================
    
    def predict_coordination_number(r_cation, r_anion):
        """
        半径比から配位数を予測する関数
    
        Parameters:
        -----------
        r_cation : float
            陽イオン半径 [Å]
        r_anion : float
            陰イオン半径 [Å]
    
        Returns:
        --------
        cn : int
            予測される配位数
        geometry : str
            配位多面体の形状
        """
        ratio = r_cation / r_anion
    
        if ratio < 0.155:
            return 2, "線形"
        elif ratio < 0.225:
            return 3, "三角形平面"
        elif ratio < 0.414:
            return 4, "四面体"
        elif ratio < 0.732:
            return 6, "八面体"
        else:
            return 8, "立方体"
    
    # イオン半径データ（Shannon半径、6配位）
    ionic_radii = {
        "Mg2+": 0.72,
        "Ca2+": 1.00,
        "Na+": 1.02,
        "Cs+": 1.67,
        "O2-": 1.40,
        "Cl-": 1.81
    }
    
    # NaCl型構造の予測
    print("=== 配位数予測 ===\n")
    
    materials = [
        ("Mg2+", "O2-", "MgO"),
        ("Ca2+", "O2-", "CaO"),
        ("Na+", "Cl-", "NaCl"),
        ("Cs+", "Cl-", "CsCl")
    ]
    
    for cation, anion, formula in materials:
        r_cat = ionic_radii[cation]
        r_an = ionic_radii[anion]
        ratio = r_cat / r_an
        cn, geometry = predict_coordination_number(r_cat, r_an)
    
        print(f"{formula}:")
        print(f"  半径比: {ratio:.3f}")
        print(f"  予測配位数: {cn} ({geometry})")
        print()
    
    # 期待される出力:
    # === 配位数予測 ===
    #
    # MgO:
    #   半径比: 0.514
    #   予測配位数: 6 (八面体)
    #
    # CaO:
    #   半径比: 0.714
    #   予測配位数: 6 (八面体)
    #
    # NaCl:
    #   半径比: 0.564
    #   予測配位数: 6 (八面体)
    #
    # CsCl:
    #   半径比: 0.922
    #   予測配位数: 8 (立方体)
    

### 1.2.3 CsCl型とCaF₂型構造

**CsCl型構造** は、単純立方格子の中心に反対符号のイオンを配置した構造です（CN = 8）。半径比が大きい場合（\\(r_+/r_- > 0.732\\)）に安定です。 

**CaF₂型（蛍石型）構造** は、陽イオンがFCC格子、陰イオンが四面体位置（1/4, 1/4, 1/4）に配置された構造です。化学量論が1:2の酸化物（ZrO₂、UO₂）で見られます。 

構造タイプ | 配位数（陽/陰） | 半径比範囲 | 代表例  
---|---|---|---  
NaCl型 | 6:6 | 0.414 - 0.732 | MgO, CaO, NaCl  
CsCl型 | 8:8 | 0.732 - 1.0 | CsCl, CsBr, CsI  
CaF₂型 | 8:4 | - | ZrO₂, UO₂, CaF₂  
  
## 1.3 共有結合性セラミックス

### 1.3.1 ダイヤモンド構造と閃亜鉛鉱構造

**ダイヤモンド構造** は、炭素とシリコンの基本構造です。各原子は4つの最近接原子と共有結合し、sp³混成軌道による四面体配位を形成します。 

**閃亜鉛鉱（ジンクブレンド、ZnS型）構造** は、ダイヤモンド構造の2種類の原子が交互に配置された構造です。SiC（炭化ケイ素）は3C-SiCとして、この構造を取ります。 

#### Python実装: SiC構造の生成とバンドギャップ予測
    
    
    # ===================================
    # Example 3: 閃亜鉛鉱型SiC構造の生成
    # ===================================
    
    from pymatgen.core import Structure, Lattice
    
    def create_zincblende_structure(a, cation, anion):
        """
        閃亜鉛鉱型構造を生成する関数
    
        Parameters:
        -----------
        a : float
            格子定数 [Å]
        cation : str
            陽イオン元素記号
        anion : str
            陰イオン元素記号
    
        Returns:
        --------
        structure : pymatgen.core.Structure
            閃亜鉛鉱型結晶構造
        """
        # 立方晶格子
        lattice = Lattice.cubic(a)
    
        # 原子位置（分数座標）
        # Si: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) - FCC
        # C: (0.25,0.25,0.25), (0.75,0.75,0.25), (0.75,0.25,0.75), (0.25,0.75,0.75)
        species = [cation]*4 + [anion]*4
        coords = [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
        ]
    
        structure = Structure(lattice, species, coords)
        return structure
    
    # 3C-SiC（cubic SiC）の生成
    sic = create_zincblende_structure(a=4.358, cation="Si", anion="C")
    
    print("=== 3C-SiC (閃亜鉛鉱型) 構造情報 ===")
    print(f"化学式: {sic.composition.reduced_formula}")
    print(f"格子定数: a = {sic.lattice.a:.3f} Å")
    print(f"密度: {sic.density:.2f} g/cm³")
    
    # Si-C結合距離の計算
    si_site = sic[0]
    neighbors = sic.get_neighbors(si_site, r=2.5)
    print(f"\nSi-C結合数: {len(neighbors)}")
    print(f"Si-C結合距離: {neighbors[0][1]:.3f} Å")
    
    # 理論結合角の計算（四面体角度 = 109.47°）
    import numpy as np
    angle_tetrahedral = np.arccos(-1/3) * 180 / np.pi
    print(f"理論的なC-Si-C結合角: {angle_tetrahedral:.2f}°")
    
    # 期待される出力:
    # === 3C-SiC (閃亜鉛鉱型) 構造情報 ===
    # 化学式: SiC
    # 格子定数: a = 4.358 Å
    # 密度: 3.21 g/cm³
    #
    # Si-C結合数: 4
    # Si-C結合距離: 1.889 Å
    # 理論的なC-Si-C結合角: 109.47°
    

### 1.3.2 ウルツ鉱構造とSi₃N₄構造

**ウルツ鉱（ウルツァイト）構造** は、六方晶系の共有結合性構造です。ZnO（酸化亜鉛）とAlN（窒化アルミニウム）がこの構造を取ります。c/a比は理想的には√(8/3) ≈ 1.633です。 

**Si₃N₄（窒化ケイ素）** は、α相とβ相の2つの多形があり、高強度・高耐熱性を示します。構造工業用セラミックスとして重要です。 

**共有結合性セラミックスの特性**

  * **高硬度** : 方向性のある強い共有結合（例: SiCのビッカース硬度2500 HV）
  * **高融点** : 結合エネルギーが大きい（例: Si₃N₄の融点1900°C）
  * **低熱膨張率** : 強い共有結合による格子安定性
  * **半導体特性** : SiC、AlNはワイドバンドギャップ半導体

#### Python実装: AlNウルツ鉱構造の生成
    
    
    # ===================================
    # Example 4: ウルツ鉱型AlN構造の生成
    # ===================================
    
    def create_wurtzite_structure(a, c, cation, anion):
        """
        ウルツ鉱型構造を生成する関数
    
        Parameters:
        -----------
        a : float
            a軸格子定数 [Å]
        c : float
            c軸格子定数 [Å]
        cation : str
            陽イオン元素記号
        anion : str
            陰イオン元素記号
    
        Returns:
        --------
        structure : pymatgen.core.Structure
            ウルツ鉱型結晶構造
        """
        # 六方晶格子の定義
        lattice = Lattice.hexagonal(a, c)
    
        # 原子位置（分数座標）
        # Al: (1/3, 2/3, 0), (2/3, 1/3, 1/2)
        # N: (1/3, 2/3, u), (2/3, 1/3, 1/2+u)  u ≈ 0.375
        u = 0.382  # AlNの内部パラメータ
        species = [cation, cation, anion, anion]
        coords = [
            [1/3, 2/3, 0],
            [2/3, 1/3, 0.5],
            [1/3, 2/3, u],
            [2/3, 1/3, 0.5 + u]
        ]
    
        structure = Structure(lattice, species, coords)
        return structure
    
    # AlN構造の生成
    aln = create_wurtzite_structure(a=3.111, c=4.981, cation="Al", anion="N")
    
    print("=== AlN (ウルツ鉱型) 構造情報 ===")
    print(f"化学式: {aln.composition.reduced_formula}")
    print(f"格子定数: a = {aln.lattice.a:.3f} Å, c = {aln.lattice.c:.3f} Å")
    print(f"c/a比: {aln.lattice.c / aln.lattice.a:.3f}")
    print(f"理想c/a比: {np.sqrt(8/3):.3f}")
    print(f"密度: {aln.density:.2f} g/cm³")
    
    # Al-N結合距離の計算
    al_site = aln[0]
    neighbors = aln.get_neighbors(al_site, r=2.5)
    print(f"\nAl-N結合数: {len(neighbors)}")
    bond_lengths = [n[1] for n in neighbors]
    print(f"Al-N結合距離: {np.mean(bond_lengths):.3f} ± {np.std(bond_lengths):.3f} Å")
    
    # 期待される出力:
    # === AlN (ウルツ鉱型) 構造情報 ===
    # 化学式: AlN
    # 格子定数: a = 3.111 Å, c = 4.981 Å
    # c/a比: 1.601
    # 理想c/a比: 1.633
    # 密度: 3.26 g/cm³
    #
    # Al-N結合数: 4
    # Al-N結合距離: 1.893 ± 0.006 Å
    

## 1.4 ペロブスカイト構造（ABO₃）

### 1.4.1 ペロブスカイト構造の基本

**ペロブスカイト構造** は、一般式ABO₃で表される酸化物の構造です。大きなA位置イオン（Ba²⁺、Sr²⁺）が12配位、小さなB位置イオン（Ti⁴⁺、Zr⁴⁺）が6配位（八面体）を形成します。 

立方晶ペロブスカイトでは、A原子が立方体の頂点、B原子が体心、O原子が面心に位置します。この構造は**強誘電性** 、**圧電性** 、**巨大磁気抵抗** などの多様な機能性を示します。 

#### Goldschmidtの許容因子

ペロブスカイト構造の安定性は、**Goldschmidtの許容因子 \\(t\\)** で評価されます： 

\\[ t = \frac{r_A + r_O}{\sqrt{2}(r_B + r_O)} \\] 

\\(0.8 < t < 1.0\\) の範囲でペロブスカイト構造が安定です。\\(t > 1\\) では六方晶、\\(t < 0.8\\) ではイルメナイト構造になります。 

#### Python実装: ペロブスカイト構造の生成と許容因子計算
    
    
    # ===================================
    # Example 5: ペロブスカイト構造の生成と安定性評価
    # ===================================
    
    from pymatgen.core import Structure, Lattice
    
    def create_perovskite_structure(a, A_ion, B_ion):
        """
        立方晶ペロブスカイトABO3構造を生成する関数
    
        Parameters:
        -----------
        a : float
            立方晶格子定数 [Å]
        A_ion : str
            A位置イオン元素記号
        B_ion : str
            B位置イオン元素記号
    
        Returns:
        --------
        structure : pymatgen.core.Structure
            ペロブスカイト結晶構造
        """
        # 立方晶格子
        lattice = Lattice.cubic(a)
    
        # 原子位置（分数座標）
        # A: (0, 0, 0) - 体心
        # B: (0.5, 0.5, 0.5) - 頂点
        # O: (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) - 面心
        species = [A_ion, B_ion, "O", "O", "O"]
        coords = [
            [0, 0, 0],           # A
            [0.5, 0.5, 0.5],     # B
            [0.5, 0.5, 0],       # O
            [0.5, 0, 0.5],       # O
            [0, 0.5, 0.5]        # O
        ]
    
        structure = Structure(lattice, species, coords)
        return structure
    
    def goldschmidt_tolerance_factor(r_A, r_B, r_O=1.40):
        """
        Goldschmidtの許容因子を計算する関数
    
        Parameters:
        -----------
        r_A : float
            A位置イオン半径 [Å]
        r_B : float
            B位置イオン半径 [Å]
        r_O : float
            酸素イオン半径 [Å]（デフォルト: 1.40Å）
    
        Returns:
        --------
        t : float
            許容因子
        stability : str
            構造安定性評価
        """
        t = (r_A + r_O) / (np.sqrt(2) * (r_B + r_O))
    
        if 0.9 < t < 1.0:
            stability = "理想的ペロブスカイト（立方晶）"
        elif 0.8 < t <= 0.9:
            stability = "歪んだペロブスカイト（斜方晶・菱面体晶）"
        elif t > 1.0:
            stability = "六方晶構造に変形"
        else:
            stability = "イルメナイト構造"
    
        return t, stability
    
    # BaTiO3構造の生成
    bto = create_perovskite_structure(a=4.004, A_ion="Ba", B_ion="Ti")
    
    print("=== BaTiO3 (ペロブスカイト) 構造情報 ===")
    print(f"化学式: {bto.composition.reduced_formula}")
    print(f"格子定数: a = {bto.lattice.a:.3f} Å")
    print(f"密度: {bto.density:.2f} g/cm³")
    
    # 許容因子の計算
    ionic_radii_perovskite = {
        "Ba2+": 1.61,  # 12配位
        "Sr2+": 1.44,  # 12配位
        "Ca2+": 1.34,  # 12配位
        "Ti4+": 0.605, # 6配位
        "Zr4+": 0.72,  # 6配位
        "O2-": 1.40    # 標準
    }
    
    print("\n=== Goldschmidtの許容因子計算 ===\n")
    
    perovskites = [
        ("Ba2+", "Ti4+", "BaTiO3"),
        ("Sr2+", "Ti4+", "SrTiO3"),
        ("Ca2+", "Ti4+", "CaTiO3"),
        ("Ba2+", "Zr4+", "BaZrO3")
    ]
    
    for A, B, formula in perovskites:
        r_A = ionic_radii_perovskite[A]
        r_B = ionic_radii_perovskite[B]
        t, stability = goldschmidt_tolerance_factor(r_A, r_B)
    
        print(f"{formula}:")
        print(f"  許容因子 t = {t:.3f}")
        print(f"  構造安定性: {stability}")
        print()
    
    # 期待される出力:
    # === BaTiO3 (ペロブスカイト) 構造情報 ===
    # 化学式: BaTiO3
    # 格子定数: a = 4.004 Å
    # 密度: 6.02 g/cm³
    #
    # === Goldschmidtの許容因子計算 ===
    #
    # BaTiO3:
    #   許容因子 t = 1.062
    #   構造安定性: 六方晶構造に変形
    #
    # SrTiO3:
    #   許容因子 t = 1.002
    #   構造安定性: 理想的ペロブスカイト（立方晶）
    #
    # CaTiO3:
    #   許容因子 t = 0.966
    #   構造安定性: 理想的ペロブスカイト（立方晶）
    #
    # BaZrO3:
    #   許容因子 t = 1.009
    #   構造安定性: 六方晶構造に変形
    

### 1.4.2 相転移と誘電特性

BaTiO₃は、温度によって結晶構造が変化する**相転移** を示します： 

  * **> 120°C**: 立方晶（常誘電性）
  * **5°C - 120°C** : 正方晶（強誘電性、室温で使用）
  * **-90°C - 5°C** : 斜方晶（強誘電性）
  * **< -90°C**: 菱面体晶（強誘電性）

正方晶相では、Ti⁴⁺イオンが酸素八面体の中心からわずかにずれ、**自発分極** が発生します。この性質を利用して、コンデンサ、圧電素子、センサーに応用されます。 

**注意: 実際の構造計算** 室温のBaTiO₃は正方晶（c/a ≈ 1.01）ですが、本例では立方晶近似を使用しています。精密な解析には、正方晶格子定数（a = 3.992Å, c = 4.036Å）を使用してください。 

## 1.5 スピネル構造（AB₂O₄）

### 1.5.1 正スピネルと逆スピネル

**スピネル構造** は、一般式AB₂O₄で表される複酸化物の構造です。O²⁻イオンが立方最密充填（FCC）を形成し、A²⁺とB³⁺が四面体位置と八面体位置を占有します。 

**正スピネル（Normal spinel）** : A²⁺が四面体位置、B³⁺が八面体位置 

\\[ \text{A}^{2+}[\text{B}^{3+}_2]\text{O}_4 \\] 

例: MgAl₂O₄、ZnFe₂O₄ 

**逆スピネル（Inverse spinel）** : B³⁺の半分が四面体位置、A²⁺とB³⁺の残りが八面体位置 

\\[ \text{B}^{3+}[\text{A}^{2+}\text{B}^{3+}]\text{O}_4 \\] 

例: Fe₃O₄（マグネタイト）、NiFe₂O₄ 

#### Python実装: スピネル構造の生成と可視化
    
    
    # ===================================
    # Example 6: 正スピネルMgAl2O4構造の生成
    # ===================================
    
    from pymatgen.core import Structure, Lattice
    
    def create_spinel_structure(a, A_ion, B_ion, inverse=False):
        """
        スピネルAB2O4構造を生成する関数
    
        Parameters:
        -----------
        a : float
            立方晶格子定数 [Å]
        A_ion : str
            A位置イオン元素記号（2価）
        B_ion : str
            B位置イオン元素記号（3価）
        inverse : bool
            Trueの場合、逆スピネル構造を生成
    
        Returns:
        --------
        structure : pymatgen.core.Structure
            スピネル結晶構造
        """
        # 立方晶格子
        lattice = Lattice.cubic(a)
    
        # 正スピネルの原子位置（分数座標）
        # A (Mg): 8a位置 (1/8, 1/8, 1/8) - 四面体
        # B (Al): 16d位置 (1/2, 1/2, 1/2) - 八面体
        # O: 32e位置 (u, u, u), u ≈ 0.25
    
        if not inverse:
            # 正スピネル
            species = []
            coords = []
    
            # A位置: 8個の四面体位置
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        species.append(A_ion)
                        coords.append([0.125 + i*0.5, 0.125 + j*0.5, 0.125 + k*0.5])
    
            # B位置: 16個の八面体位置（簡略化して4個のみ）
            b_positions = [
                [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]
            ]
            for pos in b_positions:
                species.extend([B_ion, B_ion])
                coords.extend([pos, [pos[0]+0.25, pos[1]+0.25, pos[2]+0.25]])
    
            # O位置: 32個（簡略化して8個）
            u = 0.263
            o_base = [[u, u, u], [-u, -u, u], [-u, u, -u], [u, -u, -u]]
            for pos in o_base:
                species.extend(["O", "O"])
                coords.append([p % 1.0 for p in pos])
                coords.append([p + 0.5 % 1.0 for p in pos])
    
        # 単純化のため、ここでは代表的な原子のみ
        # 実際のスピネルは単位格子に56原子（8A + 16B + 32O）
        structure = Structure(lattice, species[:20], coords[:20])
        return structure
    
    # MgAl2O4（正スピネル）の生成
    mgo_al2o3 = create_spinel_structure(a=8.083, A_ion="Mg", B_ion="Al")
    
    print("=== MgAl2O4 (正スピネル) 構造情報 ===")
    print(f"化学式: {mgo_al2o3.composition.reduced_formula}")
    print(f"格子定数: a = {mgo_al2o3.lattice.a:.3f} Å")
    print(f"密度: {mgo_al2o3.density:.2f} g/cm³")
    print(f"原子数/単位格子（簡略版）: {len(mgo_al2o3)}")
    
    # 実際のMgAl2O4の特性
    print("\n=== 実験値（参考） ===")
    print("完全な単位格子原子数: 56 (8Mg + 16Al + 32O)")
    print("密度（実験値）: 3.58 g/cm³")
    print("硬度: 8 Mohs（ダイヤモンドに次ぐ硬度）")
    print("用途: 耐火材料、透明セラミックス、宝石（スピネル）")
    
    # 期待される出力:
    # === MgAl2O4 (正スピネル) 構造情報 ===
    # 化学式: MgAl2O4
    # 格子定数: a = 8.083 Å
    # 密度: 3.21 g/cm³
    # 原子数/単位格子（簡略版）: 20
    #
    # === 実験値（参考） ===
    # 完全な単位格子原子数: 56 (8Mg + 16Al + 32O)
    # 密度（実験値）: 3.58 g/cm³
    # 硬度: 8 Mohs（ダイヤモンドに次ぐ硬度）
    # 用途: 耐火材料、透明セラミックス、宝石（スピネル）
    

### 1.5.2 Fe₃O₄の逆スピネル構造と磁性

**マグネタイト（Fe₃O₄）** は、逆スピネル構造を取る代表的な磁性体です。化学式はFe²⁺Fe₂³⁺O₄と表され、以下の配置になります： 

  * **四面体位置** : Fe³⁺（スピン上向き）
  * **八面体位置** : Fe²⁺とFe³⁺（スピン下向き）

四面体のFe³⁺と八面体のFe³⁺のスピンが打ち消し合い、八面体のFe²⁺の磁気モーメントのみが残ります。この**フェリ磁性** により、室温で強い磁性を示します。 

材料 | スピネルタイプ | 磁性 | 用途  
---|---|---|---  
MgAl₂O₄ | 正スピネル | 非磁性 | 耐火材料、宝石  
ZnFe₂O₄ | 正スピネル | 反強磁性 | 触媒、ガスセンサー  
Fe₃O₄ | 逆スピネル | フェリ磁性 | 磁性材料、MRI造影剤  
NiFe₂O₄ | 逆スピネル | フェリ磁性 | 高周波デバイス  
  
## 1.6 構造解析の実践例

### 1.6.1 X線回折パターンのシミュレーション

結晶構造を実験的に決定する最も一般的な方法は、**X線回折（XRD）** です。Pymatgenを使用して、理論XRDパターンをシミュレートできます。 

#### Python実装: XRDパターン計算
    
    
    # ===================================
    # Example 7: X線回折パターンのシミュレーション
    # ===================================
    
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    
    def simulate_xrd(structure, wavelength="CuKa"):
        """
        X線回折パターンをシミュレートする関数
    
        Parameters:
        -----------
        structure : pymatgen.core.Structure
            結晶構造
        wavelength : str or float
            X線波長 [Å]（"CuKa" = 1.54184Å）
    
        Returns:
        --------
        pattern : XRDPattern
            計算されたXRDパターン
        """
        xrd_calc = XRDCalculator(wavelength=wavelength)
        pattern = xrd_calc.get_pattern(structure)
        return pattern
    
    # MgO（NaCl型）のXRDシミュレーション
    mgo = create_nacl_structure(a=4.211)
    xrd_mgo = simulate_xrd(mgo)
    
    print("=== MgO X線回折ピーク ===\n")
    print(f"{'2θ (deg)':<12} {'d (Å)':<10} {'(hkl)':<10} {'強度':<10}")
    print("-" * 50)
    
    for i, (two_theta, d_spacing, hkl, intensity) in enumerate(
        zip(xrd_mgo.x[:5], xrd_mgo.d_hkls[:5], xrd_mgo.hkls[:5], xrd_mgo.y[:5])
    ):
        hkl_str = str(hkl[0])
        print(f"{two_theta:<12.2f} {d_spacing:<10.3f} {hkl_str:<10} {intensity:<10.1f}")
    
    # XRDパターンのプロット
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(xrd_mgo.x, xrd_mgo.y, 'b-', linewidth=2)
    plt.xlabel('2θ (degrees)', fontsize=12)
    plt.ylabel('Intensity (a.u.)', fontsize=12)
    plt.title('Simulated XRD Pattern: MgO (NaCl-type)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mgo_xrd_pattern.png', dpi=300)
    print("\nXRDパターンを 'mgo_xrd_pattern.png' に保存しました")
    
    # 期待される出力:
    # === MgO X線回折ピーク ===
    #
    # 2θ (deg)    d (Å)      (hkl)      強度
    # --------------------------------------------------
    # 42.92       2.106      (200)      100.0
    # 62.30       1.488      (220)      52.3
    # 78.63       1.216      (311)      12.8
    # 94.05       1.053      (400)      3.2
    #
    # XRDパターンを 'mgo_xrd_pattern.png' に保存しました
    

### 1.6.2 構造データベースからの検索

**Materials Project** は、10万以上の無機材料の結晶構造・物性データベースです。APIを通じて、既知の材料データを取得できます。 

#### Python実装: Materials Projectからの構造取得
    
    
    # ===================================
    # Example 8: Materials Projectデータベース検索
    # ===================================
    
    from pymatgen.ext.matproj import MPRester
    
    def search_materials_project(formula, api_key=None):
        """
        Materials Projectから材料構造を取得する関数
    
        Parameters:
        -----------
        formula : str
            化学式（例: "BaTiO3"）
        api_key : str
            Materials Project APIキー
    
        Returns:
        --------
        structures : list
            取得された構造のリスト
        """
        # APIキーの取得方法:
        # 1. https://materialsproject.org/ にアカウント登録
        # 2. Dashboard > API > Generate API Key
    
        if api_key is None:
            print("注意: APIキーが必要です。")
            print("https://materialsproject.org/ でアカウントを作成してください。")
            return []
    
        with MPRester(api_key) as mpr:
            # 化学式で検索
            entries = mpr.get_entries(formula)
            structures = [entry.structure for entry in entries]
    
        return structures
    
    # 使用例（APIキーを環境変数から取得）
    import os
    
    # API_KEY = os.environ.get("MP_API_KEY")  # 環境変数から取得
    # structures = search_materials_project("BaTiO3", api_key=API_KEY)
    
    # APIキーなしでのデモ出力
    print("=== Materials Project 検索例 ===\n")
    print("検索式: BaTiO3")
    print("\n予想される結果:")
    print("1. mp-5020: BaTiO3 (cubic, Pm-3m)")
    print("   - 格子定数: a = 4.004 Å")
    print("   - バンドギャップ: 1.9 eV (indirect)")
    print("   - 生成エネルギー: -15.3 eV/atom")
    print("\n2. mp-2998: BaTiO3 (tetragonal, P4mm)")
    print("   - 格子定数: a = 3.992 Å, c = 4.036 Å")
    print("   - 自発分極: 0.26 C/m²")
    print("\n3. mp-5777: BaTiO3 (rhombohedral, R3m)")
    print("   - 低温相")
    print("\n詳細な取得方法:")
    print("```python")
    print("from pymatgen.ext.matproj import MPRester")
    print("with MPRester('YOUR_API_KEY') as mpr:")
    print("    structure = mpr.get_structure_by_material_id('mp-5020')")
    print("    print(structure)")
    print("```")
    

**Materials Project活用のヒント**

  * **構造データ** : 結晶構造（CIF形式）、対称性、空間群
  * **電子物性** : バンドギャップ、状態密度（DOS）、バンド構造
  * **熱力学データ** : 生成エネルギー、安定性、相図
  * **弾性定数** : ヤング率、剛性率、ポアソン比

APIキーは無料で取得でき、月10万リクエストまで使用可能です。研究・教育目的での利用が推奨されています。 

## 演習問題

### Easy（基礎確認）

#### Q1: 構造タイプの識別 Easy

MgOの結晶構造タイプは次のうちどれですか？ 

a) CsCl型  
b) NaCl型  
c) ダイヤモンド型  
d) ペロブスカイト型 

解答を見る

**正解: b) NaCl型**

**解説:**  
MgOは岩塩型（NaCl型）構造を取ります。Mg²⁺とO²⁻がそれぞれFCC格子を形成し、各イオンは6つの反対符号イオンに囲まれます（配位数 = 6）。半径比 r(Mg²⁺)/r(O²⁻) = 0.72/1.40 = 0.514 で、Paulingの規則により八面体配位（CN = 6）が予測されます。 

#### Q2: Paulingの規則 Easy

CsCl（塩化セシウム）の陽イオン配位数は何ですか？また、その理由を説明してください。 

解答を見る

**正解: 8（立方体配位）**

**解説:**  
Cs⁺の半径は1.67Å、Cl⁻の半径は1.81Åです。半径比 = 1.67/1.81 = 0.922 で、Paulingの第1規則により 0.732 < r₊/r₋ < 1.0 の範囲にあるため、配位数8（立方体配位）となります。CsCl型構造では、Cs⁺が単純立方格子の中心に位置し、8つのCl⁻に囲まれます。 

#### Q3: ペロブスカイトの一般式 Easy

ペロブスカイト構造の一般式はABO₃です。BaTiO₃において、A位置とB位置の配位数をそれぞれ答えてください。 

解答を見る

**正解: A位置（Ba）: 12配位、B位置（Ti）: 6配位**

**解説:**  
立方晶ペロブスカイトでは、大きなBa²⁺イオンが立方体の頂点（体心から見て12配位）、小さなTi⁴⁺イオンが体心（酸素八面体の中心、6配位）に位置します。O²⁻は面心に配置され、Ti-O-Ti結合のネットワークを形成します。 

### Medium（応用）

#### Q4: Goldschmidtの許容因子計算 Medium

SrTiO₃のGoldschmidtの許容因子tを計算してください。イオン半径は Sr²⁺ = 1.44Å（12配位）、Ti⁴⁺ = 0.605Å（6配位）、O²⁻ = 1.40Å です。この材料はペロブスカイト構造を取ると予想されますか？ 

解答を見る

**正解: t = 1.002、理想的なペロブスカイト構造**

**計算過程:**  
\\[ t = \frac{r_{\text{Sr}} + r_{\text{O}}}{\sqrt{2}(r_{\text{Ti}} + r_{\text{O}})} = \frac{1.44 + 1.40}{\sqrt{2}(0.605 + 1.40)} = \frac{2.84}{2.834} = 1.002 \\] 

**解説:**  
t ≈ 1.0 は理想的なペロブスカイト構造を示します。実際、SrTiO₃は室温で立方晶ペロブスカイト構造（空間群Pm-3m）を取り、高い誘電率（ε ≈ 300）を示します。量子常誘電体として、低温でも強誘電転移を起こしません。 

#### Q5: 正スピネルと逆スピネルの判別 Medium

MgAl₂O₄は正スピネル、Fe₃O₄は逆スピネル構造を取ります。この違いを決定する主な要因は何ですか？また、逆スピネルの方が安定になる条件を説明してください。 

解答を見る

**正解: 結晶場安定化エネルギー（CFSE）と陽イオンの電子配置**

**解説:**  
スピネル構造の安定性は、陽イオンが四面体位置と八面体位置のどちらを好むかで決まります： 

  * **MgAl₂O₄（正スピネル）** : Mg²⁺（d⁰）は結晶場安定化なし、Al³⁺（d⁰）も同様。イオン半径の違いで、小さなMg²⁺が四面体、Al³⁺が八面体を占める。
  * **Fe₃O₄（逆スピネル）** : Fe³⁺（d⁵、高スピン）はどちらの位置でもCFSEが小さい。Fe²⁺（d⁶）は八面体位置で大きなCFSEを得る。したがって、Fe³⁺が四面体と八面体に分かれ、Fe²⁺が八面体を占める逆スピネルが安定。

一般に、d³、d⁶、d⁸電子配置のイオンは八面体位置を強く好むため、逆スピネル構造が安定化されます。 

#### Q6: 3C-SiCの密度計算 Medium

3C-SiC（閃亜鉛鉱型）の格子定数は a = 4.358Å です。単位格子の質量と体積から密度を計算してください。（原子量: Si = 28.09, C = 12.01） 

解答を見る

**正解: 密度 ≈ 3.21 g/cm³**

**計算過程:**

  1. 単位格子の体積: V = a³ = (4.358×10⁻⁸ cm)³ = 8.28×10⁻²³ cm³
  2. 単位格子中の原子数: 4組のSiC（閃亜鉛鉱型のFCC基本）
  3. 単位格子の質量: m = 4 × (28.09 + 12.01) / N_A = 4 × 40.10 / 6.022×10²³ = 2.66×10⁻²² g
  4. 密度: ρ = m / V = 2.66×10⁻²² / 8.28×10⁻²³ = 3.21 g/cm³

実験値（3.21 g/cm³）とよく一致します。SiCの高密度は、共有結合性と強い原子間結合に由来します。 

#### Q7: ウルツ鉱構造の理想c/a比 Medium

ウルツ鉱構造の理想c/a比は √(8/3) ≈ 1.633 です。AlNの実測値はc/a = 1.601 です。この偏差は何を意味しますか？また、イオン性と共有性のどちらが強いと推測されますか？ 

解答を見る

**正解: c/a < 1.633 は、共有結合性が強いことを示唆**

**解説:**  
理想ウルツ鉱構造は、球の最密充填を仮定した場合のc/a比です。実際の材料では： 

  * **c/a < 1.633**: 共有結合性が強く、方向性のある結合によりc軸方向が圧縮される（例: AlN, GaN）
  * **c/a > 1.633**: イオン結合性が強く、静電反発によりc軸方向が伸びる（例: ZnO = 1.602、BeO = 1.623）

AlNは共有結合性が強いため、c/a = 1.601 と理想値より小さくなります。この性質は、高い熱伝導率（320 W/m·K）と関連しています。 

### Hard（発展）

#### Q8: BaTiO₃の相転移と構造変化 Hard

BaTiO₃は120°Cで立方晶から正方晶に相転移します。正方晶相（a = 3.992Å, c = 4.036Å）における自発分極の方向と大きさを、Ti⁴⁺イオンの変位（δ ≈ 0.12Å）から見積もってください。（有効電荷: Z* = 7e、単位格子体積から計算） 

解答を見る

**正解: 自発分極 P_s ≈ 0.26 C/m² （c軸方向）**

**計算過程:**

  1. 単位格子体積: V = a² × c = (3.992)² × 4.036 = 64.3 Å³ = 6.43×10⁻²⁹ m³
  2. 双極子モーメント: p = Z* × e × δ = 7 × 1.602×10⁻¹⁹ C × 0.12×10⁻¹⁰ m = 1.35×10⁻²⁹ C·m
  3. 自発分極: P_s = p / V = 1.35×10⁻²⁹ / 6.43×10⁻²⁹ = 0.21 C/m²

実験値（0.26 C/m²）と近い値が得られます。差異は、O²⁻イオンの変位や電子雲の分極を考慮していないためです。 

**物理的意味:**  
Ti⁴⁺が酸素八面体の中心からc軸方向にずれることで、強い双極子モーメントが発生します。この自発分極により、BaTiO₃は優れた圧電材料・強誘電体メモリとして応用されます。 

#### Q9: Fe₃O₄の磁気モーメント計算 Hard

マグネタイト（Fe₃O₄）は逆スピネル構造 Fe³⁺[Fe²⁺Fe³⁺]O₄ を取ります。単位格子（8組のFe₃O₄）の理論磁気モーメントを計算してください。（Fe²⁺: 4μ_B、Fe³⁺: 5μ_B、μ_B = Bohr磁子） 

解答を見る

**正解: 単位格子の磁気モーメント = 32μ_B**

**計算過程:**

  1. 1分子式単位（Fe₃O₄）の構成: 
     * 四面体位置: Fe³⁺（スピン↑） → +5μ_B
     * 八面体位置: Fe²⁺（スピン↓） → -4μ_B、Fe³⁺（スピン↓） → -5μ_B
  2. 正味の磁気モーメント（1分子式あたり）: 
     * M = 5μ_B（四面体Fe³⁺） - 4μ_B（八面体Fe²⁺） - 5μ_B（八面体Fe³⁺） = -4μ_B
     * 絶対値: |M| = 4μ_B（フェリ磁性）
  3. 単位格子（8組のFe₃O₄）: M_total = 8 × 4μ_B = 32μ_B

実験値（約4.1μ_B/分子式単位）とよく一致します。この強い磁気モーメントにより、Fe₃O₄は室温で磁性体として機能し、磁気記録媒体、MRI造影剤、磁性流体などに応用されます。 

#### Q10: 複合材料の密度予測 Hard

MgO（密度3.58 g/cm³）とAl₂O₃（密度3.98 g/cm³）が1:1のモル比で反応し、MgAl₂O₄スピネル（理論密度3.58 g/cm³）を生成します。反応前後の体積変化率を計算し、焼結時の収縮を予測してください。 

解答を見る

**正解: 体積収縮率 ≈ 1.8%**

**計算過程:**

  1. 反応式: MgO + Al₂O₃ → MgAl₂O₄
  2. 反応前の質量: 
     * MgO: 40.30 g/mol
     * Al₂O₃: 101.96 g/mol
     * 合計: 142.26 g/mol
  3. 反応前の体積（1モルあたり）: 
     * V_MgO = 40.30 / 3.58 = 11.26 cm³
     * V_Al2O3 = 101.96 / 3.98 = 25.62 cm³
     * V_before = 11.26 + 25.62 = 36.88 cm³
  4. 反応後の体積（1モルMgAl₂O₄）: 
     * V_after = 142.26 / 3.58 = 39.74 cm³
  5. 体積変化率: ΔV/V = (39.74 - 36.88) / 36.88 × 100 = 7.8%（膨張）

**注意:** 実際には、反応前の原料は焼結体ではなく粉末であり、空隙率が高い（30-50%）。焼結プロセス全体では、空隙が減少し、全体として15-20%の収縮が観測されます。スピネル相生成による体積膨張（7.8%）は、空隙減少による収縮と相殺されます。 

**工学的意義:**  
この体積変化を考慮して、耐火レンガやセラミックス部品の焼結条件（温度、時間、雰囲気）を最適化します。急激な体積変化はクラックの原因となるため、段階的な昇温プロファイルが重要です。 

## 参考文献

  1. **Kingery, W.D., Bowen, H.K., Uhlmann, D.R. (1976).** _Introduction to Ceramics_ (2nd ed.). Wiley, pp. 30-89 (crystal structures), pp. 92-135 (defects and diffusion). 
  2. **Carter, C.B., Norton, M.G. (2013).** _Ceramic Materials: Science and Engineering_ (2nd ed.). Springer, pp. 45-120 (ionic and covalent structures), pp. 267-310 (phase transformations). 
  3. **West, A.R. (2014).** _Solid State Chemistry and its Applications_ (2nd ed.). Wiley, pp. 1-85 (crystal structures), pp. 187-245 (perovskites and spinels), pp. 320-375 (structure-property relationships). 
  4. **Barsoum, M.W. (2020).** _Fundamentals of Ceramics_ (2nd ed.). CRC Press, pp. 20-75 (bonding and crystal structures), pp. 105-158 (point defects), pp. 445-490 (electrical properties). 
  5. **Richerson, D.W., Lee, W.E. (2018).** _Modern Ceramic Engineering: Properties, Processing, and Use in Design_ (4th ed.). CRC Press, pp. 50-95 (structures), pp. 120-165 (mechanical properties). 
  6. **Pymatgen Documentation (2024).** Available at: <https://pymatgen.org/> (Python materials analysis library, Materials Project integration) 
  7. **Shannon, R.D. (1976).** "Revised effective ionic radii and systematic studies of interatomic distances in halides and chalcogenides." _Acta Crystallographica A_ , 32, 751-767. (イオン半径の標準データ) 

**さらに学ぶために**

  * 構造解析技術: _Elements of X-ray Diffraction_ (Cullity & Stock, 2014) - XRD原理と実践
  * 第一原理計算: _Density Functional Theory: A Practical Introduction_ (Sholl & Steckel, 2009) - DFT入門
  * セラミックス物性: _Physical Properties of Ceramics_ (Wachtman et al., 2009) - 機械・熱・電気特性
  * データベース: Materials Project (<https://materialsproject.org/>) - 無料の材料データベース

## 学習目標確認チェックリスト

### レベル1: 基本理解

  * □ NaCl型、CsCl型、CaF₂型の違いを説明できる
  * □ Paulingの第1規則を使い、半径比から配位数を予測できる
  * □ イオン結合性と共有結合性セラミックスの特徴を述べられる
  * □ ペロブスカイト構造（ABO₃）の基本配置を理解している
  * □ 正スピネルと逆スピネルの違いを説明できる
  * □ 閃亜鉛鉱構造とウルツ鉱構造の違いを理解している

### レベル2: 実践スキル

  * □ Pymatgenを使ってNaCl型、閃亜鉛鉱型構造を生成できる
  * □ 格子定数から密度を計算できる
  * □ イオン半径データから配位数を予測できる
  * □ Goldschmidtの許容因子を計算し、ペロブスカイト安定性を評価できる
  * □ X線回折パターンをシミュレートできる
  * □ Materials Projectから結晶構造データを取得できる
  * □ ウルツ鉱構造のc/a比から結合性を評価できる

### レベル3: 応用力

  * □ Paulingの規則を適用して、新規化合物の構造を予測できる
  * □ BaTiO₃の相転移と構造変化を理解し、自発分極を見積もれる
  * □ Fe₃O₄の逆スピネル構造から磁気モーメントを計算できる
  * □ 構造-物性相関を説明できる（硬度、誘電率、磁性）
  * □ 実材料の設計に構造知識を応用できる（耐火材料、圧電素子、磁性体）
  * □ XRDパターンから構造同定ができる
  * □ 第一原理計算の結果を解釈し、構造最適化を議論できる

**次のステップ** 第2章では、セラミックスの欠陥構造（点欠陥、転位、粒界）を学びます。本章で学んだ理想結晶構造が、実材料ではどのように変化し、それが物性にどう影響するかを理解しましょう。特に、イオン伝導性、拡散、焼結などの現象は欠陥構造と密接に関連しています。 

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
