---
title: 第3章：第一原理計算入門（DFT基礎）
chapter_title: 第3章：第一原理計算入門（DFT基礎）
subtitle: 密度汎関数理論で材料の電子状態を解く
difficulty: 上級
---

## この章で学ぶこと

### 学習目標（3レベル）

#### 基本レベル

  * DFTの基本原理（Hohenberg-Kohn定理、Kohn-Sham方程式）を説明できる
  * 交換相関汎関数（LDA, GGA, hybrid）の違いを理解できる
  * DFT計算の基本的なワークフローを説明できる

#### 中級レベル

  * ASE/Pymatgenを使って結晶構造を作成し、DFT計算の入力を準備できる
  * VASP入力ファイル（INCAR, POSCAR, KPOINTS, POTCAR）の役割を理解できる
  * k点メッシュとカットオフエネルギーの収束テストを実施できる

#### 上級レベル

  * 材料の特性に応じて適切な汎関数を選択できる
  * 疑擬ポテンシャルとPAW法の違いを理解し、適切に使い分けられる
  * 実際の研究で使えるDFT計算ワークフローを構築できる

## 第一原理計算とは

第一原理計算（First-principles calculation）は、経験的パラメータを用いず、量子力学の基本原理から出発して材料の性質を予測する計算手法です。実験的に未知の材料の物性を理論的に予測できるため、材料開発の強力なツールとなっています。

### 多体Schrödinger方程式の困難

理想的には、$N$個の電子と$M$個の原子核からなる系の全波動関数$\Psi(\mathbf{r}_1, \ldots, \mathbf{r}_N, \mathbf{R}_1, \ldots, \mathbf{R}_M)$を解くべきです：

$$ \hat{H}\Psi = E\Psi $$ 

しかし、多体系のSchrödinger方程式を厳密に解くことは、計算量の観点から不可能です。例えば、100個の電子を持つ系では、$3N = 300$次元の偏微分方程式を解く必要があります。

### Born-Oppenheimer近似

原子核の質量は電子の約1,000倍以上重いため、電子は原子核の運動に瞬時に追随すると仮定できます（断熱近似）。これにより、電子と原子核の運動を分離できます：

$$ \Psi(\mathbf{r}_i, \mathbf{R}_\alpha) = \psi(\mathbf{r}_i; \mathbf{R}_\alpha) \chi(\mathbf{R}_\alpha) $$ 

原子核位置$\\{\mathbf{R}_\alpha\\}$を固定して、電子系のSchrödinger方程式を解けば良いことになります。

## 密度汎関数理論（DFT）の基礎

### Hohenberg-Kohn定理（1964年）

DFTの基礎は、Hohenberg と Kohn による2つの定理です：

#### 第1定理：一意性定理

外部ポテンシャル$V_{\text{ext}}(\mathbf{r})$（原子核からのクーロンポテンシャル）は、電子密度$n(\mathbf{r})$によって一意に決まる（定数の差を除く）。

**物理的意味** ：電子密度がわかれば、系の全ての物理量が決まる。

#### 第2定理：変分原理

基底状態エネルギーは、電子密度の汎関数$E[n]$であり、真の基底状態密度で最小値を取る：

$$ E_0 = \min_{n} E[n] $$ 

**物理的意味** ：電子密度を変分パラメータとして、エネルギーを最小化できる。

この定理により、多体波動関数$\Psi(\mathbf{r}_1, \ldots, \mathbf{r}_N)$（$3N$次元）の代わりに、電子密度$n(\mathbf{r})$（3次元）を扱えるようになります。計算量の劇的な削減です。

### Kohn-Sham方程式（1965年）

Hohenberg-Kohn定理は存在定理ですが、実際にどうやって$E[n]$を求めるかは示していません。Kohn と Sham は、相互作用する多電子系を、同じ密度を持つ「非相互作用系」に写像するアイデアを提案しました。

電子密度$n(\mathbf{r})$は、$N$個のKohn-Sham軌道$\\{\phi_i(\mathbf{r})\\}$を使って表されます：

$$ n(\mathbf{r}) = \sum_{i=1}^N |\phi_i(\mathbf{r})|^2 $$ 

各軌道は、Kohn-Sham方程式を満たします：

$$ \left[ -\frac{\hbar^2}{2m}\nabla^2 + V_{\text{eff}}(\mathbf{r}) \right] \phi_i(\mathbf{r}) = \varepsilon_i \phi_i(\mathbf{r}) $$ 

ここで、有効ポテンシャル$V_{\text{eff}}$は：

$$ V_{\text{eff}}(\mathbf{r}) = V_{\text{ext}}(\mathbf{r}) + V_{\text{H}}(\mathbf{r}) + V_{\text{xc}}(\mathbf{r}) $$ 

  * **$V_{\text{ext}}$** : 外部ポテンシャル（原子核からのクーロンポテンシャル）
  * **$V_{\text{H}}$** : Hartreeポテンシャル（電子間の古典的クーロン相互作用）
  * **$V_{\text{xc}}$** : 交換相関ポテンシャル（量子力学的効果を全て含む）

### 交換相関エネルギー：DFTの心臓部

交換相関エネルギー$E_{\text{xc}}[n]$は、DFTにおける唯一の近似部分です。交換（exchange）と相関（correlation）の効果を含みます：

  * **交換エネルギー** ：Pauli排他律による同種スピンの電子間反発の低減
  * **相関エネルギー** ：電子間相互作用による運動の相関

交換相関ポテンシャルは、$E_{\text{xc}}[n]$の汎関数微分で得られます：

$$ V_{\text{xc}}(\mathbf{r}) = \frac{\delta E_{\text{xc}}[n]}{\delta n(\mathbf{r})} $$ 

## 交換相関汎関数の種類

### LDA（Local Density Approximation）

最も単純な近似。各点$\mathbf{r}$での交換相関エネルギー密度が、密度$n(\mathbf{r})$の一様電子ガスのそれに等しいと仮定します：

$$ E_{\text{xc}}^{\text{LDA}}[n] = \int n(\mathbf{r}) \varepsilon_{\text{xc}}(n(\mathbf{r})) d\mathbf{r} $$ 

**特徴** ：

  * 計算コストが低い
  * 密度がゆっくり変化する系（金属）で良い精度
  * バンドギャップを過小評価する傾向
  * 弱い相互作用（van der Waals力）を記述できない

### GGA（Generalized Gradient Approximation）

密度$n(\mathbf{r})$だけでなく、その勾配$\nabla n(\mathbf{r})$も考慮します：

$$ E_{\text{xc}}^{\text{GGA}}[n] = \int n(\mathbf{r}) \varepsilon_{\text{xc}}(n(\mathbf{r}), |\nabla n(\mathbf{r})|) d\mathbf{r} $$ 

**代表的なGGA汎関数** ：

  * **PBE** （Perdew-Burke-Ernzerhof）: 固体物性計算で最も広く使われる
  * **PW91** : PBEの前身
  * **BLYP** : 分子系でよく使われる

**特徴** ：

  * LDAより原子間結合距離や結合エネルギーが改善
  * 分子系・表面系で高精度
  * 計算コストはLDAとほぼ同等
  * バンドギャップの過小評価は依然として存在

### Hybrid汎関数

Hartree-Fock交換（厳密交換）を一部混ぜる方法。バンドギャップの精度を改善します：

$$ E_{\text{xc}}^{\text{hybrid}} = aE_{\text{x}}^{\text{HF}} + (1-a)E_{\text{x}}^{\text{DFT}} + E_{\text{c}}^{\text{DFT}} $$ 

**代表的なhybrid汎関数** ：

  * **PBE0** : HF交換を25%混合（$a=0.25$）
  * **HSE06** : 短距離のみHF交換を使用（計算コスト削減）
  * **B3LYP** : 分子系で広く使用

**特徴** ：

  * バンドギャップを正確に予測
  * 半導体・絶縁体に有効
  * 計算コストが高い（GGAの5-10倍）

    
    
    ```mermaid
    graph TD
        A[交換相関汎関数] --> B[LDA]
        A --> C[GGA]
        A --> D[Hybrid]
        A --> E[メタGGA]
    
        B --> B1[最速/低精度]
        C --> C1[標準/高精度]
        D --> D1[高精度/高コスト]
        E --> E1[研究用/最高精度]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#d4edda,stroke:#28a745,stroke-width:2px
        style D fill:#fff3cd,stroke:#ffc107,stroke-width:2px
    ```

## 疑擬ポテンシャルとPAW法

### 全電子計算の困難

原子の内殻電子（1s, 2s, 2pなど）は：

  * 化学結合にほとんど寄与しない
  * 原子核近傍で急激に振動する波動関数を持つ
  * 記述に多数の平面波基底が必要（計算コスト大）

### 疑擬ポテンシャル（Pseudopotential）

内殻電子の効果を「疑擬ポテンシャル」で置き換え、価電子のみを明示的に扱う方法です。

#### 疑擬ポテンシャルの条件

  * カットオフ半径$r_c$の外側で、全電子波動関数と一致
  * 全電子系と同じ散乱特性を持つ
  * ノルム保存性：電荷が保存される

**種類** ：

  * **ノルム保存型** （Norm-conserving）：精度が高いが転送性に課題
  * **ウルトラソフト型** （Ultrasoft）：転送性が良いが複雑
  * **PAW法** （Projector Augmented Wave）：現在の標準

### PAW（Projector Augmented Wave）法

PAW法は、全電子波動関数と疑擬波動関数を厳密に関係づける変換を導入します：

$$ |\psi\rangle = |\tilde{\psi}\rangle + \sum_i (|\phi_i\rangle - |\tilde{\phi}_i\rangle) \langle\tilde{p}_i|\tilde{\psi}\rangle $$ 

  * $|\tilde{\psi}\rangle$：疑擬波動関数（計算で扱う）
  * $|\psi\rangle$：全電子波動関数（物理量計算で使用）

**PAWの利点** ：

  * 全電子計算に近い精度
  * 計算コストは疑擬ポテンシャル並み
  * VASPの標準手法

## k点サンプリング

### Blochの定理と周期境界条件

結晶は周期構造を持つため、Blochの定理が成り立ちます：

$$ \psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} u_{n\mathbf{k}}(\mathbf{r}) $$ 

ここで、$u_{n\mathbf{k}}(\mathbf{r})$は格子周期関数です。波動関数は波数ベクトル$\mathbf{k}$でラベル付けされます。

### Brillouinゾーンとk点メッシュ

第一Brillouinゾーン内の$\mathbf{k}$点で、Kohn-Sham方程式を解く必要があります。実際には、有限個のk点でサンプリングします。

**Monkhorst-Packメッシュ** ：

Brillouinゾーンを等間隔で分割：

$$ \mathbf{k} = \frac{n_1}{N_1}\mathbf{b}_1 + \frac{n_2}{N_2}\mathbf{b}_2 + \frac{n_3}{N_3}\mathbf{b}_3 $$ 

$N_1 \times N_2 \times N_3$のメッシュ密度を指定します。

#### k点収束テストの重要性

k点メッシュが粗いと、エネルギーや物性値が収束しません。特に金属は密なメッシュが必要です（半導体より3-4倍密）。目安：

  * 半導体：4×4×4 〜 8×8×8
  * 金属：12×12×12 〜 16×16×16
  * 表面・2次元系：k_z方向は粗くて良い（例：8×8×1）

## 平面波基底とカットオフエネルギー

### 平面波展開

周期系では、Kohn-Sham軌道を平面波で展開できます：

$$ \phi_{n\mathbf{k}}(\mathbf{r}) = \sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G}) e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}} $$ 

$\mathbf{G}$は逆格子ベクトルです。原理的には無限和ですが、実際には有限項で打ち切ります。

### カットオフエネルギー$E_{\text{cut}}$

波数$|\mathbf{k}+\mathbf{G}|$に対応する運動エネルギー：

$$ E = \frac{\hbar^2}{2m}|\mathbf{k}+\mathbf{G}|^2 $$ 

が$E_{\text{cut}}$以下の平面波のみを使用します：

$$ \frac{\hbar^2}{2m}|\mathbf{k}+\mathbf{G}|^2 < E_{\text{cut}} $$ 

**典型的な値** ：

  * PAW：400-600 eV（元素による）
  * Ultrasoft PP：30-50 Ry（約400-680 eV）

#### カットオフエネルギー収束テスト

エネルギーがカットオフの増加に対して収束するまでテストします。目標：

  * 全エネルギー：1 meV/atom 以下の変化
  * 格子定数：0.01 Å 以下の変化

## Pythonで学ぶDFT：ASE/Pymatgen入門

### ASE（Atomic Simulation Environment）の基礎

ASEは、原子構造の操作とDFT計算のセットアップを統一的に扱えるPythonライブラリです。
    
    
    import numpy as np
    from ase import Atoms
    from ase.build import bulk, surface, molecule
    from ase.visualize import view
    import matplotlib.pyplot as plt
    
    # 1. 結晶構造の作成
    # Si（ダイヤモンド構造）
    si = bulk('Si', 'diamond', a=5.43)
    print("Si 結晶構造:")
    print(si)
    print(f"格子定数: {si.cell[0,0]:.3f} Å")
    print(f"原子数: {len(si)}")
    
    # 2. 様々な結晶構造の作成
    structures = {
        'Al (FCC)': bulk('Al', 'fcc', a=4.05),
        'Fe (BCC)': bulk('Fe', 'bcc', a=2.87),
        'Cu (FCC)': bulk('Cu', 'fcc', a=3.61),
        'GaAs (zincblende)': bulk('GaAs', 'zincblende', a=5.65)
    }
    
    for name, struct in structures.items():
        print(f"\n{name}:")
        print(f"  格子定数: {struct.cell[0,0]:.3f} Å")
        print(f"  原子数: {len(struct)}")
        print(f"  化学式: {struct.get_chemical_formula()}")
    
    # 3. 表面構造の作成
    # Si(111) 表面
    si_surface = surface('Si', (1,1,1), layers=4, vacuum=10.0)
    print(f"\nSi(111) 表面:")
    print(f"  原子数: {len(si_surface)}")
    print(f"  セルサイズ: {si_surface.cell.lengths()}")
    
    # 4. 分子構造の作成
    h2o = molecule('H2O')
    print(f"\nH2O 分子:")
    print(f"  原子数: {len(h2o)}")
    for i, atom in enumerate(h2o):
        print(f"  {atom.symbol}: {atom.position}")
    

### 結晶構造の詳細情報取得
    
    
    from ase.build import bulk
    import numpy as np
    
    # Si結晶の詳細解析
    si = bulk('Si', 'diamond', a=5.43)
    
    # セル情報
    print("セル行列:")
    print(si.cell)
    print(f"\nセル体積: {si.get_volume():.3f} Å³")
    
    # 原子位置（分数座標）
    print("\n原子の分数座標:")
    scaled_pos = si.get_scaled_positions()
    for i, pos in enumerate(scaled_pos):
        print(f"  原子{i}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    
    # 原子位置（デカルト座標）
    print("\n原子のデカルト座標 [Å]:")
    for i, pos in enumerate(si.positions):
        print(f"  原子{i}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    
    # 原子間距離計算
    from ase.neighborlist import NeighborList, natural_cutoffs
    
    cutoffs = natural_cutoffs(si)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(si)
    
    print("\n最近接原子間距離:")
    for i in range(len(si)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            dist = si.get_distance(i, j, mic=True)
            print(f"  原子{i} - 原子{j}: {dist:.4f} Å")
        break  # 最初の原子のみ表示
    

### Pymatgenで結晶構造を操作
    
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    
    # 1. 格子定数から結晶構造を作成
    # Si（ダイヤモンド構造）
    lattice = Lattice.cubic(5.43)
    si_struct = Structure(
        lattice,
        ["Si", "Si"],
        [[0.00, 0.00, 0.00],
         [0.25, 0.25, 0.25]]
    )
    
    print("Si 結晶構造 (Pymatgen):")
    print(si_struct)
    
    # 2. 対称性解析
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    analyzer = SpacegroupAnalyzer(si_struct)
    print(f"\n空間群: {analyzer.get_space_group_symbol()}")
    print(f"空間群番号: {analyzer.get_space_group_number()}")
    print(f"点群: {analyzer.get_point_group_symbol()}")
    
    # 3. 原始格子への変換
    primitive = analyzer.get_primitive_standard_structure()
    print(f"\n原始格子の原子数: {len(primitive)}")
    print(f"慣用格子の原子数: {len(si_struct)}")
    
    # 4. 複合材料：GaAs
    gaas_struct = Structure(
        Lattice.cubic(5.65),
        ["Ga", "As"],
        [[0.00, 0.00, 0.00],
         [0.25, 0.25, 0.25]]
    )
    print("\nGaAs 結晶構造:")
    print(gaas_struct)
    print(f"空間群: {SpacegroupAnalyzer(gaas_struct).get_space_group_symbol()}")
    

### ASEとPymatgenの相互変換
    
    
    from ase.build import bulk
    from pymatgen.io.ase import AseAtomsAdaptor
    
    # ASE → Pymatgen
    si_ase = bulk('Si', 'diamond', a=5.43)
    adaptor = AseAtomsAdaptor()
    si_pmg = adaptor.get_structure(si_ase)
    
    print("ASE から Pymatgen への変換:")
    print(si_pmg)
    
    # Pymatgen → ASE
    si_ase_back = adaptor.get_atoms(si_pmg)
    print("\nPymatgen から ASE への変換:")
    print(si_ase_back)
    
    # 両方の利点を活用
    # ASE: 計算セットアップに便利
    # Pymatgen: 対称性解析、Materials Project連携に強力
    

## VASP入力ファイルの作成

### VASPの4つの入力ファイル

VASPには4つの主要な入力ファイルがあります：

ファイル名 | 内容 | 作成方法  
---|---|---  
**INCAR** | 計算パラメータ（汎関数、k点、収束条件など） | 手動またはPymatgen  
**POSCAR** | 原子構造（格子定数、原子位置） | ASE/Pymatgenから自動生成  
**KPOINTS** | k点メッシュ設定 | 手動またはPymatgen  
**POTCAR** | 疑擬ポテンシャル（PAW） | VASP付属ファイルをコピー  
  
### POSCAR（原子構造ファイル）
    
    
    from ase.build import bulk
    from ase.io import write
    
    # Si結晶のPOSCARファイル作成
    si = bulk('Si', 'diamond', a=5.43)
    
    # 2×2×2のスーパーセルに拡大
    si_supercell = si.repeat((2, 2, 2))
    
    # POSCARファイルに書き出し
    write('POSCAR', si_supercell, format='vasp')
    
    print("POSCARファイルを作成しました")
    print(f"原子数: {len(si_supercell)}")
    
    # POSCARファイルの内容を表示
    with open('POSCAR', 'r') as f:
        print("\nPOSCAR の内容:")
        print(f.read())
    

生成されるPOSCARファイルの形式：
    
    
    Si16
    1.0
       10.8600000000    0.0000000000    0.0000000000
        0.0000000000   10.8600000000    0.0000000000
        0.0000000000    0.0000000000   10.8600000000
    Si
    16
    Direct
      0.0000000000  0.0000000000  0.0000000000
      0.1250000000  0.1250000000  0.1250000000
      ...
    

### INCARファイルの作成
    
    
    # INCARファイルのテンプレート生成
    
    def create_incar(calculation_type='scf', system_name='Si',
                     functional='PBE', encut=400, ismear=0, sigma=0.05):
        """
        VASP INCARファイルを生成
    
        Parameters:
        -----------
        calculation_type : str
            'scf' (一点計算), 'relax' (構造最適化), 'band' (バンド構造)
        functional : str
            'PBE', 'LDA', 'PBE0', 'HSE06'
        encut : float
            カットオフエネルギー [eV]
        ismear : int
            スメアリング法（0: Gaussian, 1: Methfessel-Paxton, -5: tetrahedron）
        sigma : float
            スメアリング幅 [eV]
        """
    
        incar_content = f"""SYSTEM = {system_name}
    
    # Electronic structure
    ENCUT = {encut}         # カットオフエネルギー [eV]
    PREC = Accurate         # 精度（Normal, Accurate, High）
    LREAL = Auto            # 実空間射影（Auto推奨）
    
    # Exchange-correlation
    GGA = PE                # PBE汎関数（LDAの場合は削除）
    
    # SCF convergence
    EDIFF = 1E-6            # 電子状態収束条件 [eV]
    NELM = 100              # 最大SCF iteration
    
    # Smearing（金属・半導体で異なる設定）
    ISMEAR = {ismear}       # スメアリング法
    SIGMA = {sigma}         # スメアリング幅 [eV]
    
    # Parallelization
    NCORE = 4               # コア並列数（システム依存）
    """
    
        # 構造最適化の場合
        if calculation_type == 'relax':
            incar_content += """
    # Structure relaxation
    IBRION = 2              # イオン緩和アルゴリズム（2: CG, 1: RMM-DIIS）
    NSW = 100               # 最大イオンステップ
    ISIF = 3                # セルとイオン位置を最適化
    EDIFFG = -0.01          # 力の収束条件 [eV/Å]
    """
    
        # バンド計算の場合
        elif calculation_type == 'band':
            incar_content += """
    # Band structure calculation
    ICHARG = 11             # 電荷密度を読み込み
    LORBIT = 11             # 射影DOS計算
    """
    
        return incar_content
    
    # SCF計算用INCARを生成
    incar_scf = create_incar(calculation_type='scf', system_name='Si bulk')
    with open('INCAR', 'w') as f:
        f.write(incar_scf)
    
    print("INCAR ファイルを作成しました:")
    print(incar_scf)
    

### KPOINTSファイルの作成
    
    
    # KPOINTSファイル生成
    
    def create_kpoints(kpts=(8, 8, 8), shift=(0, 0, 0), mode='Monkhorst-Pack'):
        """
        VASP KPOINTSファイルを生成
    
        Parameters:
        -----------
        kpts : tuple
            k点メッシュ (nx, ny, nz)
        shift : tuple
            メッシュのシフト（通常は(0,0,0)）
        mode : str
            'Monkhorst-Pack' or 'Gamma'
        """
    
        kpoints_content = f"""Automatic mesh
    0
    {mode}
    {kpts[0]} {kpts[1]} {kpts[2]}
    {shift[0]} {shift[1]} {shift[2]}
    """
        return kpoints_content
    
    # 8×8×8 Monkhorst-Pack メッシュ
    kpoints_dense = create_kpoints(kpts=(8, 8, 8))
    with open('KPOINTS', 'w') as f:
        f.write(kpoints_dense)
    
    print("KPOINTS ファイルを作成しました:")
    print(kpoints_dense)
    
    # Gamma点中心メッシュ（金属に推奨）
    kpoints_gamma = create_kpoints(kpts=(12, 12, 12), mode='Gamma')
    print("\nGamma点中心メッシュ:")
    print(kpoints_gamma)
    

### Pymatgenを使った入力ファイル一括生成
    
    
    from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import MPRelaxSet
    
    # 1. PymatgenでSi構造を作成
    si_struct = Structure.from_file('POSCAR')  # または直接作成
    
    # 2. Materials Project標準設定で入力ファイル一式を生成
    # （研究コミュニティで広く使われる標準設定）
    mp_set = MPRelaxSet(si_struct)
    
    # 3. ファイル書き出し
    mp_set.write_input('vasp_calc/')  # vasp_calc/ ディレクトリに全ファイル作成
    
    print("VASP入力ファイル一式を vasp_calc/ に作成しました")
    print("含まれるファイル: INCAR, POSCAR, KPOINTS, POTCAR（コピー必要）")
    
    # 4. 個別にカスタマイズ
    custom_incar = Incar({
        'SYSTEM': 'Si bulk',
        'ENCUT': 520,
        'ISMEAR': 0,
        'SIGMA': 0.05,
        'EDIFF': 1e-6,
        'PREC': 'Accurate'
    })
    
    custom_kpoints = Kpoints.gamma_automatic(kpts=(10, 10, 10))
    
    # 書き出し
    custom_incar.write_file('vasp_calc/INCAR')
    custom_kpoints.write_file('vasp_calc/KPOINTS')
    

## k点とカットオフエネルギーの収束テスト

### 収束テストの戦略
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 収束テストのシミュレーション
    # （実際のVASP計算結果を模擬）
    
    # 1. k点収束テスト
    k_values = np.array([2, 4, 6, 8, 10, 12, 14, 16])
    # 収束する総エネルギー（ダミーデータ）
    energy_k = -5.4 + 0.1 * np.exp(-k_values/4) + np.random.normal(0, 0.001, len(k_values))
    
    # 2. カットオフエネルギー収束テスト
    encut_values = np.array([200, 250, 300, 350, 400, 450, 500, 550, 600])
    # 収束する総エネルギー（ダミーデータ）
    energy_encut = -5.4 - 0.05 * np.exp(-(encut_values-200)/100) + np.random.normal(0, 0.001, len(encut_values))
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # k点収束
    ax1.plot(k_values, energy_k, 'o-', markersize=8, linewidth=2, color='#f093fb')
    ax1.axhline(y=energy_k[-1], color='red', linestyle='--', label='収束値')
    ax1.fill_between(k_values, energy_k[-1]-0.001, energy_k[-1]+0.001,
                      alpha=0.2, color='red', label='±1 meV 範囲')
    ax1.set_xlabel('k点メッシュ (k×k×k)', fontsize=12)
    ax1.set_ylabel('全エネルギー [eV/atom]', fontsize=12)
    ax1.set_title('k点収束テスト', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # カットオフエネルギー収束
    ax2.plot(encut_values, energy_encut, 's-', markersize=8, linewidth=2, color='#f5576c')
    ax2.axhline(y=energy_encut[-1], color='red', linestyle='--', label='収束値')
    ax2.fill_between(encut_values, energy_encut[-1]-0.001, energy_encut[-1]+0.001,
                      alpha=0.2, color='red', label='±1 meV 範囲')
    ax2.set_xlabel('カットオフエネルギー [eV]', fontsize=12)
    ax2.set_ylabel('全エネルギー [eV/atom]', fontsize=12)
    ax2.set_title('カットオフエネルギー収束テスト', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_tests.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 収束判定
    print("=== 収束テスト結果 ===")
    print(f"\nk点収束:")
    for i in range(1, len(k_values)):
        diff = abs(energy_k[i] - energy_k[i-1]) * 1000  # meV
        status = "✓" if diff < 1.0 else "✗"
        print(f"  k={k_values[i]:2d}: ΔE = {diff:.3f} meV {status}")
    
    print(f"\nカットオフエネルギー収束:")
    for i in range(1, len(encut_values)):
        diff = abs(energy_encut[i] - energy_encut[i-1]) * 1000  # meV
        status = "✓" if diff < 1.0 else "✗"
        print(f"  ENCUT={encut_values[i]:3d} eV: ΔE = {diff:.3f} meV {status}")
    

### 実践的な収束テストスクリプト
    
    
    import os
    import numpy as np
    from ase.build import bulk
    from ase.io import write
    
    def run_convergence_test(structure, test_type='kpoints',
                              k_range=None, encut_range=None):
        """
        収束テストを実行するためのディレクトリ構造とファイルを準備
    
        Parameters:
        -----------
        structure : ase.Atoms
            テストする結晶構造
        test_type : str
            'kpoints' または 'encut'
        k_range : list
            テストするk点メッシュのリスト（例：[4, 6, 8, 10, 12]）
        encut_range : list
            テストするカットオフエネルギーのリスト [eV]
        """
    
        if test_type == 'kpoints' and k_range is not None:
            print("=== k点収束テスト準備 ===")
            for k in k_range:
                dirname = f'ktest_{k}x{k}x{k}'
                os.makedirs(dirname, exist_ok=True)
    
                # POSCAR作成
                write(f'{dirname}/POSCAR', structure, format='vasp')
    
                # KPOINTS作成
                with open(f'{dirname}/KPOINTS', 'w') as f:
                    f.write(f"""Automatic mesh
    0
    Monkhorst-Pack
    {k} {k} {k}
    0 0 0
    """)
    
                # INCAR作成（共通設定）
                with open(f'{dirname}/INCAR', 'w') as f:
                    f.write("""SYSTEM = k-point convergence test
    ENCUT = 400
    ISMEAR = 0
    SIGMA = 0.05
    EDIFF = 1E-6
    PREC = Accurate
    """)
    
                print(f"  {dirname}/ 作成完了")
    
        elif test_type == 'encut' and encut_range is not None:
            print("=== カットオフエネルギー収束テスト準備 ===")
            for encut in encut_range:
                dirname = f'encut_{encut}'
                os.makedirs(dirname, exist_ok=True)
    
                write(f'{dirname}/POSCAR', structure, format='vasp')
    
                with open(f'{dirname}/KPOINTS', 'w') as f:
                    f.write("""Automatic mesh
    0
    Monkhorst-Pack
    8 8 8
    0 0 0
    """)
    
                with open(f'{dirname}/INCAR', 'w') as f:
                    f.write(f"""SYSTEM = ENCUT convergence test
    ENCUT = {encut}
    ISMEAR = 0
    SIGMA = 0.05
    EDIFF = 1E-6
    PREC = Accurate
    """)
    
                print(f"  {dirname}/ 作成完了")
    
    # 使用例
    si = bulk('Si', 'diamond', a=5.43)
    
    # k点収束テスト用ディレクトリ作成
    run_convergence_test(si, test_type='kpoints',
                          k_range=[4, 6, 8, 10, 12, 14])
    
    # カットオフ収束テスト用ディレクトリ作成
    run_convergence_test(si, test_type='encut',
                          encut_range=[300, 350, 400, 450, 500, 550, 600])
    
    print("\n全てのテストディレクトリを作成しました")
    print("次のステップ: 各ディレクトリでVASPを実行してください")
    

## DFT計算のワークフロー
    
    
    ```mermaid
    flowchart TD
        A[結晶構造準備] --> B[入力ファイル作成]
        B --> C{収束テストk点・ENCUT}
        C -->|未収束| B
        C -->|収束| D[SCF計算]
        D --> E[構造最適化]
        E --> F{構造変化小さい?}
        F -->|No| E
        F -->|Yes| G[物性計算]
        G --> H[バンド構造]
        G --> I[状態密度]
        G --> J[電荷密度]
        H --> K[結果解析]
        I --> K
        J --> K
        K --> L[論文・報告]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#d4edda,stroke:#28a745,stroke-width:2px
        style G fill:#fff3cd,stroke:#ffc107,stroke-width:2px
        style L fill:#d4edda,stroke:#28a745,stroke-width:2px
    ```

### 標準的なDFT計算手順
    
    
    # DFT計算の標準ワークフロー（擬似コード）
    
    def dft_workflow(material_name, structure):
        """
        標準的なDFT計算ワークフロー
        """
    
        print(f"=== {material_name} のDFT計算ワークフロー ===\n")
    
        # Step 1: 収束テスト
        print("Step 1: 収束テスト")
        print("  1-1. k点メッシュ収束テスト")
        print("       → 最適k点: 8×8×8 と判定")
        print("  1-2. カットオフエネルギー収束テスト")
        print("       → 最適ENCUT: 450 eV と判定")
    
        # Step 2: 構造最適化
        print("\nStep 2: 構造最適化（IBRION=2, ISIF=3）")
        print("  - 初期構造: 実験値")
        print("  - 最適化後:")
        print("    格子定数: 5.43 → 5.47 Å（+0.7%、PBEの過大評価）")
        print("    原子位置: 変化なし（高対称性）")
    
        # Step 3: 静的計算（高精度）
        print("\nStep 3: 静的計算（一点計算、NSW=0）")
        print("  - 最適化構造で高精度SCF")
        print("  - 電荷密度・波動関数を保存")
    
        # Step 4: バンド構造計算
        print("\nStep 4: バンド構造計算（ICHARG=11）")
        print("  - 高対称点パス: Γ-X-W-K-Γ-L")
        print("  - 間接バンドギャップ: 0.65 eV（実験値1.12 eVを過小評価）")
    
        # Step 5: 状態密度計算
        print("\nStep 5: 状態密度計算（LORBIT=11）")
        print("  - 密なk点メッシュ: 16×16×16")
        print("  - 射影DOS（原子・軌道分解）")
    
        # Step 6: 結果解析
        print("\nStep 6: 結果解析")
        print("  - バンド図プロット")
        print("  - DOSプロット")
        print("  - 実験値との比較")
    
        return {
            'optimal_k': (8, 8, 8),
            'optimal_encut': 450,
            'lattice_constant': 5.47,
            'band_gap': 0.65
        }
    
    # 実行例
    si = bulk('Si', 'diamond', a=5.43)
    results = dft_workflow('Silicon', si)
    
    print(f"\n=== 計算結果サマリ ===")
    for key, value in results.items():
        print(f"  {key}: {value}")
    

## まとめ

### この章で学んだこと

#### 理論的理解

  * DFTは多体Schrödinger方程式を電子密度で解く理論
  * Hohenberg-Kohn定理：電子密度が全ての物理量を決定
  * Kohn-Sham方程式：非相互作用系への写像で計算可能に
  * 交換相関汎関数（LDA, GGA, hybrid）が唯一の近似部分

#### 実践的スキル

  * ASE/Pymatgenで結晶構造を作成・操作できる
  * VASP入力ファイル（INCAR, POSCAR, KPOINTS）を作成できる
  * k点メッシュとカットオフエネルギーの収束テストを実施できる
  * DFT計算の標準ワークフローを理解した

#### 次章への準備

  * 第4章では、DFT計算結果から電気的・磁気的性質を計算します
  * 電気伝導、Hall効果、磁化などの具体的な物性値の計算方法を学びます

## 演習問題

#### 演習1：基礎理論（難易度：★☆☆）

**問題** ：以下の記述の正誤を判定し、誤りがあれば訂正してください。

  1. DFTでは、電子密度$n(\mathbf{r})$は3次元関数であるため、多体波動関数より扱いやすい。
  2. Kohn-Sham方程式は、相互作用する多電子系を厳密に解く方程式である。
  3. GGA汎関数は、密度勾配も考慮するため、常にLDAより高精度である。
  4. PAW法は、疑擬ポテンシャル法の一種で、全電子計算に近い精度を達成する。

**解答のポイント** ：

  1. 正しい。$3N$次元 → 3次元への劇的な削減。
  2. 誤り。Kohn-Sham方程式は「非相互作用系への写像」。交換相関汎関数$E_{\text{xc}}$が近似。
  3. 誤り。多くの場合GGAが優れるが、密度が一様に近い系（単純金属）ではLDAも良い。
  4. 正しい。PAWは全電子波動関数を復元する変換を持つ。

#### 演習2：汎関数の選択（難易度：★★☆）

**問題** ：以下の系に対して最適な交換相関汎関数を選び、理由を説明してください。

  1. Siバルク（半導体）のバンドギャップ計算
  2. Cu金属の格子定数最適化
  3. TiO₂（酸化チタン）の電子状態計算
  4. 有機分子の結合エネルギー計算

**推奨解答** ：

  1. HSE06（hybrid）：GGAではバンドギャップを約50%過小評価するため。
  2. PBE（GGA）：金属は電子密度が滑らかで、GGAで十分高精度。計算コストも低い。
  3. PBE+U または HSE06：Tiのd軌道の強相関効果を考慮する必要がある。
  4. B3LYP（hybrid）：分子系で広く検証されている。ただしPBEも可。

#### 演習3：Pythonコーディング（難易度：★★☆）

**問題** ：以下の結晶構造をASEで作成し、POSCAR形式で保存するコードを書いてください。

  1. GaN（ウルツ鉱構造、a=3.19 Å, c=5.19 Å）
  2. Fe（BCC構造、a=2.87 Å）
  3. Al₂O₃（コランダム構造、Materials Projectからダウンロード）

**ヒント** ：

  * GaNはウルツ鉱構造（wurtzite）
  * FeはBCC（body-centered cubic）
  * Materials Projectへのアクセスには`pymatgen.ext.matproj`を使用

#### 演習4：収束テストの解析（難易度：★★☆）

**問題** ：以下のk点収束テスト結果を解析し、最適なk点メッシュを推奨してください。

k点メッシュ | 全エネルギー [eV/atom] | 計算時間 [分]  
---|---|---  
4×4×4| -5.3421| 2  
6×6×6| -5.3998| 5  
8×8×8| -5.4125| 12  
10×10×10| -5.4138| 25  
12×12×12| -5.4141| 45  
14×14×14| -5.4142| 75  
  
**考察ポイント** ：

  * 1 meV/atom（0.001 eV/atom）を収束基準とする
  * 計算コストと精度のバランスを考慮
  * 金属か半導体かで判断が変わる

**推奨解答** ：

8×8×8 〜 10×10×10 を推奨。理由：

  * 8×8×8: 6×6×6から1.27 meV/atom改善（収束に近い）
  * 10×10×10: 8×8×8から0.13 meV/atom改善（ほぼ収束）
  * 12×12×12以上: 計算コストに見合う改善なし
  * 半導体なら8×8×8、金属なら10×10×10を推奨

#### 演習5：INCAR設定の最適化（難易度：★★★）

**問題** ：以下の計算目的に対して、最適なINCARパラメータを設定してください。

  1. 金属Alの格子定数最適化
  2. 半導体GaAsのバンドギャップ計算（HSE06）
  3. 磁性体Feの磁気モーメント計算

**推奨解答** ：

1\. 金属Alの格子定数最適化:
    
    
    SYSTEM = Al lattice optimization
    ENCUT = 400
    PREC = Accurate
    GGA = PE
    ISMEAR = 1          # Methfessel-Paxton（金属に最適）
    SIGMA = 0.2         # やや大きめのスメアリング
    IBRION = 2          # CG法
    ISIF = 3            # セル形状も最適化
    EDIFF = 1E-6
    EDIFFG = -0.01
    NSW = 50
    

2\. 半導体GaAsのバンドギャップ計算（HSE06）:
    
    
    SYSTEM = GaAs band gap (HSE06)
    ENCUT = 450
    PREC = Accurate
    LHFCALC = .TRUE.    # Hybrid汎関数有効化
    HFSCREEN = 0.2      # HSE06のスクリーニングパラメータ
    ISMEAR = 0          # Gaussian（半導体）
    SIGMA = 0.05        # 小さめのスメアリング
    EDIFF = 1E-7        # 高精度収束
    ALGO = Damped       # HSE06に推奨
    TIME = 0.4
    

3\. 磁性体Feの磁気モーメント計算:
    
    
    SYSTEM = Fe magnetic moment
    ENCUT = 400
    PREC = Accurate
    GGA = PE
    ISMEAR = 1
    SIGMA = 0.2
    ISPIN = 2           # スピン分極計算
    MAGMOM = 2.0        # 初期磁気モーメント（Fe: 約2.2 μB）
    LORBIT = 11         # 磁気モーメント射影
    EDIFF = 1E-6
    

#### 演習6：実践課題（難易度：★★★）

**問題** ：以下の手順で、実際にDFT計算の準備を行ってください。

  1. ASEを使ってダイヤモンド構造のSi結晶（2×2×2スーパーセル）を作成
  2. VASP入力ファイル（INCAR, POSCAR, KPOINTS）を生成
  3. k点収束テスト用のディレクトリ構造（k=4,6,8,10,12）を作成
  4. 各ディレクトリの設定ファイルを確認し、計算準備が整っているか検証

**評価基準** ：

  * POSCARファイルが正しいVASP形式であるか
  * INCARの設定が半導体Siに適しているか（ISMEAR=0など）
  * k点メッシュが各ディレクトリで正しく設定されているか
  * ディレクトリ構造が整理され、追跡可能か

## 参考文献

  1. Hohenberg, P., & Kohn, W. (1964). "Inhomogeneous Electron Gas". Physical Review, 136(3B), B864.
  2. Kohn, W., & Sham, L. J. (1965). "Self-Consistent Equations Including Exchange and Correlation Effects". Physical Review, 140(4A), A1133.
  3. Perdew, J. P., Burke, K., & Ernzerhof, M. (1996). "Generalized Gradient Approximation Made Simple". Physical Review Letters, 77, 3865.
  4. Blöchl, P. E. (1994). "Projector augmented-wave method". Physical Review B, 50, 17953.
  5. Sholl, D., & Steckel, J. A. (2011). "Density Functional Theory: A Practical Introduction". Wiley.
  6. ASE documentation: https://wiki.fysik.dtu.dk/ase/
  7. Pymatgen documentation: https://pymatgen.org/
  8. VASP manual: https://www.vasp.at/wiki/index.php/The_VASP_Manual

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
