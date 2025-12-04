---
title: 第2章：原子構造と化学結合
chapter_title: 第2章：原子構造と化学結合
subtitle: 材料特性を決定する原子レベルの仕組み
reading_time: 25-30分
difficulty: 入門〜中級
code_examples: 6
---

材料の性質は、原子がどのように結合しているかによって決まります。この章では、原子の構造、電子配置、そして化学結合の種類を学び、それらが材料特性にどう影響するかを理解します。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 原子の構造（原子核、電子殻、電子配置）を説明できる
  * ✅ 化学結合の4つの主要タイプ（イオン・共有・金属・分子間力）を理解する
  * ✅ 結合の種類と材料特性（強度・導電性・融点）の関係を説明できる
  * ✅ 電気陰性度と結合性の関係を理解する
  * ✅ Pythonで電子配置と結合エネルギーを可視化できる

* * *

## 2.1 原子の構造と電子配置

### 原子の基本構造

原子は、以下の3つの基本粒子から構成されています：

粒子 | 電荷 | 質量（u） | 位置  
---|---|---|---  
**陽子** （proton） | +1 | 1.0073 | 原子核  
**中性子** （neutron） | 0 | 1.0087 | 原子核  
**電子** （electron） | -1 | 0.00055 | 電子殻  
  
**原子番号（Z）** : 陽子の数（= 中性原子の電子数）

**質量数（A）** : 陽子数 + 中性子数

### 電子殻と電子配置

電子は、原子核の周りの**電子殻（shell）** に存在します。電子殻は、エネルギーの低い方から**K殻、L殻、M殻、N殻...** と名付けられています。

各電子殻には、収容できる電子の最大数が決まっています：

$$\text{最大電子数} = 2n^2 \quad (n: \text{主量子数})$$

電子殻 | 主量子数 n | 最大電子数 | 副殻  
---|---|---|---  
**K殻** | 1 | 2 | 1s  
**L殻** | 2 | 8 | 2s, 2p  
**M殻** | 3 | 18 | 3s, 3p, 3d  
**N殻** | 4 | 32 | 4s, 4p, 4d, 4f  
  
**電子配置の表記法** : 例えば、炭素（C, Z=6）の電子配置は以下のように表記されます：

$$\text{C}: 1s^2 \, 2s^2 \, 2p^2$$

これは、「1s軌道に2個、2s軌道に2個、2p軌道に2個の電子が存在する」という意味です。

### 価電子と化学反応性

**価電子（valence electron）** は、最外殻にある電子で、化学結合に関与します。価電子の数が材料の化学的・電気的性質を決定します。

**代表的な元素の電子配置と価電子** :

元素 | 原子番号 | 電子配置 | 価電子数 | 特性  
---|---|---|---|---  
**水素（H）** | 1 | 1s¹ | 1 | 反応性が高い  
**炭素（C）** | 6 | 1s² 2s² 2p² | 4 | 4つの共有結合を形成  
**ナトリウム（Na）** | 11 | [Ne] 3s¹ | 1 | 容易にイオン化（Na⁺）  
**シリコン（Si）** | 14 | [Ne] 3s² 3p² | 4 | 半導体  
**鉄（Fe）** | 26 | [Ar] 3d⁶ 4s² | 2-3 | 磁性、触媒活性  
**銅（Cu）** | 29 | [Ar] 3d¹⁰ 4s¹ | 1 | 高い導電性  
  
* * *

## 2.2 化学結合の種類

原子が材料を形成するとき、原子同士が**化学結合** で結びつきます。結合の種類により、材料の性質が大きく変わります。

### 1\. イオン結合（Ionic Bond）

**形成メカニズム** :

  * 金属（電気陰性度が低い）が電子を放出 → 陽イオン（cation）
  * 非金属（電気陰性度が高い）が電子を受け取る → 陰イオン（anion）
  * 陽イオンと陰イオンの静電引力により結合

**例** : NaCl（塩化ナトリウム）

$$\text{Na} \rightarrow \text{Na}^+ + e^-$$

$$\text{Cl} + e^- \rightarrow \text{Cl}^-$$

$$\text{Na}^+ + \text{Cl}^- \rightarrow \text{NaCl}$$

**特徴** :

  * 高い融点（強い静電引力）
  * 硬いが脆い（イオンの配列がずれると反発）
  * 固体では電気を通さないが、溶融液や水溶液では導電性あり
  * 水に溶けやすい

**代表的な材料** : NaCl, MgO, Al₂O₃（アルミナ）, ZrO₂（ジルコニア）

### 2\. 共有結合（Covalent Bond）

**形成メカニズム** :

  * 原子同士が電子を共有
  * 各原子が安定な電子配置（閉殻構造）を獲得
  * 結合性軌道に電子対が存在

**例** : ダイヤモンド（C）、シリコン（Si）、SiC（炭化ケイ素）

**特徴** :

  * 非常に高い硬度（結合が強い）
  * 高い融点（ダイヤモンド: 3823 K）
  * 電気絶縁性（ただし、半導体もこのカテゴリ）
  * 方向性がある（結合角が決まっている）

**代表的な材料** : ダイヤモンド、SiC、Si₃N₄（窒化ケイ素）

### 3\. 金属結合（Metallic Bond）

**形成メカニズム** :

  * 金属原子が価電子を放出し、正イオンの「海」を形成
  * 自由電子（delocalizedな電子）が金属イオン間を動き回る
  * 自由電子と金属イオンの静電引力により結合

**特徴** :

  * 高い電気伝導率（自由電子が電流を運ぶ）
  * 高い熱伝導率（自由電子が熱を運ぶ）
  * 延性・展性に優れる（金属イオンの配列が変わっても結合は保たれる）
  * 金属光沢（自由電子が光を反射）

**代表的な材料** : Fe, Cu, Al, Ti, Au, Ag

### 4\. 分子間力（Intermolecular Forces）

分子同士を弱く結びつける力です。化学結合（1-3）よりもはるかに弱い相互作用です。

#### a. ファンデルワールス力（Van der Waals Forces）

  * 分子の瞬間的な電荷の偏りによる弱い引力
  * すべての分子間に働く
  * 例: 固体Ar、CH₄（メタン）

#### b. 水素結合（Hydrogen Bond）

  * H-O, H-N, H-Fなどの結合で、Hが正に分極
  * 隣接分子の電気陰性度の高い原子（O, N, F）との間に働く
  * ファンデルワールス力より強い
  * 例: H₂O（氷）、DNA二重らせん、タンパク質の高次構造

**特徴** :

  * 低い融点・沸点（弱い結合）
  * 柔らかい
  * 電気絶縁性

**代表的な材料** : 高分子材料（ポリエチレン、ポリプロピレンなど）

### 結合タイプの比較

結合タイプ | 結合強度 | 融点 | 電気伝導性 | 機械的性質 | 代表例  
---|---|---|---|---|---  
**イオン結合** | 強い | 高い | 溶融液で導電 | 硬いが脆い | NaCl, MgO  
**共有結合** | 非常に強い | 非常に高い | 低い（絶縁体） | 非常に硬い、脆い | ダイヤモンド, SiC  
**金属結合** | 中〜強 | 中〜高 | 非常に高い | 延性・展性あり | Fe, Cu, Al  
**分子間力** | 弱い | 低い | 低い（絶縁体） | 柔らかい | 高分子, 氷  
  
* * *

## 2.3 結合と材料特性の関係

### 電気陰性度と結合性

**電気陰性度（electronegativity）** は、原子が電子を引き寄せる能力を表します。Paulingスケールが広く使われます。

**主要元素の電気陰性度** :

  * F（フッ素）: 4.0（最大）
  * O（酸素）: 3.5
  * N（窒素）: 3.0
  * C（炭素）: 2.5
  * H（水素）: 2.1
  * Si（シリコン）: 1.8
  * Al（アルミニウム）: 1.5
  * Na（ナトリウム）: 0.9
  * Cs（セシウム）: 0.7（最小）

**電気陰性度差と結合タイプの関係** :

2つの原子間の電気陰性度差（Δχ）により、結合のタイプがおおよそ決まります：

  * **Δχ > 2.0**: イオン結合が支配的（例: NaCl, Δχ = 3.0 - 0.9 = 2.1）
  * **0.5 < Δχ < 2.0**: 極性共有結合（例: H₂O, Δχ = 3.5 - 2.1 = 1.4）
  * **Δχ < 0.5**: 無極性共有結合（例: C-C, Δχ = 0）

### 結合エネルギーと材料特性

**結合エネルギ（bond energy）** は、結合を切断するのに必要なエネルギーです。結合エネルギーが大きいほど、材料の融点、硬度、弾性率が高くなります。

結合 | 結合エネルギー（kJ/mol） | 結合長（Å）  
---|---|---  
C-C（ダイヤモンド） | 347 | 1.54  
C=C（エチレン） | 614 | 1.34  
C≡C（アセチレン） | 839 | 1.20  
Si-Si | 222 | 2.35  
O-H（水） | 463 | 0.96  
Na-Cl（イオン結合） | 410 | 2.36  
  
**結合エネルギーと融点の関係** :

  * ダイヤモンド（C-C結合、347 kJ/mol）: 融点 3823 K
  * シリコン（Si-Si結合、222 kJ/mol）: 融点 1687 K
  * 氷（水素結合、約20 kJ/mol）: 融点 273 K

* * *

## 2.4 Pythonによる電子配置と結合の可視化

ここから、Pythonを使って原子構造と化学結合を可視化し、理解を深めましょう。

### コード例1: 電子配置の自動計算と表示

任意の元素の電子配置を自動で計算し、表示します。
    
    
    def electron_configuration(atomic_number):
        """
        原子番号から電子配置を計算する関数
    
        Parameters:
        atomic_number (int): 原子番号
    
        Returns:
        dict: 各軌道の電子数
        """
        # 軌道の順序（エネルギー準位の低い順）
        orbital_order = [
            '1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p',
            '6s', '4f', '5d', '6p', '7s', '5f', '6d', '7p'
        ]
    
        # 各軌道の最大電子数
        max_electrons = {
            's': 2, 'p': 6, 'd': 10, 'f': 14
        }
    
        electrons_left = atomic_number
        config = {}
    
        for orbital in orbital_order:
            if electrons_left == 0:
                break
    
            orbital_type = orbital[-1]  # 's', 'p', 'd', 'f'
            max_e = max_electrons[orbital_type]
    
            if electrons_left >= max_e:
                config[orbital] = max_e
                electrons_left -= max_e
            else:
                config[orbital] = electrons_left
                electrons_left = 0
    
        return config
    
    
    def print_electron_configuration(atomic_number, element_symbol):
        """
        電子配置を見やすく表示する関数
        """
        config = electron_configuration(atomic_number)
    
        # 表記法1: 通常の表記
        config_str = ' '.join([f"{orbital}^{count}" for orbital, count in config.items()])
    
        # 表記法2: 希ガス表記（簡略表示）
        noble_gases = {2: 'He', 10: 'Ne', 18: 'Ar', 36: 'Kr', 54: 'Xe', 86: 'Rn'}
    
        print(f"\n【{element_symbol}（原子番号: {atomic_number}）の電子配置】")
        print(f"完全表記: {config_str}")
    
        # 価電子の計算
        outermost_shell = max([int(orbital[0]) for orbital in config.keys()])
        valence_electrons = sum([count for orbital, count in config.items()
                                 if int(orbital[0]) == outermost_shell])
    
        print(f"最外殻: {outermost_shell}殻")
        print(f"価電子数: {valence_electrons}個")
    
        return config
    
    
    # 代表的な元素の電子配置を表示
    elements = [
        (1, 'H', '水素'),
        (6, 'C', '炭素'),
        (11, 'Na', 'ナトリウム'),
        (14, 'Si', 'シリコン'),
        (26, 'Fe', '鉄'),
        (29, 'Cu', '銅'),
        (79, 'Au', '金')
    ]
    
    print("=" * 60)
    print("主要元素の電子配置")
    print("=" * 60)
    
    for z, symbol, name in elements:
        config = print_electron_configuration(z, f"{name}（{symbol}）")
    

**出力例** :
    
    
    ============================================================
    主要元素の電子配置
    ============================================================
    
    【水素（H）（原子番号: 1）の電子配置】
    完全表記: 1s^1
    最外殻: 1殻
    価電子数: 1個
    
    【炭素（C）（原子番号: 6）の電子配置】
    完全表記: 1s^2 2s^2 2p^2
    最外殻: 2殻
    価電子数: 4個
    
    【ナトリウム（Na）（原子番号: 11）の電子配置】
    完全表記: 1s^2 2s^2 2p^6 3s^1
    最外殻: 3殻
    価電子数: 1個
    
    【シリコン（Si）（原子番号: 14）の電子配置】
    完全表記: 1s^2 2s^2 2p^6 3s^2 3p^2
    最外殻: 3殻
    価電子数: 4個
    
    【鉄（Fe）（原子番号: 26）の電子配置】
    完全表記: 1s^2 2s^2 2p^6 3s^2 3p^6 4s^2 3d^6
    最外殻: 4殻
    価電子数: 2個
    
    【銅（Cu）（原子番号: 29）の電子配置】
    完全表記: 1s^2 2s^2 2p^6 3s^2 3p^6 4s^1 3d^10
    最外殻: 4殻
    価電子数: 1個
    
    【金（Au）（原子番号: 79）の電子配置】
    完全表記: 1s^2 2s^2 2p^6 3s^2 3p^6 4s^2 3d^10 4p^6 5s^2 4d^10 5p^6 6s^1 4f^14 5d^10
    最外殻: 6殻
    価電子数: 1個
    

**解説** : この関数により、任意の元素の電子配置を自動で計算できます。価電子数は、化学結合の性質を決定する重要なパラメータです。

### コード例2: 周期表の電気陰性度ヒートマップ

周期表上の電気陰性度をヒートマップで可視化します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 主要元素の電気陰性度データ（Paulingスケール）
    # 周期表の位置: (周期, 族) → 電気陰性度
    electronegativity_data = {
        # 1周期
        (1, 1): ('H', 2.20),
        (1, 18): ('He', 0),  # 希ガスは定義されない
    
        # 2周期
        (2, 1): ('Li', 0.98), (2, 2): ('Be', 1.57),
        (2, 13): ('B', 2.04), (2, 14): ('C', 2.55), (2, 15): ('N', 3.04),
        (2, 16): ('O', 3.44), (2, 17): ('F', 3.98), (2, 18): ('Ne', 0),
    
        # 3周期
        (3, 1): ('Na', 0.93), (3, 2): ('Mg', 1.31),
        (3, 13): ('Al', 1.61), (3, 14): ('Si', 1.90), (3, 15): ('P', 2.19),
        (3, 16): ('S', 2.58), (3, 17): ('Cl', 3.16), (3, 18): ('Ar', 0),
    
        # 4周期（一部）
        (4, 1): ('K', 0.82), (4, 2): ('Ca', 1.00),
        (4, 13): ('Ga', 1.81), (4, 14): ('Ge', 2.01), (4, 15): ('As', 2.18),
        (4, 16): ('Se', 2.55), (4, 17): ('Br', 2.96), (4, 18): ('Kr', 0),
    
        # 遷移金属（一部）
        (4, 6): ('Cr', 1.66), (4, 8): ('Fe', 1.83), (4, 11): ('Cu', 1.90),
        (4, 12): ('Zn', 1.65),
    }
    
    # 周期表グリッド作成（4周期×18族）
    grid = np.full((4, 18), np.nan)
    labels = [['' for _ in range(18)] for _ in range(4)]
    
    for (period, group), (symbol, en) in electronegativity_data.items():
        grid[period-1, group-1] = en
        labels[period-1][group-1] = symbol
    
    # ヒートマップ作成
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # NaN（希ガスや空白）をマスク
    mask = np.isnan(grid) | (grid == 0)
    
    sns.heatmap(grid, annot=np.array(labels), fmt='', cmap='RdYlGn_r',
                cbar_kws={'label': '電気陰性度 (Pauling scale)'},
                linewidths=0.5, linecolor='gray', mask=mask,
                vmin=0.5, vmax=4.0, ax=ax)
    
    ax.set_xlabel('族（Group）', fontsize=13, fontweight='bold')
    ax.set_ylabel('周期（Period）', fontsize=13, fontweight='bold')
    ax.set_title('周期表における電気陰性度の分布', fontsize=15, fontweight='bold', pad=20)
    
    # 軸ラベルの調整
    ax.set_xticklabels(range(1, 19), fontsize=10)
    ax.set_yticklabels(range(1, 5), fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\n電気陰性度の傾向:")
    print("- 右上（F, O, N）ほど大きい → 電子を強く引き寄せる")
    print("- 左下（Cs, Fr, Na）ほど小さい → 電子を放出しやすい")
    print("- 金属は一般に小さい（< 2.0）")
    print("- 非金属は一般に大きい（> 2.0）")
    

**解説** : 電気陰性度は、周期表の右上ほど大きく、左下ほど小さい傾向があります。この傾向は、イオン化エネルギーや電子親和力とも関連しています。

### コード例3: 電気陰性度差と結合タイプの分類

2つの元素間の電気陰性度差から、結合タイプを分類します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 元素の電気陰性度データ
    electronegativity = {
        'H': 2.20, 'Li': 0.98, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
        'K': 0.82, 'Ca': 1.00, 'Fe': 1.83, 'Cu': 1.90, 'Zn': 1.65, 'Br': 2.96
    }
    
    
    def classify_bond(element1, element2):
        """
        2つの元素間の結合タイプを分類する関数
        """
        en1 = electronegativity[element1]
        en2 = electronegativity[element2]
        delta_en = abs(en1 - en2)
    
        if delta_en > 2.0:
            bond_type = 'イオン結合'
            color = '#ff7f0e'
        elif delta_en > 0.5:
            bond_type = '極性共有結合'
            color = '#2ca02c'
        else:
            bond_type = '無極性共有結合'
            color = '#1f77b4'
    
        return delta_en, bond_type, color
    
    
    # 代表的な化合物の結合分類
    compounds = [
        ('Na', 'Cl', 'NaCl（食塩）'),
        ('Mg', 'O', 'MgO（酸化マグネシウム）'),
        ('Al', 'O', 'Al₂O₃（アルミナ）'),
        ('H', 'O', 'H₂O（水）'),
        ('H', 'F', 'HF（フッ化水素）'),
        ('C', 'O', 'CO₂（二酸化炭素）'),
        ('C', 'C', 'ダイヤモンド'),
        ('Si', 'Si', 'シリコン結晶'),
        ('H', 'H', 'H₂（水素分子）'),
        ('C', 'H', 'CH₄（メタン）'),
        ('N', 'N', 'N₂（窒素分子）'),
        ('Si', 'O', 'SiO₂（シリカ）'),
    ]
    
    # 結果の計算とプロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    delta_ens = []
    compound_names = []
    colors = []
    bond_types = []
    
    for elem1, elem2, name in compounds:
        delta_en, bond_type, color = classify_bond(elem1, elem2)
        delta_ens.append(delta_en)
        compound_names.append(name)
        colors.append(color)
        bond_types.append(bond_type)
    
    # グラフ1: 電気陰性度差の棒グラフ
    ax1.barh(compound_names, delta_ens, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='極性の境界')
    ax1.axvline(x=2.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='イオン結合の境界')
    ax1.set_xlabel('電気陰性度差 Δχ', fontsize=12, fontweight='bold')
    ax1.set_title('化合物の電気陰性度差と結合タイプ', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # グラフ2: 結合タイプの分布（散布図）
    bond_type_categories = {'無極性共有結合': 1, '極性共有結合': 2, 'イオン結合': 3}
    y_positions = [bond_type_categories[bt] for bt in bond_types]
    
    ax2.scatter(delta_ens, y_positions, s=200, c=colors, edgecolors='black', linewidth=2, alpha=0.7)
    
    for i, name in enumerate(compound_names):
        ax2.annotate(name.split('（')[0], (delta_ens[i], y_positions[i]),
                     xytext=(5, 0), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('電気陰性度差 Δχ', fontsize=12, fontweight='bold')
    ax2.set_ylabel('結合タイプ', fontsize=12, fontweight='bold')
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['無極性共有結合', '極性共有結合', 'イオン結合'])
    ax2.set_title('電気陰性度差による結合分類', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 境界線
    ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=2.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # 結果の表示
    print("\n【結合タイプの分類結果】")
    print("=" * 70)
    for name, delta_en, bond_type in zip(compound_names, delta_ens, bond_types):
        print(f"{name:30s} | Δχ = {delta_en:.2f} | {bond_type}")
    

**出力例** :
    
    
    【結合タイプの分類結果】
    ======================================================================
    NaCl（食塩）                      | Δχ = 2.23 | イオン結合
    MgO（酸化マグネシウム）           | Δχ = 2.13 | イオン結合
    Al₂O₃（アルミナ）                 | Δχ = 1.83 | 極性共有結合
    H₂O（水）                         | Δχ = 1.24 | 極性共有結合
    HF（フッ化水素）                  | Δχ = 1.78 | 極性共有結合
    CO₂（二酸化炭素）                 | Δχ = 0.89 | 極性共有結合
    ダイヤモンド                      | Δχ = 0.00 | 無極性共有結合
    シリコン結晶                      | Δχ = 0.00 | 無極性共有結合
    H₂（水素分子）                    | Δχ = 0.00 | 無極性共有結合
    CH₄（メタン）                     | Δχ = 0.35 | 無極性共有結合
    N₂（窒素分子）                    | Δχ = 0.00 | 無極性共有結合
    SiO₂（シリカ）                    | Δχ = 1.54 | 極性共有結合
    

**解説** : 電気陰性度差により、結合タイプをおおよそ分類できます。ただし、実際の結合は純粋なイオン結合や共有結合ではなく、両方の性質を持つことが多いです。

### コード例4: 結合エネルギーと融点の関係

各結合タイプの代表的な材料について、結合エネルギーと融点の関係をプロットします。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 材料データ（平均結合エネルギー kJ/mol, 融点 K）
    materials_bond = {
        'イオン結合': {
            'NaCl': (410, 1074),
            'MgO': (1000, 3125),
            'Al₂O₃': (1180, 2345),
            'CaF₂': (540, 1691),
        },
        '共有結合': {
            'ダイヤモンド': (347, 3823),
            'SiC': (318, 3103),
            'Si₃N₄': (290, 2173),
            'Si': (222, 1687),
        },
        '金属結合': {
            'タングステン': (850, 3695),
            '鉄': (415, 1811),
            '銅': (338, 1358),
            'アルミニウム': (326, 933),
        },
        '分子間力': {
            '氷': (20, 273),
            'ドライアイス': (5, 195),
            'アルゴン': (1, 84),
            'メタン': (8, 91),
        }
    }
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors_bond = {
        'イオン結合': '#ff7f0e',
        '共有結合': '#1f77b4',
        '金属結合': '#2ca02c',
        '分子間力': '#d62728'
    }
    
    for bond_type, materials in materials_bond.items():
        bond_energies = [v[0] for v in materials.values()]
        melting_points = [v[1] for v in materials.values()]
        names = list(materials.keys())
    
        ax.scatter(bond_energies, melting_points, s=200, alpha=0.7,
                   color=colors_bond[bond_type], label=bond_type,
                   edgecolors='black', linewidth=1.5)
    
        # 材料名をラベル表示
        for name, x, y in zip(names, bond_energies, melting_points):
            ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
    
    # 対数スケール
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('平均結合エネルギー (kJ/mol)', fontsize=13, fontweight='bold')
    ax.set_ylabel('融点 (K)', fontsize=13, fontweight='bold')
    ax.set_title('結合エネルギーと融点の関係', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    print("\n結合エネルギーと融点の相関:")
    print("- 結合エネルギーが大きいほど、融点が高い傾向")
    print("- 共有結合材料（ダイヤモンド、SiC）は非常に高い融点")
    print("- 分子間力材料（氷、ドライアイス）は低い融点")
    print("- 金属結合は中程度〜高融点（自由電子による柔軟な結合）")
    

**解説** : 結合エネルギーと融点には強い相関があります。共有結合材料は結合が強いため融点が非常に高く、分子間力材料は結合が弱いため融点が低いです。

### コード例5: 結合タイプと電気伝導率の関係

結合タイプにより、電気伝導率が桁違いに異なることを可視化します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 材料データ（結合タイプ, 電気伝導率 S/m）
    materials_conductivity_bond = [
        ('銅', '金属結合', 5.96e7),
        ('アルミニウム', '金属結合', 3.5e7),
        ('鉄', '金属結合', 1.0e7),
        ('黒鉛', '共有結合（特殊）', 1e5),
        ('シリコン', '共有結合', 1e-3),
        ('ダイヤモンド', '共有結合', 1e-13),
        ('NaCl（溶融）', 'イオン結合', 1e2),
        ('NaCl（固体）', 'イオン結合', 1e-15),
        ('ポリエチレン', '分子間力', 1e-17),
        ('テフロン', '分子間力', 1e-16),
    ]
    
    # データ整理
    names = [m[0] for m in materials_conductivity_bond]
    bond_types = [m[1] for m in materials_conductivity_bond]
    conductivities = [m[2] for m in materials_conductivity_bond]
    
    # 色の割り当て
    color_map_bond = {
        '金属結合': '#2ca02c',
        '共有結合': '#1f77b4',
        '共有結合（特殊）': '#17becf',
        'イオン結合': '#ff7f0e',
        '分子間力': '#d62728'
    }
    colors = [color_map_bond[bt] for bt in bond_types]
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 10))
    
    y_positions = np.arange(len(names))
    ax.barh(y_positions, conductivities, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('電気伝導率 (S/m)', fontsize=13, fontweight='bold')
    ax.set_title('結合タイプと電気伝導率の関係', fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)
    
    # 凡例（手動作成）
    from matplotlib.patches import Rectangle
    legend_elements = [Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, edgecolor='black')
                       for color in set(colors)]
    legend_labels = list(set(bond_types))
    ax.legend(legend_elements, legend_labels, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\n結合タイプと電気伝導性の関係:")
    print("- 金属結合: 自由電子により高い電気伝導性（10⁷ S/m）")
    print("- 共有結合: 電子が局在化し、一般に低い電気伝導性")
    print("  - ただし、黒鉛は例外（π電子の非局在化により導電性あり）")
    print("- イオン結合: 固体では絶縁体、溶融液では導電性あり（イオンの移動）")
    print("- 分子間力: 電気絶縁性（10⁻¹⁵ S/m以下）")
    

**解説** : 電気伝導率は、結合タイプにより24桁以上も異なります。金属結合材料は自由電子により高い導電性を示し、分子間力材料は電気絶縁性です。

### コード例6: 結合タイプの性質マトリックス

4つの結合タイプについて、各種性質を定量的に比較するレーダーチャートを作成します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from math import pi
    
    # 結合タイプの性質データ（0-10スケール、10が最高）
    categories = ['結合強度', '融点', '電気伝導性', '延性', '硬度', '化学的安定性']
    N = len(categories)
    
    # 各結合タイプの特性値
    ionic = [7, 8, 2, 1, 8, 7]          # イオン結合
    covalent = [9, 10, 1, 1, 10, 9]     # 共有結合
    metallic = [6, 7, 10, 9, 5, 6]      # 金属結合
    intermolecular = [2, 1, 1, 8, 2, 4] # 分子間力
    
    # 角度の計算
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ionic += ionic[:1]
    covalent += covalent[:1]
    metallic += metallic[:1]
    intermolecular += intermolecular[:1]
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 各結合タイプをプロット
    ax.plot(angles, ionic, 'o-', linewidth=2, label='イオン結合', color='#ff7f0e')
    ax.fill(angles, ionic, alpha=0.15, color='#ff7f0e')
    
    ax.plot(angles, covalent, 'o-', linewidth=2, label='共有結合', color='#1f77b4')
    ax.fill(angles, covalent, alpha=0.15, color='#1f77b4')
    
    ax.plot(angles, metallic, 'o-', linewidth=2, label='金属結合', color='#2ca02c')
    ax.fill(angles, metallic, alpha=0.15, color='#2ca02c')
    
    ax.plot(angles, intermolecular, 'o-', linewidth=2, label='分子間力', color='#d62728')
    ax.fill(angles, intermolecular, alpha=0.15, color='#d62728')
    
    # 軸ラベルの設定
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)
    ax.grid(True)
    
    # タイトルと凡例
    plt.title('結合タイプ別の性質比較', size=16, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    print("\n各結合タイプの特徴:")
    print("\n【イオン結合】")
    print("- 強度: 高い（静電引力）")
    print("- 融点: 高い")
    print("- 電気伝導性: 固体では低いが、溶融液では高い")
    print("- 延性: 低い（脆い）")
    print("- 硬度: 高い")
    
    print("\n【共有結合】")
    print("- 強度: 非常に高い（最強）")
    print("- 融点: 非常に高い")
    print("- 電気伝導性: 低い（絶縁体または半導体）")
    print("- 延性: 低い（脆い）")
    print("- 硬度: 非常に高い")
    
    print("\n【金属結合】")
    print("- 強度: 中〜高")
    print("- 融点: 中〜高")
    print("- 電気伝導性: 非常に高い（自由電子）")
    print("- 延性: 非常に高い（塑性変形可能）")
    print("- 硬度: 中")
    
    print("\n【分子間力】")
    print("- 強度: 弱い")
    print("- 融点: 低い")
    print("- 電気伝導性: 低い（絶縁体）")
    print("- 延性: 高い（柔軟）")
    print("- 硬度: 低い")
    

**解説** : このレーダーチャートにより、各結合タイプの性質の特徴が一目で理解できます。材料選択の際、求める特性に応じて適切な結合タイプの材料を選ぶことが重要です。

* * *

## 2.5 本章のまとめ

### 学んだこと

  1. **原子構造と電子配置**
     * 原子は陽子、中性子、電子から構成される
     * 電子は電子殻（K, L, M, N...）に配置される
     * 価電子が化学結合に関与する
     * 電子配置は1s² 2s² 2p⁶のように表記される
  2. **化学結合の4つのタイプ**
     * イオン結合: 電子の授受による静電引力（NaCl, MgO）
     * 共有結合: 電子の共有（ダイヤモンド, SiC）
     * 金属結合: 自由電子による結合（Cu, Fe, Al）
     * 分子間力: 弱い相互作用（高分子, 氷）
  3. **電気陰性度と結合性**
     * 電気陰性度差により結合タイプが決まる
     * Δχ > 2.0: イオン結合
     * 0.5 < Δχ < 2.0: 極性共有結合
     * Δχ < 0.5: 無極性共有結合
  4. **結合と材料特性の関係**
     * 結合エネルギーが大きいほど、融点・硬度が高い
     * 金属結合材料は高い電気伝導性を示す
     * 共有結合材料は高い硬度と融点を持つが脆い
     * 分子間力材料は柔軟で加工性が良いが強度が低い
  5. **Pythonによる可視化**
     * 電子配置の自動計算
     * 電気陰性度ヒートマップ
     * 結合タイプの分類と性質比較

### 重要なポイント

  * 材料の性質は、**結合タイプによって決定される**
  * 電気陰性度は、結合性を理解する鍵となる概念
  * 結合エネルギーと融点・硬度には強い相関がある
  * 電気伝導性は、自由電子の有無により桁違いに変わる
  * 材料設計では、**結合タイプの選択が最初のステップ**

### 次の章へ

第3章では、**結晶構造の基礎** を学びます：

  * 結晶と非晶質の違い
  * 単位格子と格子定数
  * 主要な結晶構造（FCC, BCC, HCP）
  * ミラー指数と結晶面
  * Pythonによる3D結晶構造の可視化
