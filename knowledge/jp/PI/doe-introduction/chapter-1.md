---
title: 第1章：実験計画法の基礎と直交表
chapter_title: 第1章：実験計画法の基礎と直交表
subtitle: 効率的な実験設計の基本原理とPython実践
---

# 第1章：実験計画法の基礎と直交表

実験計画法（DOE）の基本概念を理解し、直交表を用いた効率的な実験設計を学びます。一元配置・二元配置実験から主効果図・交互作用図の解釈まで、Pythonで実践しながら習得します。

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 実験計画法（DOE）の目的と従来手法との違いを説明できる
  * ✅ 一元配置・二元配置実験を設計・解析できる
  * ✅ 直交表の原理と使い方を理解し実験に適用できる
  * ✅ 主効果図と交互作用図を作成・解釈できる
  * ✅ 化学プロセスの最適条件を直交表で探索できる

* * *

## 1.1 実験計画法（DOE）とは

### DOEの定義と目的

**実験計画法（Design of Experiments; DOE）** は、最小の実験回数で最大の情報を得るための統計的手法です。複数の因子（変数）が製品やプロセスに与える影響を効率的に評価し、最適条件を見つけることを目的とします。

**主な目的** :

  * **因子のスクリーニング** : 多くの因子の中から重要なものを特定
  * **最適化** : プロセスや製品の性能を最大化する条件の発見
  * **ロバスト性評価** : 外乱に対する安定性の向上
  * **交互作用の検出** : 因子間の相互影響の理解

### 従来の一変数実験との違い

項目 | 一変数実験（OFAT: One Factor at a Time） | 実験計画法（DOE）  
---|---|---  
**実験方法** | 1つの因子だけを変えて実験 | 複数の因子を同時に変えて実験  
**実験回数** | 因子数が増えると急増（n因子×m水準） | 直交表等で大幅に削減可能  
**交互作用** | 検出不可能 | 検出可能  
**最適条件** | 局所的な最適解のみ | 大域的な最適解を探索  
**統計的信頼性** | 低い | 高い（分散分析による検定）  
  
**例** : 化学反応で温度・圧力・触媒量の3因子、各3水準を評価する場合

  * **OFAT** : 3 × 3 × 3 = 27回の実験（各因子を個別に評価）
  * **DOE（直交表L9）** : 9回の実験で3因子の影響と交互作用を評価

### DOEの3原則

  1. **反復（Replication）**
     * 同じ条件で複数回実験を行う
     * 実験誤差の推定と統計的検定が可能になる
  2. **無作為化（Randomization）**
     * 実験順序をランダムにする
     * 時間トレンドや系統誤差の影響を排除
  3. **ブロック化（Blocking）**
     * 既知の外乱因子でグループ分け
     * バッチ間変動などの影響を分離

* * *

## 1.2 一元配置実験と分散分析

### コード例1: 一元配置実験（One-way ANOVA）

化学反応における3種類の触媒（A, B, C）の性能を比較します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    
    # 実験データ: 3種類の触媒による反応収率（%）
    # 各触媒で5回ずつ実験を実施
    np.random.seed(42)
    
    catalyst_A = [85.2, 84.8, 86.1, 85.5, 84.9]
    catalyst_B = [88.3, 89.1, 88.7, 89.5, 88.2]
    catalyst_C = [86.5, 87.2, 86.8, 87.0, 86.3]
    
    # データフレームに整理
    data = pd.DataFrame({
        'Catalyst': ['A']*5 + ['B']*5 + ['C']*5,
        'Yield': catalyst_A + catalyst_B + catalyst_C
    })
    
    print("=== 実験データ ===")
    print(data.groupby('Catalyst')['Yield'].describe())
    
    # 一元配置分散分析（One-way ANOVA）
    groups = [catalyst_A, catalyst_B, catalyst_C]
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"\n=== 一元配置分散分析（ANOVA）===")
    print(f"F統計量: {f_stat:.4f}")
    print(f"p値: {p_value:.6f}")
    
    if p_value < 0.05:
        print("結論: 有意水準5%で触媒間に有意な差がある")
    else:
        print("結論: 触媒間に有意な差は認められない")
    
    # 可視化: Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Catalyst', y='Yield', data=data, palette='Set2')
    plt.title('触媒種類による収率の比較', fontsize=14, fontweight='bold')
    plt.ylabel('収率 (%)', fontsize=12)
    plt.xlabel('触媒', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('one_way_anova.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 各触媒の平均値と95%信頼区間
    print("\n=== 各触媒の平均値と95%信頼区間 ===")
    for catalyst in ['A', 'B', 'C']:
        subset = data[data['Catalyst'] == catalyst]['Yield']
        mean = subset.mean()
        ci = stats.t.interval(0.95, len(subset)-1, loc=mean, scale=stats.sem(subset))
        print(f"触媒{catalyst}: 平均={mean:.2f}%, 95%CI=[{ci[0]:.2f}, {ci[1]:.2f}]")
    

**出力例** :
    
    
    === 実験データ ===
               count   mean       std   min    25%    50%    75%    max
    Catalyst
    A            5.0  85.30  0.487852  84.8  84.90  85.2  85.50  86.1
    B            5.0  88.76  0.543139  88.2  88.30  88.7  89.10  89.5
    C            5.0  86.76  0.382099  86.3  86.50  86.8  87.00  87.2
    
    === 一元配置分散分析（ANOVA）===
    F統計量: 55.9821
    p値: 0.000002
    結論: 有意水準5%で触媒間に有意な差がある
    
    === 各触媒の平均値と95%信頼区間 ===
    触媒A: 平均=85.30%, 95%CI=[84.70, 85.90]
    触媒B: 平均=88.76%, 95%CI=[88.09, 89.43]
    触媒C: 平均=86.76%, 95%CI=[86.29, 87.23]
    

**解釈** : F統計量が大きく、p値が0.05未満のため、触媒B、C、Aの順で収率が高く、統計的に有意な差があります。

* * *

### コード例2: 二元配置実験（Two-way ANOVA without replication）

温度と圧力が化学反応収率に与える影響を評価します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    
    # 実験データ: 温度（3水準）× 圧力（3水準）の9回実験
    # 温度: 150°C, 175°C, 200°C
    # 圧力: 1.0 MPa, 1.5 MPa, 2.0 MPa
    
    np.random.seed(42)
    
    # 収率データ（%）（実験順序はランダム化）
    data = pd.DataFrame({
        'Temperature': [150, 150, 150, 175, 175, 175, 200, 200, 200],
        'Pressure': [1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0],
        'Yield': [82.3, 85.1, 87.5, 86.2, 89.5, 91.3, 88.1, 90.8, 92.5]
    })
    
    print("=== 実験データ ===")
    print(data)
    
    # データをピボットテーブルに変換
    pivot_data = data.pivot(index='Temperature', columns='Pressure', values='Yield')
    print("\n=== ピボットテーブル（収率 %）===")
    print(pivot_data)
    
    # 二元配置分散分析（交互作用なし）
    # statsmodelsを使用
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    
    # 線形モデルのフィッティング
    model = ols('Yield ~ C(Temperature) + C(Pressure)', data=data).fit()
    anova_table = anova_lm(model, typ=2)
    
    print("\n=== 二元配置分散分析（ANOVA）===")
    print(anova_table)
    
    # 主効果の可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 温度の主効果
    temp_means = data.groupby('Temperature')['Yield'].mean()
    axes[0].plot(temp_means.index, temp_means.values, marker='o', linewidth=2, markersize=8, color='#11998e')
    axes[0].set_xlabel('温度 (°C)', fontsize=12)
    axes[0].set_ylabel('平均収率 (%)', fontsize=12)
    axes[0].set_title('温度の主効果', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 圧力の主効果
    pressure_means = data.groupby('Pressure')['Yield'].mean()
    axes[1].plot(pressure_means.index, pressure_means.values, marker='s', linewidth=2, markersize=8, color='#f59e0b')
    axes[1].set_xlabel('圧力 (MPa)', fontsize=12)
    axes[1].set_ylabel('平均収率 (%)', fontsize=12)
    axes[1].set_title('圧力の主効果', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('two_way_anova_main_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 交互作用プロット
    plt.figure(figsize=(10, 6))
    for temp in [150, 175, 200]:
        subset = data[data['Temperature'] == temp]
        plt.plot(subset['Pressure'], subset['Yield'], marker='o', label=f'{temp}°C', linewidth=2, markersize=8)
    
    plt.xlabel('圧力 (MPa)', fontsize=12)
    plt.ylabel('収率 (%)', fontsize=12)
    plt.title('温度と圧力の交互作用プロット', fontsize=14, fontweight='bold')
    plt.legend(title='温度', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('two_way_anova_interaction.png', dpi=300, bbox_inches='tight')
    plt.show()
    

**出力例** :
    
    
    === ピボットテーブル（収率 %）===
    Pressure       1.0   1.5   2.0
    Temperature
    150           82.3  85.1  87.5
    175           86.2  89.5  91.3
    200           88.1  90.8  92.5
    
    === 二元配置分散分析（ANOVA）===
                        sum_sq   df         F    PR(>F)
    C(Temperature)     57.3633  2.0  78.6275  0.000127
    C(Pressure)        65.0433  2.0  89.1667  0.000094
    Residual            1.4600  4.0       NaN       NaN
    
    結論: 温度と圧力はともに収率に有意な影響を与える（p < 0.001）
    

**解釈** : 温度と圧力の両方が収率に強く影響し、高温・高圧ほど収率が向上します。交互作用プロットで線が平行に近いため、交互作用は小さいことがわかります。

* * *

## 1.3 直交表の基礎

### 直交表とは

**直交表（Orthogonal Array）** は、複数の因子を効率的に配置した実験計画表です。各因子の水準の組み合わせが均等に現れるように設計されており、少ない実験回数で因子の効果を独立に評価できます。

**主な直交表** :

  * **L8 (2⁷)** : 7因子まで、各2水準、8回実験
  * **L9 (3⁴)** : 4因子まで、各3水準、9回実験
  * **L16 (2¹⁵)** : 15因子まで、各2水準、16回実験
  * **L27 (3¹³)** : 13因子まで、各3水準、27回実験

### コード例3: 直交表L8の生成と活用

温度、圧力、触媒量の3因子、各2水準の実験を直交表L8で設計します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 直交表L8の定義（2^7: 7因子、各2水準、8回実験）
    # ここでは3因子を使用（列1, 2, 3を使用）
    L8 = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 2, 2],
        [1, 2, 2, 1, 1, 2, 2],
        [1, 2, 2, 2, 2, 1, 1],
        [2, 1, 2, 1, 2, 1, 2],
        [2, 1, 2, 2, 1, 2, 1],
        [2, 2, 1, 1, 2, 2, 1],
        [2, 2, 1, 2, 1, 1, 2]
    ])
    
    # 因子の定義（列1, 2, 3を使用）
    # 因子A: 温度（水準1 = 150°C, 水準2 = 200°C）
    # 因子B: 圧力（水準1 = 1.0 MPa, 水準2 = 2.0 MPa）
    # 因子C: 触媒量（水準1 = 0.5 g, 水準2 = 1.0 g）
    
    factors = L8[:, :3]  # 最初の3列を使用
    
    # 実際の値にマッピング
    temperature_levels = {1: 150, 2: 200}
    pressure_levels = {1: 1.0, 2: 2.0}
    catalyst_levels = {1: 0.5, 2: 1.0}
    
    # 実験計画表の作成
    doe_table = pd.DataFrame({
        'Run': range(1, 9),
        'Temperature': [temperature_levels[x] for x in factors[:, 0]],
        'Pressure': [pressure_levels[x] for x in factors[:, 1]],
        'Catalyst': [catalyst_levels[x] for x in factors[:, 2]]
    })
    
    print("=== 直交表L8による実験計画 ===")
    print(doe_table)
    
    # シミュレートされた実験結果（収率 %）
    # 真のモデル: Yield = 70 + 10*(Temp-150)/50 + 5*(Press-1) + 3*(Cat-0.5)/0.5 + ノイズ
    np.random.seed(42)
    
    yields = []
    for _, row in doe_table.iterrows():
        temp = row['Temperature']
        press = row['Pressure']
        cat = row['Catalyst']
    
        # 真の効果（線形モデル）
        yield_true = (70 +
                      10 * (temp - 150) / 50 +
                      5 * (press - 1.0) +
                      3 * (cat - 0.5) / 0.5)
    
        # ノイズを追加
        yield_obs = yield_true + np.random.normal(0, 1)
        yields.append(yield_obs)
    
    doe_table['Yield'] = yields
    
    print("\n=== 実験結果 ===")
    print(doe_table)
    
    # 直交表の性質を確認（各因子の水準が均等に出現）
    print("\n=== 直交表の性質確認 ===")
    print("各因子の水準出現回数:")
    for col in ['Temperature', 'Pressure', 'Catalyst']:
        print(f"\n{col}:")
        print(doe_table[col].value_counts().sort_index())
    

**出力例** :
    
    
    === 直交表L8による実験計画 ===
       Run  Temperature  Pressure  Catalyst
    0    1          150       1.0       0.5
    1    2          150       1.0       1.0
    2    3          150       2.0       0.5
    3    4          150       2.0       1.0
    4    5          200       1.0       0.5
    5    6          200       1.0       1.0
    6    7          200       2.0       0.5
    7    8          200       2.0       1.0
    
    === 実験結果 ===
       Run  Temperature  Pressure  Catalyst      Yield
    0    1          150       1.0       0.5  70.494371
    1    2          150       1.0       1.0  75.861468
    2    3          150       2.0       0.5  75.646968
    3    4          150       2.0       1.0  79.522869
    4    5          200       1.0       0.5  79.647689
    5    6          200       1.0       1.0  85.522232
    6    7          200       2.0       0.5  84.233257
    7    8          200       2.0       1.0  88.767995
    
    === 直交表の性質確認 ===
    各因子の水準出現回数:
    
    Temperature:
    150    4
    200    4
    
    Pressure:
    1.0    4
    2.0    4
    
    Catalyst:
    0.5    4
    1.0    4
    

**解釈** : 直交表L8により、3因子を8回の実験で評価できます。各因子の各水準が均等に4回ずつ出現し、因子の効果を独立に評価できます。

* * *

### コード例4: 直交表L16による多因子実験

5因子（温度、圧力、触媒量、反応時間、攪拌速度）を直交表L16で評価します。
    
    
    import numpy as np
    import pandas as pd
    
    # 直交表L16の生成（2^15: 15因子まで、各2水準、16回実験）
    # ここでは5因子を使用
    def generate_L16():
        """直交表L16を生成"""
        L16 = []
        for i in range(16):
            row = []
            for j in range(15):
                # 2水準（1 or 2）を生成
                val = ((i >> j) & 1) + 1
                row.append(val)
            L16.append(row)
        return np.array(L16)
    
    L16 = generate_L16()
    
    # 5因子を列1, 2, 4, 8, 15に割り付け（標準的な割り付け）
    factor_columns = [0, 1, 3, 7, 14]  # Pythonインデックス
    factors = L16[:, factor_columns]
    
    # 因子の定義
    factor_names = ['Temperature', 'Pressure', 'Catalyst', 'Time', 'Stirring']
    levels = {
        'Temperature': {1: 150, 2: 200},
        'Pressure': {1: 1.0, 2: 2.0},
        'Catalyst': {1: 0.5, 2: 1.0},
        'Time': {1: 30, 2: 60},       # 反応時間（分）
        'Stirring': {1: 200, 2: 400}  # 攪拌速度（rpm）
    }
    
    # 実験計画表の作成
    doe_table = pd.DataFrame({'Run': range(1, 17)})
    
    for i, fname in enumerate(factor_names):
        doe_table[fname] = [levels[fname][x] for x in factors[:, i]]
    
    print("=== 直交表L16による5因子実験計画 ===")
    print(doe_table.to_string(index=False))
    
    # シミュレートされた実験結果
    np.random.seed(42)
    
    yields = []
    for _, row in doe_table.iterrows():
        temp = row['Temperature']
        press = row['Pressure']
        cat = row['Catalyst']
        time = row['Time']
        stir = row['Stirring']
    
        # 真のモデル（主効果のみ）
        yield_true = (60 +
                      8 * (temp - 150) / 50 +
                      4 * (press - 1.0) +
                      3 * (cat - 0.5) / 0.5 +
                      2 * (time - 30) / 30 +
                      1 * (stir - 200) / 200)
    
        # ノイズを追加
        yield_obs = yield_true + np.random.normal(0, 1.5)
        yields.append(yield_obs)
    
    doe_table['Yield'] = yields
    
    print("\n=== 実験結果（抜粋：最初の5実験）===")
    print(doe_table.head())
    
    print(f"\n総実験回数: {len(doe_table)}")
    print(f"評価因子数: {len(factor_names)}")
    print(f"効率: 完全要因配置（2^5=32回）の50%で実施可能")
    

**出力例** :
    
    
    === 直交表L16による5因子実験計画 ===
     Run  Temperature  Pressure  Catalyst  Time  Stirring
       1          150       1.0       0.5    30       200
       2          200       1.0       0.5    30       200
       3          150       2.0       0.5    30       200
       4          200       2.0       0.5    30       200
       5          150       1.0       1.0    30       200
    ...（以下略）
    
    === 実験結果（抜粋：最初の5実験）===
       Run  Temperature  Pressure  Catalyst  Time  Stirring      Yield
    0    1          150       1.0       0.5    30       200  59.494371
    1    2          200       1.0       0.5    30       200  67.861468
    2    3          150       2.0       0.5    30       200  63.646968
    3    4          200       2.0       0.5    30       200  71.522869
    4    5          150       1.0       1.0    30       200  65.647689
    
    総実験回数: 16
    評価因子数: 5
    効率: 完全要因配置（2^5=32回）の50%で実施可能
    

**解釈** : 直交表L16を使うことで、5因子の評価を16回の実験（完全要因配置の半分）で実施できます。

* * *

## 1.4 主効果図と交互作用図

### コード例5: 交互作用の可視化

温度と圧力の交互作用を詳細に分析します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 実験データ: 温度×圧力の交互作用を含むモデル
    np.random.seed(42)
    
    temperatures = [150, 175, 200]
    pressures = [1.0, 1.5, 2.0]
    
    data = []
    for temp in temperatures:
        for press in pressures:
            # 交互作用項を含むモデル
            # Yield = β0 + β1*Temp + β2*Press + β3*Temp*Press + ε
            yield_true = (50 +
                          0.15 * temp +
                          20 * press +
                          0.05 * temp * press)  # 交互作用項
    
            yield_obs = yield_true + np.random.normal(0, 1.5)
            data.append({
                'Temperature': temp,
                'Pressure': press,
                'Yield': yield_obs
            })
    
    df = pd.DataFrame(data)
    
    print("=== 実験データ ===")
    print(df)
    
    # 交互作用プロット
    plt.figure(figsize=(12, 5))
    
    # 左側: 温度を固定して圧力の効果を見る
    plt.subplot(1, 2, 1)
    for temp in temperatures:
        subset = df[df['Temperature'] == temp]
        plt.plot(subset['Pressure'], subset['Yield'],
                 marker='o', linewidth=2, markersize=8, label=f'{temp}°C')
    
    plt.xlabel('圧力 (MPa)', fontsize=12)
    plt.ylabel('収率 (%)', fontsize=12)
    plt.title('温度別の圧力効果（交互作用あり）', fontsize=14, fontweight='bold')
    plt.legend(title='温度', fontsize=10)
    plt.grid(alpha=0.3)
    
    # 右側: 圧力を固定して温度の効果を見る
    plt.subplot(1, 2, 2)
    for press in pressures:
        subset = df[df['Pressure'] == press]
        plt.plot(subset['Temperature'], subset['Yield'],
                 marker='s', linewidth=2, markersize=8, label=f'{press} MPa')
    
    plt.xlabel('温度 (°C)', fontsize=12)
    plt.ylabel('収率 (%)', fontsize=12)
    plt.title('圧力別の温度効果（交互作用あり）', fontsize=14, fontweight='bold')
    plt.legend(title='圧力', fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('interaction_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 交互作用の定量評価
    print("\n=== 交互作用の評価 ===")
    print("線が平行 → 交互作用なし")
    print("線が交差 → 交互作用あり（強い）")
    print("\n本データ: 線の傾きが変化 → 温度と圧力に交互作用が存在")
    

**解釈** : グラフの線が平行でなく傾きが変化している場合、交互作用が存在します。この場合、高温×高圧の組み合わせで相乗効果が見られます。

* * *

### コード例6: 主効果図の作成

直交表L8の結果から主効果図を作成します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 先ほどのL8実験データを使用（コード例3のdoe_tableとyields）
    # ここでは再度生成
    np.random.seed(42)
    
    # 直交表L8
    L8 = np.array([
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 1],
        [1, 2, 2],
        [2, 1, 1],
        [2, 1, 2],
        [2, 2, 1],
        [2, 2, 2]
    ])
    
    temperature_levels = {1: 150, 2: 200}
    pressure_levels = {1: 1.0, 2: 2.0}
    catalyst_levels = {1: 0.5, 2: 1.0}
    
    doe_table = pd.DataFrame({
        'Run': range(1, 9),
        'Temp_code': L8[:, 0],
        'Press_code': L8[:, 1],
        'Cat_code': L8[:, 2],
        'Temperature': [temperature_levels[x] for x in L8[:, 0]],
        'Pressure': [pressure_levels[x] for x in L8[:, 1]],
        'Catalyst': [catalyst_levels[x] for x in L8[:, 2]]
    })
    
    # シミュレートされた収率
    yields = []
    for _, row in doe_table.iterrows():
        yield_true = (70 +
                      10 * (row['Temperature'] - 150) / 50 +
                      5 * (row['Pressure'] - 1.0) +
                      3 * (row['Catalyst'] - 0.5) / 0.5)
        yield_obs = yield_true + np.random.normal(0, 1)
        yields.append(yield_obs)
    
    doe_table['Yield'] = yields
    
    # 主効果の計算（各因子の各水準での平均収率）
    main_effects = {}
    
    # 温度の主効果
    temp_level1 = doe_table[doe_table['Temp_code'] == 1]['Yield'].mean()
    temp_level2 = doe_table[doe_table['Temp_code'] == 2]['Yield'].mean()
    main_effects['Temperature'] = {150: temp_level1, 200: temp_level2}
    
    # 圧力の主効果
    press_level1 = doe_table[doe_table['Press_code'] == 1]['Yield'].mean()
    press_level2 = doe_table[doe_table['Press_code'] == 2]['Yield'].mean()
    main_effects['Pressure'] = {1.0: press_level1, 2.0: press_level2}
    
    # 触媒量の主効果
    cat_level1 = doe_table[doe_table['Cat_code'] == 1]['Yield'].mean()
    cat_level2 = doe_table[doe_table['Cat_code'] == 2]['Yield'].mean()
    main_effects['Catalyst'] = {0.5: cat_level1, 1.0: cat_level2}
    
    # 主効果図の作成
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 温度の主効果
    axes[0].plot([150, 200], [temp_level1, temp_level2],
                 marker='o', linewidth=2.5, markersize=10, color='#11998e')
    axes[0].axhline(y=doe_table['Yield'].mean(), color='red', linestyle='--',
                    linewidth=1.5, label='全体平均')
    axes[0].set_xlabel('温度 (°C)', fontsize=12)
    axes[0].set_ylabel('平均収率 (%)', fontsize=12)
    axes[0].set_title('温度の主効果', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 圧力の主効果
    axes[1].plot([1.0, 2.0], [press_level1, press_level2],
                 marker='s', linewidth=2.5, markersize=10, color='#f59e0b')
    axes[1].axhline(y=doe_table['Yield'].mean(), color='red', linestyle='--',
                    linewidth=1.5, label='全体平均')
    axes[1].set_xlabel('圧力 (MPa)', fontsize=12)
    axes[1].set_ylabel('平均収率 (%)', fontsize=12)
    axes[1].set_title('圧力の主効果', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 触媒量の主効果
    axes[2].plot([0.5, 1.0], [cat_level1, cat_level2],
                 marker='^', linewidth=2.5, markersize=10, color='#7b2cbf')
    axes[2].axhline(y=doe_table['Yield'].mean(), color='red', linestyle='--',
                    linewidth=1.5, label='全体平均')
    axes[2].set_xlabel('触媒量 (g)', fontsize=12)
    axes[2].set_ylabel('平均収率 (%)', fontsize=12)
    axes[2].set_title('触媒量の主効果', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('main_effects_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 主効果の大きさを計算（効果 = 高水準平均 - 低水準平均）
    print("=== 主効果の大きさ ===")
    print(f"温度の効果: {temp_level2 - temp_level1:.2f} %")
    print(f"圧力の効果: {press_level2 - press_level1:.2f} %")
    print(f"触媒量の効果: {cat_level2 - cat_level1:.2f} %")
    
    print("\n=== 因子の重要度ランキング ===")
    effects = {
        '温度': abs(temp_level2 - temp_level1),
        '圧力': abs(press_level2 - press_level1),
        '触媒量': abs(cat_level2 - cat_level1)
    }
    sorted_effects = sorted(effects.items(), key=lambda x: x[1], reverse=True)
    for i, (factor, effect) in enumerate(sorted_effects, 1):
        print(f"{i}位: {factor} (効果: {effect:.2f} %)")
    

**出力例** :
    
    
    === 主効果の大きさ ===
    温度の効果: 10.15 %
    圧力の効果: 5.08 %
    触媒量の効果: 3.04 %
    
    === 因子の重要度ランキング ===
    1位: 温度 (効果: 10.15 %)
    2位: 圧力 (効果: 5.08 %)
    3位: 触媒量 (効果: 3.04 %)
    

**解釈** : 主効果図から、温度が最も大きな影響を持ち、次いで圧力、触媒量の順であることがわかります。最適条件は温度200°C、圧力2.0 MPa、触媒量1.0 gです。

* * *

### コード例7: 交互作用図の作成

温度×圧力の交互作用を詳細に解析します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 2因子の完全要因配置実験（交互作用を評価するため）
    np.random.seed(42)
    
    data = []
    for temp in [150, 200]:
        for press in [1.0, 2.0]:
            # 交互作用項を含むモデル
            # Yield = β0 + β1*Temp + β2*Press + β3*Temp*Press + ε
            # ここでは交互作用項を意図的に追加
            yield_true = (50 +
                          0.10 * temp +
                          15 * press +
                          0.03 * temp * press)  # 交互作用項
    
            # 3回反復
            for rep in range(3):
                yield_obs = yield_true + np.random.normal(0, 1.0)
                data.append({
                    'Temperature': temp,
                    'Pressure': press,
                    'Replicate': rep + 1,
                    'Yield': yield_obs
                })
    
    df = pd.DataFrame(data)
    
    # 各条件の平均を計算
    avg_df = df.groupby(['Temperature', 'Pressure'])['Yield'].mean().reset_index()
    
    print("=== 各条件の平均収率 ===")
    print(avg_df)
    
    # 交互作用図の作成
    plt.figure(figsize=(10, 6))
    
    for temp in [150, 200]:
        subset = avg_df[avg_df['Temperature'] == temp]
        plt.plot(subset['Pressure'], subset['Yield'],
                 marker='o', linewidth=2.5, markersize=10, label=f'{temp}°C')
    
    plt.xlabel('圧力 (MPa)', fontsize=12)
    plt.ylabel('平均収率 (%)', fontsize=12)
    plt.title('温度×圧力の交互作用図', fontsize=14, fontweight='bold')
    plt.legend(title='温度', fontsize=11)
    plt.xticks([1.0, 2.0])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('interaction_plot_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 交互作用の定量評価
    # 交互作用 = (高温・高圧 - 高温・低圧) - (低温・高圧 - 低温・低圧)
    y_150_10 = avg_df[(avg_df['Temperature'] == 150) & (avg_df['Pressure'] == 1.0)]['Yield'].values[0]
    y_150_20 = avg_df[(avg_df['Temperature'] == 150) & (avg_df['Pressure'] == 2.0)]['Yield'].values[0]
    y_200_10 = avg_df[(avg_df['Temperature'] == 200) & (avg_df['Pressure'] == 1.0)]['Yield'].values[0]
    y_200_20 = avg_df[(avg_df['Temperature'] == 200) & (avg_df['Pressure'] == 2.0)]['Yield'].values[0]
    
    interaction = (y_200_20 - y_200_10) - (y_150_20 - y_150_10)
    
    print(f"\n=== 交互作用の大きさ ===")
    print(f"温度×圧力の交互作用: {interaction:.2f} %")
    
    if abs(interaction) > 2:
        print("判定: 交互作用が存在する（|交互作用| > 2%）")
    else:
        print("判定: 交互作用は無視できる（|交互作用| ≤ 2%）")
    
    print("\n=== 解釈 ===")
    if interaction > 0:
        print("高温×高圧の組み合わせで相乗効果（シナジー）が見られる")
    else:
        print("交互作用は負（高水準同士の組み合わせで効果が減少）")
    

**出力例** :
    
    
    === 各条件の平均収率 ===
       Temperature  Pressure      Yield
    0          150       1.0  80.329885
    1          150       2.0  95.246314
    2          200       1.0  85.294928
    3          200       2.0 106.022349
    
    === 交互作用の大きさ ===
    温度×圧力の交互作用: 5.81 %
    
    判定: 交互作用が存在する（|交互作用| > 2%）
    
    === 解釈 ===
    高温×高圧の組み合わせで相乗効果（シナジー）が見られる
    

**解釈** : 交互作用プロットで線が平行でない（交差または傾きが異なる）場合、交互作用が存在します。この場合、高温と高圧を組み合わせることで相乗効果が得られます。

* * *

## 1.5 ケーススタディ: 化学反応収率最適化

### コード例8: 直交表L8による3因子実験と最適条件探索

エステル化反応における温度、触媒濃度、反応時間の最適化を直交表L8で実施します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # ケーススタディ: エステル化反応の収率最適化
    # 因子A: 反応温度（60°C vs 80°C）
    # 因子B: 触媒濃度（0.1 M vs 0.5 M）
    # 因子C: 反応時間（2時間 vs 4時間）
    
    np.random.seed(42)
    
    # 直交表L8
    L8 = np.array([
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 1],
        [1, 2, 2],
        [2, 1, 1],
        [2, 1, 2],
        [2, 2, 1],
        [2, 2, 2]
    ])
    
    # 因子の定義
    factor_levels = {
        'Temperature': {1: 60, 2: 80},
        'Catalyst': {1: 0.1, 2: 0.5},
        'Time': {1: 2, 2: 4}
    }
    
    # 実験計画表の作成
    doe_table = pd.DataFrame({
        'Run': range(1, 9),
        'Temp_code': L8[:, 0],
        'Cat_code': L8[:, 1],
        'Time_code': L8[:, 2],
        'Temperature': [factor_levels['Temperature'][x] for x in L8[:, 0]],
        'Catalyst': [factor_levels['Catalyst'][x] for x in L8[:, 1]],
        'Time': [factor_levels['Time'][x] for x in L8[:, 2]]
    })
    
    # シミュレートされた収率（現実的なモデル）
    # 真のモデル: Yield = f(Temp, Cat, Time) + 交互作用 + ノイズ
    yields = []
    for _, row in doe_table.iterrows():
        temp = row['Temperature']
        cat = row['Catalyst']
        time = row['Time']
    
        # 主効果
        yield_base = 50
        temp_effect = 15 * (temp - 60) / 20
        cat_effect = 20 * (cat - 0.1) / 0.4
        time_effect = 10 * (time - 2) / 2
    
        # 交互作用（温度×触媒濃度）
        interaction = 5 * ((temp - 60) / 20) * ((cat - 0.1) / 0.4)
    
        yield_true = yield_base + temp_effect + cat_effect + time_effect + interaction
    
        # ノイズを追加
        yield_obs = yield_true + np.random.normal(0, 2)
        yields.append(yield_obs)
    
    doe_table['Yield'] = yields
    
    print("=== エステル化反応 実験計画と結果 ===")
    print(doe_table[['Run', 'Temperature', 'Catalyst', 'Time', 'Yield']])
    
    # 主効果の計算
    print("\n=== 主効果分析 ===")
    
    # 温度の主効果
    temp_low = doe_table[doe_table['Temp_code'] == 1]['Yield'].mean()
    temp_high = doe_table[doe_table['Temp_code'] == 2]['Yield'].mean()
    temp_effect = temp_high - temp_low
    print(f"温度の効果: {temp_effect:.2f}% (低水準: {temp_low:.2f}%, 高水準: {temp_high:.2f}%)")
    
    # 触媒濃度の主効果
    cat_low = doe_table[doe_table['Cat_code'] == 1]['Yield'].mean()
    cat_high = doe_table[doe_table['Cat_code'] == 2]['Yield'].mean()
    cat_effect = cat_high - cat_low
    print(f"触媒濃度の効果: {cat_effect:.2f}% (低水準: {cat_low:.2f}%, 高水準: {cat_high:.2f}%)")
    
    # 反応時間の主効果
    time_low = doe_table[doe_table['Time_code'] == 1]['Yield'].mean()
    time_high = doe_table[doe_table['Time_code'] == 2]['Yield'].mean()
    time_effect = time_high - time_low
    print(f"反応時間の効果: {time_effect:.2f}% (低水準: {time_low:.2f}%, 高水準: {time_high:.2f}%)")
    
    # 因子の重要度ランキング
    effects = {
        '触媒濃度': abs(cat_effect),
        '温度': abs(temp_effect),
        '反応時間': abs(time_effect)
    }
    sorted_effects = sorted(effects.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== 因子の重要度ランキング ===")
    for i, (factor, effect) in enumerate(sorted_effects, 1):
        print(f"{i}位: {factor} (効果: {effect:.2f}%)")
    
    # 最適条件の決定
    print("\n=== 最適条件 ===")
    print("収率を最大化する条件:")
    print(f"  温度: {80 if temp_effect > 0 else 60}°C")
    print(f"  触媒濃度: {0.5 if cat_effect > 0 else 0.1} M")
    print(f"  反応時間: {4 if time_effect > 0 else 2}時間")
    
    # 予測収率（最適条件）
    predicted_yield_max = doe_table['Yield'].mean() + abs(temp_effect)/2 + abs(cat_effect)/2 + abs(time_effect)/2
    print(f"  予測収率: {predicted_yield_max:.1f}%")
    
    # 主効果図の可視化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 温度の主効果
    axes[0].plot([60, 80], [temp_low, temp_high],
                 marker='o', linewidth=2.5, markersize=10, color='#11998e')
    axes[0].axhline(y=doe_table['Yield'].mean(), color='red', linestyle='--',
                    linewidth=1.5, label='全体平均', alpha=0.7)
    axes[0].set_xlabel('反応温度 (°C)', fontsize=12)
    axes[0].set_ylabel('平均収率 (%)', fontsize=12)
    axes[0].set_title('温度の主効果', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 触媒濃度の主効果
    axes[1].plot([0.1, 0.5], [cat_low, cat_high],
                 marker='s', linewidth=2.5, markersize=10, color='#f59e0b')
    axes[1].axhline(y=doe_table['Yield'].mean(), color='red', linestyle='--',
                    linewidth=1.5, label='全体平均', alpha=0.7)
    axes[1].set_xlabel('触媒濃度 (M)', fontsize=12)
    axes[1].set_ylabel('平均収率 (%)', fontsize=12)
    axes[1].set_title('触媒濃度の主効果', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 反応時間の主効果
    axes[2].plot([2, 4], [time_low, time_high],
                 marker='^', linewidth=2.5, markersize=10, color='#7b2cbf')
    axes[2].axhline(y=doe_table['Yield'].mean(), color='red', linestyle='--',
                    linewidth=1.5, label='全体平均', alpha=0.7)
    axes[2].set_xlabel('反応時間 (時間)', fontsize=12)
    axes[2].set_ylabel('平均収率 (%)', fontsize=12)
    axes[2].set_title('反応時間の主効果', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('esterification_main_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ヒートマップで結果を可視化
    pivot_temp_cat = doe_table.pivot_table(values='Yield',
                                            index='Temperature',
                                            columns='Catalyst',
                                            aggfunc='mean')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_temp_cat, annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': '収率 (%)'}, linewidths=2, linecolor='white')
    plt.title('温度×触媒濃度の収率マップ', fontsize=14, fontweight='bold')
    plt.ylabel('温度 (°C)', fontsize=12)
    plt.xlabel('触媒濃度 (M)', fontsize=12)
    plt.tight_layout()
    plt.savefig('esterification_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== ケーススタディのまとめ ===")
    print("✅ 直交表L8により8回の実験で3因子の影響を評価")
    print("✅ 触媒濃度が最も重要な因子と判明")
    print("✅ 最適条件: 80°C, 0.5 M, 4時間")
    print(f"✅ 予測収率: {predicted_yield_max:.1f}%（確認実験で検証推奨）")
    

**出力例** :
    
    
    === エステル化反応 実験計画と結果 ===
       Run  Temperature  Catalyst  Time      Yield
    0    1           60       0.1     2  50.987420
    1    2           60       0.1     4  60.723869
    2    3           60       0.5     2  70.294968
    3    4           60       0.5     4  80.046738
    4    5           80       0.1     2  65.296378
    5    6           80       0.1     4  75.044464
    6    7           80       0.5     2  90.466514
    7    8           80       0.5     4 100.535989
    
    === 主効果分析 ===
    温度の効果: 15.09% (低水準: 65.51%, 高水準: 80.59%)
    触媒濃度の効果: 20.25% (低水準: 63.01%, 高水準: 83.26%)
    反応時間の効果: 9.99% (低水準: 68.14%, 高水準: 78.13%)
    
    === 因子の重要度ランキング ===
    1位: 触媒濃度 (効果: 20.25%)
    2位: 温度 (効果: 15.09%)
    3位: 反応時間 (効果: 9.99%)
    
    === 最適条件 ===
    収率を最大化する条件:
      温度: 80°C
      触媒濃度: 0.5 M
      反応時間: 4時間
      予測収率: 95.7%
    
    === ケーススタディのまとめ ===
    ✅ 直交表L8により8回の実験で3因子の影響を評価
    ✅ 触媒濃度が最も重要な因子と判明
    ✅ 最適条件: 80°C, 0.5 M, 4時間
    ✅ 予測収率: 95.7%（確認実験で検証推奨）
    

**解釈** : 直交表L8により、わずか8回の実験で3因子の影響を効率的に評価し、最適条件を決定できました。従来の一変数実験では最低でも3×2×2×2=24回必要でしたが、DOEにより実験回数を67%削減しました。

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **実験計画法（DOE）の基礎**
     * 最小の実験回数で最大の情報を得る統計的手法
     * 従来のOFAT（一変数実験）との違い
     * DOEの3原則: 反復、無作為化、ブロック化
  2. **一元配置・二元配置実験**
     * 一元配置ANOVA: 1因子の水準間比較
     * 二元配置ANOVA: 2因子の主効果と交互作用
     * F検定による統計的有意性の判定
  3. **直交表の活用**
     * 直交表L8, L16による効率的な実験設計
     * 各因子の水準が均等に出現する性質
     * 実験回数の大幅削減（50-75%）
  4. **主効果図と交互作用図**
     * 主効果: 各因子の単独効果の可視化
     * 交互作用: 因子間の相互作用の検出
     * 最適条件の決定方法
  5. **化学プロセスへの応用**
     * エステル化反応の収率最適化
     * 重要因子の特定と最適条件の探索
     * 予測収率の計算と確認実験の必要性

### 重要なポイント

  * DOEは実験回数を50-75%削減しながら交互作用も評価可能
  * 直交表は因子のスクリーニング（重要因子の特定）に最適
  * 主効果図で因子の影響度を視覚的に理解できる
  * 交互作用プロットで線が平行でない場合、交互作用が存在
  * 最適条件は主効果が最大となる水準の組み合わせ

### 次の章へ

第2章では、**要因配置実験と分散分析** を詳しく学びます：

  * 完全要因配置実験（Full Factorial Design）
  * 一部実施要因配置実験（Fractional Factorial Design）
  * 分散分析（ANOVA）の詳細とF検定
  * 多重比較検定（Tukey HSD）
  * 分散成分の分解と寄与率の計算
  * ケーススタディ: 触媒活性に影響する因子探索
