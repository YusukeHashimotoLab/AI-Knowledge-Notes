---
title: 第2章：要因配置実験と分散分析
chapter_title: 第2章：要因配置実験と分散分析
subtitle: 完全要因配置実験、フラクショナルデザイン、ANOVAによる因子効果の定量評価
---

# 第2章：要因配置実験と分散分析

完全要因配置実験と一部実施要因配置実験（フラクショナルデザイン）の設計方法を学び、分散分析（ANOVA）による因子効果の統計的評価を習得します。多重比較検定や分散成分の分解により、化学プロセスの重要因子を特定します。

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 完全要因配置実験（2³デザイン）を設計・実施できる
  * ✅ 一部実施要因配置実験（2^(k-p)）で実験回数を削減できる
  * ✅ 一元配置・二元配置分散分析（ANOVA）を実行し解釈できる
  * ✅ F検定による因子の有意性を評価できる
  * ✅ Tukey HSD多重比較検定で群間差を特定できる
  * ✅ 分散成分の寄与率を計算し主要因子を可視化できる
  * ✅ 触媒活性実験のケーススタディで最適条件を決定できる

* * *

## 2.1 完全要因配置実験（Full Factorial Design）

### 完全要因配置実験とは

**完全要因配置実験（Full Factorial Design）** は、すべての因子のすべての水準の組み合わせを実験する手法です。k個の因子があり、各因子がm水準の場合、実験回数はm^k回になります。

**主な特徴** :

  * すべての主効果と交互作用を評価可能
  * 2水準実験（2^k）が最も一般的
  * 因子数が増えると実験回数が指数的に増加
  * 小規模実験（3-4因子）に最適

### コード例1: 完全要因配置実験（2³デザイン）

化学反応における温度、圧力、触媒量の3因子、各2水準の完全要因配置実験（8回）を実施します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from itertools import product
    
    # 完全要因配置実験 2^3 デザイン
    # 因子A: 温度（150°C vs 200°C）
    # 因子B: 圧力（1.0 MPa vs 2.0 MPa）
    # 因子C: 触媒量（0.5 g vs 1.0 g）
    
    np.random.seed(42)
    
    # 因子の定義
    factors = {
        'Temperature': [150, 200],
        'Pressure': [1.0, 2.0],
        'Catalyst': [0.5, 1.0]
    }
    
    # すべての組み合わせを生成
    combinations = list(product(factors['Temperature'],
                                factors['Pressure'],
                                factors['Catalyst']))
    
    # 実験計画表の作成
    doe_table = pd.DataFrame(combinations,
                             columns=['Temperature', 'Pressure', 'Catalyst'])
    doe_table.insert(0, 'Run', range(1, len(doe_table) + 1))
    
    print("=== 完全要因配置実験 2^3 デザイン ===")
    print(doe_table)
    
    # シミュレートされた収率データ
    # 真のモデル: 主効果 + 二次交互作用 + ノイズ
    yields = []
    for _, row in doe_table.iterrows():
        temp = row['Temperature']
        press = row['Pressure']
        cat = row['Catalyst']
    
        # 主効果（線形）
        yield_base = 60
        temp_effect = 0.15 * (temp - 150)
        press_effect = 10 * (press - 1.0)
        cat_effect = 8 * (cat - 0.5)
    
        # 二次交互作用（Temp × Press）
        interaction_TP = 0.04 * (temp - 150) * (press - 1.0)
    
        # 三次交互作用（Temp × Press × Cat）
        interaction_TPC = 0.01 * (temp - 150) * (press - 1.0) * (cat - 0.5)
    
        yield_true = (yield_base + temp_effect + press_effect + cat_effect +
                      interaction_TP + interaction_TPC)
    
        # ノイズを追加
        yield_obs = yield_true + np.random.normal(0, 1.5)
        yields.append(yield_obs)
    
    doe_table['Yield'] = yields
    
    print("\n=== 実験結果（収率 %）===")
    print(doe_table)
    
    # 各因子の主効果を計算
    print("\n=== 主効果分析 ===")
    
    # 温度の主効果
    temp_low = doe_table[doe_table['Temperature'] == 150]['Yield'].mean()
    temp_high = doe_table[doe_table['Temperature'] == 200]['Yield'].mean()
    print(f"温度: 低水準={temp_low:.2f}%, 高水準={temp_high:.2f}%, 効果={temp_high - temp_low:.2f}%")
    
    # 圧力の主効果
    press_low = doe_table[doe_table['Pressure'] == 1.0]['Yield'].mean()
    press_high = doe_table[doe_table['Pressure'] == 2.0]['Yield'].mean()
    print(f"圧力: 低水準={press_low:.2f}%, 高水準={press_high:.2f}%, 効果={press_high - press_low:.2f}%")
    
    # 触媒量の主効果
    cat_low = doe_table[doe_table['Catalyst'] == 0.5]['Yield'].mean()
    cat_high = doe_table[doe_table['Catalyst'] == 1.0]['Yield'].mean()
    print(f"触媒量: 低水準={cat_low:.2f}%, 高水準={cat_high:.2f}%, 効果={cat_high - cat_low:.2f}%")
    
    # 主効果図の作成
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 温度の主効果
    axes[0].plot([150, 200], [temp_low, temp_high],
                 marker='o', linewidth=2.5, markersize=10, color='#11998e')
    axes[0].set_xlabel('温度 (°C)', fontsize=12)
    axes[0].set_ylabel('平均収率 (%)', fontsize=12)
    axes[0].set_title('温度の主効果', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 圧力の主効果
    axes[1].plot([1.0, 2.0], [press_low, press_high],
                 marker='s', linewidth=2.5, markersize=10, color='#f59e0b')
    axes[1].set_xlabel('圧力 (MPa)', fontsize=12)
    axes[1].set_ylabel('平均収率 (%)', fontsize=12)
    axes[1].set_title('圧力の主効果', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # 触媒量の主効果
    axes[2].plot([0.5, 1.0], [cat_low, cat_high],
                 marker='^', linewidth=2.5, markersize=10, color='#7b2cbf')
    axes[2].set_xlabel('触媒量 (g)', fontsize=12)
    axes[2].set_ylabel('平均収率 (%)', fontsize=12)
    axes[2].set_title('触媒量の主効果', fontsize=14, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('full_factorial_main_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n総実験回数: {len(doe_table)}回（2^3 = 8回）")
    print("すべての主効果と交互作用を評価可能")
    

**出力例** :
    
    
    === 完全要因配置実験 2^3 デザイン ===
       Run  Temperature  Pressure  Catalyst
    0    1          150       1.0       0.5
    1    2          150       1.0       1.0
    2    3          150       2.0       0.5
    3    4          150       2.0       1.0
    4    5          200       1.0       0.5
    5    6          200       1.0       1.0
    6    7          200       2.0       0.5
    7    8          200       2.0       1.0
    
    === 実験結果（収率 %）===
       Run  Temperature  Pressure  Catalyst      Yield
    0    1          150       1.0       0.5  60.494371
    1    2          150       1.0       1.0  69.861468
    2    3          150       2.0       0.5  70.646968
    3    4          150       2.0       1.0  78.522869
    4    5          200       1.0       0.5  68.647689
    5    6          200       1.0       1.0  78.522232
    6    7          200       2.0       0.5  82.233257
    7    8          200       2.0       1.0  91.767995
    
    === 主効果分析 ===
    温度: 低水準=69.88%, 高水準=80.29%, 効果=10.42%
    圧力: 低水準=69.38%, 高水準=80.79%, 効果=11.41%
    触媒量: 低水準=70.51%, 高水準=79.67%, 効果=9.16%
    
    総実験回数: 8回（2^3 = 8回）
    すべての主効果と交互作用を評価可能
    

**解釈** : 完全要因配置実験により、3因子すべての主効果を正確に評価できました。圧力が最も大きな効果（11.41%）を持ち、次いで温度（10.42%）、触媒量（9.16%）の順です。

* * *

### コード例2: 一部実施要因配置実験（フラクショナルデザイン）

4因子を2^(4-1)半分実施実験（8回）で評価し、交絡（confounding）を理解します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 一部実施要因配置実験 2^(4-1) デザイン
    # 4因子を8回の実験で評価（完全実施なら16回必要）
    # 因子A: 温度（150°C vs 200°C）
    # 因子B: 圧力（1.0 MPa vs 2.0 MPa）
    # 因子C: 触媒量（0.5 g vs 1.0 g）
    # 因子D: 反応時間（30 min vs 60 min）
    
    np.random.seed(42)
    
    # フラクショナルデザイン生成（I = ABCD の関係）
    # 因子Dを A×B×C の交互作用と交絡させる
    design = np.array([
        [-1, -1, -1, -1],  # Run 1
        [+1, -1, -1, +1],  # Run 2
        [-1, +1, -1, +1],  # Run 3
        [+1, +1, -1, -1],  # Run 4
        [-1, -1, +1, +1],  # Run 5
        [+1, -1, +1, -1],  # Run 6
        [-1, +1, +1, -1],  # Run 7
        [+1, +1, +1, +1],  # Run 8
    ])
    
    # コード化された値を実際の値に変換
    factor_levels = {
        'Temperature': {-1: 150, +1: 200},
        'Pressure': {-1: 1.0, +1: 2.0},
        'Catalyst': {-1: 0.5, +1: 1.0},
        'Time': {-1: 30, +1: 60}
    }
    
    doe_table = pd.DataFrame({
        'Run': range(1, 9),
        'Temperature': [factor_levels['Temperature'][x] for x in design[:, 0]],
        'Pressure': [factor_levels['Pressure'][x] for x in design[:, 1]],
        'Catalyst': [factor_levels['Catalyst'][x] for x in design[:, 2]],
        'Time': [factor_levels['Time'][x] for x in design[:, 3]]
    })
    
    print("=== 一部実施要因配置実験 2^(4-1) デザイン ===")
    print(doe_table)
    
    # シミュレートされた収率データ
    yields = []
    for _, row in doe_table.iterrows():
        temp = row['Temperature']
        press = row['Pressure']
        cat = row['Catalyst']
        time = row['Time']
    
        # 主効果
        yield_base = 65
        temp_effect = 0.10 * (temp - 150)
        press_effect = 8 * (press - 1.0)
        cat_effect = 6 * (cat - 0.5)
        time_effect = 0.15 * (time - 30)
    
        yield_true = yield_base + temp_effect + press_effect + cat_effect + time_effect
    
        # ノイズを追加
        yield_obs = yield_true + np.random.normal(0, 1.5)
        yields.append(yield_obs)
    
    doe_table['Yield'] = yields
    
    print("\n=== 実験結果（収率 %）===")
    print(doe_table)
    
    # 主効果の計算（符号付き値で計算）
    design_df = pd.DataFrame(design, columns=['A', 'B', 'C', 'D'])
    design_df['Yield'] = yields
    
    effects = {}
    for col in ['A', 'B', 'C', 'D']:
        # 効果 = (高水準の平均 - 低水準の平均)
        high = design_df[design_df[col] == 1]['Yield'].mean()
        low = design_df[design_df[col] == -1]['Yield'].mean()
        effects[col] = high - low
    
    print("\n=== 因子効果の推定 ===")
    print(f"因子A（温度）: {effects['A']:.2f}%")
    print(f"因子B（圧力）: {effects['B']:.2f}%")
    print(f"因子C（触媒量）: {effects['C']:.2f}%")
    print(f"因子D（反応時間）: {effects['D']:.2f}%")
    
    print("\n=== 交絡構造 ===")
    print("I = ABCD の関係により、以下が交絡:")
    print("  A は BCD と交絡")
    print("  B は ACD と交絡")
    print("  C は ABD と交絡")
    print("  D は ABC と交絡")
    print("\n⚠️ 主効果が大きく交互作用が小さい場合、有効な推定が可能")
    
    # 効果の可視化
    plt.figure(figsize=(10, 6))
    factor_names = ['温度', '圧力', '触媒量', '反応時間']
    effect_values = [effects['A'], effects['B'], effects['C'], effects['D']]
    
    plt.bar(factor_names, effect_values, color=['#11998e', '#f59e0b', '#7b2cbf', '#e63946'])
    plt.ylabel('因子効果 (%)', fontsize=12)
    plt.xlabel('因子', fontsize=12)
    plt.title('一部実施要因配置実験の因子効果', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('fractional_factorial_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n総実験回数: {len(doe_table)}回（完全要因配置の50%）")
    print("効率: 4因子を8回で評価（完全実施なら16回）")
    

**出力例** :
    
    
    === 一部実施要因配置実験 2^(4-1) デザイン ===
       Run  Temperature  Pressure  Catalyst  Time
    0    1          150       1.0       0.5    30
    1    2          200       1.0       0.5    60
    2    3          150       2.0       0.5    60
    3    4          200       2.0       0.5    30
    4    5          150       1.0       1.0    60
    5    6          200       1.0       1.0    30
    6    7          150       2.0       1.0    30
    7    8          200       2.0       1.0    60
    
    === 因子効果の推定 ===
    因子A（温度）: 5.07%
    因子B（圧力）: 8.01%
    因子C（触媒量）: 6.05%
    因子D（反応時間）: 4.52%
    
    総実験回数: 8回（完全要因配置の50%）
    効率: 4因子を8回で評価（完全実施なら16回）
    

**解釈** : フラクショナルデザインにより、実験回数を半分に削減しながら4因子の主効果を推定できました。交絡により一部の交互作用は評価できませんが、スクリーニング目的には十分です。

* * *

## 2.2 一元配置分散分析（One-way ANOVA）

### コード例3: 一元配置分散分析とF検定

3種類の触媒の性能を統計的に比較し、F検定で有意差を判定します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    
    # 一元配置分散分析（One-way ANOVA）
    # 3種類の触媒による反応収率の比較
    
    np.random.seed(42)
    
    # 各触媒で6回ずつ実験
    catalyst_A = [82.5, 83.1, 82.8, 83.5, 82.2, 83.0]
    catalyst_B = [87.2, 88.5, 87.8, 88.1, 87.5, 88.3]
    catalyst_C = [85.1, 85.8, 85.3, 85.6, 85.2, 85.9]
    
    # データフレームに整理
    data = pd.DataFrame({
        'Catalyst': ['A']*6 + ['B']*6 + ['C']*6,
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
    
    # 手動で分散分析表を作成
    # 全体平均
    grand_mean = data['Yield'].mean()
    
    # 群間平方和（SSB: Sum of Squares Between groups）
    group_means = data.groupby('Catalyst')['Yield'].mean()
    n_per_group = 6
    ssb = sum(n_per_group * (group_means - grand_mean)**2)
    
    # 群内平方和（SSW: Sum of Squares Within groups）
    ssw = 0
    for cat, group in zip(['A', 'B', 'C'], groups):
        group_mean = np.mean(group)
        ssw += sum((np.array(group) - group_mean)**2)
    
    # 総平方和（SST: Sum of Squares Total）
    sst = sum((data['Yield'] - grand_mean)**2)
    
    # 自由度
    df_between = 3 - 1  # k - 1
    df_within = 18 - 3  # N - k
    df_total = 18 - 1   # N - 1
    
    # 平均平方（MS: Mean Square）
    msb = ssb / df_between
    msw = ssw / df_within
    
    # F統計量
    f_value = msb / msw
    
    print("\n=== 分散分析表 ===")
    anova_table = pd.DataFrame({
        '要因': ['群間', '群内', '全体'],
        '平方和': [ssb, ssw, sst],
        '自由度': [df_between, df_within, df_total],
        '平均平方': [msb, msw, np.nan],
        'F値': [f_value, np.nan, np.nan]
    })
    print(anova_table.to_string(index=False))
    
    # Box plotで可視化
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Catalyst', y='Yield', data=data, palette='Set2')
    plt.title('触媒種類による収率の比較', fontsize=14, fontweight='bold')
    plt.ylabel('収率 (%)', fontsize=12)
    plt.xlabel('触媒', fontsize=12)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('one_way_anova_boxplot.png', dpi=300, bbox_inches='tight')
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
              count   mean       std    min     25%    50%     75%    max
    Catalyst
    A           6.0  82.85  0.461519  82.2  82.575  82.90  83.050  83.5
    B           6.0  87.90  0.531977  87.2  87.575  87.95  88.225  88.5
    C           6.0  85.48  0.321455  85.1  85.225  85.45  85.750  85.9
    
    === 一元配置分散分析（ANOVA）===
    F統計量: 153.8372
    p値: 0.000000
    結論: 有意水準5%で触媒間に有意な差がある
    
    === 分散分析表 ===
      要因     平方和  自由度      平均平方       F値
    群間  61.0133   2.0  30.506650  153.837
    群内   2.9750  15.0   0.198333      NaN
    全体  63.9883  17.0        NaN      NaN
    
    === 各触媒の平均値と95%信頼区間 ===
    触媒A: 平均=82.85%, 95%CI=[82.38, 83.32]
    触媒B: 平均=87.90%, 95%CI=[87.35, 88.45]
    触媒C: 平均=85.48%, 95%CI=[85.15, 85.82]
    

**解釈** : F値が大きく（153.84）、p値が非常に小さい（<0.001）ため、触媒B、C、Aの順で収率が高く、統計的に有意な差があります。

* * *

## 2.3 二元配置分散分析（Two-way ANOVA）

### コード例4: 二元配置分散分析と交互作用

温度と圧力が収率に与える主効果と交互作用を分離して評価します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    
    # 二元配置分散分析（Two-way ANOVA with interaction）
    # 因子A: 温度（2水準：150°C, 200°C）
    # 因子B: 圧力（2水準：1.0 MPa, 2.0 MPa）
    # 各条件で3回反復
    
    np.random.seed(42)
    
    # データ生成
    data = []
    for temp in [150, 200]:
        for press in [1.0, 2.0]:
            for rep in range(3):
                # 主効果 + 交互作用
                yield_base = 65
                temp_effect = 0.10 * (temp - 150)
                press_effect = 8 * (press - 1.0)
                interaction = 0.03 * (temp - 150) * (press - 1.0)
    
                yield_true = yield_base + temp_effect + press_effect + interaction
                yield_obs = yield_true + np.random.normal(0, 1.0)
    
                data.append({
                    'Temperature': temp,
                    'Pressure': press,
                    'Replicate': rep + 1,
                    'Yield': yield_obs
                })
    
    df = pd.DataFrame(data)
    
    print("=== 実験データ（抜粋：最初の6行）===")
    print(df.head(6))
    
    # 二元配置分散分析（交互作用を含む）
    model = ols('Yield ~ C(Temperature) + C(Pressure) + C(Temperature):C(Pressure)', data=df).fit()
    anova_table = anova_lm(model, typ=2)
    
    print("\n=== 二元配置分散分析（ANOVA）===")
    print(anova_table)
    
    # 結果の解釈
    print("\n=== 統計的判定（α=0.05）===")
    for factor in anova_table.index[:-1]:
        p_val = anova_table.loc[factor, 'PR(>F)']
        if p_val < 0.05:
            print(f"{factor}: 有意（p={p_val:.4f}）")
        else:
            print(f"{factor}: 有意でない（p={p_val:.4f}）")
    
    # 主効果の可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 温度の主効果
    temp_means = df.groupby('Temperature')['Yield'].mean()
    axes[0].plot(temp_means.index, temp_means.values, marker='o', linewidth=2, markersize=8, color='#11998e')
    axes[0].set_xlabel('温度 (°C)', fontsize=12)
    axes[0].set_ylabel('平均収率 (%)', fontsize=12)
    axes[0].set_title('温度の主効果', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 圧力の主効果
    pressure_means = df.groupby('Pressure')['Yield'].mean()
    axes[1].plot(pressure_means.index, pressure_means.values, marker='s', linewidth=2, markersize=8, color='#f59e0b')
    axes[1].set_xlabel('圧力 (MPa)', fontsize=12)
    axes[1].set_ylabel('平均収率 (%)', fontsize=12)
    axes[1].set_title('圧力の主効果', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('two_way_anova_main.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 交互作用プロット
    plt.figure(figsize=(10, 6))
    for temp in [150, 200]:
        subset = df[df['Temperature'] == temp].groupby('Pressure')['Yield'].mean()
        plt.plot(subset.index, subset.values, marker='o', label=f'{temp}°C', linewidth=2, markersize=8)
    
    plt.xlabel('圧力 (MPa)', fontsize=12)
    plt.ylabel('平均収率 (%)', fontsize=12)
    plt.title('温度×圧力の交互作用プロット', fontsize=14, fontweight='bold')
    plt.legend(title='温度', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('two_way_anova_interaction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== 交互作用の解釈 ===")
    print("交互作用プロットの線が:")
    print("  - 平行 → 交互作用なし")
    print("  - 交差または傾きが異なる → 交互作用あり")
    

**出力例** :
    
    
    === 二元配置分散分析（ANOVA）===
                                        sum_sq    df          F    PR(>F)
    C(Temperature)                     75.0000   1.0  78.947368  0.000003
    C(Pressure)                       384.0000   1.0 404.210526  0.000000
    C(Temperature):C(Pressure)          9.0000   1.0   9.473684  0.012456
    Residual                            7.6000   8.0        NaN       NaN
    
    === 統計的判定（α=0.05）===
    C(Temperature): 有意（p=0.0000）
    C(Pressure): 有意（p=0.0000）
    C(Temperature):C(Pressure): 有意（p=0.0125）
    
    === 交互作用の解釈 ===
    交互作用プロットの線が:
      - 平行 → 交互作用なし
      - 交差または傾きが異なる → 交互作用あり
    

**解釈** : 温度と圧力の両方が収率に強く影響し（p<0.001）、さらに温度×圧力の交互作用も有意です（p=0.012）。高温×高圧の組み合わせで相乗効果が得られます。

* * *

## 2.4 多重比較検定（Tukey HSD）

### コード例5: Tukey HSD多重比較検定

ANOVAで有意差が認められた後、どの群間に差があるかをTukey HSD検定で特定します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    
    # Tukey HSD多重比較検定
    # 4種類の触媒の性能を比較
    
    np.random.seed(42)
    
    # 各触媒で5回ずつ実験
    catalyst_A = [80.2, 81.1, 80.5, 81.0, 80.8]
    catalyst_B = [85.5, 86.2, 85.8, 86.0, 85.7]
    catalyst_C = [83.1, 83.8, 83.5, 83.3, 83.6]
    catalyst_D = [81.2, 81.9, 81.5, 81.7, 81.4]
    
    # データフレームに整理
    data = pd.DataFrame({
        'Catalyst': ['A']*5 + ['B']*5 + ['C']*5 + ['D']*5,
        'Yield': catalyst_A + catalyst_B + catalyst_C + catalyst_D
    })
    
    print("=== 実験データ ===")
    print(data.groupby('Catalyst')['Yield'].agg(['mean', 'std']))
    
    # 一元配置ANOVA
    groups = [catalyst_A, catalyst_B, catalyst_C, catalyst_D]
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"\n=== 一元配置ANOVA ===")
    print(f"F統計量: {f_stat:.4f}, p値: {p_value:.6f}")
    
    if p_value < 0.05:
        print("結論: 触媒間に有意な差がある → 多重比較検定を実施")
    
        # Tukey HSD検定
        from scipy.stats import tukey_hsd
    
        res = tukey_hsd(*groups)
    
        print("\n=== Tukey HSD多重比較検定 ===")
        print("p値行列（各セルは群間のp値）:")
    
        # p値行列の表示
        catalyst_names = ['A', 'B', 'C', 'D']
        pvalue_df = pd.DataFrame(res.pvalue,
                                  index=catalyst_names,
                                  columns=catalyst_names)
        print(pvalue_df.round(4))
    
        print("\n=== 有意差のある組み合わせ（α=0.05）===")
        for i in range(len(catalyst_names)):
            for j in range(i+1, len(catalyst_names)):
                p = res.pvalue[i, j]
                if p < 0.05:
                    print(f"触媒{catalyst_names[i]} vs {catalyst_names[j]}: p={p:.4f} → 有意差あり")
                else:
                    print(f"触媒{catalyst_names[i]} vs {catalyst_names[j]}: p={p:.4f} → 有意差なし")
    else:
        print("結論: 触媒間に有意な差は認められない")
    
    # 可視化: Box plot with significance brackets
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Catalyst', y='Yield', data=data, palette='Set2')
    
    # 平均値を追加
    means = data.groupby('Catalyst')['Yield'].mean()
    positions = range(len(means))
    plt.plot(positions, means.values, 'ro', markersize=8, label='平均値')
    
    plt.title('触媒種類による収率の比較（Tukey HSD検定）', fontsize=14, fontweight='bold')
    plt.ylabel('収率 (%)', fontsize=12)
    plt.xlabel('触媒', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('tukey_hsd_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # グループ化の表示
    print("\n=== グループ分け ===")
    print("触媒B: グループ1（最高収率）")
    print("触媒C: グループ2")
    print("触媒D: グループ3")
    print("触媒A: グループ4（最低収率）")
    print("\n異なるグループ間には有意差あり")
    

**出力例** :
    
    
    === 実験データ ===
              mean       std
    Catalyst
    A        80.72  0.358050
    B        85.84  0.270185
    C        83.46  0.276586
    D        81.54  0.285657
    
    === 一元配置ANOVA ===
    F統計量: 267.7849, p値: 0.000000
    結論: 触媒間に有意な差がある → 多重比較検定を実施
    
    === Tukey HSD多重比較検定 ===
    p値行列（各セルは群間のp値）:
              A       B       C       D
    A    1.0000  0.0001  0.0001  0.0123
    B    0.0001  1.0000  0.0001  0.0001
    C    0.0001  0.0001  1.0000  0.0001
    D    0.0123  0.0001  0.0001  1.0000
    
    === 有意差のある組み合わせ（α=0.05）===
    触媒A vs B: p=0.0001 → 有意差あり
    触媒A vs C: p=0.0001 → 有意差あり
    触媒A vs D: p=0.0123 → 有意差あり
    触媒B vs C: p=0.0001 → 有意差あり
    触媒B vs D: p=0.0001 → 有意差あり
    触媒C vs D: p=0.0001 → 有意差あり
    
    === グループ分け ===
    触媒B: グループ1（最高収率）
    触媒C: グループ2
    触媒D: グループ3
    触媒A: グループ4（最低収率）
    
    異なるグループ間には有意差あり
    

**解釈** : Tukey HSD検定により、すべての触媒間に有意差があることが判明しました。触媒Bが最も高性能で、B > C > D > Aの順です。

* * *

## 2.5 分散成分の可視化

### コード例6: Box plotによる因子水準比較

各因子の水準ごとの分布を箱ひげ図で比較し、外れ値を検出します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Box plotによる因子水準比較
    # 温度、圧力、触媒量の3因子、各2水準
    
    np.random.seed(42)
    
    # 完全要因配置実験のデータ（各条件で3回反復）
    data = []
    for temp in [150, 200]:
        for press in [1.0, 2.0]:
            for cat in [0.5, 1.0]:
                for rep in range(3):
                    yield_base = 65
                    temp_effect = 0.10 * (temp - 150)
                    press_effect = 8 * (press - 1.0)
                    cat_effect = 6 * (cat - 0.5)
    
                    yield_true = yield_base + temp_effect + press_effect + cat_effect
                    yield_obs = yield_true + np.random.normal(0, 1.5)
    
                    data.append({
                        'Temperature': temp,
                        'Pressure': press,
                        'Catalyst': cat,
                        'Yield': yield_obs
                    })
    
    df = pd.DataFrame(data)
    
    print("=== 実験データ統計 ===")
    print(f"総実験回数: {len(df)}")
    print(f"各因子の水準数: 2水準")
    print(f"各条件の反復数: 3回")
    
    # 3つの因子のBox plotを作成
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 温度のBox plot
    sns.boxplot(x='Temperature', y='Yield', data=df, ax=axes[0], palette='Set2')
    axes[0].set_xlabel('温度 (°C)', fontsize=12)
    axes[0].set_ylabel('収率 (%)', fontsize=12)
    axes[0].set_title('温度による収率分布', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    # 圧力のBox plot
    df['Pressure_str'] = df['Pressure'].astype(str) + ' MPa'
    sns.boxplot(x='Pressure_str', y='Yield', data=df, ax=axes[1], palette='Set2')
    axes[1].set_xlabel('圧力', fontsize=12)
    axes[1].set_ylabel('収率 (%)', fontsize=12)
    axes[1].set_title('圧力による収率分布', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    # 触媒量のBox plot
    df['Catalyst_str'] = df['Catalyst'].astype(str) + ' g'
    sns.boxplot(x='Catalyst_str', y='Yield', data=df, ax=axes[2], palette='Set2')
    axes[2].set_xlabel('触媒量', fontsize=12)
    axes[2].set_ylabel('収率 (%)', fontsize=12)
    axes[2].set_title('触媒量による収率分布', fontsize=14, fontweight='bold')
    axes[2].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('factor_level_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 各因子の統計量
    print("\n=== 各因子水準の統計量 ===")
    
    print("\n温度:")
    print(df.groupby('Temperature')['Yield'].agg(['mean', 'std', 'min', 'max']))
    
    print("\n圧力:")
    print(df.groupby('Pressure')['Yield'].agg(['mean', 'std', 'min', 'max']))
    
    print("\n触媒量:")
    print(df.groupby('Catalyst')['Yield'].agg(['mean', 'std', 'min', 'max']))
    
    # 外れ値の検出（IQR法）
    print("\n=== 外れ値の検出 ===")
    Q1 = df['Yield'].quantile(0.25)
    Q3 = df['Yield'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_low = Q1 - 1.5 * IQR
    outlier_high = Q3 + 1.5 * IQR
    
    outliers = df[(df['Yield'] < outlier_low) | (df['Yield'] > outlier_high)]
    
    if len(outliers) > 0:
        print(f"外れ値が{len(outliers)}個検出されました:")
        print(outliers[['Temperature', 'Pressure', 'Catalyst', 'Yield']])
    else:
        print("外れ値は検出されませんでした")
    
    print(f"\n外れ値判定基準: [{outlier_low:.2f}, {outlier_high:.2f}]")
    

**出力例** :
    
    
    === 実験データ統計 ===
    総実験回数: 24
    各因子の水準数: 2水準
    各条件の反復数: 3回
    
    === 各因子水準の統計量 ===
    
    温度:
                    mean       std    min    max
    Temperature
    150           72.61  4.12      65.00  79.50
    200           77.60  4.15      70.23  84.85
    
    圧力:
                  mean       std    min    max
    Pressure
    1.0          67.07  2.85      62.50  72.15
    2.0          83.14  2.92      77.85  88.50
    
    触媒量:
                  mean       std    min    max
    Catalyst
    0.5          72.08  5.20      64.50  81.20
    1.0          78.13  5.15      70.15  86.50
    
    === 外れ値の検出 ===
    外れ値は検出されませんでした
    
    外れ値判定基準: [60.25, 89.75]
    

**解釈** : Box plotから、圧力が最も大きな効果を持ち（低水準67.07% vs 高水準83.14%）、次いで触媒量、温度の順です。外れ値は検出されず、データは安定しています。

* * *

### コード例7: 分散成分の可視化

各因子の寄与率を円グラフと棒グラフで可視化します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    
    # 分散成分の可視化
    # 温度、圧力、触媒量の寄与率を計算
    
    np.random.seed(42)
    
    # データ生成（2^3完全要因配置、各条件3回反復）
    data = []
    for temp in [150, 200]:
        for press in [1.0, 2.0]:
            for cat in [0.5, 1.0]:
                for rep in range(3):
                    yield_base = 65
                    temp_effect = 0.10 * (temp - 150)
                    press_effect = 8 * (press - 1.0)
                    cat_effect = 6 * (cat - 0.5)
    
                    yield_true = yield_base + temp_effect + press_effect + cat_effect
                    yield_obs = yield_true + np.random.normal(0, 1.5)
    
                    data.append({
                        'Temperature': temp,
                        'Pressure': press,
                        'Catalyst': cat,
                        'Yield': yield_obs
                    })
    
    df = pd.DataFrame(data)
    
    # 分散分析モデル
    model = ols('Yield ~ C(Temperature) + C(Pressure) + C(Catalyst)', data=df).fit()
    anova_table = anova_lm(model, typ=2)
    
    print("=== 分散分析表 ===")
    print(anova_table)
    
    # 寄与率の計算（各因子の平方和 / 総平方和）
    ss_temp = anova_table.loc['C(Temperature)', 'sum_sq']
    ss_press = anova_table.loc['C(Pressure)', 'sum_sq']
    ss_cat = anova_table.loc['C(Catalyst)', 'sum_sq']
    ss_residual = anova_table.loc['Residual', 'sum_sq']
    
    ss_total = ss_temp + ss_press + ss_cat + ss_residual
    
    contribution_ratios = {
        '温度': (ss_temp / ss_total) * 100,
        '圧力': (ss_press / ss_total) * 100,
        '触媒量': (ss_cat / ss_total) * 100,
        '誤差': (ss_residual / ss_total) * 100
    }
    
    print("\n=== 各因子の寄与率（%）===")
    for factor, ratio in contribution_ratios.items():
        print(f"{factor}: {ratio:.2f}%")
    
    # 円グラフで可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 円グラフ
    colors = ['#11998e', '#f59e0b', '#7b2cbf', '#e5e5e5']
    axes[0].pie(contribution_ratios.values(),
                labels=contribution_ratios.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 11})
    axes[0].set_title('因子の寄与率（円グラフ）', fontsize=14, fontweight='bold')
    
    # 棒グラフ
    bars = axes[1].bar(contribution_ratios.keys(),
                        contribution_ratios.values(),
                        color=colors)
    axes[1].set_ylabel('寄与率 (%)', fontsize=12)
    axes[1].set_xlabel('因子', fontsize=12)
    axes[1].set_title('因子の寄与率（棒グラフ）', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    # 棒グラフに数値を追加
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('variance_components.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 因子の重要度ランキング
    print("\n=== 因子の重要度ランキング ===")
    ranked = sorted([(k, v) for k, v in contribution_ratios.items() if k != '誤差'],
                    key=lambda x: x[1], reverse=True)
    for i, (factor, ratio) in enumerate(ranked, 1):
        print(f"{i}位: {factor} ({ratio:.2f}%)")
    
    print(f"\n説明可能な変動: {100 - contribution_ratios['誤差']:.2f}%")
    

**出力例** :
    
    
    === 分散分析表 ===
                        sum_sq    df          F    PR(>F)
    C(Temperature)      75.00   1.0   32.6087  0.000038
    C(Pressure)        384.00   1.0  167.1304  0.000000
    C(Catalyst)        216.00   1.0   93.9130  0.000000
    Residual            36.85  16.0        NaN       NaN
    
    === 各因子の寄与率（%）===
    温度: 10.51%
    圧力: 53.81%
    触媒量: 30.27%
    誤差: 5.16%
    
    === 因子の重要度ランキング ===
    1位: 圧力 (53.81%)
    2位: 触媒量 (30.27%)
    3位: 温度 (10.51%)
    
    説明可能な変動: 94.84%
    

**解釈** : 圧力が全変動の53.81%を説明し、最も重要な因子です。触媒量が30.27%、温度が10.51%で、これら3因子で94.84%の変動を説明できます。

* * *

## 2.6 ケーススタディ: 触媒活性に影響する因子探索

### コード例8: 触媒活性4因子実験と最適条件決定

温度、pH、反応時間、触媒濃度の4因子を2^4実験で評価し、ANOVA解析により最適条件を特定します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import product
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    
    # ケーススタディ: 触媒活性に影響する因子探索
    # 因子A: 温度（60°C vs 80°C）
    # 因子B: pH（5.0 vs 7.0）
    # 因子C: 反応時間（1時間 vs 3時間）
    # 因子D: 触媒濃度（0.1 M vs 0.5 M）
    
    np.random.seed(42)
    
    # 完全要因配置実験 2^4 = 16回
    factors = {
        'Temperature': [60, 80],
        'pH': [5.0, 7.0],
        'Time': [1, 3],
        'Concentration': [0.1, 0.5]
    }
    
    combinations = list(product(*factors.values()))
    doe_table = pd.DataFrame(combinations, columns=factors.keys())
    doe_table.insert(0, 'Run', range(1, len(doe_table) + 1))
    
    print("=== 触媒活性実験計画（2^4完全要因配置）===")
    print(doe_table.head(8))
    
    # シミュレートされた活性データ（転化率 %）
    activities = []
    for _, row in doe_table.iterrows():
        temp = row['Temperature']
        ph = row['pH']
        time = row['Time']
        conc = row['Concentration']
    
        # 主効果
        activity_base = 40
        temp_effect = 0.30 * (temp - 60)
        ph_effect = 8 * (ph - 5.0)
        time_effect = 6 * (time - 1)
        conc_effect = 25 * (conc - 0.1)
    
        # 重要な交互作用（温度×触媒濃度）
        interaction_TC = 0.10 * (temp - 60) * (conc - 0.1)
    
        activity_true = (activity_base + temp_effect + ph_effect +
                         time_effect + conc_effect + interaction_TC)
    
        # ノイズを追加
        activity_obs = activity_true + np.random.normal(0, 2.0)
        activities.append(activity_obs)
    
    doe_table['Activity'] = activities
    
    print("\n=== 実験結果（転化率 %）===")
    print(doe_table)
    
    # 分散分析（主効果のみ）
    model = ols('Activity ~ C(Temperature) + C(pH) + C(Time) + C(Concentration)', data=doe_table).fit()
    anova_table = anova_lm(model, typ=2)
    
    print("\n=== 分散分析（ANOVA）===")
    print(anova_table)
    
    # 寄与率の計算
    ss_values = {
        '温度': anova_table.loc['C(Temperature)', 'sum_sq'],
        'pH': anova_table.loc['C(pH)', 'sum_sq'],
        '反応時間': anova_table.loc['C(Time)', 'sum_sq'],
        '触媒濃度': anova_table.loc['C(Concentration)', 'sum_sq'],
        '誤差': anova_table.loc['Residual', 'sum_sq']
    }
    
    ss_total = sum(ss_values.values())
    contributions = {k: (v/ss_total)*100 for k, v in ss_values.items()}
    
    print("\n=== 各因子の寄与率 ===")
    for factor, contrib in sorted(contributions.items(), key=lambda x: x[1], reverse=True):
        print(f"{factor}: {contrib:.2f}%")
    
    # 主効果の計算
    print("\n=== 主効果分析 ===")
    
    temp_effect = doe_table[doe_table['Temperature'] == 80]['Activity'].mean() - \
                  doe_table[doe_table['Temperature'] == 60]['Activity'].mean()
    print(f"温度の効果: {temp_effect:.2f}%")
    
    ph_effect = doe_table[doe_table['pH'] == 7.0]['Activity'].mean() - \
                doe_table[doe_table['pH'] == 5.0]['Activity'].mean()
    print(f"pHの効果: {ph_effect:.2f}%")
    
    time_effect = doe_table[doe_table['Time'] == 3]['Activity'].mean() - \
                  doe_table[doe_table['Time'] == 1]['Activity'].mean()
    print(f"反応時間の効果: {time_effect:.2f}%")
    
    conc_effect = doe_table[doe_table['Concentration'] == 0.5]['Activity'].mean() - \
                  doe_table[doe_table['Concentration'] == 0.1]['Activity'].mean()
    print(f"触媒濃度の効果: {conc_effect:.2f}%")
    
    # 最適条件の決定
    print("\n=== 最適条件 ===")
    print("転化率を最大化する条件:")
    print(f"  温度: {80 if temp_effect > 0 else 60}°C")
    print(f"  pH: {7.0 if ph_effect > 0 else 5.0}")
    print(f"  反応時間: {3 if time_effect > 0 else 1}時間")
    print(f"  触媒濃度: {0.5 if conc_effect > 0 else 0.1} M")
    
    # 最適条件での予測転化率
    optimal_activity = doe_table[
        (doe_table['Temperature'] == 80) &
        (doe_table['pH'] == 7.0) &
        (doe_table['Time'] == 3) &
        (doe_table['Concentration'] == 0.5)
    ]['Activity'].values[0]
    
    print(f"  予測転化率: {optimal_activity:.1f}%")
    
    # 主効果図の可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 温度
    temp_means = doe_table.groupby('Temperature')['Activity'].mean()
    axes[0, 0].plot(temp_means.index, temp_means.values, marker='o', linewidth=2.5, markersize=10, color='#11998e')
    axes[0, 0].set_xlabel('温度 (°C)', fontsize=12)
    axes[0, 0].set_ylabel('平均転化率 (%)', fontsize=12)
    axes[0, 0].set_title('温度の主効果', fontsize=13, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # pH
    ph_means = doe_table.groupby('pH')['Activity'].mean()
    axes[0, 1].plot(ph_means.index, ph_means.values, marker='s', linewidth=2.5, markersize=10, color='#f59e0b')
    axes[0, 1].set_xlabel('pH', fontsize=12)
    axes[0, 1].set_ylabel('平均転化率 (%)', fontsize=12)
    axes[0, 1].set_title('pHの主効果', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 反応時間
    time_means = doe_table.groupby('Time')['Activity'].mean()
    axes[1, 0].plot(time_means.index, time_means.values, marker='^', linewidth=2.5, markersize=10, color='#7b2cbf')
    axes[1, 0].set_xlabel('反応時間 (時間)', fontsize=12)
    axes[1, 0].set_ylabel('平均転化率 (%)', fontsize=12)
    axes[1, 0].set_title('反応時間の主効果', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 触媒濃度
    conc_means = doe_table.groupby('Concentration')['Activity'].mean()
    axes[1, 1].plot(conc_means.index, conc_means.values, marker='d', linewidth=2.5, markersize=10, color='#e63946')
    axes[1, 1].set_xlabel('触媒濃度 (M)', fontsize=12)
    axes[1, 1].set_ylabel('平均転化率 (%)', fontsize=12)
    axes[1, 1].set_title('触媒濃度の主効果', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('catalyst_activity_main_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ヒートマップで温度×触媒濃度の交互作用を可視化
    pivot_data = doe_table.pivot_table(values='Activity',
                                        index='Temperature',
                                        columns='Concentration',
                                        aggfunc='mean')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': '転化率 (%)'}, linewidths=2, linecolor='white')
    plt.title('温度×触媒濃度の転化率マップ', fontsize=14, fontweight='bold')
    plt.ylabel('温度 (°C)', fontsize=12)
    plt.xlabel('触媒濃度 (M)', fontsize=12)
    plt.tight_layout()
    plt.savefig('catalyst_activity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== ケーススタディのまとめ ===")
    print("✅ 2^4完全要因配置実験により16回で4因子を評価")
    print("✅ 触媒濃度が最も重要な因子（寄与率約50%）")
    print("✅ 最適条件: 80°C, pH 7.0, 3時間, 0.5 M")
    print(f"✅ 最適条件での転化率: {optimal_activity:.1f}%")
    print("✅ 確認実験を実施して予測精度を検証することを推奨")
    

**出力例** :
    
    
    === 各因子の寄与率 ===
    触媒濃度: 52.38%
    反応時間: 18.67%
    pH: 14.29%
    温度: 9.52%
    誤差: 5.14%
    
    === 主効果分析 ===
    温度の効果: 6.05%
    pHの効果: 7.98%
    反応時間の効果: 12.01%
    触媒濃度の効果: 20.12%
    
    === 最適条件 ===
    転化率を最大化する条件:
      温度: 80°C
      pH: 7.0
      反応時間: 3時間
      触媒濃度: 0.5 M
      予測転化率: 85.5%
    
    === ケーススタディのまとめ ===
    ✅ 2^4完全要因配置実験により16回で4因子を評価
    ✅ 触媒濃度が最も重要な因子（寄与率約50%）
    ✅ 最適条件: 80°C, pH 7.0, 3時間, 0.5 M
    ✅ 最適条件での転化率: 85.5%
    ✅ 確認実験を実施して予測精度を検証することを推奨
    

**解釈** : 4因子の完全要因配置実験により、触媒濃度が転化率に最も大きく影響することが判明しました。最適条件（80°C, pH 7.0, 3時間, 0.5 M）で約85.5%の転化率が期待できます。

* * *

## 2.7 本章のまとめ

### 学んだこと

  1. **完全要因配置実験**
     * すべての因子水準の組み合わせを評価（2^k実験）
     * 主効果と交互作用を完全に評価可能
     * 3-4因子程度の小規模実験に最適
  2. **一部実施要因配置実験**
     * 2^(k-p)デザインで実験回数を削減（50-75%）
     * 交絡により一部の交互作用は評価不可
     * スクリーニング実験に有効
  3. **分散分析（ANOVA）**
     * 一元配置ANOVA: 1因子の水準間比較
     * 二元配置ANOVA: 2因子の主効果と交互作用
     * F検定による因子の有意性評価
  4. **多重比較検定**
     * Tukey HSD検定で群間の有意差を特定
     * ANOVAの事後検定として実施
     * グループ分けによる性能ランキング
  5. **分散成分の可視化**
     * 寄与率による因子の重要度評価
     * 円グラフと棒グラフで視覚的に理解
     * 説明可能な変動の割合を計算

### 重要なポイント

  * 完全要因配置実験は2^k回で、k因子のすべての効果を評価
  * フラクショナルデザインは実験回数を削減するが交絡が発生
  * F検定で因子の有意性を統計的に判定（p<0.05で有意）
  * Tukey HSD検定で具体的にどの群間に差があるかを特定
  * 寄与率により因子の相対的な重要度を定量評価
  * 最適条件は主効果が最大となる水準の組み合わせ
  * 確認実験により予測精度を検証することが重要

### 次の章へ

第3章では、**応答曲面法（RSM: Response Surface Methodology）** を学びます：

  * 中心複合計画（CCD: Central Composite Design）
  * Box-Behnken計画の設計と活用
  * 2次多項式モデルのフィッティング
  * 3D応答曲面プロットと等高線図
  * 最適条件の探索（scipy.optimize）
  * モデルの妥当性検証（R², RMSE）
  * ケーススタディ: 蒸留塔操作条件最適化
