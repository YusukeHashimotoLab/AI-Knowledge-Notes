---
title: "第3章: 仮説検定と検定力分析"
chapter_title: "第3章: 仮説検定と検定力分析"
subtitle: Hypothesis Testing and Power Analysis
---

[基礎数理道場](<../index.html>) > [推測統計学とベイズ統計](<index.html>) > 第3章 

## 3.1 仮説検定の枠組み

#### 📘 仮説検定の基本構造

**帰無仮説 (Null Hypothesis) \\( H_0 \\)** : 「差がない」「効果がない」という主張

**対立仮説 (Alternative Hypothesis) \\( H_1 \\)** : 「差がある」「効果がある」という主張

**第1種過誤 (Type I Error)** : \\( H_0 \\) が真なのに棄却する誤り（偽陽性）

$$ \alpha = P(\text{Type I Error}) = P(\text{reject } H_0 | H_0 \text{ is true}) $$

**第2種過誤 (Type II Error)** : \\( H_0 \\) が偽なのに棄却しない誤り（偽陰性）

$$ \beta = P(\text{Type II Error}) = P(\text{fail to reject } H_0 | H_1 \text{ is true}) $$

**検定力 (Power)** : 真の効果を正しく検出する確率

$$ \text{Power} = 1 - \beta $$

#### 💻 コード例1: z検定（母平均の検定）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # シナリオ: 新しい製造プロセスが従来より平均強度を向上させたか？
    # H0: μ = 450 (従来プロセスの平均)
    # H1: μ > 450 (片側検定)
    
    mu_0 = 450  # 帰無仮説の母平均
    sigma = 30  # 母標準偏差（既知）
    alpha = 0.05  # 有意水準
    
    # 新プロセスのデータ
    np.random.seed(42)
    n = 25
    mu_true = 465  # 実際の母平均（未知と仮定）
    data = np.random.normal(mu_true, sigma, n)
    
    # 検定統計量の計算
    x_bar = np.mean(data)
    se = sigma / np.sqrt(n)
    z_stat = (x_bar - mu_0) / se
    
    # p値の計算（片側検定）
    p_value = 1 - stats.norm.cdf(z_stat)
    
    # 棄却域
    z_critical = stats.norm.ppf(1 - alpha)
    
    print("=== z検定: 母平均の片側検定 ===")
    print(f"帰無仮説 H0: μ = {mu_0}")
    print(f"対立仮説 H1: μ > {mu_0}")
    print(f"有意水準: α = {alpha}")
    print(f"\n標本サイズ: {n}")
    print(f"標本平均: {x_bar:.2f}")
    print(f"z統計量: {z_stat:.4f}")
    print(f"臨界値: z_{{{1-alpha}}} = {z_critical:.4f}")
    print(f"p値: {p_value:.6f}")
    print(f"\n判定: {'H0を棄却' if p_value < alpha else 'H0を棄却できない'}")
    print(f"解釈: {'新プロセスは有意に強度を向上させた' if p_value < alpha else '有意な向上は確認できない'}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 帰無仮説下の分布と棄却域
    x = np.linspace(-4, 4, 300)
    axes[0].plot(x, stats.norm.pdf(x), 'b-', linewidth=2, label='H0下の分布')
    axes[0].fill_between(x, 0, stats.norm.pdf(x), where=(x >= z_critical),
                          alpha=0.3, color='red', label=f'棄却域 (α={alpha})')
    axes[0].axvline(z_stat, color='green', linestyle='--', linewidth=2,
                    label=f'観測されたz統計量: {z_stat:.2f}')
    axes[0].axvline(z_critical, color='red', linestyle=':', linewidth=2,
                    label=f'臨界値: {z_critical:.2f}')
    axes[0].set_xlabel('z値')
    axes[0].set_ylabel('確率密度')
    axes[0].set_title('仮説検定の可視化（片側検定）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 元データと帰無仮説下の分布
    x_data = np.linspace(mu_0 - 4*sigma, mu_0 + 4*sigma, 300)
    axes[1].hist(data, bins=12, density=True, alpha=0.7,
                 color='skyblue', edgecolor='black', label='観測データ')
    axes[1].plot(x_data, stats.norm.pdf(x_data, mu_0, se),
                 'r-', linewidth=2, label=f'H0: μ={mu_0} の分布')
    axes[1].axvline(x_bar, color='blue', linestyle='--', linewidth=2,
                    label=f'標本平均: {x_bar:.1f}')
    axes[1].axvline(mu_0, color='red', linestyle=':', linewidth=2,
                    label=f'H0の平均: {mu_0}')
    axes[1].set_xlabel('強度 [MPa]')
    axes[1].set_ylabel('密度')
    axes[1].set_title('データと帰無仮説の比較')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

**📌 p値の正しい解釈**  
p値は「帰無仮説が真である場合に、観測されたデータまたはそれより極端なデータが得られる確率」です。 「帰無仮説が正しい確率」ではありません。p < 0.05は「5%未満の確率でしか起こらない希な現象」を意味します。 

## 3.2 t検定（1標本・2標本・対応あり）

#### 💻 コード例2: t検定の3パターン
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    np.random.seed(42)
    
    # ===== 1標本t検定 =====
    print("=== 1標本t検定 ===")
    # 材料の目標強度500 MPaを達成しているか？
    mu_target = 500
    n1 = 15
    data_1sample = np.random.normal(510, 25, n1)
    
    t_stat_1, p_value_1 = stats.ttest_1samp(data_1sample, mu_target)
    print(f"H0: μ = {mu_target}")
    print(f"標本平均: {np.mean(data_1sample):.2f}")
    print(f"t統計量: {t_stat_1:.4f}, p値: {p_value_1:.6f}")
    print(f"判定: {'H0を棄却（目標と有意に異なる）' if p_value_1 < 0.05 else 'H0を棄却できない'}\n")
    
    # ===== 2標本t検定（独立） =====
    print("=== 2標本t検定（独立）===")
    # 2つの供給業者の材料強度に差があるか？
    n2a, n2b = 20, 22
    data_2sample_A = np.random.normal(480, 20, n2a)
    data_2sample_B = np.random.normal(495, 25, n2b)
    
    # Welchのt検定（不等分散）
    t_stat_2, p_value_2 = stats.ttest_ind(data_2sample_A, data_2sample_B, equal_var=False)
    print(f"H0: μA = μB")
    print(f"平均A: {np.mean(data_2sample_A):.2f}, 平均B: {np.mean(data_2sample_B):.2f}")
    print(f"t統計量: {t_stat_2:.4f}, p値: {p_value_2:.6f}")
    print(f"判定: {'H0を棄却（有意差あり）' if p_value_2 < 0.05 else 'H0を棄却できない'}\n")
    
    # ===== 対応のあるt検定 =====
    print("=== 対応のあるt検定 ===")
    # 熱処理前後で材料の硬度が変化したか？
    n3 = 12
    before = np.random.normal(65, 5, n3)
    after = before + np.random.normal(3, 2, n3)  # 平均3の増加
    
    t_stat_3, p_value_3 = stats.ttest_rel(after, before)
    diff = after - before
    print(f"H0: μdiff = 0 (処理前後で変化なし)")
    print(f"平均差: {np.mean(diff):.2f}")
    print(f"t統計量: {t_stat_3:.4f}, p値: {p_value_3:.6f}")
    print(f"判定: {'H0を棄却（有意な変化あり）' if p_value_3 < 0.05 else 'H0を棄却できない'}\n")
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1標本t検定
    axes[0].hist(data_1sample, bins=8, density=True, alpha=0.7,
                 color='skyblue', edgecolor='black')
    axes[0].axvline(np.mean(data_1sample), color='blue', linestyle='--',
                    linewidth=2, label=f'標本平均: {np.mean(data_1sample):.1f}')
    axes[0].axvline(mu_target, color='red', linestyle='--',
                    linewidth=2, label=f'目標値: {mu_target}')
    axes[0].set_xlabel('強度 [MPa]')
    axes[0].set_ylabel('密度')
    axes[0].set_title(f'1標本t検定\np={p_value_1:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2標本t検定
    bp = axes[1].boxplot([data_2sample_A, data_2sample_B], labels=['業者A', '業者B'],
                          patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    axes[1].set_ylabel('強度 [MPa]')
    axes[1].set_title(f'2標本t検定\np={p_value_2:.4f}')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 対応のあるt検定
    x_pos = np.arange(n3)
    axes[2].plot(x_pos, before, 'bo-', label='処理前', alpha=0.6)
    axes[2].plot(x_pos, after, 'rs-', label='処理後', alpha=0.6)
    for i in range(n3):
        axes[2].plot([i, i], [before[i], after[i]], 'k-', alpha=0.3)
    axes[2].set_xlabel('サンプル番号')
    axes[2].set_ylabel('硬度 [HV]')
    axes[2].set_title(f'対応のあるt検定\np={p_value_3:.4f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

## 3.3 カイ二乗検定（適合度・独立性）

#### 💻 コード例3: カイ二乗検定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # ===== 適合度検定 =====
    print("=== カイ二乗適合度検定 ===")
    # サイコロの公正性の検定
    observed = np.array([48, 52, 45, 50, 55, 50])  # 観測度数
    expected = np.array([50, 50, 50, 50, 50, 50])  # 期待度数（公正なら各面1/6）
    
    chi2_stat, p_value_gof = stats.chisquare(observed, expected)
    print(f"H0: サイコロは公正である")
    print(f"観測度数: {observed}")
    print(f"期待度数: {expected}")
    print(f"χ²統計量: {chi2_stat:.4f}")
    print(f"p値: {p_value_gof:.6f}")
    print(f"判定: {'H0を棄却（不公正）' if p_value_gof < 0.05 else 'H0を棄却できない'}\n")
    
    # ===== 独立性の検定 =====
    print("=== カイ二乗独立性検定 ===")
    # 材料の種類と不良品発生に関連があるか？
    # 2×3分割表
    contingency_table = np.array([
        [15, 25, 10],  # 材料A: 良品, 軽微不良, 重大不良
        [20, 18, 12]   # 材料B
    ])
    
    chi2_stat_ind, p_value_ind, dof, expected_ind = stats.chi2_contingency(contingency_table)
    
    print(f"H0: 材料の種類と不良の程度は独立")
    print(f"観測度数:\n{contingency_table}")
    print(f"期待度数:\n{expected_ind}")
    print(f"χ²統計量: {chi2_stat_ind:.4f}")
    print(f"自由度: {dof}")
    print(f"p値: {p_value_ind:.6f}")
    print(f"判定: {'H0を棄却（関連あり）' if p_value_ind < 0.05 else 'H0を棄却できない'}\n")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 適合度検定
    x = np.arange(1, 7)
    width = 0.35
    axes[0].bar(x - width/2, observed, width, label='観測', color='skyblue', edgecolor='black')
    axes[0].bar(x + width/2, expected, width, label='期待', color='orange', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('サイコロの目')
    axes[0].set_ylabel('度数')
    axes[0].set_title(f'適合度検定\nχ²={chi2_stat:.2f}, p={p_value_gof:.4f}')
    axes[0].set_xticks(x)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 独立性検定
    categories = ['良品', '軽微不良', '重大不良']
    materials = ['材料A', '材料B']
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1].bar(x - width/2, contingency_table[0], width, label='材料A',
                color='lightblue', edgecolor='black')
    axes[1].bar(x + width/2, contingency_table[1], width, label='材料B',
                color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('品質カテゴリ')
    axes[1].set_ylabel('度数')
    axes[1].set_title(f'独立性検定\nχ²={chi2_stat_ind:.2f}, p={p_value_ind:.4f}')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

## 3.4 F検定（等分散性の検定）

#### 💻 コード例4: F検定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    np.random.seed(42)
    
    # 2つのプロセスの分散が等しいか？
    n1, n2 = 20, 25
    sigma1, sigma2 = 15, 22  # 真の標準偏差
    
    data1 = np.random.normal(100, sigma1, n1)
    data2 = np.random.normal(100, sigma2, n2)
    
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)
    
    # F統計量（大きい分散/小さい分散）
    F_stat = max(var1, var2) / min(var1, var2)
    df1 = n1 - 1 if var1 > var2 else n2 - 1
    df2 = n2 - 1 if var1 > var2 else n1 - 1
    
    # 両側検定のp値
    p_value = 2 * min(stats.f.cdf(F_stat, df1, df2),
                      1 - stats.f.cdf(F_stat, df1, df2))
    
    print("=== F検定: 等分散性の検定 ===")
    print(f"H0: σ₁² = σ₂² (分散が等しい)")
    print(f"H1: σ₁² ≠ σ₂²")
    print(f"\nデータ1: n={n1}, s²={var1:.2f}")
    print(f"データ2: n={n2}, s²={var2:.2f}")
    print(f"\nF統計量: {F_stat:.4f}")
    print(f"自由度: ({df1}, {df2})")
    print(f"p値: {p_value:.6f}")
    print(f"\n判定: {'H0を棄却（分散が異なる）' if p_value < 0.05 else 'H0を棄却できない（等分散性が仮定できる）'}")
    
    # Leveneの検定（より頑健）
    levene_stat, levene_p = stats.levene(data1, data2)
    print(f"\n【Leveneの検定】（頑健性が高い）")
    print(f"統計量: {levene_stat:.4f}, p値: {levene_p:.6f}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # データの分布
    axes[0].hist(data1, bins=12, density=True, alpha=0.6,
                 color='skyblue', edgecolor='black', label=f'データ1 (s²={var1:.1f})')
    axes[0].hist(data2, bins=12, density=True, alpha=0.6,
                 color='lightgreen', edgecolor='black', label=f'データ2 (s²={var2:.1f})')
    axes[0].set_xlabel('値')
    axes[0].set_ylabel('密度')
    axes[0].set_title('2つのデータセットの分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F分布と観測されたF統計量
    x = np.linspace(0, 5, 300)
    axes[1].plot(x, stats.f.pdf(x, df1, df2), 'b-', linewidth=2,
                 label=f'F分布 ({df1}, {df2})')
    axes[1].axvline(F_stat, color='red', linestyle='--', linewidth=2,
                    label=f'観測F統計量: {F_stat:.2f}')
    
    # 臨界値（両側5%）
    f_crit_upper = stats.f.ppf(0.975, df1, df2)
    axes[1].axvline(f_crit_upper, color='orange', linestyle=':', linewidth=2,
                    label=f'臨界値(上側2.5%): {f_crit_upper:.2f}')
    axes[1].fill_between(x, 0, stats.f.pdf(x, df1, df2),
                          where=(x >= f_crit_upper), alpha=0.3, color='red',
                          label='棄却域')
    axes[1].set_xlabel('F値')
    axes[1].set_ylabel('確率密度')
    axes[1].set_title(f'F検定の可視化\np={p_value:.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

## 3.5 検定力分析とサンプルサイズ計算

#### 📘 検定力（Power）

検定力は真の効果を正しく検出する確率です：

$$ \text{Power} = 1 - \beta = P(\text{reject } H_0 | H_1 \text{ is true}) $$

検定力に影響する要因：

  * **効果量（Effect Size）** : 真の差の大きさ
  * **標本サイズ（n）** : 大きいほど検定力が高い
  * **有意水準（α）** : 大きいほど検定力が高い（第1種過誤は増加）
  * **データの分散** : 小さいほど検定力が高い

#### 💻 コード例5: 検定力分析とサンプルサイズ計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.stats.power import TTestIndPower
    
    # 検定力分析のパラメータ
    mu0 = 100  # 帰無仮説の平均
    mu1 = 105  # 対立仮説の平均
    sigma = 10  # 標準偏差
    alpha = 0.05
    
    # Cohen's d (効果量)
    effect_size = (mu1 - mu0) / sigma
    print("=== 検定力分析 ===")
    print(f"効果量 Cohen's d: {effect_size:.3f}")
    print(f"解釈: {'small' if effect_size < 0.5 else 'medium' if effect_size < 0.8 else 'large'}")
    
    # サンプルサイズと検定力の関係
    sample_sizes = np.arange(10, 201, 5)
    powers = []
    
    power_analysis = TTestIndPower()
    for n in sample_sizes:
        power = power_analysis.power(effect_size, n, alpha, alternative='two-sided')
        powers.append(power)
    
    # 目標検定力0.8を達成するサンプルサイズ
    target_power = 0.8
    required_n = power_analysis.solve_power(effect_size, power=target_power,
                                            alpha=alpha, alternative='two-sided')
    
    print(f"\n目標検定力 {target_power} を達成するサンプルサイズ: {int(np.ceil(required_n))}")
    
    # 効果量と検定力の関係
    effect_sizes = np.linspace(0.1, 1.5, 50)
    powers_by_effect = []
    n_fixed = 50
    
    for es in effect_sizes:
        power = power_analysis.power(es, n_fixed, alpha, alternative='two-sided')
        powers_by_effect.append(power)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 帰無仮説と対立仮説の分布
    x = np.linspace(80, 120, 300)
    se_example = sigma / np.sqrt(50)
    axes[0, 0].plot(x, stats.norm.pdf(x, mu0, se_example), 'b-',
                    linewidth=2, label=f'H0: μ={mu0}')
    axes[0, 0].plot(x, stats.norm.pdf(x, mu1, se_example), 'r-',
                    linewidth=2, label=f'H1: μ={mu1}')
    
    # 臨界値
    z_crit = stats.norm.ppf(1 - alpha/2)
    x_crit_upper = mu0 + z_crit * se_example
    x_crit_lower = mu0 - z_crit * se_example
    
    axes[0, 0].axvline(x_crit_upper, color='green', linestyle='--',
                       linewidth=2, label='臨界値')
    axes[0, 0].axvline(x_crit_lower, color='green', linestyle='--', linewidth=2)
    
    # β領域（第2種過誤）
    x_beta = np.linspace(x_crit_lower, x_crit_upper, 100)
    axes[0, 0].fill_between(x_beta, 0, stats.norm.pdf(x_beta, mu1, se_example),
                             alpha=0.3, color='orange', label='β (第2種過誤)')
    
    # 検定力領域
    x_power_lower = np.linspace(80, x_crit_lower, 100)
    x_power_upper = np.linspace(x_crit_upper, 120, 100)
    axes[0, 0].fill_between(x_power_lower, 0, stats.norm.pdf(x_power_lower, mu1, se_example),
                             alpha=0.3, color='green', label='検定力')
    axes[0, 0].fill_between(x_power_upper, 0, stats.norm.pdf(x_power_upper, mu1, se_example),
                             alpha=0.3, color='green')
    
    axes[0, 0].set_xlabel('値')
    axes[0, 0].set_ylabel('確率密度')
    axes[0, 0].set_title(f'検定力の概念図 (n=50, Power={powers[8]:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # サンプルサイズと検定力
    axes[0, 1].plot(sample_sizes, powers, 'b-', linewidth=2)
    axes[0, 1].axhline(target_power, color='red', linestyle='--',
                       linewidth=2, label=f'目標検定力: {target_power}')
    axes[0, 1].axvline(required_n, color='green', linestyle='--',
                       linewidth=2, label=f'必要n: {int(np.ceil(required_n))}')
    axes[0, 1].set_xlabel('サンプルサイズ (各群)')
    axes[0, 1].set_ylabel('検定力 (1-β)')
    axes[0, 1].set_title(f'サンプルサイズと検定力 (d={effect_size:.2f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 効果量と検定力
    axes[1, 0].plot(effect_sizes, powers_by_effect, 'b-', linewidth=2)
    axes[1, 0].axhline(target_power, color='red', linestyle='--',
                       linewidth=2, label=f'目標検定力: {target_power}')
    axes[1, 0].axvline(effect_size, color='orange', linestyle='--',
                       linewidth=2, label=f'現在の効果量: {effect_size:.2f}')
    axes[1, 0].set_xlabel("効果量 (Cohen's d)")
    axes[1, 0].set_ylabel('検定力 (1-β)')
    axes[1, 0].set_title(f'効果量と検定力 (n={n_fixed})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # αとβのトレードオフ
    alphas = np.linspace(0.01, 0.2, 50)
    betas_at_alphas = []
    
    for a in alphas:
        power = power_analysis.power(effect_size, n_fixed, a, alternative='two-sided')
        betas_at_alphas.append(1 - power)
    
    axes[1, 1].plot(alphas, betas_at_alphas, 'b-', linewidth=2, label='β')
    axes[1, 1].plot(alphas, alphas, 'r--', linewidth=2, label='α')
    axes[1, 1].axvline(0.05, color='green', linestyle=':', linewidth=2,
                       label='α=0.05 (慣例)')
    axes[1, 1].set_xlabel('α (第1種過誤率)')
    axes[1, 1].set_ylabel('誤り率')
    axes[1, 1].set_title(f'αとβのトレードオフ (n={n_fixed}, d={effect_size:.2f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 実践的アドバイス
    print("\n=== サンプルサイズ設計のガイドライン ===")
    print(f"小効果 (d=0.2): n ≈ {int(np.ceil(power_analysis.solve_power(0.2, power=0.8, alpha=0.05)))}")
    print(f"中効果 (d=0.5): n ≈ {int(np.ceil(power_analysis.solve_power(0.5, power=0.8, alpha=0.05)))}")
    print(f"大効果 (d=0.8): n ≈ {int(np.ceil(power_analysis.solve_power(0.8, power=0.8, alpha=0.05)))}")

## 3.6 多重比較問題と補正法

#### 📘 多重比較問題

複数の仮説検定を同時に行うと、少なくとも1つの第1種過誤が起きる確率（Family-Wise Error Rate, FWER）が増加します：

$$ \text{FWER} = 1 - (1-\alpha)^m $$

ここで \\( m \\) は検定回数です。

**Bonferroni補正** : 各検定の有意水準を \\( \alpha/m \\) に設定

**Holm法** : ステップダウン法（より検定力が高い）

**FDR制御** : Benjamini-Hochberg法（False Discovery Rateを制御）

#### 💻 コード例6: 多重比較補正
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    
    np.random.seed(42)
    
    # 10個の材料特性を同時に検定
    n_tests = 10
    n_samples = 30
    alpha = 0.05
    
    # 9個は差なし、1個だけ真の差あり
    p_values = []
    for i in range(n_tests):
        if i == 5:  # 6番目だけ真の差
            data1 = np.random.normal(100, 15, n_samples)
            data2 = np.random.normal(110, 15, n_samples)
        else:  # 他は差なし
            data1 = np.random.normal(100, 15, n_samples)
            data2 = np.random.normal(100, 15, n_samples)
        
        _, p = stats.ttest_ind(data1, data2)
        p_values.append(p)
    
    p_values = np.array(p_values)
    
    print("=== 多重比較問題 ===")
    print(f"検定回数: {n_tests}")
    print(f"各検定の有意水準: α = {alpha}")
    print(f"理論的FWER: {1 - (1-alpha)**n_tests:.4f}")
    print(f"\n元のp値:\n{p_values}")
    print(f"\n補正なし（α={alpha}）で有意: {np.sum(p_values < alpha)}個")
    
    # 各種補正法の適用
    methods = ['bonferroni', 'holm', 'fdr_bh']
    method_names = ['Bonferroni', 'Holm', 'Benjamini-Hochberg']
    
    results = {}
    for method, name in zip(methods, method_names):
        reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
        results[name] = {'reject': reject, 'p_corrected': p_corrected}
        print(f"\n【{name}法】")
        print(f"  補正後p値: {p_corrected}")
        print(f"  有意と判定: {np.sum(reject)}個 (位置: {np.where(reject)[0]})")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 元のp値
    axes[0, 0].bar(range(n_tests), p_values, color='skyblue', edgecolor='black')
    axes[0, 0].axhline(alpha, color='red', linestyle='--', linewidth=2,
                       label=f'α = {alpha}')
    axes[0, 0].set_xlabel('検定番号')
    axes[0, 0].set_ylabel('p値')
    axes[0, 0].set_title('元のp値（補正なし）')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Bonferroni補正
    reject_bonf = results['Bonferroni']['reject']
    colors = ['red' if r else 'skyblue' for r in reject_bonf]
    axes[0, 1].bar(range(n_tests), p_values, color=colors, edgecolor='black')
    axes[0, 1].axhline(alpha/n_tests, color='orange', linestyle='--',
                       linewidth=2, label=f'Bonferroni α = {alpha/n_tests:.4f}')
    axes[0, 1].set_xlabel('検定番号')
    axes[0, 1].set_ylabel('p値')
    axes[0, 1].set_title('Bonferroni補正')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Holm補正
    reject_holm = results['Holm']['reject']
    colors = ['red' if r else 'skyblue' for r in reject_holm]
    axes[1, 0].bar(range(n_tests), p_values, color=colors, edgecolor='black')
    # Holm法の段階的な閾値を表示
    sorted_idx = np.argsort(p_values)
    for rank, idx in enumerate(sorted_idx):
        holm_alpha = alpha / (n_tests - rank)
        axes[1, 0].plot(idx, holm_alpha, 'o', color='orange', markersize=8)
    
    axes[1, 0].set_xlabel('検定番号')
    axes[1, 0].set_ylabel('p値')
    axes[1, 0].set_title('Holm補正（段階的閾値）')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # FDR (Benjamini-Hochberg)
    reject_fdr = results['Benjamini-Hochberg']['reject']
    colors = ['red' if r else 'skyblue' for r in reject_fdr]
    sorted_p = np.sort(p_values)
    bh_thresholds = [(i+1)/n_tests * alpha for i in range(n_tests)]
    
    axes[1, 1].bar(range(n_tests), sorted_p, color=['red' if sorted_p[i] < bh_thresholds[i] else 'skyblue' for i in range(n_tests)],
                   edgecolor='black', alpha=0.7, label='p値')
    axes[1, 1].plot(range(n_tests), bh_thresholds, 'o-', color='orange',
                    linewidth=2, markersize=6, label='BH閾値')
    axes[1, 1].set_xlabel('検定番号（p値でソート済み）')
    axes[1, 1].set_ylabel('p値')
    axes[1, 1].set_title('Benjamini-Hochberg (FDR制御)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # まとめ
    print("\n=== 補正法の比較 ===")
    print(f"補正なし: {np.sum(p_values < alpha)}個有意")
    print(f"Bonferroni: {np.sum(results['Bonferroni']['reject'])}個有意（最も保守的）")
    print(f"Holm: {np.sum(results['Holm']['reject'])}個有意")
    print(f"Benjamini-Hochberg: {np.sum(results['Benjamini-Hochberg']['reject'])}個有意（検定力が高い）")

## 3.7 品質管理における仮説検定

#### 💻 コード例7: 品質管理データの包括的検定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # 品質管理シナリオ: 製造ラインの品質データ
    np.random.seed(42)
    
    # 従来プロセス（基準）
    baseline_mean = 500
    baseline_std = 15
    n_baseline = 50
    baseline_data = np.random.normal(baseline_mean, baseline_std, n_baseline)
    
    # 改善プロセス（検証対象）
    improved_mean = 505  # 平均が向上
    improved_std = 12    # ばらつきが減少
    n_improved = 60
    improved_data = np.random.normal(improved_mean, improved_std, n_improved)
    
    print("=== 品質管理における統計的検定 ===")
    
    # 1. 平均の検定（t検定）
    print("\n【1. 平均強度の改善検定】")
    t_stat, p_value_mean = stats.ttest_ind(improved_data, baseline_data, equal_var=False)
    print(f"従来プロセス平均: {np.mean(baseline_data):.2f} MPa")
    print(f"改善プロセス平均: {np.mean(improved_data):.2f} MPa")
    print(f"t統計量: {t_stat:.4f}, p値: {p_value_mean:.6f}")
    print(f"判定: {'有意に改善' if p_value_mean < 0.05 and t_stat > 0 else '有意な改善なし'}")
    
    # 2. 分散の検定（F検定）
    print("\n【2. ばらつきの改善検定】")
    var_baseline = np.var(baseline_data, ddof=1)
    var_improved = np.var(improved_data, ddof=1)
    F_stat = var_baseline / var_improved
    df1, df2 = n_baseline - 1, n_improved - 1
    p_value_var = stats.f.sf(F_stat, df1, df2)  # 片側検定（ばらつき減少）
    
    print(f"従来プロセス分散: {var_baseline:.2f}")
    print(f"改善プロセス分散: {var_improved:.2f}")
    print(f"F統計量: {F_stat:.4f}, p値: {p_value_var:.6f}")
    print(f"判定: {'ばらつきが有意に減少' if p_value_var < 0.05 else 'ばらつきの有意な減少なし'}")
    
    # 3. 規格外れ率の検定（比率の検定）
    print("\n【3. 規格外れ率の検定】")
    spec_lower = 470  # 下限規格
    spec_upper = 530  # 上限規格
    
    defect_baseline = np.sum((baseline_data < spec_lower) | (baseline_data > spec_upper))
    defect_improved = np.sum((improved_data < spec_lower) | (improved_data > spec_upper))
    
    p_baseline = defect_baseline / n_baseline
    p_improved = defect_improved / n_improved
    
    # 2標本の比率の検定
    pooled_p = (defect_baseline + defect_improved) / (n_baseline + n_improved)
    se_pool = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_baseline + 1/n_improved))
    z_prop = (p_baseline - p_improved) / se_pool if se_pool > 0 else 0
    p_value_prop = stats.norm.sf(z_prop)  # 片側検定
    
    print(f"従来プロセス規格外れ率: {p_baseline:.4f} ({defect_baseline}/{n_baseline})")
    print(f"改善プロセス規格外れ率: {p_improved:.4f} ({defect_improved}/{n_improved})")
    print(f"z統計量: {z_prop:.4f}, p値: {p_value_prop:.6f}")
    print(f"判定: {'規格外れ率が有意に改善' if p_value_prop < 0.05 else '有意な改善なし'}")
    
    # 4. 工程能力指数の比較
    print("\n【4. 工程能力指数 Cp, Cpk】")
    spec_range = spec_upper - spec_lower
    
    Cp_baseline = spec_range / (6 * np.std(baseline_data, ddof=1))
    Cpk_baseline_lower = (np.mean(baseline_data) - spec_lower) / (3 * np.std(baseline_data, ddof=1))
    Cpk_baseline_upper = (spec_upper - np.mean(baseline_data)) / (3 * np.std(baseline_data, ddof=1))
    Cpk_baseline = min(Cpk_baseline_lower, Cpk_baseline_upper)
    
    Cp_improved = spec_range / (6 * np.std(improved_data, ddof=1))
    Cpk_improved_lower = (np.mean(improved_data) - spec_lower) / (3 * np.std(improved_data, ddof=1))
    Cpk_improved_upper = (spec_upper - np.mean(improved_data)) / (3 * np.std(improved_data, ddof=1))
    Cpk_improved = min(Cpk_improved_lower, Cpk_improved_upper)
    
    print(f"従来プロセス: Cp={Cp_baseline:.3f}, Cpk={Cpk_baseline:.3f}")
    print(f"改善プロセス: Cp={Cp_improved:.3f}, Cpk={Cpk_improved:.3f}")
    print(f"評価: Cpk ≥ 1.33 で優秀、≥ 1.00 で許容可能")
    
    # 可視化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. データの分布比較
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(baseline_data, bins=20, density=True, alpha=0.6,
             color='skyblue', edgecolor='black', label='従来プロセス')
    ax1.hist(improved_data, bins=20, density=True, alpha=0.6,
             color='lightgreen', edgecolor='black', label='改善プロセス')
    ax1.axvline(spec_lower, color='red', linestyle='--', linewidth=2, label='規格下限')
    ax1.axvline(spec_upper, color='red', linestyle='--', linewidth=2, label='規格上限')
    ax1.axvline(np.mean(baseline_data), color='blue', linestyle=':', linewidth=2)
    ax1.axvline(np.mean(improved_data), color='green', linestyle=':', linewidth=2)
    ax1.set_xlabel('強度 [MPa]')
    ax1.set_ylabel('密度')
    ax1.set_title(f'プロセス改善の効果\n平均: {np.mean(baseline_data):.1f} → {np.mean(improved_data):.1f} (p={p_value_mean:.4f}), '
                  f'SD: {np.std(baseline_data, ddof=1):.1f} → {np.std(improved_data, ddof=1):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱ひげ図
    ax2 = fig.add_subplot(gs[1, 0])
    bp = ax2.boxplot([baseline_data, improved_data], labels=['従来', '改善'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax2.axhline(spec_lower, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(spec_upper, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('強度 [MPa]')
    ax2.set_title('ばらつきの比較')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Qプロット（正規性確認）
    ax3 = fig.add_subplot(gs[1, 1])
    stats.probplot(improved_data, dist="norm", plot=ax3)
    ax3.set_title('改善プロセスの正規Q-Qプロット')
    ax3.grid(True, alpha=0.3)
    
    # 4. 工程能力指数の比較
    ax4 = fig.add_subplot(gs[2, 0])
    metrics = ['Cp', 'Cpk']
    baseline_metrics = [Cp_baseline, Cpk_baseline]
    improved_metrics = [Cp_improved, Cpk_improved]
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, baseline_metrics, width, label='従来',
            color='lightblue', edgecolor='black')
    ax4.bar(x + width/2, improved_metrics, width, label='改善',
            color='lightgreen', edgecolor='black')
    ax4.axhline(1.33, color='green', linestyle='--', linewidth=2, label='優秀ライン')
    ax4.axhline(1.00, color='orange', linestyle='--', linewidth=2, label='許容ライン')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.set_ylabel('工程能力指数')
    ax4.set_title('工程能力の改善')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 規格外れ率の比較
    ax5 = fig.add_subplot(gs[2, 1])
    categories = ['従来プロセス', '改善プロセス']
    defect_rates = [p_baseline * 100, p_improved * 100]
    bars = ax5.bar(categories, defect_rates, color=['lightblue', 'lightgreen'],
                   edgecolor='black')
    for bar, rate in zip(bars, defect_rates):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                 f'{rate:.2f}%', ha='center', va='bottom', fontsize=12)
    ax5.set_ylabel('規格外れ率 [%]')
    ax5.set_title(f'規格外れ率の改善 (p={p_value_prop:.4f})')
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 総合評価 ===")
    improvements = []
    if p_value_mean < 0.05 and t_stat > 0:
        improvements.append("✓ 平均強度が有意に向上")
    if p_value_var < 0.05:
        improvements.append("✓ ばらつきが有意に減少")
    if p_value_prop < 0.05:
        improvements.append("✓ 規格外れ率が有意に改善")
    if Cpk_improved > Cpk_baseline:
        improvements.append(f"✓ 工程能力指数が改善 ({Cpk_baseline:.2f} → {Cpk_improved:.2f})")
    
    if improvements:
        print("プロセス改善は統計的に有意:")
        for imp in improvements:
            print(f"  {imp}")
    else:
        print("統計的に有意な改善は確認できませんでした")

#### 📝 練習問題

  1. 片側検定と両側検定の使い分けについて、具体例を挙げて説明してください。
  2. p値が0.06の場合、どのように解釈すべきか議論してください。
  3. 検定力0.8を達成するために必要なサンプルサイズを、効果量d=0.3, 0.5, 0.8で計算してください。
  4. 5つの仮説検定を行う場合、Bonferroni補正後の各検定の有意水準はいくつになりますか？

## まとめ

  * 仮説検定は帰無仮説を統計的に評価する枠組みである
  * 第1種過誤（α）と第2種過誤（β）のバランスが重要
  * p値は帰無仮説が真のときの極端さの確率であり、帰無仮説の真偽の確率ではない
  * t検定は母平均の比較に、F検定は分散の比較に用いる
  * カイ二乗検定は度数データや独立性の検定に有効
  * 検定力分析は実験計画で必須のステップである
  * 多重比較では適切な補正法（Bonferroni, Holm, FDR）を選択する
  * 品質管理では複数の統計的指標を組み合わせて評価する

[← 第2章: 区間推定と信頼区間](<chapter-2.html>) 第4章: ベイズ推論の基礎とMCMC →（準備中）

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
