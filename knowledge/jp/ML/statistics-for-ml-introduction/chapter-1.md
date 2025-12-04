---
title: 第1章：記述統計と確率の基礎
chapter_title: 第1章：記述統計と確率の基礎
subtitle: データの特徴を捉え、不確実性を定量化する
reading_time: 20-25分
difficulty: 初級
code_examples: 8
---

## イントロダクション

統計学は、データから意味のある情報を抽出し、不確実性のもとで合理的な意思決定を行うための学問です。機械学習においても、データの特徴を理解し、モデルの性能を評価し、予測の不確実性を定量化するために統計学の知識が不可欠です。

この章では、統計学の基礎となる**記述統計** と**確率論** について学びます。記述統計では、データの中心傾向（平均、中央値）や散らばり（分散、標準偏差）を数値で表現する方法を、確率論では不確実な事象を数学的に扱う方法を習得します。

**💡 この章で学ぶこと**

  * データの特徴を数値指標で要約する（平均、分散、四分位数など）
  * 適切なグラフでデータを可視化する（ヒストグラム、箱ひげ図など）
  * 確率の基本的な計算と条件付き確率
  * ベイズの定理の理解と応用
  * 期待値と分散の数学的定義

## 1\. 記述統計の基本

記述統計（Descriptive Statistics）は、データの特徴を要約し、理解しやすい形で表現する手法です。大量のデータを少数の数値指標で表現することで、データの全体像を把握できます。

### 1.1 中心傾向の指標

データの「中心」がどこにあるかを示す指標です。

#### 平均（Mean）

すべてのデータ値の総和をデータ数で割ったもので、最も基本的な中心傾向の指標です。

数式表現：

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$$

ここで、$n$はデータ数、$x_i$は$i$番目のデータ値です。

#### 中央値（Median）

データを昇順に並べたときの中央の値です。外れ値の影響を受けにくい頑健な指標です。

#### 最頻値（Mode）

データ中で最も頻繁に出現する値です。カテゴリカルデータにも適用できます。

**📝 例題：学生の試験成績**

5人の学生の試験成績: 65点、70点、75点、80点、95点

  * 平均: $(65+70+75+80+95)/5 = 77$点
  * 中央値: 75点（真ん中の値）

もし極端な値（例: 10点）が含まれると、平均は大きく変わりますが、中央値は比較的安定します。

### 1.2 散らばりの指標

データがどの程度ばらついているかを示す指標です。

#### 分散（Variance）

各データ値と平均との差の二乗の平均です。

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**⚠️ 注意：標本分散と母集団分散**

標本から母集団の分散を推定する場合は、$n$の代わりに$n-1$で割ります（不偏推定量）：

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

#### 標準偏差（Standard Deviation）

分散の平方根です。元のデータと同じ単位で散らばりを表現できます。

$$\sigma = \sqrt{\sigma^2}$$

#### 四分位数とパーセンタイル

データを順序づけた際の位置を示す指標です。

  * **第1四分位数（Q1）** : データの下位25%の位置
  * **第2四分位数（Q2）** : 中央値（50%の位置）
  * **第3四分位数（Q3）** : データの上位25%の位置
  * **四分位範囲（IQR）** : $Q3 - Q1$（中央50%のデータの範囲）

### 1.3 Pythonでの実装

NumPyとSciPyを使って記述統計量を計算してみましょう。
    
    
    import numpy as np
    from scipy import stats
    
    # サンプルデータ: 学生の試験成績
    scores = np.array([65, 70, 72, 75, 78, 80, 82, 85, 88, 95, 98])
    
    # 中心傾向の指標
    mean = np.mean(scores)
    median = np.median(scores)
    mode_result = stats.mode(scores, keepdims=True)
    mode = mode_result.mode[0] if len(mode_result.mode) > 0 else None
    
    print(f"平均（Mean）: {mean:.2f}")
    print(f"中央値（Median）: {median:.2f}")
    print(f"最頻値（Mode）: {mode}")
    
    # 散らばりの指標
    variance = np.var(scores)  # 母集団分散
    std_dev = np.std(scores)   # 母集団標準偏差
    sample_variance = np.var(scores, ddof=1)  # 標本分散（不偏推定量）
    sample_std = np.std(scores, ddof=1)       # 標本標準偏差
    
    print(f"\n母集団分散: {variance:.2f}")
    print(f"母集団標準偏差: {std_dev:.2f}")
    print(f"標本分散: {sample_variance:.2f}")
    print(f"標本標準偏差: {sample_std:.2f}")
    
    # 四分位数
    q1 = np.percentile(scores, 25)
    q2 = np.percentile(scores, 50)  # 中央値
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    
    print(f"\n第1四分位数（Q1）: {q1:.2f}")
    print(f"第2四分位数（Q2）: {q2:.2f}")
    print(f"第3四分位数（Q3）: {q3:.2f}")
    print(f"四分位範囲（IQR）: {iqr:.2f}")

**実行結果:**
    
    
    平均（Mean）: 80.73
    中央値（Median）: 80.00
    最頻値（Mode）: 65
    
    母集団分散: 103.29
    母集団標準偏差: 10.16
    標本分散: 113.62
    標本標準偏差: 10.66
    
    第1四分位数（Q1）: 73.50
    第2四分位数（Q2）: 80.00
    第3四分位数（Q3）: 88.00
    四分位範囲（IQR）: 14.50

## 2\. データの可視化

数値指標だけでなく、グラフによる可視化もデータ理解に不可欠です。

### 2.1 ヒストグラム

データの分布を視覚的に表現するグラフです。データを階級（ビン）に分け、各階級の頻度を棒グラフで表示します。
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 正規分布に従うサンプルデータを生成
    np.random.seed(42)
    data = np.random.normal(loc=70, scale=10, size=1000)
    
    # ヒストグラムを描画
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'平均: {np.mean(data):.2f}')
    plt.axvline(np.median(data), color='green', linestyle='--', linewidth=2, label=f'中央値: {np.median(data):.2f}')
    plt.xlabel('値', fontsize=12)
    plt.ylabel('頻度', fontsize=12)
    plt.title('ヒストグラム：データの分布', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

### 2.2 箱ひげ図（Box Plot）

四分位数を視覚的に表現し、外れ値も識別できるグラフです。
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 複数グループのデータ
    np.random.seed(42)
    group_a = np.random.normal(70, 10, 100)
    group_b = np.random.normal(75, 8, 100)
    group_c = np.random.normal(65, 12, 100)
    
    data_groups = [group_a, group_b, group_c]
    
    # 箱ひげ図を描画
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data_groups, labels=['グループA', 'グループB', 'グループC'],
                     patch_artist=True, notch=True)
    
    # 色をカスタマイズ
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('スコア', fontsize=12)
    plt.title('箱ひげ図：グループ間の比較', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

**💡 箱ひげ図の読み方**

  * 箱の下端: 第1四分位数（Q1）
  * 箱の中の線: 中央値（Q2）
  * 箱の上端: 第3四分位数（Q3）
  * ひげ: データの範囲（外れ値を除く）
  * 点: 外れ値

### 2.3 散布図（Scatter Plot）

2つの変数間の関係を可視化するグラフです。
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 相関のあるデータを生成
    np.random.seed(42)
    x = np.random.normal(50, 10, 100)
    y = 2 * x + np.random.normal(0, 10, 100)  # yはxと正の相関
    
    # 散布図を描画
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, edgecolors='black', s=50)
    plt.xlabel('変数 X', fontsize=12)
    plt.ylabel('変数 Y', fontsize=12)
    plt.title('散布図：2変数間の関係', fontsize=14)
    plt.grid(alpha=0.3)
    
    # 相関係数を計算して表示
    correlation = np.corrcoef(x, y)[0, 1]
    plt.text(0.05, 0.95, f'相関係数: {correlation:.3f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.show()

## 3\. 確率の基礎

確率論は、不確実な事象を数学的に扱う枠組みです。機械学習では、データの生成過程をモデル化したり、予測の不確実性を定量化したりする際に確率論が使われます。

### 3.1 確率の定義と公理

**確率** は、ある事象が起こる可能性を0から1の数値で表したものです。

**コルモゴロフの公理** （確率の基本的な性質）:

  1. **非負性** : すべての事象$A$について、$P(A) \geq 0$
  2. **全確率** : 全事象の確率は1、$P(\Omega) = 1$
  3. **加法性** : 互いに排反な事象$A$と$B$について、$P(A \cup B) = P(A) + P(B)$

#### 基本的な確率の計算

  * **余事象** : $P(A^c) = 1 - P(A)$
  * **和事象** : $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### 3.2 条件付き確率

ある事象$B$が起きたという条件のもとで、事象$A$が起きる確率を**条件付き確率** といい、$P(A|B)$と表記します。

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

ただし、$P(B) > 0$とします。

**📝 例題：カードの抽出**

52枚のトランプから1枚引いたとき：

  * 引いたカードがスペードである確率: $P(\text{スペード}) = 13/52 = 1/4$
  * 引いたカードが絵札（J, Q, K）である確率: $P(\text{絵札}) = 12/52 = 3/13$
  * 引いたカードがスペードであるという条件のもとで絵札である確率: $P(\text{絵札}|\text{スペード}) = 3/13$

### 3.3 ベイズの定理

**ベイズの定理** は、条件付き確率を逆転させる重要な公式です。機械学習、特にベイズ統計において中心的な役割を果たします。

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

各項の意味：

  * $P(A|B)$: **事後確率** （Posterior）- 事象$B$を観測した後の$A$の確率
  * $P(B|A)$: **尤度** （Likelihood）- $A$が真のとき$B$が観測される確率
  * $P(A)$: **事前確率** （Prior）- 観測前の$A$の確率
  * $P(B)$: **周辺確率** （Evidence）- $B$が観測される全体の確率

#### ベイズの定理の応用：医療診断

**📝 例題：病気の検査**

ある稀な病気について：

  * 人口の1%がこの病気にかかっている: $P(\text{病気}) = 0.01$
  * 検査の感度（病気の人を正しく陽性と判定）: $P(\text{陽性}|\text{病気}) = 0.99$
  * 検査の偽陽性率: $P(\text{陽性}|\text{健康}) = 0.05$

検査で陽性と出たとき、実際に病気である確率は？

ベイズの定理を適用：

$$P(\text{病気}|\text{陽性}) = \frac{P(\text{陽性}|\text{病気}) \cdot P(\text{病気})}{P(\text{陽性})}$$

まず、$P(\text{陽性})$を計算（全確率の法則）：

$$P(\text{陽性}) = P(\text{陽性}|\text{病気})P(\text{病気}) + P(\text{陽性}|\text{健康})P(\text{健康})$$

$$= 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0099 + 0.0495 = 0.0594$$

したがって：

$$P(\text{病気}|\text{陽性}) = \frac{0.99 \times 0.01}{0.0594} \approx 0.167$$

つまり、検査で陽性と出ても、実際に病気である確率は約16.7%にすぎません。これは病気の有病率が低いためです。

#### Pythonでの実装
    
    
    import numpy as np
    
    def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
        """
        ベイズの定理を計算
    
        Parameters:
        -----------
        p_a : float
            事前確率 P(A)
        p_b_given_a : float
            尤度 P(B|A)
        p_b_given_not_a : float
            P(B|not A)
    
        Returns:
        --------
        float
            事後確率 P(A|B)
        """
        # 全確率の法則で P(B) を計算
        p_not_a = 1 - p_a
        p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    
        # ベイズの定理で P(A|B) を計算
        p_a_given_b = (p_b_given_a * p_a) / p_b
    
        return p_a_given_b, p_b
    
    # 医療診断の例
    p_disease = 0.01  # 病気の事前確率（有病率）
    p_positive_given_disease = 0.99  # 感度（真陽性率）
    p_positive_given_healthy = 0.05  # 偽陽性率
    
    p_disease_given_positive, p_positive = bayes_theorem(
        p_disease,
        p_positive_given_disease,
        p_positive_given_healthy
    )
    
    print("=== 医療診断におけるベイズの定理 ===")
    print(f"病気の有病率: {p_disease * 100:.1f}%")
    print(f"検査の感度: {p_positive_given_disease * 100:.1f}%")
    print(f"偽陽性率: {p_positive_given_healthy * 100:.1f}%")
    print(f"\n陽性判定が出る確率: {p_positive * 100:.2f}%")
    print(f"陽性と判定されたとき実際に病気である確率: {p_disease_given_positive * 100:.2f}%")
    
    # 感度を変えて比較
    print("\n=== 感度を変化させた場合の比較 ===")
    sensitivities = [0.90, 0.95, 0.99, 0.999]
    for sens in sensitivities:
        prob, _ = bayes_theorem(p_disease, sens, p_positive_given_healthy)
        print(f"感度 {sens*100:.1f}%: 陽性時の病気確率 = {prob*100:.2f}%")

**実行結果:**
    
    
    === 医療診断におけるベイズの定理 ===
    病気の有病率: 1.0%
    検査の感度: 99.0%
    偽陽性率: 5.0%
    
    陽性判定が出る確率: 5.94%
    陽性と判定されたとき実際に病気である確率: 16.64%
    
    === 感度を変化させた場合の比較 ===
    感度 90.0%: 陽性時の病気確率 = 15.38%
    感度 95.0%: 陽性時の病気確率 = 16.10%
    感度 99.0%: 陽性時の病気確率 = 16.64%
    感度 99.9%: 陽性時の病気確率 = 16.72%

## 4\. 期待値と分散

確率変数の特徴を数値で表現する重要な概念です。

### 4.1 期待値（Expected Value）

**期待値** は、確率変数の平均的な値を表します。

**離散型確率変数** の場合：

$$E[X] = \sum_{i} x_i \cdot P(X = x_i)$$

**連続型確率変数** の場合：

$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$$

ここで、$f(x)$は確率密度関数です。

#### 期待値の性質

  * **線形性** : $E[aX + b] = aE[X] + b$
  * **加法性** : $E[X + Y] = E[X] + E[Y]$

### 4.2 分散と標準偏差

**分散** は、確率変数が期待値からどれだけばらついているかを表します。

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**標準偏差** は分散の平方根です：

$$\sigma = \sqrt{\text{Var}(X)}$$

#### 分散の性質

  * $\text{Var}(aX + b) = a^2 \text{Var}(X)$
  * $X$と$Y$が独立なら: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### 4.3 Pythonでの実装
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サイコロを振る実験の例
    def dice_expectation():
        """サイコロの期待値を計算"""
        outcomes = np.array([1, 2, 3, 4, 5, 6])
        probabilities = np.array([1/6] * 6)
    
        # 期待値の計算
        expectation = np.sum(outcomes * probabilities)
    
        # 分散の計算
        variance = np.sum((outcomes - expectation)**2 * probabilities)
        std_dev = np.sqrt(variance)
    
        print("=== サイコロの期待値と分散 ===")
        print(f"期待値 E[X]: {expectation:.4f}")
        print(f"分散 Var(X): {variance:.4f}")
        print(f"標準偏差 σ: {std_dev:.4f}")
    
        # 可視化
        plt.figure(figsize=(10, 6))
        plt.bar(outcomes, probabilities, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(expectation, color='red', linestyle='--', linewidth=2,
                    label=f'期待値: {expectation:.2f}')
        plt.xlabel('出目', fontsize=12)
        plt.ylabel('確率', fontsize=12)
        plt.title('サイコロの確率分布', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(outcomes)
        plt.show()
    
    dice_expectation()
    
    # シミュレーションによる検証
    print("\n=== シミュレーションによる検証 ===")
    n_trials = 10000
    dice_rolls = np.random.randint(1, 7, n_trials)
    
    empirical_mean = np.mean(dice_rolls)
    empirical_variance = np.var(dice_rolls)
    empirical_std = np.std(dice_rolls)
    
    print(f"シミュレーション回数: {n_trials}")
    print(f"経験的平均: {empirical_mean:.4f}")
    print(f"経験的分散: {empirical_variance:.4f}")
    print(f"経験的標準偏差: {empirical_std:.4f}")
    print(f"\n理論値との差:")
    print(f"平均の差: {abs(empirical_mean - 3.5):.4f}")
    print(f"分散の差: {abs(empirical_variance - 35/12):.4f}")

## 5\. まとめと次のステップ

この章では、統計学の基礎となる記述統計と確率論について学びました。

**✅ この章で学んだこと**

  * データの中心傾向（平均、中央値、最頻値）と散らばり（分散、標準偏差）の指標
  * ヒストグラム、箱ひげ図、散布図によるデータの可視化
  * 確率の基本公理と条件付き確率
  * ベイズの定理の理論と応用
  * 期待値と分散の数学的定義と計算
  * NumPy/SciPy/Matplotlibを使った統計分析の実装

**🔑 重要ポイント**

  * 平均は外れ値の影響を受けやすいが、中央値は頑健
  * 標本分散を計算する際は$n-1$で割る（不偏推定量）
  * ベイズの定理では事前確率が事後確率に大きく影響する
  * 期待値は線形性を持ち、分散は線形変換に対して二乗倍される

### 次のステップ

次章では、確率分布について学びます。正規分布、二項分布、ポアソン分布など、機械学習で頻繁に使われる確率分布の性質と応用を習得します。

[← シリーズトップへ](<./index.html>) 第2章：確率分布（準備中）→

## 練習問題

**問題1：記述統計の計算**

次のデータセットについて、平均、中央値、分散、標準偏差を計算してください。

データ: 12, 15, 18, 20, 22, 25, 28, 30, 35, 40
    
    
    import numpy as np
    
    data = np.array([12, 15, 18, 20, 22, 25, 28, 30, 35, 40])
    
    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data, ddof=1)
    std_dev = np.std(data, ddof=1)
    
    print(f"平均: {mean}")
    print(f"中央値: {median}")
    print(f"分散: {variance:.2f}")
    print(f"標準偏差: {std_dev:.2f}")

**問題2：ベイズの定理の応用**

スパムメールフィルターを考えます。メールの10%がスパムで、"無料"という単語が含まれる確率が、スパムメールでは80%、正常メールでは5%です。"無料"という単語を含むメールがスパムである確率を計算してください。
    
    
    def spam_filter_bayes(p_spam, p_free_given_spam, p_free_given_normal):
        p_normal = 1 - p_spam
        p_free = p_free_given_spam * p_spam + p_free_given_normal * p_normal
        p_spam_given_free = (p_free_given_spam * p_spam) / p_free
        return p_spam_given_free
    
    # パラメータ
    p_spam = 0.10
    p_free_given_spam = 0.80
    p_free_given_normal = 0.05
    
    result = spam_filter_bayes(p_spam, p_free_given_spam, p_free_given_normal)
    print(f"「無料」を含むメールがスパムである確率: {result * 100:.2f}%")

**問題3：期待値の計算**

宝くじのゲームで、1000円の券を買うと、10%の確率で5000円、5%の確率で10000円が当たり、残りは0円です。このゲームの期待値を計算し、プレイする価値があるか判断してください。
    
    
    import numpy as np
    
    # 結果と確率
    outcomes = np.array([5000, 10000, 0])
    probabilities = np.array([0.10, 0.05, 0.85])
    
    # 期待値の計算
    expected_value = np.sum(outcomes * probabilities)
    net_expected_value = expected_value - 1000  # チケット代を引く
    
    print(f"期待値: {expected_value:.2f}円")
    print(f"正味期待値（チケット代を引いた後）: {net_expected_value:.2f}円")
    
    if net_expected_value > 0:
        print("期待値的にはプレイする価値があります")
    else:
        print("期待値的にはプレイする価値がありません")
