---
title: 第1章：確率統計の基礎
chapter_title: 第1章：確率統計の基礎
---

**機械学習の基盤となる確率統計を理論と実装の両面から深く理解する**

**この章で学べること**

  * ベイズの定理と条件付き確率の数学的理解
  * 正規分布と多変量正規分布の性質と実装
  * 期待値、分散、共分散の計算と幾何的意味
  * 最尤推定とベイズ推定の理論的違い
  * 機械学習アルゴリズムへの確率統計の応用

## 1\. 確率の基礎

### 1.1 条件付き確率とベイズの定理

条件付き確率は、ある事象Bが起きたという条件の下での事象Aの確率を表します。

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$ 

ベイズの定理は、事前確率と尤度から事後確率を計算する基本定理です。

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)} = \frac{P(B|A)P(A)}{\sum_{i} P(B|A_i)P(A_i)}$$ 

ここで：

  * **P(A)** : 事前確率（prior）- 観測前の仮説の確率
  * **P(B|A)** : 尤度（likelihood）- 仮説Aの下でデータBが観測される確率
  * **P(A|B)** : 事後確率（posterior）- データ観測後の仮説の確率
  * **P(B)** : 周辺尤度（evidence）- データの正規化定数

**機械学習での応用** ベイズの定理は、ナイーブベイズ分類器、ベイズ線形回帰、ベイズ最適化など、多くの機械学習アルゴリズムの基盤となっています。 

### 実装例1：ナイーブベイズ分類器

ベイズの定理を用いた文書分類の実装例です。各単語の出現が独立と仮定します。
    
    
    import numpy as np
    from collections import defaultdict
    
    class NaiveBayesClassifier:
        """ナイーブベイズ分類器の実装"""
    
        def __init__(self, alpha=1.0):
            """
            Parameters:
            -----------
            alpha : float
                ラプラス平滑化パラメータ（加算平滑化）
            """
            self.alpha = alpha
            self.class_priors = {}
            self.word_probs = defaultdict(dict)
            self.vocab = set()
    
        def fit(self, X, y):
            """
            訓練データから確率を学習
    
            Parameters:
            -----------
            X : list of list
                各文書の単語リスト
            y : list
                各文書のクラスラベル
            """
            n_docs = len(X)
            class_counts = defaultdict(int)
            word_counts = defaultdict(lambda: defaultdict(int))
    
            # クラスごとの文書数と単語出現回数をカウント
            for doc, label in zip(X, y):
                class_counts[label] += 1
                for word in doc:
                    self.vocab.add(word)
                    word_counts[label][word] += 1
    
            # 事前確率 P(class) を計算
            for label, count in class_counts.items():
                self.class_priors[label] = count / n_docs
    
            # 尤度 P(word|class) を計算（ラプラス平滑化適用）
            vocab_size = len(self.vocab)
            for label in class_counts:
                total_words = sum(word_counts[label].values())
                for word in self.vocab:
                    word_count = word_counts[label].get(word, 0)
                    # P(word|class) with Laplace smoothing
                    self.word_probs[label][word] = (
                        (word_count + self.alpha) /
                        (total_words + self.alpha * vocab_size)
                    )
    
        def predict(self, X):
            """
            ベイズの定理で事後確率を計算し、最も確率の高いクラスを予測
    
            log P(class|doc) = log P(class) + Σ log P(word|class)
            """
            predictions = []
            for doc in X:
                class_scores = {}
                for label in self.class_priors:
                    # 対数事後確率を計算（数値安定性のため）
                    score = np.log(self.class_priors[label])
                    for word in doc:
                        if word in self.vocab:
                            score += np.log(self.word_probs[label][word])
                    class_scores[label] = score
                predictions.append(max(class_scores, key=class_scores.get))
            return predictions
    
    # 使用例
    X_train = [
        ['機械', '学習', '深層', '学習'],
        ['統計', '確率', '分布'],
        ['深層', 'ニューラル', 'ネットワーク'],
        ['確率', 'ベイズ', '統計']
    ]
    y_train = ['ML', 'Stats', 'ML', 'Stats']
    
    nb = NaiveBayesClassifier(alpha=1.0)
    nb.fit(X_train, y_train)
    
    X_test = [['機械', '学習'], ['ベイズ', '確率']]
    predictions = nb.predict(X_test)
    print(f"予測結果: {predictions}")  # ['ML', 'Stats']
    

## 2\. 確率分布

### 2.1 正規分布（ガウス分布）

正規分布は自然界や測定誤差など、多くの現象で観測される最も重要な連続確率分布です。

$$\mathcal{N}(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$ 

ここで：

  * **μ** : 平均（期待値）- 分布の中心位置
  * **σ²** : 分散 - データの散らばり具合
  * **σ** : 標準偏差 - 分散の平方根

**中心極限定理** 独立同分布な確率変数の和は、元の分布の形に関わらず、サンプル数が大きくなると正規分布に近づきます。これが正規分布が重要な理由の一つです。 

### 2.2 多変量正規分布

多次元データの確率分布を記述する多変量正規分布は、機械学習で頻繁に使用されます。

$$\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$ 

ここで：

  * **μ** : D次元平均ベクトル
  * **Σ** : D×D共分散行列（対称正定値行列）
  * **|Σ|** : 共分散行列の行列式

### 実装例2：多変量正規分布の可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    
    def plot_multivariate_gaussian():
        """2次元多変量正規分布の可視化"""
    
        # 平均ベクトルと共分散行列を定義
        mu = np.array([0, 0])
    
        # 異なる共分散行列のケース
        covariances = [
            np.array([[1, 0], [0, 1]]),           # 独立・等分散
            np.array([[2, 0], [0, 0.5]]),         # 独立・異分散
            np.array([[1, 0.8], [0.8, 1]]),       # 正の相関
            np.array([[1, -0.8], [-0.8, 1]])      # 負の相関
        ]
    
        titles = ['独立・等分散', '独立・異分散', '正の相関', '負の相関']
    
        # グリッドポイントの生成
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
    
        for idx, (cov, title) in enumerate(zip(covariances, titles)):
            # 多変量正規分布の計算
            rv = multivariate_normal(mu, cov)
            Z = rv.pdf(pos)
    
            # 等高線プロット
            axes[idx].contour(X, Y, Z, levels=10, cmap='viridis')
            axes[idx].set_title(f'{title}\nΣ = {cov.tolist()}')
            axes[idx].set_xlabel('x₁')
            axes[idx].set_ylabel('x₂')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].axis('equal')
    
        plt.tight_layout()
        plt.savefig('multivariate_gaussian.png', dpi=150, bbox_inches='tight')
        print("多変量正規分布の可視化を保存しました")
    
    # 実行
    plot_multivariate_gaussian()
    
    # 固有値分解による共分散行列の解析
    cov = np.array([[2, 1], [1, 2]])
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    print(f"\n共分散行列:\n{cov}")
    print(f"固有値: {eigenvalues}")
    print(f"固有ベクトル:\n{eigenvectors}")
    print(f"主軸の方向（第1固有ベクトル）: {eigenvectors[:, 0]}")
    

## 3\. 期待値と分散

### 3.1 期待値

期待値（平均）は確率変数の「中心的な値」を表します。

$$\mathbb{E}[X] = \sum_{x} x \cdot P(X=x) \quad \text{（離散）}$$ $$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) dx \quad \text{（連続）}$$ 

**期待値の性質：**

  * 線形性: \\(\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]\\)
  * 独立変数の積: \\(\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]\\) （X, Y が独立の場合）

### 3.2 分散と共分散

分散はデータの散らばり具合を表す指標です。

$$\text{Var}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$ 

共分散は2つの確率変数の同時変動を表します。

$$\text{Cov}[X, Y] = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$ 

相関係数は共分散を正規化した値で、-1から1の範囲を取ります。

$$\rho_{X,Y} = \frac{\text{Cov}[X, Y]}{\sqrt{\text{Var}[X]\text{Var}[Y]}}$$ 

### 実装例3：期待値・分散・共分散の計算
    
    
    import numpy as np
    
    class StatisticsCalculator:
        """確率統計の基本量を計算するクラス"""
    
        @staticmethod
        def expectation(X, P=None):
            """
            期待値の計算
    
            Parameters:
            -----------
            X : array-like
                確率変数の値
            P : array-like, optional
                各値の確率（Noneの場合は一様分布と仮定）
    
            Returns:
            --------
            float : 期待値
            """
            X = np.array(X)
            if P is None:
                return np.mean(X)
            else:
                P = np.array(P)
                assert abs(np.sum(P) - 1.0) < 1e-10, "確率の和は1でなければなりません"
                return np.sum(X * P)
    
        @staticmethod
        def variance(X, P=None):
            """
            分散の計算: Var[X] = E[X²] - (E[X])²
            """
            X = np.array(X)
            E_X = StatisticsCalculator.expectation(X, P)
            E_X2 = StatisticsCalculator.expectation(X**2, P)
            return E_X2 - E_X**2
    
        @staticmethod
        def covariance(X, Y):
            """
            共分散の計算: Cov[X,Y] = E[XY] - E[X]E[Y]
            """
            X, Y = np.array(X), np.array(Y)
            assert len(X) == len(Y), "XとYの長さが一致しません"
    
            E_X = np.mean(X)
            E_Y = np.mean(Y)
            E_XY = np.mean(X * Y)
    
            return E_XY - E_X * E_Y
    
        @staticmethod
        def correlation(X, Y):
            """
            相関係数の計算: ρ = Cov[X,Y] / (σ_X * σ_Y)
            """
            cov_XY = StatisticsCalculator.covariance(X, Y)
            std_X = np.sqrt(StatisticsCalculator.variance(X))
            std_Y = np.sqrt(StatisticsCalculator.variance(Y))
    
            return cov_XY / (std_X * std_Y)
    
        @staticmethod
        def covariance_matrix(data):
            """
            共分散行列の計算
    
            Parameters:
            -----------
            data : ndarray of shape (n_samples, n_features)
                データ行列
    
            Returns:
            --------
            ndarray : 共分散行列 (n_features, n_features)
            """
            data = np.array(data)
            n_samples, n_features = data.shape
    
            # 各特徴量の平均を計算
            means = np.mean(data, axis=0)
    
            # 中心化
            centered_data = data - means
    
            # 共分散行列: (1/n) * X^T X
            cov_matrix = (centered_data.T @ centered_data) / n_samples
    
            return cov_matrix
    
    # 使用例
    calc = StatisticsCalculator()
    
    # 離散確率変数（サイコロ）
    X = [1, 2, 3, 4, 5, 6]
    P = [1/6] * 6
    print(f"サイコロの期待値: {calc.expectation(X, P):.2f}")
    print(f"サイコロの分散: {calc.variance(X, P):.2f}")
    
    # 連続データ
    np.random.seed(42)
    data = np.random.randn(1000, 3)  # 3次元データ
    cov_matrix = calc.covariance_matrix(data)
    print(f"\n共分散行列:\n{cov_matrix}")
    
    # NumPyの関数と比較
    cov_numpy = np.cov(data.T)
    print(f"\nNumPyの共分散行列:\n{cov_numpy}")
    print(f"差の最大値: {np.max(np.abs(cov_matrix - cov_numpy)):.10f}")
    

## 4\. 最尤推定とベイズ推定

### 4.1 最尤推定（Maximum Likelihood Estimation）

最尤推定は、観測データが得られる確率（尤度）を最大化するパラメータを求める方法です。

$$\hat{\theta}_{ML} = \arg\max_{\theta} P(D|\theta) = \arg\max_{\theta} \prod_{i=1}^{N} P(x_i|\theta)$$ 

対数尤度を使うと計算が簡単になります：

$$\hat{\theta}_{ML} = \arg\max_{\theta} \log P(D|\theta) = \arg\max_{\theta} \sum_{i=1}^{N} \log P(x_i|\theta)$$ 

**正規分布の最尤推定** 正規分布のパラメータμとσ²の最尤推定値は、標本平均と標本分散に一致します： $$\hat{\mu}_{ML} = \frac{1}{N}\sum_{i=1}^{N}x_i, \quad \hat{\sigma}^2_{ML} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{\mu})^2$$ 

### 4.2 ベイズ推定とMAP推定

ベイズ推定では、事前分布とデータの尤度から事後分布を計算します。

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)} \propto P(D|\theta)P(\theta)$$ 

MAP推定（Maximum A Posteriori）は事後確率を最大化します：

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|D) = \arg\max_{\theta} P(D|\theta)P(\theta)$$ 

### 実装例4：正規分布の最尤推定とMAP推定
    
    
    import numpy as np
    from scipy.stats import norm
    
    class GaussianEstimator:
        """正規分布のパラメータ推定"""
    
        @staticmethod
        def mle(data):
            """
            最尤推定（Maximum Likelihood Estimation）
    
            Parameters:
            -----------
            data : array-like
                観測データ
    
            Returns:
            --------
            tuple : (平均の推定値, 分散の推定値)
            """
            data = np.array(data)
            n = len(data)
    
            # MLE: 標本平均と標本分散
            mu_mle = np.mean(data)
            sigma2_mle = np.mean((data - mu_mle)**2)  # 1/n * Σ(x - μ)²
    
            return mu_mle, sigma2_mle
    
        @staticmethod
        def map_estimation(data, prior_mu=0, prior_sigma=1, prior_alpha=1, prior_beta=1):
            """
            MAP推定（Maximum A Posteriori）
    
            事前分布:
            - μ ~ N(prior_mu, prior_sigma²)
            - σ² ~ InverseGamma(prior_alpha, prior_beta)
    
            Parameters:
            -----------
            data : array-like
                観測データ
            prior_mu : float
                平均の事前分布の平均
            prior_sigma : float
                平均の事前分布の標準偏差
            prior_alpha, prior_beta : float
                分散の事前分布（逆ガンマ分布）のパラメータ
    
            Returns:
            --------
            tuple : (平均のMAP推定値, 分散のMAP推定値)
            """
            data = np.array(data)
            n = len(data)
            sample_mean = np.mean(data)
            sample_var = np.mean((data - sample_mean)**2)
    
            # 平均のMAP推定（事前分布が正規分布の場合）
            # 事後分布の平均は、事前分布と尤度の精度加重平均
            precision_prior = 1 / prior_sigma**2
            precision_likelihood = n / sample_var
    
            mu_map = (precision_prior * prior_mu + precision_likelihood * sample_mean) / \
                     (precision_prior + precision_likelihood)
    
            # 分散のMAP推定（事前分布が逆ガンマ分布の場合）
            # 簡易的に、事前分布の影響を考慮した推定
            alpha_post = prior_alpha + n / 2
            beta_post = prior_beta + 0.5 * np.sum((data - mu_map)**2)
    
            sigma2_map = beta_post / (alpha_post + 1)
    
            return mu_map, sigma2_map
    
        @staticmethod
        def compare_estimators(data, true_mu=0, true_sigma=1):
            """MLE推定とMAP推定の比較"""
    
            mu_mle, sigma2_mle = GaussianEstimator.mle(data)
            mu_map, sigma2_map = GaussianEstimator.map_estimation(
                data, prior_mu=true_mu, prior_sigma=true_sigma
            )
    
            print(f"真の値: μ={true_mu:.3f}, σ²={true_sigma**2:.3f}")
            print(f"\nMLE推定:")
            print(f"  μ̂_MLE = {mu_mle:.3f}, σ̂²_MLE = {sigma2_mle:.3f}")
            print(f"  誤差: |μ-μ̂|={abs(true_mu-mu_mle):.3f}, |σ²-σ̂²|={abs(true_sigma**2-sigma2_mle):.3f}")
    
            print(f"\nMAP推定:")
            print(f"  μ̂_MAP = {mu_map:.3f}, σ̂²_MAP = {sigma2_map:.3f}")
            print(f"  誤差: |μ-μ̂|={abs(true_mu-mu_map):.3f}, |σ²-σ̂²|={abs(true_sigma**2-sigma2_map):.3f}")
    
            return (mu_mle, sigma2_mle), (mu_map, sigma2_map)
    
    # 使用例
    np.random.seed(42)
    
    # サンプルサイズが小さい場合（MAP推定が有利）
    print("=" * 50)
    print("小サンプル（n=10）での比較")
    print("=" * 50)
    data_small = np.random.normal(0, 1, size=10)
    GaussianEstimator.compare_estimators(data_small)
    
    # サンプルサイズが大きい場合（MLEとMAPが近似）
    print("\n" + "=" * 50)
    print("大サンプル（n=1000）での比較")
    print("=" * 50)
    data_large = np.random.normal(0, 1, size=1000)
    GaussianEstimator.compare_estimators(data_large)
    

### 4.3 MLEとベイズ推定の比較

観点 | 最尤推定（MLE） | ベイズ推定  
---|---|---  
パラメータの扱い | 固定値（点推定） | 確率変数（分布推定）  
事前知識 | 使用しない | 事前分布で表現  
データが少ない場合 | 過学習しやすい | 事前知識で補完  
計算コスト | 低い | 高い（積分が必要）  
不確実性の表現 | 点推定のみ | 事後分布全体  
  
## 5\. 実践応用

### 5.1 混合ガウスモデル（GMM）

混合ガウスモデルは、複数の正規分布の重み付き和で複雑な分布を表現するモデルです。

$$P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$ 

ここで、\\(\pi_k\\)は各ガウス成分の混合係数（\\(\sum_k \pi_k = 1\\)）です。

### 実装例5：GMMによるクラスタリング
    
    
    import numpy as np
    from scipy.stats import multivariate_normal
    
    class GaussianMixtureModel:
        """混合ガウスモデル（EM アルゴリズムによる学習）"""
    
        def __init__(self, n_components=2, max_iter=100, tol=1e-4):
            """
            Parameters:
            -----------
            n_components : int
                ガウス成分の数
            max_iter : int
                最大反復回数
            tol : float
                収束判定の閾値
            """
            self.n_components = n_components
            self.max_iter = max_iter
            self.tol = tol
    
        def initialize_parameters(self, X):
            """パラメータの初期化"""
            n_samples, n_features = X.shape
    
            # ランダムにデータ点を選んで初期平均とする
            random_idx = np.random.choice(n_samples, self.n_components, replace=False)
            self.means = X[random_idx]
    
            # 共分散行列を単位行列で初期化
            self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
    
            # 混合係数を均等に初期化
            self.weights = np.ones(self.n_components) / self.n_components
    
        def e_step(self, X):
            """
            Eステップ: 責任度（各データがどのガウス成分に属するか）を計算
    
            γ(z_nk) = π_k N(x_n|μ_k,Σ_k) / Σ_j π_j N(x_n|μ_j,Σ_j)
            """
            n_samples = X.shape[0]
            responsibilities = np.zeros((n_samples, self.n_components))
    
            for k in range(self.n_components):
                # 各成分の尤度を計算
                rv = multivariate_normal(self.means[k], self.covariances[k])
                responsibilities[:, k] = self.weights[k] * rv.pdf(X)
    
            # 正規化して責任度を計算
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    
            return responsibilities
    
        def m_step(self, X, responsibilities):
            """
            Mステップ: 責任度を使ってパラメータを更新
            """
            n_samples, n_features = X.shape
    
            # 各成分の有効サンプル数
            N_k = responsibilities.sum(axis=0)
    
            # パラメータの更新
            for k in range(self.n_components):
                # 混合係数の更新
                self.weights[k] = N_k[k] / n_samples
    
                # 平均の更新
                self.means[k] = (responsibilities[:, k].reshape(-1, 1) * X).sum(axis=0) / N_k[k]
    
                # 共分散行列の更新
                diff = X - self.means[k]
                self.covariances[k] = (responsibilities[:, k].reshape(-1, 1, 1) *
                                      (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :])).sum(axis=0) / N_k[k]
    
                # 数値安定性のために小さな値を加える
                self.covariances[k] += np.eye(n_features) * 1e-6
    
        def compute_log_likelihood(self, X):
            """対数尤度の計算"""
            n_samples = X.shape[0]
            log_likelihood = 0
    
            for i in range(n_samples):
                likelihood = 0
                for k in range(self.n_components):
                    rv = multivariate_normal(self.means[k], self.covariances[k])
                    likelihood += self.weights[k] * rv.pdf(X[i])
                log_likelihood += np.log(likelihood)
    
            return log_likelihood
    
        def fit(self, X):
            """EMアルゴリズムでパラメータを学習"""
            self.initialize_parameters(X)
    
            prev_log_likelihood = -np.inf
    
            for iteration in range(self.max_iter):
                # Eステップ
                responsibilities = self.e_step(X)
    
                # Mステップ
                self.m_step(X, responsibilities)
    
                # 対数尤度の計算
                log_likelihood = self.compute_log_likelihood(X)
    
                # 収束判定
                if abs(log_likelihood - prev_log_likelihood) < self.tol:
                    print(f"収束しました（{iteration+1}回の反復）")
                    break
    
                prev_log_likelihood = log_likelihood
    
                if (iteration + 1) % 10 == 0:
                    print(f"反復 {iteration+1}: 対数尤度 = {log_likelihood:.4f}")
    
            return self
    
        def predict(self, X):
            """最も責任度の高いクラスタを予測"""
            responsibilities = self.e_step(X)
            return np.argmax(responsibilities, axis=1)
    
    # 使用例
    np.random.seed(42)
    
    # 2つのガウス分布から生成されたデータ
    data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 150)
    data2 = np.random.multivariate_normal([4, 4], [[1, 0.5], [0.5, 1]], 150)
    X = np.vstack([data1, data2])
    
    # GMMの学習
    gmm = GaussianMixtureModel(n_components=2, max_iter=50)
    gmm.fit(X)
    
    # クラスタリング結果
    labels = gmm.predict(X)
    print(f"\n学習されたパラメータ:")
    for k in range(gmm.n_components):
        print(f"成分 {k+1}: 重み={gmm.weights[k]:.3f}, 平均={gmm.means[k]}")
    

### 5.2 ベイズ線形回帰

ベイズ線形回帰では、パラメータに確率分布を考え、予測の不確実性も推定できます。

$$P(\mathbf{w}|D) = \frac{P(D|\mathbf{w})P(\mathbf{w})}{P(D)}$$ 

予測分布は事後分布を周辺化して得られます：

$$P(y^*|x^*, D) = \int P(y^*|x^*, \mathbf{w})P(\mathbf{w}|D)d\mathbf{w}$$ 

### 実装例6：ベイズ線形回帰
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class BayesianLinearRegression:
        """ベイズ線形回帰の実装"""
    
        def __init__(self, alpha=1.0, beta=1.0):
            """
            Parameters:
            -----------
            alpha : float
                重みの事前分布の精度（λ = α * I）
            beta : float
                観測ノイズの精度（1/σ²）
            """
            self.alpha = alpha  # 重みの事前精度
            self.beta = beta    # ノイズの精度
    
        def fit(self, X, y):
            """
            訓練データから事後分布を計算
    
            事後分布: P(w|D) = N(w|m_N, S_N)
            S_N = (α*I + β*X^T*X)^(-1)
            m_N = β*S_N*X^T*y
            """
            X = np.array(X)
            y = np.array(y).reshape(-1, 1)
    
            n_samples, n_features = X.shape
    
            # 事前分布の精度行列
            prior_precision = self.alpha * np.eye(n_features)
    
            # 事後分布の共分散行列（精度行列の逆行列）
            posterior_precision = prior_precision + self.beta * (X.T @ X)
            self.posterior_cov = np.linalg.inv(posterior_precision)
    
            # 事後分布の平均
            self.posterior_mean = self.beta * (self.posterior_cov @ X.T @ y)
    
            return self
    
        def predict(self, X_test, return_std=False):
            """
            予測分布を計算
    
            予測分布: P(y*|x*, D) = N(y*|m_N^T*x*, σ_N²(x*))
            σ_N²(x*) = 1/β + x*^T*S_N*x*
            """
            X_test = np.array(X_test)
    
            # 予測平均
            y_pred = X_test @ self.posterior_mean
    
            if return_std:
                # 予測分散（データノイズ + パラメータの不確実性）
                y_var = 1/self.beta + np.sum(X_test @ self.posterior_cov * X_test, axis=1, keepdims=True)
                y_std = np.sqrt(y_var)
                return y_pred.flatten(), y_std.flatten()
            else:
                return y_pred.flatten()
    
        def sample_weights(self, n_samples=10):
            """事後分布からパラメータをサンプリング"""
            return np.random.multivariate_normal(
                self.posterior_mean.flatten(),
                self.posterior_cov,
                size=n_samples
            )
    
    # 使用例とベイズ推定の可視化
    np.random.seed(42)
    
    # 真のパラメータ: y = 2x + 1 + ノイズ
    def true_function(x):
        return 2 * x + 1
    
    # 訓練データ生成
    X_train = np.linspace(0, 1, 10).reshape(-1, 1)
    X_train = np.hstack([np.ones_like(X_train), X_train])  # バイアス項を追加
    y_train = true_function(X_train[:, 1]) + np.random.randn(10) * 0.3
    
    # ベイズ線形回帰の学習
    bayesian_lr = BayesianLinearRegression(alpha=2.0, beta=10.0)
    bayesian_lr.fit(X_train, y_train)
    
    # テストデータ
    X_test = np.linspace(-0.2, 1.2, 100).reshape(-1, 1)
    X_test_with_bias = np.hstack([np.ones_like(X_test), X_test])
    
    # 予測（平均と標準偏差）
    y_pred, y_std = bayesian_lr.predict(X_test_with_bias, return_std=True)
    
    # 事後分布からパラメータをサンプリング
    weight_samples = bayesian_lr.sample_weights(n_samples=20)
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    # 左図: 予測分布と信頼区間
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 1], y_train, c='red', s=50, label='訓練データ', zorder=3)
    plt.plot(X_test, true_function(X_test), 'g--', linewidth=2, label='真の関数', zorder=2)
    plt.plot(X_test, y_pred, 'b-', linewidth=2, label='予測平均', zorder=2)
    plt.fill_between(X_test.flatten(), y_pred - 2*y_std, y_pred + 2*y_std,
                     alpha=0.3, label='95%信頼区間', zorder=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ベイズ線形回帰: 予測分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右図: 事後分布からサンプリングした関数
    plt.subplot(1, 2, 2)
    plt.scatter(X_train[:, 1], y_train, c='red', s=50, label='訓練データ', zorder=3)
    for i, w in enumerate(weight_samples):
        y_sample = X_test_with_bias @ w
        plt.plot(X_test, y_sample, 'b-', alpha=0.3, linewidth=1,
                 label='サンプル' if i == 0 else '')
    plt.plot(X_test, true_function(X_test), 'g--', linewidth=2, label='真の関数')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('事後分布からサンプリングした関数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_regression.png', dpi=150, bbox_inches='tight')
    print("ベイズ線形回帰の可視化を保存しました")
    
    # パラメータの事後分布
    print(f"\nパラメータの事後分布:")
    print(f"平均: {bayesian_lr.posterior_mean.flatten()}")
    print(f"標準偏差: {np.sqrt(np.diag(bayesian_lr.posterior_cov))}")
    

## まとめ

この章では、機械学習の基盤となる確率統計の基礎を学びました。

**学習した内容**

  * **ベイズの定理** : 事前知識とデータから事後確率を計算する基本原理
  * **確率分布** : 正規分布と多変量正規分布の性質と実装
  * **統計量** : 期待値、分散、共分散の計算と解釈
  * **パラメータ推定** : 最尤推定とベイズ推定の違いと適用
  * **実践応用** : ナイーブベイズ、GMM、ベイズ線形回帰

**次章への準備** 第2章では、線形代数の基礎を学びます。特に、行列分解（固有値分解、SVD）とPCAは、この章で学んだ多変量正規分布の共分散行列と深く関連します。 

### 演習問題

  1. ベイズの定理を使って、スパムメール検出器の精度を計算してみましょう
  2. 2次元正規分布で、相関係数が0.9の場合と-0.9の場合のデータを生成し、可視化してください
  3. 小サンプル（n=5）と大サンプル（n=1000）で、最尤推定とMAP推定の違いを比較してください
  4. GMMを3成分に拡張し、3つのクラスタを持つデータで動作を確認してください
  5. ベイズ線形回帰で、事前分布の精度パラメータαを変えると予測がどう変わるか実験してください

### 参考文献

  * C.M. Bishop, "Pattern Recognition and Machine Learning" (2006)
  * Kevin P. Murphy, "Machine Learning: A Probabilistic Perspective" (2012)
  * 杉山将, "統計的機械学習の数理100問 with Python" (2020)

[← シリーズ目次に戻る](<./index.html>) [第2章：線形代数の基礎 →](<./chapter2-linear-algebra.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
