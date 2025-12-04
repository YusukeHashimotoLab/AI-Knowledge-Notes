---
title: 第4章：情報理論
chapter_title: 第4章：情報理論
---

**機械学習を支える情報理論を、エントロピーからVAEまで理論と実装で深く理解する**

**この章で学べること**

  * エントロピーと情報量の数学的定義と直感的理解
  * KLダイバージェンスと交差エントロピーの関係
  * 相互情報量による特徴選択の理論的基礎
  * VAEと情報ボトルネックの情報理論的解釈
  * 機械学習における損失関数の情報理論的意味

## 1\. エントロピー

### 1.1 情報量とシャノンエントロピー

情報理論の基礎は、「ある事象の情報量」を定量化することから始まります。事象xが起こる確率をP(x)としたとき、その**自己情報量** は次のように定義されます。

$$I(x) = -\log_2 P(x) \quad \text{[bits]}$$ 

この定義には深い意味があります：

  * **確率が低い事象ほど情報量が大きい** : 珍しい事象ほど驚きが大きい
  * **確実な事象（P(x)=1）は情報量ゼロ** : 予測できることに驚きはない
  * **独立事象の情報量は加算的** : I(x,y) = I(x) + I(y)

**シャノンエントロピー** は、確率分布全体の平均的な情報量を表します。

$$H(X) = -\sum_{x} P(x) \log_2 P(x) = \mathbb{E}_{x \sim P}[-\log P(x)]$$ 

**エントロピーの直感的理解** エントロピーは「不確実性」や「ランダムさ」の尺度です。一様分布は最大エントロピーを持ち（最も予測困難）、確定的な分布は最小エントロピー（エントロピー=0）を持ちます。 

### 1.2 条件付きエントロピー

変数Yが与えられた条件の下でのXのエントロピーを**条件付きエントロピー** と呼びます。

$$H(X|Y) = \sum_{y} P(y) H(X|Y=y) = -\sum_{x,y} P(x,y) \log P(x|y)$$ 

重要な性質として、**連鎖律（chain rule）** が成り立ちます：

$$H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$ 

### 実装例1：エントロピーの計算と可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class InformationMeasures:
        """情報理論の基本量を計算するクラス"""
    
        @staticmethod
        def entropy(p, base=2):
            """
            シャノンエントロピーの計算
    
            Parameters:
            -----------
            p : array-like
                確率分布（合計が1になる必要がある）
            base : int
                対数の底（2: bits, e: nats）
    
            Returns:
            --------
            float : エントロピー
            """
            p = np.array(p)
            # 0 * log(0) = 0 と定義（数値的に安定な実装）
            p = p[p > 0]  # 正の確率のみを考慮
    
            if base == 2:
                return -np.sum(p * np.log2(p))
            elif base == np.e:
                return -np.sum(p * np.log(p))
            else:
                return -np.sum(p * np.log(p)) / np.log(base)
    
        @staticmethod
        def conditional_entropy(joint_p, axis=1):
            """
            条件付きエントロピー H(X|Y) の計算
    
            Parameters:
            -----------
            joint_p : ndarray
                同時確率分布 P(X,Y)
            axis : int
                条件とする変数の軸（0: H(Y|X), 1: H(X|Y)）
    
            Returns:
            --------
            float : 条件付きエントロピー
            """
            joint_p = np.array(joint_p)
    
            # 周辺確率の計算
            marginal_p = np.sum(joint_p, axis=axis)
    
            # 条件付きエントロピーの計算
            h_cond = 0
            for i, p_y in enumerate(marginal_p):
                if p_y > 0:
                    if axis == 1:
                        conditional_p = joint_p[:, i] / p_y
                    else:
                        conditional_p = joint_p[i, :] / p_y
                    h_cond += p_y * InformationMeasures.entropy(conditional_p)
    
            return h_cond
    
        @staticmethod
        def joint_entropy(joint_p):
            """
            同時エントロピー H(X,Y) の計算
            """
            joint_p = np.array(joint_p).flatten()
            return InformationMeasures.entropy(joint_p)
    
    # 使用例1: 二値変数のエントロピー
    print("=" * 50)
    print("二値変数のエントロピー")
    print("=" * 50)
    
    # コインの確率分布
    probs = np.linspace(0.01, 0.99, 99)
    entropies = [InformationMeasures.entropy([p, 1-p]) for p in probs]
    
    plt.figure(figsize=(10, 5))
    plt.plot(probs, entropies, 'b-', linewidth=2)
    plt.axvline(x=0.5, color='r', linestyle='--', label='最大エントロピー (p=0.5)')
    plt.xlabel('確率 P(X=1)')
    plt.ylabel('エントロピー H(X) [bits]')
    plt.title('二値確率変数のエントロピー')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('entropy_binary.png', dpi=150, bbox_inches='tight')
    print(f"最大エントロピー: {max(entropies):.4f} bits (p=0.5)")
    
    # 使用例2: 同時確率分布と条件付きエントロピー
    print("\n" + "=" * 50)
    print("条件付きエントロピーの例")
    print("=" * 50)
    
    # 同時確率分布 P(X,Y)
    joint_prob = np.array([
        [0.2, 0.1],  # P(X=0, Y=0), P(X=0, Y=1)
        [0.15, 0.55] # P(X=1, Y=0), P(X=1, Y=1)
    ])
    
    # 周辺確率
    p_x = np.sum(joint_prob, axis=1)
    p_y = np.sum(joint_prob, axis=0)
    
    # 各種エントロピー
    h_x = InformationMeasures.entropy(p_x)
    h_y = InformationMeasures.entropy(p_y)
    h_xy = InformationMeasures.joint_entropy(joint_prob)
    h_x_given_y = InformationMeasures.conditional_entropy(joint_prob, axis=1)
    h_y_given_x = InformationMeasures.conditional_entropy(joint_prob, axis=0)
    
    print(f"H(X) = {h_x:.4f} bits")
    print(f"H(Y) = {h_y:.4f} bits")
    print(f"H(X,Y) = {h_xy:.4f} bits")
    print(f"H(X|Y) = {h_x_given_y:.4f} bits")
    print(f"H(Y|X) = {h_y_given_x:.4f} bits")
    
    # 連鎖律の検証: H(X,Y) = H(X) + H(Y|X)
    print(f"\n連鎖律の検証:")
    print(f"H(X) + H(Y|X) = {h_x + h_y_given_x:.4f}")
    print(f"H(X,Y) = {h_xy:.4f}")
    print(f"差: {abs(h_xy - (h_x + h_y_given_x)):.10f}")
    

## 2\. KLダイバージェンスと交差エントロピー

### 2.1 KLダイバージェンス（Kullback-Leibler Divergence）

KLダイバージェンスは、2つの確率分布P(x)とQ(x)の「差異」を測る指標です。

$$D_{KL}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$ 

**重要な性質：**

  * **非対称性** : \\(D_{KL}(P||Q) \neq D_{KL}(Q||P)\\)（距離ではない）
  * **非負性** : \\(D_{KL}(P||Q) \geq 0\\)、等号成立は \\(P = Q\\) のとき
  * **ギブスの不等式** : \\(\mathbb{E}_P[\log P(x)] \geq \mathbb{E}_P[\log Q(x)]\\)

### 2.2 交差エントロピー（Cross Entropy）

交差エントロピーは、真の分布Pの下でモデル分布Qを使って符号化する際の平均ビット数です。

$$H(P, Q) = -\sum_{x} P(x) \log Q(x) = H(P) + D_{KL}(P||Q)$$ 

この関係式から、**交差エントロピーを最小化することはKLダイバージェンスを最小化することと等価** であることがわかります（H(P)は定数のため）。

**機械学習での応用** 分類問題の損失関数として交差エントロピーが広く使われます。真のラベル分布P（one-hot）とモデルの予測分布Q（ソフトマックス出力）の差を最小化することで学習が進みます。 

### 実装例2：KLダイバージェンスと交差エントロピー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import softmax
    
    class DivergenceMeasures:
        """ダイバージェンス指標の計算"""
    
        @staticmethod
        def kl_divergence(p, q, epsilon=1e-10):
            """
            KLダイバージェンス D_KL(P||Q) の計算
    
            Parameters:
            -----------
            p, q : array-like
                確率分布（正規化されている必要がある）
            epsilon : float
                数値安定性のための小さな値
    
            Returns:
            --------
            float : KLダイバージェンス [nats または bits]
            """
            p = np.array(p)
            q = np.array(q)
    
            # ゼロ除算を防ぐ
            q = np.clip(q, epsilon, 1.0)
            p = np.clip(p, epsilon, 1.0)
    
            return np.sum(p * np.log(p / q))
    
        @staticmethod
        def cross_entropy(p, q, epsilon=1e-10):
            """
            交差エントロピー H(P,Q) の計算
    
            Returns:
            --------
            float : 交差エントロピー
            """
            p = np.array(p)
            q = np.array(q)
    
            q = np.clip(q, epsilon, 1.0)
    
            return -np.sum(p * np.log(q))
    
        @staticmethod
        def js_divergence(p, q):
            """
            Jensen-Shannon ダイバージェンス（対称版のKL）
    
            JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
            where M = 0.5 * (P + Q)
            """
            p = np.array(p)
            q = np.array(q)
            m = 0.5 * (p + q)
    
            return 0.5 * DivergenceMeasures.kl_divergence(p, m) + \
                   0.5 * DivergenceMeasures.kl_divergence(q, m)
    
    # 使用例1: ガウス分布のKLダイバージェンス
    print("=" * 50)
    print("ガウス分布のKLダイバージェンス")
    print("=" * 50)
    
    # 2つの正規分布
    x = np.linspace(-5, 5, 1000)
    p = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)  # N(0, 1)
    q = np.exp(-0.5 * (x - 1)**2) / np.sqrt(2 * np.pi)  # N(1, 1)
    
    # 正規化
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    kl_pq = DivergenceMeasures.kl_divergence(p, q)
    kl_qp = DivergenceMeasures.kl_divergence(q, p)
    js = DivergenceMeasures.js_divergence(p, q)
    
    print(f"D_KL(P||Q) = {kl_pq:.4f}")
    print(f"D_KL(Q||P) = {kl_qp:.4f}")
    print(f"JS(P||Q) = {js:.4f}")
    print(f"非対称性: |D_KL(P||Q) - D_KL(Q||P)| = {abs(kl_pq - kl_qp):.4f}")
    
    # 可視化
    plt.figure(figsize=(10, 5))
    plt.plot(x, p * 1000, 'b-', linewidth=2, label='P: N(0,1)')
    plt.plot(x, q * 1000, 'r-', linewidth=2, label='Q: N(1,1)')
    plt.xlabel('x')
    plt.ylabel('確率密度')
    plt.title(f'KLダイバージェンス: D_KL(P||Q) = {kl_pq:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kl_divergence.png', dpi=150, bbox_inches='tight')
    
    # 使用例2: 分類問題での交差エントロピー損失
    print("\n" + "=" * 50)
    print("分類問題での交差エントロピー損失")
    print("=" * 50)
    
    # 真のラベル（one-hot encoding）
    true_label = np.array([0, 1, 0, 0])  # クラス1が正解
    
    # モデルの予測（ロジット）
    logits_good = np.array([1.0, 3.5, 0.5, 0.8])   # 良い予測
    logits_bad = np.array([2.0, 0.5, 1.5, 1.0])    # 悪い予測
    
    # ソフトマックスで確率に変換
    pred_good = softmax(logits_good)
    pred_bad = softmax(logits_bad)
    
    # 交差エントロピー損失
    ce_good = DivergenceMeasures.cross_entropy(true_label, pred_good)
    ce_bad = DivergenceMeasures.cross_entropy(true_label, pred_bad)
    
    print(f"良い予測の確率分布: {pred_good}")
    print(f"交差エントロピー損失: {ce_good:.4f}\n")
    
    print(f"悪い予測の確率分布: {pred_bad}")
    print(f"交差エントロピー損失: {ce_bad:.4f}\n")
    
    print(f"損失の差: {ce_bad - ce_good:.4f}")
    print("→ 良い予測の方が損失が小さい")
    

## 3\. 相互情報量

### 3.1 相互情報量の定義

**相互情報量（Mutual Information）** は、2つの確率変数XとYの間の統計的依存性を測る指標です。

$$I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)} = D_{KL}(P(X,Y)||P(X)P(Y))$$ 

相互情報量は、エントロピーを使って次のようにも表せます：

$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$ 

**重要な性質：**

  * **対称性** : \\(I(X;Y) = I(Y;X)\\)
  * **非負性** : \\(I(X;Y) \geq 0\\)、等号成立はXとYが独立のとき
  * **情報の減少** : \\(I(X;Y) \leq \min(H(X), H(Y))\\)

**直感的理解** 相互情報量は「Xを観測することで、Yについてどれだけの情報が得られるか」を表します。I(X;Y)が大きいほど、XとYは強く依存しています。 

### 3.2 特徴選択への応用

機械学習では、相互情報量を使って重要な特徴を選択できます。目標変数Yと各特徴X_iの相互情報量I(X_i;Y)を計算し、値が大きい特徴を選択します。

### 実装例3：相互情報量による特徴選択
    
    
    import numpy as np
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.datasets import make_classification
    
    class MutualInformation:
        """相互情報量の計算と特徴選択"""
    
        @staticmethod
        def mutual_information_discrete(x, y):
            """
            離散変数の相互情報量を計算
    
            I(X;Y) = H(X) + H(Y) - H(X,Y)
    
            Parameters:
            -----------
            x, y : array-like
                離散値の確率変数
    
            Returns:
            --------
            float : 相互情報量
            """
            x = np.array(x)
            y = np.array(y)
    
            # 同時頻度行列を作成
            xy = np.c_[x, y]
            unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
            joint_prob = counts_xy / len(x)
    
            # 周辺確率
            unique_x, counts_x = np.unique(x, return_counts=True)
            p_x = counts_x / len(x)
    
            unique_y, counts_y = np.unique(y, return_counts=True)
            p_y = counts_y / len(y)
    
            # エントロピーの計算
            from scipy.stats import entropy
            h_x = entropy(p_x, base=2)
            h_y = entropy(p_y, base=2)
            h_xy = entropy(joint_prob, base=2)
    
            # 相互情報量
            mi = h_x + h_y - h_xy
    
            return mi
    
        @staticmethod
        def feature_selection_by_mi(X, y, n_features=5):
            """
            相互情報量による特徴選択
    
            Parameters:
            -----------
            X : ndarray of shape (n_samples, n_features)
                特徴量行列
            y : array-like
                目標変数
            n_features : int
                選択する特徴数
    
            Returns:
            --------
            selected_indices : array
                選択された特徴のインデックス
            mi_scores : array
                各特徴の相互情報量スコア
            """
            # scikit-learnの相互情報量計算
            mi_scores = mutual_info_classif(X, y, random_state=42)
    
            # スコアの高い順にソート
            selected_indices = np.argsort(mi_scores)[::-1][:n_features]
    
            return selected_indices, mi_scores
    
    # 使用例1: 離散変数の相互情報量
    print("=" * 50)
    print("離散変数の相互情報量")
    print("=" * 50)
    
    # 例: 天気（X）と傘の使用（Y）
    # 0: 晴れ/使わない, 1: 雨/使う
    np.random.seed(42)
    
    # 強い相関がある場合
    weather = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 10)
    umbrella_corr = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1] * 10)  # 天気と相関あり
    umbrella_rand = np.random.randint(0, 2, 100)  # ランダム
    
    mi_corr = MutualInformation.mutual_information_discrete(weather, umbrella_corr)
    mi_rand = MutualInformation.mutual_information_discrete(weather, umbrella_rand)
    
    print(f"天気と傘（相関あり）の相互情報量: {mi_corr:.4f} bits")
    print(f"天気と傘（ランダム）の相互情報量: {mi_rand:.4f} bits")
    print(f"→ 相関がある場合の方が相互情報量が大きい")
    
    # 使用例2: 特徴選択
    print("\n" + "=" * 50)
    print("相互情報量による特徴選択")
    print("=" * 50)
    
    # 合成データの生成（20特徴、うち5個が有用）
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,  # 有用な特徴
        n_redundant=5,    # 冗長な特徴
        n_repeated=0,
        n_classes=2,
        random_state=42
    )
    
    # 相互情報量で特徴選択
    selected_idx, mi_scores = MutualInformation.feature_selection_by_mi(X, y, n_features=5)
    
    print(f"全特徴数: {X.shape[1]}")
    print(f"\n相互情報量スコア（上位10特徴）:")
    for i in range(10):
        print(f"  特徴 {i}: MI = {mi_scores[i]:.4f}")
    
    print(f"\n選択された特徴（上位5個）: {selected_idx}")
    print(f"選択された特徴のMIスコア: {mi_scores[selected_idx]}")
    
    # 可視化
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(mi_scores)), mi_scores, color='steelblue')
    plt.bar(selected_idx, mi_scores[selected_idx], color='crimson', label='選択された特徴')
    plt.xlabel('特徴インデックス')
    plt.ylabel('相互情報量スコア')
    plt.title('各特徴の相互情報量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sorted_idx = np.argsort(mi_scores)[::-1]
    plt.plot(range(1, len(mi_scores)+1), mi_scores[sorted_idx], 'o-', linewidth=2)
    plt.axvline(x=5, color='r', linestyle='--', label='選択数=5')
    plt.xlabel('ランク')
    plt.ylabel('相互情報量スコア')
    plt.title('相互情報量スコアの順位')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mutual_information.png', dpi=150, bbox_inches='tight')
    print("\n相互情報量の可視化を保存しました")
    

## 4\. 情報理論と機械学習

### 4.1 変分オートエンコーダ（VAE）

VAEは情報理論の観点から理解できます。潜在変数zとデータxの関係を学習する際、次の目的関数を最大化します。

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x)||p(z))$$ 

この右辺は**ELBO（Evidence Lower BOund）** と呼ばれ：

  * **第1項（再構成項）** : データの再構成品質
  * **第2項（KL項）** : 潜在分布と事前分布の近さ

### 4.2 情報ボトルネック理論

情報ボトルネック理論は、表現学習を情報理論的に定式化します。入力XとラベルYに対し、表現Zは次を満たすべきです：

$$\min_{Z} I(X;Z) - \beta I(Z;Y)$$ 

これは「入力情報を圧縮しつつ（I(X;Z)を最小化）、ラベル情報を保持する（I(Z;Y)を最大化）」というトレードオフを表現しています。

### 実装例4：VAEのELBO計算
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    
    class VAE(nn.Module):
        """変分オートエンコーダ（Variational Autoencoder）"""
    
        def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
            """
            Parameters:
            -----------
            input_dim : int
                入力次元（例: 28x28=784 for MNIST）
            hidden_dim : int
                隠れ層の次元
            latent_dim : int
                潜在変数の次元
            """
            super(VAE, self).__init__()
    
            # エンコーダ q(z|x)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
            # デコーダ p(x|z)
            self.fc3 = nn.Linear(latent_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, input_dim)
    
        def encode(self, x):
            """
            エンコーダ: x → (μ, log σ²)
    
            Returns:
            --------
            mu, logvar : 潜在分布のパラメータ
            """
            h = F.relu(self.fc1(x))
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
    
        def reparameterize(self, mu, logvar):
            """
            再パラメータ化トリック: z = μ + σ * ε, ε ~ N(0,1)
    
            これにより、確率的なサンプリングを微分可能にする
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z
    
        def decode(self, z):
            """
            デコーダ: z → x̂
            """
            h = F.relu(self.fc3(z))
            x_recon = torch.sigmoid(self.fc4(h))
            return x_recon
    
        def forward(self, x):
            """
            順伝播: x → z → x̂
            """
            mu, logvar = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar
    
    def vae_loss(x, x_recon, mu, logvar, beta=1.0):
        """
        VAEの損失関数（ELBO の負値）
    
        ELBO = E[log p(x|z)] - β * D_KL(q(z|x)||p(z))
    
        Parameters:
        -----------
        x : Tensor
            元のデータ
        x_recon : Tensor
            再構成されたデータ
        mu, logvar : Tensor
            潜在分布のパラメータ
        beta : float
            KL項の重み（β-VAE）
    
        Returns:
        --------
        loss, recon_loss, kl_loss : 総損失、再構成損失、KL損失
        """
        # 再構成損失（負の対数尤度）
        # バイナリデータの場合: BCE loss
        recon_loss = F.binary_cross_entropy(
            x_recon, x.view(-1, 784), reduction='sum'
        )
    
        # KLダイバージェンス損失
        # D_KL(N(μ,σ²)||N(0,1)) = 0.5 * Σ(μ² + σ² - log(σ²) - 1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        # 総損失（ELBOの負値）
        total_loss = recon_loss + beta * kl_loss
    
        return total_loss, recon_loss, kl_loss
    
    # 使用例
    print("=" * 50)
    print("VAEのELBO計算")
    print("=" * 50)
    
    # モデルの初期化
    vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
    
    # ダミーデータ（バッチサイズ32の28x28画像）
    torch.manual_seed(42)
    x = torch.rand(32, 1, 28, 28)
    
    # 順伝播
    x_recon, mu, logvar = vae(x)
    
    # 損失計算
    loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar, beta=1.0)
    
    print(f"総損失（-ELBO）: {loss.item():.2f}")
    print(f"  再構成損失: {recon_loss.item():.2f}")
    print(f"  KL損失: {kl_loss.item():.2f}")
    
    # βの影響を確認
    print("\n" + "=" * 50)
    print("β-VAE: βパラメータの影響")
    print("=" * 50)
    
    betas = [0.5, 1.0, 2.0, 5.0]
    for beta in betas:
        loss, recon, kl = vae_loss(x, x_recon, mu, logvar, beta=beta)
        print(f"β={beta:.1f}: 総損失={loss.item():.2f}, "
              f"再構成={recon.item():.2f}, KL={kl.item():.2f}")
    
    print("\n解釈:")
    print("- βが大きい: KL項を重視 → 潜在空間が正規分布に近づく")
    print("- βが小さい: 再構成を重視 → 再構成品質が向上")
    

## 5\. 実践応用

### 5.1 交差エントロピー損失関数

分類問題で最も一般的な損失関数は、交差エントロピー損失です。

$$\mathcal{L}_{CE} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}$$ 

ここで、y_{i,c}は真のラベル（one-hot）、\\(\hat{y}_{i,c}\\)はモデルの予測確率です。

### 5.2 KL損失とラベルスムージング

ラベルスムージングは、過信を防ぐ正則化手法です。ハードラベル[0,1,0]を[ε/K, 1-ε+ε/K, ε/K]に変換します。

### 実装例5：交差エントロピーとKL損失
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    
    class LossFunctions:
        """情報理論に基づく損失関数"""
    
        @staticmethod
        def cross_entropy_loss(logits, targets):
            """
            交差エントロピー損失
    
            Parameters:
            -----------
            logits : Tensor of shape (batch_size, num_classes)
                モデルの出力（ソフトマックス前）
            targets : Tensor of shape (batch_size,)
                真のクラスラベル
    
            Returns:
            --------
            loss : Tensor
                交差エントロピー損失
            """
            return F.cross_entropy(logits, targets)
    
        @staticmethod
        def kl_div_loss(logits, target_dist):
            """
            KLダイバージェンス損失
    
            D_KL(target || pred) を計算
    
            Parameters:
            -----------
            logits : Tensor
                モデルの出力
            target_dist : Tensor
                目標分布（確率分布）
    
            Returns:
            --------
            loss : Tensor
                KL損失
            """
            log_pred = F.log_softmax(logits, dim=-1)
            return F.kl_div(log_pred, target_dist, reduction='batchmean')
    
        @staticmethod
        def label_smoothing_loss(logits, targets, smoothing=0.1):
            """
            ラベルスムージング付き交差エントロピー
    
            Parameters:
            -----------
            smoothing : float
                スムージングパラメータ（0: なし, 1: 完全に一様）
            """
            n_classes = logits.size(-1)
            log_pred = F.log_softmax(logits, dim=-1)
    
            # ラベルスムージング
            # 真のクラス: 1 - ε + ε/K
            # 他のクラス: ε/K
            with torch.no_grad():
                true_dist = torch.zeros_like(log_pred)
                true_dist.fill_(smoothing / (n_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    
            return torch.mean(torch.sum(-true_dist * log_pred, dim=-1))
    
    # 使用例1: 損失関数の比較
    print("=" * 50)
    print("損失関数の比較")
    print("=" * 50)
    
    torch.manual_seed(42)
    
    # データの準備
    batch_size = 4
    num_classes = 3
    
    logits = torch.randn(batch_size, num_classes) * 2
    targets = torch.tensor([0, 1, 2, 1])
    
    # 各損失の計算
    ce_loss = LossFunctions.cross_entropy_loss(logits, targets)
    
    # ラベルスムージング損失
    ls_loss_01 = LossFunctions.label_smoothing_loss(logits, targets, smoothing=0.1)
    ls_loss_03 = LossFunctions.label_smoothing_loss(logits, targets, smoothing=0.3)
    
    print(f"交差エントロピー損失: {ce_loss.item():.4f}")
    print(f"ラベルスムージング損失（ε=0.1）: {ls_loss_01.item():.4f}")
    print(f"ラベルスムージング損失（ε=0.3）: {ls_loss_03.item():.4f}")
    
    # 予測確率の表示
    probs = F.softmax(logits, dim=-1)
    print(f"\n予測確率:")
    for i in range(batch_size):
        print(f"  サンプル{i} (真のラベル={targets[i]}): {probs[i].numpy()}")
    
    # 使用例2: 信頼度と損失の関係
    print("\n" + "=" * 50)
    print("モデルの信頼度と損失の関係")
    print("=" * 50)
    
    # 信頼度を変化させる
    confidences = np.linspace(0.1, 0.99, 50)
    losses = []
    
    for conf in confidences:
        # 正しいクラスへの確率がconfのロジットを作成
        # [conf, (1-conf)/2, (1-conf)/2]
        logits_conf = torch.tensor([[
            np.log(conf),
            np.log((1-conf)/2),
            np.log((1-conf)/2)
        ]])
        target = torch.tensor([0])
    
        loss = LossFunctions.cross_entropy_loss(logits_conf, target)
        losses.append(loss.item())
    
    # 可視化
    plt.figure(figsize=(10, 5))
    plt.plot(confidences, losses, 'b-', linewidth=2)
    plt.xlabel('正解クラスへの予測確率')
    plt.ylabel('交差エントロピー損失')
    plt.title('予測信頼度と損失の関係')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('confidence_loss.png', dpi=150, bbox_inches='tight')
    print("信頼度と損失の関係を可視化しました")
    
    print(f"\n観察:")
    print(f"- 信頼度0.5の損失: {losses[20]:.4f}")
    print(f"- 信頼度0.9の損失: {losses[40]:.4f}")
    print(f"- 信頼度0.99の損失: {losses[-1]:.4f}")
    print("→ 信頼度が高いほど損失が小さくなる（指数的に減少）")
    

### 5.3 ELBO（Evidence Lower Bound）

生成モデルの学習で重要なELBOは、対数周辺尤度の下界を与えます。

$$\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x)||p(z))$$ 

### 実装例6：ELBOの詳細計算
    
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    class ELBOAnalysis:
        """ELBOの詳細分析"""
    
        @staticmethod
        def compute_elbo_components(x, x_recon, mu, logvar, n_samples=1000):
            """
            ELBOの各成分を詳細に計算
    
            Parameters:
            -----------
            x : Tensor
                元のデータ
            x_recon : Tensor
                再構成データ
            mu, logvar : Tensor
                エンコーダの出力（潜在分布のパラメータ）
            n_samples : int
                モンテカルロサンプル数
    
            Returns:
            --------
            dict : ELBOの各成分
            """
            batch_size = x.size(0)
            latent_dim = mu.size(1)
    
            # 1. 再構成項: E_q[log p(x|z)]
            # バイナリデータの場合の対数尤度
            recon_term = -F.binary_cross_entropy(
                x_recon, x.view(batch_size, -1), reduction='sum'
            ) / batch_size
    
            # 2. KL項（解析的計算）: D_KL(q(z|x)||p(z))
            # q(z|x) = N(μ, σ²), p(z) = N(0, I)
            kl_term = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            ) / batch_size
    
            # 3. ELBOの計算
            elbo = recon_term - kl_term
    
            # 4. モンテカルロ推定による検証
            # E_q[log p(x|z)] のサンプル推定
            std = torch.exp(0.5 * logvar)
            recon_mc = 0
            for _ in range(n_samples):
                eps = torch.randn_like(std)
                z = mu + eps * std
                # 簡略化した再構成尤度
                recon_mc += -F.binary_cross_entropy(
                    x_recon, x.view(batch_size, -1), reduction='sum'
                )
            recon_mc = recon_mc / (n_samples * batch_size)
    
            return {
                'elbo': elbo.item(),
                'reconstruction': recon_term.item(),
                'kl_divergence': kl_term.item(),
                'reconstruction_mc': recon_mc.item(),
                'log_marginal_lower_bound': elbo.item()
            }
    
        @staticmethod
        def analyze_latent_distribution(mu, logvar):
            """
            潜在分布の統計を分析
    
            Returns:
            --------
            dict : 統計量
            """
            std = torch.exp(0.5 * logvar)
    
            return {
                'mean_mu': mu.mean().item(),
                'std_mu': mu.std().item(),
                'mean_sigma': std.mean().item(),
                'std_sigma': std.std().item(),
                'min_sigma': std.min().item(),
                'max_sigma': std.max().item()
            }
    
    # 使用例
    print("=" * 50)
    print("ELBOの詳細分析")
    print("=" * 50)
    
    # ダミーVAEモデルの出力
    torch.manual_seed(42)
    batch_size = 16
    input_dim = 784
    latent_dim = 20
    
    x = torch.rand(batch_size, 1, 28, 28)
    mu = torch.randn(batch_size, latent_dim) * 0.5
    logvar = torch.randn(batch_size, latent_dim) * 0.5
    
    # 再パラメータ化
    std = torch.exp(0.5 * logvar)
    z = mu + torch.randn_like(std) * std
    x_recon = torch.sigmoid(torch.randn(batch_size, input_dim))
    
    # ELBOの計算
    elbo_components = ELBOAnalysis.compute_elbo_components(
        x, x_recon, mu, logvar, n_samples=100
    )
    
    print("ELBOの各成分:")
    for key, value in elbo_components.items():
        print(f"  {key}: {value:.4f}")
    
    # 潜在分布の分析
    latent_stats = ELBOAnalysis.analyze_latent_distribution(mu, logvar)
    
    print("\n潜在分布の統計:")
    for key, value in latent_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # 可視化: 潜在次元ごとのμとσ
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 平均μの分布
    axes[0].hist(mu.detach().numpy().flatten(), bins=30, alpha=0.7, color='blue')
    axes[0].set_xlabel('μ')
    axes[0].set_ylabel('頻度')
    axes[0].set_title('潜在変数の平均（μ）の分布')
    axes[0].axvline(x=0, color='r', linestyle='--', label='事前分布の平均')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 標準偏差σの分布
    axes[1].hist(std.detach().numpy().flatten(), bins=30, alpha=0.7, color='green')
    axes[1].set_xlabel('σ')
    axes[1].set_ylabel('頻度')
    axes[1].set_title('潜在変数の標準偏差（σ）の分布')
    axes[1].axvline(x=1, color='r', linestyle='--', label='事前分布の標準偏差')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elbo_analysis.png', dpi=150, bbox_inches='tight')
    print("\nELBO分析の可視化を保存しました")
    
    print("\n" + "=" * 50)
    print("情報理論的解釈")
    print("=" * 50)
    print("ELBO = 再構成項 - KL項")
    print("  再構成項: データを潜在変数で説明する能力")
    print("  KL項: 潜在分布が事前分布からどれだけ離れているか")
    print("  → トレードオフ: 良い再構成 vs 正則化された潜在空間")
    

## まとめ

この章では、機械学習を支える情報理論の基礎を学びました。

**学習した内容**

  * **エントロピー** : 不確実性の定量化と条件付きエントロピー
  * **KLダイバージェンス** : 確率分布の差異を測る非対称な指標
  * **交差エントロピー** : 分類問題の損失関数の理論的基礎
  * **相互情報量** : 変数間の依存性と特徴選択への応用
  * **VAEとELBO** : 生成モデルの情報理論的解釈

**次章への準備** 第5章では、機械学習の学習理論を学びます。この章で学んだKLダイバージェンスや相互情報量は、汎化誤差やモデルの複雑度を理解する上で重要な役割を果たします。 

### 演習問題

  1. サイコロ（6面）と偏ったコイン（P(表)=0.8）のエントロピーを計算し、比較してください
  2. 2つのガウス分布N(0,1)とN(2,2)のKLダイバージェンスを解析的に計算してください
  3. 相互情報量I(X;Y)=0となるのは、XとYがどのような関係のときか説明してください
  4. β-VAEでβ=0.5, 1.0, 2.0のときの潜在空間の違いを実験で確認してください
  5. ラベルスムージングがなぜモデルの過信を防ぐのか、情報理論の観点から説明してください

### 参考文献

  * Claude E. Shannon, "A Mathematical Theory of Communication" (1948)
  * Thomas M. Cover and Joy A. Thomas, "Elements of Information Theory" (2006)
  * D.P. Kingma and M. Welling, "Auto-Encoding Variational Bayes" (2013)
  * Naftali Tishby et al., "The Information Bottleneck Method" (2000)

[← 第3章：最適化理論](<./chapter3-optimization.html>) [第5章：学習理論 →](<./chapter5-learning-theory.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
