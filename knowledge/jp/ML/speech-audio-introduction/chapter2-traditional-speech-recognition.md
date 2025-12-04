---
title: 第2章：伝統的音声認識
chapter_title: 第2章：伝統的音声認識
subtitle: HMM-GMM時代の音声認識技術 - 統計的モデルによるアプローチ
reading_time: 35-40分
difficulty: 中級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 音声認識（ASR）タスクの定義と評価指標を理解する
  * ✅ Hidden Markov Model（HMM）の原理とアルゴリズムを実装できる
  * ✅ Gaussian Mixture Model（GMM）による音響モデリングを理解する
  * ✅ 言語モデル（N-gram）の構築と評価ができる
  * ✅ HMM-GMMベースのASRパイプライン全体を構築できる
  * ✅ WER（Word Error Rate）を用いてシステムを評価できる

* * *

## 2.1 音声認識の基礎

### ASRタスクの定義

**自動音声認識（Automatic Speech Recognition, ASR）** は、音声信号をテキストに変換するタスクです。

> 音声認識の目標：観測された音響信号 $X$ が与えられたとき、最も確からしい単語列 $W$ を見つけること

これはベイズの定理により以下のように定式化されます：

$$ \hat{W} = \arg\max_{W} P(W|X) = \arg\max_{W} \frac{P(X|W) P(W)}{P(X)} = \arg\max_{W} P(X|W) P(W) $$

  * $P(X|W)$: **音響モデル（Acoustic Model）** \- 単語列が与えられたときの音響信号の確率
  * $P(W)$: **言語モデル（Language Model）** \- 単語列の事前確率

### ASRシステムの構成要素
    
    
    ```mermaid
    graph LR
        A[音声信号] --> B[特徴抽出MFCC]
        B --> C[音響モデルHMM-GMM]
        C --> D[デコーディングViterbi]
        D --> E[言語モデルN-gram]
        E --> F[認識結果テキスト]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#e3f2fd
        style D fill:#f3e5f5
        style E fill:#e8f5e9
        style F fill:#c8e6c9
    ```

### 評価指標

#### Word Error Rate（WER）

**WER** は音声認識の標準評価指標です：

$$ \text{WER} = \frac{S + D + I}{N} \times 100\% $$

  * $S$: 置換エラー（Substitutions）
  * $D$: 削除エラー（Deletions）
  * $I$: 挿入エラー（Insertions）
  * $N$: 参照テキストの総単語数

#### Character Error Rate（CER）

日本語や中国語などの言語では、文字レベルのCERも使用されます：

$$ \text{CER} = \frac{S_c + D_c + I_c}{N_c} \times 100\% $$

### 実装：WER計算
    
    
    import numpy as np
    from typing import List, Tuple
    
    def levenshtein_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
        """
        レーベンシュタイン距離とエラー統計を計算
    
        Returns:
            (distance, substitutions, deletions, insertions)
        """
        m, n = len(ref), len(hyp)
    
        # DP テーブル
        dp = np.zeros((m + 1, n + 1), dtype=int)
    
        # 初期化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
    
        # DP
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j-1] + 1,  # 置換
                        dp[i-1][j] + 1,    # 削除
                        dp[i][j-1] + 1     # 挿入
                    )
    
        # バックトラック: エラー統計を計算
        i, j = m, n
        subs, dels, ins = 0, 0, 0
    
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                subs += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                dels += 1
                i -= 1
            else:
                ins += 1
                j -= 1
    
        return dp[m][n], subs, dels, ins
    
    
    def calculate_wer(reference: str, hypothesis: str) -> dict:
        """
        WER（Word Error Rate）を計算
    
        Args:
            reference: 正解テキスト
            hypothesis: 認識結果
    
        Returns:
            エラー統計を含む辞書
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
    
        dist, subs, dels, ins = levenshtein_distance(ref_words, hyp_words)
    
        n_words = len(ref_words)
        wer = (dist / n_words * 100) if n_words > 0 else 0
    
        return {
            'WER': wer,
            'substitutions': subs,
            'deletions': dels,
            'insertions': ins,
            'total_errors': dist,
            'total_words': n_words
        }
    
    
    # テスト
    reference = "the quick brown fox jumps over the lazy dog"
    hypothesis = "the quick brown fox jumped over a lazy dog"
    
    result = calculate_wer(reference, hypothesis)
    
    print("=== WER 計算例 ===")
    print(f"参照: {reference}")
    print(f"仮説: {hypothesis}")
    print(f"\nWER: {result['WER']:.2f}%")
    print(f"置換: {result['substitutions']}")
    print(f"削除: {result['deletions']}")
    print(f"挿入: {result['insertions']}")
    print(f"総エラー: {result['total_errors']}")
    print(f"総単語数: {result['total_words']}")
    

**出力** ：
    
    
    === WER 計算例 ===
    参照: the quick brown fox jumps over the lazy dog
    仮説: the quick brown fox jumped over a lazy dog
    
    WER: 22.22%
    置換: 2
    削除: 0
    挿入: 0
    総エラー: 2
    総単語数: 9
    

> **重要** : WERは100%を超える場合があります（挿入エラーが多い場合）。

* * *

## 2.2 Hidden Markov Models（HMM）

### HMMの基礎

**隠れマルコフモデル（HMM）** は、観測できない隠れ状態と観測可能な出力からなる確率モデルです。

#### HMMの構成要素

  * $N$: 状態数
  * $M$: 観測シンボル数
  * $A = \\{a_{ij}\\}$: 状態遷移確率行列 $a_{ij} = P(q_{t+1}=j | q_t=i)$
  * $B = \\{b_j(k)\\}$: 出力確率分布 $b_j(k) = P(o_t=k | q_t=j)$
  * $\pi = \\{\pi_i\\}$: 初期状態確率 $\pi_i = P(q_1=i)$

#### HMMの3つの基本問題

問題 | 説明 | アルゴリズム  
---|---|---  
**評価問題** | 観測列の確率を計算 | Forward-Backward  
**デコーディング** | 最も確からしい状態列を推定 | Viterbi  
**学習問題** | パラメータを推定 | Baum-Welch（EM）  
  
### Forward-Backwardアルゴリズム

**Forwardアルゴリズム** は観測列の確率 $P(O|\lambda)$ を計算します：

$$ \alpha_t(i) = P(o_1, o_2, \ldots, o_t, q_t=i | \lambda) $$

再帰式：

$$ \alpha_t(j) = \left[\sum_{i=1}^N \alpha_{t-1}(i) a_{ij}\right] b_j(o_t) $$

### Viterbiアルゴリズム

**Viterbiアルゴリズム** は最も確からしい状態列を見つけます：

$$ \delta_t(i) = \max_{q_1, \ldots, q_{t-1}} P(q_1, \ldots, q_{t-1}, q_t=i, o_1, \ldots, o_t | \lambda) $$

再帰式：

$$ \delta_t(j) = \left[\max_i \delta_{t-1}(i) a_{ij}\right] b_j(o_t) $$

### 実装：HMMの基本操作
    
    
    import numpy as np
    from hmmlearn import hmm
    import matplotlib.pyplot as plt
    
    # サンプル：天気モデル
    # 状態: 0=晴れ, 1=雨
    # 観測: 0=散歩, 1=買い物, 2=掃除
    
    # HMMパラメータの定義
    n_states = 2
    n_observations = 3
    
    # モデルの構築
    model = hmm.MultinomialHMM(n_components=n_states, random_state=42)
    
    # 状態遷移確率
    model.startprob_ = np.array([0.6, 0.4])  # 初期確率
    model.transmat_ = np.array([
        [0.7, 0.3],  # 晴れ → [晴れ, 雨]
        [0.4, 0.6]   # 雨 → [晴れ, 雨]
    ])
    
    # 出力確率
    model.emissionprob_ = np.array([
        [0.6, 0.3, 0.1],  # 晴れの日: [散歩, 買い物, 掃除]
        [0.1, 0.4, 0.5]   # 雨の日: [散歩, 買い物, 掃除]
    ])
    
    print("=== HMMパラメータ ===")
    print("\n初期状態確率:")
    print(f"  晴れ: {model.startprob_[0]:.2f}")
    print(f"  雨: {model.startprob_[1]:.2f}")
    
    print("\n状態遷移確率:")
    print(model.transmat_)
    
    print("\n出力確率:")
    print("       散歩   買い物  掃除")
    print(f"晴れ: {model.emissionprob_[0]}")
    print(f"雨:   {model.emissionprob_[1]}")
    
    # 観測列
    observations = np.array([[0], [1], [2], [1], [0]])  # 散歩, 買い物, 掃除, 買い物, 散歩
    
    # Forward アルゴリズム: 観測列の確率
    log_prob = model.score(observations)
    print(f"\n観測列の対数尤度: {log_prob:.4f}")
    print(f"観測列の確率: {np.exp(log_prob):.6f}")
    
    # Viterbi アルゴリズム: 最も確からしい状態列
    log_prob, states = model.decode(observations)
    state_names = ['晴れ', '雨']
    print(f"\n最も確からしい状態列:")
    for i, (obs, state) in enumerate(zip(observations.flatten(), states)):
        obs_names = ['散歩', '買い物', '掃除']
        print(f"  日{i+1}: 観測={obs_names[obs]}, 状態={state_names[state]}")
    

**出力** ：
    
    
    === HMMパラメータ ===
    
    初期状態確率:
      晴れ: 0.60
      雨: 0.40
    
    状態遷移確率:
    [[0.7 0.3]
     [0.4 0.6]]
    
    出力確率:
           散歩   買い物  掃除
    晴れ: [0.6 0.3 0.1]
    雨:   [0.1 0.4 0.5]
    
    観測列の対数尤度: -6.3218
    観測列の確率: 0.001802
    
    最も確からしい状態列:
      日1: 観測=散歩, 状態=晴れ
      日2: 観測=買い物, 状態=晴れ
      日3: 観測=掃除, 状態=雨
      日4: 観測=買い物, 状態=雨
      日5: 観測=散歩, 状態=雨
    

### HMMによる音素モデリング

音声認識では、各音素を3状態のLeft-to-Right HMMでモデル化します：
    
    
    ```mermaid
    graph LR
        Start((開始)) --> S1[状態1音素開始]
        S1 --> S1
        S1 --> S2[状態2音素中間]
        S2 --> S2
        S2 --> S3[状態3音素終了]
        S3 --> S3
        S3 --> End((終了))
    
        style Start fill:#c8e6c9
        style S1 fill:#e3f2fd
        style S2 fill:#fff3e0
        style S3 fill:#f3e5f5
        style End fill:#ffcdd2
    ```
    
    
    # Left-to-Right HMM（音素モデル）
    n_states = 3
    
    # Left-to-Right構造の遷移行列
    # 状態は前進または自己ループのみ
    lr_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", random_state=42)
    
    # 遷移確率（左から右のみ）
    lr_model.transmat_ = np.array([
        [0.5, 0.5, 0.0],  # 状態1: 自己ループまたは状態2へ
        [0.0, 0.5, 0.5],  # 状態2: 自己ループまたは状態3へ
        [0.0, 0.0, 1.0]   # 状態3: 自己ループのみ
    ])
    
    lr_model.startprob_ = np.array([1.0, 0.0, 0.0])  # 常に状態1から開始
    
    print("=== Left-to-Right HMM ===")
    print("遷移確率行列:")
    print(lr_model.transmat_)
    print("\n状態1から始まり、左から右へのみ進むことができます")
    

* * *

## 2.3 Gaussian Mixture Models（GMM）

### GMMの基礎

**ガウス混合モデル（GMM）** は複数のガウス分布の線形結合で確率分布を表現します：

$$ p(x) = \sum_{k=1}^K w_k \mathcal{N}(x | \mu_k, \Sigma_k) $$

  * $K$: 混合成分数
  * $w_k$: 混合重み（$\sum_k w_k = 1$）
  * $\mu_k$: 平均ベクトル
  * $\Sigma_k$: 共分散行列

### EMアルゴリズムによる学習

GMMのパラメータは**EM（Expectation-Maximization）アルゴリズム** で推定します：

#### Eステップ：責任度の計算

$$ \gamma_{nk} = \frac{w_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^K w_j \mathcal{N}(x_n | \mu_j, \Sigma_j)} $$

#### Mステップ：パラメータの更新

$$ \mu_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} x_n $$

$$ \Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} (x_n - \mu_k^{\text{new}})(x_n - \mu_k^{\text{new}})^T $$

$$ w_k^{\text{new}} = \frac{N_k}{N}, \quad N_k = \sum_{n=1}^N \gamma_{nk} $$

### 実装：GMMによるクラスタリング
    
    
    import numpy as np
    from sklearn.mixture import GaussianMixture
    import matplotlib.pyplot as plt
    
    # 3つのガウス分布から生成されたデータ
    np.random.seed(42)
    
    # データ生成
    n_samples = 300
    X1 = np.random.randn(n_samples // 3, 2) * 0.5 + np.array([0, 0])
    X2 = np.random.randn(n_samples // 3, 2) * 0.7 + np.array([3, 3])
    X3 = np.random.randn(n_samples // 3, 2) * 0.6 + np.array([0, 3])
    
    X = np.vstack([X1, X2, X3])
    
    # GMMによるクラスタリング
    n_components = 3
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    
    # 予測
    labels = gmm.predict(X)
    proba = gmm.predict_proba(X)
    
    print("=== GMM パラメータ ===")
    print(f"混合成分数: {n_components}")
    print(f"収束反復数: {gmm.n_iter_}")
    print(f"対数尤度: {gmm.score(X) * len(X):.2f}")
    
    print("\n混合重み:")
    for i, weight in enumerate(gmm.weights_):
        print(f"  成分{i+1}: {weight:.3f}")
    
    print("\n平均ベクトル:")
    for i, mean in enumerate(gmm.means_):
        print(f"  成分{i+1}: [{mean[0]:.2f}, {mean[1]:.2f}]")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # クラスタリング結果
    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='black')
    axes[0].scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=200, marker='X',
                    edgecolors='black', linewidths=2, label='中心')
    axes[0].set_xlabel('特徴量 1')
    axes[0].set_ylabel('特徴量 2')
    axes[0].set_title('GMMによるクラスタリング', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 確率密度の可視化
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1].contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.6)
    axes[1].scatter(X[:, 0], X[:, 1], c='white', s=10, alpha=0.5, edgecolors='black')
    axes[1].scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=200, marker='X',
                    edgecolors='black', linewidths=2)
    axes[1].set_xlabel('特徴量 1')
    axes[1].set_ylabel('特徴量 2')
    axes[1].set_title('GMM確率密度', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### GMM-HMMシステム

伝統的ASRでは、HMMの各状態の出力確率をGMMでモデル化します：

$$ b_j(o_t) = \sum_{m=1}^M c_{jm} \mathcal{N}(o_t | \mu_{jm}, \Sigma_{jm}) $$

  * $j$: HMM状態
  * $m$: GMM成分
  * $c_{jm}$: 状態 $j$ における成分 $m$ の重み

    
    
    from hmmlearn import hmm
    
    # GMMを出力分布に持つHMM
    n_states = 3
    n_mix = 4  # 各状態のGMM成分数
    
    # GaussianHMM: 各状態がガウス分布を持つ
    # n_mix > 1 で各状態がGMMになる
    gmm_hmm = hmm.GMMHMM(n_components=n_states, n_mix=n_mix,
                          covariance_type='diag', n_iter=100, random_state=42)
    
    # 訓練データ生成（2次元特徴量）
    np.random.seed(42)
    n_samples = 200
    train_data = np.random.randn(n_samples, 2) * 0.5
    
    # モデルの学習
    gmm_hmm.fit(train_data)
    
    print("\n=== GMM-HMM システム ===")
    print(f"HMM状態数: {n_states}")
    print(f"各状態のGMM成分数: {n_mix}")
    print(f"収束反復数: {gmm_hmm.monitor_.iter}")
    print(f"対数尤度: {gmm_hmm.score(train_data) * len(train_data):.2f}")
    
    # デコーディング
    test_data = np.random.randn(10, 2) * 0.5
    log_prob, states = gmm_hmm.decode(test_data)
    
    print(f"\nテストデータの状態列:")
    print(f"  {states}")
    print(f"  対数確率: {log_prob:.4f}")
    

* * *

## 2.4 言語モデル

### N-gramモデル

**N-gramモデル** は、単語列の確率を過去 $n-1$ 単語から予測します：

$$ P(w_1, w_2, \ldots, w_n) \approx \prod_{i=1}^n P(w_i | w_{i-n+1}, \ldots, w_{i-1}) $$

#### 主なN-gramモデル

モデル | 定義 | 例  
---|---|---  
**Unigram** | $P(w_i)$ | 単語の独立確率  
**Bigram** | $P(w_i | w_{i-1})$ | 直前の単語に依存  
**Trigram** | $P(w_i | w_{i-2}, w_{i-1})$ | 直前2単語に依存  
  
### 最尤推定

N-gramの確率は訓練コーパスからのカウントで推定します：

$$ P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})} $$

  * $C(w_{i-1}, w_i)$: バイグラム $(w_{i-1}, w_i)$ の出現回数
  * $C(w_{i-1})$: 単語 $w_{i-1}$ の出現回数

### パープレキシティ（Perplexity）

**パープレキシティ** は言語モデルの評価指標です：

$$ \text{PPL} = P(w_1, \ldots, w_N)^{-1/N} = \sqrt[N]{\prod_{i=1}^N \frac{1}{P(w_i | w_1, \ldots, w_{i-1})}} $$

低いほど良いモデルです。

### スムージング技術

未観測のN-gramに確率を割り当てるための技術：

#### 1\. Add-kスムージング（Laplace）

$$ P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + k|V|} $$

#### 2\. Kneser-Neyスムージング

文脈の多様性を考慮：

$$ P_{\text{KN}}(w_i | w_{i-1}) = \frac{\max(C(w_{i-1}, w_i) - \delta, 0)}{C(w_{i-1})} + \lambda(w_{i-1}) P_{\text{continuation}}(w_i) $$

### 実装：N-gram言語モデル
    
    
    import numpy as np
    from collections import defaultdict, Counter
    from typing import List, Tuple
    
    class BigramLanguageModel:
        """
        バイグラム言語モデル（Add-kスムージング付き）
        """
        def __init__(self, k: float = 1.0):
            self.k = k
            self.unigram_counts = Counter()
            self.bigram_counts = defaultdict(Counter)
            self.vocab = set()
    
        def train(self, corpus: List[List[str]]):
            """
            コーパスからモデルを学習
    
            Args:
                corpus: 文のリスト（各文は単語のリスト）
            """
            for sentence in corpus:
                # 開始・終了タグを追加
                words = ['~~'] + sentence + ['~~ ']
    
                for word in words:
                    self.vocab.add(word)
                    self.unigram_counts[word] += 1
    
                for w1, w2 in zip(words[:-1], words[1:]):
                    self.bigram_counts[w1][w2] += 1
    
            print(f"語彙サイズ: {len(self.vocab)}")
            print(f"総単語数: {sum(self.unigram_counts.values())}")
            print(f"ユニークバイグラム数: {sum(len(counts) for counts in self.bigram_counts.values())}")
    
        def probability(self, w1: str, w2: str) -> float:
            """
            バイグラム確率 P(w2|w1) を計算（Add-kスムージング）
            """
            numerator = self.bigram_counts[w1][w2] + self.k
            denominator = self.unigram_counts[w1] + self.k * len(self.vocab)
            return numerator / denominator
    
        def sentence_probability(self, sentence: List[str]) -> float:
            """
            文の確率を計算
            """
            words = ['~~'] + sentence + ['~~ ']
            prob = 1.0
    
            for w1, w2 in zip(words[:-1], words[1:]):
                prob *= self.probability(w1, w2)
    
            return prob
    
        def perplexity(self, test_corpus: List[List[str]]) -> float:
            """
            テストコーパスのパープレキシティを計算
            """
            log_prob = 0
            n_words = 0
    
            for sentence in test_corpus:
                words = ['~~'] + sentence + ['~~ ']
                n_words += len(words) - 1
    
                for w1, w2 in zip(words[:-1], words[1:]):
                    prob = self.probability(w1, w2)
                    log_prob += np.log2(prob)
    
            return 2 ** (-log_prob / n_words)
    
    
    # サンプルコーパス
    train_corpus = [
        ['I', 'love', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun'],
        ['I', 'love', 'deep', 'learning'],
        ['deep', 'learning', 'is', 'powerful'],
        ['I', 'study', 'machine', 'learning'],
    ]
    
    test_corpus = [
        ['I', 'love', 'learning'],
        ['machine', 'learning', 'is', 'interesting']
    ]
    
    # モデルの学習
    print("=== バイグラム言語モデル ===\n")
    lm = BigramLanguageModel(k=0.1)
    lm.train(train_corpus)
    
    # 確率の計算
    print("\nバイグラム確率の例:")
    bigrams = [('I', 'love'), ('love', 'learning'), ('machine', 'learning'), ('learning', 'is')]
    for w1, w2 in bigrams:
        prob = lm.probability(w1, w2)
        print(f"  P({w2}|{w1}) = {prob:.4f}")
    
    # 文の確率
    print("\n文の確率:")
    for sentence in test_corpus:
        prob = lm.sentence_probability(sentence)
        print(f"  '{' '.join(sentence)}': {prob:.6e}")
    
    # パープレキシティ
    ppl = lm.perplexity(test_corpus)
    print(f"\nテストコーパスのパープレキシティ: {ppl:.2f}")
    

**出力** ：
    
    
    === バイグラム言語モデル ===
    
    語彙サイズ: 12
    総単語数: 35
    ユニークバイグラム数: 25
    
    バイグラム確率の例:
      P(love|I) = 0.4255
      P(learning|love) = 0.3571
      P(learning|machine) = 0.6667
      P(is|learning) = 0.5000
    
    文の確率:
      'I love learning': 2.547618e-03
      'machine learning is interesting': 1.984127e-04
    
    テストコーパスのパープレキシティ: 8.91
    

### KenLMライブラリ

実用的なN-gram言語モデルには**KenLM** を使用します：
    
    
    # KenLM を使用した高度な言語モデル
    # 注: 事前にインストールが必要: pip install https://github.com/kpu/kenlm/archive/master.zip
    
    import kenlm
    
    # ARPAフォーマットの言語モデルファイルから読み込み
    # model = kenlm.Model('path/to/model.arpa')
    
    # 文のスコア計算
    # score = model.score('this is a test sentence', bos=True, eos=True)
    # perplexity = model.perplexity('this is a test sentence')
    
    print("KenLMは効率的なN-gram言語モデル実装です")
    print("大規模コーパスの学習とクエリに最適化されています")
    

* * *

## 2.5 伝統的ASRパイプライン

### 完全なパイプライン構成
    
    
    ```mermaid
    graph TD
        A[音声信号Waveform] --> B[前処理Pre-emphasis]
        B --> C[フレーム化Framing]
        C --> D[窓掛けWindowing]
        D --> E[MFCC抽出Feature Extraction]
        E --> F[デルタ特徴Delta/Delta-Delta]
        F --> G[音響モデルGMM-HMM]
        G --> H[Viterbiデコーディング+ 言語モデル]
        H --> I[認識結果Text]
    
        style A fill:#ffebee
        style E fill:#e3f2fd
        style G fill:#fff3e0
        style H fill:#f3e5f5
        style I fill:#c8e6c9
    ```

### 実装：簡易ASRシステム
    
    
    import numpy as np
    import librosa
    from hmmlearn import hmm
    from sklearn.mixture import GaussianMixture
    from typing import List, Tuple
    
    class SimpleASR:
        """
        簡易的な音声認識システム（デモ用）
        """
        def __init__(self, n_mfcc: int = 13, n_states: int = 3):
            self.n_mfcc = n_mfcc
            self.n_states = n_states
            self.models = {}  # 単語ごとのHMMモデル
    
        def extract_features(self, audio_path: str, sr: int = 16000) -> np.ndarray:
            """
            音声ファイルからMFCC特徴を抽出
            """
            # 音声読み込み
            y, sr = librosa.load(audio_path, sr=sr)
    
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
    
            # デルタ特徴
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
    
            # 結合
            features = np.vstack([mfcc, delta, delta2])
    
            return features.T  # (時間, 特徴量)
    
        def train_word_model(self, word: str, audio_files: List[str]):
            """
            特定の単語のHMMモデルを学習
            """
            # 全訓練データから特徴抽出
            all_features = []
            lengths = []
    
            for audio_file in audio_files:
                features = self.extract_features(audio_file)
                all_features.append(features)
                lengths.append(len(features))
    
            # 結合
            X = np.vstack(all_features)
    
            # Left-to-Right HMM
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type='diag',
                n_iter=100,
                random_state=42
            )
    
            # 遷移確率を制約（Left-to-Right）
            model.transmat_ = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                if i < self.n_states - 1:
                    model.transmat_[i, i] = 0.5
                    model.transmat_[i, i+1] = 0.5
                else:
                    model.transmat_[i, i] = 1.0
    
            model.startprob_ = np.zeros(self.n_states)
            model.startprob_[0] = 1.0
    
            # 学習
            model.fit(X, lengths)
    
            self.models[word] = model
    
            print(f"単語 '{word}' のモデルを学習しました")
            print(f"  訓練サンプル数: {len(audio_files)}")
            print(f"  総フレーム数: {len(X)}")
    
        def recognize(self, audio_path: str) -> Tuple[str, float]:
            """
            音声ファイルを認識
    
            Returns:
                (認識された単語, スコア)
            """
            # 特徴抽出
            features = self.extract_features(audio_path)
    
            # 各単語モデルでスコア計算
            scores = {}
            for word, model in self.models.items():
                try:
                    score = model.score(features)
                    scores[word] = score
                except:
                    scores[word] = -np.inf
    
            # 最高スコアの単語を選択
            best_word = max(scores, key=scores.get)
            best_score = scores[best_word]
    
            return best_word, best_score
    
    
    # デモ使用例（実際の音声ファイルが必要）
    print("=== 簡易ASRシステム ===\n")
    print("このシステムは以下の手順で動作します：")
    print("1. 音声からMFCC特徴（+ デルタ）を抽出")
    print("2. 各単語をLeft-to-Right HMMでモデル化")
    print("3. Viterbiアルゴリズムで最も確からしい単語を選択")
    print("\n実際の使用には音声ファイルが必要です")
    
    # asr = SimpleASR(n_mfcc=13, n_states=3)
    #
    # # 訓練（単語ごとに複数の音声サンプル）
    # asr.train_word_model('hello', ['hello1.wav', 'hello2.wav', 'hello3.wav'])
    # asr.train_word_model('world', ['world1.wav', 'world2.wav', 'world3.wav'])
    #
    # # 認識
    # word, score = asr.recognize('test.wav')
    # print(f"認識結果: {word} (スコア: {score:.2f})")
    

### 言語モデルとの統合

実際のASRでは、音響モデルと言語モデルを統合します：

$$ \hat{W} = \arg\max_W \left[\log P(X|W) + \lambda \log P(W)\right] $$

  * $\lambda$: 言語モデルの重み（Language Model Weight）

    
    
    class ASRWithLanguageModel:
        """
        言語モデル統合ASR
        """
        def __init__(self, acoustic_model, language_model, lm_weight: float = 1.0):
            self.acoustic_model = acoustic_model
            self.language_model = language_model
            self.lm_weight = lm_weight
    
        def recognize_with_lm(self, audio_features: np.ndarray,
                              previous_words: List[str] = None) -> str:
            """
            言語モデルを使用した認識
            """
            # 音響スコア（各単語候補）
            acoustic_scores = {}
            for word in self.acoustic_model.models.keys():
                acoustic_scores[word] = self.acoustic_model.models[word].score(audio_features)
    
            # 言語モデルスコア
            if previous_words:
                lm_scores = {}
                for word in acoustic_scores.keys():
                    # バイグラム確率
                    prev_word = previous_words[-1] if previous_words else '~~'
                    lm_scores[word] = np.log(self.language_model.probability(prev_word, word))
            else:
                lm_scores = {word: 0 for word in acoustic_scores.keys()}
    
            # 総合スコア
            total_scores = {
                word: acoustic_scores[word] + self.lm_weight * lm_scores[word]
                for word in acoustic_scores.keys()
            }
    
            # 最良の単語を選択
            best_word = max(total_scores, key=total_scores.get)
    
            return best_word
    
    print("\n=== 言語モデル統合 ===")
    print("音響スコアと言語スコアを組み合わせることで、")
    print("文脈を考慮した認識精度の向上が可能です")~~

* * *

## 2.6 本章のまとめ

### 学んだこと

  1. **音声認識の基礎**

     * ASRは音響モデルと言語モデルの組み合わせ
     * WER（Word Error Rate）による評価
     * レーベンシュタイン距離によるエラー計算
  2. **Hidden Markov Models**

     * 状態遷移と出力確率のモデリング
     * Forward-Backwardアルゴリズム（評価）
     * Viterbiアルゴリズム（デコーディング）
     * Baum-Welchアルゴリズム（学習）
  3. **Gaussian Mixture Models**

     * 複数のガウス分布による密度推定
     * EMアルゴリズムによるパラメータ推定
     * GMM-HMMによる音響モデリング
  4. **言語モデル**

     * N-gramによる単語列の確率モデリング
     * スムージング技術（未観測事象への対処）
     * パープレキシティによる評価
  5. **ASRパイプライン**

     * 特徴抽出（MFCC + デルタ）
     * 音響モデル（GMM-HMM）
     * デコーディング（Viterbi）
     * 言語モデル統合

### 伝統的ASRの長所と短所

長所 | 短所  
---|---  
理論的に明確 | 大量のラベル付きデータが必要  
各コンポーネントが独立 | パイプライン全体の最適化が困難  
音素レベルでの解釈可能性 | 長時間依存関係のモデル化が弱い  
少ないデータでも動作 | 特徴エンジニアリングに依存  
  
### 次の章へ

第3章では、**現代的なEnd-to-End音声認識** を学びます：

  * Deep Speech（CTC）
  * Listen, Attend and Spell
  * Transformer-based ASR
  * Wav2Vec 2.0
  * Whisper

* * *

## 演習問題

### 問題1（難易度：easy）

以下の参照文と仮説文のWERを手計算で求めてください。

  * 参照: "the cat sat on the mat"
  * 仮説: "the cat sit on mat"

解答例

**解答** ：

参照と仮説をアラインメント：
    
    
    参照: the cat sat on the mat
    仮説: the cat sit on --- mat
    

エラーのカウント：

  * 置換（S）: sat → sit （1個）
  * 削除（D）: the （1個）
  * 挿入（I）: 0個

WERの計算：

$$ \text{WER} = \frac{S + D + I}{N} = \frac{1 + 1 + 0}{6} = \frac{2}{6} = 0.333 = 33.3\% $$

答え: **33.3%**

### 問題2（難易度：medium）

3状態のHMMで、以下のパラメータが与えられたとき、観測列 [0, 1, 0] の確率をForwardアルゴリズムで計算してください。
    
    
    # 初期確率
    pi = [0.6, 0.3, 0.1]
    
    # 遷移確率
    A = [[0.7, 0.2, 0.1],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    
    # 出力確率 (観測値 0 と 1)
    B = [[0.8, 0.2],
         [0.4, 0.6],
         [0.3, 0.7]]
    

解答例
    
    
    import numpy as np
    
    # パラメータ
    pi = np.array([0.6, 0.3, 0.1])
    A = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    B = np.array([[0.8, 0.2],
                  [0.4, 0.6],
                  [0.3, 0.7]])
    
    observations = [0, 1, 0]
    T = len(observations)
    N = len(pi)
    
    # Forward変数
    alpha = np.zeros((T, N))
    
    # 初期化 (t=0)
    alpha[0] = pi * B[:, observations[0]]
    print(f"t=0: α = {alpha[0]}")
    
    # 再帰 (t=1, 2, ...)
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, observations[t]]
        print(f"t={t}: α = {alpha[t]}")
    
    # 観測列の確率
    prob = np.sum(alpha[T-1])
    
    print(f"\n観測列 {observations} の確率:")
    print(f"P(O|λ) = {prob:.6f}")
    

**出力** ：
    
    
    t=0: α = [0.48 0.12 0.03]
    t=1: α = [0.0672 0.1092 0.0588]
    t=2: α = [0.06048 0.02688 0.01512]
    
    観測列 [0, 1, 0] の確率:
    P(O|λ) = 0.102480
    

### 問題3（難易度：medium）

バイグラム言語モデルにおいて、以下のコーパスから P("learning"|"machine") を最尤推定で求めてください。
    
    
    I love machine learning
    machine learning is fun
    I study machine learning
    deep learning is great
    

解答例

**解答** ：

カウント：

  * C(machine, learning) = 3
  * C(machine) = 3

最尤推定：

$$ P(\text{learning} | \text{machine}) = \frac{C(\text{machine}, \text{learning})}{C(\text{machine})} = \frac{3}{3} = 1.0 $$

答え: **1.0（100%）**

このコーパスでは「machine」の後には必ず「learning」が続いています。

### 問題4（難易度：hard）

GMMの2成分（K=2）を使って、以下の1次元データをクラスタリングし、各成分のパラメータ（平均、分散、重み）を求めてください。
    
    
    data = np.array([1.2, 1.5, 1.8, 2.0, 2.1, 8.5, 9.0, 9.2, 9.5, 10.0])
    

解答例
    
    
    import numpy as np
    from sklearn.mixture import GaussianMixture
    import matplotlib.pyplot as plt
    
    data = np.array([1.2, 1.5, 1.8, 2.0, 2.1, 8.5, 9.0, 9.2, 9.5, 10.0])
    X = data.reshape(-1, 1)
    
    # GMMによるクラスタリング
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    
    # パラメータ
    print("=== GMM パラメータ ===")
    print(f"\n成分1:")
    print(f"  平均: {gmm.means_[0][0]:.3f}")
    print(f"  分散: {gmm.covariances_[0][0][0]:.3f}")
    print(f"  重み: {gmm.weights_[0]:.3f}")
    
    print(f"\n成分2:")
    print(f"  平均: {gmm.means_[1][0]:.3f}")
    print(f"  分散: {gmm.covariances_[1][0][0]:.3f}")
    print(f"  重み: {gmm.weights_[1]:.3f}")
    
    # クラスタラベル
    labels = gmm.predict(X)
    print(f"\nクラスタラベル:")
    for i, (val, label) in enumerate(zip(data, labels)):
        print(f"  データ{i+1}: {val:.1f} → クラスタ{label}")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(data, np.zeros_like(data), c=labels, cmap='viridis',
                s=100, alpha=0.6, edgecolors='black')
    plt.scatter(gmm.means_, [0, 0], c='red', s=200, marker='X',
                edgecolors='black', linewidths=2, label='中心')
    plt.xlabel('値')
    plt.title('GMMによる1次元データのクラスタリング', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

**出力例** ：
    
    
    === GMM パラメータ ===
    
    成分1:
      平均: 1.720
      分散: 0.124
      重み: 0.500
    
    成分2:
      平均: 9.240
      分散: 0.294
      重み: 0.500
    
    クラスタラベル:
      データ1: 1.2 → クラスタ0
      データ2: 1.5 → クラスタ0
      データ3: 1.8 → クラスタ0
      データ4: 2.0 → クラスタ0
      データ5: 2.1 → クラスタ0
      データ6: 8.5 → クラスタ1
      データ7: 9.0 → クラスタ1
      データ8: 9.2 → クラスタ1
      データ9: 9.5 → クラスタ1
      データ10: 10.0 → クラスタ1
    

### 問題5（難易度：hard）

音響モデルと言語モデルを統合したASRシステムにおいて、言語モデルの重み（LM weight）を変化させると認識結果にどのような影響があるか説明してください。また、最適な重みはどのように決定すべきか述べてください。

解答例

**解答** ：

**LM weightの影響** ：

総合スコア：

$$ \text{Score}(W) = \log P(X|W) + \lambda \log P(W) $$

LM weight $\lambda$ | 影響 | 認識傾向  
---|---|---  
**小さい（0に近い）** | 音響モデル優先 | 音響的に似た単語を選択、文法的に不自然  
**適切** | バランスが取れる | 音響と文法の両方を考慮、最良の認識精度  
**大きい** | 言語モデル優先 | 文法的には正しいが音響的に誤り、頻出単語に偏る  
  
**最適な重みの決定方法** ：

  1. **開発セットでのグリッドサーチ**
         
         lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
         best_lambda = None
         best_wer = float('inf')
         
         for lam in lambda_values:
             wer = evaluate_asr(dev_set, lm_weight=lam)
             if wer < best_wer:
                 best_wer = wer
                 best_lambda = lam
         
         print(f"最適なLM weight: {best_lambda}")
         

  2. **ドメイン依存性の考慮**

     * 読み上げ音声: 音響品質が高い → 小さめの $\lambda$
     * 自発音声: ノイズが多い → 大きめの $\lambda$
     * 専門用語が多い: 言語モデルの信頼性が低い → 小さめの $\lambda$
  3. **動的調整**

     * 信頼度スコアに基づいて動的に調整
     * 音響品質（SNR）に応じて調整

**実例** ：
    
    
    音声: "I scream"（アイスクリーム）
    
    λ = 0.1（音響優先）:
      → "I scream"（音響的に正確）
    
    λ = 5.0（言語優先）:
      → "ice cream"（文法的に自然、言語モデルで高頻度）
    
    λ = 1.0（バランス）:
      → 文脈次第で適切に選択
    

**結論** ：最適なLM weightはドメイン、音響条件、言語モデルの品質に依存し、開発セットでの実験的な調整が必要です。

* * *

## 参考文献

  1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. _Proceedings of the IEEE_ , 77(2), 257-286.
  2. Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_. Springer.
  3. Jurafsky, D., & Martin, J. H. (2023). _Speech and Language Processing_ (3rd ed.). Draft.
  4. Gales, M., & Young, S. (2008). The application of hidden Markov models in speech recognition. _Foundations and Trends in Signal Processing_ , 1(3), 195-304.
  5. Heafield, K. (2011). KenLM: Faster and smaller language model queries. _Proceedings of the Sixth Workshop on Statistical Machine Translation_ , 187-197.
