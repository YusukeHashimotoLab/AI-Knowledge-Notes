# Chapter 4 Quality Enhancements

This file contains enhancements to be integrated into chapter-4.md

## Code Reproducibility Section (add after section 4.1)

### コード再現性の確保

**環境設定**:

```python
# 第4章: アクティブラーニング戦略
# 必須ライブラリバージョン
"""
Python: 3.8+
numpy: 1.21.0
scikit-learn: 1.0.0
scipy: 1.7.0
matplotlib: 3.5.0
"""

import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

# 再現性確保
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 推奨カーネル設定（アクティブラーニング用）
kernel_default = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 Matern(length_scale=0.2, length_scale_bounds=(1e-2, 1e0), nu=2.5)

print("環境設定完了（アクティブラーニング用）")
```

---

## Practical Pitfalls Section (add after section 4.2)

### 4.3 実践的な落とし穴と対処法

#### 落とし穴1: 不確実性サンプリングの偏り

**問題**: 不確実性サンプリングが探索空間の端に集中しすぎる

**症状**:
- 境界付近ばかりサンプリング
- 内部領域の情報が不足
- 予測精度が不均一

**解決策**: エプシロン貪欲法との組み合わせ

```python
def epsilon_greedy_uncertainty_sampling(gp, X_candidate, epsilon=0.1):
    """
    εgreedy戦略を組み込んだ不確実性サンプリング

    Parameters:
    -----------
    gp : GaussianProcessRegressor
        学習済みGPモデル
    X_candidate : array (n_candidates, n_features)
        候補点
    epsilon : float
        ランダム探索の確率（0~1）

    Returns:
    --------
    next_x : array
        次のサンプリング点
    """
    if np.random.rand() < epsilon:
        # ε確率でランダムサンプリング
        next_idx = np.random.randint(len(X_candidate))
        print(f"  ランダム探索（ε={epsilon}）")
    else:
        # (1-ε)確率で不確実性サンプリング
        _, sigma = gp.predict(X_candidate, return_std=True)
        next_idx = np.argmax(sigma)
        print(f"  不確実性サンプリング（σ={sigma[next_idx]:.4f}）")

    next_x = X_candidate[next_idx]
    return next_x, next_idx

# 使用例
np.random.seed(42)
X_train = np.array([[0.1], [0.5], [0.9]])
y_train = np.sin(5 * X_train).ravel()

kernel = ConstantKernel(1.0) * RBF(length_scale=0.15)
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(X_train, y_train)

X_candidate = np.linspace(0, 1, 100).reshape(-1, 1)

# εgreedy不確実性サンプリング
for i in range(5):
    print(f"\nIteration {i+1}:")
    next_x, idx = epsilon_greedy_uncertainty_sampling(
        gp, X_candidate, epsilon=0.2  # 20%ランダム
    )
    print(f"  選択点: x={next_x[0]:.3f}")
```

---

#### 落とし穴2: 多様性サンプリングの計算コスト

**問題**: 大規模データで距離計算が遅い

**症状**:
- サンプリングに時間がかかる
- メモリ使用量が大きい
- スケールしない

**解決策**: k-means クラスタリングによる近似

```python
from sklearn.cluster import KMeans

def fast_diversity_sampling(X_sampled, X_candidate, n_clusters=10):
    """
    k-means クラスタリングによる高速多様性サンプリング

    Parameters:
    -----------
    X_sampled : array (n_sampled, n_features)
        既存サンプル
    X_candidate : array (n_candidates, n_features)
        候補点
    n_clusters : int
        クラスタ数

    Returns:
    --------
    next_x : array
        次のサンプリング点
    """
    # 候補点をクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_candidate)

    # 各クラスタ中心から最も遠い候補点を選択
    cluster_centers = kmeans.cluster_centers_
    distances_from_sampled = np.min(
        np.linalg.norm(
            cluster_centers[:, np.newaxis, :] -
            X_sampled[np.newaxis, :, :],
            axis=2
        ),
        axis=1
    )

    # 最も遠いクラスタの代表点を選択
    farthest_cluster = np.argmax(distances_from_sampled)
    cluster_mask = (kmeans.labels_ == farthest_cluster)
    candidates_in_cluster = X_candidate[cluster_mask]

    # クラスタ内でクラスタ中心に最も近い点を選択
    distances_to_center = np.linalg.norm(
        candidates_in_cluster - cluster_centers[farthest_cluster],
        axis=1
    )
    next_idx_in_cluster = np.argmin(distances_to_center)
    next_x = candidates_in_cluster[next_idx_in_cluster]

    return next_x

# ベンチマーク
import time

n_sampled = 100
n_candidates = 10000
X_sampled = np.random.rand(n_sampled, 4)
X_candidate = np.random.rand(n_candidates, 4)

# 従来法（全距離計算）
start = time.time()
from scipy.spatial.distance import cdist
distances = cdist(X_candidate, X_sampled)
min_distances = np.min(distances, axis=1)
next_idx_naive = np.argmax(min_distances)
time_naive = time.time() - start

# k-means近似法
start = time.time()
next_x_fast = fast_diversity_sampling(X_sampled, X_candidate, n_clusters=20)
time_fast = time.time() - start

print(f"従来法: {time_naive:.4f}秒")
print(f"k-means法: {time_fast:.4f}秒")
print(f"高速化率: {time_naive/time_fast:.1f}x")
```

---

#### 落とし穴3: クローズドループの実験失敗対応

**問題**: 実験失敗を考慮していない

**症状**:
- 実験失敗でループが停止
- 失敗データを活用できない
- ロバスト性が低い

**解決策**: 失敗を考慮したアクティブラーニング

```python
class RobustClosedLoopOptimizer:
    """
    実験失敗に対応したクローズドループ最適化
    """

    def __init__(self, objective_function, total_budget=50, failure_rate=0.1):
        """
        Parameters:
        -----------
        objective_function : callable
            目的関数（実験シミュレーター）
        total_budget : int
            総実験予算
        failure_rate : float
            実験失敗率（0~1）
        """
        self.objective_function = objective_function
        self.total_budget = total_budget
        self.failure_rate = failure_rate

        self.X_sampled = []
        self.y_observed = []
        self.failures = []

    def execute_experiment(self, x):
        """
        実験実行（失敗の可能性あり）

        Returns:
        --------
        success : bool
            実験成功フラグ
        result : float or None
            成功時は測定値、失敗時はNone
        """
        # 失敗シミュレーション
        if np.random.rand() < self.failure_rate:
            print(f"  実験失敗: x={x}")
            return False, None

        # 成功時は目的関数評価
        y = self.objective_function(x)
        return True, y

    def run(self):
        """クローズドループ最適化実行"""
        # 初期化
        X_init = np.random.uniform(0, 1, (5, 1))
        for x in X_init:
            success, y = self.execute_experiment(x)
            if success:
                self.X_sampled.append(x)
                self.y_observed.append(y)
                self.failures.append(False)
            else:
                self.failures.append(True)

        # メインループ
        experiments_done = len(X_init)

        while len(self.y_observed) < self.total_budget:
            if experiments_done >= self.total_budget * 1.5:
                print("実験予算超過（失敗多数）")
                break

            # GPモデル学習
            if len(self.y_observed) < 3:
                # データ不足時はランダムサンプリング
                next_x = np.random.uniform(0, 1, (1, 1))
                print(f"データ不足: ランダムサンプリング")
            else:
                kernel = ConstantKernel(1.0) * RBF(length_scale=0.15)
                gp = GaussianProcessRegressor(kernel=kernel)
                X_array = np.array(self.X_sampled)
                y_array = np.array(self.y_observed)
                gp.fit(X_array, y_array)

                # EI最大化
                X_candidate = np.linspace(0, 1, 500).reshape(-1, 1)
                mu, sigma = gp.predict(X_candidate, return_std=True)
                f_best = np.max(y_array)

                from scipy.stats import norm
                improvement = mu - f_best - 0.01
                Z = improvement / (sigma + 1e-9)
                ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

                next_idx = np.argmax(ei)
                next_x = X_candidate[next_idx:next_idx+1]

            # 実験実行
            success, y = self.execute_experiment(next_x)
            experiments_done += 1

            if success:
                self.X_sampled.append(next_x)
                self.y_observed.append(y)
                self.failures.append(False)
                print(f"成功 {len(self.y_observed)}/{self.total_budget}: "
                      f"x={next_x[0][0]:.3f}, y={y:.3f}")
            else:
                self.failures.append(True)
                print(f"失敗: 再試行します")

        # 結果サマリー
        success_rate = len(self.y_observed) / experiments_done
        print(f"\n最終結果:")
        print(f"  総実験回数: {experiments_done}")
        print(f"  成功実験数: {len(self.y_observed)}")
        print(f"  成功率: {success_rate:.1%}")
        print(f"  最良値: {np.max(self.y_observed):.4f}")

# 使用例
def noisy_objective(x):
    """ノイズのある目的関数"""
    return np.sin(5 * x[0]) * np.exp(-x[0]) + 0.1 * np.random.randn()

np.random.seed(42)
optimizer = RobustClosedLoopOptimizer(
    objective_function=noisy_objective,
    total_budget=20,
    failure_rate=0.2  # 20%失敗率
)
optimizer.run()
```

---

## End-of-Chapter Checklist (add before "演習問題")

### 4.7 章末チェックリスト

#### ✅ アクティブラーニングの理解

- [ ] アクティブラーニングとベイズ最適化の違いを説明できる
- [ ] 3つの主要戦略（不確実性、多様性、モデル変化）を理解している
- [ ] 各戦略の長所・短所を説明できる
- [ ] 問題に応じて戦略を選択できる
- [ ] 戦略を組み合わせる方法を知っている

**選択ガイド**:
```
探索空間の理解が目的        → 多様性サンプリング
予測精度の向上が目的        → 不確実性サンプリング
モデルの汎化性能向上        → 期待モデル変化
最適解の発見が目的          → ベイズ最適化（EI/UCB）
多様な候補材料の発見        → 多様性 + 不確実性の組み合わせ
```

---

#### ✅ 不確実性サンプリング（Uncertainty Sampling）

- [ ] 予測標準偏差σの意味を理解している
- [ ] 不確実性が高い領域の特定方法を知っている
- [ ] エプシロン貪欲法との組み合わせを実装できる
- [ ] 分類問題への応用（マージン、エントロピー）を理解している
- [ ] 不確実性サンプリングの限界を知っている

**実装チェック**:
```python
# このコードを完成させられますか？
def uncertainty_sampling(gp, X_candidate):
    """
    不確実性が最大の点を選択

    Returns:
    --------
    next_x : array
        次のサンプリング点
    uncertainty : float
        その点の不確実性
    """
    # あなたの実装
    _, sigma = gp.predict(X_candidate, return_std=True)
    next_idx = np.argmax(sigma)
    next_x = X_candidate[next_idx]
    uncertainty = sigma[next_idx]

    return next_x, uncertainty

# 正解！
```

---

#### ✅ 多様性サンプリング（Diversity Sampling）

- [ ] MaxMin距離の概念を理解している
- [ ] k-means クラスタリングによる近似を実装できる
- [ ] Determinantal Point Process (DPP) の基本を知っている
- [ ] 探索空間のカバー率を評価できる
- [ ] 大規模データでの高速化手法を知っている

**多様性の評価指標**:
```python
def evaluate_diversity(X_sampled, bounds):
    """
    サンプリングの多様性を評価

    Returns:
    --------
    coverage_score : float
        探索空間のカバー率（0~1）
    """
    # 探索空間を10分割してカバー率を計算
    n_dims = X_sampled.shape[1]
    n_bins = 10

    coverage_count = 0
    total_bins = n_bins ** n_dims

    # 簡易版: 1次元ごとのカバー率
    for dim in range(n_dims):
        hist, _ = np.histogram(
            X_sampled[:, dim],
            bins=n_bins,
            range=(bounds[dim, 0], bounds[dim, 1])
        )
        coverage_count += np.sum(hist > 0)

    coverage_score = coverage_count / (n_bins * n_dims)
    return coverage_score

# 使用例
bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
X_sampled = np.random.rand(20, 4)
coverage = evaluate_diversity(X_sampled, bounds)
print(f"カバー率: {coverage:.1%}")
```

---

#### ✅ クローズドループ最適化

- [ ] クローズドループシステムの構成要素を理解している
- [ ] AIエンジン、実験装置、データ管理の統合方法を知っている
- [ ] 実験失敗への対処法を実装できる
- [ ] リアルタイムモニタリングを設計できる
- [ ] 人間研究者の役割を理解している

**システム設計チェックリスト**:
```
□ 目的関数の定義と評価方法
□ 制約条件の明示
□ 初期サンプリング戦略
□ 獲得関数の選択
□ バッチサイズの決定
□ 実験失敗時の再試行ロジック
□ 異常検知と人間への通知
□ データの自動保存とバックアップ
□ 進捗の可視化
□ 終了条件の設定
```

---

#### ✅ 実世界応用の理解

- [ ] Berkeley A-Labの成果を説明できる
- [ ] RoboRXNのアプローチを理解している
- [ ] Materials Acceleration Platformの特徴を知っている
- [ ] 産業応用のROIを評価できる
- [ ] 成功要因と課題を分析できる

**ROI計算テンプレート**:
```
従来法:
  実験回数: ________ 回
  実験時間: ________ 時間/回
  人件費: ________ 円/時間
  総コスト: ________ 円
  開発期間: ________ ヶ月

AI駆動法（クローズドループ）:
  実験回数: ________ 回（ __% 削減）
  実験時間: ________ 時間/回（自動化）
  人件費: ________ 円/時間（監視のみ）
  システム構築: ________ 円（初期投資）
  総コスト: ________ 円
  開発期間: ________ ヶ月（__% 短縮）

投資回収期間: ________ ヶ月
```

---

#### ✅ 人間とAIの協働

- [ ] 人間の直感とAIの強みを理解している
- [ ] ハイブリッドアプローチの設計ができる
- [ ] 人間が介入すべき場面を判断できる
- [ ] 意思決定支援システムを構築できる
- [ ] フィードバックループを設計できる

**協働プロトコル**:
```
Phase 1: 問題定式化（人間主導）
  → 目的関数、制約、探索空間を定義
  → AIが実現可能性をチェック

Phase 2: 初期探索（AI主導）
  → AIがデータ効率的に探索
  → 人間が異常値を検証

Phase 3: 精密化（ハイブリッド）
  → AIが提案
  → 人間が物理的妥当性を評価
  → 協働で意思決定

Phase 4: 実装（人間主導）
  → 最終候補を人間が選択
  → AIが不確実性を定量化
```

---

### ✅ キャリアパスの理解

- [ ] アカデミア研究者パスを理解している
- [ ] 産業界R&Dエンジニアパスを知っている
- [ ] 自律実験専門家パスを検討できる
- [ ] 次に学ぶべきスキルを特定できる
- [ ] 自分のキャリア目標を明確化している

**次のステップ選択ガイド**:
```
理論研究志向
→ GNN入門 + 強化学習入門
→ 論文執筆、学会発表

実装・応用志向
→ ロボティクス実験自動化入門
→ 独自プロジェクト、ポートフォリオ作成

産業応用志向
→ 産業ケーススタディ深掘り
→ インターンシップ、実務経験

システム構築志向
→ クローズドループシステム構築
→ API設計、ハードウェア連携
```

---

### 合格基準

以下を達成していれば、シリーズ完了です：

1. **理論理解**: 各チェック項目の80%以上をクリア
2. **実装スキル**: 演習問題をすべて解ける
3. **応用力**: 新しい材料探索問題を定式化できる
4. **キャリア**: 次のステップが明確

**最終確認問題**:
1. 3つのアクティブラーニング戦略を実装し、性能を比較できますか？
2. クローズドループ最適化システムを設計できますか？
3. 実世界応用の成功事例から学びを抽出できますか？
4. 自分のキャリア目標に向けた次のステップを説明できますか？

すべてYESなら、おめでとうございます！
ベイズ最適化・アクティブラーニング入門シリーズを完了しました！

**次のシリーズへ**:
- ロボティクス実験自動化入門
- 強化学習入門（材料科学特化版）
- GNN入門

**継続的な学習**:
- 論文読解（週1本）
- オープンソース貢献
- コミュニティ参加
- 実プロジェクトへの応用

皆さんの成功を祈っています！
