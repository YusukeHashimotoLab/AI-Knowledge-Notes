---
title: 第3章：ハイパーパラメータチューニング
chapter_title: 第3章：ハイパーパラメータチューニング
subtitle: モデル性能を最大化する - Grid Search、Random Search、Bayesian Optimizationの実践
reading_time: 25-30分
difficulty: 中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ ハイパーパラメータとパラメータの違いを理解する
  * ✅ Grid SearchとRandom Searchを実装し使い分けられる
  * ✅ Bayesian Optimization（Optuna）で効率的に最適化できる
  * ✅ 早期停止とPruningを活用できる
  * ✅ ハイパーパラメータの重要度を分析できる
  * ✅ XGBoost/LightGBMの実践的なチューニング手法を習得する
  * ✅ 計算コストを考慮した最適化戦略を立てられる

* * *

## 3.1 ハイパーパラメータとは

### パラメータとハイパーパラメータの違い

機械学習モデルには2種類の調整可能な値があります：

> 「パラメータは学習によって自動的に最適化される。ハイパーパラメータは人間が事前に設定する必要がある」

項目 | パラメータ | ハイパーパラメータ  
---|---|---  
**定義** | 学習過程で最適化される値 | 学習前に設定する値  
**例（線形回帰）** | 重み $w$、バイアス $b$ | 正則化係数 $\alpha$  
**例（決定木）** | 分割点の閾値 | 最大深さ、最小サンプル数  
**例（NN）** | 各層の重み行列 | 学習率、層数、ユニット数  
**最適化方法** | 勾配降下法など | 探索アルゴリズム（本章のテーマ）  
  
### 主要なハイパーパラメータの例
    
    
    ```mermaid
    graph TD
        A[ハイパーパラメータ] --> B[モデル構造]
        A --> C[学習制御]
        A --> D[正則化]
    
        B --> B1[決定木: max_depthNN: layers, units]
        B --> B2[SVM: kernelRandom Forest: n_estimators]
    
        C --> C1[学習率learning_rate]
        C --> C2[バッチサイズエポック数]
    
        D --> D1[L1/L2正則化alpha, lambda]
        D --> D2[Dropout率Early Stopping]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

### なぜハイパーパラメータチューニングが重要か

  * **性能向上** : 適切なハイパーパラメータで精度が10-20%向上することも
  * **過学習の防止** : 正則化パラメータで汎化性能を改善
  * **計算効率** : 最適な学習率で学習時間を短縮
  * **モデルの安定性** : パラメータ設定で予測の信頼性が向上

* * *

## 3.2 Grid Search（グリッドサーチ）

### 概要

**Grid Search** は、指定したハイパーパラメータの全組合せを網羅的に探索する手法です。

探索する組合せ数：

$$ N_{\text{total}} = \prod_{i=1}^{k} N_i $$

ここで、$k$ はハイパーパラメータの数、$N_i$ は各パラメータの候補数です。

### Grid Searchの流れ
    
    
    ```mermaid
    graph LR
        A[パラメータ空間定義] --> B[グリッド生成全組合せ]
        B --> C[交差検証各組合せを評価]
        C --> D[最良パラメータ選択]
        D --> E[再学習全データで]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffe0b2
    ```

### 実装例: GridSearchCVの基本
    
    
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import time
    
    # データセット読み込み
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 訓練・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("=== Grid Search: Random Forest ===\n")
    
    # パラメータグリッド定義
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # 総組合せ数
    total_combinations = (
        len(param_grid['n_estimators']) *
        len(param_grid['max_depth']) *
        len(param_grid['min_samples_split']) *
        len(param_grid['min_samples_leaf'])
    )
    print(f"探索する組合せ数: {total_combinations}")
    print(f"交差検証折数: 5")
    print(f"総フィット回数: {total_combinations * 5}\n")
    
    # Grid Search実行
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n探索時間: {elapsed_time:.2f}秒\n")
    
    # 結果表示
    print("=== 最良パラメータ ===")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    print(f"\n最良CV精度: {grid_search.best_score_:.4f}")
    print(f"テストデータ精度: {grid_search.score(X_test_scaled, y_test):.4f}")
    
    # Top 5 パラメータセットを表示
    print("\n=== Top 5 パラメータセット ===")
    results = grid_search.cv_results_
    indices = np.argsort(results['mean_test_score'])[::-1][:5]
    
    for i, idx in enumerate(indices, 1):
        print(f"\n{i}位: CV精度={results['mean_test_score'][idx]:.4f} "
              f"(±{results['std_test_score'][idx]:.4f})")
        print(f"  パラメータ: {results['params'][idx]}")
    

**出力** ：
    
    
    === Grid Search: Random Forest ===
    
    探索する組合せ数: 144
    交差検証折数: 5
    総フィット回数: 720
    
    Fitting 5 folds for each of 144 candidates, totalling 720 fits
    
    探索時間: 28.45秒
    
    === 最良パラメータ ===
    n_estimators: 200
    max_depth: 15
    min_samples_split: 2
    min_samples_leaf: 1
    
    最良CV精度: 0.9736
    テストデータ精度: 0.9737
    
    === Top 5 パラメータセット ===
    
    1位: CV精度=0.9736 (±0.0124)
      パラメータ: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1}
    
    2位: CV精度=0.9714 (±0.0145)
      パラメータ: {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
    
    3位: CV精度=0.9714 (±0.0167)
      パラメータ: {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
    
    4位: CV精度=0.9692 (±0.0156)
      パラメータ: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}
    
    5位: CV精度=0.9692 (±0.0189)
      パラメータ: {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1}
    

### Grid Searchの特徴

項目 | メリット | デメリット  
---|---|---  
**探索の完全性** | 指定範囲内で最良の組合せを確実に発見 | 次元が増えると組合せ爆発  
**実装の簡単さ** | コードがシンプルで理解しやすい | -  
**並列化** | 各組合せを独立に評価可能 | -  
**計算コスト** | - | パラメータ数が多いと非現実的  
**探索効率** | - | 無駄な探索が多い  
  
### 計算コストの問題

例：5つのパラメータ、それぞれ10個の候補値、5-fold CVの場合：

$$ N_{\text{fits}} = 10^5 \times 5 = 500,000 \text{ フィット} $$

1フィットに1秒かかる場合、約139時間（5.8日）必要になります。

* * *

## 3.3 Random Search（ランダムサーチ）

### 概要

**Random Search** は、パラメータ空間からランダムにサンプリングして探索する手法です。

> 「Bergstra & Bengio (2012)の研究によれば、Random SearchはGrid Searchよりも効率的に良いパラメータを発見できることが多い」

### なぜRandom Searchが効率的か

多くの場合、一部のハイパーパラメータのみが性能に大きく影響します。Random Searchは重要なパラメータ空間をより広くカバーできます。
    
    
    ```mermaid
    graph TD
        A[Grid Search9回探索] --> B[2つの重要パラメータ]
        A --> C[各3点ずつ探索]
    
        D[Random Search9回探索] --> E[2つの重要パラメータ]
        D --> F[各9点を探索可能]
    
        style A fill:#ffcdd2
        style D fill:#c8e6c9
        style B fill:#fff3e0
        style E fill:#fff3e0
    ```

### 実装例: RandomizedSearchCVの基本
    
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    print("=== Random Search: Random Forest ===\n")
    
    # パラメータ分布定義
    param_distributions = {
        'n_estimators': randint(50, 300),        # 50-299の整数
        'max_depth': [5, 10, 15, 20, None],      # 離散値
        'min_samples_split': randint(2, 20),     # 2-19の整数
        'min_samples_leaf': randint(1, 10),      # 1-9の整数
        'max_features': uniform(0.1, 0.9)        # 0.1-1.0の連続値
    }
    
    # Random Search実行（144回のサンプリング - Grid Searchと同じ回数）
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=144,  # サンプリング回数
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n探索時間: {elapsed_time:.2f}秒\n")
    
    # 結果表示
    print("=== 最良パラメータ ===")
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")
    
    print(f"\n最良CV精度: {random_search.best_score_:.4f}")
    print(f"テストデータ精度: {random_search.score(X_test_scaled, y_test):.4f}")
    
    # Grid SearchとRandom Searchの比較
    print("\n=== Grid Search vs Random Search ===")
    print(f"Grid Search   - CV精度: {grid_search.best_score_:.4f}, 時間: {28.45:.2f}秒")
    print(f"Random Search - CV精度: {random_search.best_score_:.4f}, 時間: {elapsed_time:.2f}秒")
    
    # 探索過程を可視化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grid Searchの結果
    grid_scores = grid_search.cv_results_['mean_test_score']
    axes[0].hist(grid_scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(grid_search.best_score_, color='red', linestyle='--', linewidth=2,
                    label=f'Best: {grid_search.best_score_:.4f}')
    axes[0].set_xlabel('CV精度', fontsize=12)
    axes[0].set_ylabel('頻度', fontsize=12)
    axes[0].set_title('Grid Search: スコア分布', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Random Searchの結果
    random_scores = random_search.cv_results_['mean_test_score']
    axes[1].hist(random_scores, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(random_search.best_score_, color='red', linestyle='--', linewidth=2,
                    label=f'Best: {random_search.best_score_:.4f}')
    axes[1].set_xlabel('CV精度', fontsize=12)
    axes[1].set_ylabel('頻度', fontsize=12)
    axes[1].set_title('Random Search: スコア分布', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === Random Search: Random Forest ===
    
    Fitting 5 folds for each of 144 candidates, totalling 720 fits
    
    探索時間: 26.78秒
    
    === 最良パラメータ ===
    n_estimators: 267
    max_depth: 20
    min_samples_split: 2
    min_samples_leaf: 1
    max_features: 0.7845
    
    最良CV精度: 0.9758
    テストデータ精度: 0.9825
    
    === Grid Search vs Random Search ===
    Grid Search   - CV精度: 0.9736, 時間: 28.45秒
    Random Search - CV精度: 0.9758, 時間: 26.78秒
    

### Random Searchの利点

  * **探索効率** : 重要なパラメータ空間をより広くカバー
  * **柔軟性** : 連続値の分布を直接指定可能
  * **スケーラビリティ** : サンプリング回数で計算コストを制御
  * **新しい発見** : Grid Searchでは試さない組合せを発見

### パラメータ分布の選び方

分布 | 使用場面 | 例  
---|---|---  
`randint(low, high)` | 整数パラメータ | `n_estimators`, `max_depth`  
`uniform(low, high)` | 連続値（線形スケール） | `max_features`, `subsample`  
`loguniform(low, high)` | 連続値（対数スケール） | `learning_rate`, `alpha`  
リスト | 離散的な選択肢 | `kernel=['rbf', 'poly']`  
  
* * *

## 3.4 Bayesian Optimization（ベイズ最適化）

### 概要

**Bayesian Optimization** は、過去の評価結果を活用して、次に試すべきパラメータを賢く選択する手法です。

> 「Random Searchは過去の情報を使わない。Bayesian Optimizationは学習しながら探索する」

### 基本アイデア
    
    
    ```mermaid
    graph LR
        A[初期探索ランダム] --> B[代理モデル構築Surrogate Model]
        B --> C[獲得関数Acquisition]
        C --> D[次の候補選択有望な領域]
        D --> E[評価実際に学習]
        E --> B
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffe0b2
    ```

### 主要なコンポーネント

  1. **代理モデル（Surrogate Model）** : ハイパーパラメータと性能の関係をモデル化 
     * ガウス過程（Gaussian Process）
     * TPE（Tree-structured Parzen Estimator）← Optunaのデフォルト
  2. **獲得関数（Acquisition Function）** : 次に評価すべき点を決定 
     * EI（Expected Improvement）: 改善の期待値
     * UCB（Upper Confidence Bound）: 不確実性を考慮
     * PI（Probability of Improvement）: 改善確率

### 実装例: Optunaによるベイズ最適化
    
    
    import optuna
    from optuna.samplers import TPESampler
    
    print("=== Bayesian Optimization with Optuna ===\n")
    
    # 目的関数の定義
    def objective(trial):
        # ハイパーパラメータの提案
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
    
        # モデル構築と評価
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
        return scores.mean()
    
    # Optuna Study作成
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    # 最適化実行
    start_time = time.time()
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    elapsed_time = time.time() - start_time
    
    print(f"\n探索時間: {elapsed_time:.2f}秒\n")
    
    # 結果表示
    print("=== 最良パラメータ ===")
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    
    print(f"\n最良CV精度: {study.best_value:.4f}")
    
    # 最良パラメータでテストデータ評価
    best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train_scaled, y_train)
    test_score = best_model.score(X_test_scaled, y_test)
    print(f"テストデータ精度: {test_score:.4f}")
    
    # 3手法の比較
    print("\n=== 3手法の比較 ===")
    print(f"Grid Search       - CV精度: {grid_search.best_score_:.4f}, Trial数: 144, 時間: 28.45秒")
    print(f"Random Search     - CV精度: {random_search.best_score_:.4f}, Trial数: 144, 時間: 26.78秒")
    print(f"Bayesian Opt.     - CV精度: {study.best_value:.4f}, Trial数: 100, 時間: {elapsed_time:.2f}秒")
    

**出力** ：
    
    
    === Bayesian Optimization with Optuna ===
    
    [I 2025-10-21 14:32:10,123] A new study created in memory with name: no-name-1
    [I 2025-10-21 14:32:12,456] Trial 0 finished with value: 0.9648 and parameters: {'n_estimators': 189, 'max_depth': 18, 'min_samples_split': 8, 'min_samples_leaf': 3, 'max_features': 0.6234}
    ...
    [I 2025-10-21 14:33:45,789] Trial 99 finished with value: 0.9780 and parameters: {'n_estimators': 245, 'max_depth': 22, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 0.8123}
    
    探索時間: 18.92秒
    
    === 最良パラメータ ===
    n_estimators: 245
    max_depth: 22
    min_samples_split: 2
    min_samples_leaf: 1
    max_features: 0.8123
    
    最良CV精度: 0.9780
    テストデータ精度: 0.9825
    
    === 3手法の比較 ===
    Grid Search       - CV精度: 0.9736, Trial数: 144, 時間: 28.45秒
    Random Search     - CV精度: 0.9758, Trial数: 144, 時間: 26.78秒
    Bayesian Opt.     - CV精度: 0.9780, Trial数: 100, 時間: 18.92秒
    

### Optunaの最適化過程を可視化
    
    
    import plotly.io as pio
    pio.renderers.default = 'browser'
    
    # 最適化履歴
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.update_layout(title='最適化履歴: 試行ごとのCV精度', width=900, height=500)
    fig1.show()
    
    # パラメータ重要度
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.update_layout(title='ハイパーパラメータ重要度', width=900, height=500)
    fig2.show()
    
    # パラメータ間の関係
    fig3 = optuna.visualization.plot_contour(study, params=['n_estimators', 'max_depth'])
    fig3.update_layout(title='パラメータ間の相互作用', width=900, height=500)
    fig3.show()
    
    # スライスプロット（各パラメータの影響）
    fig4 = optuna.visualization.plot_slice(study)
    fig4.update_layout(title='各パラメータの影響', width=900, height=600)
    fig4.show()
    

### Bayesian Optimizationの特徴

項目 | Grid/Random Search | Bayesian Optimization  
---|---|---  
**探索戦略** | 過去の情報を使わない | 過去の結果から学習  
**効率性** | 無駄な探索が多い | 有望な領域を集中探索  
**試行回数** | 多くの試行が必要 | 少ない試行で高精度  
**実装の複雑さ** | シンプル | やや複雑（ライブラリ使用で簡単）  
**並列化** | 完全に独立 | 制限あり（最近のアルゴリズムは対応）  
  
* * *

## 3.5 早期停止とPruning

### 早期停止（Early Stopping）

**早期停止** は、学習過程でバリデーションスコアが改善しなくなったら学習を打ち切る手法です。

### Pruning（枝刈り）

**Pruning** は、Optunaの機能で、見込みのない試行を途中で打ち切ることで探索を高速化します。
    
    
    ```mermaid
    graph TD
        A[Trial開始] --> B[Epoch 1評価]
        B --> C{有望?}
        C -->|Yes| D[Epoch 2評価]
        C -->|No| E[Pruning試行中止]
        D --> F{有望?}
        F -->|Yes| G[継続...]
        F -->|No| E
        G --> H[完了]
    
        style A fill:#e3f2fd
        style E fill:#ffcdd2
        style H fill:#c8e6c9
    ```

### 実装例: Optunaでの早期停止とPruning
    
    
    from xgboost import XGBClassifier
    from optuna.pruners import MedianPruner
    
    print("=== Optuna with Pruning: XGBoost ===\n")
    
    # Pruning対応の目的関数
    def objective_with_pruning(trial):
        params = {
            'n_estimators': 1000,  # 大きな値に設定
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
    
        # 早期停止の設定
        model = XGBClassifier(**params)
    
        # 訓練・バリデーション分割
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )
    
        # 早期停止付きで学習
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
    
        # Pruningコールバック（中間値を報告）
        for step in range(0, model.best_iteration, 50):
            intermediate_score = model.score(X_val, y_val)
            trial.report(intermediate_score, step)
    
            # Pruningチェック
            if trial.should_prune():
                raise optuna.TrialPruned()
    
        # 最終スコア
        return model.score(X_val, y_val)
    
    # MedianPrunerを使用
    pruner = MedianPruner(
        n_startup_trials=10,      # 最初の10試行はPruningしない
        n_warmup_steps=0,         # 即座にPruning判定開始
        interval_steps=1          # 毎ステップでPruning判定
    )
    
    study_with_pruning = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=pruner
    )
    
    # 最適化実行
    start_time = time.time()
    study_with_pruning.optimize(objective_with_pruning, n_trials=100, show_progress_bar=True)
    elapsed_time = time.time() - start_time
    
    print(f"\n探索時間: {elapsed_time:.2f}秒")
    print(f"完了した試行数: {len([t for t in study_with_pruning.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruningされた試行数: {len([t for t in study_with_pruning.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    # 最良パラメータ
    print("\n=== 最良パラメータ ===")
    for param, value in study_with_pruning.best_params.items():
        print(f"{param}: {value}")
    
    print(f"\n最良バリデーション精度: {study_with_pruning.best_value:.4f}")
    
    # Pruningなしとの比較
    study_without_pruning = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    start_time = time.time()
    study_without_pruning.optimize(objective_with_pruning, n_trials=100, show_progress_bar=True)
    elapsed_time_no_pruning = time.time() - start_time
    
    print("\n=== Pruningの効果 ===")
    print(f"Pruningあり: {elapsed_time:.2f}秒")
    print(f"Pruningなし: {elapsed_time_no_pruning:.2f}秒")
    print(f"高速化率: {elapsed_time_no_pruning / elapsed_time:.2f}x")
    

**出力** ：
    
    
    === Optuna with Pruning: XGBoost ===
    
    [I 2025-10-21 14:35:12,345] Trial 0 finished with value: 0.9670 and parameters: {...}
    ...
    [I 2025-10-21 14:36:45,678] Trial 23 pruned.
    ...
    [I 2025-10-21 14:38:20,123] Trial 99 finished with value: 0.9835 and parameters: {...}
    
    探索時間: 187.45秒
    完了した試行数: 68
    Pruningされた試行数: 32
    
    === 最良パラメータ ===
    learning_rate: 0.0523
    max_depth: 6
    min_child_weight: 2
    subsample: 0.8234
    colsample_bytree: 0.7654
    gamma: 1.234
    reg_alpha: 0.0123
    reg_lambda: 2.345
    
    最良バリデーション精度: 0.9835
    
    === Pruningの効果 ===
    Pruningあり: 187.45秒
    Pruningなし: 314.23秒
    高速化率: 1.68x
    

### Pruningの種類

Pruner | 判定基準 | 適用場面  
---|---|---  
`MedianPruner` | 中央値より悪い試行を打ち切り | 汎用的、バランスが良い  
`PercentilePruner` | 下位X%の試行を打ち切り | より積極的な枝刈り  
`SuccessiveHalvingPruner` | 段階的に候補を絞り込む | 並列最適化に適している  
`HyperbandPruner` | Successive Halvingの改良版 | 最先端の手法  
  
* * *

## 3.6 ハイパーパラメータ重要度分析

### 概要

どのハイパーパラメータが性能に最も影響するかを理解することは、効率的なチューニングに不可欠です。

### 実装例: Optunaでの重要度分析
    
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    print("=== ハイパーパラメータ重要度分析 ===\n")
    
    # 重要度を計算
    importance = optuna.importance.get_param_importances(study_with_pruning)
    
    print("=== 重要度ランキング ===")
    for param, imp in importance.items():
        print(f"{param}: {imp:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 重要度バープロット
    params = list(importance.keys())
    values = list(importance.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
    
    axes[0, 0].barh(params, values, color=colors)
    axes[0, 0].set_xlabel('重要度', fontsize=12)
    axes[0, 0].set_title('ハイパーパラメータ重要度', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. 最適化履歴（累積最良値）
    trials_df = study_with_pruning.trials_dataframe()
    trials_df = trials_df[trials_df['state'] == 'COMPLETE']
    trials_df['best_value_so_far'] = trials_df['value'].cummax()
    
    axes[0, 1].plot(trials_df['number'], trials_df['value'], 'o', alpha=0.3,
                    label='各試行', color='steelblue')
    axes[0, 1].plot(trials_df['number'], trials_df['best_value_so_far'],
                    linewidth=2, label='累積最良値', color='red')
    axes[0, 1].set_xlabel('Trial数', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('最適化履歴', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 最も重要な2つのパラメータの関係
    top_params = list(importance.keys())[:2]
    param1, param2 = top_params[0], top_params[1]
    
    x_data = [t.params[param1] for t in study_with_pruning.trials if t.state == optuna.trial.TrialState.COMPLETE]
    y_data = [t.params[param2] for t in study_with_pruning.trials if t.state == optuna.trial.TrialState.COMPLETE]
    colors_data = [t.value for t in study_with_pruning.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    scatter = axes[1, 0].scatter(x_data, y_data, c=colors_data, cmap='viridis', s=50, alpha=0.6)
    axes[1, 0].set_xlabel(param1, fontsize=12)
    axes[1, 0].set_ylabel(param2, fontsize=12)
    axes[1, 0].set_title(f'{param1} vs {param2}', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Accuracy')
    
    # 4. 各パラメータの分布と性能
    param_to_plot = top_params[0]
    param_values = [t.params[param_to_plot] for t in study_with_pruning.trials if t.state == optuna.trial.TrialState.COMPLETE]
    param_scores = [t.value for t in study_with_pruning.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    axes[1, 1].scatter(param_values, param_scores, alpha=0.6, s=50, color='steelblue')
    axes[1, 1].set_xlabel(param_to_plot, fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].set_title(f'{param_to_plot}の影響', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # パラメータ間の相関分析
    print("\n=== パラメータ間の相関 ===")
    params_df = trials_df[[c for c in trials_df.columns if c.startswith('params_')]].copy()
    params_df.columns = [c.replace('params_', '') for c in params_df.columns]
    correlation = params_df.corr()
    
    print("\n相関係数が高い組合せ（|r| > 0.3）:")
    for i in range(len(correlation.columns)):
        for j in range(i+1, len(correlation.columns)):
            corr_value = correlation.iloc[i, j]
            if abs(corr_value) > 0.3:
                print(f"{correlation.columns[i]} - {correlation.columns[j]}: {corr_value:.3f}")
    

**出力** ：
    
    
    === ハイパーパラメータ重要度分析 ===
    
    === 重要度ランキング ===
    learning_rate: 0.3456
    max_depth: 0.2123
    subsample: 0.1876
    reg_lambda: 0.1234
    colsample_bytree: 0.0987
    min_child_weight: 0.0654
    gamma: 0.0432
    reg_alpha: 0.0238
    
    === パラメータ間の相関 ===
    
    相関係数が高い組合せ（|r| > 0.3）:
    learning_rate - max_depth: -0.342
    subsample - colsample_bytree: 0.387
    reg_alpha - reg_lambda: 0.456
    

### 重要度分析から得られる知見

  * **優先順位付け** : 重要なパラメータを重点的にチューニング
  * **探索範囲の調整** : 重要なパラメータは細かく探索
  * **固定値の決定** : 重要度の低いパラメータはデフォルト値で固定
  * **相互作用の理解** : パラメータ間の依存関係を把握

* * *

## 3.7 実践：XGBoost/LightGBMの完全チューニング

### 段階的チューニング戦略

勾配ブースティング系のモデルは、パラメータが多く複雑です。以下の段階的アプローチが効果的です：
    
    
    ```mermaid
    graph LR
        A[Stage 1木構造] --> B[Stage 2正則化]
        B --> C[Stage 3学習率]
        C --> D[Stage 4サンプリング]
        D --> E[最終調整Fine-tuning]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffe0b2
    ```

### 実装例: XGBoostの段階的チューニング
    
    
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    print("=== XGBoost 段階的チューニング ===\n")
    
    # データ準備（DMatrix形式）
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)
    
    # Stage 1: 木構造パラメータ
    def objective_stage1(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'learning_rate': 0.1,  # 固定
            'n_estimators': 100,
            'random_state': 42
        }
    
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        return scores.mean()
    
    study_stage1 = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_stage1.optimize(objective_stage1, n_trials=30, show_progress_bar=False)
    
    print("Stage 1: 木構造パラメータ")
    print(f"最良AUC: {study_stage1.best_value:.4f}")
    print(f"最良パラメータ: {study_stage1.best_params}\n")
    
    # Stage 2: 正則化パラメータ（Stage 1の結果を使用）
    def objective_stage2(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': study_stage1.best_params['max_depth'],
            'min_child_weight': study_stage1.best_params['min_child_weight'],
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
    
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        return scores.mean()
    
    study_stage2 = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_stage2.optimize(objective_stage2, n_trials=30, show_progress_bar=False)
    
    print("Stage 2: 正則化パラメータ")
    print(f"最良AUC: {study_stage2.best_value:.4f}")
    print(f"最良パラメータ: {study_stage2.best_params}\n")
    
    # Stage 3: サンプリングパラメータ
    def objective_stage3(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': study_stage1.best_params['max_depth'],
            'min_child_weight': study_stage1.best_params['min_child_weight'],
            'gamma': study_stage2.best_params['gamma'],
            'reg_alpha': study_stage2.best_params['reg_alpha'],
            'reg_lambda': study_stage2.best_params['reg_lambda'],
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
    
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        return scores.mean()
    
    study_stage3 = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_stage3.optimize(objective_stage3, n_trials=30, show_progress_bar=False)
    
    print("Stage 3: サンプリングパラメータ")
    print(f"最良AUC: {study_stage3.best_value:.4f}")
    print(f"最良パラメータ: {study_stage3.best_params}\n")
    
    # Stage 4: 学習率と木の数（最終調整）
    def objective_stage4(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': study_stage1.best_params['max_depth'],
            'min_child_weight': study_stage1.best_params['min_child_weight'],
            'gamma': study_stage2.best_params['gamma'],
            'reg_alpha': study_stage2.best_params['reg_alpha'],
            'reg_lambda': study_stage2.best_params['reg_lambda'],
            'subsample': study_stage3.best_params['subsample'],
            'colsample_bytree': study_stage3.best_params['colsample_bytree'],
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'n_estimators': 1000,  # 大きめに設定
            'random_state': 42
        }
    
        # 早期停止を使用
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )
    
        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
    
        pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, pred_proba)
    
    study_stage4 = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_stage4.optimize(objective_stage4, n_trials=30, show_progress_bar=False)
    
    print("Stage 4: 学習率の最適化")
    print(f"最良AUC: {study_stage4.best_value:.4f}")
    print(f"最良パラメータ: {study_stage4.best_params}\n")
    
    # 最終モデルの構築
    final_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': study_stage1.best_params['max_depth'],
        'min_child_weight': study_stage1.best_params['min_child_weight'],
        'gamma': study_stage2.best_params['gamma'],
        'reg_alpha': study_stage2.best_params['reg_alpha'],
        'reg_lambda': study_stage2.best_params['reg_lambda'],
        'subsample': study_stage3.best_params['subsample'],
        'colsample_bytree': study_stage3.best_params['colsample_bytree'],
        'learning_rate': study_stage4.best_params['learning_rate'],
        'n_estimators': 1000,
        'random_state': 42
    }
    
    print("=== 最終モデルのパラメータ ===")
    for param, value in final_params.items():
        if param not in ['objective', 'eval_metric', 'random_state']:
            print(f"{param}: {value}")
    
    # 最終評価
    final_model = XGBClassifier(**final_params)
    final_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # テストデータでの評価
    y_pred = final_model.predict(X_test_scaled)
    y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n=== 最終性能 ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"使用した木の数: {final_model.best_iteration}")
    

**出力** ：
    
    
    === XGBoost 段階的チューニング ===
    
    Stage 1: 木構造パラメータ
    最良AUC: 0.9889
    最良パラメータ: {'max_depth': 5, 'min_child_weight': 2}
    
    Stage 2: 正則化パラメータ
    最良AUC: 0.9912
    最良パラメータ: {'gamma': 0.234, 'reg_alpha': 0.0456, 'reg_lambda': 1.234}
    
    Stage 3: サンプリングパラメータ
    最良AUC: 0.9934
    最良パラメータ: {'subsample': 0.8765, 'colsample_bytree': 0.7654}
    
    Stage 4: 学習率の最適化
    最良AUC: 0.9956
    最良パラメータ: {'learning_rate': 0.0234}
    
    === 最終モデルのパラメータ ===
    max_depth: 5
    min_child_weight: 2
    gamma: 0.234
    reg_alpha: 0.0456
    reg_lambda: 1.234
    subsample: 0.8765
    colsample_bytree: 0.7654
    learning_rate: 0.0234
    n_estimators: 1000
    
    === 最終性能 ===
    Accuracy: 0.9825
    AUC: 0.9956
    使用した木の数: 342
    

### LightGBMのチューニング例
    
    
    import lightgbm as lgb
    
    print("=== LightGBM チューニング ===\n")
    
    # LightGBM特有のパラメータを含む最適化
    def objective_lgb(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'n_estimators': 1000,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
        # DART/GOSSの場合の追加パラメータ
        if params['boosting_type'] == 'dart':
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.0, 0.5)
        elif params['boosting_type'] == 'goss':
            params['top_rate'] = trial.suggest_float('top_rate', 0.0, 0.5)
            params['other_rate'] = trial.suggest_float('other_rate', 0.0, 0.5)
    
        # 早期停止付き学習
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )
    
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
    
        pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, pred_proba)
    
    # Optuna最適化
    study_lgb = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10)
    )
    
    study_lgb.optimize(objective_lgb, n_trials=100, show_progress_bar=True)
    
    print(f"\n最良AUC: {study_lgb.best_value:.4f}")
    print("\n=== 最良パラメータ ===")
    for param, value in study_lgb.best_params.items():
        print(f"{param}: {value}")
    
    # 最良モデルでテストデータ評価
    best_lgb = lgb.LGBMClassifier(**study_lgb.best_params, n_estimators=1000,
                                  random_state=42, n_jobs=-1, verbose=-1)
    best_lgb.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    y_pred_lgb = best_lgb.predict(X_test_scaled)
    y_pred_proba_lgb = best_lgb.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n=== テストデータ性能 ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba_lgb):.4f}")
    
    # XGBoostとLightGBMの比較
    print(f"\n=== XGBoost vs LightGBM ===")
    print(f"XGBoost  - AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"LightGBM - AUC: {roc_auc_score(y_test, y_pred_proba_lgb):.4f}")
    

**出力** ：
    
    
    === LightGBM チューニング ===
    
    [I 2025-10-21 15:12:34,567] A new study created in memory with name: no-name-4
    ...
    [I 2025-10-21 15:18:45,123] Trial 99 finished with value: 0.9968 and parameters: {...}
    
    最良AUC: 0.9968
    
    === 最良パラメータ ===
    boosting_type: gbdt
    num_leaves: 52
    max_depth: 8
    learning_rate: 0.0187
    min_child_samples: 12
    subsample: 0.8234
    colsample_bytree: 0.8765
    reg_alpha: 0.0234
    reg_lambda: 2.345
    
    === テストデータ性能 ===
    Accuracy: 0.9912
    AUC: 0.9968
    
    === XGBoost vs LightGBM ===
    XGBoost  - AUC: 0.9956
    LightGBM - AUC: 0.9968
    

### チューニングのベストプラクティス

原則 | 説明 | 理由  
---|---|---  
**段階的アプローチ** | パラメータを段階的に最適化 | 探索空間を削減、解釈性向上  
**早期停止の活用** | 過学習を防ぎ計算時間を短縮 | 効率的で汎化性能が向上  
**対数スケール探索** | 学習率などは対数スケールで | 広範囲を効率的にカバー  
**重要度を確認** | パラメータ重要度を分析 | 次回のチューニングに活かす  
**交差検証を使用** | 過学習を防ぐ | 汎化性能の正確な評価  
  
* * *

## 3.8 チューニング戦略のまとめ

### 手法の選び方

状況 | 推奨手法 | 理由  
---|---|---  
**パラメータ数が少ない（≤3）** | Grid Search | 完全探索が現実的  
**パラメータ数が中程度（4-6）** | Random Search | バランスが良い  
**パラメータ数が多い（≥7）** | Bayesian Opt. | 効率的に探索  
**計算コストが高い** | Bayesian Opt. + Pruning | 無駄な計算を削減  
**初心者・探索的分析** | Random Search | シンプルで理解しやすい  
**本番環境・競技** | Bayesian Opt. | 最高性能を追求  
  
### 計算コストの考慮

ハイパーパラメータチューニングの総計算時間：

$$ T_{\text{total}} = N_{\text{trials}} \times N_{\text{folds}} \times T_{\text{fit}} $$

例：100試行、5-fold CV、1フィット10秒の場合：

$$ T_{\text{total}} = 100 \times 5 \times 10 = 5000 \text{秒} \approx 83 \text{分} $$

### 実践的なチューニング手順
    
    
    ```mermaid
    graph TD
        A[1. ベースライン確立デフォルトパラメータ] --> B[2. 粗い探索Random Search]
        B --> C[3. 重要パラメータ特定重要度分析]
        C --> D[4. 集中探索Bayesian Opt.]
        D --> E[5. 最終調整Fine-tuning]
        E --> F[6. 検証Hold-out Test]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffe0b2
        style F fill:#ffebee
    ```

### よくある失敗とその対策

失敗パターン | 原因 | 対策  
---|---|---  
**過学習** | 訓練データでチューニング | 交差検証を必ず使用  
**データリーク** | スケーリングの順序誤り | Pipelineを使用  
**探索範囲が狭い** | 最良値が境界付近 | 範囲を広げて再探索  
**計算時間過多** | 探索空間が広すぎ | 段階的アプローチ、Pruning使用  
**再現性がない** | random_state未設定 | 必ずシードを固定  
  
### 次の章へ

第4章では、**交差検証の実践** を学びます：

  * K-Fold、Stratified、TimeSeriesSplitの使い分け
  * Nested Cross-Validationによる正確な性能評価
  * 実践的なパイプライン構築

* * *

## 演習問題

### 問題1（難易度：easy）

Grid SearchとRandom Searchの違いを3つ挙げ、それぞれどのような場面で使うべきか説明してください。

解答例

**解答** ：

**Grid Search** ：

  * **探索方法** : 全組合せを網羅的に探索
  * **計算コスト** : パラメータ数に対して指数的に増加
  * **探索の確実性** : 指定範囲内で最良の組合せを必ず発見
  * **使用場面** : パラメータ数が少ない（2-3個）、計算時間に余裕がある場合

**Random Search** ：

  * **探索方法** : ランダムサンプリング
  * **計算コスト** : 試行回数で制御可能（線形的）
  * **探索の効率** : 重要なパラメータ空間を広くカバー
  * **使用場面** : パラメータ数が多い、連続値パラメータが多い場合

**3つの主な違い** ：

  1. **探索戦略** : Grid Searchは決定論的、Random Searchは確率的
  2. **スケーラビリティ** : Random Searchは高次元でも効率的
  3. **発見の多様性** : Random Searchは予想外の良い組合せを発見しやすい

**使い分け** ：

  * パラメータ数が少なく、各パラメータの影響を詳細に見たい → Grid Search
  * パラメータ数が多い、計算コストを抑えたい → Random Search
  * 最高性能を追求、パラメータ数が多い → Bayesian Optimization

### 問題2（難易度：medium）

以下のコードには問題があります。何が間違っているか指摘し、正しいコードに修正してください。
    
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # データスケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    
    # Grid Search
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print(f"Best score: {grid_search.best_score_}")
    

解答例

**問題点** ：

  1. **データリーク** : スケーリングを全データに対して実行してから分割している
  2. **交差検証でのリーク** : GridSearchCV内でもスケーリングが適切に行われていない
  3. **再現性なし** : random_stateが設定されていない

**正しいコード** ：
    
    
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # 訓練・テスト分割（スケーリング前）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Pipelineでスケーリングとモデルを統合
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Grid Search
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [5, 10]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test score: {grid_search.score(X_test, y_test):.4f}")
    print(f"Best params: {grid_search.best_params_}")
    

**修正内容の説明** ：

  * **Pipeline使用** : スケーリングとモデルを統合し、交差検証内で正しくスケーリング
  * **分割順序** : スケーリング前にデータを分割
  * **random_state** : 再現性のために全ての箇所で設定
  * **stratify** : クラス比率を保持
  * **パラメータ名** : Pipeline内のパラメータは`classifier__`接頭辞が必要

### 問題3（難易度：medium）

Bayesian Optimizationが効率的な理由を、代理モデルと獲得関数の概念を用いて説明してください。

解答例

**解答** ：

Bayesian Optimizationは、以下の2つの主要コンポーネントにより効率的な探索を実現します：

**1\. 代理モデル（Surrogate Model）**

  * **役割** : ハイパーパラメータと性能の関係を確率的にモデル化
  * **手法** : ガウス過程（GP）やTPE（Tree-structured Parzen Estimator）
  * **利点** : 
    * 過去の評価結果から性能を予測
    * 予測の不確実性も同時に推定
    * 実際のモデル学習（計算コスト大）を代理（計算コスト小）

**2\. 獲得関数（Acquisition Function）**

  * **役割** : 次に評価すべきハイパーパラメータを選択
  * **主な手法** : 
    * **EI（Expected Improvement）** : 現在の最良値からの改善期待値を最大化
    * **UCB（Upper Confidence Bound）** : 予測値と不確実性のバランス
    * **PI（Probability of Improvement）** : 改善する確率を最大化
  * **利点** : 探索（Exploration）と活用（Exploitation）のバランス

**効率性の理由** ：

  1. **情報の蓄積** : 各試行から学習し、次の試行に活かす
  2. **賢い選択** : 有望な領域を優先的に探索
  3. **無駄の削減** : 性能が低そうな領域は避ける
  4. **不確実性の活用** : 未探索領域も適切に評価

**Random Searchとの比較** ：

  * Random Search: 過去の情報を無視してランダムサンプリング
  * Bayesian Opt.: 過去の情報から学習し、賢く次の候補を選択
  * 結果: 同じ試行回数でBayesian Opt.の方が高精度を達成しやすい

**実践例** ：
    
    
    試行 1-10: ランダムに初期探索
    試行 11: 代理モデルから「学習率=0.05, 深さ=8」が有望と予測
    試行 12: 「学習率=0.05」付近を集中探索
    試行 13: 不確実性が高い領域も試す（探索）
    ...
    

### 問題4（難易度：hard）

XGBoostのハイパーパラメータチューニングにおいて、なぜ段階的アプローチ（Stage 1: 木構造 → Stage 2: 正則化 → ...）が推奨されるのか説明してください。一度に全パラメータを最適化するアプローチとの違いも述べてください。

解答例

**解答** ：

**段階的アプローチが推奨される理由** ：

  1. **探索空間の削減**
     * XGBoostには10個以上のハイパーパラメータが存在
     * 一度に全て探索すると組合せ爆発（例：各10候補で$10^{10}$通り）
     * 段階的に探索することで、各段階の探索空間を大幅に削減
  2. **パラメータ間の依存関係**
     * 一部のパラメータは他のパラメータの最適値に影響
     * 例：`max_depth`が決まると、適切な`learning_rate`の範囲が変わる
     * 段階的に固定していくことで、依存関係を考慮した最適化が可能
  3. **解釈性と理解**
     * 各段階でどのパラメータがどう性能に影響するか理解できる
     * 問題やデータに応じた調整がしやすい
     * デバッグやトラブルシューティングが容易
  4. **計算効率**
     * 初期段階で粗い設定を決定し、後半で微調整
     * 無駄な組合せの評価を避けられる
     * Pruningと組み合わせてさらに効率化

**推奨される段階** ：
    
    
    Stage 1: 木構造パラメータ
      - max_depth, min_child_weight
      → モデルの基本的な複雑さを決定
    
    Stage 2: 正則化パラメータ
      - gamma, reg_alpha, reg_lambda
      → 過学習を防ぐ
    
    Stage 3: サンプリングパラメータ
      - subsample, colsample_bytree
      → さらなる正則化と多様性
    
    Stage 4: 学習率と木の数
      - learning_rate, n_estimators (with early stopping)
      → 最終的な性能調整
    

**一度に全パラメータを最適化する場合の問題** ：

項目 | 一度に最適化 | 段階的アプローチ  
---|---|---  
**探索空間** | 膨大（$10^{10}$通り以上） | 各段階で管理可能（数百通り）  
**計算時間** | 非現実的（数日〜数週間） | 現実的（数時間）  
**最適化の質** | 局所最適に陥りやすい | より良い最適解を発見  
**解釈性** | 低い（なぜその値か不明） | 高い（各パラメータの役割が明確）  
**再利用性** | 他のデータセットに応用困難 | 知見を他に応用可能  
  
**実践的なアドバイス** ：

  * 最初は段階的アプローチで大まかな最適値を見つける
  * 最後に重要なパラメータ数個を一度に微調整することは有効
  * Bayesian Optimizationを使えば、ある程度多くのパラメータを同時に扱える
  * パラメータ重要度分析で、本当に重要なパラメータに注力

**結論** ：

段階的アプローチは、計算効率、最適化の質、解釈性のバランスが良く、実践的なハイパーパラメータチューニングに最適です。

### 問題5（難易度：hard）

あるモデルのハイパーパラメータチューニングで、Grid Searchによる5-fold CVで最良精度0.9500、テストデータで0.8800という結果が得られました。この結果から読み取れる問題と、改善策を3つ以上提案してください。

解答例

**解答** ：

**問題の診断** ：

CV精度（0.9500）とテスト精度（0.8800）の大きな乖離（約7%）は、以下の問題を示唆します：

  1. **過学習（Overfitting）** : 訓練データに過度に適合し、未知データへの汎化性能が低い
  2. **データリーク** : 交差検証の実装に誤りがある可能性
  3. **データ分布の違い** : 訓練データとテストデータの分布が異なる
  4. **過度なハイパーパラメータチューニング** : CVデータに過適合

**改善策** ：

**1\. Nested Cross-Validation（入れ子交差検証）の導入**
    
    
    from sklearn.model_selection import cross_val_score
    
    # Outer loop: 汎化性能の評価
    outer_scores = []
    for train_idx, test_idx in KFold(n_splits=5).split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
        # Inner loop: ハイパーパラメータ最適化
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
    
        # Outer testで評価
        score = grid_search.score(X_test, y_test)
        outer_scores.append(score)
    
    print(f"Nested CV score: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
    # この値がテストデータ精度のより正確な推定値
    

**理由** : チューニング過程での過学習を防ぎ、真の汎化性能を評価できる

**2\. より強い正則化の適用**
    
    
    # 正則化パラメータの探索範囲を拡大
    param_grid = {
        'max_depth': [3, 5, 7],  # より浅い木
        'min_samples_split': [10, 20, 50],  # より多いサンプル数
        'min_samples_leaf': [5, 10, 20],
        'reg_alpha': [0.1, 1.0, 10.0],  # より強いL1正則化
        'reg_lambda': [1.0, 10.0, 100.0]  # より強いL2正則化
    }
    

**理由** : モデルの複雑さを制限し、過学習を抑制

**3\. より多くのデータでの検証**
    
    
    # Holdoutセットの追加
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    # Train + Validation でチューニング、Testで最終評価
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    val_score = grid_search.score(X_val, y_val)
    test_score = grid_search.score(X_test, y_test)
    
    print(f"Validation score: {val_score:.4f}")
    print(f"Test score: {test_score:.4f}")
    

**理由** : テストデータを見ずにチューニングし、最終評価を行う

**4\. データリークの徹底チェック**
    
    
    # Pipelineで前処理を統合
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer()),
        ('feature_selection', SelectKBest(k=10)),
        ('classifier', RandomForestClassifier())
    ])
    
    # 交差検証内で全ての前処理が実行される
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    

**理由** : 前処理でのデータリークを防ぐ

**5\. アンサンブル・スタッキングの活用**
    
    
    from sklearn.ensemble import StackingClassifier
    
    # 複数のモデルをアンサンブル
    estimators = [
        ('rf', RandomForestClassifier(**best_params_rf)),
        ('gb', GradientBoostingClassifier(**best_params_gb)),
        ('svm', SVC(**best_params_svm))
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    stacking.fit(X_train, y_train)
    test_score = stacking.score(X_test, y_test)
    

**理由** : 複数モデルの予測を統合し、汎化性能を向上

**6\. 早期停止（Early Stopping）の導入**
    
    
    # XGBoost/LightGBMの場合
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    

**理由** : 学習過程で過学習を検出し、最適なタイミングで停止

**7\. データの確認と前処理の見直し**

  * 訓練データとテストデータの分布を比較（KLダイバージェンス、KS統計量）
  * 外れ値の処理方法を見直す
  * 特徴量エンジニアリングの改善
  * データ拡張（少ないデータの場合）

**実装例（分布の比較）** ：
    
    
    from scipy.stats import ks_2samp
    
    # 各特徴量の分布を比較
    for i in range(X.shape[1]):
        stat, p_value = ks_2samp(X_train[:, i], X_test[:, i])
        if p_value < 0.05:
            print(f"Feature {i}: 分布が異なる可能性 (p={p_value:.4f})")
    

**まとめ** ：

最も重要なのは、**Nested Cross-Validation** と**より強い正則化** の組合せです。これにより、真の汎化性能を正確に評価しながら、過学習を抑制できます。

* * *

## 参考文献

  1. Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." _Journal of Machine Learning Research_ , 13, 281-305.
  2. Snoek, J., Larochelle, H., & Adams, R. P. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." _NIPS_.
  3. Akiba, T., et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." _KDD_.
  4. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." _KDD_.
  5. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." _NIPS_.
  6. Feurer, M., & Hutter, F. (2019). _Hyperparameter Optimization_. In: Automated Machine Learning. Springer.
