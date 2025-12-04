---
title: 第3章：高度なチューニング手法
chapter_title: 第3章：高度なチューニング手法
subtitle: Hyperband、BOHB、Population-based Trainingによる効率的な探索
reading_time: 25-30分
difficulty: 中級-上級
code_examples: 6
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Successive HalvingとHyperbandの原理を理解する
  * ✅ BOHBによるベイズ最適化とHyperbandの融合を活用できる
  * ✅ Population-based Training (PBT)で並列学習を最適化できる
  * ✅ Hyperopt、SMAC、Ax/BoTorchなど主要ライブラリの特徴を理解する
  * ✅ Ray Tuneで大規模分散チューニングを実装できる

* * *

## 3.1 Hyperband

### Successive Halvingの原理

**Successive Halving** は、限られた計算リソースを効率的に配分する手法です。基本アイデアは以下の通り：

  1. 多数の設定でトレーニングを少量のリソースで開始
  2. 性能の悪い設定を段階的に除外（半分ずつ）
  3. 残った有望な設定により多くのリソースを割り当て

> **重要** : 早期に性能が悪い設定を除外することで、計算コストを大幅に削減できます。

### アルゴリズムの流れ
    
    
    ```mermaid
    graph TD
        A[n個の設定をランダム生成] --> B[各設定をrリソースで評価]
        B --> C{性能の上位n/2を選択}
        C --> D[リソースを2倍に増やす]
        D --> E{さらに上位n/4を選択}
        E --> F[リソースを2倍に増やす]
        F --> G[最終的に最良の設定が残る]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#e3f2fd
        style D fill:#f3e5f5
        style E fill:#e3f2fd
        style F fill:#f3e5f5
        style G fill:#c8e6c9
    ```

### Hyperbandアルゴリズム

**Hyperband** は、Successive Halvingを複数の異なる設定で実行し、リソース配分戦略を最適化します。

パラメータ：

  * **R** : 1つの設定に割り当てる最大リソース（エポック数など）
  * **η** : 各ラウンドでの削減率（通常3または4）

$$ s_{\max} = \lfloor \log_\eta(R) \rfloor $$

### Optunaでの実装（HyperbandPruner）
    
    
    import optuna
    from optuna.pruners import HyperbandPruner
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    # Hyperbandの設定
    pruner = HyperbandPruner(
        min_resource=1,      # 最小リソース（エポック）
        max_resource=100,    # 最大リソース
        reduction_factor=3   # 削減率η
    )
    
    def objective(trial):
        # ハイパーパラメータの提案
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
        # データの準備
        X, y = load_iris(return_X_y=True)
    
        # 段階的にn_estimatorsを増やして評価（Hyperband対応）
        for step in range(1, 6):
            # 現在のステップに応じた木の数
            current_n_estimators = int(n_estimators * step / 5)
    
            model = RandomForestClassifier(
                n_estimators=current_n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
    
            # 交差検証スコア
            score = cross_val_score(model, X, y, cv=3, n_jobs=-1).mean()
    
            # Optunaに中間値を報告
            trial.report(score, step)
    
            # 枝刈り判定
            if trial.should_prune():
                raise optuna.TrialPruned()
    
        return score
    
    # スタディの実行
    study = optuna.create_study(
        direction='maximize',
        pruner=pruner,
        study_name='hyperband_example'
    )
    
    study.optimize(objective, n_trials=100, timeout=300)
    
    print("\n=== Hyperband 最適化結果 ===")
    print(f"最良スコア: {study.best_value:.4f}")
    print(f"最良パラメータ: {study.best_params}")
    print(f"\n完了試行: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"枝刈りされた試行: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    

**出力例** ：
    
    
    === Hyperband 最適化結果 ===
    最良スコア: 0.9733
    最良パラメータ: {'n_estimators': 142, 'max_depth': 8, 'min_samples_split': 3, 'min_samples_leaf': 1}
    
    完了試行: 28
    枝刈りされた試行: 72
    

> **効果** : 100試行のうち72試行が早期に枝刈りされ、計算時間を大幅に削減。

* * *

## 3.2 BOHB (Bayesian Optimization and HyperBand)

### HyperbandとベイズOPの融合

**BOHB** は、Hyperbandの効率的なリソース配分とベイズ最適化の賢い探索を組み合わせた手法です。

手法 | 長所 | 短所  
---|---|---  
**Hyperband** | 効率的なリソース配分 | ランダムサンプリング  
**ベイズ最適化** | 賢い探索 | 全てのリソースを割り当て  
**BOHB** | 効率的 + 賢い探索 | 実装が複雑  
  
### BOHBの動作原理

  1. **Hyperbandフレームワーク** でリソース配分を管理
  2. 各ラウンドで、**TPE（Tree-structured Parzen Estimator）** を使用してハイパーパラメータを提案
  3. 過去の試行結果から学習し、有望な領域を優先的に探索

    
    
    ```mermaid
    graph LR
        A[過去の試行データ] --> B[TPEモデル構築]
        B --> C[有望な設定を提案]
        C --> D[Successive Halvingで評価]
        D --> E[結果をフィードバック]
        E --> A
    
        style A fill:#e3f2fd
        style B fill:#f3e5f5
        style C fill:#fff3e0
        style D fill:#ffebee
        style E fill:#e8f5e9
    ```

### 実装と活用シーン
    
    
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner
    from sklearn.datasets import load_digits
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    
    # BOHB設定（TPE + Hyperband）
    sampler = TPESampler(seed=42, n_startup_trials=10)
    pruner = HyperbandPruner(
        min_resource=5,
        max_resource=100,
        reduction_factor=3
    )
    
    def objective_bohb(trial):
        # ハイパーパラメータ提案（TPEが賢く選択）
        hidden_layer_size = trial.suggest_int('hidden_layer_size', 50, 200)
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True)
    
        X, y = load_digits(return_X_y=True)
    
        # Hyperband: 段階的にmax_iterを増やす
        for step in range(1, 6):
            max_iter = int(100 * step / 5)
    
            model = MLPClassifier(
                hidden_layer_sizes=(hidden_layer_size,),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                random_state=42
            )
    
            score = cross_val_score(model, X, y, cv=3, n_jobs=-1).mean()
    
            trial.report(score, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
        return score
    
    # BOHBスタディ
    study_bohb = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='bohb_example'
    )
    
    study_bohb.optimize(objective_bohb, n_trials=50, timeout=180)
    
    print("\n=== BOHB 最適化結果 ===")
    print(f"最良スコア: {study_bohb.best_value:.4f}")
    print(f"最良パラメータ:")
    for key, value in study_bohb.best_params.items():
        print(f"  {key}: {value}")
    print(f"\n完了/枝刈り: {len([t for t in study_bohb.trials if t.state == optuna.trial.TrialState.COMPLETE])}/{len([t for t in study_bohb.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    

### 活用シーン

  * **ニューラルネットワーク** : エポック数を段階的に増やす
  * **アンサンブル学習** : 弱学習器の数を段階的に増やす
  * **大規模データ** : データサンプル数を段階的に増やす

* * *

## 3.3 Population-based Training (PBT)

### PBTの原理

**Population-based Training** は、複数のモデルを並列に学習させ、定期的に以下を実行します：

  1. **Exploit（活用）** : 性能の悪いモデルを性能の良いモデルに置き換え
  2. **Explore（探索）** : ハイパーパラメータを摂動させて新しい設定を試す

> **特徴** : 学習中にハイパーパラメータを動的に調整できる点が、従来手法との大きな違い。

### PBTのワークフロー
    
    
    ```mermaid
    graph TD
        A[Population初期化n個のモデル] --> B[各モデルを並列学習]
        B --> C{定期的な評価ポイント}
        C --> D[性能の悪いモデルを特定]
        D --> E[良いモデルの重みをコピーExploit]
        E --> F[ハイパーパラメータを摂動Explore]
        F --> G{学習終了?}
        G -->|No| B
        G -->|Yes| H[最良モデルを選択]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style E fill:#e3f2fd
        style F fill:#e8f5e9
        style H fill:#c8e6c9
    ```

### Ray Tuneでの実装
    
    
    from ray import tune
    from ray.tune.schedulers import PopulationBasedTraining
    import numpy as np
    
    def train_function(config):
        """トレーニング関数（シミュレーション）"""
        # 初期設定
        learning_rate = config["lr"]
        momentum = config["momentum"]
    
        # 学習のシミュレーション
        for step in range(100):
            # ダミーの性能指標（実際にはモデル訓練）
            # 学習率とモーメンタムが適切な範囲で良い性能
            optimal_lr = 0.01
            optimal_momentum = 0.9
    
            score = 1.0 - (
                abs(learning_rate - optimal_lr) / optimal_lr +
                abs(momentum - optimal_momentum) / optimal_momentum
            ) / 2
    
            # ノイズを追加してリアルな学習を模倣
            score += np.random.normal(0, 0.05)
    
            # Ray Tuneに結果を報告
            tune.report(score=score, lr=learning_rate, momentum=momentum)
    
    # PBTスケジューラの設定
    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="score",
        mode="max",
        perturbation_interval=10,  # 10イテレーションごとに摂動
        hyperparam_mutations={
            "lr": lambda: np.random.uniform(0.001, 0.1),
            "momentum": lambda: np.random.uniform(0.8, 0.99)
        }
    )
    
    # Ray Tuneの実行
    analysis = tune.run(
        train_function,
        name="pbt_example",
        scheduler=pbt_scheduler,
        num_samples=8,  # 8個のモデルを並列実行
        config={
            "lr": tune.uniform(0.001, 0.1),
            "momentum": tune.uniform(0.8, 0.99)
        },
        stop={"training_iteration": 100},
        verbose=1
    )
    
    print("\n=== PBT 最適化結果 ===")
    best_config = analysis.get_best_config(metric="score", mode="max")
    print(f"最良設定:")
    print(f"  Learning Rate: {best_config['lr']:.4f}")
    print(f"  Momentum: {best_config['momentum']:.4f}")
    print(f"\n最良スコア: {analysis.best_result['score']:.4f}")
    

### 並列学習との組み合わせ

PBTの最大の利点は、並列計算リソースを最大限活用できる点です：

シナリオ | 従来手法 | PBT  
---|---|---  
8 GPUで100エポック | 逐次に8設定を試す  
800エポック分の時間 | 8設定を同時学習  
100エポック分の時間  
動的調整 | 不可 | 学習中に最適化  
リソース効率 | 性能悪い設定も最後まで | 早期に良い設定に収束  
  
* * *

## 3.4 その他の高度な手法

### Hyperopt (TPE実装)

**Hyperopt** は、Tree-structured Parzen Estimator (TPE)を実装した人気のライブラリです。
    
    
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    # 探索空間の定義
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1)
    }
    
    # データ準備
    X, y = load_breast_cancer(return_X_y=True)
    
    def objective_hyperopt(params):
        """Hyperopt用の目的関数"""
        # 整数型に変換
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
    
        model = GradientBoostingClassifier(**params, random_state=42)
        score = cross_val_score(model, X, y, cv=5, n_jobs=-1).mean()
    
        # Hyperoptは最小化なので、負の値を返す
        return {'loss': -score, 'status': STATUS_OK}
    
    # 最適化実行
    trials = Trials()
    best = fmin(
        fn=objective_hyperopt,
        space=space,
        algo=tpe.suggest,  # TPEアルゴリズム
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(42)
    )
    
    print("\n=== Hyperopt (TPE) 最適化結果 ===")
    print("最良パラメータ:")
    for key, value in best.items():
        print(f"  {key}: {value}")
    print(f"\n最良スコア: {-min(trials.losses()):.4f}")
    

### SMAC (Random Forest based)

**SMAC (Sequential Model-based Algorithm Configuration)** は、ランダムフォレストをサロゲートモデルとして使用します。

特徴：

  * カテゴリカル変数と条件付きパラメータに強い
  * 不確実性推定が優れている
  * ノイズの多い目的関数にロバスト

### Ax/BoTorch (Facebook Research)

**Ax** と**BoTorch** は、Facebook Researchが開発した次世代ベイズ最適化フレームワークです。
    
    
    from ax.service.ax_client import AxClient
    from sklearn.datasets import load_wine
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    
    # Axクライアントの作成
    ax_client = AxClient()
    
    # 探索空間の定義
    ax_client.create_experiment(
        name="svm_optimization",
        parameters=[
            {"name": "C", "type": "range", "bounds": [0.1, 100.0], "log_scale": True},
            {"name": "gamma", "type": "range", "bounds": [0.0001, 1.0], "log_scale": True},
            {"name": "kernel", "type": "choice", "values": ["rbf", "poly", "sigmoid"]}
        ],
        objective_name="accuracy",
        minimize=False
    )
    
    # データ準備
    X, y = load_wine(return_X_y=True)
    
    # 最適化ループ
    for i in range(30):
        # 次の設定を提案
        parameters, trial_index = ax_client.get_next_trial()
    
        # モデル評価
        model = SVC(**parameters, random_state=42)
        score = cross_val_score(model, X, y, cv=5, n_jobs=-1).mean()
    
        # 結果を報告
        ax_client.complete_trial(trial_index=trial_index, raw_data=score)
    
    # 最良設定の取得
    best_parameters, metrics = ax_client.get_best_parameters()
    
    print("\n=== Ax/BoTorch 最適化結果 ===")
    print("最良パラメータ:")
    for key, value in best_parameters.items():
        print(f"  {key}: {value}")
    print(f"\n最良精度: {metrics[0]['accuracy']:.4f}")
    print(f"信頼区間: [{metrics[0]['accuracy'] - metrics[1]['accuracy']['accuracy']:.4f}, "
          f"{metrics[0]['accuracy'] + metrics[1]['accuracy']['accuracy']:.4f}]")
    

### 手法比較表

手法 | サロゲートモデル | 強み | 適用場面  
---|---|---|---  
**Hyperopt (TPE)** | カーネル密度推定 | シンプル、高速 | 一般的な最適化  
**SMAC** | ランダムフォレスト | 条件付きパラメータ | 複雑な探索空間  
**Ax/BoTorch** | ガウス過程 | 不確実性推定、マルチタスク | 研究・実験  
**Optuna** | TPE/GP/CMA-ES | 柔軟、枝刈り | 実用的な最適化  
  
* * *

## 3.5 実践: Ray Tuneによる大規模チューニング

### Ray Tuneセットアップ

**Ray Tune** は、分散ハイパーパラメータチューニングのための統一フレームワークです。
    
    
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.bayesopt import BayesOptSearch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Rayの初期化
    ray.init(ignore_reinit_error=True)
    
    # データ準備
    X, y = make_classification(
        n_samples=10000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # PyTorchデータセット
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    def train_model(config):
        """Ray Tune用のトレーニング関数"""
        # モデル定義
        model = nn.Sequential(
            nn.Linear(20, config["hidden_size_1"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_1"], config["hidden_size_2"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_2"], 2)
        )
    
        # 最適化器
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()
    
        # データローダー
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=256)
    
        # トレーニングループ
        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
    
            # 検証
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
    
            accuracy = correct / total
    
            # Ray Tuneに報告
            tune.report(accuracy=accuracy, epoch=epoch)
    
    # 探索空間
    search_space = {
        "hidden_size_1": tune.choice([32, 64, 128, 256]),
        "hidden_size_2": tune.choice([16, 32, 64, 128]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "dropout": tune.uniform(0.1, 0.5)
    }
    
    print("=== Ray Tune セットアップ完了 ===")
    print(f"探索空間: {len(search_space)}次元")
    

### PBTスケジューラの活用
    
    
    from ray.tune.schedulers import PopulationBasedTraining
    
    # PBTスケジューラ
    pbt = PopulationBasedTraining(
        time_attr="epoch",
        metric="accuracy",
        mode="max",
        perturbation_interval=5,
        hyperparam_mutations={
            "lr": lambda: 10 ** np.random.uniform(-4, -1),
            "dropout": lambda: np.random.uniform(0.1, 0.5)
        }
    )
    
    # Ray Tune実行（PBT）
    analysis_pbt = tune.run(
        train_model,
        name="pbt_neural_net",
        scheduler=pbt,
        num_samples=8,  # 8つのモデルを並列実行
        config=search_space,
        resources_per_trial={"cpu": 2, "gpu": 0},  # GPU利用時は変更
        verbose=1
    )
    
    print("\n=== PBT実行結果 ===")
    best_trial_pbt = analysis_pbt.get_best_trial("accuracy", "max", "last")
    print(f"最良精度: {best_trial_pbt.last_result['accuracy']:.4f}")
    print(f"最良設定:")
    for key, value in best_trial_pbt.config.items():
        print(f"  {key}: {value}")
    

### 分散環境での実行

Ray Tuneは、複数マシンでの分散実行をサポートします：
    
    
    # ASHAスケジューラ + ベイズ最適化
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.bayesopt import BayesOptSearch
    
    # ASHAスケジューラ（Hyperbandの改良版）
    asha_scheduler = ASHAScheduler(
        max_t=50,              # 最大エポック
        grace_period=5,        # 最小エポック
        reduction_factor=3     # 削減率
    )
    
    # ベイズ最適化サーチャー
    bayesopt = BayesOptSearch(
        metric="accuracy",
        mode="max"
    )
    
    # 分散実行
    analysis_distributed = tune.run(
        train_model,
        name="distributed_tuning",
        scheduler=asha_scheduler,
        search_alg=bayesopt,
        num_samples=100,  # 100試行
        config=search_space,
        resources_per_trial={"cpu": 2},
        verbose=1
    )
    
    print("\n=== 分散チューニング結果 ===")
    best_trial = analysis_distributed.get_best_trial("accuracy", "max", "last")
    print(f"最良精度: {best_trial.last_result['accuracy']:.4f}")
    print(f"\n試行統計:")
    print(f"  完了試行: {len(analysis_distributed.trials)}")
    print(f"  平均精度: {np.mean([t.last_result['accuracy'] for t in analysis_distributed.trials if 'accuracy' in t.last_result]):.4f}")
    
    # 結果の可視化
    import pandas as pd
    
    df = analysis_distributed.results_df
    print(f"\n=== トップ5設定 ===")
    top5 = df.nlargest(5, 'accuracy')[['accuracy', 'config/hidden_size_1', 'config/lr', 'config/dropout']]
    print(top5)
    
    # Rayのシャットダウン
    ray.shutdown()
    

### Ray Tuneの利点

機能 | 説明 | 利点  
---|---|---  
**統一API** | 複数スケジューラ/サーチャーを統一インターフェース | 簡単に手法を切り替え  
**分散実行** | 複数マシンで自動スケーリング | 大規模探索が可能  
**早期停止** | ASHA、Hyperband、Medianなど | 計算資源の節約  
**チェックポイント** | 中断・再開サポート | 長時間実行の安全性  
**可視化** | TensorBoard統合 | リアルタイム監視  
  
* * *

## 3.6 本章のまとめ

### 学んだこと

  1. **Hyperband**

     * Successive Halvingで効率的なリソース配分
     * 早期に性能の悪い設定を除外
     * Optunaで簡単に実装可能
  2. **BOHB**

     * HyperbandとTPEの融合
     * 効率的なリソース配分と賢い探索の両立
     * ニューラルネットワークなどで特に有効
  3. **Population-based Training**

     * 並列学習中に動的にハイパーパラメータを調整
     * Exploit（活用）とExplore（探索）のバランス
     * 大規模並列環境で真価を発揮
  4. **その他の手法**

     * Hyperopt: シンプルで高速なTPE実装
     * SMAC: 条件付きパラメータに強い
     * Ax/BoTorch: 最先端のベイズ最適化
  5. **Ray Tune**

     * 統一フレームワークで複数手法を活用
     * 分散環境での大規模チューニング
     * 実用的なツールとの統合

### 手法選択ガイドライン

シナリオ | 推奨手法 | 理由  
---|---|---  
限られた計算資源 | Hyperband | 効率的なリソース配分  
ニューラルネット | BOHB、PBT | 段階的学習と動的調整  
大規模並列環境 | PBT、Ray Tune | 並列リソースを最大活用  
条件付きパラメータ | SMAC | 複雑な探索空間に対応  
研究・実験 | Ax/BoTorch | 最先端手法とカスタマイズ性  
実用的プロジェクト | Optuna、Ray Tune | 使いやすさと実績  
  
### 次の章へ

第4章では、**実践的な最適化戦略** を学びます：

  * 探索空間の設計ベストプラクティス
  * 並列化と分散実行の最適化
  * 結果の分析と可視化
  * 本番環境へのデプロイ

* * *

## 参考文献

  1. Li, L., et al. (2018). "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization". _Journal of Machine Learning Research_ , 18(185), 1-52.
  2. Falkner, S., Klein, A., & Hutter, F. (2018). "BOHB: Robust and Efficient Hyperparameter Optimization at Scale". _ICML 2018_.
  3. Jaderberg, M., et al. (2017). "Population Based Training of Neural Networks". _arXiv:1711.09846_.
  4. Liaw, R., et al. (2018). "Tune: A Research Platform for Distributed Model Selection and Training". _arXiv:1807.05118_.
  5. Bergstra, J., et al. (2013). "Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures". _ICML 2013_.
