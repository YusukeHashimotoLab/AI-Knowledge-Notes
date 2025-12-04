---
title: 第2章：ハイパーパラメータ最適化
chapter_title: 第2章：ハイパーパラメータ最適化
subtitle: AutoMLの核心 - 最適な設定を自動探索する
reading_time: 30-35分
difficulty: 中級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ ハイパーパラメータ最適化の基礎と探索戦略を理解する
  * ✅ Optunaを使った効率的なハイパーパラメータ探索を実装できる
  * ✅ Hyperoptでベイズ最適化を活用できる
  * ✅ Ray Tuneで分散ハイパーパラメータ最適化を実行できる
  * ✅ 高度なHPO手法（ASHA、PBT、Hyperband）を理解し適用できる
  * ✅ 実際のモデルに対して最適化戦略を選択・実装できる

* * *

## 2.1 HPOの基礎

### ハイパーパラメータとは

**ハイパーパラメータ（Hyperparameter）** は、モデルの学習プロセスを制御するパラメータで、学習前に設定する必要があります。

> 「モデルのパラメータは学習中に最適化されるが、ハイパーパラメータは学習前に人間が設定する」

### ハイパーパラメータの例

アルゴリズム | 主要なハイパーパラメータ | 影響  
---|---|---  
**ランダムフォレスト** | n_estimators, max_depth, min_samples_split | 性能、計算コスト、過学習  
**勾配ブースティング** | learning_rate, n_estimators, max_depth | 収束速度、性能、過学習  
**SVM** | C, kernel, gamma | 決定境界、汎化性能  
**ニューラルネット** | learning_rate, batch_size, hidden_units | 収束、性能、計算効率  
  
### 探索空間の定義

探索空間は、各ハイパーパラメータが取りうる値の範囲と分布を定義します。
    
    
    import numpy as np
    
    # 探索空間の例
    search_space = {
        # 整数型：木の数（50から500まで）
        'n_estimators': (50, 500),
    
        # 整数型：木の深さ（3から20まで）
        'max_depth': (3, 20),
    
        # 実数型（対数スケール）：学習率
        'learning_rate': (1e-4, 1e-1, 'log'),
    
        # カテゴリカル型：ブースティングタイプ
        'boosting_type': ['gbdt', 'dart', 'goss'],
    
        # 実数型（線形スケール）：正則化パラメータ
        'reg_alpha': (0.0, 10.0),
    }
    
    print("=== 探索空間の定義 ===")
    for param, space in search_space.items():
        print(f"{param}: {space}")
    

### Grid Search vs Random Search

#### Grid Search（格子探索）

すべての組み合わせを網羅的に探索します。
    
    
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import accuracy_score
    import time
    
    # データ生成
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, n_redundant=5,
                              random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Grid Searchの探索空間
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    print("=== Grid Search ===")
    print(f"探索する組み合わせ数: {3 * 4 * 3} = 36通り")
    
    # Grid Search実行
    start_time = time.time()
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start_time
    
    # 結果
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    test_score = accuracy_score(y_test, grid_search.predict(X_test))
    
    print(f"\n最適なパラメータ: {best_params}")
    print(f"CV精度: {best_score:.4f}")
    print(f"テスト精度: {test_score:.4f}")
    print(f"実行時間: {grid_time:.2f}秒")
    

**出力** ：
    
    
    === Grid Search ===
    探索する組み合わせ数: 3 * 4 * 3 = 36通り
    Fitting 3 folds for each of 36 candidates, totalling 108 fits
    
    最適なパラメータ: {'max_depth': 15, 'min_samples_split': 2, 'n_estimators': 200}
    CV精度: 0.9162
    テスト精度: 0.9200
    実行時間: 12.34秒
    

#### Random Search（ランダム探索）

ランダムに組み合わせをサンプリングして探索します。
    
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    # Random Searchの探索空間（分布で指定）
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(5, 25),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0.3, 0.7)  # 0.3から1.0の範囲
    }
    
    print("\n=== Random Search ===")
    print(f"ランダムに試行する回数: 50回")
    
    # Random Search実行
    start_time = time.time()
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions,
        n_iter=50,  # 50回のランダム試行
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    random_time = time.time() - start_time
    
    # 結果
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    test_score = accuracy_score(y_test, random_search.predict(X_test))
    
    print(f"\n最適なパラメータ: {best_params}")
    print(f"CV精度: {best_score:.4f}")
    print(f"テスト精度: {test_score:.4f}")
    print(f"実行時間: {random_time:.2f}秒")
    
    # 比較
    print(f"\n=== Grid vs Random 比較 ===")
    print(f"Grid Search: {grid_time:.2f}秒で精度{test_score:.4f}")
    print(f"Random Search: {random_time:.2f}秒で精度{test_score:.4f}")
    print(f"時間短縮: {(1 - random_time/grid_time)*100:.1f}%")
    

### 探索戦略の分類
    
    
    ```mermaid
    graph TD
        A[HPO戦略] --> B[単純探索]
        A --> C[適応的探索]
        A --> D[多段階探索]
    
        B --> B1[Grid Search]
        B --> B2[Random Search]
    
        C --> C1[ベイズ最適化]
        C --> C2[進化的アルゴリズム]
        C --> C3[バンディットアルゴリズム]
    
        D --> D1[Hyperband]
        D --> D2[ASHA]
        D --> D3[PBT]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

### Early Stopping（早期停止）

学習中に性能が改善しない場合に学習を早期終了し、計算資源を節約します。
    
    
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score
    
    # LightGBMでEarly Stoppingの例
    print("\n=== Early Stopping デモ ===")
    
    # データセットの作成
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # パラメータ設定
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    
    # Early Stoppingあり
    print("\nEarly Stoppingあり:")
    model_es = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    print(f"実際の学習ラウンド数: {model_es.best_iteration}")
    
    # Early Stoppingなし
    print("\nEarly Stoppingなし:")
    model_no_es = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.log_evaluation(period=100)]
    )
    
    # 比較
    pred_es = (model_es.predict(X_test) > 0.5).astype(int)
    pred_no_es = (model_no_es.predict(X_test) > 0.5).astype(int)
    
    print(f"\n精度（Early Stopping）: {accuracy_score(y_test, pred_es):.4f}")
    print(f"精度（200ラウンド）: {accuracy_score(y_test, pred_no_es):.4f}")
    print(f"計算時間削減: {(1 - model_es.best_iteration/200)*100:.1f}%")
    

* * *

## 2.2 Optuna

### Optunaの特徴

**Optuna** は、次世代のハイパーパラメータ最適化フレームワークです。

特徴 | 説明 | 利点  
---|---|---  
**Define-by-run API** | 動的に探索空間を定義 | 柔軟で直感的なコード  
**Pruning** | 見込みのない試行を早期終了 | 大幅な時間短縮  
**並列化** | 複数の試行を同時実行 | 高速化  
**可視化** | 最適化過程の詳細な可視化 | 理解と診断が容易  
  
### Study and Trial

Optunaの基本概念：

  * **Study** : 最適化タスク全体を管理
  * **Trial** : 個々の試行（1つのハイパーパラメータ設定）

    
    
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    # データ準備
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, random_state=42)
    
    # 目的関数の定義
    def objective(trial):
        # ハイパーパラメータの提案
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42
        }
    
        # モデル学習と評価
        model = RandomForestClassifier(**params)
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
        return score
    
    # Studyの作成と最適化
    print("=== Optuna 基本例 ===")
    study = optuna.create_study(
        direction='maximize',  # 最大化
        study_name='random_forest_optimization'
    )
    
    # 最適化実行
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # 結果
    print(f"\n最適な精度: {study.best_value:.4f}")
    print(f"最適なパラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 統計情報
    print(f"\n総試行回数: {len(study.trials)}")
    print(f"完了した試行: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    

### Samplers（サンプラー）

Optunaは複数のサンプリング戦略をサポートしています。

#### TPE (Tree-structured Parzen Estimator)

デフォルトのサンプラーで、ベイズ最適化の一種です。
    
    
    from optuna.samplers import TPESampler
    
    # TPEサンプラー
    print("\n=== TPE Sampler ===")
    study_tpe = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    study_tpe.optimize(objective, n_trials=30)
    
    print(f"TPE最適精度: {study_tpe.best_value:.4f}")
    

#### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

進化戦略に基づくサンプラーです。
    
    
    from optuna.samplers import CmaEsSampler
    
    # CMA-ESサンプラー
    print("\n=== CMA-ES Sampler ===")
    study_cmaes = optuna.create_study(
        direction='maximize',
        sampler=CmaEsSampler(seed=42)
    )
    study_cmaes.optimize(objective, n_trials=30)
    
    print(f"CMA-ES最適精度: {study_cmaes.best_value:.4f}")
    

### Pruning Strategies（枝刈り戦略）

見込みのない試行を早期に打ち切ることで、計算時間を大幅に削減します。
    
    
    from optuna.pruners import MedianPruner
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    
    # データ準備
    X, y = make_classification(n_samples=5000, n_features=50,
                              n_informative=30, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Pruningを使う目的関数
    def objective_with_pruning(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
    
        # LightGBMデータセット
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
        # Pruningコールバック
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'binary_logloss')
    
        # 学習
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[pruning_callback, lgb.log_evaluation(period=0)]
        )
    
        # 評価
        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, (preds > 0.5).astype(int))
    
        return accuracy
    
    # Prunerを使ったStudy
    print("\n=== Pruning デモ ===")
    study_pruning = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study_pruning.optimize(objective_with_pruning, n_trials=30, timeout=60)
    
    # 統計
    n_complete = len([t for t in study_pruning.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study_pruning.trials if t.state == optuna.trial.TrialState.PRUNED])
    
    print(f"\n最適精度: {study_pruning.best_value:.4f}")
    print(f"完了した試行: {n_complete}")
    print(f"枝刈りされた試行: {n_pruned}")
    print(f"枝刈り率: {n_pruned/(n_complete+n_pruned)*100:.1f}%")
    

### Complete Optuna Example

Optunaを使った完全な最適化例です。
    
    
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate
    )
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingClassifier
    import warnings
    warnings.filterwarnings('ignore')
    
    # データ読み込み
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 目的関数
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }
    
        model = GradientBoostingClassifier(**params)
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
    
        return score
    
    # Studyの作成
    print("\n=== Complete Optuna Example ===")
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name='breast_cancer_optimization'
    )
    
    # 最適化実行
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    # 結果表示
    print(f"\n=== 最適化結果 ===")
    print(f"最適精度: {study.best_value:.4f}")
    print(f"\n最適パラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # パラメータ重要度
    print(f"\n=== パラメータ重要度 ===")
    importances = optuna.importance.get_param_importances(study)
    for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {importance:.4f}")
    
    # 可視化は実際の環境で表示されます
    # plot_optimization_history(study).show()
    # plot_param_importances(study).show()
    # plot_parallel_coordinate(study).show()
    
    print("\n✓ 最適化完了")
    

* * *

## 2.3 Hyperopt

### Tree-structured Parzen Estimator (TPE)

**Hyperopt** は、TPEアルゴリズムを用いたベイズ最適化フレームワークです。
    
    
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_classification
    import numpy as np
    
    # データ準備
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, random_state=42)
    
    # 探索空間の定義
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
        'max_depth': hp.quniform('max_depth', 3, 20, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'max_features': hp.uniform('max_features', 0.3, 1.0)
    }
    
    # 目的関数（Hyperoptは最小化するため、負の精度を返す）
    def objective(params):
        # 整数型に変換
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
    
        model = RandomForestClassifier(**params, random_state=42)
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
        return {'loss': -score, 'status': STATUS_OK}
    
    # 最適化実行
    print("=== Hyperopt TPE ===")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(42)
    )
    
    print(f"\n最適パラメータ: {best}")
    print(f"最適精度: {-min([trial['result']['loss'] for trial in trials.trials]):.4f}")
    
    # 最適化履歴
    losses = [trial['result']['loss'] for trial in trials.trials]
    print(f"\n試行回数: {len(trials.trials)}")
    print(f"最良スコアの推移:")
    best_so_far = []
    for i, loss in enumerate(losses):
        if i == 0:
            best_so_far.append(loss)
        else:
            best_so_far.append(min(best_so_far[-1], loss))
    print(f"  開始: {-best_so_far[0]:.4f}")
    print(f"  終了: {-best_so_far[-1]:.4f}")
    print(f"  改善: {(-best_so_far[-1] + best_so_far[0]):.4f}")
    

### Search Space Definition

Hyperoptは柔軟な探索空間定義をサポートします。
    
    
    from hyperopt import hp
    
    # 各種分布の定義
    search_space_detailed = {
        # 一様分布（連続値）
        'uniform_param': hp.uniform('uniform_param', 0.0, 1.0),
    
        # 一様分布（離散値）
        'quniform_param': hp.quniform('quniform_param', 10, 100, 5),  # 10, 15, 20, ...
    
        # 対数一様分布
        'loguniform_param': hp.loguniform('loguniform_param', np.log(0.001), np.log(1.0)),
    
        # 正規分布
        'normal_param': hp.normal('normal_param', 0, 1),
    
        # カテゴリカル
        'choice_param': hp.choice('choice_param', ['option1', 'option2', 'option3']),
    
        # 条件付き探索空間
        'classifier_type': hp.choice('classifier_type', [
            {
                'type': 'random_forest',
                'n_estimators': hp.quniform('rf_n_estimators', 50, 300, 1),
                'max_depth': hp.quniform('rf_max_depth', 3, 20, 1)
            },
            {
                'type': 'gradient_boosting',
                'n_estimators': hp.quniform('gb_n_estimators', 50, 300, 1),
                'learning_rate': hp.loguniform('gb_learning_rate', np.log(0.01), np.log(0.3))
            }
        ])
    }
    
    print("=== Hyperopt 探索空間の例 ===")
    for key, value in search_space_detailed.items():
        print(f"{key}: {value}")
    

### Trials Database

Trialsオブジェクトは、すべての試行履歴を保存します。
    
    
    from hyperopt import Trials
    import pandas as pd
    
    # Trials情報の詳細分析
    print("\n=== Trials 詳細分析 ===")
    
    # DataFrameに変換
    trials_df = pd.DataFrame([
        {
            'trial_id': i,
            'loss': trial['result']['loss'],
            **{k: v[0] if isinstance(v, (list, np.ndarray)) else v
               for k, v in trial['misc']['vals'].items() if v}
        }
        for i, trial in enumerate(trials.trials)
    ])
    
    print("\nTop 5 試行:")
    print(trials_df.nsmallest(5, 'loss')[['trial_id', 'loss', 'n_estimators', 'max_depth']])
    
    print("\nパラメータ統計:")
    print(trials_df.describe())
    

### Hyperopt Integration

Hyperoptと機械学習ライブラリの統合例です。
    
    
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # データ準備
    X, y = make_classification(n_samples=5000, n_features=50,
                              n_informative=30, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # LightGBM用の探索空間
    lgb_space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'num_leaves': hp.quniform('num_leaves', 20, 200, 1),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'min_child_samples': hp.quniform('min_child_samples', 5, 100, 1),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 10.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 10.0),
    }
    
    # 目的関数
    def lgb_objective(params):
        # 整数型に変換
        params['num_leaves'] = int(params['num_leaves'])
        params['max_depth'] = int(params['max_depth'])
        params['min_child_samples'] = int(params['min_child_samples'])
    
        # LightGBMパラメータ
        lgb_params = {
            **params,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
        }
    
        # データセット
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
        # 学習
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )
    
        # 評価
        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, (preds > 0.5).astype(int))
    
        return {'loss': -accuracy, 'status': STATUS_OK}
    
    # 最適化
    print("\n=== Hyperopt + LightGBM 統合 ===")
    lgb_trials = Trials()
    best_lgb = fmin(
        fn=lgb_objective,
        space=lgb_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=lgb_trials,
        rstate=np.random.default_rng(42)
    )
    
    print(f"\n最適精度: {-min([trial['result']['loss'] for trial in lgb_trials.trials]):.4f}")
    print(f"最適パラメータ:")
    for key, value in best_lgb.items():
        print(f"  {key}: {value}")
    

* * *

## 2.4 Ray Tune

### Tune API

**Ray Tune** は、分散ハイパーパラメータ最適化のためのライブラリです。
    
    
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings('ignore')
    
    # データ準備
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, random_state=42)
    
    # トレーニング関数
    def train_model(config):
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            random_state=42
        )
    
        score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    
        # Ray Tuneに結果を報告
        tune.report(accuracy=score)
    
    # 探索空間の定義
    config = {
        'n_estimators': tune.randint(50, 300),
        'max_depth': tune.randint(3, 20),
        'min_samples_split': tune.randint(2, 20)
    }
    
    # 最適化実行
    print("=== Ray Tune 基本例 ===")
    analysis = tune.run(
        train_model,
        config=config,
        num_samples=20,  # 試行回数
        resources_per_trial={'cpu': 1},
        verbose=1
    )
    
    # 結果
    best_config = analysis.get_best_config(metric='accuracy', mode='max')
    print(f"\n最適パラメータ: {best_config}")
    print(f"最適精度: {analysis.best_result['accuracy']:.4f}")
    

### Schedulers（スケジューラー）

#### ASHA (Async Successive Halving Algorithm)

ASHAは、性能の低い試行を早期に打ち切り、有望な試行にリソースを集中させます。
    
    
    from ray.tune.schedulers import ASHAScheduler
    from ray import tune
    import numpy as np
    
    # トレーニング関数（イテレーション対応）
    def train_with_iterations(config):
        # シミュレーション: イテレーションごとに性能が向上
        base_score = np.random.rand()
    
        for iteration in range(config['max_iterations']):
            # 学習曲線のシミュレーション
            score = base_score + (1 - base_score) * (1 - np.exp(-iteration / 20))
            score += np.random.randn() * 0.01  # ノイズ
    
            # 報告
            tune.report(accuracy=score, iteration=iteration)
    
    # ASHAスケジューラー
    asha_scheduler = ASHAScheduler(
        time_attr='iteration',
        metric='accuracy',
        mode='max',
        max_t=100,  # 最大イテレーション
        grace_period=10,  # 最低限実行するイテレーション
        reduction_factor=3  # 削減率
    )
    
    # 探索空間
    config_asha = {
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([16, 32, 64, 128]),
        'max_iterations': 100
    }
    
    print("\n=== ASHA Scheduler ===")
    analysis_asha = tune.run(
        train_with_iterations,
        config=config_asha,
        num_samples=30,
        scheduler=asha_scheduler,
        resources_per_trial={'cpu': 1},
        verbose=1
    )
    
    print(f"\n最適精度: {analysis_asha.best_result['accuracy']:.4f}")
    print(f"完了した試行数: {len(analysis_asha.trials)}")
    

#### PBT (Population Based Training)

PBTは、集団ベースの進化的アプローチでハイパーパラメータを動的に調整します。
    
    
    from ray.tune.schedulers import PopulationBasedTraining
    
    # PBTスケジューラー
    pbt_scheduler = PopulationBasedTraining(
        time_attr='iteration',
        metric='accuracy',
        mode='max',
        perturbation_interval=5,  # 摂動間隔
        hyperparam_mutations={
            'learning_rate': lambda: np.random.uniform(1e-4, 1e-1),
            'batch_size': [16, 32, 64, 128]
        }
    )
    
    print("\n=== PBT Scheduler ===")
    print("PBTは動的にハイパーパラメータを調整します")
    print("- 性能の良いモデルの設定を他のモデルにコピー")
    print("- ハイパーパラメータに小さな変動を加える")
    print("- 集団全体で最適化を進める")
    

### Integration with PyTorch/TensorFlow

Ray TuneはPyTorchやTensorFlowとシームレスに統合できます。
    
    
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # シンプルなニューラルネットワーク
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # データ準備
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # トレーニング関数
    def train_pytorch(config):
        # モデル構築
        model = SimpleNet(
            input_size=20,
            hidden_size=config['hidden_size'],
            output_size=2
        )
    
        # オプティマイザ
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.CrossEntropyLoss()
    
        # DataLoader
        train_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
    
        # トレーニングループ
        for epoch in range(10):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
    
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
            accuracy = correct / total
            tune.report(accuracy=accuracy, loss=total_loss/len(train_loader))
    
    # 探索空間
    pytorch_config = {
        'hidden_size': tune.choice([32, 64, 128, 256]),
        'lr': tune.loguniform(1e-4, 1e-2),
        'batch_size': tune.choice([16, 32, 64])
    }
    
    print("\n=== Ray Tune + PyTorch 統合 ===")
    analysis_pytorch = tune.run(
        train_pytorch,
        config=pytorch_config,
        num_samples=10,
        resources_per_trial={'cpu': 1},
        verbose=1
    )
    
    print(f"\n最適精度: {analysis_pytorch.best_result['accuracy']:.4f}")
    print(f"最適設定: {analysis_pytorch.get_best_config(metric='accuracy', mode='max')}")
    

### Distributed HPO

Ray Tuneは自動的に複数のCPU/GPUに処理を分散します。
    
    
    import ray
    
    # Ray初期化（複数CPUを使用）
    ray.init(num_cpus=4, ignore_reinit_error=True)
    
    # 分散実行の設定
    distributed_config = {
        'n_estimators': tune.randint(50, 300),
        'max_depth': tune.randint(3, 20),
        'min_samples_split': tune.randint(2, 20)
    }
    
    print("\n=== 分散ハイパーパラメータ最適化 ===")
    print("4つのCPUコアで並列実行")
    
    # 並列実行
    analysis_distributed = tune.run(
        train_model,
        config=distributed_config,
        num_samples=40,
        resources_per_trial={'cpu': 1},  # 1試行あたり1CPU
        verbose=1
    )
    
    print(f"\n最適精度: {analysis_distributed.best_result['accuracy']:.4f}")
    print(f"総試行数: {len(analysis_distributed.trials)}")
    
    # クリーンアップ
    ray.shutdown()
    

* * *

## 2.5 高度なHPO手法

### Bayesian Optimization（ベイズ最適化）

ベイズ最適化は、過去の試行結果を活用して次の探索点を選択します。
    
    
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from scipy.stats import norm
    import numpy as np
    
    class BayesianOptimizer:
        def __init__(self, bounds, n_init=5):
            self.bounds = bounds
            self.n_init = n_init
            self.X_obs = []
            self.y_obs = []
            self.gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42
            )
    
        def acquisition_function(self, X, xi=0.01):
            """Expected Improvement (EI)"""
            mu, sigma = self.gp.predict(X, return_std=True)
    
            if len(self.y_obs) == 0:
                return np.zeros_like(mu)
    
            mu_best = np.max(self.y_obs)
    
            with np.errstate(divide='warn'):
                imp = mu - mu_best - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
    
            return ei
    
        def propose_location(self):
            """次の探索点を提案"""
            if len(self.X_obs) < self.n_init:
                # ランダムサンプリング
                return np.random.uniform(self.bounds[0], self.bounds[1])
    
            # Acquisition Functionを最大化
            X_random = np.random.uniform(
                self.bounds[0], self.bounds[1], size=(1000, 1)
            )
            ei = self.acquisition_function(X_random)
            return X_random[np.argmax(ei)]
    
        def observe(self, X, y):
            """観測結果を記録"""
            self.X_obs.append(X)
            self.y_obs.append(y)
    
            if len(self.X_obs) >= self.n_init:
                self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))
    
    # テスト関数（最適化対象）
    def test_function(x):
        """1次元のテスト関数"""
        return -(x - 2) ** 2 + 5 + np.random.randn() * 0.1
    
    # ベイズ最適化実行
    print("=== Bayesian Optimization デモ ===")
    optimizer = BayesianOptimizer(bounds=(0, 5), n_init=3)
    
    for i in range(20):
        # 次の探索点を提案
        x_next = optimizer.propose_location()
    
        # 評価
        y_next = test_function(x_next[0])
    
        # 観測を記録
        optimizer.observe(x_next, y_next)
    
        if i % 5 == 0:
            print(f"Iteration {i}: x={x_next[0]:.3f}, y={y_next:.3f}")
    
    # 結果
    best_idx = np.argmax(optimizer.y_obs)
    print(f"\n最適解: x={optimizer.X_obs[best_idx][0]:.3f}, y={optimizer.y_obs[best_idx]:.3f}")
    print(f"真の最適値: x=2.0, y=5.0")
    

### Population-based Training (PBT)

PBTは、複数のモデルを同時に学習させ、良い設定を共有します。
    
    
    import numpy as np
    from copy import deepcopy
    
    class PBTOptimizer:
        def __init__(self, population_size=10, perturbation_factor=0.2):
            self.population_size = population_size
            self.perturbation_factor = perturbation_factor
            self.population = []
    
        def initialize_population(self, param_ranges):
            """集団の初期化"""
            for _ in range(self.population_size):
                individual = {
                    'params': {
                        key: np.random.uniform(low, high)
                        for key, (low, high) in param_ranges.items()
                    },
                    'score': 0.0,
                    'history': []
                }
                self.population.append(individual)
    
        def exploit_and_explore(self, param_ranges):
            """Exploit（良い設定をコピー）とExplore（摂動）"""
            # 性能でソート
            self.population.sort(key=lambda x: x['score'], reverse=True)
    
            # 下位20%を上位からコピー
            cutoff = int(0.2 * self.population_size)
            for i in range(self.population_size - cutoff, self.population_size):
                # 上位からランダムに選択してコピー
                source = np.random.randint(0, cutoff)
                self.population[i]['params'] = deepcopy(
                    self.population[source]['params']
                )
    
                # パラメータに摂動を加える（Explore）
                for key in self.population[i]['params']:
                    low, high = param_ranges[key]
                    current = self.population[i]['params'][key]
    
                    # ランダムに増減
                    factor = 1 + np.random.uniform(
                        -self.perturbation_factor,
                        self.perturbation_factor
                    )
                    new_value = current * factor
    
                    # 範囲内にクリップ
                    self.population[i]['params'][key] = np.clip(
                        new_value, low, high
                    )
    
        def step(self, eval_fn, param_ranges):
            """1ステップ実行"""
            # 各個体を評価
            for individual in self.population:
                score = eval_fn(individual['params'])
                individual['score'] = score
                individual['history'].append(score)
    
            # Exploit & Explore
            self.exploit_and_explore(param_ranges)
    
    # 評価関数（シミュレーション）
    def evaluate_params(params):
        """パラメータを評価（シミュレーション）"""
        # 最適値: learning_rate=0.1, batch_size=32
        lr_score = 1 - abs(params['learning_rate'] - 0.1)
        bs_score = 1 - abs(params['batch_size'] - 32) / 64
        return (lr_score + bs_score) / 2 + np.random.randn() * 0.05
    
    # PBT実行
    print("\n=== Population-based Training デモ ===")
    pbt = PBTOptimizer(population_size=10)
    param_ranges = {
        'learning_rate': (0.001, 0.3),
        'batch_size': (16, 128)
    }
    
    pbt.initialize_population(param_ranges)
    
    for step in range(20):
        pbt.step(evaluate_params, param_ranges)
    
        if step % 5 == 0:
            best = max(pbt.population, key=lambda x: x['score'])
            print(f"Step {step}: Best score={best['score']:.3f}, params={best['params']}")
    
    # 最終結果
    best_individual = max(pbt.population, key=lambda x: x['score'])
    print(f"\n最適パラメータ: {best_individual['params']}")
    print(f"最適スコア: {best_individual['score']:.3f}")
    

### Hyperband

Hyperbandは、様々な予算（イテレーション数）で多数の設定を試す手法です。
    
    
    import numpy as np
    import math
    
    class HyperbandOptimizer:
        def __init__(self, max_iter=81, eta=3):
            self.max_iter = max_iter
            self.eta = eta
            self.logeta = lambda x: math.log(x) / math.log(self.eta)
            self.s_max = int(self.logeta(self.max_iter))
            self.B = (self.s_max + 1) * self.max_iter
    
        def run(self, get_config_fn, eval_fn):
            """Hyperband実行"""
            results = []
    
            for s in reversed(range(self.s_max + 1)):
                n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
                r = self.max_iter * self.eta ** (-s)
    
                print(f"\nBracket s={s}: n={n} configs, r={r:.1f} iterations")
    
                # n個の設定を生成
                configs = [get_config_fn() for _ in range(n)]
    
                # Successive Halving
                for i in range(s + 1):
                    n_i = n * self.eta ** (-i)
                    r_i = r * self.eta ** i
    
                    print(f"  Round {i}: {int(n_i)} configs, {int(r_i)} iterations each")
    
                    # 評価
                    scores = [eval_fn(config, int(r_i)) for config in configs]
    
                    # 結果を記録
                    for config, score in zip(configs, scores):
                        results.append({
                            'config': config,
                            'score': score,
                            'iterations': int(r_i)
                        })
    
                    # 上位を選択
                    if i < s:
                        indices = np.argsort(scores)[-int(n_i / self.eta):]
                        configs = [configs[i] for i in indices]
    
            return results
    
    # 設定生成関数
    def get_random_config():
        return {
            'learning_rate': np.random.uniform(0.001, 0.3),
            'batch_size': np.random.choice([16, 32, 64, 128])
        }
    
    # 評価関数（イテレーション数に依存）
    def evaluate_config(config, iterations):
        # 最適値からの距離で性能を計算
        lr_score = 1 - abs(config['learning_rate'] - 0.1)
        bs_score = 1 - abs(config['batch_size'] - 32) / 64
        base_score = (lr_score + bs_score) / 2
    
        # イテレーション数が多いほど性能が向上（学習曲線）
        improvement = 1 - np.exp(-iterations / 20)
    
        return base_score * improvement + np.random.randn() * 0.01
    
    # Hyperband実行
    print("=== Hyperband デモ ===")
    hyperband = HyperbandOptimizer(max_iter=81, eta=3)
    results = hyperband.run(get_random_config, evaluate_config)
    
    # 最良の結果
    best_result = max(results, key=lambda x: x['score'])
    print(f"\n=== 最適結果 ===")
    print(f"設定: {best_result['config']}")
    print(f"スコア: {best_result['score']:.4f}")
    print(f"イテレーション数: {best_result['iterations']}")
    print(f"\n総評価回数: {len(results)}")
    

### Multi-fidelity Optimization

Multi-fidelity最適化は、低コストの近似評価を活用します。
    
    
    import numpy as np
    
    class MultiFidelityOptimizer:
        def __init__(self, fidelity_levels=[0.1, 0.3, 0.5, 1.0]):
            self.fidelity_levels = fidelity_levels
            self.evaluations = {level: [] for level in fidelity_levels}
    
        def evaluate_at_fidelity(self, config, fidelity, true_fn):
            """指定のfidelityで評価"""
            # 低fidelityは計算が速いが精度が低い
            # 高fidelityは計算が遅いが精度が高い
    
            true_score = true_fn(config)
            noise = (1 - fidelity) * 0.2  # 低fidelityほどノイズが大きい
    
            observed_score = true_score + np.random.randn() * noise
    
            return observed_score
    
        def optimize(self, param_ranges, true_fn, n_total_evals=100):
            """Multi-fidelity最適化"""
            # 予算配分: 低fidelityで多数、高fidelityで少数
            eval_counts = {
                0.1: int(0.5 * n_total_evals),
                0.3: int(0.3 * n_total_evals),
                0.5: int(0.15 * n_total_evals),
                1.0: int(0.05 * n_total_evals)
            }
    
            all_configs = []
    
            # 各fidelityレベルで評価
            for fidelity in self.fidelity_levels:
                n_evals = eval_counts[fidelity]
    
                if fidelity == self.fidelity_levels[0]:
                    # 最低fidelity: ランダムサンプリング
                    configs = [
                        {key: np.random.uniform(low, high)
                         for key, (low, high) in param_ranges.items()}
                        for _ in range(n_evals)
                    ]
                else:
                    # 前のfidelityの上位を次のfidelityで評価
                    prev_results = sorted(
                        self.evaluations[self.fidelity_levels[self.fidelity_levels.index(fidelity) - 1]],
                        key=lambda x: x['score'],
                        reverse=True
                    )
                    configs = [r['config'] for r in prev_results[:n_evals]]
    
                # 評価
                for config in configs:
                    score = self.evaluate_at_fidelity(config, fidelity, true_fn)
                    self.evaluations[fidelity].append({
                        'config': config,
                        'score': score,
                        'fidelity': fidelity
                    })
                    all_configs.append((config, score, fidelity))
    
            # 最高fidelityでの最良結果を返す
            best = max(
                self.evaluations[1.0],
                key=lambda x: x['score']
            )
    
            return best, all_configs
    
    # 真の目的関数
    def true_objective(config):
        lr_score = 1 - abs(config['learning_rate'] - 0.1)
        bs_score = 1 - abs(config['batch_size'] - 32) / 64
        return (lr_score + bs_score) / 2
    
    # Multi-fidelity最適化実行
    print("\n=== Multi-fidelity Optimization デモ ===")
    mf_optimizer = MultiFidelityOptimizer()
    param_ranges = {
        'learning_rate': (0.001, 0.3),
        'batch_size': (16, 128)
    }
    
    best, all_evals = mf_optimizer.optimize(param_ranges, true_objective, n_total_evals=100)
    
    print(f"\n最適設定: {best['config']}")
    print(f"最適スコア: {best['score']:.4f}")
    print(f"\nFidelityレベル別の評価数:")
    for fidelity in mf_optimizer.fidelity_levels:
        print(f"  Fidelity {fidelity}: {len(mf_optimizer.evaluations[fidelity])}回")
    

* * *

## 2.6 本章のまとめ

### 学んだこと

  1. **HPOの基礎**

     * 探索空間の定義と分布の選択
     * Grid SearchとRandom Searchの比較
     * Early Stoppingによる効率化
  2. **Optuna**

     * Define-by-run APIによる柔軟な実装
     * TPE、CMA-ESなどの高度なサンプラー
     * Pruningによる大幅な時間短縮
     * 可視化と診断機能
  3. **Hyperopt**

     * TPEベースのベイズ最適化
     * 柔軟な探索空間定義
     * Trialsによる詳細な履歴管理
  4. **Ray Tune**

     * 分散ハイパーパラメータ最適化
     * ASHA、PBTなどの高度なスケジューラー
     * PyTorch/TensorFlowとの統合
     * スケーラブルな並列実行
  5. **高度なHPO手法**

     * Bayesian Optimization: 過去の試行を活用
     * Population-based Training: 動的な設定調整
     * Hyperband: 多様な予算での探索
     * Multi-fidelity: 低コスト評価の活用

### HPOフレームワークの選択ガイド

フレームワーク | 最適な用途 | 長所 | 短所  
---|---|---|---  
**Optuna** | 汎用的なHPO、研究 | 柔軟、高機能、可視化 | 分散は限定的  
**Hyperopt** | 中規模HPO、複雑な探索空間 | 成熟、安定 | やや古い設計  
**Ray Tune** | 大規模分散HPO、DL | スケーラブル、統合 | 設定が複雑  
**scikit-learn** | シンプルなHPO | 簡単、標準的 | 機能が限定的  
  
### HPO戦略の選択基準

状況 | 推奨戦略 | 理由  
---|---|---  
小規模探索（<10パラメータ） | Grid Search | 網羅的で理解しやすい  
中規模探索（10-20パラメータ） | Random Search, TPE | 効率的で実用的  
大規模探索（>20パラメータ） | Bayesian Opt, ASHA | 高次元でも効率的  
高価な評価関数 | Bayesian Opt | 少ない試行で最適化  
安価な評価関数 | Random Search, Hyperband | 多数の試行が可能  
分散環境あり | Ray Tune + ASHA/PBT | 並列化で高速化  
  
### 次の章へ

第3章では、**Neural Architecture Search（NAS）** を学びます：

  * NASの基礎と動機
  * 検索空間の設計
  * DARTS、ENAS、NASNet
  * 効率的なNAS手法
  * 実践的なNAS実装

* * *

## 演習問題

### 問題1（難易度：easy）

Grid SearchとRandom Searchの主な違いを説明し、それぞれの長所と短所を述べてください。

解答例

**解答** ：

**Grid Search** ：

  * 探索方法: すべての組み合わせを網羅的に試す
  * 長所: 探索空間を完全に調査、再現性が高い
  * 短所: パラメータ数が増えると組み合わせ爆発、計算コストが高い

**Random Search** ：

  * 探索方法: パラメータをランダムにサンプリング
  * 長所: 高次元でも効率的、重要なパラメータを見つけやすい
  * 短所: 最適解の保証なし、試行回数の決定が難しい

**使い分け** ：

状況 | 推奨  
---|---  
パラメータ数が少ない（<5個） | Grid Search  
パラメータ数が多い（>5個） | Random Search  
計算資源が豊富 | Grid Search  
計算資源が限定的 | Random Search  
  
### 問題2（難易度：medium）

Optunaを使って、LightGBMのハイパーパラメータ最適化を実装してください。Pruningを有効にし、少なくとも5つのハイパーパラメータを最適化してください。

解答例
    
    
    import optuna
    from optuna.pruners import MedianPruner
    import lightgbm as lgb
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # データ準備
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 目的関数
    def objective(trial):
        # ハイパーパラメータの提案
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        }
    
        # データセット
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
        # Pruningコールバック
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, 'binary_logloss'
        )
    
        # 学習
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[pruning_callback, lgb.log_evaluation(period=0)]
        )
    
        # 評価
        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, (preds > 0.5).astype(int))
    
        return accuracy
    
    # Study作成
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # 最適化実行
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # 結果
    print(f"\n=== 最適化結果 ===")
    print(f"最適精度: {study.best_value:.4f}")
    print(f"\n最適パラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 統計
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\n完了: {n_complete}, 枝刈り: {n_pruned}, 枝刈り率: {n_pruned/(n_complete+n_pruned)*100:.1f}%")
    

### 問題3（難易度：medium）

ベイズ最適化における「Acquisition Function」の役割を説明してください。また、Expected Improvement (EI)とUpper Confidence Bound (UCB)の違いを述べてください。

解答例

**解答** ：

**Acquisition Functionの役割** ：

  * 目的: 次に評価すべき点を決定する
  * 機能: 探索（Exploration）と活用（Exploitation）のバランスを取る
  * 入力: 現在のガウス過程モデル（予測平均と分散）
  * 出力: 各候補点の「有用性」スコア

**Expected Improvement (EI)** ：

  * 定義: 現在の最良値からの改善の期待値
  * 数式: $EI(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)]$
  * 特徴: 確実な改善を重視、保守的
  * 長所: 安定した収束、理論的保証
  * 短所: 局所最適に陥りやすい

**Upper Confidence Bound (UCB)** ：

  * 定義: 予測平均 + 不確実性ボーナス
  * 数式: $UCB(x) = \mu(x) + \kappa \sigma(x)$
  * 特徴: 不確実性の高い領域を探索
  * 長所: 探索を促進、大域的最適化
  * 短所: パラメータ$\kappa$の調整が必要

**使い分け** ：

状況 | 推奨  
---|---  
評価が高価、確実性重視 | EI  
探索を重視、大域的最適化 | UCB  
ノイズが多い | EI  
滑らかな目的関数 | どちらでも可  
  
### 問題4（難易度：hard）

ASHAスケジューラーの仕組みを説明し、通常のRandom Searchと比較して、なぜ効率的なのかを述べてください。

解答例

**解答** ：

**ASHAの仕組み** ：

  1. **基本アイデア**

     * 多数の設定を少ないリソース（イテレーション）で試す
     * 性能の悪い設定を早期に打ち切る
     * 有望な設定にリソースを集中
  2. **アルゴリズム**

     * Rung（段階）を設定: 例えば[10, 30, 90, 270]イテレーション
     * 各Rungで上位1/η（例: η=3なら上位1/3）のみ次へ
     * 最終的に少数の設定のみ最大イテレーションまで実行
  3. **非同期実行**

     * 各試行が独立して進行
     * Rungに到達したら昇格判定
     * リソースの効率的な活用

**Random Searchとの比較** ：

側面 | Random Search | ASHA  
---|---|---  
リソース配分 | 全試行に均等 | 有望な試行に集中  
早期停止 | なし | あり（性能不良を打ち切り）  
並列化 | 簡単 | 非同期で効率的  
総計算時間 | N × max_iter | ≈ N × min_iter + 少数 × max_iter  
  
**効率性の理由** ：

  1. **無駄な計算の削減**

     * 明らかに悪い設定を早期に打ち切る
     * 有望な設定のみフル学習
  2. **探索と活用のバランス**

     * 多様な設定を試す（探索）
     * 良い設定に注力（活用）
  3. **理論的保証**

     * 最適設定を見逃す確率が低い
     * 計算量が対数的に増加

**実例** ：
    
    
    Random Search: 81試行 × 100イテレーション = 8,100計算単位
    
    ASHA (η=3):
    - Rung 0: 81試行 × 1イテレーション = 81
    - Rung 1: 27試行 × 3イテレーション = 81
    - Rung 2: 9試行 × 9イテレーション = 81
    - Rung 3: 3試行 × 27イテレーション = 81
    - Rung 4: 1試行 × 81イテレーション = 81
    合計: 405計算単位
    
    削減率: (8,100 - 405) / 8,100 = 95%
    

### 問題5（難易度：hard）

Population-based Training (PBT)とベイズ最適化の主な違いを説明し、それぞれが適している状況を述べてください。

解答例

**解答** ：

**PBTの特徴** ：

  * 複数のモデルを同時に学習
  * 学習中に動的にハイパーパラメータを調整
  * 良い設定を他のモデルにコピー（Exploit）
  * コピーした設定に摂動を加える（Explore）
  * 進化的アプローチ（遺伝的アルゴリズムに類似）

**ベイズ最適化の特徴** ：

  * 1つずつモデルを順次学習
  * ハイパーパラメータは学習前に固定
  * 過去の試行履歴からガウス過程でモデル化
  * 次の最適な探索点を確率的に選択
  * 理論的な最適化アプローチ

**主な違い** ：

側面 | PBT | ベイズ最適化  
---|---|---  
**並列性** | 高い（集団全体を同時学習） | 限定的（順次実行が基本）  
**動的調整** | あり（学習中に変更） | なし（学習前に固定）  
**計算効率** | 高い（並列実行） | 中程度（試行回数は少ない）  
**理論的保証** | 弱い（ヒューリスティック） | 強い（収束保証あり）  
**実装複雑度** | 高い（集団管理が必要） | 中程度（既存実装利用可）  
**適用範囲** | 長時間学習（DL） | 高価な評価関数  
  
**適している状況** ：

**PBTを使うべき場合** ：

  * ディープラーニングなど長時間の学習
  * 学習率スケジュールなど動的な調整が重要
  * 並列計算リソースが豊富
  * 探索空間が広い
  * 例: 強化学習、大規模ニューラルネット学習

**ベイズ最適化を使うべき場合** ：

  * 1回の評価が非常に高価
  * 評価関数が滑らか
  * 少ない試行回数で最適化したい
  * 理論的な保証が必要
  * 例: シミュレーション最適化、高価な実験設定

**ハイブリッドアプローチ** ：

  * ベイズ最適化で初期設定を探索
  * 良い設定の周辺をPBTで細かく調整
  * 両方の利点を活用

* * *

## 参考文献

  1. Bergstra, J., & Bengio, Y. (2012). _Random search for hyper-parameter optimization_. Journal of Machine Learning Research, 13(1), 281-305.
  2. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). _Optuna: A next-generation hyperparameter optimization framework_. KDD.
  3. Bergstra, J., Yamins, D., & Cox, D. (2013). _Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures_. ICML.
  4. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). _Hyperband: A novel bandit-based approach to hyperparameter optimization_. JMLR.
  5. Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., Razavi, A., ... & Kavukcuoglu, K. (2017). _Population based training of neural networks_. arXiv preprint arXiv:1711.09846.
  6. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2015). _Taking the human out of the loop: A review of Bayesian optimization_. Proceedings of the IEEE, 104(1), 148-175.
  7. Liaw, R., Liang, E., Nishihara, R., Moritz, P., Gonzalez, J. E., & Stoica, I. (2018). _Tune: A research platform for distributed model selection and training_. arXiv preprint arXiv:1807.05118.
