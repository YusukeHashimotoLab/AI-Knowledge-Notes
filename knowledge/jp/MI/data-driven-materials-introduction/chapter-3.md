---
title: Chapter
chapter_title: Chapter
subtitle: 
reading_time: 20-25分
difficulty: 初級
code_examples: 0
exercises: 0
---

# Chapter 3: モデル選択とハイパーパラメータ最適化

* * *

## 学習目標

この章を読むことで、以下を習得できます：

✅ データサイズに応じた適切なモデル選択（線形、木ベース、NN、GNN） ✅ 交差検証（K-Fold、Stratified、Time Series Split）の実践 ✅ Optunaによるベイズ最適化を用いたハイパーパラメータ自動最適化 ✅ アンサンブル学習（Bagging、Boosting、Stacking）の実装 ✅ Li-ion電池容量予測における実践的ワークフロー

* * *

## 3.1 モデル選択の戦略

材料科学における機械学習では、データの特性に応じた適切なモデル選択が重要です。

### データサイズとモデル複雑度
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import learning_curve
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    
    # サンプルデータ生成
    np.random.seed(42)
    
    def generate_material_data(n_samples, n_features=20):
        """材料データシミュレーション"""
        X = np.random.randn(n_samples, n_features)
        # 非線形関係
        y = (
            2 * X[:, 0]**2 +
            3 * X[:, 1] * X[:, 2] -
            1.5 * X[:, 3] +
            np.random.normal(0, 0.5, n_samples)
        )
        return X, y
    
    # モデル複雑度とサンプルサイズの関係
    sample_sizes = [50, 100, 200, 500, 1000]
    models = {
        'Ridge': Ridge(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
        'Neural Network': MLPRegressor(hidden_layers=(50, 50), max_iter=1000, random_state=42)
    }
    
    # 学習曲線
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (model_name, model) in enumerate(models.items()):
        X, y = generate_material_data(1000, n_features=20)
    
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
    
        train_mean = -train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = -val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
    
        axes[idx].plot(train_sizes, train_mean, 'o-',
                       color='steelblue', label='Training Error')
        axes[idx].fill_between(train_sizes,
                               train_mean - train_std,
                               train_mean + train_std,
                               alpha=0.2, color='steelblue')
    
        axes[idx].plot(train_sizes, val_mean, 'o-',
                       color='coral', label='Validation Error')
        axes[idx].fill_between(train_sizes,
                               val_mean - val_std,
                               val_mean + val_std,
                               alpha=0.2, color='coral')
    
        axes[idx].set_xlabel('Training Size', fontsize=11)
        axes[idx].set_ylabel('MSE', fontsize=11)
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("モデル選択ガイドライン：")
    print("- 小データ (<100): Ridge, Lasso（正則化線形モデル）")
    print("- 中データ (100-1000): Random Forest, Gradient Boosting")
    print("- 大データ (>1000): Neural Network, Deep Learning")
    

### 解釈性 vs 精度のトレードオフ
    
    
    # モデルの解釈性と精度の比較
    model_comparison = pd.DataFrame({
        'モデル': [
            'Linear Regression',
            'Ridge/Lasso',
            'Decision Tree',
            'Random Forest',
            'Gradient Boosting',
            'Neural Network',
            'GNN'
        ],
        '解釈性': [10, 9, 7, 4, 3, 2, 1],
        '精度': [4, 5, 5, 8, 9, 9, 10],
        '訓練速度': [10, 9, 8, 6, 5, 3, 2],
        '推論速度': [10, 10, 9, 7, 6, 8, 4]
    })
    
    # レーダーチャート
    from math import pi
    
    categories = ['解釈性', '精度', '訓練速度', '推論速度']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10),
                             subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    for idx, row in model_comparison.iterrows():
        values = row[categories].tolist()
        values += values[:1]
    
        axes[idx].plot(angles, values, 'o-', linewidth=2)
        axes[idx].fill(angles, values, alpha=0.25)
        axes[idx].set_xticks(angles[:-1])
        axes[idx].set_xticklabels(categories, size=9)
        axes[idx].set_ylim(0, 10)
        axes[idx].set_title(row['モデル'], size=11, fontweight='bold', pad=20)
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nモデル選択の判断基準：")
    print("- 解釈性重視: Ridge, Lasso, Decision Tree")
    print("- 精度重視: Gradient Boosting, Neural Network, GNN")
    print("- バランス型: Random Forest")
    

### 線形モデル、木ベース、NN、GNNの使い分け
    
    
    # 実データでの性能比較
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    
    X, y = generate_material_data(500, n_features=20)
    
    models_benchmark = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'MLP': MLPRegressor(hidden_layers=(100, 50), max_iter=1000, random_state=42)
    }
    
    results = []
    
    for model_name, model in models_benchmark.items():
        # 交差検証
        cv_scores = cross_val_score(model, X, y, cv=5,
                                    scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        mae_std = cv_scores.std()
    
        # R²
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
        r2 = cv_r2.mean()
    
        results.append({
            'Model': model_name,
            'MAE': mae,
            'MAE_std': mae_std,
            'R²': r2
        })
    
    results_df = pd.DataFrame(results)
    print("\nモデル性能比較：")
    print(results_df.to_string(index=False))
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE
    axes[0].barh(results_df['Model'], results_df['MAE'],
                 xerr=results_df['MAE_std'],
                 color='steelblue', alpha=0.7)
    axes[0].set_xlabel('MAE (lower is better)', fontsize=11)
    axes[0].set_title('予測誤差（MAE）', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # R²
    axes[1].barh(results_df['Model'], results_df['R²'],
                 color='coral', alpha=0.7)
    axes[1].set_xlabel('R² (higher is better)', fontsize=11)
    axes[1].set_title('決定係数（R²）', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.2 交差検証（Cross-Validation）

モデルの汎化性能を適切に評価するための手法です。

### K-Fold CV
    
    
    from sklearn.model_selection import KFold, cross_validate
    
    def kfold_cv_demo(X, y, model, k=5):
        """
        K-Fold交差検証のデモ
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
        fold_results = []
    
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
    
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
    
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
    
            fold_results.append({
                'Fold': fold_idx + 1,
                'MAE': mae,
                'R²': r2
            })
    
        return pd.DataFrame(fold_results)
    
    # K-Fold実行
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    fold_results = kfold_cv_demo(X, y, model, k=5)
    
    print("K-Fold CV結果：")
    print(fold_results.to_string(index=False))
    print(f"\n平均MAE: {fold_results['MAE'].mean():.4f} ± {fold_results['MAE'].std():.4f}")
    print(f"平均R²: {fold_results['R²'].mean():.4f} ± {fold_results['R²'].std():.4f}")
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(fold_results))
    
    ax.bar(x_pos, fold_results['MAE'], color='steelblue', alpha=0.7,
           label='MAE per fold')
    ax.axhline(y=fold_results['MAE'].mean(), color='red',
               linestyle='--', linewidth=2, label='Mean MAE')
    ax.fill_between(x_pos,
                    fold_results['MAE'].mean() - fold_results['MAE'].std(),
                    fold_results['MAE'].mean() + fold_results['MAE'].std(),
                    color='red', alpha=0.2, label='±1 Std')
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('K-Fold交差検証結果', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fold_results['Fold'])
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Stratified K-Fold
    
    
    from sklearn.model_selection import StratifiedKFold
    
    # 分類問題用データ
    X_class, _ = generate_material_data(500, n_features=20)
    # 3クラス分類
    y_class = np.digitize(y, bins=np.percentile(y, [33, 67]))
    
    def compare_kfold_strategies(X, y):
        """
        通常K-Fold vs Stratified K-Fold
        """
        # 通常K-Fold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        normal_distributions = []
    
        for train_idx, _ in kf.split(X):
            y_train = y[train_idx]
            class_dist = np.bincount(y_train) / len(y_train)
            normal_distributions.append(class_dist)
    
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        stratified_distributions = []
    
        for train_idx, _ in skf.split(X, y):
            y_train = y[train_idx]
            class_dist = np.bincount(y_train) / len(y_train)
            stratified_distributions.append(class_dist)
    
        return normal_distributions, stratified_distributions
    
    normal_dist, stratified_dist = compare_kfold_strategies(X_class, y_class)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 通常K-Fold
    normal_array = np.array(normal_dist)
    axes[0].bar(range(len(normal_array)), normal_array[:, 0],
                label='Class 0', alpha=0.7)
    axes[0].bar(range(len(normal_array)), normal_array[:, 1],
                bottom=normal_array[:, 0],
                label='Class 1', alpha=0.7)
    axes[0].bar(range(len(normal_array)), normal_array[:, 2],
                bottom=normal_array[:, 0] + normal_array[:, 1],
                label='Class 2', alpha=0.7)
    axes[0].set_xlabel('Fold', fontsize=11)
    axes[0].set_ylabel('Class Distribution', fontsize=11)
    axes[0].set_title('通常K-Fold', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    # Stratified K-Fold
    stratified_array = np.array(stratified_dist)
    axes[1].bar(range(len(stratified_array)), stratified_array[:, 0],
                label='Class 0', alpha=0.7)
    axes[1].bar(range(len(stratified_array)), stratified_array[:, 1],
                bottom=stratified_array[:, 0],
                label='Class 1', alpha=0.7)
    axes[1].bar(range(len(stratified_array)), stratified_array[:, 2],
                bottom=stratified_array[:, 0] + stratified_array[:, 1],
                label='Class 2', alpha=0.7)
    axes[1].set_xlabel('Fold', fontsize=11)
    axes[1].set_ylabel('Class Distribution', fontsize=11)
    axes[1].set_title('Stratified K-Fold', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print("Stratified K-Foldの利点：")
    print("- 各Foldでクラス分布が均一")
    print("- 不均衡データでも安定した評価")
    

### Time Series Split（逐次データ用）
    
    
    from sklearn.model_selection import TimeSeriesSplit
    
    # 時系列データシミュレーション
    n_time_points = 200
    time = np.arange(n_time_points)
    # トレンド + 季節性 + ノイズ
    y_timeseries = (
        0.05 * time +
        10 * np.sin(2 * np.pi * time / 50) +
        np.random.normal(0, 2, n_time_points)
    )
    X_timeseries = np.column_stack([time, np.sin(2 * np.pi * time / 50)])
    
    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_timeseries)):
        # Train
        ax.scatter(time[train_idx], y_timeseries[train_idx],
                   c=[colors[fold_idx]], s=20, alpha=0.3,
                   label=f'Fold {fold_idx+1} Train')
        # Test
        ax.scatter(time[test_idx], y_timeseries[test_idx],
                   c=[colors[fold_idx]], s=50, marker='s',
                   label=f'Fold {fold_idx+1} Test')
    
    ax.plot(time, y_timeseries, 'k-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Time Series Split', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Time Series Splitの特徴：")
    print("- 訓練データは常にテストデータより前")
    print("- 未来のデータでモデルを訓練しない（データリーク防止）")
    

### Leave-One-Out CV（小規模データ）
    
    
    from sklearn.model_selection import LeaveOneOut
    
    def loo_cv_demo(X, y, model):
        """
        Leave-One-Out CV
        小規模データ用
        """
        loo = LeaveOneOut()
        predictions = []
        actuals = []
    
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
    
            predictions.append(y_pred[0])
            actuals.append(y_test[0])
    
        return np.array(actuals), np.array(predictions)
    
    # 小規模データ
    X_small, y_small = generate_material_data(50, n_features=10)
    model_small = Ridge(alpha=1.0)
    
    y_actual, y_pred_loo = loo_cv_demo(X_small, y_small, model_small)
    
    mae_loo = mean_absolute_error(y_actual, y_pred_loo)
    r2_loo = r2_score(y_actual, y_pred_loo)
    
    print(f"LOO CV結果 (n={len(X_small)}):")
    print(f"MAE: {mae_loo:.4f}")
    print(f"R²: {r2_loo:.4f}")
    
    # 予測 vs 実測
    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_pred_loo, c='steelblue', s=50, alpha=0.6)
    plt.plot([y_actual.min(), y_actual.max()],
             [y_actual.min(), y_actual.max()],
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title('Leave-One-Out CV Results', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

* * *

## 3.3 ハイパーパラメータ最適化

モデルの性能を最大化するハイパーパラメータを探索します。

### Grid Search（全探索）
    
    
    from sklearn.model_selection import GridSearchCV
    
    def grid_search_demo(X, y):
        """
        Grid Searchによる全探索
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    
        model = RandomForestRegressor(random_state=42)
    
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
    
        grid_search.fit(X, y)
    
        return grid_search
    
    # 実行
    grid_result = grid_search_demo(X, y)
    
    print("Grid Search結果：")
    print(f"最適パラメータ: {grid_result.best_params_}")
    print(f"最良スコア (MAE): {-grid_result.best_score_:.4f}")
    print(f"\n探索空間サイズ: {len(grid_result.cv_results_['params'])}")
    
    # 結果可視化（2次元ヒートマップ）
    results = pd.DataFrame(grid_result.cv_results_)
    
    # n_estimators vs max_depth
    pivot_table = results.pivot_table(
        values='mean_test_score',
        index='param_max_depth',
        columns='param_n_estimators',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(-pivot_table, annot=True, fmt='.3f',
                cmap='YlOrRd', cbar_kws={'label': 'MAE'})
    plt.xlabel('n_estimators', fontsize=12)
    plt.ylabel('max_depth', fontsize=12)
    plt.title('Grid Search結果（MAE）', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    

### Random Search（ランダム探索）
    
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    def random_search_demo(X, y, n_iter=50):
        """
        Random Searchによるランダム探索
        """
        param_distributions = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': uniform(0.3, 0.7)
        }
    
        model = RandomForestRegressor(random_state=42)
    
        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    
        random_search.fit(X, y)
    
        return random_search
    
    # 実行
    random_result = random_search_demo(X, y, n_iter=50)
    
    print("\nRandom Search結果：")
    print(f"最適パラメータ: {random_result.best_params_}")
    print(f"最良スコア (MAE): {-random_result.best_score_:.4f}")
    
    # Grid vs Random比較
    print(f"\nGrid Search最良スコア: {-grid_result.best_score_:.4f}")
    print(f"Random Search最良スコア: {-random_result.best_score_:.4f}")
    print(f"\nRandom Searchの探索効率: {50 / len(grid_result.cv_results_['params']) * 100:.1f}%の探索で同等性能")
    

### Bayesian Optimization（Optuna, Hyperopt）
    
    
    import optuna
    
    def objective(trial, X, y):
        """
        Optuna目的関数
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42
        }
    
        model = RandomForestRegressor(**params)
    
        # 交差検証
        cv_scores = cross_val_score(
            model, X, y, cv=5,
            scoring='neg_mean_absolute_error'
        )
    
        return -cv_scores.mean()  # 最小化するため負にする
    
    # Optuna最適化
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100, show_progress_bar=True)
    
    print("\nOptuna Bayesian Optimization結果：")
    print(f"最適パラメータ: {study.best_params}")
    print(f"最良スコア (MAE): {study.best_value:.4f}")
    
    # 最適化履歴
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 最適化履歴
    axes[0].plot(study.trials_dataframe()['number'],
                 study.trials_dataframe()['value'],
                 'o-', color='steelblue', alpha=0.6, label='Trial Score')
    axes[0].plot(study.trials_dataframe()['number'],
                 study.trials_dataframe()['value'].cummin(),
                 'r-', linewidth=2, label='Best Score')
    axes[0].set_xlabel('Trial', fontsize=11)
    axes[0].set_ylabel('MAE', fontsize=11)
    axes[0].set_title('最適化履歴', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # パラメータ重要度
    importances = optuna.importance.get_param_importances(study)
    axes[1].barh(list(importances.keys()), list(importances.values()),
                 color='coral', alpha=0.7)
    axes[1].set_xlabel('Importance', fontsize=11)
    axes[1].set_title('ハイパーパラメータ重要度', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 早期終了（Early Stopping）
    
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    def early_stopping_demo(X, y):
        """
        Early Stoppingのデモ
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # GradientBoostingでステージごとの性能追跡
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
        model.fit(X_train, y_train)
    
        # ステージごとの予測
        train_scores = []
        val_scores = []
    
        for i, y_pred_train in enumerate(model.staged_predict(X_train)):
            y_pred_val = list(model.staged_predict(X_val))[i]
    
            train_mae = mean_absolute_error(y_train, y_pred_train)
            val_mae = mean_absolute_error(y_val, y_pred_val)
    
            train_scores.append(train_mae)
            val_scores.append(val_mae)
    
        # 最適なn_estimators（検証誤差最小）
        best_n_estimators = np.argmin(val_scores) + 1
    
        return train_scores, val_scores, best_n_estimators
    
    train_curve, val_curve, best_n = early_stopping_demo(X, y)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_curve)+1), train_curve,
             'b-', label='Training Error', linewidth=2)
    plt.plot(range(1, len(val_curve)+1), val_curve,
             'r-', label='Validation Error', linewidth=2)
    plt.axvline(x=best_n, color='green', linestyle='--',
                label=f'Best n_estimators={best_n}', linewidth=2)
    plt.xlabel('n_estimators', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Early Stopping', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Early Stopping結果：")
    print(f"最適n_estimators: {best_n}")
    print(f"検証MAE: {val_curve[best_n-1]:.4f}")
    print(f"過学習防止: {500 - best_n} イテレーション削減")
    

* * *

## 3.4 アンサンブル学習

複数のモデルを組み合わせて予測精度を向上させます。

### Bagging（Bootstrap Aggregating）
    
    
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
    
    # Bagging
    bagging = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=10),
        n_estimators=50,
        max_samples=0.8,
        random_state=42
    )
    
    # 比較用：単一Decision Tree
    single_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
    
    # 評価
    cv_bagging = cross_val_score(bagging, X, y, cv=5,
                                 scoring='neg_mean_absolute_error')
    cv_single = cross_val_score(single_tree, X, y, cv=5,
                                scoring='neg_mean_absolute_error')
    
    print("Bagging結果：")
    print(f"単一Decision Tree MAE: {-cv_single.mean():.4f} ± {cv_single.std():.4f}")
    print(f"Bagging MAE: {-cv_bagging.mean():.4f} ± {cv_bagging.std():.4f}")
    print(f"改善率: {(cv_single.mean() - cv_bagging.mean()) / cv_single.mean() * 100:.1f}%")
    

### Boosting（AdaBoost, Gradient Boosting, LightGBM, XGBoost）
    
    
    from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
    import lightgbm as lgb
    # import xgboost as xgb  # Optional
    
    # 各種Boostingアルゴリズム
    boosting_models = {
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        # 'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    boosting_results = []
    
    for model_name, model in boosting_models.items():
        cv_scores = cross_val_score(model, X, y, cv=5,
                                    scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        mae_std = cv_scores.std()
    
        boosting_results.append({
            'Model': model_name,
            'MAE': mae,
            'MAE_std': mae_std
        })
    
    boosting_df = pd.DataFrame(boosting_results)
    
    print("\nBoosting手法の比較：")
    print(boosting_df.to_string(index=False))
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.barh(boosting_df['Model'], boosting_df['MAE'],
             xerr=boosting_df['MAE_std'],
             color='steelblue', alpha=0.7)
    plt.xlabel('MAE', fontsize=12)
    plt.title('Boosting手法の性能比較', fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### Stacking
    
    
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge
    
    # Base models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('lgbm', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
    ]
    
    # Meta model
    meta_model = Ridge(alpha=1.0)
    
    # Stacking
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
    
    # 評価
    cv_stacking = cross_val_score(stacking, X, y, cv=5,
                                  scoring='neg_mean_absolute_error')
    
    print("\nStacking結果：")
    for name, _ in base_models:
        model_cv = cross_val_score(dict(base_models)[name], X, y, cv=5,
                                   scoring='neg_mean_absolute_error')
        print(f"{name} MAE: {-model_cv.mean():.4f}")
    
    print(f"Stacking Ensemble MAE: {-cv_stacking.mean():.4f}")
    print(f"\n改善効果: Stackingが最良単一モデルより "
          f"{((-cv_stacking.mean() / min([cross_val_score(m, X, y, cv=5, scoring='neg_mean_absolute_error').mean() for _, m in base_models])) - 1) * -100:.1f}% 改善")
    

### Voting
    
    
    from sklearn.ensemble import VotingRegressor
    
    # Voting Ensemble
    voting = VotingRegressor(
        estimators=base_models,
        weights=[1, 1.5, 1]  # GBに高い重み
    )
    
    cv_voting = cross_val_score(voting, X, y, cv=5,
                                scoring='neg_mean_absolute_error')
    
    print(f"\nVoting Ensemble MAE: {-cv_voting.mean():.4f}")
    
    # アンサンブル手法の比較
    ensemble_comparison = pd.DataFrame({
        'Method': ['Single Best', 'Bagging', 'Boosting (LightGBM)',
                   'Stacking', 'Voting'],
        'MAE': [
            min([cross_val_score(m, X, y, cv=5, scoring='neg_mean_absolute_error').mean() for _, m in base_models]),
            cv_bagging.mean(),
            boosting_df[boosting_df['Model'] == 'LightGBM']['MAE'].values[0],
            cv_stacking.mean(),
            cv_voting.mean()
        ]
    })
    
    ensemble_comparison['MAE'] = -ensemble_comparison['MAE']
    
    plt.figure(figsize=(10, 6))
    plt.barh(ensemble_comparison['Method'], ensemble_comparison['MAE'],
             color='coral', alpha=0.7)
    plt.xlabel('MAE (lower is better)', fontsize=12)
    plt.title('アンサンブル手法の性能比較', fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

* * *

## 3.5 ケーススタディ：Li-ion電池容量予測

実際のLi-ion電池データでモデル選択と最適化の全工程を実践します。
    
    
    # Li-ion電池容量データセット（シミュレーション）
    np.random.seed(42)
    n_batteries = 300
    
    battery_data = pd.DataFrame({
        '正極材料組成_Li': np.random.uniform(0.9, 1.1, n_batteries),
        '正極材料組成_Co': np.random.uniform(0, 0.6, n_batteries),
        '正極材料組成_Ni': np.random.uniform(0, 0.8, n_batteries),
        '正極材料組成_Mn': np.random.uniform(0, 0.4, n_batteries),
        '負極材料_黒鉛割合': np.random.uniform(0.8, 1.0, n_batteries),
        '電解質濃度': np.random.uniform(0.5, 2.0, n_batteries),
        '電極厚さ': np.random.uniform(50, 200, n_batteries),
        '粒子サイズ': np.random.uniform(1, 20, n_batteries),
        '焼成温度': np.random.uniform(700, 1000, n_batteries),
        'BET表面積': np.random.uniform(1, 50, n_batteries)
    })
    
    # 容量（真の関係は複雑な非線形）
    capacity = (
        150 * battery_data['正極材料組成_Ni'] +
        120 * battery_data['正極材料組成_Co'] +
        80 * battery_data['正極材料組成_Mn'] +
        30 * battery_data['電解質濃度'] -
        0.5 * battery_data['電極厚さ'] +
        2 * battery_data['BET表面積'] +
        0.1 * battery_data['焼成温度'] +
        20 * battery_data['正極材料組成_Ni'] * battery_data['電解質濃度'] +
        np.random.normal(0, 5, n_batteries)
    )
    
    battery_data['容量_mAh/g'] = capacity
    
    print("=== Li-ion電池容量予測データセット ===")
    print(f"サンプル数: {len(battery_data)}")
    print(f"特徴量数: {battery_data.shape[1] - 1}")
    print(f"\n容量統計:")
    print(battery_data['容量_mAh/g'].describe())
    
    X_battery = battery_data.drop('容量_mAh/g', axis=1)
    y_battery = battery_data['容量_mAh/g']
    

### Step 1: 5つのモデルの比較
    
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_battery, y_battery, test_size=0.2, random_state=42
    )
    
    # 5つのモデル
    models_to_compare = {
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'Stacking': StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('lgbm', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=5
        )
    }
    
    comparison_results = []
    
    for model_name, model in models_to_compare.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
        comparison_results.append({
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    print("\n=== Step 1: モデル比較結果 ===")
    print(comparison_df.to_string(index=False))
    

### Step 2: Optunaによるハイパーパラメータ自動最適化
    
    
    def objective_lightgbm(trial, X, y):
        """
        LightGBMハイパーパラメータ最適化
        """
        param = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42
        }
    
        model = lgb.LGBMRegressor(**param)
    
        cv_scores = cross_val_score(
            model, X, y, cv=5, scoring='neg_mean_absolute_error'
        )
    
        return -cv_scores.mean()
    
    # Optuna最適化
    study_battery = optuna.create_study(direction='minimize')
    study_battery.optimize(
        lambda trial: objective_lightgbm(trial, X_battery, y_battery),
        n_trials=100,
        show_progress_bar=True
    )
    
    print("\n=== Step 2: Optunaによる最適化 ===")
    print(f"最適パラメータ:")
    for key, value in study_battery.best_params.items():
        print(f"  {key}: {value}")
    print(f"\n最良MAE: {study_battery.best_value:.4f} mAh/g")
    
    # 最適化されたモデルで再評価
    best_model = lgb.LGBMRegressor(**study_battery.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    
    mae_best = mean_absolute_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)
    
    print(f"\n最適化後の性能:")
    print(f"MAE: {mae_best:.4f} mAh/g")
    print(f"R²: {r2_best:.4f}")
    

### Step 3: Stacking ensembleで最高性能達成
    
    
    # 最適化されたLightGBMを含むStacking
    optimized_stacking = StackingRegressor(
        estimators=[
            ('rf_tuned', RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=42
            )),
            ('lgbm_tuned', lgb.LGBMRegressor(**study_battery.best_params, random_state=42)),
            ('gb_tuned', GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.1, max_depth=7,
                random_state=42
            ))
        ],
        final_estimator=Ridge(alpha=0.5),
        cv=5
    )
    
    optimized_stacking.fit(X_train, y_train)
    y_pred_stack = optimized_stacking.predict(X_test)
    
    mae_stack = mean_absolute_error(y_test, y_pred_stack)
    r2_stack = r2_score(y_test, y_pred_stack)
    
    print("\n=== Step 3: 最終Stacking Ensemble ===")
    print(f"MAE: {mae_stack:.4f} mAh/g")
    print(f"R²: {r2_stack:.4f}")
    
    # 全工程の比較
    final_comparison = pd.DataFrame({
        'Stage': [
            'Baseline (Ridge)',
            'Best Single Model',
            'Optuna Optimized',
            'Final Stacking'
        ],
        'MAE': [
            comparison_df[comparison_df['Model'] == 'Ridge']['MAE'].values[0],
            comparison_df['MAE'].min(),
            mae_best,
            mae_stack
        ],
        'R²': [
            comparison_df[comparison_df['Model'] == 'Ridge']['R²'].values[0],
            comparison_df['R²'].max(),
            r2_best,
            r2_stack
        ]
    })
    
    print("\n=== 全工程の性能推移 ===")
    print(final_comparison.to_string(index=False))
    
    # 予測 vs 実測
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 最適化前の最良モデル
    best_single_idx = comparison_df['MAE'].idxmin()
    best_single_model = list(models_to_compare.values())[best_single_idx]
    y_pred_single = best_single_model.predict(X_test)
    
    axes[0].scatter(y_test, y_pred_single, c='steelblue', s=50, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('Actual Capacity (mAh/g)', fontsize=11)
    axes[0].set_ylabel('Predicted Capacity (mAh/g)', fontsize=11)
    axes[0].set_title(f'最良単一モデル (MAE={comparison_df["MAE"].min():.2f})',
                      fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 最終Stacking
    axes[1].scatter(y_test, y_pred_stack, c='coral', s=50, alpha=0.6)
    axes[1].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='Perfect')
    axes[1].set_xlabel('Actual Capacity (mAh/g)', fontsize=11)
    axes[1].set_ylabel('Predicted Capacity (mAh/g)', fontsize=11)
    axes[1].set_title(f'最終Stacking (MAE={mae_stack:.2f})',
                      fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    improvement = (comparison_df['MAE'].min() - mae_stack) / comparison_df['MAE'].min() * 100
    print(f"\n最終改善率: {improvement:.1f}%")
    

* * *

## 演習問題

### 問題1（難易度: easy）

K-Fold CVとStratified K-Fold CVを用いて、クラス不均衡データでの性能を比較してください。

解答例
    
    
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # 不均衡データ生成
    X, y = make_classification(n_samples=200, n_features=20,
                              weights=[0.9, 0.1], random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kfold = cross_val_score(model, X, y, cv=kf, scoring='f1')
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_stratified = cross_val_score(model, X, y, cv=skf, scoring='f1')
    
    print(f"K-Fold F1: {scores_kfold.mean():.4f} ± {scores_kfold.std():.4f}")
    print(f"Stratified K-Fold F1: {scores_stratified.mean():.4f} ± {scores_stratified.std():.4f}")
    

### 問題2（難易度: medium）

Optunaを用いて、Random Forestのハイパーパラメータを最適化してください。探索空間は`n_estimators`, `max_depth`, `min_samples_split`, `max_features`の4つとします。

解答例
    
    
    import optuna
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42
        }
    
        model = RandomForestRegressor(**params)
        cv_scores = cross_val_score(model, X, y, cv=5,
                                    scoring='neg_mean_absolute_error')
    
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_rf, n_trials=50)
    
    print(f"最適パラメータ: {study.best_params}")
    print(f"最良スコア: {study.best_value:.4f}")
    

### 問題3（難易度: hard）

Stacking Ensembleを構築し、3つのベースモデル（Ridge, Random Forest, LightGBM）と2つのメタモデル（Ridge, Lasso）を比較してください。どのメタモデルが最良の性能を示すか評価してください。

解答例
    
    
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge, Lasso
    import lightgbm as lgb
    
    base_models = [
        ('ridge', Ridge(alpha=1.0)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('lgbm', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
    ]
    
    meta_models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }
    
    results = []
    
    for meta_name, meta_model in meta_models.items():
        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )
    
        cv_scores = cross_val_score(stacking, X, y, cv=5,
                                    scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
    
        results.append({
            'Meta Model': meta_name,
            'MAE': mae
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    

* * *

## 3.6 モデル最適化環境と再現性

### ライブラリバージョン管理
    
    
    # モデル選択・最適化に必要なライブラリバージョン
    import sys
    import sklearn
    import optuna
    import lightgbm
    import pandas as pd
    import numpy as np
    
    reproducibility_info = {
        'Python': sys.version,
        'NumPy': np.__version__,
        'Pandas': pd.__version__,
        'scikit-learn': sklearn.__version__,
        'Optuna': optuna.__version__,
        'LightGBM': lightgbm.__version__,
        'Date': '2025-10-19'
    }
    
    print("=== モデル最適化環境 ===")
    for key, value in reproducibility_info.items():
        print(f"{key}: {value}")
    
    # 推奨バージョン
    print("\n【推奨環境】")
    recommended = """
    numpy==1.24.3
    pandas==2.0.3
    scikit-learn==1.3.0
    optuna==3.3.0
    lightgbm==4.0.0
    xgboost==2.0.0  # Optional
    matplotlib==3.7.2
    seaborn==0.12.2
    """
    print(recommended)
    
    print("\n【インストールコマンド】")
    print("```bash")
    print("pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0")
    print("pip install optuna==3.3.0 lightgbm==4.0.0")
    print("```")
    
    print("\n【注意事項】")
    print("⚠️ Optunaのバージョンで最適化アルゴリズムが異なる")
    print("⚠️ LightGBM/XGBoostはOS依存の問題あり → 同一環境で検証")
    print("⚠️ 論文再現時は必ずバージョンを明記")
    

### 乱数シード管理
    
    
    # 再現性確保のためのシード設定
    def set_all_seeds(seed=42):
        """
        全乱数生成器のシードを設定
        """
        import random
        import numpy as np
    
        random.seed(seed)
        np.random.seed(seed)
    
        # scikit-learnモデルはrandom_stateパラメータで個別指定
        print(f"✅ 乱数シード {seed} を設定")
        print("モデル訓練時は random_state={seed} を明示的に指定してください")
    
    set_all_seeds(42)
    
    # Optunaのシード管理
    print("\n【Optunaのシード管理】")
    print("```python")
    print("study = optuna.create_study(")
    print("    direction='minimize',")
    print("    sampler=optuna.samplers.TPESampler(seed=42)  # 再現性確保")
    print(")")
    print("```")
    

### 実践的な落とし穴（モデル選択・最適化編）
    
    
    print("=== モデル選択・最適化の落とし穴 ===\n")
    
    print("【落とし穴1: テストデータでのハイパーパラメータ調整】")
    print("❌ 悪い例：テストセットで最適化")
    print("```python")
    print("X_train, X_test = train_test_split(X, y)")
    print("# テストセットで複数モデルを評価し、最良を選択")
    print("for model in models:")
    print("    score = model.score(X_test, y_test)  # NG！")
    print("best_model = models[best_idx]")
    print("```")
    
    print("\n✅ 正しい例：Train/Validation/Test分割")
    print("```python")
    print("X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)")
    print("X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)")
    print("# Validationで最適化、Testは最終評価のみ")
    print("```")
    
    print("\n【落とし穴2: 交差検証時のデータリーク】")
    print("⚠️ 前処理（StandardScaler、Imputer）を全データで実行")
    print("→ テストデータの情報が訓練時に漏れる")
    
    print("\n✅ 対策：Pipelineで前処理を内包")
    print("```python")
    print("from sklearn.pipeline import Pipeline")
    print("from sklearn.preprocessing import StandardScaler")
    print("")
    print("pipeline = Pipeline([")
    print("    ('scaler', StandardScaler()),")
    print("    ('model', RandomForestRegressor())")
    print("])")
    print("cv_scores = cross_val_score(pipeline, X, y, cv=5)")
    print("```")
    
    print("\n【落とし穴3: Optunaでの過剰な試行回数】")
    print("⚠️ n_trials=1000で最適化 → 過学習リスク")
    print("→ Validationセットに過適合")
    
    print("\n✅ 対策：Early Stopping + Test評価")
    print("```python")
    print("study.optimize(objective, n_trials=100,")
    print("    callbacks=[optuna.study.MaxTrialsCallback(100)])")
    print("# 最適モデルをTestセットで最終評価")
    print("```")
    
    print("\n【落とし穴4: アンサンブルの過剰な複雑化】")
    print("⚠️ Stacking + Voting + Blending を多段に組み合わせ")
    print("→ 訓練時間増大、解釈性喪失、過学習")
    
    print("\n✅ 対策：シンプルなアンサンブル（2-3モデル）")
    print("```python")
    print("# シンプルなStacking（ベース3モデル + メタモデル1）")
    print("stacking = StackingRegressor(")
    print("    estimators=[('rf', RF), ('gb', GB), ('lgbm', LGBM)],")
    print("    final_estimator=Ridge(),")
    print("    cv=5")
    print(")")
    print("```")
    
    print("\n【落とし穴5: 小データでの複雑モデル】")
    print("⚠️ サンプル数50でNeural Network訓練")
    print("→ 過学習、不安定")
    
    print("\n✅ 対策：データサイズに応じたモデル選択")
    print("```python")
    print("if len(X) < 100:")
    print("    model = Ridge()  # 線形モデル")
    print("elif len(X) < 1000:")
    print("    model = RandomForest()  # 木ベース")
    print("else:")
    print("    model = NeuralNetwork()  # 深層学習")
    print("```")
    

* * *

## まとめ

この章では、**モデル選択とハイパーパラメータ最適化** を学びました。

**重要ポイント** ：

  1. **モデル選択** ：データサイズに応じた適切なモデル（小: Ridge、中: RF、大: NN）
  2. **交差検証** ：K-Fold、Stratified、Time Seriesを使い分け
  3. **最適化手法** ：Grid < Random < Bayesian（Optuna）の順で効率化
  4. **アンサンブル** ：Stacking > Voting > Boosting > Bagging
  5. **実践事例** ：Li-ion電池容量予測で30%以上の性能改善
  6. **環境管理** ：ライブラリバージョン、乱数シードの厳密な管理
  7. **実践的落とし穴** ：テストデータでの調整、データリーク、過剰最適化、小データでの複雑モデル

**次章予告** ： Chapter 4では、解釈可能AI（XAI）を学びます。SHAP、LIME、Attention可視化により、予測の物理的意味を理解し、実世界応用とキャリアパスを探ります。

* * *

## Chapter 3 チェックリスト

### モデル選択

  * [ ] **データサイズ評価**
  * [ ] サンプル数が100未満 → Ridge/Lasso（正則化線形モデル）
  * [ ] サンプル数が100-1000 → Random Forest/Gradient Boosting
  * [ ] サンプル数が1000以上 → Neural Network/Deep Learning
  * [ ] サンプル/特徴量比 > 10:1 を確認

  * [ ] **解釈性 vs 精度のバランス**

  * [ ] 解釈性重視（材料設計ガイドライン抽出）→ Ridge, Decision Tree
  * [ ] 精度重視（予測性能最大化）→ Gradient Boosting, Stacking
  * [ ] バランス型（実用的妥協点）→ Random Forest

  * [ ] **モデル複雑度の妥当性**

  * [ ] 学習曲線で過学習を確認（訓練誤差 << 検証誤差）
  * [ ] 正則化パラメータ（alpha, lambda）を調整
  * [ ] Early Stoppingで適切なイテレーション数を設定

### 交差検証

  * [ ] **K-Fold CV（基本）**
  * [ ] 回帰問題で標準的にK=5を使用
  * [ ] 計算リソースがあればK=10で精度向上
  * [ ] shuffle=Trueで順序効果を排除

  * [ ] **Stratified K-Fold（分類・不均衡データ）**

  * [ ] クラス不均衡データで必須
  * [ ] 各Foldのクラス分布が全体と一致するか確認
  * [ ] マイノリティクラスのサンプル数 > K を確認

  * [ ] **Time Series Split（時系列データ）**

  * [ ] 逐次的な材料実験データに適用
  * [ ] 訓練データが常にテストデータより過去か確認
  * [ ] 未来データでの訓練を絶対に行わない（データリーク防止）

  * [ ] **Leave-One-Out CV（小規模データ）**

  * [ ] サンプル数 < 50 で使用
  * [ ] 計算コスト高（n回の訓練）を認識
  * [ ] 過度に楽観的な評価に注意

### ハイパーパラメータ最適化

  * [ ] **Grid Search**
  * [ ] 探索空間が小さい（<100組み合わせ）時に使用
  * [ ] 重要パラメータ2-3個に絞る
  * [ ] 全組み合わせを網羅的に評価

  * [ ] **Random Search**

  * [ ] 探索空間が大きい時に使用
  * [ ] Grid Searchの10-20%の試行で同等性能
  * [ ] 連続値パラメータの効率的探索

  * [ ] **Bayesian Optimization（Optuna）**

  * [ ] 最も効率的（推奨）
  * [ ] n_trials=50-100で十分な性能
  * [ ] パラメータ重要度分析で解釈性向上
  * [ ] サンプリングアルゴリズム（TPESampler）のシード設定

  * [ ] **Early Stopping**

  * [ ] Boosting系モデルで必須
  * [ ] Validation誤差の最小点で停止
  * [ ] 過学習防止と計算時間削減

### アンサンブル学習

  * [ ] **Bagging**
  * [ ] Random Forest（自動的にBagging）
  * [ ] 高分散モデルの安定化
  * [ ] Decision Treeのアンサンブルで改善率20-30%

  * [ ] **Boosting**

  * [ ] Gradient Boosting, LightGBM, XGBoostを試行
  * [ ] 学習率（learning_rate）とn_estimatorsのトレードオフ
  * [ ] 過学習リスク → Early Stopping併用

  * [ ] **Stacking**

  * [ ] ベースモデル3-5個（多様性重視）
  * [ ] メタモデルはシンプル（Ridge, Lasso）
  * [ ] 交差検証（cv=5）でメタ特徴量生成
  * [ ] 最良単一モデルより5-10%改善を目標

  * [ ] **Voting**

  * [ ] Stackingより簡易
  * [ ] 重み付き平均（性能に応じた重み設定）
  * [ ] 3モデル程度で十分

### 実践的落とし穴の回避

  * [ ] **テストデータでの調整禁止**
  * [ ] Train/Validation/Test の3分割
  * [ ] Validationで最適化、Testは最終評価のみ
  * [ ] テストセットは1度だけ使用

  * [ ] **交差検証時のデータリーク防止**

  * [ ] Pipelineで前処理を内包
  * [ ] 各Foldで独立に前処理（StandardScaler.fit）
  * [ ] 特徴量選択も交差検証内で実行

  * [ ] **過剰最適化の回避**

  * [ ] Optunaのn_trials < 200（推奨100）
  * [ ] Validation性能とTest性能の乖離を監視
  * [ ] シンプルなモデルから開始

  * [ ] **アンサンブルの複雑化回避**

  * [ ] Stacking 1段まで（多段は禁止）
  * [ ] ベースモデル数 ≤ 5
  * [ ] 訓練時間と性能のトレードオフを評価

  * [ ] **小データでの複雑モデル回避**

  * [ ] サンプル数 < 100 → 線形モデル
  * [ ] サンプル数 < 1000 → 木ベースモデル
  * [ ] 深層学習は大規模データ（>1000）のみ

### 再現性の確保

  * [ ] **バージョン管理**
  * [ ] scikit-learn, Optuna, LightGBMのバージョン記録
  * [ ] requirements.txt作成
  * [ ] Docker環境で統一（推奨）

  * [ ] **乱数シード設定**

  * [ ] NumPy: np.random.seed(42)
  * [ ] scikit-learn: random_state=42
  * [ ] Optuna: TPESampler(seed=42)
  * [ ] すべてのランダム要素に同一シード使用

  * [ ] **最適化履歴の保存**

  * [ ] Optunaのstudy.trials_dataframe()をCSV保存
  * [ ] 最適パラメータをJSON保存
  * [ ] 学習曲線を画像保存

  * [ ] **モデルの永続化**

  * [ ] joblib.dump()で訓練済みモデル保存
  * [ ] 予測時に同じ前処理を適用
  * [ ] モデルのバージョン管理（Git LFS推奨）

### Li-ion電池容量予測ケーススタディ

  * [ ] **データ準備**
  * [ ] 組成・構造・プロセス条件の特徴量生成
  * [ ] Train/Test分割（80/20）
  * [ ] 欠損値・外れ値の事前処理

  * [ ] **Step 1: モデル比較**

  * [ ] 5つのモデル（Ridge, RF, GB, LGBM, Stacking）を評価
  * [ ] MAE, RMSE, R²で性能比較
  * [ ] 最良単一モデルを特定

  * [ ] **Step 2: Optuna最適化**

  * [ ] LightGBMのハイパーパラメータ8個を最適化
  * [ ] n_trials=100で実行
  * [ ] パラメータ重要度を分析

  * [ ] **Step 3: 最終Stacking**

  * [ ] 最適化モデル3個をベースに使用
  * [ ] Ridgeをメタモデルに選択
  * [ ] 最終性能がBaseline（Ridge）より30%以上改善

### モデル選択・最適化品質指標

  * [ ] **予測精度**
  * [ ] MAE（回帰）< データ標準偏差の20%
  * [ ] R² > 0.8（材料科学では0.7以上で良好）
  * [ ] RMSE確認（外れ値の影響評価）

  * [ ] **汎化性能**

  * [ ] 訓練誤差とテスト誤差の差 < 10%
  * [ ] 交差検証の標準偏差小（安定性）
  * [ ] 新規データでの性能維持確認

  * [ ] **計算効率**

  * [ ] 訓練時間 < 1時間（中規模データ）
  * [ ] 予測時間 < 1秒/サンプル
  * [ ] Optunaの最適化時間 < 10分（n_trials=100）

  * [ ] **解釈性**

  * [ ] feature_importances_で重要変数特定
  * [ ] SHAP値（次章）で物理的意味づけ
  * [ ] 専門家検証で妥当性確認

* * *

## 参考文献

  1. **Akiba, T., Sano, S., Yanase, T., et al.** (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. _Proceedings of the 25th ACM SIGKDD_ , 2623-2631. [DOI: 10.1145/3292500.3330701](<https://doi.org/10.1145/3292500.3330701>)

  2. **Bergstra, J. & Bengio, Y.** (2012). Random search for hyper-parameter optimization. _Journal of Machine Learning Research_ , 13, 281-305.

  3. **Dietterich, T. G.** (2000). Ensemble methods in machine learning. _International Workshop on Multiple Classifier Systems_ , 1-15. Springer.

  4. **Wolpert, D. H.** (1992). Stacked generalization. _Neural Networks_ , 5(2), 241-259. [DOI: 10.1016/S0893-6080(05)80023-1](<https://doi.org/10.1016/S0893-6080\(05\)80023-1>)

* * *

[← Chapter 2に戻る](<chapter-2.html>) | Chapter 4へ進む →（準備中）
