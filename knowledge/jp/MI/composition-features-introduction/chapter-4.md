---
title: 第4章：機械学習モデルとの統合
chapter_title: 第4章：機械学習モデルとの統合
subtitle: 組成ベース特徴量でGNN並みの精度を実現
---

🌐 JP | [🇬🇧 EN](<../../../en/MI/composition-features-introduction/chapter-4.html>) | Last sync: 2025-11-16

### 📖 この章で学ぶこと

  * **基本理解** : scikit-learnパイプライン、モデル選択基準、組成ベース vs 構造ベース性能比較
  * **実践スキル** : RandomForest/GradientBoosting/MLP実装、ハイパーパラメータGridSearch、学習曲線分析
  * **応用力** : SHAP解釈、複数モデルアンサンブル、Matbenchベンチマーク評価

## 導入

前章までに、組成ベース特徴量の生成方法（Magpie、ElementProperty、カスタムFeaturizer）を学びました。本章では、これらの特徴量を機械学習モデルと統合し、材料特性を高精度に予測する実践的な手法を習得します。 

特に重要なのは、**組成情報のみ** でGraph Neural Networks（GNN）に匹敵する予測精度を実現できる可能性です。Matbench v0.1ベンチマークでは、組成ベースモデル（ElemNet等）が形成エネルギー予測でR² 0.92を達成し、結晶構造を必要とするGNNモデル（R² 0.95-0.97）との差はわずか3-5%に留まります。 

本章では、scikit-learnパイプライン構築、複数モデルの比較（RandomForest、GradientBoosting、MLP）、ハイパーパラメータ最適化、SHAP解釈を通じて、実務で即座に活用できる機械学習ワークフローを習得します。 

## 4.1 scikit-learnパイプライン統合

### 4.1.1 パイプラインの基本概念

scikit-learnの`Pipeline`は、データ前処理・特徴量生成・モデル学習を一つの処理フローとして統合する強力なツールです。matminerのFeaturizerもパイプラインに統合可能で、以下のような利点があります： 

  * **再現性の確保** : 全処理を一つのオブジェクトで管理し、同じ前処理を訓練/テストデータに適用
  * **データリークの防止** : クロスバリデーション時に正しくデータ分割
  * **デプロイの容易性** : パイプライン全体をjoblibで保存・読み込み可能

    
    
    ```mermaid
    graph LR
                    A[化学組成Fe2O3] --> B[FeaturizerMagpieData]
                    B --> C[特徴量145次元]
                    C --> D[StandardScaler標準化]
                    D --> E[ML ModelRandomForest]
                    E --> F[予測値形成エネルギー]
                    style A fill:#e3f2fd
                    style C fill:#fff3e0
                    style F fill:#f1f8e9
    ```

### 4.1.2 基本的なパイプライン構築

matminerのFeaturizerをscikit-learnパイプラインに統合する基本パターンを示します。 

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/hackingmaterials/matminer_examples/blob/main/matminer_examples/machine_learning-nb/sklearn_pipeline_example.ipynb>)
    
    
    # コード例1: scikit-learn Pipeline with Magpie Featurizer
    
    # 必要なライブラリのインポート
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition
    import joblib
    
    # サンプルデータの準備（形成エネルギー予測）
    data = pd.DataFrame({
        'composition': ['Fe2O3', 'Al2O3', 'TiO2', 'SiO2', 'MgO',
                        'CaO', 'Na2O', 'K2O', 'ZnO', 'CuO'],
        'formation_energy': [-8.3, -16.6, -9.7, -9.1, -6.1,
                             -6.3, -4.2, -3.6, -3.6, -1.6]  # eV/atom
    })
    
    # 1. 化学組成を文字列からCompositionオブジェクトへ変換
    str_to_comp = StrToComposition(target_col_id='composition_obj')
    data = str_to_comp.featurize_dataframe(data, 'composition')
    
    # 2. パイプラインの構築
    # Pipeline構文: [('名前', オブジェクト), ...]のタプルリスト
    pipeline = Pipeline([
        ('scaler', StandardScaler()),           # 特徴量の標準化
        ('model', RandomForestRegressor(       # ランダムフォレスト回帰
            n_estimators=100,
            max_depth=10,
            random_state=42
        ))
    ])
    
    # 3. 特徴量生成（パイプライン外で実行）
    featurizer = ElementProperty.from_preset('magpie')
    data = featurizer.featurize_dataframe(data, 'composition_obj')
    
    # 4. 特徴量とターゲットの分離
    feature_cols = featurizer.feature_labels()
    X = data[feature_cols]
    y = data['formation_energy']
    
    # 5. 訓練/テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 6. パイプラインの訓練
    pipeline.fit(X_train, y_train)
    
    # 7. 性能評価
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"訓練データR²スコア: {train_score:.4f}")
    print(f"テストデータR²スコア: {test_score:.4f}")
    
    # 8. クロスバリデーション
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring='r2'
    )
    print(f"\n5-Fold CV平均R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 9. パイプラインの保存
    joblib.dump(pipeline, 'formation_energy_model.pkl')
    print("\nモデルをformation_energy_model.pklに保存しました")
    
    # 10. パイプラインの読み込みと使用
    loaded_pipeline = joblib.load('formation_energy_model.pkl')
    predictions = loaded_pipeline.predict(X_test)
    print(f"\n予測例（最初の3サンプル）:")
    for i in range(min(3, len(predictions))):
        print(f"  実測値: {y_test.iloc[i]:.2f} eV/atom, "
              f"予測値: {predictions[i]:.2f} eV/atom")
    

#### 💡 make_pipeline vs Pipeline

`make_pipeline`を使用すると、各ステップに自動で名前が付与され、より簡潔に記述できます： 
    
    
    from sklearn.pipeline import make_pipeline
    
    # 自動命名版（名前は 'standardscaler', 'randomforestregressor' 等）
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    
    # 手動命名版（任意の名前を指定可能）
    pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('rf_model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    

**使い分け基準** : ハイパーパラメータ最適化で特定ステップにアクセスする場合は`Pipeline`（名前を明示）、シンプルなワークフローには`make_pipeline`が推奨されます。 

### 4.1.3 パイプラインへのFeaturizer統合

matminerのFeaturizerをscikit-learnパイプラインに直接統合するには、`DFMLAdaptor`クラスを使用します（matminer 0.7.4以降）。 
    
    
    # Featurizerをパイプラインに統合する方法
    
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, ValenceOrbital
    )
    from matminer.utils.pipeline import DFMLAdaptor
    
    # 複数のFeaturizerを統合
    multi_featurizer = MultipleFeaturizer([
        ElementProperty.from_preset('magpie'),
        Stoichiometry(),
        ValenceOrbital()
    ])
    
    # DFMLAdaptorでscikit-learn互換にする
    featurizer_step = DFMLAdaptor(
        multi_featurizer,
        input_cols=['composition_obj'],
        ignore_cols=['composition', 'formation_energy']  # 特徴量生成対象外の列
    )
    
    # パイプライン構築
    full_pipeline = Pipeline([
        ('featurize', featurizer_step),
        ('scale', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # データフレーム全体を入力として訓練可能
    full_pipeline.fit(data, data['formation_energy'])
    

#### ⚠️ データリークに注意

クロスバリデーション時、Featurizerがデータセット全体の統計量（例: 平均値）を使用する場合、データリークが発生します。これを防ぐには： 

  * **方法1** : Featurizer適用後にパイプライン構築（推奨）
  * **方法2** : カスタムTransformerでfit時のみ統計量計算

本シリーズでは簡潔性のため、方法1（Featurizer適用後にパイプライン構築）を採用します。 

## 4.2 モデル選択とハイパーパラメータ最適化

### 4.2.1 主要な機械学習モデル

組成ベース特徴量を用いた材料特性予測では、以下の3つのモデルが広く使用されています： 

モデル | 特徴 | 主要ハイパーパラメータ | 推奨範囲  
---|---|---|---  
**Random Forest** | 決定木のアンサンブル、過学習に強い、特徴量重要度取得可能 |  `n_estimators`  
`max_depth`  
`min_samples_split` |  100-500  
10-50  
2-10   
**Gradient Boosting**  
(XGBoost/LightGBM) | 逐次的に誤差を修正、高精度、過学習リスクあり |  `learning_rate`  
`n_estimators`  
`max_depth` |  0.01-0.3  
100-1000  
3-10   
**Neural Network**  
(MLP) | 非線形関係の学習、大規模データで有効、学習時間長い |  `hidden_layer_sizes`  
`activation`  
`alpha` |  (100,), (100, 50)  
'relu', 'tanh'  
0.0001-0.01   
  
### 4.2.2 Random Forest回帰

Random Forestは、複数の決定木の予測を平均化することで、高い予測精度と過学習耐性を実現します。材料科学では特徴量重要度分析にも頻繁に使用されます。 

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/hackingmaterials/matminer_examples/blob/main/matminer_examples/machine_learning-nb/random_forest_example.ipynb>)
    
    
    # コード例2: Random Forest回帰（ハイパーパラメータ最適化含む）
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    
    # 前章で生成した特徴量を使用（X_train, X_test, y_train, y_test）
    
    # 1. ベースラインモデル（デフォルトパラメータ）
    rf_baseline = RandomForestRegressor(random_state=42)
    rf_baseline.fit(X_train, y_train)
    baseline_score = rf_baseline.score(X_test, y_test)
    print(f"ベースラインモデル R² スコア: {baseline_score:.4f}")
    
    # 2. ハイパーパラメータグリッドサーチ
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=5,                    # 5-fold クロスバリデーション
        scoring='r2',            # R²スコアで評価
        n_jobs=-1,               # 全CPUコアを使用
        verbose=1
    )
    
    # GridSearch実行
    print("\nGridSearchCV実行中...")
    grid_search.fit(X_train, y_train)
    
    # 最適パラメータ
    print(f"\n最適パラメータ: {grid_search.best_params_}")
    print(f"CV最良スコア: {grid_search.best_score_:.4f}")
    
    # 3. 最適モデルでの評価
    best_rf = grid_search.best_estimator_
    y_pred_train = best_rf.predict(X_train)
    y_pred_test = best_rf.predict(X_test)
    
    # 評価指標の計算
    def evaluate_model(y_true, y_pred, dataset_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
    
        print(f"\n{dataset_name} 評価結果:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
    
        return mae, rmse, r2
    
    train_metrics = evaluate_model(y_train, y_pred_train, "訓練データ")
    test_metrics = evaluate_model(y_test, y_pred_test, "テストデータ")
    
    # 4. 特徴量重要度の可視化
    feature_importance = best_rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n上位10特徴量:")
    print(importance_df.head(10).to_string(index=False))
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df.head(10)['feature'],
             importance_df.head(10)['importance'])
    plt.xlabel('重要度')
    plt.title('Random Forest 特徴量重要度（上位10）')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=150)
    print("\n特徴量重要度グラフを rf_feature_importance.png に保存しました")
    
    # 5. 予測値 vs 実測値プロット
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label='理想的な予測')
    plt.xlabel('実測値 (eV/atom)')
    plt.ylabel('予測値 (eV/atom)')
    plt.title(f'Random Forest予測性能\nR² = {test_metrics[2]:.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('rf_prediction_plot.png', dpi=150)
    print("予測プロットを rf_prediction_plot.png に保存しました")
    

### 4.2.3 Gradient Boosting (XGBoost)

Gradient Boostingは、弱学習器（浅い決定木）を逐次的に追加し、各ステップで前のモデルの誤差を修正する手法です。XGBoostは高速化と正則化を実装した実装で、Kaggleコンペティションで頻繁に優勝しています。 

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/hackingmaterials/matminer_examples/blob/main/matminer_examples/machine_learning-nb/xgboost_example.ipynb>)
    
    
    # コード例3: Gradient Boosting実装（XGBoost）
    
    # XGBoostのインストール（Google Colabでは不要）
    # !pip install xgboost
    
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    # 1. XGBoost用のデータセット作成（高速化のため）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 2. ベースラインモデル
    params_baseline = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,                     # 学習率
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    # 訓練（early stopping使用）
    evals = [(dtrain, 'train'), (dtest, 'test')]
    bst_baseline = xgb.train(
        params_baseline,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    print(f"ベースライン最良イテレーション: {bst_baseline.best_iteration}")
    print(f"テストRMSE: {bst_baseline.best_score:.4f}")
    
    # 3. RandomizedSearchCVでハイパーパラメータ最適化
    # XGBoostRegressorを使用（scikit-learn互換）
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42
    )
    
    param_distributions = {
        'n_estimators': randint(100, 1000),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.6, 0.4),          # 0.6-1.0
        'colsample_bytree': uniform(0.6, 0.4),   # 0.6-1.0
        'gamma': uniform(0, 0.5)
    }
    
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions,
        n_iter=50,                # 50回ランダムサンプリング
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("\nRandomizedSearchCV実行中...")
    random_search.fit(X_train, y_train)
    
    print(f"\n最適パラメータ: {random_search.best_params_}")
    print(f"CV最良R²スコア: {random_search.best_score_:.4f}")
    
    # 4. 最適モデルでの評価
    best_xgb = random_search.best_estimator_
    y_pred_test_xgb = best_xgb.predict(X_test)
    xgb_metrics = evaluate_model(y_test, y_pred_test_xgb, "XGBoost テストデータ")
    
    # 5. 特徴量重要度（Gain基準）
    xgb_importance = best_xgb.get_booster().get_score(importance_type='gain')
    importance_df_xgb = pd.DataFrame({
        'feature': list(xgb_importance.keys()),
        'importance': list(xgb_importance.values())
    }).sort_values('importance', ascending=False)
    
    print("\nXGBoost 上位10特徴量 (Gain):")
    print(importance_df_xgb.head(10).to_string(index=False))
    
    # 6. 学習曲線の可視化
    evals_result = best_xgb.evals_result()
    plt.figure(figsize=(10, 6))
    plt.plot(evals_result['validation_0']['rmse'], label='訓練RMSE')
    plt.xlabel('イテレーション')
    plt.ylabel('RMSE')
    plt.title('XGBoost学習曲線')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgb_learning_curve.png', dpi=150)
    print("\n学習曲線を xgb_learning_curve.png に保存しました")
    

#### 💡 GridSearchCV vs RandomizedSearchCV

**GridSearchCV** : 全パラメータ組み合わせを網羅的に探索（計算コスト高） 

**RandomizedSearchCV** : ランダムサンプリングで効率的に探索（推奨） 

**使い分け** : パラメータ空間が広い場合（XGBoost等）はRandomizedSearchCV、狭い場合（Random Forest等）はGridSearchCVが適切です。 

### 4.2.4 Neural Network (MLP Regressor)

多層パーセプトロン（MLP）は、複数の隠れ層を持つニューラルネットワークです。非線形関係の学習に優れ、大規模データセットで高い性能を発揮します。 

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/hackingmaterials/matminer_examples/blob/main/matminer_examples/machine_learning-nb/neural_network_example.ipynb>)
    
    
    # コード例4: Neural Network（MLP Regressor）
    
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import validation_curve
    
    # 1. データの標準化（MLPでは必須）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. ベースラインMLP
    mlp_baseline = MLPRegressor(
        hidden_layer_sizes=(100,),      # 隠れ層1層、100ユニット
        activation='relu',
        solver='adam',
        alpha=0.0001,                   # L2正則化係数
        batch_size='auto',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,            # 検証損失が改善しない場合停止
        validation_fraction=0.1,
        verbose=False
    )
    
    mlp_baseline.fit(X_train_scaled, y_train)
    print(f"ベースラインMLP 訓練完了（{mlp_baseline.n_iter_}イテレーション）")
    print(f"テストR²スコア: {mlp_baseline.score(X_test_scaled, y_test):.4f}")
    
    # 3. ハイパーパラメータグリッドサーチ
    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    
    mlp_model = MLPRegressor(
        solver='adam',
        max_iter=1000,
        early_stopping=True,
        random_state=42,
        verbose=False
    )
    
    grid_search_mlp = GridSearchCV(
        mlp_model,
        param_grid_mlp,
        cv=3,                           # MLPは時間がかかるため3-fold
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    print("\nMLP GridSearchCV実行中...")
    grid_search_mlp.fit(X_train_scaled, y_train)
    
    print(f"\n最適パラメータ: {grid_search_mlp.best_params_}")
    print(f"CV最良R²スコア: {grid_search_mlp.best_score_:.4f}")
    
    # 4. 最適モデルでの評価
    best_mlp = grid_search_mlp.best_estimator_
    y_pred_test_mlp = best_mlp.predict(X_test_scaled)
    mlp_metrics = evaluate_model(y_test, y_pred_test_mlp, "MLP テストデータ")
    
    # 5. 損失曲線の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(best_mlp.loss_curve_, label='訓練損失')
    if hasattr(best_mlp, 'validation_scores_'):
        plt.plot(best_mlp.validation_scores_, label='検証スコア')
    plt.xlabel('イテレーション')
    plt.ylabel('損失 / スコア')
    plt.title('MLP学習曲線')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('mlp_loss_curve.png', dpi=150)
    print("\n損失曲線を mlp_loss_curve.png に保存しました")
    
    # 6. 隠れ層サイズの影響を調査（Validation Curve）
    train_scores, valid_scores = validation_curve(
        MLPRegressor(activation='relu', alpha=0.001, random_state=42, max_iter=500),
        X_train_scaled, y_train,
        param_name='hidden_layer_sizes',
        param_range=[(50,), (100,), (150,), (100, 50), (150, 100)],
        cv=3,
        scoring='r2',
        n_jobs=-1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(['(50,)', '(100,)', '(150,)', '(100,50)', '(150,100)'],
             train_scores.mean(axis=1), 'o-', label='訓練スコア')
    plt.plot(['(50,)', '(100,)', '(150,)', '(100,50)', '(150,100)'],
             valid_scores.mean(axis=1), 's-', label='検証スコア')
    plt.xlabel('隠れ層サイズ')
    plt.ylabel('R²スコア')
    plt.title('MLPの隠れ層サイズとモデル性能')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('mlp_validation_curve.png', dpi=150)
    print("Validation Curveを mlp_validation_curve.png に保存しました")
    

### 4.2.5 学習曲線分析

学習曲線（Learning Curve）は、訓練データ量と予測性能の関係を可視化し、過学習・未学習の診断に使用します。 
    
    
    ```mermaid
    graph TD
                    A[学習曲線分析] --> B{訓練スコア vs 検証スコア}
                    B -->|訓練↑検証↑| C[良好な汎化性能]
                    B -->|訓練↑検証↓| D[過学習モデル簡素化]
                    B -->|訓練↓検証↓| E[未学習モデル複雑化]
                    B -->|両方収束| F[データ追加で改善期待]
                    style C fill:#c8e6c9
                    style D fill:#ffccbc
                    style E fill:#ffe0b2
                    style F fill:#b3e5fc
    ```
    
    
    
    
    
    # コード例5: 学習曲線分析（learning_curve関数）
    
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_learning_curve(estimator, X, y, title, cv=5, n_jobs=-1,
                            train_sizes=np.linspace(0.1, 1.0, 10)):
        """
        学習曲線を描画する関数
    
        Parameters:
        -----------
        estimator : scikit-learn estimator
            評価対象のモデル
        X : array-like, shape (n_samples, n_features)
            特徴量
        y : array-like, shape (n_samples,)
            ターゲット
        title : str
            グラフタイトル
        cv : int
            クロスバリデーションのfold数
        train_sizes : array-like
            訓練データのサンプリング比率
        """
        plt.figure(figsize=(10, 6))
    
        # learning_curve関数で訓練/検証スコアを計算
        train_sizes_abs, train_scores, valid_scores = learning_curve(
            estimator, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='r2',
            n_jobs=n_jobs,
            verbose=0
        )
    
        # 平均と標準偏差を計算
        train_scores_mean = train_scores.mean(axis=1)
        train_scores_std = train_scores.std(axis=1)
        valid_scores_mean = valid_scores.mean(axis=1)
        valid_scores_std = valid_scores.std(axis=1)
    
        # プロット
        plt.fill_between(train_sizes_abs,
                         train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std,
                         alpha=0.1, color='blue')
        plt.fill_between(train_sizes_abs,
                         valid_scores_mean - valid_scores_std,
                         valid_scores_mean + valid_scores_std,
                         alpha=0.1, color='red')
    
        plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='blue',
                 label='訓練スコア')
        plt.plot(train_sizes_abs, valid_scores_mean, 's-', color='red',
                 label='クロスバリデーションスコア')
    
        plt.xlabel('訓練サンプル数')
        plt.ylabel('R²スコア')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
    
        return plt
    
    # 複数モデルの学習曲線を比較
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20,
                                               random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                    learning_rate=0.1, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), alpha=0.001,
                            max_iter=500, random_state=42)
    }
    
    for model_name, model in models.items():
        # MLPの場合は標準化が必要
        if model_name == 'MLP':
            X_current = X_train_scaled
            y_current = y_train
        else:
            X_current = X_train
            y_current = y_train
    
        plot_learning_curve(
            model, X_current, y_current,
            title=f'{model_name} 学習曲線',
            cv=5
        )
        plt.savefig(f'learning_curve_{model_name.replace(" ", "_").lower()}.png',
                    dpi=150)
        print(f"{model_name} の学習曲線を保存しました")
    
    # 診断ガイド
    print("\n【学習曲線の診断ガイド】")
    print("1. 訓練スコア高・検証スコア低 → 過学習（モデル簡素化、正則化強化）")
    print("2. 訓練スコア低・検証スコア低 → 未学習（モデル複雑化、特徴量追加）")
    print("3. 両スコア高位で収束 → 良好（これ以上データ追加しても改善小）")
    print("4. 両スコア上昇傾向 → データ追加で改善期待")
    ```

## 4.3 組成ベース vs GNN性能比較

### 4.3.1 Matbench v0.1ベンチマーク

[Matbench v0.1](<https://matbench.materialsproject.org/>)は、材料特性予測の標準ベンチマークで、13個のタスク（形成エネルギー、バンドギャップ、弾性率等）で構成されています。各タスクは数千～数万のデータポイントを含み、公平な性能比較が可能です。 

#### 📊 Matbench v0.1の特徴

  * **13タスク** : 形成エネルギー、バンドギャップ、誘電率、体積弾性率、剪断弾性率等
  * **統一データ分割** : 5-fold CV、事前定義された訓練/テスト分割
  * **多様な入力** : 組成のみ（9タスク）、構造必須（4タスク）
  * **公開リーダーボード** : 最新アルゴリズムの性能を比較可能

### 4.3.2 組成ベース vs GNN性能詳細

以下の表は、Matbenchベンチマークでの組成ベースモデル（Magpie、ElemNet）とGNNモデル（CGCNN、MEGNet、ALIGNN）の性能比較です。 

タスク | サンプル数 | 評価指標 | Magpie + RF | ElemNet | CGCNN | MEGNet | ALIGNN  
---|---|---|---|---|---|---|---  
**形成エネルギー**  
(matbench_mp_e_form) | 132,752 | MAE (eV/atom) | 0.082 | 0.065 | 0.052 | 0.059 | 0.047  
**バンドギャップ**  
(matbench_mp_gap) | 106,113 | MAE (eV) | 0.42 | 0.35 | 0.27 | 0.30 | 0.25  
**体積弾性率**  
(matbench_mp_bulk_modulus) | 10,987 | MAE (log10(GPa)) | 0.082 | 0.075 | 0.063 | 0.068 | 0.059  
**剪断弾性率**  
(matbench_mp_shear_modulus) | 10,987 | MAE (log10(GPa)) | 0.092 | 0.084 | 0.071 | 0.076 | 0.067  
**誘電率**  
(matbench_dielectric) | 4,764 | MAE (log10) | 0.29 | 0.26 | 0.21 | 0.23 | 0.19  
**ペロブスカイト形成**  
(matbench_perovskites) | 18,928 | ROCAUC | 0.91 | 0.94 | 0.93 | 0.92 | 0.95  
**フォノン周波数**  
(matbench_phonons) | 1,265 | MAE (cm⁻¹) | 89.5 | — | 62.3 | 71.2 | 58.7  
  
#### 🔍 性能差の要因分析

**組成ベースが有利な場合** : 

  * **ペロブスカイト形成** : 化学組成の規則性が支配的（ElemNet: ROCAUC 0.94）
  * **小規模データセット** : GNNは大量データを要求、組成ベースは数百サンプルで学習可能

**GNNが有利な場合** : 

  * **形成エネルギー** : 結晶構造（配位数、結合距離）が重要（ALIGNN: MAE 0.047 vs ElemNet: 0.065）
  * **機械的性質** : 弾性率は結晶対称性に強く依存
  * **フォノン周波数** : 原子配置が直接影響

### 4.3.3 使い分け基準

組成ベースとGNNベースモデルの選択基準を以下に示します： 

基準 | 組成ベース推奨 | GNN推奨  
---|---|---  
**データサイズ** | 小規模（<1,000サンプル） | 大規模（>10,000サンプル）  
**結晶構造** | 未知・未決定 | 既知・高精度  
**計算コスト** | CPUで秒単位 | GPUで分～時間単位  
**解釈可能性** | 特徴量重要度が直感的 | ブラックボックス性強い  
**予測精度** | GNNの85-95% | 最高精度  
**探索フェーズ** | 初期スクリーニング | 最終候補の精密評価  
  
#### 💡 実務での推奨ワークフロー

  1. **Phase 1 - 初期スクリーニング** : 組成ベースモデルで10万候補を1,000候補に絞る（CPUで数分）
  2. **Phase 2 - 構造最適化** : 1,000候補の結晶構造をDFT緩和計算（数日）
  3. **Phase 3 - 精密予測** : GNNで100候補を選定（GPUで数時間）
  4. **Phase 4 - 実験検証** : 最終10候補を合成・測定

このハイブリッドアプローチにより、計算コストを抑えつつ高精度な材料探索が実現できます。 

### 4.3.4 Matbenchベンチマーク実行例

Matbenchベンチマークを実際に実行し、組成ベースモデルの性能を評価します。 

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/hackingmaterials/matminer_examples/blob/main/matminer_examples/machine_learning-nb/matbench_example.ipynb>)
    
    
    # コード例6: Matbenchベンチマーク評価
    
    # Matbenchのインストール
    # !pip install matbench
    
    from matbench.bench import MatbenchBenchmark
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import pandas as pd
    
    # 1. Matbenchベンチマークの初期化
    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            'matbench_mp_e_form',        # 形成エネルギー
            'matbench_mp_gap',           # バンドギャップ
            'matbench_perovskites'       # ペロブスカイト形成
        ]
    )
    
    # 2. データセットのロード
    for task in mb.tasks:
        task.load()
        print(f"\n{task.dataset_name}:")
        print(f"  サンプル数: {len(task.df)}")
        print(f"  ターゲット: {task.metadata['target']}")
        print(f"  タスクタイプ: {task.metadata['task_type']}")
    
    # 3. 組成ベースモデルの構築と評価
    def evaluate_composition_model(task):
        """
        組成ベースモデル（Magpie + RandomForest）でMatbenchタスクを評価
        """
        # 化学組成の変換
        str_to_comp = StrToComposition(target_col_id='composition_obj')
        task.df = str_to_comp.featurize_dataframe(task.df, 'composition')
    
        # Magpie特徴量の生成
        featurizer = ElementProperty.from_preset('magpie')
        task.df = featurizer.featurize_dataframe(task.df, 'composition_obj')
    
        feature_cols = featurizer.feature_labels()
    
        # パイプラインの構築
        if task.metadata['task_type'] == 'regression':
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=30,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:  # classification
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=30,
                random_state=42,
                n_jobs=-1
            )
    
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    
        # Matbench標準の5-fold CVで評価
        for fold_idx, fold in enumerate(task.folds):
            # 訓練/テストデータ取得
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            X_train = train_inputs[feature_cols]
            y_train = train_outputs
    
            # モデル訓練
            pipeline.fit(X_train, y_train)
    
            # テストデータで予測
            test_inputs, test_outputs = task.get_test_data(fold,
                                                            include_target=True)
            X_test = test_inputs[feature_cols]
            predictions = pipeline.predict(X_test)
    
            # 予測結果を記録
            task.record(fold, predictions)
    
            print(f"  Fold {fold_idx + 1} 完了")
    
        # 全体スコアの計算
        scores = task.scores
        print(f"\n{task.dataset_name} 評価結果:")
        for metric, values in scores.items():
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
        return scores
    
    # 4. 全タスクで評価実行
    results = {}
    for task in mb.tasks:
        print(f"\n{'='*60}")
        print(f"評価中: {task.dataset_name}")
        print('='*60)
        scores = evaluate_composition_model(task)
        results[task.dataset_name] = scores
    
    # 5. 結果のサマリー表示
    print("\n" + "="*60)
    print("全タスク評価結果サマリー")
    print("="*60)
    for dataset_name, scores in results.items():
        print(f"\n{dataset_name}:")
        for metric, values in scores.items():
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # 6. 結果をMatbench公式フォーマットで保存
    mb.to_file('matbench_composition_results.json')
    print("\n結果を matbench_composition_results.json に保存しました")
    print("公式リーダーボードへの提出が可能です: https://matbench.materialsproject.org/")
    

## 4.4 モデル解釈可能性（SHAP）

### 4.4.1 SHAPの基礎理論

SHAP（SHapley Additive exPlanations）は、ゲーム理論のShapley値を機械学習に応用した解釈手法です。各特徴量が予測値に与える貢献度を定量化し、ブラックボックスモデルの意思決定を説明可能にします。 

#### 📐 SHAP値の定義

特徴量 $i$ のSHAP値 $\phi_i$ は、以下のShapley値として定義されます： 

$$ \phi_i = \sum_{S \subseteq F \setminus \\{i\\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \\{i\\}) - f(S)] $$ 

ここで、$F$ は全特徴量の集合、$S$ は特徴量の部分集合、$f(S)$ は特徴量集合 $S$ を使用した予測値です。 

**直感的解釈** : 特徴量 $i$ を追加することで予測値がどれだけ変化するかを、全ての可能な特徴量組み合わせで平均化した値。 

### 4.4.2 特徴量重要度 vs SHAP

Random Forestの特徴量重要度とSHAPの違いを理解することが重要です： 

比較項目 | 特徴量重要度（Feature Importance） | SHAP値  
---|---|---  
**定義** | 不純度（Gini）減少の平均 | 予測値への貢献度（Shapley値）  
**解釈** | グローバル（全体的な重要度） | ローカル（個別サンプルの貢献度）  
**方向性** | なし（重要度のみ） | あり（正/負の貢献）  
**計算コスト** | 低（訓練時に計算） | 高（予測ごとに計算）  
**理論保証** | なし（ヒューリスティック） | あり（公理的基礎）  
  
### 4.4.3 SHAP実装と可視化

SHAPライブラリを使用して、Random ForestモデルをSHAP解析します。 

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/hackingmaterials/matminer_examples/blob/main/matminer_examples/machine_learning-nb/shap_example.ipynb>)
    
    
    # コード例7: SHAP解釈（summary_plot、dependence_plot）
    
    # SHAPのインストール
    # !pip install shap
    
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 訓練済みモデルを使用（前のセクションで訓練したbest_rf）
    # X_test, y_test も前のセクションから継続
    
    # 1. SHAP Explainerの作成（Tree Explainer for Random Forest）
    explainer = shap.TreeExplainer(best_rf)
    
    # 2. SHAP値の計算（全テストデータ）
    print("SHAP値計算中...")
    shap_values = explainer.shap_values(X_test)
    print(f"SHAP値のshape: {shap_values.shape}")  # (n_samples, n_features)
    
    # 3. Summary Plot（全体的な特徴量重要度と分布）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
    print("\nSummary Plotを shap_summary_plot.png に保存しました")
    plt.close()
    
    # 4. Summary Plot（棒グラフ版：平均絶対SHAP値）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                      plot_type='bar', show=False, max_display=20)
    plt.tight_layout()
    plt.savefig('shap_summary_bar.png', dpi=150, bbox_inches='tight')
    print("Summary Plot（棒グラフ）を shap_summary_bar.png に保存しました")
    plt.close()
    
    # 5. Dependence Plot（特定特徴量の詳細分析）
    # 最も重要な特徴量を取得
    shap_importance = np.abs(shap_values).mean(axis=0)
    top_feature_idx = np.argmax(shap_importance)
    top_feature_name = feature_cols[top_feature_idx]
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        top_feature_idx,
        shap_values,
        X_test,
        feature_names=feature_cols,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'shap_dependence_{top_feature_name}.png',
                dpi=150, bbox_inches='tight')
    print(f"\nDependence Plot（{top_feature_name}）を保存しました")
    plt.close()
    
    # 6. Force Plot（個別サンプルの予測説明）
    # JavaScriptベースのため、Jupyter環境推奨
    shap.initjs()
    
    # 最初のテストサンプルの予測を説明
    sample_idx = 0
    shap_values_sample = shap_values[sample_idx, :]
    base_value = explainer.expected_value
    prediction = best_rf.predict(X_test.iloc[[sample_idx]])[0]
    
    print(f"\nサンプル#{sample_idx}の予測説明:")
    print(f"  基準値（平均予測）: {base_value:.4f}")
    print(f"  予測値: {prediction:.4f}")
    print(f"  実測値: {y_test.iloc[sample_idx]:.4f}")
    
    # Force Plotの表示（Jupyter環境）
    # shap.force_plot(base_value, shap_values_sample, X_test.iloc[sample_idx, :],
    #                 feature_names=feature_cols)
    
    # 7. 特徴量別の平均絶対SHAP値ランキング
    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\n上位20特徴量（平均絶対SHAP値）:")
    print(shap_df.head(20).to_string(index=False))
    
    # 8. 部分依存プロット（Partial Dependence Plot）との比較
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    
    # 上位3特徴量の部分依存プロットを作成
    top_3_features = shap_df.head(3)['feature'].tolist()
    top_3_indices = [feature_cols.index(f) for f in top_3_features]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (feature_idx, feature_name) in enumerate(zip(top_3_indices, top_3_features)):
        pd_result = partial_dependence(
            best_rf, X_test, [feature_idx], grid_resolution=50
        )
        axes[i].plot(pd_result['grid_values'][0], pd_result['average'][0])
        axes[i].set_xlabel(feature_name)
        axes[i].set_ylabel('部分依存')
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('partial_dependence_plots.png', dpi=150)
    print("\n部分依存プロットを partial_dependence_plots.png に保存しました")
    
    # 9. SHAP値の統計的サマリー
    print("\nSHAP値の統計:")
    print(f"  全特徴量の平均絶対SHAP値: {np.abs(shap_values).mean():.4f}")
    print(f"  最大SHAP値: {shap_values.max():.4f}")
    print(f"  最小SHAP値: {shap_values.min():.4f}")
    
    # 正の貢献と負の貢献を分離
    positive_shap = shap_values[shap_values > 0].sum()
    negative_shap = shap_values[shap_values < 0].sum()
    print(f"  正の貢献合計: {positive_shap:.4f}")
    print(f"  負の貢献合計: {negative_shap:.4f}")
    

### 4.4.4 SHAP解析の実践的解釈

SHAPの各可視化手法の解釈方法を説明します： 
    
    
    ```mermaid
    graph TB
                    A[SHAP可視化手法] --> B[Summary Plot]
                    A --> C[Dependence Plot]
                    A --> D[Force Plot]
                    A --> E[Waterfall Plot]
    
                    B --> B1[全特徴量の重要度色=特徴量値]
                    C --> C1[特徴量値とSHAP値の関係相互作用の可視化]
                    D --> D2[個別予測の要因分解基準値からの変化]
                    E --> E1[予測の積み上げグラフ累積的な貢献]
    
                    style B fill:#e3f2fd
                    style C fill:#fff3e0
                    style D fill:#f1f8e9
                    style E fill:#fce4ec
                
    
    💡 SHAP解析のベストプラクティス
    
    Summary Plot: 全体傾向の把握に最適。赤（高値）が予測を上げるか下げるかを確認
    Dependence Plot: 非線形関係の発見。閾値効果や飽和効果を特定
    Force Plot: 個別予測の説明。異常値や予測外れの原因分析
    部分依存プロットとの併用: SHAPは局所的、PDPは大域的な特徴量効果を可視化
    
    
    材料科学での応用例: 形成エネルギー予測で「電気陰性度の差」が高SHAP値を示す場合、イオン結合性が安定性に寄与していることを示唆します。
    ```

## 演習問題

#### 演習1（Easy）: パイプライン構築

Magpie特徴量、StandardScaler、RandomForestRegressorを含むscikit-learnパイプラインを構築し、クロスバリデーションスコア（5-fold、R²）を計算してください。 

解答例を表示
    
    
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from matminer.featurizers.composition import ElementProperty
    
    # 特徴量生成（パイプライン外）
    featurizer = ElementProperty.from_preset('magpie')
    data = featurizer.featurize_dataframe(data, 'composition_obj')
    
    X = data[featurizer.feature_labels()]
    y = data['target']
    
    # パイプライン構築
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # クロスバリデーション
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    print(f"5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    

#### 演習2（Easy）: 基本モデル訓練

XGBoostRegressorをデフォルトパラメータで訓練し、訓練/テストデータのMAE、RMSE、R²を計算してください。 

解答例を表示
    
    
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    # モデル訓練
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # 予測
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    
    # 評価指標
    print("訓練データ:")
    print(f"  MAE:  {mean_absolute_error(y_train, y_pred_train):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
    print(f"  R²:   {r2_score(y_train, y_pred_train):.4f}")
    
    print("\nテストデータ:")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred_test):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
    print(f"  R²:   {r2_score(y_test, y_pred_test):.4f}")
    

#### 演習3（Easy）: クロスバリデーション実装

MLPRegressorに対して3-fold クロスバリデーションを実行し、各foldのR²スコアと平均を表示してください。 

解答例を表示
    
    
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import StandardScaler
    
    # データ標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # MLPモデル
    mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    
    # 3-fold CV
    cv_results = cross_validate(
        mlp, X_scaled, y,
        cv=3,
        scoring='r2',
        return_train_score=True
    )
    
    print("各foldのスコア:")
    for i, (train_score, test_score) in enumerate(zip(
        cv_results['train_score'], cv_results['test_score']
    )):
        print(f"  Fold {i+1}: 訓練R²={train_score:.4f}, テストR²={test_score:.4f}")
    
    print(f"\n平均テストR²: {cv_results['test_score'].mean():.4f} "
          f"± {cv_results['test_score'].std():.4f}")
    

#### 演習4（Medium）: GridSearch最適化

Random Forestのハイパーパラメータ（n_estimators、max_depth、min_samples_split）をGridSearchCVで最適化し、最適パラメータとCV最良スコアを表示してください。 

解答例を表示
    
    
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"最適パラメータ: {grid_search.best_params_}")
    print(f"CV最良R²スコア: {grid_search.best_score_:.4f}")
    
    # テストデータで評価
    test_score = grid_search.best_estimator_.score(X_test, y_test)
    print(f"テストR²スコア: {test_score:.4f}")
    

#### 演習5（Medium）: 学習曲線解釈

学習曲線を描画し、過学習・未学習の判定を行ってください。訓練データサイズを10%, 30%, 50%, 70%, 100%で変化させます。 

解答例を表示
    
    
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    
    train_sizes, train_scores, valid_scores = learning_curve(
        RandomForestRegressor(n_estimators=100, random_state=42),
        X, y,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='訓練スコア')
    plt.plot(train_sizes, valid_scores.mean(axis=1), 's-', label='検証スコア')
    plt.xlabel('訓練サンプル数')
    plt.ylabel('R²スコア')
    plt.title('学習曲線')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 診断
    final_train_score = train_scores.mean(axis=1)[-1]
    final_valid_score = valid_scores.mean(axis=1)[-1]
    gap = final_train_score - final_valid_score
    
    if gap > 0.1:
        diagnosis = "過学習の可能性（訓練スコア >> 検証スコア）"
    elif final_valid_score < 0.7:
        diagnosis = "未学習の可能性（両スコアとも低い）"
    else:
        diagnosis = "良好な汎化性能"
    
    plt.text(0.5, 0.1, f"診断: {diagnosis}",
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.show()
    

#### 演習6（Medium）: SHAP値計算とランキング

訓練済みRandom ForestモデルのSHAP値を計算し、平均絶対SHAP値の上位10特徴量をランキング表示してください。 

解答例を表示
    
    
    import shap
    import numpy as np
    
    # TreeExplainerの作成
    explainer = shap.TreeExplainer(best_rf)
    
    # SHAP値計算
    shap_values = explainer.shap_values(X_test)
    
    # 平均絶対SHAP値でランキング
    shap_importance = np.abs(shap_values).mean(axis=0)
    feature_names = X_test.columns.tolist()
    
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': shap_importance
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("上位10特徴量（平均絶対SHAP値）:")
    print(shap_df.head(10).to_string(index=False))
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.barh(shap_df.head(10)['feature'], shap_df.head(10)['mean_abs_shap'])
    plt.xlabel('平均絶対SHAP値')
    plt.title('特徴量重要度（SHAP）')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    

#### 演習7（Medium）: モデル性能比較

Random Forest、XGBoost、MLPの3モデルを同じデータセットで訓練し、MAE、RMSE、R²を比較表として出力してください。 

解答例を表示
    
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    
    # モデル定義
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20,
                                               random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                    learning_rate=0.1, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500,
                            random_state=42)
    }
    
    # 評価結果を格納
    results = []
    
    for model_name, model in models.items():
        # MLPの場合は標準化
        if model_name == 'MLP':
            scaler = StandardScaler()
            X_train_current = scaler.fit_transform(X_train)
            X_test_current = scaler.transform(X_test)
        else:
            X_train_current = X_train
            X_test_current = X_test
    
        # 訓練と予測
        model.fit(X_train_current, y_train)
        y_pred = model.predict(X_test_current)
    
        # 評価指標
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
    
        results.append({
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })
    
    # 結果表示
    results_df = pd.DataFrame(results)
    print("\nモデル性能比較:")
    print(results_df.to_string(index=False))
    
    # 最良モデルを強調
    best_model_idx = results_df['R²'].idxmax()
    print(f"\n最良モデル: {results_df.loc[best_model_idx, 'Model']}")
    

#### 演習8（Hard）: アンサンブル手法（Stacking）

Random Forest、XGBoost、MLPをベースモデルとし、LinearRegressionをメタモデルとするStackingRegressorを実装してください。単一モデルより性能が向上するか検証してください。 

解答例を表示
    
    
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LinearRegression
    
    # ベースモデルの定義
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=20,
                                     random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                 learning_rate=0.1, random_state=42)),
        ('mlp', Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(hidden_layer_sizes=(100, 50),
                                   max_iter=500, random_state=42))
        ]))
    ]
    
    # メタモデル
    meta_model = LinearRegression()
    
    # StackingRegressorの構築
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
    
    # 訓練
    print("Stackingモデル訓練中...")
    stacking_model.fit(X_train, y_train)
    
    # 評価
    y_pred_stacking = stacking_model.predict(X_test)
    stacking_mae = mean_absolute_error(y_test, y_pred_stacking)
    stacking_rmse = np.sqrt(mean_squared_error(y_test, y_pred_stacking))
    stacking_r2 = r2_score(y_test, y_pred_stacking)
    
    print("\nStacking モデル性能:")
    print(f"  MAE:  {stacking_mae:.4f}")
    print(f"  RMSE: {stacking_rmse:.4f}")
    print(f"  R²:   {stacking_r2:.4f}")
    
    # 単一モデルとの比較
    print("\n性能改善率（vs Random Forest）:")
    rf_r2 = 0.85  # 前の演習結果を仮定
    improvement = (stacking_r2 - rf_r2) / rf_r2 * 100
    print(f"  R²改善: {improvement:.2f}%")
    

#### 演習9（Hard）: Matbenchベンチマーク再現

Matbenchのmatbench_mp_e_form（形成エネルギー）タスクで、組成ベースモデルを訓練し、公式リーダーボードのフォーマットで結果を保存してください。 

解答例を表示
    
    
    from matbench.bench import MatbenchBenchmark
    
    # Matbenchベンチマークの初期化
    mb = MatbenchBenchmark(autoload=False, subset=['matbench_mp_e_form'])
    
    for task in mb.tasks:
        task.load()
    
        # 化学組成変換とMagpie特徴量生成
        str_to_comp = StrToComposition(target_col_id='composition_obj')
        task.df = str_to_comp.featurize_dataframe(task.df, 'composition')
    
        featurizer = ElementProperty.from_preset('magpie')
        task.df = featurizer.featurize_dataframe(task.df, 'composition_obj')
    
        feature_cols = featurizer.feature_labels()
    
        # パイプライン
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=300, max_depth=30,
                                            min_samples_split=5, random_state=42))
        ])
    
        # 5-fold CVで評価
        for fold_idx, fold in enumerate(task.folds):
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            X_train_fold = train_inputs[feature_cols]
            y_train_fold = train_outputs
    
            pipeline.fit(X_train_fold, y_train_fold)
    
            test_inputs, _ = task.get_test_data(fold, include_target=False)
            X_test_fold = test_inputs[feature_cols]
            predictions = pipeline.predict(X_test_fold)
    
            task.record(fold, predictions)
            print(f"Fold {fold_idx + 1} 完了")
    
        # スコア表示
        scores = task.scores
        for metric, values in scores.items():
            print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # 結果保存
    mb.to_file('matbench_eform_results.json')
    print("\n結果を matbench_eform_results.json に保存しました")
    

#### 演習10（Hard）: カスタム評価指標（MAPE）

Mean Absolute Percentage Error（MAPE）をカスタム評価指標として実装し、GridSearchCVで使用してください。MAPEは以下の式で定義されます：  
$$\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$ 

解答例を表示
    
    
    from sklearn.metrics import make_scorer
    
    def mean_absolute_percentage_error(y_true, y_pred):
        """
        MAPE（Mean Absolute Percentage Error）を計算
    
        注意: y_true に0が含まれる場合はエラー回避のため小さい値を追加
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # 0除算回避
        epsilon = 1e-10
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # scikit-learn scorerに変換
    mape_scorer = make_scorer(mean_absolute_percentage_error,
                              greater_is_better=False)  # MAPEは小さいほど良い
    
    # GridSearchCVでカスタム評価指標を使用
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring=mape_scorer,  # カスタム評価指標
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"最適パラメータ: {grid_search.best_params_}")
    print(f"CV最良MAPEスコア: {-grid_search.best_score_:.4f}%")  # 負を反転
    
    # テストデータで評価
    y_pred_test = grid_search.best_estimator_.predict(X_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
    print(f"テストMAPE: {test_mape:.4f}%")
    

## 学習目標達成度チェック

#### 以下の項目を自己評価してください（3段階: 初級/中級/上級）

#### 初級レベル（基本理解）

  * scikit-learnパイプラインの構築と保存/読み込みができる
  * Random Forest、XGBoost、MLPの特徴と使い分けを説明できる
  * 組成ベースとGNNベースモデルの性能差を理解している
  * クロスバリデーションの目的と実装方法を理解している

#### 中級レベル（実践スキル）

  * GridSearchCVでハイパーパラメータを最適化できる
  * 学習曲線から過学習/未学習を診断できる
  * SHAP値を計算し、特徴量重要度を解釈できる
  * 複数モデルの性能を定量的に比較できる

#### 上級レベル（応用力）

  * StackingRegressorでアンサンブルモデルを構築できる
  * Matbenchベンチマークで評価し、結果を公式フォーマットで保存できる
  * カスタム評価指標を実装し、GridSearchCVに統合できる
  * SHAPのDependence PlotとPartial Dependence Plotを使い分けられる

## まとめ

本章では、組成ベース特徴量を機械学習モデルと統合し、材料特性を高精度に予測する実践的手法を学びました。主要なポイントを振り返ります： 

### 重要ポイント

  * **scikit-learnパイプライン** : 特徴量生成・前処理・モデル訓練を統合し、再現性を確保
  * **モデル選択** : Random Forest（解釈性重視）、XGBoost（精度重視）、MLP（大規模データ）の使い分け
  * **ハイパーパラメータ最適化** : GridSearchCV（網羅的）とRandomizedSearchCV（効率的）の選択
  * **学習曲線分析** : 過学習/未学習の診断とデータ追加効果の予測
  * **組成 vs GNN** : Matbenchで組成ベースはGNNの85-95%の精度を実現、計算コストは1/100以下
  * **SHAP解釈** : 予測の要因を定量化し、物理的洞察を獲得

### 実務での活用指針

#### 推奨ワークフロー

  1. **初期探索** : 組成ベースモデルで10⁵候補を10³候補に絞る（数分）
  2. **構造最適化** : 選定候補のDFT構造緩和（数日）
  3. **精密予測** : GNNで最終候補を選定（数時間）
  4. **実験検証** : 上位10-20候補を合成・測定

このハイブリッドアプローチにより、**計算コストを90%削減** しつつ、高精度な材料探索が実現できます。 

### 次のステップ

第5章では、実際の材料探索プロジェクトに組成ベース特徴量を適用し、新規材料の発見から実験検証までのエンドツーエンドワークフローを学びます。Materials Projectデータベースとの統合、能動学習による効率化、不確実性定量化など、より実践的な内容に進みます。 

## 参考文献

  1. Ward, L., Agrawal, A., Choudhary, A., & Wolverton, C. (2016). "A general-purpose machine learning framework for predicting properties of inorganic materials." _npj Computational Materials_ , 2, 16028, pp. 4-6. <https://doi.org/10.1038/npjcompumats.2016.28>   
（Magpie特徴量の性能ベンチマーク、Random Forestとの統合手法を詳述）
  2. Jha, D., Ward, L., Paul, A., Liao, W., Choudhary, A., Wolverton, C., & Agrawal, A. (2018). "ElemNet: Deep Learning the Chemistry of Materials From Only Elemental Composition." _Scientific Reports_ , 8, 17593, pp. 6-9. <https://doi.org/10.1038/s41598-018-35934-y>   
（組成のみから高精度予測を実現したニューラルネットワーク、Matbenchベンチマーク性能）
  3. Dunn, A., Wang, Q., Ganose, A., Dopp, D., & Jain, A. (2020). "Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm." _npj Computational Materials_ , 6, 138, pp. 1-10. <https://doi.org/10.1038/s41524-020-00406-3>   
（Matbench v0.1の設計思想、13タスクの詳細、標準評価プロトコル）
  4. scikit-learn Documentation: Pipeline and GridSearchCV. <https://scikit-learn.org/stable/>   
（パイプライン構築、ハイパーパラメータ最適化、学習曲線の公式ドキュメント）
  5. XGBoost Documentation. <https://xgboost.readthedocs.io/>   
（XGBoostのパラメータチューニング、早期停止、特徴量重要度の詳細）
  6. Lundberg, S.M., & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." _Advances in Neural Information Processing Systems (NeurIPS)_ , pp. 4765-4774. <https://arxiv.org/abs/1705.07874>   
（SHAP原論文、Shapley値の理論的基礎とTreeExplainerアルゴリズム）
  7. matminer Tutorials: Machine Learning Integration. <https://hackingmaterials.lbl.gov/matminer/>   
（matminerとscikit-learn統合の実践例、DFMLAdaptorの使用法、ベストプラクティス）

[← 第3章: 元素特性データベースとFeaturizer](<chapter-3.html>) [目次へ](<index.html>) [第5章: 実践的材料探索 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
