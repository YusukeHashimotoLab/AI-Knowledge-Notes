---
title: "第4章: AutoMLツールの実践"
chapter_title: "第4章: AutoMLツールの実践"
subtitle: TPOT、Auto-sklearn、H2O AutoMLによる自動機械学習
reading_time: 40-45分
difficulty: 中級
code_examples: 10
exercises: 5
---

# 第4章：AutoMLツールの実践

**学習目標:**

  * TPOTの遺伝的プログラミングアプローチを理解する
  * Auto-sklearnのベイズ最適化とメタ学習を習得する
  * H2O AutoMLでスタックアンサンブルを構築する
  * 各AutoMLツールの特徴と使い分けを理解する
  * プロダクション環境へのデプロイ戦略を学ぶ

**読了時間** : 40-45分

* * *

## 4.1 TPOT (Tree-based Pipeline Optimization Tool)

### 4.1.1 TPOTの概要

**TPOTとは:**  
遺伝的プログラミング（Genetic Programming）を使用して、scikit-learnのパイプライン全体を自動最適化するAutoMLツール。

**開発元:** ペンシルバニア大学（Moore Lab）

**特徴:**

  * 遺伝的アルゴリズムによる探索
  * 前処理からモデル選択まで完全自動化
  * scikit-learn完全互換
  * 生成されたパイプラインコードをPythonコードとしてエクスポート可能

### 4.1.2 遺伝的プログラミングアプローチ

**遺伝的アルゴリズムの流れ:**
    
    
    1. 初期集団生成（ランダムパイプライン作成）
    2. 評価（交差検証スコア）
    3. 選択（上位個体を選択）
    4. 交叉（パイプラインを組み合わせ）
    5. 突然変異（ランダムな変更）
    6. 次世代へ
    7. 2-6を世代数分繰り返す
    

**パイプライン表現:**
    
    
    # 遺伝子型（木構造）
    Pipeline(
        SelectKBest(k=10),
        StandardScaler(),
        RandomForestClassifier(n_estimators=100)
    )
    

### 4.1.3 TPOTの基本的な使い方

**例1: 分類問題の基本例**
    
    
    from tpot import TPOTClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # データセット準備
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # TPOTClassifierの作成
    tpot = TPOTClassifier(
        generations=5,        # 進化の世代数
        population_size=20,   # 各世代の個体数
        cv=5,                 # 交差検証の分割数
        random_state=42,
        verbosity=2,          # 進捗表示レベル
        n_jobs=-1             # 並列処理
    )
    
    # 学習（数分かかる）
    tpot.fit(X_train, y_train)
    
    # 評価
    print(f'Test Accuracy: {tpot.score(X_test, y_test):.4f}')
    
    # 最適パイプラインをPythonコードとして保存
    tpot.export('tpot_iris_pipeline.py')
    

**出力例:**
    
    
    Generation 1 - Current best internal CV score: 0.9666666666666667
    Generation 2 - Current best internal CV score: 0.975
    Generation 3 - Current best internal CV score: 0.975
    Generation 4 - Current best internal CV score: 0.9833333333333333
    Generation 5 - Current best internal CV score: 0.9833333333333333
    
    Best pipeline: RandomForestClassifier(SelectKBest(input_matrix, k=2),
                                          bootstrap=True, n_estimators=100)
    Test Accuracy: 1.0000
    

### 4.1.4 TPOT設定のカスタマイズ

**例2: カスタムTPOT設定**
    
    
    from tpot import TPOTClassifier
    
    # カスタム設定でTPOT作成
    tpot_config = {
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 'log2', None],
            'min_samples_split': [2, 5, 10]
        },
        'sklearn.svm.SVC': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'sklearn.preprocessing.StandardScaler': {},
        'sklearn.feature_selection.SelectKBest': {
            'k': range(1, 11)
        }
    }
    
    tpot = TPOTClassifier(
        config_dict=tpot_config,
        generations=10,
        population_size=50,
        cv=5,
        scoring='f1_weighted',  # 評価指標をF1スコアに
        max_time_mins=30,       # 最大実行時間30分
        random_state=42,
        verbosity=2
    )
    
    tpot.fit(X_train, y_train)
    

### 4.1.5 回帰問題の例

**例3: 回帰問題でのTPOT使用**
    
    
    from tpot import TPOTRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # 回帰データセット生成
    X, y = make_regression(n_samples=1000, n_features=20,
                           n_informative=15, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # TPOTRegressor
    tpot_reg = TPOTRegressor(
        generations=5,
        population_size=20,
        cv=5,
        scoring='neg_mean_squared_error',  # MSEを最小化
        random_state=42,
        verbosity=2
    )
    
    tpot_reg.fit(X_train, y_train)
    
    # 評価
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = tpot_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Test MSE: {mse:.4f}')
    print(f'Test R²: {r2:.4f}')
    
    # パイプライン保存
    tpot_reg.export('tpot_regression_pipeline.py')
    

**エクスポートされたコードの例:**
    
    
    # tpot_regression_pipeline.py
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    # 生成されたパイプライン
    exported_pipeline = make_pipeline(
        StandardScaler(),
        GradientBoostingRegressor(
            alpha=0.9, learning_rate=0.1, loss="squared_error",
            max_depth=3, n_estimators=100
        )
    )
    
    # 使用例
    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)
    

* * *

## 4.2 Auto-sklearn

### 4.2.1 Auto-sklearnの概要

**Auto-sklearnとは:**  
ベイズ最適化、メタ学習、アンサンブル構築を組み合わせた自動機械学習ツール。

**開発元:** フライブルク大学（ドイツ）

**主要技術:**

  1. **ベイズ最適化:** SMAC (Sequential Model-based Algorithm Configuration)
  2. **メタ学習:** 過去のタスクから初期設定を学習
  3. **アンサンブル構築:** 複数モデルを自動的に組み合わせ

### 4.2.2 ベイズ最適化とメタ学習

**ベイズ最適化の流れ:**
    
    
    1. 初期設定でモデルを評価
    2. ガウス過程で性能を予測
    3. Acquisition関数で次の探索点を決定
    4. 評価してガウス過程を更新
    5. 2-4を繰り返す
    

**メタ学習:**  
過去の140+データセットでの最適設定から、類似タスクの良い初期設定を推測
    
    
    メタ知識ベース（140+ tasks）
        ↓
    類似度計算（データセット特徴量）
        ↓
    上位25設定をウォームスタート
        ↓
    ベイズ最適化で微調整
    

### 4.2.3 Auto-sklearnの基本的な使い方

**例4: Auto-sklearn分類**
    
    
    import autosklearn.classification
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # データセット準備
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    
    # Auto-sklearn分類器
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,  # 総実行時間5分
        per_run_time_limit=30,         # 1モデルあたり30秒
        ensemble_size=50,              # アンサンブルサイズ
        ensemble_nbest=200,            # アンサンブル候補数
        initial_configurations_via_metalearning=25,  # メタ学習初期設定数
        seed=42
    )
    
    # 学習
    automl.fit(X_train, y_train)
    
    # 予測と評価
    y_pred = automl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # 学習されたモデルの統計情報
    print(automl.sprint_statistics())
    
    # アンサンブルの詳細
    print(automl.show_models())
    

**出力例:**
    
    
    auto-sklearn results:
      Dataset name: digits
      Metric: accuracy
      Best validation score: 0.9832
      Number of target algorithm runs: 127
      Number of successful target algorithm runs: 115
      Number of crashed target algorithm runs: 8
      Number of target algorithms that exceeded the time limit: 4
      Number of target algorithms that exceeded the memory limit: 0
    
    Test Accuracy: 0.9806
    

### 4.2.4 Auto-sklearn 2.0の新機能

**Auto-sklearn 2.0の改良点:**

  * 実行時間の短縮（従来比50%削減）
  * デフォルト設定の改善
  * Portfolio構築の高速化
  * より効率的なアンサンブル選択

**例5: Auto-sklearn 2.0の使用**
    
    
    from autosklearn.experimental.askl2 import AutoSklearn2Classifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # データ準備
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )
    
    # Auto-sklearn 2.0（より高速）
    automl2 = AutoSklearn2Classifier(
        time_left_for_this_task=120,  # 2分
        seed=42
    )
    
    automl2.fit(X_train, y_train)
    
    # 評価
    from sklearn.metrics import classification_report
    y_pred = automl2.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # CV結果の取得
    cv_results = automl2.cv_results_
    print(f"Best model config: {automl2.get_models_with_weights()}")
    

### 4.2.5 カスタム設定と制約

**例6: モデル候補の制限**
    
    
    import autosklearn.classification
    
    # 使用するアルゴリズムを制限
    automl_custom = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,
        include={
            'classifier': ['random_forest', 'gradient_boosting', 'extra_trees'],
            'feature_preprocessor': ['no_preprocessing', 'pca', 'select_percentile']
        },
        exclude={
            'classifier': ['k_nearest_neighbors'],  # KNNは除外
        },
        seed=42
    )
    
    automl_custom.fit(X_train, y_train)
    

* * *

## 4.3 H2O AutoML

### 4.3.1 H2O AutoMLの概要

**H2O.aiとは:**  
オープンソースの分散機械学習プラットフォーム。大規模データ処理に強い。

**H2O AutoMLの特徴:**

  * 自動的なスタックアンサンブル構築
  * リーダーボード形式での結果表示
  * 大規模データ対応（分散処理）
  * モデル説明性機能（SHAP、PDP）

### 4.3.2 H2O AutoMLの基本的な使い方

**例7: H2O AutoML分類**
    
    
    import h2o
    from h2o.automl import H2OAutoML
    import pandas as pd
    
    # H2Oの初期化
    h2o.init()
    
    # データセット準備（Pandasから変換）
    from sklearn.datasets import load_wine
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    
    # H2O DataFrameに変換
    hf = h2o.H2OFrame(df)
    hf['target'] = hf['target'].asfactor()  # 分類タスクのため
    
    # 訓練/テスト分割
    train, test = hf.split_frame(ratios=[0.8], seed=42)
    
    # AutoML実行
    aml = H2OAutoML(
        max_runtime_secs=300,      # 最大実行時間5分
        max_models=20,              # 最大モデル数
        seed=42,
        sort_metric='AUC',          # 評価指標
        exclude_algos=['DeepLearning']  # ディープラーニングは除外
    )
    
    # 学習（targetが目的変数、残りが説明変数）
    x = hf.columns
    x.remove('target')
    y = 'target'
    
    aml.fit(x=x, y=y, training_frame=train)
    
    # リーダーボード表示
    lb = aml.leaderboard
    print(lb.head(rows=10))
    
    # 最良モデルでの予測
    best_model = aml.leader
    preds = best_model.predict(test)
    print(preds.head())
    
    # モデル性能
    perf = best_model.model_performance(test)
    print(perf)
    

**リーダーボード出力例:**
    
    
                                                  model_id       auc   logloss
    0  StackedEnsemble_AllModels_1_AutoML_1_20241021  0.998876  0.067234
    1  StackedEnsemble_BestOfFamily_1_AutoML_1_20241021  0.997543  0.072156
    2               GBM_1_AutoML_1_20241021_163045  0.996321  0.078432
    3                XRT_1_AutoML_1_20241021_163012  0.995234  0.081245
    4                DRF_1_AutoML_1_20241021_163001  0.993456  0.089321
    

### 4.3.3 スタックアンサンブル

**H2Oのスタッキング戦略:**
    
    
    ベースモデル層:
    - GBM（複数設定）
    - Random Forest
    - XGBoost
    - GLM
    - DeepLearning
    
        ↓ メタ特徴量
    
    メタモデル層:
    - GLM（正則化）
    - GBM
    
        ↓
    
    最終予測
    

**例8: カスタムスタックアンサンブル**
    
    
    from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator
    from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
    
    # ベースモデル1: GBM
    gbm = H2OGradientBoostingEstimator(
        ntrees=50,
        max_depth=5,
        learn_rate=0.1,
        seed=42,
        model_id='gbm_base'
    )
    gbm.train(x=x, y=y, training_frame=train)
    
    # ベースモデル2: Random Forest
    rf = H2ORandomForestEstimator(
        ntrees=50,
        max_depth=10,
        seed=42,
        model_id='rf_base'
    )
    rf.train(x=x, y=y, training_frame=train)
    
    # スタックアンサンブル構築
    ensemble = H2OStackedEnsembleEstimator(
        base_models=[gbm, rf],
        metalearner_algorithm='gbm',
        seed=42
    )
    ensemble.train(x=x, y=y, training_frame=train)
    
    # 評価
    ensemble_perf = ensemble.model_performance(test)
    print(f"Ensemble AUC: {ensemble_perf.auc()}")
    

### 4.3.4 モデル説明性（Explainability）

**例9: SHAP値とPDP可視化**
    
    
    # 最良モデルのSHAP値
    shap_values = best_model.shap_summary_plot(test)
    
    # Partial Dependence Plot（部分依存プロット）
    best_model.partial_plot(
        data=test,
        cols=['alcohol', 'flavanoids'],  # 特徴量名
        plot=True
    )
    
    # 変数重要度
    varimp = best_model.varimp(use_pandas=True)
    print(varimp.head(10))
    
    # Feature Interaction（特徴量相互作用）
    best_model.feature_interaction(max_depth=2)
    

* * *

## 4.4 その他のAutoMLツール

### 4.4.1 Google AutoML

**特徴:**

  * Google Cloud Platform上のマネージドサービス
  * ニューラルアーキテクチャ探索（NAS）を使用
  * 画像、テキスト、表形式データに対応
  * エンタープライズグレードのスケーラビリティ

**主要製品:**

  * AutoML Tables（表形式データ）
  * AutoML Vision（画像分類）
  * AutoML Natural Language（テキスト分類）
  * Vertex AI（統合プラットフォーム）

### 4.4.2 Azure AutoML

**特徴:**

  * Azure Machine Learning Studioに統合
  * コードレスUI + Pythonライブラリ
  * モデル説明性機能が充実
  * MLOpsパイプライン統合

### 4.4.3 PyCaret

**PyCaretとは:**  
Pythonのローコード機械学習ライブラリ。わずか数行でAutoMLを実行可能。

**例10: PyCaret使用例**
    
    
    from pycaret.classification import *
    import pandas as pd
    from sklearn.datasets import load_iris
    
    # データセット準備
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # PyCaret環境セットアップ
    clf_setup = setup(
        data=df,
        target='target',
        train_size=0.8,
        session_id=42,
        verbose=False
    )
    
    # 全モデル比較（自動）
    best_models = compare_models(n_select=3)  # 上位3モデル
    
    # 最良モデルの詳細評価
    best = best_models[0]
    evaluate_model(best)
    
    # ハイパーパラメータチューニング
    tuned_best = tune_model(best, n_iter=50)
    
    # アンサンブル
    bagged = ensemble_model(tuned_best, method='Bagging')
    boosted = ensemble_model(tuned_best, method='Boosting')
    
    # スタッキング
    stacked = stack_models(estimator_list=best_models[:3])
    
    # モデル保存
    save_model(stacked, 'pycaret_final_model')
    
    # 新データでの予測
    predictions = predict_model(stacked, data=df)
    print(predictions.head())
    

### 4.4.4 Ludwig

**Ludwigとは:**  
Uberが開発したコードレス深層学習ツールボックス。YAML設定ファイルでモデル構築。

**特徴:**

  * 宣言的なモデル定義（YAMLベース）
  * 多様なデータタイプ対応（画像、テキスト、表形式の混在可）
  * AutoMLモード搭載
  * 転移学習サポート

### 4.4.5 AutoMLツール比較表

ツール | 最適化手法 | 実行速度 | スケーラビリティ | 説明性 | 学習曲線 | ベストユースケース  
---|---|---|---|---|---|---  
**TPOT** | 遺伝的プログラミング | 中 | 中 | 高（コードエクスポート） | 低 | 中規模データ、パイプライン自動化  
**Auto-sklearn** | ベイズ最適化+メタ学習 | 中〜高 | 中 | 中 | 低 | 学術研究、ベンチマーク  
**H2O AutoML** | グリッドサーチ+スタッキング | 高 | 高 | 高（SHAP統合） | 中 | 大規模データ、プロダクション  
**PyCaret** | 複数手法の組み合わせ | 高 | 中 | 高 | 極低 | 迅速なプロトタイピング  
**Google AutoML** | NAS（ニューラルアーキテクチャ探索） | 高 | 極高 | 中 | 低 | クラウドベース大規模タスク  
**Azure AutoML** | 複数手法のハイブリッド | 高 | 高 | 極高 | 低 | エンタープライズMLOps  
**Ludwig** | ハイパーパラメータ探索 | 中 | 中 | 中 | 中 | マルチモーダル深層学習  
  
* * *

## 4.5 AutoMLのベストプラクティス

### 4.5.1 ツール選択基準

**データサイズによる選択:**

  * **小規模（ <10,000サンプル）:** TPOT、Auto-sklearn
  * **中規模（10,000-1,000,000）:** H2O AutoML、PyCaret
  * **大規模（ >1,000,000）:** H2O AutoML（分散モード）、Google/Azure AutoML

**タスクタイプによる選択:**

  * **表形式データ:** TPOT、Auto-sklearn、H2O、PyCaret
  * **画像・テキスト:** Google AutoML、Ludwig
  * **時系列:** Auto-sklearn、H2O、PyCaret
  * **マルチモーダル:** Ludwig

**実行時間制約:**

  * **短時間（ <10分）:** PyCaret、Auto-sklearn 2.0
  * **中時間（10分〜1時間）:** TPOT、H2O AutoML
  * **長時間OK（ >1時間）:** すべて可（より深い探索）

### 4.5.2 カスタマイズ vs フルオートメーション

**フルオートメーションが適している場合:**

  * 初期ベースライン作成
  * ドメイン知識が限られている
  * 迅速なプロトタイピング
  * 複数データセットの一括処理

**カスタマイズが必要な場合:**

  * ドメイン特有の前処理が必要
  * 特定のモデルファミリーに制限したい
  * カスタム評価指標を使用
  * 解釈可能性が最優先

**ハイブリッドアプローチ:**
    
    
    # 1. AutoMLでベースライン作成
    tpot.fit(X_train, y_train)
    baseline_score = tpot.score(X_test, y_test)
    
    # 2. エクスポートされたパイプラインを手動で改良
    from tpot_exported_pipeline import exported_pipeline
    pipeline = exported_pipeline
    
    # 3. ドメイン知識を追加
    from sklearn.preprocessing import FunctionTransformer
    
    def domain_specific_transform(X):
        # カスタム変換
        return X
    
    pipeline.steps.insert(
        0, ('domain_transform', FunctionTransformer(domain_specific_transform))
    )
    
    # 4. 再評価
    pipeline.fit(X_train, y_train)
    improved_score = pipeline.score(X_test, y_test)
    print(f'Baseline: {baseline_score:.4f}, Improved: {improved_score:.4f}')
    

### 4.5.3 プロダクション環境へのデプロイ

**デプロイ時の考慮事項:**

  1. **モデルサイズと推論速度**

     * アンサンブルモデルは高精度だが重い
     * 推論速度要件に応じてモデル選択
  2. **依存関係管理**

     * AutoMLツールの依存ライブラリをプロダクション環境に含める
     * Dockerコンテナ化推奨
  3. **バージョン管理**

     * モデルとパイプラインのバージョニング
     * MLflow、DVC等のMLOpsツール使用
  4. **モニタリング**

     * データドリフト検出
     * モデル性能のトラッキング
     * 再学習トリガー設定

**デプロイ例（Flask API）:**
    
    
    # app.py
    from flask import Flask, request, jsonify
    import joblib
    import numpy as np
    
    app = Flask(__name__)
    
    # モデルロード
    model = joblib.load('tpot_model.pkl')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        probability = model.predict_proba(features)
    
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()
        })
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)
    

### 4.5.4 コスト・時間管理

**計算コスト削減戦略:**

  1. **Early Stopping:**

     * 改善が見られない場合は早期終了
     * `max_time_mins`、`max_models`パラメータ設定
  2. **並列処理:**

     * `n_jobs=-1`で全CPUコア使用
     * クラウドの場合、適切なインスタンスタイプ選択
  3. **データサンプリング:**

     * 初期探索は小サンプルで実施
     * 有望な設定を見つけたら全データで再学習
  4. **段階的アプローチ:**

    
    
    # ステージ1: 高速探索（10分）
    quick_automl = TPOTClassifier(
        generations=3,
        population_size=10,
        max_time_mins=10
    )
    quick_automl.fit(X_train_sample, y_train_sample)
    
    # ステージ2: 詳細探索（1時間）
    if quick_automl.score(X_val, y_val) > 0.85:  # 閾値超えた場合のみ
        deep_automl = TPOTClassifier(
            generations=20,
            population_size=50,
            max_time_mins=60
        )
        deep_automl.fit(X_train, y_train)
    

**クラウドコスト管理:**

  * **Spot/Preemptible Instances:** コスト70%削減可能
  * **Auto Scaling:** 必要時のみリソース使用
  * **予算アラート:** 上限設定で予想外のコスト回避

* * *

## 4.6 まとめ

### 学んだこと

  1. **TPOT:**

     * 遺伝的プログラミングでパイプライン全体を最適化
     * Pythonコードエクスポートで透明性が高い
     * 中規模データでの探索に適している
  2. **Auto-sklearn:**

     * ベイズ最適化とメタ学習で効率的探索
     * アンサンブル自動構築
     * 学術的にも広く使用されている
  3. **H2O AutoML:**

     * 大規模データに強い
     * リーダーボードで結果比較が容易
     * モデル説明性機能が充実
  4. **ツール選択基準:**

     * データサイズ、タスクタイプ、時間制約を考慮
     * フルオートとカスタマイズのバランス
     * プロダクション要件（速度、サイズ、依存性）
  5. **ベストプラクティス:**

     * 段階的アプローチでコスト削減
     * デプロイ時のモニタリング設計
     * MLOpsツールとの統合

### 次のステップ

第5章では、特徴量エンジニアリングの自動化とFeature Toolsの使用方法を学びます：

  * 自動特徴量生成の理論
  * Feature Toolsによる深層特徴量合成
  * 時系列データの自動特徴量抽出
  * 特徴量選択の自動化

* * *

## 演習問題

**問1:** TPOTの遺伝的プログラミングアプローチにおける「交叉」と「突然変異」の役割を説明し、それぞれがパイプライン最適化にどう貢献するか述べよ。

**問2:** Auto-sklearnのメタ学習がコールドスタート問題をどのように解決するか説明せよ。また、メタ学習が効果的でない場合はどのような状況か考察せよ。

**問3:** H2O AutoMLのスタックアンサンブルと単一モデルの性能を比較する実験を設計せよ。どのようなデータセットでスタッキングの効果が最大化されるか述べよ。

**問4:** 以下のシナリオに最適なAutoMLツールを選択し、理由を説明せよ：  
(a) 10,000サンプルの医療診断データ、高い解釈可能性が必要  
(b) 10億サンプルのクリックログデータ、推論速度が重要  
(c) 画像とテキストの混在データ、迅速なプロトタイピング

**問5:** AutoMLモデルをプロダクション環境にデプロイする際の5つの主要な考慮事項を挙げ、それぞれの対策を具体的に述べよ（600字以内）。

* * *

## 参考文献

  1. Olson, R. S. et al. "TPOT: A Tree-based Pipeline Optimization Tool for Automating Machine Learning." _AutoML Workshop at ICML_ (2016).
  2. Feurer, M. et al. "Efficient and Robust Automated Machine Learning." _NIPS_ (2015).
  3. LeDell, E. & Poirier, S. "H2O AutoML: Scalable Automatic Machine Learning." _AutoML Workshop at ICML_ (2020).
  4. Hutter, F. et al. "Sequential Model-Based Optimization for General Algorithm Configuration." _LION_ (2011).
  5. Molnar, C. _Interpretable Machine Learning: A Guide for Making Black Box Models Explainable._ (2022).
  6. Lundberg, S. M. & Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." _NIPS_ (2017).
  7. He, X. et al. "AutoML: A Survey of the State-of-the-Art." _Knowledge-Based Systems_ (2021).

* * *

**次章** : 第5章：特徴量エンジニアリングの自動化（準備中）

**ライセンス** : このコンテンツはCC BY 4.0ライセンスの下で提供されています。
