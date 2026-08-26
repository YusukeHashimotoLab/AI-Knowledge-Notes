---
title: 第5章：Python実践：matminerワークフロー
chapter_title: 第5章：Python実践：matminerワークフロー
subtitle: エンドツーエンドの材料探索パイプライン構築
---

🌐 JP | [🇬🇧 EN](<../../../en/MI/composition-features-introduction/chapter-5.html>) | Last sync: 2025-11-16

## 本章の学習目標

この最終章では、Chapter 1-4で学んだ全ての知識を統合し、実際の材料探索プロジェクトで使用できる完全なワークフローを構築します。

### 学習目標

  * **基本理解** : Materials Project API、AutoFeaturizer、完全なMLパイプライン構成
  * **実践スキル** : データ取得→特徴量生成→モデル訓練→予測→可視化の実装、joblib保存/読み込み
  * **応用力** : 新規材料予測、エラー分析とモデル改善、バッチ予測システム構築

## 5.1 Materials Project APIデータ取得

Materials Projectは、DFT計算に基づく150,000以上の材料データを提供する世界最大級のオープン材料データベースです。

### 5.1.1 API Key取得と認証

Materials Project APIを使用するには、無料のAPI keyが必要です：

  1. [Materials Project](<https://materialsproject.org/>)にアクセス
  2. 右上の「Sign Up」からアカウント作成
  3. ログイン後、「Dashboard」→「API」からAPI keyを取得

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/your-repo/composition-features/blob/main/chapter5_example1.ipynb>)

#### Example 1: Materials Project APIデータ取得（10,000化合物）
    
    
    # ===================================
    # Example 1: Materials Project APIデータ取得
    # ===================================
    
    # 必要なライブラリのインポート
    from mp_api.client import MPRester
    from pymatgen.core import Composition
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    
    # API keyの設定（ご自身のkeyに置き換えてください）
    API_KEY = "your_api_key_here"
    
    def fetch_materials_data(api_key, max_compounds=10000):
        """Materials Projectから材料データを取得
        
        Args:
            api_key (str): Materials Project API key
            max_compounds (int): 取得する最大化合物数
            
        Returns:
            pd.DataFrame: 材料データ（化学式、形成エネルギー、バンドギャップ等）
        """
        with MPRester(api_key) as mpr:
            # 形成エネルギーとバンドギャップデータを取得
            # 安定性基準: エネルギーが凸包上または近傍（e_above_hull < 0.1 eV/atom）
            docs = mpr.materials.summary.search(
                energy_above_hull=(0, 0.1),  # 準安定材料を含む
                fields=["material_id", "formula_pretty", "formation_energy_per_atom",
                       "band_gap", "elements", "nelements"],
                num_chunks=10,
                chunk_size=1000
            )
            
            # DataFrameに変換
            data = []
            for doc in docs[:max_compounds]:
                data.append({
                    'material_id': doc.material_id,
                    'formula': doc.formula_pretty,
                    'formation_energy': doc.formation_energy_per_atom,
                    'band_gap': doc.band_gap,
                    'elements': ' '.join(doc.elements),
                    'n_elements': doc.nelements
                })
            
            df = pd.DataFrame(data)
            return df
    
    # データ取得実行
    df = fetch_materials_data(API_KEY, max_compounds=10000)
    
    print(f"取得データ数: {len(df)}")
    print(f"\n最初の5行:")
    print(df.head())
    
    # 統計情報
    print(f"\n形成エネルギー範囲: {df['formation_energy'].min():.3f} ~ {df['formation_energy'].max():.3f} eV/atom")
    print(f"バンドギャップ範囲: {df['band_gap'].min():.3f} ~ {df['band_gap'].max():.3f} eV")
    print(f"元素数分布:\n{df['n_elements'].value_counts().sort_index()}")
    
    # 期待される出力:
    # 取得データ数: 10000
    # 最初の5行:
    #   material_id formula  formation_energy  band_gap  ...
    # 0  mp-1234    Fe2O3    -2.543           2.18       ...
    # 1  mp-5678    TiO2     -4.889           3.25       ...
    # ...
    #
    # 形成エネルギー範囲: -5.234 ~ 0.099 eV/atom
    # バンドギャップ範囲: 0.000 ~ 9.876 eV
    # 元素数分布:
    # 2    3456
    # 3    4123
    # 4    1892
    # 5    529
    

## 5.2 AutoFeaturizerによる自動特徴量生成

matminerの`AutoFeaturizer`は、化学組成または結晶構造を自動判定し、適切な特徴量を生成します。

### 5.2.1 AutoFeaturizerの仕組み

  * **preset選択** : 
    * `express`: 高速（22特徴量、10秒/1000化合物）
    * `fast`: 中速（50特徴量、30秒/1000化合物）
    * `all`: 完全（132特徴量、120秒/1000化合物）
  * **欠損値処理** : DataCleanerで自動処理
  * **特徴量選択** : VarianceThreshold、FeatureAgglomeration統合可能

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/your-repo/composition-features/blob/main/chapter5_example2.ipynb>)

#### Example 2: AutoFeaturizer活用（preset='express'）
    
    
    # ===================================
    # Example 2: AutoFeaturizer活用
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.base import MultipleFeaturizer
    import time
    
    # 化学式文字列をCompositionオブジェクトに変換
    df = StrToComposition().featurize_dataframe(df, 'formula')
    
    # AutoFeaturizerの代わりに、expressプリセット相当を手動構築
    # (実際のAutoFeaturizerはpresetに応じて最適なFeaturizerを自動選択)
    featurizer = ElementProperty.from_preset("magpie")
    
    # 特徴量生成
    start_time = time.time()
    df = featurizer.featurize_dataframe(df, col_id='composition', ignore_errors=True)
    elapsed = time.time() - start_time
    
    print(f"特徴量生成完了: {elapsed:.2f}秒")
    print(f"生成特徴量数: {len(featurizer.feature_labels())}")
    print(f"特徴量名（最初の10個）:\n{featurizer.feature_labels()[:10]}")
    
    # DataCleanerで欠損値処理
    from matminer.utils.data import MixingInfoError
    # 欠損値を含む行を除去（本番環境では補完も検討）
    df_clean = df.dropna()
    print(f"\n欠損値処理後: {len(df_clean)}行（元: {len(df)}行）")
    
    # 期待される出力:
    # 特徴量生成完了: 8.54秒
    # 生成特徴量数: 132
    # 特徴量名（最初の10個）:
    # ['MagpieData minimum Number', 'MagpieData maximum Number', ...]
    #
    # 欠損値処理後: 9876行（元: 10000行）
    

## 5.3 完全なMLパイプライン構築

scikit-learnのPipelineを活用し、データ取得から予測までを一貫したワークフローにします。
    
    
    ```mermaid
    graph LR
        A[データ取得MP API] --> B[特徴量生成matminer]
        B --> C[前処理StandardScaler]
        C --> D[モデル訓練RandomForest]
        D --> E[評価R², MAE]
        E --> F{性能OK?}
        F -->|Yes| G[モデル保存joblib]
        F -->|No| H[ハイパーパラメータ最適化]
        H --> D
        
        style A fill:#e3f2fd
        style G fill:#e8f5e9
        style F fill:#fff3e0
    ```

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/your-repo/composition-features/blob/main/chapter5_example3.ipynb>)

#### Example 3: 完全なMLパイプライン（データ→モデル→予測）
    
    
    # ===================================
    # Example 3: 完全なMLパイプライン
    # ===================================
    
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np
    
    # 特徴量とターゲットの分離
    feature_cols = [col for col in df_clean.columns 
                    if col.startswith('MagpieData')]
    X = df_clean[feature_cols].values
    y = df_clean['formation_energy'].values
    
    print(f"特徴量行列: {X.shape}")
    print(f"ターゲット: {y.shape}")
    
    # 訓練/テストデータ分割（80/20）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Pipeline構築
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # モデル訓練
    print("\nモデル訓練中...")
    pipeline.fit(X_train, y_train)
    
    # 予測と評価
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n=== 性能評価 ===")
    print(f"MAE: {mae:.4f} eV/atom")
    print(f"R²:  {r2:.4f}")
    
    # クロスバリデーション（5分割）
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=5, scoring='neg_mean_absolute_error'
    )
    print(f"\nCV MAE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f} eV/atom")
    
    # 期待される出力:
    # 特徴量行列: (9876, 132)
    # ターゲット: (9876,)
    #
    # モデル訓練中...
    #
    # === 性能評価 ===
    # MAE: 0.1234 eV/atom
    # R²:  0.8976
    #
    # CV MAE: 0.1298 ± 0.0087 eV/atom
    

## 5.4 モデル保存と読み込み

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/your-repo/composition-features/blob/main/chapter5_example4.ipynb>)

#### Example 4: モデル保存と読み込み（joblib）
    
    
    # ===================================
    # Example 4: モデル保存と読み込み
    # ===================================
    
    import joblib
    from pathlib import Path
    
    # モデル保存
    model_path = Path('composition_formation_energy_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"モデル保存完了: {model_path}")
    print(f"ファイルサイズ: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # モデル読み込み
    loaded_pipeline = joblib.load(model_path)
    print("\nモデル読み込み完了")
    
    # 読み込んだモデルで予測（検証）
    y_pred_loaded = loaded_pipeline.predict(X_test[:5])
    y_pred_original = pipeline.predict(X_test[:5])
    
    print("\n予測値比較（最初の5サンプル）:")
    print("元のモデル:    ", y_pred_original)
    print("読み込みモデル:", y_pred_loaded)
    print("一致:", np.allclose(y_pred_original, y_pred_loaded))
    
    # 期待される出力:
    # モデル保存完了: composition_formation_energy_model.pkl
    # ファイルサイズ: 24.56 MB
    #
    # モデル読み込み完了
    #
    # 予測値比較（最初の5サンプル）:
    # 元のモデル:     [-2.543 -4.889 -1.234 -3.456 -0.987]
    # 読み込みモデル: [-2.543 -4.889 -1.234 -3.456 -0.987]
    # 一致: True
    

## 5.5 新規材料予測と可視化

[ ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](<https://colab.research.google.com/github/your-repo/composition-features/blob/main/chapter5_example4.ipynb>)

#### Example 4: モデル保存と読み込み（joblib）
    
    
    # ===================================
    # Example 4: モデル保存と読み込み
    # ===================================
    
    import joblib
    from pathlib import Path
    
    # モデル保存
    model_path = Path('composition_formation_energy_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"モデル保存完了: {model_path}")
    print(f"ファイルサイズ: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # モデル読み込み
    loaded_pipeline = joblib.load(model_path)
    print("\nモデル読み込み完了")
    
    # 読み込んだモデルで予測（検証）
    y_pred_loaded = loaded_pipeline.predict(X_test[:5])
    y_pred_original = pipeline.predict(X_test[:5])
    
    print("\n予測値比較（最初の5サンプル）:")
    print("元のモデル:    ", y_pred_original)
    print("読み込みモデル:", y_pred_loaded)
    print("一致:", np.allclose(y_pred_original, y_pred_loaded))
    
    # 期待される出力:
    # モデル保存完了: composition_formation_energy_model.pkl
    # ファイルサイズ: 24.56 MB
    #
    # モデル読み込み完了
    #
    # 予測値比較（最初の5サンプル）:
    # 元のモデル:     [-2.543 -4.889 -1.234 -3.456 -0.987]
    # 読み込みモデル: [-2.543 -4.889 -1.234 -3.456 -0.987]
    # 一致: True
    

## 5.5 新規材料予測と可視化

訓練したモデルを使用して、未知の材料の特性を予測します。Random Forestの場合、全決定木の予測分布から不確実性も推定できます。

## 学習目標の確認

この章を完了すると、以下を実行できるようになります：

### 基本理解

  * ✅ Materials Project APIの使用方法を理解できる
  * ✅ AutoFeaturizerの仕組みとpreset選択を説明できる
  * ✅ 完全なMLパイプラインの構成要素を列挙できる

### 実践スキル

  * ✅ MP APIで10,000化合物データを取得できる
  * ✅ matminerで特徴量を自動生成できる
  * ✅ scikit-learn Pipelineで訓練→評価→保存を実行できる
  * ✅ joblibでモデルの保存と読み込みができる
  * ✅ 新規材料に対して予測を実行できる

### 応用力

  * ✅ 実際の材料探索プロジェクトを設計できる
  * ✅ エラー分析からモデル改善策を提案できる
  * ✅ バッチ予測システムを構築できる

## 演習問題

### Easy（基礎確認）

**Q1** : Materials Project APIで酸化物（O含有）のみを取得する方法は？

**正解** :
    
    
    docs = mpr.materials.summary.search(
        elements=["O"],  # O含有
        energy_above_hull=(0, 0.1),
        fields=["material_id", "formula_pretty", ...]
    )

**解説** : `elements`パラメータで特定元素を含む材料に絞り込みます。

**Q2** : AutoFeaturizerのpreset='fast'と'all'の違いは？いつ'all'を使うべきか説明してください。

**正解** :

**preset='fast'** : 約50特徴量、30秒/1000化合物  
**preset='all'** : 132特徴量（Magpie完全版）、120秒/1000化合物

**'all'を使うべき場合** :

  * データ数が十分（>10,000サンプル）で過学習リスクが低い
  * 精度が最優先（計算時間は許容）
  * 特徴量重要度分析で最適な記述子を探索したい

**解説** : preset='express'（22特徴量）→ 'fast'（50特徴量）→ 'all'（132特徴量）と、特徴量数が増えるほど精度向上の可能性がある一方、計算時間と過学習リスクも増加します。

**Q3** : Materials Project APIで「準安定材料」（energy_above_hull < 0.1 eV/atom）も取得する理由は？

**正解** : 準安定材料は熱力学的に完全安定ではないが、実用上合成可能で、有用な特性を持つことが多いため。

**詳細解説** :

  * **energy_above_hull = 0** : 完全安定（凸包上）
  * **0 < e_above_hull < 0.1**: 準安定（実用的に合成可能）
  * **e_above_hull > 0.1**: 不安定（合成困難）

**実例** : ダイヤモンド（e_above_hull ≈ 0.0025 eV/atom）は準安定だが、実用材料として重要。
    
    
    # 安定性別のデータ取得例
    stable_only = mpr.materials.summary.search(
        energy_above_hull=(0, 0.001),  # 完全安定のみ
        fields=["material_id", "formula_pretty"]
    )
    
    metastable = mpr.materials.summary.search(
        energy_above_hull=(0.001, 0.1),  # 準安定材料
        fields=["material_id", "formula_pretty"]
    )
    
    print(f"完全安定材料: {len(stable_only)}")
    print(f"準安定材料: {len(metastable)}")
    # 出力例: 完全安定材料: 15,234 / 準安定材料: 42,156
    

**Q4** : scikit-learn Pipelineにmatminer Featurizerを統合する際、`ignore_errors=True`を使う理由は？

**正解** : 一部の化学式で特徴量生成に失敗しても、全体のパイプライン実行を中断せずに継続するため。

**詳細解説** :

matminer Featurizerは、以下の場合にエラーを発生させる可能性があります：

  * 未知の元素（超重元素等）
  * 元素特性データの欠損
  * 無効な酸化状態

`ignore_errors=True`を設定すると、エラーが発生した行にはNaNが入り、後で`dropna()`で除去できます。
    
    
    # エラーハンドリングの比較
    # ❌ 悪い例: ignore_errors=False（デフォルト）
    df = featurizer.featurize_dataframe(df, 'composition')
    # → 1つのエラーで全体が停止
    
    # ✅ 良い例: ignore_errors=True
    df = featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
    df_clean = df.dropna()  # エラー行を除去
    print(f"成功: {len(df_clean)} / 全体: {len(df)}")
    

### Medium（応用）

**Q5** : Random Forestの全決定木予測から不確実性を推定する方法を実装し、信頼度が低い（std > 0.3）予測を特定してください。

**解答例** :
    
    
    # Random Forest不確実性推定
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # 訓練済みRandom Forestモデル
    rf_model = pipeline.named_steps['model']
    
    # 全決定木で予測
    tree_predictions = np.array([
        tree.predict(pipeline.named_steps['scaler'].transform(X_test))
        for tree in rf_model.estimators_
    ])
    
    # 予測の平均と標準偏差
    y_pred_mean = tree_predictions.mean(axis=0)
    y_pred_std = tree_predictions.std(axis=0)
    
    # 信頼度分類
    low_confidence_mask = y_pred_std > 0.3
    high_confidence_mask = y_pred_std <= 0.3
    
    print(f"高信頼度予測: {high_confidence_mask.sum()} サンプル")
    print(f"低信頼度予測: {low_confidence_mask.sum()} サンプル")
    
    # 低信頼度サンプルの分析
    low_conf_indices = np.where(low_confidence_mask)[0]
    for idx in low_conf_indices[:5]:
        print(f"\\nサンプル {idx}:")
        print(f"  予測値: {y_pred_mean[idx]:.3f} ± {y_pred_std[idx]:.3f}")
        print(f"  実測値: {y_test[idx]:.3f}")
        print(f"  化学式: {df_clean.iloc[idx]['formula']}")
    
    # 期待される出力:
    # 高信頼度予測: 1,856 サンプル
    # 低信頼度予測: 119 サンプル
    #
    # サンプル 23:
    #   予測値: -1.234 ± 0.456
    #   実測値: -0.987
    #   化学式: CuFeO2
    

**解説** : 標準偏差が大きい予測は、訓練データに類似サンプルが少ない、または複雑な材料系であることを示唆します。このような予測には追加の実験検証が推奨されます。

**Q6** : matminerのFeaturizerをscikit-learn Pipelineに統合する際、`StandardScaler`を使う理由と、その位置（Featurizerの前 or 後）を説明してください。

**正解** : Featurizerの**後**（特徴量生成後）に配置し、特徴量のスケールを正規化するため。

**詳細解説** :

matminer Featurizerは元素特性から統計量（平均、最大、最小等）を計算します。これらは異なるスケールを持つため：

  * 原子番号: 1-118
  * 原子半径: 30-300 pm
  * 電気陰性度: 0.7-4.0
  * 融点: 14-3,683 K

StandardScalerで正規化（平均0、標準偏差1）することで：

  1. スケールの大きい特徴量が支配的になるのを防ぐ
  2. 勾配降下法の収束を高速化（Neural Network等）
  3. 距離ベースのアルゴリズム（KNN、SVM）の性能向上

    
    
    # ✅ 正しいPipeline構成
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    
    pipeline = Pipeline([
        # ('featurizer', matminer_featurizer),  # 通常はDataFrame段階で実行
        ('scaler', StandardScaler()),  # Featurizer後に正規化
        ('model', RandomForestRegressor())
    ])
    
    # ❌ 間違い: Featurizerの前にScaler
    # → Composition objectはスケールできない
    

**Q7** : 10,000化合物のMatbenchデータセットで、Chapter 4のMatbenchベンチマーク表（組成ベース vs GNN）を再現してください。Random Forestで形成エネルギー予測のMAEを計算し、CGCNN（MAE=0.039）と比較してください。

**解答例** :
    
    
    # Matbenchベンチマーク再現
    from matminer.featurizers.composition import ElementProperty
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np
    
    # Matbench形成エネルギーデータセット（簡略版、実際はmatbenchからロード）
    # from matbench.bench import MatbenchBenchmark
    # mb = MatbenchBenchmark(autoload=False)
    # task = mb.matbench_mp_e_form
    
    # 簡略化のため、MP APIから10,000化合物取得（前述のコード）
    # df, X, y = ... (前のセクションで取得済み)
    
    # Pipeline構築（Magpie特徴量 + Random Forest）
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # 5-fold Cross Validation
    cv_scores = cross_val_score(
        pipeline, X, y,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    mae_composition = -cv_scores.mean()
    mae_std = cv_scores.std()
    
    print("=== Matbenchベンチマーク比較 ===")
    print(f"組成ベース（Magpie + RF）: MAE = {mae_composition:.3f} ± {mae_std:.3f} eV/atom")
    print(f"GNN（CGCNN）:              MAE = 0.039 eV/atom（文献値）")
    print(f"\\n性能比: CGCNN / Composition = {mae_composition / 0.039:.2f}x 高精度")
    
    # 期待される出力:
    # === Matbenchベンチマーク比較 ===
    # 組成ベース（Magpie + RF）: MAE = 0.124 ± 0.008 eV/atom
    # GNN（CGCNN）:              MAE = 0.039 eV/atom（文献値）
    #
    # 性能比: CGCNN / Composition = 3.18x 高精度
    

**解説** : 組成ベース特徴量のみでも実用的な予測（MAE ~0.12 eV/atom）が可能ですが、結晶構造情報を活用するGNNは約3倍の精度を達成します。ただし、組成ベースは：

  * ✅ 結晶構造不要（未知構造材料にも適用可能）
  * ✅ 計算速度が速い（GNNの10-100倍）
  * ✅ 解釈性が高い（特徴量重要度分析が容易）

用途に応じて使い分けることが重要です。

### Hard（発展）

**Q8** : High-Entropy Alloy（HEA、例: CoCrFeNiMn）の形成エネルギーを予測し、各元素を1つずつ置換した場合（例: CuCrFeNiMn）の予測変化を分析してください。

**解答例** :
    
    
    # High-Entropy Alloy分析
    from pymatgen.core import Composition
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 元のHEA
    base_hea = "CoCrFeNiMn"
    base_comp = Composition(base_hea)
    
    # 置換候補元素
    substitute_elements = ['Cu', 'Ti', 'V', 'Zn', 'Mo', 'W', 'Al']
    
    # 各元素を順番に置換
    results = []
    for original_elem in ['Co', 'Cr', 'Fe', 'Ni', 'Mn']:
        for sub_elem in substitute_elements:
            # 置換HEAの生成
            new_formula = base_hea.replace(original_elem, sub_elem, 1)
            new_comp = Composition(new_formula)
            
            # 特徴量生成と予測
            feat_dict = {'composition': new_comp}
            feat_df = pd.DataFrame([feat_dict])
            feat_df = featurizer.featurize_dataframe(feat_df, 'composition', ignore_errors=True)
            
            if not feat_df.dropna().empty:
                X_pred = feat_df[feature_cols].values
                y_pred = loaded_pipeline.predict(X_pred)[0]
                
                results.append({
                    'Original': original_elem,
                    'Substitute': sub_elem,
                    'Formula': new_formula,
                    'Predicted_Hf': y_pred
                })
    
    # 元のHEA予測
    base_feat_df = pd.DataFrame([{'composition': base_comp}])
    base_feat_df = featurizer.featurize_dataframe(base_feat_df, 'composition')
    X_base = base_feat_df[feature_cols].values
    y_base = loaded_pipeline.predict(X_base)[0]
    
    results_df = pd.DataFrame(results)
    results_df['Delta_Hf'] = results_df['Predicted_Hf'] - y_base
    
    # 最も安定化する置換
    best_substitutions = results_df.nsmallest(5, 'Delta_Hf')
    print("=== 最も安定化する元素置換（Top 5）===")
    print(best_substitutions[['Original', 'Substitute', 'Formula', 'Delta_Hf']])
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    for original in ['Co', 'Cr', 'Fe', 'Ni', 'Mn']:
        subset = results_df[results_df['Original'] == original]
        ax.plot(subset['Substitute'], subset['Delta_Hf'], 
                marker='o', label=f'{original}置換')
    
    ax.axhline(y=0, color='r', linestyle='--', label='元のHEA')
    ax.set_xlabel('置換元素')
    ax.set_ylabel('Δ形成エネルギー (eV/atom)')
    ax.set_title('HEA元素置換の効果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hea_substitution_analysis.png', dpi=300)
    
    # 期待される出力:
    # === 最も安定化する元素置換（Top 5）===
    #   Original Substitute    Formula  Delta_Hf
    # 12       Cr          Cu  CoCuFeNiMn  -0.234
    # 18       Fe          Al  CoCrAlNiMn  -0.189
    # 23       Ni          Ti  CoCrFeTiMn  -0.156
    # ...
    

**解説** : この分析により、HEAの安定性を向上させる元素置換を予測できます。実験検証の前に、多数の候補をスクリーニングすることで、材料探索を加速できます。

**Q9** : Chapter 4のStacking Regressorを使用して、Random Forest、XGBoost、MLPの3つのモデルをアンサンブルし、単一モデルよりも高精度な予測を実現してください。

**解答例** :
    
    
    # Stacking Ensemble実装
    from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # ベース学習器
    base_estimators = [
        ('rf', RandomForestRegressor(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42
        )),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64), activation='relu', 
            max_iter=500, random_state=42
        ))
    ]
    
    # メタ学習器（Ridge回帰）
    meta_estimator = Ridge(alpha=1.0)
    
    # Stacking Regressor
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_estimator,
        cv=5,
        n_jobs=-1
    )
    
    # Pipeline with scaling
    stacking_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('stacking', stacking_model)
    ])
    
    # 訓練と評価
    stacking_pipeline.fit(X_train, y_train)
    y_pred_stacking = stacking_pipeline.predict(X_test)
    mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
    r2_stacking = r2_score(y_test, y_pred_stacking)
    
    # 個別モデルとの比較
    print("=== アンサンブル性能比較 ===")
    for name, model in base_estimators:
        pipeline_single = Pipeline([('scaler', StandardScaler()), ('model', model)])
        pipeline_single.fit(X_train, y_train)
        y_pred_single = pipeline_single.predict(X_test)
        mae_single = mean_absolute_error(y_test, y_pred_single)
        print(f"{name.upper():5s}: MAE = {mae_single:.4f} eV/atom")
    
    print(f"\\n{'STACK':5s}: MAE = {mae_stacking:.4f} eV/atom (R² = {r2_stacking:.4f})")
    
    # Cross-validation評価
    cv_scores_stacking = cross_val_score(
        stacking_pipeline, X_train, y_train,
        cv=5, scoring='neg_mean_absolute_error'
    )
    print(f"\\nCV MAE: {-cv_scores_stacking.mean():.4f} ± {cv_scores_stacking.std():.4f}")
    
    # 期待される出力:
    # === アンサンブル性能比較 ===
    # RF   : MAE = 0.1234 eV/atom
    # GB   : MAE = 0.1189 eV/atom
    # MLP  : MAE = 0.1312 eV/atom
    #
    # STACK: MAE = 0.1156 eV/atom (R² = 0.9087)
    #
    # CV MAE: 0.1178 ± 0.0065
    

**解説** : Stacking Ensembleは、各ベースモデルの強みを組み合わせることで、単一モデルよりも約3-5%精度が向上します。ただし、訓練時間は3倍以上かかるため、精度が最優先の場合に使用します。

**Q10** : 50,000化合物の大規模予測を実行し、メモリ効率的なバッチ処理（チャンクサイズ1,000）と並列化（n_jobs=-1）を実装してください。処理速度（化合物/秒）を測定してください。

**解答例** :
    
    
    # 大規模バッチ予測システム
    import time
    import numpy as np
    from tqdm import tqdm
    
    def batch_predict_optimized(model, formulas, batch_size=1000, n_jobs=-1):
        """メモリ効率的なバッチ予測
        
        Args:
            model: 訓練済みPipeline
            formulas (list): 化学式リスト
            batch_size (int): バッチサイズ
            n_jobs (int): 並列ジョブ数
            
        Returns:
            tuple: (predictions, elapsed_time, throughput)
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from pymatgen.core import Composition
        
        start_time = time.time()
        predictions = []
        n_batches = (len(formulas) + batch_size - 1) // batch_size
        
        def process_batch(batch_formulas):
            """バッチ処理関数"""
            try:
                # 化学式→Composition変換
                batch_comps = [Composition(f) for f in batch_formulas]
                batch_df = pd.DataFrame({'composition': batch_comps})
                
                # 特徴量生成
                batch_df = featurizer.featurize_dataframe(
                    batch_df, 'composition', ignore_errors=True
                )
                batch_df = batch_df.dropna(subset=feature_cols)
                
                if len(batch_df) > 0:
                    X_batch = batch_df[feature_cols].values
                    return model.predict(X_batch)
                else:
                    return np.array([])
            except Exception as e:
                print(f"Batch error: {e}")
                return np.array([])
        
        # 並列バッチ処理
        with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            futures = []
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(formulas))
                batch_formulas = formulas[start_idx:end_idx]
                futures.append(executor.submit(process_batch, batch_formulas))
            
            # 結果収集（進捗表示付き）
            for future in tqdm(as_completed(futures), total=n_batches, desc="Processing"):
                batch_preds = future.result()
                if len(batch_preds) > 0:
                    predictions.extend(batch_preds)
        
        elapsed_time = time.time() - start_time
        throughput = len(predictions) / elapsed_time if elapsed_time > 0 else 0
        
        return np.array(predictions), elapsed_time, throughput
    
    # テスト実行（50,000化合物）
    # 実際のデータ取得は省略、ダミーデータで代用
    np.random.seed(42)
    elements_pool = ['Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'Ti', 'Fe', 'Cu', 'Zn', 'O', 'S', 'N']
    test_formulas = []
    for _ in range(50000):
        n_elem = np.random.randint(2, 4)
        elem_set = np.random.choice(elements_pool, n_elem, replace=False)
        formula = ''.join([f"{e}{np.random.randint(1,3)}" for e in elem_set])
        test_formulas.append(formula)
    
    print("=== 大規模バッチ予測実行 ===")
    print(f"対象化合物数: {len(test_formulas):,}")
    print(f"バッチサイズ: 1,000")
    print(f"並列化: CPU全コア使用\n")
    
    # predictions, elapsed, throughput = batch_predict_optimized(
    #     loaded_pipeline, test_formulas, batch_size=1000, n_jobs=-1
    # )
    
    # 推定結果（実際の実行時間は環境依存）
    print("推定性能:")
    print(f"処理時間: ~5-7分（CPU 8コア）")
    print(f"スループット: ~120-150 化合物/秒")
    print(f"メモリ使用: ~2-3 GB（ピーク）")
    print(f"\\n最適化ポイント:")
    print("- バッチサイズ: 500-1,000が最適（メモリ vs 速度のトレードオフ）")
    print("- n_jobs=-1: CPU全コア活用で線形スケーリング")
    print("- ProcessPoolExecutor: GIL制約を回避")
    

**解説** : 大規模予測では、メモリ管理と並列化が重要です。バッチサイズを適切に設定し（1,000前後）、全CPUコアを活用することで、数万化合物の予測を数分で完了できます。

## 参考文献

  1. Ward, L. et al. (2018). "Matminer: An open source toolkit for materials data mining." _Computational Materials Science_ , 152, 60-69.
  2. Dunn, A. et al. (2020). "Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm." _npj Computational Materials_ , 6, 138, pp. 5-8.
  3. Ong, S.P. et al. (2015). "The Materials Application Programming Interface (API)." _Computational Materials Science_ , 97, 209-215.
  4. Materials Project API Documentation. https://docs.materialsproject.org/
  5. matminer Examples Gallery. https://hackingmaterials.lbl.gov/matminer/examples/
  6. pandas Documentation: Data manipulation. https://pandas.pydata.org/docs/
  7. matplotlib/seaborn Documentation. https://matplotlib.org/

## 次のステップ

🎉 **おめでとうございます！** 組成ベース特徴量入門シリーズを完了しました。

次の学習リソース:

  * **gnn-features-comparison** : 組成ベース vs GNN構造ベース特徴量の詳細比較
  * **Advanced MI Topics** : 転移学習、Active Learning、ベイズ最適化
  * **実践プロジェクト** : Kaggle Materials Science competitions

← 第4章: 機械学習モデルとの統合（準備中） [シリーズ目次に戻る](<index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
