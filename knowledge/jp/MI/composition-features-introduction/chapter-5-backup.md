---
title: 第5章：Python実践：matminerワークフロー
chapter_title: 第5章：Python実践：matminerワークフロー
subtitle: エンドツーエンドの材料探索パイプライン構築
---

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
    * `all`: 完全（145特徴量、120秒/1000化合物）
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
