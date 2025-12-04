---
title: "第5章: Python実践：高分子データ解析ワークフロー"
chapter_title: "第5章: Python実践：高分子データ解析ワークフロー"
---

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Polymer Materials](<../../MS/polymer-materials-introduction/index.html>)›Chapter 5

  * [目次](<index.html>)
  * ← 第4章（準備中）
  * [第5章](<chapter-5.html>)

### 学習目標

**初級:**

  * RDKitで高分子SMILES記法を理解し、構造を生成できる
  * Morgan fingerprintの基本概念と用途を説明できる
  * MDAnalysisでMDシミュレーションデータを読み込める

**中級:**

  * scikit-learnで高分子物性（Tg）予測モデルを構築できる
  * MSD（平均二乗変位）から拡散係数を計算できる
  * PolyInfo APIで高分子データベースにアクセスできる

**上級:**

  * 統合ワークフローPolymerAnalysisクラスを実装できる
  * カスタム記述子を設計し、モデル精度を向上できる
  * バッチ処理とレポート自動生成システムを構築できる

### 本章の位置づけ

第5章では、第1章から第4章で学んだ理論を統合し、実務で即戦力となるPython実践スキルを習得します。RDKitによる構造生成、機械学習による物性予測、MDシミュレーションデータ解析、そしてPolyInfoなどのデータベース連携まで、高分子材料データサイエンスの完全なワークフローを構築します。 
    
    
    ```mermaid
    flowchart TB
                        A[データ取得PolyInfo API] --> B[構造処理RDKit SMILES]
                        B --> C[特徴量抽出Morgan Fingerprint]
                        C --> D[機械学習scikit-learn]
                        D --> E[物性予測Tg, 密度]
                        F[MDシミュレーションLAMMPS/GROMACS] --> G[軌跡解析MDAnalysis]
                        G --> H[MSD計算拡散係数]
                        E --> I[統合ワークフローPolymerAnalysis]
                        H --> I
                        I --> J[可視化・レポートmatplotlib/pandas]
                        J --> K[自動化バッチ処理]
    
                        style A fill:#f093fb
                        style E fill:#f5576c
                        style I fill:#27ae60
                        style K fill:#3498db
    ```

## 5.1 RDKitによる高分子構造生成

**RDKit** は、化学情報学のオープンソースライブラリで、分子構造の生成、可視化、記述子計算が可能です。高分子材料では、**SMILES記法** でモノマー構造を表現し、繰り返し単位をモデル化します。 

### 5.1.1 高分子SMILES記法と構造生成

SMILES（Simplified Molecular Input Line Entry System）は、分子構造を文字列で表現する記法です。高分子では、繰り返し単位を`[*]`で表現します。 
    
    
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, AllChem
    import matplotlib.pyplot as plt
    import numpy as np
    
    # RDKitによる高分子構造生成
    def generate_polymer_structure(monomer_smiles, repeat_units=5, polymer_name="Polymer"):
        """
        SMILES記法から高分子構造を生成
    
        Parameters:
        - monomer_smiles: モノマーのSMILES記法（[*]で結合点を指定）
        - repeat_units: 繰り返し単位数
        - polymer_name: 高分子名
    
        Returns:
        - polymer_mol: RDKit分子オブジェクト
        - properties: 分子記述子辞書
        """
        print(f"=== {polymer_name}構造生成 ===")
        print(f"モノマーSMILES: {monomer_smiles}")
    
        # モノマー構造生成
        monomer = Chem.MolFromSmiles(monomer_smiles)
        if monomer is None:
            print("エラー: 無効なSMILES記法")
            return None, {}
    
        # 高分子SMILES生成（簡易的な繰り返し）
        # 実際の重合では結合点[*]を処理する必要がある
        # ここでは単純化のため、線形連結をシミュレート
    
        # 一般的な高分子SMILES例
        polymer_smiles_examples = {
            "Polyethylene": "CC" * repeat_units,
            "Polystyrene": "CC(c1ccccc1)" * repeat_units,
            "PMMA": "CC(C)(C(=O)OC)" * repeat_units,
            "Nylon-6": "N(CCCCC)C(=O)" * repeat_units,
            "PET": "O=C(c1ccc(cc1)C(=O)O)OCCOC" * repeat_units
        }
    
        # モノマーから高分子へ（ここでは例示用に既知のSMILESを使用）
        if polymer_name in polymer_smiles_examples:
            polymer_smiles = polymer_smiles_examples[polymer_name]
        else:
            # カスタムモノマーの場合は単純連結
            polymer_smiles = monomer_smiles.replace("[*]", "") * repeat_units
    
        # 高分子分子オブジェクト生成
        polymer_mol = Chem.MolFromSmiles(polymer_smiles)
    
        if polymer_mol is None:
            print("エラー: 高分子SMILES生成失敗")
            return None, {}
    
        # 3D構造最適化
        polymer_mol_3d = Chem.AddHs(polymer_mol)
        AllChem.EmbedMolecule(polymer_mol_3d, randomSeed=42)
        AllChem.UFFOptimizeMolecule(polymer_mol_3d)
    
        # 分子記述子計算
        properties = {
            "Molecular Weight": Descriptors.MolWt(polymer_mol),
            "LogP": Descriptors.MolLogP(polymer_mol),
            "TPSA": Descriptors.TPSA(polymer_mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(polymer_mol),
            "H-Bond Donors": Descriptors.NumHDonors(polymer_mol),
            "H-Bond Acceptors": Descriptors.NumHAcceptors(polymer_mol),
            "Heavy Atoms": Descriptors.HeavyAtomCount(polymer_mol)
        }
    
        print(f"\n繰り返し単位数: {repeat_units}")
        print("分子記述子:")
        for key, value in properties.items():
            print(f"  {key}: {value:.2f}")
    
        # 構造可視化
        img = Draw.MolToImage(polymer_mol, size=(600, 400))
    
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{polymer_name} Structure (n={repeat_units})', fontsize=14, fontweight='bold')
    
        # 記述子棒グラフ
        plt.subplot(1, 2, 2)
        descriptor_names = ['MW\n(Da)', 'LogP', 'TPSA\n(Ų)', 'Rot.\nBonds', 'HBD', 'HBA']
        descriptor_values = [
            properties["Molecular Weight"] / 100,  # スケール調整
            properties["LogP"],
            properties["TPSA"] / 10,  # スケール調整
            properties["Rotatable Bonds"],
            properties["H-Bond Donors"],
            properties["H-Bond Acceptors"]
        ]
    
        colors = ['#f093fb', '#f5576c', '#4A90E2', '#27ae60', '#f39c12', '#e74c3c']
        bars = plt.bar(descriptor_names, descriptor_values, color=colors, edgecolor='black', linewidth=2)
        plt.ylabel('Value (scaled)', fontsize=12)
        plt.title('Molecular Descriptors', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='y')
    
        for bar, val in zip(bars, descriptor_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig(f'{polymer_name}_structure.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        return polymer_mol, properties
    
    # 実行例1: ポリスチレン
    generate_polymer_structure("CC(c1ccccc1)", repeat_units=5, polymer_name="Polystyrene")
    
    # 実行例2: PMMA
    generate_polymer_structure("CC(C)(C(=O)OC)", repeat_units=5, polymer_name="PMMA")
    

### 5.1.2 Morgan Fingerprintと構造記述子

**Morgan Fingerprint** は、分子構造を固定長のビットベクトルで表現する手法で、構造類似性評価や機械学習の特徴量として利用されます。 
    
    
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Morgan Fingerprint計算
    def calculate_morgan_fingerprint(smiles_list, radius=2, n_bits=2048):
        """
        Morgan Fingerprintを計算し、類似度行列を生成
    
        Parameters:
        - smiles_list: SMILES記法のリスト（辞書形式 {名前: SMILES}）
        - radius: Morgan Fingerprintの半径（デフォルト: 2）
        - n_bits: ビット長（デフォルト: 2048）
    
        Returns:
        - fingerprints: Fingerprintリスト
        - similarity_matrix: Tanimoto類似度行列
        """
        print("=== Morgan Fingerprint計算 ===")
    
        # 分子オブジェクト生成
        mols = {}
        fingerprints = {}
    
        for name, smiles in smiles_list.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols[name] = mol
                # Morgan Fingerprint生成
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fingerprints[name] = fp
                print(f"{name}: Fingerprint生成完了（{n_bits} bits）")
            else:
                print(f"エラー: {name}のSMILES変換失敗")
    
        # Tanimoto類似度行列計算
        polymer_names = list(fingerprints.keys())
        n = len(polymer_names)
        similarity_matrix = np.zeros((n, n))
    
        for i, name1 in enumerate(polymer_names):
            for j, name2 in enumerate(polymer_names):
                fp1 = fingerprints[name1]
                fp2 = fingerprints[name2]
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                similarity_matrix[i, j] = similarity
    
        # 可視化
        plt.figure(figsize=(14, 6))
    
        # サブプロット1: 類似度ヒートマップ
        plt.subplot(1, 2, 1)
        sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=polymer_names, yticklabels=polymer_names,
                    vmin=0, vmax=1, cbar_kws={'label': 'Tanimoto Similarity'})
        plt.title('Morgan Fingerprint Similarity Matrix', fontsize=14, fontweight='bold')
    
        # サブプロット2: Fingerprintのビット分布
        plt.subplot(1, 2, 2)
        for name, fp in fingerprints.items():
            # Fingerprintをnumpy配列に変換
            arr = np.zeros((n_bits,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            # ビットの設定頻度をヒストグラム化
            bit_count = np.sum(arr)
            plt.bar(name, bit_count, edgecolor='black', linewidth=2)
    
        plt.ylabel('Number of Set Bits', fontsize=12)
        plt.title('Fingerprint Bit Density', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(alpha=0.3, axis='y')
    
        plt.tight_layout()
        plt.savefig('morgan_fingerprint_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("\n=== Tanimoto類似度行列 ===")
        for i, name1 in enumerate(polymer_names):
            for j, name2 in enumerate(polymer_names):
                if i < j:  # 上三角のみ表示
                    print(f"{name1} - {name2}: {similarity_matrix[i, j]:.3f}")
    
        return fingerprints, similarity_matrix
    
    # 実行例: 代表的高分子の構造類似性
    polymer_smiles = {
        "Polyethylene": "CC",
        "Polypropylene": "CC(C)",
        "Polystyrene": "CC(c1ccccc1)",
        "PMMA": "CC(C)(C(=O)OC)",
        "PVC": "CC(Cl)",
        "PVDF": "CC(F)(F)"
    }
    
    calculate_morgan_fingerprint(polymer_smiles, radius=2, n_bits=2048)
    

## 5.2 機械学習による物性予測

高分子材料のガラス転移温度（Tg）や密度などの物性を、機械学習モデルで予測します。ここでは**Random Forest** を用いた回帰モデルを構築します。 

### 5.2.1 Tg予測モデル（scikit-learn Random Forest）
    
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import matplotlib.pyplot as plt
    
    # Tg予測モデル構築
    def build_tg_prediction_model():
        """
        Random Forestを用いた高分子Tg予測モデル
    
        Returns:
        - model: 訓練済みモデル
        - X_test, y_test: テストデータ
        - metrics: 評価指標辞書
        """
        print("=== Tg予測モデル構築（Random Forest）===")
    
        # サンプルデータセット（実際にはPolyInfoなどから取得）
        # 特徴量: [MW, LogP, TPSA, RotBonds, HBD, HBA, Aromatic_Ratio]
        data = {
            'Polymer': ['PS', 'PMMA', 'PC', 'PET', 'Nylon-6', 'PE', 'PP', 'PVC',
                        'PVDF', 'PTFE', 'PAN', 'PVA', 'Cellulose', 'PLA', 'PEEK'],
            'MW': [104, 100, 254, 192, 113, 28, 42, 62.5, 64, 100, 53, 44, 162, 72, 288],
            'LogP': [3.2, 1.5, 2.8, 1.2, -0.5, 1.9, 2.1, 1.4, 1.0, 3.5, -0.3, -0.8, -1.2, 0.5, 3.8],
            'TPSA': [0, 26.3, 40.5, 52.6, 43.1, 0, 0, 0, 0, 0, 23.8, 20.2, 90.2, 26.3, 25.8],
            'RotBonds': [2, 3, 8, 6, 10, 1, 2, 1, 1, 0, 1, 1, 4, 2, 6],
            'HBD': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0],
            'HBA': [0, 2, 2, 4, 1, 0, 0, 0, 2, 0, 1, 1, 5, 2, 2],
            'Aromatic_Ratio': [100, 0, 80, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90],
            'Tg_K': [373, 378, 423, 343, 323, 195, 253, 354, 233, 115, 358, 358, 503, 333, 416]
        }
    
        df = pd.DataFrame(data)
    
        # 特徴量とターゲット
        feature_cols = ['MW', 'LogP', 'TPSA', 'RotBonds', 'HBD', 'HBA', 'Aromatic_Ratio']
        X = df[feature_cols].values
        y = df['Tg_K'].values
    
        # データ分割（訓練:テスト = 80:20）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # Random Forestモデル構築
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42
        )
    
        # 訓練
        model.fit(X_train, y_train)
    
        # 予測
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
        # 評価指標
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
    
        # クロスバリデーション
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
        metrics = {
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std()
        }
    
        # 可視化
        plt.figure(figsize=(14, 5))
    
        # サブプロット1: 予測 vs 実測
        plt.subplot(1, 3, 1)
        plt.scatter(y_train, y_train_pred, alpha=0.6, s=100, label='Train', edgecolors='black')
        plt.scatter(y_test, y_test_pred, alpha=0.8, s=150, label='Test',
                    edgecolors='black', linewidths=2)
    
        # 対角線（完全予測）
        min_val = min(y.min(), y_train_pred.min(), y_test_pred.min())
        max_val = max(y.max(), y_train_pred.max(), y_test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
        plt.xlabel('Actual Tg (K)', fontsize=12)
        plt.ylabel('Predicted Tg (K)', fontsize=12)
        plt.title(f'Tg Prediction (R² = {test_r2:.3f})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット2: 残差プロット
        plt.subplot(1, 3, 2)
        residuals = y_test - y_test_pred
        plt.scatter(y_test_pred, residuals, s=150, alpha=0.6, edgecolors='black', linewidths=2)
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Tg (K)', fontsize=12)
        plt.ylabel('Residuals (K)', fontsize=12)
        plt.title('Residual Plot', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
        # サブプロット3: 評価指標
        plt.subplot(1, 3, 3)
        metric_names = ['R² Train', 'R² Test', 'RMSE\nTrain', 'RMSE\nTest', 'MAE\nTest']
        metric_values = [train_r2, test_r2, train_rmse/100, test_rmse/100, test_mae/100]  # スケール調整
    
        colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        bars = plt.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=2)
        plt.ylabel('Value (R² or RMSE/100)', fontsize=12)
        plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='y')
    
        for bar, val, orig in zip(bars, metric_values, [train_r2, test_r2, train_rmse, test_rmse, test_mae]):
            height = bar.get_height()
            if 'RMSE' in metric_names[bars.index(bar)] or 'MAE' in metric_names[bars.index(bar)]:
                label_text = f'{orig:.1f}K'
            else:
                label_text = f'{orig:.3f}'
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    label_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('tg_prediction_model.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("\n=== モデル評価指標 ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    
        print("\n=== テストセット予測結果 ===")
        test_indices = df.index[len(y_train):]
        for idx, (actual, pred) in zip(test_indices, zip(y_test, y_test_pred)):
            polymer_name = df.loc[idx, 'Polymer']
            print(f"{polymer_name}: 実測 {actual:.1f}K, 予測 {pred:.1f}K, 誤差 {abs(actual-pred):.1f}K")
    
        return model, X_test, y_test, metrics, feature_cols
    
    # 実行
    model, X_test, y_test, metrics, feature_cols = build_tg_prediction_model()
    

### 5.2.2 特徴量重要度とモデル解釈
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    
    # 特徴量重要度解析
    def analyze_feature_importance(model, X_test, y_test, feature_names):
        """
        Random Forestの特徴量重要度を解析
    
        Parameters:
        - model: 訓練済みモデル
        - X_test: テストデータ特徴量
        - y_test: テストデータターゲット
        - feature_names: 特徴量名リスト
    
        Returns:
        - importances: 特徴量重要度辞書
        """
        print("=== 特徴量重要度解析 ===")
    
        # Gini重要度（ビルトイン）
        gini_importances = model.feature_importances_
    
        # Permutation重要度（より信頼性が高い）
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42
        )
        perm_importances = perm_importance.importances_mean
    
        # 可視化
        plt.figure(figsize=(14, 6))
    
        # サブプロット1: Gini重要度
        plt.subplot(1, 2, 1)
        sorted_idx = np.argsort(gini_importances)[::-1]
    
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_names)))
        bars = plt.barh(range(len(feature_names)),
                        gini_importances[sorted_idx],
                        color=colors, edgecolor='black', linewidth=2)
        plt.yticks(range(len(feature_names)), np.array(feature_names)[sorted_idx])
        plt.xlabel('Gini Importance', fontsize=12)
        plt.title('Feature Importance (Gini)', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='x')
    
        for bar, val in zip(bars, gini_importances[sorted_idx]):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
        # サブプロット2: Permutation重要度
        plt.subplot(1, 2, 2)
        sorted_idx_perm = np.argsort(perm_importances)[::-1]
    
        bars2 = plt.barh(range(len(feature_names)),
                         perm_importances[sorted_idx_perm],
                         color=colors, edgecolor='black', linewidth=2)
        plt.yticks(range(len(feature_names)), np.array(feature_names)[sorted_idx_perm])
        plt.xlabel('Permutation Importance', fontsize=12)
        plt.title('Feature Importance (Permutation)', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='x')
    
        for bar, val in zip(bars2, perm_importances[sorted_idx_perm]):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("\n=== Gini重要度ランキング ===")
        for idx in sorted_idx:
            print(f"{feature_names[idx]}: {gini_importances[idx]:.4f}")
    
        print("\n=== Permutation重要度ランキング ===")
        for idx in sorted_idx_perm:
            print(f"{feature_names[idx]}: {perm_importances[idx]:.4f}")
    
        importances = {
            'gini': dict(zip(feature_names, gini_importances)),
            'permutation': dict(zip(feature_names, perm_importances))
        }
    
        return importances
    
    # 実行
    analyze_feature_importance(model, X_test, y_test, feature_cols)
    

## 5.3 MDシミュレーションデータ解析

分子動力学（MD）シミュレーションで得られた軌跡データを**MDAnalysis** で解析し、拡散係数や構造パラメータを計算します。 

### 5.3.1 MDAnalysisトラジェクトリ解析
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # MDシミュレーションデータ解析（簡易版）
    # 実際のMDAnalysisでは: import MDAnalysis as mda
    # ここでは疑似データで動作をシミュレート
    
    def analyze_md_trajectory_simplified():
        """
        MDシミュレーション軌跡の簡易解析
        （実環境ではMDAnalysisを使用）
    
        Returns:
        - time: 時間（ps）
        - rg: 回転半径（Å）
        - end_to_end: 末端間距離（Å）
        """
        print("=== MD軌跡解析（簡易版）===")
    
        # 疑似MDデータ生成
        # 実際は: u = mda.Universe('topology.pdb', 'trajectory.dcd')
        time = np.linspace(0, 10000, 1000)  # ps
    
        # 回転半径（Radius of Gyration）
        # 高分子のコンパクトさの指標
        rg_mean = 15.0  # Å
        rg = rg_mean + np.random.normal(0, 1.5, len(time))
    
        # 末端間距離（End-to-End Distance）
        # 高分子鎖の伸びの指標
        ete_mean = 40.0  # Å
        ete = ete_mean + np.random.normal(0, 5.0, len(time))
    
        # 可視化
        plt.figure(figsize=(14, 10))
    
        # サブプロット1: 回転半径の時間変化
        plt.subplot(3, 2, 1)
        plt.plot(time, rg, linewidth=1, alpha=0.7)
        plt.axhline(rg.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {rg.mean():.2f} Å')
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('Radius of Gyration Rg (Å)', fontsize=12)
        plt.title('Rg vs Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット2: Rg分布
        plt.subplot(3, 2, 2)
        plt.hist(rg, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
        plt.axvline(rg.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {rg.mean():.2f} Å')
        plt.xlabel('Radius of Gyration Rg (Å)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Rg Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット3: 末端間距離の時間変化
        plt.subplot(3, 2, 3)
        plt.plot(time, ete, linewidth=1, alpha=0.7, color='green')
        plt.axhline(ete.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {ete.mean():.2f} Å')
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('End-to-End Distance (Å)', fontsize=12)
        plt.title('End-to-End Distance vs Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット4: 末端間距離分布
        plt.subplot(3, 2, 4)
        plt.hist(ete, bins=30, edgecolor='black', alpha=0.7, color='#27ae60')
        plt.axvline(ete.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {ete.mean():.2f} Å')
        plt.xlabel('End-to-End Distance (Å)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('End-to-End Distance Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット5: Rg vs End-to-End相関
        plt.subplot(3, 2, 5)
        plt.scatter(rg, ete, alpha=0.3, s=20, edgecolors='none')
    
        # 線形回帰
        z = np.polyfit(rg, ete, 1)
        p = np.poly1d(z)
        plt.plot(rg, p(rg), "r--", linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
    
        plt.xlabel('Radius of Gyration Rg (Å)', fontsize=12)
        plt.ylabel('End-to-End Distance (Å)', fontsize=12)
        plt.title('Rg vs End-to-End Correlation', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット6: 統計量サマリ
        plt.subplot(3, 2, 6)
        stats = {
            'Rg Mean (Å)': rg.mean(),
            'Rg Std (Å)': rg.std(),
            'EtE Mean (Å)': ete.mean(),
            'EtE Std (Å)': ete.std(),
            'Correlation': np.corrcoef(rg, ete)[0, 1]
        }
    
        stat_names = list(stats.keys())
        stat_values = list(stats.values())
    
        colors = ['#3498db', '#3498db', '#27ae60', '#27ae60', '#9b59b6']
        bars = plt.barh(stat_names, stat_values, color=colors, edgecolor='black', linewidth=2)
        plt.xlabel('Value', fontsize=12)
        plt.title('Statistical Summary', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='x')
    
        for bar, val in zip(bars, stat_values):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{val:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('md_trajectory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("\n=== 軌跡統計量 ===")
        for key, value in stats.items():
            print(f"{key}: {value:.3f}")
    
        return time, rg, ete
    
    # 実行
    time, rg, ete = analyze_md_trajectory_simplified()
    

### 5.3.2 MSD（平均二乗変位）計算と拡散係数
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # MSD計算と拡散係数導出
    def calculate_msd_diffusion_coefficient(temperature=300):
        """
        平均二乗変位（MSD）から拡散係数を計算
    
        Parameters:
        - temperature: 温度（K）
    
        Returns:
        - time: 時間（ps）
        - msd: MSD（Ų）
        - diffusion_coeff: 拡散係数（cm²/s）
        """
        print(f"=== MSD計算と拡散係数（T = {temperature}K）===")
    
        # 疑似データ生成
        # 実際のMD解析では原子座標から計算
        time = np.linspace(0, 1000, 500)  # ps
    
        # MSD = 6Dt（3次元拡散）
        # D: 拡散係数（Ų/ps）
        D_true = 5e-3  # Ų/ps（仮定値）
        msd = 6 * D_true * time + np.random.normal(0, 0.5, len(time))
        msd = np.maximum(msd, 0)  # 負の値を防ぐ
    
        # 線形フィッティング（後半の線形領域のみ使用）
        # 初期の非線形領域を除外
        linear_region = time > 200  # ps
        z = np.polyfit(time[linear_region], msd[linear_region], 1)
        slope = z[0]  # Ų/ps
    
        # 拡散係数 D = slope / 6
        D_fitted = slope / 6  # Ų/ps
    
        # 単位変換: Ų/ps → cm²/s
        # 1 Ų = 10⁻¹⁶ cm², 1 ps = 10⁻¹² s
        D_cm2_s = D_fitted * 1e-16 / 1e-12  # cm²/s
        D_cm2_s = D_fitted * 1e-4  # cm²/s
    
        # 可視化
        plt.figure(figsize=(14, 5))
    
        # サブプロット1: MSD vs Time
        plt.subplot(1, 3, 1)
        plt.plot(time, msd, 'b-', linewidth=2, label='MSD Data')
    
        # フィッティング直線
        fit_line = z[0] * time + z[1]
        plt.plot(time, fit_line, 'r--', linewidth=2,
                 label=f'Fit: MSD = {z[0]:.4f}t + {z[1]:.2f}')
    
        # 線形領域の表示
        plt.axvline(200, color='green', linestyle=':', linewidth=1.5,
                    label='Linear Region Start')
    
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('MSD (Ų)', fontsize=12)
        plt.title('Mean Square Displacement', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット2: 対数プロット（べき乗則確認）
        plt.subplot(1, 3, 2)
        plt.loglog(time[time > 0], msd[time > 0], 'b-', linewidth=2)
    
        # べき乗則 MSD ∝ t^α（α=1: 正常拡散）
        alpha = np.polyfit(np.log(time[linear_region]), np.log(msd[linear_region]), 1)[0]
    
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('MSD (Ų)', fontsize=12)
        plt.title(f'Log-Log Plot (α = {alpha:.2f})', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, which='both')
    
        # サブプロット3: 拡散係数の温度依存性（Arrhenius）
        plt.subplot(1, 3, 3)
        temperatures = np.linspace(250, 400, 20)  # K
    
        # Arrhenius式: D = D0 * exp(-Ea/RT)
        D0 = 1e-2  # cm²/s
        Ea = 30000  # J/mol
        R = 8.314  # J/mol·K
    
        D_temps = D0 * np.exp(-Ea / (R * temperatures))
    
        plt.semilogy(temperatures, D_temps, 'purple', linewidth=2, label='Arrhenius Model')
        plt.scatter([temperature], [D_cm2_s], s=200, c='red', edgecolors='black',
                    linewidths=2, zorder=5, label=f'Current ({temperature}K)')
    
        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel('Diffusion Coefficient (cm²/s)', fontsize=12)
        plt.title('Temperature Dependence of D', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3, which='both')
    
        plt.tight_layout()
        plt.savefig('msd_diffusion.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print(f"\n=== MSD解析結果 ===")
        print(f"線形フィッティング: MSD = {z[0]:.4f} * t + {z[1]:.2f}")
        print(f"拡散係数 D: {D_fitted:.2e} Ų/ps")
        print(f"拡散係数 D: {D_cm2_s:.2e} cm²/s")
        print(f"べき指数 α: {alpha:.3f}（α=1: 正常拡散）")
    
        # Einstein関係式による検証
        # D = kT / (6πηr)（Stokes-Einstein式）
        # ここでは簡易的に妥当性を確認
        print(f"\n実験的拡散係数範囲: 10⁻⁷ - 10⁻⁵ cm²/s（典型的高分子融体）")
    
        return time, msd, D_cm2_s
    
    # 実行
    calculate_msd_diffusion_coefficient(temperature=300)
    

## 5.4 統合ワークフロー構築

ここまで学んだ全ての手法を統合し、**PolymerAnalysisクラス** として実装します。このクラスは、構造生成、物性予測、MD解析を一括で処理できます。 

### 5.4.1 PolymerAnalysisクラス実装
    
    
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    
    # 統合ワークフロークラス
    class PolymerAnalysis:
        """
        高分子材料の統合解析ワークフロー
    
        機能:
        - RDKitによる構造生成と記述子計算
        - 機械学習によるTg予測
        - MDデータ解析
        - レポート自動生成
        """
    
        def __init__(self, polymer_name, smiles):
            """
            初期化
    
            Parameters:
            - polymer_name: 高分子名
            - smiles: SMILES記法
            """
            self.polymer_name = polymer_name
            self.smiles = smiles
            self.mol = Chem.MolFromSmiles(smiles)
            self.descriptors = {}
            self.predicted_tg = None
    
            print(f"=== PolymerAnalysis初期化: {polymer_name} ===")
            print(f"SMILES: {smiles}")
    
        def calculate_descriptors(self):
            """RDKit記述子計算"""
            if self.mol is None:
                print("エラー: 分子オブジェクトが無効です")
                return None
    
            self.descriptors = {
                'MW': Descriptors.MolWt(self.mol),
                'LogP': Descriptors.MolLogP(self.mol),
                'TPSA': Descriptors.TPSA(self.mol),
                'RotBonds': Descriptors.NumRotatableBonds(self.mol),
                'HBD': Descriptors.NumHDonors(self.mol),
                'HBA': Descriptors.NumHAcceptors(self.mol),
                'AromaticRings': Descriptors.NumAromaticRings(self.mol)
            }
    
            print("\n=== 分子記述子 ===")
            for key, value in self.descriptors.items():
                print(f"{key}: {value:.2f}")
    
            return self.descriptors
    
        def predict_tg(self, model=None):
            """
            Tg予測（簡易版モデル使用）
    
            Parameters:
            - model: 訓練済みモデル（Noneの場合は簡易推定）
    
            Returns:
            - predicted_tg: 予測Tg（K）
            """
            if not self.descriptors:
                self.calculate_descriptors()
    
            # 簡易推定式（実際にはモデルを使用）
            # Tg ≈ 250 + 0.5*MW + 10*HBD + 5*HBA - 20*RotBonds
            tg_estimate = (250 +
                          0.5 * self.descriptors['MW'] +
                          10 * self.descriptors['HBD'] +
                          5 * self.descriptors['HBA'] -
                          20 * self.descriptors['RotBonds'])
    
            self.predicted_tg = tg_estimate
    
            print(f"\n=== Tg予測 ===")
            print(f"予測Tg: {tg_estimate:.1f} K ({tg_estimate - 273.15:.1f}°C)")
    
            return tg_estimate
    
        def analyze_md_data(self, time_range=1000):
            """
            MD解析（疑似データ）
    
            Parameters:
            - time_range: シミュレーション時間（ps）
    
            Returns:
            - md_results: 解析結果辞書
            """
            print(f"\n=== MD解析（{time_range} ps）===")
    
            # 疑似MDデータ
            time = np.linspace(0, time_range, 500)
            rg = 15 + np.random.normal(0, 1.5, len(time))
            msd = 6 * 5e-3 * time + np.random.normal(0, 0.5, len(time))
    
            # 拡散係数計算
            linear_region = time > time_range * 0.2
            slope = np.polyfit(time[linear_region], msd[linear_region], 1)[0]
            D = slope / 6 * 1e-4  # cm²/s
    
            md_results = {
                'Rg_mean': rg.mean(),
                'Rg_std': rg.std(),
                'Diffusion_coeff': D
            }
    
            print(f"回転半径 Rg: {md_results['Rg_mean']:.2f} ± {md_results['Rg_std']:.2f} Å")
            print(f"拡散係数 D: {md_results['Diffusion_coeff']:.2e} cm²/s")
    
            return md_results
    
        def generate_report(self):
            """統合レポート生成"""
            print(f"\n{'='*60}")
            print(f"高分子材料解析レポート: {self.polymer_name}")
            print(f"{'='*60}")
    
            print(f"\n【構造情報】")
            print(f"SMILES: {self.smiles}")
    
            if self.descriptors:
                print(f"\n【分子記述子】")
                for key, value in self.descriptors.items():
                    print(f"  {key}: {value:.2f}")
    
            if self.predicted_tg:
                print(f"\n【物性予測】")
                print(f"  ガラス転移温度 Tg: {self.predicted_tg:.1f} K ({self.predicted_tg - 273.15:.1f}°C)")
    
            print(f"\n【推奨用途】")
            if self.predicted_tg and self.predicted_tg > 350:
                print("  - 高温エンジニアリングプラスチック")
                print("  - 航空宇宙材料")
            elif self.predicted_tg and self.predicted_tg > 300:
                print("  - 汎用エンジニアリングプラスチック")
                print("  - 自動車部品")
            else:
                print("  - 汎用プラスチック")
                print("  - 包装材料")
    
            print(f"\n{'='*60}\n")
    
        def visualize_summary(self):
            """サマリ可視化"""
            if not self.descriptors:
                self.calculate_descriptors()
    
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            # サブプロット1: 記述子レーダーチャート
            ax1 = axes[0, 0]
            categories = list(self.descriptors.keys())[:6]
            values = [self.descriptors[cat] for cat in categories]
    
            # 正規化
            max_vals = [500, 5, 150, 20, 5, 10]  # 各記述子の最大想定値
            values_norm = [v / m for v, m in zip(values, max_vals)]
    
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values_norm += values_norm[:1]
            angles += angles[:1]
    
            ax1 = plt.subplot(2, 2, 1, projection='polar')
            ax1.plot(angles, values_norm, 'o-', linewidth=2, color='#f093fb')
            ax1.fill(angles, values_norm, alpha=0.25, color='#f5576c')
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(categories, fontsize=10)
            ax1.set_ylim(0, 1)
            ax1.set_title(f'{self.polymer_name}: Descriptor Profile', fontsize=12, fontweight='bold', pad=20)
            ax1.grid(True)
    
            # サブプロット2: 構造式
            ax2 = axes[0, 1]
            from rdkit.Chem import Draw
            img = Draw.MolToImage(self.mol, size=(400, 300))
            ax2.imshow(img)
            ax2.axis('off')
            ax2.set_title('Chemical Structure', fontsize=12, fontweight='bold')
    
            # サブプロット3: Tg予測
            ax3 = axes[1, 0]
            if self.predicted_tg:
                bars = ax3.bar(['Predicted Tg'], [self.predicted_tg],
                              color='#27ae60', edgecolor='black', linewidth=2)
                ax3.axhline(273.15, color='blue', linestyle='--', linewidth=1.5, label='0°C')
                ax3.axhline(373.15, color='red', linestyle='--', linewidth=1.5, label='100°C')
                ax3.set_ylabel('Temperature (K)', fontsize=12)
                ax3.set_title('Glass Transition Temperature Prediction', fontsize=12, fontweight='bold')
                ax3.legend()
                ax3.grid(alpha=0.3, axis='y')
    
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}K\n({height-273.15:.1f}°C)',
                            ha='center', va='bottom', fontsize=11, fontweight='bold')
    
            # サブプロット4: 用途分類
            ax4 = axes[1, 1]
            applications = ['Engineering\nPlastic', 'General\nPurpose', 'Commodity', 'High-Temp\nSpecialty']
            scores = [0, 0, 0, 0]
    
            if self.predicted_tg:
                if self.predicted_tg > 400:
                    scores[3] = 1
                elif self.predicted_tg > 350:
                    scores[0] = 1
                elif self.predicted_tg > 300:
                    scores[1] = 1
                else:
                    scores[2] = 1
    
            colors_app = ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
            bars = ax4.barh(applications, scores, color=colors_app, edgecolor='black', linewidth=2)
            ax4.set_xlim(0, 1.2)
            ax4.set_xlabel('Suitability Score', fontsize=12)
            ax4.set_title('Application Classification', fontsize=12, fontweight='bold')
            ax4.grid(alpha=0.3, axis='x')
    
            plt.tight_layout()
            plt.savefig(f'{self.polymer_name}_analysis_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 使用例
    polymer = PolymerAnalysis("PMMA", "CC(C)(C(=O)OC)")
    polymer.calculate_descriptors()
    polymer.predict_tg()
    polymer.analyze_md_data(time_range=1000)
    polymer.generate_report()
    polymer.visualize_summary()
    

### 5.4.2 PolyInfo APIデータベース連携
    
    
    import requests
    import json
    import pandas as pd
    
    # PolyInfo API連携（簡易版）
    def fetch_polyinfo_data(polymer_name, property_type='Tg'):
        """
        PolyInfo APIから高分子データを取得（疑似実装）
    
        実際のAPI仕様はPolyInfo公式ドキュメント参照:
        https://polymer.nims.go.jp/
    
        Parameters:
        - polymer_name: 高分子名
        - property_type: 物性タイプ（'Tg', 'density', 'modulus'）
    
        Returns:
        - data: 取得データ（DataFrame）
        """
        print(f"=== PolyInfo APIデータ取得: {polymer_name} ===")
    
        # 実際のAPI呼び出し例（疑似コード）
        # api_url = "https://polymer.nims.go.jp/api/v1/polymers"
        # params = {'name': polymer_name, 'property': property_type}
        # response = requests.get(api_url, params=params)
        # data = response.json()
    
        # ここでは疑似データを返す
        sample_data = {
            'Polymer': [polymer_name] * 5,
            'Measurement_Condition': ['DSC-10K/min', 'DSC-20K/min', 'DMA-1Hz', 'DSC-10K/min', 'TMA'],
            'Tg_K': [378, 375, 380, 377, 379],
            'Reference': ['Smith2020', 'Jones2019', 'Lee2021', 'Kim2018', 'Wang2022']
        }
    
        df = pd.DataFrame(sample_data)
    
        print(f"\n取得データ数: {len(df)} 件")
        print("\nデータサンプル:")
        print(df.head())
    
        # 統計量
        print(f"\n=== {property_type}統計量 ===")
        print(f"平均: {df['Tg_K'].mean():.1f} K")
        print(f"標準偏差: {df['Tg_K'].std():.1f} K")
        print(f"最小-最大: {df['Tg_K'].min():.1f} - {df['Tg_K'].max():.1f} K")
    
        return df
    
    # 使用例
    polyinfo_data = fetch_polyinfo_data("PMMA", property_type='Tg')
    

## 5.5 実践プロジェクト例

最後に、新規高分子材料設計から特性最適化までの完全なワークフローを実践します。 

### 5.5.1 新規材料設計ワークフロー
    
    
    ```mermaid
    flowchart LR
                        A[要求仕様Tg > 150°C密度 < 1.2 g/cm³] --> B[候補構造生成RDKit]
                        B --> C[記述子計算Morgan FP]
                        C --> D[ML予測Random Forest]
                        D --> E{物性達成?}
                        E -->|No| F[構造最適化官能基修飾]
                        F --> B
                        E -->|Yes| G[MDシミュレーション詳細評価]
                        G --> H[実験検証合成・測定]
                        H --> I[データベース登録PolyInfo]
    
                        style A fill:#f093fb
                        style D fill:#3498db
                        style E fill:#f39c12
                        style I fill:#27ae60
    ```
    
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    # 新規材料設計ワークフロー
    def design_new_polymer_material(target_tg=423, max_iterations=10):
        """
        要求仕様に基づく新規高分子設計
    
        Parameters:
        - target_tg: 目標Tg（K）
        - max_iterations: 最大試行回数
    
        Returns:
        - best_candidate: 最適候補
        """
        print(f"=== 新規高分子材料設計 ===")
        print(f"目標Tg: {target_tg} K ({target_tg - 273.15}°C)")
    
        # 候補モノマーリスト
        monomers = {
            'Styrene': 'CC(c1ccccc1)',
            'MMA': 'CC(C)(C(=O)OC)',
            'Acrylonitrile': 'CC(C#N)',
            'Vinyl_Acetate': 'CC(OC(=O)C)',
            'Butadiene': 'C=CC=C',
            'Isoprene': 'CC(=C)C=C'
        }
    
        # 官能基ライブラリ
        functional_groups = {
            'Phenyl': 'c1ccccc1',
            'Methyl': 'C',
            'Cyano': 'C#N',
            'Carbonyl': 'C(=O)',
            'Hydroxyl': 'O'
        }
    
        candidates = []
    
        for iteration in range(max_iterations):
            # ランダムにモノマーと官能基を組み合わせ
            monomer_name = np.random.choice(list(monomers.keys()))
            monomer_smiles = monomers[monomer_name]
    
            # Tg簡易推定
            mol = Chem.MolFromSmiles(monomer_smiles)
            if mol is None:
                continue
    
            descriptors = {
                'MW': Descriptors.MolWt(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol)
            }
    
            # 簡易Tg推定式
            tg_estimate = (250 +
                          0.5 * descriptors['MW'] +
                          10 * descriptors['HBD'] +
                          5 * descriptors['HBA'] -
                          20 * descriptors['RotBonds'] +
                          30 * descriptors['AromaticRings'])
    
            error = abs(tg_estimate - target_tg)
    
            candidates.append({
                'Monomer': monomer_name,
                'SMILES': monomer_smiles,
                'Predicted_Tg': tg_estimate,
                'Error': error,
                'Descriptors': descriptors
            })
    
            print(f"\nIteration {iteration + 1}:")
            print(f"  Monomer: {monomer_name}")
            print(f"  Predicted Tg: {tg_estimate:.1f} K ({tg_estimate - 273.15:.1f}°C)")
            print(f"  Error: {error:.1f} K")
    
        # 最適候補選択
        candidates_sorted = sorted(candidates, key=lambda x: x['Error'])
        best_candidate = candidates_sorted[0]
    
        print(f"\n{'='*60}")
        print(f"最適候補: {best_candidate['Monomer']}")
        print(f"SMILES: {best_candidate['SMILES']}")
        print(f"予測Tg: {best_candidate['Predicted_Tg']:.1f} K ({best_candidate['Predicted_Tg'] - 273.15:.1f}°C)")
        print(f"目標からの誤差: {best_candidate['Error']:.1f} K")
        print(f"{'='*60}")
    
        # 上位3候補を可視化
        import matplotlib.pyplot as plt
    
        top_candidates = candidates_sorted[:5]
        names = [c['Monomer'] for c in top_candidates]
        tgs = [c['Predicted_Tg'] for c in top_candidates]
        errors = [c['Error'] for c in top_candidates]
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # サブプロット1: Tg比較
        ax1 = axes[0]
        colors = ['#27ae60' if i == 0 else '#3498db' for i in range(len(names))]
        bars = ax1.bar(names, tgs, color=colors, edgecolor='black', linewidth=2)
        ax1.axhline(target_tg, color='red', linestyle='--', linewidth=2,
                    label=f'Target Tg = {target_tg}K')
        ax1.set_ylabel('Predicted Tg (K)', fontsize=12)
        ax1.set_title('Top Candidate Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
        for bar, val in zip(bars, tgs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}K', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
        # サブプロット2: 誤差
        ax2 = axes[1]
        bars2 = ax2.bar(names, errors, color=colors, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Error from Target (K)', fontsize=12)
        ax2.set_title('Prediction Error', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
        for bar, val in zip(bars2, errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}K', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('polymer_design_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        return best_candidate
    
    # 実行例: Tg = 150°C（423K）を目標とする材料設計
    design_new_polymer_material(target_tg=423, max_iterations=10)
    

## 演習問題

#### 演習1: SMILES記法（Easy）

ポリプロピレン（PP）のモノマーSMILES記法を書いてください。

解答を見る
    
    
    # ポリプロピレンモノマー（プロピレン）
    smiles = "CC(C)"  # または "C=CC"（二重結合表記）
    print(f"Polypropylene SMILES: {smiles}")

#### 演習2: 分子量計算（Easy）

PMMA（"CC(C)(C(=O)OC)"）の分子量をRDKitで計算してください。

解答を見る
    
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    smiles = "CC(C)(C(=O)OC)"
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    print(f"PMMA分子量: {mw:.2f} g/mol")
    # 出力: 100.12 g/mol

#### 演習3: MSD拡散係数（Easy）

MSDのスロープが0.03 Ų/psのとき、拡散係数D（Ų/ps）を計算してください（3次元拡散）。

解答を見る
    
    
    slope = 0.03  # Ų/ps
    D = slope / 6  # 3次元: MSD = 6Dt
    print(f"拡散係数: {D:.4f} Ų/ps")
    # 出力: 0.0050 Ų/ps

#### 演習4: Morgan Fingerprint類似度（Medium）

2つのSMILES "CC" と "CCC" のTanimoto類似度を計算してください（radius=2, n_bits=1024）。

解答を見る
    
    
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    
    smiles1 = "CC"
    smiles2 = "CCC"
    
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    print(f"Tanimoto類似度: {similarity:.3f}")
    # 出力: 0.5-0.7程度（構造が類似）

#### 演習5: Tg予測モデル評価（Medium）

予測値 [350, 380, 400] K、実測値 [345, 390, 395] K のとき、RMSE（平方根平均二乗誤差）を計算してください。

解答を見る
    
    
    import numpy as np
    from sklearn.metrics import mean_squared_error
    
    y_true = np.array([345, 390, 395])
    y_pred = np.array([350, 380, 400])
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.2f} K")
    # 出力: 7.45 K

#### 演習6: 回転半径の統計量（Medium）

MD軌跡から得られたRgデータ [14.5, 15.2, 14.8, 15.5, 14.9] Åの平均と標準偏差を計算してください。

解答を見る
    
    
    import numpy as np
    
    rg_data = np.array([14.5, 15.2, 14.8, 15.5, 14.9])
    rg_mean = rg_data.mean()
    rg_std = rg_data.std()
    
    print(f"Rg平均: {rg_mean:.2f} Å")
    print(f"Rg標準偏差: {rg_std:.2f} Å")
    # 出力: 平均 14.98 Å, 標準偏差 0.35 Å

#### 演習7: カスタム記述子設計（Medium）

芳香環率（芳香族原子数/全原子数）をRDKitで計算する関数を実装してください。

解答を見る
    
    
    from rdkit import Chem
    
    def calculate_aromatic_ratio(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
    
        aromatic_atoms = sum([1 for atom in mol.GetAtoms() if atom.GetIsAromatic()])
        total_atoms = mol.GetNumAtoms()
    
        aromatic_ratio = aromatic_atoms / total_atoms if total_atoms > 0 else 0
        return aromatic_ratio
    
    # テスト: ポリスチレン
    smiles_ps = "CC(c1ccccc1)"
    ratio = calculate_aromatic_ratio(smiles_ps)
    print(f"芳香環率: {ratio:.3f}")
    # 出力: 0.667（6芳香族原子 / 9全原子）

#### 演習8: ワークフロー統合（Hard）

PolymerAnalysisクラスに、複数候補を比較する`compare_polymers()`メソッドを追加してください。

解答を見る
    
    
    class PolymerAnalysisExtended(PolymerAnalysis):
        @staticmethod
        def compare_polymers(polymer_list):
            """
            複数高分子の比較
    
            Parameters:
            - polymer_list: [(name, smiles), ...] のリスト
    
            Returns:
            - comparison_df: 比較結果DataFrame
            """
            import pandas as pd
    
            results = []
            for name, smiles in polymer_list:
                poly = PolymerAnalysis(name, smiles)
                poly.calculate_descriptors()
                poly.predict_tg()
    
                results.append({
                    'Name': name,
                    'MW': poly.descriptors['MW'],
                    'LogP': poly.descriptors['LogP'],
                    'Predicted_Tg': poly.predicted_tg
                })
    
            df = pd.DataFrame(results)
            print(df)
            return df
    
    # 使用例
    polymers = [
        ("PS", "CC(c1ccccc1)"),
        ("PMMA", "CC(C)(C(=O)OC)"),
        ("PVC", "CC(Cl)")
    ]
    
    PolymerAnalysisExtended.compare_polymers(polymers)

#### 演習9: 最適化ループ実装（Hard）

目標Tg = 400Kに最も近い構造を見つけるため、遺伝的アルゴリズムの簡易版を実装してください（世代数5、個体数10）。

解答を見る
    
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    def genetic_algorithm_tg_optimization(target_tg=400, generations=5, population_size=10):
        """
        遺伝的アルゴリズムによるTg最適化
        """
        # 初期個体群（SMILESのバリエーション）
        monomers = ["CC", "CC(C)", "CC(c1ccccc1)", "CC(C)(C(=O)OC)", "CC(Cl)", "CC(C#N)"]
    
        population = np.random.choice(monomers, size=population_size)
    
        for gen in range(generations):
            # 適応度評価
            fitness_scores = []
            for smiles in population:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    fitness_scores.append(1e6)  # ペナルティ
                    continue
    
                mw = Descriptors.MolWt(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                rot = Descriptors.NumRotatableBonds(mol)
    
                tg_est = 250 + 0.5*mw + 10*hbd + 5*hba - 20*rot
                fitness = abs(tg_est - target_tg)  # 誤差が小さいほど良い
                fitness_scores.append(fitness)
    
            # 選択（エリート保存）
            elite_idx = np.argmin(fitness_scores)
            elite = population[elite_idx]
    
            print(f"Generation {gen+1}: Best = {elite}, Fitness = {fitness_scores[elite_idx]:.1f}K")
    
            # 次世代生成（簡易版：ランダム変異）
            population = np.random.choice(monomers, size=population_size)
            population[0] = elite  # エリート保存
    
        return elite
    
    best_smiles = genetic_algorithm_tg_optimization(target_tg=400, generations=5, population_size=10)
    print(f"\n最適SMILES: {best_smiles}")

#### 演習10: レポート自動生成（Hard）

PolymerAnalysisの結果をPDF形式で出力する機能を実装してください（matplotlib + reportlab使用）。

解答を見る

**実装方針:**

  * matplotlibで可視化を生成（PNG保存）
  * reportlabでPDFキャンバスを作成
  * テキストと画像を配置

    
    
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    
    def generate_pdf_report(polymer_analysis, filename='polymer_report.pdf'):
        """
        PDF レポート生成
    
        Parameters:
        - polymer_analysis: PolymerAnalysisインスタンス
        - filename: 出力ファイル名
        """
        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4
    
        # タイトル
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, f"Polymer Analysis Report: {polymer_analysis.polymer_name}")
    
        # 基本情報
        c.setFont("Helvetica", 12)
        y = height - 100
        c.drawString(50, y, f"SMILES: {polymer_analysis.smiles}")
    
        # 記述子
        y -= 40
        c.drawString(50, y, "Molecular Descriptors:")
        y -= 20
        for key, value in polymer_analysis.descriptors.items():
            c.drawString(70, y, f"{key}: {value:.2f}")
            y -= 15
    
        # Tg予測
        y -= 20
        c.drawString(50, y, f"Predicted Tg: {polymer_analysis.predicted_tg:.1f} K")
    
        # グラフ挿入（事前に生成した画像）
        # img_path = f'{polymer_analysis.polymer_name}_structure.png'
        # c.drawImage(img_path, 50, y-300, width=400, height=250)
    
        c.save()
        print(f"PDFレポート生成: {filename}")
    
    # 使用例
    polymer = PolymerAnalysis("PMMA", "CC(C)(C(=O)OC)")
    polymer.calculate_descriptors()
    polymer.predict_tg()
    generate_pdf_report(polymer, filename='PMMA_report.pdf')

## 参考文献

  1. RDKit Documentation. (2024). _Open-Source Cheminformatics Software_. Available at: https://www.rdkit.org/docs/
  2. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. _Journal of Machine Learning Research_ , 12, 2825-2830.
  3. Michaud-Agrawal, N., Denning, E. J., Woolf, T. B., & Beckstein, O. (2011). MDAnalysis: A toolkit for the analysis of molecular dynamics simulations. _Journal of Computational Chemistry_ , 32(10), 2319-2327.
  4. Kim, C., Chandrasekaran, A., Huan, T. D., Das, D., & Ramprasad, R. (2018). Polymer Genome: A Data-Powered Polymer Informatics Platform for Property Predictions. _The Journal of Physical Chemistry C_ , 122(31), 17575-17585.
  5. Ramprasad, R., Batra, R., Pilania, G., Mannodi-Kanakkithodi, A., & Kim, C. (2017). Machine learning in materials informatics: recent applications and prospects. _npj Computational Materials_ , 3(1), 54. https://doi.org/10.1038/s41524-017-0056-5
  6. Weininger, D. (1988). SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. _Journal of Chemical Information and Computer Sciences_ , 28(1), 31-36.
  7. Rogers, D., & Hahn, M. (2010). Extended-Connectivity Fingerprints. _Journal of Chemical Information and Modeling_ , 50(5), 742-754.

### シリーズ完結

本章で、高分子材料入門シリーズは完結です。第1章の基礎理論から第5章の実践的Pythonワークフローまで、高分子材料データサイエンスの全体像を学びました。RDKitによる構造生成、機械学習による物性予測、MDシミュレーションデータ解析、そしてデータベース連携まで、実務で即戦力となるスキルを習得できました。 

**次のステップ:** 本シリーズで学んだ知識を基盤に、実際の研究プロジェクトや材料開発に応用してください。PolyInfoやMaterials Projectなどのデータベースを活用し、最新の機械学習手法（Graph Neural Networksなど）にも挑戦してみましょう。 

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
