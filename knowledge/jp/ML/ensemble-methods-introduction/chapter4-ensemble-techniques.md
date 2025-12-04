---
title: 第4章：高度なアンサンブル技術
chapter_title: 第4章：高度なアンサンブル技術
subtitle: スタッキング、ブレンディング、Kaggle戦略による実践的アンサンブル構築
reading_time: 32分
difficulty: 上級
code_examples: 8
exercises: 4
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ スタッキング（Stacking）の多層構造を理解し実装できる
  * ✅ ブレンディング（Blending）で最適な重み付けを設計できる
  * ✅ Kaggleコンペティションでの高度なアンサンブル戦略を習得できる
  * ✅ モデル多様性と特徴量エンジニアリングを活用できる
  * ✅ 実践的なコンペティション用パイプラインを構築できる
  * ✅ 本番環境でのアンサンブル運用のベストプラクティスを理解できる

* * *

## 4.1 スタッキング（Stacking）の実践

### スタッキングの概念

スタッキングは、複数のベースモデルの予測結果をメタモデルで学習する手法です。Votingと異なり、メタモデルが最適な組み合わせ方を自動的に学習します。
    
    
    ```mermaid
    graph TB
        A[訓練データ] --> B1[Random Forest]
        A --> B2[XGBoost]
        A --> B3[LightGBM]
        A --> B4[Neural Network]
    
        B1 --> C[交差検証予測]
        B2 --> C
        B3 --> C
        B4 --> C
    
        C --> D[メタモデルLogistic Regression]
        D --> E[最終予測]
    
        style A fill:#e3f2fd
        style E fill:#c8e6c9
        style D fill:#fff3e0
    ```

### 基本的なスタッキングの実装
    
    
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    # データ生成
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    
    # 訓練・テストに分割
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # ベースモデルの定義
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgb', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]
    
    # メタ特徴量の作成（交差検証による予測）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_features_test = np.zeros((X_test.shape[0], len(base_models)))
    
    for i, (name, model) in enumerate(base_models):
        print(f"Training {name}...")
    
        # 訓練データでの交差検証予測（out-of-fold predictions）
        meta_features_train[:, i] = cross_val_predict(
            model, X_train, y_train, cv=cv, method='predict_proba'
        )[:, 1]
    
        # テストデータでの予測
        model.fit(X_train, y_train)
        meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]
    
    # メタモデルの訓練
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_features_train, y_train)
    
    # 最終予測
    y_pred = meta_model.predict(meta_features_test)
    y_pred_proba = meta_model.predict_proba(meta_features_test)[:, 1]
    
    print(f"\n=== Stacking Results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # メタモデルの重み（各ベースモデルの重要度）
    print(f"\nMeta-model coefficients:")
    for (name, _), coef in zip(base_models, meta_model.coef_[0]):
        print(f"  {name}: {coef:.4f}")
    

### mlxtendを使った簡潔な実装
    
    
    from mlxtend.classifier import StackingCVClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # ベースモデル
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf3 = SVC(probability=True, random_state=42)
    
    # メタモデル
    meta_clf = LogisticRegression()
    
    # StackingCVClassifier（自動的にCV予測を行う）
    stacking = StackingCVClassifier(
        classifiers=[clf1, clf2, clf3],
        meta_classifier=meta_clf,
        cv=5,
        use_probas=True,  # 確率を使用
        random_state=42
    )
    
    # 訓練と評価
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    
    print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # 各モデルの個別性能と比較
    for name, clf in [('RF', clf1), ('GB', clf2), ('SVC', clf3)]:
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"{name} Accuracy: {acc:.4f}")
    

### 多層スタッキング（Multi-level Stacking）
    
    
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.neural_network import MLPClassifier
    
    # レベル1のベースモデル
    level1_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgb', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
    ]
    
    # レベル2のメタモデル
    level2_models = [
        ('lr', LogisticRegression(random_state=42)),
        ('ridge', RidgeClassifier(random_state=42)),
    ]
    
    # レベル3の最終メタモデル
    level3_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # レベル1: ベースモデルの予測
    meta_l1_train = np.zeros((X_train.shape[0], len(level1_models)))
    meta_l1_test = np.zeros((X_test.shape[0], len(level1_models)))
    
    for i, (name, model) in enumerate(level1_models):
        print(f"Level 1 - Training {name}...")
        meta_l1_train[:, i] = cross_val_predict(
            model, X_train, y_train, cv=cv, method='predict_proba'
        )[:, 1]
        model.fit(X_train, y_train)
        meta_l1_test[:, i] = model.predict_proba(X_test)[:, 1]
    
    # レベル2: メタモデルの予測
    meta_l2_train = np.zeros((X_train.shape[0], len(level2_models)))
    meta_l2_test = np.zeros((X_test.shape[0], len(level2_models)))
    
    for i, (name, model) in enumerate(level2_models):
        print(f"Level 2 - Training {name}...")
        meta_l2_train[:, i] = cross_val_predict(
            model, meta_l1_train, y_train, cv=cv, method='decision_function'
        )
        model.fit(meta_l1_train, y_train)
        meta_l2_test[:, i] = model.decision_function(meta_l1_test)
    
    # レベル3: 最終予測
    print("Level 3 - Training final meta-model...")
    level3_model.fit(meta_l2_train, y_train)
    y_pred = level3_model.predict(meta_l2_test)
    y_pred_proba = level3_model.predict_proba(meta_l2_test)[:, 1]
    
    print(f"\n=== Multi-level Stacking Results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    

* * *

## 4.2 ブレンディング（Blending）

### ブレンディングとスタッキングの違い

項目 | スタッキング | ブレンディング  
---|---|---  
**データ分割** | 交差検証（全データ使用） | ホールドアウト（一部をメタ訓練用）  
**計算コスト** | 高い（CV分のモデル訓練） | 低い（1回のみ）  
**データ効率** | 高い（全データ活用） | やや低い（分割が必要）  
**過学習リスク** | 低い（CVによる正則化） | やや高い（分割次第）  
**実装の簡潔性** | やや複雑 | シンプル  
  
### 重み付き平均ブレンディング
    
    
    from scipy.optimize import minimize
    from sklearn.metrics import log_loss
    
    # データを訓練/ブレンド/テストに分割
    X_train_base, X_blend, y_train_base, y_blend = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # ベースモデルの訓練と予測
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgb', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
    ]
    
    blend_train = np.zeros((X_blend.shape[0], len(base_models)))
    blend_test = np.zeros((X_test.shape[0], len(base_models)))
    
    for i, (name, model) in enumerate(base_models):
        print(f"Training {name}...")
        model.fit(X_train_base, y_train_base)
        blend_train[:, i] = model.predict_proba(X_blend)[:, 1]
        blend_test[:, i] = model.predict_proba(X_test)[:, 1]
    
    # 最適な重みを探索（log_lossを最小化）
    def blend_loss(weights):
        """重み付き平均のlog_loss"""
        weights = weights / weights.sum()  # 正規化
        blended = np.dot(blend_train, weights)
        return log_loss(y_blend, blended)
    
    # 制約：重みの合計=1、各重み≥0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(len(base_models))]
    initial_weights = np.ones(len(base_models)) / len(base_models)
    
    # 最適化
    result = minimize(blend_loss, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x / result.x.sum()
    
    print(f"\n=== Optimal Blending Weights ===")
    for (name, _), weight in zip(base_models, optimal_weights):
        print(f"{name}: {weight:.4f}")
    
    # 最終予測
    y_pred_proba = np.dot(blend_test, optimal_weights)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print(f"\nBlending Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Blending AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    

### ランク平均ブレンディング
    
    
    from scipy.stats import rankdata
    
    # 各モデルの予測をランクに変換
    rank_train = np.zeros_like(blend_train)
    rank_test = np.zeros_like(blend_test)
    
    for i in range(len(base_models)):
        rank_train[:, i] = rankdata(blend_train[:, i]) / len(blend_train)
        rank_test[:, i] = rankdata(blend_test[:, i]) / len(blend_test)
    
    # ランクの平均
    rank_avg_train = rank_train.mean(axis=1)
    rank_avg_test = rank_test.mean(axis=1)
    
    # 評価
    y_pred = (rank_avg_test > 0.5).astype(int)
    
    print(f"\n=== Rank Averaging Results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, rank_avg_test):.4f}")
    
    # 各モデルの個別性能と比較
    print(f"\n=== Individual Model Performance ===")
    for i, (name, _) in enumerate(base_models):
        pred = (blend_test[:, i] > 0.5).astype(int)
        print(f"{name} - Accuracy: {accuracy_score(y_test, pred):.4f}, "
              f"AUC: {roc_auc_score(y_test, blend_test[:, i]):.4f}")
    

* * *

## 4.3 Kaggleでのアンサンブル戦略

### モデル多様性の確保

高性能なアンサンブルには、異なる特性を持つモデルの組み合わせが不可欠です：

多様性の源泉 | 実装例 | 効果  
---|---|---  
**アルゴリズム** | ツリー系、線形、ニューラルネット | 異なる決定境界  
**特徴量セット** | 異なる特徴量エンジニアリング | 異なる情報源  
**ハイパーパラメータ** | 深さ、学習率、正則化の違い | バイアス・バリアンスの調整  
**サンプリング** | ブートストラップ、CV fold | データの異なる視点  
**ランダムシード** | 異なる初期化 | 局所解の多様性  
  
### 特徴量セットの多様化
    
    
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.decomposition import PCA
    
    # サンプルデータ（実際のKaggleではコンペのデータを使用）
    df = pd.DataFrame(X_train, columns=[f'feat_{i}' for i in range(X_train.shape[1])])
    df['target'] = y_train
    
    # 特徴量セット1: 元の特徴量 + 統計量
    def create_feature_set1(X):
        features = pd.DataFrame(X)
        features['mean'] = X.mean(axis=1)
        features['std'] = X.std(axis=1)
        features['max'] = X.max(axis=1)
        features['min'] = X.min(axis=1)
        return features.values
    
    # 特徴量セット2: 多項式特徴量
    def create_feature_set2(X):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        return poly.fit_transform(X[:, :5])  # 計算量削減のため最初の5特徴量のみ
    
    # 特徴量セット3: PCA特徴量
    def create_feature_set3(X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=10)
        return pca.fit_transform(X_scaled)
    
    # 各特徴量セットでモデルを訓練
    feature_sets = [
        ('original', lambda X: X),
        ('statistics', create_feature_set1),
        ('polynomial', create_feature_set2),
        ('pca', create_feature_set3)
    ]
    
    ensemble_predictions = []
    
    for name, feature_func in feature_sets:
        print(f"\n=== Training with {name} features ===")
    
        X_train_feat = feature_func(X_train)
        X_test_feat = feature_func(X_test)
    
        # XGBoostで訓練
        model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        model.fit(X_train_feat, y_train)
    
        pred_proba = model.predict_proba(X_test_feat)[:, 1]
        ensemble_predictions.append(pred_proba)
    
        print(f"AUC: {roc_auc_score(y_test, pred_proba):.4f}")
    
    # アンサンブル予測（単純平均）
    ensemble_avg = np.mean(ensemble_predictions, axis=0)
    print(f"\n=== Ensemble with Diverse Features ===")
    print(f"Ensemble AUC: {roc_auc_score(y_test, ensemble_avg):.4f}")
    

### Kaggle戦略のフルパイプライン
    
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    import warnings
    warnings.filterwarnings('ignore')
    
    class KaggleEnsemble:
        """Kaggleコンペ用アンサンブルパイプライン"""
    
        def __init__(self, n_folds=5, random_state=42):
            self.n_folds = n_folds
            self.random_state = random_state
            self.base_models = []
            self.meta_model = None
    
        def add_base_model(self, name, model):
            """ベースモデルを追加"""
            self.base_models.append((name, model))
    
        def set_meta_model(self, model):
            """メタモデルを設定"""
            self.meta_model = model
    
        def fit(self, X, y):
            """アンサンブルモデルの訓練"""
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                random_state=self.random_state)
    
            # メタ特徴量の初期化
            self.meta_features_ = np.zeros((X.shape[0], len(self.base_models)))
    
            # 各ベースモデルでOOF予測を生成
            for i, (name, model) in enumerate(self.base_models):
                print(f"Training {name}...")
                oof_predictions = np.zeros(X.shape[0])
    
                for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
                    model.fit(X_train_fold, y_train_fold)
                    oof_predictions[val_idx] = model.predict_proba(X_val_fold)[:, 1]
    
                self.meta_features_[:, i] = oof_predictions
    
                # 全データで再訓練
                model.fit(X, y)
    
                # OOFスコア
                auc = roc_auc_score(y, oof_predictions)
                print(f"  {name} OOF AUC: {auc:.4f}")
    
            # メタモデルの訓練
            print("Training meta-model...")
            self.meta_model.fit(self.meta_features_, y)
    
            return self
    
        def predict_proba(self, X):
            """確率予測"""
            # ベースモデルの予測
            base_predictions = np.zeros((X.shape[0], len(self.base_models)))
            for i, (name, model) in enumerate(self.base_models):
                base_predictions[:, i] = model.predict_proba(X)[:, 1]
    
            # メタモデルで最終予測
            return self.meta_model.predict_proba(base_predictions)
    
        def predict(self, X):
            """クラス予測"""
            return self.meta_model.predict(
                np.column_stack([model.predict_proba(X)[:, 1]
                               for _, model in self.base_models])
            )
    
    # 使用例
    ensemble = KaggleEnsemble(n_folds=5, random_state=42)
    
    # ベースモデルの追加（多様性を確保）
    ensemble.add_base_model('rf', RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    ))
    ensemble.add_base_model('xgb', XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss'
    ))
    ensemble.add_base_model('lgb', LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        random_state=42, verbose=-1
    ))
    
    # メタモデルの設定
    ensemble.set_meta_model(LogisticRegression(random_state=42))
    
    # 訓練
    print("=== Training Kaggle Ensemble ===\n")
    ensemble.fit(X_train, y_train)
    
    # テストデータで評価
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred = ensemble.predict(X_test)
    
    print(f"\n=== Final Ensemble Performance ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    

* * *

## 4.4 実践プロジェクト：コンペティション用パイプライン

### プロジェクト概要

二値分類コンペティションを想定した、完全なアンサンブルパイプラインを構築します。

#### 要件

  * 複数の特徴量エンジニアリング手法
  * 異なるアルゴリズムのベースモデル（5種類以上）
  * 2層スタッキング構造
  * 交差検証によるモデル選択
  * 予測結果の提出ファイル生成

    
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.neural_network import MLPClassifier
    import pandas as pd
    
    class CompetitionPipeline:
        """コンペティション用完全パイプライン"""
    
        def __init__(self, n_folds=5):
            self.n_folds = n_folds
            self.level1_models = self._create_level1_models()
            self.level2_models = self._create_level2_models()
            self.final_model = LogisticRegression(C=0.1, random_state=42)
    
        def _create_level1_models(self):
            """レベル1：多様なベースモデル"""
            return [
                ('rf', RandomForestClassifier(
                    n_estimators=300, max_depth=12, min_samples_split=5,
                    random_state=42, n_jobs=-1
                )),
                ('et', ExtraTreesClassifier(
                    n_estimators=300, max_depth=12, min_samples_split=5,
                    random_state=43, n_jobs=-1
                )),
                ('xgb1', XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, eval_metric='logloss', n_jobs=-1
                )),
                ('xgb2', XGBClassifier(
                    n_estimators=300, max_depth=8, learning_rate=0.03,
                    subsample=0.7, colsample_bytree=0.7,
                    random_state=43, eval_metric='logloss', n_jobs=-1
                )),
                ('lgb1', LGBMClassifier(
                    n_estimators=300, max_depth=8, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbose=-1, n_jobs=-1
                )),
                ('lgb2', LGBMClassifier(
                    n_estimators=300, max_depth=10, learning_rate=0.03,
                    subsample=0.7, colsample_bytree=0.7,
                    random_state=43, verbose=-1, n_jobs=-1
                ))
            ]
    
        def _create_level2_models(self):
            """レベル2：メタモデル"""
            return [
                ('lr', LogisticRegression(C=1.0, random_state=42)),
                ('gb', GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    random_state=42
                )),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=(50, 20), max_iter=1000,
                    random_state=42, early_stopping=True
                ))
            ]
    
        def fit(self, X, y, X_test=None):
            """パイプライン全体の訓練"""
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
    
            # レベル1の訓練とOOF予測
            print("=" * 60)
            print("LEVEL 1: Training Base Models")
            print("=" * 60)
    
            oof_l1 = np.zeros((X.shape[0], len(self.level1_models)))
            test_l1 = np.zeros((X_test.shape[0] if X_test is not None else 0,
                              len(self.level1_models)))
    
            for i, (name, model) in enumerate(self.level1_models):
                print(f"\nModel {i+1}/{len(self.level1_models)}: {name}")
                oof_preds = np.zeros(X.shape[0])
                test_preds = np.zeros(X_test.shape[0] if X_test is not None else 0)
    
                for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
    
                    model.fit(X_train, y_train)
                    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
                    if X_test is not None:
                        test_preds += model.predict_proba(X_test)[:, 1] / self.n_folds
    
                    fold_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                    print(f"  Fold {fold+1}: AUC = {fold_auc:.4f}")
    
                oof_l1[:, i] = oof_preds
                if X_test is not None:
                    test_l1[:, i] = test_preds
    
                oof_auc = roc_auc_score(y, oof_preds)
                print(f"  OOF AUC: {oof_auc:.4f}")
    
            # レベル2の訓練
            print("\n" + "=" * 60)
            print("LEVEL 2: Training Meta Models")
            print("=" * 60)
    
            oof_l2 = np.zeros((X.shape[0], len(self.level2_models)))
            test_l2 = np.zeros((X_test.shape[0] if X_test is not None else 0,
                              len(self.level2_models)))
    
            for i, (name, model) in enumerate(self.level2_models):
                print(f"\nMeta Model {i+1}/{len(self.level2_models)}: {name}")
                oof_preds = np.zeros(X.shape[0])
                test_preds = np.zeros(X_test.shape[0] if X_test is not None else 0)
    
                for fold, (train_idx, val_idx) in enumerate(cv.split(oof_l1, y)):
                    X_train, X_val = oof_l1[train_idx], oof_l1[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
    
                    model.fit(X_train, y_train)
                    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
                    if X_test is not None:
                        test_preds += model.predict_proba(test_l1)[:, 1] / self.n_folds
    
                oof_l2[:, i] = oof_preds
                if X_test is not None:
                    test_l2[:, i] = test_preds
    
                oof_auc = roc_auc_score(y, oof_preds)
                print(f"  OOF AUC: {oof_auc:.4f}")
    
            # 最終モデルの訓練
            print("\n" + "=" * 60)
            print("FINAL: Training Ensemble Model")
            print("=" * 60)
    
            self.final_model.fit(oof_l2, y)
            oof_final = self.final_model.predict_proba(oof_l2)[:, 1]
            final_auc = roc_auc_score(y, oof_final)
    
            print(f"\nFinal Ensemble OOF AUC: {final_auc:.4f}")
    
            if X_test is not None:
                self.test_predictions_ = self.final_model.predict_proba(test_l2)[:, 1]
    
            return self
    
        def predict_proba(self, X):
            """確率予測（テストデータ用）"""
            return self.test_predictions_
    
    # 実行例
    print("Initializing Competition Pipeline...")
    pipeline = CompetitionPipeline(n_folds=5)
    
    print("\nTraining pipeline...")
    pipeline.fit(X_train, y_train, X_test)
    
    # 最終予測
    final_predictions = pipeline.predict_proba(X_test)
    final_auc = roc_auc_score(y_test, final_predictions)
    
    print("\n" + "=" * 60)
    print(f"FINAL TEST AUC: {final_auc:.4f}")
    print("=" * 60)
    
    # 提出ファイルの生成（実際のKaggleでは適切なフォーマットに変更）
    submission = pd.DataFrame({
        'id': range(len(final_predictions)),
        'prediction': final_predictions
    })
    print("\nSubmission file preview:")
    print(submission.head())
    

* * *

## 4.5 ベストプラクティスとトラブルシューティング

### よくある失敗とその対策

問題 | 原因 | 対策  
---|---|---  
**OOF予測が訓練に使われる** | データリークで過学習 | 必ず交差検証でOOF予測を生成  
**類似モデルのみ** | 多様性不足 | 異なるアルゴリズム・特徴量を使用  
**アンサンブルが個別より悪い** | 弱いモデルが悪影響 | モデル選択・重み付けを導入  
**過度に複雑な構造** | オーバーエンジニアリング | シンプルな構造から始める  
**計算時間が長すぎる** | 非効率なパイプライン | ブレンディング・並列化を検討  
  
### 本番環境での運用Tips

モデルのバージョン管理

  * 各モデルにバージョン番号を付与
  * 訓練時のハイパーパラメータを記録
  * OOFスコアと実運用スコアを追跡
  * モデルファイルとメタデータを一緒に保存

    
    
    import pickle
    import json
    from datetime import datetime
    
    # モデルの保存
    model_metadata = {
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'oof_auc': 0.8542,
        'base_models': ['rf', 'xgb', 'lgb'],
        'hyperparameters': {
            'n_folds': 5,
            'random_state': 42
        }
    }
    
    with open('ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    
    with open('ensemble_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    

推論速度の最適化

  * 不要なモデルを削除（貢献度が低いもの）
  * モデル数を減らす（Top-Kモデルのみ）
  * 軽量なメタモデルを使用（線形モデル）
  * 予測を事前計算・キャッシュ化

デバッグ手法

  * 各ベースモデルの個別性能を確認
  * モデル間の相関を分析（高相関なら多様性不足）
  * メタ特徴量の分布を可視化
  * CV foldごとのスコア変動を確認

    
    
    # モデル間の相関を確認
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    correlation_matrix = np.corrcoef(meta_features_train.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f',
                xticklabels=[name for name, _ in base_models],
                yticklabels=[name for name, _ in base_models],
                cmap='coolwarm', center=0)
    plt.title('Base Model Predictions Correlation')
    plt.tight_layout()
    plt.show()
    
    # 高相関（>0.95）のペアを表示
    print("High correlation pairs (>0.95):")
    for i in range(len(base_models)):
        for j in range(i+1, len(base_models)):
            if correlation_matrix[i, j] > 0.95:
                print(f"{base_models[i][0]} - {base_models[j][0]}: {correlation_matrix[i, j]:.3f}")
    

### まとめ

> **アンサンブルの成功の鍵**
> 
>   * **多様性** : 異なるアルゴリズム・特徴量・ハイパーパラメータを組み合わせる
>   * **適切な検証** : 交差検証でOOF予測を生成し、データリークを防ぐ
>   * **段階的構築** : シンプルな構造から始め、必要に応じて複雑化する
>   * **性能監視** : 各モデルの貢献度を追跡し、不要なモデルを削除する
>   * **実運用を意識** : 計算コストと予測精度のバランスを取る
> 

* * *

## 練習問題

問題1：2層スタッキングの実装（難易度：中）

以下の要件で2層スタッキングモデルを実装してください：

  * レベル1: Random Forest、XGBoost、LightGBM
  * レベル2: Logistic Regression
  * 5-fold交差検証でOOF予測を生成
  * 各レベルのOOF AUCを表示

    
    
    # ヒント
    from sklearn.model_selection import cross_val_predict
    
    # レベル1のOOF予測を生成
    # meta_features_l1 = ...
    
    # レベル2のOOF予測を生成
    # meta_features_l2 = ...
    

問題2：最適なブレンド重みの探索（難易度：中）

3つのモデルの予測に対して、AUCを最大化する最適な重みを探索してください。scipy.optimize.minimizeを使用すること。
    
    
    # ヒント
    from scipy.optimize import minimize
    
    def objective(weights):
        # 重み付き平均の予測
        blended = ...
        # AUCを最大化 → 負のAUCを最小化
        return -roc_auc_score(y_true, blended)
    

問題3：モデル多様性の分析（難易度：低）

5つのベースモデルの予測値の相関行列を計算し、ヒートマップで可視化してください。相関が0.95以上のペアを特定してください。

問題4：コンペティション用パイプライン（難易度：高）

以下を含む完全なパイプラインを構築してください：

  * 最低5種類の異なるベースモデル
  * 2種類の特徴量セット（元の特徴量と加工後の特徴量）
  * スタッキングまたはブレンディング
  * 交差検証によるOOFスコア計算
  * 提出用ファイルの生成（CSV形式）

* * *

## 参考文献・リソース

### 論文・書籍

  * Wolpert, D. H. (1992). "Stacked generalization." Neural Networks, 5(2), 241-259.
  * Breiman, L. (1996). "Stacked regressions." Machine Learning, 24(1), 49-64.
  * Kaggle Ensembling Guide: <https://mlwave.com/kaggle-ensembling-guide/>

### 実装リソース

  * [mlxtend.classifier.StackingCVClassifier](<http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/>)
  * [sklearn.ensemble.StackingClassifier](<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html>)
  * [vecstack: Stacking library](<https://github.com/vecxoz/vecstack>)

### Kaggleカーネル

  * [Introduction to Ensembling/Stacking in Python](<https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python>)
  * [Stacked Regressions: Top 4% on Leaderboard](<https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard>)

* * *
