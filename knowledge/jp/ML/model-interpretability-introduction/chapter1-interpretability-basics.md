---
title: 第1章：モデル解釈性基礎
chapter_title: 第1章：モデル解釈性基礎
subtitle: 信頼できるAIシステムを構築するための解釈性の理解
reading_time: 30-35分
difficulty: 初級
code_examples: 8
exercises: 6
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ モデル解釈性が重要な理由を理解する
  * ✅ 解釈性の分類体系を把握する
  * ✅ 解釈可能なモデルの特徴を知る
  * ✅ 主要な解釈手法の概要を理解する
  * ✅ 解釈性を評価する基準を学ぶ
  * ✅ 実践的な解釈可能モデルを実装できる

* * *

## 1.1 なぜモデル解釈性が重要か

### 信頼性と説明責任

機械学習モデルの予測を信頼するためには、「なぜその予測に至ったのか」を理解する必要があります。特に高リスクな意思決定（医療診断、融資審査、刑事司法など）では、説明責任が不可欠です。

適用領域 | 解釈性が必要な理由 | リスク  
---|---|---  
**医療診断** | 医師が診断根拠を理解し、患者に説明する必要がある | 誤診による生命の危険  
**融資審査** | 拒否理由の説明義務、公正性の確保 | 差別的な判断、法的訴訟  
**刑事司法** | 再犯リスク評価の根拠を示す必要がある | 不当な判決、人権侵害  
**自動運転** | 事故時の責任追及、安全性の検証 | 人命損失、法的責任  
  
> **重要** : 「予測精度が高い」だけでは不十分です。ステークホルダーがモデルを信頼し、適切に利用するには、予測の根拠を理解できる必要があります。

### 規制要件（GDPR、AI規制）

世界中で機械学習モデルの透明性に関する規制が強化されています：

  * **GDPR（欧州一般データ保護規則）** : 自動化された意思決定に関する「説明を受ける権利」を規定（第22条）
  * **EU AI Act** : 高リスクAIシステムに対する透明性と説明可能性の要件
  * **米国公正信用報告法** : 信用スコアに関する「不利な措置の通知」義務
  * **日本の個人情報保護法** : 自動化された意思決定に関する本人への情報提供

### デバッグとモデル改善

解釈性はモデルの性能向上にも不可欠です：
    
    
    """
    例: モデルが予期しない予測をする場合の診断
    
    問題: 顧客の離反予測モデルが実運用で性能が低い
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'tenure_months': np.random.randint(1, 120, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 10000, n_samples),
        'num_support_calls': np.random.poisson(2, n_samples),
        'contract_type': np.random.choice(['month', 'year', '2year'], n_samples),
        'customer_id': np.arange(n_samples)  # データリーク！
    })
    
    # ターゲット変数（離反）
    data['churn'] = ((data['num_support_calls'] > 3) |
                     (data['monthly_charges'] > 100)).astype(int)
    
    # モデル訓練
    X = data.drop('churn', axis=1)
    X_encoded = pd.get_dummies(X, columns=['contract_type'])
    y = data['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Feature Importanceで診断
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance.head(10))
    
    # 問題発見: customer_idが最も重要な特徴量になっている（データリーク）
    print("\n⚠️ customer_idの重要度が異常に高い → データリークの可能性")
    

### バイアス検出

解釈性により、モデルが学習した不公平なパターンを発見できます：
    
    
    """
    例: 採用スクリーニングモデルのバイアス検出
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # バイアスのあるサンプルデータ
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'years_experience': np.random.randint(0, 20, n_samples),
        'education_level': np.random.randint(1, 5, n_samples),
        'skills_score': np.random.uniform(0, 100, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'age': np.random.randint(22, 65, n_samples)
    })
    
    # バイアスのあるターゲット（性別による差別が含まれる）
    data['hired'] = (
        (data['years_experience'] > 5) &
        (data['skills_score'] > 60) &
        (data['gender'] == 'M')  # 性別バイアス
    ).astype(int)
    
    # モデル訓練
    X = pd.get_dummies(data.drop('hired', axis=1), columns=['gender'])
    y = data['hired']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # 係数を確認してバイアスを検出
    coefficients = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', ascending=False)
    
    print("Model Coefficients:")
    print(coefficients)
    
    # gender_Mの係数が異常に高い → 性別バイアスを検出
    print("\n⚠️ gender_Mの係数が高い → 性別による差別の可能性")
    print("📊 公正性の評価が必要")
    

* * *

## 1.2 解釈性の分類

### グローバル解釈 vs ローカル解釈

分類 | 説明 | 質問 | 手法例  
---|---|---|---  
**グローバル解釈** | モデル全体の振る舞いを理解 | 「モデルは一般的にどう予測するか？」 | Feature Importance, Partial Dependence  
**ローカル解釈** | 個別の予測を説明 | 「なぜこの顧客は離反すると予測されたか？」 | LIME, SHAP, Counterfactual  
      
    
    """
    例: グローバル解釈 vs ローカル解釈
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.uniform(20000, 150000, n_samples),
        'debt_ratio': np.random.uniform(0, 1, n_samples),
        'credit_history_months': np.random.randint(0, 360, n_samples)
    })
    
    # ターゲット: ローン承認
    data['approved'] = (
        (data['income'] > 50000) &
        (data['debt_ratio'] < 0.5) &
        (data['credit_history_months'] > 24)
    ).astype(int)
    
    X = data.drop('approved', axis=1)
    y = data['approved']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # --- グローバル解釈: Feature Importance ---
    print("=== グローバル解釈 ===")
    print("モデル全体で最も重要な特徴量:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # --- ローカル解釈: 個別の予測説明 ---
    print("\n=== ローカル解釈 ===")
    # テストデータから1サンプル選択
    sample_idx = 0
    sample = X_test.iloc[sample_idx:sample_idx+1]
    prediction = model.predict(sample)[0]
    prediction_proba = model.predict_proba(sample)[0]
    
    print(f"サンプル {sample_idx} の特徴:")
    print(sample.T)
    print(f"\n予測: {'承認' if prediction == 1 else '却下'}")
    print(f"確率: {prediction_proba[1]:.2%}")
    
    # 簡易的なローカル重要度（ツリーベース）
    # 実際にはSHAPやLIMEを使用するのが望ましい
    print("\nこの予測に寄与した特徴（概算）:")
    for feature in X.columns:
        print(f"  {feature}: {sample[feature].values[0]:.2f}")
    

### モデル固有 vs モデル非依存

分類 | 説明 | 利点 | 欠点  
---|---|---|---  
**モデル固有** | 特定のモデルに特化した解釈 | 正確、効率的 | 他のモデルに適用できない  
**モデル非依存** | どのモデルにも適用可能 | 汎用性が高い | 計算コストが高い場合がある  
  
### 事前解釈性 vs 事後解釈性

  * **事前解釈性（Intrinsic Interpretability）** : モデル自体が解釈可能（線形回帰、決定木）
  * **事後解釈性（Post-hoc Interpretability）** : ブラックボックスモデルを後から解釈（SHAP、LIME）

### 解釈性の分類体系
    
    
    ```mermaid
    graph TB
        A[モデル解釈性] --> B[スコープ]
        A --> C[依存性]
        A --> D[タイミング]
    
        B --> B1[グローバル解釈モデル全体の振る舞い]
        B --> B2[ローカル解釈個別予測の説明]
    
        C --> C1[モデル固有特定モデル用]
        C --> C2[モデル非依存汎用的]
    
        D --> D1[事前解釈性本質的に解釈可能]
        D --> D2[事後解釈性後付け説明]
    
        style A fill:#7b2cbf,color:#fff
        style B1 fill:#e3f2fd
        style B2 fill:#e3f2fd
        style C1 fill:#fff3e0
        style C2 fill:#fff3e0
        style D1 fill:#c8e6c9
        style D2 fill:#c8e6c9
    ```

* * *

## 1.3 解釈可能なモデル

### 線形回帰

線形回帰は最も解釈しやすいモデルの一つです。各特徴量の係数が直接的に影響を示します。

**数式** :

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$$

$\beta_i$ は特徴量 $x_i$ の1単位変化に対する予測値の変化量を示します。
    
    
    """
    例: 線形回帰による住宅価格予測
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 200
    
    data = pd.DataFrame({
        'square_feet': np.random.randint(500, 4000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'age_years': np.random.randint(0, 50, n_samples),
        'distance_to_city': np.random.uniform(0, 50, n_samples)
    })
    
    # ターゲット: 価格（万円）
    data['price'] = (
        data['square_feet'] * 0.5 +
        data['bedrooms'] * 50 -
        data['age_years'] * 5 -
        data['distance_to_city'] * 10 +
        np.random.normal(0, 100, n_samples)
    )
    
    X = data.drop('price', axis=1)
    y = data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 標準化（係数の比較のため）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # モデル訓練
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 係数の解釈
    coefficients = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("線形回帰モデルの係数:")
    print(coefficients)
    print(f"\n切片: {model.intercept_:.2f}")
    
    print("\n解釈:")
    print("- square_feet の係数が最も大きい → 面積が価格に最も影響")
    print("- age_years の係数が負 → 築年数が古いほど価格が低い")
    print("- 係数が標準化されているため、直接比較可能")
    
    # 予測例
    sample = X_test_scaled[0:1]
    prediction = model.predict(sample)[0]
    print(f"\nサンプル予測価格: {prediction:.2f}万円")
    

### 決定木

決定木は人間が理解しやすいルールベースの分岐構造を持ちます。
    
    
    """
    例: 決定木によるアイリス分類
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # データ読み込み
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 決定木モデル（深さを制限して解釈しやすく）
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # 精度
    accuracy = model.score(X_test, y_test)
    print(f"精度: {accuracy:.2%}")
    
    # ルールの抽出（テキスト形式）
    from sklearn.tree import export_text
    tree_rules = export_text(model, feature_names=list(iris.feature_names))
    print("\n決定木のルール:")
    print(tree_rules[:500] + "...")  # 最初の500文字のみ表示
    
    # 解釈例
    print("\n解釈:")
    print("- petal width (cm) <= 0.8 で setosa と判定")
    print("- それ以外は petal width や petal length で versicolor/virginica を判定")
    print("- 決定境界が明確で、専門家でなくても理解可能")
    

### ルールベースモデル

IF-THENルールで構成されるモデルは、ビジネスルールとして直接利用可能です。
    
    
    """
    例: シンプルなルールベース分類器
    """
    
    import numpy as np
    import pandas as pd
    
    class SimpleRuleClassifier:
        """解釈可能なルールベース分類器"""
    
        def __init__(self):
            self.rules = []
    
        def add_rule(self, condition, prediction, description=""):
            """ルールを追加"""
            self.rules.append({
                'condition': condition,
                'prediction': prediction,
                'description': description
            })
    
        def predict(self, X):
            """予測"""
            predictions = []
            for _, row in X.iterrows():
                prediction = None
                for rule in self.rules:
                    if rule['condition'](row):
                        prediction = rule['prediction']
                        break
                predictions.append(prediction if prediction is not None else 0)
            return np.array(predictions)
    
        def explain(self):
            """ルールを説明"""
            print("分類ルール:")
            for i, rule in enumerate(self.rules, 1):
                print(f"  Rule {i}: {rule['description']} → {rule['prediction']}")
    
    # 使用例: ローン承認ルール
    classifier = SimpleRuleClassifier()
    
    # ルール1: 高収入で低負債
    classifier.add_rule(
        condition=lambda row: row['income'] > 100000 and row['debt_ratio'] < 0.3,
        prediction=1,
        description="高収入（>100K）かつ低負債率（<30%）"
    )
    
    # ルール2: 中収入で良好な信用履歴
    classifier.add_rule(
        condition=lambda row: row['income'] > 50000 and row['credit_history_months'] > 36,
        prediction=1,
        description="中収入（>50K）かつ信用履歴3年以上"
    )
    
    # ルール3: それ以外は却下
    classifier.add_rule(
        condition=lambda row: True,
        prediction=0,
        description="その他のケース"
    )
    
    # テストデータ
    test_data = pd.DataFrame({
        'income': [120000, 60000, 30000],
        'debt_ratio': [0.2, 0.4, 0.6],
        'credit_history_months': [48, 40, 12]
    })
    
    predictions = classifier.predict(test_data)
    classifier.explain()
    
    print("\n予測結果:")
    for i, (pred, income) in enumerate(zip(predictions, test_data['income'])):
        print(f"  申請者 {i+1} (収入: ${income:,.0f}): {'承認' if pred == 1 else '却下'}")
    

### GAM (Generalized Additive Models)

GAMは、各特徴量の非線形効果を可視化できる解釈可能なモデルです。

**数式** :

$$g(\mathbb{E}[y]) = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_n(x_n)$$

$f_i$ は特徴量 $x_i$ の非線形関数です。
    
    
    """
    例: GAMによる非線形関係のモデリング
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    
    # サンプルデータ生成（非線形関係）
    np.random.seed(42)
    n_samples = 300
    
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(-3, 3, n_samples)
    
    # 非線形関係: sin関数と2次関数
    y = np.sin(x1) + x2**2 + np.random.normal(0, 0.2, n_samples)
    
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    
    # 特徴量エンジニアリング: 多項式特徴を追加（GAMの近似）
    from sklearn.preprocessing import PolynomialFeatures
    
    X = data[['x1', 'x2']]
    poly = PolynomialFeatures(degree=3, include_bias=False, interaction_features=False)
    X_poly = poly.fit_transform(X)
    
    feature_names = poly.get_feature_names_out(['x1', 'x2'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, data['y'], test_size=0.2, random_state=42
    )
    
    # リッジ回帰で訓練
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    print(f"テストR²スコア: {model.score(X_test, y_test):.3f}")
    
    # 各特徴量の効果を可視化
    print("\n各特徴量の多項式係数:")
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_
    })
    print(coef_df)
    
    print("\n解釈:")
    print("- x1の奇数次の項が重要 → sin関数的な非線形性")
    print("- x2の2次の項が重要 → 2次関数的な関係")
    print("- 各変数の効果を個別に解釈可能")
    

* * *

## 1.4 解釈手法の概要

### Feature Importance

特徴量の重要度を定量化する手法。ツリーベースモデルで頻繁に使用されます。

  * **Mean Decrease Impurity** : 不純度の減少量で重要度を測定
  * **Permutation Importance** : 特徴量をシャッフルした際の性能低下で測定

### Partial Dependence Plot (PDP)

特定の特徴量とモデル予測の関係を可視化します。

**数式** :

$$\text{PDP}(x_s) = \mathbb{E}_{x_c}[f(x_s, x_c)]$$

$x_s$ は対象特徴量、$x_c$ はその他の特徴量です。

### SHAP (SHapley Additive exPlanations)

ゲーム理論のShapley値を用いて、各特徴量の貢献度を計算します。

**特徴** :

  * 一貫性のある説明
  * 局所的および大域的解釈が可能
  * モデル非依存

### LIME (Local Interpretable Model-agnostic Explanations)

個別の予測を、局所的に線形モデルで近似して説明します。

**手順** :

  1. 予測したいインスタンスの近傍にサンプルを生成
  2. ブラックボックスモデルで予測を取得
  3. 解釈可能なモデル（線形回帰など）でローカルに近似
  4. 近似モデルの係数を解釈

### Saliency Maps（顕著性マップ）

画像分類において、どのピクセルが予測に重要かを可視化します。

**計算方法** :

$$S(x) = \left| \frac{\partial f(x)}{\partial x} \right|$$

入力画像に対する勾配を計算し、重要な領域を強調表示します。

* * *

## 1.5 解釈性の評価

### Fidelity（忠実度）

解釈手法が元のモデルの振る舞いをどれだけ正確に説明しているかを測定します。

評価指標 | 説明 | 計算方法  
---|---|---  
**R²スコア** | 説明モデルと元モデルの一致度 | $R^2 = 1 - \frac{\sum(y_{\text{true}} - y_{\text{approx}})^2}{\sum(y_{\text{true}} - \bar{y})^2}$  
**Local Fidelity** | 局所的な予測の一致度 | 近傍サンプルでの予測誤差  
  
### Consistency（一貫性）

類似したインスタンスに対して類似した説明が得られるかを評価します。
    
    
    """
    例: 解釈の一貫性評価
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # サンプルデータ
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    })
    data['target'] = (data['feature1'] + data['feature2'] > 0).astype(int)
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # 類似サンプルのFeature Importance比較
    sample1 = X_test.iloc[0:1]
    sample2 = X_test.iloc[1:2]  # 類似サンプル
    
    # ツリーパスを使った簡易的なローカル重要度
    # （実際にはSHAPを使用することを推奨）
    
    print("サンプル1の特徴:")
    print(sample1.values)
    print(f"予測: {model.predict(sample1)[0]}")
    
    print("\nサンプル2の特徴:")
    print(sample2.values)
    print(f"予測: {model.predict(sample2)[0]}")
    
    # 距離計算
    distance = np.linalg.norm(sample1.values - sample2.values)
    print(f"\nサンプル間の距離: {distance:.3f}")
    print("一貫性評価: 類似サンプルに対する説明が類似しているか確認が必要")
    

### Stability（安定性）

入力データのわずかな変化に対して、解釈が大きく変わらないかを評価します。

### Comprehensibility（理解容易性）

人間が説明を理解しやすいかを評価します。定量化が難しいため、ユーザー調査が一般的です。

評価方法 | 説明  
---|---  
**ルール数** | 決定木やルールセットのルール数（少ないほど理解しやすい）  
**特徴量数** | 説明に使用される特徴量の数（少ないほど良い）  
**ユーザー調査** | 実際のユーザーによる理解度テスト  
  
* * *

## 練習問題

問題1: モデル解釈性の必要性

**問題** : 以下のシナリオで、モデル解釈性が特に重要である理由を説明してください。

  1. 銀行の融資審査システム
  2. 医療画像診断支援システム
  3. レコメンデーションシステム

**解答例** :

  1. **融資審査** : 拒否理由の説明義務（法的要件）、公正性の確保、差別的な判断の防止
  2. **医療診断** : 医師による診断根拠の理解、患者への説明、誤診のリスク軽減、医療過誤訴訟への対応
  3. **レコメンデーション** : ユーザー信頼の向上、推薦理由の透明性、バイアスの検出（フィルターバブルの回避）

問題2: グローバル解釈とローカル解釈

**問題** : 「顧客離反予測モデル」において、グローバル解釈とローカル解釈それぞれで知りたい情報の例を挙げてください。

**解答例** :

  * **グローバル解釈** : 
    * 「サポート問い合わせ回数」が離反に最も影響する特徴量である
    * 「契約期間」と離反確率の関係（長いほど離反率が低い傾向）
    * モデル全体で重要な上位5つの特徴量
  * **ローカル解釈** : 
    * 顧客A（ID=12345）が離反すると予測された理由（サポート問い合わせが10回以上、契約期間が3ヶ月未満など）
    * この顧客の離反確率を下げるために改善すべき要因

問題3: 解釈可能なモデルの選択

**問題** : 以下のシナリオで、どの解釈可能なモデルが適切か選択し、理由を説明してください。

  1. 住宅価格予測（特徴量: 面積、部屋数、築年数など）
  2. スパムメール分類（特徴量: 単語の出現頻度）
  3. 患者の再入院リスク予測（特徴量: 年齢、診断履歴、検査値など）

**解答例** :

  1. **線形回帰** : 各特徴量の係数が価格への影響を直接示すため、不動産業者や顧客が理解しやすい
  2. **決定木またはルールベース** : 「"無料"という単語が5回以上 → スパム」のようなルールが直感的
  3. **GAMまたは決定木** : 非線形な関係（例: 年齢と再入院リスクのU字型関係）を可視化できる。医師が診断ロジックを理解しやすい

問題4: データリークの検出

**問題** : Feature Importanceを使ってデータリークを検出する方法を説明し、コード例を示してください。

**解答例** :
    
    
    """
    データリークの検出方法
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    # 疑わしい特徴量のチェックリスト
    suspicious_features = [
        'id', 'timestamp', 'created_at', 'updated_at',
        'target', 'label', 'outcome'  # ターゲット変数そのものやそのリーク
    ]
    
    # Feature Importanceを計算
    model = RandomForestClassifier()
    # model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 上位の特徴量をチェック
    top_features = feature_importance.head(5)
    for _, row in top_features.iterrows():
        feature = row['feature']
        importance = row['importance']
    
        # 疑わしい特徴量が上位にあるか
        if any(suspect in feature.lower() for suspect in suspicious_features):
            print(f"⚠️ データリークの可能性: {feature} (重要度: {importance:.3f})")
    
        # 重要度が異常に高いか（>0.9）
        if importance > 0.9:
            print(f"⚠️ 異常に高い重要度: {feature} (重要度: {importance:.3f})")
    

問題5: 解釈性の評価

**問題** : 解釈手法の「Fidelity（忠実度）」を評価するために、どのような指標や方法を使用できますか？

**解答例** :

  * **R²スコア** : 説明モデル（LIMEの線形近似など）と元のブラックボックスモデルの予測の一致度
  * **予測誤差** : 説明モデルと元モデルの予測の平均絶対誤差（MAE）
  * **分類精度の比較** : 説明モデルが元モデルの予測をどれだけ正確に再現できるか
  * **ローカル忠実度** : 特定のインスタンスの近傍において、説明モデルがどれだけ正確か

問題6: 実装課題

**問題** : scikit-learnのタイタニックデータセット（または任意のデータ）を使用して、以下を実装してください。

  1. ロジスティック回帰モデルを訓練し、係数を解釈する
  2. 決定木モデルを訓練し、ルールを抽出する
  3. ランダムフォレストモデルを訓練し、Feature Importanceを可視化する
  4. 3つのモデルの解釈容易性を比較する

**ヒント** :
    
    
    from sklearn.datasets import fetch_openml
    import pandas as pd
    
    # データ読み込み
    titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
    df = titanic.frame
    
    # 前処理（欠損値処理、カテゴリカルエンコーディングなど）
    # ...
    
    # モデル訓練と解釈
    # ...
    

* * *

## まとめ

この章では、モデル解釈性の基礎について学びました：

  * ✅ **重要性** : 信頼性、規制対応、デバッグ、バイアス検出に不可欠
  * ✅ **分類** : グローバル/ローカル、モデル固有/非依存、事前/事後解釈性
  * ✅ **解釈可能モデル** : 線形回帰、決定木、ルールベース、GAM
  * ✅ **解釈手法** : Feature Importance, PDP, SHAP, LIME, Saliency Maps
  * ✅ **評価基準** : Fidelity, Consistency, Stability, Comprehensibility

次章では、Feature ImportanceとPermutation Importanceについて詳しく学びます。

* * *

## 参考文献

  * Molnar, C. (2022). _Interpretable Machine Learning: A Guide for Making Black Box Models Explainable_. <https://christophm.github.io/interpretable-ml-book/>
  * Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." _NIPS 2017_.
  * Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." _KDD 2016_.
  * Rudin, C. (2019). "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead." _Nature Machine Intelligence_.
  * European Union. (2016). _General Data Protection Regulation (GDPR)_. Article 22.
  * Doshi-Velez, F., & Kim, B. (2017). "Towards A Rigorous Science of Interpretable Machine Learning." _arXiv:1702.08608_.
