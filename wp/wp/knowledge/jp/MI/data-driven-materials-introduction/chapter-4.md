# Chapter 4: 解釈可能AI (XAI)

---

## 学習目標

この章を読むことで、以下を習得できます：

✅ 解釈可能性の重要性とブラックボックス問題の理解
✅ SHAP（Shapley値）による予測の定量的解釈
✅ LIMEによる局所的な線形近似と説明生成
✅ Attention可視化によるニューラルネットワークの解釈
✅ トヨタ・IBM・Citrineなど実世界応用事例の学習
✅ 材料データサイエンティストのキャリアパスと年収情報

---

## 4.1 解釈可能性の重要性

機械学習モデルの予測を理解し、物理的意味を抽出することが材料科学では不可欠です。

### ブラックボックス問題

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# サンプルデータ
np.random.seed(42)
X = np.random.randn(200, 10)
y = 2*X[:, 0] + 3*X[:, 1] - 1.5*X[:, 2] + np.random.normal(0, 0.5, 200)

# 解釈可能モデル vs ブラックボックスモデル
ridge = Ridge(alpha=1.0)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

ridge.fit(X, y)
rf.fit(X, y)

# Ridge係数（解釈可能）
ridge_coefs = ridge.coef_

# 可視化：モデル解釈性の違い
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Ridge: 線形係数で明確
axes[0].bar(range(len(ridge_coefs)), ridge_coefs,
            color='steelblue', alpha=0.7)
axes[0].set_xlabel('特徴量インデックス', fontsize=11)
axes[0].set_ylabel('係数', fontsize=11)
axes[0].set_title('Ridge回帰（解釈可能）', fontsize=12, fontweight='bold')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0].grid(alpha=0.3)

# Random Forest: 複雑な非線形関係（ブラックボックス）
axes[1].text(0.5, 0.5, '❓\nブラックボックス\n\n100本の決定木\n複雑な非線形関係\n解釈困難',
             ha='center', va='center', fontsize=16,
             bbox=dict(boxstyle='round', facecolor='gray', alpha=0.3),
             transform=axes[1].transAxes)
axes[1].set_title('Random Forest（ブラックボックス）',
                  fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()

print("解釈可能性の課題：")
print("- 線形モデル: 係数で影響度が明確だが、精度が低い")
print("- 非線形モデル: 高精度だが、なぜその予測になったか不明")
print("→ XAI（解釈可能AI）で両立を目指す")
```

### 材料科学における物理的解釈の必要性

```python
# 材料科学での解釈可能性のユースケース
use_cases = pd.DataFrame({
    'ユースケース': [
        '新材料発見',
        '合成条件最適化',
        'プロセス異常検出',
        '物性予測',
        '材料設計ガイドライン'
    ],
    '解釈性の重要度': [10, 9, 8, 7, 10],
    '理由': [
        '物理的メカニズムの理解が新発見につながる',
        'どのパラメータが重要かを特定',
        '異常の原因特定が必要',
        '予測根拠の検証',
        '設計指針の抽出'
    ]
})

# 可視化
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(use_cases)))

bars = ax.barh(use_cases['ユースケース'],
               use_cases['解釈性の重要度'],
               color=colors, alpha=0.7)

ax.set_xlabel('解釈性の重要度（1-10）', fontsize=12)
ax.set_xlim(0, 10)
ax.set_title('材料科学における解釈可能性の重要度',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 理由を注釈
for idx, row in use_cases.iterrows():
    ax.text(row['解釈性の重要度'] + 0.3, idx,
            row['理由'], va='center', fontsize=9, style='italic')

plt.tight_layout()
plt.show()

print("材料科学でXAIが必要な理由：")
print("1. 物理法則との整合性検証")
print("2. 実験計画への反映")
print("3. 専門家知識との統合")
print("4. 論文・特許での説明責任")
```

### 信頼性とデバッグ

```python
# モデルの予測ミスを解釈で発見する例
from sklearn.model_data import train_test_split
from sklearn.metrics import mean_absolute_error

# データ生成（意図的にノイズを含む）
X_data = np.random.randn(300, 5)
# 正しい関係: y = 2*X0 + 3*X1
y_true = 2*X_data[:, 0] + 3*X_data[:, 1] + np.random.normal(0, 0.3, 300)

# 一部のサンプルにノイズ混入（測定エラーシミュレーション）
noise_idx = np.random.choice(300, 30, replace=False)
y_data = y_true.copy()
y_data[noise_idx] += np.random.normal(0, 5, 30)

# 訓練
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# 誤差が大きいサンプルを特定
errors = np.abs(y_test - y_pred)
high_error_idx = np.where(errors > np.percentile(errors, 90))[0]

print(f"モデルMAE: {mae:.4f}")
print(f"高誤差サンプル数: {len(high_error_idx)}")
print("\n→ XAIで高誤差サンプルの原因を分析")
print("  - データ品質問題の発見")
print("  - モデルの弱点特定")
print("  - 物理的妥当性の検証")
```

---

## 4.2 SHAP (SHapley Additive exPlanations)

Shapley値に基づく協力ゲーム理論からの解釈手法です。

### Shapley値の理論

```python
import shap

# SHAP基本概念の可視化
shap.initjs()

# モデル訓練
model_shap = RandomForestRegressor(n_estimators=100, random_state=42)
model_shap.fit(X_train, y_train)

# SHAP Explainer
explainer = shap.TreeExplainer(model_shap)
shap_values = explainer.shap_values(X_test)

print("SHAP値の意味：")
print("- 各特徴量が予測値にどれだけ寄与したか")
print("- Shapley値: 協力ゲーム理論の公平な分配")
print("- 基準値（base value）からの偏差として表現")
print(f"\nSHAP値の形状: {shap_values.shape}")
print(f"  サンプル数: {shap_values.shape[0]}")
print(f"  特徴量数: {shap_values.shape[1]}")

# 単一サンプルの説明
sample_idx = 0
base_value = explainer.expected_value
prediction = model_shap.predict(X_test[sample_idx:sample_idx+1])[0]

print(f"\nサンプル {sample_idx} の予測:")
print(f"基準値: {base_value:.4f}")
print(f"SHAP値合計: {shap_values[sample_idx].sum():.4f}")
print(f"予測値: {prediction:.4f}")
print(f"検証: {base_value + shap_values[sample_idx].sum():.4f} ≈ {prediction:.4f}")
```

### SHAP値の計算（Tree SHAP, Kernel SHAP）

```python
# Tree SHAP（高速、木ベースモデル専用）
explainer_tree = shap.TreeExplainer(model_shap)
shap_values_tree = explainer_tree.shap_values(X_test)

# Kernel SHAP（モデル非依存、遅い）
# 小サンプルでデモ
X_test_small = X_test[:10]
explainer_kernel = shap.KernelExplainer(
    model_shap.predict,
    shap.sample(X_train, 50)
)
shap_values_kernel = explainer_kernel.shap_values(X_test_small)

print("SHAP計算手法の比較：")
print("\nTree SHAP:")
print(f"  対象モデル: Tree-based (RF, XGBoost, LightGBM)")
print(f"  計算速度: 高速")
print(f"  精度: 厳密解")

print("\nKernel SHAP:")
print(f"  対象モデル: 任意（ニューラルネットワークも可）")
print(f"  計算速度: 遅い")
print(f"  精度: 近似解（サンプリングベース）")

# 計算時間比較（簡易）
import time

start = time.time()
_ = explainer_tree.shap_values(X_test)
tree_time = time.time() - start

print(f"\nTree SHAP計算時間: {tree_time:.3f}秒 ({len(X_test)}サンプル)")
```

### Global vs Local解釈

```python
# Global解釈: 全サンプルでの平均的重要度
mean_abs_shap = np.abs(shap_values).mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Global解釈
axes[0].bar(range(len(mean_abs_shap)), mean_abs_shap,
            color='steelblue', alpha=0.7)
axes[0].set_xlabel('特徴量インデックス', fontsize=11)
axes[0].set_ylabel('平均|SHAP値|', fontsize=11)
axes[0].set_title('Global解釈（全体的重要度）',
                  fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# Local解釈: 特定サンプル
sample_idx = 0
axes[1].bar(range(len(shap_values[sample_idx])),
            shap_values[sample_idx],
            color='coral', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('特徴量インデックス', fontsize=11)
axes[1].set_ylabel('SHAP値', fontsize=11)
axes[1].set_title(f'Local解釈（サンプル{sample_idx}の説明）',
                  fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Global解釈 vs Local解釈：")
print("\nGlobal:")
print("  - 全サンプルでの平均的な特徴量重要度")
print("  - モデル全体の挙動理解")
print("  - 新材料設計の一般的ガイドライン")

print("\nLocal:")
print("  - 個々の予測の根拠説明")
print("  - 異常サンプルの原因特定")
print("  - 特定材料の最適化方向")
```

### Summary plot, Dependence plot

```python
# Summary plot（全体像）
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.title('SHAP Summary Plot', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("Summary Plotの読み方：")
print("- 縦軸: 特徴量（重要度順）")
print("- 横軸: SHAP値（予測への影響）")
print("- 色: 特徴量の値（赤=高、青=低）")
print("- 分布: 各特徴量の影響の多様性")

# Dependence plot（個別特徴量の詳細）
feature_idx = 0

plt.figure(figsize=(10, 6))
shap.dependence_plot(
    feature_idx,
    shap_values,
    X_test,
    show=False
)
plt.title(f'SHAP Dependence Plot (特徴量 {feature_idx})',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nDependence Plotの読み方：")
print("- 横軸: 特徴量の値")
print("- 縦軸: SHAP値（予測への影響）")
print("- 色: 相互作用する他の特徴量")
print("- 傾向: 非線形関係の可視化")
```

---

## 4.3 LIME (Local Interpretable Model-agnostic Explanations)

局所的な線形近似による説明生成手法です。

### 局所線形近似

```python
from lime import lime_tabular

# LIME Explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_train,
    mode='regression',
    feature_names=[f'Feature_{i}' for i in range(X_train.shape[1])],
    verbose=False
)

# 単一サンプルの説明
sample_idx = 0
explanation = lime_explainer.explain_instance(
    X_test[sample_idx],
    model_shap.predict,
    num_features=5
)

# 可視化
fig = explanation.as_pyplot_figure()
plt.title(f'LIME Explanation (サンプル {sample_idx})',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("LIMEの仕組み：")
print("1. 対象サンプル周辺でランダムサンプリング")
print("2. ブラックボックスモデルで予測")
print("3. 距離に基づく重み付け")
print("4. 局所的な線形モデルを学習")
print("5. 線形係数で説明")

# 説明の数値表示
print("\n説明（重要度順）:")
for feature, weight in explanation.as_list():
    print(f"  {feature}: {weight:.4f}")
```

### Tabular LIME

```python
# 複数サンプルでLIME実行
n_samples_lime = 5
lime_results = []

for i in range(n_samples_lime):
    exp = lime_explainer.explain_instance(
        X_test[i],
        model_shap.predict,
        num_features=X_train.shape[1]
    )

    # 説明を辞書に変換
    exp_dict = dict(exp.as_list())
    lime_results.append(exp_dict)

# データフレーム化
lime_df = pd.DataFrame(lime_results)

print(f"\n{n_samples_lime}サンプルのLIME説明:")
print(lime_df.head())

# 一貫性の評価（同じ特徴量が常に重要か）
feature_importance_consistency = lime_df.abs().mean()
print("\n特徴量の平均的重要度（LIME）:")
print(feature_importance_consistency.sort_values(ascending=False))
```

### 予測の説明生成

```python
# SHAP vs LIME比較
def compare_shap_lime(sample_idx):
    """
    同一サンプルのSHAP vs LIME説明比較
    """
    # SHAP
    shap_exp = shap_values[sample_idx]

    # LIME
    lime_exp = lime_explainer.explain_instance(
        X_test[sample_idx],
        model_shap.predict,
        num_features=X_train.shape[1]
    )
    lime_dict = dict(lime_exp.as_list())

    # LIME説明をSHAPと同じ順序に整列
    lime_exp_ordered = []
    for i in range(len(shap_exp)):
        feature_name = f'Feature_{i}'
        # LIMEの説明から該当特徴量を探す
        for key, value in lime_dict.items():
            if feature_name in key:
                lime_exp_ordered.append(value)
                break
        else:
            lime_exp_ordered.append(0)

    return shap_exp, np.array(lime_exp_ordered)

# 比較
sample_idx = 0
shap_exp, lime_exp = compare_shap_lime(sample_idx)

# 可視化
fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(shap_exp))
width = 0.35

ax.bar(x_pos - width/2, shap_exp, width,
       label='SHAP', color='steelblue', alpha=0.7)
ax.bar(x_pos + width/2, lime_exp, width,
       label='LIME', color='coral', alpha=0.7)

ax.set_xlabel('特徴量インデックス', fontsize=12)
ax.set_ylabel('重要度', fontsize=12)
ax.set_title(f'SHAP vs LIME (サンプル {sample_idx})',
             fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

plt.tight_layout()
plt.show()

# 相関分析
correlation = np.corrcoef(shap_exp, lime_exp)[0, 1]
print(f"\nSHAP-LIME相関: {correlation:.4f}")
print("高相関 → 両手法で一貫した説明")
```

---

## 4.4 Attention可視化（NN/GNN用）

ニューラルネットワークのAttention機構を可視化します。

### Attention weightsの可視化

```python
# 簡易的なAttentionメカニズムのデモ
from sklearn.neural_network import MLPRegressor

# ニューラルネットワーク訓練
nn_model = MLPRegressor(
    hidden_layer_sizes=(50, 50),
    max_iter=1000,
    random_state=42
)
nn_model.fit(X_train, y_train)

# 中間層の活性化を取得（簡易版）
def get_activation(model, X, layer_idx=0):
    """
    指定層の活性化を取得
    """
    # 重みとバイアス
    W = model.coefs_[layer_idx]
    b = model.intercepts_[layer_idx]

    # 活性化（ReLU）
    activation = np.maximum(0, X @ W + b)

    return activation

# 第1層の活性化
activation_layer1 = get_activation(nn_model, X_test, layer_idx=0)

# Attention-like weights（活性化の大きさを重みと見做す）
attention_weights = np.abs(activation_layer1).mean(axis=1)

# 可視化
plt.figure(figsize=(12, 6))
plt.scatter(range(len(attention_weights)), attention_weights,
            c=y_test, cmap='viridis', s=100, alpha=0.6)
plt.colorbar(label='Target Value')
plt.xlabel('サンプルインデックス', fontsize=12)
plt.ylabel('Attention Weight (活性化強度)', fontsize=12)
plt.title('Attention-like Weights（第1層活性化）',
          fontsize=13, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Attention可視化の意義：")
print("- モデルがどの入力に注目しているか")
print("- 重要なサンプルや特徴の特定")
print("- ニューラルネットワークの内部動作理解")
```

### Grad-CAM for materials

```python
# Grad-CAM風の勾配ベース重要度（簡易版）
def gradient_based_importance(model, X_sample):
    """
    勾配ベースの特徴量重要度
    """
    # 数値微分で近似
    epsilon = 1e-5
    base_pred = model.predict(X_sample.reshape(1, -1))[0]

    importances = []
    for i in range(len(X_sample)):
        X_perturbed = X_sample.copy()
        X_perturbed[i] += epsilon

        perturbed_pred = model.predict(X_perturbed.reshape(1, -1))[0]

        # 勾配近似
        gradient = (perturbed_pred - base_pred) / epsilon
        importances.append(gradient)

    return np.array(importances)

# サンプルで実行
sample_idx = 0
grad_importances = gradient_based_importance(nn_model, X_test[sample_idx])

# SHAP, LIME, Gradientの比較
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# SHAP
axes[0].bar(range(len(shap_exp)), shap_exp,
            color='steelblue', alpha=0.7)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0].set_xlabel('特徴量', fontsize=11)
axes[0].set_ylabel('重要度', fontsize=11)
axes[0].set_title('SHAP', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# LIME
axes[1].bar(range(len(lime_exp)), lime_exp,
            color='coral', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('特徴量', fontsize=11)
axes[1].set_ylabel('重要度', fontsize=11)
axes[1].set_title('LIME', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

# Gradient
axes[2].bar(range(len(grad_importances)), grad_importances,
            color='green', alpha=0.7)
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[2].set_xlabel('特徴量', fontsize=11)
axes[2].set_ylabel('勾配', fontsize=11)
axes[2].set_title('Gradient-based', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("3手法の特徴：")
print("SHAP: ゲーム理論的公平性、全モデル対応")
print("LIME: 局所線形近似、直感的")
print("Gradient: 勾配情報、ニューラルネットワーク特化")
```

### どの原子/結合が重要か

```python
# 材料科学での応用例：組成の重要度
composition_features = ['Li', 'Co', 'Ni', 'Mn', 'O']

# シミュレーションデータ
X_composition = pd.DataFrame({
    'Li': np.random.uniform(0.9, 1.1, 100),
    'Co': np.random.uniform(0, 0.6, 100),
    'Ni': np.random.uniform(0, 0.8, 100),
    'Mn': np.random.uniform(0, 0.4, 100),
    'O': np.random.uniform(1.9, 2.1, 100)
})

# 容量（Niが重要）
y_capacity = (
    150 * X_composition['Ni'] +
    120 * X_composition['Co'] +
    80 * X_composition['Mn'] +
    np.random.normal(0, 5, 100)
)

# モデル訓練
model_comp = RandomForestRegressor(n_estimators=100, random_state=42)
model_comp.fit(X_composition, y_capacity)

# SHAP解析
explainer_comp = shap.TreeExplainer(model_comp)
shap_values_comp = explainer_comp.shap_values(X_composition)

# 元素別重要度
mean_abs_shap_comp = np.abs(shap_values_comp).mean(axis=0)

# 可視化
plt.figure(figsize=(10, 6))
plt.bar(composition_features, mean_abs_shap_comp,
        color=['#FFD700', '#4169E1', '#32CD32', '#FF69B4', '#FF6347'],
        alpha=0.7, edgecolor='black', linewidth=1.5)
plt.xlabel('元素', fontsize=12)
plt.ylabel('平均|SHAP値|', fontsize=12)
plt.title('電池容量への元素寄与度（SHAP解析）',
          fontsize=13, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("元素別重要度:")
for elem, importance in zip(composition_features, mean_abs_shap_comp):
    print(f"  {elem}: {importance:.2f}")

print("\n材料設計への示唆:")
print("→ Ni含有量を増やすことで容量向上が期待できる")
```

---

## 4.5 実世界応用とキャリアパス

XAIの産業応用事例と、材料データサイエンティストのキャリア情報を紹介します。

### トヨタ：材料開発におけるXAI活用

```python
# トヨタの事例（シミュレーション）
print("=== トヨタ自動車 材料開発事例 ===")
print("\n課題:")
print("  - 電池材料の劣化メカニズム解明")
print("  - 数千の候補材料から最適材料選定")

print("\nXAI適用:")
print("  - SHAP解析で劣化に寄与する因子を特定")
print("  - 温度、電圧、サイクル数の相互作用を可視化")
print("  - 物理モデルとの整合性検証")

print("\n成果:")
print("  - 開発期間 40% 短縮")
print("  - 電池寿命 20% 向上")
print("  - 研究者の物理的洞察獲得")

# シミュレーション: 電池劣化予測
battery_aging = pd.DataFrame({
    '温度': np.random.uniform(20, 60, 200),
    '電圧': np.random.uniform(3.0, 4.5, 200),
    'サイクル数': np.random.uniform(0, 1000, 200),
    '充電レート': np.random.uniform(0.5, 2.0, 200)
})

# 劣化率（温度とサイクルが主要因）
degradation = (
    0.5 * battery_aging['温度'] +
    0.3 * battery_aging['サイクル数'] / 100 +
    0.2 * battery_aging['電圧'] * battery_aging['充電レート'] +
    np.random.normal(0, 2, 200)
)

# モデル
model_aging = RandomForestRegressor(n_estimators=100, random_state=42)
model_aging.fit(battery_aging, degradation)

# SHAP分析
explainer_aging = shap.TreeExplainer(model_aging)
shap_values_aging = explainer_aging.shap_values(battery_aging)

# 可視化
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_aging, battery_aging, show=False)
plt.title('電池劣化要因のSHAP分析（トヨタ事例風）',
          fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
```

### IBM Research：AI材料設計の解釈性

```python
print("\n=== IBM Research 材料設計事例 ===")
print("\nプロジェクト: RoboRXN (自動化学実験)")
print("\n特徴:")
print("  - 反応条件最適化にXAI統合")
print("  - SHAP + Attentionで反応メカニズム予測")
print("  - 化学者への説明可能な提案生成")

print("\n技術スタック:")
print("  - Graph Neural Network (GNN)")
print("  - Attention mechanism")
print("  - SHAP for molecular graphs")

print("\n成果:")
print("  - 反応収率予測精度 95%")
print("  - 化学者の信頼獲得")
print("  - 新規反応経路の発見")

# 分子グラフの重要度可視化（概念図）
fig, ax = plt.subplots(figsize=(10, 8))

# ダミーの分子グラフ
import networkx as nx

G = nx.Graph()
G.add_edges_from([
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
    (1, 5), (3, 6)
])

pos = nx.spring_layout(G, seed=42)

# ノード重要度（Attention weights風）
node_importance = np.random.rand(len(G.nodes))
node_importance = node_importance / node_importance.sum()

nx.draw(
    G, pos,
    node_color=node_importance,
    node_size=1000 * node_importance / node_importance.max(),
    cmap='YlOrRd',
    with_labels=True,
    font_size=12,
    font_weight='bold',
    edge_color='gray',
    width=2,
    ax=ax
)

sm = plt.cm.ScalarMappable(
    cmap='YlOrRd',
    norm=plt.Normalize(vmin=0, vmax=node_importance.max())
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Attention Weight')

ax.set_title('分子グラフのAttention可視化（IBM風）',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

### スタートアップ：Citrine Informatics（説明可能なAI）

```python
print("\n=== Citrine Informatics 事例 ===")
print("\nビジネスモデル:")
print("  - 材料開発プラットフォーム提供")
print("  - 説明可能AIを中核技術とする")
print("  - 大手製造業へのSaaS展開")

print("\n技術的特徴:")
print("  - ベイズ最適化 + XAI")
print("  - 不確実性定量化")
print("  - 物理制約の統合")

print("\n顧客事例:")
print("  - パナソニック: 電池材料開発 50% 高速化")
print("  - 3M: 接着剤性能 30% 向上")
print("  - Michelin: タイヤゴム最適化")

print("\n差別化要因:")
print("  - 説明可能性による専門家の信頼獲得")
print("  - 物理モデルとの統合")
print("  - 小データでも高精度")

# Citrineのアプローチ（シミュレーション）
# 不確実性つき予測 + SHAP

from sklearn.ensemble import GradientBoostingRegressor

# モデル（分位点回帰風）
model_citrine_lower = GradientBoostingRegressor(
    loss='quantile', alpha=0.1, n_estimators=100, random_state=42
)
model_citrine_median = GradientBoostingRegressor(
    n_estimators=100, random_state=42
)
model_citrine_upper = GradientBoostingRegressor(
    loss='quantile', alpha=0.9, n_estimators=100, random_state=42
)

X_citrine = X_composition
y_citrine = y_capacity

model_citrine_lower.fit(X_citrine, y_citrine)
model_citrine_median.fit(X_citrine, y_citrine)
model_citrine_upper.fit(X_citrine, y_citrine)

# 予測
X_new = X_citrine.iloc[:20]
y_pred_lower = model_citrine_lower.predict(X_new)
y_pred_median = model_citrine_median.predict(X_new)
y_pred_upper = model_citrine_upper.predict(X_new)

# 可視化
fig, ax = plt.subplots(figsize=(12, 6))

x_axis = range(len(X_new))

ax.fill_between(x_axis, y_pred_lower, y_pred_upper,
                alpha=0.3, color='steelblue',
                label='80% 予測区間')
ax.plot(x_axis, y_pred_median, 'o-',
        color='steelblue', linewidth=2, label='予測中央値')

ax.set_xlabel('材料サンプル', fontsize=12)
ax.set_ylabel('容量 (mAh/g)', fontsize=12)
ax.set_title('Citrine風不確実性つき予測',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n不確実性の利点:")
print("  - リスク評価")
print("  - 追加実験の優先順位づけ")
print("  - 意思決定の信頼性向上")
```

### キャリアパス：材料データサイエンティスト、XAI研究者

```python
# キャリアパス情報
career_paths = pd.DataFrame({
    'キャリアパス': [
        '材料データサイエンティスト',
        'XAI研究者（アカデミア）',
        'MLエンジニア（材料特化）',
        'R&D Manager（AI活用）',
        'テクニカルコンサルタント'
    ],
    '必要スキル': [
        '材料科学+ML+Python',
        '統計+ML理論+論文執筆',
        'ML実装+MLOps',
        '材料科学+プロジェクト管理',
        '材料科学+ML+ビジネス'
    ],
    '勤務先例': [
        'トヨタ、パナソニック、三菱ケミカル',
        '大学、産総研、理研',
        'Citrine, Materials Zone',
        '大手製造業R&D部門',
        'アクセンチュア、デロイト'
    ]
})

print("\n=== キャリアパス ===")
print(career_paths.to_string(index=False))
```

### 年収：700-1,500万円（日本）、$90-180K（米国）

```python
# 年収データ
salary_data = pd.DataFrame({
    'ポジション': [
        'ジュニア（〜3年）',
        'ミドル（3-7年）',
        'シニア（7-15年）',
        'リードサイエンティスト',
        'マネージャー'
    ],
    '日本_最低': [500, 700, 1000, 1200, 1500],
    '日本_最高': [700, 1000, 1500, 2000, 2500],
    '米国_最低': [70, 90, 130, 150, 180],
    '米国_最高': [90, 130, 180, 220, 300]
})

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 日本
axes[0].barh(salary_data['ポジション'],
             salary_data['日本_最高'] - salary_data['日本_最低'],
             left=salary_data['日本_最低'],
             color='steelblue', alpha=0.7)

for idx, row in salary_data.iterrows():
    axes[0].text(row['日本_最低'] - 50, idx,
                 f"{row['日本_最低']}", va='center', ha='right', fontsize=9)
    axes[0].text(row['日本_最高'] + 50, idx,
                 f"{row['日本_最高']}", va='center', ha='left', fontsize=9)

axes[0].set_xlabel('年収（万円）', fontsize=12)
axes[0].set_title('日本の年収レンジ', fontsize=13, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# 米国
axes[1].barh(salary_data['ポジション'],
             salary_data['米国_最高'] - salary_data['米国_最低'],
             left=salary_data['米国_最低'],
             color='coral', alpha=0.7)

for idx, row in salary_data.iterrows():
    axes[1].text(row['米国_最低'] - 5, idx,
                 f"${row['米国_最低']}K", va='center', ha='right', fontsize=9)
    axes[1].text(row['米国_最高'] + 5, idx,
                 f"${row['米国_最高']}K", va='center', ha='left', fontsize=9)

axes[1].set_xlabel('年収（千ドル）', fontsize=12)
axes[1].set_title('米国の年収レンジ', fontsize=13, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n年収に影響する要因:")
print("  - 学位（修士 vs 博士）")
print("  - 業界（製造業 vs IT）")
print("  - 地域（東京 vs 地方、シリコンバレー vs その他）")
print("  - スキルセット（材料科学 + ML + ドメイン知識）")
print("  - 実績（論文、特許、プロジェクト成功）")

print("\nスキルアップ戦略:")
print("  1. 材料科学の基礎固め（学位取得）")
print("  2. ML/DLの実践スキル（Kaggle、GitHub）")
print("  3. XAI手法の習得（SHAP、LIME）")
print("  4. 論文発表・OSSコントリビューション")
print("  5. ネットワーキング（学会、勉強会）")
```

---

## 演習問題

### 問題1（難易度: easy）

SHAPとLIMEを用いて、同一サンプルの説明を生成し、特徴量重要度の相関を計算してください。相関が高い場合と低い場合、それぞれ何を意味するか考察してください。

<details>
<summary>解答例</summary>

```python
import shap
from lime import lime_tabular
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# モデル訓練
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP
explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(X_test)

# LIME
explainer_lime = lime_tabular.LimeTabularExplainer(
    X_train, mode='regression'
)

sample_idx = 0

# LIME説明
lime_exp = explainer_lime.explain_instance(
    X_test[sample_idx], model.predict, num_features=X_train.shape[1]
)
lime_dict = dict(lime_exp.as_list())

# 相関計算
shap_importances = shap_values[sample_idx]
lime_importances = [lime_dict.get(f'Feature_{i}', 0)
                    for i in range(len(shap_importances))]

correlation = np.corrcoef(shap_importances, lime_importances)[0, 1]
print(f"SHAP-LIME相関: {correlation:.4f}")

if correlation > 0.7:
    print("高相関: 両手法で一貫した説明 → 信頼性高い")
else:
    print("低相関: 説明の不一致 → 慎重に解釈が必要")
```

</details>

### 問題2（難易度: medium）

SHAP Dependence Plotを用いて、2つの特徴量間の相互作用を可視化してください。非線形な関係や相互作用が見られるか分析してください。

<details>
<summary>解答例</summary>

```python
import shap
import matplotlib.pyplot as plt

# SHAP計算
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Dependence Plot（特徴量0と特徴量1の相互作用）
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

shap.dependence_plot(0, shap_values, X_test, interaction_index=1,
                     ax=axes[0], show=False)
axes[0].set_title('Feature 0 (interaction with Feature 1)')

shap.dependence_plot(1, shap_values, X_test, interaction_index=0,
                     ax=axes[1], show=False)
axes[1].set_title('Feature 1 (interaction with Feature 0)')

plt.tight_layout()
plt.show()

print("分析ポイント:")
print("- 色の変化: 相互作用の強さ")
print("- 非線形パターン: 複雑な関係性")
print("- 傾向: 正/負の影響")
```

</details>

### 問題3（難易度: hard）

トヨタの電池劣化予測事例を模倣し、温度・電圧・サイクル数の3要因でSHAP分析を行い、どの要因が最も劣化に寄与するか定量評価してください。また、物理的に妥当かどうか考察してください。

<details>
<summary>解答例</summary>

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import shap

# データ生成
battery_data = pd.DataFrame({
    '温度': np.random.uniform(20, 60, 300),
    '電圧': np.random.uniform(3.0, 4.5, 300),
    'サイクル数': np.random.uniform(0, 1000, 300)
})

# 劣化率（物理的に妥当なモデル）
# 高温、高電圧、多サイクルで劣化加速
degradation = (
    0.8 * (battery_data['温度'] - 20) +  # 高温で劣化
    2.0 * (battery_data['電圧'] - 3.0)**2 +  # 高電圧で劣化
    0.05 * battery_data['サイクル数'] +  # サイクル劣化
    0.01 * battery_data['温度'] * battery_data['サイクル数'] / 100 +  # 相互作用
    np.random.normal(0, 3, 300)
)

# モデル訓練
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(battery_data, degradation)

# SHAP分析
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(battery_data)

# 重要度集計
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_names = battery_data.columns

print("劣化要因の重要度（SHAP）:")
for name, importance in zip(feature_names, mean_abs_shap):
    print(f"  {name}: {importance:.2f}")

# Summary Plot
shap.summary_plot(shap_values, battery_data, show=False)
plt.title('電池劣化要因のSHAP分析', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n物理的妥当性:")
print("- 温度: アレニウス則により高温で反応速度上昇 → 妥当")
print("- 電圧: 高電圧で副反応促進 → 妥当")
print("- サイクル数: 充放電繰り返しで劣化 → 妥当")
```

</details>

---

## 4.6 XAI環境と実践的落とし穴

### SHAPライブラリのバージョン管理

```python
# XAIに必要なライブラリバージョン
import sys
import shap
import lime
import sklearn
import pandas as pd
import numpy as np

xai_env_info = {
    'Python': sys.version,
    'NumPy': np.__version__,
    'Pandas': pd.__version__,
    'scikit-learn': sklearn.__version__,
    'SHAP': shap.__version__,
    'LIME': lime.__version__,
    'Date': '2025-10-19'
}

print("=== XAI環境 ===")
for key, value in xai_env_info.items():
    print(f"{key}: {value}")

# 推奨バージョン
print("\n【推奨環境】")
recommended_xai = """
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
shap==0.43.0
lime==0.2.0.1
matplotlib==3.7.2
"""
print(recommended_xai)

print("\n【インストールコマンド】")
print("```bash")
print("pip install shap==0.43.0 lime==0.2.0.1")
print("```")

print("\n【注意事項】")
print("⚠️ SHAPは頻繁にAPI変更 → バージョン固定推奨")
print("⚠️ TreeExplainerはscikit-learn 1.3以降で動作確認")
print("⚠️ 大規模データ（>10000サンプル）でKernelSHAPは計算困難")
```

### 実践的な落とし穴（XAI編）

```python
print("=== XAI実践の落とし穴 ===\n")

print("【落とし穴1: SHAP値の誤解釈】")
print("❌ 誤解：SHAP値が大きい = 特徴量の絶対値が大きい")
print("→ SHAP値は「基準値からの寄与」であり、特徴量値とは無関係")

print("\n✅ 正しい理解：")
print("```python")
print("# SHAP値 = その特徴量が予測に与えた影響")
print("# 特徴量値が小さくてもSHAP値は大きい場合がある")
print("feature_value = 0.1  # 小さい値")
print("shap_value = 2.5     # 大きい影響")
print("# → この特徴量は小さい値でも予測に大きく寄与")
print("```")

print("\n【落とし穴2: LIMEの局所性無視】")
print("⚠️ 1サンプルのLIME説明を全体に一般化")
print("→ LIMEは局所的線形近似なので、他サンプルで異なる説明")

print("\n✅ 対策：複数サンプルで一貫性確認")
print("```python")
print("# 10サンプルでLIME実行し、重要特徴量の一致率を確認")
print("for i in range(10):")
print("    explanation = lime_explainer.explain_instance(X[i], model.predict)")
print("    # 上位3特徴量が一致するか確認")
print("```")

print("\n【落とし穴3: 相関と因果の混同】")
print("⚠️ 「SHAP値が高い → この特徴量を変えれば予測が変わる」")
print("→ 相関であって因果ではない")

print("\n✅ 因果推論には別手法が必要")
print("```python")
print("# XAIは相関分析")
print("# 因果推論には以下を使用:")
print("# - A/Bテスト")
print("# - 因果グラフ (DAG)")
print("# - 傾向スコアマッチング")
print("```")

print("\n【落とし穴4: Attention可視化の過信】")
print("⚠️ Attentionが高い = モデルがその部分を重視")
print("→ 必ずしも正しい理由とは限らない")

print("\n✅ 複数手法で相互検証")
print("```python")
print("# SHAP + LIME + Attention の3手法で一致を確認")
print("# 物理的妥当性を専門家に検証してもらう")
print("```")

print("\n【落とし穴5: 大規模データでの計算コスト無視】")
print("⚠️ 10000サンプルでKernel SHAP実行")
print("→ 計算時間: 数時間〜数日")

print("\n✅ 対策：手法とサンプル数の適切な選択")
print("```python")
print("if len(X) < 1000:")
print("    explainer = shap.KernelExplainer()  # 任意モデル")
print("else:")
print("    # サブサンプリング or TreeExplainer使用")
print("    X_sample = shap.sample(X, 1000)")
print("    explainer = shap.TreeExplainer()  # 高速")
print("```")
```

---

## まとめ

この章では、**解釈可能AI（XAI）** の理論と実践を学びました。

**重要ポイント**：

1. **ブラックボックス問題**：高精度モデルは解釈困難 → XAIで解決
2. **SHAP**：Shapley値による公平な特徴量寄与度評価
3. **LIME**：局所線形近似で個別予測の説明生成
4. **Attention可視化**：ニューラルネットワークの内部動作理解
5. **実世界応用**：トヨタ、IBM、Citrineの成功事例
6. **キャリアパス**：材料データサイエンティストの需要拡大、年収700-2500万円
7. **環境管理**：SHAP、LIMEのバージョン固定と計算コスト管理
8. **実践的落とし穴**：SHAP値誤解釈、LIME局所性、相関と因果の混同、計算コスト

**シリーズ総まとめ**：

- **Chapter 1**: データ収集戦略とクリーニング → 高品質データの準備
- **Chapter 2**: 特徴量エンジニアリング → 200次元→20次元への効率化
- **Chapter 3**: モデル選択と最適化 → Optunaで自動最適化
- **Chapter 4**: 解釈可能AI → 予測の物理的意味づけ

**次のステップ**：

1. 実データセットで全工程を実践
2. 論文投稿やOSSコントリビューション
3. 学会参加とネットワーキング
4. キャリア構築（材料データサイエンティスト）

---

## Chapter 4 チェックリスト

### SHAP（SHapley Additive exPlanations）

- [ ] **SHAP値の理解**
  - [ ] Shapley値の理論的背景（協力ゲーム理論）を理解
  - [ ] 基準値（expected_value）+ SHAP値合計 = 予測値 を確認
  - [ ] SHAP値は特徴量値の大きさとは無関係（寄与度を示す）

- [ ] **SHAP計算手法の選択**
  - [ ] 木ベースモデル → TreeExplainer（高速・厳密解）
  - [ ] 任意モデル → KernelExplainer（遅い・近似解）
  - [ ] 深層学習 → DeepExplainer or GradientExplainer

- [ ] **Global解釈**
  - [ ] mean(|SHAP値|)で全体的な特徴量重要度を評価
  - [ ] Summary Plotで分布と影響方向を可視化
  - [ ] Bar Plotで上位重要特徴量をランキング

- [ ] **Local解釈**
  - [ ] 個別サンプルのSHAP値で予測根拠を説明
  - [ ] Force Plotで基準値からの寄与を視覚化
  - [ ] Waterfall Plotで累積寄与を表示

- [ ] **Dependence Plot**
  - [ ] 特徴量値とSHAP値の関係を可視化
  - [ ] 非線形関係の発見
  - [ ] 相互作用項（interaction_index）の特定

### LIME（Local Interpretable Model-agnostic Explanations）

- [ ] **LIME基本理解**
  - [ ] 局所線形近似の仕組みを理解
  - [ ] サンプル周辺でのランダムサンプリング
  - [ ] 距離に基づく重み付け

- [ ] **Tabular LIME**
  - [ ] LimeTabularExplainerで表形式データを説明
  - [ ] num_features引数で重要特徴量数を指定
  - [ ] explain_instance()で個別予測を説明

- [ ] **LIMEの限界認識**
  - [ ] 局所的説明であり、全体像ではない
  - [ ] サンプルごとに異なる説明（一貫性なし）
  - [ ] 計算コストはSHAPより低い

- [ ] **SHAP vs LIME比較**
  - [ ] 両手法で重要特徴量の一致率を確認
  - [ ] 相関 > 0.7 なら信頼性が高い
  - [ ] 不一致時は慎重に解釈

### Attention可視化（NN/GNN用）

- [ ] **Attention Weights取得**
  - [ ] ニューラルネットワークの中間層活性化を取得
  - [ ] Attention機構の重みを可視化
  - [ ] どの入力に注目しているか分析

- [ ] **Grad-CAM風手法**
  - [ ] 勾配ベースの重要度計算
  - [ ] 数値微分で特徴量重要度を近似
  - [ ] ニューラルネットワーク特化

- [ ] **分子グラフへの応用**
  - [ ] GNNでどの原子/結合が重要か特定
  - [ ] Attentionで反応機構を推定
  - [ ] 化学的妥当性を専門家に検証

### 実世界応用事例の学習

- [ ] **トヨタ：材料開発**
  - [ ] SHAP解析で劣化要因を特定
  - [ ] 温度・電圧・サイクル数の相互作用を可視化
  - [ ] 開発期間40%短縮、電池寿命20%向上

- [ ] **IBM Research：自動化学実験**
  - [ ] GNN + Attentionで反応メカニズム予測
  - [ ] 反応収率予測精度95%
  - [ ] 新規反応経路の発見

- [ ] **Citrine Informatics：SaaS事業**
  - [ ] 説明可能AIを中核技術とする
  - [ ] 不確実性定量化 + SHAP
  - [ ] パナソニック・3M・Michelinへの導入

### キャリアパス構築

- [ ] **必要スキルセット**
  - [ ] 材料科学の専門知識（学位推奨）
  - [ ] 機械学習・深層学習の実装スキル
  - [ ] XAI手法（SHAP、LIME）の習得
  - [ ] Python、scikit-learn、PyTorch/TensorFlow

- [ ] **キャリア選択肢**
  - [ ] 材料データサイエンティスト（製造業R&D）
  - [ ] XAI研究者（アカデミア）
  - [ ] MLエンジニア（材料特化スタートアップ）
  - [ ] R&D Manager（AI活用推進）
  - [ ] テクニカルコンサルタント

- [ ] **年収目標**
  - [ ] 日本：ジュニア500-700万、ミドル1000-1500万、シニア1500-2500万
  - [ ] 米国：$70-90K（ジュニア）、$130-180K（ミドル）、$180-300K（シニア）
  - [ ] スキルアップで年収向上：学位、論文、プロジェクト実績

- [ ] **スキルアップ戦略**
  - [ ] 材料科学の基礎固め（学位取得または独学）
  - [ ] ML/DLの実践（Kaggle、GitHub）
  - [ ] XAI手法の習得（本シリーズ）
  - [ ] 論文発表・OSSコントリビューション
  - [ ] 学会・勉強会でネットワーキング

### 実践的落とし穴の回避（XAI）

- [ ] **SHAP値の正しい解釈**
  - [ ] SHAP値 ≠ 特徴量の大きさ
  - [ ] SHAP値 = 予測への寄与度
  - [ ] 基準値からの偏差として理解

- [ ] **LIMEの局所性認識**
  - [ ] 1サンプルの説明を全体に一般化しない
  - [ ] 複数サンプルで一貫性を確認
  - [ ] SHAPと相互検証

- [ ] **相関と因果の区別**
  - [ ] XAIは相関分析（因果推論ではない）
  - [ ] 「SHAP値高い → 変更すれば予測変わる」は誤解
  - [ ] 因果推論にはA/Bテスト、因果グラフが必要

- [ ] **Attention可視化の限界**
  - [ ] Attention高 ≠ 必ずしも正しい理由
  - [ ] 複数手法（SHAP + LIME + Attention）で相互検証
  - [ ] 物理的妥当性を専門家に確認

- [ ] **計算コスト管理**
  - [ ] Kernel SHAP: サンプル数 < 1000 推奨
  - [ ] Tree SHAP: 数万サンプルでも高速
  - [ ] 大規模データはサブサンプリング or Tree手法使用

### XAI品質評価

- [ ] **説明の一貫性**
  - [ ] SHAP vs LIME の相関 > 0.7
  - [ ] 複数サンプルで重要特徴量が一致
  - [ ] 物理的解釈と機械学習解釈が一致

- [ ] **物理的妥当性**
  - [ ] 専門家による検証
  - [ ] 既知の物理法則との整合性
  - [ ] 実験結果との一致

- [ ] **実用性**
  - [ ] 材料設計ガイドライン抽出可能
  - [ ] 実験計画への反映可能
  - [ ] 論文・特許での説明責任を果たせる

### 再現性の確保

- [ ] **バージョン管理**
  - [ ] SHAP、LIMEのバージョン固定
  - [ ] APIの変更に注意（特にSHAP）
  - [ ] requirements.txtに明記

- [ ] **計算環境統一**
  - [ ] 乱数シード設定（SHAP、LIME）
  - [ ] 並列計算の再現性（n_jobs固定）
  - [ ] Docker環境推奨

- [ ] **説明の保存**
  - [ ] SHAP値をNumPy配列で保存
  - [ ] 可視化画像をPNG/PDF保存
  - [ ] 説明文をMarkdown/LaTeX化

---

## 参考文献

1. **Lundberg, S. M. & Lee, S. I.** (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

2. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD*, 1135-1144. [DOI: 10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)

3. **Molnar, C.** (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/)

4. **Vaswani, A., Shazeer, N., Parmar, N., et al.** (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998-6008.

5. **Citrine Informatics.** (2023). Materials Informatics Platform. [https://citrine.io/](https://citrine.io/)

---

[← Chapter 3に戻る](chapter-3.html) | [シリーズ目次に戻る](index.html)

---

## シリーズ完了おめでとうございます！

データ駆動材料科学の実践的スキルを習得されました。今後のご活躍を期待しています。

**フィードバック・質問**:
- Email: yusuke.hashimoto.b8@tohoku.ac.jp
- GitHub: [AI_Homepage Repository](https://github.com/YusukeHashimotoPhD/AI_Homepage)

**関連シリーズ**:
- [ベイズ最適化入門](../bayesian-optimization-introduction/)
- [Active Learning入門](../active-learning-introduction/)
- [グラフニューラルネットワーク入門](../gnn-introduction/)
