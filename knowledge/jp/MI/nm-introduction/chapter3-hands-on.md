---
title: "Chapter 3: Python実践チュートリアル"
subtitle: "ナノ材料データ解析と機械学習"
chapter: 3
reading_time: "30-40分"
level: "中級"
keywords: ["Python", "機械学習", "ナノ粒子解析", "量子ドット", "ベイズ最適化", "SHAP"]
prev_chapter: "chapter2-fundamentals.html"
next_chapter: "chapter4-real-world.html"
last_updated: "2025-10-16"
---

# Chapter 3: Python実践チュートリアル

ナノ材料データ解析と機械学習

---

## 本章の学習目標

本章を学習することで、以下のスキルを習得できます：

✅ ナノ粒子データの生成・可視化・前処理の実践
✅ 5種類の回帰モデルによるナノ材料物性予測
✅ ベイズ最適化によるナノ材料の最適設計
✅ SHAP分析による機械学習モデルの解釈
✅ 多目的最適化によるトレードオフ分析
✅ TEM画像解析とサイズ分布のフィッティング
✅ 異常検知による品質管理への応用

---

## 3.1 環境構築

### 必要なライブラリ

本チュートリアルで使用する主要なPythonライブラリ：

```python
# データ処理・可視化
pandas, numpy, matplotlib, seaborn, scipy

# 機械学習
scikit-learn, lightgbm

# 最適化
scikit-optimize

# モデル解釈
shap

# 多目的最適化（オプション）
pymoo
```

### インストール方法

#### Option 1: Anaconda環境

```bash
# Anacondaで新しい環境を作成
conda create -n nanomaterials python=3.10 -y
conda activate nanomaterials

# 必要なライブラリをインストール
conda install pandas numpy matplotlib seaborn scipy scikit-learn -y
conda install -c conda-forge lightgbm scikit-optimize shap -y

# 多目的最適化用（オプション）
pip install pymoo
```

#### Option 2: venv + pip環境

```bash
# 仮想環境を作成
python -m venv nanomaterials_env

# 仮想環境を有効化
# macOS/Linux:
source nanomaterials_env/bin/activate
# Windows:
nanomaterials_env\Scripts\activate

# 必要なライブラリをインストール
pip install pandas numpy matplotlib seaborn scipy
pip install scikit-learn lightgbm scikit-optimize shap pymoo
```

#### Option 3: Google Colab

Google Colabを使用する場合、以下のコードをセルで実行：

```python
# 追加パッケージのインストール
!pip install lightgbm scikit-optimize shap pymoo

# インポートの確認
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("環境構築完了！")
```

---

## 3.2 ナノ粒子データの準備と可視化

### 【例1】合成データ生成：金ナノ粒子のサイズと光学特性

金ナノ粒子の局在表面プラズモン共鳴（LSPR）波長は、粒子サイズに依存します。この関係を模擬データで表現します。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 乱数シードの設定（再現性のため）
np.random.seed(42)

# サンプル数
n_samples = 200

# 金ナノ粒子のサイズ（nm）: 平均15 nm、標準偏差5 nm
size = np.random.normal(15, 5, n_samples)
size = np.clip(size, 5, 50)  # 5-50 nmの範囲に制限

# LSPR波長（nm）: Mie理論の簡易近似
# 基本波長520 nm + サイズ依存項 + ノイズ
lspr = 520 + 0.8 * (size - 15) + np.random.normal(0, 5, n_samples)

# 合成条件
temperature = np.random.uniform(20, 80, n_samples)  # 温度（℃）
pH = np.random.uniform(4, 10, n_samples)  # pH

# データフレームの作成
data = pd.DataFrame({
    'size_nm': size,
    'lspr_nm': lspr,
    'temperature_C': temperature,
    'pH': pH
})

print("=" * 60)
print("金ナノ粒子データの生成完了")
print("=" * 60)
print(data.head(10))
print("\n基本統計量:")
print(data.describe())
```

### 【例2】サイズ分布のヒストグラム

```python
# サイズ分布のヒストグラム
fig, ax = plt.subplots(figsize=(10, 6))

# ヒストグラムとKDE（カーネル密度推定）
ax.hist(data['size_nm'], bins=30, alpha=0.6, color='skyblue',
        edgecolor='black', density=True, label='Histogram')

# KDEプロット
from scipy.stats import gaussian_kde
kde = gaussian_kde(data['size_nm'])
x_range = np.linspace(data['size_nm'].min(), data['size_nm'].max(), 100)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

ax.set_xlabel('Particle Size (nm)', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Gold Nanoparticle Size Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"平均サイズ: {data['size_nm'].mean():.2f} nm")
print(f"標準偏差: {data['size_nm'].std():.2f} nm")
print(f"中央値: {data['size_nm'].median():.2f} nm")
```

### 【例3】散布図マトリックス

```python
# ペアプロット（散布図マトリックス）
sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6},
             height=2.5, corner=False)
plt.suptitle('Pairplot of Gold Nanoparticle Data', y=1.01, fontsize=14, fontweight='bold')
plt.show()

print("各変数間の関係を可視化しました")
```

### 【例4】相関行列のヒートマップ

```python
# 相関行列の計算
correlation_matrix = data.corr()

# ヒートマップ
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("相関係数:")
print(correlation_matrix)
print(f"\nLSPR波長とサイズの相関: {correlation_matrix.loc['lspr_nm', 'size_nm']:.3f}")
```

### 【例5】3Dプロット：サイズ vs 温度 vs LSPR

```python
from mpl_toolkits.mplot3d import Axes3D

# 3D散布図
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# カラーマップ
scatter = ax.scatter(data['size_nm'], data['temperature_C'], data['lspr_nm'],
                     c=data['pH'], cmap='viridis', s=50, alpha=0.6, edgecolors='k')

ax.set_xlabel('Size (nm)', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.set_zlabel('LSPR Wavelength (nm)', fontsize=11)
ax.set_title('3D Scatter: Size vs Temperature vs LSPR (colored by pH)',
             fontsize=13, fontweight='bold')

# カラーバー
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('pH', fontsize=10)

plt.tight_layout()
plt.show()

print("3Dプロットで多次元の関係を可視化しました")
```

---

## 3.3 前処理とデータ分割

### 【例6】欠損値処理

```python
# 欠損値を人為的に導入（実習用）
data_with_missing = data.copy()
np.random.seed(123)

# ランダムに5%の欠損値を導入
missing_indices = np.random.choice(data.index, size=int(0.05 * len(data)), replace=False)
data_with_missing.loc[missing_indices, 'temperature_C'] = np.nan

print("=" * 60)
print("欠損値の確認")
print("=" * 60)
print(f"欠損値の数:\n{data_with_missing.isnull().sum()}")

# 欠損値の処理方法1: 平均値で補完
data_filled_mean = data_with_missing.fillna(data_with_missing.mean())

# 欠損値の処理方法2: 中央値で補完
data_filled_median = data_with_missing.fillna(data_with_missing.median())

# 欠損値の処理方法3: 削除
data_dropped = data_with_missing.dropna()

print(f"\n元のデータ: {len(data_with_missing)}行")
print(f"欠損値削除後: {len(data_dropped)}行")
print(f"平均値補完後: {len(data_filled_mean)}行（欠損値なし）")

# 以降の分析では元のデータ（欠損値なし）を使用
data_clean = data.copy()
print("\n→ 以降は欠損値のないデータを使用します")
```

### 【例7】外れ値検出（IQR法）

```python
# IQR（四分位範囲）法による外れ値検出
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers, lower_bound, upper_bound

# サイズについて外れ値検出
outliers, lower, upper = detect_outliers_iqr(data_clean['size_nm'])

print("=" * 60)
print("外れ値検出（IQR法）")
print("=" * 60)
print(f"検出された外れ値の数: {outliers.sum()}")
print(f"下限: {lower:.2f} nm, 上限: {upper:.2f} nm")

# 可視化
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot([data_clean['size_nm']], labels=['Size (nm)'], vert=False)
ax.scatter(data_clean.loc[outliers, 'size_nm'],
           [1] * outliers.sum(), color='red', s=100,
           label=f'Outliers (n={outliers.sum()})', zorder=3)
ax.set_xlabel('Size (nm)', fontsize=12)
ax.set_title('Boxplot with Outliers', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("→ 外れ値は除去せず、全データを使用します")
```

### 【例8】特徴量スケーリング（StandardScaler）

```python
from sklearn.preprocessing import StandardScaler

# 特徴量とターゲットの分離
X = data_clean[['size_nm', 'temperature_C', 'pH']]
y = data_clean['lspr_nm']

# StandardScaler（平均0、標準偏差1に標準化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# スケーリング前後の比較
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print("=" * 60)
print("スケーリング前の統計量")
print("=" * 60)
print(X.describe())

print("\n" + "=" * 60)
print("スケーリング後の統計量（平均≈0、標準偏差≈1）")
print("=" * 60)
print(X_scaled_df.describe())

print("\n→ スケーリングにより各特徴量のスケールが統一されました")
```

### 【例9】訓練データとテストデータの分割

```python
from sklearn.model_selection import train_test_split

# 訓練データとテストデータに分割（80:20）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("データ分割")
print("=" * 60)
print(f"全データ数: {len(X)}サンプル")
print(f"訓練データ: {len(X_train)}サンプル ({len(X_train)/len(X)*100:.1f}%)")
print(f"テストデータ: {len(X_test)}サンプル ({len(X_test)/len(X)*100:.1f}%)")

print("\n訓練データの統計量:")
print(pd.DataFrame(X_train, columns=X.columns).describe())
```

---

## 3.4 回帰モデルによるナノ粒子物性予測

目標：サイズ、温度、pHからLSPR波長を予測

### 【例10】線形回帰（Linear Regression）

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 線形回帰モデルの構築
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# 予測
y_train_pred_lr = model_lr.predict(X_train)
y_test_pred_lr = model_lr.predict(X_test)

# 評価指標
r2_train_lr = r2_score(y_train, y_train_pred_lr)
r2_test_lr = r2_score(y_test, y_test_pred_lr)
rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
mae_test_lr = mean_absolute_error(y_test, y_test_pred_lr)

print("=" * 60)
print("線形回帰（Linear Regression）")
print("=" * 60)
print(f"訓練データ R²: {r2_train_lr:.4f}")
print(f"テストデータ R²: {r2_test_lr:.4f}")
print(f"テストデータ RMSE: {rmse_test_lr:.4f} nm")
print(f"テストデータ MAE: {mae_test_lr:.4f} nm")

# 回帰係数
print("\n回帰係数:")
for name, coef in zip(X.columns, model_lr.coef_):
    print(f"  {name}: {coef:.4f}")
print(f"  切片: {model_lr.intercept_:.4f}")

# 残差プロット
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 予測値 vs 実測値
axes[0].scatter(y_test, y_test_pred_lr, alpha=0.6, edgecolors='k')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual LSPR (nm)', fontsize=11)
axes[0].set_ylabel('Predicted LSPR (nm)', fontsize=11)
axes[0].set_title(f'Linear Regression (R² = {r2_test_lr:.3f})', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 残差プロット
residuals = y_test - y_test_pred_lr
axes[1].scatter(y_test_pred_lr, residuals, alpha=0.6, edgecolors='k')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted LSPR (nm)', fontsize=11)
axes[1].set_ylabel('Residuals (nm)', fontsize=11)
axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 【例11】ランダムフォレスト回帰（Random Forest）

```python
from sklearn.ensemble import RandomForestRegressor

# ランダムフォレスト回帰モデル
model_rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                                 random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

# 予測
y_train_pred_rf = model_rf.predict(X_train)
y_test_pred_rf = model_rf.predict(X_test)

# 評価
r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)

print("=" * 60)
print("ランダムフォレスト回帰（Random Forest）")
print("=" * 60)
print(f"訓練データ R²: {r2_train_rf:.4f}")
print(f"テストデータ R²: {r2_test_rf:.4f}")
print(f"テストデータ RMSE: {rmse_test_rf:.4f} nm")
print(f"テストデータ MAE: {mae_test_rf:.4f} nm")

# 特徴量重要度
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n特徴量重要度:")
print(feature_importance)

# 特徴量重要度の可視化
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feature_importance['Feature'], feature_importance['Importance'],
        color='steelblue', edgecolor='black')
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

### 【例12】勾配ブースティング（LightGBM）

```python
import lightgbm as lgb

# LightGBMモデルの構築
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'random_state': 42,
    'verbose': -1
}

model_lgb = lgb.LGBMRegressor(**params)
model_lgb.fit(X_train, y_train)

# 予測
y_train_pred_lgb = model_lgb.predict(X_train)
y_test_pred_lgb = model_lgb.predict(X_test)

# 評価
r2_train_lgb = r2_score(y_train, y_train_pred_lgb)
r2_test_lgb = r2_score(y_test, y_test_pred_lgb)
rmse_test_lgb = np.sqrt(mean_squared_error(y_test, y_test_pred_lgb))
mae_test_lgb = mean_absolute_error(y_test, y_test_pred_lgb)

print("=" * 60)
print("勾配ブースティング（LightGBM）")
print("=" * 60)
print(f"訓練データ R²: {r2_train_lgb:.4f}")
print(f"テストデータ R²: {r2_test_lgb:.4f}")
print(f"テストデータ RMSE: {rmse_test_lgb:.4f} nm")
print(f"テストデータ MAE: {mae_test_lgb:.4f} nm")

# 予測値 vs 実測値プロット
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_test_pred_lgb, alpha=0.6, edgecolors='k', s=60)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual LSPR (nm)', fontsize=12)
ax.set_ylabel('Predicted LSPR (nm)', fontsize=12)
ax.set_title(f'LightGBM Prediction (R² = {r2_test_lgb:.3f})',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 【例13】サポートベクター回帰（SVR）

```python
from sklearn.svm import SVR

# SVRモデル（RBFカーネル）
model_svr = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
model_svr.fit(X_train, y_train)

# 予測
y_train_pred_svr = model_svr.predict(X_train)
y_test_pred_svr = model_svr.predict(X_test)

# 評価
r2_train_svr = r2_score(y_train, y_train_pred_svr)
r2_test_svr = r2_score(y_test, y_test_pred_svr)
rmse_test_svr = np.sqrt(mean_squared_error(y_test, y_test_pred_svr))
mae_test_svr = mean_absolute_error(y_test, y_test_pred_svr)

print("=" * 60)
print("サポートベクター回帰（SVR）")
print("=" * 60)
print(f"訓練データ R²: {r2_train_svr:.4f}")
print(f"テストデータ R²: {r2_test_svr:.4f}")
print(f"テストデータ RMSE: {rmse_test_svr:.4f} nm")
print(f"テストデータ MAE: {mae_test_svr:.4f} nm")
print(f"サポートベクター数: {len(model_svr.support_)}")
```

### 【例14】ニューラルネットワーク（MLP Regressor）

```python
from sklearn.neural_network import MLPRegressor

# MLPモデル
model_mlp = MLPRegressor(hidden_layer_sizes=(100, 50),
                         activation='relu',
                         solver='adam',
                         alpha=0.001,
                         max_iter=500,
                         random_state=42,
                         early_stopping=True,
                         validation_fraction=0.1,
                         verbose=False)

model_mlp.fit(X_train, y_train)

# 予測
y_train_pred_mlp = model_mlp.predict(X_train)
y_test_pred_mlp = model_mlp.predict(X_test)

# 評価
r2_train_mlp = r2_score(y_train, y_train_pred_mlp)
r2_test_mlp = r2_score(y_test, y_test_pred_mlp)
rmse_test_mlp = np.sqrt(mean_squared_error(y_test, y_test_pred_mlp))
mae_test_mlp = mean_absolute_error(y_test, y_test_pred_mlp)

print("=" * 60)
print("ニューラルネットワーク（MLP Regressor）")
print("=" * 60)
print(f"訓練データ R²: {r2_train_mlp:.4f}")
print(f"テストデータ R²: {r2_test_mlp:.4f}")
print(f"テストデータ RMSE: {rmse_test_mlp:.4f} nm")
print(f"テストデータ MAE: {mae_test_mlp:.4f} nm")
print(f"反復回数: {model_mlp.n_iter_}")
print(f"隠れ層の構造: {model_mlp.hidden_layer_sizes}")
```

### 【例15】モデル性能比較

```python
# 全モデルの性能をまとめる
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'LightGBM', 'SVR', 'MLP'],
    'R² (Train)': [r2_train_lr, r2_train_rf, r2_train_lgb, r2_train_svr, r2_train_mlp],
    'R² (Test)': [r2_test_lr, r2_test_rf, r2_test_lgb, r2_test_svr, r2_test_mlp],
    'RMSE (Test)': [rmse_test_lr, rmse_test_rf, rmse_test_lgb, rmse_test_svr, rmse_test_mlp],
    'MAE (Test)': [mae_test_lr, mae_test_rf, mae_test_lgb, mae_test_svr, mae_test_mlp]
})

results['Overfit'] = results['R² (Train)'] - results['R² (Test)']

print("=" * 80)
print("全モデルの性能比較")
print("=" * 80)
print(results.to_string(index=False))

# 最良モデルの特定
best_model_idx = results['R² (Test)'].idxmax()
best_model_name = results.loc[best_model_idx, 'Model']
best_r2 = results.loc[best_model_idx, 'R² (Test)']

print(f"\n最良モデル: {best_model_name} (R² = {best_r2:.4f})")

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R²スコア比較
x_pos = np.arange(len(results))
axes[0].bar(x_pos, results['R² (Test)'], alpha=0.7, color='steelblue',
            edgecolor='black', label='Test R²')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(results['Model'], rotation=15, ha='right')
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('Model Comparison: R² Score', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].legend()

# RMSE比較
axes[1].bar(x_pos, results['RMSE (Test)'], alpha=0.7, color='coral',
            edgecolor='black', label='Test RMSE')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(results['Model'], rotation=15, ha='right')
axes[1].set_ylabel('RMSE (nm)', fontsize=12)
axes[1].set_title('Model Comparison: RMSE', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## 3.5 量子ドット発光波長予測

### 【例16】データ生成：CdSe量子ドット

CdSe量子ドットの発光波長は、Brus方程式に基づきサイズに依存します。

```python
# CdSe量子ドットデータの生成
np.random.seed(100)

n_qd_samples = 150

# 量子ドットのサイズ（2-10 nm）
size_qd = np.random.uniform(2, 10, n_qd_samples)

# Brus方程式の簡易近似: emission = 520 + 130/(size^0.8) + noise
emission = 520 + 130 / (size_qd ** 0.8) + np.random.normal(0, 10, n_qd_samples)

# 合成条件
synthesis_time = np.random.uniform(10, 120, n_qd_samples)  # 分
precursor_ratio = np.random.uniform(0.5, 2.0, n_qd_samples)  # モル比

# データフレーム作成
data_qd = pd.DataFrame({
    'size_nm': size_qd,
    'emission_nm': emission,
    'synthesis_time_min': synthesis_time,
    'precursor_ratio': precursor_ratio
})

print("=" * 60)
print("CdSe量子ドットデータの生成完了")
print("=" * 60)
print(data_qd.head(10))
print("\n基本統計量:")
print(data_qd.describe())

# サイズと発光波長の関係プロット
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(data_qd['size_nm'], data_qd['emission_nm'],
                     c=data_qd['synthesis_time_min'], cmap='plasma',
                     s=80, alpha=0.7, edgecolors='k')
ax.set_xlabel('Quantum Dot Size (nm)', fontsize=12)
ax.set_ylabel('Emission Wavelength (nm)', fontsize=12)
ax.set_title('CdSe Quantum Dot: Size vs Emission Wavelength',
             fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Synthesis Time (min)', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 【例17】量子ドットモデル（LightGBM）

```python
# 特徴量とターゲットの分離
X_qd = data_qd[['size_nm', 'synthesis_time_min', 'precursor_ratio']]
y_qd = data_qd['emission_nm']

# スケーリング
scaler_qd = StandardScaler()
X_qd_scaled = scaler_qd.fit_transform(X_qd)

# 訓練/テスト分割
X_qd_train, X_qd_test, y_qd_train, y_qd_test = train_test_split(
    X_qd_scaled, y_qd, test_size=0.2, random_state=42
)

# LightGBMモデル
model_qd = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=200,
    random_state=42,
    verbose=-1
)

model_qd.fit(X_qd_train, y_qd_train)

# 予測
y_qd_train_pred = model_qd.predict(X_qd_train)
y_qd_test_pred = model_qd.predict(X_qd_test)

# 評価
r2_qd_train = r2_score(y_qd_train, y_qd_train_pred)
r2_qd_test = r2_score(y_qd_test, y_qd_test_pred)
rmse_qd = np.sqrt(mean_squared_error(y_qd_test, y_qd_test_pred))
mae_qd = mean_absolute_error(y_qd_test, y_qd_test_pred)

print("=" * 60)
print("量子ドット発光波長予測モデル（LightGBM）")
print("=" * 60)
print(f"訓練データ R²: {r2_qd_train:.4f}")
print(f"テストデータ R²: {r2_qd_test:.4f}")
print(f"テストデータ RMSE: {rmse_qd:.4f} nm")
print(f"テストデータ MAE: {mae_qd:.4f} nm")
```

### 【例18】予測結果の可視化

```python
# 予測値 vs 実測値プロット（信頼区間付き）
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# テストデータのプロット
axes[0].scatter(y_qd_test, y_qd_test_pred, alpha=0.6, s=80,
                edgecolors='k', label='Test Data')
axes[0].plot([y_qd_test.min(), y_qd_test.max()],
             [y_qd_test.min(), y_qd_test.max()],
             'r--', lw=2, label='Perfect Prediction')

# ±10 nm の範囲を表示
axes[0].fill_between([y_qd_test.min(), y_qd_test.max()],
                     [y_qd_test.min()-10, y_qd_test.max()-10],
                     [y_qd_test.min()+10, y_qd_test.max()+10],
                     alpha=0.2, color='gray', label='±10 nm')

axes[0].set_xlabel('Actual Emission (nm)', fontsize=12)
axes[0].set_ylabel('Predicted Emission (nm)', fontsize=12)
axes[0].set_title(f'QD Emission Prediction (R² = {r2_qd_test:.3f})',
                  fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# サイズ別の予測精度
size_bins = [2, 4, 6, 8, 10]
size_labels = ['2-4 nm', '4-6 nm', '6-8 nm', '8-10 nm']
data_qd_test = pd.DataFrame({
    'size': X_qd.iloc[y_qd_test.index]['size_nm'].values,
    'actual': y_qd_test.values,
    'predicted': y_qd_test_pred
})
data_qd_test['size_bin'] = pd.cut(data_qd_test['size'], bins=size_bins, labels=size_labels)
data_qd_test['error'] = np.abs(data_qd_test['actual'] - data_qd_test['predicted'])

# サイズビンごとの平均誤差
error_by_size = data_qd_test.groupby('size_bin')['error'].mean()

axes[1].bar(range(len(error_by_size)), error_by_size.values,
            color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xticks(range(len(error_by_size)))
axes[1].set_xticklabels(error_by_size.index)
axes[1].set_ylabel('Mean Absolute Error (nm)', fontsize=12)
axes[1].set_xlabel('QD Size Range', fontsize=12)
axes[1].set_title('Prediction Error by QD Size', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n全体の平均絶対誤差: {mae_qd:.2f} nm")
print("サイズ別の平均絶対誤差:")
print(error_by_size)
```

---

## 3.6 特徴量重要度分析

### 【例19】特徴量重要度（LightGBM）

```python
# LightGBMモデルの特徴量重要度（ゲインベース）
importance_gain = model_lgb.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance_gain
}).sort_values('Importance', ascending=False)

print("=" * 60)
print("特徴量重要度（LightGBM）")
print("=" * 60)
print(importance_df)

# 可視化
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['steelblue', 'coral', 'lightgreen']
ax.barh(importance_df['Feature'], importance_df['Importance'],
        color=colors, edgecolor='black')
ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
ax.set_title('Feature Importance: LSPR Prediction',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print(f"\n最も重要な特徴量: {importance_df.iloc[0]['Feature']}")
```

### 【例20】SHAP分析：予測解釈

```python
import shap

# SHAP Explainerの作成
explainer = shap.Explainer(model_lgb, X_train)
shap_values = explainer(X_test)

print("=" * 60)
print("SHAP分析")
print("=" * 60)
print("SHAP値の計算完了")
print(f"SHAP値の形状: {shap_values.values.shape}")

# SHAP Summary Plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.title('SHAP Summary Plot: Feature Impact on LSPR Prediction',
          fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# SHAP Dependence Plot（最も重要な特徴量）
top_feature_idx = importance_df.index[0]
top_feature_name = X.columns[top_feature_idx]

fig, ax = plt.subplots(figsize=(10, 6))
shap.dependence_plot(top_feature_idx, shap_values.values, X_test,
                     feature_names=X.columns, show=False)
plt.title(f'SHAP Dependence Plot: {top_feature_name}',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\nSHAP分析により、{top_feature_name}がLSPR波長予測に最も影響することが確認されました")
```

---

## 3.7 ベイズ最適化によるナノ材料設計

目標：目標LSPR波長（550 nm）を実現する最適な合成条件を探索

### 【例21】探索空間の定義

```python
from skopt.space import Real

# 探索空間の定義
# サイズ: 10-40 nm、温度: 20-80°C、pH: 4-10
search_space = [
    Real(10, 40, name='size_nm'),
    Real(20, 80, name='temperature_C'),
    Real(4, 10, name='pH')
]

print("=" * 60)
print("ベイズ最適化：探索空間の定義")
print("=" * 60)
for dim in search_space:
    print(f"  {dim.name}: [{dim.bounds[0]}, {dim.bounds[1]}]")

print("\n目標: LSPR波長 = 550 nm を実現する条件を探索")
```

### 【例22】目的関数の設定

```python
# 目的関数：予測LSPR波長と目標波長（550 nm）の差の絶対値を最小化
target_lspr = 550.0

def objective_function(params):
    """
    ベイズ最適化の目的関数

    Parameters:
    -----------
    params : list
        [size_nm, temperature_C, pH]

    Returns:
    --------
    float
        目標波長との誤差（最小化する値）
    """
    # パラメータを取得
    size, temp, ph = params

    # 特徴量を構築（スケーリング適用）
    features = np.array([[size, temp, ph]])
    features_scaled = scaler.transform(features)

    # LSPR波長を予測
    predicted_lspr = model_lgb.predict(features_scaled)[0]

    # 目標波長との誤差（絶対値）
    error = abs(predicted_lspr - target_lspr)

    return error

# テスト実行
test_params = [20.0, 50.0, 7.0]
test_error = objective_function(test_params)
print(f"\nテスト実行:")
print(f"  パラメータ: size={test_params[0]} nm, temp={test_params[1]}°C, pH={test_params[2]}")
print(f"  目的関数値（誤差）: {test_error:.4f} nm")
```

### 【例23】ベイズ最適化の実行（scikit-optimize）

```python
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective

# ベイズ最適化の実行
print("\n" + "=" * 60)
print("ベイズ最適化の実行中...")
print("=" * 60)

result = gp_minimize(
    func=objective_function,
    dimensions=search_space,
    n_calls=50,  # 評価回数
    n_initial_points=10,  # ランダムサンプリング回数
    random_state=42,
    verbose=False
)

print("最適化完了！")
print("\n" + "=" * 60)
print("最適化結果")
print("=" * 60)
print(f"最小目的関数値（誤差）: {result.fun:.4f} nm")
print(f"\n最適パラメータ:")
print(f"  サイズ: {result.x[0]:.2f} nm")
print(f"  温度: {result.x[1]:.2f} °C")
print(f"  pH: {result.x[2]:.2f}")

# 最適条件での予測LSPR波長を計算
optimal_features = np.array([result.x])
optimal_features_scaled = scaler.transform(optimal_features)
predicted_optimal_lspr = model_lgb.predict(optimal_features_scaled)[0]

print(f"\n予測されるLSPR波長: {predicted_optimal_lspr:.2f} nm")
print(f"目標LSPR波長: {target_lspr} nm")
print(f"達成精度: {abs(predicted_optimal_lspr - target_lspr):.2f} nm")
```

### 【例24】最適化結果の可視化

```python
# 最適化プロセスの可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 収束プロット
plot_convergence(result, ax=axes[0])
axes[0].set_title('Convergence Plot', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Objective Value (Error, nm)', fontsize=11)
axes[0].set_xlabel('Number of Evaluations', fontsize=11)
axes[0].grid(True, alpha=0.3)

# 評価履歴のプロット
iterations = range(1, len(result.func_vals) + 1)
axes[1].plot(iterations, result.func_vals, 'o-', alpha=0.6, label='Evaluation')
axes[1].plot(iterations, np.minimum.accumulate(result.func_vals),
             'r-', linewidth=2, label='Best So Far')
axes[1].set_xlabel('Iteration', fontsize=11)
axes[1].set_ylabel('Objective Value (Error, nm)', fontsize=11)
axes[1].set_title('Optimization Progress', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 【例25】収束プロット

```python
# 詳細な収束プロット（最良値の推移）
fig, ax = plt.subplots(figsize=(10, 6))

cumulative_min = np.minimum.accumulate(result.func_vals)
iterations = np.arange(1, len(cumulative_min) + 1)

ax.plot(iterations, cumulative_min, 'b-', linewidth=2, marker='o',
        markersize=4, label='Best Error')
ax.axhline(y=result.fun, color='r', linestyle='--', linewidth=2,
           label=f'Final Best: {result.fun:.2f} nm')
ax.fill_between(iterations, 0, cumulative_min, alpha=0.2, color='blue')

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Minimum Error (nm)', fontsize=12)
ax.set_title('Bayesian Optimization: Convergence to Optimal Solution',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{len(result.func_vals)}回の評価で最適解に収束しました")
print(f"初期評価での最良誤差: {result.func_vals[0]:.2f} nm")
print(f"最終的な最良誤差: {result.fun:.2f} nm")
print(f"改善率: {(1 - result.fun/result.func_vals[0])*100:.1f}%")
```

---

## 3.8 多目的最適化：サイズと発光効率のトレードオフ

### 【例26】Pareto最適化（NSGA-II）

多目的最適化では、複数の目的を同時に最適化します。ここでは、量子ドットのサイズを最小化しつつ、発光効率（仮想的な指標）を最大化します。

```python
# pymooを使用した多目的最適化
try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    # 多目的最適化問題の定義
    class QuantumDotOptimization(Problem):
        def __init__(self):
            super().__init__(
                n_var=3,  # 変数数（size, synthesis_time, precursor_ratio）
                n_obj=2,  # 目的関数数（size最小化、emission効率最大化）
                n_constr=0,  # 制約なし
                xl=np.array([2.0, 10.0, 0.5]),  # 下限
                xu=np.array([10.0, 120.0, 2.0])  # 上限
            )

        def _evaluate(self, X, out, *args, **kwargs):
            # 目的関数1: サイズの最小化
            obj1 = X[:, 0]  # size

            # 目的関数2: 発光効率の最大化（負の値で最小化問題に変換）
            # 効率は仮想的に、emission wavelengthが550 nmに近いほど高いと仮定
            features = X  # [size, synthesis_time, precursor_ratio]
            features_scaled = scaler_qd.transform(features)
            predicted_emission = model_qd.predict(features_scaled)

            # 効率：550 nmからのずれが小さいほど高い（負の値で最大化→最小化）
            efficiency = -np.abs(predicted_emission - 550)
            obj2 = -efficiency  # 最大化を最小化問題に変換

            out["F"] = np.column_stack([obj1, obj2])

    # 問題のインスタンス化
    problem = QuantumDotOptimization()

    # NSGA-IIアルゴリズム
    algorithm = NSGA2(
        pop_size=40,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    # 最適化実行
    print("=" * 60)
    print("多目的最適化（NSGA-II）実行中...")
    print("=" * 60)

    res = pymoo_minimize(
        problem,
        algorithm,
        ('n_gen', 50),  # 世代数
        seed=42,
        verbose=False
    )

    print("多目的最適化完了！")
    print(f"\nパレート最適解の数: {len(res.F)}")

    # パレート最適解の表示（上位5つ）
    print("\n代表的なパレート最適解（上位5つ）:")
    pareto_solutions = pd.DataFrame({
        'Size (nm)': res.X[:, 0],
        'Synthesis Time (min)': res.X[:, 1],
        'Precursor Ratio': res.X[:, 2],
        'Obj1: Size': res.F[:, 0],
        'Obj2: -Efficiency': res.F[:, 1]
    }).head(5)
    print(pareto_solutions.to_string(index=False))

    PYMOO_AVAILABLE = True

except ImportError:
    print("=" * 60)
    print("pymooがインストールされていません")
    print("=" * 60)
    print("多目的最適化にはpymooが必要です:")
    print("  pip install pymoo")
    print("\n代わりに、簡易的な多目的最適化の例を表示します")

    # 簡易的なグリッドサーチによる多目的最適化の模擬
    sizes = np.linspace(2, 10, 20)
    times = np.linspace(10, 120, 20)
    ratios = np.linspace(0.5, 2.0, 20)

    # グリッドサーチ（サンプリング）
    sample_X = []
    sample_F = []

    for size in sizes[::4]:
        for time in times[::4]:
            for ratio in ratios[::4]:
                features = np.array([[size, time, ratio]])
                features_scaled = scaler_qd.transform(features)
                emission = model_qd.predict(features_scaled)[0]

                obj1 = size
                obj2 = abs(emission - 550)

                sample_X.append([size, time, ratio])
                sample_F.append([obj1, obj2])

    sample_X = np.array(sample_X)
    sample_F = np.array(sample_F)

    print("\nグリッドサーチによる解の探索完了")
    print(f"探索した解の数: {len(sample_F)}")

    res = type('Result', (), {
        'X': sample_X,
        'F': sample_F
    })()

    PYMOO_AVAILABLE = False
```

### 【例27】Paretoフロントの可視化

```python
# Paretoフロントの可視化
fig, ax = plt.subplots(figsize=(10, 7))

if PYMOO_AVAILABLE:
    # NSGA-IIの結果をプロット
    ax.scatter(res.F[:, 0], -res.F[:, 1], c='blue', s=80, alpha=0.6,
               edgecolors='black', label='Pareto Optimal Solutions')

    title_suffix = "(NSGA-II)"
else:
    # グリッドサーチの結果をプロット
    ax.scatter(res.F[:, 0], res.F[:, 1], c='blue', s=60, alpha=0.5,
               edgecolors='black', label='Sampled Solutions')

    title_suffix = "(Grid Search)"

ax.set_xlabel('Objective 1: Size (nm) [Minimize]', fontsize=12)

if PYMOO_AVAILABLE:
    ax.set_ylabel('Objective 2: Efficiency [Maximize]', fontsize=12)
else:
    ax.set_ylabel('Objective 2: Deviation from 550nm [Minimize]', fontsize=12)

ax.set_title(f'Pareto Front: Size vs Emission Efficiency {title_suffix}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nパレートフロント:")
print("  サイズを小さくすると効率が下がり、効率を上げるとサイズが大きくなる")
print("  → トレードオフ関係が明確に")
```

---

## 3.9 TEM画像解析とサイズ分布

### 【例28】模擬TEMデータの生成

TEM（透過型電子顕微鏡）で測定されるナノ粒子サイズは、しばしば対数正規分布に従います。

```python
from scipy.stats import lognorm

# 対数正規分布に従うTEMサイズデータの生成
np.random.seed(200)

# パラメータ
mean_size = 20  # 平均サイズ（nm）
cv = 0.3  # 変動係数（標準偏差/平均）

# 対数正規分布のパラメータ計算
sigma = np.sqrt(np.log(1 + cv**2))
mu = np.log(mean_size) - 0.5 * sigma**2

# サンプル生成（500粒子）
tem_sizes = lognorm.rvs(s=sigma, scale=np.exp(mu), size=500)

print("=" * 60)
print("TEM測定データの生成（対数正規分布）")
print("=" * 60)
print(f"サンプル数: {len(tem_sizes)}粒子")
print(f"平均サイズ: {tem_sizes.mean():.2f} nm")
print(f"標準偏差: {tem_sizes.std():.2f} nm")
print(f"中央値: {np.median(tem_sizes):.2f} nm")
print(f"最小値: {tem_sizes.min():.2f} nm")
print(f"最大値: {tem_sizes.max():.2f} nm")

# ヒストグラム
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(tem_sizes, bins=40, alpha=0.7, color='lightblue',
        edgecolor='black', density=True, label='TEM Data')
ax.set_xlabel('Particle Size (nm)', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('TEM Size Distribution (Lognormal)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### 【例29】対数正規分布フィッティング

```python
# 対数正規分布のフィッティング
shape_fit, loc_fit, scale_fit = lognorm.fit(tem_sizes, floc=0)

# フィッティングされた分布のパラメータ
fitted_mean = np.exp(np.log(scale_fit) + 0.5 * shape_fit**2)
fitted_std = fitted_mean * np.sqrt(np.exp(shape_fit**2) - 1)

print("=" * 60)
print("対数正規分布フィッティング結果")
print("=" * 60)
print(f"形状パラメータ (sigma): {shape_fit:.4f}")
print(f"スケールパラメータ: {scale_fit:.4f}")
print(f"フィッティングされた平均サイズ: {fitted_mean:.2f} nm")
print(f"フィッティングされた標準偏差: {fitted_std:.2f} nm")

# 実測値との比較
print(f"\n実測値との比較:")
print(f"  平均サイズ - 実測: {tem_sizes.mean():.2f} nm, フィット: {fitted_mean:.2f} nm")
print(f"  標準偏差 - 実測: {tem_sizes.std():.2f} nm, フィット: {fitted_std:.2f} nm")
```

### 【例30】フィッティング結果の可視化

```python
# フィッティング結果の詳細可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ヒストグラムとフィッティング曲線
axes[0].hist(tem_sizes, bins=40, alpha=0.6, color='lightblue',
             edgecolor='black', density=True, label='TEM Data')

# フィッティングされた対数正規分布
x_range = np.linspace(0, tem_sizes.max(), 200)
fitted_pdf = lognorm.pdf(x_range, shape_fit, loc=loc_fit, scale=scale_fit)
axes[0].plot(x_range, fitted_pdf, 'r-', linewidth=2,
             label=f'Lognormal Fit (μ={fitted_mean:.1f}, σ={fitted_std:.1f})')

axes[0].set_xlabel('Particle Size (nm)', fontsize=12)
axes[0].set_ylabel('Probability Density', fontsize=12)
axes[0].set_title('TEM Size Distribution with Lognormal Fit',
                  fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Q-Qプロット（分位点プロット）
from scipy.stats import probplot

probplot(tem_sizes, dist=lognorm, sparams=(shape_fit, loc_fit, scale_fit),
         plot=axes[1])
axes[1].set_title('Q-Q Plot: Lognormal Distribution',
                  fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nQ-Qプロット: データが直線上にあれば、対数正規分布に良く従っています")
```

---

## 3.10 分子動力学（MD）データ解析

### 【例31】MDシミュレーションデータの読み込み

分子動力学シミュレーションでは、ナノ粒子の原子配置の時間発展を追跡します。

```python
# MDシミュレーションデータの模擬生成
# 実際のMDデータはLAMMPS, GROMACS等から取得

np.random.seed(300)

n_atoms = 100  # 原子数
n_steps = 1000  # タイムステップ数
dt = 0.001  # タイムステップ（ps）

# 初期位置（nm）
positions_initial = np.random.uniform(-1, 1, (n_atoms, 3))

# 時間発展の模擬（ランダムウォーク）
positions = np.zeros((n_steps, n_atoms, 3))
positions[0] = positions_initial

for t in range(1, n_steps):
    # ランダムな変位
    displacement = np.random.normal(0, 0.01, (n_atoms, 3))
    positions[t] = positions[t-1] + displacement

print("=" * 60)
print("MDシミュレーションデータの生成")
print("=" * 60)
print(f"原子数: {n_atoms}")
print(f"タイムステップ数: {n_steps}")
print(f"シミュレーション時間: {n_steps * dt:.2f} ps")
print(f"データ形状: {positions.shape} (time, atoms, xyz)")

# 中心原子（原子0）の軌跡をプロット
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(positions[:, 0, 0], positions[:, 0, 1], positions[:, 0, 2],
         'b-', alpha=0.5, linewidth=0.5)
ax1.scatter(positions[0, 0, 0], positions[0, 0, 1], positions[0, 0, 2],
            c='green', s=100, label='Start', edgecolors='k')
ax1.scatter(positions[-1, 0, 0], positions[-1, 0, 1], positions[-1, 0, 2],
            c='red', s=100, label='End', edgecolors='k')
ax1.set_xlabel('X (nm)')
ax1.set_ylabel('Y (nm)')
ax1.set_zlabel('Z (nm)')
ax1.set_title('Atom Trajectory (Atom 0)', fontweight='bold')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.plot(np.arange(n_steps) * dt, positions[:, 0, 0], label='X')
ax2.plot(np.arange(n_steps) * dt, positions[:, 0, 1], label='Y')
ax2.plot(np.arange(n_steps) * dt, positions[:, 0, 2], label='Z')
ax2.set_xlabel('Time (ps)', fontsize=11)
ax2.set_ylabel('Position (nm)', fontsize=11)
ax2.set_title('Position vs Time (Atom 0)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 【例32】動径分布関数（RDF）の計算

動径分布関数（Radial Distribution Function, RDF）は、原子間距離の分布を表します。

```python
# 動径分布関数（RDF）の計算
def calculate_rdf(positions, r_max=2.0, n_bins=100):
    """
    動径分布関数を計算

    Parameters:
    -----------
    positions : ndarray
        原子位置 (n_atoms, 3)
    r_max : float
        最大距離（nm）
    n_bins : int
        ビン数

    Returns:
    --------
    r_bins : ndarray
        距離ビン
    rdf : ndarray
        動径分布関数
    """
    n_atoms = positions.shape[0]

    # 全原子ペア間の距離を計算
    distances = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < r_max:
                distances.append(dist)

    distances = np.array(distances)

    # ヒストグラム
    hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max))
    r_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 規格化（理想気体との比）
    dr = r_max / n_bins
    volume_shell = 4 * np.pi * r_bins**2 * dr
    n_ideal = volume_shell * (n_atoms / (4/3 * np.pi * r_max**3))

    rdf = hist / n_ideal / (n_atoms / 2)

    return r_bins, rdf

# 最終フレームでRDFを計算
final_positions = positions[-1]
r_bins, rdf = calculate_rdf(final_positions, r_max=1.5, n_bins=150)

print("=" * 60)
print("動径分布関数（RDF）")
print("=" * 60)
print(f"計算完了: {len(r_bins)}個のビン")

# RDFのプロット
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(r_bins, rdf, 'b-', linewidth=2)
ax.axhline(y=1, color='r', linestyle='--', linewidth=1, label='Ideal Gas (g(r)=1)')
ax.set_xlabel('Distance r (nm)', fontsize=12)
ax.set_ylabel('g(r)', fontsize=12)
ax.set_title('Radial Distribution Function (RDF)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(rdf) * 1.1)

plt.tight_layout()
plt.show()

# ピーク位置の検出
from scipy.signal import find_peaks

peaks, _ = find_peaks(rdf, height=1.2, distance=10)
print(f"\nRDFのピーク位置（特徴的な原子間距離）:")
for i, peak_idx in enumerate(peaks[:3], 1):
    print(f"  ピーク{i}: r = {r_bins[peak_idx]:.3f} nm, g(r) = {rdf[peak_idx]:.2f}")
```

### 【例33】拡散係数の計算（Mean Squared Displacement）

```python
# 平均二乗変位（MSD）の計算
def calculate_msd(positions):
    """
    平均二乗変位を計算

    Parameters:
    -----------
    positions : ndarray
        原子位置 (n_steps, n_atoms, 3)

    Returns:
    --------
    msd : ndarray
        平均二乗変位 (n_steps,)
    """
    n_steps, n_atoms, _ = positions.shape
    msd = np.zeros(n_steps)

    # 各タイムステップでのMSD
    for t in range(n_steps):
        displacement = positions[t] - positions[0]
        squared_displacement = np.sum(displacement**2, axis=1)
        msd[t] = np.mean(squared_displacement)

    return msd

# MSDの計算
msd = calculate_msd(positions)
time = np.arange(n_steps) * dt

print("=" * 60)
print("平均二乗変位（MSD）と拡散係数")
print("=" * 60)

# 拡散係数の計算（Einstein関係式: MSD = 6*D*t）
# 線形フィット（後半50%のデータを使用）
start_idx = n_steps // 2
fit_coeffs = np.polyfit(time[start_idx:], msd[start_idx:], 1)
slope = fit_coeffs[0]
diffusion_coefficient = slope / 6

print(f"拡散係数 D = {diffusion_coefficient:.6f} nm²/ps")
print(f"            = {diffusion_coefficient * 1e3:.6f} × 10⁻⁶ cm²/s")

# MSDプロット
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time, msd, 'b-', linewidth=2, label='MSD')
ax.plot(time[start_idx:], fit_coeffs[0] * time[start_idx:] + fit_coeffs[1],
        'r--', linewidth=2, label=f'Linear Fit (D={diffusion_coefficient:.4f} nm²/ps)')
ax.set_xlabel('Time (ps)', fontsize=12)
ax.set_ylabel('MSD (nm²)', fontsize=12)
ax.set_title('Mean Squared Displacement (MSD)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n拡散係数は、ナノ粒子の移動性を定量的に評価する重要な指標です")
```

---

## 3.11 異常検知：品質管理への応用

### 【例34】Isolation Forestによる異常ナノ粒子検出

製造プロセスで生成されたナノ粒子の品質管理に、機械学習による異常検知を適用します。

```python
from sklearn.ensemble import IsolationForest

# 正常データと異常データを混在させる
np.random.seed(400)

# 正常な金ナノ粒子データ（180サンプル）
normal_size = np.random.normal(15, 3, 180)
normal_lspr = 520 + 0.8 * (normal_size - 15) + np.random.normal(0, 3, 180)

# 異常なナノ粒子データ（20サンプル）：サイズが異常に大きいor小さい
anomaly_size = np.concatenate([
    np.random.uniform(5, 8, 10),  # 異常に小さい
    np.random.uniform(35, 50, 10)  # 異常に大きい
])
anomaly_lspr = 520 + 0.8 * (anomaly_size - 15) + np.random.normal(0, 8, 20)

# 全データを結合
all_size = np.concatenate([normal_size, anomaly_size])
all_lspr = np.concatenate([normal_lspr, anomaly_lspr])
all_data = np.column_stack([all_size, all_lspr])

# ラベル（正常=0、異常=1）
true_labels = np.concatenate([np.zeros(180), np.ones(20)])

print("=" * 60)
print("異常検知（Isolation Forest）")
print("=" * 60)
print(f"全データ数: {len(all_data)}")
print(f"正常データ: {int((true_labels == 0).sum())}サンプル")
print(f"異常データ: {int((true_labels == 1).sum())}サンプル")

# Isolation Forestモデル
iso_forest = IsolationForest(
    contamination=0.1,  # 異常データの割合（10%と仮定）
    random_state=42,
    n_estimators=100
)

# 異常検知
predictions = iso_forest.fit_predict(all_data)
anomaly_scores = iso_forest.score_samples(all_data)

# 予測結果（1: 正常、-1: 異常）
predicted_anomalies = (predictions == -1)
true_anomalies = (true_labels == 1)

# 評価指標
from sklearn.metrics import confusion_matrix, classification_report

# 予測を0/1に変換
predicted_labels = (predictions == -1).astype(int)

print("\n混同行列:")
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)

print("\n分類レポート:")
print(classification_report(true_labels, predicted_labels,
                            target_names=['Normal', 'Anomaly']))

# 検出率
detected_anomalies = np.sum(predicted_anomalies & true_anomalies)
total_anomalies = np.sum(true_anomalies)
detection_rate = detected_anomalies / total_anomalies * 100

print(f"\n異常検出率: {detection_rate:.1f}% ({detected_anomalies}/{total_anomalies})")
```

### 【例35】異常サンプルの可視化

```python
# 異常検知結果の可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 散布図（真のラベル）
axes[0].scatter(all_size[true_labels == 0], all_lspr[true_labels == 0],
                c='blue', s=60, alpha=0.6, label='Normal', edgecolors='k')
axes[0].scatter(all_size[true_labels == 1], all_lspr[true_labels == 1],
                c='red', s=100, alpha=0.8, marker='^', label='True Anomaly',
                edgecolors='k', linewidths=2)
axes[0].set_xlabel('Size (nm)', fontsize=12)
axes[0].set_ylabel('LSPR Wavelength (nm)', fontsize=12)
axes[0].set_title('True Labels', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 散布図（予測結果）
normal_mask = ~predicted_anomalies
anomaly_mask = predicted_anomalies

axes[1].scatter(all_size[normal_mask], all_lspr[normal_mask],
                c='blue', s=60, alpha=0.6, label='Predicted Normal', edgecolors='k')
axes[1].scatter(all_size[anomaly_mask], all_lspr[anomaly_mask],
                c='orange', s=100, alpha=0.8, marker='X', label='Predicted Anomaly',
                edgecolors='k', linewidths=2)

# 正しく検出された異常を強調
correctly_detected = predicted_anomalies & true_anomalies
axes[1].scatter(all_size[correctly_detected], all_lspr[correctly_detected],
                c='red', s=150, marker='*', label='Correctly Detected',
                edgecolors='black', linewidths=1.5, zorder=5)

axes[1].set_xlabel('Size (nm)', fontsize=12)
axes[1].set_ylabel('LSPR Wavelength (nm)', fontsize=12)
axes[1].set_title('Isolation Forest Predictions', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 異常スコアの分布
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(anomaly_scores[true_labels == 0], bins=30, alpha=0.6,
        color='blue', label='Normal', edgecolor='black')
ax.hist(anomaly_scores[true_labels == 1], bins=30, alpha=0.6,
        color='red', label='Anomaly', edgecolor='black')
ax.set_xlabel('Anomaly Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n異常スコアが低い（負の値が大きい）ほど、異常である可能性が高い")
```

---

## まとめ

本章では、Pythonを使ったナノ材料データ解析と機械学習の実践的手法を35個のコード例で学びました。

### 習得した主要技術

1. **データ生成と可視化**（例1-5）
   - 金ナノ粒子、量子ドットの合成データ生成
   - ヒストグラム、散布図、3Dプロット、相関分析

2. **データ前処理**（例6-9）
   - 欠損値処理、外れ値検出、スケーリング、データ分割

3. **回帰モデルによる物性予測**（例10-15）
   - 線形回帰、ランダムフォレスト、LightGBM、SVR、MLP
   - モデル性能比較（R²、RMSE、MAE）

4. **量子ドット発光予測**（例16-18）
   - Brus方程式に基づくデータ生成
   - LightGBMによる予測モデル構築

5. **特徴量重要度とモデル解釈**（例19-20）
   - LightGBM特徴量重要度
   - SHAP分析による予測の解釈

6. **ベイズ最適化**（例21-25）
   - 目標LSPR波長を実現する最適合成条件の探索
   - 収束プロット、最適化プロセスの可視化

7. **多目的最適化**（例26-27）
   - NSGA-IIによるパレート最適化
   - サイズと発光効率のトレードオフ分析

8. **TEM画像解析**（例28-30）
   - 対数正規分布によるサイズ分布フィッティング
   - Q-Qプロットによる分布の検証

9. **分子動力学データ解析**（例31-33）
   - 原子軌跡の可視化
   - 動径分布関数（RDF）の計算
   - 拡散係数の算出（MSD法）

10. **異常検知**（例34-35）
    - Isolation Forestによる品質管理
    - 異常ナノ粒子の自動検出

### 実践的な応用

これらの技術は、以下のような実際のナノ材料研究に直接応用できます：

- **材料設計**: 機械学習による物性予測と最適化による高効率材料探索
- **プロセス最適化**: ベイズ最適化による実験回数削減と最適合成条件発見
- **品質管理**: 異常検知による不良品の早期発見と歩留まり向上
- **データ解析**: TEMデータ、MDシミュレーションデータの定量的解析
- **モデル解釈**: SHAP分析による予測根拠の可視化と信頼性向上

### 次章の予告

Chapter 4では、これらの技術を実際のナノ材料研究プロジェクトに適用した5つの詳細なケーススタディを学びます。カーボンナノチューブ複合材料、量子ドット、金ナノ粒子触媒、グラフェン、ナノ医薬の実用化事例を通じて、問題解決の全体像を理解します。

---

## 演習問題

### 演習1: カーボンナノチューブの電気伝導度予測

カーボンナノチューブ（CNT）の電気伝導度は、直径、カイラリティ、長さに依存します。以下のデータを生成し、LightGBMモデルで予測してください。

**データ仕様**：
- サンプル数：150
- 特徴量：直径（1-3 nm）、長さ（100-1000 nm）、カイラリティ指標（0-1の連続値）
- ターゲット：電気伝導度（10³-10⁷ S/m、対数正規分布）

**タスク**：
1. データ生成
2. 訓練/テストデータ分割
3. LightGBMモデルの構築と評価
4. 特徴量重要度の可視化

<details>
<summary>解答例</summary>

```python
# データ生成
np.random.seed(500)
n_samples = 150

diameter = np.random.uniform(1, 3, n_samples)
length = np.random.uniform(100, 1000, n_samples)
chirality = np.random.uniform(0, 1, n_samples)

# 電気伝導度（簡易モデル: 直径とカイラリティに強く依存）
log_conductivity = 3 + 2*diameter + 3*chirality + 0.001*length + np.random.normal(0, 0.5, n_samples)
conductivity = 10 ** log_conductivity  # S/m

data_cnt = pd.DataFrame({
    'diameter_nm': diameter,
    'length_nm': length,
    'chirality': chirality,
    'conductivity_Sm': conductivity
})

# 特徴量とターゲット
X_cnt = data_cnt[['diameter_nm', 'length_nm', 'chirality']]
y_cnt = np.log10(data_cnt['conductivity_Sm'])  # 対数変換

# スケーリング
scaler_cnt = StandardScaler()
X_cnt_scaled = scaler_cnt.fit_transform(X_cnt)

# 訓練/テスト分割
X_cnt_train, X_cnt_test, y_cnt_train, y_cnt_test = train_test_split(
    X_cnt_scaled, y_cnt, test_size=0.2, random_state=42
)

# LightGBMモデル
model_cnt = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=200, random_state=42, verbose=-1)
model_cnt.fit(X_cnt_train, y_cnt_train)

# 予測と評価
y_cnt_pred = model_cnt.predict(X_cnt_test)
r2_cnt = r2_score(y_cnt_test, y_cnt_pred)
rmse_cnt = np.sqrt(mean_squared_error(y_cnt_test, y_cnt_pred))

print(f"R²: {r2_cnt:.4f}")
print(f"RMSE: {rmse_cnt:.4f}")

# 特徴量重要度
importance_cnt = pd.DataFrame({
    'Feature': X_cnt.columns,
    'Importance': model_cnt.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n特徴量重要度:")
print(importance_cnt)

# 可視化
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance_cnt['Feature'], importance_cnt['Importance'], color='steelblue', edgecolor='black')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance: CNT Conductivity Prediction')
plt.tight_layout()
plt.show()
```

</details>

### 演習2: 銀ナノ粒子の最適合成条件探索

銀ナノ粒子の抗菌活性は、サイズが小さいほど高くなります。ベイズ最適化を用いて、目標サイズ（10 nm）を実現する最適な合成温度とpHを探索してください。

**条件**：
- 温度範囲：20-80°C
- pH範囲：6-11
- 目標サイズ：10 nm

<details>
<summary>解答例</summary>

```python
# 銀ナノ粒子データの生成
np.random.seed(600)
n_ag = 100

temp_ag = np.random.uniform(20, 80, n_ag)
pH_ag = np.random.uniform(6, 11, n_ag)

# サイズモデル（温度が高く、pHが低いほど小さくなると仮定）
size_ag = 15 - 0.1*temp_ag - 0.8*pH_ag + np.random.normal(0, 1, n_ag)
size_ag = np.clip(size_ag, 5, 30)

data_ag = pd.DataFrame({
    'temperature': temp_ag,
    'pH': pH_ag,
    'size': size_ag
})

# モデル構築（LightGBM）
X_ag = data_ag[['temperature', 'pH']]
y_ag = data_ag['size']

scaler_ag = StandardScaler()
X_ag_scaled = scaler_ag.fit_transform(X_ag)

model_ag = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
model_ag.fit(X_ag_scaled, y_ag)

# ベイズ最適化
from skopt import gp_minimize
from skopt.space import Real

space_ag = [
    Real(20, 80, name='temperature'),
    Real(6, 11, name='pH')
]

target_size = 10.0

def objective_ag(params):
    temp, ph = params
    features = scaler_ag.transform([[temp, ph]])
    predicted_size = model_ag.predict(features)[0]
    return abs(predicted_size - target_size)

result_ag = gp_minimize(objective_ag, space_ag, n_calls=40, random_state=42, verbose=False)

print("=" * 60)
print("銀ナノ粒子の最適合成条件")
print("=" * 60)
print(f"最小誤差: {result_ag.fun:.2f} nm")
print(f"最適温度: {result_ag.x[0]:.1f} °C")
print(f"最適pH: {result_ag.x[1]:.2f}")

# 最適条件での予測サイズ
optimal_features = scaler_ag.transform([result_ag.x])
predicted_size = model_ag.predict(optimal_features)[0]
print(f"予測サイズ: {predicted_size:.2f} nm")
```

</details>

### 演習3: 量子ドットの多色発光設計

赤（650 nm）、緑（550 nm）、青（450 nm）の3色の発光を実現するCdSe量子ドットのサイズを、ベイズ最適化で設計してください。

**ヒント**：
- 各色ごとに最適化を実行
- 発光波長とサイズの関係を使用

<details>
<summary>解答例</summary>

```python
# 量子ドットデータ（例16のdata_qdを使用）
# model_qd と scaler_qd が構築済みと仮定

# 3色の目標波長
target_colors = {
    'Red': 650,
    'Green': 550,
    'Blue': 450
}

results_colors = {}

for color_name, target_emission in target_colors.items():
    # 探索空間
    space_qd = [
        Real(2, 10, name='size_nm'),
        Real(10, 120, name='synthesis_time_min'),
        Real(0.5, 2.0, name='precursor_ratio')
    ]

    # 目的関数
    def objective_qd(params):
        features = scaler_qd.transform([params])
        predicted_emission = model_qd.predict(features)[0]
        return abs(predicted_emission - target_emission)

    # 最適化
    result_qd_color = gp_minimize(objective_qd, space_qd, n_calls=30, random_state=42, verbose=False)

    # 結果保存
    optimal_features = scaler_qd.transform([result_qd_color.x])
    predicted_emission = model_qd.predict(optimal_features)[0]

    results_colors[color_name] = {
        'target': target_emission,
        'size': result_qd_color.x[0],
        'time': result_qd_color.x[1],
        'ratio': result_qd_color.x[2],
        'predicted': predicted_emission,
        'error': result_qd_color.fun
    }

# 結果表示
print("=" * 80)
print("量子ドット多色発光設計")
print("=" * 80)

results_df = pd.DataFrame(results_colors).T
print(results_df.to_string())

# 可視化
fig, ax = plt.subplots(figsize=(10, 6))
colors_rgb = {'Red': 'red', 'Green': 'green', 'Blue': 'blue'}

for color_name, result in results_colors.items():
    ax.scatter(result['size'], result['predicted'],
               s=200, color=colors_rgb[color_name],
               edgecolors='black', linewidths=2, label=color_name)

ax.set_xlabel('Quantum Dot Size (nm)', fontsize=12)
ax.set_ylabel('Emission Wavelength (nm)', fontsize=12)
ax.set_title('Multi-Color Quantum Dot Design', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

</details>

---

## 3.12 章末チェックリスト：ナノ材料データ解析スキルの品質保証

本章で学んだPythonによるナノ材料データ解析と機械学習の実装スキルを体系的にチェックします。

### 3.12.1 環境構築スキル（Environment Setup）

#### 基礎レベル
- [ ] Python 3.9以上がインストールされている
- [ ] 3つの環境構築オプション（Anaconda/venv/Colab）の違いを説明できる
- [ ] 自分の状況に最適な環境を選択できる
- [ ] 仮想環境を作成・有効化・無効化できる
- [ ] pip/condaでライブラリをインストールできる（pandas、numpy、matplotlib、scikit-learn、lightgbm）
- [ ] 環境検証コードを実行し、エラーなく動作確認できる

#### 応用レベル
- [ ] requirements.txtを作成・使用できる
- [ ] Google ColabでGoogle Driveをマウントしてデータを読み込める
- [ ] 複数の仮想環境を用途別に使い分けられる
- [ ] インストールエラーを自力でトラブルシューティングできる
- [ ] オプションライブラリ（pymoo、SHAP）を必要に応じてインストールできる

---

### 3.12.2 データ処理・可視化スキル（Data Processing & Visualization）

#### 基礎レベル
- [ ] NumPyで合成データを生成できる（正規分布、一様分布）
- [ ] Pandasでデータフレームを作成・操作できる
- [ ] 基本統計量（mean、std、median）を計算できる
- [ ] ヒストグラムを作成できる
- [ ] 散布図を作成できる
- [ ] 欠損値を検出できる（`isnull().sum()`）
- [ ] 欠損値を削除または補完できる（`dropna()` or `fillna()`）

#### 応用レベル
- [ ] 相関行列を計算・可視化できる（`corr()`、seaborn.heatmap）
- [ ] ペアプロット（散布図マトリックス）を作成できる（seaborn.pairplot）
- [ ] 3D散布図を作成できる（mpl_toolkits.mplot3d）
- [ ] KDE（カーネル密度推定）を使用できる
- [ ] 外れ値をIQR法で検出できる
- [ ] StandardScalerでデータを標準化できる
- [ ] train_test_splitでデータを分割できる（80% vs 20%）
- [ ] `random_state=42`で再現性を確保している

#### 上級レベル
- [ ] 対数正規分布フィッティングができる（scipy.stats.lognorm）
- [ ] Q-Qプロットで分布の適合度を検証できる
- [ ] カラーマップを効果的に使用できる（viridis、plasma、coolwarm）
- [ ] 複数のsubplotsを使った高度な可視化ができる

---

### 3.12.3 機械学習モデル実装スキル（ML Model Implementation）

#### 基礎レベル（5つのモデル実装）
- [ ] 線形回帰を実装し、係数の意味を説明できる
- [ ] ランダムフォレストを実装し、`n_estimators`の役割を説明できる
- [ ] LightGBMをインストール・実装できる
- [ ] SVRで標準化（StandardScaler）の必要性を理解している
- [ ] MLPRegressor（ニューラルネットワーク）を実装できる

#### 応用レベル（モデル選択と評価）
- [ ] MAE、R²、RMSEを計算・解釈できる
- [ ] 訓練データとテストデータの性能差を評価できる
- [ ] 過学習を検出できる（訓練R² ≫ テストR²）
- [ ] 5つのモデルの性能を比較表で整理できる
- [ ] 予測値 vs 実測値の散布図を作成できる
- [ ] 残差プロットを作成し、モデルの偏りを検出できる

#### 上級レベル
- [ ] データ特性に応じて最適なモデルを選択できる
  - 線形性が強い → 線形回帰
  - 非線形性が強い → ランダムフォレスト、LightGBM
  - データ数が少ない → SVR
- [ ] 各モデルのハイパーパラメータの役割を理解している
  - ランダムフォレスト：n_estimators、max_depth
  - LightGBM：learning_rate、num_leaves
  - SVR：C、gamma、epsilon
  - MLP：hidden_layer_sizes、alpha、early_stopping

---

### 3.12.4 特徴量重要度とモデル解釈スキル（Feature Importance & Interpretability）

#### 基礎レベル
- [ ] ランダムフォレストの特徴量重要度を取得・可視化できる（`feature_importances_`）
- [ ] LightGBMの特徴量重要度を取得・可視化できる
- [ ] 特徴量重要度の結果を解釈できる（最も影響する特徴量は何か）

#### 応用レベル
- [ ] SHAPライブラリをインストール・使用できる
- [ ] SHAP Explainerを作成できる（`shap.Explainer`）
- [ ] SHAP Summary Plotを作成・解釈できる
- [ ] SHAP Dependence Plotを作成・解釈できる
- [ ] SHAP値の正負が予測に与える影響を説明できる

#### 上級レベル
- [ ] 複数の解釈手法を使い分けられる
  - 特徴量重要度：全体的な重要度
  - SHAP：個別サンプルの予測理由
- [ ] モデルの予測根拠をステークホルダーに説明できる

---

### 3.12.5 ベイズ最適化スキル（Bayesian Optimization）

#### 基礎レベル
- [ ] scikit-optimizeをインストールできる
- [ ] 探索空間を定義できる（`Real(min, max, name)`）
- [ ] 目的関数を定義できる（パラメータを受け取り、誤差を返す）
- [ ] `gp_minimize`を実行できる
- [ ] 最適化結果（result.x、result.fun）を取得できる

#### 応用レベル
- [ ] n_callsとn_initial_pointsの役割を理解している
  - n_calls：評価回数
  - n_initial_points：ランダムサンプリング回数
- [ ] 収束プロットを作成できる（`plot_convergence`）
- [ ] 最適化プロセスの可視化ができる（評価履歴、最良値の推移）
- [ ] 目標値への達成精度を評価できる

#### 上級レベル
- [ ] 複数の目標（赤・緑・青の量子ドット）に対して最適化を実行できる
- [ ] 最適化結果を実験検証計画に活用できる
- [ ] 獲得関数（Acquisition Function）の概念を理解している

---

### 3.12.6 多目的最適化スキル（Multi-Objective Optimization）

#### 基礎レベル
- [ ] pymooライブラリをインストールできる
- [ ] 多目的最適化問題の概念を理解している（サイズ最小化 vs 効率最大化）
- [ ] パレートフロントの概念を説明できる

#### 応用レベル
- [ ] pymoo.core.problemを継承してProblemクラスを定義できる
- [ ] NSGA-IIアルゴリズムを実装できる
- [ ] パレートフロントを可視化できる
- [ ] トレードオフ関係を解釈できる

#### 上級レベル
- [ ] 複数の解から用途に応じた最適解を選択できる
  - 高性能重視
  - 環境重視
  - バランス型
- [ ] グリッドサーチによる代替実装ができる（pymooが使えない場合）

---

### 3.12.7 ナノ材料特有の解析スキル（Nanomaterial-Specific Analysis）

#### TEM画像解析
- [ ] 対数正規分布に従うサイズデータを生成できる
- [ ] 対数正規分布のパラメータ（sigma、mu）を計算できる
- [ ] `lognorm.fit`でフィッティングできる
- [ ] フィッティング結果を可視化できる（ヒストグラム + PDF曲線）
- [ ] Q-Qプロットで分布の適合度を評価できる

#### 分子動力学（MD）データ解析
- [ ] 原子軌跡データの構造を理解している（n_steps × n_atoms × 3）
- [ ] 3D軌跡プロットを作成できる
- [ ] 動径分布関数（RDF）を計算できる
- [ ] RDFのピーク位置から特徴的な原子間距離を抽出できる
- [ ] 平均二乗変位（MSD）を計算できる
- [ ] MSDから拡散係数を算出できる（Einstein関係式）

#### 異常検知
- [ ] Isolation Forestを実装できる
- [ ] contamination（異常データ割合）を設定できる
- [ ] 異常スコアを計算できる（`score_samples`）
- [ ] 混同行列で異常検出精度を評価できる
- [ ] 正常データと異常データの分布を可視化できる

---

### 3.12.8 コード品質スキル（Code Quality）

#### 基礎レベル
- [ ] すべてのコードに乱数シード（`random_state=42`）を設定している
- [ ] データ検証（shape、dtype、欠損値、範囲）を実施している
- [ ] 変数名が分かりやすい（`X_train`、`y_test`、`model_lgb`）
- [ ] コメントで処理の目的を説明している
- [ ] グラフにタイトル、軸ラベル、凡例を追加している

#### 応用レベル
- [ ] 関数化してコードを再利用可能にしている
  ```python
  def calculate_rdf(positions, r_max, n_bins):
      ...
  ```
- [ ] ドキュメンテーション文字列（Docstring）を記述している
- [ ] グラフの美観を整えている（fontsize、grid、alpha）
- [ ] try-exceptでエラーハンドリングを実装している（pymooのImportError対応）

---

### 3.12.9 トラブルシューティングスキル（Troubleshooting）

#### 基礎レベル（エラー対処）
- [ ] `ModuleNotFoundError`を解決できる（`pip install`）
- [ ] `ValueError: Input contains NaN`を解決できる（欠損値処理）
- [ ] `ConvergenceWarning`（MLPの収束エラー）を解決できる
  - `max_iter`を増やす
  - データを標準化する
  - Early Stoppingを有効化
- [ ] エラーメッセージを読み、検索して解決策を見つけられる

#### 応用レベル（性能改善）
- [ ] R² < 0.7の場合、3つ以上の改善策を実行できる
  - 特徴量エンジニアリング
  - モデル変更（線形→非線形）
  - ハイパーパラメータ調整
- [ ] 過学習を検出できる（訓練R² ≫ テストR²）
- [ ] 未学習を検出できる（訓練R²もテストR²も低い）

---

### 3.12.10 総合評価：習熟度レベル判定

以下のレベル判定で、自分の到達度を確認してください。

#### レベル1：初心者（Beginner）
- 環境構築スキル：基礎レベル 100%達成
- データ処理・可視化スキル：基礎レベル 80%以上達成
- 機械学習モデル実装スキル：基礎レベル 5つ中3つ以上実装
- トラブルシューティング：基礎レベルのエラーを自力で解決

**到達目標:** ナノ粒子データを生成・可視化し、線形回帰とランダムフォレストでLSPR波長予測を実装できる

---

#### レベル2：中級者（Intermediate）
- 環境構築スキル：応用レベル 80%以上達成
- データ処理・可視化スキル：基礎レベル 100%達成 + 応用レベル 70%以上
- 機械学習モデル実装スキル：基礎レベル 100%達成 + 応用レベル 70%以上
- 特徴量重要度とモデル解釈スキル：基礎レベル 100%達成 + 応用レベル 50%以上
- ベイズ最適化スキル：基礎レベル 100%達成 + 応用レベル 50%以上

**到達目標:** 5つの回帰モデルを比較し、ベイズ最適化で目標LSPR波長（550 nm）を達成する合成条件を発見できる

---

#### レベル3：上級者（Advanced）
- 全カテゴリ：応用レベル 100%達成
- 特徴量重要度とモデル解釈スキル：上級レベル 80%以上
- ベイズ最適化スキル：上級レベル 80%以上
- 多目的最適化スキル：応用レベル 100%達成
- ナノ材料特有の解析スキル：TEM、MD、異常検知すべて実装

**到達目標:** SHAP分析でモデル解釈し、多目的最適化でサイズと効率のトレードオフを可視化できる

---

#### レベル4：エキスパート（Expert）
- 全カテゴリ：上級レベル 80%以上達成
- コード品質：応用レベル 100%達成
- 独自のナノ材料データ（実験データまたは文献データ）に適用できる
- カスタム機械学習パイプラインを構築できる
- 研究成果を学会発表・論文投稿できる

**到達目標:**
- 実ナノ材料データ（TEM、UV-Vis、XRD）を統合解析
- 機械学習により新規ナノ粒子の物性を90%以上の精度で予測
- ベイズ最適化で実験回数を従来の1/5に削減

---

### 3.12.11 実践プロジェクトチェック：演習問題の完遂

#### 演習1完遂確認（CNT電気伝導度予測）
- [ ] データ生成（150サンプル、3特徴量）を実装
- [ ] LightGBMモデルで予測を実装
- [ ] R² > 0.8、RMSE < 0.5を達成
- [ ] 特徴量重要度を可視化
- [ ] 結果を解釈（どの特徴量が最も影響するか）

#### 演習2完遂確認（銀ナノ粒子最適合成条件）
- [ ] 銀ナノ粒子データ（100サンプル）を生成
- [ ] LightGBMモデルを構築
- [ ] ベイズ最適化を実行（40回評価）
- [ ] 目標サイズ（10 nm）との誤差 < 1 nmを達成
- [ ] 最適温度・pHを特定

#### 演習3完遂確認（量子ドット多色発光設計）
- [ ] 赤・緑・青の3色について最適化を実行
- [ ] 各色の最適サイズ・合成条件を特定
- [ ] 予測波長が目標波長±10 nm以内に収まっている
- [ ] 結果を可視化（サイズ vs 波長のプロット）

---

### 3.12.12 次のステップへの準備度チェック

#### 実世界応用（Chapter 4）への準備
- [ ] 機械学習の基本的なワークフロー（データ準備→モデル訓練→評価→最適化）を理解している
- [ ] ナノ材料特有のデータ（サイズ分布、光学特性、電気特性）を扱える
- [ ] 最適化手法（ベイズ最適化、多目的最適化）を実装できる
- [ ] モデル解釈（SHAP）の重要性を理解している

#### 深層学習・グラフニューラルネットワークへの準備
- [ ] ニューラルネットワーク（MLP）を実装し、活性化関数・最適化アルゴリズムを理解している
- [ ] 学習曲線を可視化し、過学習を検出できる
- [ ] Early Stoppingの概念を理解している

#### 実務研究への準備
- [ ] Jupyter NotebookまたはPythonスクリプトでコードを管理できる
- [ ] requirements.txtで環境を再現可能にしている
- [ ] 予測結果をグラフ化し、レポートにまとめられる
- [ ] コードにドキュメンテーションを記述している

---

**チェックリスト活用のヒント:**
1. **定期的に見直す**: 学習後、1週間後、1ヶ月後に再チェック
2. **未達成項目を優先**: チェックできない項目を集中学習
3. **レベル判定を記録**: 成長を可視化してモチベーション維持
4. **実プロジェクトでの活用**: 研究・開発プロジェクト開始前に必須スキルを確認

---

## 参考文献

1. **Pedregosa, F. et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

2. **Ke, G. et al.** (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.

3. **Lundberg, S. M. & Lee, S.-I.** (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

4. **Snoek, J., Larochelle, H., & Adams, R. P.** (2012). Practical Bayesian optimization of machine learning algorithms. *Advances in Neural Information Processing Systems*, 25, 2951-2959.

5. **Deb, K. et al.** (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197. [DOI: 10.1109/4235.996017](https://doi.org/10.1109/4235.996017)

6. **Frenkel, D. & Smit, B.** (2001). *Understanding Molecular Simulation: From Algorithms to Applications* (2nd ed.). Academic Press.

---

[← 前章：ナノ材料の基礎原理](chapter2-fundamentals.html) | [次章：実世界の応用とキャリア →](chapter4-real-world.html)
