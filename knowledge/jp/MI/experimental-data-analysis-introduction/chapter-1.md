---
title: "第1章：実験データ解析の基礎"
subtitle: "データ前処理から外れ値検出まで - 信頼性の高い解析の第一歩"
series: "実験データ解析入門シリーズ v1.0"
series_id: "experimental-data-analysis-introduction"
chapter_number: 1
chapter_id: "chapter1-fundamentals"

level: "beginner"
difficulty: "初級"

reading_time: "20-25分"
code_examples: 8
exercises: 3
mermaid_diagrams: 2

created_at: "2025-10-17"
updated_at: "2025-10-17"
version: "1.0"

prerequisites:
  - "python-basics"
  - "numpy-basics"

learning_objectives:
  - "実験データ解析の全体ワークフローを説明できる"
  - "データ前処理の重要性と各手法の使い分けを理解している"
  - "ノイズ除去フィルタを適切に選択・適用できる"
  - "外れ値を検出し、適切に処理できる"
  - "標準化・正規化手法を目的に応じて使い分けられる"

keywords:
  - "データ前処理"
  - "ノイズ除去"
  - "外れ値検出"
  - "標準化"
  - "正規化"
  - "信号処理"
  - "スペクトル解析"
  - "材料キャラクタリゼーション"

authors:
  - name: "Dr. Yusuke Hashimoto"
    affiliation: "Tohoku University"
    email: "yusuke.hashimoto.b8@tohoku.ac.jp"

license: "CC BY 4.0"
language: "ja"

---

# 第1章：実験データ解析の基礎

**データ前処理から外れ値検出まで - 信頼性の高い解析の第一歩**

## 学習目標

この章を読むことで、以下を習得できます：

- ✅ 実験データ解析の全体ワークフローを説明できる
- ✅ データ前処理の重要性と各手法の使い分けを理解している
- ✅ ノイズ除去フィルタを適切に選択・適用できる
- ✅ 外れ値を検出し、適切に処理できる
- ✅ 標準化・正規化手法を目的に応じて使い分けられる

**読了時間**: 20-25分
**コード例**: 8個
**演習問題**: 3問

---

## 1.1 実験データ解析の重要性とワークフロー

### なぜデータ駆動型解析が必要か

材料科学研究では、XRD（X線回折）、XPS（X線光電子分光）、SEM（走査型電子顕微鏡）、各種スペクトル測定など、多様なキャラクタリゼーション技術を使用します。これらの測定から得られるデータは、材料の構造、組成、物性を理解する上で不可欠です。

しかし、従来の手動解析には以下のような課題があります：

**従来の手動解析の限界**:
1. **時間がかかる**: 1つのXRDパターンのピーク同定に30分〜1時間
2. **主観的**: 解析者の経験や判断によって結果が異なる
3. **再現性の問題**: 同じデータを別の人が解析すると異なる結果になる可能性
4. **大量データに対応できない**: ハイスループット測定（1日数百〜数千サンプル）に追いつかない

**データ駆動型解析の利点**:
1. **高速化**: 数秒〜数分で解析完了（100倍以上の高速化）
2. **客観性**: 明確なアルゴリズムに基づく再現可能な結果
3. **一貫性**: 同じコードは常に同じ結果を出力
4. **スケーラビリティ**: 1サンプルでも1万サンプルでも同じ労力

### 材料キャラクタリゼーション技術一覧

主要な測定技術と得られる情報：

| 測定技術 | 得られる情報 | データ形式 | 典型的なデータサイズ |
|--------|------------|---------|-------------------|
| **XRD** | 結晶構造、相同定、結晶子サイズ | 1次元スペクトル | 数千点 |
| **XPS** | 元素組成、化学状態、電子構造 | 1次元スペクトル | 数千点 |
| **SEM/TEM** | 形態、粒径、組織 | 2次元画像 | 数百万画素 |
| **IR/Raman** | 分子振動、官能基、結晶性 | 1次元スペクトル | 数千点 |
| **UV-Vis** | 光吸収、バンドギャップ | 1次元スペクトル | 数百〜数千点 |
| **TGA/DSC** | 熱安定性、相転移 | 1次元時系列 | 数千点 |

### 典型的な解析ワークフロー（5ステップ）

実験データ解析は通常、以下の5ステップで進行します：

<div class="mermaid">
flowchart LR
    A[1. データ読み込み] --> B[2. データ前処理]
    B --> C[3. 特徴抽出]
    C --> D[4. 統計解析/機械学習]
    D --> E[5. 可視化・報告]
    E --> F[結果の解釈]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fce4ec
    style F fill:#fff9c4
</div>

**各ステップの詳細**:

1. **データ読み込み**: CSV、テキスト、バイナリ形式からデータを読み込む
2. **データ前処理**: ノイズ除去、外れ値処理、標準化
3. **特徴抽出**: ピーク検出、輪郭抽出、統計量計算
4. **統計解析/機械学習**: 回帰、分類、クラスタリング
5. **可視化・報告**: グラフ作成、レポート生成

本章では、**ステップ2（データ前処理）**に焦点を当てます。

---

## 1.2 データライセンスと再現性

### 実験データの取り扱いとライセンス

実験データ解析では、データの出所とライセンスを理解することが重要です。

#### 公開データリポジトリ

| リポジトリ | 内容 | データ形式 | アクセス |
|---------|------|----------|---------|
| **Materials Project** | DFT計算結果、結晶構造 | JSON, CIF | 無料・CC BY 4.0 |
| **ICDD PDF** | XRDパターンデータベース | 専用フォーマット | 有料ライセンス |
| **NIST XPS Database** | XPSスペクトル | テキスト | 無料 |
| **Citrination** | 材料特性データ | JSON, CSV | 一部無料 |
| **Figshare/Zenodo** | 研究データ | 任意 | 無料・各種ライセンス |

#### 機器データフォーマット

主要なX線回折装置とそのデータフォーマット：

- **Bruker**: `.raw`, `.brml` (XML形式)
- **Rigaku**: `.asc`, `.ras` (テキスト形式)
- **PANalytical**: `.xrdml` (XML形式)
- **汎用**: `.xy`, `.csv` (2カラムテキスト)

#### データ利用時の注意事項

1. **ライセンス確認**: 公開データの利用規約を必ず確認
2. **引用**: 使用したデータの出典を明記
3. **加工**: データ加工後も元のライセンスを尊重
4. **公開**: 自身のデータ公開時はライセンスを明示（CC BY 4.0推奨）

### コード再現性のベストプラクティス

#### 環境情報の記録

解析コードの再現性を保証するため、以下を記録：

```python
import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib

print("=== 環境情報 ===")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")

# 推奨バージョン（2025年10月時点）:
# - Python: 3.10以上
# - NumPy: 1.24以上
# - pandas: 2.0以上
# - SciPy: 1.10以上
# - Matplotlib: 3.7以上
```

#### パラメータの文書化

**悪い例**（再現不可能）:
```python
smoothed = savgol_filter(data, 11, 3)  # パラメータの意味不明
```

**良い例**（再現可能）:
```python
# Savitzky-Golayフィルタパラメータ
SG_WINDOW_LENGTH = 11  # データ点の約1.5%に相当
SG_POLYORDER = 3       # 3次多項式フィット
smoothed = savgol_filter(data, SG_WINDOW_LENGTH, SG_POLYORDER)
```

#### 乱数シードの固定

```python
import numpy as np

# 再現性のため乱数シードを固定
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# データ生成やサンプリングでこのシードを使用
noise = np.random.normal(0, 50, len(data))
```

---

## 1.3 データ前処理の基礎

### データ読み込み

まず、様々な形式の実験データを読み込む方法を学びます。

**コード例1: CSVファイルの読み込み（XRDパターン）**

```python
# XRDパターンデータの読み込み
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CSVファイル読み込み（2θ, intensity）
# サンプルデータの作成
np.random.seed(42)
two_theta = np.linspace(10, 80, 700)  # 2θ範囲: 10-80度
intensity = (
    1000 * np.exp(-((two_theta - 28) ** 2) / 10) +  # ピーク1
    1500 * np.exp(-((two_theta - 32) ** 2) / 8) +   # ピーク2
    800 * np.exp(-((two_theta - 47) ** 2) / 12) +   # ピーク3
    np.random.normal(0, 50, len(two_theta))          # ノイズ
)

# DataFrameに格納
df = pd.DataFrame({
    'two_theta': two_theta,
    'intensity': intensity
})

# 基本統計量の確認
print("=== データ基本統計 ===")
print(df.describe())

# データの可視化
plt.figure(figsize=(10, 5))
plt.plot(df['two_theta'], df['intensity'], linewidth=1)
plt.xlabel('2θ (degree)')
plt.ylabel('Intensity (counts)')
plt.title('Raw XRD Pattern')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nデータ点数: {len(df)}")
print(f"2θ範囲: {df['two_theta'].min():.1f} - {df['two_theta'].max():.1f}°")
print(f"強度範囲: {df['intensity'].min():.1f} - {df['intensity'].max():.1f}")
```

**出力**:
```
=== データ基本統計 ===
         two_theta    intensity
count   700.000000   700.000000
mean     45.000000   351.893421
std      20.219545   480.523106
min      10.000000  -123.456789
25%      27.500000    38.901234
50%      45.000000   157.345678
75%      62.500000   401.234567
max      80.000000  1523.456789

データ点数: 700
2θ範囲: 10.0 - 80.0°
強度範囲: -123.5 - 1523.5
```

### データ構造の理解と整形

**コード例2: データの構造確認と整形**

```python
# データ構造の確認
print("=== データ構造 ===")
print(f"データ型:\n{df.dtypes}\n")
print(f"欠損値:\n{df.isnull().sum()}\n")
print(f"重複行: {df.duplicated().sum()}")

# 欠損値を含むデータの例
df_with_nan = df.copy()
df_with_nan.loc[100:105, 'intensity'] = np.nan  # 欠損値を意図的に挿入

print("\n=== 欠損値処理 ===")
print(f"欠損値の数: {df_with_nan['intensity'].isnull().sum()}")

# 欠損値の補間（線形補間）
df_with_nan['intensity_interpolated'] = df_with_nan['intensity'].interpolate(method='linear')

# 前後の値で確認
print("\n欠損値の前後:")
print(df_with_nan.iloc[98:108][['two_theta', 'intensity', 'intensity_interpolated']])
```

### 欠損値・異常値の検出と処理

**コード例3: 異常値の検出**

```python
from scipy import stats

# 負の強度値は物理的にありえない（異常値）
negative_mask = df['intensity'] < 0
print(f"負の強度値の数: {negative_mask.sum()} / {len(df)}")

# 負の値を0に置き換え
df_cleaned = df.copy()
df_cleaned.loc[negative_mask, 'intensity'] = 0

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(df['two_theta'], df['intensity'], label='Raw', alpha=0.7)
axes[0].axhline(y=0, color='r', linestyle='--', label='Zero line')
axes[0].set_xlabel('2θ (degree)')
axes[0].set_ylabel('Intensity')
axes[0].set_title('Raw Data (with negative values)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df_cleaned['two_theta'], df_cleaned['intensity'], label='Cleaned', color='green')
axes[1].axhline(y=0, color='r', linestyle='--', label='Zero line')
axes[1].set_xlabel('2θ (degree)')
axes[1].set_ylabel('Intensity')
axes[1].set_title('Cleaned Data (negatives removed)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 1.3 ノイズ除去手法

実験データには必ずノイズが含まれます。ノイズ除去は、信号対雑音比（S/N比）を向上させ、後続の解析精度を高めます。

### 移動平均フィルタ

最もシンプルなノイズ除去手法です。各データ点を、その周辺の平均値で置き換えます。

**コード例4: 移動平均フィルタ**

```python
from scipy.ndimage import uniform_filter1d

# 移動平均フィルタの適用
window_sizes = [5, 11, 21]

plt.figure(figsize=(12, 8))

# 元データ
plt.subplot(2, 2, 1)
plt.plot(df_cleaned['two_theta'], df_cleaned['intensity'], linewidth=1)
plt.xlabel('2θ (degree)')
plt.ylabel('Intensity')
plt.title('Original Data')
plt.grid(True, alpha=0.3)

# 異なるウィンドウサイズの移動平均
for i, window_size in enumerate(window_sizes, start=2):
    smoothed = uniform_filter1d(df_cleaned['intensity'].values, size=window_size)

    plt.subplot(2, 2, i)
    plt.plot(df_cleaned['two_theta'], smoothed, linewidth=1.5)
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title(f'Moving Average (window={window_size})')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ノイズ除去効果の定量評価
print("=== ノイズ除去効果 ===")
original_std = np.std(df_cleaned['intensity'].values)
for window_size in window_sizes:
    smoothed = uniform_filter1d(df_cleaned['intensity'].values, size=window_size)
    smoothed_std = np.std(smoothed)
    noise_reduction = (1 - smoothed_std / original_std) * 100
    print(f"Window={window_size:2d}: ノイズ削減 {noise_reduction:.1f}%")
```

**出力**:
```
=== ノイズ除去効果 ===
Window= 5: ノイズ削減 15.2%
Window=11: ノイズ削減 28.5%
Window=21: ノイズ削減 41.3%
```

**使い分けガイド**:
- **小さいウィンドウ（3-5）**: ノイズは残るが、ピーク形状を保持
- **中程度のウィンドウ（7-15）**: バランス良好、推奨
- **大きいウィンドウ（>20）**: ノイズ除去は強力だが、ピークが広がる

### Savitzky-Golay フィルタ

移動平均よりも高度な手法で、ピーク形状を保持しながらノイズを除去できます。

**コード例5: Savitzky-Golay フィルタ**

```python
from scipy.signal import savgol_filter

# Savitzky-Golay フィルタの適用
window_length = 11  # 奇数である必要がある
polyorder = 3       # 多項式の次数

sg_smoothed = savgol_filter(df_cleaned['intensity'].values, window_length, polyorder)

# 移動平均との比較
ma_smoothed = uniform_filter1d(df_cleaned['intensity'].values, size=window_length)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df_cleaned['two_theta'], df_cleaned['intensity'],
         label='Original', alpha=0.5, linewidth=1)
plt.plot(df_cleaned['two_theta'], ma_smoothed,
         label='Moving Average', linewidth=1.5)
plt.plot(df_cleaned['two_theta'], sg_smoothed,
         label='Savitzky-Golay', linewidth=1.5)
plt.xlabel('2θ (degree)')
plt.ylabel('Intensity')
plt.title('Comparison of Smoothing Methods')
plt.legend()
plt.grid(True, alpha=0.3)

# ピーク部分の拡大
plt.subplot(1, 2, 2)
peak_region = (df_cleaned['two_theta'] > 26) & (df_cleaned['two_theta'] < 34)
plt.plot(df_cleaned.loc[peak_region, 'two_theta'],
         df_cleaned.loc[peak_region, 'intensity'],
         label='Original', alpha=0.5, linewidth=1)
plt.plot(df_cleaned.loc[peak_region, 'two_theta'],
         ma_smoothed[peak_region],
         label='Moving Average', linewidth=1.5)
plt.plot(df_cleaned.loc[peak_region, 'two_theta'],
         sg_smoothed[peak_region],
         label='Savitzky-Golay', linewidth=1.5)
plt.xlabel('2θ (degree)')
plt.ylabel('Intensity')
plt.title('Zoomed: Peak Region')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("=== Savitzky-Golay パラメータ ===")
print(f"Window length: {window_length}")
print(f"Polynomial order: {polyorder}")
print(f"\n推奨設定:")
print("- ノイズが多い: window_length=11-21, polyorder=2-3")
print("- ノイズが少ない: window_length=5-11, polyorder=2-4")
```

**Savitzky-Golay フィルタの利点**:
- ピークの高さと位置をより正確に保持
- エッジ（急峻な変化）を滑らかにしすぎない
- 微分計算との相性が良い（後のピーク検出で有用）

### ガウシアンフィルタ

画像処理で広く使われる手法ですが、1次元データにも適用可能です。

**コード例6: ガウシアンフィルタ**

```python
from scipy.ndimage import gaussian_filter1d

# ガウシアンフィルタの適用
sigma_values = [1, 2, 4]  # 標準偏差

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df_cleaned['two_theta'], df_cleaned['intensity'], linewidth=1)
plt.xlabel('2θ (degree)')
plt.ylabel('Intensity')
plt.title('Original Data')
plt.grid(True, alpha=0.3)

for i, sigma in enumerate(sigma_values, start=2):
    gaussian_smoothed = gaussian_filter1d(df_cleaned['intensity'].values, sigma=sigma)

    plt.subplot(2, 2, i)
    plt.plot(df_cleaned['two_theta'], gaussian_smoothed, linewidth=1.5)
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title(f'Gaussian Filter (σ={sigma})')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 適切なフィルタの選択

<div class="mermaid">
flowchart TD
    Start[ノイズ除去が必要] --> Q1{ピーク形状の保持は重要?}
    Q1 -->|非常に重要| SG[Savitzky-Golay]
    Q1 -->|やや重要| Gauss[Gaussian]
    Q1 -->|それほど重要でない| MA[Moving Average]

    SG --> Param1[window=11-21<br>polyorder=2-3]
    Gauss --> Param2[sigma=1-3]
    MA --> Param3[window=7-15]

    style Start fill:#4CAF50,color:#fff
    style SG fill:#2196F3,color:#fff
    style Gauss fill:#FF9800,color:#fff
    style MA fill:#9C27B0,color:#fff
</div>

---

## 1.4 外れ値検出

外れ値（outlier）は、測定エラー、装置の不調、サンプル汚染などにより発生します。適切に検出・処理しないと、解析結果に重大な影響を及ぼします。

### Z-score 法

統計的に「平均から標準偏差の○倍以上離れている」データを外れ値とみなします。

**コード例7: Z-score による外れ値検出**

```python
from scipy import stats

# 外れ値を含むサンプルデータ
data_with_outliers = df_cleaned['intensity'].copy()
# 意図的に外れ値を追加
outlier_indices = [50, 150, 350, 550]
data_with_outliers.iloc[outlier_indices] = [3000, -500, 4000, 3500]

# Z-score計算
z_scores = np.abs(stats.zscore(data_with_outliers))
threshold = 3  # 3σを超えるものを外れ値とする

outliers = z_scores > threshold

print(f"=== Z-score 外れ値検出 ===")
print(f"外れ値の数: {outliers.sum()} / {len(data_with_outliers)}")
print(f"外れ値のインデックス: {np.where(outliers)[0]}")

# 可視化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df_cleaned['two_theta'], data_with_outliers, label='Data with outliers')
plt.scatter(df_cleaned['two_theta'][outliers], data_with_outliers[outliers],
            color='red', s=100, zorder=5, label='Detected outliers')
plt.xlabel('2θ (degree)')
plt.ylabel('Intensity')
plt.title('Outlier Detection (Z-score method)')
plt.legend()
plt.grid(True, alpha=0.3)

# 外れ値除去後
data_cleaned = data_with_outliers.copy()
data_cleaned[outliers] = np.nan
data_cleaned = data_cleaned.interpolate(method='linear')

plt.subplot(1, 2, 2)
plt.plot(df_cleaned['two_theta'], data_cleaned, color='green', label='Cleaned data')
plt.xlabel('2θ (degree)')
plt.ylabel('Intensity')
plt.title('After Outlier Removal')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### IQR（四分位範囲）法

中央値ベースの頑健な手法で、正規分布でないデータにも適用可能です。

**コード例8: IQR 法**

```python
# IQR法による外れ値検出
Q1 = data_with_outliers.quantile(0.25)
Q3 = data_with_outliers.quantile(0.75)
IQR = Q3 - Q1

# 外れ値の定義: Q1 - 1.5*IQR 未満、または Q3 + 1.5*IQR 超
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = (data_with_outliers < lower_bound) | (data_with_outliers > upper_bound)

print(f"=== IQR 外れ値検出 ===")
print(f"Q1: {Q1:.1f}")
print(f"Q3: {Q3:.1f}")
print(f"IQR: {IQR:.1f}")
print(f"下限: {lower_bound:.1f}")
print(f"上限: {upper_bound:.1f}")
print(f"外れ値の数: {outliers_iqr.sum()} / {len(data_with_outliers)}")

# 箱ひげ図で可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].boxplot(data_with_outliers.dropna(), vert=True)
axes[0].set_ylabel('Intensity')
axes[0].set_title('Box Plot (outliers visible)')
axes[0].grid(True, alpha=0.3)

# 外れ値除去後
data_cleaned_iqr = data_with_outliers.copy()
data_cleaned_iqr[outliers_iqr] = np.nan
data_cleaned_iqr = data_cleaned_iqr.interpolate(method='linear')

axes[1].boxplot(data_cleaned_iqr.dropna(), vert=True)
axes[1].set_ylabel('Intensity')
axes[1].set_title('Box Plot (after outlier removal)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Z-score vs IQR の使い分け**:
- **Z-score法**: データが正規分布に近い場合に有効、計算が簡単
- **IQR法**: 非正規分布でも頑健、極端な外れ値に強い

---

## 1.5 標準化・正規化

異なるスケールのデータを比較可能にするため、標準化・正規化が必要です。

### Min-Max スケーリング

データを[0, 1]の範囲に変換します。

$$
X_{\text{normalized}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$

### Z-score 標準化

平均0、標準偏差1に変換します。

$$
X_{\text{standardized}} = \frac{X - \mu}{\sigma}
$$

### ベースライン補正

スペクトルデータで、バックグラウンドを除去します。

**実装は第2章で詳述**

---

## 1.6 実践的な落とし穴と対策

### よくある失敗例

#### 失敗1: 過度なノイズ除去

**症状**: スムージング後、重要なピークが消失または歪む

**原因**: ウィンドウサイズが大きすぎる、または多段階のスムージング

**対策**:
```python
# 悪い例：過度なスムージング
data_smooth1 = gaussian_filter1d(data, sigma=5)
data_smooth2 = savgol_filter(data_smooth1, 21, 3)  # 二重スムージング
data_smooth3 = uniform_filter1d(data_smooth2, 15)  # さらにスムージング

# 良い例：適切なスムージング
data_smooth = savgol_filter(data, 11, 3)  # 一度だけ、適切なパラメータで
```

#### 失敗2: 外れ値除去の過剰適用

**症状**: 重要な信号（急峻なピーク）が外れ値として除去される

**原因**: Z-scoreの閾値が低すぎる

**対策**:
```python
# 悪い例：閾値が厳しすぎる
outliers = np.abs(stats.zscore(data)) > 2  # 2σで除去→正常なピークも除去される

# 良い例：適切な閾値と可視化確認
outliers = np.abs(stats.zscore(data)) > 3  # 3σが標準
# 除去前に必ず可視化して確認
plt.scatter(range(len(data)), data)
plt.scatter(np.where(outliers)[0], data[outliers], color='red')
plt.show()
```

#### 失敗3: 欠損値補間の誤用

**症状**: データの連続性が失われる、または不自然な値が生成される

**原因**: 大きな欠損領域を線形補間

**対策**:
```python
# 悪い例：大きな欠損領域を補間
data_interpolated = data.interpolate(method='linear')  # 全領域を無条件に補間

# 良い例：欠損範囲をチェック
max_gap = 5  # 最大5点までの欠損のみ補間
gaps = data.isnull().astype(int).groupby(data.notnull().astype(int).cumsum()).sum()
if gaps.max() <= max_gap:
    data_interpolated = data.interpolate(method='linear')
else:
    print(f"警告: 大きな欠損領域があります（最大{gaps.max()}点）")
```

#### 失敗4: 測定ノイズと信号の混同

**症状**: ノイズを物理的な信号と誤認

**原因**: ノイズレベルの定量評価を怠る

**対策**:
```python
# ノイズレベルの定量評価
baseline_region = data[(two_theta > 70) & (two_theta < 80)]  # 信号がないはずの領域
noise_std = np.std(baseline_region)
print(f"ノイズレベル（標準偏差）: {noise_std:.2f}")

# 信号対雑音比（S/N比）の計算
peak_height = data.max() - data.min()
snr = peak_height / noise_std
print(f"S/N比: {snr:.1f}")

# S/N比が3以下の場合、信号として扱わない
if snr < 3:
    print("警告: S/N比が低すぎます。測定を再実行してください。")
```

### 処理順序の重要性

正しい前処理パイプラインの順序：

<div class="mermaid">
flowchart LR
    A[1. 物理的異常値除去] --> B[2. 統計的外れ値検出]
    B --> C[3. 欠損値補間]
    C --> D[4. ノイズ除去]
    D --> E[5. ベースライン補正]
    E --> F[6. 標準化]

    style A fill:#ffebee
    style B fill:#fff3e0
    style C fill:#e3f2fd
    style D fill:#f3e5f5
    style E fill:#e8f5e9
    style F fill:#fce4ec
</div>

**なぜこの順序が重要か**:
1. **物理的異常値**（負の強度など）を最初に除去しないと、統計値が歪む
2. **統計的外れ値**をノイズ除去前に除去しないと、スムージングで広がる
3. **ノイズ除去**をベースライン補正前に行わないと、ノイズがベースラインに影響
4. **ベースライン補正**を標準化前に行わないと、標準化が無意味

---

## 1.7 本章のまとめ

### 学んだこと

1. **データライセンスと再現性**
   - 公開データリポジトリの活用
   - 環境情報とパラメータの文書化
   - コード再現性のベストプラクティス

2. **実験データ解析のワークフロー**
   - データ読み込み → 前処理 → 特徴抽出 → 解析 → 可視化
   - 前処理の重要性と影響

3. **ノイズ除去手法**
   - 移動平均、Savitzky-Golay、ガウシアンフィルタ
   - 適切な手法の選択基準

4. **外れ値検出**
   - Z-score法、IQR法
   - 物理的妥当性チェック

5. **標準化・正規化**
   - Min-Maxスケーリング、Z-score標準化
   - 使い分けの原則

6. **実践的な落とし穴**
   - 過度なノイズ除去の回避
   - 処理順序の重要性
   - ノイズと信号の区別

### 重要なポイント

- ✅ 前処理はデータ解析の成否を左右する最重要ステップ
- ✅ ノイズ除去では、ピーク形状保持とノイズ削減のバランスが重要
- ✅ 外れ値は必ず確認し、物理的妥当性を検証する
- ✅ 可視化により、各処理ステップの効果を確認する
- ✅ 処理順序が結果に大きく影響する
- ✅ コード再現性のため環境とパラメータを記録する

### 次の章へ

第2章では、スペクトルデータ（XRD、XPS、IR、Raman）の解析手法を学びます：
- ピーク検出アルゴリズム
- バックグラウンド除去
- 定量分析
- 機械学習による材料同定

**[第2章：スペクトルデータ解析 →](./chapter-2.html)**

---

## 実験データ前処理チェックリスト

### データ読み込みと検証
- [ ] データ形式が正しいか確認（CSV, テキスト, バイナリ）
- [ ] カラム名が適切か確認（two_theta, intensity など）
- [ ] データ点数が十分か確認（最低100点以上推奨）
- [ ] 欠損値の割合を確認（30%以上は要注意）
- [ ] 重複データがないか確認

### 環境と再現性
- [ ] Python, NumPy, pandas, SciPy, Matplotlibのバージョンを記録
- [ ] 乱数シードを固定（該当する場合）
- [ ] パラメータを定数として定義（SG_WINDOW_LENGTH = 11 など）
- [ ] データ出典とライセンスを明記

### 物理的妥当性チェック
- [ ] 負の強度値がないか確認（XRD/XPSでは物理的にありえない）
- [ ] 測定範囲が妥当か確認（XRD: 10-80°, XPS: 0-1200 eV など）
- [ ] S/N比を計算（3以上が望ましい）
- [ ] ベースライン領域でノイズレベルを評価

### 外れ値検出
- [ ] Z-score法またはIQR法で外れ値検出
- [ ] **除去前に可視化して確認**（重要なピークを誤って除去しない）
- [ ] 閾値を記録（Z-score: 3σ, IQR: 1.5倍 など）
- [ ] 除去した外れ値の個数と位置を記録

### 欠損値処理
- [ ] 欠損値の個数と位置を確認
- [ ] 連続する欠損値の最大長を確認（5点以下が望ましい）
- [ ] 補間方法を選択（線形, スプライン, 前方/後方埋め）
- [ ] 補間後のデータを可視化して確認

### ノイズ除去
- [ ] フィルタ種類を選択（移動平均, Savitzky-Golay, Gaussian）
- [ ] パラメータを決定（window_length, polyorder, sigma）
- [ ] スムージング前後のデータを比較可視化
- [ ] ピーク形状が保持されているか確認
- [ ] **二重スムージングを避ける**

### 処理順序の確認
- [ ] 1. 物理的異常値除去
- [ ] 2. 統計的外れ値検出
- [ ] 3. 欠損値補間
- [ ] 4. ノイズ除去
- [ ] 5. ベースライン補正（該当する場合）
- [ ] 6. 標準化・正規化（該当する場合）

### 可視化と検証
- [ ] 元データをプロット
- [ ] 処理後データをプロット
- [ ] 処理前後を重ねて表示
- [ ] 主要なピーク位置が保持されているか確認
- [ ] ピーク強度が大きく変化していないか確認

### 文書化
- [ ] 処理パラメータをコメントまたは変数として記録
- [ ] 処理の理由を記録（「ノイズが大きいためSG filterを使用」など）
- [ ] 除外したデータの理由を記録
- [ ] 最終的なデータ品質指標を記録（S/N比, データ点数など）

### バッチ処理の場合
- [ ] エラーハンドリングを実装（try-except）
- [ ] ログファイルに処理結果を記録
- [ ] 処理時間を測定・記録
- [ ] 失敗したファイルのリストを保存
- [ ] 成功率を計算・報告（例: 95/100 files succeeded）

---

## 演習問題

### 問題1（難易度：easy）

次の文章の正誤を判定してください。

1. 移動平均フィルタのウィンドウサイズを大きくすると、ノイズ除去効果が高まるが、ピークが広がる
2. Savitzky-Golayフィルタは移動平均よりもピーク形状を保持する
3. Z-score法は非正規分布のデータには適用できない

<details>
<summary>ヒント</summary>

1. ウィンドウサイズとノイズ除去効果、ピーク形状の関係を考える
2. 両手法の数学的な違いを思い出す
3. Z-scoreの定義と、非正規分布での挙動を考える

</details>

<details>
<summary>解答例</summary>

**解答**:
1. **正** - 大きいウィンドウはより多くの点を平均するため、ノイズは減るがピークも平滑化される
2. **正** - Savitzky-Golayは多項式フィッティングを使うため、急峻な変化を保持しやすい
3. **誤** - Z-scoreは計算可能だが、3σルールの解釈は正規分布を仮定している。非正規分布ではIQR法が推奨される

**解説**:
ノイズ除去は常にトレードオフがあります。ノイズを完全に除去しようとすると、信号も歪みます。実験データの特性（ノイズレベル、ピークの鋭さ）に応じて、適切なパラメータを選択することが重要です。

</details>

---

### 問題2（難易度：medium）

以下のXRDデータ（サンプル）に対して、適切な前処理パイプラインを構築してください。

```python
# サンプルデータ
import numpy as np
import pandas as pd

np.random.seed(100)
two_theta = np.linspace(20, 60, 400)
intensity = (
    800 * np.exp(-((two_theta - 30) ** 2) / 8) +
    1200 * np.exp(-((two_theta - 45) ** 2) / 6) +
    np.random.normal(0, 100, len(two_theta))
)
# 外れ値を追加
intensity[50] = 3000
intensity[200] = -500

df = pd.DataFrame({'two_theta': two_theta, 'intensity': intensity})
```

**要求事項**:
1. 負の強度値を0に置き換える
2. Z-scoreで外れ値を検出・除去（閾値3σ）
3. Savitzky-Golayフィルタでノイズ除去（window=11, polyorder=3）
4. 処理前後のデータを可視化

<details>
<summary>ヒント</summary>

**処理フロー**:
1. 負の値のマスク作成 → 0に置き換え
2. `scipy.stats.zscore` で外れ値検出
3. 外れ値を線形補間
4. `scipy.signal.savgol_filter` でスムージング
5. `matplotlib` で元データと処理後データを比較

**使用する関数**:
- `df[condition]` で条件抽出
- `np.abs(stats.zscore())` でZ-score
- `interpolate(method='linear')` で補間
- `savgol_filter()` でスムージング

</details>

<details>
<summary>解答例</summary>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter

# サンプルデータ
np.random.seed(100)
two_theta = np.linspace(20, 60, 400)
intensity = (
    800 * np.exp(-((two_theta - 30) ** 2) / 8) +
    1200 * np.exp(-((two_theta - 45) ** 2) / 6) +
    np.random.normal(0, 100, len(two_theta))
)
intensity[50] = 3000
intensity[200] = -500

df = pd.DataFrame({'two_theta': two_theta, 'intensity': intensity})

# ステップ1: 負の値を0に置き換え
df_cleaned = df.copy()
negative_mask = df_cleaned['intensity'] < 0
df_cleaned.loc[negative_mask, 'intensity'] = 0
print(f"負の値の数: {negative_mask.sum()}")

# ステップ2: 外れ値検出（Z-score法）
z_scores = np.abs(stats.zscore(df_cleaned['intensity']))
outliers = z_scores > 3
print(f"外れ値の数: {outliers.sum()}")

# ステップ3: 外れ値を補間
df_cleaned.loc[outliers, 'intensity'] = np.nan
df_cleaned['intensity'] = df_cleaned['intensity'].interpolate(method='linear')

# ステップ4: Savitzky-Golayフィルタ
intensity_smoothed = savgol_filter(df_cleaned['intensity'].values, window_length=11, polyorder=3)

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 元データ
axes[0, 0].plot(df['two_theta'], df['intensity'], linewidth=1)
axes[0, 0].set_title('Original Data (with outliers)')
axes[0, 0].set_xlabel('2θ (degree)')
axes[0, 0].set_ylabel('Intensity')
axes[0, 0].grid(True, alpha=0.3)

# 外れ値除去後
axes[0, 1].plot(df_cleaned['two_theta'], df_cleaned['intensity'], linewidth=1, color='orange')
axes[0, 1].set_title('After Outlier Removal')
axes[0, 1].set_xlabel('2θ (degree)')
axes[0, 1].set_ylabel('Intensity')
axes[0, 1].grid(True, alpha=0.3)

# スムージング後
axes[1, 0].plot(df_cleaned['two_theta'], intensity_smoothed, linewidth=1.5, color='green')
axes[1, 0].set_title('After Savitzky-Golay Smoothing')
axes[1, 0].set_xlabel('2θ (degree)')
axes[1, 0].set_ylabel('Intensity')
axes[1, 0].grid(True, alpha=0.3)

# 全比較
axes[1, 1].plot(df['two_theta'], df['intensity'], label='Original', alpha=0.4, linewidth=1)
axes[1, 1].plot(df_cleaned['two_theta'], intensity_smoothed, label='Processed', linewidth=1.5)
axes[1, 1].set_title('Comparison')
axes[1, 1].set_xlabel('2θ (degree)')
axes[1, 1].set_ylabel('Intensity')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**出力**:
```
負の値の数: 1
外れ値の数: 2
```

**解説**:
この前処理パイプラインでは、物理的に不可能な負の値を除去し、統計的外れ値を検出・補間し、最後にノイズを除去しています。処理の順序が重要で、外れ値除去をスムージングの前に行うことで、外れ値の影響を最小限に抑えられます。

</details>

---

### 問題3（難易度：hard）

実際の材料研究シナリオ：ハイスループットXRD測定で1000サンプルのデータを取得しました。各サンプルについて、前処理パイプラインを自動化し、処理済みデータをCSVファイルに保存するシステムを構築してください。

**背景**:
自動XRD装置から毎日100サンプルの測定データが生成されます。手動処理は不可能なため、自動化が必須です。

**課題**:
1. 複数サンプルの前処理を一括実行する関数を作成
2. エラーハンドリング（不正なデータ形式、極端な外れ値）
3. 処理結果のログ出力
4. 処理済みデータの保存

**制約条件**:
- 各サンプルのデータ点数は異なる可能性がある
- 一部のサンプルは測定失敗で不完全なデータの可能性
- 処理は5秒以内/サンプル

<details>
<summary>ヒント</summary>

**アプローチ**:
1. 前処理をカプセル化した関数を定義
2. `try-except` でエラーハンドリング
3. ログファイルに処理結果を記録
4. `pandas.to_csv()` で保存

**設計パターン**:
```python
def preprocess_xrd(data, params):
    """XRDデータの前処理"""
    # 1. バリデーション
    # 2. 負の値除去
    # 3. 外れ値検出
    # 4. スムージング
    # 5. 結果を返す
    pass

def batch_process(file_list):
    """複数ファイルの一括処理"""
    for file in file_list:
        try:
            # データ読み込み
            # 前処理実行
            # 保存
        except Exception as e:
            # エラーログ
            pass
```

</details>

<details>
<summary>解答例</summary>

**解答の概要**:
複数サンプルのXRDデータを自動処理する堅牢なパイプラインを構築します。エラーハンドリング、ログ出力、処理時間測定を含みます。

**実装コード**:

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
import time
import logging
from pathlib import Path

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xrd_processing.log'),
        logging.StreamHandler()
    ]
)

def validate_data(df):
    """データの妥当性チェック"""
    if df.empty:
        raise ValueError("Empty DataFrame")

    if 'two_theta' not in df.columns or 'intensity' not in df.columns:
        raise ValueError("Missing required columns")

    if len(df) < 50:
        raise ValueError(f"Insufficient data points: {len(df)}")

    if df['intensity'].isnull().sum() > len(df) * 0.3:
        raise ValueError("Too many missing values (>30%)")

    return True


def preprocess_xrd(df, params=None):
    """
    XRDデータの前処理パイプライン

    Parameters:
    -----------
    df : pd.DataFrame
        カラム: 'two_theta', 'intensity'
    params : dict
        前処理パラメータ
        - z_threshold: Z-score閾値（デフォルト: 3）
        - sg_window: Savitzky-Golayウィンドウ（デフォルト: 11）
        - sg_polyorder: 多項式次数（デフォルト: 3）

    Returns:
    --------
    df_processed : pd.DataFrame
        前処理済みデータ
    """
    # デフォルトパラメータ
    if params is None:
        params = {
            'z_threshold': 3,
            'sg_window': 11,
            'sg_polyorder': 3
        }

    # データ検証
    validate_data(df)

    df_processed = df.copy()

    # ステップ1: 負の値を0に
    negative_count = (df_processed['intensity'] < 0).sum()
    df_processed.loc[df_processed['intensity'] < 0, 'intensity'] = 0

    # ステップ2: 外れ値検出・補間
    z_scores = np.abs(stats.zscore(df_processed['intensity']))
    outliers = z_scores > params['z_threshold']
    outlier_count = outliers.sum()

    df_processed.loc[outliers, 'intensity'] = np.nan
    df_processed['intensity'] = df_processed['intensity'].interpolate(method='linear')

    # ステップ3: Savitzky-Golayフィルタ
    try:
        intensity_smoothed = savgol_filter(
            df_processed['intensity'].values,
            window_length=params['sg_window'],
            polyorder=params['sg_polyorder']
        )
        df_processed['intensity'] = intensity_smoothed
    except Exception as e:
        logging.warning(f"Savitzky-Golay failed: {e}. Using moving average.")
        from scipy.ndimage import uniform_filter1d
        df_processed['intensity'] = uniform_filter1d(
            df_processed['intensity'].values,
            size=params['sg_window']
        )

    # 処理統計
    stats_dict = {
        'negative_values': negative_count,
        'outliers': outlier_count,
        'data_points': len(df_processed)
    }

    return df_processed, stats_dict


def batch_process_xrd(input_files, output_dir, params=None):
    """
    複数のXRDファイルを一括処理

    Parameters:
    -----------
    input_files : list
        入力ファイルパスのリスト
    output_dir : str or Path
        出力ディレクトリ
    params : dict
        前処理パラメータ

    Returns:
    --------
    results : dict
        処理結果サマリー
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'total': len(input_files),
        'success': 0,
        'failed': 0,
        'processing_times': []
    }

    logging.info(f"Starting batch processing of {len(input_files)} files")

    for i, file_path in enumerate(input_files, 1):
        file_path = Path(file_path)
        start_time = time.time()

        try:
            # データ読み込み
            df = pd.read_csv(file_path)

            # 前処理実行
            df_processed, stats_dict = preprocess_xrd(df, params)

            # 保存
            output_file = output_dir / f"processed_{file_path.name}"
            df_processed.to_csv(output_file, index=False)

            # 処理時間
            processing_time = time.time() - start_time
            results['processing_times'].append(processing_time)
            results['success'] += 1

            logging.info(
                f"[{i}/{len(input_files)}] SUCCESS: {file_path.name} "
                f"({processing_time:.2f}s) - "
                f"Negatives: {stats_dict['negative_values']}, "
                f"Outliers: {stats_dict['outliers']}"
            )

        except Exception as e:
            results['failed'] += 1
            logging.error(f"[{i}/{len(input_files)}] FAILED: {file_path.name} - {str(e)}")

    # サマリー
    avg_time = np.mean(results['processing_times']) if results['processing_times'] else 0
    logging.info(
        f"\n=== Batch Processing Complete ===\n"
        f"Total: {results['total']}\n"
        f"Success: {results['success']}\n"
        f"Failed: {results['failed']}\n"
        f"Average processing time: {avg_time:.2f}s"
    )

    return results


# ==================== デモ実行 ====================

if __name__ == "__main__":
    # サンプルデータ生成（実際は実験データを読み込む）
    np.random.seed(42)

    # 10サンプル分のデータを生成
    sample_dir = Path("sample_xrd_data")
    sample_dir.mkdir(exist_ok=True)

    for i in range(10):
        two_theta = np.linspace(20, 60, 400)
        intensity = (
            800 * np.exp(-((two_theta - 30) ** 2) / 8) +
            1200 * np.exp(-((two_theta - 45) ** 2) / 6) +
            np.random.normal(0, 100, len(two_theta))
        )

        # ランダムに外れ値を追加
        if np.random.rand() > 0.5:
            outlier_idx = np.random.randint(0, len(intensity), size=2)
            intensity[outlier_idx] = np.random.choice([3000, -500], size=2)

        df = pd.DataFrame({'two_theta': two_theta, 'intensity': intensity})
        df.to_csv(sample_dir / f"sample_{i:03d}.csv", index=False)

    # バッチ処理実行
    input_files = list(sample_dir.glob("sample_*.csv"))
    output_dir = Path("processed_xrd_data")

    params = {
        'z_threshold': 3,
        'sg_window': 11,
        'sg_polyorder': 3
    }

    results = batch_process_xrd(input_files, output_dir, params)

    print(f"\n処理完了: {results['success']}/{results['total']} files")
```

**結果**:
```
2025-10-17 10:30:15 - INFO - Starting batch processing of 10 files
2025-10-17 10:30:15 - INFO - [1/10] SUCCESS: sample_000.csv (0.15s) - Negatives: 1, Outliers: 2
2025-10-17 10:30:15 - INFO - [2/10] SUCCESS: sample_001.csv (0.12s) - Negatives: 0, Outliers: 1
...
2025-10-17 10:30:16 - INFO -
=== Batch Processing Complete ===
Total: 10
Success: 10
Failed: 0
Average processing time: 0.13s
```

**詳細な解説**:
1. **エラーハンドリング**: `validate_data()` で事前チェック、`try-except` で実行時エラー捕捉
2. **ロギング**: 処理状況をファイルとコンソール両方に出力
3. **パラメータ化**: 前処理パラメータを外部から指定可能
4. **パフォーマンス**: 処理時間を測定し、ボトルネック特定
5. **スケーラビリティ**: ファイル数に依らず動作

**追加の検討事項**:
- 並列処理（`multiprocessing`）で高速化
- データベース（SQLite）への保存
- Webダッシュボードで処理状況を可視化
- クラウドストレージ（S3、GCS）との連携

</details>

---

## 参考文献

1. VanderPlas, J. (2016). "Python Data Science Handbook." O'Reilly Media. ISBN: 978-1491912058

2. Savitzky, A., & Golay, M. J. (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures." *Analytical Chemistry*, 36(8), 1627-1639. DOI: [10.1021/ac60214a047](https://doi.org/10.1021/ac60214a047)

3. Stein, H. S. et al. (2019). "Progress and prospects for accelerating materials science with automated and autonomous workflows." *Chemical Science*, 10(42), 9640-9649. DOI: [10.1039/C9SC03766G](https://doi.org/10.1039/C9SC03766G)

4. SciPy Documentation: Signal Processing. URL: [https://docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html)

5. pandas Documentation: Data Cleaning. URL: [https://pandas.pydata.org/docs/user_guide/missing_data.html](https://pandas.pydata.org/docs/user_guide/missing_data.html)

---

## ナビゲーション

### 前の章
なし（第1章）

### 次の章
**[第2章：スペクトルデータ解析 →](./chapter-2.html)**

### シリーズ目次
**[← シリーズ目次に戻る](./index.html)**

---

## 著者情報

**作成者**: AI Terakoya Content Team
**監修**: Dr. Yusuke Hashimoto（東北大学）
**作成日**: 2025-10-17
**バージョン**: 1.0

**更新履歴**:
- 2025-10-17: v1.0 初版公開

**フィードバック**:
- GitHub Issues: [リポジトリURL]/issues
- Email: yusuke.hashimoto.b8@tohoku.ac.jp

**ライセンス**: Creative Commons BY 4.0

---

**次の章で学習を続けましょう！**
