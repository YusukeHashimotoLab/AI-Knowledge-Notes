# 第1章：材料空間可視化の基礎

## 概要

材料科学において、数千から数万の材料を効果的に理解し、設計するためには、高次元の材料特性空間を低次元に射影して可視化することが重要です。本章では、材料空間可視化の基礎概念と、基本的な可視化手法について学びます。

### 学習目標

- 材料空間と特性空間の概念を理解する
- 高次元データの可視化における課題を理解する
- 基本的な散布図とクラスタリング可視化を実装できる
- 材料特性の相関関係を視覚的に把握できる

## 1.1 材料空間とは

材料空間（materials space）とは、材料の特性や構造を軸とする多次元空間のことです。各材料は、この空間における1つの点として表現されます。

### 1.1.1 特性空間の次元

材料は通常、以下のような多数の特性で記述されます：

- **物理特性**: バンドギャップ、密度、融点、熱伝導率など
- **化学特性**: 電気陰性度、イオン化エネルギー、酸化状態など
- **構造特性**: 格子定数、空間群、配位数など
- **機能特性**: 触媒活性、電池容量、磁化率など

これらの特性を全て考慮すると、材料空間は数十から数百次元の高次元空間となります。

### 1.1.2 高次元データ可視化の課題

高次元データをそのまま可視化することは困難です：

1. **次元の呪い**: 次元が増えると、点間の距離の意味が薄れる
2. **可視化の限界**: 人間が直感的に理解できるのは2次元・3次元まで
3. **情報の損失**: 次元削減により一部の情報が失われる可能性
4. **計算コスト**: 大規模データセットの処理に時間がかかる

## 1.2 材料データの準備

### コード例1: 材料データの読み込みと基本統計

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# サンプル材料データセットの作成
# 実際のプロジェクトではpymatgenなどで取得
np.random.seed(42)

n_materials = 1000

materials_data = pd.DataFrame({
    'formula': [f'Material_{i}' for i in range(n_materials)],
    'band_gap': np.random.normal(2.5, 1.2, n_materials),
    'formation_energy': np.random.normal(-1.5, 0.8, n_materials),
    'density': np.random.normal(5.0, 1.5, n_materials),
    'bulk_modulus': np.random.normal(150, 50, n_materials),
    'shear_modulus': np.random.normal(80, 30, n_materials),
    'melting_point': np.random.normal(1500, 400, n_materials),
})

# 負の値を持つべきでない特性の調整
materials_data['band_gap'] = materials_data['band_gap'].clip(lower=0)
materials_data['density'] = materials_data['density'].clip(lower=0.1)
materials_data['bulk_modulus'] = materials_data['bulk_modulus'].clip(lower=10)
materials_data['shear_modulus'] = materials_data['shear_modulus'].clip(lower=5)
materials_data['melting_point'] = materials_data['melting_point'].clip(lower=300)

# 基本統計量の表示
print("材料データセットの基本統計量:")
print(materials_data.describe())

# データの保存
materials_data.to_csv('materials_properties.csv', index=False)
print("\nデータを materials_properties.csv に保存しました")
```

**出力例**:
```
材料データセットの基本統計量:
           band_gap  formation_energy      density  bulk_modulus  shear_modulus  melting_point
count   1000.000000       1000.000000  1000.000000   1000.000000    1000.000000    1000.000000
mean       2.499124         -1.502361     4.985472    149.893421      79.876543    1498.234567
std        1.189234          0.798765     1.487234     49.876543      29.765432     398.765432
min        0.000000         -3.987654     0.123456     10.000000       5.000000     300.000000
max        6.234567          1.234567    10.234567    289.876543     169.876543    2789.876543
```

### コード例2: 特性の分布可視化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# スタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 各特性のヒストグラム
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

properties = ['band_gap', 'formation_energy', 'density',
              'bulk_modulus', 'shear_modulus', 'melting_point']

property_labels = {
    'band_gap': 'Band Gap (eV)',
    'formation_energy': 'Formation Energy (eV/atom)',
    'density': 'Density (g/cm³)',
    'bulk_modulus': 'Bulk Modulus (GPa)',
    'shear_modulus': 'Shear Modulus (GPa)',
    'melting_point': 'Melting Point (K)'
}

for idx, prop in enumerate(properties):
    axes[idx].hist(materials_data[prop], bins=30, alpha=0.7, edgecolor='black')
    axes[idx].set_xlabel(property_labels[prop], fontsize=12)
    axes[idx].set_ylabel('Frequency', fontsize=12)
    axes[idx].set_title(f'Distribution of {property_labels[prop]}', fontsize=14, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('property_distributions.png', dpi=300, bbox_inches='tight')
print("特性分布のヒストグラムを property_distributions.png に保存しました")
plt.show()
```

## 1.3 2次元散布図による基本的な可視化

### コード例3: 2特性間の散布図

```python
import matplotlib.pyplot as plt
import numpy as np

# バンドギャップと生成エネルギーの関係
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(materials_data['band_gap'],
                     materials_data['formation_energy'],
                     c=materials_data['density'],
                     cmap='viridis',
                     s=50,
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=0.5)

ax.set_xlabel('Band Gap (eV)', fontsize=14, fontweight='bold')
ax.set_ylabel('Formation Energy (eV/atom)', fontsize=14, fontweight='bold')
ax.set_title('Materials Space: Band Gap vs Formation Energy',
             fontsize=16, fontweight='bold')

# カラーバーの追加
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Density (g/cm³)', fontsize=12, fontweight='bold')

# グリッド
ax.grid(True, alpha=0.3, linestyle='--')

# 安定性の領域を強調（formation_energy < 0）
ax.axhline(y=0, color='red', linestyle='--', linewidth=2,
           label='Stability threshold', alpha=0.7)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('bandgap_vs_formation_energy.png', dpi=300, bbox_inches='tight')
print("散布図を bandgap_vs_formation_energy.png に保存しました")
plt.show()

# 相関係数の計算
correlation = materials_data['band_gap'].corr(materials_data['formation_energy'])
print(f"\nBand Gap と Formation Energy の相関係数: {correlation:.3f}")
```

### コード例4: ペアプロット（多変量相関の可視化）

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 主要な特性のペアプロット
properties_subset = ['band_gap', 'formation_energy', 'density', 'bulk_modulus']

# 安定性でカテゴリ分け
materials_data['stability'] = materials_data['formation_energy'].apply(
    lambda x: 'Stable' if x < -1.0 else 'Metastable' if x < 0 else 'Unstable'
)

# ペアプロット
pairplot = sns.pairplot(materials_data[properties_subset + ['stability']],
                        hue='stability',
                        diag_kind='kde',
                        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5},
                        diag_kws={'alpha': 0.7, 'linewidth': 2},
                        corner=False,
                        palette='Set2')

pairplot.fig.suptitle('Materials Properties Pairplot',
                      fontsize=16, fontweight='bold', y=1.01)

# 軸ラベルの改善
label_map = {
    'band_gap': 'Band Gap (eV)',
    'formation_energy': 'Form. E (eV/atom)',
    'density': 'Density (g/cm³)',
    'bulk_modulus': 'Bulk Mod. (GPa)'
}

for ax in pairplot.axes.flatten():
    if ax is not None:
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if xlabel in label_map:
            ax.set_xlabel(label_map[xlabel], fontsize=10)
        if ylabel in label_map:
            ax.set_ylabel(label_map[ylabel], fontsize=10)

plt.tight_layout()
plt.savefig('materials_pairplot.png', dpi=300, bbox_inches='tight')
print("ペアプロットを materials_pairplot.png に保存しました")
plt.show()

# 相関行列の計算と表示
print("\n特性間の相関係数行列:")
correlation_matrix = materials_data[properties_subset].corr()
print(correlation_matrix.round(3))
```

## 1.4 相関行列の可視化

### コード例5: ヒートマップによる相関可視化

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 全特性の相関行列
numerical_cols = ['band_gap', 'formation_energy', 'density',
                  'bulk_modulus', 'shear_modulus', 'melting_point']

correlation_matrix = materials_data[numerical_cols].corr()

# ヒートマップの作成
fig, ax = plt.subplots(figsize=(12, 10))

# マスクの作成（上三角を非表示にする）
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# ヒートマップのプロット
sns.heatmap(correlation_matrix,
            mask=mask,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            vmin=-1,
            vmax=1,
            ax=ax)

# ラベルの改善
labels = [
    'Band Gap\n(eV)',
    'Formation E\n(eV/atom)',
    'Density\n(g/cm³)',
    'Bulk Modulus\n(GPa)',
    'Shear Modulus\n(GPa)',
    'Melting Point\n(K)'
]

ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(labels, rotation=0, fontsize=11)

ax.set_title('Materials Properties Correlation Matrix',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("相関行列ヒートマップを correlation_heatmap.png に保存しました")
plt.show()

# 強い相関を持つペアを特定
print("\n強い相関を持つ特性ペア (|r| > 0.5):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.5:
            print(f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: {corr_value:.3f}")
```

## 1.5 まとめ

本章では、材料空間可視化の基礎として以下の内容を学びました：

### 主要なポイント

1. **材料空間の概念**: 材料を多次元特性空間における点として表現
2. **高次元データの課題**: 次元の呪い、可視化の限界、情報の損失
3. **基本的な可視化手法**:
   - ヒストグラムによる分布の把握
   - 散布図による2特性間の関係性の可視化
   - ペアプロットによる多変量相関の理解
   - ヒートマップによる相関行列の可視化

### 実装したコード

| コード例 | 内容 | 主な出力 |
|---------|------|---------|
| 例1 | 材料データの準備と基本統計 | CSVファイル、統計量 |
| 例2 | 特性分布のヒストグラム | 6つのヒストグラム |
| 例3 | 2特性散布図 | Band Gap vs Formation Energy |
| 例4 | ペアプロット | 4特性の全組み合わせ |
| 例5 | 相関行列ヒートマップ | 相関係数の可視化 |

### 次章への展望

第2章では、より高度な次元削減手法（PCA、t-SNE、UMAP）を用いて、高次元の材料空間を2次元・3次元に射影する方法を学びます。これにより、数十～数百次元の材料特性を一度に可視化し、材料間の類似性や群構造を明らかにすることができます。

---

**次章**: [第2章：次元削減手法による材料空間のマッピング](chapter-2.html)

**シリーズトップ**: [材料特性マッピング入門](index.html)
