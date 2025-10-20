---
title: "Chapter 3: 特徴量抽出"
chapter_number: 3
series: 特徴量エンジニアリング入門
difficulty: 初級〜中級
reading_time: 25-30分
tags: [PCA, LDA, t-SNE, UMAP, 次元削減]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 3: 特徴量抽出

## 1. PCA（主成分分析）

データの分散を最大化する方向に射影。

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCAの実行
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 説明分散比
print(f"説明分散比: {pca.explained_variance_ratio_}")
print(f"累積寄与率: {pca.explained_variance_ratio_.cumsum()}")

# 可視化
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Target')
plt.show()
```

### 最適な成分数の決定
```python
# すべての成分で実行
pca_full = PCA()
pca_full.fit(X_scaled)

# 累積寄与率プロット
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         pca_full.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.legend()
plt.show()

# 95%の分散を保持する成分数
n_components_95 = np.argmax(pca_full.explained_variance_ratio_.cumsum() >= 0.95) + 1
print(f"95%の分散を保持する成分数: {n_components_95}")
```

## 2. LDA（線形判別分析）

クラス間分散を最大化、クラス内分散を最小化。

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDAの実行（分類タスク専用）
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# 可視化
plt.figure(figsize=(10, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.colorbar(label='Class')
plt.show()
```

## 3. t-SNE（t分布確率的近傍埋め込み）

高次元データの可視化に特化。

```python
from sklearn.manifold import TSNE

# t-SNEの実行
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# 可視化
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.title('t-SNE Visualization')
plt.colorbar(label='Class')
plt.show()
```

## 4. UMAP（均一多様体近似射影）

t-SNEより高速で、距離を保存。

```python
# UMAPのインストール: pip install umap-learn
try:
    import umap

    # UMAPの実行
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # 可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.title('UMAP Visualization')
    plt.colorbar(label='Class')
    plt.show()
except ImportError:
    print("UMAPがインストールされていません")
```

## 5. 手法の比較

```python
from sklearn.preprocessing import StandardScaler

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 各手法の実行
methods = {
    'PCA': PCA(n_components=2),
    'LDA': LinearDiscriminantAnalysis(n_components=2),
    't-SNE': TSNE(n_components=2, random_state=42)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, method) in zip(axes, methods.items()):
    if name == 'LDA':
        X_transformed = method.fit_transform(X_scaled, y)
    else:
        X_transformed = method.fit_transform(X_scaled)

    ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
               c=y, cmap='viridis', alpha=0.5)
    ax.set_title(name)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

plt.tight_layout()
plt.show()
```

**選択基準:**
- **PCA**: 次元削減全般、前処理
- **LDA**: 分類タスクの次元削減
- **t-SNE**: 可視化専用、計算コスト高
- **UMAP**: 可視化と次元削減、t-SNEより高速

---

**次へ**: [Chapter 4: ドメイン知識の活用 →](chapter-4.html)
