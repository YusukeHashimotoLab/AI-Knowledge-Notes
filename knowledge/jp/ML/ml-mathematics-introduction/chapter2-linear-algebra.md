---
title: 第2章：線形代数の基礎
chapter_title: 第2章：線形代数の基礎
---

**機械学習アルゴリズムの数学的基盤となる線形代数を理論と実装の両面から深く理解する**

**この章で学べること**

  * ベクトルと行列の基本演算と幾何的意味
  * 固有値分解、SVD、QR分解の理論と実装
  * 主成分分析（PCA）の数学的原理と応用
  * 線形変換と射影の幾何学的理解
  * 線形回帰とRidge回帰への線形代数の適用

## 1\. ベクトルと行列の基礎

### 1.1 ベクトルの内積とノルム

ベクトルの内積（ドット積）は、2つのベクトルの類似度を測る基本的な演算です。

$$\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T\mathbf{y} = \sum_{i=1}^{n} x_i y_i = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$$ 

ここで、θはベクトル間の角度です。内積の幾何的意味：

  * **正** : 同じ方向を向いている（鋭角）
  * **ゼロ** : 直交している（垂直）
  * **負** : 反対方向を向いている（鈍角）

ベクトルのノルム（長さ）は、ベクトルの大きさを表します。

$$\|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^T\mathbf{x}} = \sqrt{\sum_{i=1}^{n} x_i^2} \quad \text{（L2ノルム）}$$ $$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i| \quad \text{（L1ノルム）}$$ 

**機械学習での応用** L2ノルムはRidge回帰の正則化項、L1ノルムはLasso回帰の正則化項として使用されます。また、コサイン類似度は文書分類や推薦システムで頻繁に使われます。 

### 1.2 行列の基本演算

行列積は線形変換を合成する演算です。

$$(\mathbf{AB})_{ij} = \sum_{k=1}^{m} A_{ik}B_{kj}$$ 

**重要な性質：**

  * 結合律: \\((\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})\\)
  * 分配律: \\(\mathbf{A}(\mathbf{B}+\mathbf{C}) = \mathbf{AB} + \mathbf{AC}\\)
  * 転置: \\((\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T\\)
  * 非可換性: 一般に \\(\mathbf{AB} \neq \mathbf{BA}\\)

### 実装例1：ベクトル・行列演算の実装
    
    
    import numpy as np
    
    class LinearAlgebraOps:
        """線形代数の基本演算を実装するクラス"""
    
        @staticmethod
        def inner_product(x, y):
            """
            内積の計算: x・y = Σ x_i * y_i
    
            Parameters:
            -----------
            x, y : array-like
                入力ベクトル
    
            Returns:
            --------
            float : 内積
            """
            x, y = np.array(x), np.array(y)
            assert x.shape == y.shape, "ベクトルの次元が一致しません"
            return np.sum(x * y)
    
        @staticmethod
        def cosine_similarity(x, y):
            """
            コサイン類似度: cos(θ) = (x・y) / (||x|| * ||y||)
    
            Returns:
            --------
            float : -1から1の範囲の類似度
            """
            x, y = np.array(x), np.array(y)
            dot_product = LinearAlgebraOps.inner_product(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
    
            if norm_x == 0 or norm_y == 0:
                return 0.0
    
            return dot_product / (norm_x * norm_y)
    
        @staticmethod
        def vector_norm(x, p=2):
            """
            ベクトルのLpノルム
    
            Parameters:
            -----------
            x : array-like
                入力ベクトル
            p : int or float
                ノルムの次数（1, 2, np.inf など）
    
            Returns:
            --------
            float : ノルム
            """
            x = np.array(x)
            if p == 1:
                return np.sum(np.abs(x))
            elif p == 2:
                return np.sqrt(np.sum(x**2))
            elif p == np.inf:
                return np.max(np.abs(x))
            else:
                return np.sum(np.abs(x)**p)**(1/p)
    
        @staticmethod
        def matrix_multiply(A, B):
            """
            行列積の実装: C = AB
    
            Parameters:
            -----------
            A : ndarray of shape (m, n)
            B : ndarray of shape (n, p)
    
            Returns:
            --------
            ndarray of shape (m, p)
            """
            A, B = np.array(A), np.array(B)
            assert A.shape[1] == B.shape[0], f"行列の形状が不適切: {A.shape} と {B.shape}"
    
            m, n = A.shape
            p = B.shape[1]
            C = np.zeros((m, p))
    
            for i in range(m):
                for j in range(p):
                    C[i, j] = np.sum(A[i, :] * B[:, j])
    
            return C
    
        @staticmethod
        def outer_product(x, y):
            """
            外積（テンソル積）: xy^T
    
            Returns:
            --------
            ndarray : 行列
            """
            x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
            return x @ y.T
    
    # 使用例
    ops = LinearAlgebraOps()
    
    # ベクトル演算
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print("ベクトル演算:")
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"内積: {ops.inner_product(v1, v2)}")
    print(f"コサイン類似度: {ops.cosine_similarity(v1, v2):.4f}")
    print(f"L1ノルム: {ops.vector_norm(v1, p=1):.4f}")
    print(f"L2ノルム: {ops.vector_norm(v1, p=2):.4f}")
    
    # 行列演算
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"\n行列積:")
    print(f"A @ B =\n{ops.matrix_multiply(A, B)}")
    print(f"NumPy検証:\n{A @ B}")
    
    # 外積
    print(f"\n外積 v1 ⊗ v2 =\n{ops.outer_product(v1, v2)}")
    

## 2\. 行列分解

### 2.1 固有値分解（Eigendecomposition）

正方行列Aの固有値λと固有ベクトルvは以下を満たします：

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$ 

対称行列は以下のように対角化できます：

$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$$ 

ここで：

  * **Q** : 固有ベクトルを列に持つ直交行列
  * **Λ** : 固有値を対角成分に持つ対角行列

**幾何的意味** 固有ベクトルは行列による変換で方向が変わらないベクトルで、固有値はその拡大率を表します。固有値が大きいほど、その方向の変動が大きいことを意味します。 

### 2.2 特異値分解（SVD）

任意の行列Aは以下のように分解できます：

$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$ 

ここで：

  * **U** : 左特異ベクトル（m×m直交行列）
  * **Σ** : 特異値の対角行列（m×n）
  * **V** : 右特異ベクトル（n×n直交行列）

特異値は降順に並べられ、σ₁ ≥ σ₂ ≥ ... ≥ 0 です。

**機械学習での応用** SVDは主成分分析（PCA）、推薦システム（協調フィルタリング）、自然言語処理（LSA）、画像圧縮など幅広く使用されます。 

### 2.3 QR分解

任意の行列Aは直交行列Qと上三角行列Rの積に分解できます：

$$\mathbf{A} = \mathbf{QR}$$ 

QR分解は、最小二乗法の数値的に安定な解法として使用されます。

### 実装例2：行列分解の実装と比較
    
    
    import numpy as np
    from scipy import linalg
    
    class MatrixDecomposition:
        """行列分解の実装と応用"""
    
        @staticmethod
        def eigen_decomposition(A, symmetric=True):
            """
            固有値分解: A = QΛQ^T
    
            Parameters:
            -----------
            A : ndarray of shape (n, n)
                分解する行列
            symmetric : bool
                対称行列かどうか
    
            Returns:
            --------
            eigenvalues : ndarray
                固有値（降順）
            eigenvectors : ndarray
                対応する固有ベクトル
            """
            A = np.array(A)
            assert A.shape[0] == A.shape[1], "正方行列である必要があります"
    
            if symmetric:
                eigenvalues, eigenvectors = np.linalg.eigh(A)
            else:
                eigenvalues, eigenvectors = np.linalg.eig(A)
    
            # 固有値の降順にソート
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
    
            return eigenvalues, eigenvectors
    
        @staticmethod
        def svd_decomposition(A, full_matrices=False):
            """
            特異値分解: A = UΣV^T
    
            Parameters:
            -----------
            A : ndarray of shape (m, n)
                分解する行列
            full_matrices : bool
                完全な行列を返すか
    
            Returns:
            --------
            U : ndarray of shape (m, m) or (m, k)
                左特異ベクトル
            S : ndarray of shape (k,)
                特異値（降順）
            Vt : ndarray of shape (n, n) or (k, n)
                右特異ベクトルの転置
            """
            A = np.array(A)
            U, S, Vt = np.linalg.svd(A, full_matrices=full_matrices)
            return U, S, Vt
    
        @staticmethod
        def qr_decomposition(A):
            """
            QR分解: A = QR
    
            Parameters:
            -----------
            A : ndarray of shape (m, n)
                分解する行列
    
            Returns:
            --------
            Q : ndarray of shape (m, m)
                直交行列
            R : ndarray of shape (m, n)
                上三角行列
            """
            A = np.array(A)
            Q, R = np.linalg.qr(A)
            return Q, R
    
        @staticmethod
        def low_rank_approximation(A, k):
            """
            SVDを使った低ランク近似
    
            Parameters:
            -----------
            A : ndarray of shape (m, n)
                元の行列
            k : int
                近似のランク
    
            Returns:
            --------
            A_approx : ndarray
                ランクkの近似行列
            reconstruction_error : float
                フロベニウスノルムでの再構成誤差
            """
            U, S, Vt = MatrixDecomposition.svd_decomposition(A, full_matrices=False)
    
            # 上位k個の特異値のみ使用
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]
    
            # 低ランク近似
            A_approx = U_k @ np.diag(S_k) @ Vt_k
    
            # 再構成誤差
            reconstruction_error = np.linalg.norm(A - A_approx, 'fro')
    
            return A_approx, reconstruction_error
    
    # 使用例
    decomp = MatrixDecomposition()
    
    # 対称行列の固有値分解
    print("=" * 60)
    print("固有値分解")
    print("=" * 60)
    A_sym = np.array([[4, 2], [2, 3]])
    eigenvalues, eigenvectors = decomp.eigen_decomposition(A_sym)
    print(f"元の行列 A:\n{A_sym}\n")
    print(f"固有値: {eigenvalues}")
    print(f"固有ベクトル:\n{eigenvectors}\n")
    
    # 再構成の検証
    Lambda = np.diag(eigenvalues)
    A_reconstructed = eigenvectors @ Lambda @ eigenvectors.T
    print(f"再構成 A = QΛQ^T:\n{A_reconstructed}")
    print(f"再構成誤差: {np.linalg.norm(A_sym - A_reconstructed):.10f}\n")
    
    # SVD
    print("=" * 60)
    print("特異値分解（SVD）")
    print("=" * 60)
    A = np.array([[1, 2, 3], [4, 5, 6]])
    U, S, Vt = decomp.svd_decomposition(A, full_matrices=True)
    print(f"元の行列 A ({A.shape}):\n{A}\n")
    print(f"U ({U.shape}):\n{U}\n")
    print(f"特異値 S: {S}\n")
    print(f"V^T ({Vt.shape}):\n{Vt}\n")
    
    # 再構成
    Sigma = np.zeros_like(A, dtype=float)
    Sigma[:len(S), :len(S)] = np.diag(S)
    A_reconstructed = U @ Sigma @ Vt
    print(f"再構成 A = UΣV^T:\n{A_reconstructed}")
    print(f"再構成誤差: {np.linalg.norm(A - A_reconstructed):.10f}\n")
    
    # 低ランク近似
    print("=" * 60)
    print("低ランク近似")
    print("=" * 60)
    A_large = np.random.randn(10, 8)
    for k in [1, 2, 4, 8]:
        A_approx, error = decomp.low_rank_approximation(A_large, k)
        compression_ratio = (k * (A_large.shape[0] + A_large.shape[1])) / A_large.size
        print(f"ランク {k}: 再構成誤差 = {error:.4f}, 圧縮率 = {compression_ratio:.2%}")
    
    # QR分解
    print("\n" + "=" * 60)
    print("QR分解")
    print("=" * 60)
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    Q, R = decomp.qr_decomposition(A)
    print(f"元の行列 A:\n{A}\n")
    print(f"Q (直交行列):\n{Q}\n")
    print(f"R (上三角行列):\n{R}\n")
    print(f"Q^T Q (単位行列):\n{Q.T @ Q}")
    print(f"再構成 A = QR:\n{Q @ R}")
    

## 3\. 主成分分析（PCA）

### 3.1 PCAの数学的定式化

主成分分析は、データの分散を最大化する直交軸を見つける次元削減手法です。

**目的：** データ行列 X (n×d) を低次元空間 (n×k, k < d) に射影

$$\max_{\mathbf{w}} \mathbf{w}^T\mathbf{S}\mathbf{w} \quad \text{s.t.} \quad \|\mathbf{w}\|^2 = 1$$ 

ここで、Sは共分散行列です：

$$\mathbf{S} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T = \frac{1}{n}\mathbf{X}_c^T\mathbf{X}_c$$ 

**解法：** 共分散行列の固有値分解

$$\mathbf{S} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$$ 

主成分は固有値の大きい順の固有ベクトルです。

### 3.2 PCAの手順

  1. **中心化** : データから平均を引く
  2. **共分散行列の計算** : S = (1/n)X_c^T X_c
  3. **固有値分解** : 固有値と固有ベクトルを計算
  4. **主成分の選択** : 固有値の大きい順にk個選択
  5. **射影** : データを主成分空間に変換

**寄与率と累積寄与率** 第i主成分の寄与率 = λ_i / Σλ_j は、その主成分が説明する分散の割合です。累積寄与率が90%以上になる次元数を選ぶのが一般的です。 

### 実装例3：主成分分析（PCA）の完全実装
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class PCA:
        """主成分分析の実装"""
    
        def __init__(self, n_components=None):
            """
            Parameters:
            -----------
            n_components : int or None
                保持する主成分の数（Noneの場合は全て）
            """
            self.n_components = n_components
            self.components_ = None
            self.explained_variance_ = None
            self.explained_variance_ratio_ = None
            self.mean_ = None
    
        def fit(self, X):
            """
            主成分分析を実行
    
            Parameters:
            -----------
            X : ndarray of shape (n_samples, n_features)
                入力データ
    
            Returns:
            --------
            self
            """
            X = np.array(X)
            n_samples, n_features = X.shape
    
            # 1. データの中心化
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
    
            # 2. 共分散行列の計算
            cov_matrix = (X_centered.T @ X_centered) / n_samples
    
            # 3. 固有値分解
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
            # 4. 固有値の降順にソート
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
    
            # 5. 主成分の選択
            if self.n_components is None:
                self.n_components = n_features
            else:
                self.n_components = min(self.n_components, n_features)
    
            self.components_ = eigenvectors[:, :self.n_components].T
            self.explained_variance_ = eigenvalues[:self.n_components]
            self.explained_variance_ratio_ = (
                self.explained_variance_ / np.sum(eigenvalues)
            )
    
            return self
    
        def transform(self, X):
            """
            データを主成分空間に変換
    
            Parameters:
            -----------
            X : ndarray of shape (n_samples, n_features)
                入力データ
    
            Returns:
            --------
            ndarray of shape (n_samples, n_components)
                変換されたデータ
            """
            X = np.array(X)
            X_centered = X - self.mean_
            return X_centered @ self.components_.T
    
        def fit_transform(self, X):
            """fitとtransformを同時に実行"""
            return self.fit(X).transform(X)
    
        def inverse_transform(self, X_transformed):
            """
            主成分空間から元の空間に戻す
    
            Parameters:
            -----------
            X_transformed : ndarray of shape (n_samples, n_components)
                変換されたデータ
    
            Returns:
            --------
            ndarray of shape (n_samples, n_features)
                再構成されたデータ
            """
            return X_transformed @ self.components_ + self.mean_
    
        def reconstruction_error(self, X):
            """
            再構成誤差を計算
    
            Returns:
            --------
            float : 平均二乗再構成誤差
            """
            X_transformed = self.transform(X)
            X_reconstructed = self.inverse_transform(X_transformed)
            return np.mean((X - X_reconstructed) ** 2)
    
    # 使用例：2次元データでの可視化
    np.random.seed(42)
    
    # 相関のある2次元データを生成
    mean = [0, 0]
    cov = [[3, 1.5], [1.5, 1]]
    X = np.random.multivariate_normal(mean, cov, 300)
    
    # PCAの実行
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    
    print("=" * 60)
    print("PCA結果")
    print("=" * 60)
    print(f"主成分（固有ベクトル）:\n{pca.components_}")
    print(f"説明分散（固有値）: {pca.explained_variance_}")
    print(f"寄与率: {pca.explained_variance_ratio_}")
    print(f"累積寄与率: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 元のデータと主成分軸
    ax1 = axes[0]
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5, s=30)
    ax1.set_xlabel('特徴量1')
    ax1.set_ylabel('特徴量2')
    ax1.set_title('元のデータと主成分軸')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 主成分軸を描画
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        ax1.arrow(0, 0, comp[0]*np.sqrt(var)*3, comp[1]*np.sqrt(var)*3,
                 head_width=0.3, head_length=0.2, fc=f'C{i+1}', ec=f'C{i+1}',
                 linewidth=2, label=f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})')
    ax1.legend()
    
    # 主成分空間でのデータ
    ax2 = axes[1]
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=30)
    ax2.set_xlabel('第1主成分')
    ax2.set_ylabel('第2主成分')
    ax2.set_title('主成分空間でのデータ')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=150, bbox_inches='tight')
    print("\nPCAの可視化を保存しました")
    
    # 次元削減の効果
    print("\n" + "=" * 60)
    print("次元削減の効果")
    print("=" * 60)
    for n_comp in [1, 2]:
        pca_reduced = PCA(n_components=n_comp)
        pca_reduced.fit(X)
        error = pca_reduced.reconstruction_error(X)
        cum_var = np.sum(pca_reduced.explained_variance_ratio_)
        print(f"{n_comp}次元: 累積寄与率={cum_var:.2%}, 再構成誤差={error:.4f}")
    

## 4\. 線形変換と射影

### 4.1 線形変換の幾何学

線形変換は行列Aによってベクトルを変換する操作です：

$$\mathbf{y} = \mathbf{A}\mathbf{x}$$ 

**代表的な線形変換：**

  * **回転** : 直交行列による変換（長さ保存）
  * **拡大縮小** : 対角行列による変換
  * **せん断** : 非対角成分を持つ変換
  * **射影** : 部分空間への投影

### 4.2 射影行列

ベクトルbを列空間C(A)に射影する射影行列は：

$$\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$$ 

射影ベクトルは p = Pb で、残差は e = b - p です。

**射影行列の性質：**

  * 対称性: P^T = P
  * 冪等性: P² = P
  * 残差の直交性: A^T(b - Pb) = 0

**最小二乗法との関係** 線形回帰の最小二乗解は、yをX の列空間に射影することで得られます。これにより、残差が列空間と直交することが保証されます。 

### 実装例4：線形変換と射影の可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class LinearTransformation:
        """線形変換と射影の実装"""
    
        @staticmethod
        def rotation_matrix(theta):
            """
            2次元回転行列
    
            Parameters:
            -----------
            theta : float
                回転角（ラジアン）
    
            Returns:
            --------
            ndarray : 2x2回転行列
            """
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s], [s, c]])
    
        @staticmethod
        def scaling_matrix(sx, sy):
            """
            2次元スケーリング行列
    
            Parameters:
            -----------
            sx, sy : float
                x方向、y方向のスケール
    
            Returns:
            --------
            ndarray : 2x2スケーリング行列
            """
            return np.array([[sx, 0], [0, sy]])
    
        @staticmethod
        def projection_matrix(A):
            """
            列空間への射影行列: P = A(A^T A)^(-1)A^T
    
            Parameters:
            -----------
            A : ndarray of shape (m, n)
                基底を列に持つ行列
    
            Returns:
            --------
            ndarray of shape (m, m) : 射影行列
            """
            A = np.array(A)
            return A @ np.linalg.inv(A.T @ A) @ A.T
    
        @staticmethod
        def project_onto_subspace(b, A):
            """
            ベクトルbをA の列空間に射影
    
            Parameters:
            -----------
            b : ndarray
                射影するベクトル
            A : ndarray
                部分空間を張る行列
    
            Returns:
            --------
            projection : ndarray
                射影ベクトル
            residual : ndarray
                残差ベクトル
            """
            P = LinearTransformation.projection_matrix(A)
            projection = P @ b
            residual = b - projection
            return projection, residual
    
    # 可視化例：線形変換
    print("=" * 60)
    print("線形変換の可視化")
    print("=" * 60)
    
    # 単位正方形の頂点
    square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])
    
    # 各種変換
    transformations = {
        '回転 (45°)': LinearTransformation.rotation_matrix(np.pi/4),
        'スケーリング (2, 0.5)': LinearTransformation.scaling_matrix(2, 0.5),
        'せん断': np.array([[1, 0.5], [0, 1]]),
        '複合変換': LinearTransformation.rotation_matrix(np.pi/6) @ \
                    LinearTransformation.scaling_matrix(1.5, 0.8)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, (name, A) in enumerate(transformations.items()):
        ax = axes[idx]
    
        # 元の図形
        ax.plot(square[0], square[1], 'b-', linewidth=2, label='元の図形')
        ax.fill(square[0], square[1], 'blue', alpha=0.2)
    
        # 変換後の図形
        transformed = A @ square
        ax.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='変換後')
        ax.fill(transformed[0], transformed[1], 'red', alpha=0.2)
    
        # 基底ベクトルの変換
        basis = np.array([[1, 0], [0, 1]]).T
        transformed_basis = A @ basis
        for i in range(2):
            ax.arrow(0, 0, transformed_basis[0, i], transformed_basis[1, i],
                    head_width=0.1, head_length=0.1, fc=f'C{i+2}', ec=f'C{i+2}',
                    linewidth=2)
    
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name}\n行列式: {np.linalg.det(A):.2f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 3)
    
    plt.tight_layout()
    plt.savefig('linear_transformations.png', dpi=150, bbox_inches='tight')
    print("線形変換の可視化を保存しました")
    
    # 射影の例
    print("\n" + "=" * 60)
    print("射影の計算")
    print("=" * 60)
    
    # 2次元での1次元部分空間への射影
    a = np.array([[1], [2]])  # 部分空間の基底
    b = np.array([3, 2])      # 射影するベクトル
    
    proj, resid = LinearTransformation.project_onto_subspace(b, a)
    
    print(f"基底ベクトル a: {a.flatten()}")
    print(f"ベクトル b: {b}")
    print(f"射影 p: {proj}")
    print(f"残差 e: {resid}")
    print(f"内積 a^T e (直交性の確認): {a.T @ resid}")
    print(f"||b||²: {np.linalg.norm(b)**2:.4f}")
    print(f"||p||² + ||e||²: {np.linalg.norm(proj)**2 + np.linalg.norm(resid)**2:.4f}")
    
    # 射影の可視化
    plt.figure(figsize=(8, 8))
    plt.arrow(0, 0, b[0], b[1], head_width=0.2, head_length=0.2,
             fc='blue', ec='blue', linewidth=2, label='元のベクトル b')
    plt.arrow(0, 0, proj[0], proj[1], head_width=0.2, head_length=0.2,
             fc='green', ec='green', linewidth=2, label='射影 p')
    plt.arrow(0, 0, a[0, 0]*2, a[1, 0]*2, head_width=0.2, head_length=0.2,
             fc='red', ec='red', linewidth=2, linestyle='--', label='部分空間の基底')
    plt.plot([proj[0], b[0]], [proj[1], b[1]], 'k--', linewidth=1, label='残差 e')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ベクトルの部分空間への射影')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.xlim(-1, 4)
    plt.ylim(-1, 5)
    plt.savefig('projection_visualization.png', dpi=150, bbox_inches='tight')
    print("射影の可視化を保存しました")
    

## 5\. 実践応用

### 5.1 線形回帰の線形代数的解法

線形回帰の目的は、最小二乗誤差を最小化するパラメータwを見つけることです：

$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{Xw}\|^2$$ 

正規方程式による解：

$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$ 

これは、yをXの列空間に射影する操作と等価です。

### 5.2 Ridge回帰（L2正則化）

Ridge回帰は、過学習を防ぐためにL2正則化項を追加します：

$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{Xw}\|^2 + \lambda\|\mathbf{w}\|^2$$ 

解は以下の形になります：

$$\mathbf{w}_{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$ 

λが大きいほど、パラメータの大きさが制限されます。

### 実装例5：線形回帰とRidge回帰の実装
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class LinearRegression:
        """線形回帰の実装（行列演算による）"""
    
        def __init__(self, fit_intercept=True):
            """
            Parameters:
            -----------
            fit_intercept : bool
                切片を含めるか
            """
            self.fit_intercept = fit_intercept
            self.coef_ = None
            self.intercept_ = None
    
        def fit(self, X, y):
            """
            正規方程式で最小二乗解を計算: w = (X^T X)^(-1) X^T y
    
            Parameters:
            -----------
            X : ndarray of shape (n_samples, n_features)
                特徴量行列
            y : ndarray of shape (n_samples,)
                目的変数
    
            Returns:
            --------
            self
            """
            X, y = np.array(X), np.array(y).reshape(-1, 1)
    
            if self.fit_intercept:
                # 切片項を追加
                X = np.hstack([np.ones((X.shape[0], 1)), X])
    
            # 正規方程式: (X^T X) w = X^T y
            XtX = X.T @ X
            Xty = X.T @ y
            w = np.linalg.solve(XtX, Xty)
    
            if self.fit_intercept:
                self.intercept_ = w[0, 0]
                self.coef_ = w[1:].flatten()
            else:
                self.intercept_ = 0
                self.coef_ = w.flatten()
    
            return self
    
        def predict(self, X):
            """予測"""
            X = np.array(X)
            return X @ self.coef_ + self.intercept_
    
    class RidgeRegression:
        """Ridge回帰の実装（L2正則化）"""
    
        def __init__(self, alpha=1.0, fit_intercept=True):
            """
            Parameters:
            -----------
            alpha : float
                正則化パラメータ（λ）
            fit_intercept : bool
                切片を含めるか
            """
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            self.coef_ = None
            self.intercept_ = None
    
        def fit(self, X, y):
            """
            Ridge回帰の解を計算: w = (X^T X + λI)^(-1) X^T y
    
            Parameters:
            -----------
            X : ndarray of shape (n_samples, n_features)
                特徴量行列
            y : ndarray of shape (n_samples,)
                目的変数
    
            Returns:
            --------
            self
            """
            X, y = np.array(X), np.array(y).reshape(-1, 1)
    
            if self.fit_intercept:
                X = np.hstack([np.ones((X.shape[0], 1)), X])
    
            # Ridge回帰の解
            n_features = X.shape[1]
            ridge_matrix = X.T @ X + self.alpha * np.eye(n_features)
    
            # 切片には正則化を適用しない
            if self.fit_intercept:
                ridge_matrix[0, 0] = X.T[0] @ X[:, 0]
    
            w = np.linalg.solve(ridge_matrix, X.T @ y)
    
            if self.fit_intercept:
                self.intercept_ = w[0, 0]
                self.coef_ = w[1:].flatten()
            else:
                self.intercept_ = 0
                self.coef_ = w.flatten()
    
            return self
    
        def predict(self, X):
            """予測"""
            X = np.array(X)
            return X @ self.coef_ + self.intercept_
    
    # 使用例とQR分解による解法の比較
    def solve_with_qr(X, y):
        """QR分解を使った数値的に安定な最小二乗解"""
        Q, R = np.linalg.qr(X)
        return np.linalg.solve(R, Q.T @ y)
    
    # データ生成
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1)
    y_true = 3 * X.squeeze() + 2
    y = y_true + np.random.randn(n_samples) * 0.5
    
    # 線形回帰
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    
    print("=" * 60)
    print("線形回帰の結果")
    print("=" * 60)
    print(f"係数: {lr.coef_}")
    print(f"切片: {lr.intercept_:.4f}")
    print(f"MSE: {np.mean((y - y_pred_lr)**2):.4f}")
    
    # Ridge回帰（異なるαで比較）
    alphas = [0.01, 0.1, 1.0, 10.0]
    ridge_models = []
    
    print("\n" + "=" * 60)
    print("Ridge回帰の結果")
    print("=" * 60)
    
    for alpha in alphas:
        ridge = RidgeRegression(alpha=alpha)
        ridge.fit(X, y)
        y_pred = ridge.predict(X)
        mse = np.mean((y - y_pred)**2)
        ridge_models.append(ridge)
        print(f"α={alpha:5.2f}: 係数={ridge.coef_[0]:6.3f}, "
              f"切片={ridge.intercept_:6.3f}, MSE={mse:.4f}")
    
    # 可視化
    plt.figure(figsize=(14, 5))
    
    # 左図: 線形回帰
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, s=30, label='データ')
    plt.plot(X, y_true, 'g--', linewidth=2, label='真の関数')
    plt.plot(X, y_pred_lr, 'r-', linewidth=2, label='線形回帰')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('線形回帰')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右図: Ridge回帰の比較
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, alpha=0.5, s=30, label='データ')
    plt.plot(X, y_true, 'g--', linewidth=2, label='真の関数')
    X_sorted = np.sort(X, axis=0)
    for ridge, alpha in zip(ridge_models, alphas):
        y_line = ridge.predict(X_sorted)
        plt.plot(X_sorted, y_line, linewidth=2, label=f'Ridge (α={alpha})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ridge回帰（正則化パラメータの影響）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_ridge_regression.png', dpi=150, bbox_inches='tight')
    print("\n回帰結果の可視化を保存しました")
    
    # 多重共線性の例
    print("\n" + "=" * 60)
    print("多重共線性の例")
    print("=" * 60)
    
    # 高度に相関した特徴量を生成
    X_corr = np.random.randn(50, 1)
    X_multi = np.hstack([X_corr, X_corr + np.random.randn(50, 1) * 0.1, X_corr * 2])
    y_multi = X_corr.squeeze() + np.random.randn(50) * 0.5
    
    # 線形回帰（不安定）
    lr_multi = LinearRegression()
    lr_multi.fit(X_multi, y_multi)
    
    # Ridge回帰（安定）
    ridge_multi = RidgeRegression(alpha=1.0)
    ridge_multi.fit(X_multi, y_multi)
    
    print("線形回帰の係数:", lr_multi.coef_)
    print("Ridge回帰の係数:", ridge_multi.coef_)
    print("係数のL2ノルム:")
    print(f"  線形回帰: {np.linalg.norm(lr_multi.coef_):.4f}")
    print(f"  Ridge回帰: {np.linalg.norm(ridge_multi.coef_):.4f}")
    

### 5.3 画像データへのPCA適用

PCAは画像の次元削減と圧縮に広く使用されます。各ピクセルを特徴量として扱います。

### 実装例6：画像圧縮へのPCA/SVDの適用
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class ImageCompressionPCA:
        """画像圧縮のためのPCA/SVD実装"""
    
        @staticmethod
        def compress_with_svd(image, n_components):
            """
            SVDを使った画像圧縮
    
            Parameters:
            -----------
            image : ndarray of shape (height, width)
                グレースケール画像
            n_components : int
                保持する特異値の数
    
            Returns:
            --------
            compressed : ndarray
                圧縮された画像
            compression_ratio : float
                圧縮率
            """
            # SVD分解
            U, S, Vt = np.linalg.svd(image, full_matrices=False)
    
            # 上位n_components個のみ使用
            U_k = U[:, :n_components]
            S_k = S[:n_components]
            Vt_k = Vt[:n_components, :]
    
            # 再構成
            compressed = U_k @ np.diag(S_k) @ Vt_k
    
            # 圧縮率の計算
            original_size = image.shape[0] * image.shape[1]
            compressed_size = n_components * (image.shape[0] + image.shape[1] + 1)
            compression_ratio = compressed_size / original_size
    
            return compressed, compression_ratio
    
        @staticmethod
        def analyze_singular_values(image):
            """
            特異値の分析
    
            Returns:
            --------
            singular_values : ndarray
                特異値
            cumulative_energy : ndarray
                累積エネルギー
            """
            _, S, _ = np.linalg.svd(image, full_matrices=False)
    
            # エネルギー（各特異値の2乗）
            energy = S ** 2
            total_energy = np.sum(energy)
            cumulative_energy = np.cumsum(energy) / total_energy
    
            return S, cumulative_energy
    
    # 使用例：合成画像での実験
    print("=" * 60)
    print("画像圧縮の実験")
    print("=" * 60)
    
    # 合成画像の生成（グラデーションとパターン）
    height, width = 200, 200
    x = np.linspace(-5, 5, width)
    y = np.linspace(-5, 5, height)
    X, Y = np.meshgrid(x, y)
    
    # 複雑なパターンの画像
    image = (np.sin(X) * np.cos(Y) +
             0.5 * np.sin(2*X + Y) +
             0.3 * np.cos(X - 2*Y))
    image = (image - image.min()) / (image.max() - image.min())  # 正規化
    
    # 特異値の分析
    compressor = ImageCompressionPCA()
    singular_values, cumulative_energy = compressor.analyze_singular_values(image)
    
    print(f"画像サイズ: {image.shape}")
    print(f"総特異値数: {len(singular_values)}")
    print(f"90%エネルギーに必要な成分数: {np.argmax(cumulative_energy >= 0.90) + 1}")
    print(f"99%エネルギーに必要な成分数: {np.argmax(cumulative_energy >= 0.99) + 1}")
    
    # 異なる圧縮率での比較
    n_components_list = [5, 10, 20, 50]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # 元画像
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('元画像')
    axes[0].axis('off')
    
    # 圧縮画像
    for idx, n_comp in enumerate(n_components_list, 1):
        compressed, comp_ratio = compressor.compress_with_svd(image, n_comp)
    
        axes[idx].imshow(compressed, cmap='gray')
    
        # PSNRの計算
        mse = np.mean((image - compressed) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
        energy_retained = cumulative_energy[n_comp - 1]
    
        axes[idx].set_title(f'成分数: {n_comp}\n'
                           f'圧縮率: {comp_ratio:.1%}\n'
                           f'PSNR: {psnr:.1f}dB\n'
                           f'エネルギー: {energy_retained:.1%}')
        axes[idx].axis('off')
    
    # 特異値の減衰をプロット
    axes[5].plot(singular_values[:100], 'b-', linewidth=2)
    axes[5].set_xlabel('成分番号')
    axes[5].set_ylabel('特異値')
    axes[5].set_title('特異値の減衰')
    axes[5].grid(True, alpha=0.3)
    axes[5].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('image_compression_pca.png', dpi=150, bbox_inches='tight')
    print("\n画像圧縮の可視化を保存しました")
    
    # 累積エネルギーのプロット
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_energy[:100], linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90%エネルギー')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99%エネルギー')
    plt.xlabel('成分数')
    plt.ylabel('累積エネルギー')
    plt.title('SVD成分の累積エネルギー')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cumulative_energy.png', dpi=150, bbox_inches='tight')
    print("累積エネルギープロットを保存しました")
    
    # 実用的な圧縮率の分析
    print("\n" + "=" * 60)
    print("圧縮率とPSNRの関係")
    print("=" * 60)
    print(f"{'成分数':>8} {'圧縮率':>10} {'PSNR (dB)':>12} {'エネルギー':>12}")
    print("-" * 60)
    
    for n_comp in [1, 2, 5, 10, 20, 50, 100]:
        if n_comp <= min(image.shape):
            compressed, comp_ratio = compressor.compress_with_svd(image, n_comp)
            mse = np.mean((image - compressed) ** 2)
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            energy = cumulative_energy[n_comp - 1]
            print(f"{n_comp:8d} {comp_ratio:9.1%} {psnr:11.2f} {energy:11.1%}")
    

## まとめ

この章では、機械学習の数学的基盤となる線形代数を学びました。

**学習した内容**

  * **ベクトルと行列** : 内積、ノルム、行列演算の幾何的意味
  * **行列分解** : 固有値分解、SVD、QR分解の理論と実装
  * **主成分分析** : データの分散を最大化する次元削減手法
  * **線形変換と射影** : 最小二乗法の幾何学的理解
  * **実践応用** : 線形回帰、Ridge回帰、画像圧縮

**次章への準備** 第3章では、最適化理論を学びます。線形回帰の正規方程式は最適化問題の解析解ですが、次章では勾配降下法などの数値的解法を学び、ニューラルネットワークの学習に適用します。 

### 行列分解の比較

分解法 | 形式 | 対象行列 | 主な用途  
---|---|---|---  
固有値分解 | A = QΛQ^T | 正方対称行列 | PCA、グラフ解析  
SVD | A = UΣV^T | 任意の行列 | 次元削減、推薦システム  
QR分解 | A = QR | 任意の行列 | 最小二乗法、固有値計算  
コレスキー分解 | A = LL^T | 正定値対称行列 | 線形システム、ガウス過程  
  
### 演習問題

  1. 2つのベクトルが直交するとき、コサイン類似度がどうなるか確認してください
  2. 3×3の対称行列を作成し、固有値分解して再構成誤差を確認してください
  3. ランダムな5×3行列でSVDを実行し、ランク2近似を作成してください
  4. 3次元データでPCAを実行し、2次元に次元削減して可視化してください
  5. 多項式回帰（2次、3次）でRidge回帰の正則化効果を比較してください
  6. 実際の画像データ（グレースケール）にSVDを適用し、最適な圧縮率を見つけてください

### 参考文献

  * G. Strang, "Linear Algebra and Its Applications" (2016)
  * L.N. Trefethen and D. Bau, "Numerical Linear Algebra" (1997)
  * 斎藤正彦, "線型代数入門" 東京大学出版会 (1966)
  * I. Goodfellow et al., "Deep Learning" Chapter 2 (2016)

[← 第1章：確率統計の基礎](<./chapter1-probability-statistics.html>) [第3章：最適化理論 →](<./chapter3-optimization.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
