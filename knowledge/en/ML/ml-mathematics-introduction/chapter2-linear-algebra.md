---
title: "Chapter 2: Fundamentals of Linear Algebra"
chapter_title: "Chapter 2: Fundamentals of Linear Algebra"
---

This chapter covers the fundamentals of Fundamentals of Linear Algebra, which 1. fundamentals of vectors and matrices. You will learn essential concepts and techniques.

**Deeply understand linear algebra, the mathematical foundation of machine learning algorithms, through both theory and implementation**

**What You'll Learn in This Chapter**

  * Basic operations and geometric meanings of vectors and matrices
  * Theory and implementation of eigenvalue decomposition, SVD, and QR decomposition
  * Mathematical principles and applications of Principal Component Analysis (PCA)
  * Geometric understanding of linear transformations and projections
  * Application of linear algebra to linear regression and Ridge regression

## 1\. Fundamentals of Vectors and Matrices

### 1.1 Inner Product and Norm of Vectors

The inner product (dot product) of vectors is a fundamental operation that measures the similarity between two vectors.

$$\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T\mathbf{y} = \sum_{i=1}^{n} x_i y_i = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$$ 

Here, θ is the angle between the vectors. Geometric meaning of the inner product:

  * **Positive** : Pointing in the same direction (acute angle)
  * **Zero** : Orthogonal (perpendicular)
  * **Negative** : Pointing in opposite directions (obtuse angle)

The norm (length) of a vector represents the magnitude of the vector.

$$\|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^T\mathbf{x}} = \sqrt{\sum_{i=1}^{n} x_i^2} \quad \text{(L2 norm)}$$ $$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i| \quad \text{(L1 norm)}$$ 

**Applications in Machine Learning** The L2 norm is used as the regularization term in Ridge regression, while the L1 norm is used in Lasso regression. Cosine similarity is frequently used in document classification and recommendation systems. 

### 1.2 Basic Matrix Operations

Matrix multiplication is an operation that composes linear transformations.

$$(\mathbf{AB})_{ij} = \sum_{k=1}^{m} A_{ik}B_{kj}$$ 

**Important Properties:**

  * Associativity: \\((\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})\\)
  * Distributivity: \\(\mathbf{A}(\mathbf{B}+\mathbf{C}) = \mathbf{AB} + \mathbf{AC}\\)
  * Transpose: \\((\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T\\)
  * Non-commutativity: In general \\(\mathbf{AB} \neq \mathbf{BA}\\)

### Implementation Example 1: Vector and Matrix Operations
    
    
    import numpy as np
    
    class LinearAlgebraOps:
        """Class implementing basic linear algebra operations"""
    
        @staticmethod
        def inner_product(x, y):
            """
            Compute inner product: x·y = Σ x_i * y_i
    
            Parameters:
            -----------
            x, y : array-like
                Input vectors
    
            Returns:
            --------
            float : inner product
            """
            x, y = np.array(x), np.array(y)
            assert x.shape == y.shape, "Vector dimensions must match"
            return np.sum(x * y)
    
        @staticmethod
        def cosine_similarity(x, y):
            """
            Cosine similarity: cos(θ) = (x·y) / (||x|| * ||y||)
    
            Returns:
            --------
            float : similarity in range [-1, 1]
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
            Lp norm of vector
    
            Parameters:
            -----------
            x : array-like
                Input vector
            p : int or float
                Order of norm (1, 2, np.inf, etc.)
    
            Returns:
            --------
            float : norm
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
            Matrix multiplication implementation: C = AB
    
            Parameters:
            -----------
            A : ndarray of shape (m, n)
            B : ndarray of shape (n, p)
    
            Returns:
            --------
            ndarray of shape (m, p)
            """
            A, B = np.array(A), np.array(B)
            assert A.shape[1] == B.shape[0], f"Incompatible matrix shapes: {A.shape} and {B.shape}"
    
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
            Outer product (tensor product): xy^T
    
            Returns:
            --------
            ndarray : matrix
            """
            x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
            return x @ y.T
    
    # Usage example
    ops = LinearAlgebraOps()
    
    # Vector operations
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print("Vector Operations:")
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"Inner product: {ops.inner_product(v1, v2)}")
    print(f"Cosine similarity: {ops.cosine_similarity(v1, v2):.4f}")
    print(f"L1 norm: {ops.vector_norm(v1, p=1):.4f}")
    print(f"L2 norm: {ops.vector_norm(v1, p=2):.4f}")
    
    # Matrix operations
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"\nMatrix multiplication:")
    print(f"A @ B =\n{ops.matrix_multiply(A, B)}")
    print(f"NumPy verification:\n{A @ B}")
    
    # Outer product
    print(f"\nOuter product v1 ⊗ v2 =\n{ops.outer_product(v1, v2)}")
    

## 2\. Matrix Decomposition

### 2.1 Eigendecomposition

For a square matrix A, the eigenvalue λ and eigenvector v satisfy:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$ 

A symmetric matrix can be diagonalized as follows:

$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$$ 

Where:

  * **Q** : Orthogonal matrix with eigenvectors as columns
  * **Λ** : Diagonal matrix with eigenvalues on the diagonal

**Geometric Meaning** Eigenvectors are vectors whose direction doesn't change under matrix transformation, and eigenvalues represent the scaling factor. Larger eigenvalues indicate greater variance in that direction. 

### 2.2 Singular Value Decomposition (SVD)

Any matrix A can be decomposed as:

$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$ 

Where:

  * **U** : Left singular vectors (m×m orthogonal matrix)
  * **Σ** : Diagonal matrix of singular values (m×n)
  * **V** : Right singular vectors (n×n orthogonal matrix)

Singular values are arranged in descending order: σ₁ ≥ σ₂ ≥ ... ≥ 0.

**Applications in Machine Learning** SVD is widely used in Principal Component Analysis (PCA), recommendation systems (collaborative filtering), natural language processing (LSA), image compression, and more. 

### 2.3 QR Decomposition

Any matrix A can be decomposed into the product of an orthogonal matrix Q and an upper triangular matrix R:

$$\mathbf{A} = \mathbf{QR}$$ 

QR decomposition is used as a numerically stable solution method for least squares.

### Implementation Example 2: Matrix Decomposition Implementation and Comparison
    
    
    import numpy as np
    from scipy import linalg
    
    class MatrixDecomposition:
        """Implementation and application of matrix decomposition"""
    
        @staticmethod
        def eigen_decomposition(A, symmetric=True):
            """
            Eigendecomposition: A = QΛQ^T
    
            Parameters:
            -----------
            A : ndarray of shape (n, n)
                Matrix to decompose
            symmetric : bool
                Whether matrix is symmetric
    
            Returns:
            --------
            eigenvalues : ndarray
                Eigenvalues (descending order)
            eigenvectors : ndarray
                Corresponding eigenvectors
            """
            A = np.array(A)
            assert A.shape[0] == A.shape[1], "Matrix must be square"
    
            if symmetric:
                eigenvalues, eigenvectors = np.linalg.eigh(A)
            else:
                eigenvalues, eigenvectors = np.linalg.eig(A)
    
            # Sort by descending eigenvalues
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
    
            return eigenvalues, eigenvectors
    
        @staticmethod
        def svd_decomposition(A, full_matrices=False):
            """
            Singular Value Decomposition: A = UΣV^T
    
            Parameters:
            -----------
            A : ndarray of shape (m, n)
                Matrix to decompose
            full_matrices : bool
                Whether to return full matrices
    
            Returns:
            --------
            U : ndarray of shape (m, m) or (m, k)
                Left singular vectors
            S : ndarray of shape (k,)
                Singular values (descending order)
            Vt : ndarray of shape (n, n) or (k, n)
                Transpose of right singular vectors
            """
            A = np.array(A)
            U, S, Vt = np.linalg.svd(A, full_matrices=full_matrices)
            return U, S, Vt
    
        @staticmethod
        def qr_decomposition(A):
            """
            QR decomposition: A = QR
    
            Parameters:
            -----------
            A : ndarray of shape (m, n)
                Matrix to decompose
    
            Returns:
            --------
            Q : ndarray of shape (m, m)
                Orthogonal matrix
            R : ndarray of shape (m, n)
                Upper triangular matrix
            """
            A = np.array(A)
            Q, R = np.linalg.qr(A)
            return Q, R
    
        @staticmethod
        def low_rank_approximation(A, k):
            """
            Low-rank approximation using SVD
    
            Parameters:
            -----------
            A : ndarray of shape (m, n)
                Original matrix
            k : int
                Rank of approximation
    
            Returns:
            --------
            A_approx : ndarray
                Rank-k approximation matrix
            reconstruction_error : float
                Reconstruction error in Frobenius norm
            """
            U, S, Vt = MatrixDecomposition.svd_decomposition(A, full_matrices=False)
    
            # Use only top k singular values
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]
    
            # Low-rank approximation
            A_approx = U_k @ np.diag(S_k) @ Vt_k
    
            # Reconstruction error
            reconstruction_error = np.linalg.norm(A - A_approx, 'fro')
    
            return A_approx, reconstruction_error
    
    # Usage example
    decomp = MatrixDecomposition()
    
    # Eigendecomposition of symmetric matrix
    print("=" * 60)
    print("Eigendecomposition")
    print("=" * 60)
    A_sym = np.array([[4, 2], [2, 3]])
    eigenvalues, eigenvectors = decomp.eigen_decomposition(A_sym)
    print(f"Original matrix A:\n{A_sym}\n")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}\n")
    
    # Verify reconstruction
    Lambda = np.diag(eigenvalues)
    A_reconstructed = eigenvectors @ Lambda @ eigenvectors.T
    print(f"Reconstruction A = QΛQ^T:\n{A_reconstructed}")
    print(f"Reconstruction error: {np.linalg.norm(A_sym - A_reconstructed):.10f}\n")
    
    # SVD
    print("=" * 60)
    print("Singular Value Decomposition (SVD)")
    print("=" * 60)
    A = np.array([[1, 2, 3], [4, 5, 6]])
    U, S, Vt = decomp.svd_decomposition(A, full_matrices=True)
    print(f"Original matrix A ({A.shape}):\n{A}\n")
    print(f"U ({U.shape}):\n{U}\n")
    print(f"Singular values S: {S}\n")
    print(f"V^T ({Vt.shape}):\n{Vt}\n")
    
    # Reconstruction
    Sigma = np.zeros_like(A, dtype=float)
    Sigma[:len(S), :len(S)] = np.diag(S)
    A_reconstructed = U @ Sigma @ Vt
    print(f"Reconstruction A = UΣV^T:\n{A_reconstructed}")
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.10f}\n")
    
    # Low-rank approximation
    print("=" * 60)
    print("Low-Rank Approximation")
    print("=" * 60)
    A_large = np.random.randn(10, 8)
    for k in [1, 2, 4, 8]:
        A_approx, error = decomp.low_rank_approximation(A_large, k)
        compression_ratio = (k * (A_large.shape[0] + A_large.shape[1])) / A_large.size
        print(f"Rank {k}: Reconstruction error = {error:.4f}, Compression ratio = {compression_ratio:.2%}")
    
    # QR decomposition
    print("\n" + "=" * 60)
    print("QR Decomposition")
    print("=" * 60)
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    Q, R = decomp.qr_decomposition(A)
    print(f"Original matrix A:\n{A}\n")
    print(f"Q (orthogonal matrix):\n{Q}\n")
    print(f"R (upper triangular matrix):\n{R}\n")
    print(f"Q^T Q (identity matrix):\n{Q.T @ Q}")
    print(f"Reconstruction A = QR:\n{Q @ R}")
    

## 3\. Principal Component Analysis (PCA)

### 3.1 Mathematical Formulation of PCA

Principal Component Analysis is a dimensionality reduction technique that finds orthogonal axes maximizing data variance.

**Objective:** Project data matrix X (n×d) onto lower-dimensional space (n×k, k < d)

$$\max_{\mathbf{w}} \mathbf{w}^T\mathbf{S}\mathbf{w} \quad \text{s.t.} \quad \|\mathbf{w}\|^2 = 1$$ 

Where S is the covariance matrix:

$$\mathbf{S} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T = \frac{1}{n}\mathbf{X}_c^T\mathbf{X}_c$$ 

**Solution:** Eigendecomposition of the covariance matrix

$$\mathbf{S} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$$ 

Principal components are eigenvectors corresponding to the largest eigenvalues.

### 3.2 Steps of PCA

  1. **Centering** : Subtract mean from data
  2. **Compute covariance matrix** : S = (1/n)X_c^T X_c
  3. **Eigendecomposition** : Compute eigenvalues and eigenvectors
  4. **Select principal components** : Choose top k by eigenvalue
  5. **Projection** : Transform data to principal component space

**Variance Ratio and Cumulative Variance Ratio** The variance ratio of the i-th principal component = λ_i / Σλ_j represents the proportion of variance explained by that component. It's common to choose the number of dimensions where the cumulative variance ratio reaches 90% or more. 

### Implementation Example 3: Complete Implementation of PCA
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class PCA:
        """Principal Component Analysis implementation"""
    
        def __init__(self, n_components=None):
            """
            Parameters:
            -----------
            n_components : int or None
                Number of principal components to retain (None for all)
            """
            self.n_components = n_components
            self.components_ = None
            self.explained_variance_ = None
            self.explained_variance_ratio_ = None
            self.mean_ = None
    
        def fit(self, X):
            """
            Execute principal component analysis
    
            Parameters:
            -----------
            X : ndarray of shape (n_samples, n_features)
                Input data
    
            Returns:
            --------
            self
            """
            X = np.array(X)
            n_samples, n_features = X.shape
    
            # 1. Center the data
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
    
            # 2. Compute covariance matrix
            cov_matrix = (X_centered.T @ X_centered) / n_samples
    
            # 3. Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
            # 4. Sort by descending eigenvalues
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
    
            # 5. Select principal components
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
            Transform data to principal component space
    
            Parameters:
            -----------
            X : ndarray of shape (n_samples, n_features)
                Input data
    
            Returns:
            --------
            ndarray of shape (n_samples, n_components)
                Transformed data
            """
            X = np.array(X)
            X_centered = X - self.mean_
            return X_centered @ self.components_.T
    
        def fit_transform(self, X):
            """Execute fit and transform simultaneously"""
            return self.fit(X).transform(X)
    
        def inverse_transform(self, X_transformed):
            """
            Transform back from principal component space to original space
    
            Parameters:
            -----------
            X_transformed : ndarray of shape (n_samples, n_components)
                Transformed data
    
            Returns:
            --------
            ndarray of shape (n_samples, n_features)
                Reconstructed data
            """
            return X_transformed @ self.components_ + self.mean_
    
        def reconstruction_error(self, X):
            """
            Compute reconstruction error
    
            Returns:
            --------
            float : Mean squared reconstruction error
            """
            X_transformed = self.transform(X)
            X_reconstructed = self.inverse_transform(X_transformed)
            return np.mean((X - X_reconstructed) ** 2)
    
    # Usage example: Visualization with 2D data
    np.random.seed(42)
    
    # Generate 2D data with correlation
    mean = [0, 0]
    cov = [[3, 1.5], [1.5, 1]]
    X = np.random.multivariate_normal(mean, cov, 300)
    
    # Execute PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    
    print("=" * 60)
    print("PCA Results")
    print("=" * 60)
    print(f"Principal components (eigenvectors):\n{pca.components_}")
    print(f"Explained variance (eigenvalues): {pca.explained_variance_}")
    print(f"Variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance ratio: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original data with principal component axes
    ax1 = axes[0]
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5, s=30)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Original Data with Principal Component Axes')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Draw principal component axes
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        ax1.arrow(0, 0, comp[0]*np.sqrt(var)*3, comp[1]*np.sqrt(var)*3,
                 head_width=0.3, head_length=0.2, fc=f'C{i+1}', ec=f'C{i+1}',
                 linewidth=2, label=f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})')
    ax1.legend()
    
    # Data in principal component space
    ax2 = axes[1]
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=30)
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    ax2.set_title('Data in Principal Component Space')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=150, bbox_inches='tight')
    print("\nPCA visualization saved")
    
    # Effect of dimensionality reduction
    print("\n" + "=" * 60)
    print("Effect of Dimensionality Reduction")
    print("=" * 60)
    for n_comp in [1, 2]:
        pca_reduced = PCA(n_components=n_comp)
        pca_reduced.fit(X)
        error = pca_reduced.reconstruction_error(X)
        cum_var = np.sum(pca_reduced.explained_variance_ratio_)
        print(f"{n_comp} dimensions: Cumulative variance ratio={cum_var:.2%}, Reconstruction error={error:.4f}")
    

## 4\. Linear Transformations and Projections

### 4.1 Geometry of Linear Transformations

Linear transformation is an operation that transforms vectors by a matrix A:

$$\mathbf{y} = \mathbf{A}\mathbf{x}$$ 

**Representative linear transformations:**

  * **Rotation** : Transformation by orthogonal matrix (preserves length)
  * **Scaling** : Transformation by diagonal matrix
  * **Shear** : Transformation with off-diagonal components
  * **Projection** : Projection onto subspace

### 4.2 Projection Matrix

The projection matrix that projects vector b onto the column space C(A) is:

$$\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$$ 

The projection vector is p = Pb, and the residual is e = b - p.

**Properties of projection matrix:**

  * Symmetry: P^T = P
  * Idempotence: P² = P
  * Orthogonality of residual: A^T(b - Pb) = 0

**Relationship with Least Squares** The least squares solution of linear regression is obtained by projecting y onto the column space of X. This ensures that the residual is orthogonal to the column space. 

### Implementation Example 4: Visualization of Linear Transformations and Projections
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class LinearTransformation:
        """Implementation of linear transformations and projections"""
    
        @staticmethod
        def rotation_matrix(theta):
            """
            2D rotation matrix
    
            Parameters:
            -----------
            theta : float
                Rotation angle (radians)
    
            Returns:
            --------
            ndarray : 2x2 rotation matrix
            """
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s], [s, c]])
    
        @staticmethod
        def scaling_matrix(sx, sy):
            """
            2D scaling matrix
    
            Parameters:
            -----------
            sx, sy : float
                Scale in x and y directions
    
            Returns:
            --------
            ndarray : 2x2 scaling matrix
            """
            return np.array([[sx, 0], [0, sy]])
    
        @staticmethod
        def projection_matrix(A):
            """
            Projection matrix onto column space: P = A(A^T A)^(-1)A^T
    
            Parameters:
            -----------
            A : ndarray of shape (m, n)
                Matrix with basis as columns
    
            Returns:
            --------
            ndarray of shape (m, m) : projection matrix
            """
            A = np.array(A)
            return A @ np.linalg.inv(A.T @ A) @ A.T
    
        @staticmethod
        def project_onto_subspace(b, A):
            """
            Project vector b onto column space of A
    
            Parameters:
            -----------
            b : ndarray
                Vector to project
            A : ndarray
                Matrix spanning the subspace
    
            Returns:
            --------
            projection : ndarray
                Projection vector
            residual : ndarray
                Residual vector
            """
            P = LinearTransformation.projection_matrix(A)
            projection = P @ b
            residual = b - projection
            return projection, residual
    
    # Visualization example: Linear transformations
    print("=" * 60)
    print("Visualization of Linear Transformations")
    print("=" * 60)
    
    # Vertices of unit square
    square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])
    
    # Various transformations
    transformations = {
        'Rotation (45°)': LinearTransformation.rotation_matrix(np.pi/4),
        'Scaling (2, 0.5)': LinearTransformation.scaling_matrix(2, 0.5),
        'Shear': np.array([[1, 0.5], [0, 1]]),
        'Composite': LinearTransformation.rotation_matrix(np.pi/6) @ \
                    LinearTransformation.scaling_matrix(1.5, 0.8)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, (name, A) in enumerate(transformations.items()):
        ax = axes[idx]
    
        # Original shape
        ax.plot(square[0], square[1], 'b-', linewidth=2, label='Original')
        ax.fill(square[0], square[1], 'blue', alpha=0.2)
    
        # Transformed shape
        transformed = A @ square
        ax.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed')
        ax.fill(transformed[0], transformed[1], 'red', alpha=0.2)
    
        # Transformation of basis vectors
        basis = np.array([[1, 0], [0, 1]]).T
        transformed_basis = A @ basis
        for i in range(2):
            ax.arrow(0, 0, transformed_basis[0, i], transformed_basis[1, i],
                    head_width=0.1, head_length=0.1, fc=f'C{i+2}', ec=f'C{i+2}',
                    linewidth=2)
    
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name}\nDeterminant: {np.linalg.det(A):.2f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 3)
    
    plt.tight_layout()
    plt.savefig('linear_transformations.png', dpi=150, bbox_inches='tight')
    print("Linear transformation visualization saved")
    
    # Projection example
    print("\n" + "=" * 60)
    print("Projection Calculation")
    print("=" * 60)
    
    # Projection onto 1D subspace in 2D
    a = np.array([[1], [2]])  # Subspace basis
    b = np.array([3, 2])      # Vector to project
    
    proj, resid = LinearTransformation.project_onto_subspace(b, a)
    
    print(f"Basis vector a: {a.flatten()}")
    print(f"Vector b: {b}")
    print(f"Projection p: {proj}")
    print(f"Residual e: {resid}")
    print(f"Inner product a^T e (orthogonality check): {a.T @ resid}")
    print(f"||b||²: {np.linalg.norm(b)**2:.4f}")
    print(f"||p||² + ||e||²: {np.linalg.norm(proj)**2 + np.linalg.norm(resid)**2:.4f}")
    
    # Visualization of projection
    plt.figure(figsize=(8, 8))
    plt.arrow(0, 0, b[0], b[1], head_width=0.2, head_length=0.2,
             fc='blue', ec='blue', linewidth=2, label='Original vector b')
    plt.arrow(0, 0, proj[0], proj[1], head_width=0.2, head_length=0.2,
             fc='green', ec='green', linewidth=2, label='Projection p')
    plt.arrow(0, 0, a[0, 0]*2, a[1, 0]*2, head_width=0.2, head_length=0.2,
             fc='red', ec='red', linewidth=2, linestyle='--', label='Subspace basis')
    plt.plot([proj[0], b[0]], [proj[1], b[1]], 'k--', linewidth=1, label='Residual e')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Projection of Vector onto Subspace')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.xlim(-1, 4)
    plt.ylim(-1, 5)
    plt.savefig('projection_visualization.png', dpi=150, bbox_inches='tight')
    print("Projection visualization saved")
    

## 5\. Practical Applications

### 5.1 Linear Algebraic Solution to Linear Regression

The objective of linear regression is to find parameters w that minimize the least squares error:

$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{Xw}\|^2$$ 

Solution by normal equations:

$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$ 

This is equivalent to projecting y onto the column space of X.

### 5.2 Ridge Regression (L2 Regularization)

Ridge regression adds an L2 regularization term to prevent overfitting:

$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{Xw}\|^2 + \lambda\|\mathbf{w}\|^2$$ 

The solution takes the form:

$$\mathbf{w}_{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$ 

The larger λ is, the more the magnitude of parameters is constrained.

### Implementation Example 5: Implementation of Linear Regression and Ridge Regression
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class LinearRegression:
        """Linear regression implementation (using matrix operations)"""
    
        def __init__(self, fit_intercept=True):
            """
            Parameters:
            -----------
            fit_intercept : bool
                Whether to include intercept
            """
            self.fit_intercept = fit_intercept
            self.coef_ = None
            self.intercept_ = None
    
        def fit(self, X, y):
            """
            Compute least squares solution using normal equations: w = (X^T X)^(-1) X^T y
    
            Parameters:
            -----------
            X : ndarray of shape (n_samples, n_features)
                Feature matrix
            y : ndarray of shape (n_samples,)
                Target variable
    
            Returns:
            --------
            self
            """
            X, y = np.array(X), np.array(y).reshape(-1, 1)
    
            if self.fit_intercept:
                # Add intercept term
                X = np.hstack([np.ones((X.shape[0], 1)), X])
    
            # Normal equations: (X^T X) w = X^T y
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
            """Prediction"""
            X = np.array(X)
            return X @ self.coef_ + self.intercept_
    
    class RidgeRegression:
        """Ridge regression implementation (L2 regularization)"""
    
        def __init__(self, alpha=1.0, fit_intercept=True):
            """
            Parameters:
            -----------
            alpha : float
                Regularization parameter (λ)
            fit_intercept : bool
                Whether to include intercept
            """
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            self.coef_ = None
            self.intercept_ = None
    
        def fit(self, X, y):
            """
            Compute Ridge regression solution: w = (X^T X + λI)^(-1) X^T y
    
            Parameters:
            -----------
            X : ndarray of shape (n_samples, n_features)
                Feature matrix
            y : ndarray of shape (n_samples,)
                Target variable
    
            Returns:
            --------
            self
            """
            X, y = np.array(X), np.array(y).reshape(-1, 1)
    
            if self.fit_intercept:
                X = np.hstack([np.ones((X.shape[0], 1)), X])
    
            # Ridge regression solution
            n_features = X.shape[1]
            ridge_matrix = X.T @ X + self.alpha * np.eye(n_features)
    
            # Don't regularize intercept
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
            """Prediction"""
            X = np.array(X)
            return X @ self.coef_ + self.intercept_
    
    # Usage example and comparison with QR decomposition solution
    def solve_with_qr(X, y):
        """Numerically stable least squares solution using QR decomposition"""
        Q, R = np.linalg.qr(X)
        return np.linalg.solve(R, Q.T @ y)
    
    # Data generation
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1)
    y_true = 3 * X.squeeze() + 2
    y = y_true + np.random.randn(n_samples) * 0.5
    
    # Linear regression
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    
    print("=" * 60)
    print("Linear Regression Results")
    print("=" * 60)
    print(f"Coefficient: {lr.coef_}")
    print(f"Intercept: {lr.intercept_:.4f}")
    print(f"MSE: {np.mean((y - y_pred_lr)**2):.4f}")
    
    # Ridge regression (compare different α)
    alphas = [0.01, 0.1, 1.0, 10.0]
    ridge_models = []
    
    print("\n" + "=" * 60)
    print("Ridge Regression Results")
    print("=" * 60)
    
    for alpha in alphas:
        ridge = RidgeRegression(alpha=alpha)
        ridge.fit(X, y)
        y_pred = ridge.predict(X)
        mse = np.mean((y - y_pred)**2)
        ridge_models.append(ridge)
        print(f"α={alpha:5.2f}: Coefficient={ridge.coef_[0]:6.3f}, "
              f"Intercept={ridge.intercept_:6.3f}, MSE={mse:.4f}")
    
    # Visualization
    plt.figure(figsize=(14, 5))
    
    # Left plot: Linear regression
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, s=30, label='Data')
    plt.plot(X, y_true, 'g--', linewidth=2, label='True function')
    plt.plot(X, y_pred_lr, 'r-', linewidth=2, label='Linear regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right plot: Ridge regression comparison
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, alpha=0.5, s=30, label='Data')
    plt.plot(X, y_true, 'g--', linewidth=2, label='True function')
    X_sorted = np.sort(X, axis=0)
    for ridge, alpha in zip(ridge_models, alphas):
        y_line = ridge.predict(X_sorted)
        plt.plot(X_sorted, y_line, linewidth=2, label=f'Ridge (α={alpha})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ridge Regression (Effect of Regularization Parameter)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_ridge_regression.png', dpi=150, bbox_inches='tight')
    print("\nRegression results visualization saved")
    
    # Example of multicollinearity
    print("\n" + "=" * 60)
    print("Multicollinearity Example")
    print("=" * 60)
    
    # Generate highly correlated features
    X_corr = np.random.randn(50, 1)
    X_multi = np.hstack([X_corr, X_corr + np.random.randn(50, 1) * 0.1, X_corr * 2])
    y_multi = X_corr.squeeze() + np.random.randn(50) * 0.5
    
    # Linear regression (unstable)
    lr_multi = LinearRegression()
    lr_multi.fit(X_multi, y_multi)
    
    # Ridge regression (stable)
    ridge_multi = RidgeRegression(alpha=1.0)
    ridge_multi.fit(X_multi, y_multi)
    
    print("Linear regression coefficients:", lr_multi.coef_)
    print("Ridge regression coefficients:", ridge_multi.coef_)
    print("L2 norm of coefficients:")
    print(f"  Linear regression: {np.linalg.norm(lr_multi.coef_):.4f}")
    print(f"  Ridge regression: {np.linalg.norm(ridge_multi.coef_):.4f}")
    

### 5.3 Applying PCA to Image Data

PCA is widely used for dimensionality reduction and compression of images, treating each pixel as a feature.

### Implementation Example 6: Applying PCA/SVD to Image Compression
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class ImageCompressionPCA:
        """PCA/SVD implementation for image compression"""
    
        @staticmethod
        def compress_with_svd(image, n_components):
            """
            Image compression using SVD
    
            Parameters:
            -----------
            image : ndarray of shape (height, width)
                Grayscale image
            n_components : int
                Number of singular values to retain
    
            Returns:
            --------
            compressed : ndarray
                Compressed image
            compression_ratio : float
                Compression ratio
            """
            # SVD decomposition
            U, S, Vt = np.linalg.svd(image, full_matrices=False)
    
            # Use only top n_components
            U_k = U[:, :n_components]
            S_k = S[:n_components]
            Vt_k = Vt[:n_components, :]
    
            # Reconstruction
            compressed = U_k @ np.diag(S_k) @ Vt_k
    
            # Calculate compression ratio
            original_size = image.shape[0] * image.shape[1]
            compressed_size = n_components * (image.shape[0] + image.shape[1] + 1)
            compression_ratio = compressed_size / original_size
    
            return compressed, compression_ratio
    
        @staticmethod
        def analyze_singular_values(image):
            """
            Analyze singular values
    
            Returns:
            --------
            singular_values : ndarray
                Singular values
            cumulative_energy : ndarray
                Cumulative energy
            """
            _, S, _ = np.linalg.svd(image, full_matrices=False)
    
            # Energy (square of each singular value)
            energy = S ** 2
            total_energy = np.sum(energy)
            cumulative_energy = np.cumsum(energy) / total_energy
    
            return S, cumulative_energy
    
    # Usage example: Experiment with synthetic image
    print("=" * 60)
    print("Image Compression Experiment")
    print("=" * 60)
    
    # Generate synthetic image (gradient and pattern)
    height, width = 200, 200
    x = np.linspace(-5, 5, width)
    y = np.linspace(-5, 5, height)
    X, Y = np.meshgrid(x, y)
    
    # Image with complex pattern
    image = (np.sin(X) * np.cos(Y) +
             0.5 * np.sin(2*X + Y) +
             0.3 * np.cos(X - 2*Y))
    image = (image - image.min()) / (image.max() - image.min())  # Normalize
    
    # Analyze singular values
    compressor = ImageCompressionPCA()
    singular_values, cumulative_energy = compressor.analyze_singular_values(image)
    
    print(f"Image size: {image.shape}")
    print(f"Total singular values: {len(singular_values)}")
    print(f"Components needed for 90% energy: {np.argmax(cumulative_energy >= 0.90) + 1}")
    print(f"Components needed for 99% energy: {np.argmax(cumulative_energy >= 0.99) + 1}")
    
    # Compare different compression ratios
    n_components_list = [5, 10, 20, 50]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Compressed images
    for idx, n_comp in enumerate(n_components_list, 1):
        compressed, comp_ratio = compressor.compress_with_svd(image, n_comp)
    
        axes[idx].imshow(compressed, cmap='gray')
    
        # Calculate PSNR
        mse = np.mean((image - compressed) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
        energy_retained = cumulative_energy[n_comp - 1]
    
        axes[idx].set_title(f'Components: {n_comp}\n'
                           f'Compression ratio: {comp_ratio:.1%}\n'
                           f'PSNR: {psnr:.1f}dB\n'
                           f'Energy: {energy_retained:.1%}')
        axes[idx].axis('off')
    
    # Plot singular value decay
    axes[5].plot(singular_values[:100], 'b-', linewidth=2)
    axes[5].set_xlabel('Component Number')
    axes[5].set_ylabel('Singular Value')
    axes[5].set_title('Singular Value Decay')
    axes[5].grid(True, alpha=0.3)
    axes[5].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('image_compression_pca.png', dpi=150, bbox_inches='tight')
    print("\nImage compression visualization saved")
    
    # Plot cumulative energy
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_energy[:100], linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% energy')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% energy')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Energy')
    plt.title('Cumulative Energy of SVD Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cumulative_energy.png', dpi=150, bbox_inches='tight')
    print("Cumulative energy plot saved")
    
    # Analyze practical compression ratios
    print("\n" + "=" * 60)
    print("Relationship between Compression Ratio and PSNR")
    print("=" * 60)
    print(f"{'Components':>8} {'Compression':>10} {'PSNR (dB)':>12} {'Energy':>12}")
    print("-" * 60)
    
    for n_comp in [1, 2, 5, 10, 20, 50, 100]:
        if n_comp <= min(image.shape):
            compressed, comp_ratio = compressor.compress_with_svd(image, n_comp)
            mse = np.mean((image - compressed) ** 2)
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            energy = cumulative_energy[n_comp - 1]
            print(f"{n_comp:8d} {comp_ratio:9.1%} {psnr:11.2f} {energy:11.1%}")
    

## Summary

In this chapter, we learned linear algebra, the mathematical foundation of machine learning.

**What We Learned**

  * **Vectors and Matrices** : Geometric meaning of inner products, norms, and matrix operations
  * **Matrix Decomposition** : Theory and implementation of eigendecomposition, SVD, and QR decomposition
  * **Principal Component Analysis** : Dimensionality reduction technique that maximizes data variance
  * **Linear Transformations and Projections** : Geometric understanding of least squares
  * **Practical Applications** : Linear regression, Ridge regression, image compression

**Preparation for Next Chapter** In Chapter 3, we will learn optimization theory. While the normal equations for linear regression are analytical solutions to optimization problems, in the next chapter we will learn numerical solution methods such as gradient descent and apply them to neural network training. 

### Comparison of Matrix Decompositions

Decomposition | Form | Target Matrix | Main Applications  
---|---|---|---  
Eigendecomposition | A = QΛQ^T | Square symmetric matrix | PCA, graph analysis  
SVD | A = UΣV^T | Any matrix | Dimensionality reduction, recommendation systems  
QR decomposition | A = QR | Any matrix | Least squares, eigenvalue computation  
Cholesky decomposition | A = LL^T | Positive definite symmetric matrix | Linear systems, Gaussian processes  
  
### Exercise Problems

  1. Verify what happens to cosine similarity when two vectors are orthogonal
  2. Create a 3×3 symmetric matrix, perform eigendecomposition, and verify the reconstruction error
  3. Perform SVD on a random 5×3 matrix and create a rank-2 approximation
  4. Execute PCA on 3D data, reduce to 2D, and visualize
  5. Compare the regularization effect of Ridge regression with polynomial regression (2nd and 3rd order)
  6. Apply SVD to actual image data (grayscale) and find the optimal compression ratio

### References

  * G. Strang, "Linear Algebra and Its Applications" (2016)
  * L.N. Trefethen and D. Bau, "Numerical Linear Algebra" (1997)
  * Masahiko Saito, "Introduction to Linear Algebra" University of Tokyo Press (1966)
  * I. Goodfellow et al., "Deep Learning" Chapter 2 (2016)

[← Chapter 1: Probability and Statistics Fundamentals](<./chapter1-probability-statistics.html>) [Chapter 3: Optimization Theory →](<./chapter3-optimization.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
