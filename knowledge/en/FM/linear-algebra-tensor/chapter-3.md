---
title: "Chapter 3: Eigenvalues, Eigenvectors, and Diagonalization"
chapter_title: "Chapter 3: Eigenvalues, Eigenvectors, and Diagonalization"
subtitle: Eigenvalues, Eigenvectors, and Diagonalization
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/linear-algebra-tensor/chapter-3.html>) | Last sync: 2025-11-16

[Fundamentals of Mathematics Dojo](<../index.html>) > [Linear Algebra and Tensor Analysis](<index.html>) > Chapter 3 

## 3.1 Definition of Eigenvalues and Eigenvectors

**Definition: Eigenvalues and Eigenvectors**  
For square matrix A, non-zero vector v satisfying Av = λv is called an eigenvector, and scalar λ is called an eigenvalue.  
Geometric meaning: eigenvectors are vectors whose direction doesn't change under matrix transformation 

### Code Example 1: Calculating Eigenvalues and Eigenvectors

Python Implementation: Calculating Eigenvalues and Eigenvectors
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 1: Calculating Eigenvalues and Eigenvectors
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 2×2 matrix
    A = np.array([[4, 1],
                  [2, 3]])
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("Eigenvalue and Eigenvector Calculation:")
    print(f"A =\n{A}\n")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}\n")
    
    # Verification: Av = λv
    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]
        Av = A @ v
        lam_v = lam * v
        print(f"λ{i+1} = {lam:.4f}")
        print(f"  Av = {Av}")
        print(f"  λv = {lam_v}")
        print(f"  Av = λv? {np.allclose(Av, lam_v)}\n")

### Code Example 2: Geometric Meaning of Eigenvectors

Python Implementation: Visualizing Eigenvectors
    
    
    # Visualize eigenvectors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Random vector and eigenvectors
    random_vec = np.array([1, 1])
    eigen_vec1 = eigenvectors[:, 0]
    eigen_vec2 = eigenvectors[:, 1]
    
    # Vectors before transformation
    origin = [0, 0]
    ax1.quiver(*origin, *random_vec, angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.01, label='General Vector')
    ax1.quiver(*origin, *eigen_vec1, angles='xy', scale_units='xy', scale=1,
               color='red', width=0.01, label=f'Eigenvector 1 (λ={eigenvalues[0]:.2f})')
    ax1.quiver(*origin, *eigen_vec2, angles='xy', scale_units='xy', scale=1,
               color='green', width=0.01, label=f'Eigenvector 2 (λ={eigenvalues[1]:.2f})')
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-1, 5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Before Transformation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Vectors after transformation
    random_transformed = A @ random_vec
    eigen_transformed1 = A @ eigen_vec1
    eigen_transformed2 = A @ eigen_vec2
    
    ax2.quiver(*origin, *random_transformed, angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.01, label='General Vector (direction changed)', alpha=0.7)
    ax2.quiver(*origin, *eigen_transformed1, angles='xy', scale_units='xy', scale=1,
               color='red', width=0.01, label='Eigenvector 1 (direction unchanged)')
    ax2.quiver(*origin, *eigen_transformed2, angles='xy', scale_units='xy', scale=1,
               color='green', width=0.01, label='Eigenvector 2 (direction unchanged)')
    
    # Show original vectors faintly
    ax2.quiver(*origin, *random_vec, angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.005, alpha=0.2)
    ax2.quiver(*origin, *eigen_vec1, angles='xy', scale_units='xy', scale=1,
               color='red', width=0.005, alpha=0.2)
    ax2.quiver(*origin, *eigen_vec2, angles='xy', scale_units='xy', scale=1,
               color='green', width=0.005, alpha=0.2)
    
    ax2.set_xlim(-1, 5)
    ax2.set_ylim(-1, 5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('After Transformation by Matrix A')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

## 3.2 Characteristic Equation and Eigenvalue Calculation

**Theorem: Characteristic Equation**  
Eigenvalue λ is found as the solution of the characteristic equation det(A - λI) = 0.  
An n×n matrix has n eigenvalues (including multiplicities). 

### Code Example 3: Deriving Characteristic Equation with SymPy

Python Implementation: Symbolic Calculation of Characteristic Equation
    
    
    import sympy as sp
    
    # Symbolic variable
    lam = sp.Symbol('lambda')
    
    # Define matrix A symbolically
    A_sym = sp.Matrix([[4, 1],
                       [2, 3]])
    
    # I - identity matrix
    I = sp.eye(2)
    
    # Characteristic matrix A - λI
    char_matrix = A_sym - lam * I
    
    print("Deriving Characteristic Equation:")
    print(f"A - λI =")
    sp.pprint(char_matrix)
    
    # Characteristic equation det(A - λI) = 0
    char_poly = char_matrix.det()
    print(f"\nCharacteristic Polynomial: {char_poly} = 0")
    
    # Find eigenvalues
    eigenvals_sym = sp.solve(char_poly, lam)
    print(f"Eigenvalues: {eigenvals_sym}")
    
    # Compare with NumPy results
    print(f"\nNumPy Eigenvalues: {eigenvalues}")

## 3.3 Diagonalization

**Definition: Diagonalization**  
Matrix A is diagonalizable ⇔ there exists matrix P (with eigenvectors as columns) and diagonal matrix D (with eigenvalues on diagonal) such that P^(-1)AP = D. 

### Code Example 4: Implementation of Diagonalization

Python Implementation: Diagonalization and Applications
    
    
    # Diagonal matrix D with eigenvalues on diagonal
    D = np.diag(eigenvalues)
    
    # Matrix P with eigenvectors as columns
    P = eigenvectors
    
    # P^(-1)
    P_inv = np.linalg.inv(P)
    
    # Verification: P^(-1) A P = D
    result = P_inv @ A @ P
    
    print("Diagonalization Verification:")
    print(f"P (eigenvector matrix) =\n{P}\n")
    print(f"D (eigenvalue diagonal matrix) =\n{D}\n")
    print(f"P^(-1) A P =\n{result}\n")
    print(f"Matches D? {np.allclose(result, D)}")
    
    # Application of diagonalization: fast computation of A^n
    # A^n = P D^n P^(-1)
    n = 10
    A_n_fast = P @ np.linalg.matrix_power(D, n) @ P_inv
    A_n_direct = np.linalg.matrix_power(A, n)
    
    print(f"\nComputation of A^{n}:")
    print(f"Using diagonalization: {A_n_fast[0,0]:.4f}")
    print(f"Direct calculation:   {A_n_direct[0,0]:.4f}")
    print(f"Match? {np.allclose(A_n_fast, A_n_direct)}")

## 3.4 Properties of Symmetric Matrices

**Theorem: Spectral Theorem for Symmetric Matrices**  
Real symmetric matrices (A = A^T):  

  * All eigenvalues are real
  * Eigenvectors corresponding to different eigenvalues are orthogonal
  * Always diagonalizable, orthogonally diagonalizable

### Code Example 5: Eigenvalue Decomposition of Symmetric Matrices

Python Implementation: Eigenvalue Decomposition of Symmetric Matrices
    
    
    # Symmetric matrix
    A_sym = np.array([[3, 1],
                      [1, 3]])
    
    # Eigenvalues and eigenvectors
    eigenvals_sym, eigenvecs_sym = np.linalg.eigh(A_sym)  # For symmetric matrices
    
    print("Eigenvalue Decomposition of Symmetric Matrix:")
    print(f"A (symmetric) =\n{A_sym}\n")
    print(f"Eigenvalues: {eigenvals_sym}")
    print(f"Eigenvectors:\n{eigenvecs_sym}\n")
    
    # Confirm orthogonality of eigenvectors
    v1 = eigenvecs_sym[:, 0]
    v2 = eigenvecs_sym[:, 1]
    inner_product = np.dot(v1, v2)
    print(f"Inner product of eigenvectors: {inner_product:.10f}")
    print(f"Orthogonal? {np.abs(inner_product) < 1e-10}")
    
    # Confirm orthogonality: Q^T Q = I
    Q = eigenvecs_sym
    QTQ = Q.T @ Q
    print(f"\nQ^T Q =\n{QTQ}")
    print(f"Identity matrix? {np.allclose(QTQ, np.eye(2))}")

## 3.5 PCA (Principal Component Analysis)

**Application Example: Principal Component Analysis (PCA)**  
By eigenvalue decomposition of the data covariance matrix, the eigenvector direction corresponding to the largest eigenvalue becomes the direction of maximum variance (first principal component). 

### Code Example 6: PCA Implementation

Python Implementation: Principal Component Analysis (PCA)
    
    
    from sklearn.datasets import make_blobs
    
    # Generate sample data
    np.random.seed(42)
    X, _ = make_blobs(n_samples=100, n_features=2, centers=1, cluster_std=2.0)
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigenvalue decomposition
    eigenvals_pca, eigenvecs_pca = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order of eigenvalues
    idx = eigenvals_pca.argsort()[::-1]
    eigenvals_pca = eigenvals_pca[idx]
    eigenvecs_pca = eigenvecs_pca[:, idx]
    
    print("PCA (Principal Component Analysis):")
    print(f"Covariance Matrix:\n{cov_matrix}\n")
    print(f"Eigenvalues: {eigenvals_pca}")
    print(f"Contribution Ratio: {eigenvals_pca / eigenvals_pca.sum()}")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.5)
    
    # Draw principal component directions
    origin = [0, 0]
    scale = np.sqrt(eigenvals_pca)
    for i in range(2):
        vec = eigenvecs_pca[:, i] * scale[i] * 3
        plt.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1,
                  color=['red', 'blue'][i], width=0.01,
                  label=f'PC{i+1} (λ={eigenvals_pca[i]:.2f})')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Principal Component Analysis (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

### Code Example 7: Dimensionality Reduction with Data Compression

Python Implementation: Dimensionality Reduction and Reconstruction
    
    
    # Project onto first principal component only (2D → 1D)
    PC1 = eigenvecs_pca[:, 0].reshape(-1, 1)
    
    # Projection
    X_pca = X_centered @ PC1
    
    # Reconstruct in original space (approximate reconstruction)
    X_reconstructed = X_pca @ PC1.T
    
    # Reconstruction error
    reconstruction_error = np.mean(np.linalg.norm(X_centered - X_reconstructed, axis=1)**2)
    
    print(f"\nDimensionality Reduction:")
    print(f"Original dimensions: {X_centered.shape[1]}D")
    print(f"After reduction: 1D")
    print(f"Reconstruction error: {reconstruction_error:.4f}")
    print(f"First principal component contribution: {eigenvals_pca[0]/eigenvals_pca.sum()*100:.1f}%")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.3, label='Original Data')
    plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.5, label='Reconstructed Data', s=10)
    plt.plot([0, PC1[0,0]*scale[0]*3], [0, PC1[1,0]*scale[0]*3], 'r-', linewidth=3, label='PC1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Projection onto First Principal Component and Reconstruction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

## 3.6 Application to Materials Science: Vibration Mode Analysis

### Code Example 8: Natural Frequencies of Coupled Oscillation Systems

Python Implementation: Vibration Mode Analysis
    
    
    # 2-DOF coupled oscillation system
    # m1 = m2 = 1 kg, k1 = k2 = k3 = 1 N/m
    # Equation of motion: M x'' + K x = 0
    # Eigenvalue problem: det(K - ω² M) = 0
    
    M = np.array([[1, 0],
                  [0, 1]])  # Mass matrix
    
    K = np.array([[2, -1],
                  [-1, 2]])  # Stiffness matrix
    
    # Solve generalized eigenvalue problem
    eigenvals_vibration, eigenvecs_vibration = np.linalg.eigh(K, M)
    
    # Natural angular frequency ω = sqrt(λ)
    omega = np.sqrt(eigenvals_vibration)
    
    print("Vibration Mode Analysis:")
    print(f"Mass Matrix M:\n{M}\n")
    print(f"Stiffness Matrix K:\n{K}\n")
    print(f"Eigenvalues λ: {eigenvals_vibration}")
    print(f"Natural Angular Frequencies ω: {omega} rad/s")
    print(f"Natural Frequencies f: {omega/(2*np.pi)} Hz\n")
    
    print("Vibration Modes (Eigenvectors):")
    for i in range(2):
        print(f"Mode {i+1} (f={omega[i]/(2*np.pi):.3f} Hz):")
        print(f"  Mass 1: {eigenvecs_vibration[0,i]:.4f}")
        print(f"  Mass 2: {eigenvecs_vibration[1,i]:.4f}")
    
        if eigenvecs_vibration[0,i] * eigenvecs_vibration[1,i] > 0:
            print(f"  → In-phase vibration\n")
        else:
            print(f"  → Out-of-phase vibration\n")

## Summary

  * Eigenvectors are vectors whose direction doesn't change under linear transformation, eigenvalues are the scaling factors
  * Diagonalization makes matrix power calculations efficient, useful for dynamic system analysis
  * Symmetric matrices have real eigenvalues and can be diagonalized with orthogonal eigenvectors
  * PCA extracts principal variation directions from data via eigenvalue decomposition of covariance matrix
  * Materials science has various applications including vibration mode analysis and crystal symmetry

[← Chapter 2: Determinants](<chapter-2.html>) [Chapter 4: Singular Value Decomposition →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
