---
title: "Chapter 4: Singular Value Decomposition and Applications"
chapter_title: "Chapter 4: Singular Value Decomposition and Applications"
subtitle: Singular Value Decomposition and Applications
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/linear-algebra-tensor/chapter-4.html>) | Last sync: 2025-11-16

[Fundamentals of Mathematics Dojo](<../index.html>) > [Linear Algebra and Tensor Analysis](<index.html>) > Chapter 4 

## 4.1 Definition of Singular Value Decomposition (SVD)

**Definition: Singular Value Decomposition (SVD)**  
Any m×n matrix A can be decomposed as: \\[A = U \Sigma V^T\\] 

  * U: m×m orthogonal matrix (left singular vectors)
  * Σ: m×n diagonal matrix (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
  * V: n×n orthogonal matrix (right singular vectors)

### Code Example 1: SVD Calculation

Python Implementation: Singular Value Decomposition Calculation and Verification
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 1: SVD Calculation
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define matrix
    A = np.array([[4, 0],
                  [3, -5]])
    
    # SVD decomposition
    U, s, VT = np.linalg.svd(A, full_matrices=True)
    
    # Construct Σ matrix
    Sigma = np.zeros((2, 2))
    Sigma[:2, :2] = np.diag(s)
    
    print("Singular Value Decomposition:")
    print(f"A =\n{A}\n")
    print(f"U (left singular vectors) =\n{U}\n")
    print(f"Singular values σ: {s}")
    print(f"Σ =\n{Sigma}\n")
    print(f"V^T (right singular vectors transposed) =\n{VT}\n")
    
    # Verification of reconstruction
    A_reconstructed = U @ Sigma @ VT
    print(f"UΣV^T =\n{A_reconstructed}\n")
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")
    
    # Confirm orthogonality
    print(f"\nU^T U = I? {np.allclose(U.T @ U, np.eye(2))}")
    print(f"V^T V = I? {np.allclose(VT.T @ VT, np.eye(2))}")

## 4.2 Relationship Between SVD and Eigenvalue Decomposition

**Theorem: Relationship Between SVD and Eigenvalues**  

  * Eigenvalues of A^T A = σᵢ² (square of singular values)
  * Eigenvectors of A^T A = columns of V
  * Eigenvalues of AA^T = σᵢ²
  * Eigenvectors of AA^T = columns of U

### Code Example 2: Verification of Relationship with Eigenvalue Decomposition

Python Implementation: Relationship with Eigenvalue Decomposition
    
    
    # Eigenvalue decomposition of A^T A
    ATA = A.T @ A
    eigenvals_ATA, eigenvecs_ATA = np.linalg.eigh(ATA)
    
    print("Relationship Between SVD and Eigenvalue Decomposition:")
    print(f"A^T A =\n{ATA}\n")
    print(f"Eigenvalues of A^T A: {eigenvals_ATA}")
    print(f"Square of singular values: {s**2}")
    print(f"Match? {np.allclose(sorted(eigenvals_ATA, reverse=True), s**2)}\n")
    
    # Eigenvalue decomposition of AA^T
    AAT = A @ A.T
    eigenvals_AAT, eigenvecs_AAT = np.linalg.eigh(AAT)
    
    print(f"Eigenvalues of AA^T: {eigenvals_AAT}")
    print(f"Square of singular values: {s**2}")
    print(f"Match? {np.allclose(sorted(eigenvals_AAT, reverse=True), s**2)}")

## 4.3 Low-Rank Approximation

**Theorem: Eckart-Young Theorem**  
The best rank-k approximation of A is given by \\[A_k = \sum_{i=1}^k \sigma_i u_i v_i^T\\] using only the k largest singular values from SVD, achieving minimum error in Frobenius norm. 

### Code Example 3: Implementation of Low-Rank Approximation

Python Implementation: Low-Rank Approximation
    
    
    def low_rank_approximation(A, k):
        """Rank-k approximation"""
        U, s, VT = np.linalg.svd(A, full_matrices=False)
    
        # Use only k singular values
        s_k = s.copy()
        s_k[k:] = 0
    
        Sigma_k = np.diag(s_k)
        A_k = U @ Sigma_k @ VT
    
        return A_k
    
    # Test matrix (rank 3)
    A_test = np.array([[1, 2, 3, 4],
                       [2, 4, 6, 8],
                       [3, 6, 9, 12],
                       [1, 1, 1, 1]])
    
    # Original rank
    rank_original = np.linalg.matrix_rank(A_test)
    
    print("Low-Rank Approximation:")
    print(f"Original matrix rank: {rank_original}\n")
    
    for k in range(1, 4):
        A_approx = low_rank_approximation(A_test, k)
        error = np.linalg.norm(A_test - A_approx, 'fro')
    
        print(f"Rank {k} approximation:")
        print(f"  Frobenius norm error: {error:.4f}")
        print(f"  Approximation matrix rank: {np.linalg.matrix_rank(A_approx)}\n")

## 4.4 Application to Image Compression

### Code Example 4: SVD Image Compression

Python Implementation: Image Compression
    
    
    # Requirements:
    # - Python 3.9+
    # - scipy>=1.11.0
    
    """
    Example: Code Example 4: SVD Image Compression
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from scipy import misc
    from skimage import data
    
    # Load grayscale image
    image = data.camera()  # 512x512 sample image
    
    print(f"Image size: {image.shape}")
    print(f"Original data size: {image.size} elements\n")
    
    # SVD decomposition
    U_img, s_img, VT_img = np.linalg.svd(image, full_matrices=False)
    
    # Compress with various ranks
    ranks = [5, 10, 20, 50, 100]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Original Image (rank={np.linalg.matrix_rank(image)})')
    axes[0].axis('off')
    
    # Compressed images
    for idx, k in enumerate(ranks, 1):
        # Rank-k approximation
        img_compressed = U_img[:, :k] @ np.diag(s_img[:k]) @ VT_img[:k, :]
    
        # Calculate compression ratio
        original_size = image.size
        compressed_size = k * (U_img.shape[0] + VT_img.shape[1] + 1)
        compression_ratio = original_size / compressed_size
    
        # Reconstruction error
        error = np.linalg.norm(image - img_compressed) / np.linalg.norm(image)
    
        axes[idx].imshow(img_compressed, cmap='gray')
        axes[idx].set_title(f'Rank={k}\nCompression={compression_ratio:.1f}x, Error={error:.3f}')
        axes[idx].axis('off')
    
        print(f"Rank {k}:")
        print(f"  Compressed data size: {compressed_size} elements")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Relative error: {error:.4f}\n")
    
    plt.tight_layout()
    plt.show()
    
    # Distribution of singular values
    plt.figure(figsize=(10, 6))
    plt.semilogy(s_img, 'o-', markersize=3)
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Distribution of Singular Values')
    plt.grid(True, alpha=0.3)
    plt.show()

## 4.5 Pseudo-Inverse Matrix

**Definition: Moore-Penrose Pseudo-Inverse**  
The pseudo-inverse A⁺ of an m×n matrix A is calculated using SVD as: \\[A^+ = V \Sigma^+ U^T\\] Σ⁺: matrix with reciprocals of non-zero singular values on diagonal 

### Code Example 5: Least Squares Method Using Pseudo-Inverse

Python Implementation: Pseudo-Inverse and Least Squares
    
    
    # Overdetermined system (more equations than unknowns)
    A_overdetermined = np.array([[1, 1],
                                 [1, 2],
                                 [1, 3],
                                 [1, 4]])
    
    b_overdetermined = np.array([2, 3, 4, 5.5])
    
    # Solution using pseudo-inverse
    A_pinv = np.linalg.pinv(A_overdetermined)
    x_pinv = A_pinv @ b_overdetermined
    
    print("Least Squares Method Using Pseudo-Inverse:")
    print(f"A ({A_overdetermined.shape[0]}×{A_overdetermined.shape[1]}) =\n{A_overdetermined}\n")
    print(f"b = {b_overdetermined}\n")
    print(f"Least squares solution x = {x_pinv}")
    
    # Residual
    residual = A_overdetermined @ x_pinv - b_overdetermined
    print(f"Residual: {residual}")
    print(f"Residual norm: {np.linalg.norm(residual):.4f}")
    
    # Compare with np.linalg.lstsq
    x_lstsq = np.linalg.lstsq(A_overdetermined, b_overdetermined, rcond=None)[0]
    print(f"\nlstsq solution: {x_lstsq}")
    print(f"Match? {np.allclose(x_pinv, x_lstsq)}")

## 4.6 Application to Recommendation Systems

**Application Example: Collaborative Filtering**  
By approximating a user×item rating matrix with low rank, we can predict unrated item ratings (e.g., Netflix Prize problem). 

### Code Example 6: Matrix Completion

Python Implementation: Recommendation System
    
    
    # User×movie rating matrix (5-point scale, 0 means unrated)
    ratings = np.array([[5, 3, 0, 1],
                        [4, 0, 0, 1],
                        [1, 1, 0, 5],
                        [1, 0, 0, 4],
                        [0, 1, 5, 4]])
    
    # SVD using only observed ratings
    # Simplified version: fill 0s with mean rating
    mean_rating = ratings[ratings > 0].mean()
    ratings_filled = np.where(ratings > 0, ratings, mean_rating)
    
    # Low-rank approximation via SVD
    U_rec, s_rec, VT_rec = np.linalg.svd(ratings_filled, full_matrices=False)
    
    k = 2  # Number of latent factors
    ratings_predicted = U_rec[:, :k] @ np.diag(s_rec[:k]) @ VT_rec[:k, :]
    
    print("Recommendation System (Collaborative Filtering):")
    print(f"Original Rating Matrix (0 means unrated):\n{ratings}\n")
    print(f"Predicted Rating Matrix (rank {k} approximation):\n{np.round(ratings_predicted, 2)}\n")
    
    # Predicted values for unrated items
    print("Predictions for Unrated Items:")
    for i in range(ratings.shape[0]):
        for j in range(ratings.shape[1]):
            if ratings[i, j] == 0:
                print(f"User {i+1}, Movie {j+1}: Predicted rating = {ratings_predicted[i, j]:.2f}")

### Code Example 7: Interpretation of Latent Factors

Python Implementation: Visualizing Latent Factors
    
    
    # User factors and movie factors
    user_factors = U_rec[:, :k] @ np.diag(np.sqrt(s_rec[:k]))
    movie_factors = np.diag(np.sqrt(s_rec[:k])) @ VT_rec[:k, :]
    
    print("\nLatent Factor Analysis:")
    print(f"User Factors (5 users × 2 factors):\n{user_factors}\n")
    print(f"Movie Factors (2 factors × 4 movies):\n{movie_factors}\n")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(user_factors[:, 0], user_factors[:, 1], s=100, c='blue', marker='o', label='Users')
    
    for i in range(len(user_factors)):
        plt.annotate(f'U{i+1}', (user_factors[i, 0], user_factors[i, 1]),
                    fontsize=12, ha='right')
    
    plt.scatter(movie_factors[0, :], movie_factors[1, :], s=100, c='red', marker='s', label='Movies')
    
    for j in range(movie_factors.shape[1]):
        plt.annotate(f'M{j+1}', (movie_factors[0, j], movie_factors[1, j]),
                    fontsize=12, ha='left')
    
    plt.xlabel('Latent Factor 1')
    plt.ylabel('Latent Factor 2')
    plt.title('Latent Factor Space for Users and Movies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.show()

### Code Example 8: Application to Noise Removal

Python Implementation: Noise Removal Using SVD
    
    
    # Data with noise
    np.random.seed(42)
    true_signal = np.array([[1, 2, 3, 4, 5],
                           [2, 4, 6, 8, 10],
                           [3, 6, 9, 12, 15]])
    
    noise = np.random.randn(3, 5) * 0.5
    noisy_signal = true_signal + noise
    
    # Noise removal using SVD
    U_noise, s_noise, VT_noise = np.linalg.svd(noisy_signal, full_matrices=False)
    
    print("Noise Removal Using SVD:")
    print(f"Singular values: {s_noise}")
    
    # Use only largest singular value (rank-1 approximation)
    k_denoise = 1
    denoised = U_noise[:, :k_denoise] @ np.diag(s_noise[:k_denoise]) @ VT_noise[:k_denoise, :]
    
    print(f"\nError before noise removal: {np.linalg.norm(noisy_signal - true_signal):.4f}")
    print(f"Error after noise removal: {np.linalg.norm(denoised - true_signal):.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(true_signal, cmap='viridis', aspect='auto')
    axes[0].set_title('True Signal')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Channel')
    
    axes[1].imshow(noisy_signal, cmap='viridis', aspect='auto')
    axes[1].set_title('Noisy Signal')
    axes[1].set_xlabel('Time')
    
    axes[2].imshow(denoised, cmap='viridis', aspect='auto')
    axes[2].set_title('Denoised (Rank-1 Approximation)')
    axes[2].set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()

## Summary

  * SVD is a powerful method to decompose any matrix into three orthogonal/diagonal matrices
  * Low-rank approximation enables data compression and noise removal
  * Pseudo-inverse allows finding least squares solutions for overdetermined/underdetermined systems
  * Wide applications including image compression, recommendation systems, and signal processing
  * Latent factor analysis can discover hidden structure behind data

[← Chapter 3: Eigenvalues & Eigenvectors](<chapter-3.html>) [Chapter 5: Tensor Analysis →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
