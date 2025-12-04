---
title: "Chapter 2: Determinants and Systems of Linear Equations"
chapter_title: "Chapter 2: Determinants and Systems of Linear Equations"
subtitle: Determinants and Systems of Linear Equations
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/linear-algebra-tensor/chapter-2.html>) | Last sync: 2025-11-16

[Fundamentals of Mathematics Dojo](<../index.html>) > [Linear Algebra and Tensor Analysis](<index.html>) > Chapter 2 

## 2.1 Definition and Calculation of Determinants

**Definition: Determinant**  
The determinant det(A) or |A| of an n×n square matrix A is a scalar value satisfying the following properties:  
2×2 matrix: $\det\begin{pmatrix}a&b\\\c&d\end{pmatrix} = ad - bc$  
3×3 matrix: calculated using Sarrus's rule or cofactor expansion 

### Code Example 1: Calculating Determinants

Python Implementation: Calculating Determinants
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 1: Calculating Determinants
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # 2×2 matrix determinant
    A_2x2 = np.array([[3, 8],
                      [4, 6]])
    
    det_A = np.linalg.det(A_2x2)
    det_manual = 3*6 - 8*4  # ad - bc
    
    print("2×2 Matrix Determinant:")
    print(f"A =\n{A_2x2}")
    print(f"det(A) = {det_A:.4f}")
    print(f"Manual calculation: 3×6 - 8×4 = {det_manual}")
    
    # 3×3 matrix determinant
    A_3x3 = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    
    det_A_3x3 = np.linalg.det(A_3x3)
    print(f"\n3×3 Matrix Determinant:")
    print(f"A =\n{A_3x3}")
    print(f"det(A) = {det_A_3x3:.10f}")
    print("det(A) ≈ 0 so singular matrix (no inverse)")

**Theorem: Properties of Determinants**  

  * det(AB) = det(A) det(B) (product of determinants)
  * det(A^T) = det(A) (invariant under transpose)
  * det(kA) = k^n det(A) (scalar multiplication of n×n matrix)
  * det(A) ≠ 0 ⇔ A is non-singular (inverse exists)
  * Swapping rows changes sign

### Code Example 2: Verification of Determinant Properties

A = np.array([[2, 1], [3, 4]]) B = np.array([[5, 6], [7, 8]]) det_A = np.linalg.det(A) det_B = np.linalg.det(B) det_AB = np.linalg.det(A @ B) print("Verification of Determinant Properties:") print(f"det(A) = {det_A:.4f}") print(f"det(B) = {det_B:.4f}") print(f"det(AB) = {det_AB:.4f}") print(f"det(A) × det(B) = {det_A * det_B:.4f}") print(f"det(AB) = det(A)×det(B)? {np.isclose(det_AB, det_A * det_B)}") # Transpose det_AT = np.linalg.det(A.T) print(f"\ndet(A^T) = {det_AT:.4f}") print(f"det(A) = det(A^T)? {np.isclose(det_A, det_AT)}")

## 2.2 Solving Systems of Linear Equations

**Definition: System of Linear Equations**  
Equation system represented as Ax = b. A: coefficient matrix, x: unknown vector, b: constant term vector 

### Code Example 3: Solving Linear Systems with NumPy

# System of equations: 2x + 3y = 8, x - y = -1 A = np.array([[2, 3], [1, -1]]) b = np.array([8, -1]) # Solve with np.linalg.solve x = np.linalg.solve(A, b) print("Solution of Linear System:") print(f"2x + 3y = 8") print(f"x - y = -1") print(f"\nSolution: x = {x[0]:.4f}, y = {x[1]:.4f}") # Verification b_check = A @ x print(f"\nVerification: Ax = {b_check}") print(f"Error: {np.linalg.norm(b - b_check):.2e}")

### Code Example 4: Cramer's Rule

def cramers_rule(A, b): """ Solve linear system using Cramer's rule x_i = det(A_i) / det(A) A_i: matrix with i-th column replaced by b """ det_A = np.linalg.det(A) if np.abs(det_A) < 1e-10: raise ValueError("Determinant is zero, cannot solve") n = len(b) x = np.zeros(n) for i in range(n): A_i = A.copy() A_i[:, i] = b # Replace i-th column with b x[i] = np.linalg.det(A_i) / det_A return x # Solve the same system x_cramer = cramers_rule(A, b) print("Cramer's Rule:") print(f"Solution: x = {x_cramer[0]:.4f}, y = {x_cramer[1]:.4f}") print(f"Difference from np.linalg.solve: {np.linalg.norm(x - x_cramer):.2e}")

## 2.3 Gaussian Elimination

### Code Example 5: Implementation of Gaussian Elimination

def gaussian_elimination(A, b): """ Solve linear system using Gaussian elimination Forward elimination → Back substitution """ n = len(b) # Create augmented matrix Ab = np.hstack([A.astype(float), b.reshape(-1, 1)]) # Forward elimination for i in range(n): # Pivot selection (partial pivoting) max_row = np.argmax(np.abs(Ab[i:, i])) + i Ab[[i, max_row]] = Ab[[max_row, i]] # Normalize i-th row Ab[i] = Ab[i] / Ab[i, i] # Zero out below i-th column for j in range(i+1, n): Ab[j] = Ab[j] - Ab[j, i] * Ab[i] # Back substitution x = np.zeros(n) for i in range(n-1, -1, -1): x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:]) return x # Test A_test = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]) b_test = np.array([8.0, -11.0, -3.0]) x_gauss = gaussian_elimination(A_test.copy(), b_test.copy()) x_numpy = np.linalg.solve(A_test, b_test) print("Gaussian Elimination:") print(f"Gaussian solution: {x_gauss}") print(f"NumPy solution: {x_numpy}") print(f"Difference: {np.linalg.norm(x_gauss - x_numpy):.2e}")

## 2.4 LU Decomposition

**Definition: LU Decomposition**  
Decompose matrix A into lower triangular matrix L and upper triangular matrix U: A = LU  
Linear system Ax = b can be solved in two stages: Ly = b → Ux = y 

### Code Example 6: Solving Linear Systems with LU Decomposition

from scipy.linalg import lu # LU decomposition A_lu = np.array([[4, 3], [6, 3]]) P, L, U = lu(A_lu) print("LU Decomposition:") print(f"A =\n{A_lu}\n") print(f"P (permutation matrix) =\n{P}\n") print(f"L (lower triangular) =\n{L}\n") print(f"U (upper triangular) =\n{U}\n") # Verification: PA = LU print(f"PA =\n{P @ A_lu}\n") print(f"LU =\n{L @ U}\n") print(f"PA = LU? {np.allclose(P @ A_lu, L @ U)}") # Solve linear system using LU decomposition b_lu = np.array([10, 12]) # Step 1: Solve Ly = Pb (forward substitution) y = np.linalg.solve(L, P @ b_lu) # Step 2: Solve Ux = y (back substitution) x_lu = np.linalg.solve(U, y) print(f"\nSolution of linear system Ax = b:") print(f"x = {x_lu}") # Verification print(f"Ax = {A_lu @ x_lu}") print(f"b = {b_lu}")

## 2.5 Rank and Existence Conditions for Solutions

**Theorem: Existence Conditions for Solutions**  
For Ax = b:  

  * rank(A) = rank([A|b]) = n → unique solution
  * rank(A) = rank([A|b]) < n → infinite solutions
  * rank(A) < rank([A|b]) → no solution

### Code Example 7: Rank Calculation and Solution Classification

# Rank calculation A_rank = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) rank_A = np.linalg.matrix_rank(A_rank) print("Rank Calculation:") print(f"A =\n{A_rank}") print(f"rank(A) = {rank_A}") print(f"A is 3×3 square matrix but rank < 3 so singular matrix\n") # Example with unique solution A_unique = np.array([[1, 2], [3, 4]]) b_unique = np.array([5, 6]) Ab_unique = np.column_stack([A_unique, b_unique]) rank_A_u = np.linalg.matrix_rank(A_unique) rank_Ab_u = np.linalg.matrix_rank(Ab_unique) print("Unique Solution Example:") print(f"rank(A) = {rank_A_u}") print(f"rank([A|b]) = {rank_Ab_u}") print(f"rank(A) = rank([A|b]) = 2 → unique solution") x_unique = np.linalg.solve(A_unique, b_unique) print(f"Solution: {x_unique}\n") # Example with no solution A_no_sol = np.array([[1, 2], [2, 4]]) b_no_sol = np.array([3, 7]) # 2nd equation is 2× 1st equation but RHS inconsistent Ab_no_sol = np.column_stack([A_no_sol, b_no_sol]) rank_A_n = np.linalg.matrix_rank(A_no_sol) rank_Ab_n = np.linalg.matrix_rank(Ab_no_sol) print("No Solution Example:") print(f"rank(A) = {rank_A_n}") print(f"rank([A|b]) = {rank_Ab_n}") print(f"rank(A) < rank([A|b]) → no solution")

## 2.6 Application to Materials Science: Stoichiometry Calculations

**Application Example: Balancing Chemical Equations**  
By solving chemical equation coefficients as linear equations, we can derive reaction formulas that satisfy elemental balance. 

### Code Example 8: Balancing Chemical Equations

# Chemical equation: aFe + bO2 → cFe2O3 # Elemental balance: # Fe: a = 2c # O: 2b = 3c # Express in matrix form # Coefficient matrix (unknowns on left, knowns on right) # a - 2c = 0 # 2b - 3c = 0 # c = 1 (fix arbitrarily) # Transform to solvable form A_chem = np.array([[1, 0, -2], # Fe balance [0, 2, -3]]) # O balance # With c=1 c = 1 b_chem = np.array([2*c, 3*c]) # Find least squares solution x_chem = np.linalg.lstsq(A_chem[:, :2], b_chem, rcond=None)[0] a, b = x_chem print("Chemical Equation Balancing:") print(f"Fe + O2 → Fe2O3") print(f"\nCoefficients:") print(f"a (Fe) = {a:.1f}") print(f"b (O2) = {b:.1f}") print(f"c (Fe2O3) = {c:.1f}") print(f"\nBalanced Equation:") print(f"{int(a*2)}Fe + {int(b*2)}O2 → {int(c*2)}Fe2O3") # Verification print(f"\nElemental Balance Verification:") print(f"Fe: left = {int(a*2)}, right = {int(c*2)*2}") print(f"O: left = {int(b*2)*2}, right = {int(c*2)*3}")

## Summary

  * Determinant is an important invariant of square matrices, used to determine existence of inverse
  * Systems of linear equations are efficiently solved with NumPy's solve function
  * Gaussian elimination is important as a systematic manual calculation method
  * LU decomposition efficiently solves multiple problems with same A but different b
  * Rank is key to determining existence and uniqueness of solutions
  * Linear equations appear in many contexts in materials science such as stoichiometry calculations

[← Chapter 1: Vectors and Matrices](<chapter-1.html>) [Chapter 3: Eigenvalues & Eigenvectors →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
