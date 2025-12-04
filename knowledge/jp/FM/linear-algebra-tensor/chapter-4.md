---
title: "第4章: 特異値分解と応用"
chapter_title: "第4章: 特異値分解と応用"
subtitle: Singular Value Decomposition and Applications
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/linear-algebra-tensor/chapter-4.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [線形代数とテンソル解析](<index.html>) > 第4章 

## 4.1 特異値分解（SVD）の定義

**📐 定義: 特異値分解（SVD）**  
任意の m×n 行列 A は以下のように分解できます： \\[A = U \Sigma V^T\\] 

  * U: m×m 直交行列（左特異ベクトル）
  * Σ: m×n 対角行列（特異値 σ₁ ≥ σ₂ ≥ ... ≥ 0）
  * V: n×n 直交行列（右特異ベクトル）

### 💻 コード例1: SVDの計算

Python実装: 特異値分解の計算と検証

import numpy as np import matplotlib.pyplot as plt # 行列の定義 A = np.array([[4, 0], [3, -5]]) # SVD分解 U, s, VT = np.linalg.svd(A, full_matrices=True) # Σ行列を構築 Sigma = np.zeros((2, 2)) Sigma[:2, :2] = np.diag(s) print("特異値分解:") print(f"A =\n{A}\n") print(f"U (左特異ベクトル) =\n{U}\n") print(f"特異値 σ: {s}") print(f"Σ =\n{Sigma}\n") print(f"V^T (右特異ベクトル転置) =\n{VT}\n") # 再構成の検証 A_reconstructed = U @ Sigma @ VT print(f"UΣV^T =\n{A_reconstructed}\n") print(f"再構成誤差: {np.linalg.norm(A - A_reconstructed):.2e}") # 直交性の確認 print(f"\nU^T U = I? {np.allclose(U.T @ U, np.eye(2))}") print(f"V^T V = I? {np.allclose(VT.T @ VT, np.eye(2))}")

## 4.2 SVDと固有値分解の関係

**📐 定理: SVDと固有値の関係**  

  * A^T A の固有値 = σᵢ² （特異値の2乗）
  * A^T A の固有ベクトル = V の列ベクトル
  * AA^T の固有値 = σᵢ²
  * AA^T の固有ベクトル = U の列ベクトル

### 💻 コード例2: SVDと固有値分解の関係の検証

Python実装: 固有値分解との関係

# A^T A の固有値分解 ATA = A.T @ A eigenvals_ATA, eigenvecs_ATA = np.linalg.eigh(ATA) print("SVDと固有値分解の関係:") print(f"A^T A =\n{ATA}\n") print(f"A^T A の固有値: {eigenvals_ATA}") print(f"特異値の2乗: {s**2}") print(f"一致? {np.allclose(sorted(eigenvals_ATA, reverse=True), s**2)}\n") # AA^T の固有値分解 AAT = A @ A.T eigenvals_AAT, eigenvecs_AAT = np.linalg.eigh(AAT) print(f"AA^T の固有値: {eigenvals_AAT}") print(f"特異値の2乗: {s**2}") print(f"一致? {np.allclose(sorted(eigenvals_AAT, reverse=True), s**2)}")

## 4.3 低ランク近似

**📐 定理: Eckart-Young の定理**  
ランク k の行列による A の最良近似は、SVD の最大 k 個の特異値のみを使った \\[A_k = \sum_{i=1}^k \sigma_i u_i v_i^T\\] で与えられ、フロベニウスノルムで最小誤差を達成します。 

### 💻 コード例3: 低ランク近似の実装

Python実装: 低ランク近似

def low_rank_approximation(A, k): """ランクk近似""" U, s, VT = np.linalg.svd(A, full_matrices=False) # k個の特異値のみ使用 s_k = s.copy() s_k[k:] = 0 Sigma_k = np.diag(s_k) A_k = U @ Sigma_k @ VT return A_k # テスト行列（ランク3） A_test = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [1, 1, 1, 1]]) # 元のランク rank_original = np.linalg.matrix_rank(A_test) print("低ランク近似:") print(f"元の行列のランク: {rank_original}\n") for k in range(1, 4): A_approx = low_rank_approximation(A_test, k) error = np.linalg.norm(A_test - A_approx, 'fro') print(f"ランク {k} 近似:") print(f" フロベニウスノルム誤差: {error:.4f}") print(f" 近似行列のランク: {np.linalg.matrix_rank(A_approx)}\n")

## 4.4 画像圧縮への応用

### 💻 コード例4: 画像のSVD圧縮

Python実装: 画像圧縮

from scipy import misc from skimage import data # グレースケール画像を読み込み image = data.camera() # 512x512のサンプル画像 print(f"画像サイズ: {image.shape}") print(f"元のデータ量: {image.size} 要素\n") # SVD分解 U_img, s_img, VT_img = np.linalg.svd(image, full_matrices=False) # 様々なランクで圧縮 ranks = [5, 10, 20, 50, 100] fig, axes = plt.subplots(2, 3, figsize=(15, 10)) axes = axes.ravel() # 元画像 axes[0].imshow(image, cmap='gray') axes[0].set_title(f'元画像 (ランク={np.linalg.matrix_rank(image)})') axes[0].axis('off') # 圧縮画像 for idx, k in enumerate(ranks, 1): # ランクk近似 img_compressed = U_img[:, :k] @ np.diag(s_img[:k]) @ VT_img[:k, :] # 圧縮率の計算 original_size = image.size compressed_size = k * (U_img.shape[0] + VT_img.shape[1] + 1) compression_ratio = original_size / compressed_size # 再構成誤差 error = np.linalg.norm(image - img_compressed) / np.linalg.norm(image) axes[idx].imshow(img_compressed, cmap='gray') axes[idx].set_title(f'ランク={k}\n圧縮率={compression_ratio:.1f}x, 誤差={error:.3f}') axes[idx].axis('off') print(f"ランク {k}:") print(f" 圧縮データ量: {compressed_size} 要素") print(f" 圧縮率: {compression_ratio:.2f}倍") print(f" 相対誤差: {error:.4f}\n") plt.tight_layout() plt.show() # 特異値の分布 plt.figure(figsize=(10, 6)) plt.semilogy(s_img, 'o-', markersize=3) plt.xlabel('特異値のインデックス') plt.ylabel('特異値 (対数スケール)') plt.title('特異値の分布') plt.grid(True, alpha=0.3) plt.show()

## 4.5 擬似逆行列

**📐 定義: ムーア・ペンローズ擬似逆行列**  
m×n 行列 A の擬似逆行列 A⁺ は SVD を使って： \\[A^+ = V \Sigma^+ U^T\\] Σ⁺: 非ゼロ特異値の逆数を対角成分に持つ行列 

### 💻 コード例5: 擬似逆行列による最小二乗法

Python実装: 擬似逆行列と最小二乗法

# 過剰決定系（方程式の数 > 未知数） A_overdetermined = np.array([[1, 1], [1, 2], [1, 3], [1, 4]]) b_overdetermined = np.array([2, 3, 4, 5.5]) # 擬似逆行列による解 A_pinv = np.linalg.pinv(A_overdetermined) x_pinv = A_pinv @ b_overdetermined print("擬似逆行列による最小二乗法:") print(f"A ({A_overdetermined.shape[0]}×{A_overdetermined.shape[1]}) =\n{A_overdetermined}\n") print(f"b = {b_overdetermined}\n") print(f"最小二乗解 x = {x_pinv}") # 残差 residual = A_overdetermined @ x_pinv - b_overdetermined print(f"残差: {residual}") print(f"残差ノルム: {np.linalg.norm(residual):.4f}") # np.linalg.lstsqとの比較 x_lstsq = np.linalg.lstsq(A_overdetermined, b_overdetermined, rcond=None)[0] print(f"\nlstsq解: {x_lstsq}") print(f"一致? {np.allclose(x_pinv, x_lstsq)}")

## 4.6 推薦システムへの応用

**🔬 応用例: 協調フィルタリング**  
ユーザー×アイテムの評価行列を低ランク近似することで、 未評価アイテムの評価値を予測できます（Netflix Prize問題など）。 

### 💻 コード例6: 行列補完（Matrix Completion）

Python実装: 推薦システム

# ユーザー×映画の評価行列（5点満点、0は未評価） ratings = np.array([[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4]]) # 観測された評価のみを使ってSVD # 簡易版: 0を平均値で埋める mean_rating = ratings[ratings > 0].mean() ratings_filled = np.where(ratings > 0, ratings, mean_rating) # SVDによる低ランク近似 U_rec, s_rec, VT_rec = np.linalg.svd(ratings_filled, full_matrices=False) k = 2 # 潜在因子数 ratings_predicted = U_rec[:, :k] @ np.diag(s_rec[:k]) @ VT_rec[:k, :] print("推薦システム（協調フィルタリング）:") print(f"元の評価行列 (0は未評価):\n{ratings}\n") print(f"予測評価行列 (ランク{k}近似):\n{np.round(ratings_predicted, 2)}\n") # 未評価アイテムの予測値 print("未評価アイテムの予測:") for i in range(ratings.shape[0]): for j in range(ratings.shape[1]): if ratings[i, j] == 0: print(f"ユーザー{i+1}, 映画{j+1}: 予測評価 = {ratings_predicted[i, j]:.2f}")

### 💻 コード例7: 潜在因子の解釈

Python実装: 潜在因子の可視化

# ユーザー因子と映画因子 user_factors = U_rec[:, :k] @ np.diag(np.sqrt(s_rec[:k])) movie_factors = np.diag(np.sqrt(s_rec[:k])) @ VT_rec[:k, :] print("\n潜在因子分析:") print(f"ユーザー因子 (5ユーザー × 2因子):\n{user_factors}\n") print(f"映画因子 (2因子 × 4映画):\n{movie_factors}\n") # 可視化 plt.figure(figsize=(10, 8)) plt.scatter(user_factors[:, 0], user_factors[:, 1], s=100, c='blue', marker='o', label='ユーザー') for i in range(len(user_factors)): plt.annotate(f'U{i+1}', (user_factors[i, 0], user_factors[i, 1]), fontsize=12, ha='right') plt.scatter(movie_factors[0, :], movie_factors[1, :], s=100, c='red', marker='s', label='映画') for j in range(movie_factors.shape[1]): plt.annotate(f'M{j+1}', (movie_factors[0, j], movie_factors[1, j]), fontsize=12, ha='left') plt.xlabel('潜在因子1') plt.ylabel('潜在因子2') plt.title('ユーザーと映画の潜在因子空間') plt.legend() plt.grid(True, alpha=0.3) plt.axhline(y=0, color='k', linewidth=0.5) plt.axvline(x=0, color='k', linewidth=0.5) plt.show()

### 💻 コード例8: ノイズ除去への応用

Python実装: SVDによるノイズ除去

# ノイズを含むデータ np.random.seed(42) true_signal = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15]]) noise = np.random.randn(3, 5) * 0.5 noisy_signal = true_signal + noise # SVDによるノイズ除去 U_noise, s_noise, VT_noise = np.linalg.svd(noisy_signal, full_matrices=False) print("SVDによるノイズ除去:") print(f"特異値: {s_noise}") # 最大特異値のみ使用（ランク1近似） k_denoise = 1 denoised = U_noise[:, :k_denoise] @ np.diag(s_noise[:k_denoise]) @ VT_noise[:k_denoise, :] print(f"\nノイズ除去前の誤差: {np.linalg.norm(noisy_signal - true_signal):.4f}") print(f"ノイズ除去後の誤差: {np.linalg.norm(denoised - true_signal):.4f}") # 可視化 fig, axes = plt.subplots(1, 3, figsize=(15, 4)) axes[0].imshow(true_signal, cmap='viridis', aspect='auto') axes[0].set_title('真の信号') axes[0].set_xlabel('時間') axes[0].set_ylabel('チャネル') axes[1].imshow(noisy_signal, cmap='viridis', aspect='auto') axes[1].set_title('ノイズを含む信号') axes[1].set_xlabel('時間') axes[2].imshow(denoised, cmap='viridis', aspect='auto') axes[2].set_title('ノイズ除去後（ランク1近似）') axes[2].set_xlabel('時間') plt.tight_layout() plt.show()

## まとめ

  * SVDは任意の行列を3つの直交・対角行列の積に分解する強力な手法
  * 低ランク近似により、データ圧縮・ノイズ除去が可能
  * 擬似逆行列により、過剰決定系・劣決定系の最小二乗解が求まる
  * 画像圧縮、推薦システム、信号処理など幅広い応用がある
  * 潜在因子分析により、データの背後にある隠れた構造を発見できる

[← 第3章: 固有値・固有ベクトル](<chapter-3.html>) [第5章: テンソル解析 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
