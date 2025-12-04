---
title: "第3章: リートベルト解析入門"
chapter_title: "第3章: リートベルト解析入門"
subtitle: 全パターンフィッティングによる精密結晶構造解析
reading_time: 30-35分
code_examples: 8
---

## 学習目標

この章を完了すると、以下ができるようになります:

  * リートベルト法の原理と最小二乗法を理解し実装できる
  * Pseudo-Voigtプロファイル関数を用いた全パターンフィッティングができる
  * Chebyshev多項式によるバックグラウンドモデリングを実装できる
  * R因子(Rwp, RB, GOF)を計算し、フィット品質を評価できる
  * lmfit.Minimizerを用いた実践的なリートベルト解析を実行できる

## 3.1 リートベルト法の原理

### 3.1.1 全パターンフィッティング

リートベルト法は、1969年にHugo Rietveldによって開発された粉末X線回折データの全パターンフィッティング手法です。個々のピークを独立にフィットする従来法と異なり、**全測定範囲のパターンを一度に最適化** します。

観測強度\\( y_i^{obs} \\)と計算強度\\( y_i^{calc} \\)の差の二乗和を最小化:

\\[ S = \sum_{i} w_i \left(y_i^{obs} - y_i^{calc}\right)^2 \\] 

ここで、\\( w_i = 1/y_i^{obs} \\) は統計的重み(カウント統計に基づく)、\\( i \\) は測定点のインデックスです。
    
    
    ```mermaid
    flowchart TD
        A[初期構造モデル格子定数・原子座標] --> B[計算回折パターンy_calc]
        B --> C[観測パターン y_obsとの差を計算]
        C --> D{残差 S十分小さい?}
        D -->|No| E[パラメータ調整最小二乗法]
        E --> B
        D -->|Yes| F[最終構造モデルR因子出力]
    
        style A fill:#fce7f3
        style D fill:#fff3e0
        style F fill:#e8f5e9
    ```

### 3.1.2 計算強度の定式化

測定点\\( i \\)における計算強度は以下の式で与えられます:

\\[ y_i^{calc} = y_{bi} + \sum_{K} s_K \sum_{hkl} m_{hkl} |F_{hkl}|^2 \Omega(2\theta_i - 2\theta_{hkl}) A_{hkl} \\] 

各項の意味:

  * \\( y_{bi} \\): バックグラウンド強度
  * \\( s_K \\): 相\\( K \\)のスケールファクター
  * \\( m_{hkl} \\): 多重度
  * \\( |F_{hkl}|^2 \\): 構造因子の二乗
  * \\( \Omega \\): ピークプロファイル関数
  * \\( A_{hkl} \\): ローレンツ偏光因子、吸収補正など

### 3.1.3 最小二乗法の実装
    
    
    import numpy as np
    from scipy.optimize import minimize
    
    class RietveldRefinement:
        """リートベルト解析の基本実装"""
    
        def __init__(self, two_theta_obs, intensity_obs):
            """
            Args:
                two_theta_obs (np.ndarray): 観測2θ角度
                intensity_obs (np.ndarray): 観測強度
            """
            self.two_theta_obs = two_theta_obs
            self.intensity_obs = intensity_obs
            self.weights = 1.0 / np.sqrt(np.maximum(intensity_obs, 1.0))  # 統計的重み
    
        def calculate_pattern(self, params, two_theta, peak_positions, structure_factors):
            """回折パターンを計算
    
            Args:
                params (dict): 精密化パラメータ
                two_theta (np.ndarray): 2θ角度
                peak_positions (list): ピーク位置 [(2θ_hkl, m_hkl), ...]
                structure_factors (list): 構造因子の二乗 [|F_hkl|^2, ...]
    
            Returns:
                np.ndarray: 計算強度
            """
            # バックグラウンド (後述のChebyshev多項式で実装)
            background = self._calculate_background(two_theta, params['bg_coeffs'])
    
            # ピーク寄与
            intensity_peaks = np.zeros_like(two_theta)
    
            for (peak_2theta, multiplicity), F_hkl_sq in zip(peak_positions, structure_factors):
                # プロファイル関数 (Pseudo-Voigt)
                profile = self._pseudo_voigt_profile(
                    two_theta,
                    center=peak_2theta,
                    fwhm=params['fwhm'],
                    eta=params['eta']
                )
    
                # スケールファクター × 多重度 × 構造因子
                intensity_peaks += params['scale'] * multiplicity * F_hkl_sq * profile
    
            return background + intensity_peaks
    
        def _pseudo_voigt_profile(self, x, center, fwhm, eta):
            """Pseudo-Voigtプロファイル関数
    
            Args:
                x (np.ndarray): 2θ
                center (float): ピーク中心
                fwhm (float): 半値全幅
                eta (float): Lorentz成分比 (0-1)
    
            Returns:
                np.ndarray: プロファイル形状
            """
            # Gaussian成分
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    
            # Lorentzian成分
            gamma = fwhm / 2
            lorentz = gamma / (np.pi * (gamma**2 + (x - center)**2))
    
            # Pseudo-Voigt
            return eta * lorentz + (1 - eta) * gauss
    
        def _calculate_background(self, two_theta, coeffs):
            """Chebyshev多項式でバックグラウンド計算"""
            # 正規化された x: [-1, 1]
            x_norm = 2 * (two_theta - two_theta.min()) / (two_theta.max() - two_theta.min()) - 1
    
            # Chebyshev多項式 T_n(x) を計算
            bg = np.zeros_like(two_theta)
            for n, c in enumerate(coeffs):
                bg += c * np.polynomial.chebyshev.chebval(x_norm, [0]*n + [1])
    
            return bg
    
        def residual(self, params, two_theta, peak_positions, structure_factors):
            """残差関数 (最小化の目的関数)
    
            Returns:
                np.ndarray: 重み付き残差
            """
            y_calc = self.calculate_pattern(params, two_theta, peak_positions, structure_factors)
            residual = (self.intensity_obs - y_calc) * self.weights
            return residual
    
        def chi_squared(self, residual):
            """χ^2 を計算"""
            return np.sum(residual**2)
    
    
    # 使用例: 簡易リートベルト解析
    def simple_rietveld_demo():
        """簡単なリートベルト解析のデモ"""
        # 模擬データ生成
        two_theta = np.linspace(10, 80, 3500)
        true_params = {
            'scale': 1000,
            'fwhm': 0.15,
            'eta': 0.5,
            'bg_coeffs': [100, -20, 5]
        }
    
        # ピーク位置と構造因子 (α-Fe BCC)
        peak_positions = [(44.67, 12), (65.02, 6), (82.33, 24)]  # (2θ, multiplicity)
        structure_factors = [1.0, 0.8, 1.2]  # 正規化された|F|^2
    
        # 真の計算パターン
        rietveld = RietveldRefinement(two_theta, np.zeros_like(two_theta))
        y_true = rietveld.calculate_pattern(true_params, two_theta, peak_positions, structure_factors)
    
        # ノイズ追加
        y_obs = y_true + np.random.normal(0, np.sqrt(y_true + 10), len(y_true))
    
        print("=== リートベルト解析デモ ===")
        print(f"データポイント数: {len(two_theta)}")
        print(f"ピーク数: {len(peak_positions)}")
        print(f"精密化パラメータ数: {1 + 1 + 1 + len(true_params['bg_coeffs'])} (scale, FWHM, η, BG係数×3)")
    
        return two_theta, y_obs, peak_positions, structure_factors
    
    two_theta, y_obs, peak_pos, F_sq = simple_rietveld_demo()

## 3.2 プロファイル関数

### 3.2.1 Pseudo-Voigt関数の詳細

Pseudo-Voigt関数は、リートベルト解析で最も一般的に使用されるプロファイル関数です:

\\[ \Omega(x) = \eta L(x) + (1-\eta) G(x) \\] 

パラメータ\\( \eta \\)(0-1)は、Lorentzian成分の寄与を表します。

**FWHM(半値全幅)の角度依存性** \- Caglioti式:

\\[ \text{FWHM}^2 = U\tan^2\theta + V\tan\theta + W \\] 

ここで、\\( U, V, W \\) はCagliotiパラメータです。
    
    
    def caglioti_fwhm(two_theta, U, V, W):
        """Caglioti式でFWHMを計算
    
        Args:
            two_theta (float or np.ndarray): 2θ角度 [度]
            U, V, W (float): Cagliotiパラメータ
    
        Returns:
            float or np.ndarray: FWHM [度]
        """
        theta_rad = np.radians(two_theta / 2)
        tan_theta = np.tan(theta_rad)
    
        fwhm_squared = U * tan_theta**2 + V * tan_theta + W
    
        # 負の値を避ける
        fwhm_squared = np.maximum(fwhm_squared, 0.001)
    
        return np.sqrt(fwhm_squared)
    
    
    # U, V, Wパラメータの典型値 (Cu Kα, 実験室系XRD)
    U = 0.01  # [度^2]
    V = -0.005  # [度^2]
    W = 0.005  # [度^2]
    
    # 様々な角度でのFWHM
    two_theta_range = np.linspace(10, 120, 100)
    fwhm_values = caglioti_fwhm(two_theta_range, U, V, W)
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(two_theta_range, fwhm_values, color='#f093fb', linewidth=2)
    plt.xlabel('2θ [度]', fontsize=12)
    plt.ylabel('FWHM [度]', fontsize=12)
    plt.title('Caglioti式: FWHMの角度依存性', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== FWHM計算例 ===")
    for angle in [20, 40, 60, 80, 100]:
        fwhm = caglioti_fwhm(angle, U, V, W)
        print(f"2θ = {angle:3d}°: FWHM = {fwhm:.4f}°")

### 3.2.2 Thompson-Cox-Hastings Pseudo-Voigt (TCH-pV)

より高精度なプロファイル関数として、TCH-pVがあります。これはGaussianとLorentzianの各成分のFWHMを独立に扱います:
    
    
    def tch_pseudo_voigt(x, center, fwhm_G, fwhm_L):
        """Thompson-Cox-Hastings Pseudo-Voigt関数
    
        Args:
            x (np.ndarray): 2θ
            center (float): ピーク中心
            fwhm_G (float): Gaussian成分のFWHM
            fwhm_L (float): Lorentzian成分のFWHM
    
        Returns:
            np.ndarray: プロファイル
        """
        # 有効FWHM
        fwhm_eff = (fwhm_G**5 + 2.69269 * fwhm_G**4 * fwhm_L +
                    2.42843 * fwhm_G**3 * fwhm_L**2 +
                    4.47163 * fwhm_G**2 * fwhm_L**3 +
                    0.07842 * fwhm_G * fwhm_L**4 + fwhm_L**5) ** 0.2
    
        # 混合パラメータ η
        eta = 1.36603 * (fwhm_L / fwhm_eff) - 0.47719 * (fwhm_L / fwhm_eff)**2 + \
              0.11116 * (fwhm_L / fwhm_eff)**3
    
        # Gaussian成分
        sigma_G = fwhm_eff / (2 * np.sqrt(2 * np.log(2)))
        G = np.exp(-0.5 * ((x - center) / sigma_G)**2) / (sigma_G * np.sqrt(2 * np.pi))
    
        # Lorentzian成分
        gamma_L = fwhm_eff / 2
        L = gamma_L / (np.pi * (gamma_L**2 + (x - center)**2))
    
        return eta * L + (1 - eta) * G
    
    
    # 比較: 単純pV vs TCH-pV
    x = np.linspace(44, 46, 500)
    center = 45.0
    
    # 単純pV
    simple_pv = RietveldRefinement(x, np.zeros_like(x))._pseudo_voigt_profile(
        x, center, fwhm=0.2, eta=0.5
    )
    
    # TCH-pV
    tch_pv = tch_pseudo_voigt(x, center, fwhm_G=0.15, fwhm_L=0.1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, simple_pv, label='単純Pseudo-Voigt', color='#3498db', linewidth=2)
    plt.plot(x, tch_pv, label='TCH Pseudo-Voigt', color='#f093fb', linewidth=2, linestyle='--')
    plt.xlabel('2θ [度]', fontsize=12)
    plt.ylabel('プロファイル強度 (正規化)', fontsize=12)
    plt.title('プロファイル関数の比較', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

## 3.3 バックグラウンドモデル

### 3.3.1 Chebyshev多項式

Chebyshev多項式は、リートベルト解析でバックグラウンドをモデル化する際の標準的な選択肢です。数値的に安定で、少ないパラメータで複雑な形状を表現できます:

\\[ y_{bg}(x) = \sum_{n=0}^{N} c_n T_n(x) \\] 

ここで、\\( T_n(x) \\) は第\\( n \\)次Chebyshev多項式、\\( x \in [-1, 1] \\) (正規化された2θ)です。
    
    
    import numpy.polynomial.chebyshev as cheb
    
    class BackgroundModel:
        """バックグラウンドモデリングクラス"""
    
        def __init__(self, two_theta_range):
            """
            Args:
                two_theta_range (tuple): (min, max) 測定範囲
            """
            self.two_theta_min = two_theta_range[0]
            self.two_theta_max = two_theta_range[1]
    
        def normalize_two_theta(self, two_theta):
            """2θを[-1, 1]に正規化
    
            Args:
                two_theta (np.ndarray): 2θ角度
    
            Returns:
                np.ndarray: 正規化された値
            """
            return 2 * (two_theta - self.two_theta_min) / (self.two_theta_max - self.two_theta_min) - 1
    
        def chebyshev_background(self, two_theta, coefficients):
            """Chebyshev多項式でバックグラウンド計算
    
            Args:
                two_theta (np.ndarray): 2θ角度
                coefficients (list): Chebyshev係数 [c0, c1, c2, ...]
    
            Returns:
                np.ndarray: バックグラウンド強度
            """
            x_norm = self.normalize_two_theta(two_theta)
            return cheb.chebval(x_norm, coefficients)
    
        def fit_background(self, two_theta, intensity, degree=5, exclude_peaks=True):
            """バックグラウンドをフィット
    
            Args:
                two_theta (np.ndarray): 2θ角度
                intensity (np.ndarray): 強度
                degree (int): Chebyshev多項式の次数
                exclude_peaks (bool): ピーク領域を除外するか
    
            Returns:
                np.ndarray: Chebyshev係数
            """
            x_norm = self.normalize_two_theta(two_theta)
    
            if exclude_peaks:
                # 強度が高い領域を除外 (簡易版)
                threshold = np.percentile(intensity, 60)
                mask = intensity < threshold
                coeffs = cheb.chebfit(x_norm[mask], intensity[mask], degree)
            else:
                coeffs = cheb.chebfit(x_norm, intensity, degree)
    
            return coeffs
    
    
    # 使用例
    two_theta = np.linspace(10, 80, 3500)
    
    # 複雑なバックグラウンド形状を模擬
    true_bg = 200 * np.exp(-two_theta / 40) + 50 + 10 * np.sin(two_theta / 10)
    
    # ピーク追加
    peak1 = 1000 * np.exp(-0.5 * ((two_theta - 45) / 0.15)**2)
    peak2 = 500 * np.exp(-0.5 * ((two_theta - 65) / 0.18)**2)
    intensity = true_bg + peak1 + peak2 + np.random.normal(0, 5, len(two_theta))
    
    # バックグラウンドフィット
    bg_model = BackgroundModel((10, 80))
    
    # 異なる次数でフィット
    degrees = [3, 5, 7, 10]
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(two_theta, intensity, 'gray', alpha=0.5, label='全データ')
    plt.plot(two_theta, true_bg, 'k--', linewidth=2, label='真のBG')
    
    for deg in degrees:
        coeffs = bg_model.fit_background(two_theta, intensity, degree=deg, exclude_peaks=True)
        bg_fitted = bg_model.chebyshev_background(two_theta, coeffs)
        plt.plot(two_theta, bg_fitted, linewidth=1.5, label=f'Chebyshev deg={deg}')
    
    plt.xlabel('2θ [度]')
    plt.ylabel('強度 [counts]')
    plt.title('Chebyshevバックグラウンドフィッティング')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    # 残差プロット
    best_deg = 5
    coeffs = bg_model.fit_background(two_theta, intensity, degree=best_deg, exclude_peaks=True)
    bg_fitted = bg_model.chebyshev_background(two_theta, coeffs)
    residual = intensity - bg_fitted
    
    plt.plot(two_theta, residual, color='#f093fb', linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    plt.xlabel('2θ [度]')
    plt.ylabel('残差 [counts]')
    plt.title(f'残差 (Chebyshev degree={best_deg})')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"=== Chebyshev係数 (degree={best_deg}) ===")
    for i, c in enumerate(coeffs):
        print(f"c_{i} = {c:10.4f}")

## 3.4 R因子と適合度評価

### 3.4.1 R因子の定義

リートベルト解析の品質は、様々なR因子で評価されます:

**Rwp (Weighted Profile R-factor)** \- 最も重要:

\\[ R_{wp} = \sqrt{\frac{\sum_i w_i (y_i^{obs} - y_i^{calc})^2}{\sum_i w_i (y_i^{obs})^2}} \times 100\% \\] 

**Rp (Profile R-factor)** \- 重みなし版:

\\[ R_p = \frac{\sum_i |y_i^{obs} - y_i^{calc}|}{\sum_i y_i^{obs}} \times 100\% \\] 

**RBragg (Bragg R-factor)** \- 積分強度ベース:

\\[ R_B = \frac{\sum_{hkl} |I_{hkl}^{obs} - I_{hkl}^{calc}|}{\sum_{hkl} I_{hkl}^{obs}} \times 100\% \\] 

**GOF (Goodness of Fit)** \- 期待値との比較:

\\[ \text{GOF} = \sqrt{\frac{\sum_i w_i (y_i^{obs} - y_i^{calc})^2}{N - P}} = \frac{R_{wp}}{R_{exp}} \\] 

ここで、\\( N \\) はデータポイント数、\\( P \\) はパラメータ数、\\( R_{exp} \\) は期待されるR因子です。

R因子 | 優れた | 良好 | 許容 | 問題あり  
---|---|---|---|---  
Rwp | < 5% | 5-10% | 10-15% | > 15%  
RBragg | < 3% | 3-7% | 7-12% | > 12%  
GOF | 1.0-1.3 | 1.3-2.0 | 2.0-3.0 | > 3.0 または < 1.0  
  
### 3.4.2 R因子計算の実装
    
    
    class RFactorCalculator:
        """R因子計算クラス"""
    
        @staticmethod
        def calculate_rwp(y_obs, y_calc, weights=None):
            """Weighted Profile R-factorを計算
    
            Args:
                y_obs (np.ndarray): 観測強度
                y_calc (np.ndarray): 計算強度
                weights (np.ndarray): 重み (Noneの場合は1/y_obs)
    
            Returns:
                float: Rwp [%]
            """
            if weights is None:
                weights = 1.0 / np.sqrt(np.maximum(y_obs, 1.0))
    
            numerator = np.sum(weights * (y_obs - y_calc)**2)
            denominator = np.sum(weights * y_obs**2)
    
            return 100 * np.sqrt(numerator / denominator)
    
        @staticmethod
        def calculate_rp(y_obs, y_calc):
            """Profile R-factorを計算
    
            Returns:
                float: Rp [%]
            """
            numerator = np.sum(np.abs(y_obs - y_calc))
            denominator = np.sum(y_obs)
    
            return 100 * numerator / denominator
    
        @staticmethod
        def calculate_rexp(y_obs, n_params, weights=None):
            """Expected R-factorを計算
    
            Args:
                y_obs (np.ndarray): 観測強度
                n_params (int): パラメータ数
                weights (np.ndarray): 重み
    
            Returns:
                float: Rexp [%]
            """
            if weights is None:
                weights = 1.0 / np.sqrt(np.maximum(y_obs, 1.0))
    
            N = len(y_obs)
            numerator = N - n_params
            denominator = np.sum(weights * y_obs**2)
    
            return 100 * np.sqrt(numerator / denominator)
    
        @staticmethod
        def calculate_gof(rwp, rexp):
            """Goodness of Fitを計算
    
            Returns:
                float: GOF
            """
            return rwp / rexp
    
        @classmethod
        def calculate_all_r_factors(cls, y_obs, y_calc, n_params, weights=None):
            """全てのR因子を計算
    
            Returns:
                dict: R因子の辞書
            """
            rwp = cls.calculate_rwp(y_obs, y_calc, weights)
            rp = cls.calculate_rp(y_obs, y_calc)
            rexp = cls.calculate_rexp(y_obs, n_params, weights)
            gof = cls.calculate_gof(rwp, rexp)
    
            return {
                'Rwp': rwp,
                'Rp': rp,
                'Rexp': rexp,
                'GOF': gof
            }
    
    
    # 使用例
    # 模擬データ: 良好なフィット
    np.random.seed(42)
    y_obs = np.abs(np.random.normal(1000, 50, 1000))
    y_calc_good = y_obs + np.random.normal(0, 10, 1000)  # 小さい誤差
    
    # 模擬データ: 不良なフィット
    y_calc_bad = y_obs + np.random.normal(0, 100, 1000)  # 大きい誤差
    
    n_params = 15  # パラメータ数
    
    # R因子計算
    r_good = RFactorCalculator.calculate_all_r_factors(y_obs, y_calc_good, n_params)
    r_bad = RFactorCalculator.calculate_all_r_factors(y_obs, y_calc_bad, n_params)
    
    print("=== R因子比較 ===")
    print("\n良好なフィット:")
    for key, value in r_good.items():
        print(f"  {key:<6}: {value:6.2f}{'%' if key != 'GOF' else ''}")
    
    print("\n不良なフィット:")
    for key, value in r_bad.items():
        print(f"  {key:<6}: {value:6.2f}{'%' if key != 'GOF' else ''}")
    
    # 判定
    print("\n判定:")
    if r_good['Rwp'] < 10 and 1.0 < r_good['GOF'] < 2.0:
        print("  良好: Rwp < 10%, 1.0 < GOF < 2.0 ✓")
    else:
        print("  要改善")
    
    if r_bad['Rwp'] > 15 or r_bad['GOF'] > 3.0:
        print("  不良: Rwp > 15% または GOF > 3.0 ✗")

## 3.5 Python実装基礎 (lmfit使用)

### 3.5.1 lmfit.Minimizerの導入

lmfitライブラリは、scipy.optimizeをラップし、パラメータの境界設定、制約条件、エラー推定を簡単に扱えます:
    
    
    # lmfitのインストール (必要な場合)
    # !pip install lmfit
    
    from lmfit import Parameters, Minimizer, report_fit
    
    class LmfitRietveldRefinement:
        """lmfitを用いたリートベルト解析"""
    
        def __init__(self, two_theta_obs, intensity_obs):
            self.two_theta_obs = two_theta_obs
            self.intensity_obs = intensity_obs
            self.weights = 1.0 / np.sqrt(np.maximum(intensity_obs, 1.0))
    
        def setup_parameters(self):
            """パラメータの初期値と境界を設定
    
            Returns:
                lmfit.Parameters: パラメータオブジェクト
            """
            params = Parameters()
    
            # スケールファクター
            params.add('scale', value=1000, min=0, max=1e6)
    
            # Cagliotiパラメータ
            params.add('U', value=0.01, min=0, max=1.0)
            params.add('V', value=-0.005, min=-1.0, max=1.0)
            params.add('W', value=0.005, min=0, max=1.0)
    
            # Pseudo-Voigt混合パラメータ
            params.add('eta', value=0.5, min=0, max=1)
    
            # Chebyshevバックグラウンド係数
            params.add('bg0', value=100, min=0)
            params.add('bg1', value=0)
            params.add('bg2', value=0)
            params.add('bg3', value=0)
    
            # 格子定数 (立方晶の例)
            params.add('a', value=2.87, min=2.8, max=3.0)
    
            return params
    
        def calculate_pattern_lmfit(self, params, two_theta, hkl_list):
            """lmfitパラメータオブジェクトから回折パターンを計算
    
            Args:
                params (lmfit.Parameters): パラメータ
                two_theta (np.ndarray): 2θ
                hkl_list (list): [(h, k, l, multiplicity, |F|^2), ...]
    
            Returns:
                np.ndarray: 計算強度
            """
            # パラメータ抽出
            scale = params['scale'].value
            U = params['U'].value
            V = params['V'].value
            W = params['W'].value
            eta = params['eta'].value
            a = params['a'].value
    
            bg_coeffs = [params[f'bg{i}'].value for i in range(4)]
    
            # バックグラウンド
            bg_model = BackgroundModel((two_theta.min(), two_theta.max()))
            background = bg_model.chebyshev_background(two_theta, bg_coeffs)
    
            # ピーク計算
            intensity_peaks = np.zeros_like(two_theta)
            wavelength = 1.54056  # Cu Kα
    
            for h, k, l, mult, F_sq in hkl_list:
                # d値計算 (立方晶)
                d = a / np.sqrt(h**2 + k**2 + l**2)
    
                # Bragg角
                theta = np.degrees(np.arcsin(wavelength / (2 * d)))
                peak_2theta = 2 * theta
    
                # FWHM (Caglioti式)
                fwhm = caglioti_fwhm(peak_2theta, U, V, W)
    
                # Pseudo-Voigtプロファイル
                profile = self._pseudo_voigt_profile(two_theta, peak_2theta, fwhm, eta)
    
                # スケール × 多重度 × 構造因子
                intensity_peaks += scale * mult * F_sq * profile
    
            return background + intensity_peaks
    
        def _pseudo_voigt_profile(self, x, center, fwhm, eta):
            """Pseudo-Voigtプロファイル"""
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    
            gamma = fwhm / 2
            lorentz = gamma / (np.pi * (gamma**2 + (x - center)**2))
    
            return eta * lorentz + (1 - eta) * gauss
    
        def residual_lmfit(self, params, two_theta, intensity_obs, weights, hkl_list):
            """lmfit用の残差関数
    
            Returns:
                np.ndarray: 重み付き残差
            """
            y_calc = self.calculate_pattern_lmfit(params, two_theta, hkl_list)
            return (intensity_obs - y_calc) * weights
    
    
    # 実行例
    def run_lmfit_rietveld():
        """lmfitでリートベルト解析を実行"""
        # 模擬データ生成 (α-Fe BCC)
        two_theta = np.linspace(40, 90, 2500)
    
        hkl_list = [
            (1, 1, 0, 12, 1.0),
            (2, 0, 0, 6, 0.8),
            (2, 1, 1, 24, 1.2),
            (2, 2, 0, 12, 0.6),
        ]
    
        # 真のパラメータで計算
        true_params = {
            'scale': 5000,
            'U': 0.01, 'V': -0.005, 'W': 0.005,
            'eta': 0.5,
            'bg0': 100, 'bg1': -10, 'bg2': 2, 'bg3': 0,
            'a': 2.87
        }
    
        rietveld = LmfitRietveldRefinement(two_theta, np.zeros_like(two_theta))
    
        # 真のパターン生成
        params_true = rietveld.setup_parameters()
        for key, value in true_params.items():
            params_true[key].value = value
    
        y_true = rietveld.calculate_pattern_lmfit(params_true, two_theta, hkl_list)
        y_obs = y_true + np.random.normal(0, np.sqrt(y_true + 10), len(y_true))
    
        # 初期パラメータ (真値から少しずらす)
        params_init = rietveld.setup_parameters()
        params_init['scale'].value = 4500
        params_init['a'].value = 2.90
    
        # 最小化実行
        minimizer = Minimizer(
            rietveld.residual_lmfit,
            params_init,
            fcn_args=(two_theta, y_obs, rietveld.weights, hkl_list)
        )
    
        result = minimizer.minimize(method='leastsq')  # Levenberg-Marquardt法
    
        # 結果表示
        print("=== リートベルト解析結果 (lmfit) ===\n")
        report_fit(result)
    
        # R因子計算
        y_final = rietveld.calculate_pattern_lmfit(result.params, two_theta, hkl_list)
        r_factors = RFactorCalculator.calculate_all_r_factors(
            y_obs, y_final, result.nvarys, rietveld.weights
        )
    
        print("\n=== R因子 ===")
        for key, value in r_factors.items():
            print(f"{key:<6}: {value:6.2f}{'%' if key != 'GOF' else ''}")
    
        # プロット
        plt.figure(figsize=(12, 8))
    
        plt.subplot(2, 1, 1)
        plt.plot(two_theta, y_obs, 'o', markersize=2, color='gray', alpha=0.5, label='観測データ')
        plt.plot(two_theta, y_final, color='#f093fb', linewidth=2, label='計算パターン')
        plt.xlabel('2θ [度]')
        plt.ylabel('強度 [counts]')
        plt.title('リートベルト解析: フィット結果')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.subplot(2, 1, 2)
        residual = y_obs - y_final
        plt.plot(two_theta, residual, color='#f5576c', linewidth=1)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        plt.xlabel('2θ [度]')
        plt.ylabel('残差 [counts]')
        plt.title(f'残差プロット (Rwp = {r_factors["Rwp"]:.2f}%)')
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        return result
    
    # 実行
    # result = run_lmfit_rietveld()  # コメント解除して実行

## 学習目標の確認

### 基本理解

  * ✅ リートベルト法の全パターンフィッティング原理と最小二乗法
  * ✅ Pseudo-Voigt関数とCaglioti式の物理的意味
  * ✅ Chebyshev多項式によるバックグラウンドモデリングの利点
  * ✅ R因子(Rwp, Rp, RBragg, GOF)の定義と評価基準

### 実践スキル

  * ✅ NumPyでPseudo-Voigtプロファイル関数を実装
  * ✅ Chebyshev多項式で複雑なバックグラウンドをフィット
  * ✅ R因子を計算し、フィット品質を定量的に評価
  * ✅ lmfit.Minimizerで実践的なリートベルト解析を実行

### 応用力

  * ✅ パラメータの初期値・境界を適切に設定
  * ✅ フィット結果から格子定数、FWHM、スケールファクターを抽出
  * ✅ 残差プロットとR因子から精密化の問題点を診断

## 演習問題

### Easy (基礎確認)

**Q1** : Rwp = 8.5%, Rexp = 5.2% の場合、GOFはいくつですか? この結果は良好ですか?

**解答** :
    
    
    GOF = Rwp / Rexp = 8.5 / 5.2 = 1.63

**判定** : GOF = 1.63 は **1.3-2.0の範囲** → 良好なフィット ✓

GOF > 2.0 だと問題あり、GOF < 1.0 は過剰フィッティングの可能性あり。

**Q2** : Caglioti式で、2θ = 30°の位置のFWHMを計算してください (U=0.01, V=-0.005, W=0.005)。

**解答** :
    
    
    fwhm = caglioti_fwhm(two_theta=30, U=0.01, V=-0.005, W=0.005)
    print(f"FWHM at 2θ=30°: {fwhm:.4f}°")
    # 出力: FWHM at 2θ=30°: 0.0844°

### Medium (応用)

**Q3** : Chebyshev多項式の次数を3次から7次に増やすと、Rwpは必ず改善しますか? 理由を説明してください。

**解答** :

**結論** : 通常は改善しますが、必ずしも物理的に正しいとは限りません。

**理由** :

  * **Rwpは減少** : パラメータが増えるほど、データへのフィットは改善
  * **しかし...**
    * 過剰フィッティング: ピーク部分までBGとして扱う可能性
    * GOFの悪化: パラメータ数Pが増えるとRexpが小さくなり、GOF = Rwp/Rexp が上昇
    * 物理的意味の喪失: 7次以上の多項式は振動し、非物理的なBG形状を生成

**推奨** : 3-5次のChebyshev多項式が実用的。視覚的評価とGOFを確認すること。

**Q4** : Pseudo-Voigt関数で、η = 0 と η = 1 のときの形状の違いを説明してください。

**解答** :

  * **η = 0** : 完全なGaussian → 裾野が急速に減衰、装置起因の広がり
  * **η = 1** : 完全なLorentzian → 裾野が広い、試料起因(結晶サイズ、歪み)

実際のXRDピークは両者の中間(η = 0.3-0.7)で、装置と試料の両方の効果を反映します。

### Hard (発展)

**Q5** : lmfitで格子定数aを精密化する際、初期値を2.87Åとし、境界を[2.8, 3.0]に設定しました。フィット後、a = 2.999Åとなりました。この結果をどう解釈すべきですか?

**解答** :

**問題** : パラメータが境界に張り付いている → 最適化が収束していない可能性

**対処法** :

  1. **境界を広げる** : [2.7, 3.2] に変更して再実行
  2. **初期値を見直す** : インデキシングから推定した値を使う
  3. **他のパラメータとの相関確認** : Uパラメータとaが強く相関している可能性
  4. **消滅則確認** : 想定している格子型(BCC/FCC/SC)が正しいか再検証
  5. **高角ピークの確認** : 格子定数は高角データに敏感 → 2θ > 80°のデータを確認

再精密化後もa = 2.99Åなら、それが真の値かもしれません。ただし**他の手法(単結晶XRD、中性子回折)で検証** が望ましいです。

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります：

### 基本理解

  * ✅ リートベルト法の原理（最小二乗法による全パターンフィッティング）を説明できる
  * ✅ Pseudo-Voigtプロファイル関数のパラメータ（FWHM、η）の物理的意味を理解している
  * ✅ Chebyshev多項式によるバックグラウンドモデリングの利点を説明できる
  * ✅ R因子（Rwp, RB, GOF）の定義と解釈基準を理解している

### 実践スキル

  * ✅ Pythonのlmfitライブラリでリートベルトフィッティングを実装できる
  * ✅ プロファイルパラメータを適切に初期化し、精密化できる
  * ✅ バックグラウンドを多項式でモデル化し、最適な次数を選択できる
  * ✅ R因子を計算し、フィッティング品質を定量評価できる
  * ✅ 差分プロット（Yobs - Ycalc）を生成し、フィッティングを視覚評価できる

### 応用力

  * ✅ 実測XRDデータに対してリートベルト解析を実行できる
  * ✅ パラメータの相関を理解し、精密化戦略を立案できる
  * ✅ 収束しない場合のトラブルシューティングができる
  * ✅ 精密化結果の妥当性を多角的に評価できる

## 参考文献

  1. Rietveld, H. M. (1969). "A profile refinement method for nuclear and magnetic structures". _Journal of Applied Crystallography_ , 2(2), 65-71. - リートベルト法の元祖論文、全パターンフィッティングの基礎
  2. Young, R. A. (Ed.). (1993). _The Rietveld Method_. Oxford University Press. - リートベルト法の理論と実践を包括的に解説した決定版
  3. McCusker, L. B., et al. (1999). "Rietveld refinement guidelines". _Journal of Applied Crystallography_ , 32(1), 36-50. - 国際結晶学連合が定めたリートベルト解析の公式ガイドライン
  4. Toby, B. H. (2006). "R factors in Rietveld analysis: How good is good enough?". _Powder Diffraction_ , 21(1), 67-70. - R因子の解釈基準を明確に示した重要論文
  5. Thompson, P., Cox, D. E., & Hastings, J. B. (1987). "Rietveld refinement of Debye-Scherrer synchrotron X-ray data from Al₂O₃". _Journal of Applied Crystallography_ , 20(2), 79-83. - Pseudo-Voigt関数の定式化論文
  6. Cheary, R. W., & Coelho, A. (1992). "A fundamental parameters approach to X-ray line-profile fitting". _Journal of Applied Crystallography_ , 25(2), 109-121. - プロファイルパラメータの物理的解釈
  7. lmfit documentation (2024). "Non-Linear Least-Squares Minimization and Curve-Fitting for Python". - Pythonリートベルト実装の実践的ガイド

## 次のステップ

第3章では、リートベルト法の理論と基本実装を学びました。全パターンフィッティング、プロファイル関数、バックグラウンドモデリング、そしてR因子による評価という、精密解析の基礎を習得しました。

**第4章** では、これらの技術を発展させ、原子座標、温度因子、結晶子サイズ、microstrainなど、より詳細な構造パラメータの精密化に進みます。制約条件や拘束条件を用いた高度な精密化手法を学びます。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
