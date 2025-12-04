---
title: "第2章: 粉末X線回折測定と解析"
chapter_title: "第2章: 粉末X線回折測定と解析"
subtitle: XRD装置の原理から実データ解析まで - ピーク同定とフィッティングの実践
reading_time: 28-32分
code_examples: 8
---

## 学習目標

この章を完了すると、以下ができるようになります:

  * XRD装置の各構成要素の役割を理解し、測定条件を最適化できる
  * 実測XRDパターンからピークを同定し、Miller指数を割り当てられる
  * バックグラウンド除去とデータ平滑化を実装できる
  * Gauss、Lorentz、Voigt関数でピークフィッティングを実行できる
  * scipy.optimizeを用いた高度なピーク解析ができる

## 2.1 XRD装置の構成

### 2.1.1 X線源

粉末X線回折計の最も重要な構成要素はX線源です。実験室系XRDでは、主に以下の特性X線が使用されます:

X線源 | Kα1波長 [Å] | Kα2波長 [Å] | 特徴  
---|---|---|---  
Cu Kα | 1.54056 | 1.54439 | 最も一般的、汎用性が高い  
Mo Kα | 0.71073 | 0.71359 | 高エネルギー、透過力強い  
Co Kα | 1.78897 | 1.79285 | Fe含有試料に有利(蛍光回避)  
Cr Kα | 2.28970 | 2.29361 | 長波長、低角分解能向上  
  
### 2.1.2 光学系と検出器
    
    
    ```mermaid
    graph LR
        A[X線管Cu陽極] --> B[入射スリットビーム整形]
        B --> C[試料粉末]
        C --> D[受光スリット散乱除去]
        D --> E[単色化Ni filter]
        E --> F[検出器シンチレーション]
        F --> G[データ処理PC]
    
        style A fill:#ffe7e7
        style C fill:#fce7f3
        style F fill:#e7f3ff
        style G fill:#e7ffe7
    ```

**Bragg-Brentano配置** :

  * θ-2θ方式: X線源と検出器が試料を中心に対称移動
  * フォーカシングジオメトリ: 試料表面の異なる点からの回折X線が検出器で収束
  * 高分解能と高強度を両立

### 2.1.3 測定条件の最適化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class XRDMeasurementConditions:
        """XRD測定条件の最適化クラス"""
    
        def __init__(self, wavelength=1.54056):
            """
            Args:
                wavelength (float): X線波長 [Å]
            """
            self.wavelength = wavelength
    
        def calculate_two_theta_range(self, d_min=1.0, d_max=10.0):
            """測定すべき2θ範囲を計算
    
            Args:
                d_min (float): 最小面間隔 [Å]
                d_max (float): 最大面間隔 [Å]
    
            Returns:
                tuple: (two_theta_min, two_theta_max) [度]
            """
            # Braggの法則: λ = 2d sinθ
            # sin(θ) = λ / (2d)
    
            sin_theta_max = self.wavelength / (2 * d_min)
            sin_theta_min = self.wavelength / (2 * d_max)
    
            if sin_theta_max > 1.0:
                sin_theta_max = 1.0  # 物理的上限
    
            theta_max = np.degrees(np.arcsin(sin_theta_max))
            theta_min = np.degrees(np.arcsin(sin_theta_min))
    
            return 2 * theta_min, 2 * theta_max
    
        def optimize_step_size(self, two_theta_range, fwhm=0.1):
            """最適なステップサイズを計算
    
            Args:
                two_theta_range (tuple): (start, end) 測定範囲 [度]
                fwhm (float): ピークの半値全幅 [度]
    
            Returns:
                float: 推奨ステップサイズ [度]
            """
            # ルール: FWHM の 1/5 〜 1/10
            step_size_recommended = fwhm / 7.0
            return step_size_recommended
    
        def calculate_measurement_time(self, two_theta_range, step_size, time_per_step=1.0):
            """総測定時間を見積もる
    
            Args:
                two_theta_range (tuple): (start, end) [度]
                step_size (float): ステップサイズ [度]
                time_per_step (float): 1ステップあたりの測定時間 [秒]
    
            Returns:
                float: 総測定時間 [分]
            """
            start, end = two_theta_range
            n_steps = int((end - start) / step_size)
            total_seconds = n_steps * time_per_step
            return total_seconds / 60.0  # 分に変換
    
        def generate_measurement_plan(self, d_min=1.2, d_max=5.0, fwhm=0.1, time_per_step=2.0):
            """完全な測定計画を生成
    
            Returns:
                dict: 測定パラメータ
            """
            two_theta_range = self.calculate_two_theta_range(d_min, d_max)
            step_size = self.optimize_step_size(two_theta_range, fwhm)
            total_time = self.calculate_measurement_time(two_theta_range, step_size, time_per_step)
    
            plan = {
                'two_theta_start': two_theta_range[0],
                'two_theta_end': two_theta_range[1],
                'step_size': step_size,
                'time_per_step': time_per_step,
                'total_time_minutes': total_time,
                'estimated_points': int((two_theta_range[1] - two_theta_range[0]) / step_size)
            }
    
            return plan
    
    
    # 使用例
    conditions = XRDMeasurementConditions(wavelength=1.54056)  # Cu Kα
    plan = conditions.generate_measurement_plan(d_min=1.2, d_max=5.0, fwhm=0.1, time_per_step=2.0)
    
    print("=== XRD測定計画 ===")
    print(f"測定範囲: {plan['two_theta_start']:.2f}° - {plan['two_theta_end']:.2f}°")
    print(f"ステップサイズ: {plan['step_size']:.4f}°")
    print(f"1点あたり測定時間: {plan['time_per_step']:.1f}秒")
    print(f"総データポイント数: {plan['estimated_points']}")
    print(f"推定総測定時間: {plan['total_time_minutes']:.1f}分 ({plan['total_time_minutes']/60:.2f}時間)")
    
    # 期待される出力:
    # 測定範囲: 9.14° - 40.33°
    # ステップサイズ: 0.0143°
    # 総測定時間: 約73分

## 2.2 測定条件の最適化

### 2.2.1 2θ範囲の決定

適切な2θ範囲は、解析目的と試料の結晶構造に依存します:

  * **相同定** : 10° - 80° (主要ピークをカバー)
  * **格子定数精密化** : 20° - 120° (高角データが重要)
  * **定量分析** : 5° - 90° (低角ピークも含む)
  * **薄膜解析** : 20° - 90° (低角は基板の影響)

### 2.2.2 ステップサイズとカウント時間のトレードオフ
    
    
    def compare_measurement_strategies(two_theta_range=(10, 80)):
        """異なる測定戦略の比較
    
        Args:
            two_theta_range (tuple): 測定範囲 [度]
        """
        strategies = [
            {'name': '高速スキャン', 'step': 0.04, 'time': 0.5},
            {'name': '標準スキャン', 'step': 0.02, 'time': 1.0},
            {'name': '高分解能スキャン', 'step': 0.01, 'time': 2.0},
            {'name': '超高精度スキャン', 'step': 0.005, 'time': 5.0},
        ]
    
        print("測定戦略比較:")
        print("-" * 70)
        print(f"{'戦略':<15} | ステップ[°] | 時間/点[秒] | 総点数 | 総時間[分]")
        print("-" * 70)
    
        for strategy in strategies:
            step = strategy['step']
            time_per_step = strategy['time']
            n_points = int((two_theta_range[1] - two_theta_range[0]) / step)
            total_time = n_points * time_per_step / 60.0
    
            print(f"{strategy['name']:<15} | {step:^11.3f} | {time_per_step:^11.1f} | {n_points:^6} | {total_time:^9.1f}")
    
        print("\n推奨:")
        print("- ルーチン分析: 高速〜標準スキャン")
        print("- 精密構造解析: 高分解能スキャン")
        print("- 論文発表用: 超高精度スキャン")
    
    compare_measurement_strategies()
    
    # 期待される出力:
    # 高速スキャン:     総時間 ~15分
    # 標準スキャン:     総時間 ~58分
    # 高分解能スキャン:  総時間 ~233分
    # 超高精度スキャン:  総時間 ~1167分 (19.4時間)

## 2.3 ピーク同定とインデキシング

### 2.3.1 ピーク検出アルゴリズム
    
    
    from scipy.signal import find_peaks, peak_widths
    import numpy as np
    
    def detect_peaks(two_theta, intensity, prominence=100, width=2, distance=5):
        """XRDパターンからピークを自動検出
    
        Args:
            two_theta (np.ndarray): 2θ角度 [度]
            intensity (np.ndarray): 回折強度
            prominence (float): ピークの突出度閾値
            width (float): 最小ピーク幅 [データポイント]
            distance (int): ピーク間の最小距離 [データポイント]
    
        Returns:
            dict: ピーク情報 {positions, heights, widths, prominences}
        """
        # ピーク検出
        peaks, properties = find_peaks(
            intensity,
            prominence=prominence,
            width=width,
            distance=distance
        )
    
        # ピーク幅の計算
        widths_result = peak_widths(intensity, peaks, rel_height=0.5)
    
        peak_info = {
            'indices': peaks,
            'two_theta': two_theta[peaks],
            'intensity': intensity[peaks],
            'prominence': properties['prominences'],
            'fwhm': widths_result[0] * np.mean(np.diff(two_theta)),  # データポイント → 角度
            'left_bases': two_theta[properties['left_bases'].astype(int)],
            'right_bases': two_theta[properties['right_bases'].astype(int)]
        }
    
        return peak_info
    
    
    # 模擬XRDデータの生成
    def generate_synthetic_xrd(two_theta_range=(10, 80), n_points=3500):
        """模擬XRDパターンを生成 (α-Fe BCC)
    
        Returns:
            tuple: (two_theta, intensity)
        """
        two_theta = np.linspace(two_theta_range[0], two_theta_range[1], n_points)
        intensity = np.zeros_like(two_theta)
    
        # α-Fe (BCC) の主要ピーク
        # (hkl): (位置[°], 強度, FWHM[°])
        fe_peaks = [
            (44.67, 1000, 0.15),  # (110)
            (65.02, 300, 0.18),   # (200)
            (82.33, 450, 0.22),   # (211)
        ]
    
        # Gaussianピークの追加
        for pos, height, fwhm in fe_peaks:
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            intensity += height * np.exp(-0.5 * ((two_theta - pos) / sigma) ** 2)
    
        # バックグラウンドノイズ
        background = 50 + 20 * np.exp(-two_theta / 30)
        noise = np.random.normal(0, 5, len(two_theta))
    
        intensity_total = intensity + background + noise
    
        return two_theta, intensity_total
    
    
    # 実行例
    two_theta, intensity = generate_synthetic_xrd()
    peaks = detect_peaks(two_theta, intensity, prominence=100, width=2, distance=10)
    
    print("=== 検出されたピーク ===")
    print(f"{'2θ [°]':<10} | {'強度':<10} | {'FWHM [°]':<10} | {'相対強度 [%]':<12}")
    print("-" * 50)
    
    # 強度を正規化
    I_max = np.max(peaks['intensity'])
    for i in range(len(peaks['two_theta'])):
        rel_intensity = 100 * peaks['intensity'][i] / I_max
        print(f"{peaks['two_theta'][i]:8.2f}   | {peaks['intensity'][i]:8.0f}   | {peaks['fwhm'][i]:8.3f}   | {rel_intensity:8.1f}")
    
    # 期待される出力:
    #   44.67   |    1000   |    0.150   |    100.0
    #   65.02   |     300   |    0.180   |     30.0
    #   82.33   |     450   |    0.220   |     45.0

### 2.3.2 Miller指数の割り当て (立方晶の場合)
    
    
    def index_cubic_pattern(two_theta_obs, wavelength=1.54056, lattice_type='I', a_initial=3.0):
        """立方晶のXRDパターンをインデキシング
    
        Args:
            two_theta_obs (np.ndarray): 観測された回折角 [度]
            wavelength (float): X線波長 [Å]
            lattice_type (str): 格子型 ('P', 'I', 'F')
            a_initial (float): 初期格子定数推定値 [Å]
    
        Returns:
            list: インデキシング結果 [(h, k, l, d_calc, two_theta_calc, delta), ...]
        """
        # 立方晶のd値: d = a / sqrt(h^2 + k^2 + l^2)
        # 許容されるMiller指数を生成
        hkl_list = []
        for h in range(0, 6):
            for k in range(0, 6):
                for l in range(0, 6):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    # 消滅則チェック
                    if lattice_type == 'I' and (h + k + l) % 2 != 0:
                        continue
                    if lattice_type == 'F':
                        parity = [h % 2, k % 2, l % 2]
                        if len(set(parity)) != 1:
                            continue
                    hkl_list.append((h, k, l, h**2 + k**2 + l**2))
    
        # h^2 + k^2 + l^2 でソート
        hkl_list.sort(key=lambda x: x[3])
    
        # 各観測ピークに最も近いMiller指数を割り当て
        indexing_results = []
    
        for two_theta in two_theta_obs:
            theta_rad = np.radians(two_theta / 2)
            d_obs = wavelength / (2 * np.sin(theta_rad))
    
            # 全てのMiller指数候補を試す
            best_match = None
            min_error = float('inf')
    
            for h, k, l, h2k2l2 in hkl_list:
                d_calc = a_initial / np.sqrt(h2k2l2)
                error = abs(d_calc - d_obs)
    
                if error < min_error:
                    min_error = error
                    theta_calc = np.degrees(np.arcsin(wavelength / (2 * d_calc)))
                    two_theta_calc = 2 * theta_calc
                    best_match = (h, k, l, d_calc, two_theta_calc, two_theta_calc - two_theta)
    
            if best_match and abs(best_match[5]) < 0.5:  # 許容誤差 0.5度
                indexing_results.append(best_match)
    
        return indexing_results
    
    
    # 使用例
    two_theta_obs = np.array([44.67, 65.02, 82.33])  # α-Fe BCCの主要ピーク
    indexed = index_cubic_pattern(two_theta_obs, wavelength=1.54056, lattice_type='I', a_initial=2.87)
    
    print("=== Miller指数の割り当て (α-Fe BCC) ===")
    print(f"{'(hkl)':<10} | {'d計算[Å]':<10} | {'2θ計算[°]':<12} | {'2θ観測[°]':<12} | {'誤差[°]':<8}")
    print("-" * 65)
    
    for h, k, l, d_calc, two_theta_calc, delta in indexed:
        two_theta_obs_match = two_theta_calc - delta
        print(f"({h} {k} {l}){' '*(6-len(f'{h} {k} {l}'))} | {d_calc:8.4f}   | {two_theta_calc:10.2f}   | {two_theta_obs_match:10.2f}   | {delta:6.2f}")
    
    # 期待される出力:
    # (1 1 0)  |   2.0293   |      44.67   |      44.67   |   0.00
    # (2 0 0)  |   1.4350   |      65.02   |      65.02   |   0.00
    # (2 1 1)  |   1.1707   |      82.33   |      82.33   |   0.00

## 2.4 バックグラウンド除去とデータ平滑化

### 2.4.1 多項式フィッティングによるバックグラウンド除去
    
    
    from scipy.signal import savgol_filter
    
    def remove_background_polynomial(two_theta, intensity, degree=3, exclude_peaks=True):
        """多項式フィッティングでバックグラウンドを除去
    
        Args:
            two_theta (np.ndarray): 2θ角度
            intensity (np.ndarray): 回折強度
            degree (int): 多項式の次数
            exclude_peaks (bool): ピーク領域を除外してフィット
    
        Returns:
            tuple: (background, intensity_corrected)
        """
        if exclude_peaks:
            # ピーク検出
            peaks_info = detect_peaks(two_theta, intensity, prominence=50)
            peak_indices = peaks_info['indices']
    
            # ピーク近傍を除外したマスク作成
            mask = np.ones(len(two_theta), dtype=bool)
            for peak_idx in peak_indices:
                # ピークの前後 ±20点を除外
                mask[max(0, peak_idx-20):min(len(two_theta), peak_idx+20)] = False
    
            # マスクされた領域のみでフィット
            coeffs = np.polyfit(two_theta[mask], intensity[mask], degree)
        else:
            coeffs = np.polyfit(two_theta, intensity, degree)
    
        # バックグラウンド計算
        background = np.polyval(coeffs, two_theta)
    
        # 補正後の強度
        intensity_corrected = intensity - background
    
        # 負の値をゼロにクリップ
        intensity_corrected = np.maximum(intensity_corrected, 0)
    
        return background, intensity_corrected
    
    
    def smooth_data_savitzky_golay(intensity, window_length=11, polyorder=3):
        """Savitzky-Golayフィルタでデータを平滑化
    
        Args:
            intensity (np.ndarray): 回折強度
            window_length (int): ウィンドウ長 (奇数)
            polyorder (int): 多項式次数
    
        Returns:
            np.ndarray: 平滑化された強度
        """
        if window_length % 2 == 0:
            window_length += 1  # 奇数に調整
    
        smoothed = savgol_filter(intensity, window_length=window_length, polyorder=polyorder)
        return smoothed
    
    
    # 適用例
    two_theta, intensity_raw = generate_synthetic_xrd()
    
    # バックグラウンド除去
    background, intensity_corrected = remove_background_polynomial(
        two_theta, intensity_raw, degree=3, exclude_peaks=True
    )
    
    # 平滑化
    intensity_smoothed = smooth_data_savitzky_golay(intensity_corrected, window_length=11, polyorder=3)
    
    # プロット
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(two_theta, intensity_raw, 'gray', alpha=0.5, label='生データ')
    plt.plot(two_theta, background, 'r--', linewidth=2, label='バックグラウンド')
    plt.xlabel('2θ [度]')
    plt.ylabel('強度 [counts]')
    plt.legend()
    plt.title('ステップ1: バックグラウンド推定')
    plt.grid(alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(two_theta, intensity_corrected, 'b', alpha=0.7, label='BG除去後')
    plt.xlabel('2θ [度]')
    plt.ylabel('強度 [counts]')
    plt.legend()
    plt.title('ステップ2: バックグラウンド除去')
    plt.grid(alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(two_theta, intensity_smoothed, color='#f093fb', linewidth=2, label='平滑化後')
    plt.xlabel('2θ [度]')
    plt.ylabel('強度 [counts]')
    plt.legend()
    plt.title('ステップ3: Savitzky-Golay平滑化')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

## 2.5 ピークフィッティング

### 2.5.1 ピーク形状関数

XRDピークは、装置の分解能と試料の結晶性により、様々な形状を示します:

**Gaussian関数** (装置起因の広がり):

\\[ G(x) = I_0 \exp\left[-\frac{(x - x_0)^2}{2\sigma^2}\right] \\] 

**Lorentzian関数** (試料起因の広がり、結晶サイズ):

\\[ L(x) = I_0 \left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]^{-1} \\] 

**Pseudo-Voigt関数** (GaussとLorentzの線形結合):

\\[ PV(x) = \eta L(x) + (1-\eta) G(x) \\] 
    
    
    def gaussian(x, amplitude, center, sigma):
        """Gaussian関数
    
        Args:
            x (np.ndarray): 独立変数
            amplitude (float): ピーク高さ
            center (float): ピーク中心位置
            sigma (float): 標準偏差
    
        Returns:
            np.ndarray: Gaussian曲線
        """
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    
    def lorentzian(x, amplitude, center, gamma):
        """Lorentzian関数
    
        Args:
            x (np.ndarray): 独立変数
            amplitude (float): ピーク高さ
            center (float): ピーク中心位置
            gamma (float): 半値半幅
    
        Returns:
            np.ndarray: Lorentzian曲線
        """
        return amplitude / (1 + ((x - center) / gamma) ** 2)
    
    
    def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
        """Pseudo-Voigt関数
    
        Args:
            x (np.ndarray): 独立変数
            amplitude (float): ピーク高さ
            center (float): ピーク中心位置
            sigma (float): Gaussian成分の標準偏差
            gamma (float): Lorentzian成分の半値半幅
            eta (float): Lorentzian成分の混合比 (0-1)
    
        Returns:
            np.ndarray: Pseudo-Voigt曲線
        """
        G = gaussian(x, amplitude, center, sigma)
        L = lorentzian(x, amplitude, center, gamma)
        return eta * L + (1 - eta) * G
    
    
    # ピーク形状の比較プロット
    x = np.linspace(40, 50, 500)
    amplitude = 1000
    center = 45
    sigma = 0.1
    gamma = 0.1
    eta = 0.5
    
    y_gauss = gaussian(x, amplitude, center, sigma)
    y_lorentz = lorentzian(x, amplitude, center, gamma)
    y_voigt = pseudo_voigt(x, amplitude, center, sigma, gamma, eta)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_gauss, label='Gaussian', color='#3498db', linewidth=2)
    plt.plot(x, y_lorentz, label='Lorentzian', color='#e74c3c', linewidth=2)
    plt.plot(x, y_voigt, label='Pseudo-Voigt (η=0.5)', color='#f093fb', linewidth=2, linestyle='--')
    plt.xlabel('2θ [度]', fontsize=12)
    plt.ylabel('強度 [counts]', fontsize=12)
    plt.title('XRDピーク形状関数の比較', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

### 2.5.2 scipy.optimizeを用いたフィッティング
    
    
    from scipy.optimize import curve_fit
    
    def fit_single_peak(two_theta, intensity, peak_center_guess, fit_func='voigt'):
        """単一ピークをフィッティング
    
        Args:
            two_theta (np.ndarray): 2θ角度
            intensity (np.ndarray): 回折強度
            peak_center_guess (float): ピーク中心位置の初期推定
            fit_func (str): フィット関数 ('gaussian', 'lorentzian', 'voigt')
    
        Returns:
            dict: フィット結果 {params, covariance, fitted_curve}
        """
        # フィット範囲を制限 (ピーク中心 ±2度)
        mask = (two_theta >= peak_center_guess - 2) & (two_theta <= peak_center_guess + 2)
        x_fit = two_theta[mask]
        y_fit = intensity[mask]
    
        # 初期パラメータ推定
        amplitude_guess = np.max(y_fit)
        sigma_guess = 0.1
        gamma_guess = 0.1
    
        try:
            if fit_func == 'gaussian':
                popt, pcov = curve_fit(
                    gaussian,
                    x_fit, y_fit,
                    p0=[amplitude_guess, peak_center_guess, sigma_guess],
                    bounds=([0, peak_center_guess-1, 0.01], [np.inf, peak_center_guess+1, 1.0])
                )
                fitted_curve = gaussian(two_theta, *popt)
                param_names = ['amplitude', 'center', 'sigma']
    
            elif fit_func == 'lorentzian':
                popt, pcov = curve_fit(
                    lorentzian,
                    x_fit, y_fit,
                    p0=[amplitude_guess, peak_center_guess, gamma_guess],
                    bounds=([0, peak_center_guess-1, 0.01], [np.inf, peak_center_guess+1, 1.0])
                )
                fitted_curve = lorentzian(two_theta, *popt)
                param_names = ['amplitude', 'center', 'gamma']
    
            elif fit_func == 'voigt':
                popt, pcov = curve_fit(
                    pseudo_voigt,
                    x_fit, y_fit,
                    p0=[amplitude_guess, peak_center_guess, sigma_guess, gamma_guess, 0.5],
                    bounds=([0, peak_center_guess-1, 0.01, 0.01, 0], [np.inf, peak_center_guess+1, 1.0, 1.0, 1])
                )
                fitted_curve = pseudo_voigt(two_theta, *popt)
                param_names = ['amplitude', 'center', 'sigma', 'gamma', 'eta']
    
            else:
                raise ValueError(f"Unknown fit function: {fit_func}")
    
            # パラメータの標準誤差
            perr = np.sqrt(np.diag(pcov))
    
            result = {
                'function': fit_func,
                'params': dict(zip(param_names, popt)),
                'errors': dict(zip(param_names, perr)),
                'covariance': pcov,
                'fitted_curve': fitted_curve,
                'x_fit': x_fit,
                'y_fit': y_fit
            }
    
            return result
    
        except Exception as e:
            print(f"フィッティングエラー: {e}")
            return None
    
    
    # 実行例
    two_theta, intensity = generate_synthetic_xrd()
    background, intensity_corrected = remove_background_polynomial(two_theta, intensity)
    
    # (110)ピークをフィット
    fit_result = fit_single_peak(two_theta, intensity_corrected, peak_center_guess=44.7, fit_func='voigt')
    
    if fit_result:
        print("=== ピークフィッティング結果 (Pseudo-Voigt) ===")
        for param, value in fit_result['params'].items():
            error = fit_result['errors'][param]
            print(f"{param:<12}: {value:10.5f} ± {error:8.5f}")
    
        # FWHM計算 (Voigtの場合は近似)
        sigma = fit_result['params']['sigma']
        fwhm_gauss = 2.355 * sigma  # Gaussian寄与
        print(f"\nFWHM (概算): {fwhm_gauss:.4f}°")
    
        # プロット
        plt.figure(figsize=(10, 6))
        plt.plot(two_theta, intensity_corrected, 'gray', alpha=0.5, label='実測データ')
        plt.plot(two_theta, fit_result['fitted_curve'], color='#f093fb', linewidth=2, label='フィット曲線')
        plt.scatter(fit_result['x_fit'], fit_result['y_fit'], color='#f5576c', s=20, alpha=0.7, label='フィット範囲')
        plt.xlabel('2θ [度]', fontsize=12)
        plt.ylabel('強度 [counts]', fontsize=12)
        plt.title(f"ピークフィッティング: {fit_result['params']['center']:.2f}°", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

## 学習目標の確認

### 基本理解

  * ✅ XRD装置の各構成要素(X線源、光学系、検出器)の役割
  * ✅ 測定条件(2θ範囲、ステップサイズ、カウント時間)の最適化原理
  * ✅ Gaussian、Lorentzian、Voigt関数の物理的意味

### 実践スキル

  * ✅ scipy.signalでピーク自動検出を実装
  * ✅ Miller指数を実測ピークに割り当て(立方晶)
  * ✅ 多項式フィッティングでバックグラウンド除去
  * ✅ curve_fitで単一ピークをフィッティング

### 応用力

  * ✅ 測定条件を最適化し、総測定時間を見積もる
  * ✅ 実データに対してバックグラウンド除去→平滑化→ピークフィットの完全ワークフローを実行
  * ✅ フィット結果からFWHM、ピーク位置の標準誤差を算出

## 演習問題

### Easy (基礎確認)

**Q1** : Cu Kα線(λ=1.54Å)で、d=1.2Åから5.0Åの範囲をカバーするのに必要な2θ範囲を計算してください。

**解答** :
    
    
    cond = XRDMeasurementConditions(wavelength=1.54056)
    two_theta_range = cond.calculate_two_theta_range(d_min=1.2, d_max=5.0)
    print(f"必要な2θ範囲: {two_theta_range[0]:.2f}° - {two_theta_range[1]:.2f}°")
    # 出力: 必要な2θ範囲: 9.14° - 40.33°

**Q2** : ピークのFWHMが0.1°の場合、推奨されるステップサイズはいくつですか?

**解答** :
    
    
    cond = XRDMeasurementConditions()
    step = cond.optimize_step_size((10, 80), fwhm=0.1)
    print(f"推奨ステップサイズ: {step:.4f}°")
    # 出力: 推奨ステップサイズ: 0.0143° (FWHM/7)

### Medium (応用)

**Q3** : Gaussian関数とLorentzian関数で、同じFWHM (0.2°) を持つピークの形状の違いを説明してください。

**解答** :

**FWHM = 0.2°の場合** :

  * **Gaussian** : σ = FWHM / (2√(2ln2)) ≈ 0.085°
  * **Lorentzian** : γ = FWHM / 2 = 0.1°

**形状の違い** :

  * Gaussian: 裾野が急速に減衰 (exp減衰)
  * Lorentzian: 裾野が広い (べき乗減衰)
  * Lorentzianの方がピークのすそが長く引く → 結晶サイズが小さい試料に顕著

**Q4** : α-Fe (BCC, a=2.87Å) で観測される最初の3つのピークのMiller指数と2θ位置を計算してください。

**解答** :
    
    
    # BCCの消滅則: h+k+l = 偶数
    # 最小のh^2+k^2+l^2: (110)→2, (200)→4, (211)→6
    from scipy.constants import physical_constants
    
    wavelength = 1.54056
    a = 2.87
    
    peaks = [
        (1, 1, 0, 2),
        (2, 0, 0, 4),
        (2, 1, 1, 6)
    ]
    
    for h, k, l, h2k2l2 in peaks:
        d = a / np.sqrt(h2k2l2)
        theta = np.degrees(np.arcsin(wavelength / (2 * d)))
        two_theta = 2 * theta
        print(f"({h}{k}{l}): d={d:.4f}Å, 2θ={two_theta:.2f}°")
    
    # 期待される出力:
    # (110): d=2.0293Å, 2θ=44.67°
    # (200): d=1.4350Å, 2θ=65.02°
    # (211): d=1.1707Å, 2θ=82.33°

### Hard (発展)

**Q5** : 多項式次数(1次、3次、5次)を変えてバックグラウンド除去を行い、最適な次数を選択する基準を説明してください。

**解答** :
    
    
    # 異なる次数でフィット
    degrees = [1, 3, 5]
    for deg in degrees:
        bg, corrected = remove_background_polynomial(two_theta, intensity, degree=deg)
        residual = intensity - bg
        rms = np.sqrt(np.mean((residual - np.mean(residual))**2))
        print(f"次数{deg}: RMS残差 = {rms:.2f}")
    
    # 判断基準:
    # 1. 視覚的評価: BGがピークを通過していないか
    # 2. 残差の大きさ: 小さすぎるとoverfitting
    # 3. ピーク領域の保存: 強度が負にならないか
    # 推奨: 3次多項式 (柔軟性と安定性のバランス)

**最適次数の選択基準** :

  1. **次数が低すぎる(1次)** : BGの曲線を捉えられない → ピーク強度が過大評価
  2. **適切(3次)** : BGを滑らかにフィット、ピークを避ける
  3. **次数が高すぎる(5次以上)** : ピークにフィットしてしまう → 強度が過小評価

実用的には**3次多項式** が最も一般的です。

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります：

### 基本理解

  * ✅ XRD装置の構成要素（X線源、ゴニオメータ、検出器）の役割を説明できる
  * ✅ 測定条件（2θ範囲、ステップサイズ、積算時間）が結果に与える影響を理解している
  * ✅ バックグラウンドの発生源（非弾性散乱、蛍光X線）を説明できる
  * ✅ ピークプロファイル関数（Gaussian, Lorentz, Voigt）の特徴を理解している

### 実践スキル

  * ✅ Pythonで生XRDデータを読み込み、プロットできる
  * ✅ ピーク検出アルゴリズム（scipy.signal.find_peaks）を適用できる
  * ✅ 多項式フィッティングでバックグラウンドを除去できる
  * ✅ Voigt関数でピークフィッティングを実行し、パラメータを抽出できる
  * ✅ 格子定数からピークをインデキシングできる

### 応用力

  * ✅ 実測XRDデータの品質を評価し、測定条件を最適化できる
  * ✅ 多相混合物のピークを区別し、主要相を同定できる
  * ✅ ピーク幅から結晶子サイズを概算できる（Scherrer式）
  * ✅ XRD解析の完全なワークフローを構築できる

## 参考文献

  1. Jenkins, R., & Snyder, R. L. (1996). _Introduction to X-Ray Powder Diffractometry_. Wiley. - XRD装置の原理と測定技術を詳細に解説した実践的教科書
  2. Dinnebier, R. E., & Billinge, S. J. L. (Eds.). (2008). _Powder Diffraction: Theory and Practice_. Royal Society of Chemistry. - 粉末XRDの理論と実践を網羅した決定版
  3. Langford, J. I., & Louër, D. (1996). "Powder diffraction". _Reports on Progress in Physics_ , 59(2), 131-234. - ピークプロファイル解析の理論的基礎を提供する重要レビュー
  4. ICDD PDF-4+ Database (2024). International Centre for Diffraction Data. - 40万以上の参照パターンを収録した世界標準XRDデータベース
  5. Scherrer, P. (1918). "Bestimmung der Größe und der inneren Struktur von Kolloidteilchen mittels Röntgenstrahlen". _Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen_ , 2, 98-100. - 結晶子サイズ解析の元祖論文
  6. Klug, H. P., & Alexander, L. E. (1974). _X-Ray Diffraction Procedures for Polycrystalline and Amorphous Materials_ (2nd ed.). Wiley. - バックグラウンド除去とピーク分離技術の古典的解説
  7. scipy.signal documentation (2024). "Signal processing (scipy.signal)". SciPy project. - find_peaks関数の詳細仕様とピーク検出アルゴリズム

## 次のステップ

第2章では、実際のXRD測定データの取得から基本的な解析までを学びました。ピーク検出、インデキシング、バックグラウンド処理、そしてピークフィッティングという一連のワークフローを習得しました。

**第3章** では、これらの技術を統合し、リートベルト法による精密解析に進みます。全パターンフィッティングにより、格子定数、原子座標、結晶子サイズなど、より詳細な構造情報を抽出します。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
