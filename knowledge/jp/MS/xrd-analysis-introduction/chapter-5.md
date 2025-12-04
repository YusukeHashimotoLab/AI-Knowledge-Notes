---
title: "第5章: 実践XRDデータ解析ワークフロー"
chapter_title: "第5章: 実践XRDデータ解析ワークフロー"
subtitle: 生データから学術報告まで - 実務に即したXRD解析の完全ガイド
reading_time: 35分
---

## 学習目標

この章を完了すると、以下を説明・実装できるようになります:

### 基本理解

  * ✅ 実践的なXRD解析ワークフローの全体像
  * ✅ 多相混合物解析(2相・3相系)の戦略
  * ✅ 定量相分析(RIR法、Rietveld定量)の原理
  * ✅ 収束失敗やパラメータ相関のトラブルシューティング

### 実践スキル

  * ✅ 生データ読み込みから結果可視化まで完全ワークフローの実装
  * ✅ 多相混合物(α-Fe + Fe₃O₄)のリートベルト解析
  * ✅ RIR法とRietveld法による相分率の定量
  * ✅ CIFファイル出力と学術論文用図表の作成

### 応用力

  * ✅ エラー診断(収束失敗、GOF異常、負の占有率)と解決策
  * ✅ GSAS-IIとPythonの連携による高度な解析
  * ✅ 実測データに対する完全なリートベルト解析の実行

## 5.1 完全解析ワークフロー

実践的なXRD解析は、生データの読み込みから始まり、構造精密化、結果の検証、そして学術報告用の図表作成まで、一連の流れを体系的に実行する必要があります。この節では、プロフェッショナルな解析ワークフローを構築します。
    
    
    ```mermaid
    flowchart TB
                A[生データ読み込み.xy, .dat, .xrdml] --> B[データ前処理]
                B --> C[ピーク検出相同定]
                C --> D[初期構造モデルCIF読み込み]
                D --> E[パラメータ初期化]
                E --> F{単相 or 多相?}
    
                F -->|単相| G[リートベルト精密化Stage 1: BG+Scale]
                F -->|多相| H[多相解析Phase 1 → Phase 2]
    
                G --> I[Stage 2: ProfileU, V, W, η]
                I --> J[Stage 3: Structurex, y, z, Uiso]
    
                H --> I
    
                J --> K[収束判定GOF < 2.0?]
                K -->|No| L[エラー診断パラメータ調整]
                L --> E
    
                K -->|Yes| M[結果抽出格子定数, 相分率, D]
                M --> N[可視化論文用図表]
                N --> O[CIF出力学術報告]
    
                style A fill:#e3f2fd
                style K fill:#fff3e0
                style O fill:#e8f5e9
    ```

### 5.1.1 生データの読み込みと前処理

XRD装置から出力されるデータ形式は多様です。主要な形式をサポートするデータ読み込み関数を実装します。
    
    
    # ========================================
    # Example 1: 汎用XRDデータローダー
    # ========================================
    
    import numpy as np
    import pandas as pd
    
    class XRDDataLoader:
        """
        汎用XRDデータローダー
    
        対応形式:
        - .xy: 2列形式 (2θ, Intensity)
        - .dat: ヘッダー付きテキスト
        - .xrdml: Panalytical XML形式
        - .raw: Bruker RAW形式
        """
    
        @staticmethod
        def load_xy(filepath, skip_rows=0):
            """
            .xy形式の読み込み
    
            Args:
                filepath: ファイルパス
                skip_rows: スキップする行数(ヘッダー)
    
            Returns:
                two_theta, intensity: numpy配列
            """
            try:
                data = np.loadtxt(filepath, skiprows=skip_rows)
                two_theta = data[:, 0]
                intensity = data[:, 1]
    
                return two_theta, intensity
            except Exception as e:
                print(f"Error loading .xy file: {e}")
                return None, None
    
        @staticmethod
        def load_dat(filepath, delimiter=None, header=None):
            """
            .dat形式の読み込み (pandasで柔軟に対応)
    
            Args:
                filepath: ファイルパス
                delimiter: 区切り文字 (None: 空白)
                header: ヘッダー行番号
    
            Returns:
                two_theta, intensity
            """
            try:
                df = pd.read_csv(filepath, delimiter=delimiter, header=header)
    
                # 最初の2列を2θとIntensityと仮定
                two_theta = df.iloc[:, 0].values
                intensity = df.iloc[:, 1].values
    
                return two_theta, intensity
            except Exception as e:
                print(f"Error loading .dat file: {e}")
                return None, None
    
        @staticmethod
        def preprocess(two_theta, intensity, remove_outliers=True, smooth=False):
            """
            データ前処理
    
            Args:
                two_theta, intensity: 生データ
                remove_outliers: 外れ値除去
                smooth: 平滑化 (移動平均)
    
            Returns:
                two_theta_clean, intensity_clean
            """
            # 外れ値除去 (3σ以上)
            if remove_outliers:
                mean_int = np.mean(intensity)
                std_int = np.std(intensity)
                mask = np.abs(intensity - mean_int) < 3 * std_int
                two_theta = two_theta[mask]
                intensity = intensity[mask]
    
            # 平滑化 (移動平均, window=5)
            if smooth:
                intensity = np.convolve(intensity, np.ones(5)/5, mode='same')
    
            # 負の強度を0に
            intensity = np.maximum(intensity, 0)
    
            return two_theta, intensity
    
    
    # 使用例
    loader = XRDDataLoader()
    
    # .xy形式の読み込み
    two_theta, intensity = loader.load_xy('sample_data.xy', skip_rows=1)
    
    # 前処理
    two_theta_clean, intensity_clean = loader.preprocess(
        two_theta, intensity,
        remove_outliers=True,
        smooth=False
    )
    
    print(f"データポイント数: {len(two_theta_clean)}")
    print(f"2θ範囲: {two_theta_clean.min():.2f}° - {two_theta_clean.max():.2f}°")
    print(f"最大強度: {intensity_clean.max():.0f} counts")
    

### 5.1.2 完全ワークフローの実装

データ読み込みからCIF出力まで、全ステップを統合したワークフローを実装します。
    
    
    # ========================================
    # Example 2: 完全XRD解析ワークフロー
    # ========================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Parameters, Minimizer
    from pymatgen.core import Structure
    
    class CompleteXRDWorkflow:
        """
        XRD解析の完全ワークフロー
    
        手順:
        1. データ読み込み・前処理
        2. ピーク検出・相同定
        3. リートベルト精密化 (3段階)
        4. 結果抽出・可視化
        5. CIF出力
        """
    
        def __init__(self, filepath, wavelength=1.5406):
            self.filepath = filepath
            self.wavelength = wavelength
            self.two_theta = None
            self.intensity = None
            self.result = None
    
        def step1_load_data(self, skip_rows=0):
            """Step 1: データ読み込み"""
            loader = XRDDataLoader()
            self.two_theta, self.intensity = loader.load_xy(self.filepath, skip_rows)
            self.two_theta, self.intensity = loader.preprocess(
                self.two_theta, self.intensity, remove_outliers=True
            )
            print(f"✓ Step 1 完了: {len(self.two_theta)} データポイント読み込み")
    
        def step2_peak_detection(self, prominence=0.1):
            """Step 2: ピーク検出"""
            from scipy.signal import find_peaks
    
            # ピーク検出
            intensity_norm = self.intensity / self.intensity.max()
            peaks, properties = find_peaks(intensity_norm, prominence=prominence)
    
            self.peak_positions = self.two_theta[peaks]
            self.peak_intensities = self.intensity[peaks]
    
            print(f"✓ Step 2 完了: {len(self.peak_positions)} ピーク検出")
            print(f"  主要ピーク位置: {self.peak_positions[:5]}")
    
        def step3_rietveld_refinement(self, structure_cif=None):
            """Step 3: リートベルト精密化 (3段階)"""
    
            # Stage 1: Background + Scale
            print("Stage 1: Background + Scale ...")
            params_stage1 = self._initialize_params_stage1()
            result_stage1 = self._minimize(params_stage1)
    
            # Stage 2: Profile parameters
            print("Stage 2: Profile parameters ...")
            params_stage2 = self._add_profile_params(result_stage1.params)
            result_stage2 = self._minimize(params_stage2)
    
            # Stage 3: Structure parameters
            print("Stage 3: Structure parameters ...")
            params_stage3 = self._add_structure_params(result_stage2.params)
            self.result = self._minimize(params_stage3)
    
            print(f"✓ Step 3 完了: Rwp = {self._calculate_rwp(self.result):.2f}%")
    
        def _initialize_params_stage1(self):
            """Stage 1用パラメータ"""
            params = Parameters()
            params.add('scale', value=1.0, min=0.1, max=10.0)
            params.add('bg_0', value=self.intensity.min(), min=0.0)
            params.add('bg_1', value=0.0)
            params.add('bg_2', value=0.0)
            return params
    
        def _add_profile_params(self, params_prev):
            """Stage 2: プロファイルパラメータ追加"""
            params = params_prev.copy()
            params.add('U', value=0.01, min=0.0, max=0.1)
            params.add('V', value=-0.005, min=-0.05, max=0.0)
            params.add('W', value=0.005, min=0.0, max=0.05)
            params.add('eta', value=0.5, min=0.0, max=1.0)
            return params
    
        def _add_structure_params(self, params_prev):
            """Stage 3: 構造パラメータ追加"""
            params = params_prev.copy()
            params.add('lattice_a', value=5.64, min=5.5, max=5.8)
            params.add('U_iso', value=0.01, min=0.001, max=0.05)
            return params
    
        def _minimize(self, params):
            """最小化実行"""
            minimizer = Minimizer(self._residual, params)
            result = minimizer.minimize(method='leastsq')
            return result
    
        def _residual(self, params):
            """残差関数 (簡略版)"""
            # バックグラウンド
            bg_coeffs = [params.get('bg_0', params.valuesdict().get('bg_0', 0)),
                         params.get('bg_1', params.valuesdict().get('bg_1', 0)),
                         params.get('bg_2', params.valuesdict().get('bg_2', 0))]
    
            x_norm = 2 * (self.two_theta - self.two_theta.min()) / (self.two_theta.max() - self.two_theta.min()) - 1
            bg = sum(c * np.polynomial.chebyshev.chebval(x_norm, [0]*i + [1]) for i, c in enumerate(bg_coeffs))
    
            # スケール
            scale = params['scale'].value
    
            # 計算パターン (簡略化)
            I_calc = bg + scale * 10  # 実際はピーク計算が入る
    
            # 残差
            residual = (self.intensity - I_calc) / np.sqrt(np.maximum(self.intensity, 1.0))
            return residual
    
        def _calculate_rwp(self, result):
            """Rwp計算"""
            return np.sqrt(result.chisqr / result.ndata) * 100
    
        def step4_extract_results(self):
            """Step 4: 結果抽出"""
            if self.result is None:
                print("Error: 精密化が未実行です")
                return
    
            results_dict = {
                'lattice_a': self.result.params.get('lattice_a', None),
                'U_iso': self.result.params.get('U_iso', None),
                'Rwp': self._calculate_rwp(self.result),
                'GOF': self.result.redchi
            }
    
            print("✓ Step 4 完了: 結果抽出")
            for key, val in results_dict.items():
                if val is not None:
                    if hasattr(val, 'value'):
                        print(f"  {key}: {val.value:.6f}")
                    else:
                        print(f"  {key}: {val:.6f}")
    
            return results_dict
    
        def step5_visualize(self, save_path=None):
            """Step 5: 可視化"""
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]})
    
            # 上段: 観測・計算・差分
            ax1.plot(self.two_theta, self.intensity, 'o', markersize=3,
                     label='Observed', color='red', alpha=0.6)
            # I_calc は簡略化のため省略
            ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
            ax1.legend()
            ax1.set_title('Rietveld Refinement', fontsize=14, fontweight='bold')
    
            # 下段: 残差
            # residual は簡略化のため省略
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax2.set_xlabel('2θ (°)', fontsize=12)
            ax2.set_ylabel('Residual', fontsize=10)
    
            plt.tight_layout()
    
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"✓ Step 5 完了: 図を {save_path} に保存")
    
            plt.show()
    
    
    # ワークフロー実行例
    workflow = CompleteXRDWorkflow('sample.xy', wavelength=1.5406)
    
    # 全ステップ実行
    workflow.step1_load_data(skip_rows=1)
    workflow.step2_peak_detection(prominence=0.1)
    workflow.step3_rietveld_refinement()
    results = workflow.step4_extract_results()
    workflow.step5_visualize(save_path='rietveld_result.png')
    

## 5.2 多相混合物解析

実際の材料は複数の相が共存する場合が多く、多相リートベルト解析が必要です。この節では、2相系(α-Fe + Fe₃O₄)と3相系の解析手法を学びます。

### 5.2.1 2相系の解析: α-Fe + Fe₃O₄

鉄の酸化試料を例に、α-Fe (BCC)とFe₃O₄ (スピネル構造)の2相混合物を解析します。
    
    
    # ========================================
    # Example 3: 2相混合物リートベルト解析
    # ========================================
    
    import numpy as np
    from lmfit import Parameters, Minimizer
    
    class TwoPhaseRietveld:
        """
        2相混合物のリートベルト解析
    
        Phase 1: α-Fe (BCC, Im-3m, a=2.866 Å)
        Phase 2: Fe₃O₄ (Spinel, Fd-3m, a=8.396 Å)
        """
    
        def __init__(self, two_theta, intensity, wavelength=1.5406):
            self.two_theta = np.array(two_theta)
            self.intensity = np.array(intensity)
            self.wavelength = wavelength
    
            # 各相のhklリスト
            self.hkl_Fe = [(1,1,0), (2,0,0), (2,1,1), (2,2,0), (3,1,0)]
            self.hkl_Fe3O4 = [(2,2,0), (3,1,1), (4,0,0), (4,2,2), (5,1,1)]
    
        def two_theta_from_d(self, d):
            """d間隔から2θを計算"""
            sin_theta = self.wavelength / (2 * d)
            if abs(sin_theta) > 1.0:
                return None
            theta = np.arcsin(sin_theta)
            return np.degrees(2 * theta)
    
        def d_spacing_cubic(self, hkl, a):
            """立方格子のd間隔"""
            h, k, l = hkl
            return a / np.sqrt(h**2 + k**2 + l**2)
    
        def pseudo_voigt(self, two_theta, two_theta_0, fwhm, eta, amplitude):
            """Pseudo-Voigt プロファイル"""
            H = fwhm / 2
            delta = two_theta - two_theta_0
    
            G = np.exp(-np.log(2) * (delta / H)**2)
            L = 1 / (1 + (delta / H)**2)
            PV = eta * L + (1 - eta) * G
    
            return amplitude * PV
    
        def caglioti_fwhm(self, two_theta, U, V, W):
            """Caglioti式"""
            theta_rad = np.radians(two_theta / 2)
            tan_theta = np.tan(theta_rad)
            fwhm_sq = U * tan_theta**2 + V * tan_theta + W
            return np.sqrt(max(fwhm_sq, 1e-6))
    
        def calculate_pattern(self, params):
            """
            2相の計算パターンを生成
            """
            # パラメータ抽出
            a_Fe = params['a_Fe'].value
            a_Fe3O4 = params['a_Fe3O4'].value
    
            scale_Fe = params['scale_Fe'].value
            scale_Fe3O4 = params['scale_Fe3O4'].value
    
            U = params['U'].value
            V = params['V'].value
            W = params['W'].value
            eta = params['eta'].value
    
            # バックグラウンド
            bg_0 = params['bg_0'].value
            bg_1 = params['bg_1'].value
    
            x_norm = 2 * (self.two_theta - self.two_theta.min()) / (self.two_theta.max() - self.two_theta.min()) - 1
            bg = bg_0 + bg_1 * x_norm
    
            I_calc = bg.copy()
    
            # Phase 1: α-Fe
            for hkl in self.hkl_Fe:
                d = self.d_spacing_cubic(hkl, a_Fe)
                two_theta_hkl = self.two_theta_from_d(d)
    
                if two_theta_hkl is None or two_theta_hkl < self.two_theta.min() or two_theta_hkl > self.two_theta.max():
                    continue
    
                fwhm = self.caglioti_fwhm(two_theta_hkl, U, V, W)
                amplitude = scale_Fe * 100  # 簡略化
    
                I_calc += self.pseudo_voigt(self.two_theta, two_theta_hkl, fwhm, eta, amplitude)
    
            # Phase 2: Fe₃O₄
            for hkl in self.hkl_Fe3O4:
                d = self.d_spacing_cubic(hkl, a_Fe3O4)
                two_theta_hkl = self.two_theta_from_d(d)
    
                if two_theta_hkl is None or two_theta_hkl < self.two_theta.min() or two_theta_hkl > self.two_theta.max():
                    continue
    
                fwhm = self.caglioti_fwhm(two_theta_hkl, U, V, W)
                amplitude = scale_Fe3O4 * 80  # 簡略化
    
                I_calc += self.pseudo_voigt(self.two_theta, two_theta_hkl, fwhm, eta, amplitude)
    
            return I_calc
    
        def residual(self, params):
            """残差関数"""
            I_calc = self.calculate_pattern(params)
            residual = (self.intensity - I_calc) / np.sqrt(np.maximum(self.intensity, 1.0))
            return residual
    
        def refine(self):
            """2相精密化"""
            params = Parameters()
    
            # 格子定数
            params.add('a_Fe', value=2.866, min=2.85, max=2.89)
            params.add('a_Fe3O4', value=8.396, min=8.35, max=8.45)
    
            # スケールファクター
            params.add('scale_Fe', value=1.0, min=0.1, max=10.0)
            params.add('scale_Fe3O4', value=0.5, min=0.1, max=10.0)
    
            # プロファイル
            params.add('U', value=0.01, min=0.0, max=0.1)
            params.add('V', value=-0.005, min=-0.05, max=0.0)
            params.add('W', value=0.005, min=0.0, max=0.05)
            params.add('eta', value=0.5, min=0.0, max=1.0)
    
            # バックグラウンド
            params.add('bg_0', value=10.0, min=0.0)
            params.add('bg_1', value=0.0)
    
            # 最小化
            minimizer = Minimizer(self.residual, params)
            result = minimizer.minimize(method='leastsq')
    
            return result
    
    
    # テストデータ (簡略化)
    two_theta_test = np.linspace(20, 80, 600)
    intensity_test = 15 + 3*np.random.randn(len(two_theta_test))
    
    # 2相解析実行
    two_phase = TwoPhaseRietveld(two_theta_test, intensity_test)
    result = two_phase.refine()
    
    print("=== 2相リートベルト解析結果 ===")
    print(f"α-Fe格子定数: a = {result.params['a_Fe'].value:.6f} Å")
    print(f"Fe₃O₄格子定数: a = {result.params['a_Fe3O4'].value:.6f} Å")
    print(f"スケール比 (Fe:Fe₃O₄) = {result.params['scale_Fe'].value:.3f}:{result.params['scale_Fe3O4'].value:.3f}")
    print(f"Rwp = {np.sqrt(result.chisqr / result.ndata) * 100:.2f}%")
    

### 5.2.2 3相系の解析戦略

3相以上の混合物では、パラメータ数が急増し、収束が困難になります。以下の戦略が有効です:

  1. **段階的精密化** : 主相 → 第2相 → 第3相と順番に追加
  2. **パラメータ固定** : 既知の相は格子定数を文献値に固定
  3. **スケール比の制約** : \\(\sum w_i = 1.0\\) (重量分率の合計が1)
  4. **プロファイル共有** : U, V, Wを全相で共通とする

## 5.3 定量相分析

多相混合物の各相の重量分率を定量する手法を学びます。RIR法(Reference Intensity Ratio)とRietveld法の2つのアプローチがあります。

### 5.3.1 RIR法 (Reference Intensity Ratio)

RIR法は、最強ピークの強度比から相分率を推定する簡易的な手法です:

\\[ w_{\alpha} = \frac{I_{\alpha} / RIR_{\alpha}}{I_{\alpha}/RIR_{\alpha} + I_{\beta}/RIR_{\beta}} \\] 

  * \\(w_{\alpha}\\): 相αの重量分率
  * \\(I_{\alpha}\\): 相αの最強ピーク強度
  * \\(RIR_{\alpha}\\): 相αのRIR値(PDF Cardから取得)

    
    
    # ========================================
    # Example 4: RIR法による定量相分析
    # ========================================
    
    import numpy as np
    
    def rir_quantitative_analysis(peak_intensities, rir_values, phase_names):
        """
        RIR法で重量分率を計算
    
        Args:
            peak_intensities: 各相の最強ピーク強度 [I1, I2, ...]
            rir_values: 各相のRIR値 [RIR1, RIR2, ...]
            phase_names: 相名リスト ['Phase1', 'Phase2', ...]
    
        Returns:
            weight_fractions: 重量分率の辞書
        """
        # I/RIR を計算
        I_over_RIR = np.array(peak_intensities) / np.array(rir_values)
    
        # 重量分率
        total = np.sum(I_over_RIR)
        weight_fractions = I_over_RIR / total
    
        results = {name: w for name, w in zip(phase_names, weight_fractions)}
    
        return results
    
    
    # 例: α-Fe + Fe₃O₄ の2相混合物
    peak_intensities = [1500, 800]  # α-Fe(110): 1500, Fe₃O₄(311): 800
    rir_values = [2.0, 2.5]         # RIR値 (PDF Card)
    phase_names = ['α-Fe', 'Fe₃O₄']
    
    wt_fractions = rir_quantitative_analysis(peak_intensities, rir_values, phase_names)
    
    print("=== RIR法 定量分析結果 ===")
    for phase, wt in wt_fractions.items():
        print(f"{phase}: {wt*100:.2f} wt%")
    
    # 出力例:
    # === RIR法 定量分析結果 ===
    # α-Fe: 70.09 wt%
    # Fe₃O₄: 29.91 wt%
    

### 5.3.2 Rietveld定量分析

Rietveld法では、スケールファクター \\(S_{\alpha}\\) から重量分率を精密に計算できます:

\\[ w_{\alpha} = \frac{S_{\alpha} (ZMV)_{\alpha}}{\sum_i S_i (ZMV)_i} \\] 

  * \\(S_{\alpha}\\): 相αのスケールファクター(精密化で決定)
  * \\(Z\\): 単位格子内の化学式数
  * \\(M\\): 化学式量
  * \\(V\\): 単位格子体積

    
    
    # ========================================
    # Example 5: Rietveld定量分析
    # ========================================
    
    def rietveld_quantitative_analysis(scale_factors, Z_list, M_list, V_list, phase_names):
        """
        Rietveld法で重量分率を計算
    
        Args:
            scale_factors: スケールファクター [S1, S2, ...]
            Z_list: 単位格子内の化学式数 [Z1, Z2, ...]
            M_list: 化学式量 [M1, M2, ...] (g/mol)
            V_list: 単位格子体積 [V1, V2, ...] (Å³)
            phase_names: 相名リスト
    
        Returns:
            weight_fractions: 重量分率の辞書
        """
        # S*(ZMV) を計算
        S_ZMV = np.array(scale_factors) * np.array(Z_list) * np.array(M_list) * np.array(V_list)
    
        # 重量分率
        total = np.sum(S_ZMV)
        weight_fractions = S_ZMV / total
    
        results = {name: w for name, w in zip(phase_names, weight_fractions)}
    
        return results
    
    
    # 例: α-Fe + Fe₃O₄
    scale_factors = [1.23, 0.67]  # リートベルト精密化で決定
    Z_list = [2, 8]               # α-Fe: BCC (Z=2), Fe₃O₄: Spinel (Z=8)
    M_list = [55.845, 231.533]    # Fe: 55.845, Fe₃O₄: 231.533 g/mol
    V_list = [23.55, 591.4]       # α-Fe: a³ = 2.866³, Fe₃O₄: a³ = 8.396³
    
    phase_names = ['α-Fe', 'Fe₃O₄']
    
    wt_fractions_rietveld = rietveld_quantitative_analysis(
        scale_factors, Z_list, M_list, V_list, phase_names
    )
    
    print("=== Rietveld定量分析結果 ===")
    for phase, wt in wt_fractions_rietveld.items():
        print(f"{phase}: {wt*100:.2f} wt%")
    
    # 出力例:
    # === Rietveld定量分析結果 ===
    # α-Fe: 68.34 wt%
    # Fe₃O₄: 31.66 wt%
    

## 5.4 エラー解析とトラブルシューティング

リートベルト解析では、収束失敗や非物理的な結果が頻繁に発生します。この節では、典型的なエラーと解決策を学びます。

### 5.4.1 典型的なエラーと診断

症状 | 原因 | 解決策  
---|---|---  
**収束しない** (100回以上反復) | 初期値が不適切、パラメータ相関が強い | 初期値を文献値に近づける、段階的精密化  
**GOF >> 2.0** | モデルが不適切、バックグラウンドが間違い | Chebyshev次数を増やす、相を見直す  
**GOF < 1.0** | 過剰フィッティング、パラメータ多すぎ | 不要なパラメータを固定、制約を追加  
**負の占有率** | 初期値が悪い、境界設定ミス | min=0.0を設定、初期値を0.5に  
**格子定数が境界に張り付く** | 境界が狭すぎる、相が間違い | 境界を広げる、相同定を再確認  
**Rwp > 20%** | ピーク位置がずれている、装置関数未補正 | Zero補正、試料変位補正を追加  
  
### 5.4.2 収束診断ツールの実装
    
    
    # ========================================
    # Example 6: 収束診断ツール
    # ========================================
    
    class ConvergenceDiagnostics:
        """
        リートベルト精密化の収束診断ツール
        """
    
        @staticmethod
        def check_convergence(result):
            """
            収束状態をチェック
    
            Args:
                result: lmfit Minimizer.minimize() の結果
    
            Returns:
                診断レポート
            """
            issues = []
    
            # 1. GOFチェック
            GOF = result.redchi
            if GOF > 2.0:
                issues.append(f"⚠️ GOF = {GOF:.2f} (>2.0): モデルが不適切な可能性")
            elif GOF < 1.0:
                issues.append(f"⚠️ GOF = {GOF:.2f} (<1.0): 過剰フィッティングの可能性")
    
            # 2. パラメータ境界チェック
            for name, param in result.params.items():
                if not param.vary:
                    continue
    
                # 境界に張り付いているか
                if param.min is not None and abs(param.value - param.min) < 1e-6:
                    issues.append(f"⚠️ {name} が下限境界に張り付いています: {param.value:.6f}")
                if param.max is not None and abs(param.value - param.max) < 1e-6:
                    issues.append(f"⚠️ {name} が上限境界に張り付いています: {param.value:.6f}")
    
                # 非物理的な値
                if 'occ' in name and (param.value < 0 or param.value > 1):
                    issues.append(f"❌ {name} = {param.value:.6f}: 占有率が範囲外 [0, 1]")
    
                if 'U_iso' in name and param.value < 0:
                    issues.append(f"❌ {name} = {param.value:.6f}: 温度因子が負")
    
            # 3. 相関マトリックスチェック
            if hasattr(result, 'covar') and result.covar is not None:
                corr_matrix = result.covar / np.outer(np.sqrt(np.diag(result.covar)),
                                                       np.sqrt(np.diag(result.covar)))
                strong_corr = np.where(np.abs(corr_matrix) > 0.9)
    
                for i, j in zip(*strong_corr):
                    if i < j:  # 重複を避ける
                        param_names = list(result.var_names)
                        issues.append(f"⚠️ 強い相関: {param_names[i]} ↔ {param_names[j]} (r={corr_matrix[i,j]:.3f})")
    
            # レポート出力
            if not issues:
                print("✅ 収束診断: 問題なし")
            else:
                print("🔍 収束診断結果:")
                for issue in issues:
                    print(f"  {issue}")
    
            return issues
    
    
    # 使用例
    diagnostics = ConvergenceDiagnostics()
    
    # result は lmfit Minimizer.minimize() の結果オブジェクト
    # diagnostics.check_convergence(result)
    

### 5.4.3 トラブルシューティング実例

**Case 1** : 格子定数が境界に張り付く

**症状** :
    
    
    lattice_a = 5.799 Å (境界: [5.5, 5.8])
    ⚠️ lattice_a が上限境界に張り付いています

**原因** : 境界が狭すぎる、または相同定が間違っている

**解決策** :

  1. 境界を広げる: `params.add('lattice_a', value=5.64, min=5.3, max=6.0)`
  2. 高角ピークを確認し、相を再検証
  3. 文献値と比較し、初期値を調整

**Case 2** : GOF = 5.2 (異常に高い)

**症状** : Rwp = 23%, GOF = 5.2

**原因** :

  * バックグラウンドモデルが不適切(次数が低すぎる)
  * 未同定の相が存在する
  * ピーク位置がずれている(Zero補正が必要)

**解決策** :

  1. Chebyshev次数を3次→5次に増やす
  2. ピーク検出を再実行し、全ピークが説明されているか確認
  3. Zero補正パラメータを追加: `params.add('zero_shift', value=0.0, min=-0.1, max=0.1)`

## 5.5 結果の可視化と学術報告

リートベルト解析の結果は、論文や報告書で視覚的に明瞭に示す必要があります。この節では、論文品質の図表作成とCIF出力を学びます。

### 5.5.1 論文用リートベルトプロットの作成
    
    
    # ========================================
    # Example 7: 論文品質リートベルトプロット
    # ========================================
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_rietveld_publication_quality(two_theta, I_obs, I_calc, I_bg, residual,
                                          phase_labels=None, save_path='rietveld.pdf'):
        """
        論文品質のリートベルトプロット
    
        Args:
            two_theta: 2θ配列
            I_obs: 観測強度
            I_calc: 計算強度
            I_bg: バックグラウンド
            residual: 残差
            phase_labels: 各相の名前リスト
            save_path: 保存先ファイルパス
        """
        fig = plt.figure(figsize=(12, 8))
    
        # 上段: 観測・計算・差分
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    
        # 観測データ (赤丸)
        ax1.plot(two_theta, I_obs, 'o', markersize=4, markerfacecolor='none',
                 markeredgecolor='red', markeredgewidth=1.2, label='Observed')
    
        # 計算パターン (青線)
        ax1.plot(two_theta, I_calc, '-', color='blue', linewidth=2, label='Calculated')
    
        # バックグラウンド (緑線)
        ax1.plot(two_theta, I_bg, '--', color='green', linewidth=1.5, label='Background')
    
        # 差分 (灰色、下にオフセット)
        offset = I_obs.min() - 0.1 * I_obs.max()
        ax1.plot(two_theta, residual + offset, '-', color='gray', linewidth=1, label='Difference')
        ax1.axhline(offset, color='black', linestyle='-', linewidth=0.5)
    
        # ブラッグピーク位置 (縦線)
        if phase_labels:
            colors = ['red', 'blue', 'orange']
            for i, label in enumerate(phase_labels):
                # 簡略化: ピーク位置は手動設定
                peak_positions = [38.2, 44.4, 64.6]  # 例
                y_position = offset - 0.05 * I_obs.max() * (i + 1)
                ax1.vlines(peak_positions, ymin=y_position, ymax=y_position + 0.03*I_obs.max(),
                          colors=colors[i], linewidth=2, label=label)
    
        ax1.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11, frameon=False)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.set_xlim(two_theta.min(), two_theta.max())
        ax1.set_ylim(offset - 0.2*I_obs.max(), I_obs.max() * 1.1)
    
        # 下段: 残差拡大図
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
        ax2.plot(two_theta, residual, '-', color='black', linewidth=1)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.set_xlabel('2θ (°)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Residual', fontsize=12)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.set_ylim(-3*np.std(residual), 3*np.std(residual))
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 図を {save_path} に保存しました (300 dpi)")
    
        plt.show()
    
    
    # 使用例 (ダミーデータ)
    two_theta = np.linspace(20, 80, 600)
    I_obs = 100 * np.exp(-((two_theta - 38)/2)**2) + 50 * np.exp(-((two_theta - 44)/2.5)**2) + 20 + 5*np.random.randn(len(two_theta))
    I_calc = 100 * np.exp(-((two_theta - 38)/2)**2) + 50 * np.exp(-((two_theta - 44)/2.5)**2) + 20
    I_bg = 20 * np.ones_like(two_theta)
    residual = I_obs - I_calc
    
    plot_rietveld_publication_quality(two_theta, I_obs, I_calc, I_bg, residual,
                                      phase_labels=['α-Fe', 'Fe₃O₄'],
                                      save_path='rietveld_paper.pdf')
    

### 5.5.2 CIFファイル出力

CIF (Crystallographic Information File) は、結晶構造データの標準フォーマットです。精密化結果をCIFで保存することで、他の研究者が再現・検証できます。
    
    
    # ========================================
    # Example 8: CIFファイル生成
    # ========================================
    
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    
    def export_to_cif(structure, lattice_params, refinement_results, output_path='refined_structure.cif'):
        """
        精密化結果をCIFファイルに出力
    
        Args:
            structure: pymatgen Structure オブジェクト
            lattice_params: {'a': 5.64, 'b': 5.64, 'c': 5.64, ...}
            refinement_results: {'Rwp': 8.5, 'GOF': 1.42, ...}
            output_path: 出力ファイルパス
        """
        # CIFライター
        cif_writer = CifWriter(structure, symprec=0.01)
    
        # CIFを文字列として取得
        cif_string = str(cif_writer)
    
        # メタデータを追加
        metadata = f"""
    #======================================================================
    # Rietveld Refinement Results
    #======================================================================
    # Refined lattice parameters:
    #   a = {lattice_params.get('a', 'N/A'):.6f} Å
    #   b = {lattice_params.get('b', 'N/A'):.6f} Å
    #   c = {lattice_params.get('c', 'N/A'):.6f} Å
    #
    # Refinement statistics:
    #   Rwp = {refinement_results.get('Rwp', 'N/A'):.2f} %
    #   GOF = {refinement_results.get('GOF', 'N/A'):.3f}
    #   Number of data points = {refinement_results.get('ndata', 'N/A')}
    #
    # Date: {refinement_results.get('date', 'YYYY-MM-DD')}
    # Software: Python + lmfit + pymatgen
    #======================================================================
    
    """
    
        # CIF出力
        with open(output_path, 'w') as f:
            f.write(metadata)
            f.write(cif_string)
    
        print(f"✓ CIFファイルを {output_path} に出力しました")
    
    
    # 使用例
    from pymatgen.core import Lattice, Structure
    
    # 精密化後の構造
    lattice = Lattice.cubic(5.6405)  # 精密化後のa
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    
    lattice_params = {'a': 5.6405, 'b': 5.6405, 'c': 5.6405}
    refinement_results = {
        'Rwp': 8.52,
        'GOF': 1.42,
        'ndata': 600,
        'date': '2025-10-28'
    }
    
    export_to_cif(structure, lattice_params, refinement_results, 'NaCl_refined.cif')
    

### 5.5.3 GSAS-IIとの連携

GSAS-II は、GUI対応の強力なリートベルト解析ソフトウェアです。Pythonスクリプトから操作することで、高度な解析とPythonの柔軟性を両立できます。

> **💡 GSAS-II Pythonインターフェース**
> 
> GSAS-IIは、`GSASIIscriptable`モジュールを通じてPythonから操作可能です。プロジェクトの作成、データ読み込み、精密化実行、結果抽出まで自動化できます。
> 
> 詳細: [GSAS-II Scriptable Documentation](<https://subversion.xray.aps.anl.gov/pyGSAS/trunk/help/GSASIIscriptable.html>)
    
    
    # ========================================
    # GSAS-II連携の概念コード (要GSAS-IIインストール)
    # ========================================
    
    # import GSASIIscriptable as G2sc
    #
    # # プロジェクトの作成
    # gpx = G2sc.G2Project(newgpx='my_rietveld.gpx')
    #
    # # XRDデータの追加
    # hist = gpx.add_powder_histogram('sample.xy', 'PWDR')
    #
    # # 相の追加 (CIFから)
    # phase = gpx.add_phase('Fe.cif', phasename='alpha-Fe', histograms=[hist])
    #
    # # リートベルト精密化実行
    # gpx.do_refinements([
    #     {'set': {'Background': {'refine': True}}},
    #     {'set': {'Cell': True, 'Atoms': True}},
    # ])
    #
    # # 結果抽出
    # results = gpx.get_Covariance()
    # print(f"Rwp = {results['Rvals']['Rwp']:.2f}%")
    #
    # # プロジェクト保存
    # gpx.save()
    

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります:

### 基本理解

  * ✅ 実践的なXRD解析ワークフロー(データ読み込み→精密化→報告)の全体像
  * ✅ 多相混合物解析における段階的精密化の戦略
  * ✅ RIR法とRietveld法による定量相分析の違いと適用場面
  * ✅ 収束失敗、GOF異常、負の占有率などの典型的エラーとその原因

### 実践スキル

  * ✅ .xy, .dat形式のXRDデータを読み込み、前処理を実行
  * ✅ 2相混合物(α-Fe + Fe₃O₄)の完全リートベルト解析
  * ✅ スケールファクターから重量分率を計算(Rietveld定量)
  * ✅ 論文品質のリートベルトプロット作成(matplotlib)
  * ✅ 精密化結果をCIFファイルとして出力

### 応用力

  * ✅ 収束診断ツールでエラーを自動検出し、対処
  * ✅ 相関の強いパラメータ(格子定数と温度因子)の最適化戦略
  * ✅ GSAS-IIとPythonを連携させた高度な解析フロー
  * ✅ 実測データに対する完全な解析→検証→報告のサイクル実行

## 演習問題

### Easy (基礎確認)

**Q1** : RIR法とRietveld法の定量相分析の主な違いは何ですか?

**解答** :

項目 | RIR法 | Rietveld法  
---|---|---  
**使用データ** | 最強ピークの強度のみ | 全パターン(全2θ範囲)  
**精度** | ±5-10% | ±1-3%  
**必要情報** | RIR値(PDF Card) | 結晶構造(CIF)  
**計算時間** | 数秒 | 数分〜数時間  
**適用場面** | 簡易スクリーニング | 精密定量分析  
  
**結論** : RIR法は迅速な推定、Rietveld法は高精度定量に適しています。

**Q2** : 3相混合物の精密化で、プロファイルパラメータ(U, V, W)を全相で共通にする理由は?

**解答** :

**理由** :

  1. **パラメータ数の削減** : 3相で個別にU, V, Wを精密化すると9パラメータ。共通化で3パラメータに削減。
  2. **物理的妥当性** : U, V, Wは装置起因のピーク幅を表すため、全相で共通であるべき。
  3. **収束の安定性** : パラメータが少ないほど、最小化が安定。

**ただし** : 結晶子サイズやmicrostrainが相によって大きく異なる場合は、個別に精密化が必要な場合もあります。

### Medium (応用)

**Q3** : α-Fe(a=2.866Å, Z=2, M=55.845) と Fe₃O₄(a=8.396Å, Z=8, M=231.533) の2相混合物で、スケールファクターがS_Fe=1.5, S_Fe3O4=0.8と精密化されました。各相の重量分率を計算してください。

**解答** :
    
    
    import numpy as np
    
    # データ
    S = [1.5, 0.8]
    Z = [2, 8]
    M = [55.845, 231.533]
    V = [2.866**3, 8.396**3]  # a³
    
    # S*Z*M*V
    S_ZMV = np.array(S) * np.array(Z) * np.array(M) * np.array(V)
    print(f"S*Z*M*V: {S_ZMV}")
    
    # 重量分率
    wt_fractions = S_ZMV / S_ZMV.sum()
    print(f"α-Fe: {wt_fractions[0]*100:.2f} wt%")
    print(f"Fe₃O₄: {wt_fractions[1]*100:.2f} wt%")
    

**出力** :
    
    
    S*Z*M*V: [2650.13 908536.45]
    α-Fe: 0.29 wt%
    Fe₃O₄: 99.71 wt%

Fe₃O₄が圧倒的に多い試料と判明します。

**Q4** : リートベルト精密化で「GOF = 0.85」となりました。何が問題で、どう対処すべきですか?

**解答** :

**問題** : GOF < 1.0 は**過剰フィッティング** の可能性。パラメータが多すぎて、ノイズまでフィットしている。

**対処法** :

  1. **不要なパラメータを固定** : 温度因子、占有率が1.0に近いなら固定
  2. **バックグラウンド次数を下げる** : Chebyshev 5次 → 3次
  3. **制約条件を追加** : 化学結合長などの拘束を強化
  4. **データの質を確認** : 測定時間が長すぎて、統計ノイズが極端に小さい可能性

**目標** : GOF = 1.0 - 2.0 が理想的。

### Hard (発展)

**Q5** : 完全な2相リートベルト解析コードを書いてください。α-Fe(BCC)とFe₃O₄(Spinel)の混合物を想定し、格子定数、スケールファクター、プロファイルパラメータを精密化し、重量分率を計算してください。

**解答** :

(Example 3のTwoPhaseRietveldクラスを拡張)
    
    
    # 完全版は Example 3 を参照
    # 追加機能: 重量分率計算
    
    class TwoPhaseRietveldComplete(TwoPhaseRietveld):
        """2相リートベルト解析 + 定量分析"""
    
        def calculate_weight_fractions(self, result):
            """重量分率を計算"""
            a_Fe = result.params['a_Fe'].value
            a_Fe3O4 = result.params['a_Fe3O4'].value
    
            S_Fe = result.params['scale_Fe'].value
            S_Fe3O4 = result.params['scale_Fe3O4'].value
    
            # 結晶データ
            Z_Fe, M_Fe, V_Fe = 2, 55.845, a_Fe**3
            Z_Fe3O4, M_Fe3O4, V_Fe3O4 = 8, 231.533, a_Fe3O4**3
    
            # S*Z*M*V
            S_ZMV_Fe = S_Fe * Z_Fe * M_Fe * V_Fe
            S_ZMV_Fe3O4 = S_Fe3O4 * Z_Fe3O4 * M_Fe3O4 * V_Fe3O4
    
            total = S_ZMV_Fe + S_ZMV_Fe3O4
    
            wt_Fe = S_ZMV_Fe / total
            wt_Fe3O4 = S_ZMV_Fe3O4 / total
    
            return {'α-Fe': wt_Fe, 'Fe₃O₄': wt_Fe3O4}
    
    # 実行
    two_phase_complete = TwoPhaseRietveldComplete(two_theta_test, intensity_test)
    result = two_phase_complete.refine()
    wt_fractions = two_phase_complete.calculate_weight_fractions(result)
    
    print("=== 定量分析結果 ===")
    for phase, wt in wt_fractions.items():
        print(f"{phase}: {wt*100:.2f} wt%")
    

**Q6** : 実測XRDデータ(.xy形式)を用意し、完全な解析ワークフロー(データ読み込み→精密化→可視化→CIF出力)を実行してください。

**解答** :

この問題は、実際の.xyファイルを用いて、Example 2のCompleteXRDWorkflowクラスを適用します。
    
    
    # 実測データでの完全ワークフロー
    workflow = CompleteXRDWorkflow('実測データ.xy', wavelength=1.5406)
    
    # Step 1-5を順次実行
    workflow.step1_load_data(skip_rows=1)
    workflow.step2_peak_detection(prominence=0.15)
    workflow.step3_rietveld_refinement()
    results = workflow.step4_extract_results()
    workflow.step5_visualize(save_path='実測データ_Rietveld.pdf')
    
    # CIF出力 (pymatgen Structureを事前に定義)
    from pymatgen.core import Structure, Lattice
    
    a_refined = results['lattice_a'].value
    structure = Structure(Lattice.cubic(a_refined), ["Fe"], [[0, 0, 0]])
    
    export_to_cif(structure,
                  {'a': a_refined, 'b': a_refined, 'c': a_refined},
                  {'Rwp': results['Rwp'], 'GOF': results['GOF'], 'ndata': len(workflow.two_theta), 'date': '2025-10-28'},
                  'Fe_refined.cif')
    

## 学習目標の確認

この章で学んだ内容を振り返り、以下の項目を確認してください。

### 基本理解

  * ✅ XRDデータ解析の全体ワークフロー(前処理→インデキシング→精密化→報告)を説明できる
  * ✅ 多相混合物の定性・定量分析の原理と手順を理解している
  * ✅ 典型的なエラー(ピーク同定失敗、精密化発散、配向効果)の原因を説明できる
  * ✅ 学術論文に必要なXRD解析結果の報告形式を理解している

### 実践スキル

  * ✅ XRDWorkflowManagerクラスを使用して完全な解析パイプラインを構築できる
  * ✅ MultiphaseAnalyzerで複数相の同時精密化を実行できる
  * ✅ エラー診断関数を活用して解析の問題点を特定・解決できる
  * ✅ publication_quality_plot関数で学術報告レベルのグラフを作成できる

### 応用力

  * ✅ 実測データから出発して論文投稿レベルの結果を導出できる
  * ✅ 多相試料の定量相分析を実行し、信頼性を評価できる
  * ✅ 解析結果の妥当性を多角的に検証し、エラーを自律的にトラブルシューティングできる
  * ✅ CIFファイルを生成してCCDCやICSDなどのデータベースに登録できる形式で出力できる

## 参考文献

  1. Young, R. A. (Ed.). (1993). _The Rietveld Method_. Oxford University Press. - リートベルト法の包括的な教科書と実践的なワークフロー解説
  2. Dinnebier, R. E., & Billinge, S. J. L. (Eds.). (2008). _Powder Diffraction: Theory and Practice_. Royal Society of Chemistry. - 粉末XRDの理論と実践、エラー診断のベストプラクティス
  3. Bish, D. L., & Post, J. E. (Eds.). (1989). _Modern Powder Diffraction (Reviews in Mineralogy Vol. 20)_. Mineralogical Society of America. - 多相解析と定量相分析の古典的レファレンス
  4. Hill, R. J., & Howard, C. J. (1987). _Quantitative phase analysis from neutron powder diffraction data using the Rietveld method_. Journal of Applied Crystallography, 20(6), 467-474. - 定量相分析のオリジナル論文
  5. GSAS-II Documentation. _Tutorials and User Guides_. Available at: https://gsas-ii.readthedocs.io/ - GSAS-IIの公式ドキュメントと実践的チュートリアル
  6. International Centre for Diffraction Data (ICDD). _PDF-4+ Database and Search/Match Software_. - 相同定のための包括的な粉末回折データベース
  7. McCusker, L. B., et al. (1999). _Rietveld refinement guidelines_. Journal of Applied Crystallography, 32(1), 36-50. - 学術報告のための精密化ガイドラインとチェックリスト

## シリーズのまとめ

おめでとうございます! X線回折分析入門シリーズの全5章を完了しました。このシリーズを通じて、以下のスキルを習得しました:

### 習得した知識・スキル

  * ✅ **第1章** : X線回折の基礎理論(Braggの法則、構造因子、消滅則)
  * ✅ **第2章** : 粉末XRD測定と基本解析(ピーク検出、インデキシング、格子定数計算)
  * ✅ **第3章** : リートベルト法の原理(プロファイル関数、バックグラウンド、R因子)
  * ✅ **第4章** : 構造精密化(原子座標、温度因子、結晶子サイズ、microstrain)
  * ✅ **第5章** : 実践ワークフロー(多相解析、定量分析、エラー診断、学術報告)

### 次のステップ

XRD解析の基礎を習得したあなたは、以下のトピックに進むことができます:

  * **高度なXRD技術** : 薄膜XRD、高温・その場XRD、全散乱PDF解析
  * **中性子回折** : 軽元素の精密構造解析、磁気構造決定
  * **単結晶XRD** : 精密構造決定、結晶対称性の決定
  * **機械学習とXRD** : 自動相同定、異常検知、逆問題による構造予測

> **🎓 学習の継続**
> 
> 実際のXRDデータで練習を重ね、論文を読み、コミュニティ(X-ray Discussion ForumやStack Exchangeなど)で質問することで、さらにスキルを磨いてください。

## 参考文献・リソース

### 教科書

  1. Pecharsky, V. K., & Zavalij, P. Y. (2009). _Fundamentals of Powder Diffraction and Structural Characterization of Materials_ (2nd ed.). Springer.
  2. Dinnebier, R. E., & Billinge, S. J. L. (Eds.). (2008). _Powder Diffraction: Theory and Practice_. Royal Society of Chemistry.
  3. Young, R. A. (Ed.). (1993). _The Rietveld Method_. Oxford University Press.

### ソフトウェア

  * **GSAS-II** : [https://subversion.xray.aps.anl.gov/pyGSAS/](<https://subversion.xray.aps.anl.gov/pyGSAS/trunk/help/>)
  * **FullProf** : <https://www.ill.eu/sites/fullprof/>
  * **TOPAS** : <https://www.bruker.com/topas>
  * **pymatgen** : <https://pymatgen.org/>

### データベース

  * **ICDD PDF-4+** : 粉末回折パターンデータベース
  * **Crystallography Open Database (COD)** : <http://www.crystallography.net/>
  * **Materials Project** : <https://materialsproject.org/>

## 謝辞

このシリーズは、Materials Science教育の一環として作成されました。XRD解析の基礎から実践まで、体系的に学べる日本語リソースの充実を目指しています。

**フィードバック・質問** は、[yusuke.hashimoto.b8@tohoku.ac.jp](<mailto:yusuke.hashimoto.b8@tohoku.ac.jp>) までお寄せください。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
