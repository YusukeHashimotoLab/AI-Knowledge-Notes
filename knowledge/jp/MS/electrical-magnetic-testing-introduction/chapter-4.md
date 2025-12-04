---
title: "第4章: Python実践：電磁気データ解析ワークフロー"
chapter_title: "第4章: Python実践：電磁気データ解析ワークフロー"
subtitle: データ読み込み、前処理、フィッティング、可視化、自動レポート生成の統合パイプライン
reading_time: 55-65分
difficulty: 中級〜上級
code_examples: 7
---

この章では、実際の測定装置（VSM、SQUID、Hall測定システム）から得られるデータを扱う実践的なPythonワークフローを習得します。CSVやバイナリデータの読み込み、異常値除去、バックグラウンド減算、高度なフィッティング技術（lmfit、scipy.optimize）、誤差伝播、機械学習を用いた異常検出、publication-qualityの図の作成、自動レポート生成までの統合的なパイプラインを構築します。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 実測装置データ（CSV、DAT、バイナリ）を読み込み、前処理できる
  * ✅ 異常値検出とクリーニングを自動化できる
  * ✅ 四端子法・Hall・M-H曲線を統合解析できる
  * ✅ lmfitとscipy.optimizeで高度なフィッティングができる
  * ✅ 誤差伝播を正しく計算できる
  * ✅ publication-qualityの図を自動生成できる
  * ✅ PDFレポートを自動生成できる

## 4.1 データ読み込みとクリーニング

### 4.1.1 多形式データローダー

実際の測定装置は、様々な形式でデータを出力します：

装置 | 形式 | ヘッダー | 区切り文字  
---|---|---|---  
Quantum Design VSM | .dat | 複数行コメント（#） | タブまたはカンマ  
Keithley 2400 SMU | .csv | 1行または無し | カンマ  
Lake Shore 7400 VSM | .txt | 固定フォーマット | スペース  
カスタムLabVIEW | .bin | バイナリヘッダー | N/A  
      
    
    ```mermaid
    flowchart LR
        A[Raw DataCSV/DAT/BIN] --> B[Data Loaderpandas/numpy]
        B --> C{FormatDetection}
        C -->|CSV| D[pd.read_csv]
        C -->|DAT| E[Custom Parser]
        C -->|Binary| F[np.fromfile]
        
        D --> G[Data Cleaning]
        E --> G
        F --> G
        
        G --> H[Remove NaN/Inf]
        G --> I[Outlier Detection]
        G --> J[Unit Conversion]
        
        H --> K[Clean Dataset]
        I --> K
        J --> K
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style K fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

#### コード例4-1: 多形式データローダークラス
    
    
    import numpy as np
    import pandas as pd
    import struct
    from pathlib import Path
    from typing import Union, Tuple
    
    class UniversalDataLoader:
        """
        多形式データローダー（CSV、DAT、バイナリ対応）
        
        Attributes
        ----------
        data : pd.DataFrame
            読み込んだデータ
        metadata : dict
            メタデータ（ヘッダー情報、測定条件など）
        """
    
        def __init__(self):
            self.data = None
            self.metadata = {}
    
        def load(self, filepath: str, format: str = 'auto') -> pd.DataFrame:
            """
            データファイルを読み込む
            
            Parameters
            ----------
            filepath : str
                ファイルパス
            format : str
                'auto'（自動検出）, 'csv', 'dat', 'binary'
            
            Returns
            -------
            data : pd.DataFrame
                読み込んだデータ
            """
            path = Path(filepath)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
    
            # 形式の自動検出
            if format == 'auto':
                format = self._detect_format(path)
    
            # 形式に応じた読み込み
            if format == 'csv':
                self.data = self._load_csv(path)
            elif format == 'dat':
                self.data = self._load_dat(path)
            elif format == 'binary':
                self.data = self._load_binary(path)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
            print(f"Loaded {len(self.data)} rows from {path.name} (format: {format})")
            return self.data
    
        def _detect_format(self, path: Path) -> str:
            """ファイル拡張子から形式を推定"""
            ext = path.suffix.lower()
            if ext in ['.csv', '.txt']:
                return 'csv'
            elif ext == '.dat':
                return 'dat'
            elif ext == '.bin':
                return 'binary'
            else:
                # 中身を見て判定
                with open(path, 'rb') as f:
                    header = f.read(100)
                    if b'\x00' in header:
                        return 'binary'
                    else:
                        return 'csv'
    
        def _load_csv(self, path: Path) -> pd.DataFrame:
            """CSV形式の読み込み"""
            # ヘッダー行数の検出
            header_lines = 0
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip().startswith('#'):
                        header_lines += 1
                        self.metadata[f'comment_{i}'] = line.strip()
                    else:
                        break
    
            # データ読み込み
            try:
                data = pd.read_csv(path, skiprows=header_lines, sep=None, engine='python')
            except Exception as e:
                # カンマ区切り以外の場合
                data = pd.read_csv(path, skiprows=header_lines, delimiter=r'\s+')
    
            return data
    
        def _load_dat(self, path: Path) -> pd.DataFrame:
            """DAT形式の読み込み（Quantum Design VSM想定）"""
            # メタデータとデータを分離
            metadata_lines = []
            data_lines = []
            
            with open(path, 'r') as f:
                in_data_section = False
                for line in f:
                    line = line.strip()
                    if line.startswith('[Data]'):
                        in_data_section = True
                        continue
                    
                    if not in_data_section:
                        if line and not line.startswith('#'):
                            # メタデータ行
                            if '=' in line:
                                key, value = line.split('=', 1)
                                self.metadata[key.strip()] = value.strip()
                    else:
                        if line and not line.startswith('#'):
                            data_lines.append(line)
    
            # データをDataFrameに変換
            if data_lines:
                # カラム名（最初の行）
                header = data_lines[0].split('\t')
                
                # データ行
                data_values = []
                for line in data_lines[1:]:
                    values = line.split('\t')
                    data_values.append([float(v) if v else np.nan for v in values])
    
                data = pd.DataFrame(data_values, columns=header)
            else:
                data = pd.DataFrame()
    
            return data
    
        def _load_binary(self, path: Path) -> pd.DataFrame:
            """
            バイナリ形式の読み込み（カスタムフォーマット）
            
            仮定：
            - ヘッダー：最初の100バイト（メタデータ）
            - データ：float64配列、3カラム（H, M, T）
            """
            with open(path, 'rb') as f:
                # ヘッダー読み込み
                header_bytes = f.read(100)
                # メタデータ解析（ここでは省略）
                
                # データ読み込み
                remaining = f.read()
                n_cols = 3
                n_rows = len(remaining) // (8 * n_cols)  # float64 = 8 bytes
                
                data_array = np.frombuffer(remaining, dtype=np.float64)
                data_array = data_array.reshape((n_rows, n_cols))
                
                data = pd.DataFrame(data_array, columns=['H', 'M', 'T'])
    
            return data
    
        def clean_data(self, remove_nan=True, remove_inf=True, outlier_method='iqr', threshold=3.0):
            """
            データクリーニング
            
            Parameters
            ----------
            remove_nan : bool
                NaNを削除するか
            remove_inf : bool
                Infを削除するか
            outlier_method : str
                外れ値除去法（'iqr', 'zscore', 'none'）
            threshold : float
                外れ値判定の閾値
            """
            if self.data is None:
                raise ValueError("No data loaded. Call load() first.")
    
            original_len = len(self.data)
    
            # NaN/Inf除去
            if remove_nan:
                self.data = self.data.dropna()
            if remove_inf:
                self.data = self.data.replace([np.inf, -np.inf], np.nan).dropna()
    
            # 外れ値除去
            if outlier_method == 'iqr':
                self.data = self._remove_outliers_iqr(self.data, threshold)
            elif outlier_method == 'zscore':
                self.data = self._remove_outliers_zscore(self.data, threshold)
    
            print(f"Data cleaning: {original_len} → {len(self.data)} rows (removed {original_len - len(self.data)})")
    
            return self.data
    
        def _remove_outliers_iqr(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
            """IQR法による外れ値除去"""
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = ((data >= lower_bound) & (data <= upper_bound)).all(axis=1)
            return data[mask]
    
        def _remove_outliers_zscore(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
            """Z-score法による外れ値除去"""
            z_scores = np.abs((data - data.mean()) / data.std())
            mask = (z_scores < threshold).all(axis=1)
            return data[mask]
    
    # 使用例
    loader = UniversalDataLoader()
    
    # CSV読み込み
    data = loader.load('vsm_measurement.csv', format='auto')
    print(data.head())
    print(f"\nMetadata: {loader.metadata}")
    
    # クリーニング
    data_clean = loader.clean_data(outlier_method='iqr', threshold=3.0)
    print(f"\nCleaned data shape: {data_clean.shape}")
    

## 4.2 統合解析パイプライン

### 4.2.1 四端子法 + Hall + M-H統合解析

実験では、同一試料に対して複数の測定を行います。これらを統合的に解析することで、材料の電気的・磁気的特性を完全に理解できます。

#### コード例4-2: 統合解析パイプライン
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve, curve_fit
    from lmfit import Model
    
    class IntegratedAnalysisPipeline:
        """
        四端子法 + Hall + M-H統合解析パイプライン
        
        Attributes
        ----------
        thickness : float
            試料厚さ [m]
        mass : float
            試料質量 [g]
        results : dict
            解析結果（σ, n, μ, M_s, H_c, M_r, K）
        """
    
        def __init__(self, thickness: float, mass: float = None):
            self.t = thickness
            self.mass = mass
            self.results = {}
            self.data = {}
    
        def load_four_probe_data(self, R_AB_CD: float, R_BC_DA: float):
            """van der Pauw四端子データ読み込み"""
            self.data['R_AB_CD'] = R_AB_CD
            self.data['R_BC_DA'] = R_BC_DA
    
        def load_hall_data(self, I: float, B: float, V_pos: float, V_neg: float):
            """Hallデータ読み込み"""
            self.data['I'] = I
            self.data['B'] = B
            self.data['V_hall_pos'] = V_pos
            self.data['V_hall_neg'] = V_neg
    
        def load_mh_data(self, H: np.ndarray, M: np.ndarray):
            """M-Hデータ読み込み"""
            self.data['H'] = H
            self.data['M'] = M
    
        def analyze_electrical_properties(self):
            """電気的特性解析（四端子法 + Hall）"""
            # van der Pauw シート抵抗
            def vdp_eq(Rs, R1, R2):
                return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
    
            R1 = self.data['R_AB_CD']
            R2 = self.data['R_BC_DA']
            R_initial = (R1 + R2) / 2 * np.pi / np.log(2)
            R_s = fsolve(vdp_eq, R_initial, args=(R1, R2))[0]
    
            # 電気伝導率
            sigma = 1 / (R_s * self.t)
    
            # Hall係数
            V_H = 0.5 * (self.data['V_hall_pos'] - self.data['V_hall_neg'])
            R_H = V_H * self.t / (self.data['I'] * self.data['B'])
    
            # キャリア密度
            e = 1.60218e-19
            n = 1 / (np.abs(R_H) * e)
            carrier_type = 'electron' if R_H < 0 else 'hole'
    
            # 移動度
            mu = sigma * np.abs(R_H)
    
            # 結果保存
            self.results.update({
                'R_s': R_s,
                'sigma': sigma,
                'rho': 1 / sigma,
                'R_H': R_H,
                'n': n,
                'carrier_type': carrier_type,
                'mu': mu
            })
    
            return self.results
    
        def analyze_magnetic_properties(self):
            """磁気的特性解析（M-H曲線）"""
            H = self.data['H']
            M = self.data['M']
    
            # バックグラウンド減算
            H_high = H[H > 0.8 * np.max(H)]
            M_high = M[H > 0.8 * np.max(H)]
            slope = np.polyfit(H_high, M_high, 1)[0]
            M_corrected = M - slope * H
    
            # 飽和磁化
            M_s = np.mean(M_corrected[H > 0.8 * np.max(H)])
    
            # 保磁力
            from scipy.interpolate import interp1d
            from scipy.optimize import brentq
            interp_func = interp1d(H, M_corrected, kind='linear')
            H_c = brentq(interp_func, np.min(H[H < 0]), np.max(H[H > 0]))
    
            # 残留磁化
            M_r = interp_func(0)
    
            # 角形比
            S = M_r / M_s if M_s != 0 else 0
    
            # 磁気異方性定数
            K = H_c * M_s / 2  # CGS単位 [erg/cm^3]
    
            # 結果保存
            self.results.update({
                'M_s': M_s,
                'H_c': H_c,
                'M_r': M_r,
                'S': S,
                'K': K
            })
    
            return self.results
    
        def generate_report(self):
            """統合レポート生成"""
            print("=" * 80)
            print("INTEGRATED ELECTRICAL & MAGNETIC PROPERTIES ANALYSIS")
            print("=" * 80)
            
            print("\n[Electrical Properties]")
            print(f"  Sheet Resistance R_s = {self.results['R_s']:.2f} Ω/sq")
            print(f"  Conductivity σ = {self.results['sigma']:.2e} S/m")
            print(f"  Resistivity ρ = {self.results['rho']:.2e} Ω·m = {self.results['rho'] * 1e8:.2f} μΩ·cm")
            print(f"  Carrier Type: {self.results['carrier_type']}")
            print(f"  Carrier Density n = {self.results['n']:.2e} m⁻³ = {self.results['n'] / 1e6:.2e} cm⁻³")
            print(f"  Mobility μ = {self.results['mu']:.2e} m²/(V·s) = {self.results['mu'] * 1e4:.1f} cm²/(V·s)")
    
            print("\n[Magnetic Properties]")
            print(f"  Saturation Magnetization M_s = {self.results['M_s']:.2f} emu/g")
            print(f"  Coercivity H_c = {self.results['H_c']:.2f} Oe = {self.results['H_c'] / 79.5775:.2f} kA/m")
            print(f"  Remanence M_r = {self.results['M_r']:.2f} emu/g")
            print(f"  Squareness S = {self.results['S']:.3f}")
            print(f"  Anisotropy Constant K = {self.results['K']:.2e} erg/cm³ = {self.results['K'] * 1e3:.2e} J/m³")
    
            print("\n[Material Classification]")
            if self.results['H_c'] < 100:
                print("  → Soft magnetic material (transformers, inductors)")
            elif self.results['H_c'] > 1000:
                print("  → Hard magnetic material (permanent magnets)")
            else:
                print("  → Medium coercivity (recording media)")
    
            if self.results['mu'] * 1e4 > 1000:
                print("  → High mobility material (high-performance electronics)")
            else:
                print("  → Moderate mobility (standard electronics)")
    
            print("=" * 80)
    
    # 使用例
    pipeline = IntegratedAnalysisPipeline(thickness=200e-9, mass=0.005)
    
    # データ読み込み
    pipeline.load_four_probe_data(R_AB_CD=1000, R_BC_DA=950)
    pipeline.load_hall_data(I=100e-6, B=0.5, V_pos=-5.0e-3, V_neg=4.8e-3)
    
    H_data = np.linspace(-5000, 5000, 200)
    M_data = 50 * np.tanh(H_data / 1000) + 0.5 * np.random.randn(200)
    pipeline.load_mh_data(H=H_data, M=M_data)
    
    # 解析実行
    pipeline.analyze_electrical_properties()
    pipeline.analyze_magnetic_properties()
    
    # レポート生成
    pipeline.generate_report()
    

## 4.3 高度なフィッティング技術

### 4.3.1 lmfitによる制約付きフィッティング

**lmfit** は、scipy.optimizeのラッパーで、パラメータの制約、誤差推定、相関行列の計算が容易です。

#### コード例4-3: lmfitによる複雑なフィッティング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model, Parameters
    
    def temperature_dependent_hall(T, n0, Ea, mu0, alpha):
        """
        温度依存性Hallモデル
        
        n(T) = n0 * exp(Ea / (k_B * T))  # キャリア密度
        μ(T) = μ0 * (T / 300)^(-α)      # 移動度
        R_H(T) = 1 / (n(T) * e)
        
        Parameters
        ----------
        T : array-like
            温度 [K]
        n0 : float
            基準キャリア密度 [m^-3]
        Ea : float
            活性化エネルギー [eV]
        mu0 : float
            基準移動度（300 K）[m^2/(V·s)]
        alpha : float
            移動度の温度指数
        
        Returns
        -------
        R_H : array-like
            Hall係数 [m^3/C]
        """
        k_B = 8.617e-5  # [eV/K]
        e = 1.60218e-19  # [C]
        
        n = n0 * np.exp(-Ea / (k_B * T))
        R_H = 1 / (n * e)
        
        return R_H
    
    # シミュレーションデータ生成
    T_range = np.linspace(200, 400, 25)
    n0_true = 1e21  # [m^-3]
    Ea_true = 0.3  # [eV]
    mu0_true = 0.05  # [m^2/(V·s)]
    alpha_true = 1.5
    
    R_H_data = temperature_dependent_hall(T_range, n0_true, Ea_true, mu0_true, alpha_true)
    R_H_data_noise = R_H_data * (1 + 0.08 * np.random.randn(len(T_range)))
    
    # lmfitモデル定義
    model = Model(temperature_dependent_hall)
    
    # パラメータ設定（初期値、範囲、制約）
    params = model.make_params(
        n0 = {'value': 5e20, 'min': 1e19, 'max': 1e23},
        Ea = {'value': 0.4, 'min': 0.1, 'max': 1.0},
        mu0 = {'value': 0.1, 'min': 0.01, 'max': 1.0},
        alpha = {'value': 1.0, 'min': 0.5, 'max': 3.0}
    )
    
    # フィッティング実行
    result = model.fit(R_H_data_noise, params, T=T_range)
    
    # 結果表示
    print("=" * 80)
    print("ADVANCED FITTING WITH LMFIT")
    print("=" * 80)
    print(result.fit_report())
    
    # パラメータ相関行列
    print("\nParameter Correlation Matrix:")
    print(result.params.correlation)
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上：R_H vs T
    axes[0, 0].scatter(T_range, R_H_data_noise, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data (with noise)', color='#f093fb')
    axes[0, 0].plot(T_range, result.best_fit, linewidth=2.5, label='Fit', color='#f5576c')
    axes[0, 0].plot(T_range, R_H_data, linewidth=2, linestyle='--', label='True (no noise)', color='green')
    axes[0, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 0].set_ylabel('Hall Coefficient R$_H$ [m$^3$/C]', fontsize=12)
    axes[0, 0].set_title('Hall Coefficient vs Temperature', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 右上：残差
    residuals = R_H_data_noise - result.best_fit
    axes[0, 1].scatter(T_range, residuals, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, color='#99ccff')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[0, 1].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 1].set_ylabel('Residuals [m$^3$/C]', fontsize=12)
    axes[0, 1].set_title('Fit Residuals', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 左下：キャリア密度
    k_B = 8.617e-5
    e = 1.60218e-19
    n_fit = result.params['n0'].value * np.exp(-result.params['Ea'].value / (k_B * T_range))
    axes[1, 0].semilogy(T_range, n_fit / 1e6, linewidth=2.5, color='#ffa500', label='Fitted n(T)')
    axes[1, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[1, 0].set_ylabel('Carrier Density n [cm$^{-3}$]', fontsize=12)
    axes[1, 0].set_title('Carrier Density (from fit)', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    
    # 右下：移動度
    mu_fit = result.params['mu0'].value * (T_range / 300)**(-result.params['alpha'].value)
    axes[1, 1].loglog(T_range, mu_fit * 1e4, linewidth=2.5, color='#99ff99', label='Fitted μ(T)')
    axes[1, 1].set_xlabel('Temperature T [K]', fontsize=12)
    axes[1, 1].set_ylabel('Mobility μ [cm$^2$/(V·s)]', fontsize=12)
    axes[1, 1].set_title('Mobility (from fit)', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # 物理的解釈
    print("\n" + "=" * 80)
    print("PHYSICAL INTERPRETATION")
    print("=" * 80)
    print(f"Activation Energy E_a = {result.params['Ea'].value:.3f} ± {result.params['Ea'].stderr:.3f} eV")
    print(f"  → Band gap or dopant ionization energy")
    print(f"Mobility Exponent α = {result.params['alpha'].value:.2f} ± {result.params['alpha'].stderr:.2f}")
    print(f"  → Scattering mechanism: α ≈ 1.5 suggests acoustic phonon scattering")
    print(f"Carrier Density (300 K) = {result.params['n0'].value * np.exp(-result.params['Ea'].value / (k_B * 300)):.2e} m⁻³")
    print(f"Mobility (300 K) = {result.params['mu0'].value * 1e4:.1f} cm²/(V·s)")
    

## 4.4 誤差伝播と不確かさ評価

### 4.4.1 自動誤差伝播

測定値の不確かさは、導出量（移動度、異方性定数など）に伝播します。**uncertainties** パッケージを使うと、自動的に誤差伝播を計算できます。

#### コード例4-4: 自動誤差伝播
    
    
    from uncertainties import ufloat, umath
    import numpy as np
    
    def propagate_uncertainties_example():
        """
        誤差伝播の実例
        
        Hall測定から移動度を計算し、不確かさを自動伝播
        """
        # 測定値（値 ± 不確かさ）
        R_H = ufloat(-2.5e-3, 0.1e-3)  # Hall係数 [m^3/C]
        sigma = ufloat(1e4, 100)  # 電気伝導率 [S/m]
        I = ufloat(100e-6, 1e-6)  # 電流 [A]
        B = ufloat(0.5, 0.01)  # 磁場 [T]
        t = ufloat(200e-9, 5e-9)  # 厚さ [m]
    
        # 移動度: μ = σ * |R_H|
        mu = sigma * umath.fabs(R_H)
    
        # キャリア密度: n = 1 / (e * |R_H|)
        e = 1.60218e-19  # 定数（不確かさなし）
        n = 1 / (e * umath.fabs(R_H))
    
        # 磁気異方性定数: K = H_c * M_s / 2
        H_c = ufloat(500, 20)  # [Oe]
        M_s = ufloat(50, 2)  # [emu/g]
        K = H_c * M_s / 2
    
        print("=" * 80)
        print("AUTOMATIC ERROR PROPAGATION")
        print("=" * 80)
        
        print("\n[Input Measurements]")
        print(f"  Hall Coefficient R_H = {R_H} m³/C")
        print(f"  Conductivity σ = {sigma} S/m")
        print(f"  Current I = {I} A")
        print(f"  Magnetic Field B = {B} T")
        print(f"  Thickness t = {t} m")
    
        print("\n[Derived Quantities with Propagated Uncertainties]")
        print(f"  Mobility μ = {mu} m²/(V·s)")
        print(f"            = ({mu.nominal_value * 1e4:.1f} ± {mu.std_dev * 1e4:.1f}) cm²/(V·s)")
        print(f"  Carrier Density n = {n} m⁻³")
        print(f"                    = ({n.nominal_value / 1e6:.2e} ± {n.std_dev / 1e6:.2e}) cm⁻³")
        print(f"  Anisotropy Constant K = {K} erg/cm³")
    
        print("\n[Relative Uncertainties]")
        print(f"  μ: {mu.std_dev / mu.nominal_value * 100:.2f}%")
        print(f"  n: {n.std_dev / n.nominal_value * 100:.2f}%")
        print(f"  K: {K.std_dev / K.nominal_value * 100:.2f}%")
    
        # 不確かさの寄与分析（手動計算で確認）
        print("\n[Uncertainty Budget for μ = σ * |R_H|]")
        rel_sigma = (100 / 1e4)**2
        rel_R_H = (0.1e-3 / 2.5e-3)**2
        rel_mu_manual = np.sqrt(rel_sigma + rel_R_H)
        print(f"  Contribution from σ: {np.sqrt(rel_sigma) * 100:.2f}%")
        print(f"  Contribution from R_H: {np.sqrt(rel_R_H) * 100:.2f}%")
        print(f"  Total (manual): {rel_mu_manual * 100:.2f}%")
        print(f"  Total (automatic): {mu.std_dev / mu.nominal_value * 100:.2f}%")
    
    propagate_uncertainties_example()
    

## 4.5 Publication-Quality図の作成

### 4.5.1 matplotlibベストプラクティス

論文掲載可能な図を作成するためのポイント：

  * フォントサイズ：ラベル 12-14pt、タイトル 14-16pt、凡例 10-12pt
  * 線幅：データ線 2-3pt、グリッド 0.5-1pt
  * マーカーサイズ：80-150（scatter）
  * 色：colorblind-friendlyなカラーパレット（例：seaborn、viridis）
  * 解像度：DPI 300以上（印刷用）
  * フォーマット：PDF（ベクター）またはPNG（ラスター）

#### コード例4-5: Publication-Quality図生成
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    
    # Publication設定
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.width'] = 1.0
    mpl.rcParams['ytick.minor.width'] = 1.0
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['ytick.major.size'] = 6
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['ytick.minor.size'] = 3
    
    def create_publication_figure():
        """
        Publication-qualityの図を作成
        
        例：温度依存性Hall測定結果
        """
        # データ生成
        T = np.linspace(100, 400, 20)
        n = 1e22 * np.exp(-0.3 / (8.617e-5 * T))
        mu = 0.05 * (300 / T)**1.5
        sigma = n * 1.60218e-19 * mu
    
        # 図作成（2x2レイアウト）
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
        # (a) キャリア密度
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(T, n / 1e6, 'o-', linewidth=2.5, markersize=8, color='#2E86AB', 
                     markeredgecolor='black', markeredgewidth=1.5, label='Carrier density')
        ax1.set_xlabel('Temperature (K)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Carrier Density (cm$^{-3}$)', fontsize=14, fontweight='bold')
        ax1.set_title('(a) Temperature-Dependent Carrier Density', fontsize=14, fontweight='bold', loc='left')
        ax1.legend(fontsize=12, frameon=True, shadow=True)
        ax1.grid(True, which='both', alpha=0.3, linestyle='--')
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
    
        # (b) 移動度
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.loglog(T, mu * 1e4, 's-', linewidth=2.5, markersize=8, color='#A23B72',
                   markeredgecolor='black', markeredgewidth=1.5, label='Mobility')
        ax2.set_xlabel('Temperature (K)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Mobility (cm$^2$/(V·s))', fontsize=14, fontweight='bold')
        ax2.set_title('(b) Temperature-Dependent Mobility', fontsize=14, fontweight='bold', loc='left')
        ax2.legend(fontsize=12, frameon=True, shadow=True)
        ax2.grid(True, which='both', alpha=0.3, linestyle='--')
    
        # (c) 電気伝導率
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.semilogy(T, sigma, '^-', linewidth=2.5, markersize=8, color='#F18F01',
                     markeredgecolor='black', markeredgewidth=1.5, label='Conductivity')
        ax3.set_xlabel('Temperature (K)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Conductivity (S/m)', fontsize=14, fontweight='bold')
        ax3.set_title('(c) Electrical Conductivity', fontsize=14, fontweight='bold', loc='left')
        ax3.legend(fontsize=12, frameon=True, shadow=True)
        ax3.grid(True, which='both', alpha=0.3, linestyle='--')
        ax3.xaxis.set_minor_locator(AutoMinorLocator())
    
        # (d) Arrheniusプロット
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.semilogy(1000 / T, n / 1e6, 'D-', linewidth=2.5, markersize=8, color='#C73E1D',
                     markeredgecolor='black', markeredgewidth=1.5, label='Arrhenius plot')
        ax4.set_xlabel('1000/T (K$^{-1}$)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Carrier Density (cm$^{-3}$)', fontsize=14, fontweight='bold')
        ax4.set_title('(d) Arrhenius Plot', fontsize=14, fontweight='bold', loc='left')
        ax4.legend(fontsize=12, frameon=True, shadow=True)
        ax4.grid(True, which='both', alpha=0.3, linestyle='--')
    
        # 全体調整
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='both', which='major', labelsize=12, direction='in', top=True, right=True)
            ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True)
    
        # 保存（高解像度PDF + PNG）
        plt.savefig('hall_measurement_publication.pdf', dpi=300, bbox_inches='tight', format='pdf')
        plt.savefig('hall_measurement_publication.png', dpi=300, bbox_inches='tight', format='png')
        
        print("Figures saved:")
        print("  - hall_measurement_publication.pdf (vector, for publication)")
        print("  - hall_measurement_publication.png (raster, for preview)")
    
        plt.show()
    
    create_publication_figure()
    

## 4.6 自動レポート生成

### 4.6.1 PDF自動レポート

**matplotlib** と**matplotlib.backends.backend_pdf** を使って、複数ページのPDFレポートを自動生成します。

#### コード例4-6: PDFレポート自動生成
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime
    
    class AutoReportGenerator:
        """
        自動PDFレポート生成クラス
        
        Attributes
        ----------
        filename : str
            出力PDFファイル名
        metadata : dict
            レポートメタデータ
        """
    
        def __init__(self, filename='analysis_report.pdf'):
            self.filename = filename
            self.metadata = {
                'Title': 'Electrical & Magnetic Properties Analysis Report',
                'Author': 'MS Terakoya Analysis Pipeline',
                'Subject': 'Automated Data Analysis',
                'Keywords': 'Hall effect, Magnetometry, Python',
                'CreationDate': datetime.now()
            }
    
        def generate_report(self, results: dict):
            """
            完全なレポートを生成
            
            Parameters
            ----------
            results : dict
                解析結果（電気的・磁気的特性）
            """
            with PdfPages(self.filename) as pdf:
                # ページ1：サマリー
                self._add_summary_page(pdf, results)
                
                # ページ2：電気的特性グラフ
                self._add_electrical_plots(pdf, results)
                
                # ページ3：磁気的特性グラフ
                self._add_magnetic_plots(pdf, results)
                
                # ページ4：統計情報
                self._add_statistics_page(pdf, results)
    
                # メタデータ設定
                d = pdf.infodict()
                for key, value in self.metadata.items():
                    d[key] = value
    
            print(f"Report generated: {self.filename}")
    
        def _add_summary_page(self, pdf, results):
            """ページ1：サマリー（テキストのみ）"""
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.95, 'ANALYSIS SUMMARY', ha='center', fontsize=20, fontweight='bold')
            fig.text(0.5, 0.90, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ha='center', fontsize=10)
    
            # 電気的特性
            y_start = 0.75
            fig.text(0.1, y_start, 'ELECTRICAL PROPERTIES', fontsize=16, fontweight='bold', color='#2E86AB')
            fig.text(0.1, y_start - 0.05, f"Sheet Resistance R_s = {results['R_s']:.2f} Ω/sq", fontsize=12)
            fig.text(0.1, y_start - 0.10, f"Conductivity σ = {results['sigma']:.2e} S/m", fontsize=12)
            fig.text(0.1, y_start - 0.15, f"Carrier Type: {results['carrier_type']}", fontsize=12)
            fig.text(0.1, y_start - 0.20, f"Carrier Density n = {results['n']:.2e} m⁻³", fontsize=12)
            fig.text(0.1, y_start - 0.25, f"Mobility μ = {results['mu'] * 1e4:.1f} cm²/(V·s)", fontsize=12)
    
            # 磁気的特性
            y_start = 0.45
            fig.text(0.1, y_start, 'MAGNETIC PROPERTIES', fontsize=16, fontweight='bold', color='#A23B72')
            fig.text(0.1, y_start - 0.05, f"Saturation Magnetization M_s = {results['M_s']:.2f} emu/g", fontsize=12)
            fig.text(0.1, y_start - 0.10, f"Coercivity H_c = {results['H_c']:.2f} Oe", fontsize=12)
            fig.text(0.1, y_start - 0.15, f"Remanence M_r = {results['M_r']:.2f} emu/g", fontsize=12)
            fig.text(0.1, y_start - 0.20, f"Squareness S = {results['S']:.3f}", fontsize=12)
            fig.text(0.1, y_start - 0.25, f"Anisotropy Constant K = {results['K']:.2e} erg/cm³", fontsize=12)
    
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
        def _add_electrical_plots(self, pdf, results):
            """ページ2：電気的特性グラフ"""
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            
            # ダミーデータ（実際にはresultsから取得）
            T = np.linspace(100, 400, 20)
            n = results['n'] * np.exp(-0.1 / (8.617e-5 * T))
            mu = results['mu'] * (300 / T)**1.5
            sigma = n * 1.60218e-19 * mu
            R_H = 1 / (n * 1.60218e-19)
    
            # プロット
            axes[0, 0].semilogy(T, n / 1e6, 'o-', linewidth=2.5)
            axes[0, 0].set_xlabel('Temperature (K)')
            axes[0, 0].set_ylabel('n (cm⁻³)')
            axes[0, 0].set_title('Carrier Density vs T')
            axes[0, 0].grid(alpha=0.3)
    
            axes[0, 1].loglog(T, mu * 1e4, 's-', linewidth=2.5, color='#A23B72')
            axes[0, 1].set_xlabel('Temperature (K)')
            axes[0, 1].set_ylabel('μ (cm²/(V·s))')
            axes[0, 1].set_title('Mobility vs T')
            axes[0, 1].grid(alpha=0.3)
    
            axes[1, 0].semilogy(T, sigma, '^-', linewidth=2.5, color='#F18F01')
            axes[1, 0].set_xlabel('Temperature (K)')
            axes[1, 0].set_ylabel('σ (S/m)')
            axes[1, 0].set_title('Conductivity vs T')
            axes[1, 0].grid(alpha=0.3)
    
            axes[1, 1].plot(T, R_H, 'D-', linewidth=2.5, color='#C73E1D')
            axes[1, 1].set_xlabel('Temperature (K)')
            axes[1, 1].set_ylabel('R_H (m³/C)')
            axes[1, 1].set_title('Hall Coefficient vs T')
            axes[1, 1].grid(alpha=0.3)
    
            plt.suptitle('Electrical Properties', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
        def _add_magnetic_plots(self, pdf, results):
            """ページ3：磁気的特性グラフ"""
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            
            # M-H曲線（ダミー）
            H = np.linspace(-5000, 5000, 100)
            M = results['M_s'] * np.tanh(H / 1000)
    
            axes[0, 0].plot(H, M, linewidth=2.5, color='#f093fb')
            axes[0, 0].axhline(results['M_s'], linestyle='--', color='green', label=f"M_s = {results['M_s']:.1f}")
            axes[0, 0].axvline(results['H_c'], linestyle='--', color='red', label=f"H_c = {results['H_c']:.0f}")
            axes[0, 0].set_xlabel('H (Oe)')
            axes[0, 0].set_ylabel('M (emu/g)')
            axes[0, 0].set_title('M-H Hysteresis Loop')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
    
            # 他のプロットは省略（実際には温度依存性など）
            for ax in axes.flat[1:]:
                ax.text(0.5, 0.5, 'Additional magnetic\nproperties plots', ha='center', va='center', fontsize=14)
                ax.axis('off')
    
            plt.suptitle('Magnetic Properties', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
        def _add_statistics_page(self, pdf, results):
            """ページ4：統計情報"""
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.95, 'STATISTICAL SUMMARY', ha='center', fontsize=20, fontweight='bold')
            
            # 簡単な統計表（実際にはもっと詳細に）
            fig.text(0.1, 0.80, 'Measurement Quality Metrics:', fontsize=14, fontweight='bold')
            fig.text(0.1, 0.75, '  - Data points: 200', fontsize=12)
            fig.text(0.1, 0.70, '  - Outliers removed: 5 (2.5%)', fontsize=12)
            fig.text(0.1, 0.65, '  - Fit R²: 0.998', fontsize=12)
            fig.text(0.1, 0.60, '  - Residual std: 0.05', fontsize=12)
    
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    # 使用例
    results_example = {
        'R_s': 1370,
        'sigma': 2.43e3,
        'carrier_type': 'electron',
        'n': 2.36e20,
        'mu': 0.064,
        'M_s': 50,
        'H_c': 300,
        'M_r': 40,
        'S': 0.8,
        'K': 7.5e5
    }
    
    reporter = AutoReportGenerator(filename='integrated_analysis_report.pdf')
    reporter.generate_report(results_example)
    

## 4.7 完全なワークフロー統合

#### コード例4-7: エンドツーエンド解析パイプライン
    
    
    """
    完全なエンドツーエンド解析パイプライン
    
    ワークフロー:
    1. データ読み込み（CSV/DAT/Binary）
    2. データクリーニング（外れ値除去）
    3. 四端子法 + Hall + M-H統合解析
    4. 高度なフィッティング
    5. 誤差伝播
    6. Publication-quality図生成
    7. PDFレポート生成
    """
    
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    class EndToEndPipeline:
        """
        完全な解析パイプライン
        """
    
        def __init__(self, project_name: str):
            self.project_name = project_name
            self.loader = UniversalDataLoader()
            self.analyzer = IntegratedAnalysisPipeline(thickness=200e-9, mass=0.005)
            self.reporter = AutoReportGenerator(filename=f'{project_name}_report.pdf')
    
        def run(self, data_files: dict):
            """
            パイプライン実行
            
            Parameters
            ----------
            data_files : dict
                {'four_probe': 'path/to/file.csv',
                 'hall': 'path/to/file.csv',
                 'mh': 'path/to/file.csv'}
            """
            print("=" * 80)
            print(f"STARTING END-TO-END ANALYSIS PIPELINE: {self.project_name}")
            print("=" * 80)
    
            # Step 1: データ読み込み
            print("\n[Step 1] Loading data...")
            data_four_probe = self.loader.load(data_files['four_probe'])
            data_hall = self.loader.load(data_files['hall'])
            data_mh = self.loader.load(data_files['mh'])
    
            # Step 2: データクリーニング
            print("\n[Step 2] Cleaning data...")
            data_four_probe_clean = self.loader.clean_data(outlier_method='iqr')
            
            self.loader.data = data_hall
            data_hall_clean = self.loader.clean_data(outlier_method='iqr')
            
            self.loader.data = data_mh
            data_mh_clean = self.loader.clean_data(outlier_method='iqr')
    
            # Step 3: 統合解析
            print("\n[Step 3] Integrated analysis...")
            self.analyzer.load_four_probe_data(
                R_AB_CD=data_four_probe_clean['R_AB_CD'].mean(),
                R_BC_DA=data_four_probe_clean['R_BC_DA'].mean()
            )
            self.analyzer.load_hall_data(
                I=data_hall_clean['I'].mean(),
                B=data_hall_clean['B'].mean(),
                V_pos=data_hall_clean['V_pos'].mean(),
                V_neg=data_hall_clean['V_neg'].mean()
            )
            self.analyzer.load_mh_data(
                H=data_mh_clean['H'].values,
                M=data_mh_clean['M'].values
            )
    
            self.analyzer.analyze_electrical_properties()
            self.analyzer.analyze_magnetic_properties()
    
            # Step 4: レポート生成
            print("\n[Step 4] Generating report...")
            self.analyzer.generate_report()
            self.reporter.generate_report(self.analyzer.results)
    
            print("\n" + "=" * 80)
            print("PIPELINE COMPLETED")
            print("=" * 80)
    
            return self.analyzer.results
    
    # 使用例（モックデータで実行）
    if __name__ == '__main__':
        # モックデータファイル作成（実際にはここは不要）
        np.random.seed(42)
        pd.DataFrame({
            'R_AB_CD': 1000 + 50 * np.random.randn(30),
            'R_BC_DA': 950 + 45 * np.random.randn(30)
        }).to_csv('four_probe_data.csv', index=False)
    
        pd.DataFrame({
            'I': 100e-6 + 1e-6 * np.random.randn(30),
            'B': 0.5 + 0.01 * np.random.randn(30),
            'V_pos': -5.0e-3 + 0.1e-3 * np.random.randn(30),
            'V_neg': 4.8e-3 + 0.1e-3 * np.random.randn(30)
        }).to_csv('hall_data.csv', index=False)
    
        H = np.linspace(-5000, 5000, 200)
        M = 50 * np.tanh(H / 1000) + 0.5 * np.random.randn(200)
        pd.DataFrame({'H': H, 'M': M}).to_csv('mh_data.csv', index=False)
    
        # パイプライン実行
        pipeline = EndToEndPipeline(project_name='Sample_Material_XYZ')
        results = pipeline.run({
            'four_probe': 'four_probe_data.csv',
            'hall': 'hall_data.csv',
            'mh': 'mh_data.csv'
        })
    
        print("\nFinal results saved to:")
        print("  - Sample_Material_XYZ_report.pdf")
    

## 4.8 演習問題

### 演習4-1: データローダーの拡張（Easy）

Easy **問題** ：`UniversalDataLoader`クラスに、JSON形式のデータを読み込むメソッド `_load_json()` を追加せよ。

**解答例を表示**
    
    
    import json
    
    def _load_json(self, path: Path) -> pd.DataFrame:
        """JSON形式の読み込み"""
        with open(path, 'r') as f:
            data_dict = json.load(f)
        
        # 辞書からDataFrameに変換
        data = pd.DataFrame(data_dict)
        return data
    
    # クラスに追加
    UniversalDataLoader._load_json = _load_json
    
    # テスト
    loader = UniversalDataLoader()
    # loader.load('data.json', format='json')
    

### 演習4-2: 外れ値検出の可視化（Easy）

Easy **問題** ：IQR法で検出された外れ値を、元のデータと一緒にプロットせよ（外れ値を赤色でマーク）。

**解答例を表示**
    
    
    import matplotlib.pyplot as plt
    
    data = np.random.randn(100)
    data[95:] = 10  # 外れ値を追加
    
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = (data < lower) | (data > upper)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data)), data, c=outliers, cmap='RdYlGn_r', s=50, edgecolors='black')
    plt.axhline(lower, color='blue', linestyle='--', label='Lower bound')
    plt.axhline(upper, color='blue', linestyle='--', label='Upper bound')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Outlier Detection (IQR method)')
    plt.legend()
    plt.colorbar(label='Outlier (True=Red)')
    plt.show()
    

### 演習4-3: フィッティング残差の分析（Medium）

Medium **問題** ：lmfitのフィッティング結果から、残差の正規性をQ-Qプロットで確認せよ。

**解答例を表示**
    
    
    import scipy.stats as stats
    
    # フィッティング実行（前のコード例から）
    residuals = result.residual
    
    # Q-Qプロット
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Residuals Normality Check', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.show()
    
    # 統計検定（Shapiro-Wilk検定）
    stat, p_value = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test: statistic={stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print("  → Residuals are normally distributed (p > 0.05)")
    else:
        print("  → Residuals are NOT normally distributed (p < 0.05)")
    

### 演習4-4: 誤差伝播の手動計算（Medium）

Medium **問題** ：移動度 $\mu = \sigma |R_H|$ の不確かさを、偏微分を使って手動計算し、`uncertainties`パッケージの結果と比較せよ。

**解答例を表示**
    
    
    sigma_val = 1e4
    sigma_err = 100
    R_H_val = 2.5e-3
    R_H_err = 0.1e-3
    
    # 手動計算：δμ = sqrt((∂μ/∂σ)² δσ² + (∂μ/∂R_H)² δR_H²)
    # ∂μ/∂σ = |R_H|
    # ∂μ/∂R_H = σ * sign(R_H)
    
    dmu_dsigma = R_H_val
    dmu_dRH = sigma_val
    
    delta_mu_manual = np.sqrt((dmu_dsigma * sigma_err)**2 + (dmu_dRH * R_H_err)**2)
    mu_val = sigma_val * R_H_val
    
    print(f"手動計算:")
    print(f"  μ = {mu_val:.2e} m²/(V·s)")
    print(f"  Δμ = {delta_mu_manual:.2e} m²/(V·s)")
    print(f"  相対不確かさ = {delta_mu_manual / mu_val * 100:.2f}%")
    
    # uncertaintiesパッケージ
    from uncertainties import ufloat
    sigma_u = ufloat(sigma_val, sigma_err)
    R_H_u = ufloat(R_H_val, R_H_err)
    mu_u = sigma_u * R_H_u
    
    print(f"\nuncertaintiesパッケージ:")
    print(f"  μ = {mu_u}")
    print(f"  相対不確かさ = {mu_u.std_dev / mu_u.nominal_value * 100:.2f}%")
    

### 演習4-5: カスタムレポートテンプレート作成（Medium）

Medium **問題** ：`AutoReportGenerator`を拡張し、ユーザー定義の図を追加できるメソッド `add_custom_page()` を実装せよ。

**解答例を表示**
    
    
    def add_custom_page(self, pdf, fig):
        """
        カスタム図をレポートに追加
        
        Parameters
        ----------
        pdf : PdfPages
            PDFページオブジェクト
        fig : matplotlib.figure.Figure
            追加する図
        """
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # クラスに追加
    AutoReportGenerator.add_custom_page = add_custom_page
    
    # 使用例
    reporter = AutoReportGenerator('custom_report.pdf')
    with PdfPages(reporter.filename) as pdf:
        # カスタムページ1
        fig1 = plt.figure(figsize=(8.5, 11))
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.title('Custom Plot 1')
        reporter.add_custom_page(pdf, fig1)
        
        # カスタムページ2
        fig2 = plt.figure(figsize=(8.5, 11))
        plt.scatter([1, 2, 3], [6, 5, 4])
        plt.title('Custom Plot 2')
        reporter.add_custom_page(pdf, fig2)
    
    print("Custom report generated")
    

### 演習4-6: バッチ処理パイプライン（Hard）

Hard **問題** ：複数試料のデータファイルを一括処理し、結果を1つのExcelファイル（複数シート）にまとめるパイプラインを作成せよ。

**解答例を表示**
    
    
    import pandas as pd
    from pathlib import Path
    
    def batch_process_samples(data_dir: str, output_file: str = 'batch_results.xlsx'):
        """
        複数試料を一括処理
        
        Parameters
        ----------
        data_dir : str
            データディレクトリ（各試料のサブディレクトリを含む）
        output_file : str
            出力Excelファイル名
        """
        data_path = Path(data_dir)
        sample_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        all_results = []
        
        for sample_dir in sample_dirs:
            sample_name = sample_dir.name
            print(f"Processing: {sample_name}")
            
            try:
                # パイプライン実行
                pipeline = EndToEndPipeline(project_name=sample_name)
                results = pipeline.run({
                    'four_probe': sample_dir / 'four_probe.csv',
                    'hall': sample_dir / 'hall.csv',
                    'mh': sample_dir / 'mh.csv'
                })
                
                results['sample_name'] = sample_name
                all_results.append(results)
            
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        
        # 結果をExcelに保存
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # サマリーシート
            df_summary = pd.DataFrame(all_results)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # 個別試料シート
            for result in all_results:
                sample_name = result['sample_name']
                df_sample = pd.DataFrame([result])
                df_sample.to_excel(writer, sheet_name=sample_name[:31], index=False)  # Excel制限
        
        print(f"\nBatch results saved to: {output_file}")
    
    # 使用例（ディレクトリ構造想定）
    # data/
    #   sample1/
    #     four_probe.csv
    #     hall.csv
    #     mh.csv
    #   sample2/
    #     ...
    
    # batch_process_samples('data/', 'all_samples_results.xlsx')
    

### 演習4-7: 機械学習を用いた異常検出（Hard）

Hard **問題** ：scikit-learnのIsolation Forestを使って、M-H曲線の異常なデータポイントを検出せよ。

**解答例を表示**
    
    
    from sklearn.ensemble import IsolationForest
    import numpy as np
    import matplotlib.pyplot as plt
    
    # M-Hデータ生成（一部に異常値）
    H = np.linspace(-5000, 5000, 200)
    M = 50 * np.tanh(H / 1000)
    M[50:55] += 20  # 異常値追加
    M[150:155] -= 15
    
    # 特徴量作成（H, M, dM/dH）
    dM_dH = np.gradient(M, H)
    X = np.column_stack([H, M, dM_dH])
    
    # Isolation Forest
    clf = IsolationForest(contamination=0.05, random_state=42)
    anomalies = clf.fit_predict(X)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図：M-H曲線
    colors = ['red' if a == -1 else 'blue' for a in anomalies]
    ax1.scatter(H, M, c=colors, s=50, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('H [Oe]')
    ax1.set_ylabel('M [emu/g]')
    ax1.set_title('M-H Curve with Anomaly Detection')
    ax1.grid(alpha=0.3)
    
    # 右図：異常スコア
    scores = clf.decision_function(X)
    ax2.plot(H, scores, linewidth=2, color='purple')
    ax2.axhline(0, color='red', linestyle='--', label='Threshold')
    ax2.set_xlabel('H [Oe]')
    ax2.set_ylabel('Anomaly Score')
    ax2.set_title('Isolation Forest Anomaly Scores')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Detected {np.sum(anomalies == -1)} anomalous points out of {len(anomalies)}")
    

### 演習4-8: 実験計画最適化（Hard）

Hard **問題** ：測定時間と精度のトレードオフを考慮し、最小限の測定点数で目標精度（誤差 < 5%）を達成する実験計画を提案せよ（シミュレーションベース）。

**解答例を表示**

**アプローチ** ：

  1. 異なる測定点数（N = 10, 20, 50, 100）でデータ生成
  2. 各Nでフィッティング精度を評価（R²、残差標準偏差）
  3. 測定時間を推定（1点 = 1分と仮定）
  4. 精度 vs 時間のトレードオフ曲線を作成

    
    
    N_range = [10, 20, 30, 50, 100, 200]
    fit_quality = []
    measurement_time = []
    
    for N in N_range:
        # データ生成
        T = np.linspace(100, 400, N)
        sigma_true = lambda T: 1e4 * np.exp(-0.2 / (8.617e-5 * T))
        sigma_data = sigma_true(T) * (1 + 0.05 * np.random.randn(N))
        
        # フィッティング
        from scipy.optimize import curve_fit
        def model(T, A, Ea):
            return A * np.exp(-Ea / (8.617e-5 * T))
        
        params, _ = curve_fit(model, T, sigma_data)
        sigma_fit = model(T, *params)
        
        # 精度評価
        R2 = 1 - np.sum((sigma_data - sigma_fit)**2) / np.sum((sigma_data - np.mean(sigma_data))**2)
        fit_quality.append(R2)
        
        # 測定時間（1点 = 1分）
        measurement_time.append(N * 1)
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(measurement_time, fit_quality, 'o-', linewidth=2.5, markersize=10, color='#f093fb')
    ax.axhline(0.95, color='red', linestyle='--', linewidth=2, label='Target R² = 0.95')
    ax.set_xlabel('Measurement Time [minutes]', fontsize=12)
    ax.set_ylabel('Fit Quality (R²)', fontsize=12)
    ax.set_title('Trade-off: Measurement Time vs Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.show()
    
    # 推奨測定点数
    optimal_idx = np.argmin(np.abs(np.array(fit_quality) - 0.95))
    print(f"Recommended: N = {N_range[optimal_idx]} points ({measurement_time[optimal_idx]} minutes)")
    

## 4.9 学習の確認

以下のチェックリストで理解度を確認しましょう：

### 基本理解

  * 実測装置データの多様な形式を理解している
  * データクリーニングの必要性と手法を説明できる
  * 四端子法・Hall・M-H測定の統合的な意義を理解している
  * lmfitによるフィッティングの利点を説明できる
  * 誤差伝播の重要性を理解している

### 実践スキル

  * 多形式データローダーを実装できる
  * IQR法・Z-score法で外れ値を除去できる
  * 統合解析パイプラインを構築できる
  * lmfitで制約付きフィッティングができる
  * uncertaintiesパッケージで誤差伝播を計算できる
  * Publication-qualityの図を作成できる
  * PDFレポートを自動生成できる

### 応用力

  * 完全なエンドツーエンドパイプラインを設計できる
  * 機械学習で異常検出ができる
  * 実験計画を最適化できる（精度 vs 時間のトレードオフ）
  * カスタムレポートテンプレートを作成できる

## 4.10 参考文献

  1. McKinney, W. (2017). _Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython_ (2nd ed.). O'Reilly. - pandasによるデータ処理
  2. VanderPlas, J. (2016). _Python Data Science Handbook_. O'Reilly. - 科学計算の総合ガイド
  3. Newville, M., et al. (2014). _LMFIT: Non-Linear Least-Square Minimization and Curve-Fitting for Python_. Zenodo. - lmfitドキュメント
  4. Hunter, J. D. (2007). _Matplotlib: A 2D Graphics Environment_. Computing in Science & Engineering, 9(3), 90-95. - matplotlibの原論文
  5. Lebigot, E. O. (2010). _Uncertainties: a Python package for calculations with uncertainties_. - 誤差伝播パッケージ
  6. Pedregosa, F., et al. (2011). _Scikit-learn: Machine Learning in Python_. Journal of Machine Learning Research, 12, 2825-2830. - scikit-learn
  7. Schroder, D. K. (2006). _Semiconductor Material and Device Characterization_ (3rd ed.). Wiley. - 測定データ解析の実践

## 4.11 まとめと次のステップ

この章では、実験データの読み込みから最終レポート生成までの完全なPythonワークフローを学びました。これで、電気・磁気測定の**全工程を自動化** できるようになりました。

**シリーズ全体の習得内容** ：

  * **第1章** ：四端子法、van der Pauw法、温度依存性測定
  * **第2章** ：Hall効果、キャリア密度・移動度決定、two-band model
  * **第3章** ：VSM/SQUID磁気測定、M-H曲線解析、FC/ZFC
  * **第4章** ：統合データ解析パイプライン、自動レポート生成

**次のステップ** ：

  * 実際の測定データでパイプラインを実行
  * 機械学習を用いた材料特性予測モデルの構築
  * リアルタイム測定システムとの統合
  * Webアプリケーション化（Streamlit、Dash）
