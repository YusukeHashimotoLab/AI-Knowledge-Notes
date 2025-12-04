---
title: 第5章：Python実践：プロセスデータ解析ワークフロー
chapter_title: 第5章：Python実践：プロセスデータ解析ワークフロー
subtitle: SPC, DOE, Machine Learning, Automated Reporting
reading_time: 35-45分
difficulty: 中級〜上級
code_examples: 7
---

プロセスデータ解析は、材料製造の品質管理と最適化の核心です。この章では、統計的プロセス制御（SPC）、実験計画法（DOE）、機械学習による予測モデル構築、異常検知、自動レポート生成までを統合したPythonワークフローを実践し、実務で即使えるスキルを習得します。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 多様なプロセスデータ形式（CSV, JSON, Excel, 装置固有フォーマット）の読み込みと前処理ができる
  * ✅ SPC（Statistical Process Control）チャート（X-bar, R-chart, Cp/Cpk）を生成・解釈できる
  * ✅ 実験計画法（DOE: Design of Experiments）を設計し、応答曲面法（RSM）で最適化できる
  * ✅ 機械学習（回帰・分類）でプロセス結果を予測し、特徴量重要度を評価できる
  * ✅ 異常検知（Isolation Forest, One-Class SVM）で不良品を早期発見できる
  * ✅ 自動レポート生成（matplotlib, seaborn, Jinja2）で日次/週次報告を効率化できる
  * ✅ 完全統合ワークフロー（データ → 解析 → 最適化 → レポート）を構築できる

## 5.1 プロセスデータの読み込みと前処理

### 5.1.1 多様なデータ形式への対応

実際のプロセスデータは、装置ログ（CSV, TXT）、データベースエクスポート（JSON, Excel）、独自フォーマット（バイナリ）など多岐にわたります。

**主要なデータ形式** ：

  * **CSV/TSV** ：最も一般的。pandas.read_csv()で読み込み
  * **Excel (.xlsx, .xls)** ：pandas.read_excel()で読み込み
  * **JSON** ：pandas.read_json()またはjson.load()で読み込み
  * **HDF5** ：大規模データ。pandas.read_hdf()で読み込み
  * **SQL Database** ：pandas.read_sql()で直接クエリ

#### コード例5-1: 多形式データローダー（バッチ処理）
    
    
    import pandas as pd
    import numpy as np
    import json
    import glob
    from pathlib import Path
    
    class ProcessDataLoader:
        """
        プロセスデータの統合ローダー
    
        複数形式・複数ファイルのバッチ処理をサポート
        """
    
        def __init__(self, data_dir='./process_data'):
            """
            Parameters
            ----------
            data_dir : str or Path
                データディレクトリのパス
            """
            self.data_dir = Path(data_dir)
            self.supported_formats = ['.csv', '.xlsx', '.json', '.txt']
    
        def load_single_file(self, filepath):
            """
            単一ファイルの読み込み
    
            Parameters
            ----------
            filepath : str or Path
                ファイルパス
    
            Returns
            -------
            df : pd.DataFrame
                読み込まれたデータ
            """
            filepath = Path(filepath)
            ext = filepath.suffix.lower()
    
            try:
                if ext == '.csv' or ext == '.txt':
                    # CSV/TXTの読み込み（区切り文字自動検出）
                    df = pd.read_csv(filepath, sep=None, engine='python')
                elif ext == '.xlsx' or ext == '.xls':
                    # Excelの読み込み（最初のシートのみ）
                    df = pd.read_excel(filepath)
                elif ext == '.json':
                    # JSONの読み込み
                    df = pd.read_json(filepath)
                else:
                    raise ValueError(f"Unsupported file format: {ext}")
    
                # メタデータ追加
                df['source_file'] = filepath.name
                df['load_timestamp'] = pd.Timestamp.now()
    
                print(f"Loaded: {filepath.name} ({len(df)} rows, {len(df.columns)} columns)")
                return df
    
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return None
    
        def load_batch(self, pattern='*', file_extension='.csv'):
            """
            バッチ読み込み（複数ファイルを統合）
    
            Parameters
            ----------
            pattern : str
                ファイル名パターン（ワイルドカード可）
            file_extension : str
                ファイル拡張子フィルタ
    
            Returns
            -------
            df_combined : pd.DataFrame
                統合されたデータフレーム
            """
            search_pattern = str(self.data_dir / f"{pattern}{file_extension}")
            files = glob.glob(search_pattern)
    
            if not files:
                print(f"No files found matching: {search_pattern}")
                return None
    
            print(f"Found {len(files)} files matching pattern '{pattern}{file_extension}'")
    
            dfs = []
            for filepath in sorted(files):
                df = self.load_single_file(filepath)
                if df is not None:
                    dfs.append(df)
    
            if not dfs:
                print("No data loaded successfully")
                return None
    
            # 統合
            df_combined = pd.concat(dfs, ignore_index=True)
            print(f"\nCombined data: {len(df_combined)} rows, {len(df_combined.columns)} columns")
    
            return df_combined
    
        def preprocess(self, df, dropna_thresh=0.5, drop_duplicates=True):
            """
            基本的な前処理
    
            Parameters
            ----------
            df : pd.DataFrame
                入力データフレーム
            dropna_thresh : float
                欠損値がこの割合以上の列を削除（0-1）
            drop_duplicates : bool
                重複行を削除するか
    
            Returns
            -------
            df_clean : pd.DataFrame
                クリーニング済みデータ
            """
            df_clean = df.copy()
    
            # 元のサイズ
            n_rows_orig, n_cols_orig = df_clean.shape
    
            # 1. 欠損値が多い列の削除
            thresh = int(len(df_clean) * dropna_thresh)
            df_clean = df_clean.dropna(thresh=thresh, axis=1)
    
            # 2. 完全に欠損している行の削除
            df_clean = df_clean.dropna(how='all', axis=0)
    
            # 3. 重複行の削除
            if drop_duplicates:
                df_clean = df_clean.drop_duplicates()
    
            # 4. データ型の自動推定
            df_clean = df_clean.infer_objects()
    
            # 5. 数値カラムの異常値検出（簡易版：±5σ）
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                lower = mean - 5 * std
                upper = mean + 5 * std
                outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
                if outliers.sum() > 0:
                    print(f"  {col}: {outliers.sum()} outliers detected (outside ±5σ)")
                    # 異常値をNaNに置換（オプション）
                    # df_clean.loc[outliers, col] = np.nan
    
            n_rows_clean, n_cols_clean = df_clean.shape
    
            print(f"\nPreprocessing summary:")
            print(f"  Rows: {n_rows_orig} → {n_rows_clean} ({n_rows_orig - n_rows_clean} removed)")
            print(f"  Columns: {n_cols_orig} → {n_cols_clean} ({n_cols_orig - n_cols_clean} removed)")
    
            return df_clean
    
    # 使用例
    if __name__ == "__main__":
        # サンプルデータの生成（実際にはファイルから読み込む）
        import os
        os.makedirs('./process_data', exist_ok=True)
    
        # サンプルCSVファイルを作成
        for i in range(3):
            df_sample = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01', periods=100, freq='h'),
                'temperature': np.random.normal(400, 10, 100),
                'pressure': np.random.normal(0.5, 0.05, 100),
                'power': np.random.normal(300, 20, 100),
                'thickness': np.random.normal(100, 5, 100)
            })
            df_sample.to_csv(f'./process_data/run_{i+1}.csv', index=False)
    
        # ローダーの使用
        loader = ProcessDataLoader(data_dir='./process_data')
    
        # バッチ読み込み
        df = loader.load_batch(pattern='run_*', file_extension='.csv')
    
        if df is not None:
            # 前処理
            df_clean = loader.preprocess(df, dropna_thresh=0.5, drop_duplicates=True)
    
            # 統計サマリー
            print("\nData summary:")
            print(df_clean.describe())
    

## 5.2 統計的プロセス制御（SPC: Statistical Process Control）

### 5.2.1 SPCの基礎

SPCは、プロセスの変動を統計的に監視し、異常を早期発見する手法です。

**主要な管理図（Control Charts）** ：

  * **X-bar チャート** ：サンプル平均の推移を監視
  * **R チャート** ：サンプル範囲（Range）の推移を監視
  * **S チャート** ：サンプル標準偏差の推移を監視
  * **I-MR チャート** ：個別値と移動範囲（Individual & Moving Range）

**管理限界（Control Limits）** ：

$$ \text{UCL} = \bar{X} + 3\sigma, \quad \text{LCL} = \bar{X} - 3\sigma $$ 

  * UCL: Upper Control Limit（上方管理限界）
  * LCL: Lower Control Limit（下方管理限界）
  * $\bar{X}$: プロセス平均
  * $\sigma$: プロセス標準偏差

### 5.2.2 プロセス能力指数（Cp/Cpk）

プロセスが規格を満たす能力を評価する指標です。

**Cp（Process Capability）** ：

$$ C_p = \frac{\text{USL} - \text{LSL}}{6\sigma} $$ 

  * USL: Upper Specification Limit（上限規格）
  * LSL: Lower Specification Limit（下限規格）

**Cpk（Process Capability Index）** ：

$$ C_{pk} = \min\left(\frac{\text{USL} - \mu}{3\sigma}, \frac{\mu - \text{LSL}}{3\sigma}\right) $$ 

  * $\mu$: プロセス平均

**評価基準** ：

  * Cpk ≥ 1.33: 優秀（不良率 <64 ppm）
  * Cpk ≥ 1.00: 十分（不良率 <2700 ppm）
  * Cpk < 1.00: 不十分（プロセス改善必要）

#### コード例5-2: SPCチャート生成（X-bar, R-chart, Cp/Cpk）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    
    class SPCAnalyzer:
        """
        統計的プロセス制御（SPC）解析クラス
        """
    
        def __init__(self, data, sample_size=5):
            """
            Parameters
            ----------
            data : array-like
                プロセスデータ（時系列）
            sample_size : int
                サンプルサイズ（サブグループサイズ）
            """
            self.data = np.array(data)
            self.sample_size = sample_size
            self.n_samples = len(data) // sample_size
    
            # サブグループに分割
            self.samples = self.data[:self.n_samples * sample_size].reshape(-1, sample_size)
    
        def calculate_xbar_r(self):
            """
            X-bar チャートとRチャートの統計量を計算
    
            Returns
            -------
            stats_dict : dict
                統計量（xbar, R, UCL, LCL）
            """
            # サンプル平均とサンプル範囲
            xbar = np.mean(self.samples, axis=1)
            R = np.ptp(self.samples, axis=1)  # Range (max - min)
    
            # 全体平均と平均範囲
            xbar_mean = np.mean(xbar)
            R_mean = np.mean(R)
    
            # 管理図定数（n=5の場合）
            # A2, D3, D4は統計表から取得（JIS Z 9020-2）
            control_constants = {
                2: {'A2': 1.880, 'D3': 0, 'D4': 3.267},
                3: {'A2': 1.023, 'D3': 0, 'D4': 2.574},
                4: {'A2': 0.729, 'D3': 0, 'D4': 2.282},
                5: {'A2': 0.577, 'D3': 0, 'D4': 2.114},
                6: {'A2': 0.483, 'D3': 0, 'D4': 2.004},
                7: {'A2': 0.419, 'D3': 0.076, 'D4': 1.924},
                8: {'A2': 0.373, 'D3': 0.136, 'D4': 1.864},
                9: {'A2': 0.337, 'D3': 0.184, 'D4': 1.816},
                10: {'A2': 0.308, 'D3': 0.223, 'D4': 1.777}
            }
    
            if self.sample_size not in control_constants:
                raise ValueError(f"Sample size {self.sample_size} not supported (use 2-10)")
    
            consts = control_constants[self.sample_size]
    
            # X-barチャートの管理限界
            xbar_UCL = xbar_mean + consts['A2'] * R_mean
            xbar_LCL = xbar_mean - consts['A2'] * R_mean
    
            # Rチャートの管理限界
            R_UCL = consts['D4'] * R_mean
            R_LCL = consts['D3'] * R_mean
    
            return {
                'xbar': xbar,
                'xbar_mean': xbar_mean,
                'xbar_UCL': xbar_UCL,
                'xbar_LCL': xbar_LCL,
                'R': R,
                'R_mean': R_mean,
                'R_UCL': R_UCL,
                'R_LCL': R_LCL
            }
    
        def calculate_cp_cpk(self, USL, LSL):
            """
            プロセス能力指数（Cp, Cpk）を計算
    
            Parameters
            ----------
            USL : float
                上限規格
            LSL : float
                下限規格
    
            Returns
            -------
            cp_cpk : dict
                {'Cp': float, 'Cpk': float, 'ppm': float}
            """
            mu = np.mean(self.data)
            sigma = np.std(self.data, ddof=1)  # 標本標準偏差
    
            # Cp
            Cp = (USL - LSL) / (6 * sigma)
    
            # Cpk
            Cpk_upper = (USL - mu) / (3 * sigma)
            Cpk_lower = (mu - LSL) / (3 * sigma)
            Cpk = min(Cpk_upper, Cpk_lower)
    
            # 不良率の推定（ppm: parts per million）
            # 正規分布を仮定
            z_USL = (USL - mu) / sigma
            z_LSL = (LSL - mu) / sigma
    
            ppm_upper = (1 - stats.norm.cdf(z_USL)) * 1e6
            ppm_lower = stats.norm.cdf(z_LSL) * 1e6
            ppm_total = ppm_upper + ppm_lower
    
            return {
                'Cp': Cp,
                'Cpk': Cpk,
                'ppm': ppm_total,
                'sigma': sigma,
                'mu': mu
            }
    
        def plot_control_charts(self, USL=None, LSL=None):
            """
            管理図のプロット
    
            Parameters
            ----------
            USL, LSL : float, optional
                規格限界（Cp/Cpk計算用）
            """
            stats_dict = self.calculate_xbar_r()
    
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
            sample_indices = np.arange(1, len(stats_dict['xbar']) + 1)
    
            # X-barチャート
            ax1.plot(sample_indices, stats_dict['xbar'], 'bo-', linewidth=2, markersize=6,
                    label='Sample Mean')
            ax1.axhline(stats_dict['xbar_mean'], color='green', linestyle='-', linewidth=2,
                       label=f"Center Line: {stats_dict['xbar_mean']:.2f}")
            ax1.axhline(stats_dict['xbar_UCL'], color='red', linestyle='--', linewidth=2,
                       label=f"UCL: {stats_dict['xbar_UCL']:.2f}")
            ax1.axhline(stats_dict['xbar_LCL'], color='red', linestyle='--', linewidth=2,
                       label=f"LCL: {stats_dict['xbar_LCL']:.2f}")
    
            # 管理限界外の点をハイライト
            out_of_control = (stats_dict['xbar'] > stats_dict['xbar_UCL']) | \
                             (stats_dict['xbar'] < stats_dict['xbar_LCL'])
            if out_of_control.any():
                ax1.scatter(sample_indices[out_of_control], stats_dict['xbar'][out_of_control],
                           color='red', s=150, marker='x', linewidths=3, zorder=5,
                           label='Out of Control')
    
            ax1.set_xlabel('Sample Number', fontsize=12)
            ax1.set_ylabel('Sample Mean', fontsize=12)
            ax1.set_title('X-bar Control Chart', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3)
    
            # Rチャート
            ax2.plot(sample_indices, stats_dict['R'], 'go-', linewidth=2, markersize=6,
                    label='Sample Range')
            ax2.axhline(stats_dict['R_mean'], color='blue', linestyle='-', linewidth=2,
                       label=f"Center Line: {stats_dict['R_mean']:.2f}")
            ax2.axhline(stats_dict['R_UCL'], color='red', linestyle='--', linewidth=2,
                       label=f"UCL: {stats_dict['R_UCL']:.2f}")
            ax2.axhline(stats_dict['R_LCL'], color='red', linestyle='--', linewidth=2,
                       label=f"LCL: {stats_dict['R_LCL']:.2f}")
    
            ax2.set_xlabel('Sample Number', fontsize=12)
            ax2.set_ylabel('Sample Range', fontsize=12)
            ax2.set_title('R Control Chart', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.3)
    
            # ヒストグラムとプロセス能力
            ax3.hist(self.data, bins=30, alpha=0.7, color='skyblue', edgecolor='black',
                    density=True, label='Data Distribution')
    
            # 正規分布フィット
            mu = np.mean(self.data)
            sigma = np.std(self.data, ddof=1)
            x_range = np.linspace(self.data.min(), self.data.max(), 200)
            ax3.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2,
                    label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')
    
            # 規格限界
            if USL is not None and LSL is not None:
                ax3.axvline(USL, color='red', linestyle='--', linewidth=2, label=f'USL: {USL}')
                ax3.axvline(LSL, color='red', linestyle='--', linewidth=2, label=f'LSL: {LSL}')
    
                # Cp/Cpk計算
                cp_cpk = self.calculate_cp_cpk(USL, LSL)
                textstr = f"Cp = {cp_cpk['Cp']:.2f}\nCpk = {cp_cpk['Cpk']:.2f}\nDefect Rate ≈ {cp_cpk['ppm']:.1f} ppm"
                ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
            ax3.set_xlabel('Value', fontsize=12)
            ax3.set_ylabel('Density', fontsize=12)
            ax3.set_title('Process Distribution & Capability', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
    # 実行例
    if __name__ == "__main__":
        # サンプルデータ生成（プロセスデータシミュレーション）
        np.random.seed(42)
    
        # 正常プロセス
        data_normal = np.random.normal(100, 2, 100)
    
        # 異常が混入（90-110番目でシフト）
        data_shift = np.random.normal(105, 2, 20)
        data = np.concatenate([data_normal[:90], data_shift, data_normal[90:]])
    
        # SPC解析
        spc = SPCAnalyzer(data, sample_size=5)
    
        # 管理図プロット（規格: 95-105）
        spc.plot_control_charts(USL=105, LSL=95)
    
        print("\nSPC Analysis Summary:")
        print(f"Total samples: {len(data)}")
        print(f"Subgroups: {spc.n_samples}")
        cp_cpk = spc.calculate_cp_cpk(USL=105, LSL=95)
        print(f"Cp = {cp_cpk['Cp']:.3f}")
        print(f"Cpk = {cp_cpk['Cpk']:.3f}")
        print(f"Estimated defect rate: {cp_cpk['ppm']:.1f} ppm")
    
        if cp_cpk['Cpk'] >= 1.33:
            print("Process capability: Excellent")
        elif cp_cpk['Cpk'] >= 1.00:
            print("Process capability: Adequate")
        else:
            print("Process capability: Poor - Improvement needed")
    

## 5.3 実験計画法（DOE: Design of Experiments）

### 5.3.1 DOEの基礎

DOEは、少ない実験回数で多くのパラメータの影響を効率的に調査する手法です。

**主要な実験計画** ：

  * **全因子実験** ：全組み合わせを実験（2k実験）
  * **部分実施計画** ：主効果のみ評価（2k-p実験）
  * **直交表** ：タグチメソッド（L8, L16, L27など）
  * **中心複合計画（CCD）** ：応答曲面法（RSM）用

### 5.3.2 応答曲面法（RSM: Response Surface Methodology）

RSMは、応答変数（目的関数）と説明変数（パラメータ）の関係を2次多項式でモデル化します。

$$ y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i=1}^{k} \beta_{ii} x_i^2 + \sum_{i
* $y$: 応答変数（膜厚、応力、品質スコアなど）
* $x_i$: 説明変数（温度、圧力、パワーなど）
* $\beta$: 回帰係数

#### コード例5-3: 実験計画法（2因子全因子実験+RSM）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from itertools import product
    
    def full_factorial_design(factors, levels):
        """
        全因子実験の実験計画を生成
    
        Parameters
        ----------
        factors : dict
            {'factor_name': [level1, level2, ...]}
        levels : int
            各因子の水準数（2水準、3水準など）
    
        Returns
        -------
        design : pd.DataFrame
            実験計画表
        """
        factor_names = list(factors.keys())
        factor_values = [factors[name] for name in factor_names]
    
        # 全組み合わせ生成
        combinations = list(product(*factor_values))
    
        design = pd.DataFrame(combinations, columns=factor_names)
    
        return design
    
    def response_surface_model(X, y, degree=2):
        """
        応答曲面モデル（多項式回帰）
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            説明変数
        y : array-like, shape (n_samples,)
            応答変数
        degree : int
            多項式次数（通常2）
    
        Returns
        -------
        model : sklearn model
            フィット済みモデル
        poly : PolynomialFeatures
            多項式変換器
        """
        # 多項式特徴量生成
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(X)
    
        # 線形回帰
        model = LinearRegression()
        model.fit(X_poly, y)
    
        print(f"R² score: {model.score(X_poly, y):.3f}")
    
        return model, poly
    
    # 実験計画の設定
    factors = {
        'Temperature': [300, 350, 400, 450, 500],  # [°C]
        'Pressure': [0.2, 0.35, 0.5, 0.65, 0.8]     # [Pa]
    }
    
    design = full_factorial_design(factors, levels=5)
    print("Experimental Design (Full Factorial):")
    print(design.head(10))
    print(f"Total experiments: {len(design)}")
    
    # 応答変数のシミュレーション（実際には実験で測定）
    # 真のモデル: y = 100 + 0.2*T + 50*P - 0.0002*T^2 - 50*P^2 + 0.05*T*P
    def true_response(T, P):
        """真の応答関数（未知として扱う）"""
        y = 100 + 0.2*T + 50*P - 0.0002*T**2 - 50*P**2 + 0.05*T*P
        # ノイズ追加
        y += np.random.normal(0, 2, len(T))
        return y
    
    design['Response'] = true_response(design['Temperature'], design['Pressure'])
    
    # データ準備
    X = design[['Temperature', 'Pressure']].values
    y = design['Response'].values
    
    # 応答曲面モデルのフィッティング
    model, poly = response_surface_model(X, y, degree=2)
    
    # 予測グリッド生成
    T_range = np.linspace(300, 500, 50)
    P_range = np.linspace(0.2, 0.8, 50)
    T_grid, P_grid = np.meshgrid(T_range, P_range)
    
    X_grid = np.c_[T_grid.ravel(), P_grid.ravel()]
    X_grid_poly = poly.transform(X_grid)
    y_pred_grid = model.predict(X_grid_poly).reshape(T_grid.shape)
    
    # 可視化
    fig = plt.figure(figsize=(16, 6))
    
    # 左図: 3D応答曲面
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax1.plot_surface(T_grid, P_grid, y_pred_grid, cmap='viridis',
                            alpha=0.8, edgecolor='none')
    ax1.scatter(X[:, 0], X[:, 1], y, color='red', s=50, marker='o',
               edgecolors='black', linewidths=1.5, label='Experimental Data')
    ax1.set_xlabel('Temperature [°C]', fontsize=11)
    ax1.set_ylabel('Pressure [Pa]', fontsize=11)
    ax1.set_zlabel('Response', fontsize=11)
    ax1.set_title('Response Surface (3D)', fontsize=13, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    
    # 中央図: 等高線図
    ax2 = fig.add_subplot(1, 3, 2)
    contour = ax2.contourf(T_grid, P_grid, y_pred_grid, levels=20, cmap='viridis', alpha=0.8)
    contour_lines = ax2.contour(T_grid, P_grid, y_pred_grid, levels=10,
                                 colors='white', linewidths=1, alpha=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8)
    ax2.scatter(X[:, 0], X[:, 1], color='red', s=50, marker='o',
               edgecolors='black', linewidths=1.5, label='Exp. Points')
    
    # 最適点
    optimal_idx = np.argmax(y_pred_grid)
    T_opt = T_grid.ravel()[optimal_idx]
    P_opt = P_grid.ravel()[optimal_idx]
    y_opt = y_pred_grid.ravel()[optimal_idx]
    
    ax2.scatter(T_opt, P_opt, color='yellow', s=300, marker='*',
               edgecolors='black', linewidths=2, label=f'Optimum: {y_opt:.1f}', zorder=5)
    
    ax2.set_xlabel('Temperature [°C]', fontsize=12)
    ax2.set_ylabel('Pressure [Pa]', fontsize=12)
    ax2.set_title('Response Surface (Contour)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    fig.colorbar(contour, ax=ax2, label='Response')
    
    # 右図: 主効果プロット
    ax3 = fig.add_subplot(1, 3, 3)
    
    # 温度の主効果（圧力を中央値に固定）
    P_center = np.median(factors['Pressure'])
    T_effect = np.linspace(300, 500, 50)
    X_effect_T = np.c_[T_effect, np.full(50, P_center)]
    X_effect_T_poly = poly.transform(X_effect_T)
    y_effect_T = model.predict(X_effect_T_poly)
    
    ax3.plot(T_effect, y_effect_T, 'b-', linewidth=2, label=f'Temperature (P={P_center} Pa)')
    
    # 圧力の主効果（温度を中央値に固定）
    T_center = np.median(factors['Temperature'])
    P_effect = np.linspace(0.2, 0.8, 50)
    X_effect_P = np.c_[np.full(50, T_center), P_effect]
    X_effect_P_poly = poly.transform(X_effect_P)
    y_effect_P = model.predict(X_effect_P_poly)
    
    # 右軸
    ax3_twin = ax3.twinx()
    ax3_twin.plot(P_effect*500, y_effect_P, 'r-', linewidth=2,
                 label=f'Pressure (T={T_center}°C)')
    
    ax3.set_xlabel('Temperature [°C]', fontsize=12, color='blue')
    ax3_twin.set_xlabel('Pressure [Pa] (scaled ×500)', fontsize=12, color='red')
    ax3.set_ylabel('Response (Temperature effect)', fontsize=12, color='blue')
    ax3_twin.set_ylabel('Response (Pressure effect)', fontsize=12, color='red')
    ax3.set_title('Main Effects Plot', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', labelcolor='blue')
    ax3_twin.tick_params(axis='x', labelcolor='red')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nOptimal Conditions:")
    print(f"  Temperature: {T_opt:.1f} °C")
    print(f"  Pressure: {P_opt:.3f} Pa")
    print(f"  Predicted Response: {y_opt:.2f}")
    
    # 回帰係数の表示
    coef_names = poly.get_feature_names_out(['T', 'P'])
    print(f"\nRegression Coefficients:")
    for name, coef in zip(coef_names, [model.intercept_] + list(model.coef_[1:])):
        print(f"  {name}: {coef:.4f}")
    

## 5.4 機械学習によるプロセス予測

### 5.4.1 回帰モデルによる品質予測

機械学習を用いて、プロセスパラメータから製品品質を予測します。

**主要なアルゴリズム** ：

  * **ランダムフォレスト回帰** ：高精度、解釈性（特徴量重要度）
  * **勾配ブースティング（XGBoost, LightGBM）** ：最高精度
  * **ニューラルネットワーク** ：非線形パターン学習
  * **サポートベクター回帰（SVR）** ：小データセット向け

#### コード例5-4: ランダムフォレストによるプロセス品質予測
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import seaborn as sns
    
    # サンプルデータセット生成（実際には実験データを使用）
    np.random.seed(42)
    n_samples = 200
    
    # プロセスパラメータ
    data = pd.DataFrame({
        'Temperature': np.random.uniform(300, 500, n_samples),
        'Pressure': np.random.uniform(0.2, 0.8, n_samples),
        'Power': np.random.uniform(100, 400, n_samples),
        'Flow_Rate': np.random.uniform(50, 150, n_samples),
        'Time': np.random.uniform(30, 120, n_samples)
    })
    
    # 目的変数（膜厚）のシミュレーション
    # 真のモデル: 複雑な非線形関係
    data['Thickness'] = (
        0.5 * data['Temperature'] +
        100 * data['Pressure'] +
        0.3 * data['Power'] +
        0.2 * data['Flow_Rate'] +
        1.0 * data['Time'] +
        0.001 * data['Temperature'] * data['Pressure'] -
        0.0005 * data['Temperature']**2 +
        np.random.normal(0, 10, n_samples)  # ノイズ
    )
    
    print("Dataset shape:", data.shape)
    print("\nFeature summary:")
    print(data.describe())
    
    # 特徴量と目的変数の分割
    X = data.drop('Thickness', axis=1)
    y = data['Thickness']
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ランダムフォレストモデルの訓練
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # 予測
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # 評価指標
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("\n" + "="*60)
    print("Model Performance:")
    print("="*60)
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    # 交差検証
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    print(f"\n5-Fold CV R² score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("="*60)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 左上: 予測vs実測（訓練データ）
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                   'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Thickness', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Thickness', fontsize=12)
    axes[0, 0].set_title(f'Training Set\nR² = {train_r2:.3f}, RMSE = {train_rmse:.2f}',
                        fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # 右上: 予測vs実測（テストデータ）
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, s=30, color='green',
                      edgecolors='black', linewidth=0.5)
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('Actual Thickness', fontsize=12)
    axes[0, 1].set_ylabel('Predicted Thickness', fontsize=12)
    axes[0, 1].set_title(f'Test Set\nR² = {test_r2:.3f}, RMSE = {test_rmse:.2f}',
                        fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # 左下: 特徴量重要度
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    bars = axes[1, 0].barh(feature_importance['Feature'], feature_importance['Importance'],
                           color='skyblue', edgecolor='black')
    axes[1, 0].set_xlabel('Importance', fontsize=12)
    axes[1, 0].set_title('Feature Importance', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # 値をバーに表示
    for bar, importance in zip(bars, feature_importance['Importance']):
        axes[1, 0].text(importance + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{importance:.3f}', va='center', fontsize=10)
    
    # 右下: 残差プロット
    residuals = y_test - y_test_pred
    axes[1, 1].scatter(y_test_pred, residuals, alpha=0.6, s=30, color='orange',
                      edgecolors='black', linewidth=0.5)
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Thickness', fontsize=12)
    axes[1, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[1, 1].set_title('Residual Plot (Test Set)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nFeature Importance Ranking:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    print("\nInterpretation:")
    print("  - Time has the highest importance (longer deposition → thicker film)")
    print("  - Temperature and Pressure also significant")
    print("  - Model can predict thickness with ~±10 nm accuracy")
    

### 5.4.2 分類モデルによる不良品検出

プロセスパラメータから不良品を事前に予測します。

#### コード例5-5: ロジスティック回帰による不良品予測
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    import seaborn as sns
    
    # サンプルデータセット生成
    np.random.seed(42)
    n_samples = 300
    
    # 良品（70%）と不良品（30%）
    data = pd.DataFrame({
        'Temperature': np.random.normal(400, 30, n_samples),
        'Pressure': np.random.normal(0.5, 0.1, n_samples),
        'Power': np.random.normal(250, 50, n_samples)
    })
    
    # 不良品ラベル生成（規格外条件で不良率が上がる）
    # 良品条件: 380 380) & (data['Temperature'] < 420) &
        (data['Pressure'] > 0.4) & (data['Pressure'] < 0.6) &
        (data['Power'] > 200) & (data['Power'] < 300)
    )
    
    # 条件外でも確率的に良品になることがある
    data['Defect'] = 0
    data.loc[~good_condition, 'Defect'] = np.random.choice([0, 1], size=(~good_condition).sum(),
                                                            p=[0.3, 0.7])
    
    print(f"Dataset: {len(data)} samples")
    print(f"Defect rate: {data['Defect'].mean()*100:.1f}%")
    
    # 特徴量と目的変数
    X = data[['Temperature', 'Pressure', 'Power']]
    y = data['Defect']
    
    # 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    
    # ロジスティック回帰モデル
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # 予測
    y_test_pred = lr_model.predict(X_test)
    y_test_prob = lr_model.predict_proba(X_test)[:, 1]
    
    # 評価
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(y_test, y_test_pred, target_names=['Good', 'Defect']))
    
    auc_score = roc_auc_score(y_test, y_test_prob)
    print(f"ROC-AUC Score: {auc_score:.3f}")
    print("="*60)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 左上: 混同行列
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 0],
               xticklabels=['Good', 'Defect'], yticklabels=['Good', 'Defect'])
    axes[0, 0].set_xlabel('Predicted', fontsize=12)
    axes[0, 0].set_ylabel('Actual', fontsize=12)
    axes[0, 0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    # 右上: ROC曲線
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title('ROC Curve', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(alpha=0.3)
    
    # 左下: 特徴量係数
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    bars = axes[1, 0].barh(coef_df['Feature'], coef_df['Coefficient'],
                           color=['red' if c > 0 else 'blue' for c in coef_df['Coefficient']],
                           edgecolor='black')
    axes[1, 0].axvline(0, color='black', linewidth=1)
    axes[1, 0].set_xlabel('Coefficient (Defect Risk)', fontsize=12)
    axes[1, 0].set_title('Feature Coefficients\n(Positive = Increases Defect Risk)',
                        fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # 右下: 確率分布
    axes[1, 1].hist(y_test_prob[y_test == 0], bins=20, alpha=0.6, label='Good',
                   color='green', edgecolor='black')
    axes[1, 1].hist(y_test_prob[y_test == 1], bins=20, alpha=0.6, label='Defect',
                   color='red', edgecolor='black')
    axes[1, 1].axvline(0.5, color='black', linestyle='--', linewidth=2,
                      label='Decision Threshold')
    axes[1, 1].set_xlabel('Predicted Probability (Defect)', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Predicted Probability Distribution', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nModel Interpretation:")
    for idx, row in coef_df.iterrows():
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"  {row['Feature']}: {row['Coefficient']:+.4f} → {direction} defect risk")
    

## 5.5 異常検知（Anomaly Detection）

### 5.5.1 Isolation Forestによる異常検知

Isolation Forestは、教師なし学習で異常データを検出します。正常データのパターンから逸脱したデータを「異常」として識別します。

#### コード例5-6: Isolation Forestによる異常プロセス検出
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # サンプルデータ生成
    np.random.seed(42)
    
    # 正常データ（200サンプル）
    normal_data = pd.DataFrame({
        'Temperature': np.random.normal(400, 10, 200),
        'Pressure': np.random.normal(0.5, 0.05, 200),
        'Power': np.random.normal(250, 20, 200),
        'Thickness': np.random.normal(100, 5, 200)
    })
    
    # 異常データ（20サンプル）
    anomaly_data = pd.DataFrame({
        'Temperature': np.random.uniform(350, 450, 20),
        'Pressure': np.random.uniform(0.3, 0.7, 20),
        'Power': np.random.uniform(150, 350, 20),
        'Thickness': np.random.uniform(70, 130, 20)
    })
    
    # データ統合
    data = pd.concat([normal_data, anomaly_data], ignore_index=True)
    true_labels = np.array([0]*len(normal_data) + [1]*len(anomaly_data))  # 0: normal, 1: anomaly
    
    print(f"Total samples: {len(data)}")
    print(f"Anomaly rate: {(true_labels == 1).mean()*100:.1f}%")
    
    # 特徴量の標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Isolation Forestモデル
    iso_forest = IsolationForest(
        contamination=0.1,  # 想定異常率
        random_state=42,
        n_estimators=100
    )
    
    # 予測（-1: 異常, 1: 正常）
    predictions = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.score_samples(X_scaled)
    
    # 予測ラベルを0/1に変換
    pred_labels = (predictions == -1).astype(int)
    
    # 評価（真のラベルがある場合）
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + "="*60)
    print("Anomaly Detection Results:")
    print("="*60)
    print(classification_report(true_labels, pred_labels,
                              target_names=['Normal', 'Anomaly']))
    print("="*60)
    
    # PCAで2次元に圧縮（可視化用）
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 左上: PCA空間での異常検知結果
    scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=pred_labels,
                                 cmap='RdYlGn_r', s=80, alpha=0.7,
                                 edgecolors='black', linewidth=1)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                         fontsize=11)
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
                         fontsize=11)
    axes[0, 0].set_title('Anomaly Detection (Predicted)', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('0: Normal, 1: Anomaly', fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # 右上: 真のラベル（比較用）
    scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels,
                                 cmap='RdYlGn_r', s=80, alpha=0.7,
                                 edgecolors='black', linewidth=1)
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                         fontsize=11)
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
                         fontsize=11)
    axes[0, 1].set_title('Ground Truth Labels', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('0: Normal, 1: Anomaly', fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # 左下: 異常スコア分布
    axes[1, 0].hist(anomaly_scores[true_labels == 0], bins=30, alpha=0.6,
                   label='Normal', color='green', edgecolor='black')
    axes[1, 0].hist(anomaly_scores[true_labels == 1], bins=30, alpha=0.6,
                   label='Anomaly', color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Anomaly Score (lower = more anomalous)', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    
    # 右下: 混同行列
    cm = confusion_matrix(true_labels, pred_labels)
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1, 1],
               xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    axes[1, 1].set_xlabel('Predicted', fontsize=12)
    axes[1, 1].set_ylabel('Actual', fontsize=12)
    axes[1, 1].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 最も異常なサンプルをリスト
    top_anomalies = np.argsort(anomaly_scores)[:5]
    print("\nTop 5 Most Anomalous Samples:")
    print(data.iloc[top_anomalies])
    print("\nAnomaly Scores:")
    print(anomaly_scores[top_anomalies])
    

## 5.6 自動レポート生成

### 5.6.1 日次/週次プロセスレポートの自動化

解析結果を自動的にPDFレポートやHTMLダッシュボードに出力します。

#### コード例5-7: 完全統合ワークフロー（データ → 解析 → レポート）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    from matplotlib.backends.backend_pdf import PdfPages
    import warnings
    warnings.filterwarnings('ignore')
    
    class ProcessReportGenerator:
        """
        プロセス解析の自動レポート生成クラス
        """
    
        def __init__(self, data, report_title="Process Analysis Report"):
            """
            Parameters
            ----------
            data : pd.DataFrame
                プロセスデータ
            report_title : str
                レポートタイトル
            """
            self.data = data
            self.report_title = report_title
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        def generate_summary_statistics(self):
            """統計サマリーの生成"""
            summary = self.data.describe().T
            summary['missing'] = self.data.isnull().sum()
            summary['missing_pct'] = (summary['missing'] / len(self.data) * 100).round(2)
    
            return summary
    
        def plot_time_series(self, ax, column):
            """時系列プロット"""
            if 'timestamp' in self.data.columns:
                x = self.data['timestamp']
            else:
                x = np.arange(len(self.data))
    
            ax.plot(x, self.data[column], 'b-', linewidth=1.5, alpha=0.7)
    
            # 管理限界（±3σ）
            mean = self.data[column].mean()
            std = self.data[column].std()
            ucl = mean + 3*std
            lcl = mean - 3*std
    
            ax.axhline(mean, color='green', linestyle='-', linewidth=2, label='Mean')
            ax.axhline(ucl, color='red', linestyle='--', linewidth=2, label='UCL (±3σ)')
            ax.axhline(lcl, color='red', linestyle='--', linewidth=2, label='LCL')
    
            # 管理限界外の点をハイライト
            out_of_control = (self.data[column] > ucl) | (self.data[column] < lcl)
            if out_of_control.any():
                ax.scatter(np.where(out_of_control)[0], self.data.loc[out_of_control, column],
                          color='red', s=100, marker='x', linewidths=3, zorder=5,
                          label='Out of Control')
    
            ax.set_xlabel('Sample Index', fontsize=10)
            ax.set_ylabel(column, fontsize=10)
            ax.set_title(f'Time Series: {column}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
    
        def plot_distribution(self, ax, column, bins=30):
            """分布プロット"""
            ax.hist(self.data[column], bins=bins, alpha=0.7, color='skyblue',
                   edgecolor='black', density=True)
    
            # 正規分布フィット
            from scipy import stats
            mu = self.data[column].mean()
            sigma = self.data[column].std()
            x_range = np.linspace(self.data[column].min(), self.data[column].max(), 100)
            ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2,
                   label=f'Normal Fit\nμ={mu:.2f}, σ={sigma:.2f}')
    
            ax.set_xlabel(column, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Distribution: {column}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
    
        def plot_correlation_matrix(self, ax):
            """相関行列ヒートマップ"""
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            corr = self.data[numeric_cols].corr()
    
            import seaborn as sns
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
        def generate_pdf_report(self, filename='process_report.pdf'):
            """
            PDFレポートの生成
    
            Parameters
            ----------
            filename : str
                出力PDFファイル名
            """
            with PdfPages(filename) as pdf:
                # ページ1: タイトルと統計サマリー
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle(self.report_title, fontsize=18, fontweight='bold', y=0.98)
    
                # タイムスタンプ
                fig.text(0.5, 0.94, f'Generated: {self.timestamp}', ha='center',
                        fontsize=10, style='italic')
    
                # 統計サマリーテーブル
                ax_table = fig.add_subplot(111)
                ax_table.axis('off')
    
                summary = self.generate_summary_statistics()
                summary_display = summary[['mean', 'std', 'min', 'max', 'missing_pct']]
                summary_display.columns = ['Mean', 'Std', 'Min', 'Max', 'Missing%']
    
                table = ax_table.table(cellText=summary_display.round(2).values,
                                      rowLabels=summary_display.index,
                                      colLabels=summary_display.columns,
                                      cellLoc='center', rowLoc='center',
                                      loc='center', bbox=[0.1, 0.3, 0.8, 0.6])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
    
                # ヘッダー行の装飾
                for i in range(len(summary_display.columns)):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')
    
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
                # ページ2-N: 各変数の時系列と分布
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
    
                for col in numeric_cols:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
    
                    self.plot_time_series(ax1, col)
                    self.plot_distribution(ax2, col)
    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
    
                # 最終ページ: 相関行列
                fig, ax = plt.subplots(figsize=(11, 8.5))
                self.plot_correlation_matrix(ax)
    
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
            print(f"PDF report generated: {filename}")
    
    # 使用例
    if __name__ == "__main__":
        # サンプルデータ生成
        np.random.seed(42)
        n_samples = 100
    
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='h'),
            'Temperature': np.random.normal(400, 10, n_samples),
            'Pressure': np.random.normal(0.5, 0.05, n_samples),
            'Power': np.random.normal(250, 20, n_samples),
            'Thickness': np.random.normal(100, 5, n_samples),
            'Uniformity': np.random.normal(95, 2, n_samples)
        })
    
        # 一部に異常を挿入
        data.loc[50:55, 'Temperature'] = np.random.normal(430, 5, 6)
        data.loc[50:55, 'Thickness'] = np.random.normal(110, 3, 6)
    
        # レポート生成
        report_gen = ProcessReportGenerator(data, report_title="Weekly Process Analysis Report")
    
        # 統計サマリー表示
        print("Statistical Summary:")
        print(report_gen.generate_summary_statistics())
    
        # PDFレポート生成
        report_gen.generate_pdf_report('process_weekly_report.pdf')
    
        print("\nReport generation complete!")
        print("  - PDF: process_weekly_report.pdf")
        print("  - Contains: Time series, distributions, correlation matrix")
    
    
    
    ```mermaid
    flowchart TD
        A[Raw Process DataCSV/Excel/JSON] --> B[Data LoadingProcessDataLoader]
        B --> C[PreprocessingClean & Standardize]
        C --> D[SPC AnalysisControl Charts]
        C --> E[DOE/RSMOptimization]
        C --> F[ML PredictionQuality Forecast]
        C --> G[Anomaly DetectionIsolation Forest]
        D --> H[Report GenerationPDF/HTML]
        E --> H
        F --> H
        G --> H
        H --> I[Automated ReportDaily/Weekly]
    
        style A fill:#99ccff,stroke:#0066cc
        style H fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
        style I fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

## 5.7 演習問題

### 演習5-1: Cp/Cpk計算（易）

**問題** ：膜厚データが平均100 nm、標準偏差3 nm、規格が95-105 nmのとき、CpとCpkを計算せよ。プロセスは適切か？

**解答例を表示**
    
    
    import numpy as np
    
    mu = 100  # [nm]
    sigma = 3  # [nm]
    USL = 105  # [nm]
    LSL = 95   # [nm]
    
    # Cp
    Cp = (USL - LSL) / (6 * sigma)
    
    # Cpk
    Cpk_upper = (USL - mu) / (3 * sigma)
    Cpk_lower = (mu - LSL) / (3 * sigma)
    Cpk = min(Cpk_upper, Cpk_lower)
    
    print(f"Cp = {Cp:.3f}")
    print(f"Cpk = {Cpk:.3f}")
    
    if Cpk >= 1.33:
        print("プロセス能力: 優秀")
    elif Cpk >= 1.00:
        print("プロセス能力: 十分")
    else:
        print("プロセス能力: 不十分（改善必要）")
    
    # 不良率推定
    from scipy import stats
    ppm = (stats.norm.cdf((LSL - mu) / sigma) +
           (1 - stats.norm.cdf((USL - mu) / sigma))) * 1e6
    print(f"推定不良率: {ppm:.1f} ppm")
    

### 演習5-2: 2因子実験計画（中）

**問題** ：温度（300, 400, 500°C）と圧力（0.3, 0.5, 0.7 Pa）の2因子3水準実験を設計し、全組み合わせの実験計画表を作成せよ。

**解答例を表示**
    
    
    import pandas as pd
    from itertools import product
    
    # 因子と水準
    factors = {
        'Temperature': [300, 400, 500],
        'Pressure': [0.3, 0.5, 0.7]
    }
    
    # 全組み合わせ生成
    combinations = list(product(factors['Temperature'], factors['Pressure']))
    design = pd.DataFrame(combinations, columns=['Temperature', 'Pressure'])
    
    print("Experimental Design (Full Factorial):")
    print(design)
    print(f"\nTotal experiments: {len(design)}")
    

### 演習5-3: ランダムフォレスト特徴量重要度（中）

**問題** ：5つのプロセスパラメータ（温度、圧力、パワー、流量、時間）から膜厚を予測するモデルで、最も重要な因子を特定せよ。

**解答例を表示**

**コード例5-4のランダムフォレストモデル** を実行し、特徴量重要度を確認：
    
    
    # feature_importanceから重要度を確認
    print(feature_importance)
    
    # 典型的な結果:
    #   Time: 0.35 (最重要: 成膜時間が長いほど厚い)
    #   Temperature: 0.25 (成長速度に影響)
    #   Power: 0.20 (スパッタ収率に影響)
    #   Pressure: 0.15 (ガス散乱に影響)
    #   Flow_Rate: 0.05 (間接的影響)
    

### 演習5-4: 異常検知の閾値設定（中）

**問題** ：Isolation Forestの`contamination`パラメータを0.05, 0.1, 0.2で変えた場合、検出される異常数はどう変化するか？

**解答例を表示**
    
    
    from sklearn.ensemble import IsolationForest
    import numpy as np
    
    # サンプルデータ
    data = np.random.normal(0, 1, (100, 4))
    
    contaminations = [0.05, 0.1, 0.2]
    
    for cont in contaminations:
        iso_forest = IsolationForest(contamination=cont, random_state=42)
        predictions = iso_forest.fit_predict(data)
        n_anomalies = (predictions == -1).sum()
    
        print(f"Contamination = {cont}: {n_anomalies} anomalies detected ({n_anomalies/len(data)*100:.1f}%)")
    
    # 出力例:
    # Contamination = 0.05: 5 anomalies (5.0%)
    # Contamination = 0.1: 10 anomalies (10.0%)
    # Contamination = 0.2: 20 anomalies (20.0%)
    
    print("\n解釈: contaminationは想定異常率。高いほど多く検出（偽陽性リスク増）")
    

### 演習5-5: 応答曲面法の最適化（難）

**問題** ：温度と圧力の2因子で膜厚を最大化したい。応答曲面モデルから最適条件（温度、圧力）を数値的に求めよ。

**解答例を表示**
    
    
    from scipy.optimize import minimize
    
    # コード例5-3のモデルとpolyを使用
    def objective(x):
        """最小化目標関数（最大化のため負号）"""
        X_input = np.array(x).reshape(1, -1)
        X_poly = poly.transform(X_input)
        y_pred = model.predict(X_poly)
        return -y_pred[0]  # 最大化のため負号
    
    # 初期値と境界
    x0 = [400, 0.5]  # [Temperature, Pressure]
    bounds = [(300, 500), (0.2, 0.8)]
    
    # 最適化
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    T_opt = result.x[0]
    P_opt = result.x[1]
    y_opt = -result.fun
    
    print(f"Optimal conditions:")
    print(f"  Temperature: {T_opt:.1f} °C")
    print(f"  Pressure: {P_opt:.3f} Pa")
    print(f"  Predicted maximum response: {y_opt:.2f}")
    

### 演習5-6: データ前処理の影響（難）

**問題** ：欠損値を含むデータセットで、(a)欠損行削除、(b)平均値補完、(c)KNN補完の3手法を比較し、機械学習モデルの精度への影響を評価せよ。

**解答例を表示**
    
    
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    # サンプルデータ（欠損値あり）
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'X1': np.random.normal(0, 1, n),
        'X2': np.random.normal(0, 1, n),
        'X3': np.random.normal(0, 1, n),
        'y': np.random.normal(0, 1, n)
    })
    
    # 欠損値をランダムに導入（10%）
    for col in ['X1', 'X2', 'X3']:
        missing_idx = np.random.choice(n, size=int(n*0.1), replace=False)
        data.loc[missing_idx, col] = np.nan
    
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    methods = {
        'Dropna': data.dropna(),
        'Mean Imputation': pd.DataFrame(
            SimpleImputer(strategy='mean').fit_transform(data),
            columns=data.columns
        ),
        'KNN Imputation': pd.DataFrame(
            KNNImputer(n_neighbors=5).fit_transform(data),
            columns=data.columns
        )
    }
    
    for method_name, df in methods.items():
        X = df.drop('y', axis=1)
        y = df['y']
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        r2 = r2_score(y_test, y_pred)
        print(f"{method_name}: R² = {r2:.3f}, n_samples = {len(df)}")
    
    print("\n結論: データ量とモデル精度のトレードオフを考慮して選択")
    

### 演習5-7: SPC管理限界の調整（難）

**問題** ：3σ管理限界（99.73%）の代わりに2σ（95.45%）を使う場合、偽警報率と見逃し率はどう変化するか？どちらが適切か議論せよ。

**解答例を表示**

**理論的比較** ：

  * **3σ管理限界** ： 
    * 偽警報率（α）: 0.27%（正常なのに異常と判定）
    * 見逃し率（β）: 高い（異常なのに検出漏れ）
    * 適用: 安定プロセス、調整コストが高い場合
  * **2σ管理限界** ： 
    * 偽警報率（α）: 4.55%（偽警報が増える）
    * 見逃し率（β）: 低い（早期検出可能）
    * 適用: 不安定プロセス、早期警告が重要な場合

**実践的選択** ：
    
    
    # 偽警報率の計算
    from scipy import stats
    
    alpha_3sigma = 2 * (1 - stats.norm.cdf(3))
    alpha_2sigma = 2 * (1 - stats.norm.cdf(2))
    
    print(f"3σ管理限界: 偽警報率 = {alpha_3sigma*100:.2f}%")
    print(f"2σ管理限界: 偽警報率 = {alpha_2sigma*100:.2f}%")
    
    print("\n推奨:")
    print("  - 3σ: 成熟プロセス、調整コストが高い（半導体製造など）")
    print("  - 2σ: 新規プロセス、品質重視、早期介入が必要（医薬品など）")
    

### 演習5-8: 統合ワークフロー設計（難）

**問題** ：実際の工場で、データ取得 → SPC監視 → 異常検知 → 自動アラート → 週次レポートの完全自動化システムを設計せよ。必要なコンポーネントとデータフローを示せ。

**解答例を表示**

**システムアーキテクチャ** ：

  1. **データ取得層** ： 
     * 装置からのリアルタイムログ収集（OPC UA, MQTT）
     * データベース格納（TimescaleDB, InfluxDB）
  2. **処理層** ： 
     * 定期実行（cron, Airflow）
     * SPC解析（Python + pandas）
     * 異常検知（Isolation Forest）
  3. **アラート層** ： 
     * 管理限界外検出 → メール/Slack通知
     * 異常スコア閾値超過 → 緊急アラート
  4. **レポート層** ： 
     * 週次レポート自動生成（PDF）
     * Webダッシュボード（Dash, Streamlit）

**実装例（擬似コード）** ：
    
    
    # daily_monitoring.py (cron: 毎日1回実行)
    def daily_monitoring():
        # 1. データ取得
        data = load_data_from_database(start_date=yesterday, end_date=today)
    
        # 2. SPC解析
        spc = SPCAnalyzer(data)
        out_of_control = spc.detect_out_of_control()
    
        # 3. 異常検知
        anomalies = detect_anomalies(data)
    
        # 4. アラート
        if out_of_control or anomalies:
            send_alert(subject="Process Anomaly Detected",
                      body=f"Out of control: {out_of_control}\nAnomalies: {anomalies}")
    
        # 5. ログ保存
        save_monitoring_log(data, spc_results, anomalies)
    
    # weekly_report.py (cron: 毎週月曜実行)
    def weekly_report():
        data = load_data_from_database(start_date=last_week, end_date=today)
        report_gen = ProcessReportGenerator(data)
        report_gen.generate_pdf_report('weekly_report.pdf')
        send_email(to='manager@example.com', attachment='weekly_report.pdf')
    

## 5.8 学習の確認

### 基本理解度チェック

  1. CSV, Excel, JSONなど多様なフォーマットからデータを読み込めますか？
  2. X-barチャートとRチャートの意味と使い分けを理解していますか？
  3. Cp/Cpkの違いと、プロセス能力評価の基準を説明できますか？
  4. 全因子実験と応答曲面法の違いを理解していますか？
  5. ランダムフォレストの特徴量重要度の解釈ができますか？
  6. Isolation Forestによる異常検知の原理を理解していますか？

### 実践スキル確認

  1. 実際のプロセスデータからSPCチャートを生成し、管理限界外の点を特定できますか？
  2. 2因子以上の実験計画を設計し、応答曲面モデルで最適化できますか？
  3. 機械学習モデル（回帰・分類）でプロセス品質を予測できますか？
  4. 異常検知アルゴリズムで不良品を早期発見できますか？
  5. 解析結果を自動的にPDFレポートに出力できますか？

### 応用力確認

  1. 完全統合ワークフロー（データ → 解析 → 最適化 → レポート）を設計・実装できますか？
  2. 実際の工場データで、品質問題の根本原因をデータから特定できますか？
  3. 新規プロセスの立ち上げ時に、DOEとMLを組み合わせた最適化戦略を提案できますか？

## 5.9 参考文献

  1. Montgomery, D.C. (2012). _Statistical Quality Control_ (7th ed.). Wiley. pp. 156-234 (Control charts), pp. 289-345 (Process capability).
  2. Box, G.E.P., Hunter, J.S., Hunter, W.G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley. pp. 123-189 (Factorial designs), pp. 289-345 (Response surface methods).
  3. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). _An Introduction to Statistical Learning with Applications in Python_. Springer. pp. 303-335 (Random forests), pp. 445-489 (Unsupervised learning).
  4. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." _Journal of Machine Learning Research_ , 12:2825-2830. - scikit-learn documentation.
  5. McKinney, W. (2017). _Python for Data Analysis_ (2nd ed.). O'Reilly. pp. 89-156 (pandas basics), pp. 234-289 (Data cleaning and preparation).
  6. Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). "Isolation Forest." _Proceedings of the 8th IEEE International Conference on Data Mining_ , pp. 413-422. DOI: 10.1109/ICDM.2008.17
  7. Hunter, J.D. (2007). "Matplotlib: A 2D Graphics Environment." _Computing in Science & Engineering_, 9(3):90-95. DOI: 10.1109/MCSE.2007.55
  8. Waskom, M. (2021). "seaborn: statistical data visualization." _Journal of Open Source Software_ , 6(60):3021. DOI: 10.21105/joss.03021

## 5.10 次のステップ

この章で学んだプロセスデータ解析の実践スキルは、材料科学研究と産業応用の両方で即座に活用できます。次のステップとして：

  * **実データへの適用** ：自身の研究室や工場のプロセスデータで、学んだ手法を実践
  * **発展的機械学習** ：深層学習（LSTM, Transformer）による時系列予測
  * **リアルタイム監視** ：ストリーミングデータ処理（Apache Kafka, Flink）の導入
  * **統合システム開発** ：Webダッシュボード（Dash, Streamlit）での可視化
  * **自動化の深化** ：CI/CDパイプライン構築（GitHub Actions）で完全自動運用

プロセス技術入門シリーズを完走されたあなたは、薄膜成長・プロセス制御・データ解析の全領域を統合的に理解し、実践できる力を身につけました。この知識を基盤に、さらなる専門性を深め、材料科学の最前線で活躍されることを期待します！
