---
title: 第5章：Pythonpractice：process data analysisworkflow
chapter_title: 第5章：Pythonpractice：process data analysisworkflow
subtitle: SPC, DOE, Machine Learning, Automated Reporting
reading_time: 35-45分
difficulty: medium級〜上級
code_examples: 7
---

process data analysis、material manufacturing withquality controlandoptimization withcore。In this chapter、Statistical Process Control（SPC）、Design of Experiments（DOE）、machine learningwithpredictive modelconstruction、anomaly detection、automated report generationintegratedPythonworkflowpractice and、immediately applicableskillsacquire。 

## learn目標

こ with章読むこand for、以下acquire：

  * ✅ diverseprocess dataformats（CSV, JSON, Excel, equipment-specificformat） withloadingandpreprocessing
  * ✅ SPC（Statistical Process Control）charts（X-bar, R-chart, Cp/Cpk）generate・interpret
  * ✅ Design of Experiments（DOE: Design of Experiments）design、Response Surface Methodology（RSM） foroptimization
  * ✅ machine learning（regression・classification） forprocess outcomespredict、feature importanceevaluate
  * ✅ anomaly detection（Isolation Forest, One-Class SVM） fordefective productsearly detection
  * ✅ automated report generation（matplotlib, seaborn, Jinja2） fordaily/weeklyreportingstreamline
  * ✅ fully integratedworkflow（data → analysis → optimization → report）construction

## 5.1 process data withloadingandpreprocessing

### 5.1.1 diversedataformatson対応

actualprocess data、equipment logs（CSV, TXT）、dataベースエクスポート（JSON, Excel）、proprietaryformat（binary）etc.various。

**majordataformats** ：

  * **CSV/TSV** ：most common。pandas.read_csv() forloading
  * **Excel (.xlsx, .xls)** ：pandas.read_excel() forloading
  * **JSON** ：pandas.read_json()orjson.load() forloading
  * **HDF5** ：大規模data。pandas.read_hdf() forloading
  * **SQL Database** ：pandas.read_sql() forquery directly

#### Code Example5-1: 多formatsdataローダー（Batch Processing）
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import pandas as pd
    import numpy as np
    import json
    import glob
    from pathlib import Path
    
    class ProcessDataLoader:
        """
        process data withcombineローダー
    
        複数formats・multiple files withBatch Processingsupports
        """
    
        def __init__(self, data_dir='./process_data'):
            """
            Parameters
            ----------
            data_dir : str or Path
                dataディレクトリpath
            """
            self.data_dir = Path(data_dir)
            self.supported_formats = ['.csv', '.xlsx', '.json', '.txt']
    
        def load_single_file(self, filepath):
            """
            single file withloading
    
            Parameters
            ----------
            filepath : str or Path
                file path
    
            Returns
            -------
            df : pd.DataFrame
                loadeddata
            """
            filepath = Path(filepath)
            ext = filepath.suffix.lower()
    
            try:
                if ext == '.csv' or ext == '.txt':
                    # CSV/TXT withloading（区切り文字自動detected）
                    df = pd.read_csv(filepath, sep=None, engine='python')
                elif ext == '.xlsx' or ext == '.xls':
                    # Excel withloading（最初 withシート withみ）
                    df = pd.read_excel(filepath)
                elif ext == '.json':
                    # JSON withloading
                    df = pd.read_json(filepath)
                else:
                    raise ValueError(f"Unsupported file format: {ext}")
    
                # メタdata追加
                df['source_file'] = filepath.name
                df['load_timestamp'] = pd.Timestamp.now()
    
                print(f"Loaded: {filepath.name} ({len(df)} rows, {len(df.columns)} columns)")
                return df
    
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return None
    
        def load_batch(self, pattern='*', file_extension='.csv'):
            """
            バッチloading（multiple filescombine）
    
            Parameters
            ----------
            pattern : str
                file name pattern（wildcards allowed）
            file_extension : str
                file extensionfilter
    
            Returns
            -------
            df_combined : pd.DataFrame
                combineddataフレーム
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
    
            # combine
            df_combined = pd.concat(dfs, ignore_index=True)
            print(f"\nCombined data: {len(df_combined)} rows, {len(df_combined.columns)} columns")
    
            return df_combined
    
        def preprocess(self, df, dropna_thresh=0.5, drop_duplicates=True):
            """
            basicpreprocessing
    
            Parameters
            ----------
            df : pd.DataFrame
                inputdataフレーム
            dropna_thresh : float
                missing valuesabove this ratiodrop columns（0-1）
            drop_duplicates : bool
                duplicate rowswhether to drop
    
            Returns
            -------
            df_clean : pd.DataFrame
                cleaneddata
            """
            df_clean = df.copy()
    
            # original size
            n_rows_orig, n_cols_orig = df_clean.shape
    
            # 1. missing values多いcolumns withdelete
            thresh = int(len(df_clean) * dropna_thresh)
            df_clean = df_clean.dropna(thresh=thresh, axis=1)
    
            # 2. completely emptydrop rows
            df_clean = df_clean.dropna(how='all', axis=0)
    
            # 3. duplicate rows withdelete
            if drop_duplicates:
                df_clean = df_clean.drop_duplicates()
    
            # 4. data型 withauto-infer
            df_clean = df_clean.infer_objects()
    
            # 5. numeric columns withdetect outliers（simple version：±5σ）
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                lower = mean - 5 * std
                upper = mean + 5 * std
                outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
                if outliers.sum() > 0:
                    print(f"  {col}: {outliers.sum()} outliers detected (outside ±5σ)")
                    # outliersreplace with NaN（optional）
                    # df_clean.loc[outliers, col] = np.nan
    
            n_rows_clean, n_cols_clean = df_clean.shape
    
            print(f"\nPreprocessing summary:")
            print(f"  Rows: {n_rows_orig} → {n_rows_clean} ({n_rows_orig - n_rows_clean} removed)")
            print(f"  Columns: {n_cols_orig} → {n_cols_clean} ({n_cols_orig - n_cols_clean} removed)")
    
            return df_clean
    
    # usage example
    if __name__ == "__main__":
        # sampledata withgenerate（normallyload from files）
        import os
        os.makedirs('./process_data', exist_ok=True)
    
        # sampleCSVcreate files
        for i in range(3):
            df_sample = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01', periods=100, freq='h'),
                'temperature': np.random.normal(400, 10, 100),
                'pressure': np.random.normal(0.5, 0.05, 100),
                'power': np.random.normal(300, 20, 100),
                'thickness': np.random.normal(100, 5, 100)
            })
            df_sample.to_csv(f'./process_data/run_{i+1}.csv', index=False)
    
        # use loader
        loader = ProcessDataLoader(data_dir='./process_data')
    
        # バッチloading
        df = loader.load_batch(pattern='run_*', file_extension='.csv')
    
        if df is not None:
            # preprocessing
            df_clean = loader.preprocess(df, dropna_thresh=0.5, drop_duplicates=True)
    
            # statistical summary
            print("\nData summary:")
            print(df_clean.describe())
    

## 5.2 Statistical Process Control（SPC: Statistical Process Control）

### 5.2.1 SPC withfundamentals

SPC、process variationstatisticallymonitor、異常early detectionするmethod。

**majorcontrol charts（Control Charts）** ：

  * **X-bar charts** ：sample平均 withtrendsmonitor
  * **R charts** ：sample範囲（Range） withtrendsmonitor
  * **S charts** ：sample標準偏差 withtrendsmonitor
  * **I-MR charts** ：individual valuesandmoving range（Individual & Moving Range）

**control limits（Control Limits）** ：

$$ \text{UCL} = \bar{X} + 3\sigma, \quad \text{LCL} = \bar{X} - 3\sigma $$ 

  * UCL: Upper Control Limit（uppercontrol limits）
  * LCL: Lower Control Limit（lowercontrol limits）
  * $\bar{X}$: process mean
  * $\sigma$: process standard deviation

### 5.2.2 process capability indices（Cp/Cpk）

プロセスspecifications満たす能力evaluateする指標。

**Cp（Process Capability）** ：

$$ C_p = \frac{\text{USL} - \text{LSL}}{6\sigma} $$ 

  * USL: Upper Specification Limit（upper specification limit）
  * LSL: Lower Specification Limit（lower specification limit）

**Cpk（Process Capability Index）** ：

$$ C_{pk} = \min\left(\frac{\text{USL} - \mu}{3\sigma}, \frac{\mu - \text{LSL}}{3\sigma}\right) $$ 

  * $\mu$: process mean

**evaluate基準** ：

  * Cpk ≥ 1.33: excellent（defect rate <64 ppm）
  * Cpk ≥ 1.00: adequate（defect rate <2700 ppm）
  * Cpk < 1.00: 不adequate（process improvement needed）

#### Code Example5-2: SPCchartsgenerate（X-bar, R-chart, Cp/Cpk）
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    
    class SPCAnalyzer:
        """
        Statistical Process Control（SPC）analysisclass
        """
    
        def __init__(self, data, sample_size=5):
            """
            Parameters
            ----------
            data : array-like
                process data（時系columns）
            sample_size : int
                sampleサイズ（subgroup size）
            """
            self.data = np.array(data)
            self.sample_size = sample_size
            self.n_samples = len(data) // sample_size
    
            # split into subgroups
            self.samples = self.data[:self.n_samples * sample_size].reshape(-1, sample_size)
    
        def calculate_xbar_r(self):
            """
            X-bar chartsandRcharts withstatisticscalculate
    
            Returns
            -------
            stats_dict : dict
                statistics（xbar, R, UCL, LCL）
            """
            # sample平均andsample範囲
            xbar = np.mean(self.samples, axis=1)
            R = np.ptp(self.samples, axis=1)  # Range (max - min)
    
            # overall meanandaverage range
            xbar_mean = np.mean(xbar)
            R_mean = np.mean(R)
    
            # control chartsconstants（n=5 withcase）
            # A2, D3, D4from statistical tables（JIS Z 9020-2）
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
    
            # X-barcharts withcontrol limits
            xbar_UCL = xbar_mean + consts['A2'] * R_mean
            xbar_LCL = xbar_mean - consts['A2'] * R_mean
    
            # Rcharts withcontrol limits
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
            process capability indices（Cp, Cpk）calculate
    
            Parameters
            ----------
            USL : float
                upper specification limit
            LSL : float
                lower specification limit
    
            Returns
            -------
            cp_cpk : dict
                {'Cp': float, 'Cpk': float, 'ppm': float}
            """
            mu = np.mean(self.data)
            sigma = np.std(self.data, ddof=1)  # sample standard deviation
    
            # Cp
            Cp = (USL - LSL) / (6 * sigma)
    
            # Cpk
            Cpk_upper = (USL - mu) / (3 * sigma)
            Cpk_lower = (mu - LSL) / (3 * sigma)
            Cpk = min(Cpk_upper, Cpk_lower)
    
            # defect rate withestimate（ppm: parts per million）
            # assume normal distribution
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
            control chartsplot
    
            Parameters
            ----------
            USL, LSL : float, optional
                specification limits（Cp/Cpkcalculatefor）
            """
            stats_dict = self.calculate_xbar_r()
    
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
            sample_indices = np.arange(1, len(stats_dict['xbar']) + 1)
    
            # X-barcharts
            ax1.plot(sample_indices, stats_dict['xbar'], 'bo-', linewidth=2, markersize=6,
                    label='Sample Mean')
            ax1.axhline(stats_dict['xbar_mean'], color='green', linestyle='-', linewidth=2,
                       label=f"Center Line: {stats_dict['xbar_mean']:.2f}")
            ax1.axhline(stats_dict['xbar_UCL'], color='red', linestyle='--', linewidth=2,
                       label=f"UCL: {stats_dict['xbar_UCL']:.2f}")
            ax1.axhline(stats_dict['xbar_LCL'], color='red', linestyle='--', linewidth=2,
                       label=f"LCL: {stats_dict['xbar_LCL']:.2f}")
    
            # control limits外 with点highlight
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
    
            # Rcharts
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
    
            # histogramandprocess capability
            ax3.hist(self.data, bins=30, alpha=0.7, color='skyblue', edgecolor='black',
                    density=True, label='Data Distribution')
    
            # normal distribution fit
            mu = np.mean(self.data)
            sigma = np.std(self.data, ddof=1)
            x_range = np.linspace(self.data.min(), self.data.max(), 200)
            ax3.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2,
                    label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')
    
            # specification limits
            if USL is not None and LSL is not None:
                ax3.axvline(USL, color='red', linestyle='--', linewidth=2, label=f'USL: {USL}')
                ax3.axvline(LSL, color='red', linestyle='--', linewidth=2, label=f'LSL: {LSL}')
    
                # Cp/Cpkcalculate
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
    
    # 実rowsexample
    if __name__ == "__main__":
        # sampledatagenerate（process datasimulation）
        np.random.seed(42)
    
        # normalプロセス
        data_normal = np.random.normal(100, 2, 100)
    
        # insert anomaly（90-110番目shift at）
        data_shift = np.random.normal(105, 2, 20)
        data = np.concatenate([data_normal[:90], data_shift, data_normal[90:]])
    
        # SPCanalysis
        spc = SPCAnalyzer(data, sample_size=5)
    
        # control chartsplot（specifications: 95-105）
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
    

## 5.3 Design of Experiments（DOE: Design of Experiments）

### 5.3.1 DOE withfundamentals

DOE、minimal experiments formultiple parameters witheffectsefficientlyinvestigatemethod。

**majorexperiments計画** ：

  * **full factorial design** ：all combinationsexperiments（2kexperiments）
  * **fractional factorial design** ：main effects onlyevaluate（2k-pexperiments）
  * **orthogonal array** ：Taguchi method（L8, L16, L27etc.）
  * **central composite design（CCD）** ：Response Surface Methodology（RSM）for

### 5.3.2 Response Surface Methodology（RSM: Response Surface Methodology）

RSM、response variable（objective function）andexplanatory variables（parameters） withrelationship2order polynomialmodel using。

$$ y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i=1}^{k} \beta_{ii} x_i^2 + \sum_{i
* $y$: response variable（film thickness、stress、quality scoreetc.）
* $x_i$: explanatory variables（temperature、pressure、poweretc.）
* $\beta$: regressioncoefficients

#### Code Example5-3: Design of Experiments（2factorfull factorial design+RSM）
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from itertools import product
    
    def full_factorial_design(factors, levels):
        """
        full factorial design withexperiments計画generate
    
        Parameters
        ----------
        factors : dict
            {'factor_name': [level1, level2, ...]}
        levels : int
            各factor withnumber of levels（2level、3leveletc.）
    
        Returns
        -------
        design : pd.DataFrame
            experiments計画表
        """
        factor_names = list(factors.keys())
        factor_values = [factors[name] for name in factor_names]
    
        # all combinationsgenerate
        combinations = list(product(*factor_values))
    
        design = pd.DataFrame(combinations, columns=factor_names)
    
        return design
    
    def response_surface_model(X, y, degree=2):
        """
        response surfacemodel（多項式regression）
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            explanatory variables
        y : array-like, shape (n_samples,)
            response variable
        degree : int
            polynomial degree（usually2）
    
        Returns
        -------
        model : sklearn model
            fittedmodel
        poly : PolynomialFeatures
            polynomial transformer
        """
        # polynomial featuresgenerate
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(X)
    
        # 線形regression
        model = LinearRegression()
        model.fit(X_poly, y)
    
        print(f"R² score: {model.score(X_poly, y):.3f}")
    
        return model, poly
    
    # experiments計画 with設定
    factors = {
        'Temperature': [300, 350, 400, 450, 500],  # [°C]
        'Pressure': [0.2, 0.35, 0.5, 0.65, 0.8]     # [Pa]
    }
    
    design = full_factorial_design(factors, levels=5)
    print("Experimental Design (Full Factorial):")
    print(design.head(10))
    print(f"Total experiments: {len(design)}")
    
    # response variable withsimulation（normallyexperiments for測定）
    # truemodel: y = 100 + 0.2*T + 50*P - 0.0002*T^2 - 50*P^2 + 0.05*T*P
    def true_response(T, P):
        """true応答関数（未知and andて扱う）"""
        y = 100 + 0.2*T + 50*P - 0.0002*T**2 - 50*P**2 + 0.05*T*P
        # add noise
        y += np.random.normal(0, 2, len(T))
        return y
    
    design['Response'] = true_response(design['Temperature'], design['Pressure'])
    
    # data準備
    X = design[['Temperature', 'Pressure']].values
    y = design['Response'].values
    
    # response surfacemodelfit
    model, poly = response_surface_model(X, y, degree=2)
    
    # prediction gridgenerate
    T_range = np.linspace(300, 500, 50)
    P_range = np.linspace(0.2, 0.8, 50)
    T_grid, P_grid = np.meshgrid(T_range, P_range)
    
    X_grid = np.c_[T_grid.ravel(), P_grid.ravel()]
    X_grid_poly = poly.transform(X_grid)
    y_pred_grid = model.predict(X_grid_poly).reshape(T_grid.shape)
    
    # visualization
    fig = plt.figure(figsize=(16, 6))
    
    # left plot: 3Dresponse surface
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
    
    # center plot: contour plot
    ax2 = fig.add_subplot(1, 3, 2)
    contour = ax2.contourf(T_grid, P_grid, y_pred_grid, levels=20, cmap='viridis', alpha=0.8)
    contour_lines = ax2.contour(T_grid, P_grid, y_pred_grid, levels=10,
                                 colors='white', linewidths=1, alpha=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8)
    ax2.scatter(X[:, 0], X[:, 1], color='red', s=50, marker='o',
               edgecolors='black', linewidths=1.5, label='Exp. Points')
    
    # optimal point
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
    
    # right plot: main effects plot
    ax3 = fig.add_subplot(1, 3, 3)
    
    # temperature main effect（pressure fixed at median）
    P_center = np.median(factors['Pressure'])
    T_effect = np.linspace(300, 500, 50)
    X_effect_T = np.c_[T_effect, np.full(50, P_center)]
    X_effect_T_poly = poly.transform(X_effect_T)
    y_effect_T = model.predict(X_effect_T_poly)
    
    ax3.plot(T_effect, y_effect_T, 'b-', linewidth=2, label=f'Temperature (P={P_center} Pa)')
    
    # pressure main effect（temperature fixed at median）
    T_center = np.median(factors['Temperature'])
    P_effect = np.linspace(0.2, 0.8, 50)
    X_effect_P = np.c_[np.full(50, T_center), P_effect]
    X_effect_P_poly = poly.transform(X_effect_P)
    y_effect_P = model.predict(X_effect_P_poly)
    
    # right axis
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
    
    # regressioncoefficients withdisplay
    coef_names = poly.get_feature_names_out(['T', 'P'])
    print(f"\nRegression Coefficients:")
    for name, coef in zip(coef_names, [model.intercept_] + list(model.coef_[1:])):
        print(f"  {name}: {coef:.4f}")
    

## 5.4 machine learningwithプロセス予測

### 5.4.1 regressionmodelwith品質予測

machine learningforいて、プロセスparametersfromproduct quality予測。

**majoralgorithms** ：

  * **ランダムフォレストregression** ：high accuracy、interpret性（feature importance）
  * **gradient boosting（XGBoost, LightGBM）** ：最high accuracy
  * **neural networks** ：nonlinear patternslearn
  * **サポートベクターregression（SVR）** ：小datasetfor

#### Code Example5-4: ランダムフォレストwithprocess quality予測
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example5-4: ランダムフォレストwithprocess quality予測
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import seaborn as sns
    
    # sampledatasetgenerate（normallyexperimentsdata使for）
    np.random.seed(42)
    n_samples = 200
    
    # プロセスparameters
    data = pd.DataFrame({
        'Temperature': np.random.uniform(300, 500, n_samples),
        'Pressure': np.random.uniform(0.2, 0.8, n_samples),
        'Power': np.random.uniform(100, 400, n_samples),
        'Flow_Rate': np.random.uniform(50, 150, n_samples),
        'Time': np.random.uniform(30, 120, n_samples)
    })
    
    # target variable（film thickness） withsimulation
    # truemodel: 複雑な非線形relationship
    data['Thickness'] = (
        0.5 * data['Temperature'] +
        100 * data['Pressure'] +
        0.3 * data['Power'] +
        0.2 * data['Flow_Rate'] +
        1.0 * data['Time'] +
        0.001 * data['Temperature'] * data['Pressure'] -
        0.0005 * data['Temperature']**2 +
        np.random.normal(0, 10, n_samples)  # noise
    )
    
    print("Dataset shape:", data.shape)
    print("\nFeature summary:")
    print(data.describe())
    
    # featuresandtarget variablesplit
    X = data.drop('Thickness', axis=1)
    y = data['Thickness']
    
    # 訓練dataandテストdatasplit into
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ランダムフォレストmodeltrain
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
    
    # evaluate指標
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
    
    # cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    print(f"\n5-Fold CV R² score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("="*60)
    
    # visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # top-left: 予測vsactual（訓練data）
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                   'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Thickness', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Thickness', fontsize=12)
    axes[0, 0].set_title(f'Training Set\nR² = {train_r2:.3f}, RMSE = {train_rmse:.2f}',
                        fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # top-right: 予測vsactual（テストdata）
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
    
    # bottom-left: feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    bars = axes[1, 0].barh(feature_importance['Feature'], feature_importance['Importance'],
                           color='skyblue', edgecolor='black')
    axes[1, 0].set_xlabel('Importance', fontsize=12)
    axes[1, 0].set_title('Feature Importance', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # 値to barsdisplay
    for bar, importance in zip(bars, feature_importance['Importance']):
        axes[1, 0].text(importance + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{importance:.3f}', va='center', fontsize=10)
    
    # bottom-right: residualsplot
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
    

### 5.4.2 classificationmodelwithdefective productsdetected

プロセスparametersfromdefective productsin advance予測。

#### Code Example5-5: ロジスティックregressionwithdefective products予測
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example5-5: ロジスティックregressionwithdefective products予測
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    import seaborn as sns
    
    # sampledatasetgenerate
    np.random.seed(42)
    n_samples = 300
    
    # good products（70%）anddefective products（30%）
    data = pd.DataFrame({
        'Temperature': np.random.normal(400, 30, n_samples),
        'Pressure': np.random.normal(0.5, 0.1, n_samples),
        'Power': np.random.normal(250, 50, n_samples)
    })
    
    # defective productslabelsgenerate（specifications外conditions fordefect rate increases）
    # good productsconditions: 380<t<420, (data['temperature']="" 0.4<p<0.6,="" 200<power<300="" good_condition="("> 380) & (data['Temperature'] < 420) &
        (data['Pressure'] > 0.4) & (data['Pressure'] < 0.6) &
        (data['Power'] > 200) & (data['Power'] < 300)
    )
    
    # conditions外 forもprobabilisticallygood productsになるこandある
    data['Defect'] = 0
    data.loc[~good_condition, 'Defect'] = np.random.choice([0, 1], size=(~good_condition).sum(),
                                                            p=[0.3, 0.7])
    
    print(f"Dataset: {len(data)} samples")
    print(f"Defect rate: {data['Defect'].mean()*100:.1f}%")
    
    # featuresandtarget variable
    X = data[['Temperature', 'Pressure', 'Power']]
    y = data['Defect']
    
    # train-test分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    
    # ロジスティックregressionmodel
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # 予測
    y_test_pred = lr_model.predict(X_test)
    y_test_prob = lr_model.predict_proba(X_test)[:, 1]
    
    # evaluate
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(y_test, y_test_pred, target_names=['Good', 'Defect']))
    
    auc_score = roc_auc_score(y_test, y_test_prob)
    print(f"ROC-AUC Score: {auc_score:.3f}")
    print("="*60)
    
    # visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # top-left: 混同rowscolumns
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 0],
               xticklabels=['Good', 'Defect'], yticklabels=['Good', 'Defect'])
    axes[0, 0].set_xlabel('Predicted', fontsize=12)
    axes[0, 0].set_ylabel('Actual', fontsize=12)
    axes[0, 0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    # top-right: ROCcurve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title('ROC Curve', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(alpha=0.3)
    
    # bottom-left: featurescoefficients
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
    
    # bottom-right: probability distribution
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
    </t<420,>

## 5.5 anomaly detection（Anomaly Detection）

### 5.5.1 Isolation Forestwithanomaly detection

Isolation Forest、教師な andlearn for異常datadetected。normaldata patternsfromdeviate fromdata「異常」and andてidentify。

#### Code Example5-6: Isolation Forestwith異常プロセスdetected
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example5-6: Isolation Forestwith異常プロセスdetected
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # sampledatagenerate
    np.random.seed(42)
    
    # normaldata（200sample）
    normal_data = pd.DataFrame({
        'Temperature': np.random.normal(400, 10, 200),
        'Pressure': np.random.normal(0.5, 0.05, 200),
        'Power': np.random.normal(250, 20, 200),
        'Thickness': np.random.normal(100, 5, 200)
    })
    
    # 異常data（20sample）
    anomaly_data = pd.DataFrame({
        'Temperature': np.random.uniform(350, 450, 20),
        'Pressure': np.random.uniform(0.3, 0.7, 20),
        'Power': np.random.uniform(150, 350, 20),
        'Thickness': np.random.uniform(70, 130, 20)
    })
    
    # datacombine
    data = pd.concat([normal_data, anomaly_data], ignore_index=True)
    true_labels = np.array([0]*len(normal_data) + [1]*len(anomaly_data))  # 0: normal, 1: anomaly
    
    print(f"Total samples: {len(data)}")
    print(f"Anomaly rate: {(true_labels == 1).mean()*100:.1f}%")
    
    # features withstandardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Isolation Forestmodel
    iso_forest = IsolationForest(
        contamination=0.1,  # expected異常rate
        random_state=42,
        n_estimators=100
    )
    
    # 予測（-1: 異常, 1: normal）
    predictions = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.score_samples(X_scaled)
    
    # 予測labels0/1convert to
    pred_labels = (predictions == -1).astype(int)
    
    # evaluate（truelabelswhen available）
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + "="*60)
    print("Anomaly Detection Results:")
    print("="*60)
    print(classification_report(true_labels, pred_labels,
                              target_names=['Normal', 'Anomaly']))
    print("="*60)
    
    # PCA for2dimensionsにcompress（visualizationfor）
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # top-left: PCAspace for withanomaly detection結果
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
    
    # top-right: truelabels（比較for）
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
    
    # bottom-left: 異常score分布
    axes[1, 0].hist(anomaly_scores[true_labels == 0], bins=30, alpha=0.6,
                   label='Normal', color='green', edgecolor='black')
    axes[1, 0].hist(anomaly_scores[true_labels == 1], bins=30, alpha=0.6,
                   label='Anomaly', color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Anomaly Score (lower = more anomalous)', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    
    # bottom-right: 混同rowscolumns
    cm = confusion_matrix(true_labels, pred_labels)
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1, 1],
               xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    axes[1, 1].set_xlabel('Predicted', fontsize=12)
    axes[1, 1].set_ylabel('Actual', fontsize=12)
    axes[1, 1].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # most異常なsampleリスト
    top_anomalies = np.argsort(anomaly_scores)[:5]
    print("\nTop 5 Most Anomalous Samples:")
    print(data.iloc[top_anomalies])
    print("\nAnomaly Scores:")
    print(anomaly_scores[top_anomalies])
    

## 5.6 automated report generation

### 5.6.1 daily/weeklyプロセスreportautomated化

analysis結果automaticallyPDFreportor HTML dashboards。

#### Code Example5-7: fully integratedworkflow（data → analysis → report）
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    # - seaborn>=0.12.0
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    from matplotlib.backends.backend_pdf import PdfPages
    import warnings
    warnings.filterwarnings('ignore')
    
    class ProcessReportGenerator:
        """
        プロセスanalysis withautomated report generationclass
        """
    
        def __init__(self, data, report_title="Process Analysis Report"):
            """
            Parameters
            ----------
            data : pd.DataFrame
                process data
            report_title : str
                reporttitle
            """
            self.data = data
            self.report_title = report_title
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        def generate_summary_statistics(self):
            """statistical summary withgenerate"""
            summary = self.data.describe().T
            summary['missing'] = self.data.isnull().sum()
            summary['missing_pct'] = (summary['missing'] / len(self.data) * 100).round(2)
    
            return summary
    
        def plot_time_series(self, ax, column):
            """時系columnsplot"""
            if 'timestamp' in self.data.columns:
                x = self.data['timestamp']
            else:
                x = np.arange(len(self.data))
    
            ax.plot(x, self.data[column], 'b-', linewidth=1.5, alpha=0.7)
    
            # control limits（±3σ）
            mean = self.data[column].mean()
            std = self.data[column].std()
            ucl = mean + 3*std
            lcl = mean - 3*std
    
            ax.axhline(mean, color='green', linestyle='-', linewidth=2, label='Mean')
            ax.axhline(ucl, color='red', linestyle='--', linewidth=2, label='UCL (±3σ)')
            ax.axhline(lcl, color='red', linestyle='--', linewidth=2, label='LCL')
    
            # control limits外 with点highlight
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
            """分布plot"""
            ax.hist(self.data[column], bins=bins, alpha=0.7, color='skyblue',
                   edgecolor='black', density=True)
    
            # normal distribution fit
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
            """相関rowscolumnsヒートマップ"""
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            corr = self.data[numeric_cols].corr()
    
            import seaborn as sns
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
        def generate_pdf_report(self, filename='process_report.pdf'):
            """
            PDFreport withgenerate
    
            Parameters
            ----------
            filename : str
                出力PDFファイル名
            """
            with PdfPages(filename) as pdf:
                # ページ1: titleandstatistical summary
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle(self.report_title, fontsize=18, fontweight='bold', y=0.98)
    
                # タイムスタンプ
                fig.text(0.5, 0.94, f'Generated: {self.timestamp}', ha='center',
                        fontsize=10, style='italic')
    
                # statistical summaryテーブル
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
    
                # ヘッダーrows with装飾
                for i in range(len(summary_display.columns)):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')
    
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
                # ページ2-N: 各変数 with時系columnsand分布
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
    
                for col in numeric_cols:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
    
                    self.plot_time_series(ax1, col)
                    self.plot_distribution(ax2, col)
    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
    
                # 最終ページ: 相関rowscolumns
                fig, ax = plt.subplots(figsize=(11, 8.5))
                self.plot_correlation_matrix(ax)
    
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
            print(f"PDF report generated: {filename}")
    
    # usage example
    if __name__ == "__main__":
        # sampledatagenerate
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
    
        # some異常insert
        data.loc[50:55, 'Temperature'] = np.random.normal(430, 5, 6)
        data.loc[50:55, 'Thickness'] = np.random.normal(110, 3, 6)
    
        # reportgenerate
        report_gen = ProcessReportGenerator(data, report_title="Weekly Process Analysis Report")
    
        # statistical summarydisplay
        print("Statistical Summary:")
        print(report_gen.generate_summary_statistics())
    
        # PDFreportgenerate
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

## 5.7 exerciseproblem

### exercise5-1: Cp/Cpkcalculate（easy）

**problem** ：film thicknessdata平均100 nm、標準偏差3 nm、specifications95-105 nm withandき、CpandCpkcalculate。プロセス適切？

**answerdisplay**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    """
    Example: problem：film thicknessdata平均100 nm、標準偏差3 nm、specifications95
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
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
        print("process capability: excellent")
    elif Cpk >= 1.00:
        print("process capability: adequate")
    else:
        print("process capability: 不adequate（improvement needed）")
    
    # defect rateestimate
    from scipy import stats
    ppm = (stats.norm.cdf((LSL - mu) / sigma) +
           (1 - stats.norm.cdf((USL - mu) / sigma))) * 1e6
    print(f"estimatedefect rate: {ppm:.1f} ppm")
    

### exercise5-2: 2factorexperiments計画（medium）

**problem** ：temperature（300, 400, 500°C）andpressure（0.3, 0.5, 0.7 Pa） with2factor3levelexperimentsdesign、all combinations withexperiments計画表作成。

**answerdisplay**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: problem：temperature（300, 400, 500°C）andpressure（0.3, 0.5, 0.
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    from itertools import product
    
    # factorandlevel
    factors = {
        'Temperature': [300, 400, 500],
        'Pressure': [0.3, 0.5, 0.7]
    }
    
    # all combinationsgenerate
    combinations = list(product(factors['Temperature'], factors['Pressure']))
    design = pd.DataFrame(combinations, columns=['Temperature', 'Pressure'])
    
    print("Experimental Design (Full Factorial):")
    print(design)
    print(f"\nTotal experiments: {len(design)}")
    

### exercise5-3: ランダムフォレストfeature importance（medium）

**problem** ：5つ withプロセスparameters（temperature、pressure、power、流量、時間）fromfilm thickness予測するmodel for、mostimportantfactoridentify。

**answerdisplay**

**Code Example5-4 withランダムフォレストmodel** 実rows and、feature importancecheck：
    
    
    # feature_importancefromimportancecheck
    print(feature_importance)
    
    # 典型的な結果:
    #   Time: 0.35 (最重要: longer deposition = thicker film)
    #   Temperature: 0.25 (成長速度にeffects)
    #   Power: 0.20 (スパッタ収rateにeffects)
    #   Pressure: 0.15 (ガス散乱にeffects)
    #   Flow_Rate: 0.05 (間接的effects)
    

### exercise5-4: anomaly detection with閾値設定（medium）

**problem** ：Isolation Forest with`contamination`parameters0.05, 0.1, 0.2 for変えたcase、detectedされる異常数どう変化する？

**answerdisplay**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: problem：Isolation Forest with`contamination`parameters0.05, 
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from sklearn.ensemble import IsolationForest
    import numpy as np
    
    # sampledata
    data = np.random.normal(0, 1, (100, 4))
    
    contaminations = [0.05, 0.1, 0.2]
    
    for cont in contaminations:
        iso_forest = IsolationForest(contamination=cont, random_state=42)
        predictions = iso_forest.fit_predict(data)
        n_anomalies = (predictions == -1).sum()
    
        print(f"Contamination = {cont}: {n_anomalies} anomalies detected ({n_anomalies/len(data)*100:.1f}%)")
    
    # 出力example:
    # Contamination = 0.05: 5 anomalies (5.0%)
    # Contamination = 0.1: 10 anomalies (10.0%)
    # Contamination = 0.2: 20 anomalies (20.0%)
    
    print("\ninterpret: contaminationexpected異常rate。high多くdetected（偽陽性リスク増）")
    

### exercise5-5: Response Surface Methodology withoptimization（hard）

**problem** ：temperatureandpressure with2factor forfilm thicknessmaximize andたい。response surfacemodelfromoptimumconditions（temperature、pressure）numericallyfind。

**answerdisplay**
    
    
    from scipy.optimize import minimize
    
    # Code Example5-3 withmodelandpoly使for
    def objective(x):
        """最小化目標関数（maximize withため負号）"""
        X_input = np.array(x).reshape(1, -1)
        X_poly = poly.transform(X_input)
        y_pred = model.predict(X_poly)
        return -y_pred[0]  # maximize withため負号
    
    # initial guessandbounds
    x0 = [400, 0.5]  # [Temperature, Pressure]
    bounds = [(300, 500), (0.2, 0.8)]
    
    # optimization
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    T_opt = result.x[0]
    P_opt = result.x[1]
    y_opt = -result.fun
    
    print(f"Optimal conditions:")
    print(f"  Temperature: {T_opt:.1f} °C")
    print(f"  Pressure: {P_opt:.3f} Pa")
    print(f"  Predicted maximum response: {y_opt:.2f}")
    

### exercise5-6: datapreprocessing witheffects（hard）

**problem** ：missing valueswithdataset for、(a)欠損rowsdelete、(b)平均値imputation、(c)KNNimputation with3method比較 and、machine learningmodel with精度oneffectsevaluate。

**answerdisplay**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: problem：missing valueswithdataset for、(a)欠損rowsdelete、(b)平均値
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    # sampledata（missing valuesあり）
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'X1': np.random.normal(0, 1, n),
        'X2': np.random.normal(0, 1, n),
        'X3': np.random.normal(0, 1, n),
        'y': np.random.normal(0, 1, n)
    })
    
    # missing valuesランダムに導入（10%）
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
    
    print("\nconclusion: data量andmodel精度 withトレードオフ考慮 andて選択")
    

### exercise5-7: SPCcontrol limits withadjusting（hard）

**problem** ：3σcontrol limits（99.73%） instead of2σ（95.45%）when using、偽警報rateand見逃 andrateどう変化する？どちら適切discuss which。

**answerdisplay**

**theoretical比較** ：

  * **3σcontrol limits** ： 
    * 偽警報rate（α）: 0.27%（normalな withに異常and判定）
    * 見逃 andrate（β）: high（異常な withにdetected漏れ）
    * 適for: stableプロセス、adjustingコストhighcase
  * **2σcontrol limits** ： 
    * 偽警報rate（α）: 4.55%（偽警報増える）
    * 見逃 andrate（β）: low（earlydetected）
    * 適for: 不stableプロセス、early警告importantcase

**practice的選択** ：
    
    
    # Requirements:
    # - Python 3.9+
    # - scipy>=1.11.0
    
    """
    Example: practice的選択：
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    # 偽警報rate withcalculate
    from scipy import stats
    
    alpha_3sigma = 2 * (1 - stats.norm.cdf(3))
    alpha_2sigma = 2 * (1 - stats.norm.cdf(2))
    
    print(f"3σcontrol limits: 偽警報rate = {alpha_3sigma*100:.2f}%")
    print(f"2σcontrol limits: 偽警報rate = {alpha_2sigma*100:.2f}%")
    
    print("\n推奨:")
    print("  - 3σ: matureプロセス、adjustingコストhigh（半導体製造etc.）")
    print("  - 2σ: newプロセス、品質critical、early介入needed（医薬品etc.）")
    

### exercise5-8: combineworkflowdesign（hard）

**problem** ：actual工場 for、dataacquisition → SPCmonitoring → anomaly detection → 自動アラート → weeklyreport with完全自動化システムdesign。neededなコンポーネントanddataフローdescribe。

**answerdisplay**

**システムarchitecture** ：

  1. **dataacquisitionlayer** ： 
     * 装置from withreal-timeログ収集（OPC UA, MQTT）
     * dataベースstorage（TimescaleDB, InfluxDB）
  2. **処理layer** ： 
     * 定期実rows（cron, Airflow）
     * SPCanalysis（Python + pandas）
     * anomaly detection（Isolation Forest）
  3. **アラートlayer** ： 
     * control limits外detected → メール/Slack通知
     * 異常score閾値exceeded → emergencyアラート
  4. **reportlayer** ： 
     * weeklyreport自動generate（PDF）
     * Webダッシュボード（Dash, Streamlit）

**実装example（pseudo-code）** ：
    
    
    # daily_monitoring.py (cron: daily1回実rows)
    def daily_monitoring():
        # 1. dataacquisition
        data = load_data_from_database(start_date=yesterday, end_date=today)
    
        # 2. SPCanalysis
        spc = SPCAnalyzer(data)
        out_of_control = spc.detect_out_of_control()
    
        # 3. anomaly detection
        anomalies = detect_anomalies(data)
    
        # 4. アラート
        if out_of_control or anomalies:
            send_alert(subject="Process Anomaly Detected",
                      body=f"Out of control: {out_of_control}\nAnomalies: {anomalies}")
    
        # 5. ログsave
        save_monitoring_log(data, spc_results, anomalies)
    
    # weekly_report.py (cron: every月曜実rows)
    def weekly_report():
        data = load_data_from_database(start_date=last_week, end_date=today)
        report_gen = ProcessReportGenerator(data)
        report_gen.generate_pdf_report('weekly_report.pdf')
        send_email(to='manager@example.com', attachment='weekly_report.pdf')
    

## 5.8 learn withcheck

### 基本理解度check

  1. CSV, Excel, JSONetc.diverseformatfromdata読み込めます？
  2. X-barchartsandRcharts with意味and使い分け理解 andています？
  3. Cp/Cpk with違いand、process capabilityevaluate with基準説明？
  4. full factorial designandResponse Surface Methodology with違い理解 andています？
  5. ランダムフォレスト withfeature importance withinterpret？
  6. Isolation Forestwithanomaly detection with原理理解 andています？

### practiceskillscheck

  1. actualprocess datafromSPCchartsgenerate and、control limits外 with点identify？
  2. 2factor以上 withexperiments計画design、response surfacemodel foroptimization？
  3. machine learningmodel（regression・classification） forprocess quality予測？
  4. anomaly detectionalgorithms fordefective productsearly detection？
  5. analysis結果automaticallyPDFreportに出力？

### 応for力check

  1. fully integratedworkflow（data → analysis → optimization → report）design・実装？
  2. actual工場data for、品質problem with根本原因datafromidentify？
  3. newプロセス with立ち上げ時に、DOEandML組み合わせたoptimization戦略提案？

## 5.9 references

  1. Montgomery, D.C. (2012). _Statistical Quality Control_ (7th ed.). Wiley. pp. 156-234 (Control charts), pp. 289-345 (Process capability).
  2. Box, G.E.P., Hunter, J.S., Hunter, W.G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley. pp. 123-189 (Factorial designs), pp. 289-345 (Response surface methods).
  3. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). _An Introduction to Statistical Learning with Applications in Python_. Springer. pp. 303-335 (Random forests), pp. 445-489 (Unsupervised learning).
  4. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." _Journal of Machine Learning Research_ , 12:2825-2830. - scikit-learn documentation.
  5. McKinney, W. (2017). _Python for Data Analysis_ (2nd ed.). O'Reilly. pp. 89-156 (pandas basics), pp. 234-289 (Data cleaning and preparation).
  6. Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). "Isolation Forest." _Proceedings of the 8th IEEE International Conference on Data Mining_ , pp. 413-422. DOI: 10.1109/ICDM.2008.17
  7. Hunter, J.D. (2007). "Matplotlib: A 2D Graphics Environment." _Computing in Science & Engineering_, 9(3):90-95. DOI: 10.1109/MCSE.2007.55
  8. Waskom, M. (2021). "seaborn: statistical data visualization." _Journal of Open Source Software_ , 6(60):3021. DOI: 10.21105/joss.03021

## 5.10 次 withsteps

こ with章 for学んだprocess data analysis withpracticeskills、材料科学研究and産業応for with両方 forimmediately活for。次 withstepsand andて：

  * **実dataon適for** ：自身 with研究室や工場 withprocess data for、学んだmethodpractice
  * **advancedmachine learning** ：深layerlearn（LSTM, Transformer）with時系columns予測
  * **real-timemonitoring** ：ストリーミングdata処理（Apache Kafka, Flink） with導入
  * **combineシステムdevelopment** ：Webダッシュボード（Dash, Streamlit） for withvisualization
  * **自動化 with深化** ：CI/CDパイプラインconstruction（GitHub Actions） for完全自動運for

プロセス技術入門シリーズ完走されたあなた、薄膜成長・プロセス制御・dataanalysis with全領域combine的に理解 and、practice力身につけま andた。こ with知識基盤に、さらなる専門性深め、材料科学 with最前線 for活躍されるこand期待！
