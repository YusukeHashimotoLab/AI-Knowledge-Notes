---
title: "Chapter 3: PAT and Real-Time Quality Control"
chapter_title: "Chapter 3: PAT and Real-Time Quality Control"
subtitle: Process Analytical Technology and Real-Time Quality Control
---

[AI Terakoya Top](<../index.html>)›[Process Informatics](<../../index.html>)›[Pharma Manufacturing AI](<../../PI/pharma-manufacturing-ai/index.html>)›Chapter 3

🌐 EN | [🇯🇵 JP](<../../../jp/PI/pharma-manufacturing-ai/chapter-3.html>) | Last sync: 2025-11-16

[← Back to Series Index](<index.html>)

## 📖 Chapter Overview

Process Analytical Technology (PAT) is an FDA-recommended real-time quality control approach. This chapter covers PAT tools such as NIR/Raman spectroscopy, Multivariate Statistical Process Control (MSPC), and Real-Time Release Testing (RTRT) implementation methods to achieve Quality by Design. 

### 🎯 Learning Objectives

  * Basic concepts of PAT (Process Analytical Technology) and FDA guidance
  * Preprocessing and feature extraction of NIR/Raman spectroscopic data
  * Quantitative analysis using PLS (Partial Least Squares) regression
  * Implementation of Multivariate Statistical Process Control (MSPC)
  * Construction of Hotelling's T² and SPE control charts
  * Design of Real-Time Release Testing (RTRT)
  * PAT system validation strategies

## 🔬 3.1 Fundamentals of PAT (Process Analytical Technology)

### FDA PAT Initiative

The FDA (U.S. Food and Drug Administration) issued PAT Guidance in 2004, promoting a paradigm shift: "Building quality into the process rather than testing for quality." 

**🏭 Four PAT Tools**  
1\. **Multivariate Tools** : PCA, PLS, Neural Networks  
2\. **Process Analyzers** : NIR, Raman, UV-Vis Spectrometers  
3\. **Process Control Tools** : Feedback Control, Adaptive Control  
4\. **Continuous Improvement & Knowledge Management**: Databases, Statistical Analysis 

### Principles of NIR/Raman Spectroscopy

  * **NIR (Near-Infrared Spectroscopy)** : Non-destructive, solid/liquid measurement capability, effective for moisture/content determination
  * **Raman Spectroscopy** : Molecular vibration spectra, minimal water interference, effective for crystalline polymorph identification

### 💻 Code Example 3.1: NIR Spectral Data Preprocessing and PLS Regression
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler
    from scipy.signal import savgol_filter
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    class NIRAnalyzer:
        """NIR spectroscopy data analysis class"""
    
        def __init__(self, wavelengths):
            """
            Args:
                wavelengths: Wavelength array (nm)
            """
            self.wavelengths = wavelengths
            self.scaler = StandardScaler()
            self.pls_model = None
    
        def generate_nir_spectra(self, n_samples=100):
            """Generate NIR spectral data (simulated)"""
            np.random.seed(42)
    
            # API content (85-115% range)
            api_content = np.random.uniform(85, 115, n_samples)
    
            spectra = []
            for content in api_content:
                # Baseline spectrum
                baseline = 0.5 + 0.001 * self.wavelengths
    
                # API absorption peaks (around 1450nm, 1900nm)
                peak1 = 0.3 * (content / 100) * np.exp(-((self.wavelengths - 1450) ** 2) / (50 ** 2))
                peak2 = 0.2 * (content / 100) * np.exp(-((self.wavelengths - 1900) ** 2) / (80 ** 2))
    
                # Excipient effects
                excipient = 0.1 * np.exp(-((self.wavelengths - 1700) ** 2) / (100 ** 2))
    
                # Noise
                noise = np.random.normal(0, 0.01, len(self.wavelengths))
    
                spectrum = baseline + peak1 + peak2 + excipient + noise
                spectra.append(spectrum)
    
            return np.array(spectra), api_content
    
        def preprocess_spectra(self, spectra, method='snv'):
            """
            Spectral preprocessing
    
            Args:
                spectra: Spectral data (n_samples × n_wavelengths)
                method: Preprocessing method ('snv', 'msc', 'derivative')
    
            Returns:
                Preprocessed spectra
            """
            if method == 'snv':
                # Standard Normal Variate (SNV)
                mean = np.mean(spectra, axis=1, keepdims=True)
                std = np.std(spectra, axis=1, keepdims=True)
                processed = (spectra - mean) / std
    
            elif method == 'msc':
                # Multiplicative Scatter Correction (MSC)
                ref_spectrum = np.mean(spectra, axis=0)
                processed = np.zeros_like(spectra)
    
                for i in range(spectra.shape[0]):
                    # Remove scaling and offset using linear regression
                    fit = np.polyfit(ref_spectrum, spectra[i], 1)
                    processed[i] = (spectra[i] - fit[1]) / fit[0]
    
            elif method == 'derivative':
                # Savitzky-Golay 1st derivative
                processed = np.array([savgol_filter(s, window_length=11, polyorder=2, deriv=1)
                                      for s in spectra])
    
            else:
                processed = spectra
    
            return processed
    
        def build_pls_model(self, X_train, y_train, n_components=5):
            """
            Build PLS model
    
            Args:
                X_train: Training spectral data
                y_train: Training labels (API content)
                n_components: Number of PLS components
            """
            # Data standardization
            X_scaled = self.scaler.fit_transform(X_train)
    
            # PLS model
            self.pls_model = PLSRegression(n_components=n_components)
            self.pls_model.fit(X_scaled, y_train)
    
            # Cross-validation
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.pls_model, X_scaled, y_train,
                                         cv=kfold, scoring='r2')
    
            return cv_scores
    
        def predict(self, X_test):
            """Prediction"""
            X_scaled = self.scaler.transform(X_test)
            return self.pls_model.predict(X_scaled)
    
        def plot_nir_analysis(self, spectra, api_content, X_test, y_test, y_pred):
            """Visualize NIR analysis results"""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            # NIR spectra (samples)
            for i in range(0, len(spectra), 20):
                axes[0, 0].plot(self.wavelengths, spectra[i], alpha=0.6,
                                label=f'{api_content[i]:.1f}%' if i < 80 else None)
    
            axes[0, 0].set_xlabel('Wavelength (nm)')
            axes[0, 0].set_ylabel('Absorbance')
            axes[0, 0].set_title('NIR Spectra (Raw Data)', fontsize=12, fontweight='bold')
            axes[0, 0].legend(fontsize=8, loc='upper right')
            axes[0, 0].grid(alpha=0.3)
    
            # Preprocessed spectra
            processed = self.preprocess_spectra(spectra, method='snv')
            for i in range(0, len(processed), 20):
                axes[0, 1].plot(self.wavelengths, processed[i], alpha=0.6)
    
            axes[0, 1].set_xlabel('Wavelength (nm)')
            axes[0, 1].set_ylabel('SNV-Processed Absorbance')
            axes[0, 1].set_title('NIR Spectra (After SNV Preprocessing)', fontsize=12, fontweight='bold')
            axes[0, 1].grid(alpha=0.3)
    
            # PLS prediction accuracy
            axes[1, 0].scatter(y_test, y_pred, alpha=0.6, s=50, color='#11998e')
            axes[1, 0].plot([85, 115], [85, 115], 'r--', linewidth=2, label='Ideal Line')
    
            # ±5% tolerance range
            axes[1, 0].plot([85, 115], [90, 120], 'orange', linestyle=':', linewidth=1.5, alpha=0.7)
            axes[1, 0].plot([85, 115], [80, 110], 'orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
            # Calculate R² and RMSE
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
            axes[1, 0].text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}%',
                            transform=axes[1, 0].transAxes, fontsize=11,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
            axes[1, 0].set_xlabel('Actual API Content (%)')
            axes[1, 0].set_ylabel('Predicted API Content (%)')
            axes[1, 0].set_title('PLS Model Prediction Accuracy', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
    
            # Distribution of prediction errors
            errors = y_pred.flatten() - y_test
            axes[1, 1].hist(errors, bins=20, color='#38ef7d', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            axes[1, 1].set_xlabel('Prediction Error (%)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('nir_pls_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Execution example
    print("=" * 60)
    print("NIR-PLS Analysis System (PAT Implementation)")
    print("=" * 60)
    
    # Define wavelength array (1100-2500nm, 2nm intervals)
    wavelengths = np.arange(1100, 2501, 2)
    
    # Initialize NIR analyzer
    nir_analyzer = NIRAnalyzer(wavelengths)
    
    # Generate NIR spectral data
    spectra, api_content = nir_analyzer.generate_nir_spectra(n_samples=100)
    
    print(f"\nNumber of samples: {len(spectra)}")
    print(f"Number of wavelength points: {len(wavelengths)}")
    print(f"Wavelength range: {wavelengths[0]}-{wavelengths[-1]} nm")
    print(f"API content range: {api_content.min():.1f}-{api_content.max():.1f}%")
    
    # Split training/test data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, api_content, test_size=0.3, random_state=42
    )
    
    # Spectral preprocessing
    X_train_processed = nir_analyzer.preprocess_spectra(X_train, method='snv')
    X_test_processed = nir_analyzer.preprocess_spectra(X_test, method='snv')
    
    # Build PLS model
    cv_scores = nir_analyzer.build_pls_model(X_train_processed, y_train, n_components=5)
    
    print(f"\nPLS Model (5 components):")
    print(f"Cross-validation R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Prediction
    y_pred = nir_analyzer.predict(X_test_processed)
    
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nTest Set Performance:")
    print(f"R² = {r2:.4f}")
    print(f"RMSE = {rmse:.2f}%")
    print(f"Relative Error = {rmse / api_content.mean() * 100:.2f}%")
    
    # Visualization
    nir_analyzer.plot_nir_analysis(spectra, api_content, X_test_processed, y_test, y_pred)
    

**Implementation Points:**

  * NIR spectral preprocessing (SNV, MSC, derivative) for scatter removal
  * Building multivariate regression models using PLS
  * Model evaluation using cross-validation
  * Real-time API content prediction
  * Quantitative evaluation of prediction accuracy (R², RMSE)

## 📊 3.2 Multivariate Statistical Process Control (MSPC)

### MSPC Principles

Multivariate Statistical Process Control (MSPC) is a method for integrated monitoring of multiple process variables. Using Principal Component Analysis (PCA), it learns the data space during normal operation and detects anomalies. 

#### Hotelling's T² Statistic

$$ T^2 = \mathbf{t}^\top \mathbf{\Lambda}^{-1} \mathbf{t} $$ 

where \\( \mathbf{t} \\) is the PCA score vector, \\( \mathbf{\Lambda} \\) is the covariance matrix of scores

#### SPE (Squared Prediction Error)

$$ \text{SPE} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 $$ 

\\( \mathbf{x} \\) is the original data, \\( \hat{\mathbf{x}} \\) is the data reconstructed by the PCA model

### 💻 Code Example 3.2: MSPC Control Charts (Hotelling's T² and SPE)
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    class MSPCMonitor:
        """Multivariate Statistical Process Control (MSPC) class"""
    
        def __init__(self, n_components=3, alpha=0.05):
            """
            Args:
                n_components: Number of PCA principal components
                alpha: Significance level (for control limit calculation)
            """
            self.n_components = n_components
            self.alpha = alpha
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=n_components)
            self.T2_limit = None
            self.SPE_limit = None
    
        def fit(self, X_normal):
            """
            Train model with normal operation data
    
            Args:
                X_normal: Process data during normal operation (n_samples × n_features)
            """
            # Standardization
            X_scaled = self.scaler.fit_transform(X_normal)
    
            # Build PCA model
            self.pca.fit(X_scaled)
    
            # Calculate T² and SPE for training data
            T2_train = self._calculate_T2(X_scaled)
            SPE_train = self._calculate_SPE(X_scaled)
    
            # Calculate control limits
            self.T2_limit = self._calculate_T2_limit(len(X_normal))
            self.SPE_limit = self._calculate_SPE_limit(SPE_train)
    
            return T2_train, SPE_train
    
        def _calculate_T2(self, X_scaled):
            """Calculate Hotelling's T² statistic"""
            scores = self.pca.transform(X_scaled)
    
            # Inverse of score covariance matrix
            cov_scores = np.cov(scores.T)
            cov_inv = np.linalg.inv(cov_scores)
    
            # T² calculation
            T2 = np.sum(scores @ cov_inv * scores, axis=1)
    
            return T2
    
        def _calculate_SPE(self, X_scaled):
            """Calculate SPE (Squared Prediction Error)"""
            # Reconstruction by PCA model
            scores = self.pca.transform(X_scaled)
            X_reconstructed = self.pca.inverse_transform(scores)
    
            # SPE calculation
            residuals = X_scaled - X_reconstructed
            SPE = np.sum(residuals ** 2, axis=1)
    
            return SPE
    
        def _calculate_T2_limit(self, n_samples):
            """T² control limit (F-distribution based)"""
            k = self.n_components
            n = n_samples
    
            F_crit = stats.f.ppf(1 - self.alpha, k, n - k)
            T2_limit = (k * (n - 1) / (n - k)) * F_crit
    
            return T2_limit
    
        def _calculate_SPE_limit(self, SPE_train):
            """SPE control limit (empirical method)"""
            # Mean and percentile
            SPE_limit = np.percentile(SPE_train, (1 - self.alpha) * 100)
    
            return SPE_limit
    
        def monitor(self, X_new):
            """
            Monitor new data
    
            Args:
                X_new: Data to be monitored
    
            Returns:
                T2, SPE, anomaly flags
            """
            X_scaled = self.scaler.transform(X_new)
    
            T2 = self._calculate_T2(X_scaled)
            SPE = self._calculate_SPE(X_scaled)
    
            # Anomaly detection
            T2_alarm = T2 > self.T2_limit
            SPE_alarm = SPE > self.SPE_limit
    
            return T2, SPE, T2_alarm, SPE_alarm
    
        def plot_mspc_charts(self, T2, SPE, T2_alarm, SPE_alarm):
            """Visualize MSPC control charts"""
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
            sample_indices = range(len(T2))
    
            # T² control chart
            colors_t2 = ['red' if alarm else '#11998e' for alarm in T2_alarm]
            axes[0].scatter(sample_indices, T2, c=colors_t2, s=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[0].plot(sample_indices, T2, color='#11998e', alpha=0.3, linewidth=1)
            axes[0].axhline(y=self.T2_limit, color='red', linestyle='--', linewidth=2,
                            label=f'Control Limit (T² = {self.T2_limit:.2f})')
    
            axes[0].set_xlabel('Sample Number')
            axes[0].set_ylabel("Hotelling's T²")
            axes[0].set_title("Multivariate Control Chart: Hotelling's T²", fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
    
            # SPE control chart
            colors_spe = ['red' if alarm else '#38ef7d' for alarm in SPE_alarm]
            axes[1].scatter(sample_indices, SPE, c=colors_spe, s=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[1].plot(sample_indices, SPE, color='#38ef7d', alpha=0.3, linewidth=1)
            axes[1].axhline(y=self.SPE_limit, color='red', linestyle='--', linewidth=2,
                            label=f'Control Limit (SPE = {self.SPE_limit:.2f})')
    
            axes[1].set_xlabel('Sample Number')
            axes[1].set_ylabel('SPE (Squared Prediction Error)')
            axes[1].set_title('Multivariate Control Chart: SPE', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('mspc_control_charts.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Execution example
    print("=" * 60)
    print("Multivariate Statistical Process Control (MSPC) System")
    print("=" * 60)
    
    # Generate process data
    np.random.seed(42)
    n_normal = 100
    n_abnormal = 30
    n_features = 6  # Temperature, Pressure, Flow rate, pH, Concentration, Viscosity
    
    # Normal operation data
    mean_normal = [80, 2.0, 100, 6.5, 5.0, 1000]
    cov_normal = np.diag([4, 0.04, 100, 0.09, 0.25, 10000])
    X_normal = np.random.multivariate_normal(mean_normal, cov_normal, n_normal)
    
    # Train MSPC model
    mspc = MSPCMonitor(n_components=3, alpha=0.05)
    T2_train, SPE_train = mspc.fit(X_normal)
    
    print(f"\nPCA Model:")
    print(f"Number of components: {mspc.n_components}")
    print(f"Cumulative variance ratio: {mspc.pca.explained_variance_ratio_.sum():.2%}")
    print(f"\nControl Limits:")
    print(f"T² Limit = {mspc.T2_limit:.2f}")
    print(f"SPE Limit = {mspc.SPE_limit:.2f}")
    
    # Generate monitoring data (normal + abnormal)
    X_monitor = np.vstack([
        X_normal[:50],  # Normal
        np.random.multivariate_normal([85, 2.2, 110, 6.8, 5.5, 1200], cov_normal, n_abnormal)  # Abnormal
    ])
    
    # Execute monitoring
    T2, SPE, T2_alarm, SPE_alarm = mspc.monitor(X_monitor)
    
    # Results summary
    total_alarms = np.sum(T2_alarm | SPE_alarm)
    print(f"\nMonitoring Results:")
    print(f"Total samples: {len(X_monitor)}")
    print(f"T² anomalies: {np.sum(T2_alarm)} cases")
    print(f"SPE anomalies: {np.sum(SPE_alarm)} cases")
    print(f"Total anomalies detected: {total_alarms} cases")
    
    # Visualization
    mspc.plot_mspc_charts(T2, SPE, T2_alarm, SPE_alarm)
    

**Implementation Points:**

  * Dimensionality reduction and anomaly detection of multivariate data using PCA
  * Comprehensive process evaluation using Hotelling's T² statistic
  * Monitoring model fit using SPE
  * Statistical control limit setting based on F-distribution
  * Real-time monitoring and alarm functionality

## 📚 Summary

In this chapter, we learned about PAT and real-time quality control.

### Key Points

  * Real-time quality measurement using NIR/Raman spectroscopy
  * Building quantitative models using PLS regression
  * Implementation of Multivariate Statistical Process Control (MSPC)
  * Anomaly detection using Hotelling's T² and SPE control charts
  * Deepening process understanding through integration of PAT tools

**🎯 Next Chapter Preview**  
Chapter 4 will cover the transition from batch production to continuous production and Quality by Design (QbD) implementation. You will master more strategic quality control methods including DoE (Design of Experiments), design space, and risk-based approaches. 

[← Chapter 2: Electronic Batch Record Analysis](<chapter-2.html>) [Chapter 4: Continuous Production and QbD Implementation →](<chapter-4.html>)

## References

  1. Montgomery, D. C. (2019). _Design and Analysis of Experiments_ (9th ed.). Wiley.
  2. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley.
  3. Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). _Process Dynamics and Control_ (4th ed.). Wiley.
  4. McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." _Technometrics_ , 42(1), 55-61.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
