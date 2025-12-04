---
title: 第3章 PATとリアルタイム品質管理
chapter_title: 第3章 PATとリアルタイム品質管理
subtitle: Process Analytical Technology and Real-Time Quality Control
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/pharma-manufacturing-ai/chapter-3.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Pharma Manufacturing Ai](<../../PI/pharma-manufacturing-ai/index.html>)›Chapter 3

[← シリーズ目次に戻る](<index.html>)

## 📖 本章の概要

プロセス分析技術（PAT: Process Analytical Technology）は、FDA推奨のリアルタイム品質管理手法です。 本章では、NIR/Raman分光法などのPATツール、多変量統計的プロセス管理（MSPC）、 リアルタイムリリース試験（RTRT）の実装方法を学び、品質の作り込み（Quality by Design）を実現します。 

### 🎯 学習目標

  * PAT（Process Analytical Technology）の基本概念とFDAガイダンス
  * NIR/Raman分光データの前処理と特徴抽出
  * PLS（Partial Least Squares）回帰による定量分析
  * 多変量統計的プロセス管理（MSPC）の実装
  * Hotelling's T²とSPE管理図の構築
  * リアルタイムリリース試験（RTRT）の設計
  * PATシステムのバリデーション戦略

## 🔬 3.1 PAT（Process Analytical Technology）の基礎

### FDAのPATイニシアチブ

FDA（米国食品医薬品局）は2004年にPATガイダンスを発行し、 「品質をテストするのではなく、品質をプロセスに作り込む」という パラダイムシフトを推進しています。 

**🏭 PAT 4つのツール**  
1\. **多変量ツール** : PCA、PLS、ニューラルネットワーク  
2\. **プロセス分析器** : NIR、Raman、UV-Vis分光計  
3\. **プロセス制御ツール** : フィードバック制御、適応制御  
4\. **継続的改善・知識管理** : データベース、統計解析 

### NIR/Raman分光法の原理

  * **NIR（近赤外分光法）** : 非破壊、固体・液体測定可能、水分・含量測定に有効
  * **Raman分光法** : 分子振動スペクトル、水の影響少ない、結晶多形判定に有効

### 💻 コード例3.1: NIRスペクトルデータの前処理とPLS回帰
    
    
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
        """NIR分光データ解析クラス"""
    
        def __init__(self, wavelengths):
            """
            Args:
                wavelengths: 波長配列（nm）
            """
            self.wavelengths = wavelengths
            self.scaler = StandardScaler()
            self.pls_model = None
    
        def generate_nir_spectra(self, n_samples=100):
            """NIRスペクトルデータの生成（模擬）"""
            np.random.seed(42)
    
            # APIの含量（85-115%の範囲）
            api_content = np.random.uniform(85, 115, n_samples)
    
            spectra = []
            for content in api_content:
                # ベースラインスペクトル
                baseline = 0.5 + 0.001 * self.wavelengths
    
                # APIの吸収ピーク（1450nm, 1900nm付近）
                peak1 = 0.3 * (content / 100) * np.exp(-((self.wavelengths - 1450) ** 2) / (50 ** 2))
                peak2 = 0.2 * (content / 100) * np.exp(-((self.wavelengths - 1900) ** 2) / (80 ** 2))
    
                # 賦形剤の影響
                excipient = 0.1 * np.exp(-((self.wavelengths - 1700) ** 2) / (100 ** 2))
    
                # ノイズ
                noise = np.random.normal(0, 0.01, len(self.wavelengths))
    
                spectrum = baseline + peak1 + peak2 + excipient + noise
                spectra.append(spectrum)
    
            return np.array(spectra), api_content
    
        def preprocess_spectra(self, spectra, method='snv'):
            """
            スペクトルの前処理
    
            Args:
                spectra: スペクトルデータ（n_samples × n_wavelengths）
                method: 前処理方法 ('snv', 'msc', 'derivative')
    
            Returns:
                前処理後のスペクトル
            """
            if method == 'snv':
                # Standard Normal Variate（SNV）
                mean = np.mean(spectra, axis=1, keepdims=True)
                std = np.std(spectra, axis=1, keepdims=True)
                processed = (spectra - mean) / std
    
            elif method == 'msc':
                # Multiplicative Scatter Correction（MSC）
                ref_spectrum = np.mean(spectra, axis=0)
                processed = np.zeros_like(spectra)
    
                for i in range(spectra.shape[0]):
                    # 線形回帰でスケーリング・オフセット除去
                    fit = np.polyfit(ref_spectrum, spectra[i], 1)
                    processed[i] = (spectra[i] - fit[1]) / fit[0]
    
            elif method == 'derivative':
                # Savitzky-Golay 1次微分
                processed = np.array([savgol_filter(s, window_length=11, polyorder=2, deriv=1)
                                      for s in spectra])
    
            else:
                processed = spectra
    
            return processed
    
        def build_pls_model(self, X_train, y_train, n_components=5):
            """
            PLSモデルの構築
    
            Args:
                X_train: 訓練スペクトルデータ
                y_train: 訓練ラベル（API含量）
                n_components: PLS成分数
            """
            # データの標準化
            X_scaled = self.scaler.fit_transform(X_train)
    
            # PLSモデル
            self.pls_model = PLSRegression(n_components=n_components)
            self.pls_model.fit(X_scaled, y_train)
    
            # クロスバリデーション
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.pls_model, X_scaled, y_train,
                                         cv=kfold, scoring='r2')
    
            return cv_scores
    
        def predict(self, X_test):
            """予測"""
            X_scaled = self.scaler.transform(X_test)
            return self.pls_model.predict(X_scaled)
    
        def plot_nir_analysis(self, spectra, api_content, X_test, y_test, y_pred):
            """NIR解析結果の可視化"""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            # NIRスペクトル（サンプル）
            for i in range(0, len(spectra), 20):
                axes[0, 0].plot(self.wavelengths, spectra[i], alpha=0.6,
                                label=f'{api_content[i]:.1f}%' if i < 80 else None)
    
            axes[0, 0].set_xlabel('波長（nm）')
            axes[0, 0].set_ylabel('吸光度')
            axes[0, 0].set_title('NIRスペクトル（生データ）', fontsize=12, fontweight='bold')
            axes[0, 0].legend(fontsize=8, loc='upper right')
            axes[0, 0].grid(alpha=0.3)
    
            # 前処理後のスペクトル
            processed = self.preprocess_spectra(spectra, method='snv')
            for i in range(0, len(processed), 20):
                axes[0, 1].plot(self.wavelengths, processed[i], alpha=0.6)
    
            axes[0, 1].set_xlabel('波長（nm）')
            axes[0, 1].set_ylabel('SNV処理後の吸光度')
            axes[0, 1].set_title('NIRスペクトル（SNV前処理後）', fontsize=12, fontweight='bold')
            axes[0, 1].grid(alpha=0.3)
    
            # PLS予測精度
            axes[1, 0].scatter(y_test, y_pred, alpha=0.6, s=50, color='#11998e')
            axes[1, 0].plot([85, 115], [85, 115], 'r--', linewidth=2, label='理想直線')
    
            # ±5%の許容範囲
            axes[1, 0].plot([85, 115], [90, 120], 'orange', linestyle=':', linewidth=1.5, alpha=0.7)
            axes[1, 0].plot([85, 115], [80, 110], 'orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
            # R²とRMSEの計算
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
            axes[1, 0].text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}%',
                            transform=axes[1, 0].transAxes, fontsize=11,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
            axes[1, 0].set_xlabel('実測API含量（%）')
            axes[1, 0].set_ylabel('予測API含量（%）')
            axes[1, 0].set_title('PLSモデル予測精度', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
    
            # 予測誤差の分布
            errors = y_pred.flatten() - y_test
            axes[1, 1].hist(errors, bins=20, color='#38ef7d', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='誤差ゼロ')
            axes[1, 1].set_xlabel('予測誤差（%）')
            axes[1, 1].set_ylabel('頻度')
            axes[1, 1].set_title('予測誤差分布', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('nir_pls_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 実行例
    print("=" * 60)
    print("NIR-PLS分析システム（PAT実装）")
    print("=" * 60)
    
    # 波長配列の定義（1100-2500nm、2nm間隔）
    wavelengths = np.arange(1100, 2501, 2)
    
    # NIRアナライザーの初期化
    nir_analyzer = NIRAnalyzer(wavelengths)
    
    # NIRスペクトルデータの生成
    spectra, api_content = nir_analyzer.generate_nir_spectra(n_samples=100)
    
    print(f"\nサンプル数: {len(spectra)}")
    print(f"波長ポイント数: {len(wavelengths)}")
    print(f"波長範囲: {wavelengths[0]}-{wavelengths[-1]} nm")
    print(f"API含量範囲: {api_content.min():.1f}-{api_content.max():.1f}%")
    
    # 訓練/テストデータの分割
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, api_content, test_size=0.3, random_state=42
    )
    
    # スペクトルの前処理
    X_train_processed = nir_analyzer.preprocess_spectra(X_train, method='snv')
    X_test_processed = nir_analyzer.preprocess_spectra(X_test, method='snv')
    
    # PLSモデルの構築
    cv_scores = nir_analyzer.build_pls_model(X_train_processed, y_train, n_components=5)
    
    print(f"\nPLSモデル（5成分）:")
    print(f"クロスバリデーション R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 予測
    y_pred = nir_analyzer.predict(X_test_processed)
    
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nテストセット性能:")
    print(f"R² = {r2:.4f}")
    print(f"RMSE = {rmse:.2f}%")
    print(f"相対誤差 = {rmse / api_content.mean() * 100:.2f}%")
    
    # 可視化
    nir_analyzer.plot_nir_analysis(spectra, api_content, X_test_processed, y_test, y_pred)
    

**実装のポイント:**

  * NIRスペクトルの前処理（SNV、MSC、微分）による散乱除去
  * PLSによる多変量回帰モデル構築
  * クロスバリデーションによるモデル評価
  * リアルタイムAPI含量予測の実現
  * 予測精度の定量評価（R²、RMSE）

## 📊 3.2 多変量統計的プロセス管理（MSPC）

### MSPCの原理

多変量統計的プロセス管理（MSPC）は、複数のプロセス変数を統合的に監視する手法です。 主成分分析（PCA）を用いて、正常運転時のデータ空間を学習し、 異常を検出します。 

#### Hotelling's T²統計量

$$ T^2 = \mathbf{t}^\top \mathbf{\Lambda}^{-1} \mathbf{t} $$ 

ここで、\\( \mathbf{t} \\) はPCAスコアベクトル、\\( \mathbf{\Lambda} \\) はスコアの共分散行列

#### SPE（Squared Prediction Error）

$$ \text{SPE} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 $$ 

\\( \mathbf{x} \\) は元データ、\\( \hat{\mathbf{x}} \\) はPCAモデルによる再構成データ

### 💻 コード例3.2: MSPC管理図（Hotelling's T²とSPE）
    
    
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
        """多変量統計的プロセス管理（MSPC）クラス"""
    
        def __init__(self, n_components=3, alpha=0.05):
            """
            Args:
                n_components: PCA主成分数
                alpha: 有意水準（管理限界計算用）
            """
            self.n_components = n_components
            self.alpha = alpha
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=n_components)
            self.T2_limit = None
            self.SPE_limit = None
    
        def fit(self, X_normal):
            """
            正常運転データでモデル訓練
    
            Args:
                X_normal: 正常運転時のプロセスデータ（n_samples × n_features）
            """
            # 標準化
            X_scaled = self.scaler.fit_transform(X_normal)
    
            # PCAモデル構築
            self.pca.fit(X_scaled)
    
            # 訓練データのT²とSPE計算
            T2_train = self._calculate_T2(X_scaled)
            SPE_train = self._calculate_SPE(X_scaled)
    
            # 管理限界の計算
            self.T2_limit = self._calculate_T2_limit(len(X_normal))
            self.SPE_limit = self._calculate_SPE_limit(SPE_train)
    
            return T2_train, SPE_train
    
        def _calculate_T2(self, X_scaled):
            """Hotelling's T²統計量の計算"""
            scores = self.pca.transform(X_scaled)
    
            # スコアの共分散行列の逆行列
            cov_scores = np.cov(scores.T)
            cov_inv = np.linalg.inv(cov_scores)
    
            # T²計算
            T2 = np.sum(scores @ cov_inv * scores, axis=1)
    
            return T2
    
        def _calculate_SPE(self, X_scaled):
            """SPE（二乗予測誤差）の計算"""
            # PCAモデルによる再構成
            scores = self.pca.transform(X_scaled)
            X_reconstructed = self.pca.inverse_transform(scores)
    
            # SPE計算
            residuals = X_scaled - X_reconstructed
            SPE = np.sum(residuals ** 2, axis=1)
    
            return SPE
    
        def _calculate_T2_limit(self, n_samples):
            """T²管理限界（F分布ベース）"""
            k = self.n_components
            n = n_samples
    
            F_crit = stats.f.ppf(1 - self.alpha, k, n - k)
            T2_limit = (k * (n - 1) / (n - k)) * F_crit
    
            return T2_limit
    
        def _calculate_SPE_limit(self, SPE_train):
            """SPE管理限界（経験的方法）"""
            # 平均とパーセンタイル
            SPE_limit = np.percentile(SPE_train, (1 - self.alpha) * 100)
    
            return SPE_limit
    
        def monitor(self, X_new):
            """
            新規データの監視
    
            Args:
                X_new: 監視対象データ
    
            Returns:
                T2, SPE, 異常フラグ
            """
            X_scaled = self.scaler.transform(X_new)
    
            T2 = self._calculate_T2(X_scaled)
            SPE = self._calculate_SPE(X_scaled)
    
            # 異常判定
            T2_alarm = T2 > self.T2_limit
            SPE_alarm = SPE > self.SPE_limit
    
            return T2, SPE, T2_alarm, SPE_alarm
    
        def plot_mspc_charts(self, T2, SPE, T2_alarm, SPE_alarm):
            """MSPC管理図の可視化"""
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
            sample_indices = range(len(T2))
    
            # T²管理図
            colors_t2 = ['red' if alarm else '#11998e' for alarm in T2_alarm]
            axes[0].scatter(sample_indices, T2, c=colors_t2, s=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[0].plot(sample_indices, T2, color='#11998e', alpha=0.3, linewidth=1)
            axes[0].axhline(y=self.T2_limit, color='red', linestyle='--', linewidth=2,
                            label=f'管理限界 (T² = {self.T2_limit:.2f})')
    
            axes[0].set_xlabel('サンプル番号')
            axes[0].set_ylabel("Hotelling's T²")
            axes[0].set_title("多変量管理図: Hotelling's T²", fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
    
            # SPE管理図
            colors_spe = ['red' if alarm else '#38ef7d' for alarm in SPE_alarm]
            axes[1].scatter(sample_indices, SPE, c=colors_spe, s=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[1].plot(sample_indices, SPE, color='#38ef7d', alpha=0.3, linewidth=1)
            axes[1].axhline(y=self.SPE_limit, color='red', linestyle='--', linewidth=2,
                            label=f'管理限界 (SPE = {self.SPE_limit:.2f})')
    
            axes[1].set_xlabel('サンプル番号')
            axes[1].set_ylabel('SPE（二乗予測誤差）')
            axes[1].set_title('多変量管理図: SPE', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('mspc_control_charts.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 実行例
    print("=" * 60)
    print("多変量統計的プロセス管理（MSPC）システム")
    print("=" * 60)
    
    # プロセスデータの生成
    np.random.seed(42)
    n_normal = 100
    n_abnormal = 30
    n_features = 6  # 温度、圧力、流量、pH、濃度、粘度
    
    # 正常運転データ
    mean_normal = [80, 2.0, 100, 6.5, 5.0, 1000]
    cov_normal = np.diag([4, 0.04, 100, 0.09, 0.25, 10000])
    X_normal = np.random.multivariate_normal(mean_normal, cov_normal, n_normal)
    
    # MSPCモデルの訓練
    mspc = MSPCMonitor(n_components=3, alpha=0.05)
    T2_train, SPE_train = mspc.fit(X_normal)
    
    print(f"\nPCAモデル:")
    print(f"主成分数: {mspc.n_components}")
    print(f"累積寄与率: {mspc.pca.explained_variance_ratio_.sum():.2%}")
    print(f"\n管理限界:")
    print(f"T² 限界 = {mspc.T2_limit:.2f}")
    print(f"SPE 限界 = {mspc.SPE_limit:.2f}")
    
    # 監視データの生成（正常 + 異常）
    X_monitor = np.vstack([
        X_normal[:50],  # 正常
        np.random.multivariate_normal([85, 2.2, 110, 6.8, 5.5, 1200], cov_normal, n_abnormal)  # 異常
    ])
    
    # 監視実行
    T2, SPE, T2_alarm, SPE_alarm = mspc.monitor(X_monitor)
    
    # 結果サマリー
    total_alarms = np.sum(T2_alarm | SPE_alarm)
    print(f"\n監視結果:")
    print(f"総サンプル数: {len(X_monitor)}")
    print(f"T²異常: {np.sum(T2_alarm)} 件")
    print(f"SPE異常: {np.sum(SPE_alarm)} 件")
    print(f"総異常検出数: {total_alarms} 件")
    
    # 可視化
    mspc.plot_mspc_charts(T2, SPE, T2_alarm, SPE_alarm)
    

**実装のポイント:**

  * PCAによる多変量データの次元削減と異常検出
  * Hotelling's T²統計量による総合的なプロセス評価
  * SPEによるモデル適合度の監視
  * F分布ベースの統計的管理限界設定
  * リアルタイム監視とアラーム機能

## 📚 まとめ

本章では、PATとリアルタイム品質管理について学びました。

### 主要なポイント

  * NIR/Raman分光法によるリアルタイム品質測定
  * PLS回帰による定量モデルの構築
  * 多変量統計的プロセス管理（MSPC）の実装
  * Hotelling's T²とSPE管理図による異常検出
  * PATツールの統合によるプロセス理解の深化

**🎯 次章予告**  
第4章では、バッチ生産から連続生産への移行とQbD（Quality by Design）実装について学びます。 DoE（実験計画法）、デザインスペース、リスクベースアプローチなど、 より戦略的な品質管理手法を習得します。 

[← 第2章: 電子バッチ記録解析](<chapter-2.html>) 第4章: 連続生産とQbD実装 →（準備中）

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
