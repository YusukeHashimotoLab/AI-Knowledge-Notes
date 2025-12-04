---
title: 第4章：プロセスデータの特徴量エンジニアリング
chapter_title: 第4章：プロセスデータの特徴量エンジニアリング
subtitle: 時系列データから価値ある特徴を抽出する技術
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 時間領域特徴量（統計量、モーメント）の計算と解釈ができる
  * ✅ 周波数領域特徴量（FFT、スペクトルエントロピー）を抽出できる
  * ✅ ウェーブレット変換による多解像度特徴量を理解する
  * ✅ ドメイン知識を活用した特徴量設計ができる
  * ✅ 特徴量選択手法（RFE、LASSO、相互情報量）を実装できる
  * ✅ tsfreshによる自動特徴量抽出を活用できる

* * *

## 4.1 特徴量エンジニアリングの重要性

### なぜ特徴量エンジニアリングが重要か

プロセスデータ解析において、**特徴量エンジニアリング** は機械学習モデルの性能を決定する最も重要な要素です。生の時系列データから、プロセスの状態を適切に表現する特徴量を抽出することで、予測精度が大幅に向上します。

特徴量タイプ | 情報内容 | 適用例  
---|---|---  
時間領域特徴 | 統計的性質、トレンド | 平均温度、標準偏差、傾き  
周波数領域特徴 | 周期性、振動成分 | FFT係数、スペクトル密度  
ウェーブレット特徴 | 時間-周波数局在 | 過渡現象、突発イベント  
ドメイン特徴 | プロセス固有の知識 | 滞留時間、転化率、収率  
相互作用特徴 | 変数間の関係 | 温度×圧力、比率特徴  
  
* * *

## 4.2 時間領域特徴量

### コード例1: 基本統計量特徴の抽出
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    
    def extract_time_domain_features(data, window_size=100):
        """
        時間領域の基本統計特徴を抽出
    
        Parameters:
        -----------
        data : array-like
            時系列データ
        window_size : int
            特徴抽出用のウィンドウサイズ
    
        Returns:
        --------
        features : dict
            抽出された特徴量の辞書
        """
        features = {}
    
        # 基本統計量
        features['mean'] = np.mean(data)
        features['std'] = np.std(data)
        features['variance'] = np.var(data)
        features['min'] = np.min(data)
        features['max'] = np.max(data)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(data)
    
        # パーセンタイル
        features['q25'] = np.percentile(data, 25)
        features['q75'] = np.percentile(data, 75)
        features['iqr'] = features['q75'] - features['q25']
    
        # 高次モーメント
        features['skewness'] = stats.skew(data)  # 歪度
        features['kurtosis'] = stats.kurtosis(data)  # 尖度
    
        # エネルギーとパワー
        features['energy'] = np.sum(data ** 2)
        features['power'] = features['energy'] / len(data)
        features['rms'] = np.sqrt(features['power'])  # Root Mean Square
    
        # 変動係数
        features['cv'] = features['std'] / features['mean'] if features['mean'] != 0 else 0
    
        return features
    
    
    # サンプルデータ生成（反応器温度データのシミュレーション）
    np.random.seed(42)
    time = np.linspace(0, 100, 1000)
    temperature = 180 + 5 * np.sin(0.1 * time) + np.random.normal(0, 1, 1000)
    
    # 特徴抽出
    features = extract_time_domain_features(temperature)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 時系列データ
    axes[0, 0].plot(time, temperature, color='#11998e', linewidth=1.5)
    axes[0, 0].axhline(y=features['mean'], color='red', linestyle='--',
                        label=f"Mean: {features['mean']:.2f}")
    axes[0, 0].axhline(y=features['mean'] + features['std'], color='orange',
                        linestyle='--', alpha=0.7, label=f"±1σ")
    axes[0, 0].axhline(y=features['mean'] - features['std'], color='orange',
                        linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Temperature [°C]')
    axes[0, 0].set_title('Time Series with Statistical Features')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # ヒストグラム
    axes[0, 1].hist(temperature, bins=50, color='#11998e', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=features['mean'], color='red', linestyle='--',
                        linewidth=2, label=f"Mean: {features['mean']:.2f}")
    axes[0, 1].axvline(x=features['median'], color='blue', linestyle='--',
                        linewidth=2, label=f"Median: {features['median']:.2f}")
    axes[0, 1].set_xlabel('Temperature [°C]')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Distribution (Skewness: {features["skewness"]:.2f})')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 特徴量の可視化
    feature_names = ['mean', 'std', 'variance', 'range', 'iqr',
                     'skewness', 'kurtosis', 'rms']
    feature_values = [features[name] for name in feature_names]
    
    axes[1, 0].barh(feature_names, feature_values, color='#38ef7d', edgecolor='black')
    axes[1, 0].set_xlabel('Feature Value')
    axes[1, 0].set_title('Extracted Time-Domain Features')
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # ボックスプロット
    axes[1, 1].boxplot(temperature, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='#c8e6c9', color='black'),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'))
    axes[1, 1].set_ylabel('Temperature [°C]')
    axes[1, 1].set_title('Box Plot Summary')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # 特徴量の出力
    print("Extracted Time-Domain Features:")
    print("=" * 50)
    for key, value in features.items():
        print(f"{key:15s}: {value:10.4f}")
    

**出力例:**
    
    
    Extracted Time-Domain Features:
    ==================================================
    mean           :   179.9682
    std            :     5.1558
    variance       :    26.5824
    min            :   164.9012
    max            :   194.8533
    range          :    29.9521
    median         :   179.9449
    q25            :   176.5168
    q75            :   183.4302
    iqr            :     6.9134
    skewness       :     0.0187
    kurtosis       :    -0.1023
    energy         : 32405692.8145
    power          : 32405.6928
    rms            :   180.0158
    cv             :     0.0286
    

**解説:** 時間領域特徴は、データの統計的性質を捉えます。平均と標準偏差は中心傾向と散らばりを、歪度と尖度は分布の形状を表します。これらは異常検知や状態分類に有効です。

* * *

### コード例2: ローリング統計特徴とラグ特徴
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    def create_rolling_features(data, windows=[10, 30, 60]):
        """
        ローリング統計特徴とラグ特徴を生成
    
        Parameters:
        -----------
        data : pd.Series
            時系列データ
        windows : list
            ローリングウィンドウサイズのリスト
    
        Returns:
        --------
        df : pd.DataFrame
            特徴量を含むDataFrame
        """
        df = pd.DataFrame({'original': data})
    
        for window in windows:
            # ローリング統計量
            df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = data.rolling(window=window).std()
            df[f'rolling_min_{window}'] = data.rolling(window=window).min()
            df[f'rolling_max_{window}'] = data.rolling(window=window).max()
            df[f'rolling_median_{window}'] = data.rolling(window=window).median()
    
        # ラグ特徴（過去の値）
        for lag in [1, 5, 10, 20]:
            df[f'lag_{lag}'] = data.shift(lag)
    
        # 差分特徴
        df['diff_1'] = data.diff(1)  # 1次差分
        df['diff_2'] = data.diff(2)  # 2次差分
    
        # 勾配（傾き）
        df['gradient'] = np.gradient(data)
    
        return df
    
    
    # サンプルデータ（反応器圧力のトレンド付きデータ）
    np.random.seed(42)
    time = np.arange(500)
    pressure = 2.5 + 0.001 * time + 0.1 * np.sin(0.05 * time) + np.random.normal(0, 0.05, 500)
    pressure_series = pd.Series(pressure)
    
    # ローリング特徴の生成
    df_features = create_rolling_features(pressure_series)
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # オリジナルデータとローリング平均
    axes[0].plot(time, df_features['original'], label='Original',
                 color='gray', alpha=0.5, linewidth=1)
    axes[0].plot(time, df_features['rolling_mean_10'], label='Rolling Mean (10)',
                 color='#11998e', linewidth=2)
    axes[0].plot(time, df_features['rolling_mean_30'], label='Rolling Mean (30)',
                 color='#38ef7d', linewidth=2)
    axes[0].plot(time, df_features['rolling_mean_60'], label='Rolling Mean (60)',
                 color='orange', linewidth=2)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Pressure [bar]')
    axes[0].set_title('Rolling Mean Features')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # ローリング標準偏差（変動性の検出）
    axes[1].plot(time, df_features['rolling_std_10'], label='Rolling Std (10)',
                 color='#11998e', linewidth=2)
    axes[1].plot(time, df_features['rolling_std_30'], label='Rolling Std (30)',
                 color='#38ef7d', linewidth=2)
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Std [bar]')
    axes[1].set_title('Rolling Standard Deviation (Variability Detection)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 差分とグラデーション（トレンド検出）
    axes[2].plot(time, df_features['diff_1'], label='1st Difference',
                 color='#11998e', alpha=0.7, linewidth=1)
    axes[2].plot(time, df_features['gradient'], label='Gradient',
                 color='orange', linewidth=2)
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Change Rate')
    axes[2].set_title('Trend Detection Features')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 特徴量の一部を表示
    print("\nSample of Rolling Features:")
    print(df_features.iloc[60:70][['original', 'rolling_mean_30', 'rolling_std_30',
                                     'lag_10', 'diff_1', 'gradient']].to_string())
    

**解説:** ローリング統計は時間的な文脈を捉え、短期的な変動や長期的なトレンドを検出できます。ラグ特徴は時系列の自己相関を捉え、差分特徴はトレンドや変化点を強調します。

* * *

## 4.3 周波数領域特徴量

### コード例3: FFT（高速フーリエ変換）による周波数特徴抽出
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    from scipy.signal import welch
    
    def extract_frequency_features(signal, sampling_rate=1.0, n_coeffs=10):
        """
        FFTによる周波数領域特徴を抽出
    
        Parameters:
        -----------
        signal : array-like
            時系列信号
        sampling_rate : float
            サンプリングレート [Hz]
        n_coeffs : int
            抽出するFFT係数の数
    
        Returns:
        --------
        features : dict
            周波数領域特徴
        """
        n = len(signal)
    
        # FFT計算
        fft_vals = fft(signal)
        fft_magnitude = np.abs(fft_vals)[:n//2]  # 正の周波数のみ
        fft_power = fft_magnitude ** 2
        frequencies = fftfreq(n, d=1/sampling_rate)[:n//2]
    
        features = {}
    
        # 主要なFFT係数（低周波成分）
        for i in range(n_coeffs):
            features[f'fft_coeff_{i}'] = fft_magnitude[i]
    
        # スペクトルエネルギー
        features['spectral_energy'] = np.sum(fft_power)
    
        # スペクトル重心（周波数の重心）
        features['spectral_centroid'] = np.sum(frequencies * fft_magnitude) / np.sum(fft_magnitude)
    
        # スペクトル分散
        features['spectral_variance'] = np.sum(((frequencies - features['spectral_centroid']) ** 2) * fft_magnitude) / np.sum(fft_magnitude)
    
        # スペクトルエントロピー
        psd_norm = fft_power / np.sum(fft_power)
        psd_norm = psd_norm[psd_norm > 0]  # ゼロを除去
        features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
    
        # 支配的な周波数（パワー最大のピーク）
        dominant_freq_idx = np.argmax(fft_magnitude[1:]) + 1  # DC成分を除く
        features['dominant_frequency'] = frequencies[dominant_freq_idx]
        features['dominant_power'] = fft_magnitude[dominant_freq_idx]
    
        return features, frequencies, fft_magnitude
    
    
    # サンプルデータ（周期的な振動を含む信号）
    np.random.seed(42)
    sampling_rate = 100  # Hz
    duration = 10  # 秒
    time = np.linspace(0, duration, sampling_rate * duration)
    
    # 複数の周波数成分を含む信号
    signal = (2.0 * np.sin(2 * np.pi * 5 * time) +    # 5 Hz成分
              1.0 * np.sin(2 * np.pi * 12 * time) +   # 12 Hz成分
              0.5 * np.sin(2 * np.pi * 25 * time) +   # 25 Hz成分
              0.3 * np.random.randn(len(time)))       # ノイズ
    
    # 特徴抽出
    features, frequencies, fft_magnitude = extract_frequency_features(signal, sampling_rate)
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 時間領域の信号
    axes[0].plot(time, signal, color='#11998e', linewidth=1.5)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time-Domain Signal (5Hz + 12Hz + 25Hz components)')
    axes[0].grid(alpha=0.3)
    
    # 周波数スペクトル
    axes[1].plot(frequencies, fft_magnitude, color='#11998e', linewidth=2)
    axes[1].axvline(x=features['dominant_frequency'], color='red', linestyle='--',
                    linewidth=2, label=f"Dominant: {features['dominant_frequency']:.1f} Hz")
    axes[1].axvline(x=features['spectral_centroid'], color='orange', linestyle='--',
                    linewidth=2, label=f"Centroid: {features['spectral_centroid']:.1f} Hz")
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Frequency Spectrum (FFT)')
    axes[1].set_xlim(0, 50)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 特徴量の出力
    print("\nExtracted Frequency-Domain Features:")
    print("=" * 50)
    for key in ['spectral_energy', 'spectral_centroid', 'spectral_variance',
                'spectral_entropy', 'dominant_frequency', 'dominant_power']:
        print(f"{key:25s}: {features[key]:12.4f}")
    
    print("\nTop 5 FFT Coefficients:")
    for i in range(5):
        print(f"fft_coeff_{i:2d}: {features[f'fft_coeff_{i}']:12.4f}")
    

**解説:** FFTは時系列データを周波数成分に分解し、周期的なパターンを検出します。スペクトル重心やエントロピーは、信号の周波数特性を要約する有用な特徴量です。プロセスデータでは、ポンプやコンプレッサーの振動周波数、制御ループの振動などを検出できます。

* * *

### コード例4: パワースペクトル密度とスペクトログラム
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import welch, spectrogram
    
    def analyze_power_spectrum(signal, sampling_rate=1.0):
        """
        パワースペクトル密度の解析
    
        Parameters:
        -----------
        signal : array-like
            時系列信号
        sampling_rate : float
            サンプリングレート [Hz]
    
        Returns:
        --------
        frequencies : array
            周波数配列
        psd : array
            パワースペクトル密度
        """
        # Welch法によるパワースペクトル密度推定
        frequencies, psd = welch(signal, fs=sampling_rate, nperseg=256)
    
        # パワーの累積分布
        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]
    
        # 90%パワーを含む周波数帯域
        freq_90 = frequencies[np.where(cumulative_power >= 0.9 * total_power)[0][0]]
    
        features = {
            'total_power': total_power,
            'freq_90_power': freq_90,
            'peak_frequency': frequencies[np.argmax(psd)]
        }
    
        return frequencies, psd, features
    
    
    # 非定常信号のサンプル（周波数が時間変化）
    sampling_rate = 200
    duration = 5
    time = np.linspace(0, duration, sampling_rate * duration)
    
    # チャープ信号（周波数が時間とともに増加）
    frequency_sweep = np.linspace(5, 50, len(time))
    signal_nonstationary = np.sin(2 * np.pi * frequency_sweep * time)
    
    # パワースペクトル解析
    frequencies, psd, psd_features = analyze_power_spectrum(signal_nonstationary, sampling_rate)
    
    # スペクトログラム（時間-周波数解析）
    f_spec, t_spec, Sxx = spectrogram(signal_nonstationary, fs=sampling_rate,
                                       nperseg=128, noverlap=64)
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 時間領域信号
    axes[0].plot(time, signal_nonstationary, color='#11998e', linewidth=1.5)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Non-Stationary Signal (Frequency Sweep: 5-50 Hz)')
    axes[0].grid(alpha=0.3)
    
    # パワースペクトル密度
    axes[1].semilogy(frequencies, psd, color='#11998e', linewidth=2)
    axes[1].axvline(x=psd_features['peak_frequency'], color='red', linestyle='--',
                    linewidth=2, label=f"Peak: {psd_features['peak_frequency']:.1f} Hz")
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('Power Spectral Density [V²/Hz]')
    axes[1].set_title('Power Spectral Density (Welch Method)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # スペクトログラム
    im = axes[2].pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx),
                             shading='gouraud', cmap='viridis')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Frequency [Hz]')
    axes[2].set_title('Spectrogram (Time-Frequency Analysis)')
    axes[2].set_ylim(0, 60)
    cbar = plt.colorbar(im, ax=axes[2])
    cbar.set_label('Power [dB]')
    
    plt.tight_layout()
    plt.show()
    
    print("\nPower Spectrum Features:")
    print("=" * 50)
    for key, value in psd_features.items():
        print(f"{key:20s}: {value:12.4f}")
    

**解説:** パワースペクトル密度（PSD）は周波数ごとのパワー分布を示し、スペクトログラムは時間変化する周波数成分を可視化します。非定常信号の解析に有効で、プロセスの動的な挙動を把握できます。

* * *

## 4.4 ウェーブレット変換特徴

### コード例5: ウェーブレット変換による多解像度特徴抽出
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pywt
    
    def extract_wavelet_features(signal, wavelet='db4', level=4):
        """
        ウェーブレット変換による特徴抽出
    
        Parameters:
        -----------
        signal : array-like
            時系列信号
        wavelet : str
            ウェーブレット関数の種類 ('db4', 'sym5', 'coif3'など)
        level : int
            分解レベル
    
        Returns:
        --------
        features : dict
            ウェーブレット特徴
        coeffs : list
            ウェーブレット係数
        """
        # ウェーブレット分解
        coeffs = pywt.wavedec(signal, wavelet, level=level)
    
        features = {}
    
        # 各レベルの係数から特徴を抽出
        for i, coeff in enumerate(coeffs):
            prefix = 'approx' if i == 0 else f'detail_{i}'
    
            # エネルギー
            features[f'{prefix}_energy'] = np.sum(coeff ** 2)
    
            # 平均絶対値
            features[f'{prefix}_mean_abs'] = np.mean(np.abs(coeff))
    
            # 標準偏差
            features[f'{prefix}_std'] = np.std(coeff)
    
            # エントロピー
            coeff_norm = (coeff ** 2) / np.sum(coeff ** 2)
            coeff_norm = coeff_norm[coeff_norm > 0]
            features[f'{prefix}_entropy'] = -np.sum(coeff_norm * np.log2(coeff_norm))
    
        # 全エネルギーに対する各レベルの比率
        total_energy = sum([np.sum(c ** 2) for c in coeffs])
        for i, coeff in enumerate(coeffs):
            prefix = 'approx' if i == 0 else f'detail_{i}'
            features[f'{prefix}_energy_ratio'] = np.sum(coeff ** 2) / total_energy
    
        return features, coeffs
    
    
    # サンプルデータ（過渡現象を含む信号）
    np.random.seed(42)
    time = np.linspace(0, 10, 1000)
    
    # ベース信号 + 突発イベント
    signal = np.sin(2 * np.pi * 2 * time) + 0.2 * np.random.randn(1000)
    # 3秒と7秒の位置に突発イベント（スパイク）
    signal[300:320] += 5 * np.exp(-0.5 * ((np.arange(20) - 10) / 3)**2)
    signal[700:720] += -4 * np.exp(-0.5 * ((np.arange(20) - 10) / 3)**2)
    
    # ウェーブレット特徴抽出
    features, coeffs = extract_wavelet_features(signal, wavelet='db4', level=4)
    
    # 可視化
    fig, axes = plt.subplots(6, 1, figsize=(14, 16))
    
    # オリジナル信号
    axes[0].plot(time, signal, color='#11998e', linewidth=1.5)
    axes[0].set_title('Original Signal with Transient Events', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(alpha=0.3)
    
    # 近似係数（低周波成分）
    axes[1].plot(coeffs[0], color='#11998e', linewidth=2)
    axes[1].set_title(f'Approximation Coefficients (cA{len(coeffs)-1})', fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(alpha=0.3)
    
    # 詳細係数（高周波成分、レベル1-4）
    for i in range(1, 5):
        axes[i+1].plot(coeffs[i], color=f'C{i}', linewidth=1.5)
        axes[i+1].set_title(f'Detail Coefficients (cD{5-i})', fontweight='bold')
        axes[i+1].set_ylabel('Amplitude')
        axes[i+1].grid(alpha=0.3)
    
    axes[5].set_xlabel('Coefficient Index')
    
    plt.tight_layout()
    plt.show()
    
    # エネルギー比率の可視化
    energy_ratios = [features[f'{("approx" if i==0 else f"detail_{i}")}_energy_ratio']
                     for i in range(len(coeffs))]
    labels = ['Approx'] + [f'Detail {i}' for i in range(1, len(coeffs))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, energy_ratios, color='#38ef7d', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Energy Ratio')
    ax.set_title('Wavelet Energy Distribution Across Levels', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # 特徴量の出力
    print("\nWavelet Features (Energy and Entropy):")
    print("=" * 60)
    for level in ['approx'] + [f'detail_{i}' for i in range(1, 5)]:
        energy = features[f'{level}_energy']
        entropy = features[f'{level}_entropy']
        ratio = features[f'{level}_energy_ratio']
        print(f"{level:12s}: Energy={energy:10.2f}, Entropy={entropy:6.3f}, Ratio={ratio:6.4f}")
    

**解説:** ウェーブレット変換は、時間と周波数の両方の情報を保持しながら信号を分解します。異なる解像度レベルで、低周波のトレンドから高周波の突発イベントまで検出できます。プロセスの異常検知や過渡現象の解析に特に有効です。

* * *

## 4.5 ドメイン知識に基づく特徴量

### コード例6: プロセス特有の特徴量設計
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    def calculate_process_features(df):
        """
        化学プロセス特有の特徴量を計算
    
        Parameters:
        -----------
        df : pd.DataFrame
            プロセスデータ（温度、圧力、流量、濃度などを含む）
    
        Returns:
        --------
        df_features : pd.DataFrame
            追加された特徴量を含むDataFrame
        """
        df_features = df.copy()
    
        # 1. 滞留時間（Residence Time）
        # 滞留時間 = 反応器容積 / 流量
        reactor_volume = 100  # リットル
        df_features['residence_time'] = reactor_volume / df['flow_rate']
    
        # 2. 空間速度（Space Velocity）
        # LHSV (Liquid Hourly Space Velocity) = 流量 / 反応器容積
        df_features['space_velocity'] = df['flow_rate'] / reactor_volume
    
        # 3. 転化率（Conversion）
        # 転化率 = (入口濃度 - 出口濃度) / 入口濃度
        df_features['conversion'] = ((df['inlet_concentration'] - df['outlet_concentration']) /
                                      df['inlet_concentration'])
    
        # 4. 収率（Yield）
        # 収率 = 目的生成物濃度 / 入口原料濃度
        df_features['yield'] = df['product_concentration'] / df['inlet_concentration']
    
        # 5. 選択率（Selectivity）
        # 選択率 = 目的生成物 / 全生成物
        df_features['selectivity'] = (df['product_concentration'] /
                                      (df['product_concentration'] + df['byproduct_concentration']))
    
        # 6. 熱収支関連
        # 反応熱（簡略化モデル）
        cp = 4.18  # 比熱 [kJ/(kg·K)]
        df_features['heat_generation'] = (df['flow_rate'] * cp *
                                          (df['outlet_temperature'] - df['inlet_temperature']))
    
        # 7. 圧力損失
        df_features['pressure_drop'] = df['inlet_pressure'] - df['outlet_pressure']
    
        # 8. エネルギー効率
        # エネルギー効率 = 製品価値 / エネルギー投入
        df_features['energy_efficiency'] = df['product_concentration'] / df['energy_input']
    
        # 9. 比率特徴（物質収支チェック）
        df_features['mass_balance_ratio'] = ((df['inlet_concentration'] * df['flow_rate']) /
                                             ((df['outlet_concentration'] + df['product_concentration']) * df['flow_rate']))
    
        # 10. 温度-圧力相関指標
        df_features['T_P_interaction'] = df['temperature'] * df['pressure']
    
        return df_features
    
    
    # サンプルデータ生成（反応器の運転データ）
    np.random.seed(42)
    n_samples = 200
    
    process_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
        'flow_rate': 50 + 10 * np.random.randn(n_samples),  # L/h
        'temperature': 180 + 5 * np.random.randn(n_samples),  # °C
        'pressure': 3.0 + 0.2 * np.random.randn(n_samples),  # bar
        'inlet_temperature': 150 + 3 * np.random.randn(n_samples),  # °C
        'outlet_temperature': 185 + 5 * np.random.randn(n_samples),  # °C
        'inlet_pressure': 3.2 + 0.15 * np.random.randn(n_samples),  # bar
        'outlet_pressure': 2.8 + 0.15 * np.random.randn(n_samples),  # bar
        'inlet_concentration': 2.0 + 0.1 * np.random.randn(n_samples),  # mol/L
        'outlet_concentration': 0.5 + 0.1 * np.random.randn(n_samples),  # mol/L
        'product_concentration': 1.3 + 0.15 * np.random.randn(n_samples),  # mol/L
        'byproduct_concentration': 0.2 + 0.05 * np.random.randn(n_samples),  # mol/L
        'energy_input': 100 + 15 * np.random.randn(n_samples)  # kW
    })
    
    # 特徴量計算
    df_with_features = calculate_process_features(process_data)
    
    # 可視化
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 滞留時間
    axes[0, 0].plot(df_with_features.index, df_with_features['residence_time'],
                    color='#11998e', linewidth=1.5)
    axes[0, 0].set_ylabel('Residence Time [h]')
    axes[0, 0].set_title('Residence Time', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 転化率
    axes[0, 1].plot(df_with_features.index, df_with_features['conversion'] * 100,
                    color='#38ef7d', linewidth=1.5)
    axes[0, 1].set_ylabel('Conversion [%]')
    axes[0, 1].set_title('Conversion Rate', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 収率
    axes[1, 0].plot(df_with_features.index, df_with_features['yield'] * 100,
                    color='orange', linewidth=1.5)
    axes[1, 0].set_ylabel('Yield [%]')
    axes[1, 0].set_title('Product Yield', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 選択率
    axes[1, 1].plot(df_with_features.index, df_with_features['selectivity'] * 100,
                    color='purple', linewidth=1.5)
    axes[1, 1].set_ylabel('Selectivity [%]')
    axes[1, 1].set_title('Product Selectivity', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # エネルギー効率
    axes[2, 0].plot(df_with_features.index, df_with_features['energy_efficiency'],
                    color='red', linewidth=1.5)
    axes[2, 0].set_ylabel('Efficiency [mol·L⁻¹/kW]')
    axes[2, 0].set_title('Energy Efficiency', fontweight='bold')
    axes[2, 0].set_xlabel('Sample Index')
    axes[2, 0].grid(alpha=0.3)
    
    # 圧力損失
    axes[2, 1].plot(df_with_features.index, df_with_features['pressure_drop'],
                    color='brown', linewidth=1.5)
    axes[2, 1].set_ylabel('Pressure Drop [bar]')
    axes[2, 1].set_title('Pressure Drop', fontweight='bold')
    axes[2, 1].set_xlabel('Sample Index')
    axes[2, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 統計サマリー
    print("\nProcess Feature Statistics:")
    print("=" * 70)
    feature_cols = ['residence_time', 'conversion', 'yield', 'selectivity',
                    'energy_efficiency', 'pressure_drop']
    print(df_with_features[feature_cols].describe().to_string())
    

**解説:** プロセス工学の知識に基づく特徴量は、物理的な意味を持ち解釈可能性が高い点が重要です。転化率、収率、選択率などは、プロセスの性能を直接表現し、異常検知や最適化において極めて有用です。

* * *

## 4.6 相互作用特徴と多項式特徴

### コード例7: 相互作用特徴の生成
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from itertools import combinations
    
    def create_interaction_features(df, feature_cols, degree=2):
        """
        相互作用特徴と多項式特徴を生成
    
        Parameters:
        -----------
        df : pd.DataFrame
            元のデータ
        feature_cols : list
            特徴量のカラム名リスト
        degree : int
            多項式の次数
    
        Returns:
        --------
        df_extended : pd.DataFrame
            相互作用特徴を追加したDataFrame
        """
        df_extended = df.copy()
    
        # 1. 積（Product）特徴
        for col1, col2 in combinations(feature_cols, 2):
            df_extended[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
        # 2. 比率（Ratio）特徴
        for col1, col2 in combinations(feature_cols, 2):
            # ゼロ除算を避ける
            df_extended[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
    
        # 3. 差分特徴
        for col1, col2 in combinations(feature_cols, 2):
            df_extended[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
    
        # 4. 多項式特徴（sklearn使用）
        if degree >= 2:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(df[feature_cols])
            poly_feature_names = poly.get_feature_names_out(feature_cols)
    
            # 新しい特徴のみ追加（元の特徴は除く）
            for i, name in enumerate(poly_feature_names):
                if name not in feature_cols:
                    df_extended[f'poly_{name}'] = X_poly[:, i]
    
        return df_extended
    
    
    # サンプルデータ
    np.random.seed(42)
    n_samples = 300
    
    data = pd.DataFrame({
        'temperature': 180 + 10 * np.random.randn(n_samples),
        'pressure': 3.0 + 0.5 * np.random.randn(n_samples),
        'flow_rate': 50 + 8 * np.random.randn(n_samples)
    })
    
    # 非線形な目的変数（温度と圧力の相互作用を含む）
    data['product_quality'] = (0.5 * data['temperature'] +
                               2.0 * data['pressure'] +
                               0.01 * data['temperature'] * data['pressure'] +  # 相互作用項
                               0.002 * data['temperature']**2 +  # 2次項
                               np.random.randn(n_samples) * 5)
    
    # 相互作用特徴の生成
    df_interaction = create_interaction_features(data,
                                                 ['temperature', 'pressure', 'flow_rate'],
                                                 degree=2)
    
    # 可視化：相関分析
    import seaborn as sns
    
    # 主要な特徴量の相関マトリックス
    key_features = ['temperature', 'pressure', 'flow_rate',
                    'temperature_x_pressure', 'poly_temperature^2',
                    'temperature_div_pressure', 'product_quality']
    correlation_matrix = df_interaction[key_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix with Interaction Features',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 散布図行列
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 温度 vs 品質
    axes[0, 0].scatter(df_interaction['temperature'], df_interaction['product_quality'],
                       alpha=0.5, color='#11998e', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Temperature [°C]')
    axes[0, 0].set_ylabel('Product Quality')
    axes[0, 0].set_title('Temperature vs Quality')
    axes[0, 0].grid(alpha=0.3)
    
    # 圧力 vs 品質
    axes[0, 1].scatter(df_interaction['pressure'], df_interaction['product_quality'],
                       alpha=0.5, color='#38ef7d', edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Pressure [bar]')
    axes[0, 1].set_ylabel('Product Quality')
    axes[0, 1].set_title('Pressure vs Quality')
    axes[0, 1].grid(alpha=0.3)
    
    # 温度×圧力（相互作用） vs 品質
    axes[1, 0].scatter(df_interaction['temperature_x_pressure'],
                       df_interaction['product_quality'],
                       alpha=0.5, color='orange', edgecolor='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Temperature × Pressure')
    axes[1, 0].set_ylabel('Product Quality')
    axes[1, 0].set_title('Interaction Feature vs Quality')
    axes[1, 0].grid(alpha=0.3)
    
    # 温度^2（2次項） vs 品質
    axes[1, 1].scatter(df_interaction['poly_temperature^2'],
                       df_interaction['product_quality'],
                       alpha=0.5, color='purple', edgecolor='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Temperature²')
    axes[1, 1].set_ylabel('Product Quality')
    axes[1, 1].set_title('Polynomial Feature vs Quality')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nOriginal features: {len(['temperature', 'pressure', 'flow_rate'])}")
    print(f"Total features after interaction: {len(df_interaction.columns)}")
    print(f"\nSample of new interaction features:")
    interaction_cols = [col for col in df_interaction.columns if '_x_' in col or '_div_' in col or 'poly_' in col]
    print(interaction_cols[:10])
    

**解説:** 相互作用特徴は、変数間の非線形な関係を捉えます。温度と圧力の積項は、両者が同時に高い場合の効果を表現し、線形モデルでも非線形な関係をモデル化できるようになります。

* * *

## 4.7 特徴量の正規化とスケーリング

### コード例8: 特徴量スケーリング手法の比較
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                       RobustScaler, PowerTransformer)
    
    # サンプルデータ（スケールの異なる特徴量）
    np.random.seed(42)
    n_samples = 300
    
    data = pd.DataFrame({
        'temperature': 180 + 20 * np.random.randn(n_samples),  # 平均180, スケール大
        'pressure': 3.0 + 0.5 * np.random.randn(n_samples),    # 平均3, スケール小
        'flow_rate': 5000 + 500 * np.random.randn(n_samples)   # 平均5000, スケール非常に大
    })
    
    # 外れ値を追加
    data.loc[10:15, 'temperature'] = 250  # 異常に高い温度
    data.loc[50:55, 'pressure'] = 8.0     # 異常に高い圧力
    
    # 各種スケーリング手法を適用
    scalers = {
        'Original': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'PowerTransformer': PowerTransformer(method='yeo-johnson')
    }
    
    scaled_data = {}
    for name, scaler in scalers.items():
        if scaler is None:
            scaled_data[name] = data.copy()
        else:
            scaled_data[name] = pd.DataFrame(
                scaler.fit_transform(data),
                columns=data.columns
            )
    
    # 可視化
    fig, axes = plt.subplots(3, 5, figsize=(18, 12))
    
    for i, feature in enumerate(data.columns):
        for j, (name, df) in enumerate(scaled_data.items()):
            axes[i, j].hist(df[feature], bins=30, color='#11998e',
                            alpha=0.7, edgecolor='black')
            axes[i, j].set_title(f'{name}\n{feature}', fontsize=10)
            axes[i, j].grid(alpha=0.3)
    
            if i == 0:
                axes[i, j].set_title(f'{name}\n{feature}', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 統計量の比較
    print("\nScaling Methods Comparison:")
    print("=" * 80)
    for name, df in scaled_data.items():
        print(f"\n{name}:")
        print(df.describe().loc[['mean', 'std', 'min', 'max']].to_string())
    

**解説:**

  * **StandardScaler** : 平均0、標準偏差1に正規化。外れ値の影響を受けやすい。
  * **MinMaxScaler** : [0, 1]の範囲にスケーリング。外れ値の影響を最も受けやすい。
  * **RobustScaler** : 中央値と四分位範囲を使用。外れ値に頑健。
  * **PowerTransformer** : データを正規分布に近づける変換。歪んだ分布に有効。

* * *

## 4.8 特徴量選択

### コード例9: 特徴量選択手法（相互情報量、RFE、LASSO）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import (mutual_info_regression, RFE,
                                           SelectFromModel)
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    
    # 特徴量生成（一部は目的変数と無関係）
    X = np.random.randn(n_samples, n_features)
    
    # 目的変数（一部の特徴量のみに依存）
    y = (3 * X[:, 0] +      # 重要な特徴
         2 * X[:, 1] +      # 重要な特徴
         1.5 * X[:, 2] +    # やや重要
         0.5 * X[:, 5] +    # わずかに重要
         np.random.randn(n_samples) * 0.5)  # ノイズ
    # X[:, 3], X[:, 4], X[:, 6-19]は無関係
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    
    # 1. 相互情報量（Mutual Information）
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_scores = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
    
    # 2. RFE（Recursive Feature Elimination）
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rfe = RFE(estimator=rf_model, n_features_to_select=10)
    rfe.fit(X, y)
    rfe_ranking = pd.Series(rfe.ranking_, index=feature_names).sort_values()
    
    # 3. LASSO（L1正則化）
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X, y)
    lasso_coefs = pd.Series(np.abs(lasso.coef_), index=feature_names).sort_values(ascending=False)
    
    # 4. Random Forestの特徴重要度
    rf_model.fit(X, y)
    rf_importances = pd.Series(rf_model.feature_importances_,
                               index=feature_names).sort_values(ascending=False)
    
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 相互情報量
    axes[0, 0].barh(mi_scores.index[:10], mi_scores.values[:10],
                    color='#11998e', edgecolor='black')
    axes[0, 0].set_xlabel('Mutual Information Score')
    axes[0, 0].set_title('Mutual Information (Top 10 Features)', fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(alpha=0.3, axis='x')
    
    # RFEランキング
    selected_features_rfe = rfe_ranking[rfe_ranking == 1].index.tolist()
    axes[0, 1].barh(rfe_ranking.index[:10], rfe_ranking.values[:10],
                    color='#38ef7d', edgecolor='black')
    axes[0, 1].set_xlabel('RFE Ranking (1 = selected)')
    axes[0, 1].set_title(f'RFE Ranking (Selected: {len(selected_features_rfe)})',
                         fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(alpha=0.3, axis='x')
    
    # LASSO係数
    axes[1, 0].barh(lasso_coefs.index[:10], lasso_coefs.values[:10],
                    color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('|LASSO Coefficient|')
    axes[1, 0].set_title('LASSO Feature Selection (Top 10)', fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # Random Forest重要度
    axes[1, 1].barh(rf_importances.index[:10], rf_importances.values[:10],
                    color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Feature Importance')
    axes[1, 1].set_title('Random Forest Importances (Top 10)', fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # 特徴選択結果の比較
    print("\nFeature Selection Results Comparison:")
    print("=" * 70)
    print("\nTop 5 Features by Each Method:")
    print(f"\nMutual Information:\n{mi_scores.head(5).to_string()}")
    print(f"\nRFE Selected Features:\n{selected_features_rfe[:5]}")
    print(f"\nLASSO:\n{lasso_coefs.head(5).to_string()}")
    print(f"\nRandom Forest:\n{rf_importances.head(5).to_string()}")
    
    # 真に重要な特徴（Feature_0, Feature_1, Feature_2, Feature_5）との比較
    true_important = ['Feature_0', 'Feature_1', 'Feature_2', 'Feature_5']
    print(f"\n\nTrue Important Features: {true_important}")
    print(f"MI correctly identified: {[f for f in mi_scores.head(5).index if f in true_important]}")
    print(f"LASSO correctly identified: {[f for f in lasso_coefs.head(5).index if f in true_important]}")
    print(f"RF correctly identified: {[f for f in rf_importances.head(5).index if f in true_important]}")
    

**解説:**

  * **相互情報量** : 非線形な関係も検出可能。計算コストが高い。
  * **RFE** : モデルベースの選択。反復的に重要度の低い特徴を除去。
  * **LASSO** : L1正則化により自動的に不要な特徴の係数をゼロに。線形関係を前提。
  * **Random Forest** : 非線形関係を捉える。アンサンブルモデルの副産物として得られる。

* * *

## 4.9 自動特徴量抽出（tsfresh）

### コード例10: tsfreshによる包括的特徴量抽出
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tsfresh import extract_features
    from tsfresh.feature_extraction import ComprehensiveFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    
    # サンプル時系列データ（複数のセンサー、複数の時系列セグメント）
    np.random.seed(42)
    n_segments = 50  # セグメント数（バッチ数）
    n_timesteps = 100  # 各セグメントの時系列長
    
    data_list = []
    
    for seg_id in range(n_segments):
        # 各セグメントは異なる特性を持つ
        time = np.arange(n_timesteps)
    
        # ラベル（品質：良品=1、不良品=0）
        label = np.random.choice([0, 1], p=[0.3, 0.7])
    
        # 良品と不良品で統計的特性を変える
        if label == 1:  # 良品
            temperature = 180 + 2 * np.sin(0.1 * time) + np.random.randn(n_timesteps) * 0.5
            pressure = 3.0 + 0.1 * np.cos(0.08 * time) + np.random.randn(n_timesteps) * 0.1
        else:  # 不良品（より不規則）
            temperature = 180 + 5 * np.sin(0.1 * time) + np.random.randn(n_timesteps) * 2.0
            pressure = 3.0 + 0.3 * np.cos(0.08 * time) + np.random.randn(n_timesteps) * 0.5
    
        for t in range(n_timesteps):
            data_list.append({
                'segment_id': seg_id,
                'time': t,
                'temperature': temperature[t],
                'pressure': pressure[t],
                'label': label
            })
    
    df_timeseries = pd.DataFrame(data_list)
    
    # tsfreshで特徴量抽出
    print("Extracting features with tsfresh...")
    extraction_settings = ComprehensiveFCParameters()
    
    # 特徴量抽出（segment_idごとに集約）
    df_features = extract_features(
        df_timeseries[['segment_id', 'time', 'temperature', 'pressure']],
        column_id='segment_id',
        column_sort='time',
        default_fc_parameters=extraction_settings,
        impute_function=impute,
        n_jobs=4
    )
    
    # ラベル情報を追加
    df_labels = df_timeseries.groupby('segment_id')['label'].first()
    df_features = df_features.join(df_labels)
    
    print(f"\nExtracted {len(df_features.columns)-1} features from {n_segments} segments")
    print(f"Feature examples:\n{df_features.columns[:10].tolist()}")
    
    # 特徴量の重要度分析（簡易版：分散に基づく）
    feature_variance = df_features.drop('label', axis=1).var().sort_values(ascending=False)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. サンプル時系列（良品 vs 不良品）
    good_example = df_timeseries[df_timeseries['label'] == 1]['segment_id'].iloc[0]
    bad_example = df_timeseries[df_timeseries['label'] == 0]['segment_id'].iloc[0]
    
    good_data = df_timeseries[df_timeseries['segment_id'] == good_example]
    bad_data = df_timeseries[df_timeseries['segment_id'] == bad_example]
    
    axes[0, 0].plot(good_data['time'], good_data['temperature'],
                    color='green', linewidth=2, label='Good Product')
    axes[0, 0].plot(bad_data['time'], bad_data['temperature'],
                    color='red', linewidth=2, alpha=0.7, label='Defective Product')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Temperature [°C]')
    axes[0, 0].set_title('Example Time Series (Temperature)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. 圧力の比較
    axes[0, 1].plot(good_data['time'], good_data['pressure'],
                    color='green', linewidth=2, label='Good Product')
    axes[0, 1].plot(bad_data['time'], bad_data['pressure'],
                    color='red', linewidth=2, alpha=0.7, label='Defective Product')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Pressure [bar]')
    axes[0, 1].set_title('Example Time Series (Pressure)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. 分散が大きい特徴量トップ20
    axes[1, 0].barh(range(20), feature_variance.head(20).values,
                    color='#11998e', edgecolor='black')
    axes[1, 0].set_yticks(range(20))
    axes[1, 0].set_yticklabels([name[:40] + '...' if len(name) > 40 else name
                                for name in feature_variance.head(20).index], fontsize=8)
    axes[1, 0].set_xlabel('Variance')
    axes[1, 0].set_title('Top 20 Features by Variance', fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # 4. 特徴量の分布（良品 vs 不良品）
    # 分散が最大の特徴を選択
    top_feature = feature_variance.index[0]
    axes[1, 1].hist(df_features[df_features['label'] == 1][top_feature].dropna(),
                    bins=20, alpha=0.6, color='green', edgecolor='black', label='Good')
    axes[1, 1].hist(df_features[df_features['label'] == 0][top_feature].dropna(),
                    bins=20, alpha=0.6, color='red', edgecolor='black', label='Defective')
    axes[1, 1].set_xlabel('Feature Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Feature Distribution: {top_feature[:50]}...', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 統計サマリー
    print("\nFeature Extraction Summary:")
    print("=" * 70)
    print(f"Number of segments: {n_segments}")
    print(f"Timesteps per segment: {n_timesteps}")
    print(f"Total extracted features: {len(df_features.columns)-1}")
    print(f"\nSample features:\n{df_features.iloc[0, :5].to_string()}")
    

**解説:** tsfreshは時系列データから数百〜数千の統計的特徴量を自動的に抽出するライブラリです。フーリエ変換係数、自己相関、エントロピー、トレンド指標など、包括的な特徴セットを生成し、特徴選択機能も提供します。プロセスデータの異常検知や品質予測において強力なツールです。

* * *

## 4.10 本章のまとめ

### 学んだこと

  1. **時間領域特徴量**
     * 基本統計量（平均、標準偏差、歪度、尖度）
     * ローリング統計とラグ特徴
     * 差分と勾配による変化検出
  2. **周波数領域特徴量**
     * FFTによる周波数成分抽出
     * スペクトル重心、エントロピー
     * パワースペクトル密度とスペクトログラム
  3. **ウェーブレット特徴量**
     * 多解像度分解
     * 過渡現象と突発イベントの検出
  4. **ドメイン知識に基づく特徴量**
     * 滞留時間、転化率、収率、選択率
     * 熱収支、物質収支指標
  5. **相互作用特徴と多項式特徴**
     * 変数間の非線形関係のモデル化
  6. **特徴量選択**
     * 相互情報量、RFE、LASSO、Random Forest重要度
  7. **自動特徴量抽出**
     * tsfreshによる包括的な特徴生成

### 重要なポイント

  * 特徴量エンジニアリングは機械学習モデルの性能を決定する最重要ステップ
  * 時間領域と周波数領域の特徴は相補的な情報を提供する
  * プロセス知識に基づく特徴は解釈可能性が高く、実用的
  * 相互作用特徴により、線形モデルでも非線形関係を捉えられる
  * 適切な特徴選択により、モデルの汎化性能と計算効率が向上
  * tsfreshなどの自動化ツールは探索的分析に有用だが、ドメイン知識も重要

### 次の章へ

第5章では、**リアルタイムデータ解析と可視化** を学びます：

  * ストリーミングデータ処理
  * リアルタイム統計モニタリング
  * オンライン機械学習モデル
  * リアルタイムダッシュボード構築
  * 本番環境での監視システム実装
