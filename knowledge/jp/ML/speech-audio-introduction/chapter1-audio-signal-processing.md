---
title: 第1章：音声信号処理の基礎
chapter_title: 第1章：音声信号処理の基礎
subtitle: デジタル音声からスペクトログラムまで - 音響信号の理解と処理
reading_time: 35-40分
difficulty: 初級〜中級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ デジタル音声の基本概念（サンプリング、量子化）を理解する
  * ✅ 時間領域と周波数領域での音声分析手法を実装できる
  * ✅ スペクトログラムとメルスペクトログラムの違いを理解する
  * ✅ MFCCなどの音響特徴量を抽出できる
  * ✅ librosaを使った音声処理パイプラインを構築できる
  * ✅ 音声データの前処理とデータ拡張を実装できる

* * *

## 1.1 デジタル音声の基礎

### 音声信号とは

**音声信号（Audio Signal）** は、空気の振動を電気信号や数値データに変換したものです。機械学習で扱うためには、連続的なアナログ信号をデジタル化する必要があります。

> 「音声は時間の関数である」という特性を理解することが、音声処理の第一歩です。

### デジタル化のプロセス
    
    
    ```mermaid
    graph LR
        A[アナログ音声連続信号] --> B[サンプリング離散化]
        B --> C[量子化数値化]
        C --> D[デジタル音声数値配列]
        D --> E[音声ファイルWAV/MP3/FLAC]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#e8f5e9
    ```

### 1.1.1 サンプリングレート

**サンプリングレート（Sampling Rate）** は、1秒間に何回音声を記録するかを表す値で、単位はHz（ヘルツ）です。

$$ f_s = \frac{\text{サンプル数}}{\text{秒}} $$

**ナイキスト定理** : 信号を正確に再現するには、最高周波数の2倍以上のサンプリングレートが必要です。

$$ f_s \geq 2 \times f_{\max} $$

#### 一般的なサンプリングレート

サンプリングレート | 用途 | 周波数帯域  
---|---|---  
**8,000 Hz** | 電話音声 | 0-4 kHz  
**16,000 Hz** | 音声認識 | 0-8 kHz  
**22,050 Hz** | 低品質音楽 | 0-11 kHz  
**44,100 Hz** | CD品質音楽 | 0-22 kHz  
**48,000 Hz** | プロオーディオ | 0-24 kHz  
  
### 1.1.2 量子化とビット深度

**量子化（Quantization）** は、連続的な振幅値を離散的な数値に変換するプロセスです。**ビット深度（Bit Depth）** は、各サンプルを何ビットで表現するかを示します。

$$ \text{表現可能なレベル数} = 2^{\text{ビット深度}} $$

ビット深度 | レベル数 | ダイナミックレンジ | 用途  
---|---|---|---  
**8-bit** | 256 | 48 dB | 低品質  
**16-bit** | 65,536 | 96 dB | CD品質  
**24-bit** | 16,777,216 | 144 dB | プロ録音  
**32-bit float** | - | 1,528 dB | 処理用  
  
### 1.1.3 音声ファイル形式

形式 | 圧縮 | 品質 | 用途  
---|---|---|---  
**WAV** | 非圧縮 | 最高 | 録音・処理  
**FLAC** | 可逆圧縮 | 最高 | アーカイブ  
**MP3** | 非可逆圧縮 | 中〜高 | 配布・再生  
**AAC** | 非可逆圧縮 | 高 | ストリーミング  
**OGG** | 非可逆圧縮 | 中〜高 | オープン形式  
  
### 1.1.4 librosaの基本
    
    
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 音声ファイルの読み込み
    # librosaはデフォルトでモノラル、22050Hzにリサンプリング
    audio_path = "sample.wav"
    y, sr = librosa.load(audio_path)
    
    print("=== 音声ファイル情報 ===")
    print(f"サンプリングレート: {sr} Hz")
    print(f"サンプル数: {len(y)}")
    print(f"時間長: {len(y) / sr:.2f} 秒")
    print(f"データ型: {y.dtype}")
    print(f"振幅範囲: [{y.min():.3f}, {y.max():.3f}]")
    
    # オリジナルのサンプリングレートで読み込み
    y_orig, sr_orig = librosa.load(audio_path, sr=None)
    print(f"\nオリジナルサンプリングレート: {sr_orig} Hz")
    
    # 特定のサンプリングレートで読み込み
    y_16k, sr_16k = librosa.load(audio_path, sr=16000)
    print(f"16kHzリサンプリング: {len(y_16k)} サンプル")
    
    # ステレオで読み込み
    y_stereo, sr = librosa.load(audio_path, mono=False)
    print(f"ステレオ: shape = {y_stereo.shape}")
    

**出力** ：
    
    
    === 音声ファイル情報 ===
    サンプリングレート: 22050 Hz
    サンプル数: 66150
    時間長: 3.00 秒
    データ型: float32
    振幅範囲: [-0.523, 0.487]
    
    オリジナルサンプリングレート: 44100 Hz
    16kHzリサンプリング: 48000 サンプル
    ステレオ: shape = (2, 66150)
    

* * *

## 1.2 時間領域の処理

### 時間領域とは

**時間領域（Time Domain）** は、音声信号を時間の関数として扱います。横軸が時間、縦軸が振幅を表します。

### 1.2.1 振幅とエネルギー

**振幅（Amplitude）** は音の大きさを表します。**エネルギー（Energy）** は振幅の二乗の総和です。

$$ E = \sum_{n=1}^{N} |x[n]|^2 $$

**RMS（Root Mean Square）** は平均的な振幅を表します：

$$ \text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x[n]^2} $$
    
    
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプル音声の生成（440Hz正弦波 + ノイズ）
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    
    print("=== 振幅とエネルギー ===")
    
    # 振幅統計
    print(f"最大振幅: {np.max(np.abs(y)):.3f}")
    print(f"平均振幅: {np.mean(np.abs(y)):.3f}")
    
    # エネルギー
    energy = np.sum(y**2)
    print(f"総エネルギー: {energy:.2f}")
    
    # RMS
    rms = np.sqrt(np.mean(y**2))
    print(f"RMS: {rms:.3f}")
    
    # フレームごとのRMS計算
    frame_length = 2048
    hop_length = 512
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    print(f"\nフレーム数: {len(rms_frames)}")
    print(f"RMS範囲: [{rms_frames.min():.3f}, {rms_frames.max():.3f}]")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # 波形
    axes[0].plot(t, y, linewidth=0.5)
    axes[0].set_xlabel('時間 (秒)')
    axes[0].set_ylabel('振幅')
    axes[0].set_title('音声波形')
    axes[0].grid(True, alpha=0.3)
    
    # RMS
    times = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=hop_length)
    axes[1].plot(times, rms_frames)
    axes[1].set_xlabel('時間 (秒)')
    axes[1].set_ylabel('RMS')
    axes[1].set_title('RMSエネルギー')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('amplitude_rms.png', dpi=150, bbox_inches='tight')
    print("\n図を保存しました: amplitude_rms.png")
    

### 1.2.2 Zero Crossing Rate

**Zero Crossing Rate (ZCR)** は、信号がゼロを横切る頻度を表します。音声の種類（有声音/無声音）や楽器の判別に有用です。

$$ \text{ZCR} = \frac{1}{T-1} \sum_{t=1}^{T-1} \mathbb{1}_{x[t] \cdot x[t-1] < 0} $$
    
    
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプル音声の読み込み
    y, sr = librosa.load('sample.wav', duration=5.0)
    
    # Zero Crossing Rateの計算
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
    
    print("=== Zero Crossing Rate ===")
    print(f"ZCR平均: {np.mean(zcr):.4f}")
    print(f"ZCR範囲: [{zcr.min():.4f}, {zcr.max():.4f}]")
    
    # 時間軸
    times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=512)
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # 波形
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('音声波形')
    axes[0].set_ylabel('振幅')
    
    # ZCR
    axes[1].plot(times, zcr)
    axes[1].set_xlabel('時間 (秒)')
    axes[1].set_ylabel('ZCR')
    axes[1].set_title('Zero Crossing Rate')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('zcr_analysis.png', dpi=150, bbox_inches='tight')
    print("図を保存しました: zcr_analysis.png")
    
    # 特徴：高ZCR = 無声音/ノイズ、低ZCR = 有声音/音楽
    print("\n解釈:")
    print("- 高ZCR (>0.1): 無声音、ホワイトノイズ、シンバル等")
    print("- 低ZCR (<0.05): 有声音、音楽、低周波成分")
    

### 1.2.3 音声の読み込みと可視化
    
    
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 音声ファイルの読み込み
    audio_path = 'sample.wav'
    y, sr = librosa.load(audio_path, sr=None)
    
    print("=== 音声データの可視化 ===")
    print(f"サンプリングレート: {sr} Hz")
    print(f"時間長: {len(y)/sr:.2f} 秒")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. 波形表示
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('音声波形（全体）', fontsize=14)
    axes[0].set_ylabel('振幅')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 拡大波形（最初の0.1秒）
    n_samples = int(0.1 * sr)
    time_zoom = np.arange(n_samples) / sr
    axes[1].plot(time_zoom, y[:n_samples])
    axes[1].set_title('音声波形（拡大：最初の0.1秒）', fontsize=14)
    axes[1].set_xlabel('時間 (秒)')
    axes[1].set_ylabel('振幅')
    axes[1].grid(True, alpha=0.3)
    
    # 3. エンベロープ（包絡線）
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    axes[2].plot(times, rms, label='RMS Energy', color='red', linewidth=2)
    axes[2].set_title('音声エンベロープ（RMS）', fontsize=14)
    axes[2].set_xlabel('時間 (秒)')
    axes[2].set_ylabel('RMS振幅')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('audio_visualization.png', dpi=150, bbox_inches='tight')
    print("図を保存しました: audio_visualization.png")
    
    # 統計情報
    print("\n音声統計:")
    print(f"  最大振幅: {np.max(np.abs(y)):.3f}")
    print(f"  RMS: {np.sqrt(np.mean(y**2)):.3f}")
    print(f"  ダイナミックレンジ: {20*np.log10(np.max(np.abs(y))/np.min(np.abs(y[y!=0]))):.1f} dB")
    

* * *

## 1.3 周波数領域の処理

### 周波数領域とは

**周波数領域（Frequency Domain）** は、音声信号をどのような周波数成分で構成されているかで表現します。フーリエ変換により時間領域から変換できます。

### 1.3.1 フーリエ変換（FFT）

**離散フーリエ変換（DFT: Discrete Fourier Transform）** は、時間領域の信号を周波数領域に変換します：

$$ X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i 2\pi k n / N} $$

**高速フーリエ変換（FFT: Fast Fourier Transform）** は、DFTを高速に計算するアルゴリズムです。
    
    
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプル音声（複数周波数の正弦波）
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 440Hz（A4）+ 880Hz（A5）+ 1320Hz（E6）
    y = (0.5 * np.sin(2 * np.pi * 440 * t) +
         0.3 * np.sin(2 * np.pi * 880 * t) +
         0.2 * np.sin(2 * np.pi * 1320 * t))
    
    print("=== フーリエ変換（FFT）===")
    
    # FFTの計算
    n_fft = 2048
    fft_result = np.fft.fft(y[:n_fft])
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    
    # 周波数軸
    frequencies = np.fft.fftfreq(n_fft, 1/sr)
    
    # 正の周波数のみ（対称性のため）
    positive_frequencies = frequencies[:n_fft//2]
    positive_magnitude = magnitude[:n_fft//2]
    
    print(f"FFTサイズ: {n_fft}")
    print(f"周波数分解能: {sr/n_fft:.2f} Hz")
    
    # ピーク周波数の検出
    peak_indices = np.argsort(positive_magnitude)[-3:][::-1]
    print("\n検出されたピーク周波数:")
    for i, idx in enumerate(peak_indices, 1):
        print(f"  {i}. {positive_frequencies[idx]:.1f} Hz (振幅: {positive_magnitude[idx]:.1f})")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 時間領域
    axes[0].plot(t[:1000], y[:1000])
    axes[0].set_xlabel('時間 (秒)')
    axes[0].set_ylabel('振幅')
    axes[0].set_title('時間領域の信号')
    axes[0].grid(True, alpha=0.3)
    
    # 周波数領域（振幅スペクトル）
    axes[1].plot(positive_frequencies, positive_magnitude)
    axes[1].set_xlabel('周波数 (Hz)')
    axes[1].set_ylabel('振幅')
    axes[1].set_title('振幅スペクトル')
    axes[1].set_xlim([0, 2000])
    axes[1].grid(True, alpha=0.3)
    
    # 対数スケール
    axes[2].plot(positive_frequencies, 20*np.log10(positive_magnitude + 1e-10))
    axes[2].set_xlabel('周波数 (Hz)')
    axes[2].set_ylabel('振幅 (dB)')
    axes[2].set_title('振幅スペクトル（対数スケール）')
    axes[2].set_xlim([0, 2000])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fft_analysis.png', dpi=150, bbox_inches='tight')
    print("\n図を保存しました: fft_analysis.png")
    

### 1.3.2 スペクトログラム

**スペクトログラム（Spectrogram）** は、時間-周波数領域での音声表現です。短時間フーリエ変換（STFT）を使用します。

$$ \text{STFT}(t, f) = \sum_{n=-\infty}^{\infty} x[n] \cdot w[n-t] \cdot e^{-i 2\pi f n} $$

$w[n]$はウィンドウ関数（ハミング窓、ハニング窓など）です。
    
    
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 音声ファイルの読み込み
    y, sr = librosa.load('sample.wav', duration=5.0)
    
    print("=== スペクトログラム ===")
    
    # STFTの計算
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # 振幅スペクトログラム
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    print(f"スペクトログラムの形状: {S_db.shape}")
    print(f"  周波数ビン数: {S_db.shape[0]}")
    print(f"  時間フレーム数: {S_db.shape[1]}")
    print(f"  周波数分解能: {sr/n_fft:.2f} Hz")
    print(f"  時間分解能: {hop_length/sr*1000:.2f} ms")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 波形
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('音声波形', fontsize=14)
    axes[0].set_ylabel('振幅')
    
    # スペクトログラム
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                    x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
    axes[1].set_title('スペクトログラム', fontsize=14)
    axes[1].set_ylabel('周波数 (Hz)')
    axes[1].set_xlabel('時間 (秒)')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('spectrogram.png', dpi=150, bbox_inches='tight')
    print("図を保存しました: spectrogram.png")
    
    # パワースペクトログラム
    S_power = np.abs(D)**2
    print(f"\nパワー範囲: [{S_power.min():.2e}, {S_power.max():.2e}]")
    

### 1.3.3 メルスペクトログラム

**メルスペクトログラム（Mel Spectrogram）** は、人間の聴覚特性を考慮した周波数スケールを使用します。

**メルスケール** は、人間が知覚する音高の尺度です：

$$ m = 2595 \log_{10}\left(1 + \frac{f}{700}\right) $$
    
    
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 音声の読み込み
    y, sr = librosa.load('sample.wav', duration=5.0)
    
    print("=== メルスペクトログラム ===")
    
    # メルスペクトログラムの計算
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                            hop_length=hop_length, n_mels=n_mels)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    
    print(f"メルスペクトログラムの形状: {S_mel_db.shape}")
    print(f"  メルバンド数: {n_mels}")
    print(f"  時間フレーム数: {S_mel_db.shape[1]}")
    
    # 通常のスペクトログラム
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # 可視化：比較
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 通常のスペクトログラム
    img1 = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                      x_axis='time', y_axis='hz', ax=axes[0], cmap='viridis')
    axes[0].set_title('通常のスペクトログラム（線形周波数）', fontsize=14)
    axes[0].set_ylabel('周波数 (Hz)')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # メルスペクトログラム
    img2 = librosa.display.specshow(S_mel_db, sr=sr, hop_length=hop_length,
                                      x_axis='time', y_axis='mel', ax=axes[1], cmap='viridis')
    axes[1].set_title('メルスペクトログラム（メルスケール）', fontsize=14)
    axes[1].set_ylabel('周波数 (メル)')
    axes[1].set_xlabel('時間 (秒)')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('mel_spectrogram.png', dpi=150, bbox_inches='tight')
    print("図を保存しました: mel_spectrogram.png")
    
    # メルスケールの特性
    print("\nメルスケールの特性:")
    print("- 低周波数: 高い周波数分解能")
    print("- 高周波数: 低い周波数分解能")
    print("- 人間の聴覚特性に近い表現")
    

* * *

## 1.4 音響特徴量

### 音響特徴量とは

**音響特徴量（Acoustic Features）** は、音声信号から抽出される特徴的な数値表現です。機械学習モデルの入力として使用されます。

### 1.4.1 MFCC（Mel-Frequency Cepstral Coefficients）

**MFCC** は、音声認識で最も広く使われる特徴量です。メルスペクトログラムに対数と離散コサイン変換（DCT）を適用します。
    
    
    ```mermaid
    graph LR
        A[音声信号] --> B[プリエンファシス]
        B --> C[フレーム分割]
        C --> D[窓関数適用]
        D --> E[FFT]
        E --> F[メルフィルタバンク]
        F --> G[対数]
        G --> H[DCT]
        H --> I[MFCC係数]
    
        style A fill:#ffebee
        style I fill:#e8f5e9
    ```
    
    
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 音声の読み込み
    y, sr = librosa.load('sample.wav', duration=5.0)
    
    print("=== MFCC（Mel-Frequency Cepstral Coefficients）===")
    
    # MFCCの計算
    n_mfcc = 13  # 一般的には13または20
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    print(f"MFCCの形状: {mfccs.shape}")
    print(f"  MFCC係数数: {n_mfcc}")
    print(f"  時間フレーム数: {mfccs.shape[1]}")
    
    # 統計量
    print(f"\nMFCC統計:")
    print(f"  平均: {np.mean(mfccs, axis=1)[:5]}")  # 最初の5係数
    print(f"  標準偏差: {np.std(mfccs, axis=1)[:5]}")
    
    # デルタ特徴量（1次微分）
    mfccs_delta = librosa.feature.delta(mfccs)
    print(f"\nデルタMFCC形状: {mfccs_delta.shape}")
    
    # デルタデルタ特徴量（2次微分）
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    print(f"デルタデルタMFCC形状: {mfccs_delta2.shape}")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # MFCC
    img1 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[0], cmap='coolwarm')
    axes[0].set_title('MFCC', fontsize=14)
    axes[0].set_ylabel('MFCC係数')
    fig.colorbar(img1, ax=axes[0])
    
    # デルタMFCC
    img2 = librosa.display.specshow(mfccs_delta, sr=sr, x_axis='time', ax=axes[1], cmap='coolwarm')
    axes[1].set_title('デルタMFCC（1次微分）', fontsize=14)
    axes[1].set_ylabel('デルタMFCC係数')
    fig.colorbar(img2, ax=axes[1])
    
    # デルタデルタMFCC
    img3 = librosa.display.specshow(mfccs_delta2, sr=sr, x_axis='time', ax=axes[2], cmap='coolwarm')
    axes[2].set_title('デルタデルタMFCC（2次微分）', fontsize=14)
    axes[2].set_ylabel('デルタデルタMFCC係数')
    axes[2].set_xlabel('時間 (秒)')
    fig.colorbar(img3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('mfcc_features.png', dpi=150, bbox_inches='tight')
    print("\n図を保存しました: mfcc_features.png")
    
    # 特徴ベクトルの結合（音声認識で一般的）
    combined_features = np.vstack([mfccs, mfccs_delta, mfccs_delta2])
    print(f"\n結合特徴量の形状: {combined_features.shape}")
    print(f"  (13 MFCC + 13 デルタ + 13 デルタデルタ = 39次元)")
    

### 1.4.2 Chroma Features

**クロマ特徴量（Chroma Features）** は、音楽の調性や和音を表現します。12個の音階クラス（C, C#, D, ..., B）ごとのエネルギーを計算します。
    
    
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 音楽ファイルの読み込み
    y, sr = librosa.load('music.wav', duration=10.0)
    
    print("=== Chroma Features ===")
    
    # クロマグラムの計算
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    
    print(f"クロマグラムの形状: {chromagram.shape}")
    print(f"  音階クラス数: 12 (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)")
    print(f"  時間フレーム数: {chromagram.shape[1]}")
    
    # CQT（Constant-Q Transform）ベースのクロマ
    chromagram_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    print(f"\nCQTクロマグラムの形状: {chromagram_cqt.shape}")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 波形
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('音声波形', fontsize=14)
    axes[0].set_ylabel('振幅')
    
    # STFTベースのクロマ
    img1 = librosa.display.specshow(chromagram, sr=sr, x_axis='time',
                                      y_axis='chroma', ax=axes[1], cmap='coolwarm')
    axes[1].set_title('クロマグラム（STFT）', fontsize=14)
    axes[1].set_ylabel('ピッチクラス')
    fig.colorbar(img1, ax=axes[1])
    
    # CQTベースのクロマ
    img2 = librosa.display.specshow(chromagram_cqt, sr=sr, x_axis='time',
                                      y_axis='chroma', ax=axes[2], cmap='coolwarm')
    axes[2].set_title('クロマグラム（CQT）', fontsize=14)
    axes[2].set_ylabel('ピッチクラス')
    axes[2].set_xlabel('時間 (秒)')
    fig.colorbar(img2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('chroma_features.png', dpi=150, bbox_inches='tight')
    print("図を保存しました: chroma_features.png")
    
    # 各ピッチクラスの平均エネルギー
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    mean_energy = np.mean(chromagram, axis=1)
    
    print("\n各ピッチクラスの平均エネルギー:")
    for pc, energy in zip(pitch_classes, mean_energy):
        print(f"  {pc}: {energy:.3f}")
    

### 1.4.3 Spectral Features

**スペクトル特徴量** は、周波数分布の特性を捉えます。
    
    
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 音声の読み込み
    y, sr = librosa.load('sample.wav', duration=5.0)
    
    print("=== Spectral Features ===")
    
    # 1. Spectral Centroid（スペクトル重心）
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    print(f"Spectral Centroid: 平均 {np.mean(spectral_centroids):.2f} Hz")
    
    # 2. Spectral Rolloff（スペクトルロールオフ）
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    print(f"Spectral Rolloff: 平均 {np.mean(spectral_rolloff):.2f} Hz")
    
    # 3. Spectral Bandwidth（スペクトル帯域幅）
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    print(f"Spectral Bandwidth: 平均 {np.mean(spectral_bandwidth):.2f} Hz")
    
    # 4. Spectral Contrast（スペクトルコントラスト）
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    print(f"Spectral Contrast形状: {spectral_contrast.shape}")
    
    # 5. Spectral Flatness（スペクトル平坦度）
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    print(f"Spectral Flatness: 平均 {np.mean(spectral_flatness):.4f}")
    
    # 時間軸
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames, sr=sr)
    
    # 可視化
    fig, axes = plt.subplots(5, 1, figsize=(14, 14))
    
    # スペクトログラム（背景）
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # 1. Spectral Centroid
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[0], cmap='gray_r', alpha=0.5)
    axes[0].plot(t, spectral_centroids, color='red', linewidth=2, label='Spectral Centroid')
    axes[0].set_title('Spectral Centroid', fontsize=14)
    axes[0].set_ylabel('周波数 (Hz)')
    axes[0].legend()
    
    # 2. Spectral Rolloff
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='gray_r', alpha=0.5)
    axes[1].plot(t, spectral_rolloff, color='blue', linewidth=2, label='Spectral Rolloff')
    axes[1].set_title('Spectral Rolloff', fontsize=14)
    axes[1].set_ylabel('周波数 (Hz)')
    axes[1].legend()
    
    # 3. Spectral Bandwidth
    axes[2].plot(t, spectral_bandwidth, color='green', linewidth=2)
    axes[2].set_title('Spectral Bandwidth', fontsize=14)
    axes[2].set_ylabel('帯域幅 (Hz)')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Spectral Contrast
    img = librosa.display.specshow(spectral_contrast, sr=sr, x_axis='time', ax=axes[3], cmap='coolwarm')
    axes[3].set_title('Spectral Contrast', fontsize=14)
    axes[3].set_ylabel('周波数バンド')
    fig.colorbar(img, ax=axes[3])
    
    # 5. Spectral Flatness
    axes[4].plot(t, spectral_flatness, color='purple', linewidth=2)
    axes[4].set_title('Spectral Flatness（0=トーン的, 1=ノイズ的）', fontsize=14)
    axes[4].set_ylabel('平坦度')
    axes[4].set_xlabel('時間 (秒)')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectral_features.png', dpi=150, bbox_inches='tight')
    print("\n図を保存しました: spectral_features.png")
    
    print("\n解釈:")
    print("- Centroid: 音の「明るさ」")
    print("- Rolloff: エネルギーの集中度")
    print("- Bandwidth: 周波数の広がり")
    print("- Contrast: 周波数帯域間のコントラスト")
    print("- Flatness: トーン的（0）かノイズ的（1）か")
    

### 1.4.4 特徴抽出パイプライン
    
    
    import librosa
    import numpy as np
    
    def extract_audio_features(audio_path, sr=22050):
        """
        音声ファイルから包括的な特徴量を抽出
        """
        # 音声の読み込み
        y, sr = librosa.load(audio_path, sr=sr)
    
        # 特徴量辞書
        features = {}
    
        # 1. 基本統計
        features['duration'] = len(y) / sr
        features['rms_mean'] = float(np.mean(librosa.feature.rms(y=y)))
        features['zcr_mean'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
        # 2. MFCC（統計量）
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
    
        # 3. スペクトル特徴
        features['spectral_centroid_mean'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features['spectral_rolloff_mean'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        features['spectral_bandwidth_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        features['spectral_flatness_mean'] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    
        # 4. クロマ特徴
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
    
        # 5. テンポ
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
    
        return features
    
    # 使用例
    print("=== 音響特徴抽出パイプライン ===")
    audio_file = 'sample.wav'
    features = extract_audio_features(audio_file)
    
    print(f"\n抽出された特徴量の数: {len(features)}")
    print("\n主な特徴量:")
    for key in list(features.keys())[:10]:
        print(f"  {key}: {features[key]:.4f}")
    
    # 特徴ベクトルとして取得
    feature_vector = np.array(list(features.values()))
    print(f"\n特徴ベクトルの次元: {len(feature_vector)}")
    print(f"特徴ベクトル（最初の10次元）: {feature_vector[:10]}")
    
    # 複数ファイルからの特徴抽出
    audio_files = ['sample1.wav', 'sample2.wav', 'sample3.wav']
    all_features = []
    
    print("\n複数ファイルからの特徴抽出:")
    for audio_file in audio_files:
        try:
            feats = extract_audio_features(audio_file)
            all_features.append(list(feats.values()))
            print(f"  ✓ {audio_file}: {len(feats)} 特徴量")
        except Exception as e:
            print(f"  ✗ {audio_file}: エラー ({e})")
    
    if all_features:
        # NumPy配列に変換（機械学習モデルの入力として使用）
        X = np.array(all_features)
        print(f"\n特徴行列の形状: {X.shape}")
        print(f"  (サンプル数, 特徴量数)")
    

* * *

## 1.5 音声データの前処理

### 前処理の重要性

音声データの前処理は、モデル性能に大きく影響します。ノイズ除去、正規化、データ拡張などが含まれます。

### 1.5.1 Noise Reduction
    
    
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import wiener
    
    # 音声の読み込み
    y, sr = librosa.load('sample.wav')
    
    print("=== ノイズ除去 ===")
    
    # 1. ウィーナーフィルタ
    y_wiener = wiener(y)
    print(f"ウィーナーフィルタ適用")
    
    # 2. スペクトルゲーティング（簡易版）
    # ノイズプロファイルの推定（最初の0.5秒をノイズと仮定）
    noise_sample = y[:int(0.5 * sr)]
    noise_spec = np.abs(librosa.stft(noise_sample))
    noise_profile = np.mean(noise_spec, axis=1, keepdims=True)
    
    # 全体のスペクトログラム
    D = librosa.stft(y)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # ノイズ除去（スペクトルサブトラクション）
    magnitude_clean = np.maximum(magnitude - 2.0 * noise_profile, 0.0)
    D_clean = magnitude_clean * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean)
    
    print(f"スペクトルサブトラクション適用")
    
    # 3. ハイパスフィルタ（低周波ノイズ除去）
    from scipy.signal import butter, filtfilt
    
    def highpass_filter(data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)
    
    y_highpass = highpass_filter(y, cutoff=80, fs=sr)
    print(f"ハイパスフィルタ適用（80Hz以下をカット）")
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # オリジナル
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='blue')
    axes[0].set_title('オリジナル音声', fontsize=14)
    axes[0].set_ylabel('振幅')
    
    # ウィーナーフィルタ
    librosa.display.waveshow(y_wiener, sr=sr, ax=axes[1], color='green')
    axes[1].set_title('ウィーナーフィルタ適用後', fontsize=14)
    axes[1].set_ylabel('振幅')
    
    # スペクトルサブトラクション
    librosa.display.waveshow(y_clean, sr=sr, ax=axes[2], color='red')
    axes[2].set_title('スペクトルサブトラクション適用後', fontsize=14)
    axes[2].set_ylabel('振幅')
    
    # ハイパスフィルタ
    librosa.display.waveshow(y_highpass, sr=sr, ax=axes[3], color='purple')
    axes[3].set_title('ハイパスフィルタ適用後', fontsize=14)
    axes[3].set_ylabel('振幅')
    axes[3].set_xlabel('時間 (秒)')
    
    plt.tight_layout()
    plt.savefig('noise_reduction.png', dpi=150, bbox_inches='tight')
    print("\n図を保存しました: noise_reduction.png")
    
    # SNR計算（簡易版）
    def calculate_snr(signal, noise):
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        return 10 * np.log10(signal_power / noise_power)
    
    print(f"\nSNR改善:")
    print(f"  オリジナル: 基準")
    print(f"  ウィーナー: +{calculate_snr(y_wiener, y-y_wiener):.2f} dB")
    print(f"  スペクトルサブトラクション: +{calculate_snr(y_clean, y-y_clean):.2f} dB")
    

### 1.5.2 Normalization
    
    
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 音声の読み込み
    y, sr = librosa.load('sample.wav')
    
    print("=== 正規化（Normalization）===")
    
    # 1. ピーク正規化（最大振幅を1.0に）
    y_peak_norm = y / np.max(np.abs(y))
    print(f"ピーク正規化: 最大値 = {np.max(np.abs(y_peak_norm)):.3f}")
    
    # 2. RMS正規化（目標RMSレベルに調整）
    target_rms = 0.1
    current_rms = np.sqrt(np.mean(y**2))
    y_rms_norm = y * (target_rms / current_rms)
    new_rms = np.sqrt(np.mean(y_rms_norm**2))
    print(f"RMS正規化: {current_rms:.4f} → {new_rms:.4f}")
    
    # 3. Z-score正規化（平均0、標準偏差1）
    y_zscore = (y - np.mean(y)) / np.std(y)
    print(f"Z-score正規化: 平均 = {np.mean(y_zscore):.6f}, 標準偏差 = {np.std(y_zscore):.3f}")
    
    # 4. Min-Max正規化（-1 to 1）
    y_minmax = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1
    print(f"Min-Max正規化: 範囲 = [{np.min(y_minmax):.3f}, {np.max(y_minmax):.3f}]")
    
    # 可視化
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    normalizations = [
        ('ピーク正規化', y_peak_norm),
        ('RMS正規化', y_rms_norm),
        ('Z-score正規化', y_zscore),
        ('Min-Max正規化', y_minmax)
    ]
    
    for idx, (title, y_norm) in enumerate(normalizations):
        # 波形
        librosa.display.waveshow(y_norm, sr=sr, ax=axes[idx, 0])
        axes[idx, 0].set_title(f'{title} - 波形', fontsize=12)
        axes[idx, 0].set_ylabel('振幅')
    
        # ヒストグラム
        axes[idx, 1].hist(y_norm, bins=50, alpha=0.7, color='blue')
        axes[idx, 1].set_title(f'{title} - 振幅分布', fontsize=12)
        axes[idx, 1].set_xlabel('振幅')
        axes[idx, 1].set_ylabel('頻度')
        axes[idx, 1].grid(True, alpha=0.3)
    
    axes[3, 0].set_xlabel('時間 (秒)')
    
    plt.tight_layout()
    plt.savefig('normalization.png', dpi=150, bbox_inches='tight')
    print("\n図を保存しました: normalization.png")
    
    # 統計情報
    print("\n正規化後の統計:")
    for title, y_norm in normalizations:
        print(f"{title}:")
        print(f"  平均: {np.mean(y_norm):.6f}")
        print(f"  標準偏差: {np.std(y_norm):.6f}")
        print(f"  範囲: [{np.min(y_norm):.3f}, {np.max(y_norm):.3f}]")
    

### 1.5.3 Trimming and Padding
    
    
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 音声の読み込み
    y, sr = librosa.load('sample.wav')
    
    print("=== トリミングとパディング ===")
    
    # 1. 無音部分のトリミング
    y_trimmed, index = librosa.effects.trim(y, top_db=20)
    print(f"トリミング:")
    print(f"  元の長さ: {len(y)/sr:.2f} 秒")
    print(f"  トリミング後: {len(y_trimmed)/sr:.2f} 秒")
    print(f"  削減: {(1 - len(y_trimmed)/len(y))*100:.1f}%")
    
    # 2. 固定長へのパディング/トリミング
    target_length = sr * 5  # 5秒
    
    def pad_or_trim(audio, target_length):
        """固定長にパディングまたはトリミング"""
        if len(audio) < target_length:
            # パディング
            pad_length = target_length - len(audio)
            audio_padded = np.pad(audio, (0, pad_length), mode='constant')
            return audio_padded, 'padded'
        else:
            # トリミング
            return audio[:target_length], 'trimmed'
    
    y_fixed, action = pad_or_trim(y, target_length)
    print(f"\n固定長処理 (5秒):")
    print(f"  アクション: {action}")
    print(f"  結果の長さ: {len(y_fixed)/sr:.2f} 秒")
    
    # 3. 中央パディング
    def center_pad(audio, target_length):
        """中央にパディング"""
        if len(audio) >= target_length:
            return audio[:target_length]
    
        pad_length = target_length - len(audio)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        return np.pad(audio, (pad_left, pad_right), mode='constant')
    
    y_center_padded = center_pad(y_trimmed, target_length)
    print(f"\n中央パディング:")
    print(f"  結果の長さ: {len(y_center_padded)/sr:.2f} 秒")
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # オリジナル
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='blue')
    axes[0].set_title('オリジナル音声', fontsize=14)
    axes[0].set_ylabel('振幅')
    axes[0].axvline(x=index[0]/sr, color='red', linestyle='--', alpha=0.5)
    axes[0].axvline(x=index[1]/sr, color='red', linestyle='--', alpha=0.5)
    
    # トリミング後
    librosa.display.waveshow(y_trimmed, sr=sr, ax=axes[1], color='green')
    axes[1].set_title('無音トリミング後', fontsize=14)
    axes[1].set_ylabel('振幅')
    
    # 固定長
    librosa.display.waveshow(y_fixed, sr=sr, ax=axes[2], color='red')
    axes[2].set_title(f'固定長処理後 (5秒, {action})', fontsize=14)
    axes[2].set_ylabel('振幅')
    
    # 中央パディング
    librosa.display.waveshow(y_center_padded, sr=sr, ax=axes[3], color='purple')
    axes[3].set_title('中央パディング', fontsize=14)
    axes[3].set_ylabel('振幅')
    axes[3].set_xlabel('時間 (秒)')
    
    plt.tight_layout()
    plt.savefig('trim_pad.png', dpi=150, bbox_inches='tight')
    print("\n図を保存しました: trim_pad.png")
    

### 1.5.4 Data Augmentation
    
    
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 音声の読み込み
    y, sr = librosa.load('sample.wav', duration=3.0)
    
    print("=== データ拡張（Data Augmentation）===")
    
    # 1. Pitch Shifting（ピッチシフト）
    y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)  # 4半音上げる
    y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4)  # 4半音下げる
    print("ピッチシフト: ±4半音")
    
    # 2. Time Stretching（時間伸縮）
    y_fast = librosa.effects.time_stretch(y, rate=1.5)  # 1.5倍速
    y_slow = librosa.effects.time_stretch(y, rate=0.8)  # 0.8倍速
    print(f"タイムストレッチ: 1.5倍速({len(y_fast)/sr:.2f}秒), 0.8倍速({len(y_slow)/sr:.2f}秒)")
    
    # 3. ノイズ追加
    noise_factor = 0.005
    noise = np.random.randn(len(y))
    y_noise = y + noise_factor * noise
    print(f"ホワイトノイズ追加: ノイズレベル = {noise_factor}")
    
    # 4. ゲイン調整（音量変化）
    y_louder = y * 1.5
    y_quieter = y * 0.5
    print("ゲイン調整: 1.5倍, 0.5倍")
    
    # 5. Time Shift（時間シフト）
    shift_samples = int(0.2 * sr)  # 0.2秒シフト
    y_shift = np.roll(y, shift_samples)
    print(f"時間シフト: {shift_samples/sr:.2f}秒")
    
    # 可視化
    fig, axes = plt.subplots(6, 1, figsize=(14, 16))
    
    augmentations = [
        ('オリジナル', y),
        ('ピッチシフト (+4半音)', y_pitch_up),
        ('タイムストレッチ (1.5x)', y_fast),
        ('ホワイトノイズ追加', y_noise),
        ('ゲイン調整 (1.5x)', y_louder),
        ('時間シフト (0.2秒)', y_shift)
    ]
    
    for idx, (title, y_aug) in enumerate(augmentations):
        librosa.display.waveshow(y_aug, sr=sr, ax=axes[idx])
        axes[idx].set_title(title, fontsize=14)
        axes[idx].set_ylabel('振幅')
    
    axes[5].set_xlabel('時間 (秒)')
    
    plt.tight_layout()
    plt.savefig('data_augmentation.png', dpi=150, bbox_inches='tight')
    print("\n図を保存しました: data_augmentation.png")
    
    # データ拡張パイプライン
    def augment_audio(y, sr):
        """ランダムにデータ拡張を適用"""
        augmentations = []
    
        # ピッチシフト (50%の確率)
        if np.random.rand() > 0.5:
            n_steps = np.random.randint(-3, 4)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            augmentations.append(f'pitch_shift({n_steps})')
    
        # タイムストレッチ (50%の確率)
        if np.random.rand() > 0.5:
            rate = np.random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y, rate=rate)
            augmentations.append(f'time_stretch({rate:.2f})')
    
        # ノイズ追加 (30%の確率)
        if np.random.rand() > 0.7:
            noise = np.random.randn(len(y))
            y = y + 0.005 * noise
            augmentations.append('add_noise')
    
        # ゲイン調整 (50%の確率)
        if np.random.rand() > 0.5:
            gain = np.random.uniform(0.7, 1.3)
            y = y * gain
            augmentations.append(f'gain({gain:.2f})')
    
        return y, augmentations
    
    # 使用例
    print("\nランダム拡張の例:")
    for i in range(3):
        y_aug, augs = augment_audio(y, sr)
        print(f"  試行 {i+1}: {', '.join(augs) if augs else 'なし'}")
    

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **デジタル音声の基礎**

     * サンプリングレート：時間方向の離散化
     * 量子化とビット深度：振幅の離散化
     * 音声ファイル形式：WAV、MP3、FLAC
     * librosaによる音声の読み込みと基本操作
  2. **時間領域の処理**

     * 振幅、エネルギー、RMS
     * Zero Crossing Rate（ZCR）
     * 音声波形の可視化
  3. **周波数領域の処理**

     * フーリエ変換（FFT）：時間→周波数
     * スペクトログラム：時間-周波数表現
     * メルスペクトログラム：人間の聴覚特性を考慮
  4. **音響特徴量**

     * MFCC：音声認識の標準特徴量
     * Chroma：音楽の調性表現
     * Spectral Features：周波数分布の特性
     * 特徴抽出パイプライン
  5. **音声データの前処理**

     * ノイズ除去：ウィーナーフィルタ、スペクトルサブトラクション
     * 正規化：ピーク、RMS、Z-score
     * トリミングとパディング：固定長化
     * データ拡張：ピッチシフト、タイムストレッチ

### 処理パイプラインの例
    
    
    ```mermaid
    graph LR
        A[音声ファイル] --> B[読み込みlibrosa.load]
        B --> C[前処理トリミング・正規化]
        C --> D[特徴抽出MFCC/Mel-Spec]
        D --> E[データ拡張ピッチ・時間]
        E --> F[モデル入力NumPy配列]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#e8f5e9
        style F fill:#c8e6c9
    ```

### 特徴量の選択ガイドライン

タスク | 推奨特徴量 | 理由  
---|---|---  
音声認識 | MFCC + デルタ | 音素の特徴を捉える  
話者識別 | MFCC + ピッチ | 話者固有の特性  
音楽ジャンル分類 | Chroma + Spectral | 調性とテクスチャ  
環境音認識 | Mel-Spectrogram | 時間-周波数パターン  
感情認識 | MFCC + 韻律特徴 | 音声の表現力  
  
### 次の章へ

第2章では、**音声認識の基礎** を学びます：

  * 音声認識の歴史と発展
  * HMM（隠れマルコフモデル）
  * ディープラーニングベースの音声認識
  * CTC（Connectionist Temporal Classification）
  * End-to-Endモデル（Wav2Vec、Whisper）

* * *

## 演習問題

### 問題1（難易度：easy）

サンプリングレート44100Hzで録音された3秒の音声ファイルは、何サンプルで構成されますか？また、ナイキスト定理に基づいて、この音声ファイルが正確に表現できる最高周波数はいくらですか？

解答例

**解答** ：

**サンプル数の計算** ：

$$ \text{サンプル数} = \text{サンプリングレート} \times \text{時間長} $$

$$ = 44100 \text{ Hz} \times 3 \text{ 秒} = 132300 \text{ サンプル} $$

**最高周波数の計算（ナイキスト定理）** ：

$$ f_{\max} = \frac{f_s}{2} = \frac{44100}{2} = 22050 \text{ Hz} $$

**答え** ：

  * サンプル数: 132,300サンプル
  * 最高周波数: 22,050 Hz (22.05 kHz)

**補足** ：人間の可聴域は約20-20,000 Hzなので、44,100 Hzのサンプリングレートは音楽CDに十分な品質です。

### 問題2（難易度：medium）

以下のコードを完成させて、音声ファイルからMFCCを抽出し、その統計量（平均と標準偏差）を計算してください。
    
    
    import librosa
    import numpy as np
    
    # 音声の読み込み
    y, sr = librosa.load('sample.wav')
    
    # MFCCの抽出（13係数）
    # TODO: ここにコードを追加
    
    # 各MFCC係数の平均と標準偏差を計算
    # TODO: ここにコードを追加
    
    # 結果の表示
    # TODO: ここにコードを追加
    

解答例
    
    
    import librosa
    import numpy as np
    
    # 音声の読み込み
    y, sr = librosa.load('sample.wav')
    
    # MFCCの抽出（13係数）
    n_mfcc = 13
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    print(f"=== MFCC統計量 ===")
    print(f"MFCCの形状: {mfccs.shape}")
    print(f"  係数数: {n_mfcc}")
    print(f"  時間フレーム数: {mfccs.shape[1]}")
    
    # 各MFCC係数の平均と標準偏差を計算
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # 結果の表示
    print("\n各MFCC係数の統計量:")
    print(f"{'係数':<10} {'平均':<15} {'標準偏差':<15}")
    print("-" * 40)
    for i in range(n_mfcc):
        print(f"MFCC-{i:<4} {mfcc_mean[i]:<15.4f} {mfcc_std[i]:<15.4f}")
    
    # 特徴ベクトルとして結合（機械学習で使用）
    feature_vector = np.concatenate([mfcc_mean, mfcc_std])
    print(f"\n特徴ベクトルの次元: {len(feature_vector)}")
    print(f"  (13平均 + 13標準偏差 = 26次元)")
    
    # 可視化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # MFCCヒートマップ
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[0], cmap='coolwarm')
    axes[0].set_title('MFCC', fontsize=14)
    axes[0].set_ylabel('MFCC係数')
    fig.colorbar(img, ax=axes[0])
    
    # 統計量のバープロット
    x = np.arange(n_mfcc)
    width = 0.35
    axes[1].bar(x - width/2, mfcc_mean, width, label='平均', alpha=0.8)
    axes[1].bar(x + width/2, mfcc_std, width, label='標準偏差', alpha=0.8)
    axes[1].set_xlabel('MFCC係数')
    axes[1].set_ylabel('値')
    axes[1].set_title('MFCC統計量', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mfcc_statistics.png', dpi=150)
    print("\n図を保存しました: mfcc_statistics.png")
    

**出力例** ：
    
    
    === MFCC統計量 ===
    MFCCの形状: (13, 130)
      係数数: 13
      時間フレーム数: 130
    
    各MFCC係数の統計量:
    係数        平均             標準偏差
    ----------------------------------------
    MFCC-0    -123.4567       45.2341
    MFCC-1     32.1234        12.5678
    MFCC-2     -8.9012        7.3456
    MFCC-3     5.6789         6.1234
    ...
    
    特徴ベクトルの次元: 26
      (13平均 + 13標準偏差 = 26次元)
    

### 問題3（難易度：medium）

スペクトログラムとメルスペクトログラムの違いを説明し、なぜメルスケールが音声処理で有用なのか述べてください。

解答例

**解答** ：

**スペクトログラム vs メルスペクトログラム** ：

特性 | スペクトログラム | メルスペクトログラム  
---|---|---  
**周波数スケール** | 線形（Hz） | メルスケール（対数的）  
**周波数分解能** | 全周波数で均一 | 低周波数で高く、高周波数で低い  
**次元数** | 高い（n_fft/2 + 1） | 低い（通常40-128）  
**計算方法** | STFT | STFT + メルフィルタバンク  
  
**メルスケールの有用性** ：

  1. **人間の聴覚特性に適合**

     * 人間の耳は低周波数の変化に敏感、高周波数では鈍感
     * メルスケールはこの特性を数学的にモデル化
  2. **次元削減**

     * 線形スペクトログラム：1025次元（n_fft=2048の場合）
     * メルスペクトログラム：40-128次元
     * 計算効率と学習効率の向上
  3. **音声認識への適性**

     * 音声の重要な情報（低〜中周波数）に焦点
     * 高周波数のノイズの影響を軽減
  4. **変換式**

$$ m = 2595 \log_{10}\left(1 + \frac{f}{700}\right) $$

     * 低周波数（例：100Hz→150Hz）：大きなメル変化
     * 高周波数（例：5000Hz→5100Hz）：小さなメル変化

**実用例** ：

  * 音声認識：メルスペクトログラム → MFCC
  * 音楽情報検索：メルスペクトログラム直接使用
  * 音声合成：メルスペクトログラムから音声生成（WaveNet、Tacotron）

### 問題4（難易度：hard）

音声ファイルに対して、以下のデータ拡張を適用し、元の音声と拡張後の音声のMFCCを比較してください：

  1. ピッチシフト（+3半音）
  2. タイムストレッチ（1.2倍速）
  3. ホワイトノイズ追加（SNR=20dB）

解答例
    
    
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 音声の読み込み
    y, sr = librosa.load('sample.wav', duration=3.0)
    
    print("=== データ拡張とMFCC比較 ===")
    
    # オリジナルのMFCC
    mfcc_original = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 1. ピッチシフト（+3半音）
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=3)
    mfcc_pitch = librosa.feature.mfcc(y=y_pitch, sr=sr, n_mfcc=13)
    print("1. ピッチシフト: +3半音")
    
    # 2. タイムストレッチ（1.2倍速）
    y_stretch = librosa.effects.time_stretch(y, rate=1.2)
    mfcc_stretch = librosa.feature.mfcc(y=y_stretch, sr=sr, n_mfcc=13)
    print("2. タイムストレッチ: 1.2倍速")
    
    # 3. ホワイトノイズ追加（SNR=20dB）
    def add_noise(signal, snr_db):
        """指定したSNRでホワイトノイズを追加"""
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return signal + noise
    
    y_noise = add_noise(y, snr_db=20)
    mfcc_noise = librosa.feature.mfcc(y=y_noise, sr=sr, n_mfcc=13)
    print("3. ホワイトノイズ追加: SNR=20dB")
    
    # MFCC統計の比較
    def mfcc_statistics(mfcc):
        return {
            'mean': np.mean(mfcc, axis=1),
            'std': np.std(mfcc, axis=1)
        }
    
    stats_original = mfcc_statistics(mfcc_original)
    stats_pitch = mfcc_statistics(mfcc_pitch)
    stats_stretch = mfcc_statistics(mfcc_stretch)
    stats_noise = mfcc_statistics(mfcc_noise)
    
    print("\nMFCC統計の比較（最初の5係数）:")
    print(f"{'係数':<10} {'元の平均':<15} {'ピッチ平均':<15} {'ストレッチ平均':<15} {'ノイズ平均':<15}")
    print("-" * 70)
    for i in range(5):
        print(f"MFCC-{i:<4} {stats_original['mean'][i]:<15.3f} {stats_pitch['mean'][i]:<15.3f} "
              f"{stats_stretch['mean'][i]:<15.3f} {stats_noise['mean'][i]:<15.3f}")
    
    # ユークリッド距離の計算
    def euclidean_distance(mfcc1, mfcc2):
        """2つのMFCC間のユークリッド距離"""
        # 時間方向で平均を取る
        mean1 = np.mean(mfcc1, axis=1)
        mean2 = np.mean(mfcc2, axis=1)
        return np.linalg.norm(mean1 - mean2)
    
    dist_pitch = euclidean_distance(mfcc_original, mfcc_pitch)
    dist_stretch = euclidean_distance(mfcc_original, mfcc_stretch)
    dist_noise = euclidean_distance(mfcc_original, mfcc_noise)
    
    print("\n元のMFCCとの距離:")
    print(f"  ピッチシフト: {dist_pitch:.3f}")
    print(f"  タイムストレッチ: {dist_stretch:.3f}")
    print(f"  ノイズ追加: {dist_noise:.3f}")
    
    # 可視化
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    
    augmentations = [
        ('オリジナル', y, mfcc_original),
        ('ピッチシフト (+3半音)', y_pitch, mfcc_pitch),
        ('タイムストレッチ (1.2x)', y_stretch, mfcc_stretch),
        ('ホワイトノイズ (SNR=20dB)', y_noise, mfcc_noise)
    ]
    
    for idx, (title, audio, mfcc) in enumerate(augmentations):
        # 波形
        librosa.display.waveshow(audio, sr=sr, ax=axes[idx, 0])
        axes[idx, 0].set_title(f'{title} - 波形', fontsize=12)
        axes[idx, 0].set_ylabel('振幅')
    
        # MFCC
        img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[idx, 1], cmap='coolwarm')
        axes[idx, 1].set_title(f'{title} - MFCC', fontsize=12)
        axes[idx, 1].set_ylabel('MFCC係数')
        fig.colorbar(img, ax=axes[idx, 1])
    
    axes[3, 0].set_xlabel('時間 (秒)')
    axes[3, 1].set_xlabel('時間 (秒)')
    
    plt.tight_layout()
    plt.savefig('augmentation_mfcc_comparison.png', dpi=150, bbox_inches='tight')
    print("\n図を保存しました: augmentation_mfcc_comparison.png")
    
    # 考察
    print("\n=== 考察 ===")
    print("1. ピッチシフト:")
    print("   - MFCCの値が変化（特に低次係数）")
    print("   - 話者や音高情報が変化")
    print("\n2. タイムストレッチ:")
    print("   - フレーム数が変化")
    print("   - MFCC統計は比較的類似")
    print("\n3. ノイズ追加:")
    print("   - 高次MFCC係数が影響を受ける")
    print("   - 低次係数は比較的ロバスト")
    

**出力例** ：
    
    
    === データ拡張とMFCC比較 ===
    1. ピッチシフト: +3半音
    2. タイムストレッチ: 1.2倍速
    3. ホワイトノイズ追加: SNR=20dB
    
    MFCC統計の比較（最初の5係数）:
    係数        元の平均         ピッチ平均       ストレッチ平均   ノイズ平均
    ----------------------------------------------------------------------
    MFCC-0    -123.456        -125.234        -123.789        -122.891
    MFCC-1     32.123          35.678          32.456          31.234
    MFCC-2     -8.901          -10.234         -8.756          -9.123
    MFCC-3     5.678           6.234           5.789           5.456
    MFCC-4     -3.456          -4.123          -3.567          -3.891
    
    元のMFCCとの距離:
      ピッチシフト: 12.345
      タイムストレッチ: 2.567
      ノイズ追加: 4.891
    
    === 考察 ===
    1. ピッチシフト:
       - MFCCの値が変化（特に低次係数）
       - 話者や音高情報が変化
    
    2. タイムストレッチ:
       - フレーム数が変化
       - MFCC統計は比較的類似
    
    3. ノイズ追加:
       - 高次MFCC係数が影響を受ける
       - 低次係数は比較的ロバスト
    

### 問題5（難易度：hard）

音声データの前処理パイプラインを実装してください。パイプラインには、以下を含めてください：

  1. 無音部分のトリミング
  2. RMS正規化（目標RMS=0.1）
  3. 固定長への調整（5秒）
  4. メルスペクトログラムの抽出
  5. 正規化（平均0、標準偏差1）

解答例
    
    
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    class AudioPreprocessor:
        """音声前処理パイプライン"""
    
        def __init__(self, sr=22050, target_length=5.0, target_rms=0.1,
                     n_mels=128, n_fft=2048, hop_length=512):
            """
            Parameters:
            -----------
            sr : int
                サンプリングレート
            target_length : float
                目標の時間長（秒）
            target_rms : float
                目標のRMSレベル
            n_mels : int
                メルバンド数
            n_fft : int
                FFTサイズ
            hop_length : int
                ホップ長
            """
            self.sr = sr
            self.target_length = target_length
            self.target_samples = int(sr * target_length)
            self.target_rms = target_rms
            self.n_mels = n_mels
            self.n_fft = n_fft
            self.hop_length = hop_length
    
        def trim_silence(self, y, top_db=20):
            """1. 無音部分のトリミング"""
            y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
            return y_trimmed
    
        def rms_normalize(self, y):
            """2. RMS正規化"""
            current_rms = np.sqrt(np.mean(y**2))
            if current_rms > 0:
                y_normalized = y * (self.target_rms / current_rms)
            else:
                y_normalized = y
            return y_normalized
    
        def fix_length(self, y):
            """3. 固定長への調整"""
            if len(y) < self.target_samples:
                # パディング
                pad_length = self.target_samples - len(y)
                y_fixed = np.pad(y, (0, pad_length), mode='constant')
            else:
                # トリミング
                y_fixed = y[:self.target_samples]
            return y_fixed
    
        def extract_mel_spectrogram(self, y):
            """4. メルスペクトログラムの抽出"""
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_mels=self.n_mels,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            # 対数スケール
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
    
        def standardize(self, mel_spec):
            """5. 正規化（平均0、標準偏差1）"""
            mean = np.mean(mel_spec)
            std = np.std(mel_spec)
            if std > 0:
                mel_spec_normalized = (mel_spec - mean) / std
            else:
                mel_spec_normalized = mel_spec - mean
            return mel_spec_normalized
    
        def process(self, audio_path, verbose=True):
            """完全な前処理パイプライン"""
            if verbose:
                print(f"=== 音声前処理パイプライン ===")
                print(f"入力: {audio_path}")
    
            # 音声の読み込み
            y, sr = librosa.load(audio_path, sr=self.sr)
            if verbose:
                print(f"0. 読み込み: {len(y)} サンプル ({len(y)/sr:.2f}秒)")
    
            # 1. 無音トリミング
            y = self.trim_silence(y)
            if verbose:
                print(f"1. トリミング後: {len(y)} サンプル ({len(y)/sr:.2f}秒)")
    
            # 2. RMS正規化
            y = self.rms_normalize(y)
            current_rms = np.sqrt(np.mean(y**2))
            if verbose:
                print(f"2. RMS正規化: {current_rms:.4f} (目標: {self.target_rms})")
    
            # 3. 固定長調整
            y = self.fix_length(y)
            if verbose:
                print(f"3. 固定長調整: {len(y)} サンプル ({len(y)/sr:.2f}秒)")
    
            # 4. メルスペクトログラム抽出
            mel_spec = self.extract_mel_spectrogram(y)
            if verbose:
                print(f"4. メルスペクトログラム: {mel_spec.shape}")
    
            # 5. 標準化
            mel_spec_normalized = self.standardize(mel_spec)
            if verbose:
                mean = np.mean(mel_spec_normalized)
                std = np.std(mel_spec_normalized)
                print(f"5. 標準化: 平均={mean:.6f}, 標準偏差={std:.6f}")
    
            return {
                'audio': y,
                'mel_spectrogram_raw': mel_spec,
                'mel_spectrogram': mel_spec_normalized,
                'shape': mel_spec_normalized.shape
            }
    
    # 使用例
    preprocessor = AudioPreprocessor(
        sr=22050,
        target_length=5.0,
        target_rms=0.1,
        n_mels=128
    )
    
    # 前処理の実行
    result = preprocessor.process('sample.wav')
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 波形
    librosa.display.waveshow(result['audio'], sr=preprocessor.sr, ax=axes[0])
    axes[0].set_title('前処理後の波形', fontsize=14)
    axes[0].set_ylabel('振幅')
    
    # 2. 生のメルスペクトログラム
    img1 = librosa.display.specshow(result['mel_spectrogram_raw'],
                                      sr=preprocessor.sr,
                                      hop_length=preprocessor.hop_length,
                                      x_axis='time', y_axis='mel',
                                      ax=axes[1], cmap='viridis')
    axes[1].set_title('メルスペクトログラム（dBスケール）', fontsize=14)
    axes[1].set_ylabel('周波数 (メル)')
    fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
    
    # 3. 正規化後のメルスペクトログラム
    img2 = librosa.display.specshow(result['mel_spectrogram'],
                                      sr=preprocessor.sr,
                                      hop_length=preprocessor.hop_length,
                                      x_axis='time', y_axis='mel',
                                      ax=axes[2], cmap='viridis')
    axes[2].set_title('正規化後のメルスペクトログラム', fontsize=14)
    axes[2].set_ylabel('周波数 (メル)')
    axes[2].set_xlabel('時間 (秒)')
    fig.colorbar(img2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('preprocessing_pipeline.png', dpi=150, bbox_inches='tight')
    print("\n図を保存しました: preprocessing_pipeline.png")
    
    # バッチ処理
    print("\n=== バッチ処理 ===")
    audio_files = ['sample1.wav', 'sample2.wav', 'sample3.wav']
    batch_results = []
    
    for audio_file in audio_files:
        try:
            result = preprocessor.process(audio_file, verbose=False)
            batch_results.append(result['mel_spectrogram'])
            print(f"✓ {audio_file}: {result['shape']}")
        except Exception as e:
            print(f"✗ {audio_file}: エラー ({e})")
    
    # NumPy配列に変換（モデルの入力として使用）
    if batch_results:
        X = np.array(batch_results)
        print(f"\nバッチの形状: {X.shape}")
        print(f"  (バッチサイズ, メルバンド数, 時間フレーム数)")
        print(f"\nこのデータは機械学習モデルの入力として使用可能です。")
    

**出力例** ：
    
    
    === 音声前処理パイプライン ===
    入力: sample.wav
    0. 読み込み: 66150 サンプル (3.00秒)
    1. トリミング後: 55125 サンプル (2.50秒)
    2. RMS正規化: 0.1000 (目標: 0.1)
    3. 固定長調整: 110250 サンプル (5.00秒)
    4. メルスペクトログラム: (128, 216)
    5. 標準化: 平均=0.000000, 標準偏差=1.000000
    
    図を保存しました: preprocessing_pipeline.png
    
    === バッチ処理 ===
    ✓ sample1.wav: (128, 216)
    ✓ sample2.wav: (128, 216)
    ✓ sample3.wav: (128, 216)
    
    バッチの形状: (3, 128, 216)
      (バッチサイズ, メルバンド数, 時間フレーム数)
    
    このデータは機械学習モデルの入力として使用可能です。
    

* * *

## 参考文献

  1. Rabiner, L., & Schafer, R. (2010). _Theory and Applications of Digital Speech Processing_. Pearson.
  2. Müller, M. (2015). _Fundamentals of Music Processing_. Springer.
  3. McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python." _Proceedings of the 14th Python in Science Conference_.
  4. Davis, S., & Mermelstein, P. (1980). "Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences." _IEEE Transactions on ASSP_.
  5. Logan, B. (2000). "Mel Frequency Cepstral Coefficients for Music Modeling." _ISMIR_.
  6. Oppenheim, A. V., & Schafer, R. W. (2009). _Discrete-Time Signal Processing_ (3rd ed.). Pearson.
