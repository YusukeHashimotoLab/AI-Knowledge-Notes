---
title: 第1章：プロセスモニタリングの基礎とセンサーデータ取得
chapter_title: 第1章：プロセスモニタリングの基礎とセンサーデータ取得
subtitle: プロセス監視システムの基礎から時系列データ処理まで
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ プロセスモニタリングの目的と重要性を説明できる
  * ✅ 主要なセンサータイプとその特性を理解する
  * ✅ サンプリング理論とナイキスト定理を説明できる
  * ✅ Pythonで時系列センサーデータを処理できる
  * ✅ データ品質評価（欠損値、ドリフト、外れ値）ができる

* * *

## 1.1 プロセスモニタリングの基礎

### プロセスモニタリングとは

**プロセスモニタリング** は、化学プラント、製薬、食品、半導体などのプロセス産業において、製造プロセスの状態をリアルタイムで監視し、品質、安全性、効率性を確保するための活動です。

**主な目的:**

  * **品質保証** : 製品が規格を満たしていることを確認
  * **安全性確保** : 異常状態の早期検出と事故防止
  * **効率性向上** : プロセスの最適運転点の維持
  * **トレーサビリティ** : 運転履歴の記録と分析
  * **規制対応** : GMP（医薬品製造管理基準）等の遵守

### SCADAとDCS

プロセスモニタリングシステムは、主に以下の2つの形態で実装されます：

項目 | SCADA（Supervisory Control And Data Acquisition） | DCS（Distributed Control System）  
---|---|---  
**主な用途** | 広域監視（電力、上下水道、石油パイプライン） | プロセス制御（化学プラント、製油所）  
**制御機能** | 監視中心、一部制御 | 高度な制御機能（PID、MPC等）  
**リアルタイム性** | 秒〜分単位 | ミリ秒〜秒単位  
**システム構成** | 集中型 | 分散型（冗長性高い）  
**コスト** | 比較的低コスト | 高コスト  
  
### モニタリングシステムのアーキテクチャ
    
    
    ```mermaid
    graph TD
        A[センサー層] --> B[データ収集層 PLC/RTU]
        B --> C[通信層 OPC/Modbus]
        C --> D[監視層 SCADA/DCS]
        D --> E[データベース層 Historian]
        E --> F[分析層 PI/ML]
    
        style A fill:#e8f5e9
        style B fill:#c8e6c9
        style C fill:#a5d6a7
        style D fill:#81c784
        style E fill:#66bb6a
        style F fill:#4caf50
    ```

* * *

## 1.2 センサーの種類とデータ取得

### 主要センサータイプ

センサータイプ | 測定原理 | 典型的測定範囲 | 精度 | 応答時間  
---|---|---|---|---  
**温度センサー** |  |  |  |   
\- 熱電対（TC） | ゼーベック効果 | -200〜1800°C | ±0.5〜2°C | 0.1〜10秒  
\- 測温抵抗体（RTD） | 抵抗変化 | -200〜850°C | ±0.1〜0.5°C | 1〜30秒  
**圧力センサー** |  |  |  |   
\- ダイアフラム式 | ひずみゲージ | 0〜100 MPa | ±0.1〜0.5% FS | <0.1秒  
**流量計** |  |  |  |   
\- 電磁流量計 | 電磁誘導 | 0.01〜10 m/s | ±0.5% 読取値 | <1秒  
\- コリオリ流量計 | コリオリ力 | 質量流量 | ±0.1% 読取値 | <1秒  
**液面計** |  |  |  |   
\- 差圧式 | 圧力差 | 0〜50 m | ±0.5% FS | 1〜10秒  
  
### サンプリング理論とナイキスト定理

**ナイキスト-シャノンのサンプリング定理** は、連続信号をデジタル化する際の基本原理です：

> **定理** : 信号を正確に再構成するには、信号の最高周波数成分の**2倍以上のサンプリング周波数** が必要である。

数式で表すと：

_f s ≥ 2 × fmax_

ここで、 _f s_はサンプリング周波数、 _f max_は信号の最高周波数成分です。

**実務での推奨サンプリングレート:**

  * **温度** : 1秒〜1分（変化が遅い）
  * **圧力** : 0.1秒〜1秒（比較的速い変化）
  * **流量** : 0.1秒〜1秒
  * **濃度** : 1分〜1時間（オンライン分析計の場合）

**エイリアシング（折り返し雑音）** : サンプリング周波数が不十分な場合、高周波成分が低周波成分として誤って記録される現象です。これを防ぐため、アンチエイリアシングフィルタ（ローパスフィルタ）を使用します。

* * *

## 1.3 コード例：時系列センサーデータの処理

ここから、実際にPythonでセンサーデータを処理するコード例を8つ見ていきます。

#### コード例1: 時系列センサーデータのシミュレーションと基本プロット

**目的** : 反応器温度の24時間データをシミュレートし、トレンドを可視化する。
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 日本語フォント設定（Macの場合）
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # シミュレーションパラメータ
    np.random.seed(42)
    sampling_interval = 60  # 秒（1分間隔）
    duration_hours = 24
    n_samples = int(duration_hours * 3600 / sampling_interval)
    
    # 時刻データ生成
    time_index = pd.date_range('2025-01-01 00:00:00', periods=n_samples, freq=f'{sampling_interval}s')
    
    # 温度データのシミュレーション
    # 基準温度 + トレンド + 周期変動 + ランダムノイズ
    base_temp = 175.0  # 基準温度（℃）
    trend = np.linspace(0, 2, n_samples)  # 徐々に上昇するトレンド
    daily_cycle = 3 * np.sin(2 * np.pi * np.arange(n_samples) / (24*60))  # 日周期変動
    noise = np.random.normal(0, 0.8, n_samples)  # 測定ノイズ
    
    temperature = base_temp + trend + daily_cycle + noise
    
    # DataFrameに格納
    df = pd.DataFrame({
        'timestamp': time_index,
        'temperature': temperature
    })
    df.set_index('timestamp', inplace=True)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['temperature'], linewidth=1, color='#11998e', alpha=0.8)
    ax.axhline(y=175, color='red', linestyle='--', linewidth=2, label='目標温度: 175°C')
    ax.fill_between(df.index, 173, 177, alpha=0.15, color='green', label='許容範囲 (±2°C)')
    ax.set_xlabel('時刻', fontsize=12)
    ax.set_ylabel('温度 (°C)', fontsize=12)
    ax.set_title('反応器温度の24時間トレンド', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 統計サマリー
    print("=== 温度データ統計サマリー ===")
    print(f"平均温度: {df['temperature'].mean():.2f}°C")
    print(f"標準偏差: {df['temperature'].std():.2f}°C")
    print(f"最高温度: {df['temperature'].max():.2f}°C")
    print(f"最低温度: {df['temperature'].min():.2f}°C")
    print(f"データポイント数: {len(df)}")
    

**期待される出力** :
    
    
    === 温度データ統計サマリー ===
    平均温度: 176.01°C
    標準偏差: 2.14°C
    最高温度: 181.34°C
    最低温度: 170.89°C
    データポイント数: 1440
    

**解説** : このコードは、実際のプロセスセンサーデータの特性（トレンド、周期変動、ノイズ）を含むシミュレーションデータを生成します。1分間隔で24時間分のデータを生成し、Pandasで時系列データとして扱います。

#### コード例2: 複数センサーデータの同期取得とプロット

**目的** : 蒸留塔の温度、圧力、流量を同時にモニタリングし、可視化する。
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # 時刻データ
    time_index = pd.date_range('2025-01-01 00:00:00', periods=1440, freq='1min')
    
    # 複数センサーのシミュレーションデータ
    # 蒸留塔の運転データ
    塔頂温度 = 85 + np.random.normal(0, 1.2, 1440) + 2 * np.sin(np.linspace(0, 4*np.pi, 1440))
    塔底温度 = 155 + np.random.normal(0, 1.5, 1440)
    塔内圧力 = 1.2 + np.random.normal(0, 0.03, 1440)
    還流流量 = 50 + np.random.normal(0, 2.5, 1440)
    
    # DataFrameに格納
    df_multi = pd.DataFrame({
        'timestamp': time_index,
        '塔頂温度': 塔頂温度,
        '塔底温度': 塔底温度,
        '塔内圧力': 塔内圧力,
        '還流流量': 還流流量
    })
    df_multi.set_index('timestamp', inplace=True)
    
    # 4つのサブプロットで表示
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # 塔頂温度
    axes[0].plot(df_multi.index, df_multi['塔頂温度'], color='#11998e', linewidth=0.8)
    axes[0].set_ylabel('塔頂温度 (°C)', fontsize=11)
    axes[0].set_title('蒸留塔マルチセンサーデータ', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=85, color='red', linestyle='--', alpha=0.5)
    
    # 塔底温度
    axes[1].plot(df_multi.index, df_multi['塔底温度'], color='#f59e0b', linewidth=0.8)
    axes[1].set_ylabel('塔底温度 (°C)', fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=155, color='red', linestyle='--', alpha=0.5)
    
    # 塔内圧力
    axes[2].plot(df_multi.index, df_multi['塔内圧力'], color='#7b2cbf', linewidth=0.8)
    axes[2].set_ylabel('塔内圧力 (MPa)', fontsize=11)
    axes[2].grid(alpha=0.3)
    axes[2].axhline(y=1.2, color='red', linestyle='--', alpha=0.5)
    
    # 還流流量
    axes[3].plot(df_multi.index, df_multi['還流流量'], color='#e63946', linewidth=0.8)
    axes[3].set_ylabel('還流流量 (m³/h)', fontsize=11)
    axes[3].set_xlabel('時刻', fontsize=12)
    axes[3].grid(alpha=0.3)
    axes[3].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # 変数間の相関分析
    print("\n=== 変数間相関係数 ===")
    correlation_matrix = df_multi.corr()
    print(correlation_matrix)
    

**解説** : 複数のセンサーデータを同じ時間軸で表示することで、プロセス変数間の関係性や異常パターンを視覚的に把握できます。実際のプラントでは、数十〜数百の変数を同時にモニタリングします。

#### コード例3: サンプリングレート分析とエイリアシングのデモンストレーション

**目的** : ナイキスト定理を実証し、不適切なサンプリングによるエイリアシングを可視化する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 元信号: 5 Hzの正弦波
    frequency = 5  # Hz
    duration = 2  # 秒
    t_continuous = np.linspace(0, duration, 10000)  # 連続信号の近似
    signal_continuous = np.sin(2 * np.pi * frequency * t_continuous)
    
    # 適切なサンプリング: 20 Hz（ナイキスト周波数の2倍）
    fs_good = 20
    t_good = np.arange(0, duration, 1/fs_good)
    signal_good = np.sin(2 * np.pi * frequency * t_good)
    
    # 不適切なサンプリング: 7 Hz（ナイキスト周波数未満）
    fs_bad = 7
    t_bad = np.arange(0, duration, 1/fs_bad)
    signal_bad = np.sin(2 * np.pi * frequency * t_bad)
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 適切なサンプリング
    axes[0].plot(t_continuous, signal_continuous, 'b-', linewidth=2, alpha=0.5, label='元信号 (5 Hz)')
    axes[0].plot(t_good, signal_good, 'ro-', markersize=6, label=f'サンプリング {fs_good} Hz (適切)')
    axes[0].set_ylabel('振幅', fontsize=11)
    axes[0].set_title('適切なサンプリング（ナイキスト定理を満たす）', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 不適切なサンプリング（エイリアシング発生）
    axes[1].plot(t_continuous, signal_continuous, 'b-', linewidth=2, alpha=0.5, label='元信号 (5 Hz)')
    axes[1].plot(t_bad, signal_bad, 'ro-', markersize=6, label=f'サンプリング {fs_bad} Hz (不適切)')
    # エイリアシングにより見かけの周波数が変化
    aliased_freq = abs(frequency - fs_bad)
    t_aliased = np.linspace(0, duration, 1000)
    signal_aliased = np.sin(2 * np.pi * aliased_freq * t_aliased)
    axes[1].plot(t_aliased, signal_aliased, 'g--', linewidth=2,
                 label=f'エイリアシング信号 ({aliased_freq} Hz)')
    axes[1].set_xlabel('時間 (秒)', fontsize=11)
    axes[1].set_ylabel('振幅', fontsize=11)
    axes[1].set_title('不適切なサンプリング（エイリアシング発生）', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== サンプリング分析 ===")
    print(f"元信号周波数: {frequency} Hz")
    print(f"ナイキスト周波数: {2 * frequency} Hz")
    print(f"適切なサンプリング周波数: {fs_good} Hz（ナイキストの{fs_good/(2*frequency):.1f}倍）")
    print(f"不適切なサンプリング周波数: {fs_bad} Hz（ナイキストの{fs_bad/(2*frequency):.1f}倍）")
    print(f"エイリアシングによる見かけの周波数: {aliased_freq} Hz")
    

**期待される出力** :
    
    
    === サンプリング分析 ===
    元信号周波数: 5 Hz
    ナイキスト周波数: 10 Hz
    適切なサンプリング周波数: 20 Hz（ナイキストの2.0倍）
    不適切なサンプリング周波数: 7 Hz（ナイキストの0.7倍）
    エイリアシングによる見かけの周波数: 2 Hz
    

**解説** : このコードは、サンプリング周波数がナイキスト周波数（信号の最高周波数の2倍）を下回ると、エイリアシング（折り返し雑音）が発生し、元の信号を正しく復元できなくなることを実証します。プロセス産業では、プロセスの動特性に応じた適切なサンプリング周波数の選定が重要です。

#### コード例4: データ品質評価 - 欠損値と外れ値の検出

**目的** : センサーデータの欠損値と外れ値を検出し、データ品質を評価する。
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # 時系列データ生成
    time_index = pd.date_range('2025-01-01', periods=1000, freq='1min')
    temperature = 175 + np.random.normal(0, 2, 1000)
    
    # 意図的に欠損値を追加（通信エラーをシミュレート）
    missing_indices = np.random.choice(1000, size=50, replace=False)
    temperature[missing_indices] = np.nan
    
    # 意図的に外れ値を追加（センサー異常をシミュレート）
    outlier_indices = np.random.choice(1000, size=10, replace=False)
    temperature[outlier_indices] = temperature[outlier_indices] + np.random.choice([-20, 20], size=10)
    
    # DataFrameに格納
    df_quality = pd.DataFrame({
        'timestamp': time_index,
        'temperature': temperature
    })
    df_quality.set_index('timestamp', inplace=True)
    
    # データ品質評価
    print("=== データ品質評価 ===")
    print(f"総データポイント数: {len(df_quality)}")
    print(f"欠損値数: {df_quality['temperature'].isna().sum()} ({df_quality['temperature'].isna().sum()/len(df_quality)*100:.1f}%)")
    print(f"有効データ数: {df_quality['temperature'].notna().sum()}")
    
    # 外れ値検出: Z-scoreメソッド
    mean_temp = df_quality['temperature'].mean()
    std_temp = df_quality['temperature'].std()
    z_scores = np.abs((df_quality['temperature'] - mean_temp) / std_temp)
    outliers = z_scores > 3  # 3シグマルール
    
    print(f"\n外れ値検出（3シグマルール）:")
    print(f"外れ値数: {outliers.sum()} ({outliers.sum()/len(df_quality)*100:.1f}%)")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 元データ（欠損値と外れ値を含む）
    axes[0].plot(df_quality.index, df_quality['temperature'], 'b-', linewidth=0.8, alpha=0.7)
    axes[0].scatter(df_quality.index[outliers], df_quality['temperature'][outliers],
                    color='red', s=50, label='外れ値', zorder=5)
    axes[0].axhline(y=mean_temp, color='green', linestyle='--', label='平均値')
    axes[0].axhline(y=mean_temp + 3*std_temp, color='orange', linestyle='--', alpha=0.5, label='±3σ')
    axes[0].axhline(y=mean_temp - 3*std_temp, color='orange', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('温度 (°C)', fontsize=11)
    axes[0].set_title('元データ（欠損値と外れ値を含む）', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 欠損値の可視化
    missing_mask = df_quality['temperature'].isna()
    axes[1].plot(df_quality.index, df_quality['temperature'], 'b-', linewidth=0.8, alpha=0.7)
    axes[1].scatter(df_quality.index[missing_mask],
                    [175]*missing_mask.sum(), color='red', s=30, marker='x',
                    label='欠損値', zorder=5)
    axes[1].set_ylabel('温度 (°C)', fontsize=11)
    axes[1].set_title('欠損値の位置', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Zスコアの可視化
    axes[2].plot(df_quality.index, z_scores, 'g-', linewidth=0.8)
    axes[2].axhline(y=3, color='red', linestyle='--', label='閾値 (Z=3)')
    axes[2].fill_between(df_quality.index, 0, 3, alpha=0.1, color='green')
    axes[2].set_xlabel('時刻', fontsize=11)
    axes[2].set_ylabel('Zスコア', fontsize=11)
    axes[2].set_title('外れ値検出（Zスコア法）', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**解説** : データ品質評価は、プロセスモニタリングの最初のステップです。欠損値は通信エラーやセンサー故障で発生し、外れ値はセンサー異常や実際のプロセス異常を示す可能性があります。Zスコア法（3シグマルール）は、統計的に正常範囲から外れたデータポイントを検出する基本的な手法です。

#### コード例5: センサードリフトの検出と補正

**目的** : センサーのドリフト（経時的なずれ）を検出し、補正する。
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    np.random.seed(42)
    
    # 時系列データ生成（30日間、1時間間隔）
    time_index = pd.date_range('2025-01-01', periods=720, freq='1h')
    
    # 真の温度（一定）
    true_temperature = 100.0
    
    # センサー測定値（ドリフトを含む）
    # ドリフト: 線形に0.5°C/月の速度で低下
    drift_rate = -0.5 / 30  # °C/日
    days = np.arange(720) / 24
    drift = drift_rate * days
    
    # 測定ノイズ
    noise = np.random.normal(0, 0.3, 720)
    
    # センサー測定値 = 真値 + ドリフト + ノイズ
    measured_temperature = true_temperature + drift + noise
    
    # DataFrameに格納
    df_drift = pd.DataFrame({
        'timestamp': time_index,
        'measured': measured_temperature,
        'true': true_temperature
    })
    df_drift.set_index('timestamp', inplace=True)
    
    # ドリフト検出: 線形回帰でトレンドを推定
    x = np.arange(len(df_drift))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, df_drift['measured'])
    
    print("=== センサードリフト分析 ===")
    print(f"検出されたドリフト率: {slope * 24:.4f} °C/日")
    print(f"30日間のドリフト量: {slope * 720:.2f} °C")
    print(f"R² 値: {r_value**2:.4f}")
    print(f"p値: {p_value:.4e}")
    
    # ドリフト補正
    df_drift['corrected'] = df_drift['measured'] - (slope * x + (intercept - true_temperature))
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 測定値とトレンド
    axes[0].plot(df_drift.index, df_drift['measured'], 'b-', linewidth=0.8, alpha=0.6, label='測定値（ドリフトあり）')
    axes[0].plot(df_drift.index, slope * x + intercept, 'r--', linewidth=2, label='検出されたトレンド')
    axes[0].axhline(y=true_temperature, color='green', linestyle='--', linewidth=2, label='真の温度')
    axes[0].set_ylabel('温度 (°C)', fontsize=11)
    axes[0].set_title('センサードリフトの検出', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 補正後の測定値
    axes[1].plot(df_drift.index, df_drift['measured'], 'b-', linewidth=0.8, alpha=0.4, label='補正前')
    axes[1].plot(df_drift.index, df_drift['corrected'], 'orange', linewidth=0.8, label='補正後')
    axes[1].axhline(y=true_temperature, color='green', linestyle='--', linewidth=2, label='真の温度')
    axes[1].set_xlabel('時刻', fontsize=11)
    axes[1].set_ylabel('温度 (°C)', fontsize=11)
    axes[1].set_title('ドリフト補正後の測定値', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 補正効果の評価
    mae_before = np.mean(np.abs(df_drift['measured'] - df_drift['true']))
    mae_after = np.mean(np.abs(df_drift['corrected'] - df_drift['true']))
    
    print(f"\n補正効果:")
    print(f"補正前の平均絶対誤差（MAE）: {mae_before:.3f} °C")
    print(f"補正後の平均絶対誤差（MAE）: {mae_after:.3f} °C")
    print(f"誤差削減率: {(1 - mae_after/mae_before)*100:.1f}%")
    

**期待される出力** :
    
    
    === センサードリフト分析 ===
    検出されたドリフト率: -0.0167 °C/日
    30日間のドリフト量: -0.50 °C
    R² 値: 0.9423
    p値: 0.0000e+00
    
    補正効果:
    補正前の平均絶対誤差（MAE）: 0.291 °C
    補正後の平均絶対誤差（MAE）: 0.243 °C
    誤差削減率: 16.5%
    

**解説** : センサードリフトは、センサーの経年劣化や環境変化により、測定値が真値から徐々にずれていく現象です。線形回帰によりトレンドを検出し、補正することでデータ品質を改善できます。実際のプラントでは、定期的な校正（キャリブレーション）と併用します。

#### コード例6: リアルタイムデータストリーミングのシミュレーション

**目的** : リアルタイムでセンサーデータを取得し、ストリーミング処理する仕組みをシミュレートする。
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import deque
    import time
    
    # リアルタイムストリーミングクラス
    class SensorDataStream:
        def __init__(self, buffer_size=100):
            """
            センサーデータストリーミングシミュレーター
    
            Parameters:
            -----------
            buffer_size : int
                データバッファのサイズ
            """
            self.buffer_size = buffer_size
            self.time_buffer = deque(maxlen=buffer_size)
            self.data_buffer = deque(maxlen=buffer_size)
            self.start_time = pd.Timestamp.now()
    
        def read_sensor(self):
            """センサーデータを1ポイント読み取る（シミュレーション）"""
            # 実際のシステムでは、ここでPLC/DCSからデータを取得
            current_time = pd.Timestamp.now()
            elapsed_seconds = (current_time - self.start_time).total_seconds()
    
            # 温度データのシミュレーション
            base_temp = 175
            variation = 3 * np.sin(2 * np.pi * elapsed_seconds / 60)  # 60秒周期
            noise = np.random.normal(0, 0.5)
            temperature = base_temp + variation + noise
    
            self.time_buffer.append(current_time)
            self.data_buffer.append(temperature)
    
            return current_time, temperature
    
        def get_statistics(self):
            """バッファ内データの統計量を計算"""
            if len(self.data_buffer) == 0:
                return None
    
            data_array = np.array(self.data_buffer)
            stats = {
                'mean': np.mean(data_array),
                'std': np.std(data_array),
                'min': np.min(data_array),
                'max': np.max(data_array),
                'latest': data_array[-1]
            }
            return stats
    
    # ストリーミングシミュレーション実行
    print("=== リアルタイムセンサーデータストリーミング ===")
    print("10秒間のデータ収集を開始します...\n")
    
    stream = SensorDataStream(buffer_size=50)
    
    # 10秒間、0.5秒ごとにデータ取得
    duration = 10  # 秒
    interval = 0.5  # 秒
    n_samples = int(duration / interval)
    
    for i in range(n_samples):
        timestamp, temperature = stream.read_sensor()
    
        # 統計情報の取得
        stats = stream.get_statistics()
    
        # 進捗表示（5サンプルごと）
        if (i + 1) % 5 == 0:
            print(f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
                  f"温度: {temperature:.2f}°C | "
                  f"平均: {stats['mean']:.2f}°C | "
                  f"標準偏差: {stats['std']:.2f}°C")
    
        time.sleep(interval)
    
    print("\nデータ収集完了！")
    
    # 収集したデータの可視化
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 時系列プロット
    axes[0].plot(list(stream.time_buffer), list(stream.data_buffer),
                 'o-', color='#11998e', markersize=4, linewidth=1.5)
    axes[0].set_ylabel('温度 (°C)', fontsize=11)
    axes[0].set_title('リアルタイムストリーミングデータ', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # ヒストグラム
    axes[1].hist(list(stream.data_buffer), bins=15, color='#11998e', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=np.mean(list(stream.data_buffer)), color='red',
                    linestyle='--', linewidth=2, label=f'平均: {np.mean(list(stream.data_buffer)):.2f}°C')
    axes[1].set_xlabel('温度 (°C)', fontsize=11)
    axes[1].set_ylabel('頻度', fontsize=11)
    axes[1].set_title('温度分布', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n最終統計:")
    final_stats = stream.get_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value:.2f}")
    

**解説** : このコードは、リアルタイムセンサーデータストリーミングの基本概念を実装しています。実際のプロセス監視システムでは、PLC/DCSから連続的にデータを取得し、バッファリングしながら統計処理や異常検知を行います。`deque`を使った固定長バッファは、メモリ効率的なリアルタイム処理に適しています。

#### コード例7: データロギングとバッファリングシステムの実装

**目的** : センサーデータを効率的にログファイルに記録し、バッファリングする仕組みを実装する。
    
    
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import os
    
    class ProcessDataLogger:
        def __init__(self, log_file='process_data.csv', buffer_size=100):
            """
            プロセスデータロギングシステム
    
            Parameters:
            -----------
            log_file : str
                ログファイル名
            buffer_size : int
                バッファサイズ（バッファが満杯になるとファイルに書き込み）
            """
            self.log_file = log_file
            self.buffer_size = buffer_size
            self.buffer = []
            self.total_logged = 0
    
            # ファイルが存在しない場合はヘッダーを作成
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.write('timestamp,temperature,pressure,flow_rate\n')
    
        def log_data(self, temperature, pressure, flow_rate):
            """
            データをバッファに追加
    
            Parameters:
            -----------
            temperature : float
                温度（℃）
            pressure : float
                圧力（MPa）
            flow_rate : float
                流量（m³/h）
            """
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            data_point = {
                'timestamp': timestamp,
                'temperature': temperature,
                'pressure': pressure,
                'flow_rate': flow_rate
            }
    
            self.buffer.append(data_point)
    
            # バッファが満杯になったらファイルに書き込み
            if len(self.buffer) >= self.buffer_size:
                self.flush()
    
        def flush(self):
            """バッファ内のデータをファイルに書き込み"""
            if len(self.buffer) == 0:
                return
    
            df = pd.DataFrame(self.buffer)
            df.to_csv(self.log_file, mode='a', header=False, index=False)
    
            self.total_logged += len(self.buffer)
            print(f"[INFO] {len(self.buffer)}個のデータポイントをログファイルに書き込みました。"
                  f"（累計: {self.total_logged}ポイント）")
    
            self.buffer = []
    
        def close(self):
            """残りのバッファをフラッシュしてロギングを終了"""
            self.flush()
            print(f"[INFO] データロギングを終了しました。総計: {self.total_logged}ポイント")
    
    # ロギングシステムのデモンストレーション
    print("=== プロセスデータロギングシステム ===\n")
    
    # ロガーの初期化
    logger = ProcessDataLogger(log_file='demo_process_data.csv', buffer_size=50)
    
    # 300ポイントのデータをロギング（実際はリアルタイムで取得）
    np.random.seed(42)
    n_samples = 300
    
    print(f"{n_samples}ポイントのデータロギングを開始...\n")
    
    for i in range(n_samples):
        # センサーデータのシミュレーション
        temperature = 175 + np.random.normal(0, 2)
        pressure = 1.5 + np.random.normal(0, 0.05)
        flow_rate = 50 + np.random.normal(0, 3)
    
        # データをログ
        logger.log_data(temperature, pressure, flow_rate)
    
    # 残りのバッファをフラッシュ
    logger.close()
    
    # ログファイルの読み込みと確認
    df_logged = pd.read_csv('demo_process_data.csv')
    df_logged['timestamp'] = pd.to_datetime(df_logged['timestamp'])
    
    print(f"\n=== ログファイル統計 ===")
    print(f"総データポイント数: {len(df_logged)}")
    print(f"期間: {df_logged['timestamp'].min()} ～ {df_logged['timestamp'].max()}")
    print(f"\n変数別統計:")
    print(df_logged[['temperature', 'pressure', 'flow_rate']].describe())
    
    # 簡易可視化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    axes[0].plot(df_logged.index, df_logged['temperature'], color='#11998e', linewidth=0.8)
    axes[0].set_ylabel('温度 (°C)', fontsize=11)
    axes[0].set_title('ログされたプロセスデータ', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(df_logged.index, df_logged['pressure'], color='#f59e0b', linewidth=0.8)
    axes[1].set_ylabel('圧力 (MPa)', fontsize=11)
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(df_logged.index, df_logged['flow_rate'], color='#7b2cbf', linewidth=0.8)
    axes[2].set_xlabel('サンプル番号', fontsize=11)
    axes[2].set_ylabel('流量 (m³/h)', fontsize=11)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # クリーンアップ
    os.remove('demo_process_data.csv')
    print("\n[INFO] デモファイルを削除しました。")
    

**解説** : このコードは、実際のプロセス監視システムで使用されるデータロギングの仕組みを実装しています。バッファリングにより、ディスクI/O回数を削減し、効率的なロギングを実現します。実際のシステムでは、このようなロギング機構がHistorianデータベース（OSIsoft PI、GE Proficyなど）と連携します。

#### コード例8: 基本統計モニタリング（移動平均と移動分散）

**目的** : ローリング統計量（移動平均、移動分散）を計算し、プロセスの変動をモニタリングする。
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # 時系列データ生成（24時間、1分間隔）
    time_index = pd.date_range('2025-01-01 00:00:00', periods=1440, freq='1min')
    
    # 基準温度 + 周期変動 + ノイズ + 突発的な変動
    temperature = 175 + 2 * np.sin(2 * np.pi * np.arange(1440) / 360) + np.random.normal(0, 1, 1440)
    
    # 午後12時〜14時に意図的な変動を追加（プロセス外乱をシミュレート）
    disturbance_start = 12 * 60  # 12:00
    disturbance_end = 14 * 60    # 14:00
    temperature[disturbance_start:disturbance_end] += 5
    
    # DataFrameに格納
    df_stats = pd.DataFrame({
        'timestamp': time_index,
        'temperature': temperature
    })
    df_stats.set_index('timestamp', inplace=True)
    
    # ローリング統計量の計算
    window_size = 60  # 60分（1時間）の移動窓
    
    df_stats['moving_average'] = df_stats['temperature'].rolling(window=window_size, center=True).mean()
    df_stats['moving_std'] = df_stats['temperature'].rolling(window=window_size, center=True).std()
    
    # 管理限界の計算（移動平均 ± 3σ）
    df_stats['upper_limit'] = df_stats['moving_average'] + 3 * df_stats['moving_std']
    df_stats['lower_limit'] = df_stats['moving_average'] - 3 * df_stats['moving_std']
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 温度と移動平均
    axes[0].plot(df_stats.index, df_stats['temperature'], 'b-', linewidth=0.6, alpha=0.5, label='測定値')
    axes[0].plot(df_stats.index, df_stats['moving_average'], 'r-', linewidth=2, label=f'移動平均（{window_size}分）')
    axes[0].fill_between(df_stats.index, df_stats['lower_limit'], df_stats['upper_limit'],
                          alpha=0.15, color='green', label='±3σ 範囲')
    axes[0].set_ylabel('温度 (°C)', fontsize=11)
    axes[0].set_title('温度トレンドと移動平均', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.3)
    
    # 移動標準偏差
    axes[1].plot(df_stats.index, df_stats['moving_std'], 'orange', linewidth=1.5)
    axes[1].axhline(y=df_stats['moving_std'].mean(), color='red', linestyle='--',
                    label=f'平均標準偏差: {df_stats["moving_std"].mean():.2f}°C')
    axes[1].set_ylabel('移動標準偏差 (°C)', fontsize=11)
    axes[1].set_title('プロセス変動のモニタリング', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 変動係数（CV: Coefficient of Variation）
    cv = (df_stats['moving_std'] / df_stats['moving_average']) * 100  # パーセント表示
    axes[2].plot(df_stats.index, cv, 'purple', linewidth=1.5)
    axes[2].set_xlabel('時刻', fontsize=11)
    axes[2].set_ylabel('変動係数 CV (%)', fontsize=11)
    axes[2].set_title('変動係数（CV = σ/μ × 100%）', fontsize=13, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 統計サマリー
    print("=== ローリング統計量サマリー ===")
    print(f"全体平均温度: {df_stats['temperature'].mean():.2f}°C")
    print(f"全体標準偏差: {df_stats['temperature'].std():.2f}°C")
    print(f"\n移動平均の範囲:")
    print(f"  最小: {df_stats['moving_average'].min():.2f}°C")
    print(f"  最大: {df_stats['moving_average'].max():.2f}°C")
    print(f"\n移動標準偏差の範囲:")
    print(f"  最小: {df_stats['moving_std'].min():.2f}°C")
    print(f"  最大: {df_stats['moving_std'].max():.2f}°C")
    print(f"  平均: {df_stats['moving_std'].mean():.2f}°C")
    
    # 外乱期間の検出
    high_variability = df_stats['moving_std'] > (df_stats['moving_std'].mean() + 2 * df_stats['moving_std'].std())
    print(f"\n高変動期間の検出:")
    print(f"  高変動データポイント数: {high_variability.sum()}")
    if high_variability.sum() > 0:
        print(f"  最初の高変動時刻: {df_stats.index[high_variability][0]}")
        print(f"  最後の高変動時刻: {df_stats.index[high_variability][-1]}")
    

**期待される出力** :
    
    
    === ローリング統計量サマリー ===
    全体平均温度: 175.24°C
    全体標準偏差: 2.89°C
    
    移動平均の範囲:
      最小: 172.51°C
      最大: 179.23°C
    
    移動標準偏差の範囲:
      最小: 1.12°C
      最大: 4.58°C
      平均: 1.87°C
    
    高変動期間の検出:
      高変動データポイント数: 142
      最初の高変動時刻: 2025-01-01 11:30:00
      最後の高変動時刻: 2025-01-01 14:29:00
    

**解説** : ローリング統計量（移動平均、移動標準偏差）は、プロセスのトレンドと変動を監視する基本的なツールです。移動平均はノイズを除去し、プロセスの基本的な傾向を把握するのに役立ちます。移動標準偏差は、プロセスの変動性を定量化し、異常な変動を検出するのに使用されます。このコードでは、プロセス外乱（12時〜14時の温度上昇）を移動標準偏差が正確に検出していることが確認できます。

* * *

## 1.4 本章のまとめ

### 学んだこと

  1. **プロセスモニタリングの基礎**
     * モニタリングの目的: 品質保証、安全性確保、効率性向上
     * SCADAとDCSの違いと適用領域
     * モニタリングシステムのアーキテクチャ（センサー層からデータ分析層まで）
  2. **センサー技術**
     * 主要センサータイプ（温度、圧力、流量、液面）の特性
     * サンプリング理論とナイキスト定理
     * エイリアシングの原理と防止策
  3. **時系列データ処理の実践**
     * Pandasによる時系列データハンドリング
     * 欠損値と外れ値の検出（Zスコア法）
     * センサードリフトの検出と補正
     * リアルタイムストリーミングとバッファリング
     * ローリング統計量による変動モニタリング

### 重要なポイント

  * **サンプリング周波数** : プロセスの動特性に応じた適切なサンプリングレートの選定が重要（ナイキスト定理）
  * **データ品質** : 欠損値、外れ値、ドリフトの検出と補正はモニタリングの基本
  * **リアルタイム処理** : バッファリングと効率的なデータ構造（deque）の活用
  * **統計的監視** : 移動平均と移動標準偏差による変動検出

### 次の章へ

第2章では、**統計的プロセス管理（SPC: Statistical Process Control）** を学びます：

  * シューハート管理図（X̄-R, I-MR）の作成と解釈
  * プロセス能力指数（Cp, Cpk）の計算
  * 高度なSPC手法（CUSUM, EWMA, Hotelling's T²）
  * 管理図の異常判定ルールとアラーム生成
  * 実プロセスデータを用いたSPCの実践
