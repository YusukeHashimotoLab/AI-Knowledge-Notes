---
title: 第1章 ウェハプロセス統計的管理
chapter_title: 第1章 ウェハプロセス統計的管理
subtitle: Wafer Process Statistical Control and R2R Management
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/semiconductor-manufacturing-ai/chapter-1.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Semiconductor Manufacturing Ai](<../../PI/semiconductor-manufacturing-ai/index.html>)›Chapter 1

[← シリーズ目次に戻る](<index.html>)

## 📖 本章の概要

半導体製造では、ナノメートルオーダーの精密制御が求められます。本章では、 Run-to-Run（R2R）制御、Virtual Metrology（VM）、プロセスドリフト検出など、 ウェハプロセスの統計的管理とAI技術を学びます。 

### 🎯 学習目標

  * 半導体プロセスの特性とウェハレベル統計管理
  * Run-to-Run（R2R）制御の原理と実装
  * Virtual Metrology（VM）による予測計測
  * EWMA制御とプロセスドリフト補正
  * 多変量統計的プロセス管理（MSPC）
  * ウェハマップ解析と空間パターン認識

## ⚙️ 1.1 Run-to-Run（R2R）制御の基礎

### R2R制御の原理

Run-to-Run制御は、前回のウェハ（ロット）の計測結果をフィードバックし、 次のウェハの製造条件を調整する適応制御手法です。 

#### EWMA（指数加重移動平均）制御

$$ u_k = u_{k-1} + K \cdot (T - y_k) $$ 

\\( u_k \\): 第k回の制御入力（プロセスパラメータ）  
\\( y_k \\): 第k回の測定値  
\\( T \\): 目標値  
\\( K \\): 制御ゲイン（0 < K < 1） 

**💡 半導体プロセスでのR2R適用例**  
・**エッチング** : エッチング時間調整によるCD（Critical Dimension）制御  
・**CVD成膜** : 成膜時間調整による膜厚制御  
・**CMP** : 研磨時間調整による平坦化制御  
・**リソグラフィ** : 露光量・焦点調整によるパターン精度制御 

### 💻 コード例1.1: EWMA R2R制御システム
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    class R2RController:
        """Run-to-Run EWMA制御システム"""
    
        def __init__(self, target, initial_input, control_gain=0.5):
            """
            Args:
                target: 目標値（例: 膜厚 100nm）
                initial_input: 初期制御入力（例: 成膜時間 60秒）
                control_gain: 制御ゲイン K（0 < K < 1）
            """
            self.target = target
            self.u = initial_input  # 現在の制御入力
            self.K = control_gain
            self.history = {
                'run': [],
                'input': [],
                'output': [],
                'error': []
            }
    
        def process_model(self, u, drift=0, noise_std=1.0):
            """
            プロセスモデル（簡略化）
    
            Args:
                u: 制御入力（成膜時間など）
                drift: プロセスドリフト
                noise_std: プロセスノイズ標準偏差
    
            Returns:
                測定値（膜厚など）
            """
            # 線形モデル: y = 1.5 * u + drift + noise
            gain = 1.5  # プロセスゲイン
            y = gain * u + drift + np.random.normal(0, noise_std)
            return y
    
        def update_control(self, measurement):
            """
            制御入力の更新（EWMA制御）
    
            Args:
                measurement: 測定値
    
            Returns:
                次回の制御入力
            """
            error = self.target - measurement
    
            # EWMA制御則
            u_next = self.u + self.K * error
    
            # 制御入力の制約（物理的制約）
            u_next = np.clip(u_next, 30, 90)  # 30-90秒の範囲
    
            self.u = u_next
            return u_next
    
        def run_simulation(self, n_runs=100, drift_start=50, drift_rate=0.1):
            """
            R2R制御シミュレーション
    
            Args:
                n_runs: 実行回数（ウェハ枚数）
                drift_start: ドリフト開始ラン
                drift_rate: ドリフト速度（/run）
            """
            for run in range(n_runs):
                # プロセスドリフトの計算
                if run >= drift_start:
                    drift = (run - drift_start) * drift_rate
                else:
                    drift = 0
    
                # プロセス実行と測定
                y = self.process_model(self.u, drift=drift, noise_std=2.0)
    
                # 履歴記録
                error = y - self.target
                self.history['run'].append(run + 1)
                self.history['input'].append(self.u)
                self.history['output'].append(y)
                self.history['error'].append(error)
    
                # 次回の制御入力更新
                self.update_control(y)
    
            return pd.DataFrame(self.history)
    
        def plot_results(self, df):
            """制御結果の可視化"""
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
            # 測定値のトレンド
            axes[0].plot(df['run'], df['output'], marker='o', color='#11998e',
                         linewidth=1.5, markersize=4, label='測定値')
            axes[0].axhline(y=self.target, color='red', linestyle='--',
                            linewidth=2, label=f'目標値 ({self.target}nm)')
            axes[0].fill_between(df['run'], self.target - 3, self.target + 3,
                                  alpha=0.2, color='green', label='許容範囲 (±3nm)')
            axes[0].set_xlabel('ラン番号（ウェハ）')
            axes[0].set_ylabel('膜厚（nm）')
            axes[0].set_title('R2R制御による膜厚管理', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
    
            # 制御入力のトレンド
            axes[1].plot(df['run'], df['input'], marker='s', color='#38ef7d',
                         linewidth=1.5, markersize=4, label='成膜時間')
            axes[1].set_xlabel('ラン番号（ウェハ）')
            axes[1].set_ylabel('成膜時間（秒）')
            axes[1].set_title('R2R制御入力', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
    
            # 制御誤差
            axes[2].plot(df['run'], df['error'], marker='^', color='#f38181',
                         linewidth=1.5, markersize=4, label='制御誤差')
            axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
            axes[2].fill_between(df['run'], -3, 3, alpha=0.2, color='green',
                                  label='許容誤差 (±3nm)')
            axes[2].set_xlabel('ラン番号（ウェハ）')
            axes[2].set_ylabel('誤差（nm）')
            axes[2].set_title('制御誤差', fontsize=12, fontweight='bold')
            axes[2].legend()
            axes[2].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('r2r_control_results.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 実行例
    print("=" * 60)
    print("Run-to-Run EWMA制御システム（CVD成膜プロセス）")
    print("=" * 60)
    
    # R2R制御システムの初期化
    r2r = R2RController(
        target=100.0,        # 目標膜厚 100nm
        initial_input=60.0,  # 初期成膜時間 60秒
        control_gain=0.5     # 制御ゲイン K=0.5
    )
    
    # シミュレーション実行
    df_results = r2r.run_simulation(
        n_runs=100,
        drift_start=50,   # 50ラン目からドリフト開始
        drift_rate=0.1    # 0.1nm/runのドリフト
    )
    
    # 性能評価
    print(f"\n制御性能評価:")
    print(f"平均膜厚: {df_results['output'].mean():.2f} nm")
    print(f"膜厚標準偏差: {df_results['output'].std():.2f} nm")
    print(f"平均絶対誤差: {df_results['error'].abs().mean():.2f} nm")
    print(f"最大誤差: {df_results['error'].abs().max():.2f} nm")
    
    # 規格内率の計算
    in_spec = df_results[(df_results['output'] >= 97) & (df_results['output'] <= 103)]
    print(f"規格内率 (100±3nm): {len(in_spec)/len(df_results)*100:.1f}%")
    
    # ドリフト補正前後の比較
    before_drift = df_results[df_results['run'] < 50]
    after_drift = df_results[df_results['run'] >= 50]
    print(f"\nドリフト発生前（1-49ラン）平均: {before_drift['output'].mean():.2f} nm")
    print(f"ドリフト発生後（50-100ラン）平均: {after_drift['output'].mean():.2f} nm")
    
    # 可視化
    r2r.plot_results(df_results)
    

**実装のポイント:**

  * EWMA制御によるプロセスドリフトの自動補正
  * 制御ゲインKの調整による応答性と安定性のバランス
  * 物理的制約（制御入力の上下限）の考慮
  * 定量的な制御性能評価（平均誤差、標準偏差、規格内率）

## 🔮 1.2 Virtual Metrology（仮想計測）

### Virtual Metrologyの原理

Virtual Metrology（VM）は、プロセス装置のセンサデータ（温度、圧力、流量など）から、 計測器を使わずにウェハの品質特性（膜厚、CD、電気特性など）を予測する技術です。 

**💡 VMの利点**  
・全数計測によるリアルタイム品質管理  
・計測コストと時間の削減（計測器不要）  
・インライン制御の実現  
・計測不可能な中間工程での品質予測 

### 💻 コード例1.2: Random ForestによるVirtual Metrology
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    class VirtualMetrologySystem:
        """Virtual Metrology（仮想計測）システム"""
    
        def __init__(self):
            self.model = None
            self.feature_names = None
    
        def generate_process_data(self, n_samples=500):
            """
            プロセスセンサデータの生成（エッチングプロセスを想定）
    
            Returns:
                (センサデータ, CD測定値)
            """
            np.random.seed(42)
    
            # プロセス条件（装置センサデータ）
            rf_power = np.random.normal(1000, 50, n_samples)      # RFパワー（W）
            pressure = np.random.normal(50, 5, n_samples)          # 圧力（mTorr）
            gas_flow_ar = np.random.normal(200, 10, n_samples)    # Arガス流量（sccm）
            gas_flow_cf4 = np.random.normal(50, 5, n_samples)     # CF4ガス流量（sccm）
            temp_chamber = np.random.normal(60, 3, n_samples)     # チャンバー温度（℃）
            etch_time = np.random.normal(120, 5, n_samples)       # エッチング時間（秒）
    
            # CD（Critical Dimension）の物理モデル
            # CD = f(RF power, pressure, gas flows, temp, time) + noise
            cd_base = 90  # ベースCD（nm）
    
            cd = (cd_base
                  - 0.002 * (rf_power - 1000)      # RFパワー↑ → CD↓
                  + 0.05 * (pressure - 50)          # 圧力↑ → CD↑
                  - 0.01 * (gas_flow_cf4 - 50)     # CF4↑ → CD↓（エッチング促進）
                  + 0.005 * (temp_chamber - 60)    # 温度↑ → CD↑
                  - 0.03 * (etch_time - 120)       # 時間↑ → CD↓
                  + np.random.normal(0, 1, n_samples))  # ノイズ
    
            # データフレーム作成
            df = pd.DataFrame({
                'rf_power': rf_power,
                'pressure': pressure,
                'gas_flow_ar': gas_flow_ar,
                'gas_flow_cf4': gas_flow_cf4,
                'temp_chamber': temp_chamber,
                'etch_time': etch_time,
                'cd_measured': cd
            })
    
            return df
    
        def train_vm_model(self, df, feature_columns, target_column='cd_measured'):
            """
            VMモデルの訓練
    
            Args:
                df: プロセスデータ
                feature_columns: 特徴量カラム
                target_column: 予測対象カラム
            """
            X = df[feature_columns].values
            y = df[target_column].values
    
            # 訓練/テストデータ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
    
            # Random Forestモデル
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            self.feature_names = feature_columns
    
            # 予測と評価
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
    
            # 性能指標
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)
    
            results = {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
                'r2_train': r2_train, 'r2_test': r2_test,
                'rmse_test': rmse_test, 'mae_test': mae_test
            }
    
            return results
    
        def predict_cd(self, process_conditions):
            """
            プロセス条件からCDを予測
    
            Args:
                process_conditions: プロセス条件の配列またはDataFrame
    
            Returns:
                予測CD値
            """
            if self.model is None:
                raise ValueError("モデルが訓練されていません")
    
            return self.model.predict(process_conditions)
    
        def plot_vm_results(self, results):
            """VM結果の可視化"""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            # 予測精度プロット（訓練データ）
            axes[0, 0].scatter(results['y_train'], results['y_pred_train'],
                               alpha=0.5, s=20, color='#11998e', label='訓練データ')
            axes[0, 0].plot([85, 95], [85, 95], 'r--', linewidth=2, label='理想直線')
            axes[0, 0].set_xlabel('実測CD（nm）')
            axes[0, 0].set_ylabel('予測CD（nm）')
            axes[0, 0].set_title(f'訓練データ予測精度 (R²={results["r2_train"]:.4f})',
                                 fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
    
            # 予測精度プロット（テストデータ）
            axes[0, 1].scatter(results['y_test'], results['y_pred_test'],
                               alpha=0.5, s=20, color='#38ef7d', label='テストデータ')
            axes[0, 1].plot([85, 95], [85, 95], 'r--', linewidth=2, label='理想直線')
            axes[0, 1].set_xlabel('実測CD（nm）')
            axes[0, 1].set_ylabel('予測CD（nm）')
            axes[0, 1].set_title(f'テストデータ予測精度 (R²={results["r2_test"]:.4f})',
                                 fontsize=12, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
    
            # 予測誤差の分布
            errors = results['y_pred_test'] - results['y_test']
            axes[1, 0].hist(errors, bins=30, color='#4ecdc4', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='誤差ゼロ')
            axes[1, 0].set_xlabel('予測誤差（nm）')
            axes[1, 0].set_ylabel('頻度')
            axes[1, 0].set_title(f'予測誤差分布 (MAE={results["mae_test"]:.2f}nm)',
                                 fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
    
            # 特徴量重要度
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
    
            axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'],
                            color='#f38181', alpha=0.8)
            axes[1, 1].set_xlabel('重要度')
            axes[1, 1].set_title('特徴量重要度', fontsize=12, fontweight='bold')
            axes[1, 1].grid(axis='x', alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('virtual_metrology_results.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 実行例
    print("=" * 60)
    print("Virtual Metrologyシステム（エッチングプロセス）")
    print("=" * 60)
    
    # VMシステムの初期化
    vm = VirtualMetrologySystem()
    
    # プロセスデータの生成
    df_process = vm.generate_process_data(n_samples=500)
    
    print(f"\n生成データ数: {len(df_process)}")
    print(f"CD範囲: {df_process['cd_measured'].min():.2f} - {df_process['cd_measured'].max():.2f} nm")
    
    # 特徴量の定義
    feature_cols = ['rf_power', 'pressure', 'gas_flow_ar', 'gas_flow_cf4',
                    'temp_chamber', 'etch_time']
    
    # VMモデルの訓練
    results = vm.train_vm_model(df_process, feature_cols)
    
    print(f"\nVMモデル性能:")
    print(f"訓練データ R² = {results['r2_train']:.4f}")
    print(f"テストデータ R² = {results['r2_test']:.4f}")
    print(f"RMSE = {results['rmse_test']:.2f} nm")
    print(f"MAE = {results['mae_test']:.2f} nm")
    
    # 新規ウェハの予測例
    print(f"\n新規ウェハのCD予測:")
    new_wafer = np.array([[1020, 52, 205, 48, 61, 118]])  # 新しいプロセス条件
    predicted_cd = vm.predict_cd(new_wafer)
    print(f"予測CD: {predicted_cd[0]:.2f} nm")
    
    # 可視化
    vm.plot_vm_results(results)
    

**実装のポイント:**

  * プロセス装置センサデータから品質特性を高精度予測（R² > 0.95）
  * Random Forestによる非線形関係のモデリング
  * 特徴量重要度分析による支配的プロセスパラメータの特定
  * リアルタイム予測によるインライン品質管理の実現

## 📚 まとめ

本章では、ウェハプロセスの統計的管理について学びました。

### 主要なポイント

  * Run-to-Run EWMA制御によるプロセスドリフトの自動補正
  * Virtual Metrologyによる全数品質予測と計測コスト削減
  * 制御ゲインと応答性のトレードオフ
  * 機械学習モデルによる高精度なプロセス-品質関係のモデリング

**🎯 次章予告**  
第2章では、AIによる欠陥検査とAOI（Automated Optical Inspection）について学びます。 深層学習（CNN）による欠陥分類、セマンティックセグメンテーション、 異物検出など、画像ベースの品質管理技術を習得します。 

[← シリーズ目次](<index.html>) [第2章: AI欠陥検査とAOI →](<chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
