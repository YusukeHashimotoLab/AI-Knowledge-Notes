---
title: 第4章 連続生産とQbD実装
chapter_title: 第4章 連続生産とQbD実装
subtitle: Continuous Manufacturing and Quality by Design Implementation
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/pharma-manufacturing-ai/chapter-4.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Pharma Manufacturing Ai](<../../PI/pharma-manufacturing-ai/index.html>)›Chapter 4

[← シリーズ目次に戻る](<index.html>)

## 📖 本章の概要

医薬品製造は従来のバッチ生産から連続生産への移行が進んでいます。 本章では、QbD（Quality by Design：品質の設計）の概念に基づき、 実験計画法（DoE）、デザインスペース、リスクベースアプローチを活用した 連続生産システムの最適化と実装方法を学びます。 

### 🎯 学習目標

  * QbD（Quality by Design）の基本概念とICH Q8-Q11ガイドライン
  * 実験計画法（DoE）による効率的なプロセス最適化
  * デザインスペースの構築と可視化
  * 連続生産システムの設計と制御戦略
  * リアルタイム制御とフィードバック制御の実装
  * バッチ生産から連続生産への移行戦略
  * リスクベースアプローチ（ICH Q9）の適用

## 🎯 4.1 QbD（Quality by Design）の基礎

### QbDの3つの柱

ICH Q8ガイドラインで定義されるQbDの核心概念：

  1. **QTPP（Quality Target Product Profile）** : 目標製品品質プロファイル
  2. **CQA（Critical Quality Attributes）** : 重要品質特性
  3. **CPP（Critical Process Parameters）** : 重要工程パラメータ

**🏭 QbDの利点**  
・科学的理解に基づいたプロセス設計  
・品質リスクの事前評価と低減  
・規制当局との継続的な対話  
・デザインスペース内での柔軟な製造  
・ライフサイクル全体での継続的改善 

### デザインスペースの定義

**デザインスペース（ICH Q8定義）**

「インプット変数（原材料属性など）とプロセスパラメータの多次元的組み合わせおよび相互作用であり、 品質を保証することが実証されている範囲」 

$$ \text{Design Space} = \\{(\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n) \mid \text{CQA}_i \in [\text{LSL}_i, \text{USL}_i] \, \forall i\\} $$ 

### 💻 コード例4.1: 実験計画法（DoE）とデザインスペース構築
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    class QbDDesigner:
        """QbDデザインスペース設計クラス"""
    
        def __init__(self):
            self.model = None
            self.poly_features = None
    
        def generate_doe_design(self, design_type='central_composite'):
            """
            実験計画の生成
    
            Args:
                design_type: 計画タイプ ('full_factorial', 'central_composite', 'box_behnken')
    
            Returns:
                実験計画のDataFrame
            """
            if design_type == 'central_composite':
                # 中心複合計画（3因子）
                # 因子: 反応温度（70-90℃）、反応時間（60-180分）、触媒量（1-5%）
    
                # 立方点（2^3 = 8点）
                cube_points = np.array([
                    [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
                ])
    
                # 軸点（2×3 = 6点、α=1.682）
                alpha = 1.682
                axial_points = np.array([
                    [-alpha, 0, 0], [alpha, 0, 0],
                    [0, -alpha, 0], [0, alpha, 0],
                    [0, 0, -alpha], [0, 0, alpha]
                ])
    
                # 中心点（6反復）
                center_points = np.array([[0, 0, 0]] * 6)
    
                # 統合
                design_matrix = np.vstack([cube_points, axial_points, center_points])
    
                # 実スケールへの変換
                temp_range = (70, 90)
                time_range = (60, 180)
                catalyst_range = (1, 5)
    
                temperature = (design_matrix[:, 0] + 1) / 2 * (temp_range[1] - temp_range[0]) + temp_range[0]
                time = (design_matrix[:, 1] + 1) / 2 * (time_range[1] - time_range[0]) + time_range[0]
                catalyst = (design_matrix[:, 2] + 1) / 2 * (catalyst_range[1] - catalyst_range[0]) + catalyst_range[0]
    
                df_design = pd.DataFrame({
                    'run': range(1, len(design_matrix) + 1),
                    'temperature': temperature,
                    'time': time,
                    'catalyst': catalyst
                })
    
            return df_design
    
        def simulate_response(self, df_design):
            """
            応答変数のシミュレーション（収率とpurity）
    
            Args:
                df_design: 実験計画のDataFrame
    
            Returns:
                応答変数を追加したDataFrame
            """
            np.random.seed(42)
    
            # 収率モデル（2次式 + ノイズ）
            # 最適条件: 温度80℃、時間120分、触媒3%で最大収率
            temp_centered = (df_design['temperature'] - 80) / 10
            time_centered = (df_design['time'] - 120) / 60
            catalyst_centered = (df_design['catalyst'] - 3) / 2
    
            yield_base = (
                95  # ベース収率
                - 3 * temp_centered ** 2  # 温度の2次効果
                - 2 * time_centered ** 2  # 時間の2次効果
                - 4 * catalyst_centered ** 2  # 触媒の2次効果
                + 1.5 * temp_centered * time_centered  # 交互作用
                + np.random.normal(0, 1, len(df_design))  # ノイズ
            )
    
            # Purity（純度）モデル
            purity_base = (
                99.5  # ベース純度
                - 0.5 * temp_centered ** 2
                - 0.3 * time_centered ** 2
                - 0.4 * catalyst_centered ** 2
                + np.random.normal(0, 0.1, len(df_design))
            )
    
            df_design['yield'] = np.clip(yield_base, 0, 100)
            df_design['purity'] = np.clip(purity_base, 95, 100)
    
            return df_design
    
        def build_response_surface_model(self, df_design, response_var):
            """
            応答曲面モデルの構築（2次多項式）
    
            Args:
                df_design: 実験データ
                response_var: 応答変数名
    
            Returns:
                訓練されたモデル
            """
            X = df_design[['temperature', 'time', 'catalyst']].values
            y = df_design[response_var].values
    
            # 2次多項式特徴量の生成
            self.poly_features = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = self.poly_features.fit_transform(X)
    
            # 線形回帰モデル
            self.model = LinearRegression()
            self.model.fit(X_poly, y)
    
            # 予測と評価
            y_pred = self.model.predict(X_poly)
            r2 = r2_score(y, y_pred)
    
            return self.model, r2
    
        def predict_response(self, temperature, time, catalyst):
            """応答予測"""
            X_new = np.array([[temperature, time, catalyst]])
            X_poly = self.poly_features.transform(X_new)
            return self.model.predict(X_poly)[0]
    
        def plot_design_space(self, df_design, temp_range, time_range, catalyst_fixed=3):
            """デザインスペースの可視化"""
            fig = plt.figure(figsize=(16, 6))
    
            # 温度-時間平面でのデザインスペース（触媒量固定）
            temp_grid = np.linspace(temp_range[0], temp_range[1], 50)
            time_grid = np.linspace(time_range[0], time_range[1], 50)
            T_mesh, Ti_mesh = np.meshgrid(temp_grid, time_grid)
    
            # 収率予測
            yield_grid = np.zeros_like(T_mesh)
            purity_grid = np.zeros_like(T_mesh)
    
            for i in range(len(temp_grid)):
                for j in range(len(time_grid)):
                    yield_grid[j, i] = self.predict_response(T_mesh[j, i], Ti_mesh[j, i], catalyst_fixed)
                    purity_grid[j, i] = self.predict_response(T_mesh[j, i], Ti_mesh[j, i], catalyst_fixed)
    
            # 収率の等高線図
            ax1 = fig.add_subplot(131)
            contour_yield = ax1.contourf(T_mesh, Ti_mesh, yield_grid, levels=20, cmap='RdYlGn')
            ax1.contour(T_mesh, Ti_mesh, yield_grid, levels=[90, 92, 94], colors='black',
                        linewidths=1.5, linestyles='dashed')
            ax1.scatter(df_design['temperature'], df_design['time'], c='blue', s=50,
                        edgecolor='black', linewidth=1, label='実験点', zorder=5)
            ax1.set_xlabel('反応温度（℃）')
            ax1.set_ylabel('反応時間（分）')
            ax1.set_title(f'収率（触媒量={catalyst_fixed}%）', fontsize=12, fontweight='bold')
            ax1.legend()
            plt.colorbar(contour_yield, ax=ax1, label='収率（%）')
    
            # 純度の等高線図
            ax2 = fig.add_subplot(132)
            contour_purity = ax2.contourf(T_mesh, Ti_mesh, purity_grid, levels=20, cmap='Blues')
            ax2.contour(T_mesh, Ti_mesh, purity_grid, levels=[99.0, 99.2, 99.4], colors='black',
                        linewidths=1.5, linestyles='dashed')
            ax2.scatter(df_design['temperature'], df_design['time'], c='blue', s=50,
                        edgecolor='black', linewidth=1, label='実験点', zorder=5)
            ax2.set_xlabel('反応温度（℃）')
            ax2.set_ylabel('反応時間（分）')
            ax2.set_title(f'純度（触媒量={catalyst_fixed}%）', fontsize=12, fontweight='bold')
            ax2.legend()
            plt.colorbar(contour_purity, ax=ax2, label='純度（%）')
    
            # デザインスペース（収率≥90% AND 純度≥99.0%）
            ax3 = fig.add_subplot(133)
            design_space_mask = (yield_grid >= 90) & (purity_grid >= 99.0)
    
            ax3.contourf(T_mesh, Ti_mesh, design_space_mask.astype(int), levels=[0, 0.5, 1],
                         colors=['red', 'green'], alpha=0.3)
            ax3.contour(T_mesh, Ti_mesh, yield_grid, levels=[90], colors='blue',
                        linewidths=2, linestyles='solid', label='収率=90%')
            ax3.contour(T_mesh, Ti_mesh, purity_grid, levels=[99.0], colors='orange',
                        linewidths=2, linestyles='solid', label='純度=99.0%')
            ax3.scatter(df_design['temperature'], df_design['time'], c='blue', s=50,
                        edgecolor='black', linewidth=1, label='実験点', zorder=5)
            ax3.set_xlabel('反応温度（℃）')
            ax3.set_ylabel('反応時間（分）')
            ax3.set_title('デザインスペース（緑=許容領域）', fontsize=12, fontweight='bold')
            ax3.legend()
    
            plt.tight_layout()
            plt.savefig('qbd_design_space.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 実行例
    print("=" * 60)
    print("QbDデザインスペース設計システム")
    print("=" * 60)
    
    # QbD設計インスタンス
    qbd = QbDDesigner()
    
    # 中心複合計画の生成
    df_design = qbd.generate_doe_design(design_type='central_composite')
    
    print(f"\n実験計画法（中心複合計画）:")
    print(f"実験数: {len(df_design)}")
    print(f"\n因子レベル:")
    print(f"  反応温度: 70-90℃")
    print(f"  反応時間: 60-180分")
    print(f"  触媒量: 1-5%")
    
    # 応答変数のシミュレーション
    df_design = qbd.simulate_response(df_design)
    
    print(f"\n応答変数:")
    print(f"  収率: {df_design['yield'].min():.1f}-{df_design['yield'].max():.1f}%")
    print(f"  純度: {df_design['purity'].min():.2f}-{df_design['purity'].max():.2f}%")
    
    # 応答曲面モデルの構築
    model_yield, r2_yield = qbd.build_response_surface_model(df_design, 'yield')
    print(f"\n収率モデル R² = {r2_yield:.4f}")
    
    model_purity, r2_purity = qbd.build_response_surface_model(df_design, 'purity')
    print(f"純度モデル R² = {r2_purity:.4f}")
    
    # 最適条件の探索
    print(f"\n最適条件予測:")
    optimal_yield = qbd.predict_response(80, 120, 3)
    print(f"  温度=80℃, 時間=120分, 触媒=3%")
    print(f"  → 収率 = {optimal_yield:.2f}%")
    
    # デザインスペース可視化
    qbd.plot_design_space(df_design, temp_range=(70, 90), time_range=(60, 180), catalyst_fixed=3)
    

**実装のポイント:**

  * 中心複合計画（CCD）による効率的な実験設計
  * 2次多項式モデルによる応答曲面の構築
  * 多目的最適化（収率と純度の両立）
  * デザインスペースの可視化と許容領域の特定
  * 科学的根拠に基づくプロセスパラメータ設定

## ⚙️ 4.2 連続生産システムの制御戦略

### バッチ vs 連続生産

項目 | バッチ生産 | 連続生産  
---|---|---  
生産形態 | 間欠的 | 連続的  
リードタイム | 数日～数週間 | 数時間  
設備規模 | 大型 | 小型  
柔軟性 | 高い | 製品切替に時間  
品質管理 | バッチ単位 | リアルタイム  
在庫 | 中間在庫大 | 最小化  
  
### 💻 コード例4.2: 連続生産のリアルタイム制御シミュレーション
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    class ContinuousManufacturingSimulator:
        """連続生産シミュレータ"""
    
        def __init__(self, target_concentration=5.0):
            """
            Args:
                target_concentration: 目標濃度（mol/L）
            """
            self.target = target_concentration
            self.Kp = 2.0  # 比例ゲイン
            self.Ki = 0.5  # 積分ゲイン
            self.Kd = 0.1  # 微分ゲイン
            self.integral_error = 0
            self.previous_error = 0
    
        def reactor_dynamics(self, state, t, feed_rate, T):
            """
            連続反応器の動特性（CSTR: Continuous Stirred-Tank Reactor）
    
            Args:
                state: [濃度 C, 温度 T_reactor]
                t: 時間
                feed_rate: 供給流量
                T: 設定温度
    
            Returns:
                状態変化率 [dC/dt, dT/dt]
            """
            C, T_reactor = state
    
            # パラメータ
            V = 10.0  # 反応器容積（L）
            k0 = 1e10  # 頻度因子
            Ea = 8000  # 活性化エネルギー（J/mol）
            R = 8.314  # 気体定数
            C_in = 10.0  # 供給濃度（mol/L）
    
            # 反応速度定数（Arrhenius式）
            k = k0 * np.exp(-Ea / (R * (T_reactor + 273.15)))
    
            # 物質収支
            dC_dt = feed_rate / V * (C_in - C) - k * C
    
            # エネルギー収支（簡略化）
            dT_dt = 0.1 * (T - T_reactor)  # 温度制御
    
            return [dC_dt, dT_dt]
    
        def pid_controller(self, error, dt=0.1):
            """
            PID制御器
    
            Args:
                error: 偏差（目標値 - 現在値）
                dt: サンプリング時間
    
            Returns:
                制御出力（供給流量調整）
            """
            # 積分項
            self.integral_error += error * dt
    
            # 微分項
            derivative_error = (error - self.previous_error) / dt
    
            # PID出力
            output = (self.Kp * error +
                      self.Ki * self.integral_error +
                      self.Kd * derivative_error)
    
            self.previous_error = error
    
            return output
    
        def simulate_continuous_process(self, duration=100, disturbance_times=None):
            """
            連続生産プロセスのシミュレーション
    
            Args:
                duration: シミュレーション時間（分）
                disturbance_times: 外乱発生時刻のリスト
    
            Returns:
                時系列データのDataFrame
            """
            dt = 0.1  # サンプリング時間（分）
            t = np.arange(0, duration, dt)
    
            # 初期状態
            C = 5.0  # 初期濃度
            T_reactor = 80.0  # 初期反応器温度
            state = [C, T_reactor]
    
            # 記録用リスト
            concentrations = []
            temperatures = []
            feed_rates = []
            errors = []
            setpoints = []
    
            # 基準供給流量
            base_feed_rate = 1.0  # L/min
    
            for i, time in enumerate(t):
                # 目標値（セットポイント変更のシミュレーション）
                if time < 30:
                    target = 5.0
                elif time < 60:
                    target = 6.0
                else:
                    target = 5.5
    
                # 外乱の追加
                if disturbance_times and any(abs(time - dt_time) < 0.5 for dt_time in disturbance_times):
                    state[0] += np.random.uniform(-0.5, 0.5)  # 濃度外乱
    
                # PID制御
                error = target - state[0]
                control_output = self.pid_controller(error, dt)
    
                # 供給流量の更新（制約付き）
                feed_rate = np.clip(base_feed_rate + control_output, 0.5, 2.0)
    
                # 反応器ダイナミクスの更新
                state = odeint(self.reactor_dynamics, state, [time, time + dt],
                               args=(feed_rate, 80.0))[-1]
    
                # 記録
                concentrations.append(state[0])
                temperatures.append(state[1])
                feed_rates.append(feed_rate)
                errors.append(error)
                setpoints.append(target)
    
            df_results = pd.DataFrame({
                'time': t,
                'concentration': concentrations,
                'temperature': temperatures,
                'feed_rate': feed_rates,
                'error': errors,
                'setpoint': setpoints
            })
    
            return df_results
    
        def plot_control_performance(self, df_results):
            """制御性能の可視化"""
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
            # 濃度プロファイル
            axes[0].plot(df_results['time'], df_results['concentration'], color='#11998e',
                         linewidth=2, label='実測濃度')
            axes[0].plot(df_results['time'], df_results['setpoint'], color='red',
                         linestyle='--', linewidth=2, label='目標濃度')
            axes[0].fill_between(df_results['time'],
                                  df_results['setpoint'] - 0.1,
                                  df_results['setpoint'] + 0.1,
                                  alpha=0.2, color='green', label='許容範囲 (±0.1)')
            axes[0].set_xlabel('時間（分）')
            axes[0].set_ylabel('濃度（mol/L）')
            axes[0].set_title('連続反応器の濃度制御', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
    
            # 供給流量
            axes[1].plot(df_results['time'], df_results['feed_rate'], color='#38ef7d',
                         linewidth=2)
            axes[1].set_xlabel('時間（分）')
            axes[1].set_ylabel('供給流量（L/min）')
            axes[1].set_title('PID制御器出力（供給流量）', fontsize=12, fontweight='bold')
            axes[1].grid(alpha=0.3)
    
            # 制御誤差
            axes[2].plot(df_results['time'], df_results['error'], color='#f38181',
                         linewidth=1.5)
            axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
            axes[2].fill_between(df_results['time'], -0.1, 0.1,
                                  alpha=0.2, color='green', label='許容誤差')
            axes[2].set_xlabel('時間（分）')
            axes[2].set_ylabel('制御誤差（mol/L）')
            axes[2].set_title('制御誤差', fontsize=12, fontweight='bold')
            axes[2].legend()
            axes[2].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('continuous_manufacturing_control.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 実行例
    print("=" * 60)
    print("連続生産リアルタイム制御シミュレーション")
    print("=" * 60)
    
    # シミュレータの初期化
    cm_sim = ContinuousManufacturingSimulator(target_concentration=5.0)
    
    # 外乱発生時刻の設定
    disturbance_times = [40, 70]
    
    # シミュレーション実行
    df_results = cm_sim.simulate_continuous_process(duration=100, disturbance_times=disturbance_times)
    
    # 性能評価
    print(f"\nシミュレーション結果:")
    print(f"シミュレーション時間: {df_results['time'].max():.1f} 分")
    print(f"濃度範囲: {df_results['concentration'].min():.2f}-{df_results['concentration'].max():.2f} mol/L")
    print(f"平均制御誤差: {df_results['error'].abs().mean():.4f} mol/L")
    print(f"最大制御誤差: {df_results['error'].abs().max():.4f} mol/L")
    
    # 定常偏差の評価（最後の10分間）
    steady_state_data = df_results[df_results['time'] > 90]
    steady_state_error = steady_state_data['error'].abs().mean()
    print(f"定常偏差: {steady_state_error:.4f} mol/L")
    
    # 可視化
    cm_sim.plot_control_performance(df_results)
    

**実装のポイント:**

  * 連続反応器（CSTR）の動的モデリング
  * PID制御によるリアルタイムフィードバック制御
  * セットポイント変更と外乱への応答性評価
  * 制御性能の定量評価（定常偏差、応答時間）
  * バッチ生産にない連続生産特有の制御課題への対応

## 📚 まとめ

本章では、連続生産とQbD実装について学びました。

### 主要なポイント

  * QbDの3つの柱（QTPP、CQA、CPP）と科学的アプローチ
  * 実験計画法（DoE）による効率的なプロセス最適化
  * デザインスペースの構築と可視化
  * 連続生産システムのリアルタイム制御戦略
  * PID制御による安定した品質保証

**🎯 次章予告**  
第5章では、AIシステムの規制対応とCSV（Computer System Validation）実装戦略について学びます。 21 CFR Part 11準拠、監査証跡、電子署名、データインテグリティの実装など、 GMP環境でのAIシステム導入に必須の知識を習得します。 

[← 第3章: PATとリアルタイム品質管理](<chapter-3.html>) [第5章: 規制対応とCSV実装 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
