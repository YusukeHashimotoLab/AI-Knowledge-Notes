---
title: 第5章 ケーススタディ
chapter_title: 第5章 ケーススタディ
subtitle: Case Studies in Food Process AI
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/food-process-ai/chapter-5.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Food Process Ai](<../../PI/food-process-ai/index.html>)›Chapter 5

[← シリーズ目次に戻る](<index.html>)

## 📖 本章の概要

本章では、第1章から第4章で学んだAI技術を実際の食品製造プロセスに適用した 具体的なケーススタディを紹介します。乳製品、飲料、スナック食品、調味料など、 様々な食品カテゴリにおけるAI導入事例を通じて、技術の実践的な応用方法と その効果を学びます。各事例では、課題の特定、AI技術の選定、実装、効果測定までの 一連のプロセスを詳しく解説します。 

### 🎯 学習目標

  * 実際の食品製造現場におけるAI導入のプロセス理解
  * 業種・製品特性に応じた技術選定の考え方
  * ROI（投資対効果）の評価手法
  * 導入時の課題と解決策
  * 組織変革とデータ文化の醸成

## 🥛 5.1 ケーススタディ1: ヨーグルト製造の品質管理AI

### 📋 企業プロファイル

  * **業種** : 乳製品メーカー（従業員300名）
  * **製品** : 発酵乳製品（ヨーグルト、飲むヨーグルト）
  * **生産量** : 日産50トン

### 🚨 課題

  * 発酵プロセスのばらつきによる品質不安定性（酸度、粘度、風味）
  * 季節変動による乳原料の成分変化への対応困難
  * 経験豊富なオペレータの退職による技能伝承の問題
  * ロット不良による廃棄率3.5%（年間約600万円の損失）

### 💡 導入したAIソリューション

  1. **発酵条件最適化AI** : ベイズ最適化による温度・時間の自動調整
  2. **品質予測モデル** : 乳原料成分から最終製品品質を事前予測
  3. **異常検出システム** : 発酵中の温度・pH変化をリアルタイム監視

### 💻 コード例5.1: ヨーグルト発酵プロセスの最適化
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ヨーグルト発酵プロセスのシミュレーション
    class YogurtFermentationSimulator:
        """ヨーグルト発酵プロセスシミュレータ"""
    
        def __init__(self):
            # 最適条件（真の値、実験では不明）
            self.optimal_temp = 42.5  # ℃
            self.optimal_time = 5.0   # 時間
            self.optimal_pH = 6.2
    
        def simulate_quality(self, temperature, fermentation_time, initial_pH, lactose_content=4.5):
            """
            発酵条件から品質スコアをシミュレート
    
            Args:
                temperature: 発酵温度（℃）
                fermentation_time: 発酵時間（時間）
                initial_pH: 初期pH
                lactose_content: 乳糖含量（%）
    
            Returns:
                quality_score: 品質スコア（0-100）
            """
            # 温度の影響（40-45℃が最適）
            temp_factor = 1.0 - 0.2 * ((temperature - self.optimal_temp) / 5) ** 2
    
            # 時間の影響（4-6時間が最適）
            time_factor = 1.0 - 0.15 * ((fermentation_time - self.optimal_time) / 2) ** 2
    
            # pHの影響（6.0-6.5が最適）
            pH_factor = 1.0 - 0.1 * ((initial_pH - self.optimal_pH) / 0.5) ** 2
    
            # 乳糖含量の影響（4.0-5.0%が最適）
            lactose_factor = 1.0 - 0.05 * ((lactose_content - 4.5) / 0.5) ** 2
    
            # 総合品質スコア
            base_quality = 85
            quality_score = base_quality * temp_factor * time_factor * pH_factor * lactose_factor
    
            # ランダムノイズ（プロセス変動）
            noise = np.random.normal(0, 2)
            quality_score += noise
    
            # 0-100の範囲にクリップ
            quality_score = np.clip(quality_score, 0, 100)
    
            return quality_score
    
        def simulate_acidity(self, temperature, fermentation_time):
            """発酵後の酸度を計算（°T）"""
            # 温度と時間が高いほど酸度が増加
            acidity = 60 + (temperature - 40) * 2 + fermentation_time * 5
            acidity += np.random.normal(0, 3)
            return np.clip(acidity, 40, 100)
    
        def simulate_viscosity(self, temperature, protein_content=3.5):
            """粘度の計算（mPa·s）"""
            # 温度が高いと粘度が下がる
            viscosity = 5000 - (temperature - 42) * 200 + protein_content * 300
            viscosity += np.random.normal(0, 200)
            return np.clip(viscosity, 2000, 8000)
    
    # シミュレータの初期化
    simulator = YogurtFermentationSimulator()
    
    # 実験データの生成（過去の生産データを模擬）
    np.random.seed(42)
    n_experiments = 50
    
    experimental_data = []
    for i in range(n_experiments):
        temp = np.random.uniform(38, 46)
        time = np.random.uniform(3, 7)
        pH = np.random.uniform(5.8, 6.6)
        lactose = np.random.uniform(4.0, 5.0)
    
        quality = simulator.simulate_quality(temp, time, pH, lactose)
        acidity = simulator.simulate_acidity(temp, time)
        viscosity = simulator.simulate_viscosity(temp)
    
        experimental_data.append({
            'temperature': temp,
            'fermentation_time': time,
            'initial_pH': pH,
            'lactose_content': lactose,
            'quality_score': quality,
            'acidity': acidity,
            'viscosity': viscosity
        })
    
    df_experiments = pd.DataFrame(experimental_data)
    
    # ベイズ最適化の実装
    class BayesianOptimizationYogurt:
        """ヨーグルト発酵条件のベイズ最適化"""
    
        def __init__(self, bounds, simulator, n_init=10):
            self.bounds = np.array(bounds)
            self.simulator = simulator
            self.n_init = n_init
            self.X_sample = []
            self.y_sample = []
    
            # ガウス過程回帰モデル
            kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2))
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                               alpha=1e-6, normalize_y=True)
    
        def acquisition_function(self, X, xi=0.01):
            """Expected Improvement獲得関数"""
            X = np.atleast_2d(X)
            mu, sigma = self.gp.predict(X, return_std=True)
    
            if len(self.y_sample) > 0:
                mu_sample_opt = np.max(self.y_sample)
            else:
                mu_sample_opt = 0
    
            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                Z = imp / sigma
                ei = imp * self._norm_cdf(Z) + sigma * self._norm_pdf(Z)
                ei[sigma == 0.0] = 0.0
    
            return ei
    
        def _norm_pdf(self, x):
            """標準正規分布のPDF"""
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
        def _norm_cdf(self, x):
            """標準正規分布のCDF"""
            return 0.5 * (1 + np.vectorize(lambda t: np.sign(t) * np.sqrt(1 - np.exp(-2*t**2/np.pi)))(x))
    
        def propose_location(self):
            """次の実験点を提案"""
            def min_obj(X):
                return -self.acquisition_function(X)
    
            min_val = float('inf')
            min_x = None
    
            # ランダムスタートで最適化
            for _ in range(25):
                x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
    
                if res.fun < min_val:
                    min_val = res.fun
                    min_x = res.x
    
            return min_x
    
        def optimize(self, n_iter=20, initial_pH=6.2, lactose_content=4.5):
            """最適化の実行"""
            # 初期ランダムサンプリング
            for _ in range(self.n_init):
                x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                y = self.simulator.simulate_quality(x[0], x[1], initial_pH, lactose_content)
                self.X_sample.append(x)
                self.y_sample.append(y)
    
            # ベイズ最適化のメインループ
            for iteration in range(n_iter):
                # GPモデルの更新
                self.gp.fit(np.array(self.X_sample), np.array(self.y_sample))
    
                # 次の実験点を提案
                x_next = self.propose_location()
    
                # 実験実施（シミュレーション）
                y_next = self.simulator.simulate_quality(x_next[0], x_next[1], initial_pH, lactose_content)
    
                # データ追加
                self.X_sample.append(x_next)
                self.y_sample.append(y_next)
    
                if (iteration + 1) % 5 == 0:
                    best_idx = np.argmax(self.y_sample)
                    best_x = self.X_sample[best_idx]
                    best_y = self.y_sample[best_idx]
                    print(f"反復 {iteration + 1}: 現在の最良 = 品質スコア {best_y:.2f} "
                          f"(温度: {best_x[0]:.1f}℃, 時間: {best_x[1]:.1f}h)")
    
            # 最適条件の抽出
            best_idx = np.argmax(self.y_sample)
            best_params = self.X_sample[best_idx]
            best_quality = self.y_sample[best_idx]
    
            return best_params, best_quality
    
    # ベイズ最適化の実行
    print("=" * 60)
    print("ヨーグルト発酵プロセス最適化（ベイズ最適化）")
    print("=" * 60)
    
    bounds = [[38, 46], [3, 7]]  # [温度範囲, 時間範囲]
    optimizer = BayesianOptimizationYogurt(bounds, simulator, n_init=10)
    
    print("\n最適化開始...")
    best_params, best_quality = optimizer.optimize(n_iter=20, initial_pH=6.2, lactose_content=4.5)
    
    print("\n" + "=" * 60)
    print("最適化結果")
    print("=" * 60)
    print(f"最適温度: {best_params[0]:.2f} ℃")
    print(f"最適発酵時間: {best_params[1]:.2f} 時間")
    print(f"予測品質スコア: {best_quality:.2f}")
    print(f"\n参考: 真の最適条件")
    print(f"最適温度: {simulator.optimal_temp} ℃")
    print(f"最適発酵時間: {simulator.optimal_time} 時間")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 最適化の履歴
    iterations = range(1, len(optimizer.y_sample) + 1)
    cumulative_best = [max(optimizer.y_sample[:i+1]) for i in range(len(optimizer.y_sample))]
    
    axes[0, 0].plot(iterations, optimizer.y_sample, 'o-', color='#11998e', alpha=0.6, label='各実験の品質')
    axes[0, 0].plot(iterations, cumulative_best, 'r-', linewidth=2, label='累積最良値')
    axes[0, 0].set_xlabel('実験回数')
    axes[0, 0].set_ylabel('品質スコア')
    axes[0, 0].set_title('ベイズ最適化の収束過程', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. 温度-時間マップ
    temp_grid = np.linspace(38, 46, 50)
    time_grid = np.linspace(3, 7, 50)
    T, Ti = np.meshgrid(temp_grid, time_grid)
    Z = np.zeros_like(T)
    
    for i in range(len(temp_grid)):
        for j in range(len(time_grid)):
            Z[j, i] = simulator.simulate_quality(T[j, i], Ti[j, i], 6.2, 4.5)
    
    contour = axes[0, 1].contourf(T, Ti, Z, levels=20, cmap='RdYlGn')
    axes[0, 1].scatter([x[0] for x in optimizer.X_sample],
                       [x[1] for x in optimizer.X_sample],
                       c='blue', s=50, edgecolor='black', linewidth=1, label='実験点', zorder=5)
    axes[0, 1].scatter(best_params[0], best_params[1], c='red', s=200, marker='*',
                       edgecolor='black', linewidth=2, label='最適点', zorder=6)
    axes[0, 1].set_xlabel('発酵温度（℃）')
    axes[0, 1].set_ylabel('発酵時間（時間）')
    axes[0, 1].set_title('品質スコアの等高線図', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    plt.colorbar(contour, ax=axes[0, 1], label='品質スコア')
    
    # 3. 温度の影響
    temp_range = np.linspace(38, 46, 30)
    quality_temp = [simulator.simulate_quality(t, best_params[1], 6.2, 4.5) for t in temp_range]
    
    axes[1, 0].plot(temp_range, quality_temp, color='#38ef7d', linewidth=2)
    axes[1, 0].axvline(x=best_params[0], color='red', linestyle='--', linewidth=2, label=f'最適温度: {best_params[0]:.1f}℃')
    axes[1, 0].set_xlabel('発酵温度（℃）')
    axes[1, 0].set_ylabel('品質スコア')
    axes[1, 0].set_title('温度と品質の関係', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 時間の影響
    time_range = np.linspace(3, 7, 30)
    quality_time = [simulator.simulate_quality(best_params[0], t, 6.2, 4.5) for t in time_range]
    
    axes[1, 1].plot(time_range, quality_time, color='#11998e', linewidth=2)
    axes[1, 1].axvline(x=best_params[1], color='red', linestyle='--', linewidth=2, label=f'最適時間: {best_params[1]:.1f}h')
    axes[1, 1].set_xlabel('発酵時間（時間）')
    axes[1, 1].set_ylabel('品質スコア')
    axes[1, 1].set_title('発酵時間と品質の関係', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('yogurt_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 📊 導入効果（6ヶ月後）

指標 | 導入前 | 導入後 | 改善率  
---|---|---|---  
廃棄率 | 3.5% | 1.2% | ▼ 65.7%  
品質スコア平均 | 78.5 | 89.2 | ▲ 13.6%  
ばらつき（標準偏差） | 8.3 | 3.1 | ▼ 62.7%  
年間コスト削減 | - | 約400万円 | -  
  
### 🔑 成功のポイント

  * 既存センサデータの活用により初期投資を抑制
  * ベイズ最適化により少ない実験回数で最適条件を発見
  * 現場オペレータとのコミュニケーションを重視し、AIの意思決定を可視化
  * 段階的導入（1ライン→全ライン展開）によるリスク管理

## 🥤 5.2 ケーススタディ2: 清涼飲料水の予知保全システム

### 📋 企業プロファイル

  * **業種** : 飲料メーカー（従業員500名）
  * **製品** : 炭酸飲料、ジュース、スポーツドリンク
  * **生産量** : 日産200万本

### 🚨 課題

  * 充填機の突然の故障による生産停止（年間15回、平均停止時間4時間）
  * 計画外停止による機会損失と納期遅延
  * 過剰な予防保全による保全コスト増大
  * 設備停止時の原因特定に時間がかかる（平均1.5時間）

### 💡 導入したAIソリューション

  1. **設備故障予測AI** : Random Forestによる24時間先までの故障リスク予測
  2. **異常検出システム** : Isolation Forestによるリアルタイム監視
  3. **根本原因分析ツール** : 決定木による故障要因の自動特定

### 💻 コード例5.2: 充填機の故障予測ダッシュボード
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 充填機センサデータのシミュレーション（24時間分）
    np.random.seed(42)
    hours = 24
    data_points_per_hour = 60  # 1分毎
    total_points = hours * data_points_per_hour
    
    # タイムスタンプの生成
    start_time = datetime(2025, 10, 27, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(total_points)]
    
    # 正常運転データ生成（0-18時間）
    normal_hours = 18 * data_points_per_hour
    
    temp_normal = np.random.normal(65, 2, normal_hours)
    vibration_normal = np.random.normal(0.3, 0.05, normal_hours)
    pressure_normal = np.random.normal(4.0, 0.1, normal_hours)
    flow_rate_normal = np.random.normal(1000, 20, normal_hours)
    motor_current_normal = np.random.normal(25, 1, normal_hours)
    
    # 異常の兆候（18-24時間：徐々に劣化）
    degradation_hours = total_points - normal_hours
    t_degrade = np.linspace(0, 1, degradation_hours)
    
    temp_degrade = 65 + t_degrade * 15 + np.random.normal(0, 3, degradation_hours)
    vibration_degrade = 0.3 + t_degrade * 0.5 + np.random.normal(0, 0.1, degradation_hours)
    pressure_degrade = 4.0 - t_degrade * 0.8 + np.random.normal(0, 0.15, degradation_hours)
    flow_rate_degrade = 1000 - t_degrade * 150 + np.random.normal(0, 30, degradation_hours)
    motor_current_degrade = 25 + t_degrade * 10 + np.random.normal(0, 2, degradation_hours)
    
    # データフレーム作成
    sensor_data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.concatenate([temp_normal, temp_degrade]),
        'vibration': np.concatenate([vibration_normal, vibration_degrade]),
        'pressure': np.concatenate([pressure_normal, pressure_degrade]),
        'flow_rate': np.concatenate([flow_rate_normal, flow_rate_degrade]),
        'motor_current': np.concatenate([motor_current_normal, motor_current_degrade])
    })
    
    # 故障リスクスコアの計算（簡易版）
    def calculate_failure_risk(row):
        """センサ値から故障リスクスコアを計算（0-100）"""
        # 各パラメータの正常範囲からの逸脱度
        temp_risk = max(0, (row['temperature'] - 65) / 20) * 100
        vib_risk = max(0, (row['vibration'] - 0.3) / 0.7) * 100
        pressure_risk = max(0, (4.0 - row['pressure']) / 2.0) * 100
        flow_risk = max(0, (1000 - row['flow_rate']) / 200) * 100
        current_risk = max(0, (row['motor_current'] - 25) / 15) * 100
    
        # 総合リスクスコア（最大値を採用）
        total_risk = max(temp_risk, vib_risk, pressure_risk, flow_risk, current_risk)
        return min(100, total_risk)
    
    sensor_data['failure_risk'] = sensor_data.apply(calculate_failure_risk, axis=1)
    
    # リスクレベルの分類
    def classify_risk_level(risk_score):
        if risk_score < 20:
            return 'Low'
        elif risk_score < 50:
            return 'Medium'
        elif risk_score < 80:
            return 'High'
        else:
            return 'Critical'
    
    sensor_data['risk_level'] = sensor_data['failure_risk'].apply(classify_risk_level)
    
    # 統計サマリー
    print("=" * 60)
    print("充填機予知保全ダッシュボード")
    print("=" * 60)
    print(f"\n監視期間: {timestamps[0].strftime('%Y-%m-%d %H:%M')} ~ {timestamps[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"総データポイント数: {total_points}")
    
    current_risk = sensor_data.iloc[-1]['failure_risk']
    current_level = sensor_data.iloc[-1]['risk_level']
    print(f"\n現在の故障リスク: {current_risk:.1f} ({current_level})")
    
    # リスクレベルごとの集計
    risk_counts = sensor_data['risk_level'].value_counts()
    print(f"\nリスクレベル分布:")
    for level in ['Low', 'Medium', 'High', 'Critical']:
        if level in risk_counts.index:
            count = risk_counts[level]
            percentage = count / total_points * 100
            print(f"  {level}: {count}件 ({percentage:.1f}%)")
    
    # 警告メッセージ
    if current_risk >= 80:
        print(f"\n🚨 【緊急警告】故障リスクが危険域に到達しています！")
        print(f"   推奨アクション: 直ちに生産を停止し、設備点検を実施してください")
    elif current_risk >= 50:
        print(f"\n⚠️ 【警告】故障リスクが上昇しています")
        print(f"   推奨アクション: 次回の休憩時間に設備点検を計画してください")
    elif current_risk >= 20:
        print(f"\n📝 【注意】わずかな異常の兆候が検出されています")
        print(f"   推奨アクション: 継続的な監視を実施してください")
    else:
        print(f"\n✅ 【正常】設備は正常に稼働しています")
    
    # 可視化ダッシュボード
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 時刻データ（X軸用）
    time_hours = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]
    
    # 1. 温度トレンド
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_hours, sensor_data['temperature'], color='#ff6b6b', linewidth=1)
    ax1.axhline(y=65, color='green', linestyle='--', alpha=0.5, label='正常値')
    ax1.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='警告閾値')
    ax1.axhline(y=85, color='red', linestyle='--', alpha=0.5, label='危険閾値')
    ax1.set_xlabel('経過時間（時間）')
    ax1.set_ylabel('温度（℃）')
    ax1.set_title('充填ヘッド温度', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    
    # 2. 振動トレンド
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_hours, sensor_data['vibration'], color='#4ecdc4', linewidth=1)
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('経過時間（時間）')
    ax2.set_ylabel('振動（mm/s）')
    ax2.set_title('振動レベル', fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. 圧力トレンド
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_hours, sensor_data['pressure'], color='#95e1d3', linewidth=1)
    ax3.axhline(y=4.0, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(y=3.5, color='orange', linestyle='--', alpha=0.5)
    ax3.axhline(y=3.0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('経過時間（時間）')
    ax3.set_ylabel('圧力（MPa）')
    ax3.set_title('充填圧力', fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. 流量トレンド
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_hours, sensor_data['flow_rate'], color='#f38181', linewidth=1)
    ax4.axhline(y=1000, color='green', linestyle='--', alpha=0.5)
    ax4.axhline(y=900, color='orange', linestyle='--', alpha=0.5)
    ax4.axhline(y=800, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('経過時間（時間）')
    ax4.set_ylabel('流量（本/分）')
    ax4.set_title('充填流量', fontsize=11, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 5. モーター電流トレンド
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(time_hours, sensor_data['motor_current'], color='#aa96da', linewidth=1)
    ax5.axhline(y=25, color='green', linestyle='--', alpha=0.5)
    ax5.axhline(y=30, color='orange', linestyle='--', alpha=0.5)
    ax5.axhline(y=35, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('経過時間（時間）')
    ax5.set_ylabel('電流（A）')
    ax5.set_title('モーター電流', fontsize=11, fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # 6. 故障リスクスコア
    ax6 = fig.add_subplot(gs[2, 1])
    colors = []
    for risk in sensor_data['failure_risk']:
        if risk < 20:
            colors.append('green')
        elif risk < 50:
            colors.append('yellow')
        elif risk < 80:
            colors.append('orange')
        else:
            colors.append('red')
    
    ax6.scatter(time_hours, sensor_data['failure_risk'], c=colors, s=5, alpha=0.6)
    ax6.axhline(y=20, color='yellow', linestyle='--', alpha=0.5, label='注意')
    ax6.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='警告')
    ax6.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='危険')
    ax6.set_xlabel('経過時間（時間）')
    ax6.set_ylabel('故障リスクスコア')
    ax6.set_title('統合故障リスク評価', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
    
    # 7. リスクレベル推移（積み上げ面グラフ）
    ax7 = fig.add_subplot(gs[3, :])
    
    # 1時間ごとにリサンプリング
    sensor_data['hour'] = sensor_data['timestamp'].dt.hour
    risk_by_hour = sensor_data.groupby(['hour', 'risk_level']).size().unstack(fill_value=0)
    
    # 積み上げ棒グラフ
    risk_by_hour.plot(kind='bar', stacked=True, ax=ax7,
                      color={'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Critical': 'red'},
                      width=0.8)
    ax7.set_xlabel('時刻')
    ax7.set_ylabel('データポイント数')
    ax7.set_title('時間帯別リスクレベル分布', fontsize=11, fontweight='bold')
    ax7.legend(title='リスクレベル', fontsize=8)
    ax7.grid(axis='y', alpha=0.3)
    
    plt.suptitle('充填機予知保全ダッシュボード - リアルタイム監視', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig('filling_machine_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保全推奨スケジュール
    print("\n" + "=" * 60)
    print("推奨保全スケジュール")
    print("=" * 60)
    
    if current_risk >= 80:
        print("⏰ 緊急保全: 2時間以内")
        print("📝 点検項目: 全システム総合点検、部品交換準備")
    elif current_risk >= 50:
        print("⏰ 計画保全: 24時間以内")
        print("📝 点検項目: 振動・温度センサ周辺、モーター軸受")
    elif current_risk >= 20:
        print("⏰ 予防保全: 1週間以内")
        print("📝 点検項目: 定期清掃、潤滑油補充")
    else:
        print("⏰ 次回定期保全: 通常スケジュール通り")
        print("📝 点検項目: 標準点検項目")
    

### 📊 導入効果（12ヶ月後）

指標 | 導入前 | 導入後 | 改善率  
---|---|---|---  
計画外停止回数 | 15回/年 | 3回/年 | ▼ 80%  
平均故障対応時間 | 4.0時間 | 1.5時間 | ▼ 62.5%  
保全コスト | 年間1200万円 | 年間850万円 | ▼ 29.2%  
設備稼働率 | 92.5% | 97.8% | ▲ 5.7%  
  
### 🔑 成功のポイント

  * IoTセンサの追加投資により、リアルタイムデータ収集を実現
  * 保全担当者向けダッシュボードで、専門知識がなくても状況を把握可能
  * 段階的なアラート設定により、過検知を抑制
  * 保全履歴データとAI予測を組み合わせた根本原因分析

## 🍪 5.3 その他の食品カテゴリ事例サマリー

### スナック菓子製造（フライ工程）

**課題** : フライ油の劣化による品質低下、油交換タイミングの最適化困難

**AI技術** : 画像認識による製品色調分析 + 時系列予測による油劣化予測

**効果** : 油交換頻度の20%削減、廃油コスト削減（年間250万円）

### パン製造（発酵・焼成）

**課題** : 気候変動による発酵速度の変化、焼きムラの発生

**AI技術** : 機械学習による発酵時間自動調整 + サーモグラフィ画像解析

**効果** : 焼きムラ不良率を2.8%→0.6%に削減

### 調味料製造（発酵調味料）

**課題** : 長期発酵プロセス（6ヶ月～1年）の品質予測が困難

**AI技術** : 深層学習（LSTM）による発酵終了時の品質予測

**効果** : 発酵3ヶ月時点で最終品質を±5%精度で予測、不良ロットの早期検出

## 🎯 5.4 AI導入の成功要因と教訓

### 成功の共通要因

  1. **経営層のコミットメント** : トップダウンでのAI戦略推進
  2. **現場との協働** : オペレータの知見とAIの融合
  3. **スモールスタート** : 1ライン・1プロセスから開始し、成功事例を横展開
  4. **データ品質の確保** : センサ校正、データクレンジング
  5. **継続的改善** : モデルの定期的な再訓練と精度向上

### 失敗から学んだ教訓

  * **過度な期待値設定** : AIは万能ではない。適用範囲の明確化が重要
  * **データ不足** : 最低6ヶ月～1年分のデータが必要（季節変動を含む）
  * **ブラックボックス化** : 説明可能性の欠如は現場の不信感を招く
  * **保守体制の不備** : 導入後のメンテナンス計画が不可欠

### ROI評価のポイント

#### 投資対効果の計算式

$$ \text{ROI} = \frac{\text{年間コスト削減額} + \text{生産性向上による増益}}{\text{初期投資額} + \text{年間運用コスト}} \times 100 (\%) $$ 

  * **初期投資** : センサ設置、システム開発、教育訓練
  * **コスト削減** : 廃棄削減、保全コスト削減、エネルギー削減
  * **増益** : 品質向上による付加価値増、生産量増加

**ベンチマーク** : 2-3年でのROI回収が一般的な目標

## 📚 まとめ

本章では、食品製造プロセスにおける実際のAI導入事例を学びました。

### 主要なポイント

  * 業種・製品特性に応じたAI技術の選定と適用
  * ベイズ最適化、故障予測、異常検出など、複数技術の統合活用
  * 定量的な効果測定とROI評価の重要性
  * 現場との協働と継続的改善の文化醸成
  * スモールスタートによるリスク管理

**🎓 シリーズ完了**  
本シリーズ「食品プロセスAI入門」では、食品製造現場におけるAI活用の基礎から実践まで、 包括的に学びました。今後は、自社の課題に応じて、適切なAI技術を選定・導入し、 データ駆動型の製造プロセス改善を推進してください。  
  
継続学習のリソース:  
・プロセス・インフォマティクス道場の他シリーズ  
・機械学習道場の基礎シリーズ  
・業界カンファレンスやワークショップへの参加 

← 第4章: 予知保全（準備中） [シリーズ目次へ →](<index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
