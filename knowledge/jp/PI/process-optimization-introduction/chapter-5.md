---
title: 第5章：ケーススタディ - 化学プロセスの最適運転条件探索
chapter_title: 第5章：ケーススタディ - 化学プロセスの最適運転条件探索
subtitle: 完全な産業最適化ワークフロー：経済的目的関数、プロセスモデリング、ロバスト最適化、実時間最適化
---

## はじめに

この最終章では、これまで学んだすべての技術を統合して、実際の化学プロセスの最適運転条件探索に取り組みます。完全な最適化ワークフロー、経済的目的関数の設計、感度分析、ロバスト最適化、そして実時間最適化フレームワークまで、産業実務に直結する内容を扱います。

#### この章で学ぶこと

  * **完全な最適化ワークフロー** : 問題定義から結果検証まで
  * **CSTR最適化の完全実装** : 多変数最適化と経済的目的関数
  * **感度分析** : パラメータ変動に対する最適解の頑健性評価
  * **ロバスト最適化** : 不確実性下での最適化
  * **実時間最適化** : オンラインデータに基づく適応的最適化
  * **蒸留塔最適化** : 総合ケーススタディ

## 5.1 完全な最適化ワークフロー

産業プロセスの最適化は、以下の体系的なワークフローに従って実行されます。
    
    
    flowchart TD
        A[1. 問題定義とゴール設定] --> B[2. プロセスモデルの開発]
        B --> C[3. 制約条件の定義]
        C --> D[4. 最適化問題の定式化]
        D --> E[5. アルゴリズム選択と実行]
        E --> F[6. 結果の検証と解釈]
        F --> G{目標達成?}
        G -->|No| H[7. モデル改善]
        H --> B
        G -->|Yes| I[8. 感度分析]
        I --> J[9. 実装と監視]
    
        style A fill:#e8f5e9
        style D fill:#fff9c4
        style E fill:#ffe0b2
        style F fill:#f8bbd0
        style I fill:#c5cae9
        style J fill:#b2dfdb
    

### 各ステップの詳細

**ステップ1: 問題定義とゴール設定**

  * 最適化の目的を明確化（コスト削減、収率向上、エネルギー効率改善等）
  * KPI（Key Performance Indicator）の定義
  * ベースライン性能の測定
  * 目標値の設定（定量的かつ達成可能な目標）

**ステップ2: プロセスモデルの開発**

  * 第一原理モデル（物質収支、エネルギー収支、反応速度式）
  * データ駆動モデル（機械学習、統計モデル）
  * ハイブリッドモデル（第一原理+データ駆動）
  * モデルの検証と妥当性確認

**ステップ3: 制約条件の定義**

  * 安全制約（温度、圧力、流量の上下限）
  * 製品規格制約（純度、品質指標）
  * 環境制約（排出基準、エネルギー使用量）
  * 運転制約（設備能力、物理的限界）

**ステップ4-9** : 残りのステップは以降のセクションで詳しく説明します。

## 5.2 CSTR最適化の完全実装

連続撹拌槽型反応器（CSTR）の最適運転条件探索を、完全な実装例で学びます。

### 問題設定

**反応システム** : 単純な可逆反応 A → B（エキソサーミック反応）

**目的** : 利益を最大化（製品収益 - 原料コスト - エネルギーコスト）

**決定変数** :

  * **T** : 反応温度 [°C] (50-300)
  * **τ** : 滞留時間 [min] (30-180)
  * **C A0**: 供給濃度 [mol/L] (1-5)

**制約条件** :

  * 安全制約: T ≤ 350°C
  * 純度制約: XA (転化率) ≥ 0.95
  * 流量制約: 100 ≤ F ≤ 400 L/h

#### コード例1: CSTR問題の完全定式化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import pandas as pd
    
    class CSTROptimization:
        """
        連続撹拌槽型反応器（CSTR）の最適化クラス
    
        反応: A → B (エキソサーミック反応)
        目的: 利益最大化（収益 - 原料コスト - エネルギーコスト）
        """
    
        def __init__(self):
            # 経済パラメータ（現実的な価格設定）
            self.price_B = 1200.0      # 製品B価格 [¥/kg]
            self.price_A = 600.0       # 原料A価格 [¥/kg]
            self.energy_cost = 12.0    # エネルギーコスト [¥/kWh]
    
            # 物性パラメータ
            self.MW_A = 60.0           # 分子量A [g/mol]
            self.MW_B = 60.0           # 分子量B [g/mol]
            self.rho = 1000.0          # 密度 [kg/m³]
            self.Cp = 4.18             # 比熱 [kJ/kg·K]
    
            # 反応速度パラメータ（Arrhenius式）
            self.k0 = 1e10             # 頻度因子 [1/min]
            self.Ea = 80000.0          # 活性化エネルギー [J/mol]
            self.R = 8.314             # ガス定数 [J/mol·K]
            self.delta_H = -50000.0    # 反応熱 [J/mol] (発熱反応)
    
            # 制約条件パラメータ
            self.T_max = 350.0         # 最大許容温度 [°C]
            self.X_min = 0.95          # 最小転化率 [-]
            self.F_min = 100.0         # 最小流量 [L/h]
            self.F_max = 400.0         # 最大流量 [L/h]
    
        def reaction_rate_constant(self, T_celsius):
            """
            反応速度定数を計算（Arrhenius式）
    
            Parameters:
            -----------
            T_celsius : float
                温度 [°C]
    
            Returns:
            --------
            k : float
                反応速度定数 [1/min]
            """
            T_K = T_celsius + 273.15
            k = self.k0 * np.exp(-self.Ea / (self.R * T_K))
            return k
    
        def conversion(self, T, tau):
            """
            CSTR転化率を計算
    
            Parameters:
            -----------
            T : float
                温度 [°C]
            tau : float
                滞留時間 [min]
    
            Returns:
            --------
            X_A : float
                転化率 [-]
            """
            k = self.reaction_rate_constant(T)
            X_A = (k * tau) / (1 + k * tau)  # CSTR転化率式
            return X_A
    
        def profit(self, x):
            """
            利益関数（最大化対象）
    
            Parameters:
            -----------
            x : array_like
                [T, tau, C_A0]
                T: 温度 [°C]
                tau: 滞留時間 [min]
                C_A0: 供給濃度 [mol/L]
    
            Returns:
            --------
            profit : float
                利益 [¥/h]（負の値を返す理由：scipy.minimizeは最小化するため）
            """
            T, tau, C_A0 = x
    
            # 転化率を計算
            X_A = self.conversion(T, tau)
    
            # 反応器体積と流量
            V = 1000.0  # 固定体積 [L]（1 m³）
            F = V / tau  # 体積流量 [L/h]
    
            # 生産速度 [mol/h]
            production_rate_B = F * C_A0 * X_A
    
            # 収益 [¥/h]
            revenue = production_rate_B * (self.MW_B / 1000.0) * self.price_B
    
            # 原料コスト [¥/h]
            raw_material_cost = F * C_A0 * (self.MW_A / 1000.0) * self.price_A
    
            # エネルギーコスト [¥/h]（反応熱除去のための冷却）
            Q_reaction = abs(self.delta_H) * production_rate_B  # [J/h]
            energy_cost = (Q_reaction / 3.6e6) * self.energy_cost  # [¥/h]（J→kWhに変換）
    
            # 利益 = 収益 - 原料コスト - エネルギーコスト
            profit = revenue - raw_material_cost - energy_cost
    
            # 最小化のため負の値を返す
            return -profit
    
        def constraints(self, x):
            """
            制約条件を定義
    
            Returns:
            --------
            constraints : list of dict
                scipy.optimizeの制約条件形式
            """
            T, tau, C_A0 = x
            V = 1000.0
            F = V / tau
            X_A = self.conversion(T, tau)
    
            cons = [
                {'type': 'ineq', 'fun': lambda x: self.T_max - x[0]},           # T ≤ 350°C
                {'type': 'ineq', 'fun': lambda x: self.conversion(x[0], x[1]) - self.X_min},  # X_A ≥ 0.95
                {'type': 'ineq', 'fun': lambda x: V / x[1] - self.F_min},       # F ≥ 100 L/h
                {'type': 'ineq', 'fun': lambda x: self.F_max - V / x[1]}        # F ≤ 400 L/h
            ]
    
            return cons
    
    # インスタンス化とテスト
    cstr = CSTROptimization()
    
    # 運転条件の例：T=200°C, τ=60min, C_A0=3.0 mol/L
    test_conditions = [200.0, 60.0, 3.0]
    profit_value = -cstr.profit(test_conditions)
    conversion_value = cstr.conversion(test_conditions[0], test_conditions[1])
    
    print("=" * 60)
    print("CSTR最適化問題の定式化")
    print("=" * 60)
    print(f"\n運転条件: T={test_conditions[0]:.1f}°C, τ={test_conditions[1]:.1f}min, C_A0={test_conditions[2]:.1f}mol/L")
    print(f"転化率: {conversion_value:.3f} ({conversion_value*100:.1f}%)")
    print(f"利益: ¥{profit_value:,.0f}/h")
    print("\n目的: 利益を最大化")
    print("制約: T≤350°C, X_A≥0.95, 100≤F≤400 L/h")
    

#### コード例2: プロセスモデルの開発（反応速度式と物質収支）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    class CSTRProcessModel:
        """
        CSTRプロセスモデル: 反応速度式と物質収支
    
        モデル:
        - Arrhenius速度式: k(T) = k0 * exp(-Ea/RT)
        - CSTR物質収支: X_A = (k*τ) / (1 + k*τ)
        - エネルギー収支: Q_reaction + Q_cooling = 0
        """
    
        def __init__(self):
            self.k0 = 1e10
            self.Ea = 80000.0
            self.R = 8.314
    
        def k(self, T_celsius):
            """反応速度定数 [1/min]"""
            T_K = T_celsius + 273.15
            return self.k0 * np.exp(-self.Ea / (self.R * T_K))
    
        def conversion_vs_temperature(self, tau=60.0):
            """温度 vs 転化率のプロット"""
            T_range = np.linspace(50, 300, 100)
            X_A = np.array([self.k(T) * tau / (1 + self.k(T) * tau) for T in T_range])
    
            plt.figure(figsize=(10, 6))
            plt.plot(T_range, X_A * 100, 'b-', linewidth=2, label=f'τ = {tau} min')
            plt.axhline(y=95, color='r', linestyle='--', linewidth=1.5, label='目標転化率 95%')
            plt.axvline(x=350, color='orange', linestyle='--', linewidth=1.5, label='最大温度 350°C')
    
            plt.xlabel('温度 T [°C]', fontsize=12, fontweight='bold')
            plt.ylabel('転化率 X_A [%]', fontsize=12, fontweight='bold')
            plt.title('CSTR転化率 vs 反応温度', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig('cstr_conversion_vs_temp.png', dpi=150, bbox_inches='tight')
            plt.show()
    
        def conversion_surface(self):
            """温度・滞留時間 vs 転化率の3Dサーフェス"""
            T_range = np.linspace(50, 300, 50)
            tau_range = np.linspace(30, 180, 50)
            T_grid, tau_grid = np.meshgrid(T_range, tau_range)
    
            X_A_grid = np.zeros_like(T_grid)
            for i in range(len(T_range)):
                for j in range(len(tau_range)):
                    k_val = self.k(T_grid[j, i])
                    X_A_grid[j, i] = (k_val * tau_grid[j, i]) / (1 + k_val * tau_grid[j, i])
    
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
    
            surf = ax.plot_surface(T_grid, tau_grid, X_A_grid * 100, cmap='viridis',
                                   alpha=0.9, edgecolor='none')
    
            # 転化率95%の等値面
            contour = ax.contour(T_grid, tau_grid, X_A_grid * 100, levels=[95],
                                 colors='red', linewidths=3)
    
            ax.set_xlabel('温度 T [°C]', fontsize=12, fontweight='bold')
            ax.set_ylabel('滞留時間 τ [min]', fontsize=12, fontweight='bold')
            ax.set_zlabel('転化率 X_A [%]', fontsize=12, fontweight='bold')
            ax.set_title('CSTR転化率の3D表面プロット', fontsize=14, fontweight='bold')
    
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='転化率 [%]')
            plt.tight_layout()
            plt.savefig('cstr_conversion_3d.png', dpi=150, bbox_inches='tight')
            plt.show()
    
            print("=" * 60)
            print("CSTRプロセスモデル可視化")
            print("=" * 60)
            print("\n転化率95%を達成する運転条件:")
            print("- 高温 + 短滞留時間")
            print("- 低温 + 長滞留時間")
            print("\n最適化では経済性も考慮して最適点を選択")
    
    # 実行
    model = CSTRProcessModel()
    model.conversion_vs_temperature(tau=60.0)
    model.conversion_surface()
    

#### コード例3: 多変数最適化の実行
    
    
    from scipy.optimize import minimize
    import numpy as np
    
    class CSTRMultivariableOptimization:
        """
        CSTR多変数最適化の実行と結果解釈
    
        決定変数:
        - T: 温度 [°C]
        - τ: 滞留時間 [min]
        - C_A0: 供給濃度 [mol/L]
        """
    
        def __init__(self):
            self.cstr = CSTROptimization()
    
        def optimize(self):
            """最適化を実行"""
    
            # 初期推定値
            x0 = np.array([200.0, 60.0, 3.0])  # [T, tau, C_A0]
    
            # 変数の境界
            bounds = [
                (50.0, 300.0),    # T [°C]
                (30.0, 180.0),    # tau [min]
                (1.0, 5.0)        # C_A0 [mol/L]
            ]
    
            # 制約条件
            constraints = self.cstr.constraints(x0)
    
            print("=" * 60)
            print("CSTR多変数最適化の実行")
            print("=" * 60)
            print(f"\n初期推定値:")
            print(f"  T   = {x0[0]:.1f} °C")
            print(f"  τ   = {x0[1]:.1f} min")
            print(f"  C_A0 = {x0[2]:.2f} mol/L")
            print(f"\n初期利益: ¥{-self.cstr.profit(x0):,.0f}/h")
    
            # SLSQP法で最適化
            result = minimize(
                self.cstr.profit,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': True, 'maxiter': 100}
            )
    
            # 結果の解釈
            self.interpret_results(result)
    
            return result
    
        def interpret_results(self, result):
            """最適化結果の解釈と出力"""
    
            print("\n" + "=" * 60)
            print("最適化結果")
            print("=" * 60)
    
            if result.success:
                print("\n✓ 最適化成功！")
            else:
                print("\n✗ 最適化失敗")
                print(f"理由: {result.message}")
                return
    
            T_opt, tau_opt, C_A0_opt = result.x
            profit_opt = -result.fun
            X_A_opt = self.cstr.conversion(T_opt, tau_opt)
            V = 1000.0
            F_opt = V / tau_opt
    
            print(f"\n【最適運転条件】")
            print(f"  温度 T*        : {T_opt:.2f} °C")
            print(f"  滞留時間 τ*    : {tau_opt:.2f} min")
            print(f"  供給濃度 C_A0* : {C_A0_opt:.3f} mol/L")
            print(f"  流量 F*        : {F_opt:.2f} L/h")
    
            print(f"\n【プロセス性能】")
            print(f"  転化率 X_A     : {X_A_opt:.4f} ({X_A_opt*100:.2f}%)")
            print(f"  最大利益       : ¥{profit_opt:,.0f}/h")
    
            # 年間利益換算
            annual_profit = profit_opt * 24 * 365
            print(f"  年間利益換算   : ¥{annual_profit:,.0f}/year")
    
            # 制約条件のマージン確認
            print(f"\n【制約条件マージン】")
            print(f"  温度制約   : T* = {T_opt:.1f}°C ≤ {self.cstr.T_max}°C (マージン: {self.cstr.T_max - T_opt:.1f}°C)")
            print(f"  転化率制約 : X_A* = {X_A_opt:.3f} ≥ {self.cstr.X_min} (マージン: {X_A_opt - self.cstr.X_min:.4f})")
            print(f"  流量制約   : F* = {F_opt:.1f} L/h ∈ [{self.cstr.F_min}, {self.cstr.F_max}]")
    
            # 反復回数
            print(f"\n【最適化統計】")
            print(f"  反復回数       : {result.nit}")
            print(f"  関数評価回数   : {result.nfev}")
    
    # 実行
    optimizer = CSTRMultivariableOptimization()
    result_optimal = optimizer.optimize()
    

#### コード例4: 包括的な制約条件の実装
    
    
    import numpy as np
    from scipy.optimize import minimize
    
    class CSTRComprehensiveConstraints:
        """
        包括的な制約条件を持つCSTR最適化
    
        制約:
        1. 安全制約: 温度上限、圧力上限
        2. 純度制約: 製品純度、転化率
        3. 流量制約: 最小・最大流量
        4. エネルギー制約: 冷却能力
        5. 環境制約: 排出基準
        """
    
        def __init__(self):
            self.cstr = CSTROptimization()
    
            # 追加制約パラメータ
            self.Q_cooling_max = 500000.0  # 最大冷却能力 [J/h]
            self.emission_limit = 10.0     # CO2排出上限 [kg/h]
            self.purity_min = 0.98         # 製品純度下限 [-]
    
        def comprehensive_constraints(self, x):
            """包括的な制約条件"""
            T, tau, C_A0 = x
            V = 1000.0
            F = V / tau
            X_A = self.cstr.conversion(T, tau)
    
            # 生産速度
            production_rate_B = F * C_A0 * X_A
    
            # 反応熱
            Q_reaction = abs(self.cstr.delta_H) * production_rate_B
    
            # CO2排出量（エネルギー使用に比例すると仮定）
            CO2_emission = (Q_reaction / 3.6e6) * 0.5  # [kg/h]（仮定: 0.5 kg-CO2/kWh）
    
            # 製品純度（簡略化: 転化率に比例）
            purity = 0.90 + 0.10 * X_A  # X_A=1.0で純度100%
    
            constraints = [
                # 既存の制約
                {'type': 'ineq', 'fun': lambda x: self.cstr.T_max - x[0]},
                {'type': 'ineq', 'fun': lambda x: self.cstr.conversion(x[0], x[1]) - self.cstr.X_min},
                {'type': 'ineq', 'fun': lambda x: V / x[1] - self.cstr.F_min},
                {'type': 'ineq', 'fun': lambda x: self.cstr.F_max - V / x[1]},
    
                # 追加制約
                {'type': 'ineq', 'fun': lambda x: self.Q_cooling_max - abs(self.cstr.delta_H) * V / x[1] * x[2] * self.cstr.conversion(x[0], x[1])},  # 冷却能力
                {'type': 'ineq', 'fun': lambda x: self.emission_limit - (abs(self.cstr.delta_H) * V / x[1] * x[2] * self.cstr.conversion(x[0], x[1])) / 3.6e6 * 0.5},  # CO2排出
                {'type': 'ineq', 'fun': lambda x: (0.90 + 0.10 * self.cstr.conversion(x[0], x[1])) - self.purity_min}  # 製品純度
            ]
    
            return constraints
    
        def optimize_with_comprehensive_constraints(self):
            """包括的制約下での最適化"""
    
            x0 = np.array([200.0, 60.0, 3.0])
            bounds = [(50.0, 300.0), (30.0, 180.0), (1.0, 5.0)]
    
            constraints = self.comprehensive_constraints(x0)
    
            print("=" * 60)
            print("包括的制約条件下でのCSTR最適化")
            print("=" * 60)
            print("\n制約条件:")
            print("  1. 安全制約: T ≤ 350°C")
            print("  2. 転化率制約: X_A ≥ 0.95")
            print("  3. 流量制約: 100 ≤ F ≤ 400 L/h")
            print("  4. 冷却能力制約: Q_reaction ≤ 500,000 J/h")
            print("  5. 環境制約: CO2排出 ≤ 10 kg/h")
            print("  6. 純度制約: 製品純度 ≥ 0.98")
    
            result = minimize(
                self.cstr.profit,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200}
            )
    
            if result.success:
                T_opt, tau_opt, C_A0_opt = result.x
                profit_opt = -result.fun
                X_A_opt = self.cstr.conversion(T_opt, tau_opt)
                purity_opt = 0.90 + 0.10 * X_A_opt
                V = 1000.0
                F_opt = V / tau_opt
                Q_reaction = abs(self.cstr.delta_H) * F_opt * C_A0_opt * X_A_opt
                CO2 = (Q_reaction / 3.6e6) * 0.5
    
                print(f"\n✓ 最適化成功")
                print(f"\n【最適運転条件】")
                print(f"  温度: {T_opt:.2f}°C, 滞留時間: {tau_opt:.2f}min, 供給濃度: {C_A0_opt:.3f}mol/L")
                print(f"\n【性能指標】")
                print(f"  利益: ¥{profit_opt:,.0f}/h")
                print(f"  転化率: {X_A_opt*100:.2f}%")
                print(f"  製品純度: {purity_opt*100:.2f}%")
                print(f"  CO2排出: {CO2:.2f} kg/h")
                print(f"  冷却負荷: {Q_reaction:,.0f} J/h ({Q_reaction/self.Q_cooling_max*100:.1f}% of max)")
            else:
                print(f"\n✗ 最適化失敗: {result.message}")
    
            return result
    
    # 実行
    comprehensive_opt = CSTRComprehensiveConstraints()
    result_comp = comprehensive_opt.optimize_with_comprehensive_constraints()
    

## 5.3 感度分析とロバスト最適化

実プロセスでは、パラメータ（原料価格、エネルギーコスト等）が変動します。感度分析により、最適解がこれらの変動に対してどの程度頑健かを評価します。

#### コード例5: 最適化実行と結果解釈

（このコードはコード例3で既に実装済み）

#### コード例6: 感度分析（パラメータ摂動）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    class SensitivityAnalysis:
        """
        最適解のパラメータ感度分析
    
        分析対象:
        - 原料価格の変動（±10%）
        - エネルギーコストの変動（±20%）
        - 反応速度パラメータの不確実性（±15%）
        """
    
        def __init__(self):
            self.cstr = CSTROptimization()
            self.base_price_A = self.cstr.price_A
            self.base_energy_cost = self.cstr.energy_cost
            self.base_k0 = self.cstr.k0
    
        def sensitivity_raw_material_price(self, optimal_x):
            """原料価格の感度分析"""
    
            # 価格変動範囲: ±10%
            price_variation = np.linspace(0.90, 1.10, 21)
            profits = []
    
            for factor in price_variation:
                self.cstr.price_A = self.base_price_A * factor
                profit = -self.cstr.profit(optimal_x)
                profits.append(profit)
    
            # リセット
            self.cstr.price_A = self.base_price_A
    
            return price_variation, profits
    
        def sensitivity_energy_cost(self, optimal_x):
            """エネルギーコストの感度分析"""
    
            # コスト変動範囲: ±20%
            cost_variation = np.linspace(0.80, 1.20, 21)
            profits = []
    
            for factor in cost_variation:
                self.cstr.energy_cost = self.base_energy_cost * factor
                profit = -self.cstr.profit(optimal_x)
                profits.append(profit)
    
            # リセット
            self.cstr.energy_cost = self.base_energy_cost
    
            return cost_variation, profits
    
        def sensitivity_reaction_rate(self, optimal_x):
            """反応速度パラメータの感度分析"""
    
            # パラメータ変動範囲: ±15%
            k0_variation = np.linspace(0.85, 1.15, 21)
            profits = []
            conversions = []
    
            for factor in k0_variation:
                self.cstr.k0 = self.base_k0 * factor
                profit = -self.cstr.profit(optimal_x)
                X_A = self.cstr.conversion(optimal_x[0], optimal_x[1])
                profits.append(profit)
                conversions.append(X_A)
    
            # リセット
            self.cstr.k0 = self.base_k0
    
            return k0_variation, profits, conversions
    
        def run_sensitivity_analysis(self):
            """感度分析の実行と可視化"""
    
            # まず最適化を実行して最適解を取得
            optimizer = CSTRMultivariableOptimization()
            result = optimizer.optimize()
    
            if not result.success:
                print("最適化失敗のため感度分析を実行できません")
                return
    
            optimal_x = result.x
            base_profit = -result.fun
    
            print("\n" + "=" * 60)
            print("感度分析")
            print("=" * 60)
            print(f"\nベース最適解: T={optimal_x[0]:.1f}°C, τ={optimal_x[1]:.1f}min, C_A0={optimal_x[2]:.2f}mol/L")
            print(f"ベース利益: ¥{base_profit:,.0f}/h")
    
            # 各パラメータの感度分析
            price_var, price_profits = self.sensitivity_raw_material_price(optimal_x)
            energy_var, energy_profits = self.sensitivity_energy_cost(optimal_x)
            k0_var, k0_profits, k0_conversions = self.sensitivity_reaction_rate(optimal_x)
    
            # 可視化
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            # 1. 原料価格感度
            ax1 = axes[0, 0]
            ax1.plot((price_var - 1) * 100, price_profits, 'b-', linewidth=2, marker='o', markersize=4)
            ax1.axhline(y=base_profit, color='r', linestyle='--', linewidth=1.5, label='ベース利益')
            ax1.axvline(x=0, color='gray', linestyle=':', linewidth=1)
            ax1.set_xlabel('原料価格変動 [%]', fontsize=11, fontweight='bold')
            ax1.set_ylabel('利益 [¥/h]', fontsize=11, fontweight='bold')
            ax1.set_title('原料価格の感度分析', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
    
            # 2. エネルギーコスト感度
            ax2 = axes[0, 1]
            ax2.plot((energy_var - 1) * 100, energy_profits, 'g-', linewidth=2, marker='s', markersize=4)
            ax2.axhline(y=base_profit, color='r', linestyle='--', linewidth=1.5, label='ベース利益')
            ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1)
            ax2.set_xlabel('エネルギーコスト変動 [%]', fontsize=11, fontweight='bold')
            ax2.set_ylabel('利益 [¥/h]', fontsize=11, fontweight='bold')
            ax2.set_title('エネルギーコストの感度分析', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
    
            # 3. 反応速度パラメータ感度（利益）
            ax3 = axes[1, 0]
            ax3.plot((k0_var - 1) * 100, k0_profits, 'm-', linewidth=2, marker='^', markersize=4)
            ax3.axhline(y=base_profit, color='r', linestyle='--', linewidth=1.5, label='ベース利益')
            ax3.axvline(x=0, color='gray', linestyle=':', linewidth=1)
            ax3.set_xlabel('反応速度定数k0変動 [%]', fontsize=11, fontweight='bold')
            ax3.set_ylabel('利益 [¥/h]', fontsize=11, fontweight='bold')
            ax3.set_title('反応速度パラメータ感度（利益）', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
    
            # 4. 反応速度パラメータ感度（転化率）
            ax4 = axes[1, 1]
            ax4.plot((k0_var - 1) * 100, np.array(k0_conversions) * 100, 'c-', linewidth=2, marker='d', markersize=4)
            ax4.axhline(y=95, color='r', linestyle='--', linewidth=1.5, label='最小転化率 95%')
            ax4.axvline(x=0, color='gray', linestyle=':', linewidth=1)
            ax4.set_xlabel('反応速度定数k0変動 [%]', fontsize=11, fontweight='bold')
            ax4.set_ylabel('転化率 [%]', fontsize=11, fontweight='bold')
            ax4.set_title('反応速度パラメータ感度（転化率）', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
    
            plt.tight_layout()
            plt.savefig('cstr_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()
    
            # 感度係数の計算
            print("\n【感度係数】（パラメータ1%変動あたりの利益変動）")
    
            # 原料価格感度
            d_profit_price = (price_profits[11] - price_profits[9]) / (2 * 0.01 * base_profit)
            print(f"  原料価格: {d_profit_price:.3f} (1%上昇 → 利益{abs(d_profit_price):.2f}%減少)")
    
            # エネルギーコスト感度
            d_profit_energy = (energy_profits[11] - energy_profits[9]) / (2 * 0.01 * base_profit)
            print(f"  エネルギーコスト: {d_profit_energy:.3f} (1%上昇 → 利益{abs(d_profit_energy):.2f}%減少)")
    
            # 反応速度感度
            d_profit_k0 = (k0_profits[11] - k0_profits[9]) / (2 * 0.01 * base_profit)
            print(f"  反応速度定数: {d_profit_k0:.3f} (1%上昇 → 利益{d_profit_k0:.2f}%増加)")
    
            print("\n結論:")
            print("  - 最も影響が大きいのは原料価格")
            print("  - エネルギーコストの影響は比較的小さい")
            print("  - 反応速度パラメータの不確実性に注意が必要")
    
    # 実行
    sensitivity = SensitivityAnalysis()
    sensitivity.run_sensitivity_analysis()
    

#### コード例7: ロバスト最適化（不確実性下での最適化）
    
    
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    
    class RobustOptimization:
        """
        ロバスト最適化: 不確実性下での最適化
    
        アプローチ:
        - ワーストケース最適化
        - 期待値最適化（モンテカルロ法）
        - 確率的制約の扱い
        """
    
        def __init__(self, n_samples=100):
            self.cstr = CSTROptimization()
            self.n_samples = n_samples
    
            # 不確実性パラメータ（標準偏差）
            self.sigma_price_A = 0.05 * self.cstr.price_A      # 原料価格の不確実性 ±5%
            self.sigma_energy = 0.10 * self.cstr.energy_cost   # エネルギーコストの不確実性 ±10%
            self.sigma_k0 = 0.10 * self.cstr.k0                # 反応速度の不確実性 ±10%
    
        def expected_profit(self, x):
            """
            期待利益（モンテカルロ法）
    
            Parameters:
            -----------
            x : array_like
                [T, tau, C_A0]
    
            Returns:
            --------
            expected_profit : float
                期待利益の負値（最小化のため）
            """
            np.random.seed(42)  # 再現性のため
    
            profits = []
    
            for _ in range(self.n_samples):
                # パラメータをサンプリング（正規分布）
                price_A_sample = np.random.normal(self.cstr.price_A, self.sigma_price_A)
                energy_cost_sample = np.random.normal(self.cstr.energy_cost, self.sigma_energy)
                k0_sample = np.random.normal(self.cstr.k0, self.sigma_k0)
    
                # 一時的にパラメータを更新
                original_price_A = self.cstr.price_A
                original_energy_cost = self.cstr.energy_cost
                original_k0 = self.cstr.k0
    
                self.cstr.price_A = max(0, price_A_sample)
                self.cstr.energy_cost = max(0, energy_cost_sample)
                self.cstr.k0 = max(0, k0_sample)
    
                # 利益を計算
                profit = -self.cstr.profit(x)
                profits.append(profit)
    
                # パラメータをリセット
                self.cstr.price_A = original_price_A
                self.cstr.energy_cost = original_energy_cost
                self.cstr.k0 = original_k0
    
            # 期待値を計算
            expected_profit_value = np.mean(profits)
    
            return -expected_profit_value  # 最小化のため負値
    
        def worst_case_profit(self, x):
            """
            ワーストケース利益
    
            パラメータが最悪の組み合わせ（利益最小）となる場合を想定
            """
            # ワーストケース: 原料価格↑、エネルギーコスト↑、反応速度↓
            original_price_A = self.cstr.price_A
            original_energy_cost = self.cstr.energy_cost
            original_k0 = self.cstr.k0
    
            self.cstr.price_A = self.cstr.price_A + 2 * self.sigma_price_A
            self.cstr.energy_cost = self.cstr.energy_cost + 2 * self.sigma_energy
            self.cstr.k0 = self.cstr.k0 - 2 * self.sigma_k0
    
            profit_worst = -self.cstr.profit(x)
    
            # リセット
            self.cstr.price_A = original_price_A
            self.cstr.energy_cost = original_energy_cost
            self.cstr.k0 = original_k0
    
            return -profit_worst  # 最小化のため負値
    
        def optimize_robust(self, method='expected'):
            """
            ロバスト最適化の実行
    
            Parameters:
            -----------
            method : str
                'expected' or 'worst_case'
            """
    
            x0 = np.array([200.0, 60.0, 3.0])
            bounds = [(50.0, 300.0), (30.0, 180.0), (1.0, 5.0)]
    
            constraints = self.cstr.constraints(x0)
    
            print("=" * 60)
            print(f"ロバスト最適化: {method}法")
            print("=" * 60)
            print(f"\n不確実性:")
            print(f"  原料価格: ±{self.sigma_price_A/self.cstr.price_A*100:.1f}%")
            print(f"  エネルギーコスト: ±{self.sigma_energy/self.cstr.energy_cost*100:.1f}%")
            print(f"  反応速度定数: ±{self.sigma_k0/self.cstr.k0*100:.1f}%")
    
            if method == 'expected':
                objective = self.expected_profit
                print(f"\n目的: 期待利益を最大化（モンテカルロサンプル数: {self.n_samples}）")
            elif method == 'worst_case':
                objective = self.worst_case_profit
                print(f"\n目的: ワーストケース利益を最大化")
            else:
                raise ValueError("method must be 'expected' or 'worst_case'")
    
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )
    
            if result.success:
                T_opt, tau_opt, C_A0_opt = result.x
    
                # 各種シナリオでの利益を評価
                profit_nominal = -self.cstr.profit(result.x)
                profit_expected = -self.expected_profit(result.x)
                profit_worst = -self.worst_case_profit(result.x)
    
                print(f"\n✓ ロバスト最適化成功")
                print(f"\n【ロバスト最適運転条件】")
                print(f"  温度: {T_opt:.2f}°C")
                print(f"  滞留時間: {tau_opt:.2f}min")
                print(f"  供給濃度: {C_A0_opt:.3f}mol/L")
    
                print(f"\n【各シナリオでの利益】")
                print(f"  ノミナルケース: ¥{profit_nominal:,.0f}/h")
                print(f"  期待値: ¥{profit_expected:,.0f}/h")
                print(f"  ワーストケース: ¥{profit_worst:,.0f}/h")
    
                profit_range = profit_nominal - profit_worst
                print(f"\n  利益変動幅: ¥{profit_range:,.0f}/h ({profit_range/profit_nominal*100:.1f}%)")
            else:
                print(f"\n✗ 最適化失敗: {result.message}")
    
            return result
    
        def compare_robust_strategies(self):
            """ロバスト最適化戦略の比較"""
    
            print("\n" + "=" * 60)
            print("ロバスト最適化戦略の比較")
            print("=" * 60)
    
            # 通常の最適化（不確実性を考慮しない）
            optimizer_nominal = CSTRMultivariableOptimization()
            result_nominal = optimizer_nominal.optimize()
    
            # 期待値ロバスト最適化
            result_expected = self.optimize_robust(method='expected')
    
            # ワーストケースロバスト最適化
            result_worst = self.optimize_robust(method='worst_case')
    
            # 比較表を作成
            strategies = ['通常最適化', '期待値ロバスト', 'ワーストケースロバスト']
            results = [result_nominal, result_expected, result_worst]
    
            comparison_data = []
            for strategy, result in zip(strategies, results):
                if result.success:
                    T, tau, C_A0 = result.x
                    profit_nom = -self.cstr.profit(result.x)
                    profit_exp = -self.expected_profit(result.x)
                    profit_worst = -self.worst_case_profit(result.x)
    
                    comparison_data.append({
                        '戦略': strategy,
                        'T [°C]': f"{T:.1f}",
                        'τ [min]': f"{tau:.1f}",
                        'C_A0 [mol/L]': f"{C_A0:.2f}",
                        'ノミナル利益 [¥/h]': f"{profit_nom:,.0f}",
                        '期待利益 [¥/h]': f"{profit_exp:,.0f}",
                        'ワースト利益 [¥/h]': f"{profit_worst:,.0f}"
                    })
    
            df_comparison = pd.DataFrame(comparison_data)
            print("\n")
            print(df_comparison.to_string(index=False))
    
            print("\n【推奨】")
            print("  - 安定運転を重視: ワーストケースロバスト最適化")
            print("  - 平均利益を重視: 期待値ロバスト最適化")
            print("  - パラメータが正確: 通常最適化")
    
    # 実行
    robust_opt = RobustOptimization(n_samples=100)
    robust_opt.compare_robust_strategies()
    

## 5.4 実時間最適化フレームワーク

実時間最適化（Real-Time Optimization, RTO）は、オンラインで取得されるプロセスデータに基づいてモデルパラメータを更新し、定期的に最適化を再実行する適応的な最適化手法です。
    
    
    flowchart LR
        A[プロセス運転] -->|測定データ| B[データ収集]
        B --> C[モデルパラメータ更新]
        C --> D[最適化実行]
        D --> E[最適運転条件]
        E -->|制御指令| A
    
        D --> F{目標達成?}
        F -->|No| G[モデル再評価]
        G --> C
        F -->|Yes| H[次のサイクルへ]
        H --> B
    
        style A fill:#e8f5e9
        style D fill:#fff9c4
        style E fill:#ffe0b2
        style C fill:#f8bbd0
    

#### コード例8: 実時間最適化フレームワーク
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import time
    
    class RealTimeOptimization:
        """
        実時間最適化（RTO）フレームワーク
    
        機能:
        1. オンラインデータ取得
        2. モデルパラメータ更新
        3. 最適化再実行
        4. 制御指令生成
        """
    
        def __init__(self):
            self.cstr = CSTROptimization()
    
            # RTOパラメータ
            self.update_interval = 60  # 更新間隔 [min]
            self.n_iterations = 10     # シミュレーション反復回数
    
            # データ履歴
            self.time_history = []
            self.profit_history = []
            self.T_history = []
            self.tau_history = []
            self.C_A0_history = []
            self.X_A_history = []
    
            # 現在の運転条件
            self.current_x = np.array([200.0, 60.0, 3.0])
    
        def simulate_online_data(self, iteration):
            """
            オンラインデータをシミュレート
    
            実プロセスでは:
            - DCS（分散制御システム）からデータ取得
            - センサー測定値のフィルタリング
            - 異常値の検出と除去
            """
            # パラメータドリフトをシミュレート
            # 実際には経年劣化、触媒活性低下等で反応速度が変化
            k0_drift = 1.0 - 0.05 * (iteration / self.n_iterations)  # 5%の劣化
    
            # ノイズを追加（測定誤差）
            noise = np.random.normal(0, 0.02)
            k0_measured = self.cstr.k0 * k0_drift * (1 + noise)
    
            return {'k0': k0_measured, 'drift_factor': k0_drift}
    
        def update_model_parameters(self, online_data):
            """
            モデルパラメータを更新
    
            実装:
            - 移動窓推定（Moving Horizon Estimation, MHE）
            - 再帰最小二乗法（Recursive Least Squares, RLS）
            - カルマンフィルタ
            """
            # 簡略化: 測定値でパラメータを直接更新
            self.cstr.k0 = online_data['k0']
    
            print(f"  モデルパラメータ更新: k0 = {self.cstr.k0:.3e} (劣化率: {(1-online_data['drift_factor'])*100:.1f}%)")
    
        def optimize_current_conditions(self):
            """現在の条件で最適化を実行"""
    
            x0 = self.current_x
            bounds = [(50.0, 300.0), (30.0, 180.0), (1.0, 5.0)]
            constraints = self.cstr.constraints(x0)
    
            result = minimize(
                self.cstr.profit,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 50}
            )
    
            if result.success:
                return result.x
            else:
                print(f"  警告: 最適化失敗 ({result.message}), 前回の条件を維持")
                return self.current_x
    
        def run_rto_cycle(self, iteration):
            """RTOサイクルを1回実行"""
    
            print(f"\n--- RTO Cycle {iteration+1}/{self.n_iterations} (時刻: {iteration*self.update_interval}分) ---")
    
            # ステップ1: オンラインデータ取得
            online_data = self.simulate_online_data(iteration)
    
            # ステップ2: モデルパラメータ更新
            self.update_model_parameters(online_data)
    
            # ステップ3: 最適化実行
            optimal_x = self.optimize_current_conditions()
    
            # ステップ4: 制御指令生成（実装では徐々に移行）
            # 実際にはMPCやPIDで緩やかに設定値変更
            self.current_x = optimal_x
    
            # ステップ5: 性能評価
            profit = -self.cstr.profit(optimal_x)
            X_A = self.cstr.conversion(optimal_x[0], optimal_x[1])
    
            print(f"  最適運転条件: T={optimal_x[0]:.1f}°C, τ={optimal_x[1]:.1f}min, C_A0={optimal_x[2]:.2f}mol/L")
            print(f"  性能: 利益=¥{profit:,.0f}/h, 転化率={X_A*100:.2f}%")
    
            # 履歴に記録
            self.time_history.append(iteration * self.update_interval)
            self.profit_history.append(profit)
            self.T_history.append(optimal_x[0])
            self.tau_history.append(optimal_x[1])
            self.C_A0_history.append(optimal_x[2])
            self.X_A_history.append(X_A)
    
        def run_rto_simulation(self):
            """RTO全体のシミュレーション"""
    
            print("=" * 60)
            print("実時間最適化（RTO）シミュレーション")
            print("=" * 60)
            print(f"\n設定:")
            print(f"  更新間隔: {self.update_interval}分")
            print(f"  シミュレーション回数: {self.n_iterations}")
            print(f"  総運転時間: {self.n_iterations * self.update_interval}分")
    
            for iteration in range(self.n_iterations):
                self.run_rto_cycle(iteration)
    
            # 結果の可視化
            self.visualize_rto_results()
    
        def visualize_rto_results(self):
            """RTO結果の可視化"""
    
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
            # 1. 利益の推移
            ax1 = axes[0, 0]
            ax1.plot(self.time_history, self.profit_history, 'b-o', linewidth=2, markersize=5)
            ax1.set_xlabel('時間 [min]', fontsize=11, fontweight='bold')
            ax1.set_ylabel('利益 [¥/h]', fontsize=11, fontweight='bold')
            ax1.set_title('利益の時間推移', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
    
            # 2. 転化率の推移
            ax2 = axes[0, 1]
            ax2.plot(self.time_history, np.array(self.X_A_history) * 100, 'g-o', linewidth=2, markersize=5)
            ax2.axhline(y=95, color='r', linestyle='--', linewidth=1.5, label='最小転化率 95%')
            ax2.set_xlabel('時間 [min]', fontsize=11, fontweight='bold')
            ax2.set_ylabel('転化率 [%]', fontsize=11, fontweight='bold')
            ax2.set_title('転化率の時間推移', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
    
            # 3. 温度の推移
            ax3 = axes[1, 0]
            ax3.plot(self.time_history, self.T_history, 'm-o', linewidth=2, markersize=5)
            ax3.set_xlabel('時間 [min]', fontsize=11, fontweight='bold')
            ax3.set_ylabel('温度 [°C]', fontsize=11, fontweight='bold')
            ax3.set_title('反応温度の時間推移', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
    
            # 4. 滞留時間の推移
            ax4 = axes[1, 1]
            ax4.plot(self.time_history, self.tau_history, 'c-o', linewidth=2, markersize=5)
            ax4.set_xlabel('時間 [min]', fontsize=11, fontweight='bold')
            ax4.set_ylabel('滞留時間 [min]', fontsize=11, fontweight='bold')
            ax4.set_title('滞留時間の時間推移', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
    
            # 5. 供給濃度の推移
            ax5 = axes[2, 0]
            ax5.plot(self.time_history, self.C_A0_history, 'y-o', linewidth=2, markersize=5)
            ax5.set_xlabel('時間 [min]', fontsize=11, fontweight='bold')
            ax5.set_ylabel('供給濃度 [mol/L]', fontsize=11, fontweight='bold')
            ax5.set_title('供給濃度の時間推移', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
    
            # 6. 利益累積
            ax6 = axes[2, 1]
            cumulative_profit = np.cumsum(self.profit_history) * (self.update_interval / 60)  # [¥]
            ax6.plot(self.time_history, cumulative_profit, 'r-o', linewidth=2, markersize=5)
            ax6.set_xlabel('時間 [min]', fontsize=11, fontweight='bold')
            ax6.set_ylabel('累積利益 [¥]', fontsize=11, fontweight='bold')
            ax6.set_title('累積利益', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('rto_simulation_results.png', dpi=150, bbox_inches='tight')
            plt.show()
    
            print("\n" + "=" * 60)
            print("RTOシミュレーション結果サマリー")
            print("=" * 60)
            print(f"\n初期利益: ¥{self.profit_history[0]:,.0f}/h")
            print(f"最終利益: ¥{self.profit_history[-1]:,.0f}/h")
            print(f"平均利益: ¥{np.mean(self.profit_history):,.0f}/h")
            print(f"利益変動: {(self.profit_history[-1] - self.profit_history[0]) / self.profit_history[0] * 100:.1f}%")
            print(f"\n総累積利益: ¥{cumulative_profit[-1]:,.0f}")
            print(f"\n結論: RTOにより触媒劣化に適応し、最適運転条件を維持")
    
    # 実行
    rto = RealTimeOptimization()
    rto.run_rto_simulation()
    

## 5.5 総合ケーススタディ: 多成分蒸留塔の最適化

最後に、より複雑な蒸留塔の最適化を総合ケーススタディとして取り組みます。

### 問題設定

**システム** : 5成分混合物の連続蒸留塔

**目的** : エネルギーコストを最小化しつつ、製品純度規格を満たす

**決定変数** :

  * **R** : 還流比（Reflux Ratio） [-] (1.5-4.0)
  * **Q R**: リボイラー熱負荷 [kW] (500-2000)
  * **N stages**: 段数 [-] (20-50, 整数)

**制約条件** :

  * 製品純度: xproduct ≥ 0.98
  * 回収率: recovery ≥ 0.95
  * 圧力損失: ΔP ≤ 50 kPa
  * 物質収支: F = D + B

#### コード例9: 蒸留塔最適化の完全実装
    
    
    import numpy as np
    from scipy.optimize import minimize, differential_evolution
    import matplotlib.pyplot as plt
    
    class DistillationColumnOptimization:
        """
        多成分蒸留塔の最適化
    
        目的: エネルギーコストを最小化
        制約: 製品純度、回収率、圧力損失
        """
    
        def __init__(self):
            # 経済パラメータ
            self.energy_cost = 15.0         # スチームコスト [¥/kWh]
            self.cooling_cost = 3.0         # 冷却水コスト [¥/kWh]
            self.product_price = 2000.0     # 製品価格 [¥/kg]
    
            # プロセスパラメータ
            self.F = 1000.0                 # 供給流量 [kg/h]
            self.x_F = 0.50                 # 供給組成（軽質成分） [-]
            self.MW_avg = 80.0              # 平均分子量 [g/mol]
    
            # 物性パラメータ（簡略化）
            self.lambda_vap = 300.0         # 蒸発潜熱 [kJ/kg]
            self.Cp = 2.5                   # 比熱 [kJ/kg·K]
    
            # 制約パラメータ
            self.purity_min = 0.98          # 最小製品純度 [-]
            self.recovery_min = 0.95        # 最小回収率 [-]
            self.dP_max = 50.0              # 最大圧力損失 [kPa]
    
        def column_model(self, R, Q_R, N_stages):
            """
            蒸留塔モデル（簡略化）
    
            実際には:
            - MESH方程式（Material, Equilibrium, Summation, Heat balance）
            - 状態方程式（VLE: Vapor-Liquid Equilibrium）
            - Aspenなどのシミュレータで計算
    
            ここでは簡略化したショートカット法を使用
            """
            # Fenske-Underwood-Gilliland法に基づく簡略計算
    
            # 最小還流比（Underwood式の簡略版）
            alpha = 2.5  # 比揮発度（簡略化）
            R_min = (alpha - 1) / alpha
    
            # 段効率（経験式）
            efficiency = 0.60 + 0.008 * N_stages  # 段数が多いほど効率向上
            efficiency = min(efficiency, 0.85)
    
            # 実効段数
            N_eff = N_stages * efficiency
    
            # 製品純度（経験式、実際にはMESH方程式で計算）
            purity = 1.0 - np.exp(-(N_eff / 30.0) * (R / R_min - 1.0))
            purity = min(purity, 0.995)  # 物理的上限
    
            # 回収率
            recovery = 0.85 + 0.10 * (R / (R + 1))
            recovery = min(recovery, 0.98)
    
            # 留出量（物質収支）
            D = self.F * self.x_F * recovery / purity  # [kg/h]
            B = self.F - D  # [kg/h]
    
            # 圧力損失（経験式）
            dP = 0.5 * N_stages + 10.0  # [kPa]
    
            # コンデンサー熱負荷
            Q_C = (R + 1) * D * self.lambda_vap / 3600.0  # [kW]（kg/h→kg/sの変換）
    
            return {
                'purity': purity,
                'recovery': recovery,
                'D': D,
                'B': B,
                'Q_C': Q_C,
                'Q_R': Q_R,
                'dP': dP
            }
    
        def objective(self, x):
            """
            目的関数: 総運転コストを最小化
    
            コスト = リボイラーコスト + コンデンサーコスト
            """
            R, Q_R, N_stages = x
            N_stages = int(round(N_stages))  # 整数化
    
            # モデル計算
            results = self.column_model(R, Q_R, N_stages)
    
            # 運転コスト [¥/h]
            reboiler_cost = Q_R * self.energy_cost
            condenser_cost = results['Q_C'] * self.cooling_cost
    
            total_cost = reboiler_cost + condenser_cost
    
            return total_cost
    
        def constraints_func(self, x):
            """制約条件"""
            R, Q_R, N_stages = x
            N_stages = int(round(N_stages))
    
            results = self.column_model(R, Q_R, N_stages)
    
            # 不等式制約（≥0となるように定義）
            constraints = []
    
            # 純度制約
            constraints.append(results['purity'] - self.purity_min)
    
            # 回収率制約
            constraints.append(results['recovery'] - self.recovery_min)
    
            # 圧力損失制約
            constraints.append(self.dP_max - results['dP'])
    
            # 物理的制約: Q_Rは留出量に対して十分であること
            Q_R_min = results['D'] * self.lambda_vap / 3600.0 * 0.5
            constraints.append(Q_R - Q_R_min)
    
            return constraints
    
        def optimize_distillation(self):
            """蒸留塔最適化の実行"""
    
            print("=" * 60)
            print("多成分蒸留塔の最適化")
            print("=" * 60)
            print(f"\n目的: エネルギーコスト最小化")
            print(f"\n供給条件:")
            print(f"  流量: {self.F:.0f} kg/h")
            print(f"  組成（軽質成分）: {self.x_F*100:.0f}%")
    
            print(f"\n制約条件:")
            print(f"  製品純度 ≥ {self.purity_min*100:.0f}%")
            print(f"  回収率 ≥ {self.recovery_min*100:.0f}%")
            print(f"  圧力損失 ≤ {self.dP_max:.0f} kPa")
    
            # 初期推定値
            x0 = np.array([2.5, 1000.0, 30.0])
    
            # 変数の境界
            bounds = [
                (1.5, 4.0),       # R
                (500.0, 2000.0),  # Q_R [kW]
                (20, 50)          # N_stages
            ]
    
            # 制約条件（scipy形式）
            constraints = [
                {'type': 'ineq', 'fun': lambda x: self.constraints_func(x)[i]}
                for i in range(4)
            ]
    
            print(f"\n最適化開始...")
    
            # 混合整数非線形計画問題（MINLP）
            # N_stagesが整数のため、differential_evolutionを使用（大域的最適化）
            result = differential_evolution(
                self.objective,
                bounds,
                strategy='best1bin',
                maxiter=100,
                popsize=15,
                constraints=constraints,
                seed=42
            )
    
            # 結果の解釈
            self.interpret_distillation_results(result)
    
            return result
    
        def interpret_distillation_results(self, result):
            """蒸留塔最適化結果の解釈"""
    
            print("\n" + "=" * 60)
            print("蒸留塔最適化結果")
            print("=" * 60)
    
            if result.success:
                print("\n✓ 最適化成功")
            else:
                print(f"\n✗ 最適化失敗: {result.message}")
                return
    
            R_opt, Q_R_opt, N_stages_opt = result.x
            N_stages_opt = int(round(N_stages_opt))
    
            results = self.column_model(R_opt, Q_R_opt, N_stages_opt)
    
            total_cost = result.fun
            reboiler_cost = Q_R_opt * self.energy_cost
            condenser_cost = results['Q_C'] * self.cooling_cost
    
            print(f"\n【最適設計・運転条件】")
            print(f"  還流比 R*: {R_opt:.3f}")
            print(f"  リボイラー熱負荷 Q_R*: {Q_R_opt:.1f} kW")
            print(f"  段数 N*: {N_stages_opt}")
    
            print(f"\n【プロセス性能】")
            print(f"  製品純度: {results['purity']*100:.2f}% (要求: ≥{self.purity_min*100:.0f}%)")
            print(f"  回収率: {results['recovery']*100:.2f}% (要求: ≥{self.recovery_min*100:.0f}%)")
            print(f"  留出量: {results['D']:.1f} kg/h")
            print(f"  缶出量: {results['B']:.1f} kg/h")
            print(f"  圧力損失: {results['dP']:.1f} kPa (上限: {self.dP_max:.0f} kPa)")
    
            print(f"\n【エネルギーコスト】")
            print(f"  リボイラーコスト: ¥{reboiler_cost:,.0f}/h")
            print(f"  コンデンサーコスト: ¥{condenser_cost:,.0f}/h")
            print(f"  総運転コスト: ¥{total_cost:,.0f}/h")
    
            # 年間コスト
            annual_cost = total_cost * 24 * 365
            print(f"  年間運転コスト: ¥{annual_cost:,.0f}/year")
    
            # 製品価値
            annual_product_value = results['D'] * 24 * 365 * self.product_price
            annual_profit = annual_product_value - annual_cost
    
            print(f"\n【経済性評価】")
            print(f"  年間製品価値: ¥{annual_product_value:,.0f}/year")
            print(f"  年間利益（粗利）: ¥{annual_profit:,.0f}/year")
    
            # 感度分析のヒント
            print(f"\n【最適化に関する考察】")
            print(f"  - 還流比R*={R_opt:.2f}: エネルギーコストと分離性能のバランス")
            print(f"  - 段数N*={N_stages_opt}: 設備投資（CAPEX）と運転コスト（OPEX）のトレードオフ")
            print(f"  - さらなる改善: 熱統合、中間熱供給、圧力最適化")
    
            # 可視化
            self.visualize_optimization_landscape(R_opt, Q_R_opt, N_stages_opt)
    
        def visualize_optimization_landscape(self, R_opt, Q_R_opt, N_stages_opt):
            """最適化ランドスケープの可視化"""
    
            # 還流比 vs コストのプロット（Q_RとN_stagesは最適値で固定）
            R_range = np.linspace(1.5, 4.0, 30)
            costs_R = []
            purities_R = []
    
            for R in R_range:
                try:
                    cost = self.objective([R, Q_R_opt, N_stages_opt])
                    results = self.column_model(R, Q_R_opt, int(N_stages_opt))
                    costs_R.append(cost)
                    purities_R.append(results['purity'] * 100)
                except:
                    costs_R.append(np.nan)
                    purities_R.append(np.nan)
    
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
            # 1. 還流比 vs コスト
            ax1.plot(R_range, costs_R, 'b-', linewidth=2)
            ax1.axvline(x=R_opt, color='r', linestyle='--', linewidth=2, label=f'最適値 R*={R_opt:.2f}')
            ax1.set_xlabel('還流比 R [-]', fontsize=12, fontweight='bold')
            ax1.set_ylabel('総運転コスト [¥/h]', fontsize=12, fontweight='bold')
            ax1.set_title('還流比 vs 運転コスト', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
    
            # 2. 還流比 vs 製品純度
            ax2.plot(R_range, purities_R, 'g-', linewidth=2)
            ax2.axhline(y=self.purity_min*100, color='r', linestyle='--', linewidth=2, label=f'最小純度 {self.purity_min*100:.0f}%')
            ax2.axvline(x=R_opt, color='orange', linestyle='--', linewidth=2, label=f'最適値 R*={R_opt:.2f}')
            ax2.set_xlabel('還流比 R [-]', fontsize=12, fontweight='bold')
            ax2.set_ylabel('製品純度 [%]', fontsize=12, fontweight='bold')
            ax2.set_title('還流比 vs 製品純度', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
    
            plt.tight_layout()
            plt.savefig('distillation_optimization_landscape.png', dpi=150, bbox_inches='tight')
            plt.show()
    
            print("\n可視化完了: distillation_optimization_landscape.png")
    
    # 実行
    distillation_opt = DistillationColumnOptimization()
    result_distillation = distillation_opt.optimize_distillation()
    

## まとめ

#### 第5章で学んだこと

  * **完全な最適化ワークフロー** : 問題定義からモデル開発、最適化実行、結果検証までの体系的プロセス
  * **CSTR最適化** : 経済的目的関数、多変数最適化、包括的制約条件の実装
  * **感度分析** : パラメータ変動に対する最適解の頑健性評価
  * **ロバスト最適化** : 不確実性下での期待値最適化とワーストケース最適化
  * **実時間最適化** : オンラインデータに基づく適応的最適化フレームワーク
  * **蒸留塔最適化** : 複雑な多変数・制約付き最適化の総合ケーススタディ

### 重要なポイント

  1. **経済的目的関数の設計** : 利益最大化 = 収益 - 原料コスト - エネルギーコスト
  2. **現実的な制約条件** : 安全制約、製品規格、環境基準、運転制約の包括的実装
  3. **感度分析の重要性** : パラメータ不確実性が最適解に与える影響の定量評価
  4. **ロバスト最適化** : 不確実性下でも性能を保証する運転条件の探索
  5. **実時間最適化（RTO）** : プロセス変動に適応する動的な最適化アプローチ

### 実務への適用

**短期的アクション（1-3ヶ月）**

  * 自社プロセスの最適化機会を評価（エネルギーコスト、収率、品質）
  * 簡単な単変数最適化から開始（温度、圧力、流量の調整）
  * DCS/PLCデータを活用した現状性能のベースライン測定

**中期的アクション（3-12ヶ月）**

  * プロセスモデルの開発（第一原理モデルまたはデータ駆動モデル）
  * 多変数最適化の実装と検証（シミュレーション環境）
  * 感度分析とロバスト最適化の実践

**長期的アクション（1-2年）**

  * 実時間最適化（RTO）システムの構築と実装
  * プラント全体の統合最適化（複数ユニットの連携）
  * モデル予測制御（MPC）との統合

### さらなる学習リソース

**推奨書籍**

  * **Biegler, L.T. (2010)** : "Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes"
  * **Edgar, T.F., et al. (2001)** : "Optimization of Chemical Processes"
  * **Seborg, D.E., et al. (2016)** : "Process Dynamics and Control" (MPC関連)

**次のステップ**

  * **動的最適化** : バッチプロセス、スタートアップ最適化
  * **モデル予測制御（MPC）** : 実時間制御と最適化の統合
  * **ベイズ最適化** : ブラックボックスプロセスの最適化
  * **機械学習統合** : サロゲートモデル、強化学習

#### 実装のヒント

**段階的アプローチ** : まずオフライン最適化で効果を検証 → 小規模パイロット試験 → 段階的な本格実装

**安全第一** : すべての最適化は安全制約を最優先し、緊急停止ロジックを実装

**継続的改善** : 定期的なモデル更新と最適化結果の検証を習慣化

## おわりに

この「プロセス最適化入門シリーズ」全5章を通じて、最適化問題の定式化から実際の化学プロセスの最適運転条件探索まで、包括的な知識とスキルを習得しました。

最適化は単なる数学的技法ではなく、プロセス産業の競争力を高め、持続可能性を実現するための強力なツールです。本シリーズで学んだ技術を実務に適用し、プロセスの経済性向上、エネルギー効率改善、環境負荷低減に貢献されることを期待しています。

**最適化の旅はここで終わりではなく、始まりです！**

[← 第4章：制約条件下での最適化](<./chapter-4.html>) [シリーズトップに戻る →](<./index.html>)
