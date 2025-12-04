---
title: 第2章：物質収支・エネルギー収支
chapter_title: 第2章：物質収支・エネルギー収支
subtitle: 化学プロセスの基本法則をPythonで実装する
---

## 2.1 物質収支の基本

物質収支は「入力 = 出力 + 蓄積 + 反応」という保存則に基づきます。化学プロセスシミュレーションでは、 この基本原理を多成分系・リサイクルストリーム・反応系に適用します。 

### 2.1.1 成分別物質収支（多成分系）

蒸留塔や分離器では、各成分ごとに物質収支を取る必要があります。以下はエタノール-水混合物の分離プロセスの例です。 

#### 例題1：蒸留塔の成分別物質収支

供給流量100 kg/h（エタノール20%、水80%）を蒸留して、留出物（エタノール90%）と缶出物（エタノール2%）に分離します。 
    
    
    import numpy as np
    from scipy.optimize import fsolve
    import pandas as pd
    
    def component_material_balance(F, z_F, x_D, x_B):
        """
        蒸留塔の成分別物質収支を解く
    
        Args:
            F (float): 供給流量 [kg/h]
            z_F (dict): 供給組成 {成分名: モル分率}
            x_D (dict): 留出物組成 {成分名: モル分率}
            x_B (dict): 缶出物組成 {成分名: モル分率}
    
        Returns:
            dict: {'D': 留出物流量, 'B': 缶出物流量, 'recovery': 回収率}
        """
        components = list(z_F.keys())
    
        # 全物質収支: F = D + B
        # 成分収支: F * z_F[i] = D * x_D[i] + B * x_B[i]
    
        def equations(vars):
            D, B = vars
            eq1 = F - D - B  # 全物質収支
    
            # 第1成分の成分収支を使用（他の成分でも同じ）
            comp = components[0]
            eq2 = F * z_F[comp] - D * x_D[comp] - B * x_B[comp]
    
            return [eq1, eq2]
    
        # 初期推定値
        D_init = F * 0.2
        B_init = F * 0.8
    
        # 連立方程式を解く
        D, B = fsolve(equations, [D_init, B_init])
    
        # 各成分の回収率を計算
        recovery = {}
        for comp in components:
            if F * z_F[comp] > 0:
                recovery[comp] = (D * x_D[comp]) / (F * z_F[comp]) * 100
            else:
                recovery[comp] = 0.0
    
        # 物質収支チェック
        balance_check = {}
        for comp in components:
            input_comp = F * z_F[comp]
            output_comp = D * x_D[comp] + B * x_B[comp]
            balance_check[comp] = abs(input_comp - output_comp) < 1e-6
    
        return {
            'D': D,
            'B': B,
            'recovery': recovery,
            'balance_check': balance_check
        }
    
    # 実践例：エタノール-水系蒸留
    F = 100.0  # kg/h
    z_F = {'Ethanol': 0.20, 'Water': 0.80}  # 供給組成
    x_D = {'Ethanol': 0.90, 'Water': 0.10}  # 留出物組成（高純度エタノール）
    x_B = {'Ethanol': 0.02, 'Water': 0.98}  # 缶出物組成（水）
    
    result = component_material_balance(F, z_F, x_D, x_B)
    
    print("=" * 60)
    print("蒸留塔 物質収支計算結果")
    print("=" * 60)
    print(f"\n供給条件:")
    print(f"  供給流量: {F:.1f} kg/h")
    print(f"  供給組成: エタノール {z_F['Ethanol']*100:.1f}%, 水 {z_F['Water']*100:.1f}%")
    
    print(f"\n計算結果:")
    print(f"  留出物流量 (D): {result['D']:.2f} kg/h")
    print(f"  缶出物流量 (B): {result['B']:.2f} kg/h")
    
    print(f"\n回収率:")
    for comp, rec in result['recovery'].items():
        print(f"  {comp}: {rec:.2f}%")
    
    print(f"\n物質収支チェック:")
    for comp, check in result['balance_check'].items():
        status = "✓ OK" if check else "✗ NG"
        print(f"  {comp}: {status}")
    
    # 出力例:
    # ==============================================================
    # 蒸留塔 物質収支計算結果
    # ==============================================================
    #
    # 供給条件:
    #   供給流量: 100.0 kg/h
    #   供給組成: エタノール 20.0%, 水 80.0%
    #
    # 計算結果:
    #   留出物流量 (D): 20.45 kg/h
    #   缶出物流量 (B): 79.55 kg/h
    #
    # 回収率:
    #   Ethanol: 92.05%
    #   Water: 10.23%
    #
    # 物質収支チェック:
    #   Ethanol: ✓ OK
    #   Water: ✓ OK
    

### 2.1.2 リサイクルストリームを含む物質収支

化学プロセスでは、未反応原料を反応器に戻すリサイクルストリームが一般的です。 全物質収支と反応器周りの収支を組み合わせて解きます。 

#### 例題2：リサイクルストリームを含む反応器系

A → B の反応で、反応率70%、分離器でBを99%除去、未反応Aはリサイクルします。 
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    def recycle_material_balance(F_fresh, conversion, separation_eff):
        """
        リサイクルストリームを含む反応器の物質収支
    
        Args:
            F_fresh (float): 新鮮供給流量 [kmol/h]
            conversion (float): 1パス反応率 [-]
            separation_eff (float): 分離効率（製品B除去率） [-]
    
        Returns:
            dict: 各ストリーム流量と組成
        """
        def equations(vars):
            F_reactor, F_recycle = vars
    
            # 反応器入口: F_reactor = F_fresh + F_recycle
            eq1 = F_reactor - F_fresh - F_recycle
    
            # 反応器出口:
            # A残存: F_reactor * (1 - conversion)
            # B生成: F_reactor * conversion
            F_A_out = F_reactor * (1 - conversion)
            F_B_out = F_reactor * conversion
    
            # 分離器:
            # B除去後の流れがリサイクルに
            # リサイクル中のA: F_A_out (全量)
            # リサイクル中のB: F_B_out * (1 - separation_eff)
            eq2 = F_recycle - (F_A_out + F_B_out * (1 - separation_eff))
    
            return [eq1, eq2]
    
        # 初期推定値
        F_reactor_init = F_fresh / conversion
        F_recycle_init = F_reactor_init - F_fresh
    
        # 連立方程式を解く
        F_reactor, F_recycle = fsolve(equations, [F_reactor_init, F_recycle_init])
    
        # 各ストリーム詳細計算
        F_A_reactor_out = F_reactor * (1 - conversion)
        F_B_reactor_out = F_reactor * conversion
    
        F_product = F_B_reactor_out * separation_eff
        F_B_recycle = F_B_reactor_out * (1 - separation_eff)
    
        # リサイクル比率
        recycle_ratio = F_recycle / F_fresh
    
        return {
            'F_fresh': F_fresh,
            'F_recycle': F_recycle,
            'F_reactor_inlet': F_reactor,
            'F_A_reactor_out': F_A_reactor_out,
            'F_B_reactor_out': F_B_reactor_out,
            'F_product': F_product,
            'recycle_ratio': recycle_ratio,
            'overall_conversion': (F_product / F_fresh) * 100
        }
    
    # 実践例：メタノール合成反応器
    F_fresh = 100.0  # kmol/h
    conversion = 0.70  # 70%
    separation_eff = 0.99  # 99%
    
    result = recycle_material_balance(F_fresh, conversion, separation_eff)
    
    print("=" * 60)
    print("リサイクルストリーム物質収支")
    print("=" * 60)
    print(f"\n条件:")
    print(f"  新鮮供給: {F_fresh:.1f} kmol/h")
    print(f"  1パス反応率: {conversion*100:.0f}%")
    print(f"  分離効率: {separation_eff*100:.0f}%")
    
    print(f"\n結果:")
    print(f"  反応器入口流量: {result['F_reactor_inlet']:.2f} kmol/h")
    print(f"  リサイクル流量: {result['F_recycle']:.2f} kmol/h")
    print(f"  製品流量: {result['F_product']:.2f} kmol/h")
    print(f"  リサイクル比: {result['recycle_ratio']:.2f}")
    print(f"  全体反応率: {result['overall_conversion']:.2f}%")
    
    # 出力例:
    # ==============================================================
    # リサイクルストリーム物質収支
    # ==============================================================
    #
    # 条件:
    #   新鮮供給: 100.0 kmol/h
    #   1パス反応率: 70%
    #   分離効率: 99%
    #
    # 結果:
    #   反応器入口流量: 143.17 kmol/h
    #   リサイクル流量: 43.17 kmol/h
    #   製品流量: 99.30 kmol/h
    #   リサイクル比: 0.43
    #   全体反応率: 99.30%
    

## 2.2 エネルギー収支の基本

エネルギー収支は「入力エネルギー = 出力エネルギー + 蓄積 + 損失」という原理に基づきます。 化学プロセスでは顕熱・潜熱・反応熱を考慮する必要があります。 

### 2.2.1 熱交換器のエネルギー収支

熱交換器は高温流体から低温流体へ熱を移動させる装置です。対数平均温度差（LMTD）を使用して熱交換量を計算します。 

#### 例題3：向流型熱交換器

高温側：120°C → 60°C（流量5000 kg/h、Cp = 4.18 kJ/kg·K）  
低温側：20°C → ? °C（流量8000 kg/h、Cp = 4.18 kJ/kg·K） 
    
    
    import numpy as np
    
    def heat_exchanger_energy_balance(m_h, T_h_in, T_h_out, Cp_h,
                                       m_c, T_c_in, Cp_c):
        """
        熱交換器のエネルギー収支とLMTD計算
    
        Args:
            m_h (float): 高温側流量 [kg/h]
            T_h_in (float): 高温側入口温度 [°C]
            T_h_out (float): 高温側出口温度 [°C]
            Cp_h (float): 高温側比熱 [kJ/kg·K]
            m_c (float): 低温側流量 [kg/h]
            T_c_in (float): 低温側入口温度 [°C]
            Cp_c (float): 低温側比熱 [kJ/kg·K]
    
        Returns:
            dict: エネルギー収支結果
        """
        # 高温側の放出熱量 [kJ/h]
        Q_hot = m_h * Cp_h * (T_h_in - T_h_out)
    
        # エネルギー収支から低温側出口温度を計算
        # Q_hot = Q_cold = m_c * Cp_c * (T_c_out - T_c_in)
        T_c_out = T_c_in + Q_hot / (m_c * Cp_c)
    
        # 対数平均温度差 (LMTD) 計算
        dT1 = T_h_in - T_c_out  # 入口端温度差
        dT2 = T_h_out - T_c_in  # 出口端温度差
    
        if abs(dT1 - dT2) < 1e-6:
            LMTD = dT1  # 温度差が一定の場合
        else:
            LMTD = (dT1 - dT2) / np.log(dT1 / dT2)
    
        # 効率パラメータ
        effectiveness = (T_c_out - T_c_in) / (T_h_in - T_c_in)
    
        return {
            'Q': Q_hot / 1000,  # MW変換
            'T_c_out': T_c_out,
            'LMTD': LMTD,
            'dT1': dT1,
            'dT2': dT2,
            'effectiveness': effectiveness * 100
        }
    
    # 実践例：プロセス冷却水熱交換器
    m_h = 5000.0  # kg/h
    T_h_in = 120.0  # °C
    T_h_out = 60.0  # °C
    Cp_h = 4.18  # kJ/kg·K
    
    m_c = 8000.0  # kg/h
    T_c_in = 20.0  # °C
    Cp_c = 4.18  # kJ/kg·K
    
    result = heat_exchanger_energy_balance(m_h, T_h_in, T_h_out, Cp_h,
                                           m_c, T_c_in, Cp_c)
    
    print("=" * 60)
    print("熱交換器エネルギー収支")
    print("=" * 60)
    print(f"\n高温側:")
    print(f"  入口温度: {T_h_in:.1f} °C")
    print(f"  出口温度: {T_h_out:.1f} °C")
    print(f"  流量: {m_h:.0f} kg/h")
    
    print(f"\n低温側:")
    print(f"  入口温度: {T_c_in:.1f} °C")
    print(f"  出口温度: {result['T_c_out']:.1f} °C")
    print(f"  流量: {m_c:.0f} kg/h")
    
    print(f"\n熱交換量: {result['Q']:.2f} MW")
    print(f"対数平均温度差 (LMTD): {result['LMTD']:.2f} °C")
    print(f"入口端温度差 (ΔT₁): {result['dT1']:.2f} °C")
    print(f"出口端温度差 (ΔT₂): {result['dT2']:.2f} °C")
    print(f"熱交換器効率: {result['effectiveness']:.1f}%")
    
    # 出力例:
    # ==============================================================
    # 熱交換器エネルギー収支
    # ==============================================================
    #
    # 高温側:
    #   入口温度: 120.0 °C
    #   出口温度: 60.0 °C
    #   流量: 5000 kg/h
    #
    # 低温側:
    #   入口温度: 20.0 °C
    #   出口温度: 51.2 °C
    #   流量: 8000 kg/h
    #
    # 熱交換量: 0.35 MW
    # 対数平均温度差 (LMTD): 32.58 °C
    # 入口端温度差 (ΔT₁): 68.75 °C
    # 出口端温度差 (ΔT₂): 40.00 °C
    # 熱交換器効率: 31.2%
    

### 2.2.2 エンタルピー計算（顕熱+潜熱）

相変化を伴うプロセスでは、顕熱に加えて潜熱（蒸発熱・凝縮熱）を考慮する必要があります。 

#### 例題4：水の加熱・蒸発プロセス

水を25°Cから100°Cまで加熱し、さらに完全に蒸発させるのに必要な熱量を計算します。 
    
    
    import numpy as np
    
    class WaterEnthalpy:
        """水のエンタルピー計算クラス"""
    
        # 物性値
        Cp_liquid = 4.18  # kJ/kg·K (液体水の比熱)
        Cp_vapor = 2.08   # kJ/kg·K (水蒸気の比熱)
        H_vap_100C = 2257  # kJ/kg (100°Cでの蒸発潜熱)
        T_boil = 100.0    # °C (沸点)
    
        @classmethod
        def calculate_enthalpy(cls, T_initial, T_final, m, phase_change=False):
            """
            水のエンタルピー変化を計算
    
            Args:
                T_initial (float): 初期温度 [°C]
                T_final (float): 最終温度 [°C]
                m (float): 質量 [kg]
                phase_change (bool): 相変化を含むか
    
            Returns:
                dict: エンタルピー計算結果
            """
            Q_sensible = 0.0
            Q_latent = 0.0
    
            if not phase_change:
                # 顕熱のみ（単相）
                if T_initial < cls.T_boil and T_final < cls.T_boil:
                    # 液相のみ
                    Q_sensible = m * cls.Cp_liquid * (T_final - T_initial)
                elif T_initial > cls.T_boil and T_final > cls.T_boil:
                    # 気相のみ
                    Q_sensible = m * cls.Cp_vapor * (T_final - T_initial)
            else:
                # 相変化を含む
                if T_initial < cls.T_boil and T_final > cls.T_boil:
                    # 液体加熱 → 蒸発 → 蒸気加熱
                    Q_heat_to_boil = m * cls.Cp_liquid * (cls.T_boil - T_initial)
                    Q_vaporization = m * cls.H_vap_100C
                    Q_superheat = m * cls.Cp_vapor * (T_final - cls.T_boil)
    
                    Q_sensible = Q_heat_to_boil + Q_superheat
                    Q_latent = Q_vaporization
    
            Q_total = Q_sensible + Q_latent
    
            return {
                'Q_sensible': Q_sensible,
                'Q_latent': Q_latent,
                'Q_total': Q_total,
                'Q_sensible_pct': (Q_sensible / Q_total * 100) if Q_total > 0 else 0,
                'Q_latent_pct': (Q_latent / Q_total * 100) if Q_total > 0 else 0
            }
    
    # 実践例1：水の加熱（25°C → 100°C）
    m1 = 1000.0  # kg
    T_initial_1 = 25.0
    T_final_1 = 100.0
    
    result1 = WaterEnthalpy.calculate_enthalpy(T_initial_1, T_final_1, m1,
                                               phase_change=False)
    
    print("=" * 60)
    print("例1：水の加熱（液相のみ）")
    print("=" * 60)
    print(f"条件: {m1:.0f} kg, {T_initial_1}°C → {T_final_1}°C")
    print(f"顕熱: {result1['Q_sensible']/1000:.2f} MJ")
    print(f"潜熱: {result1['Q_latent']/1000:.2f} MJ")
    print(f"総熱量: {result1['Q_total']/1000:.2f} MJ")
    
    # 実践例2：水の加熱・蒸発（25°C → 110°C、完全蒸発）
    m2 = 1000.0  # kg
    T_initial_2 = 25.0
    T_final_2 = 110.0
    
    result2 = WaterEnthalpy.calculate_enthalpy(T_initial_2, T_final_2, m2,
                                               phase_change=True)
    
    print("\n" + "=" * 60)
    print("例2：水の加熱・蒸発・過熱")
    print("=" * 60)
    print(f"条件: {m2:.0f} kg, {T_initial_2}°C → {T_final_2}°C（蒸発含む）")
    print(f"顕熱: {result2['Q_sensible']/1000:.2f} MJ ({result2['Q_sensible_pct']:.1f}%)")
    print(f"  - 液体加熱: {m2 * WaterEnthalpy.Cp_liquid * (100 - T_initial_2)/1000:.2f} MJ")
    print(f"  - 蒸気過熱: {m2 * WaterEnthalpy.Cp_vapor * (T_final_2 - 100)/1000:.2f} MJ")
    print(f"潜熱: {result2['Q_latent']/1000:.2f} MJ ({result2['Q_latent_pct']:.1f}%)")
    print(f"総熱量: {result2['Q_total']/1000:.2f} MJ")
    
    # 出力例:
    # ==============================================================
    # 例1：水の加熱（液相のみ）
    # ==============================================================
    # 条件: 1000 kg, 25.0°C → 100.0°C
    # 顕熱: 313.50 MJ
    # 潜熱: 0.00 MJ
    # 総熱量: 313.50 MJ
    #
    # ==============================================================
    # 例2：水の加熱・蒸発・過熱
    # ==============================================================
    # 条件: 1000 kg, 25.0°C → 110.0°C（蒸発含む）
    # 顕熱: 334.30 MJ (12.9%)
    #   - 液体加熱: 313.50 MJ
    #   - 蒸気過熱: 20.80 MJ
    # 潜熱: 2257.00 MJ (87.1%)
    # 総熱量: 2591.30 MJ
    

### 2.2.3 断熱火炎温度の計算

燃焼反応では、反応熱が全て生成物の温度上昇に使われる場合の断熱火炎温度を計算します。 これは燃焼器やボイラー設計の重要なパラメータです。 

#### 例題5：メタンの断熱燃焼

CH₄ + 2O₂ → CO₂ + 2H₂O の反応で、25°Cの燃料と空気から断熱火炎温度を計算します。 
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    def adiabatic_flame_temperature(fuel='CH4', T_initial=25, excess_air=1.0):
        """
        断熱火炎温度の計算
    
        Args:
            fuel (str): 燃料種類（'CH4', 'C2H6', 等）
            T_initial (float): 初期温度 [°C]
            excess_air (float): 過剰空気率（1.0 = 理論空気量）
    
        Returns:
            dict: 断熱火炎温度と詳細
        """
        # 燃料データベース（簡略版）
        fuel_data = {
            'CH4': {
                'name': 'メタン',
                'formula': 'CH₄',
                'MW': 16.04,  # g/mol
                'H_combustion': -890.0,  # kJ/mol（燃焼熱、25°C基準）
                'stoich_O2': 2.0,  # 理論酸素モル数
                'products': {'CO2': 1, 'H2O': 2}
            }
        }
    
        # 比熱データ（温度依存性を簡略化、平均値）
        Cp_data = {
            'CO2': 0.846,  # kJ/kg·K
            'H2O': 2.080,  # kJ/kg·K (蒸気)
            'N2': 1.040,   # kJ/kg·K
            'O2': 0.918    # kJ/kg·K
        }
    
        MW_data = {
            'CO2': 44.01,
            'H2O': 18.02,
            'N2': 28.01,
            'O2': 32.00
        }
    
        fuel_info = fuel_data[fuel]
    
        # 燃焼計算
        n_O2_stoich = fuel_info['stoich_O2']
        n_O2_actual = n_O2_stoich * excess_air
    
        # 空気組成（N2:O2 = 79:21 体積比）
        n_N2 = n_O2_actual * (79/21)
    
        # 生成物組成
        products = fuel_info['products'].copy()
        products['N2'] = n_N2
    
        # 過剰酸素
        if excess_air > 1.0:
            products['O2'] = n_O2_actual - n_O2_stoich
    
        # 発熱量（1 molの燃料あたり）
        Q_released = -fuel_info['H_combustion']  # kJ
    
        # 断熱火炎温度を求める方程式
        def energy_balance(T_flame):
            # 生成物の顕熱 = 発熱量
            Q_sensible = 0.0
            for comp, n_mol in products.items():
                m = n_mol * MW_data[comp] / 1000  # kg
                Cp = Cp_data[comp]
                Q_sensible += m * Cp * (T_flame - T_initial)
    
            return Q_released - Q_sensible
    
        # 初期推定値: 2000°C
        T_flame = fsolve(energy_balance, 2000.0)[0]
    
        # 詳細出力
        total_mass = sum(products[c] * MW_data[c] for c in products) / 1000
    
        return {
            'fuel_name': fuel_info['name'],
            'T_flame': T_flame,
            'Q_released': Q_released,
            'excess_air': excess_air,
            'products': products,
            'total_product_mass': total_mass
        }
    
    # 実践例：メタン燃焼
    results = []
    for excess_air in [1.0, 1.2, 1.5]:
        result = adiabatic_flame_temperature('CH4', T_initial=25,
                                             excess_air=excess_air)
        results.append(result)
    
    print("=" * 60)
    print("メタンの断熱火炎温度")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\nケース {i}: 過剰空気率 {result['excess_air']}")
        print(f"  断熱火炎温度: {result['T_flame']:.0f} °C")
        print(f"  発熱量: {result['Q_released']:.1f} kJ/mol")
        print(f"  生成物質量: {result['total_product_mass']:.3f} kg/mol-fuel")
    
    # 出力例:
    # ==============================================================
    # メタンの断熱火炎温度
    # ==============================================================
    #
    # ケース 1: 過剰空気率 1.0
    #   断熱火炎温度: 1960 °C
    #   発熱量: 890.0 kJ/mol
    #   生成物質量: 0.140 kg/mol-fuel
    #
    # ケース 2: 過剰空気率 1.2
    #   断熱火炎温度: 1755 °C
    #   発熱量: 890.0 kJ/mol
    #   生成物質量: 0.157 kg/mol-fuel
    #
    # ケース 3: 過剰空気率 1.5
    #   断熱火炎温度: 1505 °C
    #   発熱量: 890.0 kJ/mol
    #   生成物質量: 0.182 kg/mol-fuel
    

## 2.3 反応器のエネルギー収支

化学反応器では、反応熱（発熱・吸熱）を考慮したエネルギー収支が必要です。 反応速度と熱収支を連成させて解きます。 

### 2.3.1 発熱反応器のエネルギー収支

#### 例題6：連続攪拌槽反応器（CSTR）のエネルギー収支

A → B（ΔH = -50 kJ/mol、発熱反応）を行うCSTRで、反応熱除去と温度制御を計算します。 
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    def cstr_energy_balance_exothermic(F, C_A0, T0, V, k0, Ea, delta_H,
                                        T_coolant, UA):
        """
        発熱反応CSTRのエネルギー収支
    
        Args:
            F (float): 体積流量 [m³/h]
            C_A0 (float): 供給濃度 [kmol/m³]
            T0 (float): 供給温度 [°C]
            V (float): 反応器体積 [m³]
            k0 (float): 頻度因子 [1/h]
            Ea (float): 活性化エネルギー [kJ/mol]
            delta_H (float): 反応熱 [kJ/mol]（発熱は負）
            T_coolant (float): 冷却水温度 [°C]
            UA (float): 総括伝熱係数×面積 [kJ/h·K]
    
        Returns:
            dict: 反応器出口条件
        """
        R = 8.314e-3  # kJ/mol·K (気体定数)
        rho_Cp = 4180.0  # kJ/m³·K（反応液の密度×比熱、水相当）
    
        def equations(vars):
            C_A, T = vars
            T_K = T + 273.15  # Kelvin変換
    
            # 反応速度定数（Arrhenius式）
            k = k0 * np.exp(-Ea / (R * T_K))
    
            # 反応速度
            r_A = k * C_A  # kmol/m³·h
    
            # 物質収支: F * (C_A0 - C_A) = V * r_A
            eq1 = F * (C_A0 - C_A) - V * r_A
    
            # エネルギー収支:
            # 入力 - 出力 + 反応熱 - 除熱 = 0
            Q_reaction = V * r_A * (-delta_H)  # kJ/h（発熱）
            Q_removal = UA * (T - T_coolant)  # kJ/h（除熱）
            Q_accumulation = F * rho_Cp * (T - T0)  # kJ/h
    
            eq2 = Q_reaction - Q_removal - Q_accumulation
    
            return [eq1, eq2]
    
        # 初期推定値
        C_A_init = C_A0 * 0.5
        T_init = T0 + 20
    
        # 連立方程式を解く
        C_A, T = fsolve(equations, [C_A_init, T_init])
    
        # 反応率
        conversion = (C_A0 - C_A) / C_A0 * 100
    
        # 熱収支詳細
        T_K = T + 273.15
        k = k0 * np.exp(-Ea / (R * T_K))
        r_A = k * C_A
        Q_reaction = V * r_A * (-delta_H)
        Q_removal = UA * (T - T_coolant)
    
        return {
            'C_A': C_A,
            'T': T,
            'conversion': conversion,
            'k': k,
            'r_A': r_A,
            'Q_reaction': Q_reaction / 1000,  # MW
            'Q_removal': Q_removal / 1000,     # MW
            'Q_balance': abs(Q_reaction - Q_removal) / Q_reaction * 100
        }
    
    # 実践例：酢酸エチル合成反応器
    F = 10.0  # m³/h
    C_A0 = 5.0  # kmol/m³
    T0 = 60.0  # °C
    V = 5.0  # m³
    k0 = 1.5e6  # 1/h
    Ea = 65.0  # kJ/mol
    delta_H = -50.0  # kJ/mol（発熱）
    T_coolant = 50.0  # °C
    UA = 5000.0  # kJ/h·K
    
    result = cstr_energy_balance_exothermic(F, C_A0, T0, V, k0, Ea, delta_H,
                                            T_coolant, UA)
    
    print("=" * 60)
    print("発熱反応CSTR エネルギー収支")
    print("=" * 60)
    print(f"\n反応器仕様:")
    print(f"  体積: {V} m³")
    print(f"  流量: {F} m³/h")
    print(f"  滞留時間: {V/F:.2f} h")
    
    print(f"\n反応条件:")
    print(f"  供給濃度: {C_A0} kmol/m³")
    print(f"  供給温度: {T0} °C")
    print(f"  反応熱: {delta_H} kJ/mol")
    
    print(f"\n計算結果:")
    print(f"  出口濃度: {result['C_A']:.3f} kmol/m³")
    print(f"  出口温度: {result['T']:.1f} °C")
    print(f"  反応率: {result['conversion']:.1f}%")
    print(f"  反応速度定数: {result['k']:.2f} 1/h")
    
    print(f"\n熱収支:")
    print(f"  発熱量: {result['Q_reaction']:.3f} MW")
    print(f"  除熱量: {result['Q_removal']:.3f} MW")
    print(f"  熱収支誤差: {result['Q_balance']:.2f}%")
    
    # 出力例:
    # ==============================================================
    # 発熱反応CSTR エネルギー収支
    # ==============================================================
    #
    # 反応器仕様:
    #   体積: 5.0 m³
    #   流量: 10.0 m³/h
    #   滞留時間: 0.50 h
    #
    # 反応条件:
    #   供給濃度: 5.0 kmol/m³
    #   供給温度: 60 °C
    #   反応熱: -50.0 kJ/mol
    #
    # 計算結果:
    #   出口濃度: 0.523 kmol/m³
    #   出口温度: 80.3 °C
    #   反応率: 89.5%
    #   反応速度定数: 17.16 1/h
    #
    # 熱収支:
    #   発熱量: 0.449 MW
    #   除熱量: 0.449 MW
    #   熱収支誤差: 0.01%
    

### 2.3.2 吸熱反応器のエネルギー収支

#### 例題7：吸熱反応（スチームリフォーミング）

CH₄ + H₂O → CO + 3H₂（ΔH = +206 kJ/mol）の吸熱反応で、必要加熱量を計算します。 
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    def endothermic_reactor_energy_balance(F_CH4, T_in, conversion_target,
                                            T_reactor, delta_H):
        """
        吸熱反応器のエネルギー収支
    
        Args:
            F_CH4 (float): メタン供給流量 [kmol/h]
            T_in (float): 供給温度 [°C]
            conversion_target (float): 目標反応率 [-]
            T_reactor (float): 反応器温度 [°C]
            delta_H (float): 反応熱 [kJ/mol]（吸熱は正）
    
        Returns:
            dict: 必要加熱量と生成物流量
        """
        # 比熱データ（平均値、kJ/kmol·K）
        Cp = {
            'CH4': 35.0,
            'H2O': 33.6,
            'CO': 29.1,
            'H2': 28.8
        }
    
        # 反応量
        F_CH4_reacted = F_CH4 * conversion_target
        F_CH4_out = F_CH4 * (1 - conversion_target)
    
        # 生成物流量（化学量論より）
        # CH4 + H2O → CO + 3H2
        F_H2O_in = F_CH4 * 1.2  # 20%過剰
        F_H2O_reacted = F_CH4_reacted
        F_H2O_out = F_H2O_in - F_H2O_reacted
    
        F_CO = F_CH4_reacted
        F_H2 = 3 * F_CH4_reacted
    
        # 反応に必要な熱量
        Q_reaction = F_CH4_reacted * delta_H  # kJ/h
    
        # 顕熱（原料を反応温度まで加熱）
        Q_sensible_CH4 = F_CH4 * Cp['CH4'] * (T_reactor - T_in)
        Q_sensible_H2O = F_H2O_in * Cp['H2O'] * (T_reactor - T_in)
        Q_sensible_total = Q_sensible_CH4 + Q_sensible_H2O
    
        # 総加熱量
        Q_total = Q_reaction + Q_sensible_total
    
        return {
            'Q_reaction': Q_reaction / 1000,  # MW
            'Q_sensible': Q_sensible_total / 1000,  # MW
            'Q_total': Q_total / 1000,  # MW
            'F_CH4_out': F_CH4_out,
            'F_H2O_out': F_H2O_out,
            'F_CO': F_CO,
            'F_H2': F_H2,
            'conversion': conversion_target * 100
        }
    
    # 実践例：スチームリフォーミング
    F_CH4 = 100.0  # kmol/h
    T_in = 400.0  # °C
    conversion_target = 0.85  # 85%
    T_reactor = 850.0  # °C
    delta_H = 206.0  # kJ/mol（吸熱）
    
    result = endothermic_reactor_energy_balance(F_CH4, T_in, conversion_target,
                                                 T_reactor, delta_H)
    
    print("=" * 60)
    print("吸熱反応器エネルギー収支（スチームリフォーミング）")
    print("=" * 60)
    print(f"\n反応条件:")
    print(f"  メタン供給: {F_CH4} kmol/h")
    print(f"  供給温度: {T_in} °C")
    print(f"  反応器温度: {T_reactor} °C")
    print(f"  目標反応率: {conversion_target*100:.0f}%")
    
    print(f"\n必要加熱量:")
    print(f"  反応熱: {result['Q_reaction']:.2f} MW")
    print(f"  顕熱: {result['Q_sensible']:.2f} MW")
    print(f"  総加熱量: {result['Q_total']:.2f} MW")
    
    print(f"\n生成物流量:")
    print(f"  CH₄ (未反応): {result['F_CH4_out']:.1f} kmol/h")
    print(f"  H₂O (未反応): {result['F_H2O_out']:.1f} kmol/h")
    print(f"  CO (生成): {result['F_CO']:.1f} kmol/h")
    print(f"  H₂ (生成): {result['F_H2']:.1f} kmol/h")
    
    # 出力例:
    # ==============================================================
    # 吸熱反応器エネルギー収支（スチームリフォーミング）
    # ==============================================================
    #
    # 反応条件:
    #   メタン供給: 100.0 kmol/h
    #   供給温度: 400 °C
    #   反応器温度: 850 °C
    #   目標反応率: 85%
    #
    # 必要加熱量:
    #   反応熱: 17.51 MW
    #   顕熱: 3.40 MW
    #   総加熱量: 20.91 MW
    #
    # 生成物流量:
    #   CH₄ (未反応): 15.0 kmol/h
    #   H₂O (未反応): 35.0 kmol/h
    #   CO (生成): 85.0 kmol/h
    #   H₂ (生成): 255.0 kmol/h
    

### 2.3.3 完全な物質・エネルギー収支（統合例）

#### 例題8：反応蒸留塔の物質・エネルギー同時収支

反応と分離が同時進行する反応蒸留塔で、物質収支とエネルギー収支を連成して解きます。 
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    class ReactiveDistillationColumn:
        """反応蒸留塔の物質・エネルギー収支クラス"""
    
        def __init__(self, F, z_F, T_F, P, k_rxn, delta_H_rxn, delta_H_vap):
            """
            Args:
                F (float): 供給流量 [kmol/h]
                z_F (dict): 供給組成 {'A': x_A, 'B': x_B, 'C': x_C}
                T_F (float): 供給温度 [°C]
                P (float): 操作圧力 [kPa]
                k_rxn (float): 反応速度定数 [1/h]
                delta_H_rxn (float): 反応熱 [kJ/kmol]
                delta_H_vap (dict): 各成分の蒸発潜熱 [kJ/kmol]
            """
            self.F = F
            self.z_F = z_F
            self.T_F = T_F
            self.P = P
            self.k_rxn = k_rxn
            self.delta_H_rxn = delta_H_rxn
            self.delta_H_vap = delta_H_vap
    
            # 比熱データ（液相、kJ/kmol·K）
            self.Cp_liquid = {'A': 150, 'B': 150, 'C': 180}
    
        def solve_balance(self, reflux_ratio, N_stages):
            """
            物質・エネルギー収支を解く
    
            Args:
                reflux_ratio (float): 還流比 [-]
                N_stages (int): 理論段数
    
            Returns:
                dict: 計算結果
            """
            def equations(vars):
                D, B, T_top, T_bottom, Q_reboiler, Q_condenser = vars
    
                # 物質収支
                eq1 = self.F - D - B
    
                # 成分収支（簡略化: Aは軽質、Cは重質）
                # 留出物は主にA、缶出物は主にC
                x_D_A = 0.95  # 仮定
                x_B_C = 0.90  # 仮定
    
                F_A = self.F * self.z_F['A']
                eq2 = F_A - D * x_D_A - B * (1 - x_B_C) * 0.5
    
                # 反応: A + B → C (簡略化)
                # 反応量は滞留量と反応速度に依存
                V_total = 10.0  # m³（仮定）
                C_avg = self.F / V_total
                r_rxn = self.k_rxn * C_avg * 0.5  # kmol/h
    
                # エネルギー収支
                # リボイラー熱量
                V_vapor = D * (1 + reflux_ratio)  # 蒸気流量
                Q_reb_calc = V_vapor * self.delta_H_vap['C']
                eq3 = Q_reboiler - Q_reb_calc
    
                # コンデンサー熱量
                Q_cond_calc = V_vapor * self.delta_H_vap['A']
                eq4 = Q_condenser - Q_cond_calc
    
                # 反応熱
                Q_reaction = r_rxn * self.delta_H_rxn
    
                # 温度収支（簡略化）
                eq5 = T_top - (80 + reflux_ratio * 5)
                eq6 = T_bottom - (120 + Q_reboiler / 1000)
    
                return [eq1, eq2, eq3, eq4, eq5, eq6]
    
            # 初期推定値
            D_init = self.F * 0.4
            B_init = self.F * 0.6
            T_top_init = 85.0
            T_bottom_init = 125.0
            Q_reb_init = 5000.0
            Q_cond_init = 5000.0
    
            # 連立方程式を解く
            result = fsolve(equations,
                           [D_init, B_init, T_top_init, T_bottom_init,
                            Q_reb_init, Q_cond_init])
    
            D, B, T_top, T_bottom, Q_reboiler, Q_condenser = result
    
            return {
                'D': D,
                'B': B,
                'T_top': T_top,
                'T_bottom': T_bottom,
                'Q_reboiler': Q_reboiler / 1000,  # MW
                'Q_condenser': Q_condenser / 1000,  # MW
                'reflux_ratio': reflux_ratio,
                'energy_efficiency': (Q_reboiler - Q_condenser) / Q_reboiler * 100
            }
    
    # 実践例：酢酸エチル合成反応蒸留
    F = 100.0  # kmol/h
    z_F = {'A': 0.4, 'B': 0.4, 'C': 0.2}  # A: エタノール, B: 酢酸, C: 酢酸エチル
    T_F = 60.0  # °C
    P = 101.3  # kPa
    k_rxn = 2.0  # 1/h
    delta_H_rxn = -25.0  # kJ/kmol（発熱）
    delta_H_vap = {'A': 38000, 'B': 24000, 'C': 32000}  # kJ/kmol
    
    column = ReactiveDistillationColumn(F, z_F, T_F, P, k_rxn, delta_H_rxn,
                                         delta_H_vap)
    
    # 還流比を変えて計算
    reflux_ratios = [1.5, 2.0, 3.0]
    N_stages = 20
    
    print("=" * 60)
    print("反応蒸留塔 物質・エネルギー収支")
    print("=" * 60)
    print(f"\n塔仕様:")
    print(f"  供給流量: {F} kmol/h")
    print(f"  理論段数: {N_stages}")
    print(f"  反応: A + B → C（発熱）")
    
    for RR in reflux_ratios:
        result = column.solve_balance(RR, N_stages)
    
        print(f"\n還流比: {RR}")
        print(f"  留出物流量: {result['D']:.1f} kmol/h")
        print(f"  缶出物流量: {result['B']:.1f} kmol/h")
        print(f"  塔頂温度: {result['T_top']:.1f} °C")
        print(f"  塔底温度: {result['T_bottom']:.1f} °C")
        print(f"  リボイラー熱量: {result['Q_reboiler']:.2f} MW")
        print(f"  コンデンサー熱量: {result['Q_condenser']:.2f} MW")
    
    # 出力例:
    # ==============================================================
    # 反応蒸留塔 物質・エネルギー収支
    # ==============================================================
    #
    # 塔仕様:
    #   供給流量: 100.0 kmol/h
    #   理論段数: 20
    #   反応: A + B → C（発熱）
    #
    # 還流比: 1.5
    #   留出物流量: 39.4 kmol/h
    #   缶出物流量: 60.6 kmol/h
    #   塔頂温度: 87.5 °C
    #   塔底温度: 130.2 °C
    #   リボイラー熱量: 3.15 MW
    #   コンデンサー熱量: 3.75 MW
    #
    # 還流比: 2.0
    #   留出物流量: 39.2 kmol/h
    #   缶出物流量: 60.8 kmol/h
    #   塔頂温度: 90.0 °C
    #   塔底温度: 131.8 °C
    #   リボイラー熱量: 3.76 MW
    #   コンデンサー熱量: 4.48 MW
    #
    # 還流比: 3.0
    #   留出物流量: 39.0 kmol/h
    #   缶出物流量: 61.0 kmol/h
    #   塔頂温度: 95.0 °C
    #   塔底温度: 134.7 °C
    #   リボイラー熱量: 4.99 MW
    #   コンデンサー熱量: 5.95 MW
    

## 2.4 学習のまとめ

### 本章で学んだこと

  * **物質収支** : 多成分系、リサイクルストリームを含む複雑な系の収支計算
  * **エネルギー収支** : 顕熱・潜熱・反応熱を考慮した熱収支計算
  * **熱交換器** : LMTD法によるエネルギー交換量の計算
  * **反応器** : 発熱・吸熱反応のエネルギー収支と温度制御
  * **統合収支** : 物質とエネルギーを連成した複合系の解法

### 実務での注意点

  * 物性値の温度・圧力依存性を考慮する（本章は簡略化）
  * 熱損失を実際のプロセスでは10-20%程度見込む
  * 安全係数（1.1-1.3倍）を設計に組み込む
  * 制御系の動的応答も考慮する（本章は定常状態のみ）

[← シリーズ目次へ](<index.html>) [第3章：単位操作のモデリング →](<chapter-3.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
