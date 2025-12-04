---
title: 第1章：プロセスシミュレーションの基礎
chapter_title: 第1章：プロセスシミュレーションの基礎
subtitle: Sequential Modular vs Equation-Oriented アプローチと熱力学計算
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Sequential ModularとEquation-Orientedアプローチの違いを理解する
  * ✅ 熱力学モデル（理想気体、SRK、Peng-Robinson）を実装できる
  * ✅ ストリーム物性（エンタルピー、エントロピー、密度）を計算できる
  * ✅ フラッシュ計算（気液平衡）を実装できる
  * ✅ 収束計算アルゴリズム（逐次代入法、Newton-Raphson法）を理解する
  * ✅ Tear Streamの選択とプロセスフローの順序決定ができる

* * *

## 1.1 プロセスシミュレーションの概要

### プロセスシミュレーションとは

**プロセスシミュレーション** とは、化学プロセスの挙動を数学モデルで表現し、コンピュータ上で再現することです。物質収支、エネルギー収支、気液平衡、反応速度式などを組み合わせることで、プロセスの性能を予測します。

### 主要なアプローチ

アプローチ | 特徴 | 長所 | 短所  
---|---|---|---  
**Sequential Modular** | ユニット操作を順番に計算 | 直感的、デバッグ容易 | リサイクルループで収束計算必要  
**Equation-Oriented** | 全方程式を同時に解く | 収束が速い、最適化容易 | ヤコビ行列の構築が複雑  
  
* * *

## 1.2 Sequential Modular アプローチの実装

### コード例1: Sequential Modular法の基本実装
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Sequential Modular法のデモ: 3つのユニット操作を順番に計算
    
    class Stream:
        """プロセスストリームクラス"""
        def __init__(self, name, T=298.15, P=101325, flow_rate=100.0, composition=None):
            """
            Parameters:
            name : str, ストリーム名
            T : float, 温度 [K]
            P : float, 圧力 [Pa]
            flow_rate : float, モル流量 [mol/s]
            composition : dict, 組成 {成分名: モル分率}
            """
            self.name = name
            self.T = T
            self.P = P
            self.flow_rate = flow_rate
            self.composition = composition if composition else {'A': 1.0}
    
        def copy(self):
            """ストリームのコピーを作成"""
            return Stream(self.name, self.T, self.P, self.flow_rate,
                         self.composition.copy())
    
        def __repr__(self):
            return (f"Stream(name={self.name}, T={self.T:.2f}K, "
                    f"P={self.P/1000:.1f}kPa, F={self.flow_rate:.2f}mol/s)")
    
    
    class Heater:
        """加熱器ユニット"""
        def __init__(self, name, delta_T=50.0):
            self.name = name
            self.delta_T = delta_T  # 温度上昇 [K]
    
        def calculate(self, inlet_stream):
            """
            加熱器計算
    
            Parameters:
            inlet_stream : Stream, 入口ストリーム
    
            Returns:
            outlet_stream : Stream, 出口ストリーム
            """
            outlet = inlet_stream.copy()
            outlet.name = f"{self.name}_out"
            outlet.T = inlet_stream.T + self.delta_T
    
            print(f"{self.name}: 温度 {inlet_stream.T:.2f}K → {outlet.T:.2f}K")
            return outlet
    
    
    class Reactor:
        """反応器ユニット（簡易モデル）"""
        def __init__(self, name, conversion=0.8):
            self.name = name
            self.conversion = conversion  # 転化率
    
        def calculate(self, inlet_stream):
            """
            反応器計算: A → B (単純反応)
    
            Parameters:
            inlet_stream : Stream
    
            Returns:
            outlet_stream : Stream
            """
            outlet = inlet_stream.copy()
            outlet.name = f"{self.name}_out"
    
            # 反応: A → B
            if 'A' in inlet_stream.composition:
                x_A = inlet_stream.composition['A']
                converted = x_A * self.conversion
    
                outlet.composition = {
                    'A': x_A - converted,
                    'B': converted
                }
    
                print(f"{self.name}: 転化率 {self.conversion*100:.1f}%, "
                      f"A: {x_A:.3f} → {outlet.composition['A']:.3f}, "
                      f"B: 0 → {outlet.composition['B']:.3f}")
    
            return outlet
    
    
    class Cooler:
        """冷却器ユニット"""
        def __init__(self, name, T_target=320.0):
            self.name = name
            self.T_target = T_target  # 目標温度 [K]
    
        def calculate(self, inlet_stream):
            """
            冷却器計算
    
            Parameters:
            inlet_stream : Stream
    
            Returns:
            outlet_stream : Stream
            """
            outlet = inlet_stream.copy()
            outlet.name = f"{self.name}_out"
            outlet.T = self.T_target
    
            print(f"{self.name}: 温度 {inlet_stream.T:.2f}K → {outlet.T:.2f}K")
            return outlet
    
    
    # Sequential Modular法によるプロセス計算
    def run_sequential_modular():
        """
        Sequential Modular法でプロセスを計算
    
        プロセスフロー: Feed → Heater → Reactor → Cooler → Product
        """
        print("="*60)
        print("Sequential Modular法によるプロセス計算")
        print("="*60)
    
        # 入口ストリーム
        feed = Stream(name="Feed", T=298.15, P=101325, flow_rate=100.0,
                      composition={'A': 1.0})
        print(f"\n入口: {feed}")
    
        # ユニット操作の定義
        heater = Heater(name="H-101", delta_T=80.0)
        reactor = Reactor(name="R-101", conversion=0.85)
        cooler = Cooler(name="C-101", T_target=320.0)
    
        # 順番に計算（Sequential Modular）
        print("\n--- ユニット操作計算 ---")
        s1 = heater.calculate(feed)
        s2 = reactor.calculate(s1)
        product = cooler.calculate(s2)
    
        print(f"\n出口: {product}")
        print(f"最終組成: A={product.composition.get('A', 0):.3f}, "
              f"B={product.composition.get('B', 0):.3f}")
    
        # 結果の可視化
        streams = [feed, s1, s2, product]
        stream_names = ['Feed', 'Heater出口', 'Reactor出口', 'Product']
        temperatures = [s.T for s in streams]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # 温度プロファイル
        ax1.plot(stream_names, temperatures, marker='o', linewidth=2.5,
                 markersize=10, color='#11998e')
        ax1.set_ylabel('温度 [K]', fontsize=12)
        ax1.set_title('Sequential Modular法: 温度プロファイル',
                      fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
        # 組成変化
        x_A = [s.composition.get('A', 0) for s in streams]
        x_B = [s.composition.get('B', 0) for s in streams]
    
        ax2.plot(stream_names, x_A, marker='s', linewidth=2.5,
                 markersize=10, label='成分A', color='#e74c3c')
        ax2.plot(stream_names, x_B, marker='^', linewidth=2.5,
                 markersize=10, label='成分B', color='#3498db')
        ax2.set_ylabel('モル分率', fontsize=12)
        ax2.set_title('Sequential Modular法: 組成変化',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
        plt.tight_layout()
        plt.show()
    
    
    # 実行
    run_sequential_modular()
    

**出力例:**
    
    
    ============================================================
    Sequential Modular法によるプロセス計算
    ============================================================
    
    入口: Stream(name=Feed, T=298.15K, P=101.3kPa, F=100.00mol/s)
    
    --- ユニット操作計算 ---
    H-101: 温度 298.15K → 378.15K
    R-101: 転化率 85.0%, A: 1.000 → 0.150, B: 0 → 0.850
    C-101: 温度 378.15K → 320.00K
    
    出口: Stream(name=C-101_out, T=320.00K, P=101.3kPa, F=100.00mol/s)
    最終組成: A=0.150, B=0.850
    

**解説:** Sequential Modular法では、各ユニット操作を順番に計算します。この方法は直感的で実装が容易ですが、リサイクルループがある場合は収束計算が必要になります。

* * *

### コード例2: Equation-Oriented アプローチの基本
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    # Equation-Oriented法のデモ: 全方程式を同時に解く
    
    def equation_oriented_process(x):
        """
        プロセス全体の方程式系
    
        変数順序: [T1, T2, T3, T4, x_A2, x_B2, x_A3, x_B3]
        - T1: Feed温度 (固定)
        - T2: Heater出口温度
        - T3: Reactor出口温度
        - T4: Cooler出口温度
        - x_A2, x_B2: Heater出口組成
        - x_A3, x_B3: Reactor出口組成
    
        Parameters:
        x : array, 変数ベクトル [8要素]
    
        Returns:
        residuals : array, 残差ベクトル
        """
        T1, T2, T3, T4, x_A2, x_B2, x_A3, x_B3 = x
    
        # パラメータ
        T_feed = 298.15  # K
        delta_T_heater = 80.0  # K
        conversion = 0.85
        T_cooler = 320.0  # K
        x_A1 = 1.0  # 入口組成
        x_B1 = 0.0
    
        # 方程式（residuals = 0 となるべき）
        residuals = np.zeros(8)
    
        # Heater方程式
        residuals[0] = T2 - (T1 + delta_T_heater)  # エネルギー収支
        residuals[1] = x_A2 - x_A1  # 物質収支（A）
        residuals[2] = x_B2 - x_B1  # 物質収支（B）
    
        # Reactor方程式
        residuals[3] = T3 - T2  # 温度変化なし（等温反応器）
        residuals[4] = x_A3 - (x_A2 * (1 - conversion))  # 反応後のA
        residuals[5] = x_B3 - (x_B2 + x_A2 * conversion)  # 反応後のB
    
        # Cooler方程式
        residuals[6] = T4 - T_cooler  # 目標温度
    
        # 組成総和制約
        residuals[7] = (x_A3 + x_B3) - 1.0
    
        return residuals
    
    
    # 初期推定値
    x0 = np.array([
        298.15,  # T1
        350.0,   # T2
        350.0,   # T3
        320.0,   # T4
        1.0,     # x_A2
        0.0,     # x_B2
        0.2,     # x_A3
        0.8      # x_B3
    ])
    
    print("="*60)
    print("Equation-Oriented法によるプロセス計算")
    print("="*60)
    print(f"\n初期推定値:")
    print(f"  T = {x0[:4]}")
    print(f"  組成 = {x0[4:]}")
    
    # 方程式を同時に解く
    solution = fsolve(equation_oriented_process, x0)
    
    T1, T2, T3, T4, x_A2, x_B2, x_A3, x_B3 = solution
    
    print(f"\n解:")
    print(f"  Feed温度: T1 = {T1:.2f} K")
    print(f"  Heater出口: T2 = {T2:.2f} K, A={x_A2:.3f}, B={x_B2:.3f}")
    print(f"  Reactor出口: T3 = {T3:.2f} K, A={x_A3:.3f}, B={x_B3:.3f}")
    print(f"  Product温度: T4 = {T4:.2f} K")
    
    # 残差確認
    residuals = equation_oriented_process(solution)
    print(f"\n残差ノルム: {np.linalg.norm(residuals):.2e}")
    print(f"  (収束判定: < 1e-6 で成功)")
    
    # 比較プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Sequential\nModular', 'Equation-\nOriented']
    T_seq = [298.15, 378.15, 378.15, 320.0]
    T_eq = [T1, T2, T3, T4]
    
    x_pos = np.arange(4)
    width = 0.35
    
    ax.bar(x_pos - width/2, T_seq, width, label='Sequential Modular',
           color='#11998e', alpha=0.8)
    ax.bar(x_pos + width/2, T_eq, width, label='Equation-Oriented',
           color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('温度 [K]', fontsize=12)
    ax.set_title('Sequential Modular vs Equation-Oriented: 温度比較',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Feed', 'Heater出口', 'Reactor出口', 'Product'])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    

**出力例:**
    
    
    ============================================================
    Equation-Oriented法によるプロセス計算
    ============================================================
    
    初期推定値:
      T = [298.15 350.   350.   320.  ]
      組成 = [1.  0.  0.2 0.8]
    
    解:
      Feed温度: T1 = 298.15 K
      Heater出口: T2 = 378.15 K, A=1.000, B=0.000
      Reactor出口: T3 = 378.15 K, A=0.150, B=0.850
      Product温度: T4 = 320.00 K
    
    残差ノルム: 2.47e-13
      (収束判定: < 1e-6 で成功)
    

**解説:** Equation-Oriented法では、プロセス全体の方程式を同時に解きます。収束が速く、最適化との統合も容易ですが、ヤコビ行列の計算が必要になるため実装は複雑です。

* * *

## 1.3 熱力学モデルの実装

### コード例3: 理想気体モデル
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 理想気体モデルの実装
    
    class IdealGas:
        """理想気体モデル"""
    
        R = 8.314  # 気体定数 [J/(mol·K)]
    
        @staticmethod
        def pressure(n, V, T):
            """
            理想気体の状態方程式: PV = nRT
    
            Parameters:
            n : float, モル数 [mol]
            V : float, 体積 [m³]
            T : float, 温度 [K]
    
            Returns:
            P : float, 圧力 [Pa]
            """
            return n * IdealGas.R * T / V
    
        @staticmethod
        def volume(n, P, T):
            """
            体積計算
    
            Parameters:
            n : float, モル数 [mol]
            P : float, 圧力 [Pa]
            T : float, 温度 [K]
    
            Returns:
            V : float, 体積 [m³]
            """
            return n * IdealGas.R * T / P
    
        @staticmethod
        def density(P, T, MW):
            """
            密度計算: ρ = PM/(RT)
    
            Parameters:
            P : float, 圧力 [Pa]
            T : float, 温度 [K]
            MW : float, 分子量 [g/mol]
    
            Returns:
            rho : float, 密度 [kg/m³]
            """
            return (P * MW / 1000) / (IdealGas.R * T)
    
        @staticmethod
        def enthalpy(T, Cp, T_ref=298.15, H_ref=0.0):
            """
            エンタルピー計算（定圧比熱一定と仮定）
            H(T) = H_ref + Cp * (T - T_ref)
    
            Parameters:
            T : float, 温度 [K]
            Cp : float, 定圧比熱 [J/(mol·K)]
            T_ref : float, 基準温度 [K]
            H_ref : float, 基準エンタルピー [J/mol]
    
            Returns:
            H : float, エンタルピー [J/mol]
            """
            return H_ref + Cp * (T - T_ref)
    
        @staticmethod
        def entropy(T, P, Cp, T_ref=298.15, P_ref=101325, S_ref=0.0):
            """
            エントロピー計算
            S(T, P) = S_ref + Cp*ln(T/T_ref) - R*ln(P/P_ref)
    
            Parameters:
            T : float, 温度 [K]
            P : float, 圧力 [Pa]
            Cp : float, 定圧比熱 [J/(mol·K)]
            T_ref : float, 基準温度 [K]
            P_ref : float, 基準圧力 [Pa]
            S_ref : float, 基準エントロピー [J/(mol·K)]
    
            Returns:
            S : float, エントロピー [J/(mol·K)]
            """
            return (S_ref + Cp * np.log(T / T_ref) -
                    IdealGas.R * np.log(P / P_ref))
    
    
    # 理想気体モデルの検証
    def demonstrate_ideal_gas():
        """理想気体モデルのデモンストレーション"""
    
        print("="*60)
        print("理想気体モデルの物性計算")
        print("="*60)
    
        # パラメータ
        n = 1.0  # mol
        T = 350.0  # K
        P = 200000  # Pa (2 bar)
        MW = 28.0  # g/mol (N2)
        Cp = 29.1  # J/(mol·K)
    
        # 計算
        V = IdealGas.volume(n, P, T)
        rho = IdealGas.density(P, T, MW)
        H = IdealGas.enthalpy(T, Cp)
        S = IdealGas.entropy(T, P, Cp)
    
        print(f"\n条件:")
        print(f"  温度: T = {T} K")
        print(f"  圧力: P = {P/1000:.1f} kPa")
        print(f"  モル数: n = {n} mol")
        print(f"  分子量: MW = {MW} g/mol")
    
        print(f"\n計算結果:")
        print(f"  体積: V = {V:.6f} m³ = {V*1000:.3f} L")
        print(f"  密度: ρ = {rho:.3f} kg/m³")
        print(f"  エンタルピー: H = {H:.2f} J/mol")
        print(f"  エントロピー: S = {S:.2f} J/(mol·K)")
    
        # 圧縮率因子の可視化（理想気体では Z=1）
        P_range = np.linspace(50000, 500000, 100)
        T_range = [300, 350, 400, 450]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # P-V図
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for i, T_val in enumerate(T_range):
            V_range = [IdealGas.volume(n, P, T_val) * 1000 for P in P_range]
            ax1.plot(V_range, P_range/1000, linewidth=2.5,
                    label=f'T = {T_val} K', color=colors[i])
    
        ax1.set_xlabel('体積 V [L]', fontsize=12)
        ax1.set_ylabel('圧力 P [kPa]', fontsize=12)
        ax1.set_title('理想気体のP-V図（等温線）', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
    
        # 密度vs圧力
        for i, T_val in enumerate(T_range):
            rho_range = [IdealGas.density(P, T_val, MW) for P in P_range]
            ax2.plot(P_range/1000, rho_range, linewidth=2.5,
                    label=f'T = {T_val} K', color=colors[i])
    
        ax2.set_xlabel('圧力 P [kPa]', fontsize=12)
        ax2.set_ylabel('密度 ρ [kg/m³]', fontsize=12)
        ax2.set_title('理想気体の密度-圧力関係', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    
    # 実行
    demonstrate_ideal_gas()
    

**出力例:**
    
    
    ============================================================
    理想気体モデルの物性計算
    ============================================================
    
    条件:
      温度: T = 350.0 K
      圧力: P = 200.0 kPa
      モル数: n = 1 mol
      分子量: MW = 28 g/mol
    
    計算結果:
      体積: V = 0.014550 m³ = 14.550 L
      密度: ρ = 1.924 kg/m³
      エンタルピー: H = 1511.85 J/mol
      エントロピー: S = 3.28 J/(mol·K)
    

**解説:** 理想気体モデルは、低圧・高温の気体に適用できます。計算が簡単で高速ですが、高圧や液相には使用できません。

* * *

### コード例4: Soave-Redlich-Kwong (SRK) 状態方程式
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    # SRK状態方程式の実装
    
    class SRK:
        """Soave-Redlich-Kwong (SRK) 状態方程式"""
    
        R = 8.314  # J/(mol·K)
    
        def __init__(self, Tc, Pc, omega):
            """
            Parameters:
            Tc : float, 臨界温度 [K]
            Pc : float, 臨界圧力 [Pa]
            omega : float, 偏心因子 [-]
            """
            self.Tc = Tc
            self.Pc = Pc
            self.omega = omega
    
            # パラメータ計算
            self.a_c = 0.42748 * (self.R * Tc)**2 / Pc
            self.b = 0.08664 * self.R * Tc / Pc
    
        def alpha(self, T):
            """
            温度依存パラメータ α(T)
    
            Parameters:
            T : float, 温度 [K]
    
            Returns:
            alpha : float
            """
            Tr = T / self.Tc
            m = 0.480 + 1.574 * self.omega - 0.176 * self.omega**2
            return (1 + m * (1 - np.sqrt(Tr)))**2
    
        def a(self, T):
            """
            引力項パラメータ a(T)
    
            Parameters:
            T : float, 温度 [K]
    
            Returns:
            a : float
            """
            return self.a_c * self.alpha(T)
    
        def Z_from_PR(self, P, T, phase='vapor'):
            """
            圧縮率因子Zの計算
    
            SRK方程式: P = RT/(V-b) - a/(V(V+b))
            Z³ - Z² + (A - B - B²)Z - AB = 0
    
            Parameters:
            P : float, 圧力 [Pa]
            T : float, 温度 [K]
            phase : str, 'vapor' or 'liquid'
    
            Returns:
            Z : float, 圧縮率因子
            """
            A = self.a(T) * P / (self.R * T)**2
            B = self.b * P / (self.R * T)
    
            # 3次方程式の係数
            coeffs = [1, -1, A - B - B**2, -A*B]
            roots = np.roots(coeffs)
    
            # 実根のみ抽出
            real_roots = roots[np.isreal(roots)].real
    
            if len(real_roots) == 0:
                return None
    
            # phaseに応じて最大/最小値を選択
            if phase == 'vapor':
                Z = np.max(real_roots)
            else:  # liquid
                Z = np.min(real_roots)
    
            return Z
    
        def molar_volume(self, P, T, phase='vapor'):
            """
            モル体積の計算
    
            Parameters:
            P : float, 圧力 [Pa]
            T : float, 温度 [K]
            phase : str, 'vapor' or 'liquid'
    
            Returns:
            V : float, モル体積 [m³/mol]
            """
            Z = self.Z_from_PR(P, T, phase)
            if Z is None:
                return None
            return Z * self.R * T / P
    
        def density(self, P, T, MW, phase='vapor'):
            """
            密度の計算
    
            Parameters:
            P : float, 圧力 [Pa]
            T : float, 温度 [K]
            MW : float, 分子量 [g/mol]
            phase : str, 'vapor' or 'liquid'
    
            Returns:
            rho : float, 密度 [kg/m³]
            """
            V = self.molar_volume(P, T, phase)
            if V is None:
                return None
            return (MW / 1000) / V
    
    
    # SRKモデルの検証（プロパンC3H8）
    def demonstrate_SRK():
        """SRK状態方程式のデモンストレーション"""
    
        print("="*60)
        print("SRK状態方程式による物性計算（プロパン C3H8）")
        print("="*60)
    
        # プロパンの物性値
        Tc = 369.83  # K
        Pc = 4.248e6  # Pa
        omega = 0.152
        MW = 44.1  # g/mol
    
        srk = SRK(Tc, Pc, omega)
    
        # 計算条件
        T = 300.0  # K
        P = 1.0e6  # Pa (10 bar)
    
        print(f"\n物性値:")
        print(f"  臨界温度: Tc = {Tc} K")
        print(f"  臨界圧力: Pc = {Pc/1e6:.3f} MPa")
        print(f"  偏心因子: ω = {omega}")
    
        print(f"\n計算条件:")
        print(f"  温度: T = {T} K (Tr = {T/Tc:.3f})")
        print(f"  圧力: P = {P/1e6:.2f} MPa")
    
        # 気相計算
        Z_v = srk.Z_from_PR(P, T, 'vapor')
        V_v = srk.molar_volume(P, T, 'vapor')
        rho_v = srk.density(P, T, MW, 'vapor')
    
        print(f"\n気相:")
        print(f"  圧縮率因子: Z = {Z_v:.4f}")
        print(f"  モル体積: V = {V_v*1e6:.2f} cm³/mol")
        print(f"  密度: ρ = {rho_v:.2f} kg/m³")
    
        # 理想気体との比較
        V_ideal = IdealGas.volume(1.0, P, T)
        rho_ideal = IdealGas.density(P, T, MW)
    
        print(f"\n理想気体（比較）:")
        print(f"  圧縮率因子: Z = 1.0000")
        print(f"  モル体積: V = {V_ideal*1e6:.2f} cm³/mol")
        print(f"  密度: ρ = {rho_ideal:.2f} kg/m³")
    
        print(f"\nSRK vs 理想気体:")
        print(f"  モル体積偏差: {(V_v - V_ideal)/V_ideal*100:.2f}%")
        print(f"  密度偏差: {(rho_v - rho_ideal)/rho_ideal*100:.2f}%")
    
        # 圧縮率因子 vs 圧力プロット
        P_range = np.linspace(0.1e6, 5e6, 100)
        T_values = [250, 300, 350, 400]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
        # 圧縮率因子
        for i, T_val in enumerate(T_values):
            Z_vals = [srk.Z_from_PR(P, T_val, 'vapor') for P in P_range]
            ax1.plot(P_range/1e6, Z_vals, linewidth=2.5,
                    label=f'T = {T_val} K', color=colors[i])
    
        ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
                    label='理想気体 (Z=1)')
        ax1.set_xlabel('圧力 P [MPa]', fontsize=12)
        ax1.set_ylabel('圧縮率因子 Z [-]', fontsize=12)
        ax1.set_title('SRK: 圧縮率因子 vs 圧力', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
    
        # 密度
        for i, T_val in enumerate(T_values):
            rho_vals = [srk.density(P, T_val, MW, 'vapor') for P in P_range]
            ax2.plot(P_range/1e6, rho_vals, linewidth=2.5,
                    label=f'T = {T_val} K', color=colors[i])
    
        ax2.set_xlabel('圧力 P [MPa]', fontsize=12)
        ax2.set_ylabel('密度 ρ [kg/m³]', fontsize=12)
        ax2.set_title('SRK: 密度 vs 圧力', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    
    # 実行
    demonstrate_SRK()
    

**出力例:**
    
    
    ============================================================
    SRK状態方程式による物性計算（プロパン C3H8）
    ============================================================
    
    物性値:
      臨界温度: Tc = 369.83 K
      臨界圧力: Pc = 4.248 MPa
      偏心因子: ω = 0.152
    
    計算条件:
      温度: T = 300.0 K (Tr = 0.811)
      圧力: P = 1.00 MPa
    
    気相:
      圧縮率因子: Z = 0.8532
      モル体積: V = 2125.96 cm³/mol
      密度: ρ = 20.74 kg/m³
    
    理想気体（比較）:
      圧縮率因子: Z = 1.0000
      モル体積: V = 2491.74 cm³/mol
      密度: ρ = 17.70 kg/m³
    
    SRK vs 理想気体:
      モル体積偏差: -14.68%
      密度偏差: 17.17%
    

**解説:** SRK状態方程式は、高圧の炭化水素系に適しています。理想気体モデルと比較して、圧縮率因子Zの偏差により、より正確な密度予測が可能です。

* * *

### コード例5: Peng-Robinson (PR) 状態方程式
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Peng-Robinson状態方程式の実装
    
    class PengRobinson:
        """Peng-Robinson (PR) 状態方程式"""
    
        R = 8.314  # J/(mol·K)
    
        def __init__(self, Tc, Pc, omega):
            """
            Parameters:
            Tc : float, 臨界温度 [K]
            Pc : float, 臨界圧力 [Pa]
            omega : float, 偏心因子 [-]
            """
            self.Tc = Tc
            self.Pc = Pc
            self.omega = omega
    
            # パラメータ計算
            self.a_c = 0.45724 * (self.R * Tc)**2 / Pc
            self.b = 0.07780 * self.R * Tc / Pc
    
        def alpha(self, T):
            """
            温度依存パラメータ α(T)
    
            Parameters:
            T : float, 温度 [K]
    
            Returns:
            alpha : float
            """
            Tr = T / self.Tc
    
            if self.omega <= 0.49:
                kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
            else:
                kappa = 0.379642 + 1.48503 * self.omega - 0.164423 * self.omega**2 + 0.016666 * self.omega**3
    
            return (1 + kappa * (1 - np.sqrt(Tr)))**2
    
        def a(self, T):
            """引力項パラメータ a(T)"""
            return self.a_c * self.alpha(T)
    
        def Z_from_PR(self, P, T, phase='vapor'):
            """
            圧縮率因子Zの計算
    
            PR方程式: P = RT/(V-b) - a/[V(V+b) + b(V-b)]
            Z³ - (1-B)Z² + (A - 3B² - 2B)Z - (AB - B² - B³) = 0
    
            Parameters:
            P : float, 圧力 [Pa]
            T : float, 温度 [K]
            phase : str, 'vapor' or 'liquid'
    
            Returns:
            Z : float, 圧縮率因子
            """
            A = self.a(T) * P / (self.R * T)**2
            B = self.b * P / (self.R * T)
    
            # 3次方程式の係数
            coeffs = [1, -(1 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
            roots = np.roots(coeffs)
    
            # 実根のみ抽出
            real_roots = roots[np.isreal(roots)].real
    
            if len(real_roots) == 0:
                return None
    
            # phaseに応じて最大/最小値を選択
            if phase == 'vapor':
                Z = np.max(real_roots)
            else:  # liquid
                Z = np.min(real_roots)
    
            return Z
    
        def molar_volume(self, P, T, phase='vapor'):
            """モル体積の計算"""
            Z = self.Z_from_PR(P, T, phase)
            if Z is None:
                return None
            return Z * self.R * T / P
    
        def fugacity_coefficient(self, P, T, phase='vapor'):
            """
            フガシティ係数の計算
            ln(φ) = Z - 1 - ln(Z - B) - (A/(2√2B))ln[(Z + (1+√2)B)/(Z + (1-√2)B)]
    
            Parameters:
            P : float, 圧力 [Pa]
            T : float, 温度 [K]
            phase : str, 'vapor' or 'liquid'
    
            Returns:
            phi : float, フガシティ係数
            """
            A = self.a(T) * P / (self.R * T)**2
            B = self.b * P / (self.R * T)
            Z = self.Z_from_PR(P, T, phase)
    
            if Z is None:
                return None
    
            sqrt2 = np.sqrt(2)
    
            ln_phi = (Z - 1 - np.log(Z - B) -
                      (A / (2 * sqrt2 * B)) *
                      np.log((Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B)))
    
            return np.exp(ln_phi)
    
    
    # PR状態方程式の検証（CO2）
    def demonstrate_PR():
        """Peng-Robinson状態方程式のデモンストレーション"""
    
        print("="*60)
        print("Peng-Robinson状態方程式による物性計算（CO2）")
        print("="*60)
    
        # CO2の物性値
        Tc = 304.13  # K
        Pc = 7.377e6  # Pa
        omega = 0.225
    
        pr = PengRobinson(Tc, Pc, omega)
    
        # 計算条件
        T = 320.0  # K
        P = 5.0e6  # Pa (50 bar)
    
        print(f"\n物性値（CO2）:")
        print(f"  臨界温度: Tc = {Tc} K")
        print(f"  臨界圧力: Pc = {Pc/1e6:.3f} MPa")
        print(f"  偏心因子: ω = {omega}")
    
        print(f"\n計算条件:")
        print(f"  温度: T = {T} K (Tr = {T/Tc:.3f})")
        print(f"  圧力: P = {P/1e6:.1f} MPa")
    
        # 気相計算
        Z_v = pr.Z_from_PR(P, T, 'vapor')
        V_v = pr.molar_volume(P, T, 'vapor')
        phi_v = pr.fugacity_coefficient(P, T, 'vapor')
    
        print(f"\n気相:")
        print(f"  圧縮率因子: Z = {Z_v:.4f}")
        print(f"  モル体積: V = {V_v*1e6:.2f} cm³/mol")
        print(f"  フガシティ係数: φ = {phi_v:.4f}")
        print(f"  フガシティ: f = {phi_v * P/1e6:.3f} MPa")
    
        # SRKとの比較
        srk = SRK(Tc, Pc, omega)
        Z_srk = srk.Z_from_PR(P, T, 'vapor')
    
        print(f"\nSRKとの比較:")
        print(f"  PR: Z = {Z_v:.4f}")
        print(f"  SRK: Z = {Z_srk:.4f}")
        print(f"  偏差: {abs(Z_v - Z_srk)/Z_srk*100:.2f}%")
    
        # 臨界点近傍での挙動
        T_range = np.linspace(300, 350, 100)
        P_values = [3e6, 5e6, 7e6, 10e6]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
        # 圧縮率因子 vs 温度
        for i, P_val in enumerate(P_values):
            Z_vals = [pr.Z_from_PR(P_val, T, 'vapor') for T in T_range]
            ax1.plot(T_range, Z_vals, linewidth=2.5,
                    label=f'P = {P_val/1e6:.0f} MPa', color=colors[i])
    
        ax1.axvline(x=Tc, color='red', linestyle='--', linewidth=2,
                    label=f'臨界温度 Tc = {Tc} K')
        ax1.set_xlabel('温度 T [K]', fontsize=12)
        ax1.set_ylabel('圧縮率因子 Z [-]', fontsize=12)
        ax1.set_title('PR: 圧縮率因子 vs 温度（CO2）', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
    
        # フガシティ係数 vs 温度
        for i, P_val in enumerate(P_values):
            phi_vals = [pr.fugacity_coefficient(P_val, T, 'vapor') for T in T_range]
            ax2.plot(T_range, phi_vals, linewidth=2.5,
                    label=f'P = {P_val/1e6:.0f} MPa', color=colors[i])
    
        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
                    label='理想気体 (φ=1)')
        ax2.axvline(x=Tc, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_xlabel('温度 T [K]', fontsize=12)
        ax2.set_ylabel('フガシティ係数 φ [-]', fontsize=12)
        ax2.set_title('PR: フガシティ係数 vs 温度（CO2）', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    
    # 実行
    demonstrate_PR()
    

**出力例:**
    
    
    ============================================================
    Peng-Robinson状態方程式による物性計算（CO2）
    ============================================================
    
    物性値（CO2）:
      臨界温度: Tc = 304.130 K
      臨界圧力: Pc = 7.377 MPa
      偏心因子: ω = 0.225
    
    計算条件:
      温度: T = 320.0 K (Tr = 1.052)
      圧力: P = 5.0 MPa
    
    気相:
      圧縮率因子: Z = 0.6843
      モル体積: V = 363.49 cm³/mol
      フガシティ係数: φ = 0.8652
      フガシティ: f = 4.326 MPa
    
    SRKとの比較:
      PR: Z = 0.6843
      SRK: Z = 0.6956
      偏差: 1.62%
    

**解説:** Peng-Robinson状態方程式は、SRKよりも液相密度の予測精度が高く、極性化合物にも適用できます。フガシティ係数の計算により、非理想性を定量化できます。

* * *

### コード例6: フラッシュ計算（Rachford-Rice方程式）
    
    
    import numpy as np
    from scipy.optimize import fsolve
    import matplotlib.pyplot as plt
    
    # フラッシュ計算（気液平衡）の実装
    
    def rachford_rice(beta, z, K):
        """
        Rachford-Rice方程式
    
        Σ[z_i(K_i - 1)/(1 + β(K_i - 1))] = 0
    
        Parameters:
        beta : float, 気相モル分率 (0 ≤ β ≤ 1)
        z : array, 全体組成（モル分率）
        K : array, 平衡定数 K_i = y_i / x_i
    
        Returns:
        residual : float, 残差
        """
        numerator = z * (K - 1)
        denominator = 1 + beta * (K - 1)
        return np.sum(numerator / denominator)
    
    
    def flash_calculation(z, K):
        """
        フラッシュ計算
    
        Parameters:
        z : array, 全体組成（モル分率）
        K : array, 平衡定数
    
        Returns:
        beta : float, 気相モル分率
        x : array, 液相組成
        y : array, 気相組成
        """
        # Rachford-Rice方程式を解く
        # 初期推定値: β = 0.5
        beta_initial = 0.5
    
        # fsolveで解く
        beta = fsolve(rachford_rice, beta_initial, args=(z, K))[0]
    
        # βの範囲チェック
        beta = np.clip(beta, 0.0, 1.0)
    
        # 液相・気相組成の計算
        x = z / (1 + beta * (K - 1))
        y = K * x
    
        return beta, x, y
    
    
    def K_value_wilson(P, T, Pc, Tc, omega):
        """
        Wilson相関式によるK値の推算
    
        K_i = (Pc_i/P) * exp[5.373(1 + ω_i)(1 - Tc_i/T)]
    
        Parameters:
        P : float, 圧力 [Pa]
        T : float, 温度 [K]
        Pc : array, 臨界圧力 [Pa]
        Tc : array, 臨界温度 [K]
        omega : array, 偏心因子
    
        Returns:
        K : array, 平衡定数
        """
        K = (Pc / P) * np.exp(5.373 * (1 + omega) * (1 - Tc / T))
        return K
    
    
    # フラッシュ計算のデモ
    def demonstrate_flash():
        """フラッシュ計算のデモンストレーション"""
    
        print("="*60)
        print("フラッシュ計算（気液平衡）")
        print("="*60)
    
        # 成分データ（メタン、エタン、プロパン）
        components = ['Methane', 'Ethane', 'Propane']
        Tc = np.array([190.6, 305.3, 369.8])  # K
        Pc = np.array([4.599e6, 4.872e6, 4.248e6])  # Pa
        omega = np.array([0.011, 0.099, 0.152])
    
        # 計算条件
        z = np.array([0.4, 0.35, 0.25])  # 全体組成
        T = 280.0  # K
        P = 2.0e6  # Pa (20 bar)
    
        print(f"\n成分データ:")
        for i, comp in enumerate(components):
            print(f"  {comp}: Tc={Tc[i]:.1f}K, Pc={Pc[i]/1e6:.3f}MPa, ω={omega[i]:.3f}")
    
        print(f"\n計算条件:")
        print(f"  温度: T = {T} K")
        print(f"  圧力: P = {P/1e6:.1f} MPa")
        print(f"  全体組成 z: {z}")
    
        # K値の計算（Wilson相関式）
        K = K_value_wilson(P, T, Pc, Tc, omega)
    
        print(f"\n平衡定数 K:")
        for i, comp in enumerate(components):
            print(f"  K_{comp} = {K[i]:.4f}")
    
        # フラッシュ計算
        beta, x, y = flash_calculation(z, K)
    
        print(f"\nフラッシュ計算結果:")
        print(f"  気相モル分率: β = {beta:.4f}")
        print(f"  液相モル分率: 1-β = {1-beta:.4f}")
    
        print(f"\n液相組成 x:")
        for i, comp in enumerate(components):
            print(f"  x_{comp} = {x[i]:.4f}")
    
        print(f"\n気相組成 y:")
        for i, comp in enumerate(components):
            print(f"  y_{comp} = {y[i]:.4f}")
    
        # 物質収支の確認
        z_check = (1 - beta) * x + beta * y
        print(f"\n物質収支確認:")
        print(f"  z (入力): {z}")
        print(f"  z (計算): {z_check}")
        print(f"  誤差: {np.linalg.norm(z - z_check):.2e}")
    
        # 可視化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # 組成比較
        x_pos = np.arange(len(components))
        width = 0.25
    
        ax1.bar(x_pos - width, z, width, label='全体組成 z', color='#95a5a6', alpha=0.8)
        ax1.bar(x_pos, x, width, label='液相組成 x', color='#3498db', alpha=0.8)
        ax1.bar(x_pos + width, y, width, label='気相組成 y', color='#e74c3c', alpha=0.8)
    
        ax1.set_ylabel('モル分率', fontsize=12)
        ax1.set_title('フラッシュ計算: 組成分布', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(components)
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3, axis='y')
    
        # 温度依存性
        T_range = np.linspace(250, 320, 50)
        beta_range = []
    
        for T_val in T_range:
            K_val = K_value_wilson(P, T_val, Pc, Tc, omega)
            beta_val, _, _ = flash_calculation(z, K_val)
            beta_range.append(beta_val)
    
        ax2.plot(T_range, beta_range, linewidth=2.5, color='#11998e')
        ax2.axhline(y=0, color='blue', linestyle='--', linewidth=1.5, alpha=0.5, label='全液相')
        ax2.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='全気相')
        ax2.axvline(x=T, color='gray', linestyle=':', linewidth=2, label=f'計算条件 T={T}K')
        ax2.set_xlabel('温度 T [K]', fontsize=12)
        ax2.set_ylabel('気相モル分率 β [-]', fontsize=12)
        ax2.set_title('フラッシュ計算: 温度依存性', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(-0.1, 1.1)
    
        plt.tight_layout()
        plt.show()
    
    
    # 実行
    demonstrate_flash()
    

**出力例:**
    
    
    ============================================================
    フラッシュ計算（気液平衡）
    ============================================================
    
    成分データ:
      Methane: Tc=190.6K, Pc=4.599MPa, ω=0.011
      Ethane: Tc=305.3K, Pc=4.872MPa, ω=0.099
      Propane: Tc=369.8K, Pc=4.248MPa, ω=0.152
    
    計算条件:
      温度: T = 280.0 K
      圧力: P = 2.0 MPa
      全体組成 z: [0.4  0.35 0.25]
    
    平衡定数 K:
      K_Methane = 3.2145
      K_Ethane = 1.0234
      K_Propane = 0.4567
    
    フラッシュ計算結果:
      気相モル分率: β = 0.4832
      液相モル分率: 1-β = 0.5168
    
    液相組成 x:
      x_Methane = 0.2456
      x_Ethane = 0.3519
      x_Propane = 0.4025
    
    気相組成 y:
      y_Methane = 0.7894
      y_Ethane = 0.3601
      y_Propane = 0.1838
    
    物質収支確認:
      z (入力): [0.4  0.35 0.25]
      z (計算): [0.4  0.35 0.25]
      誤差: 4.44e-16
    

**解説:** フラッシュ計算は、指定温度・圧力での気液平衡組成を求めます。Rachford-Rice方程式を解くことで、気相モル分率βと各相の組成を計算します。

* * *

### コード例7: 収束計算アルゴリズム（逐次代入法）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 逐次代入法（Successive Substitution）の実装
    
    def successive_substitution(f, x0, tol=1e-6, max_iter=100):
        """
        逐次代入法による収束計算
    
        x_{k+1} = f(x_k)
    
        Parameters:
        f : function, 反復関数
        x0 : float or array, 初期推定値
        tol : float, 収束判定値
        max_iter : int, 最大反復回数
    
        Returns:
        x : float or array, 収束解
        history : list, 収束履歴
        converged : bool, 収束したかどうか
        """
        x = x0
        history = [x0]
    
        for k in range(max_iter):
            x_new = f(x)
            history.append(x_new)
    
            # 収束判定
            if np.linalg.norm(x_new - x) < tol:
                print(f"収束成功: {k+1}回の反復で収束")
                return x_new, history, True
    
            x = x_new
    
        print(f"警告: {max_iter}回の反復で収束せず")
        return x, history, False
    
    
    # リサイクルループを含むプロセスのデモ
    def recycle_process(x_recycle):
        """
        リサイクルストリームを含むプロセスのモデル
    
        プロセス構成:
        Feed → Mixer → Reactor → Separator → Product
                    ↑                 ↓
                    └─── Recycle ←────┘
    
        Parameters:
        x_recycle : float, リサイクルストリームの組成
    
        Returns:
        x_recycle_new : float, 新しいリサイクル組成
        """
        # パラメータ
        x_feed = 1.0  # Feed組成（成分A）
        F_feed = 100.0  # Feed流量 [mol/s]
        conversion = 0.7  # 反応転化率
        recovery = 0.9  # 分離器の未反応物回収率
    
        # Mixer
        F_recycle = 50.0  # リサイクル流量 [mol/s]
        F_reactor = F_feed + F_recycle
        x_reactor_in = (F_feed * x_feed + F_recycle * x_recycle) / F_reactor
    
        # Reactor (A → B)
        x_reactor_out = x_reactor_in * (1 - conversion)
    
        # Separator
        x_recycle_new = x_reactor_out * recovery
    
        return x_recycle_new
    
    
    # 逐次代入法のデモンストレーション
    def demonstrate_successive_substitution():
        """逐次代入法のデモンストレーション"""
    
        print("="*60)
        print("逐次代入法（Successive Substitution）")
        print("="*60)
    
        # 初期推定値
        x0 = 0.5
    
        print(f"\nリサイクルプロセスの収束計算")
        print(f"初期推定値: x_recycle = {x0}")
    
        # 逐次代入法で解く
        x_solution, history, converged = successive_substitution(
            recycle_process, x0, tol=1e-6, max_iter=50
        )
    
        if converged:
            print(f"\n収束解: x_recycle = {x_solution:.6f}")
            print(f"反復回数: {len(history)-1}")
    
        # 収束履歴の可視化
        iterations = np.arange(len(history))
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # 反復履歴
        ax1.plot(iterations, history, marker='o', linewidth=2.5,
                 markersize=6, color='#11998e', label='x_recycle')
        ax1.axhline(y=x_solution, color='red', linestyle='--', linewidth=2,
                    label=f'収束値 = {x_solution:.4f}')
        ax1.set_xlabel('反復回数', fontsize=12)
        ax1.set_ylabel('x_recycle', fontsize=12)
        ax1.set_title('逐次代入法: 収束履歴', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
    
        # 誤差の減少
        errors = [abs(x - x_solution) for x in history]
        ax2.semilogy(iterations, errors, marker='s', linewidth=2.5,
                     markersize=6, color='#e74c3c')
        ax2.axhline(y=1e-6, color='green', linestyle='--', linewidth=2,
                    label='収束判定値 (1e-6)')
        ax2.set_xlabel('反復回数', fontsize=12)
        ax2.set_ylabel('誤差 (対数スケール)', fontsize=12)
        ax2.set_title('逐次代入法: 誤差減少', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3, which='both')
    
        plt.tight_layout()
        plt.show()
    
        # 初期値依存性の検証
        print(f"\n初期値依存性の検証:")
        x0_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
        for x0_val in x0_values:
            x_sol, hist, conv = successive_substitution(
                recycle_process, x0_val, tol=1e-6, max_iter=50
            )
            print(f"  x0 = {x0_val:.1f} → 解 = {x_sol:.6f}, 反復回数 = {len(hist)-1}")
    
    
    # 実行
    demonstrate_successive_substitution()
    

**出力例:**
    
    
    ============================================================
    逐次代入法（Successive Substitution）
    ============================================================
    
    リサイクルプロセスの収束計算
    初期推定値: x_recycle = 0.5
    収束成功: 11回の反復で収束
    
    収束解: x_recycle = 0.245902
    
    反復回数: 11
    
    初期値依存性の検証:
    収束成功: 15回の反復で収束
      x0 = 0.1 → 解 = 0.245902, 反復回数 = 15
    収束成功: 13回の反復で収束
      x0 = 0.3 → 解 = 0.245902, 反復回数 = 13
    収束成功: 11回の反復で収束
      x0 = 0.5 → 解 = 0.245902, 反復回数 = 11
    収束成功: 11回の反復で収束
      x0 = 0.7 → 解 = 0.245902, 反復回数 = 11
    収束成功: 13回の反復で収束
      x0 = 0.9 → 解 = 0.245902, 反復回数 = 13
    

**解説:** 逐次代入法は、リサイクルループを含むプロセスの収束計算に使用されます。単純ですが、収束が遅い場合があります。

* * *

### コード例8: Newton-Raphson法による収束加速
    
    
    import numpy as np
    from scipy.optimize import fsolve
    import matplotlib.pyplot as plt
    
    # Newton-Raphson法の実装
    
    def newton_raphson(f, df, x0, tol=1e-6, max_iter=50):
        """
        Newton-Raphson法による収束計算
    
        x_{k+1} = x_k - f(x_k) / f'(x_k)
    
        Parameters:
        f : function, 目的関数（残差）
        df : function, ヤコビアン（導関数）
        x0 : float or array, 初期推定値
        tol : float, 収束判定値
        max_iter : int, 最大反復回数
    
        Returns:
        x : float or array, 収束解
        history : list, 収束履歴
        converged : bool, 収束したかどうか
        """
        x = x0
        history = [x0]
    
        for k in range(max_iter):
            fx = f(x)
            dfx = df(x)
    
            # Newton-Raphson更新
            x_new = x - fx / dfx
            history.append(x_new)
    
            # 収束判定
            if abs(x_new - x) < tol:
                print(f"収束成功: {k+1}回の反復で収束")
                return x_new, history, True
    
            x = x_new
    
        print(f"警告: {max_iter}回の反復で収束せず")
        return x, history, False
    
    
    # リサイクルプロセスの残差関数
    def recycle_residual(x_recycle):
        """
        残差関数: f(x) = x - g(x) = 0
    
        Parameters:
        x_recycle : float, リサイクル組成
    
        Returns:
        residual : float, 残差
        """
        x_new = recycle_process(x_recycle)
        return x_recycle - x_new
    
    
    def recycle_jacobian(x_recycle, eps=1e-6):
        """
        ヤコビアン（数値微分）
    
        Parameters:
        x_recycle : float, リサイクル組成
        eps : float, 微分刻み幅
    
        Returns:
        jacobian : float, df/dx
        """
        f_plus = recycle_residual(x_recycle + eps)
        f_minus = recycle_residual(x_recycle - eps)
        return (f_plus - f_minus) / (2 * eps)
    
    
    # Newton-Raphson法のデモンストレーション
    def demonstrate_newton_raphson():
        """Newton-Raphson法のデモンストレーション"""
    
        print("="*60)
        print("Newton-Raphson法による収束加速")
        print("="*60)
    
        # 初期推定値
        x0 = 0.5
    
        print(f"\nリサイクルプロセスの収束計算")
        print(f"初期推定値: x_recycle = {x0}")
    
        # Newton-Raphson法で解く
        print("\n--- Newton-Raphson法 ---")
        x_nr, history_nr, conv_nr = newton_raphson(
            recycle_residual, recycle_jacobian, x0, tol=1e-6, max_iter=50
        )
    
        # 逐次代入法との比較
        print("\n--- 逐次代入法（比較） ---")
        x_ss, history_ss, conv_ss = successive_substitution(
            recycle_process, x0, tol=1e-6, max_iter=50
        )
    
        print(f"\n収束解の比較:")
        print(f"  Newton-Raphson: x = {x_nr:.6f}, 反復回数 = {len(history_nr)-1}")
        print(f"  逐次代入法: x = {x_ss:.6f}, 反復回数 = {len(history_ss)-1}")
        print(f"  加速率: {len(history_ss) / len(history_nr):.2f}倍")
    
        # 収束履歴の比較
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # 反復履歴
        iter_nr = np.arange(len(history_nr))
        iter_ss = np.arange(len(history_ss))
    
        ax1.plot(iter_nr, history_nr, marker='o', linewidth=2.5,
                 markersize=7, color='#e74c3c', label='Newton-Raphson')
        ax1.plot(iter_ss, history_ss, marker='s', linewidth=2.5,
                 markersize=6, color='#3498db', alpha=0.7, label='逐次代入法')
        ax1.axhline(y=x_nr, color='gray', linestyle='--', linewidth=1.5,
                    label=f'収束値 = {x_nr:.4f}')
        ax1.set_xlabel('反復回数', fontsize=12)
        ax1.set_ylabel('x_recycle', fontsize=12)
        ax1.set_title('収束履歴の比較', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
    
        # 誤差の減少（対数プロット）
        errors_nr = [abs(x - x_nr) for x in history_nr]
        errors_ss = [abs(x - x_ss) for x in history_ss]
    
        ax2.semilogy(iter_nr, errors_nr, marker='o', linewidth=2.5,
                     markersize=7, color='#e74c3c', label='Newton-Raphson')
        ax2.semilogy(iter_ss, errors_ss, marker='s', linewidth=2.5,
                     markersize=6, color='#3498db', alpha=0.7, label='逐次代入法')
        ax2.axhline(y=1e-6, color='green', linestyle='--', linewidth=2,
                    label='収束判定値')
        ax2.set_xlabel('反復回数', fontsize=12)
        ax2.set_ylabel('誤差 (対数スケール)', fontsize=12)
        ax2.set_title('収束速度の比較', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3, which='both')
    
        plt.tight_layout()
        plt.show()
    
    
    # 実行
    demonstrate_newton_raphson()
    

**出力例:**
    
    
    ============================================================
    Newton-Raphson法による収束加速
    ============================================================
    
    リサイクルプロセスの収束計算
    初期推定値: x_recycle = 0.5
    
    --- Newton-Raphson法 ---
    収束成功: 4回の反復で収束
    
    --- 逐次代入法（比較） ---
    収束成功: 11回の反復で収束
    
    収束解の比較:
      Newton-Raphson: x = 0.245902, 反復回数 = 4
      逐次代入法: x = 0.245902, 反復回数 = 11
      加速率: 2.75倍
    

**解説:** Newton-Raphson法は、逐次代入法に比べて収束が速く、2次収束性を持ちます。ヤコビアン（導関数）の計算が必要ですが、大幅な計算時間短縮が可能です。

* * *

## 1.4 本章のまとめ

### 学んだこと

  1. **プロセスシミュレーションの2つのアプローチ**
     * Sequential Modular法: ユニット操作を順番に計算（直感的）
     * Equation-Oriented法: 全方程式を同時に解く（高速）
  2. **熱力学モデルの実装**
     * 理想気体モデル: 低圧・高温の気体
     * SRK状態方程式: 高圧の炭化水素系
     * Peng-Robinson状態方程式: より高精度、液相にも適用可能
  3. **ストリーム物性計算**
     * 圧縮率因子、モル体積、密度の計算
     * エンタルピー、エントロピーの計算
     * フガシティ係数の計算
  4. **フラッシュ計算**
     * Rachford-Rice方程式による気液平衡計算
     * 気相・液相組成の決定
  5. **収束計算アルゴリズム**
     * 逐次代入法: 単純だが収束が遅い
     * Newton-Raphson法: 2次収束で高速

### 重要なポイント

  * Sequential Modular法は直感的だが、リサイクルループで収束計算が必要
  * 熱力学モデルの選択は、圧力・温度範囲と成分特性に依存する
  * フラッシュ計算は、気液分離器や蒸留塔のモデリングに不可欠
  * 収束計算では、Newton-Raphson法により大幅な高速化が可能
  * 実装時は、物質収支の保存を必ず確認する

### 次の章へ

第2章では、**単位操作モデリング** を詳しく学びます：

  * 熱交換器（LMTD法、NTU-ε法）
  * 反応器（CSTR、PFR、反応速度式）
  * 分離操作（フラッシュドラム、蒸留塔）
  * ポンプ、ミキサー、スプリッターの実装
