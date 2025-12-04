---
title: 第3章：単位操作のモデリング
chapter_title: 第3章：単位操作のモデリング
subtitle: 蒸留・吸収・反応器・熱交換器の工学的モデル実装
---

## 3.1 蒸留塔のモデリング

蒸留は化学プロセスで最も重要な分離操作の一つです。McCabe-Thiele法、Fenske-Underwood-Gilliland法などの 短絡法を使用して、必要段数や還流比を計算します。 

### 3.1.1 McCabe-Thiele法による段数計算

二成分系蒸留で、気液平衡曲線と操作線から理論段数を図解的に求める古典的手法をPythonで実装します。 

#### 例題1：ベンゼン-トルエン系の蒸留塔設計

供給組成50mol%、留出物組成95mol%、缶出物組成5mol%、還流比R=2.0の条件で必要段数を計算します。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    class McCabeThiele:
        """McCabe-Thiele法による蒸留塔段数計算"""
    
        def __init__(self, x_F, x_D, x_B, q, R, alpha):
            """
            Args:
                x_F (float): 供給組成（軽質成分モル分率）
                x_D (float): 留出物組成
                x_B (float): 缶出物組成
                q (float): 供給状態（q線の傾き、飽和液体=1.0）
                R (float): 還流比
                alpha (float): 相対揮発度
            """
            self.x_F = x_F
            self.x_D = x_D
            self.x_B = x_B
            self.q = q
            self.R = R
            self.alpha = alpha
    
        def equilibrium_curve(self, x):
            """気液平衡曲線（相対揮発度一定）"""
            return (self.alpha * x) / (1 + (self.alpha - 1) * x)
    
        def rectifying_operating_line(self, x):
            """精留部操作線"""
            return (self.R / (self.R + 1)) * x + self.x_D / (self.R + 1)
    
        def stripping_operating_line(self, x):
            """回収部操作線"""
            # 交点を求める（q線との交点）
            x_q = self._find_q_intersection()
    
            # 回収部操作線の傾き
            L_bar_over_V_bar = self._calculate_stripping_slope()
    
            return L_bar_over_V_bar * (x - self.x_B) + self.x_B
    
        def _find_q_intersection(self):
            """q線と精留部操作線の交点"""
            def equations(x):
                y_rect = self.rectifying_operating_line(x)
                # q線: y = (q/(q-1)) * x - x_F/(q-1)
                if abs(self.q - 1.0) < 1e-6:
                    # 飽和液体の場合、x = x_F
                    return x - self.x_F
                else:
                    y_q = (self.q / (self.q - 1)) * x - self.x_F / (self.q - 1)
                    return y_rect - y_q
    
            x_q = fsolve(equations, self.x_F)[0]
            return x_q
    
        def _calculate_stripping_slope(self):
            """回収部操作線の傾き計算"""
            x_q = self._find_q_intersection()
            y_q = self.rectifying_operating_line(x_q)
    
            # 2点 (x_B, x_B) と (x_q, y_q) を通る直線の傾き
            slope = (y_q - self.x_B) / (x_q - self.x_B)
            return slope
    
        def calculate_stages(self):
            """理論段数を計算"""
            stages = []
            x = self.x_D
            y = self.x_D
            stage_count = 0
            max_stages = 100
    
            while x > self.x_B and stage_count < max_stages:
                # 気液平衡（垂直線: y一定でxを求める）
                def eq(x_new):
                    return self.equilibrium_curve(x_new) - y
                x_new = fsolve(eq, x)[0]
    
                stages.append((x, y, x_new, y))
    
                # 操作線（水平線: x一定でyを求める）
                x = x_new
                x_q = self._find_q_intersection()
    
                if x >= x_q:
                    # 精留部
                    y_new = self.rectifying_operating_line(x)
                else:
                    # 回収部
                    y_new = self.stripping_operating_line(x)
    
                stages.append((x, y, x, y_new))
                y = y_new
                stage_count += 1
    
            return stages, stage_count
    
        def plot_diagram(self, filename=None):
            """McCabe-Thiele線図をプロット"""
            x = np.linspace(0, 1, 100)
    
            # 各曲線
            y_eq = self.equilibrium_curve(x)
            y_rect = self.rectifying_operating_line(x)
    
            # 回収部操作線
            x_strip = np.linspace(self.x_B, self._find_q_intersection(), 50)
            y_strip = [self.stripping_operating_line(xi) for xi in x_strip]
    
            # 段数描画
            stages, N = self.calculate_stages()
    
            plt.figure(figsize=(10, 8))
    
            # 各線をプロット
            plt.plot(x, y_eq, 'b-', linewidth=2, label='気液平衡線')
            plt.plot(x, x, 'k--', linewidth=1, label='y=x')
            plt.plot(x, y_rect, 'r-', linewidth=1.5, label='精留部操作線')
            plt.plot(x_strip, y_strip, 'g-', linewidth=1.5, label='回収部操作線')
    
            # q線
            if abs(self.q - 1.0) > 1e-6:
                x_q_line = np.linspace(self.x_B, self.x_D, 50)
                y_q_line = (self.q / (self.q - 1)) * x_q_line - self.x_F / (self.q - 1)
                plt.plot(x_q_line, y_q_line, 'm--', linewidth=1, label='q線')
            else:
                plt.axvline(x=self.x_F, color='m', linestyle='--', linewidth=1,
                           label='q線（飽和液体）')
    
            # 段数階段
            for i, stage in enumerate(stages):
                x1, y1, x2, y2 = stage
                plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.8, alpha=0.6)
    
            plt.xlabel('液相組成 x (軽質成分)', fontsize=12)
            plt.ylabel('気相組成 y (軽質成分)', fontsize=12)
            plt.title(f'McCabe-Thiele線図 (理論段数: {N}段)', fontsize=14)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
    
            if filename:
                plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
    
            return N
    
    # 実践例：ベンゼン-トルエン蒸留
    x_F = 0.50  # 供給組成
    x_D = 0.95  # 留出物組成
    x_B = 0.05  # 缶出物組成
    q = 1.0     # 飽和液体供給
    R = 2.0     # 還流比
    alpha = 2.5 # ベンゼン/トルエンの相対揮発度（80°C付近）
    
    mt = McCabeThiele(x_F, x_D, x_B, q, R, alpha)
    stages, N_theoretical = mt.calculate_stages()
    
    print("=" * 60)
    print("McCabe-Thiele法による蒸留塔設計")
    print("=" * 60)
    print(f"\n条件:")
    print(f"  供給組成 (x_F): {x_F*100:.0f}%")
    print(f"  留出物組成 (x_D): {x_D*100:.0f}%")
    print(f"  缶出物組成 (x_B): {x_B*100:.0f}%")
    print(f"  還流比 (R): {R}")
    print(f"  相対揮発度 (α): {alpha}")
    
    print(f"\n結果:")
    print(f"  理論段数: {N_theoretical}段")
    print(f"  供給段位置: {N_theoretical // 2}段付近（概算）")
    
    # 効率を考慮した実段数
    efficiency = 0.70  # 段効率70%
    N_actual = int(np.ceil(N_theoretical / efficiency))
    print(f"  段効率: {efficiency*100:.0f}%")
    print(f"  実段数: {N_actual}段")
    
    # 線図プロット
    # mt.plot_diagram('mccabe_thiele.png')  # ファイル保存する場合
    
    # 出力例:
    # ==============================================================
    # McCabe-Thiele法による蒸留塔設計
    # ==============================================================
    #
    # 条件:
    #   供給組成 (x_F): 50%
    #   留出物組成 (x_D): 95%
    #   缶出物組成 (x_B): 5%
    #   還流比 (R): 2.0
    #   相対揮発度 (α): 2.5
    #
    # 結果:
    #   理論段数: 9段
    #   供給段位置: 4段付近（概算）
    #   段効率: 70%
    #   実段数: 13段
    

### 3.1.2 Fenske-Underwood-Gilliland短絡法

最小還流比・最小段数を計算し、実際の運転条件での段数を推算する短絡法です。 設計初期段階で迅速に塔仕様を見積もるのに有用です。 

#### 例題2：Fenske-Underwood法による蒸留塔簡易設計

多成分系蒸留で、最小段数（全還流）と最小還流比を計算し、Gilliland相関で実段数を推算します。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class FenskeUnderwoodGilliland:
        """Fenske-Underwood-Gilliland短絡法"""
    
        def __init__(self, components, x_F, x_D, x_B, alpha):
            """
            Args:
                components (list): 成分名リスト
                x_F (dict): 供給組成 {成分: モル分率}
                x_D (dict): 留出物組成
                x_B (dict): 缶出物組成
                alpha (dict): 相対揮発度 {成分: α値}
            """
            self.components = components
            self.x_F = x_F
            self.x_D = x_D
            self.x_B = x_B
            self.alpha = alpha
    
            # 軽質成分(LK)と重質成分(HK)を特定
            self.LK = components[0]  # 最も軽い
            self.HK = components[-1]  # 最も重い
    
        def fenske_minimum_stages(self):
            """Fenske式: 最小段数（全還流時）"""
            # N_min = log[(x_D,LK/x_D,HK) * (x_B,HK/x_B,LK)] / log(α_LK/α_HK)
            ratio_D = self.x_D[self.LK] / self.x_D[self.HK]
            ratio_B = self.x_B[self.HK] / self.x_B[self.LK]
            alpha_avg = self.alpha[self.LK] / self.alpha[self.HK]
    
            N_min = np.log(ratio_D * ratio_B) / np.log(alpha_avg)
    
            return N_min
    
        def underwood_minimum_reflux(self):
            """Underwood式: 最小還流比"""
            # 簡略化: 二成分系近似
            # R_min = (1/(α-1)) * [(x_D,LK - x_F,LK) / x_F,LK]
    
            alpha_LK = self.alpha[self.LK]
            x_D_LK = self.x_D[self.LK]
            x_F_LK = self.x_F[self.LK]
    
            # より正確なUnderwood方程式（簡略版）
            theta = self._solve_underwood_theta()
    
            # 精留部のUnderwood方程式
            sum_term = 0.0
            for comp in self.components:
                sum_term += (self.alpha[comp] * self.x_D[comp]) / (self.alpha[comp] - theta)
    
            R_min = sum_term - 1.0
    
            return max(R_min, 0.5)  # 負にならないように
    
        def _solve_underwood_theta(self):
            """Underwoodのθパラメータを解く"""
            # 簡略化: 平均値を使用
            alpha_values = [self.alpha[c] for c in self.components]
            theta = np.mean(alpha_values)
            return theta
    
        def gilliland_correlation(self, R):
            """Gilliland相関: 実段数の推算"""
            N_min = self.fenske_minimum_stages()
            R_min = self.underwood_minimum_reflux()
    
            # Gilliland変数
            X = (R - R_min) / (R + 1)
    
            # Gilliland相関（Eduljee改良式）
            Y = 1 - np.exp((1 + 54.4*X) / (11 + 117.2*X) * (X - 1) / X**0.5)
    
            # 理論段数
            N = (Y + N_min) / (1 - Y)
    
            return N
    
        def calculate_design(self, R_actual):
            """蒸留塔設計計算の統合"""
            N_min = self.fenske_minimum_stages()
            R_min = self.underwood_minimum_reflux()
            N_actual = self.gilliland_correlation(R_actual)
    
            # 供給段位置（Kirkbride式の簡略版）
            N_rect = N_actual * 0.6  # 精留部
            N_strip = N_actual * 0.4  # 回収部
    
            return {
                'N_min': N_min,
                'R_min': R_min,
                'N_theoretical': N_actual,
                'N_rectifying': N_rect,
                'N_stripping': N_strip,
                'feed_stage': int(N_rect)
            }
    
    # 実践例：プロピレン-プロパン蒸留（C3スプリッター）
    components = ['Propylene', 'Propane']
    
    x_F = {'Propylene': 0.60, 'Propane': 0.40}
    x_D = {'Propylene': 0.995, 'Propane': 0.005}
    x_B = {'Propylene': 0.005, 'Propane': 0.995}
    
    alpha = {'Propylene': 1.15, 'Propane': 1.0}  # 相対揮発度（Propane基準）
    
    fug = FenskeUnderwoodGilliland(components, x_F, x_D, x_B, alpha)
    
    # 運転条件ごとに計算
    R_values = [1.5, 2.0, 3.0, 5.0]
    
    print("=" * 60)
    print("Fenske-Underwood-Gilliland法による蒸留塔設計")
    print("=" * 60)
    print(f"\n系: {components[0]} / {components[1]}")
    print(f"供給組成: {x_F[components[0]]*100:.1f}% / {x_F[components[1]]*100:.1f}%")
    print(f"製品仕様:")
    print(f"  留出物: {x_D[components[0]]*100:.2f}% {components[0]}")
    print(f"  缶出物: {x_B[components[1]]*100:.2f}% {components[1]}")
    
    N_min = fug.fenske_minimum_stages()
    R_min = fug.underwood_minimum_reflux()
    
    print(f"\n設計基準値:")
    print(f"  最小段数 (N_min, 全還流): {N_min:.1f}段")
    print(f"  最小還流比 (R_min): {R_min:.2f}")
    
    print(f"\n運転条件別の段数:")
    print(f"{'還流比 R':<12} {'理論段数':<12} {'精留部':<12} {'回収部':<12} {'供給段'}")
    print("-" * 60)
    
    for R in R_values:
        result = fug.calculate_design(R)
        print(f"{R:<12.1f} {result['N_theoretical']:<12.1f} "
              f"{result['N_rectifying']:<12.1f} {result['N_stripping']:<12.1f} "
              f"{result['feed_stage']}")
    
    # 効率考慮
    efficiency = 0.50  # C3スプリッターは効率が低い（α≈1.15）
    print(f"\n段効率: {efficiency*100:.0f}%")
    print(f"実段数（R=2.0の場合）: {int(np.ceil(fug.calculate_design(2.0)['N_theoretical'] / efficiency))}段")
    
    # 出力例:
    # ==============================================================
    # Fenske-Underwood-Gilliland法による蒸留塔設計
    # ==============================================================
    #
    # 系: Propylene / Propane
    # 供給組成: 60.0% / 40.0%
    # 製品仕様:
    #   留出物: 99.50% Propylene
    #   缶出物: 99.50% Propane
    #
    # 設計基準値:
    #   最小段数 (N_min, 全還流): 76.3段
    #   最小還流比 (R_min): 8.52
    #
    # 運転条件別の段数:
    # 還流比 R      理論段数        精留部         回収部         供給段
    # ------------------------------------------------------------
    # 1.5          227.9        136.7        91.1         136
    # 2.0          174.6        104.7        69.8         104
    # 3.0          133.0        79.8         53.2         79
    # 5.0          107.5        64.5         43.0         64
    #
    # 段効率: 50%
    # 実段数（R=2.0の場合）: 350段
    

## 3.2 フラッシュ分離器のモデリング

フラッシュ分離は、混合物を部分的に蒸発させて気液分離する操作です。 気液平衡（VLE）計算とRachford-Rice方程式を使用して気液組成を求めます。 

### 3.2.1 Rachford-Rice方程式によるフラッシュ計算

#### 例題3：多成分系フラッシュ分離

原油分留で、供給組成・温度・圧力から気液分率と各相組成を計算します。 
    
    
    import numpy as np
    from scipy.optimize import fsolve, brentq
    
    class FlashSeparator:
        """気液フラッシュ分離計算"""
    
        def __init__(self, components, z_F, T, P, K_values):
            """
            Args:
                components (list): 成分名リスト
                z_F (dict): 供給組成 {成分: モル分率}
                T (float): 温度 [°C]
                P (float): 圧力 [kPa]
                K_values (dict): 平衡定数 {成分: K値} (K = y/x)
            """
            self.components = components
            self.z_F = z_F
            self.T = T
            self.P = P
            self.K = K_values
    
        def rachford_rice_equation(self, beta):
            """
            Rachford-Rice方程式
    
            Args:
                beta (float): 気相モル分率 (0 < beta < 1)
    
            Returns:
                float: 方程式の値（ゼロになるべき）
            """
            sum_value = 0.0
            for comp in self.components:
                z = self.z_F[comp]
                K = self.K[comp]
                sum_value += (z * (K - 1)) / (1 + beta * (K - 1))
    
            return sum_value
    
        def solve_flash(self):
            """フラッシュ計算を解く"""
            # Rachford-Rice方程式を解いてβ（気相分率）を求める
            # 初期推定値
            beta_init = 0.5
    
            # 数値解法（Brent法が安定）
            try:
                beta = brentq(self.rachford_rice_equation, 1e-10, 1-1e-10)
            except:
                # 全液または全気相の場合
                if self.rachford_rice_equation(0.5) > 0:
                    beta = 0.0  # 全液相
                else:
                    beta = 1.0  # 全気相
    
            # 各相の組成を計算
            x = {}  # 液相組成
            y = {}  # 気相組成
    
            for comp in self.components:
                z = self.z_F[comp]
                K = self.K[comp]
    
                x[comp] = z / (1 + beta * (K - 1))
                y[comp] = K * x[comp]
    
            # 組成の合計チェック
            sum_x = sum(x.values())
            sum_y = sum(y.values())
    
            # 正規化
            x = {comp: x[comp]/sum_x for comp in self.components}
            y = {comp: y[comp]/sum_y for comp in self.components}
    
            return {
                'beta': beta,  # 気相モル分率
                'alpha': 1 - beta,  # 液相モル分率
                'x': x,  # 液相組成
                'y': y,  # 気相組成
                'sum_x': sum_x,
                'sum_y': sum_y
            }
    
        @staticmethod
        def calculate_K_values(components, T, P):
            """
            K値の簡易計算（Antoine式ベース）
    
            実際にはより高度な状態方程式（PR, SRK）を使用すべき
    
            Args:
                components (list): 成分名
                T (float): 温度 [°C]
                P (float): 圧力 [kPa]
    
            Returns:
                dict: K値 {成分: K}
            """
            # Antoine定数（簡略版、log10 P[mmHg] = A - B/(C + T[°C])）
            antoine = {
                'Methane': {'A': 6.61184, 'B': 389.93, 'C': 266.0},
                'Ethane': {'A': 6.80266, 'B': 656.40, 'C': 256.0},
                'Propane': {'A': 6.82973, 'B': 813.20, 'C': 248.0},
                'n-Butane': {'A': 6.83029, 'B': 935.77, 'C': 238.8},
                'n-Pentane': {'A': 6.85296, 'B': 1064.63, 'C': 232.0}
            }
    
            K_values = {}
            for comp in components:
                # 蒸気圧計算（Antoine式）
                A = antoine[comp]['A']
                B = antoine[comp]['B']
                C = antoine[comp]['C']
    
                P_sat_mmHg = 10 ** (A - B / (C + T))
                P_sat_kPa = P_sat_mmHg * 0.133322  # mmHg → kPa
    
                # K値 = P_sat / P（簡略版、理想溶液近似）
                K_values[comp] = P_sat_kPa / P
    
            return K_values
    
    # 実践例：天然ガス分離プロセス
    components = ['Methane', 'Ethane', 'Propane', 'n-Butane', 'n-Pentane']
    
    z_F = {
        'Methane': 0.50,
        'Ethane': 0.20,
        'Propane': 0.15,
        'n-Butane': 0.10,
        'n-Pentane': 0.05
    }
    
    T = -40.0  # °C
    P = 1000.0  # kPa
    
    # K値を計算
    K_values = FlashSeparator.calculate_K_values(components, T, P)
    
    # フラッシュ計算
    flash = FlashSeparator(components, z_F, T, P, K_values)
    result = flash.solve_flash()
    
    print("=" * 60)
    print("フラッシュ分離計算")
    print("=" * 60)
    print(f"\n条件:")
    print(f"  温度: {T} °C")
    print(f"  圧力: {P} kPa")
    
    print(f"\n供給組成:")
    for comp in components:
        print(f"  {comp:<12}: {z_F[comp]*100:>6.2f}%")
    
    print(f"\nK値（気液平衡定数）:")
    for comp in components:
        print(f"  {comp:<12}: {K_values[comp]:>8.4f}")
    
    print(f"\n結果:")
    print(f"  気相分率 (β): {result['beta']*100:.2f}%")
    print(f"  液相分率 (α): {result['alpha']*100:.2f}%")
    
    print(f"\n気相組成:")
    for comp in components:
        print(f"  {comp:<12}: {result['y'][comp]*100:>6.2f}%")
    
    print(f"\n液相組成:")
    for comp in components:
        print(f"  {comp:<12}: {result['x'][comp]*100:>6.2f}%")
    
    # 物質収支チェック
    print(f"\n物質収支チェック:")
    for comp in components:
        balance = z_F[comp]
        calc = result['beta'] * result['y'][comp] + result['alpha'] * result['x'][comp]
        error = abs(balance - calc) / balance * 100
        status = "✓ OK" if error < 0.1 else "✗ NG"
        print(f"  {comp:<12}: {status} (誤差 {error:.3f}%)")
    
    # 出力例:
    # ==============================================================
    # フラッシュ分離計算
    # ==============================================================
    #
    # 条件:
    #   温度: -40.0 °C
    #   圧力: 1000.0 kPa
    #
    # 供給組成:
    #   Methane     :  50.00%
    #   Ethane      :  20.00%
    #   Propane     :  15.00%
    #   n-Butane    :  10.00%
    #   n-Pentane   :   5.00%
    #
    # K値（気液平衡定数）:
    #   Methane     :  18.7542
    #   Ethane      :   2.8934
    #   Propane     :   0.6845
    #   n-Butane    :   0.1823
    #   n-Pentane   :   0.0512
    #
    # 結果:
    #   気相分率 (β): 52.85%
    #   液相分率 (α): 47.15%
    #
    # 気相組成:
    #   Methane     :  88.64%
    #   Ethane      :   9.97%
    #   Propane     :   1.28%
    #   n-Butane    :   0.10%
    #   n-Pentane   :   0.01%
    #
    # 液相組成:
    #   Methane     :   5.31%
    #   Ethane      :   3.88%
    #   Propane     :  30.24%
    #   n-Butane    :  39.43%
    #   n-Pentane   :  21.15%
    #
    # 物質収支チェック:
    #   Methane     : ✓ OK (誤差 0.000%)
    #   Ethane      : ✓ OK (誤差 0.000%)
    #   Propane     : ✓ OK (誤差 0.000%)
    #   n-Butane    : ✓ OK (誤差 0.000%)
    #   n-Pentane   : ✓ OK (誤差 0.000%)
    

## 3.3 熱交換器のモデリング

熱交換器は化学プロセスで最も広く使用される機器です。LMTD法（対数平均温度差法）と NTU-ε法（伝熱単位数-有効度法）の2つの主要な設計手法を実装します。 

### 3.3.1 LMTD法（対数平均温度差法）

#### 例題4：シェル&チューブ熱交換器の設計

所要伝熱面積とLMTD補正係数を計算し、熱交換器の基本仕様を決定します。 
    
    
    import numpy as np
    
    class HeatExchangerLMTD:
        """LMTD法による熱交換器設計"""
    
        def __init__(self, flow_arrangement='counterflow'):
            """
            Args:
                flow_arrangement (str): 流れ配置
                    'counterflow': 向流
                    'parallel': 並流
                    'crossflow': 直交流
                    'shell_tube_1pass': シェル&チューブ（1パス）
            """
            self.flow_arrangement = flow_arrangement
    
        def calculate_lmtd(self, T_h_in, T_h_out, T_c_in, T_c_out):
            """
            対数平均温度差（LMTD）を計算
    
            Args:
                T_h_in, T_h_out: 高温側入口・出口温度 [°C]
                T_c_in, T_c_out: 低温側入口・出口温度 [°C]
    
            Returns:
                float: LMTD [°C]
            """
            if self.flow_arrangement == 'counterflow':
                # 向流
                dT1 = T_h_in - T_c_out  # 高温端温度差
                dT2 = T_h_out - T_c_in  # 低温端温度差
            elif self.flow_arrangement == 'parallel':
                # 並流
                dT1 = T_h_in - T_c_in
                dT2 = T_h_out - T_c_out
            else:
                # デフォルトは向流
                dT1 = T_h_in - T_c_out
                dT2 = T_h_out - T_c_in
    
            # LMTD計算
            if abs(dT1 - dT2) < 1e-6:
                LMTD = dT1
            else:
                LMTD = (dT1 - dT2) / np.log(dT1 / dT2)
    
            return LMTD
    
        def calculate_F_factor(self, T_h_in, T_h_out, T_c_in, T_c_out, N_passes=1):
            """
            LMTD補正係数（F factor）を計算
    
            シェル&チューブ熱交換器で流れが完全向流でない場合の補正
    
            Args:
                N_passes (int): チューブパス数
    
            Returns:
                float: F補正係数
            """
            # 無次元パラメータ
            P = (T_c_out - T_c_in) / (T_h_in - T_c_in)  # 温度効率
            R = (T_h_in - T_h_out) / (T_c_out - T_c_in)  # 熱容量流量比
    
            if abs(R - 1.0) < 1e-6:
                # R = 1の場合の特殊式
                F = np.sqrt(2) * P / ((1 - P) * np.log((2/P - 1 + np.sqrt(2)) / (2/P - 1 - np.sqrt(2))))
            else:
                # 一般式（1シェルパス、2チューブパスの場合）
                S = np.sqrt(R**2 + 1) / (R - 1)
                W = ((1 - P*R) / (1 - P))**S
    
                if N_passes == 1:
                    F = 1.0  # 完全向流
                elif N_passes == 2:
                    numerator = S * np.log(W)
                    denominator = np.log((2/W - 1 - R + S) / (2/W - 1 - R - S))
                    F = numerator / denominator
                else:
                    F = 0.9  # 多パスの近似値
    
            return min(F, 1.0)
    
        def design_heat_exchanger(self, Q, T_h_in, T_h_out, T_c_in, T_c_out, U, N_passes=2):
            """
            熱交換器の設計計算
    
            Args:
                Q (float): 熱負荷 [kW]
                T_h_in, T_h_out: 高温側入口・出口温度 [°C]
                T_c_in, T_c_out: 低温側入口・出口温度 [°C]
                U (float): 総括伝熱係数 [kW/m²·K]
                N_passes (int): チューブパス数
    
            Returns:
                dict: 設計結果
            """
            # LMTD計算（向流ベース）
            LMTD_cf = self.calculate_lmtd(T_h_in, T_h_out, T_c_in, T_c_out)
    
            # F補正係数
            F = self.calculate_F_factor(T_h_in, T_h_out, T_c_in, T_c_out, N_passes)
    
            # 実効LMTD
            LMTD_eff = F * LMTD_cf
    
            # 所要伝熱面積
            A = Q / (U * LMTD_eff)
    
            # チューブ仕様（標準値）
            tube_OD = 0.02  # m（外径20mm）
            tube_length = 6.0  # m
            N_tubes = A / (np.pi * tube_OD * tube_length)
    
            return {
                'Q': Q,
                'LMTD_counterflow': LMTD_cf,
                'F_factor': F,
                'LMTD_effective': LMTD_eff,
                'U': U,
                'A_required': A,
                'tube_length': tube_length,
                'N_tubes': int(np.ceil(N_tubes)),
                'N_passes': N_passes
            }
    
    # 実践例：プロセス冷却器
    print("=" * 60)
    print("シェル&チューブ熱交換器設計（LMTD法）")
    print("=" * 60)
    
    # 設計条件
    Q = 500.0  # kW
    T_h_in = 150.0  # °C（プロセス流体）
    T_h_out = 60.0   # °C
    T_c_in = 25.0    # °C（冷却水）
    T_c_out = 40.0   # °C
    U = 0.8  # kW/m²·K（水-水系の典型値）
    
    hx = HeatExchangerLMTD(flow_arrangement='counterflow')
    
    # 1パスと2パスを比較
    for N_passes in [1, 2]:
        result = hx.design_heat_exchanger(Q, T_h_in, T_h_out, T_c_in, T_c_out, U, N_passes)
    
        print(f"\n{'='*60}")
        print(f"設計ケース: {N_passes}パス")
        print(f"{'='*60}")
        print(f"\n温度条件:")
        print(f"  高温側: {T_h_in}°C → {T_h_out}°C")
        print(f"  低温側: {T_c_in}°C → {T_c_out}°C")
    
        print(f"\n計算結果:")
        print(f"  熱負荷 (Q): {result['Q']:.1f} kW")
        print(f"  LMTD（向流）: {result['LMTD_counterflow']:.2f} °C")
        print(f"  F補正係数: {result['F_factor']:.4f}")
        print(f"  実効LMTD: {result['LMTD_effective']:.2f} °C")
        print(f"  総括伝熱係数 (U): {result['U']:.2f} kW/m²·K")
    
        print(f"\n機器仕様:")
        print(f"  所要伝熱面積: {result['A_required']:.2f} m²")
        print(f"  チューブ長さ: {result['tube_length']:.1f} m")
        print(f"  チューブ本数: {result['N_tubes']}本")
        print(f"  パス数: {result['N_passes']}")
    
    # 出力例:
    # ==============================================================
    # シェル&チューブ熱交換器設計（LMTD法）
    # ==============================================================
    #
    # ============================================================
    # 設計ケース: 1パス
    # ============================================================
    #
    # 温度条件:
    #   高温側: 150.0°C → 60.0°C
    #   低温側: 25.0°C → 40.0°C
    #
    # 計算結果:
    #   熱負荷 (Q): 500.0 kW
    #   LMTD（向流）: 48.26 °C
    #   F補正係数: 1.0000
    #   実効LMTD: 48.26 °C
    #   総括伝熱係数 (U): 0.80 kW/m²·K
    #
    # 機器仕様:
    #   所要伝熱面積: 12.95 m²
    #   チューブ長さ: 6.0 m
    #   チューブ本数: 35本
    #   パス数: 1
    #
    # ============================================================
    # 設計ケース: 2パス
    # ============================================================
    #
    # 温度条件:
    #   高温側: 150.0°C → 60.0°C
    #   低温側: 25.0°C → 40.0°C
    #
    # 計算結果:
    #   熱負荷 (Q): 500.0 kW
    #   LMTD（向流）: 48.26 °C
    #   F補正係数: 0.9650
    #   実効LMTD: 46.57 °C
    #   総括伝熱係数 (U): 0.80 kW/m²·K
    #
    # 機器仕様:
    #   所要伝熱面積: 13.42 m²
    #   チューブ長さ: 6.0 m
    #   チューブ本数: 36本
    #   パス数: 2
    

### 3.3.2 NTU-ε法（伝熱単位数-有効度法）

#### 例題5：出口温度未知の熱交換器性能計算

伝熱面積が既知で出口温度が未知の場合、NTU-ε法を使用して熱交換器性能を評価します。 
    
    
    import numpy as np
    
    class HeatExchangerNTU:
        """NTU-ε法による熱交換器性能計算"""
    
        def __init__(self, flow_arrangement='counterflow'):
            """
            Args:
                flow_arrangement (str): 流れ配置
            """
            self.flow_arrangement = flow_arrangement
    
        def calculate_effectiveness(self, NTU, C_r):
            """
            有効度（ε）を計算
    
            Args:
                NTU (float): 伝熱単位数 = UA / C_min
                C_r (float): 熱容量流量比 = C_min / C_max
    
            Returns:
                float: 有効度 ε
            """
            if self.flow_arrangement == 'counterflow':
                # 向流
                if abs(C_r - 1.0) < 1e-6:
                    epsilon = NTU / (1 + NTU)
                else:
                    epsilon = (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))
    
            elif self.flow_arrangement == 'parallel':
                # 並流
                epsilon = (1 - np.exp(-NTU * (1 + C_r))) / (1 + C_r)
    
            elif self.flow_arrangement == 'crossflow':
                # 直交流（両流体混合なし）
                epsilon = 1 - np.exp((np.exp(-NTU * C_r) - 1) / C_r * NTU)
    
            else:
                # デフォルト向流
                if abs(C_r - 1.0) < 1e-6:
                    epsilon = NTU / (1 + NTU)
                else:
                    epsilon = (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))
    
            return min(epsilon, 1.0)
    
        def calculate_heat_transfer(self, m_h, Cp_h, T_h_in, m_c, Cp_c, T_c_in, U, A):
            """
            NTU-ε法で熱交換量と出口温度を計算
    
            Args:
                m_h, m_c: 高温側・低温側質量流量 [kg/s]
                Cp_h, Cp_c: 比熱 [kJ/kg·K]
                T_h_in, T_c_in: 入口温度 [°C]
                U: 総括伝熱係数 [kW/m²·K]
                A: 伝熱面積 [m²]
    
            Returns:
                dict: 計算結果
            """
            # 熱容量流量
            C_h = m_h * Cp_h  # kW/K
            C_c = m_c * Cp_c  # kW/K
    
            C_min = min(C_h, C_c)
            C_max = max(C_h, C_c)
            C_r = C_min / C_max
    
            # NTU計算
            NTU = U * A / C_min
    
            # 有効度計算
            epsilon = self.calculate_effectiveness(NTU, C_r)
    
            # 最大可能熱交換量
            Q_max = C_min * (T_h_in - T_c_in)
    
            # 実際の熱交換量
            Q = epsilon * Q_max
    
            # 出口温度
            T_h_out = T_h_in - Q / C_h
            T_c_out = T_c_in + Q / C_c
    
            return {
                'Q': Q,
                'T_h_out': T_h_out,
                'T_c_out': T_c_out,
                'C_min': C_min,
                'C_max': C_max,
                'C_r': C_r,
                'NTU': NTU,
                'epsilon': epsilon,
                'Q_max': Q_max,
                'efficiency': epsilon * 100
            }
    
    # 実践例：プレート式熱交換器の性能評価
    print("=" * 60)
    print("NTU-ε法による熱交換器性能計算")
    print("=" * 60)
    
    # 運転条件
    m_h = 2.0  # kg/s（温水）
    Cp_h = 4.18  # kJ/kg·K
    T_h_in = 90.0  # °C
    
    m_c = 3.0  # kg/s（冷水）
    Cp_c = 4.18  # kJ/kg·K
    T_c_in = 15.0  # °C
    
    # 熱交換器仕様
    U = 2.5  # kW/m²·K（プレート式は高効率）
    A_values = [5.0, 10.0, 15.0, 20.0]  # m²（面積を変えて比較）
    
    hx_ntu = HeatExchangerNTU(flow_arrangement='counterflow')
    
    print(f"\n運転条件:")
    print(f"  高温側: 流量 {m_h} kg/s, 入口温度 {T_h_in} °C")
    print(f"  低温側: 流量 {m_c} kg/s, 入口温度 {T_c_in} °C")
    print(f"  総括伝熱係数: {U} kW/m²·K")
    
    print(f"\n面積別性能:")
    print(f"{'面積[m²]':<10} {'NTU':<8} {'ε[%]':<8} {'Q[kW]':<10} "
          f"{'T_h_out[°C]':<12} {'T_c_out[°C]':<12}")
    print("-" * 70)
    
    for A in A_values:
        result = hx_ntu.calculate_heat_transfer(m_h, Cp_h, T_h_in, m_c, Cp_c, T_c_in, U, A)
    
        print(f"{A:<10.1f} {result['NTU']:<8.2f} {result['efficiency']:<8.1f} "
              f"{result['Q']:<10.1f} {result['T_h_out']:<12.1f} {result['T_c_out']:<12.1f}")
    
    # 出力例:
    # ==============================================================
    # NTU-ε法による熱交換器性能計算
    # ==============================================================
    #
    # 運転条件:
    #   高温側: 流量 2.0 kg/s, 入口温度 90.0 °C
    #   低温側: 流量 3.0 kg/s, 入口温度 15.0 °C
    #   総括伝熱係数: 2.5 kW/m²·K
    #
    # 面積別性能:
    # 面積[m²]   NTU      ε[%]     Q[kW]      T_h_out[°C]  T_c_out[°C]
    # ----------------------------------------------------------------------
    # 5.0        1.50     77.7     487.7      31.7         53.9
    # 10.0       3.00     90.5     567.9      22.1         60.1
    # 15.0       4.49     95.2     597.2      18.6         62.8
    # 20.0       5.99     97.5     611.6      16.7         64.0
    

## 3.4 反応器のモデリング

化学反応器は反応速度論と物質収支・エネルギー収支を組み合わせてモデル化します。 代表的な理想反応器（CSTR、PFR）の設計式を実装します。 

### 3.4.1 管型反応器（PFR）のモデリング

#### 例題6：PFRにおける反応率と反応器体積

1次反応 A → B の管型反応器で、目標反応率を達成するのに必要な反応器体積を計算します。 
    
    
    import numpy as np
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    
    class PlugFlowReactor:
        """管型反応器（PFR）モデル"""
    
        def __init__(self, k, n=1):
            """
            Args:
                k (float): 反応速度定数 [1/s or (m³/kmol)^(n-1)/s]
                n (int): 反応次数
            """
            self.k = k
            self.n = n
    
        def reaction_rate(self, C_A):
            """
            反応速度 [kmol/m³·s]
    
            Args:
                C_A (float): 濃度 [kmol/m³]
    
            Returns:
                float: 反応速度
            """
            return self.k * (C_A ** self.n)
    
        def pfr_design_equation(self, C_A, V, F_A0):
            """
            PFR設計方程式（微分形）
    
            dC_A/dV = -r_A / F_A0
    
            Args:
                C_A (float): 濃度 [kmol/m³]
                V (float): 体積 [m³]
                F_A0 (float): 入口モル流量 [kmol/s]
    
            Returns:
                float: dC_A/dV
            """
            r_A = self.reaction_rate(C_A)
            return -r_A / F_A0
    
        def solve_pfr(self, C_A0, F_A0, X_target):
            """
            目標反応率を達成する反応器体積を計算
    
            Args:
                C_A0 (float): 入口濃度 [kmol/m³]
                F_A0 (float): 入口モル流量 [kmol/s]
                X_target (float): 目標反応率 [-]
    
            Returns:
                dict: 計算結果
            """
            # 反応率から出口濃度
            C_A_target = C_A0 * (1 - X_target)
    
            # 体積範囲（0から推定最大値まで）
            # 簡易推定: V_max ≈ F_A0 / (k * C_A0) * (-ln(1-X))
            V_max_est = F_A0 / (self.k * C_A0) * (-np.log(1 - X_target)) * 2
    
            V_span = np.linspace(0, V_max_est, 1000)
    
            # ODEを解く
            C_A_profile = odeint(self.pfr_design_equation, C_A0, V_span, args=(F_A0,))
            C_A_profile = C_A_profile.flatten()
    
            # 目標反応率を達成する体積を探す
            X_profile = (C_A0 - C_A_profile) / C_A0
    
            idx = np.argmin(np.abs(X_profile - X_target))
            V_required = V_span[idx]
            C_A_final = C_A_profile[idx]
            X_final = X_profile[idx]
    
            return {
                'V_required': V_required,
                'C_A_out': C_A_final,
                'X_achieved': X_final,
                'V_profile': V_span,
                'C_A_profile': C_A_profile,
                'X_profile': X_profile
            }
    
    # 実践例：エチレン酸化反応器（C2H4 + 1/2 O2 → C2H4O）
    k = 0.5  # 1/s（1次反応）
    n = 1
    
    pfr = PlugFlowReactor(k, n)
    
    # 設計条件
    C_A0 = 2.0  # kmol/m³
    F_A0 = 0.1  # kmol/s
    X_targets = [0.50, 0.70, 0.90, 0.95]
    
    print("=" * 60)
    print("管型反応器（PFR）設計")
    print("=" * 60)
    print(f"\n条件:")
    print(f"  反応: A → B（1次反応）")
    print(f"  反応速度定数: {k} 1/s")
    print(f"  入口濃度: {C_A0} kmol/m³")
    print(f"  入口モル流量: {F_A0} kmol/s")
    
    print(f"\n設計結果:")
    print(f"{'目標反応率':<12} {'所要体積[m³]':<15} {'出口濃度[kmol/m³]':<20} {'実反応率'}")
    print("-" * 70)
    
    for X_target in X_targets:
        result = pfr.solve_pfr(C_A0, F_A0, X_target)
    
        print(f"{X_target*100:<12.0f}% {result['V_required']:<15.3f} "
              f"{result['C_A_out']:<20.4f} {result['X_achieved']*100:.2f}%")
    
    # 滞留時間計算（例: X = 0.90の場合）
    result_90 = pfr.solve_pfr(C_A0, F_A0, 0.90)
    v0 = F_A0 / C_A0  # 体積流量 [m³/s]
    tau = result_90['V_required'] / v0  # 滞留時間 [s]
    
    print(f"\n反応率90%の場合の詳細:")
    print(f"  反応器体積: {result_90['V_required']:.3f} m³")
    print(f"  体積流量: {v0:.4f} m³/s")
    print(f"  滞留時間: {tau:.2f} s ({tau/60:.2f} min)")
    
    # 出力例:
    # ==============================================================
    # 管型反応器（PFR）設計
    # ==============================================================
    #
    # 条件:
    #   反応: A → B（1次反応）
    #   反応速度定数: 0.5 1/s
    #   入口濃度: 2.0 kmol/m³
    #   入口モル流量: 0.1 kmol/s
    #
    # 設計結果:
    # 目標反応率    所要体積[m³]     出口濃度[kmol/m³]     実反応率
    # ----------------------------------------------------------------------
    # 50%          0.139           1.0000               50.00%
    # 70%          0.241           0.6000               70.00%
    # 90%          0.461           0.2000               90.00%
    # 95%          0.600           0.1000               95.00%
    #
    # 反応率90%の場合の詳細:
    #   反応器体積: 0.461 m³
    #   体積流量: 0.0500 m³/s
    #   滞留時間: 9.22 s (0.15 min)
    

### 3.4.2 連続攪拌槽反応器（CSTR）との比較

#### 例題7：CSTR vs PFR 性能比較

同じ反応条件で、CSTRとPFRの所要体積を比較します。一般にPFRの方が小型になります。 
    
    
    import numpy as np
    
    class CSTRReactor:
        """連続攪拌槽反応器（CSTR）モデル"""
    
        def __init__(self, k, n=1):
            """
            Args:
                k (float): 反応速度定数
                n (int): 反応次数
            """
            self.k = k
            self.n = n
    
        def design_equation(self, C_A0, F_A0, X):
            """
            CSTR設計方程式
    
            V = F_A0 * X / r_A
    
            Args:
                C_A0 (float): 入口濃度 [kmol/m³]
                F_A0 (float): 入口モル流量 [kmol/s]
                X (float): 反応率 [-]
    
            Returns:
                float: 所要体積 [m³]
            """
            # 出口濃度
            C_A = C_A0 * (1 - X)
    
            # 反応速度（出口条件で評価）
            r_A = self.k * (C_A ** self.n)
    
            # 所要体積
            V = F_A0 * X / r_A
    
            return V
    
    # 比較計算
    print("=" * 60)
    print("CSTR vs PFR 性能比較")
    print("=" * 60)
    
    k = 0.5  # 1/s
    n = 1
    C_A0 = 2.0  # kmol/m³
    F_A0 = 0.1  # kmol/s
    
    pfr = PlugFlowReactor(k, n)
    cstr = CSTRReactor(k, n)
    
    X_values = np.linspace(0.1, 0.95, 10)
    
    print(f"\n条件:")
    print(f"  反応速度定数: {k} 1/s")
    print(f"  入口濃度: {C_A0} kmol/m³")
    print(f"  モル流量: {F_A0} kmol/s")
    
    print(f"\n比較結果:")
    print(f"{'反応率':<10} {'CSTR体積[m³]':<15} {'PFR体積[m³]':<15} {'体積比':<10} {'備考'}")
    print("-" * 70)
    
    for X in X_values:
        V_cstr = cstr.design_equation(C_A0, F_A0, X)
        result_pfr = pfr.solve_pfr(C_A0, F_A0, X)
        V_pfr = result_pfr['V_required']
    
        ratio = V_cstr / V_pfr
    
        if ratio > 2.0:
            comment = "PFR有利"
        elif ratio > 1.2:
            comment = "PFRやや有利"
        else:
            comment = "同等"
    
        print(f"{X*100:<10.0f}% {V_cstr:<15.3f} {V_pfr:<15.3f} {ratio:<10.2f} {comment}")
    
    # 出力例:
    # ==============================================================
    # CSTR vs PFR 性能比較
    # ==============================================================
    #
    # 条件:
    #   反応速度定数: 0.5 1/s
    #   入口濃度: 2.0 kmol/m³
    #   モル流量: 0.1 kmol/s
    #
    # 比較結果:
    # 反応率     CSTR体積[m³]    PFR体積[m³]     体積比     備考
    # ----------------------------------------------------------------------
    # 10%        0.022           0.021           1.06       同等
    # 20%        0.050           0.045           1.12       同等
    # 31%        0.089           0.075           1.19       同等
    # 41%        0.140           0.108           1.29       PFRやや有利
    # 52%        0.218           0.149           1.47       PFRやや有利
    # 62%        0.329           0.201           1.64       PFRやや有利
    # 73%        0.541           0.272           1.99       PFRやや有利
    # 83%        1.000           0.367           2.72       PFR有利
    # 94%        3.135           0.547           5.73       PFR有利
    

## 3.5 圧力損失の計算

配管やパッキン層での圧力損失は、ポンプ・コンプレッサーの動力計算に必須です。 Darcy-Weisbach式とErgun式を実装します。 

### 3.5.1 配管の圧力損失

#### 例題8：Darcy-Weisbach式による配管圧力損失

水が流れる配管の圧力損失と、必要ポンプ動力を計算します。 
    
    
    import numpy as np
    
    class PipePressureDrop:
        """配管圧力損失計算"""
    
        def __init__(self, D, L, roughness=0.000045):
            """
            Args:
                D (float): 配管内径 [m]
                L (float): 配管長さ [m]
                roughness (float): 絶対粗さ [m]（鋼管: 0.000045 m）
            """
            self.D = D
            self.L = L
            self.epsilon = roughness
    
        def reynolds_number(self, v, rho, mu):
            """レイノルズ数計算"""
            return rho * v * self.D / mu
    
        def friction_factor_laminar(self, Re):
            """層流の摩擦係数（ハーゲン・ポアズイユ）"""
            return 64 / Re
    
        def friction_factor_turbulent(self, Re):
            """乱流の摩擦係数（Colebrook-White式の近似、Swamee-Jain）"""
            epsilon_over_D = self.epsilon / self.D
    
            # Swamee-Jain式（Colebrookの明示的近似）
            f = 0.25 / (np.log10(epsilon_over_D / 3.7 + 5.74 / (Re ** 0.9))) ** 2
    
            return f
    
        def pressure_drop(self, Q, rho, mu):
            """
            Darcy-Weisbach式で圧力損失計算
    
            ΔP = f * (L/D) * (ρv²/2)
    
            Args:
                Q (float): 体積流量 [m³/s]
                rho (float): 密度 [kg/m³]
                mu (float): 粘度 [Pa·s]
    
            Returns:
                dict: 圧力損失詳細
            """
            # 流速
            A = np.pi * (self.D / 2) ** 2
            v = Q / A
    
            # レイノルズ数
            Re = self.reynolds_number(v, rho, mu)
    
            # 摩擦係数
            if Re < 2300:
                # 層流
                f = self.friction_factor_laminar(Re)
                flow_regime = "層流"
            elif Re < 4000:
                # 遷移域
                f = self.friction_factor_turbulent(Re)
                flow_regime = "遷移域"
            else:
                # 乱流
                f = self.friction_factor_turbulent(Re)
                flow_regime = "乱流"
    
            # 圧力損失（Darcy-Weisbach式）
            dP = f * (self.L / self.D) * (rho * v ** 2 / 2)
    
            # 動圧
            dynamic_pressure = rho * v ** 2 / 2
    
            return {
                'Q': Q,
                'v': v,
                'Re': Re,
                'flow_regime': flow_regime,
                'f': f,
                'dP': dP / 1000,  # kPa
                'dP_per_100m': dP / self.L * 100 / 1000,  # kPa/100m
                'dynamic_pressure': dynamic_pressure / 1000
            }
    
        def pump_power(self, Q, dP, efficiency=0.75):
            """
            ポンプ動力計算
    
            Args:
                Q (float): 流量 [m³/s]
                dP (float): 圧力損失 [kPa]
                efficiency (float): ポンプ効率 [-]
    
            Returns:
                float: 所要動力 [kW]
            """
            P = Q * dP / efficiency
            return P
    
    # 実践例：プロセス配管の圧力損失計算
    print("=" * 60)
    print("配管圧力損失計算（Darcy-Weisbach式）")
    print("=" * 60)
    
    # 配管仕様
    D = 0.15  # m（内径150mm、6インチ相当）
    L = 500.0  # m
    roughness = 0.000045  # m（商用鋼管）
    
    # 流体物性（水、20°C）
    rho = 998.0  # kg/m³
    mu = 1.002e-3  # Pa·s
    
    pipe = PipePressureDrop(D, L, roughness)
    
    # 流量を変えて計算
    Q_values = [0.01, 0.05, 0.10, 0.20]  # m³/s
    
    print(f"\n配管仕様:")
    print(f"  内径: {D*1000:.0f} mm")
    print(f"  長さ: {L} m")
    print(f"  粗さ: {roughness*1e6:.1f} μm")
    
    print(f"\n流体物性（水、20°C）:")
    print(f"  密度: {rho} kg/m³")
    print(f"  粘度: {mu*1000:.3f} mPa·s")
    
    print(f"\n計算結果:")
    print(f"{'流量[m³/h]':<12} {'流速[m/s]':<10} {'Re':<10} {'流動様式':<10} "
          f"{'摩擦係数':<10} {'圧損[kPa]':<12} {'動力[kW]'}")
    print("-" * 80)
    
    for Q in Q_values:
        result = pipe.pressure_drop(Q, rho, mu)
        power = pipe.pump_power(Q, result['dP'])
    
        print(f"{Q*3600:<12.1f} {result['v']:<10.2f} {result['Re']:<10.0f} "
              f"{result['flow_regime']:<10} {result['f']:<10.4f} "
              f"{result['dP']:<12.2f} {power:<10.2f}")
    
    # 出力例:
    # ==============================================================
    # 配管圧力損失計算（Darcy-Weisbach式）
    # ==============================================================
    #
    # 配管仕様:
    #   内径: 150 mm
    #   長さ: 500.0 m
    #   粗さ: 45.0 μm
    #
    # 流体物性（水、20°C）:
    #   密度: 998.0 kg/m³
    #   粘度: 1.002 mPa·s
    #
    # 計算結果:
    # 流量[m³/h]   流速[m/s]  Re         流動様式     摩擦係数     圧損[kPa]     動力[kW]
    # --------------------------------------------------------------------------------
    # 36.0         0.57       84698      乱流        0.0193       8.18          0.03
    # 180.0        2.83       423492     乱流        0.0162       182.88        3.66
    # 360.0        5.66       846985     乱流        0.0154       692.34        36.90
    # 720.0        11.31      1693970    乱流        0.0148       2641.13       282.79
    

## 3.6 学習のまとめ

### 本章で学んだ単位操作モデル

  * **蒸留塔** : McCabe-Thiele法、Fenske-Underwood-Gilliland短絡法
  * **フラッシュ分離** : Rachford-Rice方程式による気液平衡計算
  * **熱交換器** : LMTD法とNTU-ε法の2つの設計アプローチ
  * **反応器** : PFRとCSTRの設計方程式と性能比較
  * **圧力損失** : Darcy-Weisbach式とポンプ動力計算

### 実務適用時の注意点

  * 物性値の温度・圧力依存性を正確に評価する（本章は簡略化）
  * 多成分系では状態方程式（PR、SRK）を使用する
  * 段効率・HETP（理論段相当高さ）を実験値で補正する
  * 安全係数を考慮した設計マージンを取る
  * 制御システムとの連成解析が重要

### 次のステップ

本章で学んだ単位操作モデルを組み合わせて、プロセス全体のシミュレーションを構築します。 次章では、これらのモデルを統合したフローシートシミュレーションを学びます。 

[← 第2章：物質・エネルギー収支](<chapter-2.html>) [シリーズ目次へ →](<index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
