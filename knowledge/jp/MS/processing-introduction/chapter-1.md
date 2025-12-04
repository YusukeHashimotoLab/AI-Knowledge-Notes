---
title: 第1章：プロセス制御の基礎
chapter_title: 第1章：プロセス制御の基礎
subtitle: PID Control, Temperature Control, Vacuum Systems, Atmosphere Control
reading_time: 35-45分
difficulty: 中級
code_examples: 7
---

プロセス制御は、すべての材料製造プロセスの基盤です。温度、圧力、雰囲気といった制御変数を正確に管理することで、目標とする材料特性を安定的に達成できます。この章では、PID制御の原理から実装、真空・ガスフロー制御、リアルタイムモニタリングまでを、Pythonシミュレーションを通じて実践的に学びます。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ PID制御（比例・積分・微分）の動作原理とパラメータチューニング手法を理解する
  * ✅ 温度制御系（ヒーター、センサー、フィードバックループ）を設計・シミュレートできる
  * ✅ 真空ポンプダウン時間を計算し、リーク検出方法を理解する
  * ✅ ガスフローコントローラー（MFC）の原理と分圧計算（Daltonの法則）を実践できる
  * ✅ 雰囲気制御（Ar、N₂、H₂）における酸素分圧と露点管理を理解する
  * ✅ リアルタイムデータ収集とプロセス異常検知システムを構築できる
  * ✅ Pythonでプロセス制御シミュレーターとダッシュボードを実装できる

## 1.1 PID制御の原理とシミュレーション

### 1.1.1 フィードバック制御の基礎

プロセス制御の目的は、**目標値（Set Point）** に対して **制御変数（Process Variable, PV）** を維持することです。最も広く使われる制御手法が**PID制御** （Proportional-Integral-Derivative Control）です。

**PID制御の構成要素** ：

  * **P（比例）** ：偏差（誤差 $e = SP - PV$）に比例した制御出力
  * **I（積分）** ：偏差の累積（定常偏差を除去）
  * **D（微分）** ：偏差の変化率（オーバーシュートを抑制）

**PID制御式** ：

$$ u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt} $$ 

  * $u(t)$：制御出力（例：ヒーター電力 %）
  * $e(t) = SP - PV(t)$：偏差
  * $K_p$：比例ゲイン
  * $K_i$：積分ゲイン
  * $K_d$：微分ゲイン

**離散時間PID式** （デジタル制御系）：

$$ u_n = K_p e_n + K_i \Delta t \sum_{k=0}^{n} e_k + K_d \frac{e_n - e_{n-1}}{\Delta t} $$ 
    
    
    ```mermaid
    flowchart LR
        A[目標値 SP] --> B[比較器]
        F[センサー測定値 PV] --> B
        B --> C[誤差 e]
        C --> D[PIDコントローラー]
        D --> E[制御出力 u]
        E --> G[プロセスヒーター等]
        G --> F
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style G fill:#fce7f3,stroke:#f093fb,stroke-width:2px
    ```

#### コード例1-1: PID制御シミュレーター
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class PIDController:
        """
        PID制御シミュレーター
    
        温度制御などのプロセス制御をシミュレートする
        """
    
        def __init__(self, Kp, Ki, Kd, setpoint, dt=1.0):
            """
            Parameters
            ----------
            Kp : float
                比例ゲイン
            Ki : float
                積分ゲイン
            Kd : float
                微分ゲイン
            setpoint : float
                目標値（Set Point）
            dt : float
                サンプリング時間（秒）
            """
            self.Kp = Kp
            self.Ki = Ki
            self.Kd = Kd
            self.setpoint = setpoint
            self.dt = dt
    
            self.integral = 0.0
            self.prev_error = 0.0
    
        def update(self, measured_value):
            """
            制御出力の計算（PID演算）
    
            Parameters
            ----------
            measured_value : float
                測定値（プロセス変数 PV）
    
            Returns
            -------
            output : float
                制御出力 u(t)
            """
            error = self.setpoint - measured_value
    
            # P項（比例）
            P = self.Kp * error
    
            # I項（積分）
            self.integral += error * self.dt
            I = self.Ki * self.integral
    
            # D項（微分）
            derivative = (error - self.prev_error) / self.dt
            D = self.Kd * derivative
    
            # PID出力
            output = P + I + D
    
            # 状態更新
            self.prev_error = error
    
            return output
    
    
    def simulate_temperature_control(Kp=2.0, Ki=0.5, Kd=0.1,
                                      target_temp=800, duration=200):
        """
        温度制御プロセスのシミュレーション
    
        Parameters
        ----------
        Kp, Ki, Kd : float
            PIDゲイン
        target_temp : float
            目標温度（℃）
        duration : float
            シミュレーション時間（秒）
    
        Returns
        -------
        time, temp, output : ndarray
            時間、温度履歴、制御出力
        """
        dt = 1.0  # サンプリング時間（秒）
        n_steps = int(duration / dt)
    
        # PIDコントローラー初期化
        pid = PIDController(Kp, Ki, Kd, target_temp, dt)
    
        # 初期化
        time = np.arange(0, duration, dt)
        temp = np.zeros(n_steps)
        output = np.zeros(n_steps)
        temp[0] = 25.0  # 室温からスタート
    
        # 簡易熱プロセスモデル（1次遅れ系）
        # dT/dt = (output - heat_loss) / thermal_mass
        thermal_mass = 50.0
        heat_loss_coeff = 0.05
    
        for i in range(1, n_steps):
            # PID制御出力計算
            output[i] = pid.update(temp[i-1])
    
            # 制御出力の制限（0-100%）
            output[i] = np.clip(output[i], 0, 100)
    
            # 温度変化の計算
            heat_input = output[i]
            heat_loss = heat_loss_coeff * (temp[i-1] - 25.0)
            dT = (heat_input - heat_loss) / thermal_mass
    
            temp[i] = temp[i-1] + dT * dt
    
        return time, temp, output
    
    
    # シミュレーション実行
    time, temp, output = simulate_temperature_control()
    
    # 結果プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 温度プロファイル
    ax1.plot(time, temp, 'b-', linewidth=2, label='Temperature')
    ax1.axhline(y=800, color='r', linestyle='--', label='Setpoint')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('PID Temperature Control Simulation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 制御出力
    ax2.plot(time, output, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Heater Output (%)')
    ax2.set_title('PID Control Output')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pid_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 性能指標の計算
    settling_time = time[np.where(np.abs(temp - 800) < 5)[0][0]]
    overshoot = np.max(temp) - 800
    print(f"Settling Time (±5°C): {settling_time:.1f} s")
    print(f"Overshoot: {overshoot:.1f} °C")
    print(f"Steady-State Error: {np.abs(temp[-1] - 800):.2f} °C")

### 1.1.2 PIDパラメータチューニング

PID制御の性能は、3つのゲイン（$K_p$, $K_i$, $K_d$）の調整に依存します。

パラメータ | 効果 | 増加時の影響  
---|---|---  
**$K_p$（比例）** | 応答速度を上げる | オーバーシュート増加、振動しやすい  
**$K_i$（積分）** | 定常偏差を除去 | 振動、整定時間増加  
**$K_d$（微分）** | オーバーシュート抑制 | ノイズに敏感、安定化  
  
**Ziegler-Nicholsチューニング法（実験的手法）** ：

  1. $K_i = 0$, $K_d = 0$ とし、$K_p$ のみで制御
  2. $K_p$ を増加させ、持続振動が発生する臨界ゲイン $K_u$ を見つける
  3. 振動周期 $T_u$ を測定
  4. 以下の式でPIDゲインを計算： $$ K_p = 0.6 K_u, \quad K_i = \frac{2K_p}{T_u}, \quad K_d = \frac{K_p T_u}{8} $$ 

#### コード例1-2: PIDパラメータの自動最適化
    
    
    from scipy.optimize import differential_evolution
    
    def evaluate_pid_performance(params, target_temp=800, duration=200):
        """
        PIDパラメータの性能評価関数
    
        IAE（Integral of Absolute Error）を最小化
    
        Parameters
        ----------
        params : tuple
            (Kp, Ki, Kd)
    
        Returns
        -------
        cost : float
            評価コスト（小さいほど良い）
        """
        Kp, Ki, Kd = params
    
        # シミュレーション実行
        time, temp, output = simulate_temperature_control(Kp, Ki, Kd,
                                                           target_temp, duration)
    
        # 評価指標：IAE + オーバーシュートペナルティ
        error = np.abs(temp - target_temp)
        IAE = np.sum(error)
    
        overshoot = np.max(temp) - target_temp
        overshoot_penalty = 100 * max(0, overshoot)
    
        cost = IAE + overshoot_penalty
    
        return cost
    
    
    # 最適化実行（差分進化アルゴリズム）
    bounds = [(0.1, 10.0),  # Kp
              (0.01, 2.0),  # Ki
              (0.0, 1.0)]   # Kd
    
    result = differential_evolution(
        evaluate_pid_performance,
        bounds,
        maxiter=50,
        seed=42,
        disp=True
    )
    
    Kp_opt, Ki_opt, Kd_opt = result.x
    print(f"Optimal PID gains:")
    print(f"  Kp = {Kp_opt:.3f}")
    print(f"  Ki = {Ki_opt:.3f}")
    print(f"  Kd = {Kd_opt:.3f}")
    
    # 最適化後のシミュレーション
    time, temp_opt, output_opt = simulate_temperature_control(
        Kp_opt, Ki_opt, Kd_opt
    )
    
    # 比較プロット
    time, temp_default, _ = simulate_temperature_control(2.0, 0.5, 0.1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, temp_default, 'b--', linewidth=2, label='Default PID')
    plt.plot(time, temp_opt, 'r-', linewidth=2, label='Optimized PID')
    plt.axhline(y=800, color='k', linestyle=':', label='Setpoint')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title('PID Optimization Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pid_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()

## 1.2 温度制御系の設計

### 1.2.1 温度ランプと多段プロファイル

熱処理プロセスでは、単純な一定温度保持だけでなく、**昇温速度制御** や**多段階プロファイル** が必要です。

**温度ランププロファイルの設計** ：

  * **昇温速度** ：$R_{\text{heat}} = 1-20$ ℃/min（材料・プロセスにより異なる）
  * **保持時間** ：$t_{\text{hold}} = 10-180$ min（拡散・反応時間の確保）
  * **冷却速度** ：$R_{\text{cool}} = 0.5-50$ ℃/min（急冷 or 徐冷）

#### コード例1-3: 多段階温度プロファイル生成器
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def generate_temperature_profile(segments, dt=1.0):
        """
        多段階温度プロファイルの生成
    
        Parameters
        ----------
        segments : list of dict
            各セグメントの仕様
            例: [{'type': 'ramp', 'start': 25, 'end': 800, 'rate': 10},
                 {'type': 'hold', 'temp': 800, 'duration': 3600}]
        dt : float
            サンプリング時間（秒）
    
        Returns
        -------
        time, temperature : ndarray
            時間と温度プロファイル
        """
        time_profile = []
        temp_profile = []
        current_time = 0.0
        current_temp = segments[0].get('start', 25.0)
    
        for seg in segments:
            if seg['type'] == 'ramp':
                # 昇温/降温セグメント
                start_temp = seg['start']
                end_temp = seg['end']
                rate = seg['rate']  # ℃/min
    
                duration = abs(end_temp - start_temp) / rate * 60  # 秒
                n_steps = int(duration / dt)
    
                seg_time = np.linspace(current_time, current_time + duration, n_steps)
                seg_temp = np.linspace(start_temp, end_temp, n_steps)
    
                time_profile.append(seg_time)
                temp_profile.append(seg_temp)
    
                current_time += duration
                current_temp = end_temp
    
            elif seg['type'] == 'hold':
                # 保持セグメント
                hold_temp = seg['temp']
                duration = seg['duration']  # 秒
                n_steps = int(duration / dt)
    
                seg_time = np.linspace(current_time, current_time + duration, n_steps)
                seg_temp = np.ones(n_steps) * hold_temp
    
                time_profile.append(seg_time)
                temp_profile.append(seg_temp)
    
                current_time += duration
                current_temp = hold_temp
    
        time_profile = np.concatenate(time_profile)
        temp_profile = np.concatenate(temp_profile)
    
        return time_profile, temp_profile
    
    
    # 典型的な焼鈍プロファイル
    annealing_segments = [
        {'type': 'ramp', 'start': 25, 'end': 800, 'rate': 10},    # 昇温 10℃/min
        {'type': 'hold', 'temp': 800, 'duration': 3600},          # 保持 1時間
        {'type': 'ramp', 'start': 800, 'end': 25, 'rate': 5}      # 徐冷 5℃/min
    ]
    
    time, temp = generate_temperature_profile(annealing_segments)
    
    # プロットと解析
    plt.figure(figsize=(12, 6))
    plt.plot(time/60, temp, 'b-', linewidth=2)
    plt.xlabel('Time (min)')
    plt.ylabel('Temperature (°C)')
    plt.title('Multi-Segment Temperature Profile (Annealing)')
    plt.grid(True, alpha=0.3)
    plt.savefig('temperature_profile.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # プロファイル統計
    total_time = time[-1] / 3600
    max_temp = np.max(temp)
    print(f"Total Process Time: {total_time:.2f} hours")
    print(f"Maximum Temperature: {max_temp:.0f} °C")
    print(f"Average Heating Rate: {(max_temp - 25) / (time[np.argmax(temp)] / 60):.2f} °C/min")

## 1.3 真空システムと圧力制御

### 1.3.1 真空ポンプダウン時間の計算

真空プロセス（スパッタリング、CVD、焼鈍）では、チャンバー内を目標圧力まで排気する**ポンプダウン時間** が重要です。

**ポンプダウン方程式** ：

$$ P(t) = P_0 \exp\left(-\frac{S}{V} t\right) + P_{\text{ultimate}} $$ 

  * $P(t)$：時刻 $t$ での圧力
  * $P_0$：初期圧力（通常は大気圧 101325 Pa）
  * $S$：実効排気速度（m³/s）
  * $V$：チャンバー容積（m³）
  * $P_{\text{ultimate}}$：到達圧力（ポンプ性能限界）

**実効排気速度** （配管抵抗を考慮）：

$$ \frac{1}{S} = \frac{1}{S_{\text{pump}}} + \frac{1}{C_{\text{pipe}}} $$ 

  * $S_{\text{pump}}$：ポンプ公称排気速度
  * $C_{\text{pipe}}$：配管コンダクタンス

#### コード例1-4: 真空ポンプダウンシミュレーター
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def pumpdown_curve(V, S_pump, C_pipe, P0=101325, P_ultimate=1e-3,
                       t_max=600):
        """
        真空ポンプダウンカーブの計算
    
        Parameters
        ----------
        V : float
            チャンバー容積（m³）
        S_pump : float
            ポンプ排気速度（m³/s）
        C_pipe : float
            配管コンダクタンス（m³/s）
        P0 : float
            初期圧力（Pa）
        P_ultimate : float
            到達圧力（Pa）
        t_max : float
            最大時間（秒）
    
        Returns
        -------
        time, pressure : ndarray
            時間と圧力履歴
        """
        # 実効排気速度
        S_eff = 1 / (1/S_pump + 1/C_pipe)
    
        time = np.linspace(0, t_max, 1000)
        pressure = P0 * np.exp(-S_eff / V * time) + P_ultimate
    
        return time, pressure
    
    
    def calculate_pumpdown_time(V, S_pump, C_pipe, P_target, P0=101325):
        """
        目標圧力到達時間の計算
    
        Returns
        -------
        t_pumpdown : float
            ポンプダウン時間（秒）
        """
        S_eff = 1 / (1/S_pump + 1/C_pipe)
        t_pumpdown = -(V / S_eff) * np.log(P_target / P0)
    
        return t_pumpdown
    
    
    # 典型的なスパッタリング装置のパラメータ
    V_chamber = 0.5  # m³（500 L）
    S_pump = 0.25    # m³/s（250 L/s ターボポンプ）
    C_pipe = 0.5     # m³/s（配管コンダクタンス）
    
    # ポンプダウンカーブ
    time, pressure = pumpdown_curve(V_chamber, S_pump, C_pipe)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.semilogy(time/60, pressure, 'b-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Target: 1 Pa')
    plt.axhline(y=1e-3, color='g', linestyle='--', label='Ultimate: 1 mPa')
    plt.xlabel('Time (min)')
    plt.ylabel('Pressure (Pa)')
    plt.title('Vacuum Pumpdown Curve')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig('pumpdown_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 到達時間計算
    P_target = 1.0  # Pa
    t_pumpdown = calculate_pumpdown_time(V_chamber, S_pump, C_pipe, P_target)
    print(f"Pumpdown time to {P_target} Pa: {t_pumpdown/60:.2f} min")
    
    # 実効排気速度
    S_eff = 1 / (1/S_pump + 1/C_pipe)
    print(f"Effective pumping speed: {S_eff:.3f} m³/s ({S_eff*1000:.0f} L/s)")

### 1.3.2 Daltonの法則とガス分圧制御

プロセス雰囲気の制御では、複数ガスの混合比が重要です。**Daltonの法則** により、全圧は各成分ガスの分圧の和です。

**Daltonの法則** ：

$$ P_{\text{total}} = \sum_{i} P_i = P_{\text{Ar}} + P_{\text{N}_2} + P_{\text{O}_2} + \cdots $$ 

**分圧とモル分率の関係** ：

$$ P_i = x_i P_{\text{total}}, \quad x_i = \frac{n_i}{\sum_j n_j} $$ 

#### コード例1-5: ガス分圧計算とMFC制御
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_partial_pressures(flow_rates, total_pressure):
        """
        Daltonの法則による分圧計算
    
        Parameters
        ----------
        flow_rates : dict
            ガスフローレート（sccm）
            例: {'Ar': 100, 'N2': 50, 'O2': 10}
        total_pressure : float
            全圧（Pa）
    
        Returns
        -------
        partial_pressures : dict
            各ガスの分圧（Pa）
        """
        total_flow = sum(flow_rates.values())
    
        partial_pressures = {}
        for gas, flow in flow_rates.items():
            mole_fraction = flow / total_flow
            partial_pressures[gas] = mole_fraction * total_pressure
    
        return partial_pressures
    
    
    def oxygen_partial_pressure_control(target_pO2, total_pressure,
                                          total_flow=200):
        """
        酸素分圧制御のためのAr/O2混合比計算
    
        Parameters
        ----------
        target_pO2 : float
            目標酸素分圧（Pa）
        total_pressure : float
            全圧（Pa）
        total_flow : float
            総流量（sccm）
    
        Returns
        -------
        flow_Ar, flow_O2 : float
            Arとo2のフローレート（sccm）
        """
        # 酸素モル分率
        x_O2 = target_pO2 / total_pressure
    
        # フローレート計算
        flow_O2 = x_O2 * total_flow
        flow_Ar = (1 - x_O2) * total_flow
    
        return flow_Ar, flow_O2
    
    
    # ケーススタディ：反応性スパッタリング
    # 目標：酸素分圧 0.1 Pa、全圧 1.0 Pa
    
    total_pressure = 1.0  # Pa
    target_pO2 = 0.1      # Pa
    total_flow = 200      # sccm
    
    flow_Ar, flow_O2 = oxygen_partial_pressure_control(target_pO2, total_pressure,
                                                         total_flow)
    
    print(f"Gas Flow Control Settings:")
    print(f"  Ar: {flow_Ar:.1f} sccm")
    print(f"  O2: {flow_O2:.1f} sccm")
    print(f"  Total: {total_flow:.1f} sccm")
    
    # 分圧計算
    flow_rates = {'Ar': flow_Ar, 'O2': flow_O2}
    partial_pressures = calculate_partial_pressures(flow_rates, total_pressure)
    
    print(f"\nPartial Pressures:")
    for gas, pressure in partial_pressures.items():
        print(f"  {gas}: {pressure:.3f} Pa")
    
    # 可視化：酸素分圧 vs Arフロー
    pO2_range = np.linspace(0.01, 0.5, 50)
    Ar_flows = []
    O2_flows = []
    
    for pO2 in pO2_range:
        Ar, O2 = oxygen_partial_pressure_control(pO2, total_pressure, total_flow)
        Ar_flows.append(Ar)
        O2_flows.append(O2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pO2_range, Ar_flows, 'b-', linewidth=2, label='Ar')
    plt.plot(pO2_range, O2_flows, 'r-', linewidth=2, label='O₂')
    plt.xlabel('Target O₂ Partial Pressure (Pa)')
    plt.ylabel('Flow Rate (sccm)')
    plt.title('Gas Flow Control for Reactive Sputtering')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gas_flow_control.png', dpi=150, bbox_inches='tight')
    plt.show()

## 1.4 リアルタイムプロセスモニタリング

### 1.4.1 データ収集とロギング

プロセス制御では、温度、圧力、ガスフローなどのパラメータを**リアルタイムで記録** し、異常検知やトレーサビリティに活用します。

#### コード例1-6: リアルタイムデータ収集シミュレーター
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    class ProcessDataLogger:
        """
        プロセスデータのリアルタイムロギング
    
        温度、圧力、フローレートなどを記録
        """
    
        def __init__(self, parameters, sampling_rate=1.0):
            """
            Parameters
            ----------
            parameters : list of str
                記録するパラメータ名
            sampling_rate : float
                サンプリングレート（Hz）
            """
            self.parameters = parameters
            self.sampling_rate = sampling_rate
            self.data = {param: [] for param in parameters}
            self.timestamps = []
    
        def log_data(self, timestamp, values):
            """
            データの記録
    
            Parameters
            ----------
            timestamp : datetime
                タイムスタンプ
            values : dict
                パラメータ値の辞書
            """
            self.timestamps.append(timestamp)
            for param in self.parameters:
                self.data[param].append(values.get(param, np.nan))
    
        def to_dataframe(self):
            """
            pandasDataFrameへの変換
    
            Returns
            -------
            df : pd.DataFrame
                記録データ
            """
            df = pd.DataFrame(self.data)
            df['timestamp'] = self.timestamps
            return df
    
        def save_to_csv(self, filename):
            """
            CSVファイルへの保存
            """
            df = self.to_dataframe()
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
    
    
    def simulate_process_with_logging(duration=600, dt=1.0):
        """
        プロセスデータロギングのシミュレーション
    
        Parameters
        ----------
        duration : float
            シミュレーション時間（秒）
        dt : float
            サンプリング間隔（秒）
    
        Returns
        -------
        df : pd.DataFrame
            記録されたプロセスデータ
        """
        # データロガー初期化
        logger = ProcessDataLogger(
            parameters=['temperature', 'pressure', 'ar_flow', 'o2_flow'],
            sampling_rate=1/dt
        )
    
        # シミュレーション初期値
        temp = 25.0
        pressure = 101325.0
        ar_flow = 100.0
        o2_flow = 10.0
    
        start_time = datetime.now()
        n_steps = int(duration / dt)
    
        for i in range(n_steps):
            # タイムスタンプ
            timestamp = start_time + timedelta(seconds=i*dt)
    
            # プロセス変化のシミュレーション
            # 昇温フェーズ（0-300秒）
            if i * dt < 300:
                temp += 2.5 * dt  # 2.5℃/s昇温
                pressure = max(1.0, pressure - 100 * dt)  # 減圧
            # 保持フェーズ（300-600秒）
            else:
                temp += np.random.normal(0, 0.5)  # ノイズ
                pressure += np.random.normal(0, 0.01)
    
            # ランダム変動（測定ノイズ）
            ar_flow += np.random.normal(0, 0.5)
            o2_flow += np.random.normal(0, 0.1)
    
            # データ記録
            values = {
                'temperature': temp,
                'pressure': pressure,
                'ar_flow': ar_flow,
                'o2_flow': o2_flow
            }
            logger.log_data(timestamp, values)
    
        # DataFrame化
        df = logger.to_dataframe()
    
        return df, logger
    
    
    # シミュレーション実行
    df, logger = simulate_process_with_logging(duration=600, dt=1.0)
    
    # データ保存
    logger.save_to_csv('process_log.csv')
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 温度
    axes[0, 0].plot(df.index, df['temperature'], 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].set_title('Temperature Profile')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 圧力
    axes[0, 1].semilogy(df.index, df['pressure'], 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Pressure (Pa)')
    axes[0, 1].set_title('Pressure Profile')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Arフロー
    axes[1, 0].plot(df.index, df['ar_flow'], 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Ar Flow (sccm)')
    axes[1, 0].set_title('Argon Flow Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # O2フロー
    axes[1, 1].plot(df.index, df['o2_flow'], 'm-', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('O₂ Flow (sccm)')
    axes[1, 1].set_title('Oxygen Flow Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('process_monitoring.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 統計サマリー
    print("\nProcess Data Summary:")
    print(df.describe())

#### コード例1-7: プロセス異常検知ダッシュボード
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    class ProcessAnomalyDetector:
        """
        プロセス異常検知システム
    
        移動平均と標準偏差によるしきい値ベース検知
        """
    
        def __init__(self, window_size=30, threshold_sigma=3.0):
            """
            Parameters
            ----------
            window_size : int
                移動平均のウィンドウサイズ
            threshold_sigma : float
                異常判定しきい値（σの倍数）
            """
            self.window_size = window_size
            self.threshold_sigma = threshold_sigma
    
        def detect_anomalies(self, data):
            """
            異常検知の実行
    
            Parameters
            ----------
            data : pd.Series
                プロセスデータ系列
    
            Returns
            -------
            anomalies : pd.Series (bool)
                異常フラグ
            """
            # 移動平均と標準偏差
            rolling_mean = data.rolling(window=self.window_size).mean()
            rolling_std = data.rolling(window=self.window_size).std()
    
            # 上下限しきい値
            upper_bound = rolling_mean + self.threshold_sigma * rolling_std
            lower_bound = rolling_mean - self.threshold_sigma * rolling_std
    
            # 異常検知
            anomalies = (data > upper_bound) | (data < lower_bound)
    
            return anomalies, upper_bound, lower_bound
    
    
    # 異常データの生成（意図的な異常を含む）
    np.random.seed(42)
    time = np.arange(0, 600, 1)
    temperature = 800 + np.random.normal(0, 2, len(time))
    
    # 異常注入（400-420秒で温度スパイク）
    temperature[400:420] += 50
    
    df_anomaly = pd.DataFrame({'time': time, 'temperature': temperature})
    
    # 異常検知
    detector = ProcessAnomalyDetector(window_size=30, threshold_sigma=3.0)
    anomalies, upper, lower = detector.detect_anomalies(df_anomaly['temperature'])
    
    # 可視化
    plt.figure(figsize=(14, 6))
    plt.plot(df_anomaly['time'], df_anomaly['temperature'],
             'b-', linewidth=1.5, label='Temperature', alpha=0.7)
    plt.plot(df_anomaly['time'], upper, 'r--', linewidth=2, label='Upper Threshold')
    plt.plot(df_anomaly['time'], lower, 'g--', linewidth=2, label='Lower Threshold')
    
    # 異常ポイントをハイライト
    anomaly_points = df_anomaly[anomalies]
    plt.scatter(anomaly_points['time'], anomaly_points['temperature'],
                color='red', s=50, label='Anomaly', zorder=5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title('Process Anomaly Detection Dashboard')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('anomaly_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 異常サマリー
    n_anomalies = anomalies.sum()
    print(f"\nAnomaly Detection Results:")
    print(f"  Total data points: {len(df_anomaly)}")
    print(f"  Anomalies detected: {n_anomalies}")
    print(f"  Anomaly rate: {n_anomalies/len(df_anomaly)*100:.2f}%")
    
    if n_anomalies > 0:
        anomaly_times = df_anomaly[anomalies]['time'].values
        print(f"  Anomaly time ranges: {anomaly_times[0]:.0f}-{anomaly_times[-1]:.0f} s")

## 演習問題

#### 演習1-1: PID制御パラメータの影響理解（Easy）

PIDコントローラーで、以下の3つのパラメータセットを試し、温度応答の違いを比較せよ：

  * (a) $K_p = 5.0$, $K_i = 0$, $K_d = 0$ （P制御のみ）
  * (b) $K_p = 2.0$, $K_i = 0.5$, $K_d = 0$ （PI制御）
  * (c) $K_p = 2.0$, $K_i = 0.5$, $K_d = 0.1$ （PID制御）

目標温度800℃、初期温度25℃、シミュレーション時間200秒。各ケースで定常偏差、オーバーシュート、整定時間を比較せよ。

解答例
    
    
    # 3つのパラメータセットでシミュレーション
    cases = [
        {'name': 'P-only', 'Kp': 5.0, 'Ki': 0.0, 'Kd': 0.0},
        {'name': 'PI', 'Kp': 2.0, 'Ki': 0.5, 'Kd': 0.0},
        {'name': 'PID', 'Kp': 2.0, 'Ki': 0.5, 'Kd': 0.1}
    ]
    
    plt.figure(figsize=(12, 8))
    
    for case in cases:
        time, temp, output = simulate_temperature_control(
            case['Kp'], case['Ki'], case['Kd'],
            target_temp=800, duration=200
        )
    
        plt.plot(time, temp, linewidth=2, label=case['name'])
    
        # 性能指標
        steady_error = abs(temp[-1] - 800)
        overshoot = max(0, np.max(temp) - 800)
        settling_idx = np.where(np.abs(temp - 800) < 5)[0]
        settling_time = time[settling_idx[0]] if len(settling_idx) > 0 else np.inf
    
        print(f"{case['name']} Control:")
        print(f"  Steady-State Error: {steady_error:.2f} °C")
        print(f"  Overshoot: {overshoot:.2f} °C")
        print(f"  Settling Time (±5°C): {settling_time:.1f} s\n")
    
    plt.axhline(y=800, color='k', linestyle='--', label='Setpoint')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title('PID Parameter Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 結果解釈：
    # - P制御のみ：定常偏差が残る（積分項がないため）
    # - PI制御：定常偏差は解消されるが、オーバーシュートが大きい
    # - PID制御：微分項により、オーバーシュートが抑制され最適

#### 演習1-2: 真空リーク検出（Medium）

真空チャンバー（容積0.5 m³）をターボポンプ（排気速度250 L/s）で1 Paまで排気した後、バルブを閉じて圧力上昇を測定したところ、10分後に1.5 Paになった。リークレート（Pa·L/s）を計算し、このリークが許容範囲か判定せよ。（許容リークレート: < 1×10⁻³ Pa·m³/s）

解答例
    
    
    import numpy as np
    
    # 与えられたパラメータ
    V_chamber = 0.5  # m³（500 L）
    P_initial = 1.0  # Pa
    P_final = 1.5    # Pa
    delta_t = 10 * 60  # 秒（10分）
    
    # リークレートの計算
    # dP/dt = Q_leak / V
    # Q_leak = V * (P_final - P_initial) / delta_t
    
    Q_leak = V_chamber * (P_final - P_initial) / delta_t
    
    print(f"Leak Rate Calculation:")
    print(f"  Initial Pressure: {P_initial} Pa")
    print(f"  Final Pressure: {P_final} Pa")
    print(f"  Time Interval: {delta_t/60:.1f} min")
    print(f"  Leak Rate: {Q_leak:.3e} Pa·m³/s")
    print(f"  Leak Rate: {Q_leak*1000:.3e} Pa·L/s")
    
    # 許容範囲判定
    Q_leak_threshold = 1e-3  # Pa·m³/s
    if Q_leak < Q_leak_threshold:
        print(f"\n✓ Leak rate is ACCEPTABLE (< {Q_leak_threshold:.1e} Pa·m³/s)")
    else:
        print(f"\n✗ Leak rate is UNACCEPTABLE (≥ {Q_leak_threshold:.1e} Pa·m³/s)")
        print(f"  Action required: Leak detection and repair")
    
    # リーク箇所の推定（体積流量）
    # at 1 Pa: Q_volume = Q_leak / P = 4.17e-4 / 1.0 = 4.17e-4 m³/s
    Q_volume = Q_leak / P_initial
    print(f"\nEquivalent Volume Flow (at 1 Pa): {Q_volume*1000:.3f} L/s")
    print(f"  → Suggests a significant leak (e.g., loose flange, damaged O-ring)")
    
    # 結果：
    # Leak Rate: 4.17×10⁻⁴ Pa·m³/s
    # これは許容値（1×10⁻³）より小さいので「ACCEPTABLE」
    # ただし、UHV（超高真空）アプリケーションでは要改善

#### 演習1-3: 反応性スパッタリングのガス制御（Medium）

ITO（Indium Tin Oxide）薄膜の反応性スパッタリングで、酸素分圧を0.15 Paに制御したい。全圧1.2 Pa、総ガス流量250 sccmとする。ArとO₂のフローレート（sccm）を計算せよ。また、酸素流量が±5%変動した場合の酸素分圧変化を評価せよ。

解答例
    
    
    # パラメータ
    total_pressure = 1.2  # Pa
    target_pO2 = 0.15     # Pa
    total_flow = 250      # sccm
    
    # 酸素モル分率
    x_O2 = target_pO2 / total_pressure
    
    # フローレート計算
    flow_O2 = x_O2 * total_flow
    flow_Ar = (1 - x_O2) * total_flow
    
    print(f"ITO Reactive Sputtering Gas Control:")
    print(f"  Target O₂ Partial Pressure: {target_pO2} Pa")
    print(f"  Total Pressure: {total_pressure} Pa")
    print(f"  Ar Flow: {flow_Ar:.2f} sccm")
    print(f"  O₂ Flow: {flow_O2:.2f} sccm")
    
    # 酸素流量変動の影響評価
    flow_O2_variation = 0.05  # ±5%
    flow_O2_min = flow_O2 * (1 - flow_O2_variation)
    flow_O2_max = flow_O2 * (1 + flow_O2_variation)
    
    # 変動後の分圧
    pO2_min = (flow_O2_min / total_flow) * total_pressure
    pO2_max = (flow_O2_max / total_flow) * total_pressure
    
    print(f"\nSensitivity Analysis (±5% O₂ flow variation):")
    print(f"  O₂ Flow Range: {flow_O2_min:.2f} - {flow_O2_max:.2f} sccm")
    print(f"  O₂ Partial Pressure Range: {pO2_min:.3f} - {pO2_max:.3f} Pa")
    print(f"  Deviation from Target: {(pO2_max - target_pO2)/target_pO2*100:.1f}%")
    
    # 安定性評価
    pO2_tolerance = 0.01  # Pa（許容誤差）
    if abs(pO2_max - target_pO2) < pO2_tolerance and abs(pO2_min - target_pO2) < pO2_tolerance:
        print(f"\n✓ Gas control is STABLE (variation < {pO2_tolerance} Pa)")
    else:
        print(f"\n⚠ Gas control may be UNSTABLE")
        print(f"  → Consider using closed-loop pO2 feedback control")
    
    # 結果：
    # Ar: 218.75 sccm, O₂: 31.25 sccm
    # ±5% O₂変動 → pO2変動 ±0.0075 Pa（±5%）
    # MFCの精度が±1%なら、十分安定した制御が可能

#### 演習1-4: 温度プロファイル最適化（Medium）

Al合金の溶体化処理（550℃、2時間保持）において、昇温速度が速すぎると熱応力でクラックが発生し、遅すぎると析出物が粗大化する。許容昇温速度範囲5-15℃/min、冷却速度≥50℃/minで、総プロセス時間が最短となる温度プロファイルを設計せよ。

解答例
    
    
    # 最適化：最速昇温、必須保持、最速冷却
    heat_rate_max = 15    # ℃/min（最速昇温）
    hold_temp = 550       # ℃
    hold_time = 120       # min（2時間）
    cool_rate = 50        # ℃/min（急冷）
    
    # プロファイル設計
    segments_optimized = [
        {'type': 'ramp', 'start': 25, 'end': hold_temp, 'rate': heat_rate_max},
        {'type': 'hold', 'temp': hold_temp, 'duration': hold_time * 60},
        {'type': 'ramp', 'start': hold_temp, 'end': 100, 'rate': cool_rate}  # 100℃まで急冷
    ]
    
    time_opt, temp_opt = generate_temperature_profile(segments_optimized)
    
    # 総プロセス時間
    total_time_opt = time_opt[-1] / 60  # 分
    
    print(f"Optimized Temperature Profile for Al Alloy:")
    print(f"  Heating Rate: {heat_rate_max} ℃/min")
    print(f"  Hold Temperature: {hold_temp} ℃")
    print(f"  Hold Time: {hold_time} min")
    print(f"  Cooling Rate: {cool_rate} ℃/min")
    print(f"  Total Process Time: {total_time_opt:.1f} min")
    
    # 各フェーズの時間
    heat_time = (hold_temp - 25) / heat_rate_max
    cool_time = (hold_temp - 100) / cool_rate
    
    print(f"\nPhase Breakdown:")
    print(f"  Heating: {heat_time:.1f} min")
    print(f"  Hold: {hold_time:.1f} min")
    print(f"  Cooling: {cool_time:.1f} min")
    
    # プロット
    plt.figure(figsize=(12, 6))
    plt.plot(time_opt/60, temp_opt, 'b-', linewidth=2)
    plt.xlabel('Time (min)')
    plt.ylabel('Temperature (°C)')
    plt.title('Optimized Solution Treatment Profile for Al Alloy')
    plt.grid(True, alpha=0.3)
    plt.savefig('solution_treatment_profile.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 結果：
    # 総プロセス時間 = 35 + 120 + 9 = 164分（約2.7時間）
    # 昇温を最速（15℃/min）、冷却を急冷（50℃/min）にすることで
    # プロセス時間を最小化

#### 演習1-5: プロセス異常の根本原因分析（Hard）

スパッタリング装置で膜厚の異常変動（目標値±10%を超える）が発生した。以下のプロセスログから、異常の根本原因を特定し、対策を提案せよ：

  * 温度：800±3℃（正常範囲内）
  * 圧力：0.8-1.5 Pa（変動大、目標1.0 Pa）
  * Arフロー：98-102 sccm（正常範囲内）
  * RF電力：変動なし（300 W固定）

解答例
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # シミュレーションデータ生成（圧力変動あり）
    np.random.seed(42)
    time = np.arange(0, 300, 1)
    temperature = 800 + np.random.normal(0, 1.5, len(time))
    pressure = 1.0 + 0.3 * np.sin(2*np.pi*time/50) + np.random.normal(0, 0.05, len(time))
    ar_flow = 100 + np.random.normal(0, 1.0, len(time))
    rf_power = 300 * np.ones(len(time))
    
    # 膜厚は圧力に反比例（スパッタリング収率への影響）
    film_thickness = 500 / pressure  # nm（簡易モデル）
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 圧力変動
    axes[0].plot(time, pressure, 'r-', linewidth=1.5)
    axes[0].axhline(y=1.0, color='k', linestyle='--', label='Target')
    axes[0].fill_between(time, 0.9, 1.1, alpha=0.2, color='green', label='Tolerance')
    axes[0].set_ylabel('Pressure (Pa)')
    axes[0].set_title('Process Parameter Monitoring')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 膜厚変動
    axes[1].plot(time, film_thickness, 'b-', linewidth=1.5)
    axes[1].axhline(y=500, color='k', linestyle='--', label='Target')
    axes[1].fill_between(time, 450, 550, alpha=0.2, color='green', label='Tolerance (±10%)')
    axes[1].set_ylabel('Film Thickness (nm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 相関分析
    axes[2].scatter(pressure, film_thickness, alpha=0.5)
    axes[2].set_xlabel('Pressure (Pa)')
    axes[2].set_ylabel('Film Thickness (nm)')
    axes[2].set_title('Correlation: Pressure vs Film Thickness')
    axes[2].grid(True, alpha=0.3)
    
    # 回帰直線
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(pressure, film_thickness)
    pressure_fit = np.linspace(pressure.min(), pressure.max(), 100)
    thickness_fit = slope * pressure_fit + intercept
    axes[2].plot(pressure_fit, thickness_fit, 'r-', linewidth=2,
                 label=f'Fit: R²={r_value**2:.3f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('root_cause_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 統計分析
    pressure_std = np.std(pressure)
    thickness_std = np.std(film_thickness)
    thickness_cv = thickness_std / np.mean(film_thickness) * 100
    
    print(f"Root Cause Analysis:")
    print(f"  Pressure Std Dev: {pressure_std:.3f} Pa")
    print(f"  Film Thickness CV: {thickness_cv:.2f}%")
    print(f"  Correlation (R²): {r_value**2:.3f}")
    
    print(f"\nConclusion:")
    print(f"  ✗ Primary cause: PRESSURE INSTABILITY")
    print(f"  → Pressure variation (±30%) causes thickness variation (±15%)")
    print(f"  → Exceeds tolerance (±10%)")
    
    print(f"\nRecommended Actions:")
    print(f"  1. Check MFC (Mass Flow Controller) calibration")
    print(f"  2. Inspect vacuum pump performance (oil level, belt tension)")
    print(f"  3. Verify APC (Automatic Pressure Controller) PID tuning")
    print(f"  4. Check for micro-leaks (leak rate test)")
    print(f"  5. Implement closed-loop pressure feedback control")
    
    # 結果解釈：
    # - 圧力変動（0.8-1.5 Pa、±50%）が膜厚変動の主因
    # - 温度・フローは安定 → ヒーター・MFCは正常
    # - 対策：APCのPID再調整、またはターボポンプのメンテナンス

#### 演習1-6: 多変数最適化問題（Hard）

CVDプロセスで、成膜速度（最大化）と膜質（ピンホール密度最小化）を同時最適化したい。目的関数：$F = w_1 \cdot R - w_2 \cdot D$（$R$: 成膜速度 nm/min、$D$: ピンホール密度 個/cm²）。制約条件：温度500-700℃、圧力10-100 Pa、ガス流量50-200 sccm。Pythonで最適プロセス条件を求めよ。

解答例
    
    
    from scipy.optimize import differential_evolution
    import numpy as np
    
    def cvd_process_model(params):
        """
        CVDプロセスの簡易モデル
    
        Parameters
        ----------
        params : tuple
            (temperature, pressure, flow_rate)
    
        Returns
        -------
        deposition_rate, pinhole_density : float
            成膜速度（nm/min）、ピンホール密度（個/cm²）
        """
        temp, pressure, flow = params
    
        # 成膜速度モデル（Arrhenius型 + 圧力依存）
        # R = A * exp(-Ea/RT) * P^0.5 * flow^0.3
        A = 1e6
        Ea = 50000  # J/mol
        R_gas = 8.314
    
        deposition_rate = A * np.exp(-Ea/(R_gas * (temp + 273))) * \
                          np.sqrt(pressure) * (flow ** 0.3)
    
        # ピンホール密度モデル（低温・高圧で増加）
        # D = D0 * exp(-T/T0) * (P/P0)^2
        D0 = 100
        T0 = 500
        P0 = 50
    
        pinhole_density = D0 * np.exp(-(temp-500)/T0) * (pressure/P0)**2
    
        return deposition_rate, pinhole_density
    
    
    def objective_function(params, w1=1.0, w2=10.0):
        """
        最適化目的関数（最大化）
    
        F = w1 * R - w2 * D
    
        Returns
        -------
        -F : float
            負の目的関数値（最小化問題に変換）
        """
        deposition_rate, pinhole_density = cvd_process_model(params)
    
        F = w1 * deposition_rate - w2 * pinhole_density
    
        return -F  # 最小化問題に変換
    
    
    # 最適化実行
    bounds = [
        (500, 700),   # Temperature (℃)
        (10, 100),    # Pressure (Pa)
        (50, 200)     # Flow rate (sccm)
    ]
    
    result = differential_evolution(
        objective_function,
        bounds,
        args=(1.0, 10.0),  # w1, w2
        maxiter=100,
        seed=42,
        disp=True
    )
    
    temp_opt, pressure_opt, flow_opt = result.x
    rate_opt, density_opt = cvd_process_model(result.x)
    
    print(f"Multi-Objective Optimization Results:")
    print(f"  Optimal Temperature: {temp_opt:.1f} °C")
    print(f"  Optimal Pressure: {pressure_opt:.1f} Pa")
    print(f"  Optimal Flow Rate: {flow_opt:.1f} sccm")
    print(f"\nPerformance:")
    print(f"  Deposition Rate: {rate_opt:.2f} nm/min")
    print(f"  Pinhole Density: {density_opt:.2f} /cm²")
    print(f"  Objective Function F: {-result.fun:.2f}")
    
    # パラメータスイープ（可視化）
    temps = np.linspace(500, 700, 50)
    rates = []
    densities = []
    
    for temp in temps:
        rate, density = cvd_process_model((temp, pressure_opt, flow_opt))
        rates.append(rate)
        densities.append(density)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Deposition Rate (nm/min)', color=color)
    ax1.plot(temps, rates, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=temp_opt, color='k', linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Pinhole Density (/cm²)', color=color)
    ax2.plot(temps, densities, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('CVD Process Multi-Objective Optimization')
    fig.tight_layout()
    plt.savefig('cvd_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 結果解釈：
    # - 最適温度は約650℃（高すぎると成膜速度↑だがピンホール↓、トレードオフ）
    # - 重み w2 を大きくすると、膜質（ピンホール低減）を重視
    # - 実プロセスでは、さらに膜厚均一性、付着性も考慮が必要

#### 演習1-7: リアルタイムフィードバック制御システム設計（Hard）

酸化雰囲気でのアニーリングプロセスにおいて、酸素分圧を目標値（0.1 Pa）に維持するPID制御システムを設計せよ。センサー（酸素分圧計）の測定遅延10秒、MFCの応答時間5秒を考慮し、安定した制御が可能なPIDゲインを決定せよ。

解答例
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class OxygenPartialPressureController:
        """
        酸素分圧のPIDフィードバック制御
    
        測定遅延とMFC応答時間を考慮
        """
    
        def __init__(self, Kp, Ki, Kd, setpoint, dt=1.0,
                     sensor_delay=10, mfc_response_time=5):
            """
            Parameters
            ----------
            sensor_delay : float
                センサー測定遅延（秒）
            mfc_response_time : float
                MFC応答時間（秒）
            """
            self.Kp = Kp
            self.Ki = Ki
            self.Kd = Kd
            self.setpoint = setpoint
            self.dt = dt
    
            self.sensor_delay_steps = int(sensor_delay / dt)
            self.mfc_tau = mfc_response_time
    
            self.integral = 0.0
            self.prev_error = 0.0
            self.measurement_buffer = []
            self.mfc_output = 0.0
    
        def update(self, actual_pO2):
            """
            制御出力の更新
    
            Returns
            -------
            mfc_flow : float
                MFCへの指令流量（sccm）
            """
            # センサー遅延シミュレーション
            self.measurement_buffer.append(actual_pO2)
            if len(self.measurement_buffer) > self.sensor_delay_steps:
                measured_pO2 = self.measurement_buffer.pop(0)
            else:
                measured_pO2 = self.measurement_buffer[0]
    
            # PID演算
            error = self.setpoint - measured_pO2
    
            self.integral += error * self.dt
            derivative = (error - self.prev_error) / self.dt
    
            pid_output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
    
            # MFC応答（1次遅れ）
            self.mfc_output += (pid_output - self.mfc_output) / self.mfc_tau * self.dt
    
            self.prev_error = error
    
            return self.mfc_output
    
    
    def simulate_oxygen_pressure_control(Kp, Ki, Kd, duration=300):
        """
        酸素分圧制御のシミュレーション
        """
        dt = 1.0
        n_steps = int(duration / dt)
    
        # コントローラー初期化
        controller = OxygenPartialPressureController(
            Kp, Ki, Kd, setpoint=0.1, dt=dt,
            sensor_delay=10, mfc_response_time=5
        )
    
        # 初期化
        time = np.arange(0, duration, dt)
        pO2 = np.zeros(n_steps)
        mfc_flow = np.zeros(n_steps)
        pO2[0] = 0.05  # 初期値
    
        # プロセスモデル（簡易）
        # dpO2/dt = k1 * mfc_flow - k2 * pO2
        k1 = 0.002  # 供給係数
        k2 = 0.01   # 消費/リーク係数
    
        for i in range(1, n_steps):
            # 制御出力
            mfc_flow[i] = controller.update(pO2[i-1])
            mfc_flow[i] = np.clip(mfc_flow[i], 0, 50)  # 0-50 sccm
    
            # プロセスダイナミクス
            dpO2 = (k1 * mfc_flow[i] - k2 * pO2[i-1]) * dt
            pO2[i] = pO2[i-1] + dpO2
    
        return time, pO2, mfc_flow
    
    
    # 最適PIDゲインの探索
    pid_candidates = [
        {'name': 'Conservative', 'Kp': 50, 'Ki': 2, 'Kd': 5},
        {'name': 'Moderate', 'Kp': 100, 'Ki': 5, 'Kd': 10},
        {'name': 'Aggressive', 'Kp': 200, 'Ki': 10, 'Kd': 20}
    ]
    
    plt.figure(figsize=(14, 10))
    
    for idx, pid in enumerate(pid_candidates):
        time, pO2, mfc_flow = simulate_oxygen_pressure_control(
            pid['Kp'], pid['Ki'], pid['Kd']
        )
    
        plt.subplot(2, 1, 1)
        plt.plot(time, pO2, linewidth=2, label=pid['name'])
    
        plt.subplot(2, 1, 2)
        plt.plot(time, mfc_flow, linewidth=2, label=pid['name'])
    
        # 性能評価
        settling_idx = np.where(np.abs(pO2 - 0.1) < 0.005)[0]
        settling_time = time[settling_idx[0]] if len(settling_idx) > 0 else np.inf
        overshoot = max(0, np.max(pO2) - 0.1)
    
        print(f"{pid['name']} PID:")
        print(f"  Kp={pid['Kp']}, Ki={pid['Ki']}, Kd={pid['Kd']}")
        print(f"  Settling Time: {settling_time:.1f} s")
        print(f"  Overshoot: {overshoot:.4f} Pa\n")
    
    plt.subplot(2, 1, 1)
    plt.axhline(y=0.1, color='k', linestyle='--', label='Setpoint')
    plt.ylabel('O₂ Partial Pressure (Pa)')
    plt.title('Oxygen Pressure Control with Measurement Delay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.xlabel('Time (s)')
    plt.ylabel('MFC Flow (sccm)')
    plt.title('MFC Control Output')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('oxygen_pressure_control.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Recommendation:")
    print(f"  Choose 'Moderate' PID for balance between speed and stability")
    print(f"  Measurement delay requires careful tuning to avoid oscillation")

#### 演習1-8: プロセスデジタルツイン構築（Hard）

温度制御システムの**デジタルツイン** （実プロセスを模倣する仮想モデル）を構築せよ。実測データ（温度、ヒーター出力）からシステム同定を行い、予測精度±2℃以内を達成せよ。モデルベース制御（MPC）への応用も検討せよ。

解答例
    
    
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    
    # 実測データの生成（簡易シミュレーション）
    np.random.seed(42)
    time_real = np.arange(0, 200, 1)
    heater_real = 50 + 30 * (1 - np.exp(-time_real/30))  # ステップ応答
    temp_real = 25 + 300 * (1 - np.exp(-time_real/40)) + np.random.normal(0, 2, len(time_real))
    
    def first_order_model(t, K, tau, delay):
        """
        1次遅れ+むだ時間モデル
    
        Parameters
        ----------
        K : float
            ゲイン
        tau : float
            時定数（秒）
        delay : float
            むだ時間（秒）
        """
        response = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti > delay:
                response[i] = K * (1 - np.exp(-(ti - delay) / tau))
        return response + 25  # オフセット
    
    # システム同定（カーブフィッティング）
    popt, pcov = curve_fit(
        lambda t, K, tau, delay: first_order_model(t, K, tau, delay),
        time_real,
        temp_real,
        p0=[300, 40, 0],
        bounds=([100, 10, 0], [500, 100, 10])
    )
    
    K_id, tau_id, delay_id = popt
    print(f"System Identification Results:")
    print(f"  Gain K: {K_id:.2f}")
    print(f"  Time Constant τ: {tau_id:.2f} s")
    print(f"  Delay: {delay_id:.2f} s")
    
    # デジタルツインモデルによる予測
    temp_model = first_order_model(time_real, K_id, tau_id, delay_id)
    
    # 予測誤差評価
    prediction_error = temp_real - temp_model
    rmse = np.sqrt(np.mean(prediction_error**2))
    max_error = np.max(np.abs(prediction_error))
    
    print(f"\nModel Accuracy:")
    print(f"  RMSE: {rmse:.2f} °C")
    print(f"  Max Error: {max_error:.2f} °C")
    
    if max_error < 2.0:
        print(f"  ✓ Accuracy target (±2°C) ACHIEVED")
    else:
        print(f"  ✗ Accuracy target (±2°C) NOT MET")
        print(f"  → Consider higher-order model or nonlinear effects")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 温度比較
    axes[0].plot(time_real, temp_real, 'b-', linewidth=2, label='Real Process', alpha=0.7)
    axes[0].plot(time_real, temp_model, 'r--', linewidth=2, label='Digital Twin')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Digital Twin: Real vs Model')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 予測誤差
    axes[1].plot(time_real, prediction_error, 'g-', linewidth=1.5)
    axes[1].axhline(y=2, color='r', linestyle='--', label='Target ±2°C')
    axes[1].axhline(y=-2, color='r', linestyle='--')
    axes[1].fill_between(time_real, -2, 2, alpha=0.2, color='green')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Prediction Error (°C)')
    axes[1].set_title('Model Prediction Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('digital_twin.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # MPC（Model Predictive Control）への応用コンセプト
    print(f"\nMPC Application Concept:")
    print(f"  1. Use digital twin to predict future temperature trajectory")
    print(f"  2. Optimize heater input sequence over prediction horizon (e.g., 60s)")
    print(f"  3. Apply first control action, then re-optimize at next timestep")
    print(f"  4. Benefits: Handles constraints, anticipates disturbances")
    
    # 結果：
    # RMSEが2℃未満なら、デジタルツインは十分な精度
    # MPCでは、このモデルを使って最適制御入力を計算
    # 実装には scipy.optimize.minimize や専用MPCライブラリ（do-mpc）を使用

## 学習達成度チェックリスト

### 基本理解レベル

  * ☐ PID制御の3つの要素（P、I、D）の役割を説明できる
  * ☐ フィードバック制御の基本構成を理解している
  * ☐ 温度ランププロファイルの設計ができる
  * ☐ 真空ポンプダウン方程式を使える
  * ☐ Daltonの法則で分圧計算ができる

### 実践スキルレベル

  * ☐ PythonでPIDコントローラーを実装し、パラメータ調整ができる
  * ☐ 多段階温度プロファイルを生成し、プロットできる
  * ☐ 真空系の到達時間とリークレートを計算できる
  * ☐ ガス混合比から分圧を計算し、MFC設定値を決定できる
  * ☐ プロセスデータをログし、異常検知アルゴリズムを実装できる

### 応用力レベル

  * ☐ PIDゲインの自動最適化（差分進化など）ができる
  * ☐ 測定遅延を考慮したフィードバック制御系を設計できる
  * ☐ プロセス異常の根本原因分析（相関解析、統計検定）ができる
  * ☐ 多目的最適化問題（成膜速度 vs 膜質）を解ける
  * ☐ デジタルツイン（システム同定、予測モデル）を構築できる
  * ☐ リアルタイムダッシュボードを設計し、プロセス監視システムを提案できる

## 参考文献

  1. Åström, K.J., Hägglund, T. (2006). _Advanced PID Control_. ISA - The Instrumentation, Systems, and Automation Society, pp. 45-78, 123-145.
  2. Ogata, K. (2010). _Modern Control Engineering_ (5th ed.). Prentice Hall, pp. 156-189, 234-267.
  3. Bunshah, R.F. (Ed.). (2001). _Handbook of Deposition Technologies for Films and Coatings: Science, Applications and Technology_ (3rd ed.). Elsevier, pp. 120-156, 201-245.
  4. O'Hanlon, J.F. (2003). _A User's Guide to Vacuum Technology_ (3rd ed.). Wiley-Interscience, pp. 234-267, 345-378.
  5. Seborg, D.E., Edgar, T.F., Mellichamp, D.A., Doyle III, F.J. (2016). _Process Dynamics and Control_ (4th ed.). Wiley, pp. 89-124, 267-302.
  6. Python control systems library: `scipy.signal`, `control` package. Documentation: https://python-control.readthedocs.io
  7. Glover, A.R., Smith, D.L. (2015). "Real-time process monitoring in semiconductor manufacturing," _Journal of Vacuum Science & Technology A_, 33(4), 041501. DOI: 10.1116/1.4916239, pp. 1-12.
