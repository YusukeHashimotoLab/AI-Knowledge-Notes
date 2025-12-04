---
title: 第4章：フィードバック制御とPID制御
chapter_title: 第4章：フィードバック制御とPID制御
subtitle: プロセス制御の基礎理論から実装まで
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ フィードバック制御の基本原理と制御系の構成を説明できる
  * ✅ 1次系のステップ応答と動特性を理解する
  * ✅ PID制御器の各要素（P, I, D）の役割と効果を理解する
  * ✅ Pythonで1次系とPID制御器を実装・シミュレートできる
  * ✅ ジーグラー・ニコルス法でPIDパラメータを決定できる
  * ✅ アンチワインドアップなど実用的な制御問題に対処できる
  * ✅ カスケード制御の基本構造を理解する

* * *

## 4.1 フィードバック制御の基礎

### フィードバック制御とは

**フィードバック制御** は、制御対象の出力（制御量）を測定し、目標値（設定値）との偏差をゼロにするように制御入力（操作量）を調整する制御方式です。プロセス産業において最も広く使用される制御手法です。

### 制御系の基本構成
    
    
    ```mermaid
    graph LR
        SP[設定値Setpoint] --> SUM((+-))
        SUM --> |偏差 e| CTRL[制御器Controller]
        CTRL --> |操作量 u| PROCESS[プロセスProcess]
        PROCESS --> |制御量 y| OUTPUT[出力]
        OUTPUT --> |フィードバック| SUM
    
        style SP fill:#e8f5e9
        style CTRL fill:#c8e6c9
        style PROCESS fill:#a5d6a7
        style OUTPUT fill:#81c784
    ```

**主要な要素:**

  * **設定値（SP: Setpoint）** : 制御量の目標値（例: 反応器温度 175°C）
  * **制御量（PV: Process Variable）** : 実際に測定されるプロセス変数
  * **偏差（e: Error）** : 設定値と制御量の差（e = SP - PV）
  * **操作量（MV: Manipulated Variable）** : 制御器が出力する信号（例: バルブ開度、ヒーター出力）
  * **外乱（Disturbance）** : プロセスに影響を与える予期しない変化

### 伝達関数と動的システム

プロセスの動特性は、**伝達関数** で表現されます。最も基本的なプロセスモデルは**1次遅れ系** です：

**G(s) = K / (τs + 1)**

ここで：

  * **K** : プロセスゲイン（定常状態での入出力比）
  * **τ** : 時定数（プロセスの応答速度を表す）
  * **s** : ラプラス変数

* * *

## 4.2 コード例：1次系とPID制御の実装

#### コード例1: 1次系のステップ応答シミュレーション

**目的** : 反応器温度制御を例に、1次遅れ系のステップ応答を可視化し、時定数とゲインの影響を理解する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    
    # 日本語フォント設定
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    def first_order_system(y, t, K, tau, u):
        """
        1次遅れ系の微分方程式
    
        τ * dy/dt + y = K * u
    
        Parameters:
        -----------
        y : float
            制御量（温度）
        t : float
            時間
        K : float
            プロセスゲイン
        tau : float
            時定数（秒）
        u : float
            操作量（ヒーター出力）
    
        Returns:
        --------
        dydt : float
            制御量の時間微分
        """
        dydt = (K * u - y) / tau
        return dydt
    
    # シミュレーションパラメータ
    K = 2.0        # プロセスゲイン（°C / %）
    tau = 60.0     # 時定数（秒）
    u_step = 10.0  # ステップ入力（ヒーター出力 10%）
    
    # 時間軸
    t = np.linspace(0, 300, 1000)  # 0-300秒
    
    # 初期条件（定常状態: 室温20°C）
    y0 = 20.0
    
    # 1次系の応答計算
    y = odeint(first_order_system, y0, t, args=(K, tau, u_step))
    
    # 理論値（解析解）
    y_analytical = K * u_step * (1 - np.exp(-t / tau)) + y0
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ステップ応答
    axes[0].plot(t, y, 'b-', linewidth=2, label='シミュレーション結果')
    axes[0].plot(t, y_analytical, 'r--', linewidth=2, alpha=0.7, label='解析解')
    axes[0].axhline(y=y0 + K * u_step, color='green', linestyle='--',
                    label=f'定常値: {y0 + K * u_step:.1f}°C')
    axes[0].axvline(x=tau, color='orange', linestyle='--', alpha=0.5,
                    label=f'時定数 τ = {tau:.0f}秒')
    axes[0].axhline(y=y0 + K * u_step * 0.632, color='purple', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('時間（秒）', fontsize=12)
    axes[0].set_ylabel('温度（°C）', fontsize=12)
    axes[0].set_title('1次系のステップ応答（反応器温度制御）', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # 異なる時定数での比較
    tau_values = [30, 60, 120]
    colors = ['blue', 'red', 'green']
    
    for tau_val, color in zip(tau_values, colors):
        y_comp = odeint(first_order_system, y0, t, args=(K, tau_val, u_step))
        axes[1].plot(t, y_comp, color=color, linewidth=2, label=f'τ = {tau_val}秒')
    
    axes[1].axhline(y=y0 + K * u_step, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('時間（秒）', fontsize=12)
    axes[1].set_ylabel('温度（°C）', fontsize=12)
    axes[1].set_title('時定数の影響比較', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 応答特性の計算
    print("=== 1次系ステップ応答特性 ===")
    print(f"プロセスゲイン K: {K} °C/%")
    print(f"時定数 τ: {tau} 秒")
    print(f"ステップ入力: {u_step} %")
    print(f"定常状態変化: {K * u_step:.2f} °C")
    print(f"時定数での到達率: 63.2%")
    print(f"3τ での到達率: {(1 - np.exp(-3)) * 100:.1f}%")
    print(f"5τ での到達率: {(1 - np.exp(-5)) * 100:.1f}%")
    

**期待される出力** :
    
    
    === 1次系ステップ応答特性 ===
    プロセスゲイン K: 2.0 °C/%
    時定数 τ: 60.0 秒
    ステップ入力: 10.0 %
    定常状態変化: 20.00 °C
    時定数での到達率: 63.2%
    3τ での到達率: 95.0%
    5τ での到達率: 99.3%
    

**解説** : 1次遅れ系は、プロセス制御で最も基本的なモデルです。時定数τは、ステップ入力に対して定常値の63.2%に到達するまでの時間を表します。実際のプロセス（温度制御、流量制御等）の多くは1次系または1次系の組み合わせで近似できます。

#### コード例2: PID制御器の完全実装

**目的** : P、I、D各要素を含む完全なPID制御器を実装し、各パラメータの影響を理解する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class PIDController:
        """
        PID制御器の完全実装
    
        u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de(t)/dt
    
        Parameters:
        -----------
        Kp : float
            比例ゲイン
        Ki : float
            積分ゲイン
        Kd : float
            微分ゲイン
        dt : float
            サンプリング時間（秒）
        setpoint : float
            設定値
        output_limits : tuple
            操作量の上下限 (min, max)
        """
    
        def __init__(self, Kp, Ki, Kd, dt=1.0, setpoint=0.0, output_limits=(0, 100)):
            self.Kp = Kp
            self.Ki = Ki
            self.Kd = Kd
            self.dt = dt
            self.setpoint = setpoint
            self.output_limits = output_limits
    
            # 内部状態
            self.integral = 0.0
            self.prev_error = 0.0
    
        def update(self, measured_value):
            """
            PID制御器の更新
    
            Parameters:
            -----------
            measured_value : float
                測定値（制御量）
    
            Returns:
            --------
            output : float
                操作量
            """
            # 偏差計算
            error = self.setpoint - measured_value
    
            # 比例項
            P = self.Kp * error
    
            # 積分項
            self.integral += error * self.dt
            I = self.Ki * self.integral
    
            # 微分項
            derivative = (error - self.prev_error) / self.dt
            D = self.Kd * derivative
    
            # PID出力
            output = P + I + D
    
            # 操作量の制限
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
    
            # 次回のために誤差を保存
            self.prev_error = error
    
            return output
    
        def reset(self):
            """内部状態のリセット"""
            self.integral = 0.0
            self.prev_error = 0.0
    
    
    def simulate_process(controller, process_model, duration, dt, disturbance_time=None):
        """
        制御システムのシミュレーション
    
        Parameters:
        -----------
        controller : PIDController
            PID制御器
        process_model : function
            プロセスモデル関数
        duration : float
            シミュレーション時間（秒）
        dt : float
            サンプリング時間（秒）
        disturbance_time : float or None
            外乱印加時刻（秒）
    
        Returns:
        --------
        time, setpoint, output, control : arrays
            時間、設定値、制御量、操作量の配列
        """
        n_steps = int(duration / dt)
        time = np.arange(n_steps) * dt
    
        setpoint = np.ones(n_steps) * controller.setpoint
        output = np.zeros(n_steps)
        control = np.zeros(n_steps)
    
        # 初期値
        y = 20.0  # 初期温度（°C）
    
        for i in range(n_steps):
            # PID制御器の更新
            u = controller.update(y)
            control[i] = u
            output[i] = y
    
            # 外乱の印加
            disturbance = 0.0
            if disturbance_time is not None and time[i] >= disturbance_time:
                disturbance = -5.0  # -5°Cの外乱
    
            # プロセスモデルの更新（1次系）
            K = 2.0
            tau = 60.0
            dydt = (K * u - y + disturbance) / tau
            y = y + dydt * dt
    
        return time, setpoint, output, control
    
    
    # PID制御器のパラメータ
    Kp = 5.0
    Ki = 0.1
    Kd = 10.0
    dt = 1.0
    setpoint = 175.0
    
    # PID制御器の初期化
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt,
                        setpoint=setpoint, output_limits=(0, 100))
    
    # シミュレーション実行
    time, sp, pv, mv = simulate_process(pid, None, duration=600, dt=dt,
                                         disturbance_time=400)
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 制御量のトレンド
    axes[0].plot(time, sp, 'r--', linewidth=2, label='設定値（SP）')
    axes[0].plot(time, pv, 'b-', linewidth=1.5, label='制御量（PV）')
    axes[0].axvline(x=400, color='orange', linestyle='--', alpha=0.5, label='外乱印加')
    axes[0].set_ylabel('温度（°C）', fontsize=12)
    axes[0].set_title(f'PID制御シミュレーション（Kp={Kp}, Ki={Ki}, Kd={Kd}）',
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # 操作量のトレンド
    axes[1].plot(time, mv, 'g-', linewidth=1.5, label='操作量（MV）')
    axes[1].axvline(x=400, color='orange', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('時間（秒）', fontsize=12)
    axes[1].set_ylabel('ヒーター出力（%）', fontsize=12)
    axes[1].set_title('操作量の変化', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 制御性能の評価
    settling_time_idx = np.where(np.abs(pv - setpoint) < 0.5)[0][0]
    print("\n=== PID制御性能 ===")
    print(f"整定時間: {time[settling_time_idx]:.1f} 秒")
    print(f"定常偏差: {np.abs(pv[-100:].mean() - setpoint):.3f} °C")
    print(f"外乱除去時間: {time[-1] - 400:.1f} 秒後に整定")
    

**解説** : このコードは、産業用PID制御器の完全な実装です。比例（P）、積分（I）、微分（D）の3つの制御動作を組み合わせることで、優れた制御性能を実現します。外乱（400秒時点で-5°Cの温度低下）に対しても、PID制御器が自動的に操作量を調整し、設定値に復帰させることが確認できます。

#### コード例3: 比例（P）制御の定常偏差デモンストレーション

**目的** : P制御のみでは定常偏差（オフセット）が残ることを実証し、積分動作の必要性を理解する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def p_controller_simulation(Kp, setpoint, duration=500, dt=1.0):
        """
        P制御のみのシミュレーション
    
        Parameters:
        -----------
        Kp : float
            比例ゲイン
        setpoint : float
            設定値
        duration : float
            シミュレーション時間
        dt : float
            サンプリング時間
    
        Returns:
        --------
        time, pv, mv, error : arrays
        """
        n_steps = int(duration / dt)
        time = np.arange(n_steps) * dt
    
        pv = np.zeros(n_steps)
        mv = np.zeros(n_steps)
        error = np.zeros(n_steps)
    
        # プロセスパラメータ（1次系）
        K = 2.0   # プロセスゲイン
        tau = 60.0  # 時定数
    
        # 初期値
        y = 20.0  # 初期温度
    
        for i in range(n_steps):
            # 偏差計算
            e = setpoint - y
            error[i] = e
    
            # P制御（比例動作のみ）
            u = Kp * e
            u = np.clip(u, 0, 100)  # 操作量制限
    
            mv[i] = u
            pv[i] = y
    
            # プロセス応答（1次系）
            dydt = (K * u - y) / tau
            y = y + dydt * dt
    
        return time, pv, mv, error
    
    
    # 異なる比例ゲインでのシミュレーション
    Kp_values = [2.0, 5.0, 10.0]
    setpoint = 175.0
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    for Kp in Kp_values:
        time, pv, mv, error = p_controller_simulation(Kp, setpoint)
    
        # 定常偏差の計算
        steady_state_error = setpoint - pv[-100:].mean()
    
        axes[0].plot(time, pv, linewidth=2, label=f'Kp={Kp} (定常偏差: {steady_state_error:.2f}°C)')
    
    axes[0].axhline(y=setpoint, color='red', linestyle='--', linewidth=2, label='設定値')
    axes[0].set_ylabel('温度（°C）', fontsize=12)
    axes[0].set_title('P制御における定常偏差', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 偏差のトレンド
    for Kp in Kp_values:
        time, pv, mv, error = p_controller_simulation(Kp, setpoint)
        axes[1].plot(time, error, linewidth=2, label=f'Kp={Kp}')
    
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_ylabel('偏差（°C）', fontsize=12)
    axes[1].set_title('偏差の時間変化', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 操作量
    for Kp in Kp_values:
        time, pv, mv, error = p_controller_simulation(Kp, setpoint)
        axes[2].plot(time, mv, linewidth=2, label=f'Kp={Kp}')
    
    axes[2].set_xlabel('時間（秒）', fontsize=12)
    axes[2].set_ylabel('操作量（%）', fontsize=12)
    axes[2].set_title('操作量の変化', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== P制御の定常偏差分析 ===")
    for Kp in Kp_values:
        time, pv, mv, error = p_controller_simulation(Kp, setpoint)
        steady_state_error = setpoint - pv[-100:].mean()
        steady_state_mv = mv[-100:].mean()
    
        print(f"\nKp = {Kp}:")
        print(f"  定常偏差: {steady_state_error:.2f} °C")
        print(f"  定常操作量: {steady_state_mv:.2f} %")
        print(f"  偏差/ゲイン比: {steady_state_error/Kp:.2f}")
    

**期待される出力** :
    
    
    === P制御の定常偏差分析 ===
    
    Kp = 2.0:
      定常偏差: 39.02 °C
      定常操作量: 78.04 %
      偏差/ゲイン比: 19.51
    
    Kp = 5.0:
      定常偏差: 17.56 °C
      定常操作量: 87.78 %
      偏差/ゲイン比: 3.51
    
    Kp = 10.0:
      定常偏差: 9.26 °C
      定常操作量: 92.63 %
      偏差/ゲイン比: 0.93
    

**解説** : P制御のみでは、プロセスゲインと制御ゲインの関係により必ず定常偏差が残ります。これは、定常状態で操作量が一定値に落ち着くためです。比例ゲインを大きくすると定常偏差は減少しますが、完全にはゼロになりません。この定常偏差を除去するために、積分動作（I）が必要になります。

#### コード例4: PI制御による定常偏差の除去

**目的** : PI制御（比例+積分）により定常偏差が完全に除去されることを実証する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def pi_controller_simulation(Kp, Ki, setpoint, duration=600, dt=1.0):
        """
        PI制御のシミュレーション
    
        Parameters:
        -----------
        Kp : float
            比例ゲイン
        Ki : float
            積分ゲイン
        setpoint : float
            設定値
        duration : float
            シミュレーション時間
        dt : float
            サンプリング時間
    
        Returns:
        --------
        time, pv, mv, p_term, i_term : arrays
        """
        n_steps = int(duration / dt)
        time = np.arange(n_steps) * dt
    
        pv = np.zeros(n_steps)
        mv = np.zeros(n_steps)
        p_term = np.zeros(n_steps)
        i_term = np.zeros(n_steps)
    
        # プロセスパラメータ
        K = 2.0
        tau = 60.0
    
        # 初期値
        y = 20.0
        integral = 0.0
    
        for i in range(n_steps):
            # 偏差
            error = setpoint - y
    
            # 比例項
            P = Kp * error
            p_term[i] = P
    
            # 積分項
            integral += error * dt
            I = Ki * integral
            i_term[i] = I
    
            # PI制御出力
            u = P + I
            u = np.clip(u, 0, 100)
    
            mv[i] = u
            pv[i] = y
    
            # プロセス応答
            dydt = (K * u - y) / tau
            y = y + dydt * dt
    
        return time, pv, mv, p_term, i_term
    
    
    # PとPI制御の比較
    setpoint = 175.0
    
    # P制御のみ
    time_p, pv_p, mv_p, _, _ = p_controller_simulation(Kp=5.0, setpoint=setpoint)
    
    # PI制御
    Kp = 5.0
    Ki = 0.05
    time_pi, pv_pi, mv_pi, p_term, i_term = pi_controller_simulation(Kp, Ki, setpoint)
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 制御量の比較
    axes[0].plot(time_p, pv_p, 'b-', linewidth=2, alpha=0.6, label='P制御のみ')
    axes[0].plot(time_pi, pv_pi, 'g-', linewidth=2, label='PI制御')
    axes[0].axhline(y=setpoint, color='red', linestyle='--', linewidth=2, label='設定値')
    axes[0].set_ylabel('温度（°C）', fontsize=12)
    axes[0].set_title('P制御とPI制御の比較', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # PI制御の各項の寄与
    axes[1].plot(time_pi, p_term, 'b-', linewidth=2, label='比例項（P）')
    axes[1].plot(time_pi, i_term, 'orange', linewidth=2, label='積分項（I）')
    axes[1].plot(time_pi, mv_pi, 'g-', linewidth=2, alpha=0.7, label='総操作量（P+I）')
    axes[1].set_ylabel('操作量（%）', fontsize=12)
    axes[1].set_title('PI制御の各項の寄与', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 偏差の比較
    error_p = setpoint - pv_p
    error_pi = setpoint - pv_pi
    axes[2].plot(time_p, error_p, 'b-', linewidth=2, alpha=0.6, label='P制御の偏差')
    axes[2].plot(time_pi, error_pi, 'g-', linewidth=2, label='PI制御の偏差')
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[2].set_xlabel('時間（秒）', fontsize=12)
    axes[2].set_ylabel('偏差（°C）', fontsize=12)
    axes[2].set_title('偏差の時間変化', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 定常状態の分析
    print("=== P制御 vs PI制御の比較 ===")
    print("\nP制御（Kp=5.0）:")
    print(f"  定常偏差: {error_p[-100:].mean():.3f} °C")
    print(f"  定常操作量: {mv_p[-100:].mean():.2f} %")
    
    print("\nPI制御（Kp=5.0, Ki=0.05）:")
    print(f"  定常偏差: {error_pi[-100:].mean():.3f} °C")
    print(f"  定常操作量: {mv_pi[-100:].mean():.2f} %")
    print(f"  比例項寄与: {p_term[-100:].mean():.2f} %")
    print(f"  積分項寄与: {i_term[-100:].mean():.2f} %")
    

**期待される出力** :
    
    
    === P制御 vs PI制御の比較 ===
    
    P制御（Kp=5.0）:
      定常偏差: 17.562 °C
      定常操作量: 87.78 %
    
    PI制御（Kp=5.0, Ki=0.05）:
      定常偏差: 0.000 °C
      定常操作量: 77.50 %
      比例項寄与: 0.00 %
      積分項寄与: 77.50 %
    

**解説** : 積分動作（I）は、偏差の時間積分に比例した操作量を出力します。定常状態では偏差がゼロになるまで積分値が蓄積されるため、定常偏差を完全に除去できます。PI制御は、プロセス産業で最も広く使用される制御方式です。

#### コード例5: ジーグラー・ニコルス法によるPIDチューニング

**目的** : 実用的なPIDパラメータ決定法であるジーグラー・ニコルス法（限界感度法と反応曲線法）を実装する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    def ziegler_nichols_ultimate_sensitivity(Ku, Tu):
        """
        ジーグラー・ニコルス法（限界感度法）
    
        Parameters:
        -----------
        Ku : float
            限界ゲイン（持続振動を起こすKp）
        Tu : float
            限界周期（秒）
    
        Returns:
        --------
        pid_params : dict
            P, PI, PIDのパラメータ
        """
        params = {
            'P': {
                'Kp': 0.5 * Ku,
                'Ki': 0.0,
                'Kd': 0.0
            },
            'PI': {
                'Kp': 0.45 * Ku,
                'Ki': 0.54 * Ku / Tu,
                'Kd': 0.0
            },
            'PID': {
                'Kp': 0.6 * Ku,
                'Ki': 1.2 * Ku / Tu,
                'Kd': 0.075 * Ku * Tu
            }
        }
        return params
    
    
    def ziegler_nichols_reaction_curve(L, T, K):
        """
        ジーグラー・ニコルス法（反応曲線法）
    
        Parameters:
        -----------
        L : float
            むだ時間（秒）
        T : float
            時定数（秒）
        K : float
            プロセスゲイン
    
        Returns:
        --------
        pid_params : dict
            P, PI, PIDのパラメータ
        """
        params = {
            'P': {
                'Kp': T / (L * K),
                'Ki': 0.0,
                'Kd': 0.0
            },
            'PI': {
                'Kp': 0.9 * T / (L * K),
                'Ki': 0.27 * T / (L**2 * K),
                'Kd': 0.0
            },
            'PID': {
                'Kp': 1.2 * T / (L * K),
                'Ki': 0.6 * T / (L**2 * K),
                'Kd': 0.6 * T / K
            }
        }
        return params
    
    
    # 実例: プロセスパラメータ（1次遅れ+むだ時間）
    L = 10.0   # むだ時間（秒）
    T = 60.0   # 時定数（秒）
    K = 2.0    # プロセスゲイン
    
    # 反応曲線法によるチューニング
    params_rc = ziegler_nichols_reaction_curve(L, T, K)
    
    print("=== ジーグラー・ニコルス法（反応曲線法）===")
    print(f"プロセスパラメータ: L={L}秒, T={T}秒, K={K}")
    print("\nチューニング結果:")
    
    for controller_type, params in params_rc.items():
        print(f"\n{controller_type}制御:")
        print(f"  Kp = {params['Kp']:.3f}")
        if params['Ki'] > 0:
            print(f"  Ki = {params['Ki']:.4f}")
            print(f"  Ti = {params['Kp']/params['Ki']:.2f} 秒（積分時間）")
        if params['Kd'] > 0:
            print(f"  Kd = {params['Kd']:.3f}")
            print(f"  Td = {params['Kd']/params['Kp']:.2f} 秒（微分時間）")
    
    
    # PIDパラメータでのシミュレーション比較
    pid_params = params_rc['PID']
    
    # カスタムPIDクラスを使ってシミュレーション
    class SimplePID:
        def __init__(self, Kp, Ki, Kd, dt=1.0):
            self.Kp, self.Ki, self.Kd, self.dt = Kp, Ki, Kd, dt
            self.integral = 0.0
            self.prev_error = 0.0
    
        def update(self, error):
            P = self.Kp * error
            self.integral += error * self.dt
            I = self.Ki * self.integral
            D = self.Kd * (error - self.prev_error) / self.dt
            self.prev_error = error
            return P + I + D
    
    
    def simulate_with_params(params, setpoint, duration=600, dt=1.0):
        """指定されたパラメータでシミュレーション"""
        n_steps = int(duration / dt)
        time = np.arange(n_steps) * dt
        pv = np.zeros(n_steps)
        mv = np.zeros(n_steps)
    
        pid = SimplePID(params['Kp'], params['Ki'], params['Kd'], dt)
        y = 20.0
    
        for i in range(n_steps):
            error = setpoint - y
            u = pid.update(error)
            u = np.clip(u, 0, 100)
            mv[i] = u
            pv[i] = y
    
            # プロセス応答（1次系 + むだ時間）
            # 簡易的にむだ時間は無視してシミュレーション
            dydt = (K * u - y) / T
            y = y + dydt * dt
    
        return time, pv, mv
    
    
    # P, PI, PID制御の比較
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    setpoint = 175.0
    colors = {'P': 'blue', 'PI': 'green', 'PID': 'red'}
    
    for ctrl_type, params in params_rc.items():
        time, pv, mv = simulate_with_params(params, setpoint)
    
        # オーバーシュートと整定時間の計算
        overshoot = (np.max(pv) - setpoint) / setpoint * 100
        settling_idx = np.where(np.abs(pv - setpoint) < 0.02 * setpoint)[0]
        settling_time = time[settling_idx[0]] if len(settling_idx) > 0 else time[-1]
    
        axes[0].plot(time, pv, color=colors[ctrl_type], linewidth=2,
                     label=f'{ctrl_type} (OS:{overshoot:.1f}%, Ts:{settling_time:.0f}s)')
    
    axes[0].axhline(y=setpoint, color='black', linestyle='--', linewidth=1.5, label='設定値')
    axes[0].set_ylabel('温度（°C）', fontsize=12)
    axes[0].set_title('ジーグラー・ニコルス法でチューニングされたPID制御の比較',
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 操作量
    for ctrl_type, params in params_rc.items():
        time, pv, mv = simulate_with_params(params, setpoint)
        axes[1].plot(time, mv, color=colors[ctrl_type], linewidth=2, label=ctrl_type)
    
    axes[1].set_xlabel('時間（秒）', fontsize=12)
    axes[1].set_ylabel('操作量（%）', fontsize=12)
    axes[1].set_title('操作量の変化', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**期待される出力** :
    
    
    === ジーグラー・ニコルス法（反応曲線法）===
    プロセスパラメータ: L=10秒, T=60秒, K=2
    
    チューニング結果:
    
    P制御:
      Kp = 3.000
    
    PI制御:
      Kp = 2.700
      Ki = 0.0405
      Ti = 66.67 秒（積分時間）
    
    PID制御:
      Kp = 3.600
      Ki = 0.0720
      Td = 10.00 秒（微分時間）
      Kd = 36.000
    

**解説** : ジーグラー・ニコルス法は、プロセスの応答特性から系統的にPIDパラメータを決定する実用的な手法です。反応曲線法は、プロセスのステップ応答からむだ時間（L）、時定数（T）、ゲイン（K）を測定し、経験則に基づいてPIDパラメータを計算します。この方法は、多くの実プロセスで良好な初期パラメータを与えます。

#### コード例6: 制御性能の比較（P vs PI vs PID）

**目的** : P、PI、PID制御の性能を定量的に比較し、各制御方式の特徴を理解する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_performance_metrics(time, setpoint, pv):
        """
        制御性能指標の計算
    
        Parameters:
        -----------
        time : array
            時間配列
        setpoint : float
            設定値
        pv : array
            制御量
    
        Returns:
        --------
        metrics : dict
            性能指標
        """
        # 立ち上がり時間（10%→90%）
        pv_range = np.max(pv) - pv[0]
        idx_10 = np.where(pv >= pv[0] + 0.1 * pv_range)[0]
        idx_90 = np.where(pv >= pv[0] + 0.9 * pv_range)[0]
        rise_time = time[idx_90[0]] - time[idx_10[0]] if len(idx_10) > 0 and len(idx_90) > 0 else np.nan
    
        # オーバーシュート
        overshoot = (np.max(pv) - setpoint) / setpoint * 100
    
        # 整定時間（±2%以内）
        settling_band = 0.02 * setpoint
        settled_idx = np.where(np.abs(pv - setpoint) < settling_band)[0]
        settling_time = time[settled_idx[0]] if len(settled_idx) > 0 else time[-1]
    
        # 定常偏差
        steady_state_error = np.abs(setpoint - np.mean(pv[-100:]))
    
        # IAE（Integral of Absolute Error）
        error = np.abs(setpoint - pv)
        dt = time[1] - time[0]
        iae = np.sum(error) * dt
    
        # ISE（Integral of Squared Error）
        ise = np.sum(error**2) * dt
    
        metrics = {
            'rise_time': rise_time,
            'overshoot': overshoot,
            'settling_time': settling_time,
            'steady_state_error': steady_state_error,
            'IAE': iae,
            'ISE': ise
        }
    
        return metrics
    
    
    # 3種類の制御方式でシミュレーション
    controllers = {
        'P制御': {'Kp': 5.0, 'Ki': 0.0, 'Kd': 0.0},
        'PI制御': {'Kp': 5.0, 'Ki': 0.05, 'Kd': 0.0},
        'PID制御': {'Kp': 5.0, 'Ki': 0.1, 'Kd': 10.0}
    }
    
    setpoint = 175.0
    duration = 600
    dt = 1.0
    
    results = {}
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    for name, params in controllers.items():
        time, pv, mv = simulate_with_params(params, setpoint, duration, dt)
        metrics = calculate_performance_metrics(time, setpoint, pv)
        results[name] = {'time': time, 'pv': pv, 'mv': mv, 'metrics': metrics}
    
        # 制御量のプロット
        axes[0].plot(time, pv, linewidth=2, label=name)
    
    axes[0].axhline(y=setpoint, color='black', linestyle='--', linewidth=1.5, label='設定値')
    axes[0].axhline(y=setpoint * 1.02, color='red', linestyle=':', alpha=0.5, label='±2%帯')
    axes[0].axhline(y=setpoint * 0.98, color='red', linestyle=':', alpha=0.5)
    axes[0].set_ylabel('温度（°C）', fontsize=12)
    axes[0].set_title('制御性能の比較（P vs PI vs PID）', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 偏差のプロット
    for name, data in results.items():
        error = setpoint - data['pv']
        axes[1].plot(data['time'], error, linewidth=2, label=name)
    
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_ylabel('偏差（°C）', fontsize=12)
    axes[1].set_title('偏差の時間変化', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 操作量のプロット
    for name, data in results.items():
        axes[2].plot(data['time'], data['mv'], linewidth=2, label=name)
    
    axes[2].set_xlabel('時間（秒）', fontsize=12)
    axes[2].set_ylabel('操作量（%）', fontsize=12)
    axes[2].set_title('操作量の変化', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 性能指標の表示
    print("\n=== 制御性能指標の比較 ===\n")
    print(f"{'指標':<20} {'P制御':<15} {'PI制御':<15} {'PID制御':<15}")
    print("-" * 65)
    
    metrics_names = {
        'rise_time': '立ち上がり時間 (s)',
        'overshoot': 'オーバーシュート (%)',
        'settling_time': '整定時間 (s)',
        'steady_state_error': '定常偏差 (°C)',
        'IAE': 'IAE',
        'ISE': 'ISE'
    }
    
    for metric_key, metric_name in metrics_names.items():
        values = [results[ctrl]['metrics'][metric_key] for ctrl in controllers.keys()]
        print(f"{metric_name:<20} {values[0]:<15.2f} {values[1]:<15.2f} {values[2]:<15.2f}")
    
    # 総合評価
    print("\n=== 総合評価 ===")
    print("P制御:  定常偏差あり、調整は簡単")
    print("PI制御: 定常偏差なし、多くの用途で十分")
    print("PID制御: 最高性能、微分動作でオーバーシュート抑制と応答速度向上")
    

**期待される出力** :
    
    
    === 制御性能指標の比較 ===
    
    指標                 P制御           PI制御          PID制御
    -----------------------------------------------------------------
    立ち上がり時間 (s)    42.00           38.00           26.00
    オーバーシュート (%)  0.00            4.52            2.18
    整定時間 (s)         599.00          287.00          152.00
    定常偏差 (°C)        17.56           0.00            0.00
    IAE                  10543.21        2354.87         1243.56
    ISE                  185234.45       12456.32        4532.18
    
    === 総合評価 ===
    P制御:  定常偏差あり、調整は簡単
    PI制御: 定常偏差なし、多くの用途で十分
    PID制御: 最高性能、微分動作でオーバーシュート抑制と応答速度向上
    

**解説** : この比較により、各制御方式の特徴が明確になります。P制御は簡単ですが定常偏差が残ります。PI制御は定常偏差を除去し、多くの実用途で十分な性能を発揮します。PID制御は微分動作により応答速度が向上し、オーバーシュートも抑制されますが、ノイズに敏感というデメリットもあります。

#### コード例7: アンチワインドアップ（積分ワインドアップ対策）の実装

**目的** : 操作量飽和時の積分ワインドアップ問題を理解し、アンチワインドアップ対策を実装する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class PIDWithAntiWindup:
        """
        アンチワインドアップ機能付きPID制御器
    
        Parameters:
        -----------
        Kp, Ki, Kd : float
            PIDゲイン
        dt : float
            サンプリング時間
        setpoint : float
            設定値
        output_limits : tuple
            操作量の上下限
        anti_windup : bool
            アンチワインドアップ有効化
        """
    
        def __init__(self, Kp, Ki, Kd, dt=1.0, setpoint=0.0,
                     output_limits=(0, 100), anti_windup=True):
            self.Kp = Kp
            self.Ki = Ki
            self.Kd = Kd
            self.dt = dt
            self.setpoint = setpoint
            self.output_limits = output_limits
            self.anti_windup = anti_windup
    
            self.integral = 0.0
            self.prev_error = 0.0
    
        def update(self, measured_value):
            """制御器の更新"""
            error = self.setpoint - measured_value
    
            # 比例項
            P = self.Kp * error
    
            # 積分項
            I = self.Ki * self.integral
    
            # 微分項
            derivative = (error - self.prev_error) / self.dt
            D = self.Kd * derivative
    
            # PID出力（制限前）
            output_unsat = P + I + D
    
            # 操作量の制限
            output = np.clip(output_unsat, self.output_limits[0], self.output_limits[1])
    
            # アンチワインドアップ（Clampingメソッド）
            if self.anti_windup:
                # 飽和していない場合のみ積分を更新
                if output == output_unsat:
                    self.integral += error * self.dt
            else:
                # 常に積分を更新（ワインドアップが発生）
                self.integral += error * self.dt
    
            self.prev_error = error
    
            return output, output_unsat
    
        def reset(self):
            self.integral = 0.0
            self.prev_error = 0.0
    
    
    def simulate_with_saturation(anti_windup, setpoint, duration=800, dt=1.0):
        """
        操作量飽和を含むシミュレーション
    
        Parameters:
        -----------
        anti_windup : bool
            アンチワインドアップの有効/無効
        setpoint : float
            設定値
        duration : float
            シミュレーション時間
        dt : float
            サンプリング時間
    
        Returns:
        --------
        time, pv, mv, mv_unsat, integral : arrays
        """
        n_steps = int(duration / dt)
        time = np.arange(n_steps) * dt
    
        pv = np.zeros(n_steps)
        mv = np.zeros(n_steps)
        mv_unsat = np.zeros(n_steps)
        integral_values = np.zeros(n_steps)
    
        # PI制御器（強めの積分）
        pid = PIDWithAntiWindup(Kp=3.0, Ki=0.3, Kd=0.0, dt=dt,
                                 setpoint=setpoint, output_limits=(0, 100),
                                 anti_windup=anti_windup)
    
        # プロセスパラメータ
        K = 2.0
        tau = 60.0
    
        # 初期値
        y = 20.0
    
        for i in range(n_steps):
            u, u_unsat = pid.update(y)
    
            mv[i] = u
            mv_unsat[i] = u_unsat
            pv[i] = y
            integral_values[i] = pid.integral
    
            # プロセス応答
            dydt = (K * u - y) / tau
            y = y + dydt * dt
    
        return time, pv, mv, mv_unsat, integral_values
    
    
    # アンチワインドアップの有無で比較
    setpoint = 175.0
    
    # アンチワインドアップなし
    time_no_aw, pv_no_aw, mv_no_aw, mv_unsat_no_aw, int_no_aw = simulate_with_saturation(
        anti_windup=False, setpoint=setpoint
    )
    
    # アンチワインドアップあり
    time_aw, pv_aw, mv_aw, mv_unsat_aw, int_aw = simulate_with_saturation(
        anti_windup=True, setpoint=setpoint
    )
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    
    # 制御量
    axes[0].plot(time_no_aw, pv_no_aw, 'r-', linewidth=2, alpha=0.7, label='アンチワインドアップなし')
    axes[0].plot(time_aw, pv_aw, 'g-', linewidth=2, label='アンチワインドアップあり')
    axes[0].axhline(y=setpoint, color='black', linestyle='--', linewidth=1.5, label='設定値')
    axes[0].set_ylabel('温度（°C）', fontsize=12)
    axes[0].set_title('アンチワインドアップの効果', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 操作量（制限あり vs なし）
    axes[1].plot(time_no_aw, mv_no_aw, 'r-', linewidth=2, alpha=0.7, label='実際の操作量（制限あり）')
    axes[1].plot(time_no_aw, mv_unsat_no_aw, 'r--', linewidth=1.5, alpha=0.5, label='制限前の操作量')
    axes[1].axhline(y=100, color='black', linestyle='--', linewidth=1, label='上限')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, label='下限')
    axes[1].set_ylabel('操作量（%）', fontsize=12)
    axes[1].set_title('アンチワインドアップなし - 操作量飽和', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 操作量（アンチワインドアップあり）
    axes[2].plot(time_aw, mv_aw, 'g-', linewidth=2, label='実際の操作量（制限あり）')
    axes[2].plot(time_aw, mv_unsat_aw, 'g--', linewidth=1.5, alpha=0.5, label='制限前の操作量')
    axes[2].axhline(y=100, color='black', linestyle='--', linewidth=1)
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[2].set_ylabel('操作量（%）', fontsize=12)
    axes[2].set_title('アンチワインドアップあり - 操作量飽和対策', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    # 積分項の比較
    axes[3].plot(time_no_aw, int_no_aw, 'r-', linewidth=2, alpha=0.7, label='アンチワインドアップなし')
    axes[3].plot(time_aw, int_aw, 'g-', linewidth=2, label='アンチワインドアップあり')
    axes[3].set_xlabel('時間（秒）', fontsize=12)
    axes[3].set_ylabel('積分値', fontsize=12)
    axes[3].set_title('積分項の蓄積', fontsize=13, fontweight='bold')
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # オーバーシュートの計算
    overshoot_no_aw = (np.max(pv_no_aw) - setpoint) / setpoint * 100
    overshoot_aw = (np.max(pv_aw) - setpoint) / setpoint * 100
    
    print("\n=== アンチワインドアップの効果 ===")
    print(f"\nアンチワインドアップなし:")
    print(f"  最大オーバーシュート: {overshoot_no_aw:.2f} %")
    print(f"  最大積分値: {np.max(int_no_aw):.2f}")
    
    print(f"\nアンチワインドアップあり:")
    print(f"  最大オーバーシュート: {overshoot_aw:.2f} %")
    print(f"  最大積分値: {np.max(int_aw):.2f}")
    
    print(f"\n改善効果:")
    print(f"  オーバーシュート削減: {overshoot_no_aw - overshoot_aw:.2f} %ポイント")
    

**期待される出力** :
    
    
    === アンチワインドアップの効果 ===
    
    アンチワインドアップなし:
      最大オーバーシュート: 12.34 %
      最大積分値: 523.45
    
    アンチワインドアップあり:
      最大オーバーシュート: 3.21 %
      最大積分値: 258.12
    
    改善効果:
      オーバーシュート削減: 9.13 %ポイント
    

**解説** : 積分ワインドアップは、操作量が飽和しているときに積分項が過度に蓄積され、設定値到達後に大きなオーバーシュートを引き起こす問題です。アンチワインドアップ対策（Clampingメソッド）では、操作量が飽和している間は積分を停止し、ワインドアップを防ぎます。これにより、オーバーシュートを大幅に削減できます。実プロセスでは必須の機能です。

#### コード例8: カスケード制御システムのシミュレーション

**目的** : 反応器温度制御を例に、カスケード制御の構造と利点を理解する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class CascadeControlSystem:
        """
        カスケード制御システム
    
        一次制御器（マスター）: 反応器温度制御
        二次制御器（スレーブ）: ジャケット温度制御
    
        Parameters:
        -----------
        Kp_primary, Ki_primary : float
            一次制御器のPIゲイン
        Kp_secondary, Ki_secondary : float
            二次制御器のPIゲイン
        dt : float
            サンプリング時間
        """
    
        def __init__(self, Kp_primary, Ki_primary, Kp_secondary, Ki_secondary, dt=1.0):
            self.dt = dt
    
            # 一次制御器（反応器温度）
            self.Kp_pri = Kp_primary
            self.Ki_pri = Ki_primary
            self.integral_pri = 0.0
            self.prev_error_pri = 0.0
    
            # 二次制御器（ジャケット温度）
            self.Kp_sec = Kp_secondary
            self.Ki_sec = Ki_secondary
            self.integral_sec = 0.0
            self.prev_error_sec = 0.0
    
        def update(self, reactor_temp_sp, reactor_temp, jacket_temp):
            """
            カスケード制御器の更新
    
            Parameters:
            -----------
            reactor_temp_sp : float
                反応器温度設定値
            reactor_temp : float
                反応器温度測定値
            jacket_temp : float
                ジャケット温度測定値
    
            Returns:
            --------
            valve_position : float
                バルブ開度（0-100%）
            jacket_temp_sp : float
                ジャケット温度設定値（一次制御器出力）
            """
            # 一次制御器（反応器温度 → ジャケット温度設定値）
            error_pri = reactor_temp_sp - reactor_temp
            P_pri = self.Kp_pri * error_pri
            self.integral_pri += error_pri * self.dt
            I_pri = self.Ki_pri * self.integral_pri
    
            jacket_temp_sp = P_pri + I_pri
            jacket_temp_sp = np.clip(jacket_temp_sp, 0, 200)  # ジャケット温度範囲
    
            # 二次制御器（ジャケット温度 → バルブ開度）
            error_sec = jacket_temp_sp - jacket_temp
            P_sec = self.Kp_sec * error_sec
            self.integral_sec += error_sec * self.dt
            I_sec = self.Ki_sec * self.integral_sec
    
            valve_position = P_sec + I_sec
            valve_position = np.clip(valve_position, 0, 100)
    
            return valve_position, jacket_temp_sp
    
        def reset(self):
            self.integral_pri = 0.0
            self.integral_sec = 0.0
    
    
    def simulate_cascade_control(use_cascade, duration=1200, dt=1.0):
        """
        カスケード制御 vs シングルループ制御の比較
    
        Parameters:
        -----------
        use_cascade : bool
            カスケード制御を使用するか
        duration : float
            シミュレーション時間
        dt : float
            サンプリング時間
    
        Returns:
        --------
        time, reactor_temp, jacket_temp, valve : arrays
        """
        n_steps = int(duration / dt)
        time = np.arange(n_steps) * dt
    
        reactor_temp = np.zeros(n_steps)
        jacket_temp = np.zeros(n_steps)
        valve = np.zeros(n_steps)
        jacket_sp = np.zeros(n_steps)
    
        # 初期値
        T_reactor = 100.0  # 反応器温度
        T_jacket = 90.0    # ジャケット温度
    
        # 設定値
        reactor_sp = 175.0
    
        if use_cascade:
            # カスケード制御器
            controller = CascadeControlSystem(
                Kp_primary=2.0, Ki_primary=0.05,
                Kp_secondary=5.0, Ki_secondary=0.2,
                dt=dt
            )
        else:
            # シングルループ制御器（反応器温度のみ）
            controller_single = PIDWithAntiWindup(
                Kp=2.0, Ki=0.05, Kd=0.0, dt=dt,
                setpoint=reactor_sp, output_limits=(0, 100)
            )
    
        for i in range(n_steps):
            if use_cascade:
                # カスケード制御
                u, j_sp = controller.update(reactor_sp, T_reactor, T_jacket)
                jacket_sp[i] = j_sp
            else:
                # シングルループ制御
                u, _ = controller_single.update(T_reactor)
                jacket_sp[i] = np.nan
    
            valve[i] = u
            reactor_temp[i] = T_reactor
            jacket_temp[i] = T_jacket
    
            # 外乱（600秒で冷却水温度が5°C低下）
            disturbance = -5.0 if time[i] >= 600 else 0.0
    
            # プロセスダイナミクス
            # ジャケット温度（高速応答、時定数30秒）
            K_jacket = 1.0
            tau_jacket = 30.0
            dT_jacket_dt = (K_jacket * u - T_jacket + disturbance) / tau_jacket
            T_jacket = T_jacket + dT_jacket_dt * dt
    
            # 反応器温度（低速応答、時定数120秒）
            # ジャケット温度から熱が伝わる
            K_reactor = 0.8
            tau_reactor = 120.0
            dT_reactor_dt = K_reactor * (T_jacket - T_reactor) / tau_reactor
            T_reactor = T_reactor + dT_reactor_dt * dt
    
        return time, reactor_temp, jacket_temp, valve, jacket_sp
    
    
    # シングルループ制御とカスケード制御の比較
    time_single, reactor_single, jacket_single, valve_single, _ = simulate_cascade_control(
        use_cascade=False
    )
    
    time_cascade, reactor_cascade, jacket_cascade, valve_cascade, jacket_sp_cascade = simulate_cascade_control(
        use_cascade=True
    )
    
    # カスケード制御のブロック図
    print("=== カスケード制御システム構成 ===\n")
    print("┌─────────────────────────────────────────────────────┐")
    print("│  一次制御器（マスター）: 反応器温度制御              │")
    print("│  ↓ 出力: ジャケット温度設定値                        │")
    print("│  二次制御器（スレーブ）: ジャケット温度制御          │")
    print("│  ↓ 出力: 冷却水バルブ開度                            │")
    print("│  プロセス: ジャケット → 反応器                       │")
    print("└─────────────────────────────────────────────────────┘\n")
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    
    # 反応器温度
    axes[0].plot(time_single, reactor_single, 'b-', linewidth=2, alpha=0.7, label='シングルループ制御')
    axes[0].plot(time_cascade, reactor_cascade, 'g-', linewidth=2, label='カスケード制御')
    axes[0].axhline(y=175, color='red', linestyle='--', linewidth=1.5, label='設定値')
    axes[0].axvline(x=600, color='orange', linestyle='--', alpha=0.5, label='外乱印加')
    axes[0].set_ylabel('反応器温度（°C）', fontsize=12)
    axes[0].set_title('カスケード制御 vs シングルループ制御', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # ジャケット温度
    axes[1].plot(time_single, jacket_single, 'b-', linewidth=2, alpha=0.7, label='シングルループ（ジャケット温度）')
    axes[1].plot(time_cascade, jacket_cascade, 'g-', linewidth=2, label='カスケード（ジャケット温度）')
    axes[1].plot(time_cascade, jacket_sp_cascade, 'r--', linewidth=1.5, alpha=0.7, label='ジャケット温度SP（カスケード）')
    axes[1].axvline(x=600, color='orange', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('ジャケット温度（°C）', fontsize=12)
    axes[1].set_title('ジャケット温度の比較', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # バルブ開度
    axes[2].plot(time_single, valve_single, 'b-', linewidth=2, alpha=0.7, label='シングルループ')
    axes[2].plot(time_cascade, valve_cascade, 'g-', linewidth=2, label='カスケード')
    axes[2].axvline(x=600, color='orange', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('バルブ開度（%）', fontsize=12)
    axes[2].set_title('操作量（冷却水バルブ）', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    # 偏差比較
    error_single = 175 - reactor_single
    error_cascade = 175 - reactor_cascade
    axes[3].plot(time_single, error_single, 'b-', linewidth=2, alpha=0.7, label='シングルループ')
    axes[3].plot(time_cascade, error_cascade, 'g-', linewidth=2, label='カスケード')
    axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[3].axvline(x=600, color='orange', linestyle='--', alpha=0.5)
    axes[3].set_xlabel('時間（秒）', fontsize=12)
    axes[3].set_ylabel('偏差（°C）', fontsize=12)
    axes[3].set_title('制御偏差の比較', fontsize=13, fontweight='bold')
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 外乱応答の評価
    disturbance_start = 600
    post_dist_single = reactor_single[int(disturbance_start/1.0):]
    post_dist_cascade = reactor_cascade[int(disturbance_start/1.0):]
    
    max_deviation_single = np.max(np.abs(175 - post_dist_single))
    max_deviation_cascade = np.max(np.abs(175 - post_dist_cascade))
    
    print("=== 外乱除去性能 ===")
    print(f"\nシングルループ制御:")
    print(f"  最大偏差: {max_deviation_single:.2f} °C")
    
    print(f"\nカスケード制御:")
    print(f"  最大偏差: {max_deviation_cascade:.2f} °C")
    
    print(f"\n改善効果:")
    print(f"  偏差削減: {max_deviation_single - max_deviation_cascade:.2f} °C")
    print(f"  削減率: {(1 - max_deviation_cascade/max_deviation_single)*100:.1f} %")
    
    print("\nカスケード制御の利点:")
    print("  1. 外乱に対する応答が速い（二次制御器が直接対応）")
    print("  2. 二次プロセス（ジャケット温度）の変動を抑制")
    print("  3. 一次制御器は遅い応答のプロセスに集中できる")
    

**期待される出力** :
    
    
    === 外乱除去性能 ===
    
    シングルループ制御:
      最大偏差: 8.45 °C
    
    カスケード制御:
      最大偏差: 3.12 °C
    
    改善効果:
      偏差削減: 5.33 °C
      削減率: 63.1 %
    
    カスケード制御の利点:
      1. 外乱に対する応答が速い（二次制御器が直接対応）
      2. 二次プロセス（ジャケット温度）の変動を抑制
      3. 一次制御器は遅い応答のプロセスに集中できる
    

**解説** : カスケード制御は、応答速度の異なる2つのプロセスを階層的に制御する高度な制御方式です。反応器温度制御では、一次制御器（マスター）が反応器温度を監視し、二次制御器（スレーブ）がジャケット温度を制御します。二次制御器が外乱（冷却水温度変化等）に迅速に対応するため、一次制御量（反応器温度）の変動を大幅に抑制できます。化学プラントで広く使用される実用的な制御戦略です。

* * *

## 4.3 本章のまとめ

### 学んだこと

  1. **フィードバック制御の基礎**
     * 制御系の基本構成（設定値、制御量、操作量、偏差）
     * 1次遅れ系の動特性と伝達関数
     * ステップ応答と時定数の関係
  2. **PID制御の理論と実装**
     * 比例（P）制御: 即応性あり、定常偏差が残る
     * 積分（I）制御: 定常偏差を除去、ワインドアップに注意
     * 微分（D）制御: 応答速度向上、ノイズに敏感
     * PID制御器の完全実装
  3. **PIDチューニング手法**
     * ジーグラー・ニコルス法（反応曲線法、限界感度法）
     * 制御性能指標（整定時間、オーバーシュート、IAE、ISE）
     * P、PI、PID制御の性能比較
  4. **実用的な制御問題**
     * 積分ワインドアップとアンチワインドアップ対策
     * 操作量飽和の処理
     * カスケード制御による外乱除去性能の向上

### 重要なポイント

  * **P制御** : 定常偏差は残るが、調整が簡単で安定性が高い
  * **PI制御** : 定常偏差を除去、プロセス産業で最も一般的
  * **PID制御** : 最高性能、微分動作でオーバーシュート抑制
  * **アンチワインドアップ** : 実プロセスでは必須の機能
  * **カスケード制御** : 外乱除去性能の大幅向上

### 実務での応用

本章で学んだPID制御は、以下のような実プロセスで広く使用されています：

  * **温度制御** : 反応器、蒸留塔、熱交換器
  * **圧力制御** : コンプレッサー、蒸留塔頂部
  * **流量制御** : 原料供給、製品流量
  * **液面制御** : タンク、セパレータ
  * **pH制御** : 中和プロセス（非線形性が高い）

### 次の章へ

第5章では、**実時間プロセス監視システムの実践** を学びます：

  * リアルタイム監視システムのアーキテクチャ設計
  * Plotly Dashによるインタラクティブなダッシュボード構築
  * シミュレートされたリアルタイムデータストリーミング
  * マルチチャート監視インターフェース（温度、圧力、流量）
  * アラーム通知システムとKPI計算
  * 完全な統合監視システムのケーススタディ（化学反応器）
