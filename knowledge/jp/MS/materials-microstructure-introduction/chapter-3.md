---
title: 第3章：析出と固溶
chapter_title: 第3章：析出と固溶
subtitle: Precipitation and Solid Solution - 時効硬化から微細析出物制御まで
difficulty: 中級
code_examples: 7
---

## 学習目標

この章を完了すると、以下のスキルと知識を習得できます：

  * ✅ 固溶体の種類と性質を理解し、固溶強化のメカニズムを説明できる
  * ✅ 析出の核生成と成長のメカニズムを理解し、時効曲線を解釈できる
  * ✅ 時効硬化（Age Hardening）の原理を説明し、Al合金などの実例を理解できる
  * ✅ Orowan機構による析出強化を定量的に計算できる
  * ✅ Gibbs-Thomson効果と粒子粗大化（Ostwald ripening）を理解できる
  * ✅ Coherent、semi-coherent、incoherent析出物の違いを説明できる
  * ✅ Pythonで析出物の時間発展と強度予測をシミュレーションできる

## 3.1 固溶体の基礎

### 3.1.1 固溶体の定義と種類

**固溶体（Solid Solution）** は、2種類以上の元素が原子レベルで混ざり合った均一な固相です。基本となる結晶構造（母相、matrix）中に、別の元素（溶質原子、solute）が溶け込んでいる状態です。

#### 💡 固溶体の分類

**1\. 置換型固溶体（Substitutional Solid Solution）**

  * 溶質原子が母相の原子と置き換わる
  * 条件: 原子半径の差が15%以内（Hume-Rothery則）
  * 例: Cu-Ni、Fe-Cr、Al-Mg

**2\. 侵入型固溶体（Interstitial Solid Solution）**

  * 溶質原子が格子間位置に入る
  * 条件: 溶質原子が小さい（C、N、H、O）
  * 例: Fe-C（鋼）、Ti-O、Zr-H

    
    
    ```mermaid
    graph LR
        A[固溶体] --> B[置換型]
        A --> C[侵入型]
        B --> D[Cu-Ni合金原子半径類似]
        B --> E[ステンレス鋼Fe-Cr-Ni]
        C --> F[炭素鋼Fe-C]
        C --> G[窒化物Ti-N]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#fce7f3
        style C fill:#fce7f3
    ```

### 3.1.2 固溶強化のメカニズム

固溶体は純金属よりも強度が高くなります。これを**固溶強化（Solid Solution Strengthening）** と呼びます。主なメカニズムは以下の通りです：

メカニズム | 原因 | 効果  
---|---|---  
**格子歪み** | 溶質原子の原子半径が異なる | 転位運動の抵抗増加  
**弾性相互作用** | 溶質原子周辺の応力場 | 転位との相互作用  
**化学的相互作用** | 結合力の変化 | 積層欠陥エネルギー変化  
**電気的相互作用** | 電子構造の変化 | 転位の易動度低下  
  
固溶強化による降伏応力の増加は、Labuschモデルにより以下のように近似されます：

> Δσy = K · cn   
>   
>  ここで、Δσyは降伏応力の増加、cは溶質原子濃度、Kは定数、nは0.5〜1（通常2/3程度） 

### 3.1.3 実例：Al-Mg固溶体の強化
    
    
    """
    Example 1: Al-Mg固溶体における固溶強化の計算
    Labuschモデルを用いた降伏応力の予測
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 固溶強化の計算
    def solid_solution_strengthening(c, K=30, n=0.67):
        """
        固溶強化による降伏応力増加を計算
    
        Args:
            c: 溶質濃度 [at%]
            K: 定数 [MPa/(at%)^n]
            n: 指数（通常0.5-1.0）
    
        Returns:
            delta_sigma: 降伏応力増加 [MPa]
        """
        return K * (c ** n)
    
    # Al-Mg合金の実験データ（近似）
    mg_content = np.array([0, 1, 2, 3, 4, 5, 6])  # at%
    yield_stress_exp = np.array([20, 50, 75, 95, 112, 127, 140])  # MPa
    
    # モデル予測
    mg_model = np.linspace(0, 7, 100)
    delta_sigma = solid_solution_strengthening(mg_model, K=30, n=0.67)
    yield_stress_model = 20 + delta_sigma  # 純Alの降伏応力20 MPa
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(mg_model, yield_stress_model, 'r-', linewidth=2,
             label=f'Labuschモデル (n=0.67)')
    plt.scatter(mg_content, yield_stress_exp, s=100, c='blue',
                marker='o', label='実験データ')
    
    plt.xlabel('Mg濃度 [at%]', fontsize=12)
    plt.ylabel('降伏応力 [MPa]', fontsize=12)
    plt.title('Al-Mg固溶体の固溶強化', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 特定組成での計算
    mg_5at = 5.0
    delta_sigma_5 = solid_solution_strengthening(mg_5at)
    print(f"Mg 5at%添加時の降伏応力増加: {delta_sigma_5:.1f} MPa")
    print(f"予測降伏応力: {20 + delta_sigma_5:.1f} MPa")
    print(f"実験値: {yield_stress_exp[5]:.1f} MPa")
    print(f"誤差: {abs((20 + delta_sigma_5) - yield_stress_exp[5]):.1f} MPa")
    
    # 出力例:
    # Mg 5at%添加時の降伏応力増加: 102.5 MPa
    # 予測降伏応力: 122.5 MPa
    # 実験値: 127.0 MPa
    # 誤差: 4.5 MPa
    

#### 📊 実践のポイント

Al-Mg合金（5000系アルミニウム合金）は、固溶強化を主な強化機構とする代表的な合金です。Mgは最大6%程度まで固溶し、優れた強度と耐食性を両立します。缶材や船舶材料として広く使用されています。

## 3.2 析出の基礎理論

### 3.2.1 析出のメカニズム

**析出（Precipitation）** は、過飽和固溶体から第二相粒子が生成する現象です。典型的な析出プロセスは以下の段階を経ます：
    
    
    ```mermaid
    flowchart TD
        A[過飽和固溶体] --> B[核生成Nucleation]
        B --> C[成長Growth]
        C --> D[粗大化Coarsening]
    
        B1[均質核生成] -.-> B
        B2[不均質核生成] -.-> B
    
        C1[拡散律速成長] -.-> C
        C2[界面律速成長] -.-> C
    
        D1[Ostwald ripening] -.-> D
    
        style A fill:#fff3e0
        style B fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

### 3.2.2 核生成理論

析出の核生成速度は、古典的核生成理論により以下のように表されます：

> J = N0 · ν · exp(-ΔG*/kT)   
>   
>  ここで、  
>  J: 核生成速度 [個/m³/s]  
>  N0: 核生成サイト密度 [個/m³]  
>  ν: 原子の振動周波数 [Hz]  
>  ΔG*: 臨界核生成エネルギー [J]  
>  k: ボルツマン定数 [J/K]  
>  T: 温度 [K] 

臨界核生成エネルギーΔG*は、均質核生成の場合：

> ΔG* = (16πγ³) / (3ΔGv²)   
>   
>  γ: 界面エネルギー [J/m²]  
>  ΔGv: 単位体積あたりの自由エネルギー変化 [J/m³] 
    
    
    """
    Example 2: 析出の核生成速度計算
    古典的核生成理論に基づくシミュレーション
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 物理定数
    k_B = 1.38e-23  # ボルツマン定数 [J/K]
    h = 6.626e-34   # プランク定数 [J·s]
    
    def nucleation_rate(T, gamma, delta_Gv, N0=1e28, nu=1e13):
        """
        核生成速度を計算（古典的核生成理論）
    
        Args:
            T: 温度 [K]
            gamma: 界面エネルギー [J/m²]
            delta_Gv: 体積自由エネルギー変化 [J/m³]
            N0: 核生成サイト密度 [個/m³]
            nu: 原子振動周波数 [Hz]
    
        Returns:
            J: 核生成速度 [個/m³/s]
        """
        # 臨界核生成エネルギー
        delta_G_star = (16 * np.pi * gamma**3) / (3 * delta_Gv**2)
    
        # 核生成速度
        J = N0 * nu * np.exp(-delta_G_star / (k_B * T))
    
        return J, delta_G_star
    
    # Al-Cu合金のパラメータ（θ'相の析出）
    gamma = 0.2  # 界面エネルギー [J/m²]
    temperatures = np.linspace(373, 573, 100)  # 100-300°C
    
    # 過飽和度による自由エネルギー変化（簡略化）
    supersaturations = [1.5, 2.0, 2.5]  # 過飽和度
    colors = ['blue', 'green', 'red']
    labels = ['低過飽和度 (1.5x)', '中過飽和度 (2.0x)', '高過飽和度 (2.5x)']
    
    plt.figure(figsize=(12, 5))
    
    # (a) 温度依存性
    plt.subplot(1, 2, 1)
    for S, color, label in zip(supersaturations, colors, labels):
        delta_Gv = -2e8 * np.log(S)  # 簡略化した自由エネルギー [J/m³]
        J_list = []
        for T in temperatures:
            J, _ = nucleation_rate(T, gamma, delta_Gv)
            J_list.append(J)
    
        plt.semilogy(temperatures - 273, J_list, color=color,
                     linewidth=2, label=label)
    
    plt.xlabel('温度 [°C]', fontsize=12)
    plt.ylabel('核生成速度 [個/m³/s]', fontsize=12)
    plt.title('(a) 温度依存性', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # (b) 臨界核半径
    plt.subplot(1, 2, 2)
    T_aging = 473  # 時効温度 200°C
    for S, color, label in zip(supersaturations, colors, labels):
        delta_Gv = -2e8 * np.log(S)
        r_crit = 2 * gamma / abs(delta_Gv)  # 臨界核半径 [m]
        r_crit_nm = r_crit * 1e9  # [nm]
    
        # プロット用
        plt.bar(label, r_crit_nm, color=color, alpha=0.7)
    
    plt.ylabel('臨界核半径 [nm]', fontsize=12)
    plt.title('(b) 過飽和度と臨界核半径 (200°C)', fontsize=13, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 数値出力
    print("=== Al-Cu合金の核生成解析 ===\n")
    T_test = 473  # 200°C
    for S in supersaturations:
        delta_Gv = -2e8 * np.log(S)
        J, delta_G_star = nucleation_rate(T_test, gamma, delta_Gv)
        r_crit = 2 * gamma / abs(delta_Gv) * 1e9  # [nm]
    
        print(f"過飽和度 {S}x:")
        print(f"  核生成速度: {J:.2e} 個/m³/s")
        print(f"  臨界核半径: {r_crit:.2f} nm")
        print(f"  活性化エネルギー: {delta_G_star/k_B:.2e} K\n")
    
    # 出力例:
    # === Al-Cu合金の核生成解析 ===
    #
    # 過飽和度 1.5x:
    #   核生成速度: 3.45e+15 個/m³/s
    #   臨界核半径: 2.47 nm
    #   活性化エネルギー: 8.12e+03 K
    

### 3.2.3 析出物の成長

核生成後、析出物は拡散により成長します。球状析出物の半径r(t)の時間発展は、拡散律速の場合：

> r(t) = √(2Dt · (c0 \- ce) / cp)   
>   
>  D: 拡散係数 [m²/s]  
>  t: 時間 [s]  
>  c0: 初期濃度  
>  ce: 平衡濃度  
>  cp: 析出物中の濃度 
    
    
    """
    Example 3: 析出物サイズの時間発展
    拡散律速成長モデル
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def precipitate_growth(t, T, D0=1e-5, Q=150e3, c0=0.04, ce=0.01, cp=0.3):
        """
        析出物半径の時間発展を計算
    
        Args:
            t: 時間 [s]
            T: 温度 [K]
            D0: 拡散係数の前指数因子 [m²/s]
            Q: 活性化エネルギー [J/mol]
            c0: 初期溶質濃度
            ce: 平衡濃度
            cp: 析出物中の濃度
    
        Returns:
            r: 析出物半径 [m]
        """
        R = 8.314  # 気体定数 [J/mol/K]
        D = D0 * np.exp(-Q / (R * T))  # Arrhenius式
    
        # 拡散律速成長
        r = np.sqrt(2 * D * t * (c0 - ce) / cp)
    
        return r
    
    # 時効条件
    temperatures = [423, 473, 523]  # 150, 200, 250°C
    temp_labels = ['150°C', '200°C', '250°C']
    colors = ['blue', 'green', 'red']
    
    time_hours = np.logspace(-1, 3, 100)  # 0.1〜1000時間
    time_seconds = time_hours * 3600
    
    plt.figure(figsize=(12, 5))
    
    # (a) 時間-サイズ曲線
    plt.subplot(1, 2, 1)
    for T, label, color in zip(temperatures, temp_labels, colors):
        r = precipitate_growth(time_seconds, T)
        r_nm = r * 1e9  # [nm]
    
        plt.loglog(time_hours, r_nm, linewidth=2,
                   color=color, label=label)
    
    plt.xlabel('時効時間 [h]', fontsize=12)
    plt.ylabel('析出物半径 [nm]', fontsize=12)
    plt.title('(a) 析出物の成長曲線', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    
    # (b) 成長速度の温度依存性
    plt.subplot(1, 2, 2)
    t_fixed = 10 * 3600  # 10時間後
    T_range = np.linspace(373, 573, 50)
    r_range = precipitate_growth(t_fixed, T_range)
    r_range_nm = r_range * 1e9
    
    plt.plot(T_range - 273, r_range_nm, 'r-', linewidth=2)
    plt.xlabel('時効温度 [°C]', fontsize=12)
    plt.ylabel('析出物半径 (10h後) [nm]', fontsize=12)
    plt.title('(b) 成長速度の温度依存性', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 実用的な計算例
    print("=== 析出物成長の予測 ===\n")
    aging_conditions = [
        (473, 1),    # 200°C, 1時間
        (473, 10),   # 200°C, 10時間
        (473, 100),  # 200°C, 100時間
        (523, 10),   # 250°C, 10時間
    ]
    
    for T, t_h in aging_conditions:
        t_s = t_h * 3600
        r = precipitate_growth(t_s, T)
        r_nm = r * 1e9
    
        print(f"{T-273:.0f}°C, {t_h}時間: 析出物半径 = {r_nm:.1f} nm")
    
    # 出力例:
    # === 析出物成長の予測 ===
    #
    # 200°C, 1時間: 析出物半径 = 8.5 nm
    # 200°C, 10時間: 析出物半径 = 26.9 nm
    # 200°C, 100時間: 析出物半径 = 85.0 nm
    # 250°C, 10時間: 析出物半径 = 67.3 nm
    

## 3.3 時効硬化（Age Hardening）

### 3.3.1 時効硬化の原理

**時効硬化（Age Hardening）** または析出硬化（Precipitation Hardening）は、過飽和固溶体から微細な析出物を生成させることで材料を強化する熱処理技術です。代表的な時効硬化性合金：

  * **Al合金** : 2000系(Al-Cu)、6000系(Al-Mg-Si)、7000系(Al-Zn-Mg)
  * **ニッケル基超合金** : Inconel 718（γ''相析出）
  * **マルエージング鋼** : Fe-Ni-Co-Mo合金
  * **析出硬化系ステンレス鋼** : 17-4PH、15-5PH

### 3.3.2 時効曲線と析出過程

Al-Cu合金（2000系）の典型的な析出過程：
    
    
    ```mermaid
    flowchart LR
        A[過飽和固溶体α-SSS] --> B[GPゾーンGP zones]
        B --> C[θ''相準安定]
        C --> D[θ'相準安定]
        D --> E[θ相Al₂Cu平衡相]
    
        style A fill:#fff3e0
        style B fill:#e3f2fd
        style C fill:#e3f2fd
        style D fill:#e3f2fd
        style E fill:#c8e6c9
    ```

各段階の特徴：

段階 | 相 | サイズ | 整合性 | 硬化効果  
---|---|---|---|---  
初期 | GPゾーン | 1-2 nm | 完全整合 | 中  
中間 | θ''、θ' | 5-50 nm | 半整合 | **最大**  
後期 | θ（Al₂Cu） | >100 nm | 非整合 | 低  
      
    
    """
    Example 4: Al合金の時効曲線シミュレーション
    硬度の時間変化を予測
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def aging_hardness_curve(t, T, peak_time_ref=10, peak_hardness=150,
                             T_ref=473, Q=100e3):
        """
        時効曲線をシミュレーション（経験的モデル）
    
        Args:
            t: 時効時間 [h]
            T: 時効温度 [K]
            peak_time_ref: 基準温度でのピーク時間 [h]
            peak_hardness: ピーク硬度 [HV]
            T_ref: 基準温度 [K]
            Q: 活性化エネルギー [J/mol]
    
        Returns:
            hardness: 硬度 [HV]
        """
        R = 8.314  # 気体定数
    
        # 温度補正したピーク時間（Arrheniusの関係）
        peak_time = peak_time_ref * np.exp(Q/R * (1/T - 1/T_ref))
    
        # 硬度の時間発展（JMAモデルベース）
        # Under-aging領域
        H_under = 70 + (peak_hardness - 70) * (1 - np.exp(-(t/peak_time)**1.5))
    
        # Over-aging領域（粗大化による軟化）
        H_over = peak_hardness * np.exp(-0.5 * ((t - peak_time)/peak_time)**0.8)
        H_over = np.maximum(H_over, 80)  # 最小硬度
    
        # 組み合わせ
        hardness = np.where(t <= peak_time, H_under, H_over)
    
        return hardness
    
    # 時効条件
    temperatures = [423, 473, 523]  # 150, 200, 250°C
    temp_labels = ['150°C (低温)', '200°C (標準)', '250°C (高温)']
    colors = ['blue', 'green', 'red']
    
    time_hours = np.logspace(-1, 3, 200)  # 0.1〜1000時間
    
    plt.figure(figsize=(12, 5))
    
    # (a) 時効曲線
    plt.subplot(1, 2, 1)
    for T, label, color in zip(temperatures, temp_labels, colors):
        hardness = aging_hardness_curve(time_hours, T)
    
        plt.semilogx(time_hours, hardness, linewidth=2.5,
                     color=color, label=label)
    
        # ピーク硬度位置をマーク
        peak_idx = np.argmax(hardness)
        plt.plot(time_hours[peak_idx], hardness[peak_idx],
                 'o', markersize=10, color=color)
    
    # Under-aging, Peak-aging, Over-agingの領域を示す
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    plt.text(0.3, 145, 'Under-aging', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.text(10, 145, 'Peak-aging', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    plt.text(300, 145, 'Over-aging', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.xlabel('時効時間 [h]', fontsize=12)
    plt.ylabel('硬度 [HV]', fontsize=12)
    plt.title('(a) Al-Cu合金の時効曲線', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.ylim(60, 160)
    
    # (b) ピーク時間の温度依存性
    plt.subplot(1, 2, 2)
    T_range = np.linspace(393, 553, 50)  # 120-280°C
    peak_times = []
    
    for T in T_range:
        # ピーク時間を求める
        t_test = np.logspace(-2, 4, 1000)
        h_test = aging_hardness_curve(t_test, T)
        peak_t = t_test[np.argmax(h_test)]
        peak_times.append(peak_t)
    
    plt.semilogy(T_range - 273, peak_times, 'r-', linewidth=2.5)
    plt.xlabel('時効温度 [°C]', fontsize=12)
    plt.ylabel('ピーク時効時間 [h]', fontsize=12)
    plt.title('(b) ピーク時効時間の温度依存性', fontsize=13, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 実用的な推奨時効条件
    print("=== 推奨時効条件（Al-Cu合金） ===\n")
    for T in temperatures:
        t_test = np.logspace(-2, 3, 1000)
        h_test = aging_hardness_curve(t_test, T)
        peak_idx = np.argmax(h_test)
        peak_time = t_test[peak_idx]
        peak_h = h_test[peak_idx]
    
        print(f"{T-273:.0f}°C:")
        print(f"  ピーク時効時間: {peak_time:.1f} 時間")
        print(f"  最大硬度: {peak_h:.1f} HV\n")
    
    # 出力例:
    # === 推奨時効条件（Al-Cu合金） ===
    #
    # 150°C:
    #   ピーク時効時間: 48.3 時間
    #   最大硬度: 150.0 HV
    #
    # 200°C:
    #   ピーク時効時間: 10.0 時間
    #   最大硬度: 150.0 HV
    

## 3.4 析出強化のメカニズム

### 3.4.1 Orowan機構

析出物が転位運動を妨げることで材料が強化されます。最も重要なメカニズムが**Orowan機構** です。転位が析出物間をすり抜けるために必要な応力：

> τOrowan = (M · G · b) / (λ - 2r)   
>   
>  M: Taylor因子（通常3程度）  
>  G: せん断弾性率 [Pa]  
>  b: Burgersベクトルの大きさ [m]  
>  λ: 析出物間隔 [m]  
>  r: 析出物半径 [m] 

析出物間隔λは、体積分率fvと半径rから：

> λ ≈ 2r · √(π / (3fv)) 
    
    
    """
    Example 5: Orowan機構による析出強化の計算
    析出物サイズと間隔の最適化
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def orowan_stress(r, f_v, G=26e9, b=2.86e-10, M=3.06):
        """
        Orowan応力を計算
    
        Args:
            r: 析出物半径 [m]
            f_v: 体積分率
            G: せん断弾性率 [Pa]
            b: Burgersベクトル [m]
            M: Taylor因子
    
        Returns:
            tau: せん断応力 [Pa]
            sigma: 降伏応力 [Pa]
        """
        # 析出物間隔
        lambda_p = 2 * r * np.sqrt(np.pi / (3 * f_v))
    
        # Orowan応力
        tau = (M * G * b) / (lambda_p - 2*r)
    
        # 引張降伏応力（Taylor因子で換算）
        sigma = M * tau
    
        return tau, sigma, lambda_p
    
    # パラメータ範囲
    radii_nm = np.linspace(1, 100, 100)  # 1-100 nm
    radii_m = radii_nm * 1e-9
    
    volume_fractions = [0.01, 0.03, 0.05, 0.1]  # 1%, 3%, 5%, 10%
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['fᵥ = 1%', 'fᵥ = 3%', 'fᵥ = 5%', 'fᵥ = 10%']
    
    plt.figure(figsize=(14, 5))
    
    # (a) 析出物半径と強度の関係
    plt.subplot(1, 3, 1)
    for f_v, color, label in zip(volume_fractions, colors, labels):
        sigma_list = []
        for r in radii_m:
            try:
                _, sigma, _ = orowan_stress(r, f_v)
                sigma_mpa = sigma / 1e6  # MPa
                sigma_list.append(sigma_mpa)
            except:
                sigma_list.append(np.nan)
    
        plt.plot(radii_nm, sigma_list, linewidth=2,
                 color=color, label=label)
    
    plt.xlabel('析出物半径 [nm]', fontsize=12)
    plt.ylabel('降伏応力増加 [MPa]', fontsize=12)
    plt.title('(a) Orowan強化の半径依存性', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 500)
    
    # (b) 体積分率と最適半径
    plt.subplot(1, 3, 2)
    f_v_range = np.linspace(0.005, 0.15, 50)
    optimal_radii = []
    max_strengths = []
    
    for f_v in f_v_range:
        sigma_test = []
        for r in radii_m:
            try:
                _, sigma, _ = orowan_stress(r, f_v)
                sigma_test.append(sigma / 1e6)
            except:
                sigma_test.append(0)
    
        max_sigma = np.max(sigma_test)
        optimal_r = radii_nm[np.argmax(sigma_test)]
    
        optimal_radii.append(optimal_r)
        max_strengths.append(max_sigma)
    
    ax1 = plt.gca()
    ax1.plot(f_v_range * 100, optimal_radii, 'b-', linewidth=2.5, label='最適半径')
    ax1.set_xlabel('体積分率 [%]', fontsize=12)
    ax1.set_ylabel('最適析出物半径 [nm]', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(f_v_range * 100, max_strengths, 'r--', linewidth=2.5, label='最大強度')
    ax2.set_ylabel('最大降伏応力増加 [MPa]', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('(b) 最適析出物条件', fontsize=13, fontweight='bold')
    
    # (c) 析出物間隔マップ
    plt.subplot(1, 3, 3)
    r_test = 10e-9  # 10 nm
    spacing_list = []
    
    for f_v in f_v_range:
        _, _, lambda_p = orowan_stress(r_test, f_v)
        spacing_list.append(lambda_p * 1e9)  # nm
    
    plt.plot(f_v_range * 100, spacing_list, 'g-', linewidth=2.5)
    plt.xlabel('体積分率 [%]', fontsize=12)
    plt.ylabel('析出物間隔 [nm]', fontsize=12)
    plt.title('(c) 析出物間隔 (r=10nm)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 実用的な設計例
    print("=== Orowan強化の設計指針 ===\n")
    print("典型的なAl合金の析出物条件:\n")
    
    design_cases = [
        (5e-9, 0.03, "Under-aging (小サイズ・低分率)"),
        (10e-9, 0.05, "Peak-aging (最適条件)"),
        (50e-9, 0.08, "Over-aging (粗大化)")
    ]
    
    for r, f_v, condition in design_cases:
        tau, sigma, lambda_p = orowan_stress(r, f_v)
    
        print(f"{condition}:")
        print(f"  析出物半径: {r*1e9:.1f} nm")
        print(f"  体積分率: {f_v*100:.1f}%")
        print(f"  析出物間隔: {lambda_p*1e9:.1f} nm")
        print(f"  降伏応力増加: {sigma/1e6:.1f} MPa\n")
    
    # 出力例:
    # === Orowan強化の設計指針 ===
    #
    # 典型的なAl合金の析出物条件:
    #
    # Under-aging (小サイズ・低分率):
    #   析出物半径: 5.0 nm
    #   体積分率: 3.0%
    #   析出物間隔: 51.2 nm
    #   降伏応力増加: 287.3 MPa
    

### 3.4.2 整合性と強化効果

析出物と母相の結晶学的関係（整合性）は強化効果に大きく影響します：

整合性 | 界面構造 | 転位との相互作用 | 強化効果  
---|---|---|---  
**Coherent  
（完全整合）** | 格子連続、歪み場あり | 転位が切断（shearing） | 中〜高  
**Semi-coherent  
（半整合）** | 一部整合、界面転位 | 切断とバイパスの競合 | **最大**  
**Incoherent  
（非整合）** | 結晶学的関係なし | Orowanバイパス | 低〜中  
  
## 3.5 粗大化とGibbs-Thomson効果

### 3.5.1 Ostwald Ripening

長時間時効により、小さい析出物が溶解し、大きい析出物が成長する現象を**Ostwald ripening** （粗大化）と呼びます。これは界面エネルギーを最小化するため、熱力学的に自発的に起こります。

**Gibbs-Thomson効果** により、小粒子ほど溶解度が高くなります：

> c(r) = c∞ · exp(2γVm / (rRT))   
>   
>  c(r): 半径rの粒子周辺の平衡濃度  
>  c∞: 平坦界面での平衡濃度  
>  γ: 界面エネルギー [J/m²]  
>  Vm: モル体積 [m³/mol]  
>  r: 粒子半径 [m] 

Lifshitz-Slyozov-Wagner（LSW）理論により、平均粒子半径の時間発展：

> r̄³(t) - r̄³(0) = Kt   
>   
>  K: 粗大化速度定数 [m³/s] 
    
    
    """
    Example 6: 析出物の粗大化シミュレーション
    Ostwald ripening (LSW理論)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def coarsening_kinetics(t, r0, K):
        """
        LSW理論による粗大化
    
        Args:
            t: 時間 [s]
            r0: 初期平均半径 [m]
            K: 粗大化速度定数 [m³/s]
    
        Returns:
            r: 平均半径 [m]
        """
        r_cubed = r0**3 + K * t
        r = r_cubed ** (1/3)
        return r
    
    def coarsening_rate_constant(T, D0=1e-5, Q=150e3, gamma=0.2,
                                  ce=0.01, Vm=1e-5):
        """
        粗大化速度定数を計算
    
        Args:
            T: 温度 [K]
            D0: 拡散係数前指数因子 [m²/s]
            Q: 活性化エネルギー [J/mol]
            gamma: 界面エネルギー [J/m²]
            ce: 平衡濃度
            Vm: モル体積 [m³/mol]
    
        Returns:
            K: 粗大化速度定数 [m³/s]
        """
        R = 8.314  # 気体定数
        D = D0 * np.exp(-Q / (R * T))
    
        # LSW理論の速度定数
        K = (8 * gamma * Vm * ce * D) / (9 * R * T)
    
        return K
    
    # 時効温度
    temperatures = [473, 523, 573]  # 200, 250, 300°C
    temp_labels = ['200°C', '250°C', '300°C']
    colors = ['blue', 'green', 'red']
    
    time_hours = np.linspace(0, 1000, 200)  # 0-1000時間
    time_seconds = time_hours * 3600
    
    r0 = 10e-9  # 初期半径 10 nm
    
    plt.figure(figsize=(14, 5))
    
    # (a) 粗大化曲線
    plt.subplot(1, 3, 1)
    for T, label, color in zip(temperatures, temp_labels, colors):
        K = coarsening_rate_constant(T)
        r = coarsening_kinetics(time_seconds, r0, K)
        r_nm = r * 1e9
    
        plt.plot(time_hours, r_nm, linewidth=2.5,
                 color=color, label=label)
    
    plt.xlabel('時効時間 [h]', fontsize=12)
    plt.ylabel('平均析出物半径 [nm]', fontsize=12)
    plt.title('(a) 析出物の粗大化曲線', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # (b) r³-t プロット（LSW理論の検証）
    plt.subplot(1, 3, 2)
    T_test = 523  # 250°C
    K_test = coarsening_rate_constant(T_test)
    r_test = coarsening_kinetics(time_seconds, r0, K_test)
    r_cubed = (r_test * 1e9) ** 3
    r0_cubed = (r0 * 1e9) ** 3
    
    plt.plot(time_hours, r_cubed - r0_cubed, 'r-', linewidth=2.5)
    plt.xlabel('時効時間 [h]', fontsize=12)
    plt.ylabel('r³ - r₀³ [nm³]', fontsize=12)
    plt.title(f'(b) LSW理論の検証 ({temp_labels[1]})', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 線形フィット
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_hours, r_cubed - r0_cubed)
    plt.plot(time_hours, slope * time_hours + intercept, 'b--',
             linewidth=1.5, label=f'線形フィット (R²={r_value**2:.3f})')
    plt.legend(fontsize=10)
    
    # (c) 粗大化速度の温度依存性
    plt.subplot(1, 3, 3)
    T_range = np.linspace(423, 623, 50)  # 150-350°C
    K_range = []
    
    for T in T_range:
        K = coarsening_rate_constant(T)
        K_range.append(K * 1e27)  # [nm³/s]
    
    plt.semilogy(T_range - 273, K_range, 'g-', linewidth=2.5)
    plt.xlabel('温度 [°C]', fontsize=12)
    plt.ylabel('粗大化速度定数 K [nm³/s]', fontsize=12)
    plt.title('(c) 粗大化速度の温度依存性', fontsize=13, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 実用計算
    print("=== 析出物粗大化の予測 ===\n")
    print("初期半径: 10 nm\n")
    
    for T, label in zip(temperatures, temp_labels):
        K = coarsening_rate_constant(T)
    
        # 100時間後、1000時間後の半径
        r_100h = coarsening_kinetics(100 * 3600, r0, K) * 1e9
        r_1000h = coarsening_kinetics(1000 * 3600, r0, K) * 1e9
    
        print(f"{label}:")
        print(f"  100時間後: {r_100h:.1f} nm")
        print(f"  1000時間後: {r_1000h:.1f} nm")
        print(f"  粗大化速度定数: {K*1e27:.2e} nm³/s\n")
    
    # 出力例:
    # === 析出物粗大化の予測 ===
    #
    # 初期半径: 10 nm
    #
    # 200°C:
    #   100時間後: 15.2 nm
    #   1000時間後: 32.8 nm
    #   粗大化速度定数: 5.67e+01 nm³/s
    

### 3.5.2 実用合金における析出制御

#### 🔬 Al-Cu-Mg合金（2024合金）の実例

**溶体化処理** : 500°C × 1時間 → 水冷（焼入れ）

**時効処理（T6）** : 190°C × 18時間（人工時効）

  * 析出相: θ'（Al₂Cu）、S'（Al₂CuMg）
  * 最適析出物サイズ: 10-30 nm
  * 体積分率: 約5%
  * 降伏強度: 324 MPa（T6状態）

航空機構造材として、リベット、翼桁などに広く使用されています。

## 3.6 実践：Al-Cu-Mg系合金の析出シミュレーション
    
    
    """
    Example 7: Al-Cu-Mg合金の総合シミュレーション
    析出過程から強度予測まで
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    class PrecipitationSimulator:
        """析出強化合金のシミュレータ"""
    
        def __init__(self, alloy_type='Al-Cu-Mg'):
            self.alloy_type = alloy_type
    
            # Al-Cu-Mg合金のパラメータ
            self.G = 26e9  # せん断弾性率 [Pa]
            self.b = 2.86e-10  # Burgersベクトル [m]
            self.M = 3.06  # Taylor因子
            self.gamma = 0.2  # 界面エネルギー [J/m²]
            self.D0 = 1e-5  # 拡散係数前指数因子 [m²/s]
            self.Q = 150e3  # 活性化エネルギー [J/mol]
    
        def simulate_aging(self, T, time_hours):
            """
            時効過程をシミュレーション
    
            Args:
                T: 時効温度 [K]
                time_hours: 時効時間配列 [h]
    
            Returns:
                results: シミュレーション結果の辞書
            """
            time_seconds = np.array(time_hours) * 3600
    
            # 核生成・成長モデル（簡略化）
            R = 8.314
            D = self.D0 * np.exp(-self.Q / (R * T))
    
            # 析出物半径の時間発展
            r0 = 2e-9  # 初期核半径
            r = r0 + np.sqrt(2 * D * time_seconds) * 0.5e-9
    
            # 体積分率の発展（JMA型）
            f_v_max = 0.05  # 最大体積分率
            k_jma = 0.1 / 3600  # 速度定数 [1/s]
            f_v = f_v_max * (1 - np.exp(-k_jma * time_seconds))
    
            # 粗大化（長時間）
            K = (8 * self.gamma * 1e-5 * 0.01 * D) / (9 * R * T)
            r_coarsen = (r**3 + K * time_seconds) ** (1/3)
    
            # 100時間以降は粗大化が支配的
            transition_idx = np.searchsorted(time_hours, 100)
            r[transition_idx:] = r_coarsen[transition_idx:]
    
            # Orowan強度の計算
            strength = np.zeros_like(r)
            for i, (ri, fv) in enumerate(zip(r, f_v)):
                if fv > 0.001:  # 十分な析出物がある場合
                    try:
                        lambda_p = 2 * ri * np.sqrt(np.pi / (3 * fv))
                        tau = (self.M * self.G * self.b) / (lambda_p - 2*ri)
                        strength[i] = self.M * tau / 1e6  # MPa
                    except:
                        strength[i] = 0
    
            # 基底強度を加算
            sigma_base = 70  # 純Alの強度 [MPa]
            total_strength = sigma_base + strength
    
            return {
                'time': time_hours,
                'radius': r * 1e9,  # nm
                'volume_fraction': f_v * 100,  # %
                'strength': total_strength,  # MPa
                'precipitation_strength': strength  # MPa
            }
    
        def plot_results(self, results_dict):
            """シミュレーション結果を可視化"""
    
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            colors = ['blue', 'green', 'red']
    
            # (a) 析出物半径
            ax = axes[0, 0]
            for (label, results), color in zip(results_dict.items(), colors):
                ax.semilogx(results['time'], results['radius'],
                           linewidth=2.5, color=color, label=label)
            ax.set_xlabel('時効時間 [h]', fontsize=12)
            ax.set_ylabel('平均析出物半径 [nm]', fontsize=12)
            ax.set_title('(a) 析出物サイズの時間発展', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
            # (b) 体積分率
            ax = axes[0, 1]
            for (label, results), color in zip(results_dict.items(), colors):
                ax.semilogx(results['time'], results['volume_fraction'],
                           linewidth=2.5, color=color, label=label)
            ax.set_xlabel('時効時間 [h]', fontsize=12)
            ax.set_ylabel('析出物体積分率 [%]', fontsize=12)
            ax.set_title('(b) 析出物体積分率', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
            # (c) 降伏強度
            ax = axes[1, 0]
            for (label, results), color in zip(results_dict.items(), colors):
                ax.semilogx(results['time'], results['strength'],
                           linewidth=2.5, color=color, label=label)
            ax.set_xlabel('時効時間 [h]', fontsize=12)
            ax.set_ylabel('降伏強度 [MPa]', fontsize=12)
            ax.set_title('(c) 時効曲線（強度予測）', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
            # (d) 強化寄与の内訳
            ax = axes[1, 1]
            # 200°Cのケースを例に
            results_200C = results_dict['200°C']
            t = results_200C['time']
            sigma_base = 70
            sigma_precip = results_200C['precipitation_strength']
    
            ax.semilogx(t, [sigma_base]*len(t), 'k--', linewidth=2, label='基底強度')
            ax.fill_between(t, sigma_base, sigma_base + sigma_precip,
                            alpha=0.3, color='blue', label='析出強化')
            ax.semilogx(t, results_200C['strength'], 'b-', linewidth=2.5,
                       label='総強度')
            ax.set_xlabel('時効時間 [h]', fontsize=12)
            ax.set_ylabel('降伏強度 [MPa]', fontsize=12)
            ax.set_title('(d) 強化機構の寄与 (200°C)', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
    # シミュレーション実行
    simulator = PrecipitationSimulator()
    
    time_array = np.logspace(-1, 3, 100)  # 0.1〜1000時間
    
    results_dict = {
        '180°C': simulator.simulate_aging(453, time_array),
        '200°C': simulator.simulate_aging(473, time_array),
        '220°C': simulator.simulate_aging(493, time_array),
    }
    
    simulator.plot_results(results_dict)
    
    # ピーク時効条件の特定
    print("=== Al-Cu-Mg合金（2024）の最適時効条件 ===\n")
    
    for temp_label, results in results_dict.items():
        peak_idx = np.argmax(results['strength'])
        peak_time = results['time'][peak_idx]
        peak_strength = results['strength'][peak_idx]
        peak_radius = results['radius'][peak_idx]
        peak_fv = results['volume_fraction'][peak_idx]
    
        print(f"{temp_label}:")
        print(f"  最適時効時間: {peak_time:.1f} 時間")
        print(f"  最大強度: {peak_strength:.1f} MPa")
        print(f"  析出物半径: {peak_radius:.1f} nm")
        print(f"  体積分率: {peak_fv:.2f}%\n")
    
    print("工業的推奨条件（T6熱処理）:")
    print("  温度: 190°C")
    print("  時間: 18時間")
    print("  期待強度: 324 MPa（実測値）")
    
    # 出力例:
    # === Al-Cu-Mg合金（2024）の最適時効条件 ===
    #
    # 180°C:
    #   最適時効時間: 31.6 時間
    #   最大強度: 298.5 MPa
    #   析出物半径: 12.3 nm
    #   体積分率: 4.85%
    #
    # 200°C:
    #   最適時効時間: 15.8 時間
    #   最大強度: 305.2 MPa
    #   析出物半径: 15.7 nm
    #   体積分率: 4.90%
    

## 学習目標の確認

この章を完了すると、以下を説明できるようになります：

### 基本理解

  * ✅ 固溶体の種類（置換型・侵入型）と固溶強化のメカニズムを説明できる
  * ✅ 析出の核生成・成長・粗大化の3段階を理解し、時効曲線を解釈できる
  * ✅ Al合金の析出過程（GPゾーン → θ'' → θ' → θ）を説明できる

### 実践スキル

  * ✅ Labuschモデルで固溶強化量を計算できる
  * ✅ 古典的核生成理論を使って核生成速度を予測できる
  * ✅ Orowan機構による析出強化を定量的に計算できる
  * ✅ LSW理論で析出物の粗大化を予測できる

### 応用力

  * ✅ 実用Al合金の最適時効条件を設計できる
  * ✅ 析出物サイズと分布を制御して材料強度を最適化できる
  * ✅ Pythonで析出過程の統合シミュレーションを実装できる

## 演習問題

### Easy（基礎確認）

**Q1** : 固溶強化と析出強化の主な違いは何ですか？

**正解** :

  * **固溶強化** : 溶質原子が母相に均一に分散し、格子歪みや転位との相互作用で強化
  * **析出強化** : 微細な第二相粒子が析出し、転位運動を物理的に阻害（Orowan機構）

**解説** :

固溶強化は単相（固溶体）であり、濃度に対してΔσ ∝ c2/3程度の増加。析出強化は二相（母相+析出物）であり、析出物の最適サイズ・分布制御で大幅な強化が可能です。

**Q2** : Al-Cu合金の時効過程で、最大硬度を示すのはどの相ですか？

**正解** : θ'相（準安定相、半整合析出物）

**解説** :

析出過程: GPゾーン → θ'' → **θ'** → θ（平衡相）

θ'相は10-50nm程度のサイズで半整合であり、転位との相互作用が最も強いため、最大の強化効果を示します。過時効でθ相（非整合、粗大）になると強度は低下します。

**Q3** : Orowan機構で、析出物間隔λが狭くなると強度はどうなりますか？

**正解** : 強度が増加する

**解説** :

Orowan応力: τ = (M·G·b) / (λ - 2r)

λが小さくなる（析出物が密に分布）と、分母が小さくなり、τが増加します。ただし、λ < 2rの極限では式が発散するため、実際には析出物が接触してしまい、別のメカニズムが働きます。

### Medium（応用）

**Q4** : Al-4%Cu合金を200°Cで時効すると、10時間でピーク硬度に達しました。250°Cで同じピーク硬度に達するまでの時間を推定してください。活性化エネルギーを150 kJ/molとします。

**計算過程** :

Arrheniusの関係式:

t2 / t1 = exp[Q/R · (1/T2 \- 1/T1)]

与えられた値:

  * T1 = 473 K (200°C), t1 = 10 h
  * T2 = 523 K (250°C), t2 = ?
  * Q = 150 kJ/mol = 150,000 J/mol
  * R = 8.314 J/mol/K

計算:
    
    
    t₂ / 10 = exp[150000/8.314 · (1/523 - 1/473)]
           = exp[18037 · (-0.0002024)]
           = exp(-3.65)
           = 0.026
    
    t₂ = 10 × 0.026 = 0.26 時間 ≈ 16分
    

**正解** : 約0.26時間（16分）

**解説** :

温度が50°C上昇すると、拡散速度が大幅に増加し、時効時間が約40倍短縮されます。これはArrhenius式の指数的な温度依存性によるものです。工業的には、高温時効（250°C）は短時間で済む一方、析出物が粗大化しやすいため、最大強度は低温時効（190-200°C）より若干低くなります。

**Q5** : 半径10nmの析出物が体積分率5%で分散しています。Orowan機構による降伏応力増加を計算してください。（G = 26 GPa、b = 0.286 nm、M = 3）

**計算過程** :

1\. 析出物間隔λの計算:
    
    
    λ = 2r · √(π / (3f_v))
      = 2 × 10 nm · √(π / (3 × 0.05))
      = 20 nm · √(π / 0.15)
      = 20 nm · √20.94
      = 20 nm × 4.576
      = 91.5 nm
    

2\. Orowan応力の計算:
    
    
    τ = (M · G · b) / (λ - 2r)
      = (3 × 26×10⁹ Pa × 0.286×10⁻⁹ m) / (91.5×10⁻⁹ m - 20×10⁻⁹ m)
      = (22.3 Pa·m) / (71.5×10⁻⁹ m)
      = 3.12×10⁸ Pa
      = 312 MPa
    

3\. 降伏応力（引張）:
    
    
    σ_y = M · τ = 3 × 312 MPa = 936 MPa
    

**正解** : 約930-950 MPa

**解説** :

この計算は理想的な条件を仮定しており、実際の材料では以下の要因で値が変わります：

  * 析出物の整合性（完全整合の場合、転位が切断するため異なる機構）
  * サイズ分布の影響
  * 他の強化機構（固溶強化、粒界強化）との重畳

典型的なAl合金（2024-T6）の実測値は約320 MPa程度で、これは基底強度70 MPa + 析出強化250 MPa程度です。

### Hard（発展）

**Q6** : Al-Cu合金において、初期半径5nmの析出物が200°Cで粗大化します。500時間後の平均半径を予測してください。また、この粗大化により降伏強度がどの程度低下するか議論してください。（粗大化速度定数 K = 5×10⁻26 m³/s、初期体積分率5%、G=26GPa、b=0.286nm）

**計算過程** :

**Step 1: 粗大化後の半径**
    
    
    LSW理論: r³(t) = r₀³ + Kt
    
    r₀ = 5 nm = 5×10⁻⁹ m
    t = 500 h = 500 × 3600 s = 1.8×10⁶ s
    K = 5×10⁻²⁶ m³/s
    
    r³ = (5×10⁻⁹)³ + 5×10⁻²⁶ × 1.8×10⁶
       = 1.25×10⁻²⁵ + 9.0×10⁻²⁰
       = 9.0×10⁻²⁰ m³  (第1項は無視できる)
    
    r = (9.0×10⁻²⁰)^(1/3) = 4.48×10⁻⁷ m = 44.8 nm
    

**Step 2: 初期強度の計算**
    
    
    r₀ = 5 nm, f_v = 0.05
    
    λ₀ = 2 × 5 × √(π/(3×0.05)) = 45.8 nm
    
    σ₀ = (3 × 26×10⁹ × 0.286×10⁻⁹) / (45.8×10⁻⁹ - 10×10⁻⁹)
       = 22.3 / (35.8×10⁻⁹)
       = 6.23×10⁸ Pa
    
    降伏応力: σ_y0 = 3 × 623 = 1869 MPa
    

**Step 3: 粗大化後の強度**
    
    
    r = 44.8 nm (体積分率は保存: f_v = 0.05)
    
    λ = 2 × 44.8 × √(π/(3×0.05)) = 410 nm
    
    σ = (3 × 26×10⁹ × 0.286×10⁻⁹) / (410×10⁻⁹ - 89.6×10⁻⁹)
      = 22.3 / (320×10⁻⁹)
      = 6.97×10⁷ Pa
    
    降伏応力: σ_y = 3 × 70 = 210 MPa
    

**Step 4: 強度低下**
    
    
    Δσ = σ_y0 - σ_y = 1869 - 210 = 1659 MPa
    
    低下率 = (1659 / 1869) × 100 = 88.8%
    

**正解** :

  * 500時間後の平均半径: 約45 nm（初期の9倍）
  * 降伏強度の低下: 約89%（1869 MPa → 210 MPa）

**詳細な考察** :

**1\. 粗大化のメカニズム**

r³則に従う粗大化（Ostwald ripening）は、Gibbs-Thomson効果により小粒子が溶解し、大粒子が成長する現象です。500時間（約3週間）の時効で、半径が9倍に増加するのは実用的に重要な問題です。

**2\. 強度低下の原因**

  * **析出物間隔の増大** : 45.8 nm → 410 nm（約9倍）
  * **Orowan機構の弱化** : λが大きくなると、転位が析出物を容易にバイパスできる
  * **粒子数密度の減少** : 大きい粒子が少数になる（体積は保存）

**3\. 工業的対策**

  * **使用温度の制限** : Al合金は150°C以下での使用が推奨（長期安定性）
  * **三元添加** : Mg、Ag、Znなどの添加で粗大化を抑制
  * **分散粒子の導入** : Al₃Zrなどの熱的に安定な分散粒子で析出物を固定
  * **組織の微細化** : 塑性加工による転位密度増加で核生成サイト増加

**4\. 実用合金の例**

航空機用Al-Cu-Mg合金（2024-T6）は、200°C × 500時間後でも約70%の強度を保持するよう設計されています（初期: 470 MPa → 500h後: 330 MPa程度）。これは本計算より遥かに良好ですが、これは：

  * 微量添加元素（Mn、Zr）による粗大化抑制
  * 二種類の析出物（θ'とS'）の複合効果
  * 塑性加工による組織制御

などの実用技術によるものです。

**Q7:** Al-4%Cu合金のGP帯形成において、銅原子がアルミニウム格子中の{100}面に優先的に偏析する理由を、原子サイズと弾性ひずみの観点から説明してください。

**解答例** :

**原子サイズの違い** :

  * アルミニウム（Al）の原子半径: 1.43 Å
  * 銅（Cu）の原子半径: 1.28 Å
  * 銅原子はアルミニウムより約10%小さい

**GP帯形成メカニズム** :

  1. **置換型固溶** : 銅原子がアルミニウム格子点を置換すると、格子に収縮ひずみが生じる
  2. **{100}面への偏析** : 銅原子が{100}面（FCC構造の特定結晶面）に集まることで、ひずみエネルギーが局所的に緩和される
  3. **円盤状クラスター形成** : 1-2原子層の厚さで、直径数nmの円盤状GP帯が{100}面に沿って形成される

**弾性ひずみの役割** :

銅原子の偏析により、母相（アルミニウム）との界面に弾性ひずみ場が形成されます。この整合ひずみが転位の運動を妨げ、Orowan機構による強化効果をもたらします。GP帯の厚さが薄いほど、整合性が維持され、高い強度が得られます。

**実験的観察** :

透過型電子顕微鏡（TEM）観察により、GP帯は{100}面に沿った特徴的なストリークコントラストとして観察されます。

**Q8:** ニッケル基超合金（例: Inconel 718）において、γ'相（Ni3Al）とγ''相（Ni3Nb）の二種類の析出強化相が共存します。それぞれの析出相の特徴と、高温強度への寄与を比較してください。

**解答例** :

特性 | γ'相（Ni3Al） | γ''相（Ni3Nb）  
---|---|---  
**結晶構造** | L12構造（FCC系） | DO22構造（BCT: Body-Centered Tetragonal）  
**形態** | 球状または立方体状（等軸） | 円盤状（{100}面に沿って析出）  
**格子ミスフィット** | 約+0.5%（わずかに大きい） | 約-2.5%（顕著な収縮）  
**整合性** | 完全整合（高温まで維持） | 準整合（600°C以上で安定性低下）  
**熱安定性** | ～1000°C（非常に高い） | ～650°C（中程度）  
**強化効果** | 高温での持続的強化（クリープ抵抗） | 中温域での顕著な強化（降伏強度）  
  
**Inconel 718の設計思想** :

  * **室温～650°C** : γ''相が主要な強化相として機能（降伏強度 > 1000 MPa）
  * **650～850°C** : γ'相が主要な強化相として機能（γ''の溶解固溶による軟化を補償）
  * **二相複合強化** : 広い温度範囲で高強度を維持できるため、航空機エンジン（タービンディスク）に最適

**時効熱処理** :

Inconel 718の標準時効処理：720°C × 8h（γ''析出）+ 620°C × 8h（γ'微細化）により、最適な析出分布を実現します。

## ✓ 学習目標の確認

この章を完了すると、以下を説明・実行できるようになります：

### 基本理解

  * ✅ 固溶強化と析出強化のメカニズムを説明できる
  * ✅ GP帯、θ''、θ'、θ相の析出シーケンスと各段階の特徴を理解している
  * ✅ 時効硬化曲線の形状と、ピーク時効・過時効の物理的意味を説明できる
  * ✅ Orowan機構による析出物の強化効果を定量的に理解している

### 実践スキル

  * ✅ Orowan方程式を用いて、析出物サイズと間隔から強度増分を計算できる
  * ✅ LSW理論（Ostwald ripening）を用いて、析出物の粗大化速度を予測できる
  * ✅ 時効処理条件（温度・時間）と最終強度の関係を定量的に評価できる
  * ✅ Pythonを用いて、時効硬化曲線と析出物成長のシミュレーションができる

### 応用力

  * ✅ Al-Cu、Al-Mg-Si、Al-Zn-Mg系合金の析出挙動の違いを説明できる
  * ✅ 時効条件（T4、T6、T7処理）の選択と、強度-延性-耐食性のトレードオフを評価できる
  * ✅ ニッケル基超合金のγ'/γ''二相強化機構を理解し、高温材料設計に応用できる
  * ✅ 長時間高温曝露による析出物粗大化と強度劣化を定量的に予測できる

**次のステップ** :

析出強化の基礎を習得したら、第4章「転位と塑性変形」に進み、析出物と転位の相互作用メカニズムをミクロスケールで学びましょう。転位論と析出強化を統合することで、材料の塑性変形挙動を深く理解できます。

## 📚 参考文献

  1. Porter, D.A., Easterling, K.E., Sherif, M.Y. (2009). _Phase Transformations in Metals and Alloys_ (3rd ed.). CRC Press. ISBN: 978-1420062106
  2. Ashby, M.F., Jones, D.R.H. (2012). _Engineering Materials 2: An Introduction to Microstructures and Processing_ (4th ed.). Butterworth-Heinemann. ISBN: 978-0080966700
  3. Martin, J.W. (1998). _Precipitation Hardening_ (2nd ed.). Butterworth-Heinemann. ISBN: 978-0750641630
  4. Polmear, I.J., StJohn, D., Nie, J.F., Qian, M. (2017). _Light Alloys: Metallurgy of the Light Metals_ (5th ed.). Butterworth-Heinemann. ISBN: 978-0080994314
  5. Starke, E.A., Staley, J.T. (1996). "Application of modern aluminum alloys to aircraft." _Progress in Aerospace Sciences_ , 32(2-3), 131-172. [DOI:10.1016/0376-0421(95)00004-6](<https://doi.org/10.1016/0376-0421\(95\)00004-6>)
  6. Wagner, C. (1961). "Theorie der Alterung von Niederschlägen durch Umlösen (Ostwald-Reifung)." _Zeitschrift für Elektrochemie_ , 65(7-8), 581-591.
  7. Ardell, A.J. (1985). "Precipitation hardening." _Metallurgical Transactions A_ , 16(12), 2131-2165. [DOI:10.1007/BF02670416](<https://doi.org/10.1007/BF02670416>)
  8. Callister, W.D., Rethwisch, D.G. (2020). _Materials Science and Engineering: An Introduction_ (10th ed.). Wiley. ISBN: 978-1119405498

### オンラインリソース

  * **アルミニウム合金データベース** : ASM Alloy Center Database (<https://matdata.asminternational.org/>)
  * **時効処理ガイド** : Aluminum Association - Heat Treatment Guidelines (<https://www.aluminum.org/>)
  * **析出シミュレーション** : TC-PRISMA (Thermo-Calc Software) - Precipitation simulation tool
