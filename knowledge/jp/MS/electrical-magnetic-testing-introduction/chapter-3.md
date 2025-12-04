---
title: "第3章: 磁気測定"
chapter_title: "第3章: 磁気測定"
subtitle: VSM・SQUIDによる磁化測定、M-H曲線解析、磁気異方性評価、PPMS統合測定
reading_time: 50-60分
difficulty: 中級〜上級
code_examples: 7
---

磁気測定は、材料の磁気的性質（磁化、磁気モーメント、磁気異方性）を定量的に評価する技術です。この章では、VSM（Vibrating Sample Magnetometer）とSQUID（超伝導量子干渉計）の原理、M-H曲線の解析（飽和磁化、保磁力、残留磁化）、Curie-Weiss則やLangevin関数によるフィッティング、温度依存性測定（FC/ZFC）、PPMS（Physical Property Measurement System）による統合測定技術を学び、Pythonで実践的な磁気データ解析を行います。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 反磁性・常磁性・強磁性の基本原理を理解できる
  * ✅ VSMとSQUIDの測定原理と特徴を説明できる
  * ✅ M-H曲線から飽和磁化、保磁力、残留磁化を抽出できる
  * ✅ Curie-Weiss則（χ = C/(T - θ)）をフィッティングできる
  * ✅ Langevin関数で常磁性を解析できる
  * ✅ 磁気異方性エネルギーを計算できる
  * ✅ Pythonで完全なVSM/SQUIDデータ処理ワークフローを構築できる

## 3.1 磁性の基礎

### 3.1.1 磁化と磁場の関係

**磁化** $M$ は、単位体積あたりの磁気モーメントです：

$$ M = \frac{\sum \mu_i}{V} $$ 

ここで、$\mu_i$ は個々の原子・分子の磁気モーメント、$V$ は体積です。

**磁化率** $\chi$ は、磁場 $H$ に対する磁化の応答：

$$ M = \chi H $$ 

**磁場の種類** ：

  * **磁場 $H$** （magnetic field strength）：単位 A/m または Oe（エルステッド）
  * **磁束密度 $B$** （magnetic flux density）：単位 T（テスラ）
  * 関係式：$B = \mu_0(H + M)$（SI単位系）、$\mu_0 = 4\pi \times 10^{-7}$ H/m

### 3.1.2 磁性の分類

磁性 | 磁化率 $\chi$ | 温度依存性 | 特徴 | 材料例  
---|---|---|---|---  
**反磁性**  
（Diamagnetism） | $\chi < 0$  
（〜$-10^{-5}$） | ほぼ無し | 外部磁場に逆らう  
微弱な磁化 | Cu, Au, H₂O  
超伝導体  
**常磁性**  
（Paramagnetism） | $\chi > 0$  
（〜$10^{-5}$-$10^{-3}$） | $\chi \propto 1/T$  
（Curie則） | 磁場方向に  
微弱に磁化 | Al, Pt, O₂  
希土類イオン  
**強磁性**  
（Ferromagnetism） | $\chi \gg 1$  
（〜$10^2$-$10^5$） | T < T$_C$ で大  
T > T$_C$ で常磁性 | 自発磁化  
ヒステリシス | Fe, Co, Ni  
NdFeB, SmCo  
**反強磁性**  
（Antiferromagnetism） | $\chi > 0$  
（小） | T = T$_N$ でピーク | 隣接スピンが  
反平行 | MnO, Cr, FeO  
**フェリ磁性**  
（Ferrimagnetism） | $\chi \gg 1$ | 強磁性と類似 | 不等な反平行  
スピン配列 | Fe₃O₄（マグネタイト）  
フェライト  
      
    
    ```mermaid
    flowchart TD
        A[外部磁場 H] --> B{材料の磁気応答}
        B --> C[反磁性M ↓ H]
        B --> D[常磁性M ↑ HM ∝ H/T]
        B --> E[強磁性自発磁化M >> H]
        
        E --> F[飽和磁化 M_s]
        E --> G[保磁力 H_c]
        E --> H[残留磁化 M_r]
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style C fill:#ffeb99,stroke:#ffa500,stroke-width:2px
        style D fill:#99ff99,stroke:#00cc00,stroke-width:2px
        style E fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

## 3.2 VSM（Vibrating Sample Magnetometer）

### 3.2.1 VSM測定原理

**VSM** （1956年、Simon Foner開発）は、試料を振動させ、その磁気モーメントによって検出コイルに誘起される電圧を測定する手法です。

**測定原理** ：

  1. 試料を均一磁場中に設置し、周波数 $f$（典型的には 10-100 Hz）で上下に振動させる
  2. 試料の磁気モーメント $\mu$ が振動すると、周囲の検出コイルに時間変化する磁束 $\Phi(t)$ が生じる
  3. Faradayの電磁誘導則により、コイルに誘起電圧 $V(t)$ が発生： $$ V(t) = -\frac{d\Phi}{dt} \propto \mu \cdot f $$ 
  4. ロックイン検出により、振動周波数 $f$ の成分を抽出し、磁気モーメントを定量

**VSMの特徴** ：

  * **感度** ：$10^{-6}$ - $10^{-8}$ emu（electromagnetic unit）
  * **磁場範囲** ：0 - 3 T（典型的な電磁石）
  * **温度範囲** ：4 K - 1000 K（液体Heクライオスタット使用時）
  * **測定時間** ：1点あたり 0.1 - 1秒（比較的高速）

#### コード例3-1: Curie-Weiss則のフィッティング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    
    def curie_weiss(T, C, theta):
        """
        Curie-Weiss則: χ = C / (T - θ)
        
        Parameters
        ----------
        T : array-like
            温度 [K]
        C : float
            Curie定数
        theta : float
            Curie-Weiss温度（Weiss定数）[K]
        
        Returns
        -------
        chi : array-like
            磁化率
        """
        return C / (T - theta)
    
    # シミュレーションデータ生成（常磁性材料）
    T_range = np.linspace(100, 400, 30)  # [K]
    C_true = 1.5  # Curie定数
    theta_true = -10  # Curie-Weiss温度 [K]（負 → 反強磁性的相互作用）
    
    chi_data = curie_weiss(T_range, C_true, theta_true)
    chi_data_noise = chi_data * (1 + 0.05 * np.random.randn(len(T_range)))  # 5%ノイズ
    
    # フィッティング
    model = Model(curie_weiss)
    params = model.make_params(C=1.0, theta=0)
    result = model.fit(chi_data_noise, params, T=T_range)
    
    print("Curie-Weiss則フィッティング結果:")
    print(result.fit_report())
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図：χ vs T
    ax1.scatter(T_range, chi_data_noise, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Measured data', color='#f093fb')
    ax1.plot(T_range, result.best_fit, linewidth=2.5, label=f'Fit: C={result.params["C"].value:.2f}, θ={result.params["theta"].value:.1f} K', color='#f5576c')
    ax1.set_xlabel('Temperature T [K]', fontsize=12)
    ax1.set_ylabel('Magnetic Susceptibility χ', fontsize=12)
    ax1.set_title('Curie-Weiss Law: χ vs T', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # 右図：1/χ vs T（Curie-Weissプロット）
    ax2.scatter(T_range, 1/chi_data_noise, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data (1/χ)', color='#ffa500')
    T_fit_extended = np.linspace(0, 450, 100)
    chi_fit_extended = curie_weiss(T_fit_extended, result.params['C'].value, result.params['theta'].value)
    ax2.plot(T_fit_extended, 1/chi_fit_extended, linewidth=2.5, label='Linear fit (extended)', color='#f5576c')
    
    # θ を強調表示
    theta_fit = result.params['theta'].value
    ax2.axvline(theta_fit, color='red', linestyle='--', linewidth=2, label=f'θ = {theta_fit:.1f} K')
    ax2.scatter([theta_fit], [0], s=200, c='red', edgecolors='black', linewidth=2, marker='X', zorder=5)
    
    ax2.set_xlabel('Temperature T [K]', fontsize=12)
    ax2.set_ylabel('1/χ', fontsize=12)
    ax2.set_title('Curie-Weiss Plot: 1/χ vs T', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 450)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n物理的解釈:")
    print(f"  Curie定数 C = {result.params['C'].value:.2f}")
    print(f"  Curie-Weiss温度 θ = {result.params['theta'].value:.1f} K")
    if result.params['theta'].value < 0:
        print(f"  θ < 0 → 反強磁性的相互作用が支配的")
    elif result.params['theta'].value > 0:
        print(f"  θ > 0 → 強磁性的相互作用が支配的")
    else:
        print(f"  θ ≈ 0 → 理想的な常磁性（相互作用なし）")
    

## 3.3 SQUID（超伝導量子干渉計）

### 3.3.1 SQUID測定原理

**SQUID** （Superconducting Quantum Interference Device）は、超伝導リングとJosephson接合を利用した超高感度磁気センサーです。

**測定原理** ：

  1. 超伝導リングに磁束 $\Phi$ が貫通すると、量子化された超伝導電流が流れる
  2. Josephson接合を通る臨界電流が、磁束に対して周期的に変化： $$ I_c(\Phi) = I_0 \left|\cos\left(\frac{\pi\Phi}{\Phi_0}\right)\right| $$ ここで、$\Phi_0 = h/(2e) \approx 2.07 \times 10^{-15}$ Wb は磁束量子 
  3. 磁束の微小変化を、臨界電流の変化として極めて高感度に検出

**SQUIDの特徴** ：

  * **感度** ：$10^{-10}$ - $10^{-12}$ emu（VSMの1000倍以上）
  * **磁場範囲** ：0 - 7 T（超伝導マグネット）
  * **温度範囲** ：1.8 K - 400 K（Heガス使用時）
  * **測定時間** ：1点あたり 1 - 10秒（VSMより遅いが、高精度）
  * **ノイズレベル** ：$10^{-14}$ T/√Hz（世界最高感度の磁気センサー）

#### コード例3-2: Langevin関数フィッティング（常磁性）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    
    def langevin(H, M_s, T, mu):
        """
        Langevin関数（古典常磁性）
        
        Parameters
        ----------
        H : array-like
            磁場 [A/m または Oe]
        M_s : float
            飽和磁化 [emu/g または Am^2/kg]
        T : float
            温度 [K]
        mu : float
            粒子あたりの磁気モーメント [Bohr magneton単位]
        
        Returns
        -------
        M : array-like
            磁化 [emu/g]
        """
        mu_B = 9.274e-24  # Bohr magneton [J/T]
        mu_0 = 4 * np.pi * 1e-7  # 真空の透磁率 [H/m]
        k_B = 1.38065e-23  # Boltzmann定数 [J/K]
        
        # 磁場を SI単位に変換（Oe → T）
        H_T = H * 1e-4  # 1 Oe = 10^-4 T
        
        # Langevinパラメータ
        xi = mu * mu_B * H_T / (k_B * T)
        
        # Langevin関数: L(ξ) = coth(ξ) - 1/ξ
        L = np.where(np.abs(xi) < 1e-3, xi/3, 1/np.tanh(xi) - 1/xi)  # 小さいξでの発散を回避
        
        M = M_s * L
        return M
    
    # シミュレーションデータ生成（超常磁性ナノ粒子）
    H_range = np.linspace(-10000, 10000, 100)  # [Oe]
    M_s_true = 50  # 飽和磁化 [emu/g]
    T_true = 300  # 温度 [K]
    mu_true = 5000  # 粒子磁気モーメント [μ_B]
    
    M_data = langevin(H_range, M_s_true, T_true, mu_true)
    M_data_noise = M_data + 0.5 * np.random.randn(len(H_range))  # ノイズ追加
    
    # フィッティング
    model = Model(langevin)
    params = model.make_params(M_s=40, T=300, mu=3000)
    params['T'].vary = False  # 温度は固定
    
    result = model.fit(M_data_noise, params, H=H_range)
    
    print("Langevin関数フィッティング結果:")
    print(result.fit_report())
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図：M vs H
    ax1.scatter(H_range, M_data_noise, s=50, alpha=0.6, edgecolors='black', linewidths=1, label='Data (with noise)', color='#f093fb')
    ax1.plot(H_range, result.best_fit, linewidth=2.5, label=f'Fit: M_s={result.params["M_s"].value:.1f} emu/g, μ={result.params["mu"].value:.0f} μ_B', color='#f5576c')
    ax1.axhline(M_s_true, color='green', linestyle='--', linewidth=1.5, label=f'True M_s = {M_s_true} emu/g')
    ax1.axhline(-M_s_true, color='green', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Magnetic Field H [Oe]', fontsize=12)
    ax1.set_ylabel('Magnetization M [emu/g]', fontsize=12)
    ax1.set_title('Langevin Function Fit (Paramagnetism)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 右図：温度依存性（異なる温度でのLangevin曲線）
    T_list = [50, 100, 200, 300, 400]
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(T_list)))
    
    for T, color in zip(T_list, colors):
        M_T = langevin(H_range, result.params['M_s'].value, T, result.params['mu'].value)
        ax2.plot(H_range, M_T, linewidth=2.5, label=f'T = {T} K', color=color)
    
    ax2.set_xlabel('Magnetic Field H [Oe]', fontsize=12)
    ax2.set_ylabel('Magnetization M [emu/g]', fontsize=12)
    ax2.set_title('Temperature Dependence of Langevin Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n物理的解釈:")
    print(f"  飽和磁化 M_s = {result.params['M_s'].value:.1f} emu/g")
    print(f"  粒子磁気モーメント μ = {result.params['mu'].value:.0f} μ_B")
    print(f"  温度が低いほど、低磁場で飽和に近づく（熱揺らぎ減少）")
    

## 3.4 M-H曲線解析

### 3.4.1 M-H曲線の特徴量

**強磁性材料のM-H曲線** （磁化曲線、ヒステリシスループ）から、以下の特性値を抽出します：

特性値 | 記号 | 定義 | 物理的意味  
---|---|---|---  
**飽和磁化** | $M_s$ | 高磁場での磁化 | 全磁気モーメントが  
磁場方向に整列  
**残留磁化** | $M_r$ | H = 0 での磁化 | 磁場除去後に  
残る磁化  
**保磁力** | $H_c$ | M = 0 となる磁場 | 磁化を反転させる  
必要な磁場  
**角形比** | $S = M_r/M_s$ | 残留磁化と  
飽和磁化の比 | S ≈ 1：角形性良好  
（永久磁石向き）  
      
    
    ```mermaid
    flowchart LR
        A[M-H曲線測定] --> B[バックグラウンド減算]
        B --> C[飽和磁化 M_s高磁場外挿]
        B --> D[保磁力 H_cM=0 交点]
        B --> E[残留磁化 M_rH=0 での M]
        
        C --> F[磁気モーメントμ = M_s × 質量]
        D --> G[磁気異方性K ∝ H_c × M_s]
        E --> H[角形比S = M_r / M_s]
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#ffa500,stroke:#ff8c00,stroke-width:2px
        style E fill:#99ff99,stroke:#00cc00,stroke-width:2px
    ```

#### コード例3-3: M-H曲線の処理と特性値抽出
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    
    def generate_mh_curve(H, M_s, H_c, M_r, slope_high_field=0):
        """
        理想的な強磁性M-H曲線を生成（簡略化モデル）
        """
        # 初磁化曲線（簡略化tanh関数）
        M_init = M_s * np.tanh(H / (0.3 * H_c))
        
        # ヒステリシス（実際は複雑だが、簡易的にシフト）
        M_up = M_s * np.tanh((H - H_c) / (0.3 * H_c)) + M_r
        M_down = -M_s * np.tanh((-H - H_c) / (0.3 * H_c)) - M_r
        
        # 高磁場での線形バックグラウンド（常磁性成分）
        M_total = np.where(H > 0, M_up, M_down) + slope_high_field * H
        
        return M_total
    
    def extract_mh_parameters(H, M):
        """
        M-H曲線から特性値を抽出
        
        Parameters
        ----------
        H : array-like
            磁場 [Oe]
        M : array-like
            磁化 [emu/g]
        
        Returns
        -------
        params : dict
            M_s, H_c, M_r, S（角形比）
        """
        # 飽和磁化：高磁場での平均値
        M_s = np.mean(M[H > 0.8 * np.max(H)])
        
        # 保磁力：M = 0 の交点
        interp_func = interp1d(H, M, kind='linear')
        H_c = brentq(interp_func, np.min(H[H < 0]), np.max(H[H > 0]))
        
        # 残留磁化：H = 0 での値
        M_r = interp_func(0)
        
        # 角形比
        S = M_r / M_s if M_s != 0 else 0
        
        params = {
            'M_s': M_s,
            'H_c': H_c,
            'M_r': M_r,
            'S': S
        }
        
        return params
    
    # シミュレーションデータ生成
    H_range = np.linspace(-5000, 5000, 200)  # [Oe]
    M_s_true = 100  # [emu/g]
    H_c_true = 500  # [Oe]
    M_r_true = 80  # [emu/g]
    slope_bg = 1e-3  # 高磁場バックグラウンド
    
    M_data = generate_mh_curve(H_range, M_s_true, H_c_true, M_r_true, slope_bg)
    M_data_noise = M_data + 1.0 * np.random.randn(len(H_range))
    
    # バックグラウンド減算（高磁場の線形成分）
    H_high = H_range[H_range > 3000]
    M_high = M_data_noise[H_range > 3000]
    slope_fit = np.polyfit(H_high, M_high, 1)[0]
    M_corrected = M_data_noise - slope_fit * H_range
    
    # 特性値抽出
    params = extract_mh_parameters(H_range, M_corrected)
    
    print("M-H曲線解析結果:")
    print("=" * 60)
    print(f"飽和磁化 M_s = {params['M_s']:.2f} emu/g")
    print(f"保磁力 H_c = {params['H_c']:.2f} Oe = {params['H_c'] / 79.5775:.2f} kA/m")
    print(f"残留磁化 M_r = {params['M_r']:.2f} emu/g")
    print(f"角形比 S = M_r/M_s = {params['S']:.3f}")
    print(f"\n材料評価:")
    if params['H_c'] < 100:
        print("  軟磁性材料（トランスフォーマー、インダクタ向き）")
    elif params['H_c'] > 1000:
        print("  硬磁性材料（永久磁石向き）")
    else:
        print("  中程度の保磁力（記録媒体向き）")
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図：完全なM-H曲線
    ax1.plot(H_range, M_corrected, linewidth=2.5, color='#f093fb', label='M-H curve (background subtracted)')
    ax1.axhline(params['M_s'], color='green', linestyle='--', linewidth=1.5, label=f'M_s = {params["M_s"]:.1f} emu/g')
    ax1.axhline(params['M_r'], color='orange', linestyle='--', linewidth=1.5, label=f'M_r = {params["M_r"]:.1f} emu/g')
    ax1.axvline(params['H_c'], color='red', linestyle='--', linewidth=1.5, label=f'H_c = {params["H_c"]:.0f} Oe')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.axvline(0, color='black', linestyle='-', linewidth=1)
    ax1.scatter([params['H_c']], [0], s=150, c='red', edgecolors='black', linewidth=2, marker='o', zorder=5)
    ax1.scatter([0], [params['M_r']], s=150, c='orange', edgecolors='black', linewidth=2, marker='s', zorder=5)
    ax1.set_xlabel('Magnetic Field H [Oe]', fontsize=12)
    ax1.set_ylabel('Magnetization M [emu/g]', fontsize=12)
    ax1.set_title('M-H Hysteresis Loop', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(alpha=0.3)
    
    # 右図：ヒステリシスループのズーム（低磁場領域）
    H_zoom = H_range[np.abs(H_range) < 1500]
    M_zoom = M_corrected[np.abs(H_range) < 1500]
    
    ax2.plot(H_zoom, M_zoom, linewidth=2.5, color='#f5576c')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(params['H_c'], color='red', linestyle='--', linewidth=2, label=f'H_c = {params["H_c"]:.0f} Oe')
    ax2.axvline(-params['H_c'], color='red', linestyle='--', linewidth=2)
    ax2.scatter([params['H_c']], [0], s=150, c='red', edgecolors='black', linewidth=2, marker='o', zorder=5)
    ax2.scatter([0], [params['M_r']], s=150, c='orange', edgecolors='black', linewidth=2, marker='s', zorder=5, label=f'M_r = {params["M_r"]:.1f} emu/g')
    ax2.set_xlabel('Magnetic Field H [Oe]', fontsize=12)
    ax2.set_ylabel('Magnetization M [emu/g]', fontsize=12)
    ax2.set_title('Hysteresis Loop (Zoom)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

## 3.5 磁気異方性エネルギー

### 3.5.1 磁気異方性の起源

**磁気異方性** は、磁化が特定の結晶方位に向きやすい性質です。保磁力の主要因の一つです。

**磁気異方性エネルギー** $K$ は、単軸異方性の場合：

$$ E = K \sin^2\theta $$ 

ここで、$\theta$ は磁化と異方性軸（easy axis）のなす角度です。

**保磁力との関係** （Stoner-Wohlfarth理論、単磁区粒子）：

$$ H_c = \frac{2K}{M_s} $$ 

**異方性の種類** ：

  * **結晶異方性** ：結晶構造に起因（立方晶、六方晶など）
  * **形状異方性** ：試料形状による反磁場効果
  * **誘導異方性** ：磁場中での熱処理で誘起
  * **表面・界面異方性** ：ナノ材料で重要

#### コード例3-4: 磁気異方性エネルギーの計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_anisotropy_constant(H_c, M_s):
        """
        保磁力から磁気異方性定数を計算
        
        Parameters
        ----------
        H_c : float
            保磁力 [Oe]
        M_s : float
            飽和磁化 [emu/g] または [emu/cm^3]
        
        Returns
        -------
        K : float
            磁気異方性定数 [erg/cm^3]（CGS単位）
            または [J/m^3]（SI単位）に変換可能（× 10^3）
        """
        # Stoner-Wohlfarth理論: H_c = 2K / M_s
        K = H_c * M_s / 2
        return K
    
    # 材料例
    materials = {
        'Soft magnetic (Fe-Si)': {'H_c': 5, 'M_s': 1700},  # [Oe], [emu/cm^3]
        'Recording media (CoCrPt)': {'H_c': 3000, 'M_s': 400},
        'Hard magnet (NdFeB)': {'H_c': 12000, 'M_s': 1280},
        'Nanoparticle (Fe3O4)': {'H_c': 300, 'M_s': 480}
    }
    
    print("磁気異方性定数の計算:")
    print("=" * 80)
    print(f"{'Material':<30} {'H_c [Oe]':<12} {'M_s [emu/cm³]':<18} {'K [10⁶ erg/cm³]':<20}")
    print("-" * 80)
    
    K_values = {}
    for name, props in materials.items():
        K = calculate_anisotropy_constant(props['H_c'], props['M_s'])
        K_values[name] = K
        print(f"{name:<30} {props['H_c']:<12.0f} {props['M_s']:<18.0f} {K / 1e6:<20.2f}")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図：異方性エネルギー曲線
    theta = np.linspace(0, 180, 100)  # [deg]
    theta_rad = np.deg2rad(theta)
    
    # 異なるKでのエネルギー曲線
    K_list = [1e5, 5e5, 1e6, 5e6]  # [erg/cm^3]
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(K_list)))
    
    for K, color in zip(K_list, colors):
        E = K * np.sin(theta_rad)**2
        ax1.plot(theta, E / 1e6, linewidth=2.5, label=f'K = {K / 1e6:.1f} × 10⁶ erg/cm³', color=color)
    
    ax1.set_xlabel('Angle θ [deg]', fontsize=12)
    ax1.set_ylabel('Anisotropy Energy E [10⁶ erg/cm³]', fontsize=12)
    ax1.set_title('Uniaxial Anisotropy Energy: E = K sin²θ', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 180)
    
    # 右図：材料比較（K vs H_c）
    names = list(K_values.keys())
    K_vals = [K_values[n] / 1e6 for n in names]
    H_c_vals = [materials[n]['H_c'] for n in names]
    
    ax2.scatter(H_c_vals, K_vals, s=200, edgecolors='black', linewidths=2, c=range(len(names)), cmap='plasma', zorder=5)
    for i, name in enumerate(names):
        ax2.annotate(name.split('(')[1].replace(')', ''), (H_c_vals[i], K_vals[i]), 
                     xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Coercivity H$_c$ [Oe]', fontsize=12)
    ax2.set_ylabel('Anisotropy Constant K [10⁶ erg/cm³]', fontsize=12)
    ax2.set_title('Material Comparison: K vs H$_c$', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n物理的解釈:")
    print(f"  軟磁性材料（Fe-Si）: K ≈ 10⁴ erg/cm³（低異方性 → 低保磁力）")
    print(f"  硬磁性材料（NdFeB）: K ≈ 10⁷ erg/cm³（高異方性 → 高保磁力）")
    print(f"  K が大きいほど、磁化を反転させるのに大きなエネルギーが必要")
    

## 3.6 温度依存性磁気測定（FC/ZFC）

### 3.6.1 FC/ZFC測定法

**FC（Field-Cooled）/ ZFC（Zero-Field-Cooled）測定** は、超常磁性転移や磁気相転移を調べる標準的な手法です。

**測定手順** ：

  1. **ZFC（ゼロ磁場冷却）** ： 
     * 磁場 H = 0 で高温から低温（例：5 K）まで冷却
     * 低温で小さな磁場（例：100 Oe）を印加
     * 温度を上げながら磁化 $M_{\text{ZFC}}(T)$ を測定
  2. **FC（磁場中冷却）** ： 
     * 測定磁場（100 Oe）を印加したまま高温から低温まで冷却
     * 温度を上げながら磁化 $M_{\text{FC}}(T)$ を測定

**典型的なFC/ZFCパターン** ：

  * **超常磁性ナノ粒子** ：ZFCはT$_B$（ブロッキング温度）でピーク、FCは単調減少
  * **スピングラス** ：ZFC/FCが分岐し、T$_f$（凍結温度）以下で不可逆
  * **強磁性転移** ：T$_C$（Curie温度）で急激な減少

#### コード例3-5: FC/ZFC曲線の解析（ブロッキング温度）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def generate_fc_zfc(T, T_B, M_high, tau_0=1e-9, K=1e5, V=1e-24):
        """
        超常磁性ナノ粒子のFC/ZFC曲線を生成
        
        Parameters
        ----------
        T : array-like
            温度 [K]
        T_B : float
            ブロッキング温度 [K]
        M_high : float
            高温での磁化 [emu/g]
        tau_0 : float
            測定時間スケール [s]
        K : float
            異方性定数 [erg/cm^3]
        V : float
            粒子体積 [cm^3]
        
        Returns
        -------
        M_FC, M_ZFC : array-like
            FC/ZFC磁化曲線
        """
        k_B = 1.38065e-16  # Boltzmann定数 [erg/K]
        
        # Néel緩和時間: τ = τ₀ exp(KV / k_B T)
        # ブロッキング温度: T_B ≈ KV / (25 k_B) for τ_measure = 100 s
        
        # ZFC: T < T_B でブロック（磁化小）、T > T_B で超常磁性（磁化増加）
        M_ZFC = M_high * (1 - np.exp(-(T / T_B)**3))
        
        # FC: 低温でも磁化は保持（磁場中で冷却）
        M_FC = M_high * (1 - 0.3 * np.exp(-(T / (1.5 * T_B))**2))
        
        return M_FC, M_ZFC
    
    # シミュレーションデータ生成
    T_range = np.linspace(5, 300, 100)  # [K]
    T_B_true = 50  # ブロッキング温度 [K]
    M_high_true = 20  # 高温磁化 [emu/g]
    
    M_FC, M_ZFC = generate_fc_zfc(T_range, T_B_true, M_high_true)
    M_FC_noise = M_FC + 0.3 * np.random.randn(len(T_range))
    M_ZFC_noise = M_ZFC + 0.3 * np.random.randn(len(T_range))
    
    # ブロッキング温度抽出（ZFCピーク）
    T_B_measured = T_range[np.argmax(M_ZFC_noise)]
    
    print("FC/ZFC測定解析:")
    print("=" * 60)
    print(f"ブロッキング温度 T_B = {T_B_measured:.1f} K（ZFCピーク位置）")
    print(f"真の値 T_B = {T_B_true} K")
    print(f"\n物理的解釈:")
    print(f"  T < T_B: 磁気モーメントがブロック状態（測定時間内で反転しない）")
    print(f"  T > T_B: 超常磁性状態（磁気モーメントが熱揺らぎで反転）")
    print(f"  FC/ZFCの分岐 → 磁気相互作用の存在を示唆")
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図：FC/ZFC曲線
    ax1.plot(T_range, M_FC_noise, 'o-', linewidth=2.5, markersize=5, label='FC (Field-Cooled)', color='#f5576c')
    ax1.plot(T_range, M_ZFC_noise, 's-', linewidth=2.5, markersize=5, label='ZFC (Zero-Field-Cooled)', color='#99ccff')
    ax1.axvline(T_B_measured, color='red', linestyle='--', linewidth=2, label=f'T_B = {T_B_measured:.1f} K')
    ax1.scatter([T_B_measured], [np.max(M_ZFC_noise)], s=200, c='red', edgecolors='black', linewidth=2, marker='X', zorder=5)
    ax1.set_xlabel('Temperature T [K]', fontsize=12)
    ax1.set_ylabel('Magnetization M [emu/g]', fontsize=12)
    ax1.set_title('FC/ZFC Magnetization (Superparamagnetic Nanoparticles)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # 右図：FC-ZFC差（不可逆性）
    Delta_M = M_FC_noise - M_ZFC_noise
    ax2.plot(T_range, Delta_M, linewidth=2.5, color='#ffa500')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(T_B_measured, color='red', linestyle='--', linewidth=2, label=f'T_B = {T_B_measured:.1f} K')
    ax2.fill_between(T_range, 0, Delta_M, where=(Delta_M > 0), alpha=0.3, color='#ffa500', label='Irreversibility region')
    ax2.set_xlabel('Temperature T [K]', fontsize=12)
    ax2.set_ylabel('ΔM = M$_{FC}$ - M$_{ZFC}$ [emu/g]', fontsize=12)
    ax2.set_title('FC-ZFC Irreversibility', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

## 3.7 PPMS（Physical Property Measurement System）

### 3.7.1 PPMSの概要

**PPMS** （Quantum Design社）は、磁気・電気・熱物性を統合測定できる汎用システムです。

**測定可能な物性** ：

  * 磁化（VSMモジュール、感度 $10^{-6}$ emu）
  * 電気伝導率・Hall効果（ACトランスポートモジュール）
  * 熱容量（熱緩和法）
  * 比熱（Semi-adiabatic法）
  * 熱電性能（Seebeck係数、熱伝導率）

**測定範囲** ：

  * **温度** ：1.8 K - 400 K（$^3$He冷凍機で 0.4 K まで拡張可）
  * **磁場** ：±9 T（または ±14 T）
  * **自動制御** ：温度・磁場・測定シーケンスを完全自動化

#### コード例3-6: 完全な磁気データ処理ワークフロー（VSM/SQUID）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    
    class MagneticDataProcessor:
        """VSM/SQUIDデータの完全処理クラス"""
    
        def __init__(self, sample_mass, sample_volume=None):
            """
            Parameters
            ----------
            sample_mass : float
                試料質量 [g]
            sample_volume : float or None
                試料体積 [cm^3]（質量磁化を体積磁化に変換する場合）
            """
            self.mass = sample_mass
            self.volume = sample_volume
            self.data = {}
    
        def load_data(self, filename=None, mock_data=True):
            """測定データの読み込み"""
            if mock_data:
                # モックデータ生成（M-H曲線）
                H = np.linspace(-5000, 5000, 100)  # [Oe]
                M_s = 50  # [emu/g]
                H_c = 300
                M_r = 40
                
                M_raw = M_s * np.tanh(H / (0.3 * H_c))
                M = np.where(H > 0, 
                             M_s * np.tanh((H - H_c) / (0.3 * H_c)) + M_r,
                             -M_s * np.tanh((-H - H_c) / (0.3 * H_c)) - M_r)
                M += 0.002 * H  # 常磁性バックグラウンド
                M += 0.5 * np.random.randn(len(H))  # ノイズ
    
                self.data = pd.DataFrame({'H': H, 'M_raw': M})
            else:
                # 実データをCSVから読み込み
                self.data = pd.read_csv(filename)
    
            return self.data
    
        def subtract_background(self, H_min=3000):
            """高磁場バックグラウンドの減算"""
            H_high = self.data['H'][self.data['H'] > H_min]
            M_high = self.data['M_raw'][self.data['H'] > H_min]
            
            # 線形フィッティング
            slope = np.polyfit(H_high, M_high, 1)[0]
            self.data['M'] = self.data['M_raw'] - slope * self.data['H']
            
            print(f"Background subtracted: slope = {slope:.2e}")
    
        def extract_parameters(self):
            """M-H曲線パラメータの抽出"""
            H = self.data['H'].values
            M = self.data['M'].values
    
            # 飽和磁化
            self.M_s = np.mean(M[H > 0.8 * np.max(H)])
    
            # 保磁力
            interp_func = interp1d(H, M, kind='linear')
            self.H_c = brentq(interp_func, np.min(H[H < 0]), np.max(H[H > 0]))
    
            # 残留磁化
            self.M_r = interp_func(0)
    
            # 角形比
            self.S = self.M_r / self.M_s if self.M_s != 0 else 0
    
            # 磁気異方性定数
            self.K = self.H_c * self.M_s / 2  # [erg/cm^3]（CGS）
    
            results = {
                'M_s': self.M_s,
                'H_c': self.H_c,
                'M_r': self.M_r,
                'S': self.S,
                'K': self.K
            }
    
            return results
    
        def plot_mh_curve(self):
            """M-H曲線の可視化"""
            fig, ax = plt.subplots(figsize=(10, 7))
    
            ax.plot(self.data['H'], self.data['M'], linewidth=2.5, color='#f093fb', label='M-H curve')
            ax.axhline(self.M_s, color='green', linestyle='--', linewidth=1.5, label=f'M_s = {self.M_s:.1f} emu/g')
            ax.axhline(self.M_r, color='orange', linestyle='--', linewidth=1.5, label=f'M_r = {self.M_r:.1f} emu/g')
            ax.axvline(self.H_c, color='red', linestyle='--', linewidth=1.5, label=f'H_c = {self.H_c:.0f} Oe')
            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
    
            ax.scatter([self.H_c], [0], s=150, c='red', edgecolors='black', linewidth=2, marker='o', zorder=5)
            ax.scatter([0], [self.M_r], s=150, c='orange', edgecolors='black', linewidth=2, marker='s', zorder=5)
    
            ax.set_xlabel('Magnetic Field H [Oe]', fontsize=12)
            ax.set_ylabel('Magnetization M [emu/g]', fontsize=12)
            ax.set_title('M-H Hysteresis Loop Analysis', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
        def save_results(self, filename='magnetic_results.csv'):
            """結果をCSVに保存"""
            self.data.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
    
    # 使用例
    processor = MagneticDataProcessor(sample_mass=0.005)  # 5 mg
    processor.load_data(mock_data=True)
    processor.subtract_background(H_min=3000)
    results = processor.extract_parameters()
    
    print("\n磁気測定解析結果:")
    print("=" * 60)
    print(f"飽和磁化 M_s = {results['M_s']:.2f} emu/g")
    print(f"保磁力 H_c = {results['H_c']:.2f} Oe = {results['H_c'] / 79.5775:.2f} kA/m")
    print(f"残留磁化 M_r = {results['M_r']:.2f} emu/g")
    print(f"角形比 S = {results['S']:.3f}")
    print(f"磁気異方性定数 K = {results['K']:.2e} erg/cm³ = {results['K'] * 1e3:.2e} J/m³")
    
    processor.plot_mh_curve()
    processor.save_results('vsm_analysis_output.csv')
    

#### コード例3-7: 多温度M-H曲線の一括解析
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def analyze_multi_temperature_mh(T_list, H_range):
        """
        複数温度でのM-H曲線を生成・解析
        
        Parameters
        ----------
        T_list : list
            温度リスト [K]
        H_range : array-like
            磁場範囲 [Oe]
        
        Returns
        -------
        results : dict
            各温度での M_s, H_c, M_r
        """
        results = {'T': T_list, 'M_s': [], 'H_c': [], 'M_r': []}
    
        for T in T_list:
            # 温度依存性を含むM-H曲線生成
            M_s_T = 100 * (1 - (T / 400)**2)  # 飽和磁化の温度依存性
            H_c_T = 500 * (1 - (T / 400)**1.5)  # 保磁力の温度依存性
            M_r_T = 0.8 * M_s_T  # 残留磁化
    
            # 簡易M-H曲線
            M = np.where(H_range > 0,
                         M_s_T * np.tanh((H_range - H_c_T) / (0.3 * H_c_T)) + M_r_T,
                         -M_s_T * np.tanh((-H_range - H_c_T) / (0.3 * H_c_T)) - M_r_T)
    
            results['M_s'].append(M_s_T)
            results['H_c'].append(H_c_T)
            results['M_r'].append(M_r_T)
    
        return results
    
    # 温度範囲
    T_list = [50, 100, 150, 200, 250, 300]  # [K]
    H_range = np.linspace(-5000, 5000, 200)
    
    # 解析実行
    results = analyze_multi_temperature_mh(T_list, H_range)
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上：M_s vs T
    axes[0, 0].plot(results['T'], results['M_s'], 'o-', linewidth=2.5, markersize=8, color='#f093fb')
    axes[0, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 0].set_ylabel('Saturation Magnetization M$_s$ [emu/g]', fontsize=12)
    axes[0, 0].set_title('M$_s$ vs Temperature', fontsize=13, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 右上：H_c vs T
    axes[0, 1].plot(results['T'], results['H_c'], 's-', linewidth=2.5, markersize=8, color='#f5576c')
    axes[0, 1].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 1].set_ylabel('Coercivity H$_c$ [Oe]', fontsize=12)
    axes[0, 1].set_title('H$_c$ vs Temperature', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 左下：M_r vs T
    axes[1, 0].plot(results['T'], results['M_r'], '^-', linewidth=2.5, markersize=8, color='#ffa500')
    axes[1, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[1, 0].set_ylabel('Remanence M$_r$ [emu/g]', fontsize=12)
    axes[1, 0].set_title('M$_r$ vs Temperature', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 右下：角形比 S vs T
    S = np.array(results['M_r']) / np.array(results['M_s'])
    axes[1, 1].plot(results['T'], S, 'd-', linewidth=2.5, markersize=8, color='#99ccff')
    axes[1, 1].set_xlabel('Temperature T [K]', fontsize=12)
    axes[1, 1].set_ylabel('Squareness S = M$_r$ / M$_s$', fontsize=12)
    axes[1, 1].set_title('Squareness vs Temperature', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("温度依存性解析結果:")
    print("=" * 80)
    print(f"{'T [K]':<10} {'M_s [emu/g]':<15} {'H_c [Oe]':<15} {'M_r [emu/g]':<15} {'S':<10}")
    print("-" * 80)
    for i, T in enumerate(results['T']):
        print(f"{T:<10.0f} {results['M_s'][i]:<15.1f} {results['H_c'][i]:<15.1f} {results['M_r'][i]:<15.1f} {S[i]:<10.3f}")
    
    print(f"\n傾向:")
    print(f"  温度上昇 → M_s 減少（熱揺らぎ増加）")
    print(f"  温度上昇 → H_c 減少（熱活性化による磁化反転促進）")
    print(f"  Curie温度に近づくと急激に減少")
    

## 3.8 演習問題

### 演習3-1: Curie定数の計算（Easy）

Easy **問題** ：磁化率が T = 200 K で χ = 0.01、300 K で χ = 0.0067 であった。Curie則 $\chi = C/T$ を仮定し、Curie定数 C を求めよ。

**解答例を表示**
    
    
    import numpy as np
    
    T1, chi1 = 200, 0.01
    T2, chi2 = 300, 0.0067
    
    # C = χ × T
    C1 = chi1 * T1
    C2 = chi2 * T2
    
    print(f"T = {T1} K: C = {C1:.2f}")
    print(f"T = {T2} K: C = {C2:.2f}")
    print(f"平均 C = {(C1 + C2) / 2:.2f}")
    

**解答** ：C ≈ 2.0（ほぼ一致 → Curie則が成立）

### 演習3-2: 飽和磁化の計算（Easy）

Easy **問題** ：Feの飽和磁化が 220 emu/g、試料質量が 5 mg のとき、試料の総磁気モーメントを emu 単位で求めよ。

**解答例を表示**
    
    
    M_s = 220  # [emu/g]
    mass = 0.005  # [g] = 5 mg
    
    total_moment = M_s * mass
    print(f"総磁気モーメント = {total_moment:.3f} emu")
    

**解答** ：1.100 emu

### 演習3-3: Langevin関数の評価（Easy）

Easy **問題** ：磁場 H = 5000 Oe、温度 T = 300 K、粒子磁気モーメント μ = 10,000 μ$_B$ のとき、Langevinパラメータ ξ = μBH/(k$_B$T) を計算せよ（μ$_B$ = 9.274 × 10$^{-24}$ J/T）。

**解答例を表示**
    
    
    mu_B = 9.274e-24  # [J/T]
    k_B = 1.38065e-23  # [J/K]
    H = 5000 * 1e-4  # [Oe] → [T]
    T = 300  # [K]
    mu = 10000  # [μ_B]
    
    xi = mu * mu_B * H / (k_B * T)
    print(f"Langevinパラメータ ξ = {xi:.3f}")
    print(f"ξ < 1 → 線形領域（M ≈ χH）")
    print(f"ξ >> 1 → 飽和領域（M ≈ M_s）")
    

**解答** ：ξ ≈ 1.12（線形から飽和への遷移領域）

### 演習3-4: 保磁力から異方性定数を計算（Medium）

Medium **問題** ：保磁力 H$_c$ = 2000 Oe、飽和磁化 M$_s$ = 800 emu/cm$^3$ の材料の磁気異方性定数 K を計算せよ（単位：erg/cm$^3$ と J/m$^3$）。

**解答例を表示**
    
    
    H_c = 2000  # [Oe]
    M_s = 800  # [emu/cm^3]
    
    # K = H_c * M_s / 2（CGS単位）
    K_cgs = H_c * M_s / 2
    print(f"磁気異方性定数 K = {K_cgs:.2e} erg/cm³")
    
    # SI単位に変換：1 erg/cm³ = 10^-1 J/m³
    K_si = K_cgs * 1e-1
    print(f"                  = {K_si:.2e} J/m³")
    

**解答** ：K = 8.0 × 10$^5$ erg/cm$^3$ = 8.0 × 10$^4$ J/m$^3$

### 演習3-5: FC/ZFC曲線からブロッキング温度を抽出（Medium）

Medium **問題** ：ZFC曲線が T = 10, 20, 30, 40, 50, 60, 70 K で M = 5, 10, 15, 18, 16, 12, 8 emu/g と測定された。ブロッキング温度 T$_B$ を求めよ。

**解答例を表示**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    T = np.array([10, 20, 30, 40, 50, 60, 70])  # [K]
    M_ZFC = np.array([5, 10, 15, 18, 16, 12, 8])  # [emu/g]
    
    # ピーク温度
    T_B = T[np.argmax(M_ZFC)]
    M_peak = np.max(M_ZFC)
    
    print(f"ブロッキング温度 T_B = {T_B} K")
    print(f"ピーク磁化 M_peak = {M_peak} emu/g")
    
    # プロット
    plt.figure(figsize=(8, 6))
    plt.plot(T, M_ZFC, 'o-', linewidth=2.5, markersize=8, color='#99ccff', label='ZFC')
    plt.axvline(T_B, color='red', linestyle='--', linewidth=2, label=f'T_B = {T_B} K')
    plt.scatter([T_B], [M_peak], s=200, c='red', edgecolors='black', linewidth=2, marker='X', zorder=5)
    plt.xlabel('Temperature T [K]', fontsize=12)
    plt.ylabel('Magnetization M [emu/g]', fontsize=12)
    plt.title('ZFC Curve: Blocking Temperature', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.show()
    

**解答** ：T$_B$ = 40 K

### 演習3-6: M-H曲線からエネルギー積を計算（Medium）

Medium **問題** ：永久磁石のM-H曲線から、最大エネルギー積 (BH)$_{\text{max}}$ を推定せよ（M$_r$ = 12 kG、H$_c$ = 10 kOe と仮定）。

**解答例を表示**
    
    
    M_r = 12000  # [G] = 12 kG
    H_c = 10000  # [Oe] = 10 kOe
    
    # 簡易的な推定：(BH)_max ≈ (B_r × H_c) / 4
    # B_r ≈ M_r（CGS単位）
    BH_max = (M_r * H_c) / 4 / 1e6  # MGOe単位
    
    print(f"最大エネルギー積 (BH)_max ≈ {BH_max:.1f} MGOe")
    print(f"\n実用磁石の比較:")
    print(f"  Ferrite: ~3-5 MGOe")
    print(f"  SmCo: ~20-30 MGOe")
    print(f"  NdFeB: ~40-50 MGOe")
    

**解答** ：(BH)$_{\text{max}}$ ≈ 30 MGOe（SmCo級の性能）

### 演習3-7: 多温度データのCurie温度推定（Hard）

Hard **問題** ：M$_s$(T) が T = 100, 200, 300, 400, 500 K で 95, 85, 70, 45, 10 emu/g と測定された。Curie温度 T$_C$ を推定せよ（ヒント：M$_s$(T) ∝ (T$_C$ - T)$^β$、β ≈ 0.5）。

**解答例を表示**
    
    
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    
    T = np.array([100, 200, 300, 400, 500])  # [K]
    M_s = np.array([95, 85, 70, 45, 10])  # [emu/g]
    
    # モデル：M_s(T) = M_0 * (1 - T/T_C)^β
    def curie_temperature_model(T, M_0, T_C, beta):
        return M_0 * np.maximum(1 - T / T_C, 0)**beta
    
    # フィッティング
    params, _ = curve_fit(curie_temperature_model, T, M_s, p0=[100, 600, 0.5])
    M_0, T_C, beta = params
    
    print(f"フィッティング結果:")
    print(f"  M_0 = {M_0:.1f} emu/g")
    print(f"  Curie温度 T_C = {T_C:.1f} K")
    print(f"  臨界指数 β = {beta:.2f}")
    
    # プロット
    T_fit = np.linspace(0, 600, 100)
    M_fit = curie_temperature_model(T_fit, M_0, T_C, beta)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(T, M_s, s=100, edgecolors='black', linewidths=2, label='Measured data', color='#f093fb', zorder=5)
    plt.plot(T_fit, M_fit, linewidth=2.5, label=f'Fit: T_C = {T_C:.1f} K, β = {beta:.2f}', color='#f5576c')
    plt.axvline(T_C, color='red', linestyle='--', linewidth=2, label=f'T_C = {T_C:.1f} K')
    plt.xlabel('Temperature T [K]', fontsize=12)
    plt.ylabel('Saturation Magnetization M$_s$ [emu/g]', fontsize=12)
    plt.title('Curie Temperature Estimation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim(0, 650)
    plt.ylim(0, 110)
    plt.show()
    

**解答** ：T$_C$ ≈ 550-570 K（フィッティングに依存）、β ≈ 0.4-0.5（理論値 0.5 に近い）

### 演習3-8: SQUIDとVSMの感度比較（Hard）

Hard **問題** ：試料質量 1 mg、飽和磁化 10 emu/g の材料を測定する。VSM（感度 $10^{-6}$ emu）とSQUID（感度 $10^{-8}$ emu）で、S/N比が何倍異なるか評価せよ。

**解答例を表示**
    
    
    mass = 0.001  # [g] = 1 mg
    M_s = 10  # [emu/g]
    
    # 試料の総磁気モーメント
    moment = M_s * mass
    print(f"試料の総磁気モーメント = {moment:.2e} emu")
    
    # VSM
    sensitivity_VSM = 1e-6  # [emu]
    SNR_VSM = moment / sensitivity_VSM
    print(f"\nVSM:")
    print(f"  感度: {sensitivity_VSM:.0e} emu")
    print(f"  S/N比: {SNR_VSM:.1f}")
    
    # SQUID
    sensitivity_SQUID = 1e-8  # [emu]
    SNR_SQUID = moment / sensitivity_SQUID
    print(f"\nSQUID:")
    print(f"  感度: {sensitivity_SQUID:.0e} emu")
    print(f"  S/N比: {SNR_SQUID:.1f}")
    
    # 比較
    ratio = SNR_SQUID / SNR_VSM
    print(f"\nSQUIDのS/N比はVSMの {ratio:.0f}倍")
    print(f"→ SQUIDは微量試料や微弱磁性の測定に有利")
    

**解答** ：S/N比はSQUIDがVSMの100倍（感度の比に対応）

### 演習3-9: 実験計画（Hard）

Hard **問題** ：未知の磁性ナノ粒子（質量 2 mg）の磁気特性を完全に評価する実験計画を立案せよ。VSM/SQUID測定、温度範囲、磁場範囲、データ解析手法を具体的に説明せよ。

**解答例を表示**

**実験計画** ：

  1. **測定装置選定** ： 
     * 質量 2 mg → 総磁気モーメント $\sim 10^{-3}$ - $10^{-2}$ emu（予想）
     * VSM（感度 $10^{-6}$ emu）で十分だが、超常磁性の可能性を考慮してSQUID推奨
  2. **室温M-H曲線測定** （T = 300 K）： 
     * 磁場範囲：-2 T 〜 +2 T（±20 kOe）、ステップ 100 Oe
     * 目的：M$_s$、H$_c$、M$_r$、磁性の種類（常磁性/強磁性）を判定
  3. **FC/ZFC測定** （超常磁性評価）： 
     * 温度範囲：5 K - 350 K、測定磁場 100 Oe
     * ZFCピークからブロッキング温度 T$_B$ を決定
     * FC/ZFC分岐 → 粒子間相互作用の評価
  4. **温度依存性M-H曲線** ： 
     * 温度：10 K, 50 K, 100 K, 200 K, 300 K
     * 各温度で完全なヒステリシスループ測定
     * H$_c$(T)、M$_s$(T) の温度依存性から磁気異方性とCurie温度を評価
  5. **データ解析** ： 
     * M-H曲線：バックグラウンド減算、特性値抽出（M$_s$、H$_c$、M$_r$、S）
     * 常磁性の場合：Langevin関数フィッティング → 粒子サイズ推定
     * 強磁性の場合：磁気異方性定数 K 計算、Curie温度推定
     * FC/ZFC解析：T$_B$ 決定、Néel緩和時間 τ の推定

**期待される結果** ：

  * **超常磁性ナノ粒子** ：H$_c$ ≈ 0（室温）、ZFCピーク有り、T$_B$ < 300 K
  * **強磁性ナノ粒子** ：H$_c$ > 100 Oe、角形性あり、T$_B$ > 300 K
  * **常磁性ナノ粒子** ：線形M-H、ヒステリシスなし

## 3.9 学習の確認

以下のチェックリストで理解度を確認しましょう：

### 基本理解

  * 反磁性・常磁性・強磁性の違いを説明できる
  * VSMとSQUIDの測定原理を理解している
  * M-H曲線から飽和磁化、保磁力、残留磁化を抽出できる
  * Curie-Weiss則とLangevin関数の物理的意味を理解している
  * FC/ZFC測定の目的と解釈を説明できる

### 実践スキル

  * M-H曲線のバックグラウンド減算ができる
  * Curie-Weiss則とLangevin関数をフィッティングできる
  * 磁気異方性定数を計算できる
  * Pythonで完全なVSM/SQUIDデータ処理ワークフローを実装できる
  * ブロッキング温度をFC/ZFC曲線から抽出できる

### 応用力

  * 材料の磁性の種類を判定できる（軟磁性/硬磁性/超常磁性）
  * 温度依存性からCurie温度を推定できる
  * 実験計画を立案し、適切な測定条件を決定できる
  * 測定の限界（感度、ノイズ）を評価し、対策を提案できる

## 3.10 参考文献

  1. Foner, S. (1959). _Versatile and Sensitive Vibrating-Sample Magnetometer_. Review of Scientific Instruments, 30(7), 548-557. - VSMの原論文
  2. Clarke, J., & Braginski, A. I. (Eds.). (2004). _The SQUID Handbook_. Wiley-VCH. - SQUID技術の包括的解説
  3. Cullity, B. D., & Graham, C. D. (2009). _Introduction to Magnetic Materials_ (2nd ed.). Wiley-IEEE Press. - 磁性材料の標準教科書
  4. Jiles, D. (2015). _Introduction to Magnetism and Magnetic Materials_ (3rd ed.). CRC Press. - 磁性物理学の詳細
  5. Néel, L. (1949). _Théorie du traînage magnétique des ferromagnétiques en grains fins avec applications aux terres cuites_. Annales de Géophysique, 5, 99-136. - 超常磁性理論の原論文
  6. Stoner, E. C., & Wohlfarth, E. P. (1948). _A Mechanism of Magnetic Hysteresis in Heterogeneous Alloys_. Philosophical Transactions of the Royal Society A, 240(826), 599-642. - Stoner-Wohlfarth理論
  7. Quantum Design. _PPMS DynaCool User's Manual_. - PPMS測定技術の実践的ガイド

## 3.11 次章へ

次章では、**Pythonによる電磁気データ解析の実践ワークフロー** を学びます。実際の測定装置データの読み込み、前処理、高度なフィッティング技術、機械学習を用いた異常検出、レポート自動生成までの統合的なデータ解析パイプラインを構築します。
